"""V4 do output operacional — usa o modelo V12 treinado.

Diferenças vs V3:
- Carrega dqn_v12.pt e env_v12 (forecasters embutidos)
- Estado tem 150 features (era 130 no V11)
- Resto da pipeline (rollout, agrupamento de campanhas, par dinâmico,
  enrichment com eventos comerciais) é idêntico ao V3.

Pipeline:
1. Carrega V12 treinado (dqn_v12.pt)
2. Constrói env_v12 iniciando na data atual + horizonte
3. Rollout determinístico (ε=0); captura par dinâmico via info['combo_par']
4. Agrupa decisões consecutivas em CAMPANHAS (≥2 dias, ≤7 dias)
5. Calcula uplift esperado de cada campanha
6. Salva JSON + Markdown
"""
import argparse
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env_v12 import construir_env_v12
from dqn import BranchingDQN

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results' / 'v12'
RESULTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--horizonte', type=int, default=60)
parser.add_argument('--data', type=str, default='2026-05-12')
parser.add_argument('--modelo', type=str, default='results/v12/dqn_v12.pt')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--min_dias_campanha', type=int, default=2)
parser.add_argument('--max_dias_campanha', type=int, default=7)
args = parser.parse_args()

DATA_HOJE = date.fromisoformat(args.data)
HORIZONTE = args.horizonte

# ── Carrega modelo ──────────────────────────────────────────────────────────

ckpt = torch.load(ROOT / args.modelo, map_location='cpu', weights_only=False)
cfg = ckpt['config']
modelo = BranchingDQN(cfg['obs_dim'], cfg['n_produtos'], cfg['n_intensidades'])
modelo.load_state_dict(ckpt['state_dict'])
modelo.eval()

env = construir_env_v12(modo='validacao')

# Forçar data inicial específica
env.estoque = env.estoque_inicial.copy()
env.idade = np.zeros(env.N, dtype=np.float32)
env.promo_ant = np.zeros(env.N, dtype=np.float32)
env.acao_ant = (0, 0)
env.turno = 0
env.passo = 0
env.data_atual = DATA_HOJE
env._np_random = np.random.default_rng(args.seed)
# Forecaster buffer
env.demanda_buffer = np.tile(env.receita_media_cat, (28, 1)).astype(np.float32)
env.receita_dia_atual = np.zeros(env.N, dtype=np.float32)
env._fc_cache_date = None

# ── Rollout ──────────────────────────────────────────────────────────────

print(f"V12 rollout determinístico {HORIZONTE} dias a partir de {DATA_HOJE}...")
decisoes = []
n_steps = HORIZONTE * 3
obs = env._get_obs()
for step in range(n_steps):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    ap, ai = modelo.select_action(obs_t, eps=0.0,
                                    rng=np.random.default_rng(0))
    data_pre = env.data_atual.isoformat()
    turno_pre = env.turno
    obs, _, term, trunc, info = env.step((ap, ai))
    decisoes.append({
        'step': step,
        'data': data_pre,
        'turno': turno_pre,
        'produto_idx': ap,
        'intensidade_idx': ai,
        'categoria': env.cats[ap - 1]['categoria'] if ap > 0 else None,
        'combo_par': info.get('combo_par'),
    })
    if term or trunc:
        break

df_dec = pd.DataFrame(decisoes)
print(f"  {len(df_dec)} decisões coletadas ({df_dec['data'].nunique()} dias)")

# ── Agrupar em campanhas ──────────────────────────────────────────────────

INTENS_NOMES = ['nada', 'desc5%', 'desc10%', 'combo', 'liq25%']
INTENS_DESCONTO = [0, 5, 10, 10, 25]

decisoes_por_dia = (df_dec.groupby('data')
                            .apply(lambda g: (
                                g[(g['produto_idx'] > 0)
                                   & (g['intensidade_idx'] > 0)]
                                  .groupby(['categoria', 'intensidade_idx'])
                                  .size().idxmax()
                                if ((g['produto_idx'] > 0)
                                     & (g['intensidade_idx'] > 0)).any()
                                else (None, 0)),
                                    include_groups=False)
                            .to_dict())

dias_ordenados = sorted(decisoes_por_dia.keys())
campanhas = []
em_campanha = False
atual = None

for d in dias_ordenados:
    cat, intens = decisoes_por_dia[d]
    if cat is None:
        if em_campanha and atual['dias_total'] >= args.min_dias_campanha:
            campanhas.append(atual)
        em_campanha = False
        atual = None
        continue

    if not em_campanha:
        em_campanha = True
        atual = {
            'data_inicio': d,
            'data_fim': d,
            'categoria': cat,
            'intensidade_idx': intens,
            'dias_total': 1,
        }
    elif cat == atual['categoria'] and intens == atual['intensidade_idx']:
        atual['data_fim'] = d
        atual['dias_total'] += 1
        if atual['dias_total'] >= args.max_dias_campanha:
            campanhas.append(atual)
            em_campanha = False
            atual = None
    else:
        if atual['dias_total'] >= args.min_dias_campanha:
            campanhas.append(atual)
        atual = {
            'data_inicio': d,
            'data_fim': d,
            'categoria': cat,
            'intensidade_idx': intens,
            'dias_total': 1,
        }

if em_campanha and atual and atual['dias_total'] >= args.min_dias_campanha:
    campanhas.append(atual)

print(f"  {len(campanhas)} campanhas identificadas")

# ── Enriquece campanhas ───────────────────────────────────────────────────

cal_dict = {ev['data']: ev for ev in env.cfg['calendario_comercial']}

for c in campanhas:
    cat_nome = c['categoria']
    cat_idx = env.nome_para_idx[cat_nome]
    cat_cfg = env.cats[cat_idx]

    demanda_dia = cat_cfg['demanda_base_dia']
    intens = c['intensidade_idx']
    intens_nome = INTENS_NOMES[intens]
    desc = INTENS_DESCONTO[intens]

    elast = abs(cat_cfg['elasticidade_promo'])
    if intens == 1:
        boost = 1 + elast * 0.05
    elif intens == 2:
        boost = 1 + elast * 0.10
    elif intens == 3:
        boost = 1.12
    elif intens == 4:
        boost = 1 + elast * 0.25
    else:
        boost = 1.0

    uplift_un = demanda_dia * c['dias_total'] * (boost - 1)
    lucro_adicional = uplift_un * cat_cfg['margem'] * (1 - desc / 100)

    eventos_na_janela = []
    d_ini = date.fromisoformat(c['data_inicio'])
    d_fim = date.fromisoformat(c['data_fim'])
    for ev_data, ev in cal_dict.items():
        ev_d = date.fromisoformat(ev_data)
        pre = int(ev['janela_pre_dias'])
        pos = int(ev['janela_pos_dias'])
        janela_ini = ev_d - timedelta(days=pre)
        janela_fim = ev_d + timedelta(days=pos)
        if not (d_fim < janela_ini or d_ini > janela_fim):
            eventos_na_janela.append(ev['nome_evento'])

    c['intensidade'] = intens_nome
    c['desconto_pct'] = desc
    c['demanda_base_dia'] = round(demanda_dia, 1)
    c['uplift_unidades_estimado'] = round(uplift_un, 1)
    c['lucro_adicional_estimado_R$'] = round(lucro_adicional, 2)
    c['preco_unitario'] = round(cat_cfg['preco_venda'], 2)
    c['margem_unitaria'] = round(cat_cfg['margem'], 2)
    c['eventos_comerciais_na_janela'] = eventos_na_janela
    if intens == 3:
        mask = ((df_dec['data'] >= c['data_inicio'])
                 & (df_dec['data'] <= c['data_fim'])
                 & (df_dec['categoria'] == cat_nome)
                 & (df_dec['intensidade_idx'] == 3)
                 & (df_dec['combo_par'].notna()))
        pares = df_dec.loc[mask, 'combo_par']
        if not pares.empty:
            c['produto_complementar'] = pares.mode().iloc[0]
        elif cat_cfg.get('par_combo'):
            c['produto_complementar'] = cat_cfg['par_combo']

# ── Salva JSON ──────────────────────────────────────────────────────────────

calendario = {
    'versao': 'V4-modelo-V12',
    'modelo_base': 'DQN V12 — env_v12 consolidado (forecaster Ridge + harmonia categorial + harmonia evento)',
    'gerado_em': str(date.today()),
    'data_inicio': DATA_HOJE.isoformat(),
    'data_fim': (DATA_HOJE + timedelta(days=HORIZONTE - 1)).isoformat(),
    'horizonte_dias': HORIZONTE,
    'n_campanhas': len(campanhas),
    'lucro_adicional_total_R$': round(sum(c['lucro_adicional_estimado_R$']
                                            for c in campanhas), 2),
    'campanhas': campanhas,
    'limitacoes': [
        'Forecaster Ridge calibrado em vendas históricas por CATEGORIA (não SKU)',
        'Lag features iniciam com receita média (sem warmup) — primeiros 28 dias são proxies',
        'Validade típica heurística (esperando ERP)',
        'Elasticidade da literatura + Dunnhumby — validação real só com A/B',
    ],
}

with open(RESULTS / 'calendario_v4.json', 'w', encoding='utf-8') as f:
    json.dump(calendario, f, indent=2, ensure_ascii=False, default=str)

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("CALENDÁRIO V4 GERADO (Modelo V12)")
print("=" * 70)
print(f"  Período: {DATA_HOJE.strftime('%d/%m/%Y')} a "
      f"{(DATA_HOJE + timedelta(days=HORIZONTE-1)).strftime('%d/%m/%Y')}")
print(f"  Campanhas: {len(campanhas)}")
print(f"  Lucro adicional estimado: R$ {calendario['lucro_adicional_total_R$']:,.2f}")
print()
if campanhas:
    print("Top 5 campanhas:")
    top = sorted(campanhas, key=lambda c: -c['lucro_adicional_estimado_R$'])[:5]
    for c in top:
        d_ini = date.fromisoformat(c['data_inicio']).strftime('%d/%m')
        d_fim = date.fromisoformat(c['data_fim']).strftime('%d/%m')
        par = f" + {c.get('produto_complementar', '')}" if c.get('produto_complementar') else ''
        evs = f" [{c['eventos_comerciais_na_janela'][0]}]" if c['eventos_comerciais_na_janela'] else ''
        print(f"  {d_ini}-{d_fim} ({c['dias_total']}d)  {c['categoria']}{par:<25s} "
              f"{c['intensidade']:<8s}  R$ {c['lucro_adicional_estimado_R$']:>8.2f}{evs}")
else:
    print("  ⚠ Modelo não recomendou nenhuma campanha")
print()
print(f"✓ Saídas:")
print(f"  results/v12/calendario_v4.json")

# ── Hook automático: regenera HTML premium ───────────────────────────────
try:
    import subprocess
    print()
    print("Gerando HTML premium V12...")
    subprocess.run([sys.executable, str(ROOT / 'gerar_html_premium_v12.py')],
                    check=True, cwd=str(ROOT))
    print(f"✓ HTML: results/v12/calendario_premium_v12.html")
except Exception as e:
    print(f"⚠ HTML premium não gerou: {e}")
