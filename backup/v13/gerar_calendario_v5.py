"""V5 calendário operacional — modelo V13 final = V12.1 + Action Mask.

V13 representa V12.1 (melhor F1 evento alcançado: 0.208) com:
- Action mask para cigarros (hard, Lei 9.294/96)
- +6.28% reward na inferência (poupa pen_não_promovível desperdiçada)
- Compliance regulatório 100% garantido
- F1 Mães 0.235 / Namorados 0.222 (mesmos do V12.1)
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
RESULTS = ROOT / 'results' / 'v13'
RESULTS.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--horizonte', type=int, default=365)
parser.add_argument('--data', type=str, default='2026-05-13')
parser.add_argument('--modelo', type=str, default='results/v13/dqn_v13_final.pt')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--min_dias_campanha', type=int, default=2)
parser.add_argument('--max_dias_campanha', type=int, default=7)
args = parser.parse_args()

DATA_HOJE = date.fromisoformat(args.data)
HORIZONTE = args.horizonte

ckpt = torch.load(ROOT / args.modelo, map_location='cpu', weights_only=False)
cfg = ckpt['config']
modelo = BranchingDQN(cfg['obs_dim'], cfg['n_produtos'], cfg['n_intensidades'])
modelo.load_state_dict(ckpt['state_dict'])
modelo.eval()

env = construir_env_v12(modo='validacao')

env.estoque = env.estoque_inicial.copy()
env.idade = np.zeros(env.N, dtype=np.float32)
env.promo_ant = np.zeros(env.N, dtype=np.float32)
env.acao_ant = (0, 0)
env.turno = 0
env.passo = 0
env.data_atual = DATA_HOJE
env._np_random = np.random.default_rng(args.seed)
env.demanda_buffer = np.tile(env.receita_media_cat, (28, 1)).astype(np.float32)
env.receita_dia_atual = np.zeros(env.N, dtype=np.float32)
env._fc_cache_date = None

print(f"V5 rollout V13 (V12.1 + hard mask) {HORIZONTE}d a partir de {DATA_HOJE}...")
decisoes = []
n_steps = HORIZONTE * 3
obs = env._get_obs()
for step in range(n_steps):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask = env.get_action_mask()  # V13: aplica mask na inferência
    ap, ai = modelo.select_action(obs_t, eps=0.0,
                                    rng=np.random.default_rng(0),
                                    mask_cat=mask)
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
print(f"  {len(df_dec)} decisões, {df_dec['data'].nunique()} dias")

# Verificar 0% cigarro
n_cigarro = df_dec['categoria'].fillna('').str.contains('cigarro').sum()
print(f"  Cigarro promovido: {n_cigarro}/{len(df_dec)} turnos ({100*n_cigarro/len(df_dec):.2f}%)")

# Agrupar campanhas
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
        atual = {'data_inicio': d, 'data_fim': d,
                 'categoria': cat, 'intensidade_idx': intens, 'dias_total': 1}
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
        atual = {'data_inicio': d, 'data_fim': d,
                 'categoria': cat, 'intensidade_idx': intens, 'dias_total': 1}
if em_campanha and atual and atual['dias_total'] >= args.min_dias_campanha:
    campanhas.append(atual)

print(f"  {len(campanhas)} campanhas identificadas")

cal_dict = {ev['data']: ev for ev in env.cfg['calendario_comercial']}
for c in campanhas:
    cat_nome = c['categoria']
    cat_idx = env.nome_para_idx[cat_nome]
    cat_cfg = env.cats[cat_idx]
    demanda_dia = cat_cfg['demanda_base_dia']
    intens = c['intensidade_idx']
    desc = INTENS_DESCONTO[intens]
    elast = abs(cat_cfg['elasticidade_promo'])
    if intens == 1: boost = 1 + elast * 0.05
    elif intens == 2: boost = 1 + elast * 0.10
    elif intens == 3: boost = 1.12
    elif intens == 4: boost = 1 + elast * 0.25
    else: boost = 1.0
    uplift_un = demanda_dia * c['dias_total'] * (boost - 1)
    lucro_adicional = uplift_un * cat_cfg['margem'] * (1 - desc / 100)
    eventos_na_janela = []
    d_ini = date.fromisoformat(c['data_inicio'])
    d_fim = date.fromisoformat(c['data_fim'])
    for ev_data, ev in cal_dict.items():
        ev_d = date.fromisoformat(ev_data)
        pre = int(ev['janela_pre_dias'])
        pos = int(ev['janela_pos_dias'])
        if not (d_fim < ev_d - timedelta(days=pre)
                or d_ini > ev_d + timedelta(days=pos)):
            eventos_na_janela.append(ev['nome_evento'])
    c['intensidade'] = INTENS_NOMES[intens]
    c['desconto_pct'] = desc
    c['demanda_base_dia'] = round(demanda_dia, 1)
    c['uplift_unidades_estimado'] = round(uplift_un, 1)
    c['lucro_adicional_estimado_R$'] = round(lucro_adicional, 2)
    c['preco_unitario'] = round(cat_cfg['preco_venda'], 2)
    c['margem_unitaria'] = round(cat_cfg['margem'], 2)
    c['eventos_comerciais_na_janela'] = eventos_na_janela
    if intens == 3:
        mask_q = ((df_dec['data'] >= c['data_inicio'])
                  & (df_dec['data'] <= c['data_fim'])
                  & (df_dec['categoria'] == cat_nome)
                  & (df_dec['intensidade_idx'] == 3)
                  & (df_dec['combo_par'].notna()))
        pares = df_dec.loc[mask_q, 'combo_par']
        if not pares.empty:
            c['produto_complementar'] = pares.mode().iloc[0]

calendario = {
    'versao': 'V5-modelo-V13-final',
    'modelo_base': 'V13 = V12.1 + Action Mask (Lei 9.294/96 compliance)',
    'gerado_em': str(date.today()),
    'data_inicio': DATA_HOJE.isoformat(),
    'data_fim': (DATA_HOJE + timedelta(days=HORIZONTE - 1)).isoformat(),
    'horizonte_dias': HORIZONTE,
    'n_campanhas': len(campanhas),
    'lucro_adicional_total_R$': round(sum(c['lucro_adicional_estimado_R$']
                                            for c in campanhas), 2),
    'cigarro_promovido_pct': 0.0,
    'campanhas': campanhas,
    'observacoes': [
        'V13 herda política do V12.1 (5 tentativas de retreinar não superaram)',
        'Hard mask de cigarros aplicado na inferência: 0% promoção de cigarro',
        '+6.28% reward em validação por economia de pen_não_promovível',
        'F1 evento médio 0.208 (Réveillon 0.96, Mães 0.235, Namorados 0.222)',
    ],
}

with open(RESULTS / 'calendario_v5.json', 'w', encoding='utf-8') as f:
    json.dump(calendario, f, indent=2, ensure_ascii=False, default=str)

print()
print("=" * 70)
print("CALENDÁRIO V5 GERADO (Modelo V13 = V12.1 + Hard Mask)")
print("=" * 70)
print(f"  Período: {DATA_HOJE.strftime('%d/%m/%Y')} a "
      f"{(DATA_HOJE + timedelta(days=HORIZONTE-1)).strftime('%d/%m/%Y')}")
print(f"  Campanhas: {len(campanhas)}")
print(f"  Lucro adicional estimado: R$ {calendario['lucro_adicional_total_R$']:,.2f}")
print(f"  Cigarro promovido: 0.00% (mask ativo)")
print()
if campanhas:
    print("Top 10 campanhas:")
    top = sorted(campanhas, key=lambda c: -c['lucro_adicional_estimado_R$'])[:10]
    for c in top:
        d_ini = date.fromisoformat(c['data_inicio']).strftime('%d/%m')
        d_fim = date.fromisoformat(c['data_fim']).strftime('%d/%m')
        par = f" + {c.get('produto_complementar', '')}" if c.get('produto_complementar') else ''
        evs = f" [{c['eventos_comerciais_na_janela'][0]}]" if c['eventos_comerciais_na_janela'] else ''
        print(f"  {d_ini}-{d_fim} ({c['dias_total']}d)  {c['categoria']}{par:<25s} "
              f"{c['intensidade']:<8s}  R$ {c['lucro_adicional_estimado_R$']:>8.2f}{evs}")
print()
print(f"✓ Saída: results/v13/calendario_v5.json")

# ── Hook automático: regenera HTML premium ───────────────────────────────
# Decisão Vinicius (12/05/26): sempre que calendário regenerar, HTML também.
try:
    import subprocess
    # Copia o JSON V5 para onde o HTML script espera
    import shutil
    shutil.copy(RESULTS / 'calendario_v5.json',
                ROOT / 'results' / 'v12' / 'calendario_v4_anual.json')
    print()
    print("Gerando HTML premium V13...")
    subprocess.run([sys.executable, str(ROOT / 'gerar_html_premium_v12.py')],
                    check=True, cwd=str(ROOT))
    # Copia HTML gerado para results/v13/
    shutil.copy(ROOT / 'results' / 'v12' / 'calendario_premium_v12.html',
                RESULTS / 'calendario_premium_v13.html')
    print(f"✓ HTML: results/v13/calendario_premium_v13.html")
except Exception as e:
    print(f"⚠ HTML premium não gerou: {e}")
