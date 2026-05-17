"""V6 calendário operacional — modelo V14 ensemble (5 seeds).

Gera calendário usando voto/Q-mean dos 5 modelos V14 treinados.
Hook automático: regenera HTML ao final.
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
from dqn import BranchingDQN, NEG_INF

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V14 = ROOT / 'results' / 'v14'
V14.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--horizonte', type=int, default=365)
parser.add_argument('--data', type=str, default='2026-05-13')
parser.add_argument('--seeds', type=int, nargs='+',
                     default=[42, 123, 999, 2025, 7777])
parser.add_argument('--mode', type=str, default='qmean', choices=['vote', 'qmean'])
parser.add_argument('--min_dias_campanha', type=int, default=2)
parser.add_argument('--max_dias_campanha', type=int, default=7)
parser.add_argument('--no_repeat_window', type=int, default=2)
args = parser.parse_args()

DATA_HOJE = date.fromisoformat(args.data)
HORIZONTE = args.horizonte


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    m = BranchingDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m


print(f"Carregando ensemble de {len(args.seeds)} seeds…")
modelos = []
for s in args.seeds:
    p = ROOT / f'results/v14/dqn_v14_seed{s}.pt'
    if p.exists():
        modelos.append(load_model(p))
        print(f"  ✓ seed {s}")

env = construir_env_v12(modo='validacao')

# Reset manual para data específica
env.estoque = env.estoque_inicial.copy()
env.idade = np.zeros(env.N, dtype=np.float32)
env.promo_ant = np.zeros(env.N, dtype=np.float32)
env.acao_ant = (0, 0)
env.turno = 0
env.passo = 0
env.historico_categorias = []
env.data_atual = DATA_HOJE
env._np_random = np.random.default_rng(0)
env.demanda_buffer = np.tile(env.receita_media_cat, (28, 1)).astype(np.float32)
env.receita_dia_atual = np.zeros(env.N, dtype=np.float32)
env._fc_cache_date = None


def ensemble_action(obs_t, mask_cat):
    """Q-mean ensemble: soma Q-values dos modelos, argmax do agregado."""
    q_total = None
    with torch.no_grad():
        for m in modelos:
            q = m.q_values(obs_t)
            if mask_cat is not None:
                mt = torch.tensor(mask_cat, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
                q = q.masked_fill(~mt, NEG_INF)
            q_total = q if q_total is None else q_total + q
    flat = q_total.view(-1)
    idx = int(flat.argmax().item())
    n_i = modelos[0].n_intensidades
    return idx // n_i, idx % n_i


print(f"\nV6 rollout V14 ensemble ({args.mode}, {len(modelos)} seeds) "
      f"{HORIZONTE}d a partir de {DATA_HOJE}...")
decisoes = []
n_steps = HORIZONTE * 3
obs = env._get_obs()
for step in range(n_steps):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask = env.get_action_mask(no_repeat_window=args.no_repeat_window)
    ap, ai = ensemble_action(obs_t, mask)
    data_pre = env.data_atual.isoformat()
    turno_pre = env.turno
    obs, _, term, trunc, info = env.step((ap, ai))
    decisoes.append({
        'step': step, 'data': data_pre, 'turno': turno_pre,
        'produto_idx': ap, 'intensidade_idx': ai,
        'categoria': env.cats[ap - 1]['categoria'] if ap > 0 else None,
        'combo_par': info.get('combo_par'),
    })
    if term or trunc: break

df_dec = pd.DataFrame(decisoes)
print(f"  {len(df_dec)} decisões, {df_dec['data'].nunique()} dias")
n_cigarro = df_dec['categoria'].fillna('').str.contains('cigarro').sum()
print(f"  Cigarro promovido: {n_cigarro}/{len(df_dec)} turnos ({100*n_cigarro/len(df_dec):.2f}%)")

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
campanhas, em_camp, atual = [], False, None
for d in dias_ordenados:
    cat, intens = decisoes_por_dia[d]
    if cat is None:
        if em_camp and atual['dias_total'] >= args.min_dias_campanha:
            campanhas.append(atual)
        em_camp = False; atual = None; continue
    if not em_camp:
        em_camp = True
        atual = {'data_inicio': d, 'data_fim': d,
                 'categoria': cat, 'intensidade_idx': intens, 'dias_total': 1}
    elif cat == atual['categoria'] and intens == atual['intensidade_idx']:
        atual['data_fim'] = d
        atual['dias_total'] += 1
        if atual['dias_total'] >= args.max_dias_campanha:
            campanhas.append(atual); em_camp = False; atual = None
    else:
        if atual['dias_total'] >= args.min_dias_campanha:
            campanhas.append(atual)
        atual = {'data_inicio': d, 'data_fim': d,
                 'categoria': cat, 'intensidade_idx': intens, 'dias_total': 1}
if em_camp and atual and atual['dias_total'] >= args.min_dias_campanha:
    campanhas.append(atual)

print(f"  {len(campanhas)} campanhas")

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
    uplift = demanda_dia * c['dias_total'] * (boost - 1)
    lucro_adic = uplift * cat_cfg['margem'] * (1 - desc / 100)
    evs = []
    d_ini = date.fromisoformat(c['data_inicio'])
    d_fim = date.fromisoformat(c['data_fim'])
    for ev_data, ev in cal_dict.items():
        ev_d = date.fromisoformat(ev_data)
        if not (d_fim < ev_d - timedelta(days=int(ev['janela_pre_dias']))
                or d_ini > ev_d + timedelta(days=int(ev['janela_pos_dias']))):
            evs.append(ev['nome_evento'])
    c['intensidade'] = INTENS_NOMES[intens]
    c['desconto_pct'] = desc
    c['demanda_base_dia'] = round(demanda_dia, 1)
    c['uplift_unidades_estimado'] = round(uplift, 1)
    c['lucro_adicional_estimado_R$'] = round(lucro_adic, 2)
    c['preco_unitario'] = round(cat_cfg['preco_venda'], 2)
    c['margem_unitaria'] = round(cat_cfg['margem'], 2)
    c['eventos_comerciais_na_janela'] = evs
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
    'versao': 'V6-modelo-V14-ensemble',
    'modelo_base': f'V14 ensemble de {len(modelos)} seeds ({args.mode}) — PER + curriculum + mask dinâmico',
    'gerado_em': str(date.today()),
    'data_inicio': DATA_HOJE.isoformat(),
    'data_fim': (DATA_HOJE + timedelta(days=HORIZONTE - 1)).isoformat(),
    'horizonte_dias': HORIZONTE,
    'n_campanhas': len(campanhas),
    'lucro_adicional_total_R$': round(sum(c['lucro_adicional_estimado_R$']
                                            for c in campanhas), 2),
    'cigarro_promovido_pct': round(100 * n_cigarro / len(df_dec), 2),
    'seeds': args.seeds,
    'campanhas': campanhas,
}

with open(V14 / 'calendario_v6.json', 'w', encoding='utf-8') as f:
    json.dump(calendario, f, indent=2, ensure_ascii=False, default=str)

print()
print("=" * 70)
print(f"CALENDÁRIO V6 GERADO (V14 ensemble, {len(modelos)} seeds)")
print("=" * 70)
print(f"  Período: {DATA_HOJE} a {DATA_HOJE + timedelta(days=HORIZONTE-1)}")
print(f"  Campanhas: {len(campanhas)}")
print(f"  Lucro adicional estimado: R$ {calendario['lucro_adicional_total_R$']:,.2f}")
print(f"  Cigarro: {calendario['cigarro_promovido_pct']:.2f}%")
if campanhas:
    print("\nTop 10 campanhas:")
    top = sorted(campanhas, key=lambda c: -c['lucro_adicional_estimado_R$'])[:10]
    for c in top:
        d_ini = date.fromisoformat(c['data_inicio']).strftime('%d/%m')
        d_fim = date.fromisoformat(c['data_fim']).strftime('%d/%m')
        par = f" + {c.get('produto_complementar', '')}" if c.get('produto_complementar') else ''
        ev = f" [{c['eventos_comerciais_na_janela'][0]}]" if c['eventos_comerciais_na_janela'] else ''
        print(f"  {d_ini}-{d_fim} ({c['dias_total']}d) {c['categoria']}{par:<22s} "
              f"{c['intensidade']:<8s} R$ {c['lucro_adicional_estimado_R$']:>8.2f}{ev}")
print()
print(f"✓ Saída: results/v14/calendario_v6.json")

# ── Hook automático: regenera HTML premium ───────────────────────────────
try:
    import subprocess
    import shutil
    # Reusar template HTML do V12 apontando para V14 JSON
    shutil.copy(V14 / 'calendario_v6.json',
                ROOT / 'results' / 'v12' / 'calendario_v4_anual.json')
    print()
    print("Gerando HTML premium V14...")
    subprocess.run([sys.executable, str(ROOT / 'gerar_html_premium_v12.py')],
                    check=True, cwd=str(ROOT))
    shutil.copy(ROOT / 'results' / 'v12' / 'calendario_premium_v12.html',
                V14 / 'calendario_premium_v14.html')
    print(f"✓ HTML: results/v14/calendario_premium_v14.html")
except Exception as e:
    print(f"⚠ HTML premium não gerou: {e}")
