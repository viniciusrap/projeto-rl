"""Valida V13 (CondPolicyDQN) e compara com V12.1.

Métricas: reward, lucro, perdas, F1 timing, F1 evento médio + F1 nas datas
problemáticas (Mães, Namorados, Mulher).

Saídas em results/v13/:
  - validacao_metricas_v13.csv
  - validacao_eventos_v13.csv
  - comparacao_v12_vs_v13.csv

Uso: python validar_v13.py [--n_episodios N] [--suffix v1]
"""
import argparse
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env_v12 import construir_env_v12
from dqn import CondPolicyDQN, BranchingDQN, BranchingDQNCross

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V12 = ROOT / 'results' / 'v12'
V13 = ROOT / 'results' / 'v13'
V13.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_episodios', type=int, default=15)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_steps', type=int, default=1095)
parser.add_argument('--suffix', type=str, default='v1')
parser.add_argument('--modelo_v12', type=str, default='results/v12/dqn_v12.pt')
args = parser.parse_args()


def load_v13(path):
    """Detecta arquitetura via config['arch']. Default: CondPolicyDQN."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    arch = c.get('arch', 'CondPolicyDQN')
    if arch == 'BranchingDQNCross':
        m = BranchingDQNCross(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    elif arch == 'BranchingDQN':
        m = BranchingDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    else:
        m = CondPolicyDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'],
                           emb_dim=c.get('emb_dim', 16))
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m


load_cond = load_v13  # backward compat


def load_branching(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    m = BranchingDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m


def rollout(env, modelo, n_ep, max_steps, seed_base, use_mask=True):
    rewards, lucros, perdas, rupturas = [], [], [], []
    decisoes = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed_base + ep * 100)
        ep_r, ep_l, ep_p, ep_rup = 0.0, 0.0, 0.0, 0.0
        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = env.get_action_mask() if use_mask else None
            ap, ai = modelo.select_action(obs_t, eps=0.0,
                                            rng=np.random.default_rng(0),
                                            mask_cat=mask)
            obs2, r, term, trunc, info = env.step((ap, ai))
            ep_r += r
            ep_l += info['lucro']
            ep_p += float(info['perdas'].sum())
            ep_rup += float(info['rupturas'].sum())
            decisoes.append({
                'ep': ep,
                'step': step,
                'data': info['data'],
                'turno': info['turno'],
                'produto': ap,
                'intensidade': ai,
            })
            obs = obs2
            if term or trunc:
                break
        rewards.append(ep_r)
        lucros.append(ep_l)
        perdas.append(ep_p)
        rupturas.append(ep_rup)
    return {
        'reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'lucro': float(np.mean(lucros)),
        'perdas': float(np.mean(perdas)),
        'rupturas': float(np.mean(rupturas)),
        'decisoes': pd.DataFrame(decisoes),
    }


# ── Carregar modelos ─────────────────────────────────────────────

env = construir_env_v12(modo='validacao')

print(f"Carregando V13.{args.suffix}…")
modelo_v13 = load_cond(ROOT / f'results/v13/dqn_v13_{args.suffix}.pt')

print(f"Carregando V12.1 (baseline)…")
modelo_v12 = load_branching(ROOT / args.modelo_v12)

# ── Rollouts ──────────────────────────────────────────────────

print(f"\n=== AVALIAÇÃO ({args.n_episodios} ep × {args.max_steps} steps) ===")

print(f"  V13.{args.suffix} (CondPolicyDQN + mask)…")
out_v13 = rollout(env, modelo_v13, args.n_episodios, args.max_steps, args.seed)

print(f"  V12.1 (BranchingDQN + mask)…")
out_v12 = rollout(env, modelo_v12, args.n_episodios, args.max_steps, args.seed)

print(f"  Sem promoção…")
out_sp = rollout(env, _no_op := type('P', (), {
    'select_action': lambda *a, **k: (0, 0)
})(), args.n_episodios, args.max_steps, args.seed, use_mask=False)


# ── Tabela comparativa ────────────────────────────────

dados = []
for nome, r in [('V13.' + args.suffix, out_v13),
                 ('V12.1', out_v12),
                 ('Sem promo', out_sp)]:
    dados.append({
        'modelo': nome,
        'reward': round(r['reward'], 2),
        'reward_std': round(r['reward_std'], 2),
        'lucro': round(r['lucro'], 2),
        'd_lucro_pct': round((r['lucro'] - out_sp['lucro']) / out_sp['lucro'] * 100, 3),
        'perdas': round(r['perdas'], 1),
        'd_perdas_pct': round((r['perdas'] - out_sp['perdas']) / max(out_sp['perdas'], 1) * 100, 3),
    })
df_cmp = pd.DataFrame(dados)
df_cmp.to_csv(V13 / f'comparacao_v12_vs_v13_{args.suffix}.csv', index=False, encoding='utf-8')

print()
print("=" * 80)
print(df_cmp.to_string(index=False))
print("=" * 80)


# ── F1 por evento ─────────────────────────────────────

def f1_por_evento(df_dec, env):
    eventos_por_data = env._eventos_por_data
    nomes_eventos = sorted({ev['evento'] for evs in eventos_por_data.values() for ev in evs})
    rows = []
    for nome_ev in nomes_eventos:
        datas_janela = []
        cats_alvo = set()
        for d, evs in eventos_por_data.items():
            for ev in evs:
                if ev['evento'] == nome_ev:
                    datas_janela.append(d.isoformat())
                    cats_alvo.update(ev['categorias'])
        if not datas_janela:
            continue
        cats_alvo_idx = [env.nome_para_idx[c] for c in cats_alvo
                          if c in env.nome_para_idx]
        if not cats_alvo_idx:
            continue
        df_jan = df_dec[df_dec['data'].isin(datas_janela)]
        if len(df_jan) == 0:
            continue
        n_total = len(df_jan)
        n_acertos = int(df_jan['produto'].apply(
            lambda p: (p - 1) in cats_alvo_idx if p > 0 else False
        ).sum())
        n_promovidos = int((df_jan['produto'] > 0).sum())
        precision = n_acertos / n_promovidos if n_promovidos else 0
        recall = n_acertos / n_total if n_total else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        rows.append({
            'evento': nome_ev,
            'n_turnos': n_total,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
        })
    return pd.DataFrame(rows)


print(f"\n=== F1 POR EVENTO ===")
f1_v13 = f1_por_evento(out_v13['decisoes'], env).rename(columns={'f1': 'f1_v13'})
f1_v12 = f1_por_evento(out_v12['decisoes'], env).rename(columns={'f1': 'f1_v12'})

f1_merged = f1_v13[['evento', 'f1_v13']].merge(
    f1_v12[['evento', 'f1_v12']], on='evento', how='outer'
).fillna(0)
f1_merged['delta'] = f1_merged['f1_v13'] - f1_merged['f1_v12']
f1_merged = f1_merged.sort_values('f1_v13', ascending=False)
f1_merged.to_csv(V13 / f'validacao_eventos_v13_{args.suffix}.csv', index=False, encoding='utf-8')

print(f1_merged.to_string(index=False))

# F1 médio
f1_v13_medio = f1_merged['f1_v13'].mean()
f1_v12_medio = f1_merged['f1_v12'].mean()
print(f"\nF1 evento médio:")
print(f"  V12.1:        {f1_v12_medio:.3f}")
print(f"  V13.{args.suffix}: {f1_v13_medio:.3f}  ({(f1_v13_medio-f1_v12_medio)*100:+.1f}p.p.)")

# F1 em datas críticas
datas_chave = ['Dia das Mães', 'Dia dos Namorados', 'Dia Internacional da Mulher']
print(f"\nF1 em datas-chave (V13.{args.suffix} vs V12.1):")
for dc in datas_chave:
    row = f1_merged[f1_merged['evento'].str.contains(dc.split()[-1], regex=False, na=False)]
    if not row.empty:
        f1_13 = float(row['f1_v13'].iloc[0])
        f1_12 = float(row['f1_v12'].iloc[0])
        print(f"  {dc:30s}: V12.1={f1_12:.3f}  V13={f1_13:.3f}")

# ── Distribuição de ações ──────────────────────────────

print(f"\n=== TOP AÇÕES V13.{args.suffix} ===")
df_dec = out_v13['decisoes']
total = len(df_dec)
counts = df_dec.groupby(['produto', 'intensidade']).size().reset_index(name='n')
counts['pct'] = counts['n'] / total * 100
intens_n = ['nada', 'desc5%', 'desc10%', 'combo', 'liq25%']
counts['acao'] = counts.apply(
    lambda r: f"{env.cats[int(r['produto'])-1]['categoria']:>22s} + {intens_n[int(r['intensidade'])]}"
              if r['produto'] > 0 else f"{'_sem_promo':>22s} + {intens_n[int(r['intensidade'])]}",
    axis=1
)
top = counts.sort_values('pct', ascending=False).head(7)
for _, r in top.iterrows():
    print(f"  {r['acao']}  {r['pct']:5.1f}%")

# ── Métricas finais V13 ─────────────────────────────

metricas = {
    'modelo': f'V13.{args.suffix}',
    'data_validacao': date.today().isoformat(),
    'reward_medio': round(out_v13['reward'], 2),
    'lucro_medio': round(out_v13['lucro'], 2),
    'delta_lucro_pct': round((out_v13['lucro'] - out_sp['lucro']) / out_sp['lucro'] * 100, 3),
    'perdas_medio': round(out_v13['perdas'], 1),
    'delta_perdas_pct': round((out_v13['perdas'] - out_sp['perdas']) / max(out_sp['perdas'], 1) * 100, 3),
    'f1_evento_medio': round(f1_v13_medio, 3),
    'f1_evento_medio_v12_baseline': round(f1_v12_medio, 3),
    'f1_evento_delta_v12': round(f1_v13_medio - f1_v12_medio, 3),
    'reward_delta_v12_pct': round((out_v13['reward'] - out_v12['reward']) / abs(out_v12['reward']) * 100, 3),
    'pct_intensidades_usadas': round(
        len(counts[counts['produto'] > 0]['intensidade'].unique()) / 4 * 100, 1
    ),
    'n_categorias_promovidas': int(len(counts[counts['produto'] > 0]['produto'].unique())),
}

pd.DataFrame([metricas]).to_csv(
    V13 / f'validacao_metricas_v13_{args.suffix}.csv', index=False, encoding='utf-8'
)

# ── Sumário de aceitação ────────────────────────────

CRITERIOS = {
    'F1 evento médio ≥ 0.25': f1_v13_medio >= 0.25,
    'Lucro vs sem-promo ≥ +0.20%': metricas['delta_lucro_pct'] >= 0.20,
    'F1 Mães ≥ 0.30': float(f1_merged[f1_merged['evento'].str.contains('Mães', na=False)]['f1_v13'].iloc[0]
                              if not f1_merged[f1_merged['evento'].str.contains('Mães', na=False)].empty
                              else 0) >= 0.30,
    'F1 Namorados ≥ 0.30': float(f1_merged[f1_merged['evento'].str.contains('Namorados', na=False)]['f1_v13'].iloc[0]
                                   if not f1_merged[f1_merged['evento'].str.contains('Namorados', na=False)].empty
                                   else 0) >= 0.30,
}

print()
print("=" * 80)
print(f"CRITÉRIOS DE ACEITAÇÃO V13.{args.suffix}")
print("=" * 80)
for crit, ok in CRITERIOS.items():
    print(f"  {'✓' if ok else '✗'}  {crit}")

aprovado = sum(CRITERIOS.values())
print(f"\n  {aprovado}/{len(CRITERIOS)} critérios atingidos")
if aprovado == len(CRITERIOS):
    print(f"  🎉 V13.{args.suffix} APROVADO — modelo satisfatório!")
elif aprovado >= 3:
    print(f"  ⚠ V13.{args.suffix} parcialmente OK — vale iterar")
else:
    print(f"  ✗ V13.{args.suffix} insuficiente — ajustar hiperparâmetros")
print("=" * 80)
print(f"\n✓ Saídas em {V13}/")
