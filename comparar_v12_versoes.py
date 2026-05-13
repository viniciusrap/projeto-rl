"""Compara V12 base, V12.1 e V12.2 todos no MESMO env V12.2.

V12 base: forecast só
V12.1: + cigarros NPM + K_EVENTO_PRESENTE
V12.2: + harmonia categoria + harmonia evento + janela reduzida

Métricas para cada: reward, lucro, perdas, F1 timing, F1 evento médio,
F1 nas datas-presente (Mães, Namorados, Mulher, Pais, Crianças, Páscoa).

Output:
  results/v12/comparacao_versoes_v12.csv (sumário)
  results/v12/comparacao_versoes_f1_eventos.csv (F1 por evento × modelo)
"""
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env_v3 import construir_env_v3
from treinar_v11 import BranchingDQN

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V12 = ROOT / 'results' / 'v12'

N_EP = 20
MAX_STEPS = 1095
SEED = 42

MODELOS = {
    'V12 base': 'results/v12/dqn_v12_base.pt',
    'V12.1': 'results/v12/dqn_v12_1.pt',
    'V12.2': 'results/v12/dqn_v12_2.pt',
}

def load_dqn(path):
    ckpt = torch.load(ROOT / path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    m = BranchingDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m


def rollout(env, modelo, n_ep, max_steps, seed_base):
    rewards, lucros, perdas, rupturas = [], [], [], []
    decisoes = []
    for ep in range(n_ep):
        obs, info = env.reset(seed=seed_base + ep * 100)
        ep_r, ep_l, ep_p, ep_rup = 0.0, 0.0, 0.0, 0.0
        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            ap, ai = modelo.select_action(obs_t, eps=0.0,
                                            rng=np.random.default_rng(0))
            obs2, r, term, trunc, info = env.step((ap, ai))
            ep_r += r
            ep_l += info['lucro']
            ep_p += float(info['perdas'].sum())
            ep_rup += float(info['rupturas'].sum())
            decisoes.append({
                'ep': ep,
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
    return rewards, lucros, perdas, rupturas, pd.DataFrame(decisoes)


# ── Roda os 3 modelos ─────────────────────────────────────────────────────

env = construir_env_v3(modo='validacao')
print(f"\n=== AVALIAÇÃO ({N_EP} ep × {MAX_STEPS} steps) ===")
print(f"Env: V12.2 (cigarros NPM + K_EVENTO_PRESENTE + harmonia + janela)\n")

resultados = {}
for nome, path in MODELOS.items():
    print(f"  Rodando {nome}...")
    if not (ROOT / path).exists():
        print(f"    ⚠ {path} não existe, pulando")
        continue
    modelo = load_dqn(path)
    r, l, p, rup, df_dec = rollout(env, modelo, N_EP, MAX_STEPS, SEED)
    resultados[nome] = {
        'reward_medio': float(np.mean(r)),
        'reward_std': float(np.std(r)),
        'lucro_medio': float(np.mean(l)),
        'perdas_medio': float(np.mean(p)),
        'rupturas_medio': float(np.mean(rup)),
        'decisoes': df_dec,
    }

# ── Baseline sem promoção ────────────────────────────────────────────────

def pol_sem_promo_rollout(env, n_ep, max_steps, seed_base):
    rewards, lucros, perdas, rupturas = [], [], [], []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed_base + ep * 100)
        ep_r, ep_l, ep_p, ep_rup = 0.0, 0.0, 0.0, 0.0
        for step in range(max_steps):
            obs2, r, term, trunc, info = env.step((0, 0))
            ep_r += r
            ep_l += info['lucro']
            ep_p += float(info['perdas'].sum())
            ep_rup += float(info['rupturas'].sum())
            obs = obs2
            if term or trunc:
                break
        rewards.append(ep_r)
        lucros.append(ep_l)
        perdas.append(ep_p)
        rupturas.append(ep_rup)
    return rewards, lucros, perdas, rupturas

print(f"  Rodando Sem promoção...")
r, l, p, rup = pol_sem_promo_rollout(env, N_EP, MAX_STEPS, SEED)
resultados['Sem promoção'] = {
    'reward_medio': float(np.mean(r)),
    'reward_std': float(np.std(r)),
    'lucro_medio': float(np.mean(l)),
    'perdas_medio': float(np.mean(p)),
    'rupturas_medio': float(np.mean(rup)),
    'decisoes': None,
}

# ── Tabela de comparação ────────────────────────────────────────────────

linhas = []
lucro_sp = resultados['Sem promoção']['lucro_medio']
perdas_sp = resultados['Sem promoção']['perdas_medio']
for nome, r in resultados.items():
    linhas.append({
        'modelo': nome,
        'reward_medio': round(r['reward_medio'], 2),
        'reward_std': round(r['reward_std'], 2),
        'lucro_medio': round(r['lucro_medio'], 2),
        'delta_lucro_%': round((r['lucro_medio'] - lucro_sp) / lucro_sp * 100, 3),
        'perdas_medio': round(r['perdas_medio'], 2),
        'delta_perdas_%': round((r['perdas_medio'] - perdas_sp) / max(perdas_sp, 1) * 100, 3),
        'rupturas_medio': round(r['rupturas_medio'], 2),
    })

df_comp = pd.DataFrame(linhas)
df_comp.to_csv(V12 / 'comparacao_versoes_v12.csv', index=False, encoding='utf-8')

print()
print("=" * 100)
print("COMPARAÇÃO V12 base × V12.1 × V12.2 (todos em env V12.2)")
print("=" * 100)
print(df_comp.to_string(index=False))

# ── F1 por evento × modelo ───────────────────────────────────────────────

print()
print("=" * 100)
print("F1 POR EVENTO COMERCIAL (cada modelo)")
print("=" * 100)

eventos_por_data = env._eventos_por_data
nomes_eventos = set()
for evs in eventos_por_data.values():
    for ev in evs:
        nomes_eventos.add(ev['evento'])

f1_por_evento = []
for nome_modelo in ['V12 base', 'V12.1', 'V12.2']:
    if nome_modelo not in resultados or resultados[nome_modelo]['decisoes'] is None:
        continue
    df_dec = resultados[nome_modelo]['decisoes']
    for nome_ev in sorted(nomes_eventos):
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
        f1_por_evento.append({
            'modelo': nome_modelo,
            'evento': nome_ev,
            'n_turnos': n_total,
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
        })

df_f1 = pd.DataFrame(f1_por_evento)
# Pivot para comparação lado a lado
piv = df_f1.pivot(index='evento', columns='modelo', values='f1').fillna(0)
piv = piv[['V12 base', 'V12.1', 'V12.2']]  # ordem
piv['delta_v12_2_vs_base'] = piv['V12.2'] - piv['V12 base']
piv = piv.sort_values('V12.2', ascending=False)
piv.to_csv(V12 / 'comparacao_versoes_f1_eventos.csv', encoding='utf-8')

print(piv.to_string())

# F1 médio
print()
print("F1 evento MÉDIO por modelo:")
for col in ['V12 base', 'V12.1', 'V12.2']:
    print(f"  {col:10s}: {piv[col].mean():.3f}")

# ── Distribuição de ações dos 3 ──────────────────────────────────────────

print()
print("=" * 100)
print("TOP 5 AÇÕES (categoria, intensidade) POR MODELO")
print("=" * 100)

for nome_modelo in ['V12 base', 'V12.1', 'V12.2']:
    if nome_modelo not in resultados or resultados[nome_modelo]['decisoes'] is None:
        continue
    df_dec = resultados[nome_modelo]['decisoes']
    total = len(df_dec)
    counts = df_dec.groupby(['produto', 'intensidade']).size().reset_index(name='n')
    counts['pct'] = counts['n'] / total * 100
    counts = counts.sort_values('pct', ascending=False).head(5)
    print(f"\n{nome_modelo}:")
    intens_n = ['nada', 'desc5%', 'desc10%', 'combo', 'liq25%']
    for _, r in counts.iterrows():
        cat = env.cats[int(r['produto']) - 1]['categoria'] if r['produto'] > 0 else '_sem_promo'
        print(f"  {cat:20s} {intens_n[int(r['intensidade'])]:8s}  {r['pct']:5.1f}%")

print()
print(f"✓ Saídas em {V12}/")
print(f"  comparacao_versoes_v12.csv")
print(f"  comparacao_versoes_f1_eventos.csv")
