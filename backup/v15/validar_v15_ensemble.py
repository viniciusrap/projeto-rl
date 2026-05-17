"""V15 — Validação ensemble + comparação 3-way: V13 vs V14 vs V15.

V15 = meio-termo: V13 architecture + PER + ensemble (sem curriculum).
Hipótese: F1 Mães/Namorados similar V13 + variance baixa V14.
"""
import argparse
import io
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

from env_v12 import construir_env_v12
from dqn import BranchingDQN, NEG_INF

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V15 = ROOT / 'results' / 'v15'
V15.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 999, 2025, 7777])
parser.add_argument('--n_episodios', type=int, default=20)
parser.add_argument('--max_steps', type=int, default=1095)
parser.add_argument('--seed_eval', type=int, default=42)
parser.add_argument('--modelo_v13', type=str, default='results/v13/dqn_v13_final.pt')
args = parser.parse_args()


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    m = BranchingDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m


def ensemble_action(modelos, obs_t, mask_cat):
    q_total = None
    with torch.no_grad():
        for m in modelos:
            q = m.q_values(obs_t)
            if mask_cat is not None:
                mt = torch.tensor(mask_cat, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
                q = q.masked_fill(~mt, NEG_INF)
            q_total = q.clone() if q_total is None else q_total + q
    flat = q_total.view(-1)
    idx = int(flat.argmax().item())
    n_i = modelos[0].n_intensidades
    return idx // n_i, idx % n_i


def rollout(env, decide_fn, n_ep, max_steps, seed_base, no_repeat=0):
    rewards, lucros, perdas, rupturas = [], [], [], []
    decisoes = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed_base + ep * 100)
        ep_r = ep_l = ep_p = ep_rup = 0.0
        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = env.get_action_mask(no_repeat_window=no_repeat)
            ap, ai = decide_fn(obs_t, mask)
            obs2, r, term, trunc, info = env.step((ap, ai))
            ep_r += r; ep_l += info['lucro']
            ep_p += float(info['perdas'].sum())
            ep_rup += float(info['rupturas'].sum())
            decisoes.append({'ep': ep, 'data': info['data'], 'turno': info['turno'],
                             'produto': ap, 'intensidade': ai})
            obs = obs2
            if term or trunc: break
        rewards.append(ep_r); lucros.append(ep_l)
        perdas.append(ep_p); rupturas.append(ep_rup)
    return {
        'reward': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'lucro': float(np.mean(lucros)),
        'perdas': float(np.mean(perdas)),
        'rupturas': float(np.mean(rupturas)),
        'decisoes': pd.DataFrame(decisoes),
    }


def f1_por_evento(df_dec, env):
    eventos_por_data = env._eventos_por_data
    nomes = sorted({ev['evento'] for evs in eventos_por_data.values() for ev in evs})
    rows = []
    for nome in nomes:
        datas, cats_alvo = [], set()
        for d, evs in eventos_por_data.items():
            for ev in evs:
                if ev['evento'] == nome:
                    datas.append(d.isoformat())
                    cats_alvo.update(ev['categorias'])
        if not datas: continue
        idxs = [env.nome_para_idx[c] for c in cats_alvo if c in env.nome_para_idx]
        if not idxs: continue
        df_jan = df_dec[df_dec['data'].isin(datas)]
        if len(df_jan) == 0: continue
        n_total = len(df_jan)
        n_ac = int(df_jan['produto'].apply(lambda p: (p-1) in idxs if p > 0 else False).sum())
        n_pr = int((df_jan['produto'] > 0).sum())
        prec = n_ac / n_pr if n_pr else 0
        rec = n_ac / n_total if n_total else 0
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        rows.append({'evento': nome, 'f1': round(f1, 3)})
    return pd.DataFrame(rows)


print(f"Carregando V15 ({len(args.seeds)} seeds), V13, V14 ensemble…")
v15_models = []
for s in args.seeds:
    p = ROOT / f'results/v15/dqn_v15_seed{s}.pt'
    if p.exists():
        v15_models.append(load_model(p))
        print(f"  ✓ V15 seed {s}")
    else:
        print(f"  ✗ V15 seed {s} (faltando)")

v14_models = []
for s in args.seeds:
    p = ROOT / f'results/v14/dqn_v14_seed{s}.pt'
    if p.exists():
        v14_models.append(load_model(p))
        print(f"  ✓ V14 seed {s}")

modelo_v13 = load_model(ROOT / args.modelo_v13)
print(f"  ✓ V13 baseline")

env = construir_env_v12(modo='validacao')

print(f"\n=== AVALIAÇÃO 3-WAY ({args.n_episodios} ep × {args.max_steps} steps) ===")

# V15 ensemble
print(f"  V15 ensemble ({len(v15_models)} seeds, sem mask dinâmico)…")
def dec_v15(obs_t, mask): return ensemble_action(v15_models, obs_t, mask)
out_v15 = rollout(env, dec_v15, args.n_episodios, args.max_steps, args.seed_eval, no_repeat=0)

# V14 ensemble
print(f"  V14 ensemble ({len(v14_models)} seeds, sem mask dinâmico)…")
def dec_v14(obs_t, mask): return ensemble_action(v14_models, obs_t, mask)
out_v14 = rollout(env, dec_v14, args.n_episodios, args.max_steps, args.seed_eval, no_repeat=0)

# V13
print(f"  V13 baseline…")
def dec_v13(obs_t, mask):
    return modelo_v13.select_action(obs_t, eps=0.0,
                                      rng=np.random.default_rng(0), mask_cat=mask)
out_v13 = rollout(env, dec_v13, args.n_episodios, args.max_steps, args.seed_eval, no_repeat=0)

# V15 individuais (para medir variance)
print(f"  V15 individuais (variance entre seeds)…")
out_v15_ind = {}
for i, m in enumerate(v15_models):
    s = args.seeds[i]
    def dec_s(obs_t, mask, mm=m):
        return mm.select_action(obs_t, eps=0.0, rng=np.random.default_rng(0), mask_cat=mask)
    out_v15_ind[s] = rollout(env, dec_s, args.n_episodios, args.max_steps, args.seed_eval, no_repeat=0)

# Sem promo
def dec_sp(obs_t, mask): return (0, 0)
out_sp = rollout(env, dec_sp, args.n_episodios, args.max_steps, args.seed_eval, no_repeat=0)

# ── Tabela 3-way ─────────────────────────────────────────

rows = []
for nome, r in [('V15 ensemble', out_v15), ('V14 ensemble', out_v14),
                 ('V13', out_v13), ('Sem promo', out_sp)]:
    rows.append({
        'modelo': nome,
        'reward': round(r['reward'], 2),
        'reward_std': round(r['reward_std'], 2),
        'lucro': round(r['lucro'], 2),
        'd_lucro%': round((r['lucro'] - out_sp['lucro']) / out_sp['lucro'] * 100, 3),
        'perdas': round(r['perdas'], 1),
        'd_perdas%': round((r['perdas'] - out_sp['perdas']) / max(out_sp['perdas'], 1) * 100, 3),
    })
df = pd.DataFrame(rows)
df.to_csv(V15 / 'comparacao_v13_v14_v15.csv', index=False, encoding='utf-8')
print()
print("=" * 90)
print(df.to_string(index=False))
print("=" * 90)

# ── F1 por evento ──────────────────────────────────────

f1_v15 = f1_por_evento(out_v15['decisoes'], env).rename(columns={'f1': 'V15'})
f1_v14 = f1_por_evento(out_v14['decisoes'], env).rename(columns={'f1': 'V14'})
f1_v13 = f1_por_evento(out_v13['decisoes'], env).rename(columns={'f1': 'V13'})
f1m = f1_v15.merge(f1_v14, on='evento', how='outer').merge(f1_v13, on='evento', how='outer').fillna(0)
f1m['v15_vs_v13'] = f1m['V15'] - f1m['V13']
f1m = f1m.sort_values('V13', ascending=False)
f1m.to_csv(V15 / 'f1_eventos_3way.csv', index=False, encoding='utf-8')

print(f"\n=== F1 POR EVENTO (3-WAY) ===")
print(f1m.to_string(index=False))

print(f"\nF1 evento médio:")
print(f"  V13:           {f1m['V13'].mean():.3f}")
print(f"  V14 ensemble:  {f1m['V14'].mean():.3f}")
print(f"  V15 ensemble:  {f1m['V15'].mean():.3f}")

# Variance V15
print(f"\nVariance V15 entre seeds (lucro):")
lucros = [out_v15_ind[s]['lucro'] for s in args.seeds if s in out_v15_ind]
print(f"  Média:  R$ {np.mean(lucros):,.2f}")
print(f"  Std:    R$ {np.std(lucros):,.2f}")

# ── Critérios meio-termo ───────────────────────────────

CRIT = {
    f'V15 F1 evento ≥ V13 ({f1m["V13"].mean():.3f})':
        f1m['V15'].mean() >= f1m['V13'].mean(),
    f'V15 Mães ≥ 0.15':
        float(f1m[f1m['evento'].str.contains('Mães', na=False)]['V15'].iloc[0]
              if not f1m[f1m['evento'].str.contains('Mães', na=False)].empty else 0) >= 0.15,
    f'V15 Namorados ≥ 0.15':
        float(f1m[f1m['evento'].str.contains('Namorados', na=False)]['V15'].iloc[0]
              if not f1m[f1m['evento'].str.contains('Namorados', na=False)].empty else 0) >= 0.15,
    f'V15 variance < 5%':
        (np.std(lucros) / max(np.mean(lucros), 1)) < 0.05 if lucros else False,
    f'V15 lucro ≥ V14':
        out_v15['lucro'] >= out_v14['lucro'],
}

print(f"\n{'='*80}\nCRITÉRIOS MEIO-TERMO V15\n{'='*80}")
for c, ok in CRIT.items():
    print(f"  {'✓' if ok else '✗'} {c}")
total_ok = sum(CRIT.values())
print(f"\n  {total_ok}/{len(CRIT)} critérios atingidos")
if total_ok >= 4:
    print(f"  🎉 V15 É O MEIO-TERMO BUSCADO")
elif total_ok >= 3:
    print(f"  ⚠ V15 parcialmente OK")
else:
    print(f"  ✗ V15 insuficiente")
print("="*80)
print(f"\n✓ Saídas em {V15}/")
