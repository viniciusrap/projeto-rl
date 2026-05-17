"""V14 — Validação ensemble de múltiplos seeds.

Carrega N modelos V14 (1 por seed), faz inferência por:
- voto majoritário (moda da ação proposta por cada modelo)
- Q-value médio (soma Q values, argmax do resultado)

Compara com V13 (baseline). Critério: V14 ensemble ≥ V13 em todas as
métricas + variance menor.

Uso: python validar_v14_ensemble.py --seeds 42 123 999 2025 7777
"""
import argparse
import io
import sys
from datetime import date
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch

from env_v12 import construir_env_v12
from dqn import BranchingDQN

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V14 = ROOT / 'results' / 'v14'
V14.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 999, 2025, 7777])
parser.add_argument('--n_episodios', type=int, default=15)
parser.add_argument('--max_steps', type=int, default=1095)
parser.add_argument('--seed_eval', type=int, default=42)
parser.add_argument('--mode', type=str, default='qmean',
                     choices=['vote', 'qmean'])
parser.add_argument('--modelo_v13', type=str, default='results/v13/dqn_v13_final.pt')
args = parser.parse_args()


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    c = ckpt['config']
    m = BranchingDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m


def ensemble_action(modelos, obs_t, mask_cat, mode='qmean'):
    """Decide ação por ensemble.

    mode='vote': cada modelo vota (ap, ai); retorna moda.
    mode='qmean': soma Q-values, argmax do agregado.
    """
    if mode == 'vote':
        votes = []
        for m in modelos:
            ap, ai = m.select_action(obs_t, eps=0.0,
                                      rng=np.random.default_rng(0),
                                      mask_cat=mask_cat)
            votes.append((ap, ai))
        return Counter(votes).most_common(1)[0][0]
    else:  # qmean
        from dqn import NEG_INF
        q_total = None
        with torch.no_grad():
            for m in modelos:
                q = m.q_values(obs_t)
                if mask_cat is not None:
                    mt = torch.tensor(mask_cat, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
                    q = q.masked_fill(~mt, NEG_INF)
                if q_total is None:
                    q_total = q.clone()
                else:
                    q_total = q_total + q
            flat = q_total.view(-1)
            idx = int(flat.argmax().item())
            n_i = modelos[0].n_intensidades
            return idx // n_i, idx % n_i


def rollout(env, decide_fn, n_ep, max_steps, seed_base, no_repeat_window=2):
    rewards, lucros, perdas, rupturas = [], [], [], []
    decisoes = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed_base + ep * 100)
        ep_r = ep_l = ep_p = ep_rup = 0.0
        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask = env.get_action_mask(no_repeat_window=no_repeat_window)
            ap, ai = decide_fn(obs_t, mask)
            obs2, r, term, trunc, info = env.step((ap, ai))
            ep_r += r
            ep_l += info['lucro']
            ep_p += float(info['perdas'].sum())
            ep_rup += float(info['rupturas'].sum())
            decisoes.append({
                'ep': ep, 'data': info['data'], 'turno': info['turno'],
                'produto': ap, 'intensidade': ai,
            })
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
        datas = []
        cats_alvo = set()
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
        rows.append({'evento': nome, 'n_turnos': n_total,
                     'precision': round(prec, 3), 'recall': round(rec, 3),
                     'f1': round(f1, 3)})
    return pd.DataFrame(rows)


# ── Carrega modelos ───────────────────────────────────────────────

print(f"Carregando {len(args.seeds)} modelos V14 + V13 baseline…")
modelos = []
seeds_carregados = []
for s in args.seeds:
    p = ROOT / f'results/v14/dqn_v14_seed{s}.pt'
    if p.exists():
        modelos.append(load_model(p))
        seeds_carregados.append(s)
        print(f"  ✓ seed {s}")
    else:
        print(f"  ✗ seed {s} (arquivo ausente — pulando)")
if not modelos:
    print("ERRO: nenhum modelo V14 encontrado.")
    sys.exit(1)

modelo_v13 = load_model(ROOT / args.modelo_v13)
print(f"  ✓ V13 (baseline)")

env = construir_env_v12(modo='validacao')

# ── Rollouts ─────────────────────────────────────────────────────

print(f"\n=== AVALIAÇÃO ({args.n_episodios} ep × {args.max_steps} steps) ===")

# V14 ensemble
print(f"  V14 ensemble ({args.mode}, {len(modelos)} seeds)…")
def dec_ensemble(obs_t, mask):
    return ensemble_action(modelos, obs_t, mask, mode=args.mode)
out_v14 = rollout(env, dec_ensemble, args.n_episodios, args.max_steps, args.seed_eval)

# V14 individual seeds
out_individual = {}
for i, m in enumerate(modelos):
    s = seeds_carregados[i]
    print(f"  V14 seed {s} (individual)…")
    def dec_single(obs_t, mask, mm=m):
        return mm.select_action(obs_t, eps=0.0,
                                  rng=np.random.default_rng(0), mask_cat=mask)
    out_individual[s] = rollout(env, dec_single, args.n_episodios, args.max_steps, args.seed_eval)

# V13 baseline
print(f"  V13 baseline…")
def dec_v13(obs_t, mask):
    return modelo_v13.select_action(obs_t, eps=0.0,
                                      rng=np.random.default_rng(0), mask_cat=mask)
out_v13 = rollout(env, dec_v13, args.n_episodios, args.max_steps, args.seed_eval)

# Sem promo
print(f"  Sem promoção…")
def dec_sp(obs_t, mask): return (0, 0)
out_sp = rollout(env, dec_sp, args.n_episodios, args.max_steps, args.seed_eval)

# ── Tabela ──────────────────────────────────────────────────────

rows = []
for nome, r in [(f'V14 ensemble ({args.mode})', out_v14), ('V13', out_v13), ('Sem promo', out_sp)]:
    rows.append({
        'modelo': nome,
        'reward': round(r['reward'], 2),
        'reward_std': round(r['reward_std'], 2),
        'lucro': round(r['lucro'], 2),
        'd_lucro%': round((r['lucro'] - out_sp['lucro']) / out_sp['lucro'] * 100, 3),
        'perdas': round(r['perdas'], 1),
        'd_perdas%': round((r['perdas'] - out_sp['perdas']) / max(out_sp['perdas'], 1) * 100, 3),
    })
# Adiciona individuais (mostra variance)
for s, r in out_individual.items():
    rows.append({
        'modelo': f'V14 seed {s}',
        'reward': round(r['reward'], 2),
        'reward_std': round(r['reward_std'], 2),
        'lucro': round(r['lucro'], 2),
        'd_lucro%': round((r['lucro'] - out_sp['lucro']) / out_sp['lucro'] * 100, 3),
        'perdas': round(r['perdas'], 1),
        'd_perdas%': round((r['perdas'] - out_sp['perdas']) / max(out_sp['perdas'], 1) * 100, 3),
    })

df = pd.DataFrame(rows)
df.to_csv(V14 / f'comparacao_v14_ensemble.csv', index=False, encoding='utf-8')
print()
print("="*90)
print(df.to_string(index=False))
print("="*90)

# ── F1 por evento ─────────────────────────────────────────────

print(f"\n=== F1 POR EVENTO (V14 ensemble vs V13) ===")
f1_v14 = f1_por_evento(out_v14['decisoes'], env).rename(columns={'f1': 'f1_v14'})
f1_v13 = f1_por_evento(out_v13['decisoes'], env).rename(columns={'f1': 'f1_v13'})
f1m = f1_v14[['evento', 'f1_v14']].merge(f1_v13[['evento', 'f1_v13']], on='evento', how='outer').fillna(0)
f1m['delta'] = f1m['f1_v14'] - f1m['f1_v13']
f1m = f1m.sort_values('f1_v14', ascending=False)
f1m.to_csv(V14 / 'f1_eventos_ensemble.csv', index=False, encoding='utf-8')
print(f1m.to_string(index=False))

f1_v14_med = f1m['f1_v14'].mean()
f1_v13_med = f1m['f1_v13'].mean()
print(f"\nF1 evento médio:")
print(f"  V13:           {f1_v13_med:.3f}")
print(f"  V14 ensemble:  {f1_v14_med:.3f}  ({(f1_v14_med-f1_v13_med)*100:+.1f}p.p.)")

# Variance entre seeds
print(f"\nVariance entre seeds (lucro):")
lucros_seeds = [out_individual[s]['lucro'] for s in seeds_carregados]
print(f"  Média:  R$ {np.mean(lucros_seeds):,.2f}")
print(f"  Std:    R$ {np.std(lucros_seeds):,.2f}")
print(f"  Min:    R$ {np.min(lucros_seeds):,.2f}")
print(f"  Max:    R$ {np.max(lucros_seeds):,.2f}")

# ── Critérios ────────────────────────────────────────────────

CRIT = {
    f'V14 ensemble F1 evento ≥ V13 ({f1_v13_med:.3f})':
        f1_v14_med >= f1_v13_med,
    f'V14 ensemble lucro ≥ V13 (R$ {out_v13["lucro"]:,.0f})':
        out_v14['lucro'] >= out_v13['lucro'],
    f'V14 ensemble perdas ≤ V13 ({out_v13["perdas"]:.0f})':
        out_v14['perdas'] <= out_v13['perdas'],
    'Variance lucro entre seeds < 5%':
        np.std(lucros_seeds) / max(np.mean(lucros_seeds), 1) < 0.05,
}

print(f"\n{'='*80}\nCRITÉRIOS V14 ENSEMBLE\n{'='*80}")
for c, ok in CRIT.items():
    print(f"  {'✓' if ok else '✗'} {c}")
total_ok = sum(CRIT.values())
print(f"\n  {total_ok}/{len(CRIT)} critérios atingidos")
if total_ok == len(CRIT):
    print(f"  🎉 V14 ENSEMBLE APROVADO")
elif total_ok >= 2:
    print(f"  ⚠ V14 parcialmente OK")
else:
    print(f"  ✗ V14 insuficiente")
print("="*80)
print(f"\n✓ Saídas em {V14}/")
