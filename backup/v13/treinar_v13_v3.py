"""V13.v3 — BranchingDQN PURO + Action Mask + K_EVENTO_PRESENTE=1000.

Estratégia mais conservadora: replica V12.1 architecture, adiciona apenas:
1. Action mask para cigarros (hard)
2. K_EVENTO_PRESENTE aumentado 600 → 1000 (via calibrar_v2.py)

Diagnóstico V13.v1/v2: arquiteturas novas (CondPolicy, BranchingCross) destravam
diversidade mas perdem foco em datas-presente porque K=600 não compete com
novas opções economicamente atraentes. V13.v3 mantém arquitetura comprovada
e apenas reforça o sinal de evento.
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(8)
torch.set_num_interop_threads(4)

from env_v12 import construir_env_v12
from dqn import BranchingDQN, ReplayBuffer, NEG_INF

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results' / 'v13'
RESULTS.mkdir(parents=True, exist_ok=True)


def treinar_seed(seed, n_episodios, max_steps, DEVICE=None):
    if DEVICE is None:
        DEVICE = torch.device('cpu')
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = construir_env_v12(modo='treino')
    obs_dim = env.observation_space.shape[0]
    n_p = env.action_space.nvec[0]
    n_i = env.action_space.nvec[1]

    online = BranchingDQN(obs_dim, n_p, n_i).to(DEVICE)
    target = BranchingDQN(obs_dim, n_p, n_i).to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()

    opt = torch.optim.Adam(online.parameters(), lr=3e-4)
    loss_fn = nn.SmoothL1Loss()
    buf = ReplayBuffer(50_000)

    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.985
    gamma = 0.99
    batch_size = 64
    target_update_every = 5

    log = []
    t0 = time.time()
    mask_estatica = env.get_action_mask()
    mask_t_batch = torch.tensor(mask_estatica, dtype=torch.bool, device=DEVICE)

    for ep in range(n_episodios):
        obs, _ = env.reset(seed=seed + ep * 1000)
        ep_reward = ep_lucro = ep_perdas = ep_rupturas = 0.0
        losses = []
        action_counts = np.zeros((n_p, n_i), dtype=np.int32)

        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask = env.get_action_mask()
            ap, ai = online.select_action(obs_t, eps, rng, mask_cat=mask)
            obs2, reward, term, trunc, info = env.step((ap, ai))
            buf.push(obs, ap, ai, reward, obs2, term or trunc)
            obs = obs2
            ep_reward += reward
            ep_lucro += info['lucro']
            ep_perdas += float(info['perdas'].sum())
            ep_rupturas += float(info['rupturas'].sum())
            action_counts[ap, ai] += 1

            if len(buf) >= 1000:
                s, _ap, _ai, r, s2, d = buf.sample(batch_size, rng)
                s_t = torch.tensor(s, device=DEVICE)
                ap_t = torch.tensor(_ap, device=DEVICE)
                ai_t = torch.tensor(_ai, device=DEVICE)
                r_t = torch.tensor(r, device=DEVICE)
                s2_t = torch.tensor(s2, device=DEVICE)
                d_t = torch.tensor(d, device=DEVICE)

                q = online.q_values(s_t)
                q_sel = q[torch.arange(batch_size), ap_t, ai_t]

                with torch.no_grad():
                    q2_online = online.q_values(s2_t)
                    q2_online = q2_online.masked_fill(
                        ~mask_t_batch.unsqueeze(0).unsqueeze(-1), NEG_INF
                    )
                    flat = q2_online.view(batch_size, -1)
                    best_idx = flat.argmax(dim=-1)
                    best_ap = best_idx // n_i
                    best_ai = best_idx % n_i
                    q2_target = target.q_values(s2_t)
                    q2_val = q2_target[torch.arange(batch_size), best_ap, best_ai]
                    y = r_t + gamma * q2_val * (1 - d_t)

                loss = loss_fn(q_sel, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.item()))

            if term or trunc:
                break

        if ep % target_update_every == 0:
            target.load_state_dict(online.state_dict())

        eps = max(eps_min, eps * eps_decay)
        n_steps = step + 1
        promo_pct = action_counts[1:, :].sum() / n_steps * 100
        log.append({
            'seed': seed, 'episodio': ep, 'steps': n_steps,
            'reward_total': round(ep_reward, 2),
            'lucro_total': round(ep_lucro, 2),
            'perdas_total': round(ep_perdas, 1),
            'rupturas_total': round(ep_rupturas, 1),
            'epsilon': round(eps, 4),
            'loss_media': round(np.mean(losses) if losses else 0, 4),
            'pct_promove': round(promo_pct, 1),
        })
        if ep % 10 == 0 or ep == n_episodios - 1:
            elapsed = time.time() - t0
            print(f"  seed {seed} ep {ep:>3d}  reward={ep_reward:>10,.0f}  "
                  f"lucro={ep_lucro:>10,.0f}  perdas={ep_perdas:>5.0f}  "
                  f"promo={promo_pct:>4.1f}%  ε={eps:.3f}  ({elapsed:.0f}s)")
    return online, log


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodios', type=int, default=150)
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--max_steps_per_ep', type=int, default=1095)
    parser.add_argument('--suffix', type=str, default='v3')
    args = parser.parse_args()

    print(f"V13.{args.suffix} — BranchingDQN + Mask + K_EVENTO_PRESENTE=1000")
    print(f"Config: {args.episodios} ep × {args.seeds} seeds, {args.max_steps_per_ep} steps/ep")

    todos_logs = []
    modelo_final = None
    for so in range(args.seeds):
        seed = 42 + so * 1000
        print(f"\n=== SEED {seed} ===")
        modelo_final, log = treinar_seed(seed, args.episodios, args.max_steps_per_ep)
        todos_logs.extend(log)

    suf = args.suffix
    df_log = pd.DataFrame(todos_logs)
    df_log.to_csv(RESULTS / f'training_log_v13_{suf}.csv', index=False, encoding='utf-8')

    torch.save({
        'state_dict': modelo_final.state_dict(),
        'config': {
            'obs_dim': modelo_final.trunk[0].in_features,
            'n_produtos': modelo_final.n_produtos,
            'n_intensidades': modelo_final.n_intensidades,
            'arch': 'BranchingDQN',
        },
    }, RESULTS / f'dqn_v13_{suf}.pt')

    print(f"\n✓ Treino V13.{suf} concluído")
    ultimo = df_log[df_log['episodio'] >= max(0, args.episodios - 10)]
    print(f"\nMédia últimos 10 ep:")
    print(f"  reward:  R$ {ultimo['reward_total'].mean():>10,.0f}")
    print(f"  lucro:   R$ {ultimo['lucro_total'].mean():>10,.0f}")
    print(f"  perdas:  {ultimo['perdas_total'].mean():>10.1f} un")
    print(f"  promove: {ultimo['pct_promove'].mean():>10.1f} %")


if __name__ == '__main__':
    main()
