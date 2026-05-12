"""Treina Double DQN no ConvenienceStoreEnvV2 (V11).

Diferenças vs V10:
- Estado: 122 features (era 47)
- Ação: MultiDiscrete [N+1, 5] = 95 ações para N=18 (era 5)
- Episódio: 1095 turnos = 1 ano calendário (era 90 turnos abstratos)

Arquitetura da rede:
- Shared trunk: Linear(122 → 256 → ReLU → 128 → ReLU)
- 2 cabeças decompostas:
    head_produto:    Linear(128 → 64 → ReLU → N+1)
    head_intensidade: Linear(128 → 64 → ReLU → 5)
- Q(s,a) = Q_prod(s)[a_prod] + Q_int(s)[a_int]
  (decomposição aditiva tipo Branching DQN — reduz espaço de saída de
   N×5=95 para N+5=24 outputs)

Treino:
- 100 episódios × 3 seeds (reduzido vs V10 — episódio 12× mais longo)
- Double DQN, HuberLoss, ε-decay agressivo
- Salva: results/v11/dqn_v11.pt + training_log_v11.csv

Uso: python treinar_v11.py [--episodios N] [--seeds N] [--device cpu|cuda]
"""
import argparse
import io
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from env_v2 import construir_env_v2

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results' / 'v11'
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Modelo: Branching DQN ───────────────────────────────────────────────────

class BranchingDQN(nn.Module):
    """DQN com cabeças decompostas para ação MultiDiscrete.

    Q(s,(p,i)) = V(s) + A_prod(s,p) + A_int(s,i)
    (dueling-style decomposition; reduz parâmetros vs Q(s,a) flat)
    """

    def __init__(self, obs_dim: int, n_produtos: int, n_intensidades: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.adv_prod = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                        nn.Linear(64, n_produtos))
        self.adv_int = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                       nn.Linear(64, n_intensidades))
        self.n_produtos = n_produtos
        self.n_intensidades = n_intensidades

    def forward(self, x):
        h = self.trunk(x)
        v = self.value(h)
        ap = self.adv_prod(h) - self.adv_prod(h).mean(dim=-1, keepdim=True)
        ai = self.adv_int(h) - self.adv_int(h).mean(dim=-1, keepdim=True)
        return v, ap, ai

    def q_values(self, x):
        """Q(s, p, i) tensor de shape (B, N_p, N_i)."""
        v, ap, ai = self.forward(x)
        # broadcast: Q[b, p, i] = V[b] + A_p[b, p] + A_i[b, i]
        return v.unsqueeze(-1) + ap.unsqueeze(-1) + ai.unsqueeze(-2)

    def select_action(self, x, eps: float, rng):
        """ε-greedy: (produto, intensidade)."""
        if rng.random() < eps:
            return (int(rng.integers(0, self.n_produtos)),
                    int(rng.integers(0, self.n_intensidades)))
        with torch.no_grad():
            q = self.q_values(x)  # (1, N_p, N_i)
            flat = q.view(-1)
            idx = int(flat.argmax().item())
            p = idx // self.n_intensidades
            i = idx % self.n_intensidades
            return p, i


# ── Replay buffer ──────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, ap, ai, r, s2, done):
        self.buf.append((s, ap, ai, r, s2, done))

    def sample(self, batch_size: int, rng):
        idxs = rng.integers(0, len(self.buf), size=batch_size)
        batch = [self.buf[i] for i in idxs]
        s = np.array([b[0] for b in batch], dtype=np.float32)
        ap = np.array([b[1] for b in batch], dtype=np.int64)
        ai = np.array([b[2] for b in batch], dtype=np.int64)
        r = np.array([b[3] for b in batch], dtype=np.float32)
        s2 = np.array([b[4] for b in batch], dtype=np.float32)
        d = np.array([b[5] for b in batch], dtype=np.float32)
        return s, ap, ai, r, s2, d

    def __len__(self):
        return len(self.buf)


# ── Treino de 1 seed ────────────────────────────────────────────────────────

def treinar_seed(seed: int, n_episodios: int, max_steps: int, DEVICE=None):
    if DEVICE is None:
        DEVICE = torch.device('cpu')
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = construir_env_v2(modo='treino')
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
    eps_decay = 0.99  # ~100 eps para chegar em ε=0.05
    gamma = 0.99
    batch_size = 64
    target_update_every = 5  # episódios

    log = []
    t0 = time.time()
    for ep in range(n_episodios):
        obs, _ = env.reset(seed=seed + ep * 1000)
        ep_reward = 0.0
        ep_lucro = 0.0
        ep_perdas = 0.0
        ep_rupturas = 0.0
        losses = []
        action_counts = np.zeros((n_p, n_i), dtype=np.int32)

        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            ap, ai = online.select_action(obs_t, eps, rng)
            obs2, reward, term, trunc, info = env.step((ap, ai))
            buf.push(obs, ap, ai, reward, obs2, term or trunc)
            obs = obs2
            ep_reward += reward
            ep_lucro += info['lucro']
            ep_perdas += float(info['perdas'].sum())
            ep_rupturas += float(info['rupturas'].sum())
            action_counts[ap, ai] += 1

            # Optimize
            if len(buf) >= 1000:
                s, _ap, _ai, r, s2, d = buf.sample(batch_size, rng)
                s_t = torch.tensor(s, device=DEVICE)
                ap_t = torch.tensor(_ap, device=DEVICE)
                ai_t = torch.tensor(_ai, device=DEVICE)
                r_t = torch.tensor(r, device=DEVICE)
                s2_t = torch.tensor(s2, device=DEVICE)
                d_t = torch.tensor(d, device=DEVICE)

                # Q atual
                q = online.q_values(s_t)
                q_sel = q[torch.arange(batch_size), ap_t, ai_t]

                # Double DQN: online seleciona melhor ação em s2, target avalia
                with torch.no_grad():
                    q2_online = online.q_values(s2_t)
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

        # Update target
        if ep % target_update_every == 0:
            target.load_state_dict(online.state_dict())

        # Logging
        eps = max(eps_min, eps * eps_decay)
        n_steps = step + 1
        promo_pct = action_counts[1:, :].sum() / n_steps * 100  # % de turnos com promoção
        log.append({
            'seed': seed,
            'episodio': ep,
            'steps': n_steps,
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
    parser.add_argument('--episodios', type=int, default=100)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--max_steps_per_ep', type=int, default=1095)
    parser.add_argument('--quick', action='store_true',
                         help='Modo rápido: 10 episódios, 1 seed, 200 steps/ep')
    args = parser.parse_args()

    if args.quick:
        args.episodios = 10
        args.seeds = 1
        args.max_steps_per_ep = 200

    DEVICE = torch.device(args.device)
    print(f"Device: {DEVICE}")
    print(f"Configuração: {args.episodios} episódios × {args.seeds} seeds, "
          f"{args.max_steps_per_ep} steps/ep")

    todos_logs = []
    modelo_final = None
    for seed_offset in range(args.seeds):
        seed = 42 + seed_offset * 1000
        print(f"\n=== SEED {seed} ===")
        modelo_final, log = treinar_seed(seed, args.episodios,
                                            args.max_steps_per_ep, DEVICE)
        todos_logs.extend(log)

    df_log = pd.DataFrame(todos_logs)
    df_log.to_csv(RESULTS / 'training_log_v11.csv', index=False, encoding='utf-8')

    torch.save({
        'state_dict': modelo_final.state_dict(),
        'config': {
            'obs_dim': modelo_final.trunk[0].in_features,
            'n_produtos': modelo_final.n_produtos,
            'n_intensidades': modelo_final.n_intensidades,
        },
    }, RESULTS / 'dqn_v11.pt')

    print(f"\n✓ Treino concluído")
    print(f"  results/v11/training_log_v11.csv  ({len(df_log)} linhas)")
    print(f"  results/v11/dqn_v11.pt")

    ultimo = df_log[df_log['episodio'] >= max(0, args.episodios - 10)]
    print(f"\nMédia dos últimos 10 episódios (entre seeds):")
    print(f"  reward:    R$ {ultimo['reward_total'].mean():>10,.0f}")
    print(f"  lucro:     R$ {ultimo['lucro_total'].mean():>10,.0f}")
    print(f"  perdas:    {ultimo['perdas_total'].mean():>10.1f} un")
    print(f"  rupturas:  {ultimo['rupturas_total'].mean():>10.1f} un")
    print(f"  promove:   {ultimo['pct_promove'].mean():>10.1f} %")


if __name__ == '__main__':
    main()
