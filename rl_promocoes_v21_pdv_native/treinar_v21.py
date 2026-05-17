"""Treino do agente V20.

300 episódios × N seeds. Cada episódio = 30 turnos (decisões de campanha).
Log de reward, loss, eps, distribuição de ações.
Save do melhor modelo (best avg reward últimos 30 eps).
"""
import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from env_rl_promocoes_v21 import EnvRLPromocoes
from branching_dqn import BranchingDQNAgent


def treinar(seed: int, n_episodios: int, log_path: Path, model_path: Path,
              verbose: bool = True):
    env = EnvRLPromocoes(seed=seed)
    state_dim = env.state_dim
    action_dims = list(env.action_space.nvec)

    agent = BranchingDQNAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=5e-4,
        gamma=0.95,
        eps_start=1.0,
        eps_end=0.10,           # ITER 4: 0.05→0.10 (mantém + exploração)
        eps_decay=0.997,        # ITER 4: 0.995→0.997 (mais lento)
        buffer_size=50_000,
        batch_size=64,
        target_update_steps=500,
        hidden=128,
        device='cpu',
    )

    # ITER 10: pré-treino mais intenso (300 ep, mais variedade de estados)
    loss_pre = agent.pretreinar_cabeca_complementar(env, n_epochs=300, lr=1e-3)
    if verbose:
        print(f"  Pre-treino cabeça COMPLEMENTAR: loss final = {loss_pre:.4f}")

    rewards_hist = []
    losses_hist = []
    best_avg = -float('inf')

    # Conta distribuição de ações por cabeça
    acoes_intensidade = np.zeros(action_dims[0], dtype=int)

    log_rows = []
    t0 = time.time()

    for ep in range(n_episodios):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        ep_reward = 0.0
        ep_loss_acc = []
        ep_acoes_intensidade = np.zeros(action_dims[0], dtype=int)
        ep_acoes_combo = 0
        ep_promoveu = 0
        ep_combo_invalido_pdv = 0

        for t in range(env.EPISODIO_TURNOS):
            # ITER 7: prior + mask na cabeça COMPLEMENTAR
            prior_comp = env.prior_complementar()
            mask_comp = env.mask_complementar_valida()
            action = agent.act(obs, prior_complementar=prior_comp,
                                mask_complementar=mask_comp)
            next_obs, reward, done, _, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                ep_loss_acc.append(loss)
            obs = next_obs
            ep_reward += reward
            # Stats da ação
            ep_acoes_intensidade[action[0]] += 1
            acoes_intensidade[action[0]] += 1
            if action[0] == 5:  # combo
                ep_acoes_combo += 1
            if action[0] > 0:  # promoveu
                ep_promoveu += 1
            if info.get('eh_combo_invalido_pdv'):
                ep_combo_invalido_pdv += 1
            if done:
                break

        agent.decay_eps()
        rewards_hist.append(ep_reward)
        ep_loss_mean = np.mean(ep_loss_acc) if ep_loss_acc else 0.0
        losses_hist.append(ep_loss_mean)

        # Avg recente
        last30 = rewards_hist[-30:]
        avg30 = np.mean(last30)

        # Save best
        if avg30 > best_avg and ep >= 30:
            best_avg = avg30
            agent.save(str(model_path))

        log_rows.append({
            'episodio': ep,
            'seed': seed,
            'reward': ep_reward,
            'loss': ep_loss_mean,
            'eps': agent.eps,
            'avg_reward_30ep': avg30,
            'promoveu_pct': ep_promoveu / env.EPISODIO_TURNOS,
            'combo_pct': ep_acoes_combo / env.EPISODIO_TURNOS,
            'combo_invalido_pdv': ep_combo_invalido_pdv,
            'acao_nada': ep_acoes_intensidade[0],
            'acao_desc3': ep_acoes_intensidade[1],
            'acao_desc5': ep_acoes_intensidade[2],
            'acao_desc7': ep_acoes_intensidade[3],
            'acao_desc10': ep_acoes_intensidade[4],
            'acao_combo': ep_acoes_intensidade[5],
        })

        if verbose and (ep + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[seed {seed}] Ep {ep+1:>3d}/{n_episodios} | "
                    f"R={ep_reward:>7.1f} avg30={avg30:>7.1f} | "
                    f"loss={ep_loss_mean:>6.2f} eps={agent.eps:.3f} | "
                    f"combo={ep_acoes_combo}/30 | pdv_inv={ep_combo_invalido_pdv} | "
                    f"{elapsed:.0f}s")

    # Salva log
    if log_rows:
        with open(log_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

    # Resumo final
    if verbose:
        print(f"\n[seed {seed}] TREINO CONCLUÍDO")
        print(f"  Tempo: {time.time()-t0:.1f}s")
        print(f"  Reward inicial (eps 1-10):     {np.mean(rewards_hist[:10]):>7.1f}")
        print(f"  Reward final (últimos 30):     {np.mean(rewards_hist[-30:]):>7.1f}")
        print(f"  Best avg30:                    {best_avg:>7.1f}")
        print(f"  Eps final:                     {agent.eps:.3f}")
        print(f"  Distribuição de ações (intensidade):")
        total_acoes = acoes_intensidade.sum()
        for i, label in enumerate(['nada', 'desc3%', 'desc5%', 'desc7%', 'desc10%', 'combo']):
            pct = acoes_intensidade[i] / total_acoes * 100 if total_acoes > 0 else 0
            print(f"    {label:<8s}: {acoes_intensidade[i]:>5d} ({pct:>5.1f}%)")

    return rewards_hist, agent


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--episodios', type=int, default=300)
    args = parser.parse_args()

    HERE = Path(__file__).parent
    LOG_DIR = HERE / 'logs'
    MODEL_DIR = HERE / 'models'
    LOG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 90)
    print(f"TREINO V20 — {args.seeds} seeds × {args.episodios} episódios")
    print("=" * 90)

    all_rewards = []
    for seed in range(args.seeds):
        print(f"\n{'─'*30} SEED {seed} {'─'*30}")
        rewards, _ = treinar(
            seed=seed,
            n_episodios=args.episodios,
            log_path=LOG_DIR / f'training_seed_{seed}.csv',
            model_path=MODEL_DIR / f'best_seed_{seed}.pt',
        )
        all_rewards.append(rewards)

    # Resumo cross-seeds
    arr = np.array(all_rewards)
    print(f"\n{'='*90}")
    print(f"RESUMO CROSS-SEEDS")
    print(f"{'='*90}")
    print(f"  Reward final médio (últimos 30 eps):")
    for i, r in enumerate(all_rewards):
        print(f"    Seed {i}: {np.mean(r[-30:]):>7.1f}")
    print(f"  Média entre seeds: {np.mean(arr[:, -30:]):>7.1f}")
    print(f"  CV (desvio/média): {np.std(arr[:, -30:]) / max(np.mean(arr[:, -30:]), 1e-6):.3f}")
