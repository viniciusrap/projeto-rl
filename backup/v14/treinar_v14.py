"""V14 — V13 + 5 melhorias incrementais.

1. **PrioritizedReplayBuffer (PER)** — Schaul 2016. Sample por |TD-error|.
   Transições raras (eventos comerciais) são vistas mais → F1 evento sobe.

2. **Action mask dinâmico** — não promover MESMA categoria nos últimos 2
   turnos efetivos. Força diversidade sem custo de treino.

3. **DRCR Reward** (opcional, --drcr) — Alibaba 2019. Substitui lucro
   absoluto por delta relativo. Reduz colapso em produtos de alto volume.

4. **Curriculum Learning** — peso dos termos do reward varia por fase:
   - Fase 1 (0-33%): só lucro/perdas/ruptura (básico)
   - Fase 2 (33-66%): + bonus_timing + bonus_evento (timing)
   - Fase 3 (66-100%): + harmonia + regras de oferta/procura (refino)

5. **Multi-seed ensemble** — script orquestrador roda 5 seeds; inferência
   vota por moda. Reduz variance estocástica.

Bellman update suporta IS weights do PER.
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
from dqn import BranchingDQN, PrioritizedReplayBuffer, NEG_INF
from drcr_wrapper import DRCRWrapper

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results' / 'v14'
RESULTS.mkdir(parents=True, exist_ok=True)


def get_curriculum_weights(ep: int, total_ep: int) -> dict:
    """V14 — pesos por fase do curriculum.

    Fase 1 (0-33%): foco em fundamentos (lucro, ruptura, vencimento, giro).
    Fase 2 (33-66%): adiciona timing e eventos comerciais.
    Fase 3 (66-100%): tudo ativo, incluindo harmonia e regras finas.
    """
    progress = ep / total_ep
    if progress < 0.33:
        return {
            'timing': 0.3,
            'evento': 0.0,
            'padrao': 0.0,
            'desc_alta_saudavel': 0.5,
            'combo_alta': 0.0,
            'combo_data_pico': 0.0,
            'desc_vencimento': 1.0,
            'desc_baixa': 0.5,
            'instabilidade': 0.5,
            'dia_semana_categoria': 0.5,
        }
    elif progress < 0.66:
        return {
            'timing': 1.0,
            'evento': 1.0,
            'padrao': 0.7,
            'desc_alta_saudavel': 1.0,
            'combo_alta': 0.5,
            'combo_data_pico': 0.5,
            'desc_vencimento': 1.0,
            'desc_baixa': 1.0,
            'instabilidade': 1.0,
            'dia_semana_categoria': 1.0,
        }
    else:
        return {
            'timing': 1.0,
            'evento': 1.0,
            'padrao': 1.0,
            'desc_alta_saudavel': 1.0,
            'combo_alta': 1.0,
            'combo_data_pico': 1.0,
            'desc_vencimento': 1.0,
            'desc_baixa': 1.0,
            'instabilidade': 1.0,
            'dia_semana_categoria': 1.0,
        }


def reward_curriculum(info: dict, weights: dict) -> float:
    """Recalcula reward ponderado pelo curriculum, a partir do info do env."""
    base = (info['lucro']
            - info['pen_venc']
            - info['pen_ruptura']
            - info['pen_desconto']
            + info['bonus_giro']
            - info['pen_nao_promovivel'])
    bonus = (weights['timing'] * info['bonus_timing']
             + weights['evento'] * info['bonus_evento']
             + weights['padrao'] * info['bonus_padrao']
             - weights['instabilidade'] * info['pen_instabilidade']
             - weights['desc_alta_saudavel'] * info['pen_desc_alta_saudavel']
             + weights['combo_alta'] * info['bonus_combo_alta']
             + weights['combo_data_pico'] * info['bonus_combo_data_pico']
             + weights['desc_vencimento'] * info['bonus_desc_vencimento']
             + weights['desc_baixa'] * info['bonus_desc_baixa']
             + weights['dia_semana_categoria'] * info['bonus_dia_semana_categoria'])
    return base + bonus


def treinar_seed(seed, n_episodios, max_steps, use_drcr=False, use_curriculum=True,
                  no_repeat_window=2, DEVICE=None):
    if DEVICE is None:
        DEVICE = torch.device('cpu')
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env_base = construir_env_v12(modo='treino')
    env = DRCRWrapper(env_base) if use_drcr else env_base

    obs_dim = env_base.observation_space.shape[0]
    n_p = env_base.action_space.nvec[0]
    n_i = env_base.action_space.nvec[1]

    online = BranchingDQN(obs_dim, n_p, n_i).to(DEVICE)
    target = BranchingDQN(obs_dim, n_p, n_i).to(DEVICE)
    target.load_state_dict(online.state_dict())
    target.eval()

    opt = torch.optim.Adam(online.parameters(), lr=3e-4)
    # V14: PER em vez de buffer uniforme
    buf = PrioritizedReplayBuffer(50_000, alpha=0.6, beta_start=0.4,
                                    beta_increment=2e-5)

    eps = 1.0
    eps_min = 0.05
    eps_decay = 0.985
    gamma = 0.99
    batch_size = 64
    target_update_every = 5

    log = []
    t0 = time.time()

    for ep in range(n_episodios):
        obs, _ = env.reset(seed=seed + ep * 1000)
        weights = get_curriculum_weights(ep, n_episodios) if use_curriculum else None

        ep_reward = ep_lucro = ep_perdas = ep_rupturas = 0.0
        losses = []
        action_counts = np.zeros((n_p, n_i), dtype=np.int32)

        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mask = env_base.get_action_mask(no_repeat_window=no_repeat_window)
            ap, ai = online.select_action(obs_t, eps, rng, mask_cat=mask)
            obs2, reward_env, term, trunc, info = env.step((ap, ai))

            # V14: aplica curriculum nas recompensas
            if use_curriculum:
                reward_train = reward_curriculum(info, weights)
            else:
                reward_train = reward_env

            buf.push(obs, ap, ai, reward_train, obs2, term or trunc)
            obs = obs2
            ep_reward += reward_train
            ep_lucro += info['lucro']
            ep_perdas += float(info['perdas'].sum())
            ep_rupturas += float(info['rupturas'].sum())
            action_counts[ap, ai] += 1

            if len(buf) >= 1000:
                s, _ap, _ai, r, s2, d, indices, is_weights = buf.sample(batch_size, rng)
                s_t = torch.tensor(s, device=DEVICE)
                ap_t = torch.tensor(_ap, device=DEVICE)
                ai_t = torch.tensor(_ai, device=DEVICE)
                r_t = torch.tensor(r, device=DEVICE)
                s2_t = torch.tensor(s2, device=DEVICE)
                d_t = torch.tensor(d, device=DEVICE)
                w_t = torch.tensor(is_weights, device=DEVICE)

                q = online.q_values(s_t)
                q_sel = q[torch.arange(batch_size), ap_t, ai_t]

                with torch.no_grad():
                    q2_online = online.q_values(s2_t)
                    mask_t_b = torch.tensor(env_base.get_action_mask(no_repeat_window=0),
                                              dtype=torch.bool, device=DEVICE)
                    q2_online = q2_online.masked_fill(
                        ~mask_t_b.unsqueeze(0).unsqueeze(-1), NEG_INF
                    )
                    flat = q2_online.view(batch_size, -1)
                    best_idx = flat.argmax(dim=-1)
                    best_ap = best_idx // n_i
                    best_ai = best_idx % n_i
                    q2_target = target.q_values(s2_t)
                    q2_val = q2_target[torch.arange(batch_size), best_ap, best_ai]
                    y = r_t + gamma * q2_val * (1 - d_t)

                # Huber loss ponderada por IS weights (PER)
                td_errors = (y - q_sel).detach().cpu().numpy()
                loss_each = nn.functional.smooth_l1_loss(q_sel, y, reduction='none')
                loss = (loss_each * w_t).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                opt.step()
                losses.append(float(loss.item()))

                # PER: atualiza prioridades com novos TD errors
                buf.update_priorities(indices, td_errors)

            if term or trunc:
                break

        if ep % target_update_every == 0:
            target.load_state_dict(online.state_dict())

        eps = max(eps_min, eps * eps_decay)
        n_steps = step + 1
        promo_pct = action_counts[1:, :].sum() / n_steps * 100
        phase = 1 if ep / n_episodios < 0.33 else (2 if ep / n_episodios < 0.66 else 3)
        log.append({
            'seed': seed, 'episodio': ep, 'phase': phase, 'steps': n_steps,
            'reward_total': round(ep_reward, 2),
            'lucro_total': round(ep_lucro, 2),
            'perdas_total': round(ep_perdas, 1),
            'rupturas_total': round(ep_rupturas, 1),
            'epsilon': round(eps, 4),
            'loss_media': round(np.mean(losses) if losses else 0, 4),
            'pct_promove': round(promo_pct, 1),
            'per_beta': round(buf.beta, 3),
        })
        if ep % 10 == 0 or ep == n_episodios - 1:
            elapsed = time.time() - t0
            print(f"  seed {seed} ep {ep:>3d} F{phase} reward={ep_reward:>10,.0f} "
                  f"lucro={ep_lucro:>10,.0f} perdas={ep_perdas:>5.0f} "
                  f"promo={promo_pct:>4.1f}% ε={eps:.3f} β={buf.beta:.2f} ({elapsed:.0f}s)")

    return online, log


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodios', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_steps_per_ep', type=int, default=1095)
    parser.add_argument('--suffix', type=str, default='seed42')
    parser.add_argument('--drcr', action='store_true', help='Usar DRCR reward')
    parser.add_argument('--no_curriculum', action='store_true', help='Desativar curriculum')
    parser.add_argument('--no_repeat_window', type=int, default=2)
    args = parser.parse_args()

    print(f"V14.{args.suffix} — PER + curriculum + mask dinâmico"
          + (" + DRCR" if args.drcr else ""))
    print(f"Config: {args.episodios} ep × seed={args.seed}, {args.max_steps_per_ep} steps/ep")

    print(f"\n=== SEED {args.seed} ===")
    modelo_final, log = treinar_seed(args.seed, args.episodios, args.max_steps_per_ep,
                                       use_drcr=args.drcr,
                                       use_curriculum=not args.no_curriculum,
                                       no_repeat_window=args.no_repeat_window)

    suf = args.suffix
    df_log = pd.DataFrame(log)
    df_log.to_csv(RESULTS / f'training_log_v14_{suf}.csv', index=False, encoding='utf-8')

    torch.save({
        'state_dict': modelo_final.state_dict(),
        'config': {
            'obs_dim': modelo_final.trunk[0].in_features,
            'n_produtos': modelo_final.n_produtos,
            'n_intensidades': modelo_final.n_intensidades,
            'arch': 'BranchingDQN',
            'seed': args.seed,
            'drcr': args.drcr,
            'curriculum': not args.no_curriculum,
            'no_repeat_window': args.no_repeat_window,
        },
    }, RESULTS / f'dqn_v14_{suf}.pt')

    print(f"\n✓ Treino V14.{suf} concluído")
    ultimo = df_log[df_log['episodio'] >= max(0, args.episodios - 10)]
    print(f"\nMédia últimos 10 ep:")
    print(f"  reward:  R$ {ultimo['reward_total'].mean():>10,.0f}")
    print(f"  lucro:   R$ {ultimo['lucro_total'].mean():>10,.0f}")
    print(f"  perdas:  {ultimo['perdas_total'].mean():>10.1f} un")
    print(f"  promove: {ultimo['pct_promove'].mean():>10.1f} %")


if __name__ == '__main__':
    main()
