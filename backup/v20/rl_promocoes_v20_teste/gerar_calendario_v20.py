"""Gerar calendário V20 — rollout determinístico do agente treinado.

Roda múltiplos episódios em DATAS REAIS do calendário e coleta as decisões
do agente. Resultado: calendário operacional para apresentar ao posto.
"""
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from env_rl_promocoes import EnvRLPromocoes, INTENSIDADE_LABEL
from branching_dqn import BranchingDQNAgent


def rollout_deterministico(env: EnvRLPromocoes, agent: BranchingDQNAgent,
                             data_inicio: date, dias: int) -> list:
    """Rollout puro greedy com action masking (V20 final)."""
    obs, _ = env.reset(seed=0)
    env.data_atual = data_inicio
    obs = env._observar()

    decisoes = []
    for d in range(dias):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qs = agent.online(s)
        # Action masking V20: harmonia >= 1.4 na cabeça complementar
        mask = env.mask_complementar_valida()
        action = []
        for h_idx, q in enumerate(qs):
            q_arr = q.squeeze(0).cpu().numpy().copy() if q.dim() == 2 else q.cpu().numpy().copy()
            if h_idx == 1:
                q_arr[~mask] = -1e9
            action.append(int(np.argmax(q_arr)))
        action = np.array(action)

        obs_next, reward, done, _, info = env.step(action)

        # Salva decisão (excluindo turnos "nada")
        info['data_decisao'] = (data_inicio + timedelta(days=d)).isoformat()
        info['acao_indices'] = action.tolist()
        info['reward_final'] = reward
        decisoes.append(info)

        obs = obs_next
        if done:
            # Reset e continua o ano
            obs, _ = env.reset(seed=0)
            env.data_atual = data_inicio + timedelta(days=d + 1)
            obs = env._observar()
    return decisoes


def main():
    HERE = Path(__file__).parent
    PROJ_ROOT = HERE.parent
    model_path = HERE / 'models' / 'v20_final.pt'   # ITER 12 (9/9 cenários)
    out_dir = PROJ_ROOT / 'results' / 'v20'
    out_dir.mkdir(parents=True, exist_ok=True)

    env = EnvRLPromocoes(seed=0)
    agent = BranchingDQNAgent(env.state_dim, list(env.action_space.nvec),
                                hidden=128, device='cpu')
    agent.load(str(model_path))
    agent.online.eval()

    # Rollout de 365 dias começando em 2026-05-15
    DATA_INICIO = date(2026, 5, 15)
    DIAS = 365

    print(f"\n{'='*100}")
    print(f"ROLLOUT DETERMINÍSTICO V20 — {DIAS} dias a partir de {DATA_INICIO}")
    print(f"{'='*100}\n")

    decisoes = rollout_deterministico(env, agent, DATA_INICIO, DIAS)

    # Agrupa em campanhas (decisões consecutivas no mesmo (categoria, intensidade))
    campanhas = []
    cur = None
    for d in decisoes:
        if d['intensidade'] == 'nada':
            if cur:
                campanhas.append(cur)
                cur = None
            continue
        key = (d['categoria'], d['intensidade'], d.get('par_combo'))
        if cur and (cur['categoria'], cur['intensidade'], cur['par_combo']) == key:
            cur['dias'] += 1
            cur['lucro_total_acumulado'] += d['lucro_total']
            cur['reward_total_acumulado'] += d['reward_final']
            cur['data_fim'] = d['data_decisao']
        else:
            if cur:
                campanhas.append(cur)
            cur = {
                'data_inicio': d['data_decisao'],
                'data_fim': d['data_decisao'],
                'dias': 1,
                'categoria': d['categoria'],
                'intensidade': d['intensidade'],
                'par_combo': d.get('par_combo'),
                'alvo_desconto': d.get('alvo_desconto'),
                'demanda_base_anual': d['demanda_base_anual'],
                'demanda_base_contextual': d['demanda_base_contextual'],
                'demanda_promocional': d['demanda_promocional'],
                'uplift_pct': d['uplift_pct'],
                'canib_pct': d['canib_pct'],
                'lucro_total_acumulado': d['lucro_total'],
                'reward_total_acumulado': d['reward_final'],
                'evento_proximo': d.get('evento_proximo'),
                'eh_combo_invalido_pdv': d.get('eh_combo_invalido_pdv', False),
                'eh_alta_natural': d.get('eh_alta_natural', False),
            }
    if cur:
        campanhas.append(cur)

    # Stats
    lucro_total = sum(c['lucro_total_acumulado'] for c in campanhas)
    reward_total = sum(c['reward_total_acumulado'] for c in campanhas)
    n_combo = sum(1 for c in campanhas if c['intensidade'] == 'combo')
    n_pdv_inv = sum(1 for c in campanhas if c['eh_combo_invalido_pdv'])
    cats_unicas = len({c['categoria'] for c in campanhas})

    print(f"  Total turnos:          {len(decisoes)}")
    print(f"  Total campanhas:       {len(campanhas)}")
    print(f"  Combos:                {n_combo}")
    print(f"  PDV-inválidos:         {n_pdv_inv} (deveria ser 0 — agente aprendeu)")
    print(f"  Categorias distintas:  {cats_unicas} de {env.N_CATEGORIAS}")
    print(f"  Lucro total estimado:  R$ {lucro_total:,.2f}")
    print(f"  Reward total:          {reward_total:,.1f}")
    print(f"  Lucro médio/campanha:  R$ {lucro_total/max(len(campanhas),1):.2f}")

    print(f"\nDistribuição de intensidade:")
    from collections import Counter
    intens_count = Counter(c['intensidade'] for c in campanhas)
    for k, v in intens_count.most_common():
        print(f"    {k:<10s}: {v:>3d} ({v/len(campanhas)*100:.0f}%)")

    print(f"\nDistribuição de categoria:")
    cat_count = Counter(c['categoria'] for c in campanhas)
    for k, v in cat_count.most_common(10):
        print(f"    {k:<22s}: {v:>3d}")

    # Top 15 campanhas por lucro
    print(f"\nTOP 15 campanhas por lucro:")
    top15 = sorted(campanhas, key=lambda x: -x['lucro_total_acumulado'])[:15]
    for c in top15:
        evt = f" [{c['evento_proximo']}]" if c.get('evento_proximo') else ''
        print(f"  {c['data_inicio']} → {c['data_fim']} ({c['dias']}d) "
                f"{c['categoria']:<22s} {c['intensidade']:<8s} "
                f"par={str(c['par_combo'])[:14]:<14s} "
                f"R$ {c['lucro_total_acumulado']:>7.2f}{evt}")

    # Salva JSON
    out = {
        'versao': 'V20 — RL Branching DQN treinado',
        'data_inicio_rollout': DATA_INICIO.isoformat(),
        'dias_simulados': DIAS,
        'sumario': {
            'total_turnos': len(decisoes),
            'total_campanhas': len(campanhas),
            'combos': n_combo,
            'pdv_invalidos': n_pdv_inv,
            'lucro_total_R$': round(lucro_total, 2),
            'reward_total': round(reward_total, 2),
            'categorias_distintas': cats_unicas,
        },
        'campanhas': campanhas,
    }
    out_path = out_dir / 'calendario_v20.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Calendário V20 salvo: {out_path}")


if __name__ == '__main__':
    main()
