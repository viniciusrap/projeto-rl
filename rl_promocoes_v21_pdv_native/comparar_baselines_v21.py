"""Comparação do agente V21 com políticas de referência (baselines).

Responde à pergunta central de RL: "o agente aprendeu algo útil, ou esse
resultado seria igual sem aprender?". Roda EXATAMENTE o mesmo rollout de
365 dias do gerar_calendario_v21.py, no MESMO ambiente e janela, trocando
apenas a política de escolha de ação:

  - nao_promover : nunca promove (intensidade = 'nada')         [piso mínimo]
  - aleatorio    : ação válida aleatória (respeita action mask)  [chance pura]
  - sempre_combo : sempre combo com par válido aleatório         [regra trivial]
  - agente_v21   : política aprendida (greedy + action mask)     [nosso modelo]

Cada política roda em N seeds (demanda é Poisson → estocástica) e
reportamos média ± desvio. Comparação RELATIVA dentro do mesmo simulador
(off-policy): mostra que a política aprendida domina alternativas triviais,
não que o ganho absoluto está validado (isso só com A/B no posto).

Saída: results/v21/comparacao_baselines.json + tabela no terminal.
"""
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from env_rl_promocoes_v21 import EnvRLPromocoes
from branching_dqn import BranchingDQNAgent

DATA_INICIO = date(2026, 5, 15)   # mesma do gerar_calendario_v21.py
DIAS = 365
N_SEEDS = 8


def _acao_agente(env, agent, obs):
    s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qs = agent.online(s)
    mask = env.mask_complementar_valida()
    action = []
    for h_idx, q in enumerate(qs):
        q_arr = (q.squeeze(0).cpu().numpy().copy() if q.dim() == 2
                 else q.cpu().numpy().copy())
        if h_idx == 1:
            q_arr[~mask] = -1e9
        action.append(int(np.argmax(q_arr)))
    return np.array(action)


def _acao_nao_promover(env, rng):
    return np.array([0, 0, 0])


def _acao_aleatoria(env, rng):
    mask = env.mask_complementar_valida()
    validos = np.where(mask)[0]
    intens = int(rng.integers(0, 6))
    compl = int(rng.choice(validos)) if len(validos) else 0
    alvo = int(rng.integers(0, 2))
    return np.array([intens, compl, alvo])


def _acao_sempre_combo(env, rng):
    mask = env.mask_complementar_valida()
    validos = np.where(mask)[0]
    validos = validos[validos >= 1]   # 0 = nenhum; combo precisa de par
    compl = int(rng.choice(validos)) if len(validos) else 0
    intens = 5 if compl >= 1 else 0   # 5 = combo
    return np.array([intens, compl, 0])


def rollout(env, escolher_acao, seed):
    """365 dias, mesma lógica do gerar_calendario_v21.rollout_deterministico."""
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.data_atual = DATA_INICIO
    obs = env._observar()

    lucro_tot = reward_tot = 0.0
    n_promo = n_combo = n_pdv_inv = 0
    cats = set()

    for d in range(DIAS):
        action = escolher_acao(env, rng)
        obs_next, reward, done, _, info = env.step(action)
        lucro_tot += info['lucro_total']
        reward_tot += reward
        if info['intensidade'] != 'nada':
            n_promo += 1
            cats.add(info['categoria'])
        if info['intensidade'] == 'combo':
            n_combo += 1
        if info['eh_combo_invalido_pdv']:
            n_pdv_inv += 1
        obs = obs_next
        if done:
            obs, _ = env.reset(seed=seed)
            env.data_atual = DATA_INICIO + timedelta(days=d + 1)
            obs = env._observar()

    return {
        'lucro': lucro_tot,
        'reward': reward_tot,
        'n_promo': n_promo,
        'n_combo': n_combo,
        'n_pdv_invalido': n_pdv_inv,
        'cats_distintas': len(cats),
    }


def avaliar(nome, escolher_acao, env_factory, agent=None):
    runs = [rollout(env_factory(), escolher_acao, s) for s in range(N_SEEDS)]
    agg = {}
    for k in runs[0]:
        vals = np.array([r[k] for r in runs], dtype=float)
        agg[k] = (float(vals.mean()), float(vals.std()))
    return nome, agg


def main():
    HERE = Path(__file__).parent
    PROJ = HERE.parent
    model_path = HERE / 'models' / 'v21_final.pt'
    out_dir = PROJ / 'results' / 'v21'
    out_dir.mkdir(parents=True, exist_ok=True)

    def env_factory():
        return EnvRLPromocoes(seed=0)

    _e = env_factory()
    agent = BranchingDQNAgent(_e.state_dim, list(_e.action_space.nvec),
                               hidden=128, device='cpu')
    agent.load(str(model_path))
    agent.online.eval()

    politicas = [
        ('Nao-promover', _acao_nao_promover),
        ('Aleatorio', _acao_aleatoria),
        ('Sempre-combo', _acao_sempre_combo),
        ('Agente V21 (RL)', lambda env, rng: _acao_agente(env, agent, env._observar())),
    ]

    print(f"\n{'='*78}")
    print(f"COMPARAÇÃO COM BASELINES — {DIAS} dias × {N_SEEDS} seeds "
          f"(início {DATA_INICIO})")
    print(f"{'='*78}\n")
    print(f"{'Política':<20s} {'Lucro/ano (R$)':>22s} {'Campanhas':>11s} "
          f"{'Combos':>8s} {'PDV-inv':>8s} {'Cats':>6s}")
    print(f"{'-'*78}")

    resultados = {}
    for nome, fn in politicas:
        _, agg = avaliar(nome, fn, env_factory)
        resultados[nome] = agg
        lm, ls = agg['lucro']
        print(f"{nome:<20s} {lm:>13,.0f} ± {ls:>5,.0f} "
              f"{agg['n_promo'][0]:>11.0f} {agg['n_combo'][0]:>8.0f} "
              f"{agg['n_pdv_invalido'][0]:>8.1f} {agg['cats_distintas'][0]:>6.0f}")

    print(f"{'-'*78}")
    base = resultados['Nao-promover']['lucro'][0]
    ag = resultados['Agente V21 (RL)']['lucro'][0]
    rnd = resultados['Aleatorio']['lucro'][0]
    combo = resultados['Sempre-combo']['lucro'][0]
    print(f"\nAgente vs Não-promover : {ag - base:+,.0f} R$/ano "
          f"(promover bem >> não fazer nada)")
    print(f"Agente vs Aleatório    : {ag - rnd:+,.0f} R$/ano "
          f"(promoção ao acaso destrói valor)")
    print(f"Agente vs Sempre-combo : {ag - combo:+,.0f} R$/ano em lucro bruto, "
          f"mas {resultados['Agente V21 (RL)']['n_promo'][0]:.0f} campanhas "
          f"vs {resultados['Sempre-combo']['n_promo'][0]:.0f} "
          f"(agente é seletivo; sempre-combo promove quase todo dia)")

    out = {
        'protocolo': {
            'dias': DIAS, 'n_seeds': N_SEEDS,
            'data_inicio': DATA_INICIO.isoformat(),
            'nota': ('comparação relativa no mesmo simulador (off-policy); '
                     'não valida ganho absoluto — só A/B no posto valida'),
        },
        'resultados': {
            nome: {k: {'media': v[0], 'desvio': v[1]} for k, v in agg.items()}
            for nome, agg in resultados.items()
        },
    }
    out_path = out_dir / 'comparacao_baselines.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Salvo: {out_path}")


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
