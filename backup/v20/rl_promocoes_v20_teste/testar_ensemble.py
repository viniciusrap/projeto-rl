"""Testa ensemble dos 2 melhores modelos (iter 12 + iter 16, ambos 9/9)."""
import io
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from env_rl_promocoes import EnvRLPromocoes
from ensemble import EnsembleAgent
from iterar_v20 import CENARIOS_VALIDACAO, avaliar_cenario, avaliar_rollout, analisar_rollout


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    HERE = Path(__file__).parent
    PROJ = HERE.parent

    env = EnvRLPromocoes(seed=42)
    state_dim = env.state_dim
    action_dims = list(env.action_space.nvec)

    # Modelos top
    model_paths = [
        HERE / 'models' / 'iter_iter12_seed_0.pt',
        HERE / 'models' / 'iter_iter16_seed_0.pt',
    ]
    print("Carregando ensemble:")
    for p in model_paths:
        print(f"  - {p.name}")

    agent = EnsembleAgent(state_dim, action_dims, model_paths, hidden=128)

    # 9 cenários
    print("\n9 CENÁRIOS (ensemble iter 12 + iter 16):")
    n_ok = 0
    for cen in CENARIOS_VALIDACAO:
        r = avaliar_cenario(env, agent, cen)
        ok = cen['check_acao_correta'](r)
        if ok:
            n_ok += 1
        marker = '✓' if ok else '✗'
        comp = f"par={r['acao_complementar']}" if r['acao_intensidade'] == 'combo' else ''
        print(f"  {marker} {cen['id']:<40s} {r['acao_intensidade']:<10s} {comp}")
    print(f"\n  SCORE: {n_ok}/9")

    # Rollout 365 dias
    decisoes = avaliar_rollout(env, agent)
    stats = analisar_rollout(decisoes, env)

    print(f"\nROLLOUT 365 dias:")
    print(f"  Lucro total:       R$ {stats['lucro_total']:>10,.2f}")
    print(f"  Campanhas:         {stats['n_promo']}")
    print(f"  PDV-inválidos:     {stats['pdv_invalidos']}")
    print(f"  Cats distintas:    {stats['cats_distintas']}")
    print(f"\n  Top 10 pares de combo:")
    for (a, b), cnt in stats['top_pares'][:10]:
        print(f"    {a:<22s} + {b:<22s} ({cnt}x)")


if __name__ == '__main__':
    main()
