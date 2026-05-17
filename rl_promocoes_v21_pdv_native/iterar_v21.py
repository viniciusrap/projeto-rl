"""Script de iteração automática: treina → valida → analisa → diagnostica.

Para cada iteração, gera relatório do que está funcionando e o que falhou.
Útil para iteração noturna autônoma.
"""
import io
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch

from env_rl_promocoes_v21 import EnvRLPromocoes, INTENSIDADE_LABEL
from branching_dqn import BranchingDQNAgent


# CENÁRIOS DE VALIDAÇÃO V21 — adaptados para 20 categorias PDV-native
# Cada um valida UM aspecto do conhecimento de PDV.
# OBJETIVO: que os combos do agente sejam aqueles que cliente REALMENTE leva
# no balcão de posto (não combos de cesta de supermercado).
CENARIOS_VALIDACAO = [
    {
        'id': 'C1_cafe_padaria_manha',
        'categoria': 'cafe', 'data': date(2026, 7, 13),  # segunda 7h
        'estoque': 1.0, 'validade': 1.0,
        'check_acao_correta': lambda r: r['acao_intensidade'] == 'combo' and r['acao_complementar'] == 'padaria',
        'descricao': 'Café+Padaria manhã (clássico commuter)',
    },
    {
        'id': 'C2_gelo_reveillon_NAO_destilados',
        'categoria': 'gelo', 'data': date(2026, 12, 30),
        'estoque': 1.0, 'validade': 1.0,
        'check_acao_correta': lambda r: r['acao_complementar'] != 'destilados',
        'descricao': 'Gelo Réveillon — NÃO destilados',
    },
    {
        'id': 'C3_choc_caixa_maes',
        'categoria': 'chocolate_caixa', 'data': date(2027, 5, 5),
        'estoque': 1.0, 'validade': 1.0,
        # Em PDV, chocolate_caixa em Mães combina com doce_balcao, biscoito, café — NÃO vinho
        'check_acao_correta': lambda r: r['acao_intensidade'] == 'combo' and r['acao_complementar'] in ('doce_balcao', 'biscoito', 'cafe', 'chocolate_unit'),
        'descricao': 'Chocolate caixa Mães — combo PDV (não vinho!)',
    },
    {
        'id': 'C4_isotonico_dia_comum',
        'categoria': 'isotonico', 'data': date(2026, 7, 15),
        'estoque': 1.0, 'validade': 1.0,
        'check_acao_correta': lambda r: r['acao_intensidade'] == 'nada',
        'descricao': 'Isotônico dia comum',
    },
    {
        'id': 'C5_choc_unit_baixa',
        'categoria': 'chocolate_unit', 'data': date(2026, 9, 14),
        'estoque': 1.0, 'validade': 1.0,
        'check_acao_correta': lambda r: r['acao_intensidade'] in ('nada', 'desc3%', 'desc5%'),
        'descricao': 'Chocolate unit baixa demanda',
    },
    {
        'id': 'C6_cerveja_sex_alta',
        'categoria': 'cerveja', 'data': date(2026, 8, 21),
        'estoque': 1.0, 'validade': 1.0,
        'check_acao_correta': lambda r: r['acao_intensidade'] in ('nada', 'combo'),
        'descricao': 'Cerveja sex alta natural — não desc direto',
    },
    {
        'id': 'C7_sorvete_parado',
        'categoria': 'sorvete', 'data': date(2027, 1, 15),
        'estoque': 2.0, 'validade': 0.8,
        'check_acao_correta': lambda r: r['acao_intensidade'] in ('desc5%', 'desc7%', 'desc10%', 'combo'),
        'descricao': 'Sorvete parado',
    },
    {
        'id': 'C8_sorvete_vencimento',
        'categoria': 'sorvete', 'data': date(2026, 5, 20),
        'estoque': 1.5, 'validade': 0.15,
        'check_acao_correta': lambda r: r['acao_intensidade'] in ('desc10%', 'combo'),
        'descricao': 'Sorvete vencimento — liquida',
    },
    {
        'id': 'C9_cafe_cerveja_absurdo',
        'categoria': 'cafe', 'data': date(2026, 9, 5),
        'estoque': 1.0, 'validade': 1.0,
        'check_acao_correta': lambda r: not (r['acao_intensidade'] == 'combo' and r['acao_complementar'] == 'cerveja'),
        'descricao': 'Café+cerveja — não escolher cerveja',
    },
    # NOVOS V21 — específicos PDV-native
    {
        'id': 'C10_refri_snack_lanche',
        'categoria': 'refrigerante', 'data': date(2026, 7, 22),  # quarta tarde
        'estoque': 1.0, 'validade': 1.0,
        'check_acao_correta': lambda r: r['acao_intensidade'] == 'combo' and r['acao_complementar'] == 'snack_salgado',
        'descricao': 'Refri+Snack lanche da tarde (clássico)',
    },
    {
        'id': 'C11_gelo_whisky_NAO',
        'categoria': 'gelo', 'data': date(2026, 12, 30),
        'estoque': 1.0, 'validade': 1.0,
        # Saco gelo + garrafa whisky no balcão de posto = absurdo
        'check_acao_correta': lambda r: r['acao_complementar'] != 'whisky',
        'descricao': 'Gelo NÃO escolher whisky (compra de mercado, não posto)',
    },
    {
        'id': 'C12_padaria_cerveja_NAO',
        'categoria': 'padaria', 'data': date(2026, 8, 7),  # sexta manhã
        'estoque': 1.0, 'validade': 1.0,
        # Padaria (manhã) + cerveja (noite) = ocasiões opostas
        'check_acao_correta': lambda r: r['acao_complementar'] != 'cerveja',
        'descricao': 'Padaria NÃO combinar com cerveja (manhã vs noite)',
    },
]


def avaliar_cenario(env, agent, cen):
    obs, _ = env.reset(seed=42)
    env.produto_atual_idx = env.cat_idx[cen['categoria']]
    env.data_atual = cen['data']
    cat_idx = env.cat_idx[cen['categoria']]
    env.estoque_rel[cat_idx] = cen['estoque']
    env.validade_rel[cat_idx] = cen['validade']
    env.hist_categorias = []
    obs = env._observar()
    s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qs = agent.online(s)
    # Aplica máscara V20 ITER 7 na cabeça complementar
    mask = env.mask_complementar_valida()
    action = []
    for h_idx, q in enumerate(qs):
        q_arr = q.squeeze(0).cpu().numpy().copy() if q.dim() == 2 else q.cpu().numpy().copy()
        if h_idx == 1:
            q_arr[~mask] = -1e9
        action.append(int(np.argmax(q_arr)))
    action = np.array(action)
    obs2, reward, _, _, info = env.step(action)
    return {
        'id': cen['id'],
        'descricao': cen['descricao'],
        'acao_intensidade': INTENSIDADE_LABEL[action[0]],
        'acao_complementar_idx': int(action[1]),
        'acao_complementar': env.categorias[action[1] - 1] if action[1] > 0 else 'nenhum',
        'acao_alvo': 'principal' if action[2] == 0 else 'complementar',
        'lucro_total': info['lucro_total'],
        'reward': info['reward'],
        'evento': info.get('evento_proximo'),
        'eh_pdv_invalido': info.get('eh_combo_invalido_pdv', False),
    }


def avaliar_rollout(env, agent, dias=365, data_inicio=date(2026, 5, 15)):
    """Roda rollout 365 dias e mede stats."""
    obs, _ = env.reset(seed=0)
    env.data_atual = data_inicio
    obs = env._observar()

    decisoes = []
    for d in range(dias):
        s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qs = agent.online(s)
        mask = env.mask_complementar_valida()
        action = []
        for h_idx, q in enumerate(qs):
            q_arr = q.squeeze(0).cpu().numpy().copy() if q.dim() == 2 else q.cpu().numpy().copy()
            if h_idx == 1:
                q_arr[~mask] = -1e9
            action.append(int(np.argmax(q_arr)))
        action = np.array(action)
        obs_next, reward, done, _, info = env.step(action)
        info['acao_indices'] = action.tolist()
        decisoes.append(info)
        obs = obs_next
        if done:
            obs, _ = env.reset(seed=0)
            obs = env._observar()
    return decisoes


def analisar_rollout(decisoes, env):
    """Gera stats agregados do rollout."""
    # Combos por par
    pares_combo = Counter()
    intensidades = Counter()
    cats_promovidas = Counter()
    pdv_inv = 0
    lucro_total = 0.0
    n_promo = 0
    eventos_capturados = []

    pares_invalidos = {
        frozenset(['gelo', 'destilados']), frozenset(['gelo', 'vinho']),
        frozenset(['gelo', 'sorvete']),    frozenset(['cafe', 'cerveja']),
        frozenset(['cafe', 'destilados']), frozenset(['cafe', 'vinho']),
        frozenset(['sorvete', 'cerveja']), frozenset(['padaria', 'cerveja']),
        frozenset(['padaria', 'destilados']),
    }

    for d in decisoes:
        intensidades[d['intensidade']] += 1
        if d['intensidade'] != 'nada':
            n_promo += 1
            cats_promovidas[d['categoria']] += 1
            lucro_total += d['lucro_total']
            if d['intensidade'] == 'combo' and d.get('par_combo'):
                par = (d['categoria'], d['par_combo'])
                pares_combo[par] += 1
                if frozenset([d['categoria'], d['par_combo']]) in pares_invalidos:
                    pdv_inv += 1
            if d.get('evento_proximo'):
                eventos_capturados.append((d['categoria'], d['evento_proximo']))

    return {
        'lucro_total': lucro_total,
        'n_promo': n_promo,
        'n_total': len(decisoes),
        'pct_promove': n_promo / len(decisoes),
        'pdv_invalidos': pdv_inv,
        'cats_distintas': len(cats_promovidas),
        'intensidades': dict(intensidades),
        'top_pares': pares_combo.most_common(10),
        'top_cats': cats_promovidas.most_common(10),
        'eventos_capturados': Counter([e[1] for e in eventos_capturados]).most_common(10),
    }


def rodar_iteracao(iter_nome, n_episodios=400, seed=0, n_seeds=1):
    """Treina + valida + roda rollout. Se n_seeds>1, faz ensemble."""
    HERE = Path(__file__).parent
    LOG_DIR = HERE / 'logs'
    MODEL_DIR = HERE / 'models'
    LOG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*100}")
    print(f"ITER {iter_nome} — treino {n_episodios} ep, n_seeds={n_seeds}")
    print(f"{'='*100}\n")

    from treinar_v21 import treinar
    model_paths = []
    rewards_finais = []
    for s in range(n_seeds):
        seed_atual = seed + s
        print(f"\n--- Treinando seed {seed_atual} ---")
        rewards, _ = treinar(
            seed=seed_atual,
            n_episodios=n_episodios,
            log_path=LOG_DIR / f'iter_{iter_nome}_seed_{seed_atual}.csv',
            model_path=MODEL_DIR / f'iter_{iter_nome}_seed_{seed_atual}.pt',
            verbose=False,
        )
        model_paths.append(MODEL_DIR / f'iter_{iter_nome}_seed_{seed_atual}.pt')
        rewards_finais.append(float(np.mean(rewards[-30:])))
        print(f"  Seed {seed_atual} reward final (últimos 30): {rewards_finais[-1]:.1f}")

    # Carrega ensemble se n_seeds>1, single se n_seeds==1
    env = EnvRLPromocoes(seed=42)
    if n_seeds == 1:
        agent = BranchingDQNAgent(env.state_dim, list(env.action_space.nvec),
                                    hidden=128, device='cpu')
        agent.load(str(model_paths[0]))
        agent.online.eval()
    else:
        from ensemble import EnsembleAgent
        agent = EnsembleAgent(env.state_dim, list(env.action_space.nvec),
                                model_paths, hidden=128)
    rewards = rewards_finais if n_seeds > 1 else rewards
    if n_seeds == 1:
        rewards = locals().get('rewards', [])  # do treinar

    # Avalia 9 cenários
    print(f"\n  9 CENÁRIOS:")
    cenarios_resultados = []
    n_ok = 0
    for cen in CENARIOS_VALIDACAO:
        r = avaliar_cenario(env, agent, cen)
        ok = cen['check_acao_correta'](r)
        if ok:
            n_ok += 1
        marker = '✓' if ok else '✗'
        comp_str = f"par={r['acao_complementar']}" if r['acao_intensidade'] == 'combo' else ''
        print(f"    {marker} {cen['id']:<40s} {r['acao_intensidade']:<10s} {comp_str}")
        cenarios_resultados.append({**r, 'ok': ok, 'descricao': cen['descricao']})

    # Avalia rollout 365 dias
    decisoes = avaliar_rollout(env, agent)
    stats = analisar_rollout(decisoes, env)

    print(f"\n  ROLLOUT 365 dias:")
    print(f"    Lucro total:       R$ {stats['lucro_total']:>10,.2f}")
    print(f"    Campanhas:         {stats['n_promo']} ({stats['pct_promove']*100:.0f}%)")
    print(f"    PDV-inválidos:     {stats['pdv_invalidos']}")
    print(f"    Cats distintas:    {stats['cats_distintas']}")
    print(f"    Top 5 pares de combo:")
    for (a, b), cnt in stats['top_pares'][:5]:
        print(f"      {a:<22s} + {b:<22s} ({cnt}x)")
    print(f"    Eventos capturados:")
    for ev, cnt in stats['eventos_capturados'][:5]:
        print(f"      {ev:<30s} ({cnt}x)")
    print(f"\n  SCORE 9 cenários: {n_ok}/9")

    return {
        'iter': iter_nome,
        'rewards_final': float(np.mean(rewards[-30:])),
        'rewards_inicio': float(np.mean(rewards[:10])),
        'cenarios': cenarios_resultados,
        'cenarios_ok': n_ok,
        'rollout_stats': stats,
    }


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter_nome', type=str, required=True)
    parser.add_argument('--episodios', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_seeds', type=int, default=1)
    args = parser.parse_args()

    resultado = rodar_iteracao(args.iter_nome, args.episodios, args.seed, args.n_seeds)

    # Salva resumo
    HERE = Path(__file__).parent
    out_path = HERE / 'logs' / f'iter_{args.iter_nome}_resumo.json'

    # Serializa
    res_ser = {
        'iter': resultado['iter'],
        'rewards_final': resultado['rewards_final'],
        'rewards_inicio': resultado['rewards_inicio'],
        'cenarios_ok': resultado['cenarios_ok'],
        'cenarios': resultado['cenarios'],
        'rollout': {
            'lucro_total': resultado['rollout_stats']['lucro_total'],
            'n_promo': resultado['rollout_stats']['n_promo'],
            'pdv_invalidos': resultado['rollout_stats']['pdv_invalidos'],
            'cats_distintas': resultado['rollout_stats']['cats_distintas'],
            'intensidades': resultado['rollout_stats']['intensidades'],
            'top_pares': [(f"{a}+{b}", c) for (a, b), c in resultado['rollout_stats']['top_pares']],
            'top_cats': resultado['rollout_stats']['top_cats'],
            'eventos_capturados': resultado['rollout_stats']['eventos_capturados'],
        },
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res_ser, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Resumo salvo: {out_path}")
