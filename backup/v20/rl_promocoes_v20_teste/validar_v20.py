"""Validação V20 — testa política aprendida em cenários específicos do briefing.

9 cenários definidos pelo Vinicius:
  1. Gelo + Cerveja em fim de semana quente
  2. Gelo + Destilados no Réveillon
  3. Chocolate Premium + Vinho
  4. Isotônico com desconto em dia comum
  5. Chocolate Impulso com baixa demanda
  6. Produto de alta demanda com desconto direto
  7. Produto parado em estoque
  8. Produto perto do vencimento
  9. Combo fraco entre produtos sem relação

Para cada: força o estado, mostra Q-values, ação escolhida, breakdown reward.
"""
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from env_rl_promocoes import (EnvRLPromocoes, INTENSIDADE_LABEL,
                                INTENSIDADE_DESCONTO_PCT)
from branching_dqn import BranchingDQNAgent


# Cenários do briefing — cada um força um contexto específico
CENARIOS = [
    {
        'nome': '1. Gelo + Cerveja em fim de semana quente',
        'categoria': 'gelo',
        'data': date(2027, 1, 9),       # sábado de janeiro (verão)
        'estoque': 1.0, 'validade': 1.0,
        'expectativa': 'agente deveria fazer combo, complementar=cerveja',
    },
    {
        'nome': '2. Gelo + Destilados no Réveillon',
        'categoria': 'gelo',
        'data': date(2026, 12, 30),     # quarta antes do Réveillon
        'estoque': 1.0, 'validade': 1.0,
        'expectativa': 'agente deveria FUGIR de destilados (PDV inválido), combo c/ cerveja',
    },
    {
        'nome': '3. Chocolate Premium + Vinho (Dia das Mães)',
        'categoria': 'chocolate_premium',
        'data': date(2027, 5, 5),        # 5 dias antes de Mães
        'estoque': 1.0, 'validade': 1.0,
        'expectativa': 'agente deveria fazer combo, complementar=vinho, intensidade alta',
    },
    {
        'nome': '4. Isotônico com desconto em dia comum',
        'categoria': 'isotonico',
        'data': date(2026, 7, 15),       # quarta sem evento
        'estoque': 1.0, 'validade': 1.0,
        'expectativa': 'agente deveria NÃO promover (commodity inelástica, sem evento)',
    },
    {
        'nome': '5. Chocolate Impulso com baixa demanda',
        'categoria': 'chocolate_impulso',
        'data': date(2026, 9, 14),       # segunda comum, sem evento
        'estoque': 1.0, 'validade': 1.0,
        'expectativa': 'agente pode fazer desc pequeno OU não promover',
    },
    {
        'nome': '6. Cerveja em sex (alta natural) com desc direto',
        'categoria': 'cerveja',
        'data': date(2026, 8, 21),       # sexta de agosto
        'estoque': 1.0, 'validade': 1.0,
        'expectativa': 'agente NÃO deveria descontar direto (alta natural); combo OK',
    },
    {
        'nome': '7. Sorvete parado em estoque (verão)',
        'categoria': 'sorvete',
        'data': date(2027, 1, 15),       # sexta de janeiro
        'estoque': 2.0, 'validade': 0.8,  # estoque alto
        'expectativa': 'agente pode dar desc para girar estoque',
    },
    {
        'nome': '8. Sorvete perto do vencimento',
        'categoria': 'sorvete',
        'data': date(2026, 5, 20),       # outono
        'estoque': 1.5, 'validade': 0.15,  # validade BAIXA
        'expectativa': 'agente deveria fazer liquidação (desc10% ou maior)',
    },
    {
        'nome': '9. Combo fraco: café + cerveja',
        'categoria': 'cafe',
        'data': date(2026, 9, 5),
        'estoque': 1.0, 'validade': 1.0,
        'expectativa': 'agente NÃO deveria escolher combo c/ cerveja (PDV inválido + sem harmonia)',
    },
]


def avaliar_cenario(env: EnvRLPromocoes, agent: BranchingDQNAgent,
                     cen: dict) -> dict:
    """Força o estado de um cenário, captura ação greedy + Q-values."""
    obs, _ = env.reset(seed=42)

    # Força o produto e contexto
    env.produto_atual_idx = env.cat_idx[cen['categoria']]
    env.data_atual = cen['data']
    cat_idx = env.cat_idx[cen['categoria']]
    env.estoque_rel[cat_idx] = cen['estoque']
    env.validade_rel[cat_idx] = cen['validade']
    env.hist_categorias = []  # zera histórico

    obs = env._observar()

    # Q-values + ação greedy
    s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        qs = agent.online(s)
    q_intensidade = qs[0].squeeze(0).numpy()
    q_complementar = qs[1].squeeze(0).numpy()
    q_alvo = qs[2].squeeze(0).numpy()
    action = np.array([int(q_intensidade.argmax()),
                        int(q_complementar.argmax()),
                        int(q_alvo.argmax())])

    # Executa o step pra obter info detalhada
    obs2, reward, _, _, info = env.step(action)

    return {
        'cenario': cen['nome'],
        'expectativa': cen['expectativa'],
        'acao_intensidade': INTENSIDADE_LABEL[action[0]],
        'acao_complementar_idx': int(action[1]),
        'acao_complementar': env.categorias[action[1] - 1] if action[1] > 0 else 'nenhum',
        'acao_alvo': 'principal' if action[2] == 0 else 'complementar',
        'demanda_base_anual': info['demanda_base_anual'],
        'demanda_base_ctx': info['demanda_base_contextual'],
        'demanda_promo': info['demanda_promocional'],
        'uplift_pct': info['uplift_pct'],
        'canib_pct': info['canib_pct'],
        'lucro_total': info['lucro_total'],
        'reward': info['reward'],
        'reward_breakdown': info['reward_breakdown'],
        'eh_pdv_invalido': info['eh_combo_invalido_pdv'],
        'evento': info['evento_proximo'],
        'q_intensidade': q_intensidade.tolist(),
        'q_top3_combo': sorted(enumerate(q_complementar),
                                key=lambda x: -x[1])[:3],
    }


def validar(model_path: Path, out_path: Path):
    env = EnvRLPromocoes(seed=42)
    state_dim = env.state_dim
    action_dims = list(env.action_space.nvec)
    agent = BranchingDQNAgent(state_dim, action_dims, hidden=128, device='cpu')
    agent.load(str(model_path))
    agent.online.eval()

    print(f"\n{'='*100}")
    print(f"VALIDAÇÃO V20 — modelo: {model_path.name}")
    print(f"{'='*100}\n")

    resultados = []
    for cen in CENARIOS:
        r = avaliar_cenario(env, agent, cen)
        resultados.append(r)
        print(f"📌 {r['cenario']}")
        print(f"   Expectativa: {r['expectativa']}")
        print(f"   AÇÃO: {r['acao_intensidade']:<8s}  par={r['acao_complementar']:<22s}  alvo={r['acao_alvo']}")
        print(f"   Demanda: {r['demanda_base_anual']:.1f} → {r['demanda_base_ctx']:.1f} → {r['demanda_promo']:.1f} (uplift +{r['uplift_pct']:.0f}%)")
        if r['evento']:
            print(f"   Evento próximo: {r['evento']}")
        print(f"   Lucro estimado: R$ {r['lucro_total']:>7.2f}/turno  |  Reward: {r['reward']:>7.1f}")
        if r['reward_breakdown']:
            partes = [f"{k}={v:+.0f}" for k, v in r['reward_breakdown'].items() if k != 'lucro_base']
            if partes:
                print(f"   Shaping: {' '.join(partes)}")
        if r['eh_pdv_invalido']:
            print(f"   ⚠ AGENTE ESCOLHEU COMBO PDV-INVÁLIDO (penalidade aplicada — deveria APRENDER a evitar)")
        # Top 3 combos por Q-value
        print(f"   Top 3 complementares (Q-value):")
        for idx, q in r['q_top3_combo']:
            nome = env.categorias[idx - 1] if idx > 0 else 'nenhum'
            print(f"      {nome:<22s} Q={q:>7.2f}")
        # Top 3 intensidades
        intensidades_ordenadas = sorted(enumerate(r['q_intensidade']), key=lambda x: -x[1])[:3]
        print(f"   Top 3 intensidades (Q-value):")
        for idx, q in intensidades_ordenadas:
            print(f"      {INTENSIDADE_LABEL[idx]:<10s} Q={q:>7.2f}")
        print()

    # Salva JSON
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Converte tuplas em listas para serializar
    for r in resultados:
        r['q_top3_combo'] = [(int(i), float(q)) for i, q in r['q_top3_combo']]
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"✓ Resultados salvos: {out_path}")
    return resultados


if __name__ == '__main__':
    HERE = Path(__file__).parent
    PROJ_ROOT = HERE.parent
    # Usa o melhor modelo de qualquer seed (o de menor seed por convenção)
    model_path = HERE / 'models' / 'best_seed_0.pt'
    if not model_path.exists():
        print(f'❌ Modelo não encontrado: {model_path}')
        print('   Rode treinar_v20.py primeiro.')
        sys.exit(1)

    out_path = PROJ_ROOT / 'results' / 'v20' / 'validacao_v20.json'
    validar(model_path, out_path)
