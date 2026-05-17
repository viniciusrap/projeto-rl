"""Bateria de testes isolados do DemandAgent (V3 final).

12 cenários cobrindo:
- Combo bom (gelo+cerveja verão, chocolate+vinho Namorados)
- Combo médio (Réveillon)
- Combo ruim (sem harmonia)
- Desc direto em puxador (regra do dono)
- Desc direto em impulso
- Desc direto em commodity
- Liquidação defensiva (vencimento próximo)
- Estoque parado
- Eventos diferenciados (presente vs consumo)
- Proibida (cigarros)

Cada teste tem:
- expectativa esperada
- saída completa do agente
- veredito (✓/✗)
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from demand_agent import DemandAgent

ROOT = Path(__file__).parent


CENARIOS = [
    # ─── CASOS BONS (deveriam ser BOA) ───
    {
        'id': 'BOA-1',
        'desc': 'Chocolate Premium + Vinho no Dia dos Namorados (match perfeito)',
        'esperado': 'BOA',
        'campanha': {
            'categoria': 'chocolate_premium', 'produto_complementar': 'vinho',
            'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 4,
            'data_inicio': '2026-06-08', 'data_fim': '2026-06-11',
            'eventos_comerciais_na_janela': ['Dia dos Namorados'],
        }
    },
    {
        'id': 'BOA-2',
        'desc': 'Gelo + Cerveja em sábado VERÃO',
        'esperado': 'BOA',
        'campanha': {
            'categoria': 'gelo', 'produto_complementar': 'cerveja',
            'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 3,
            'data_inicio': '2027-01-09', 'data_fim': '2027-01-11',
            'eventos_comerciais_na_janela': [],
        }
    },
    {
        'id': 'BOA-3',
        'desc': 'Sorvete liquidação (validade próxima)',
        'esperado': 'BOA',
        'campanha': {
            'categoria': 'sorvete', 'intensidade': 'liq25%',
            'desconto_pct': 25, 'dias_total': 3,
            'data_inicio': '2026-05-18', 'data_fim': '2026-05-20',
            'eventos_comerciais_na_janela': [],
            'validade_restante_pct': 0.15, 'estoque_pct_normal': 1.4,
        }
    },
    # ─── CASOS MÉDIOS ───
    {
        'id': 'MED-1',
        'desc': 'Gelo + Destilados no Réveillon (consumo intenso)',
        'esperado': 'MÉDIA',  # ou BOA — Réveillon é evento forte
        'campanha': {
            'categoria': 'gelo', 'produto_complementar': 'destilados',
            'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 4,
            'data_inicio': '2026-12-28', 'data_fim': '2026-12-31',
            'eventos_comerciais_na_janela': ['Réveillon'],
        }
    },
    {
        'id': 'MED-2',
        'desc': 'Chocolate impulso desc10% segunda fraca',
        'esperado': 'MÉDIA',  # impulso responde a desc
        'campanha': {
            'categoria': 'chocolate_impulso', 'intensidade': 'desc10%',
            'desconto_pct': 10, 'dias_total': 3,
            'data_inicio': '2026-09-14', 'data_fim': '2026-09-16',
            'eventos_comerciais_na_janela': [],
        }
    },
    {
        'id': 'MED-3',
        'desc': 'Vinho parado com desc10%',
        'esperado': 'MÉDIA',  # estoque alto reforça desc
        'campanha': {
            'categoria': 'vinho', 'intensidade': 'desc10%',
            'desconto_pct': 10, 'dias_total': 5,
            'data_inicio': '2026-04-13', 'data_fim': '2026-04-17',
            'eventos_comerciais_na_janela': [],
            'estoque_pct_normal': 2.0, 'giro_alto': False,
        }
    },
    # ─── CASOS RUINS ───
    {
        'id': 'RUIM-1',
        'desc': 'Cerveja desc direto em SEX (alta natural)',
        'esperado': 'RUIM',  # regra do dono
        'campanha': {
            'categoria': 'cerveja', 'intensidade': 'desc10%',
            'desconto_pct': 10, 'dias_total': 2,
            'data_inicio': '2026-08-21', 'data_fim': '2026-08-22',
            'eventos_comerciais_na_janela': [],
            'estoque_pct_normal': 1.0, 'giro_alto': True,
        }
    },
    {
        'id': 'RUIM-2',
        'desc': 'Isotônico desc5% dia comum (commodity)',
        'esperado': 'RUIM',  # commodity baixa elast
        'campanha': {
            'categoria': 'isotonico', 'intensidade': 'desc5%',
            'desconto_pct': 5, 'dias_total': 5,
            'data_inicio': '2026-07-15', 'data_fim': '2026-07-19',
            'eventos_comerciais_na_janela': [],
        }
    },
    {
        'id': 'RUIM-3',
        'desc': 'Combo ABSURDO café + cerveja',
        'esperado': 'RUIM',  # combo antagônico
        'campanha': {
            'categoria': 'cafe', 'produto_complementar': 'cerveja',
            'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 3,
            'data_inicio': '2026-05-13', 'data_fim': '2026-05-15',
            'eventos_comerciais_na_janela': [],
        }
    },
    {
        'id': 'RUIM-4',
        'desc': 'Destilados desc10% sem evento (puxador premium)',
        'esperado': 'RUIM',  # regra do dono
        'campanha': {
            'categoria': 'destilados', 'intensidade': 'desc10%',
            'desconto_pct': 10, 'dias_total': 3,
            'data_inicio': '2026-09-13', 'data_fim': '2026-09-15',
            'eventos_comerciais_na_janela': [],
            'giro_alto': True,
        }
    },
    # ─── CONTRASTE: mesma campanha, contexto diferente ───
    {
        'id': 'CONTR-1',
        'desc': 'Chocolate Impulso em Dia das Crianças (deveria virar MED/BOA)',
        'esperado': 'BOA',  # match impulso × presente
        'campanha': {
            'categoria': 'chocolate_impulso', 'intensidade': 'desc10%',
            'desconto_pct': 10, 'dias_total': 4,
            'data_inicio': '2026-10-08', 'data_fim': '2026-10-11',
            'eventos_comerciais_na_janela': ['Dia das Crianças'],
        }
    },
    # ─── PROIBIDA ───
    {
        'id': 'PROIB-1',
        'desc': 'Cigarro com desc (Lei 9.294/96)',
        'esperado': 'PROIBIDA',
        'campanha': {
            'categoria': 'cigarro_souza_cruz', 'intensidade': 'desc5%',
            'desconto_pct': 5, 'dias_total': 3,
            'data_inicio': '2026-05-13', 'data_fim': '2026-05-15',
            'eventos_comerciais_na_janela': [],
        }
    },
]


def simplificar_classificacao(classif: str) -> str:
    if 'BOA' in classif: return 'BOA'
    if 'MÉDIA' in classif: return 'MÉDIA'
    if 'RUIM' in classif: return 'RUIM'
    if 'PROIBIDA' in classif: return 'PROIBIDA'
    return '?'


def main():
    agent = DemandAgent()

    print("=" * 115)
    print("TESTES ISOLADOS — DemandAgent V3 (responsabilidade única: demanda)")
    print("=" * 115)

    resultados = []
    acertos = 0

    for cen in CENARIOS:
        est = agent.estimar(cen['campanha'])
        obtido = simplificar_classificacao(est.qualidade_promocao)
        bate = obtido == cen['esperado']
        if bate: acertos += 1

        # Print compacto
        ok = '✓' if bate else '✗'
        print()
        print(f"┌─ {ok} {cen['id']} ─ {cen['desc']}")
        print(f"│  Esperado: {cen['esperado']:8s} | Obtido: {obtido:8s} "
              f"({est.qualidade_promocao} {est.confianca})")
        print(f"│  Demanda: {est.demanda_base_dia}→{est.demanda_promocional_dia}/dia "
              f"({est.uplift_pct:+.1f}%)")
        print(f"│  Canib: {est.canibalizacao_estimada_pct}% ({est.nivel_risco_canibalizacao})")
        comp = est.componentes
        if 'boost_preco' in comp:
            print(f"│  Boosts: preço×{comp['boost_preco']} combo×{comp['boost_combo']} "
                  f"evento×{comp['boost_evento']} clima×{comp['boost_clima']} "
                  f"dow×{comp['boost_dow']} eg×{comp['boost_estoque_giro']}")
            if comp.get('penalidade_combo_fraco', 1.0) < 1.0:
                print(f"│         × penalidade_combo_fraco {comp['penalidade_combo_fraco']}")
        if est.cap_aplicado:
            print(f"│  ⚠ Cap aplicado (uplift bruto era {comp.get('uplift_bruto_pre_cap', '?')}%)")
        if est.flags:
            print(f"│  Flags: {', '.join(est.flags)}")
        print(f"│  💬 {est.motivo}")
        print(f"└─")

        resultados.append({
            'id': cen['id'],
            'cenario': cen['desc'][:42],
            'esperado': cen['esperado'],
            'obtido': obtido,
            'bate': bate,
            'd_base': est.demanda_base_dia,
            'd_promo': est.demanda_promocional_dia,
            'uplift_%': est.uplift_pct,
            'canib_%': est.canibalizacao_estimada_pct,
            'tipo_produto': est.tipo_produto,
            'tipo_evento': est.tipo_evento,
            'qualidade': est.qualidade_promocao,
            'flags': ';'.join(est.flags),
        })

    df = pd.DataFrame(resultados)
    df.to_csv(ROOT / 'testes_demand_agent_resultados.csv', index=False, encoding='utf-8')

    # ─── Sumário ───
    print()
    print("=" * 115)
    print(f"SUMÁRIO — {acertos}/{len(CENARIOS)} testes corretos "
          f"({acertos/len(CENARIOS)*100:.0f}%)")
    print("=" * 115)
    print()
    print(df[['id', 'cenario', 'esperado', 'obtido', 'bate',
              'uplift_%', 'canib_%']].to_string(index=False))

    print()
    if acertos == len(CENARIOS):
        print("🎉 100% de acerto! Agente aprovado para integração.")
    elif acertos >= len(CENARIOS) - 2:
        print(f"⚠ {len(CENARIOS) - acertos} casos border-line — revisar antes de integrar.")
    else:
        print(f"❌ {len(CENARIOS) - acertos} casos errados — agente precisa refinamento.")

    print()
    print(f"✓ Resultados salvos em testes_demand_agent_resultados.csv")


if __name__ == '__main__':
    main()
