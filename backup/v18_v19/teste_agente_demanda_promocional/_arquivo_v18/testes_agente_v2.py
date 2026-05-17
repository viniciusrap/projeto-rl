"""Testes isolados do DemandPromotionAgentV2.

Cobre os 9 cenários pedidos pelo Vinicius para validar se o agente
faz julgamentos realistas para uma conveniência de posto.

Saída: testes_agente_v2_resultados.csv + print formatado.
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from demand_promotion_agent_v2 import DemandPromotionAgentV2

ROOT = Path(__file__).parent


CENARIOS = [
    {
        'id': 'A',
        'nome': 'Gelo + Cerveja em fim de semana QUENTE',
        'esperado': 'BOA — produto puxador, combo natural, contexto perfeito',
        'campanha': {
            'categoria': 'gelo',
            'produto_complementar': 'cerveja',
            'intensidade': 'combo',
            'desconto_pct': 10,
            'dias_total': 2,
            'data_inicio': '2027-01-10',  # sáb verão
            'data_fim': '2027-01-11',
            'eventos_comerciais_na_janela': [],
        }
    },
    {
        'id': 'B',
        'nome': 'Gelo + Destilados no Réveillon',
        'esperado': 'BOA — evento consumo intenso, combo harmonia alta',
        'campanha': {
            'categoria': 'gelo',
            'produto_complementar': 'destilados',
            'intensidade': 'combo',
            'desconto_pct': 10,
            'dias_total': 4,
            'data_inicio': '2026-12-28',
            'data_fim': '2026-12-31',
            'eventos_comerciais_na_janela': ['Réveillon'],
        }
    },
    {
        'id': 'C',
        'nome': 'Chocolate Premium + Vinho no Dia dos Namorados',
        'esperado': 'BOA — match perfeito puxador_presente + evento presente',
        'campanha': {
            'categoria': 'chocolate_premium',
            'produto_complementar': 'vinho',
            'intensidade': 'combo',
            'desconto_pct': 10,
            'dias_total': 4,
            'data_inicio': '2026-06-08',
            'data_fim': '2026-06-11',
            'eventos_comerciais_na_janela': ['Dia dos Namorados'],
        }
    },
    {
        'id': 'D',
        'nome': 'Isotônico com desconto em dia COMUM',
        'esperado': 'RUIM — commodity, sem evento, desc pequeno',
        'campanha': {
            'categoria': 'isotonico',
            'produto_complementar': '',
            'intensidade': 'desc5%',
            'desconto_pct': 5,
            'dias_total': 5,
            'data_inicio': '2026-07-15',
            'data_fim': '2026-07-19',
            'eventos_comerciais_na_janela': [],
        }
    },
    {
        'id': 'E',
        'nome': 'Chocolate Impulso com desconto por BAIXA DEMANDA (segunda-feira)',
        'esperado': 'MÉDIA — produto impulso, contexto fraco mas desc move',
        'campanha': {
            'categoria': 'chocolate_impulso',
            'produto_complementar': '',
            'intensidade': 'desc10%',
            'desconto_pct': 10,
            'dias_total': 3,
            'data_inicio': '2026-09-14',  # segunda
            'data_fim': '2026-09-16',
            'eventos_comerciais_na_janela': [],
        }
    },
    {
        'id': 'F',
        'nome': 'Produto de ALTA DEMANDA (cerveja sex/sáb) com desconto direto 10%',
        'esperado': 'RUIM — cerveja já vende, dar desc destrói margem',
        'campanha': {
            'categoria': 'cerveja',
            'produto_complementar': '',
            'intensidade': 'desc10%',
            'desconto_pct': 10,
            'dias_total': 2,
            'data_inicio': '2026-08-21',  # sex
            'data_fim': '2026-08-22',
            'eventos_comerciais_na_janela': [],
            # Sinais explícitos (campanha em alta natural)
            'estoque_pct_normal': 1.0,
            'giro_alto': True,
        }
    },
    {
        'id': 'G',
        'nome': 'Produto PERTO DO VENCIMENTO com liquidação 25%',
        'esperado': 'BOA — liquidação é defensiva, evita perda total',
        'campanha': {
            'categoria': 'sorvete',
            'produto_complementar': '',
            'intensidade': 'liq25%',
            'desconto_pct': 25,
            'dias_total': 3,
            'data_inicio': '2026-05-18',
            'data_fim': '2026-05-20',
            'eventos_comerciais_na_janela': [],
            'validade_restante_pct': 0.15,  # 15% da validade restante
            'estoque_pct_normal': 1.4,
            'giro_alto': True,
        }
    },
    {
        'id': 'H',
        'nome': 'Produto PARADO em estoque com desconto 10%',
        'esperado': 'MÉDIA — estoque alto + desc converte, mas margem cai',
        'campanha': {
            'categoria': 'vinho',  # baixo giro tradicional
            'produto_complementar': '',
            'intensidade': 'desc10%',
            'desconto_pct': 10,
            'dias_total': 5,
            'data_inicio': '2026-04-13',
            'data_fim': '2026-04-17',
            'eventos_comerciais_na_janela': [],
            'estoque_pct_normal': 2.0,  # estoque dobro do normal
            'giro_alto': False,
        }
    },
    {
        'id': 'I',
        'nome': 'Combo FRACO entre produtos sem relação (café + cerveja)',
        'esperado': 'RUIM — sem harmonia, combo não converte',
        'campanha': {
            'categoria': 'cafe',
            'produto_complementar': 'cerveja',  # combo absurdo
            'intensidade': 'combo',
            'desconto_pct': 10,
            'dias_total': 3,
            'data_inicio': '2026-05-13',
            'data_fim': '2026-05-15',
            'eventos_comerciais_na_janela': [],
        }
    },
    # Bônus: produto proibido
    {
        'id': 'J',
        'nome': 'Cigarro com desconto (Lei 9.294/96)',
        'esperado': 'PROIBIDA — não deve ser promovido',
        'campanha': {
            'categoria': 'cigarro_souza_cruz',
            'produto_complementar': '',
            'intensidade': 'desc5%',
            'desconto_pct': 5,
            'dias_total': 3,
            'data_inicio': '2026-05-13',
            'data_fim': '2026-05-15',
            'eventos_comerciais_na_janela': [],
        }
    },
]


def main():
    agent = DemandPromotionAgentV2()

    print("=" * 110)
    print("TESTES ISOLADOS DO AGENTE DE DEMANDA PROMOCIONAL — V2")
    print("=" * 110)

    resultados = []

    for cenario in CENARIOS:
        c = cenario['campanha']
        est = agent.estimar(c)

        # Print formatado
        print()
        print(f"┌─ CENÁRIO {cenario['id']} ─ {cenario['nome']}")
        print(f"│  Esperado: {cenario['esperado']}")
        print(f"│  Resultado: {est.classificacao}  ({est.confianca} confiança)")
        print(f"│")
        print(f"│  TIPO: {est.tipo_produto} | EVENTO: {est.tipo_evento}")
        print(f"│  Demanda: {est.demanda_base_dia} → {est.demanda_promocional_dia} un/dia ({est.uplift_pct:+.1f}%)")
        print(f"│  Boosts:  elast×{est.boost_elasticidade} combo×{est.boost_combo_harmonia} "
              f"evento×{est.boost_evento} clima×{est.boost_clima}")
        print(f"│          dow×{est.boost_dow} estoque×{est.boost_estoque_giro}")
        if est.cap_aplicado:
            print(f"│  ⚠ CAP de uplift atingido!")
        print(f"│  Canibalização: {est.canibalizacao_pct}% | Halo: +{est.halo_pct}%")
        print(f"│  Lucro/dia: R$ {est.lucro_liquido_dia:.2f} (uplift {est.lucro_uplift:.2f} "
              f"+ halo {est.lucro_halo:.2f} − canib {-est.lucro_canibalizacao:.2f})")
        print(f"│  Custo operacional: R$ {est.custo_operacional:.2f}")
        print(f"│  LUCRO TOTAL: R$ {est.lucro_total_campanha:.2f} | ROI: {est.roi}×")
        if est.flags:
            print(f"│  ⚠ Flags: {', '.join(est.flags)}")
        print(f"│  📝 {est.justificativa}")
        print(f"└─────────────────────────────────────────────────")

        resultados.append({
            'id': cenario['id'],
            'cenario': cenario['nome'][:40],
            'esperado': cenario['esperado'][:30],
            'classificacao': est.classificacao,
            'tipo_produto': est.tipo_produto,
            'tipo_evento': est.tipo_evento,
            'd_base': est.demanda_base_dia,
            'd_promo': est.demanda_promocional_dia,
            'uplift_%': est.uplift_pct,
            'roi': est.roi,
            'lucro_total': est.lucro_total_campanha,
            'cap': est.cap_aplicado,
            'flags': ';'.join(est.flags),
        })

    # ───── Sumário ─────
    df = pd.DataFrame(resultados)
    df.to_csv(ROOT / 'testes_agente_v2_resultados.csv', index=False, encoding='utf-8')

    print()
    print("=" * 110)
    print("SUMÁRIO DOS TESTES")
    print("=" * 110)
    print(df[['id', 'cenario', 'classificacao', 'uplift_%', 'roi', 'lucro_total']].to_string(index=False))

    # Validação: o agente está classificando como esperado?
    print()
    print("=" * 110)
    print("VALIDAÇÃO: CLASSIFICAÇÃO vs EXPECTATIVA")
    print("=" * 110)
    for r, cen in zip(resultados, CENARIOS):
        esperado_simplificado = 'BOA' if 'BOA' in cen['esperado'] else \
                                'MÉDIA' if 'MÉDIA' in cen['esperado'] else \
                                'RUIM' if 'RUIM' in cen['esperado'] else \
                                'PROIBIDA'
        obtido_simplificado = 'BOA' if 'BOA' in r['classificacao'] else \
                              'MÉDIA' if 'MÉDIA' in r['classificacao'] else \
                              'RUIM' if 'RUIM' in r['classificacao'] else \
                              'PROIBIDA'
        ok = '✓' if esperado_simplificado == obtido_simplificado else '✗'
        print(f"  {ok} Cenário {r['id']}: esperado {esperado_simplificado:8s} → "
              f"obtido {obtido_simplificado:8s}")

    print()
    print(f"✓ Resultados salvos em testes_agente_v2_resultados.csv")


if __name__ == '__main__':
    main()
