"""Stress test do DemandAgent V3 — 25 cenários adversariais.

Testa casos limítrofes:
- Datas extremas (1 jan, 31 dez)
- Eventos sobrepostos
- Combos exóticos
- Inputs incompletos (sem evento, sem par)
- Valores limites (uplift estourando cap, canib máxima)
- Combinações raras (puxador_premium em consumo_intenso)

Objetivo: confirmar que o agente é ESTÁVEL (não quebra, não retorna NaN,
não dá uplift absurdo, sempre classifica algo).
"""
import io
import sys
import math
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from demand_agent import DemandAgent

ROOT = Path(__file__).parent

CENARIOS_STRESS = [
    # ─── DATAS EXTREMAS ───
    {'id': 'STR-01', 'desc': 'Réveillon 31/12', 'campanha': {
        'categoria': 'gelo', 'produto_complementar': 'cerveja', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 1, 'data_inicio': '2026-12-31', 'data_fim': '2026-12-31',
        'eventos_comerciais_na_janela': ['Réveillon'],
    }},
    {'id': 'STR-02', 'desc': '1º de janeiro pós-festa', 'campanha': {
        'categoria': 'agua', 'intensidade': 'desc5%',
        'desconto_pct': 5, 'dias_total': 1, 'data_inicio': '2026-01-01', 'data_fim': '2026-01-01',
        'eventos_comerciais_na_janela': [],
    }},
    {'id': 'STR-03', 'desc': 'Campanha de 30 dias (longa)', 'campanha': {
        'categoria': 'biscoito', 'intensidade': 'desc5%',
        'desconto_pct': 5, 'dias_total': 30, 'data_inicio': '2026-09-01', 'data_fim': '2026-09-30',
        'eventos_comerciais_na_janela': [],
    }},
    # ─── EVENTOS SOBREPOSTOS ───
    {'id': 'STR-04', 'desc': 'Réveillon + Natal sobrepostos', 'campanha': {
        'categoria': 'chocolate_premium', 'produto_complementar': 'vinho', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 7, 'data_inicio': '2026-12-25', 'data_fim': '2026-12-31',
        'eventos_comerciais_na_janela': ['Véspera de Natal', 'Réveillon'],
    }},
    # ─── COMBOS EXÓTICOS ───
    {'id': 'STR-05', 'desc': 'Combo vinho + água (estranho mas possível)', 'campanha': {
        'categoria': 'vinho', 'produto_complementar': 'agua', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 3, 'data_inicio': '2026-05-15', 'data_fim': '2026-05-17',
        'eventos_comerciais_na_janela': [],
    }},
    {'id': 'STR-06', 'desc': 'Combo sorvete + sorvete (mesmo produto)', 'campanha': {
        'categoria': 'sorvete', 'produto_complementar': 'sorvete', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 3, 'data_inicio': '2026-12-15', 'data_fim': '2026-12-17',
        'eventos_comerciais_na_janela': [],
    }},
    {'id': 'STR-07', 'desc': 'Combo isotônico + energético (consumo coerente)', 'campanha': {
        'categoria': 'isotonico', 'produto_complementar': 'energetico', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 2, 'data_inicio': '2026-06-15', 'data_fim': '2026-06-16',
        'eventos_comerciais_na_janela': ['Copa 2026 — Estreia provável Brasil'],
    }},
    # ─── INPUTS INCOMPLETOS ───
    {'id': 'STR-08', 'desc': 'Combo SEM par definido', 'campanha': {
        'categoria': 'cerveja', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 3, 'data_inicio': '2026-08-15', 'data_fim': '2026-08-17',
    }},
    {'id': 'STR-09', 'desc': 'Sem desconto (desc=0)', 'campanha': {
        'categoria': 'cafe', 'intensidade': 'nada',
        'desconto_pct': 0, 'dias_total': 3, 'data_inicio': '2026-05-15', 'data_fim': '2026-05-17',
    }},
    {'id': 'STR-10', 'desc': 'Intensidade desconhecida (inválida)', 'campanha': {
        'categoria': 'gelo', 'intensidade': 'super_promo_99',  # inexistente
        'desconto_pct': 50, 'dias_total': 3, 'data_inicio': '2026-08-15', 'data_fim': '2026-08-17',
    }},
    # ─── VALORES LIMITES ───
    {'id': 'STR-11', 'desc': 'Desc 25% combo (cap forte)', 'campanha': {
        'categoria': 'gelo', 'produto_complementar': 'cerveja', 'intensidade': 'combo',
        'desconto_pct': 25, 'dias_total': 4, 'data_inicio': '2027-01-15', 'data_fim': '2027-01-18',
        'eventos_comerciais_na_janela': [],
    }},
    {'id': 'STR-12', 'desc': 'Liquidação 25% validade 5% restante (defensiva extrema)', 'campanha': {
        'categoria': 'cerveja', 'intensidade': 'liq25%',
        'desconto_pct': 25, 'dias_total': 2, 'data_inicio': '2026-06-15', 'data_fim': '2026-06-16',
        'validade_restante_pct': 0.05, 'estoque_pct_normal': 2.5,
    }},
    {'id': 'STR-13', 'desc': 'Estoque 5x normal (super parado)', 'campanha': {
        'categoria': 'biscoito', 'intensidade': 'desc10%',
        'desconto_pct': 10, 'dias_total': 5, 'data_inicio': '2026-09-15', 'data_fim': '2026-09-19',
        'estoque_pct_normal': 5.0, 'giro_alto': False,
    }},
    # ─── COMBINAÇÕES RARAS ───
    {'id': 'STR-14', 'desc': 'Puxador premium em consumo intenso (vinho na Copa)', 'campanha': {
        'categoria': 'vinho', 'intensidade': 'desc10%',
        'desconto_pct': 10, 'dias_total': 3, 'data_inicio': '2026-06-15', 'data_fim': '2026-06-17',
        'eventos_comerciais_na_janela': ['Copa 2026 — Estreia provável Brasil'],
    }},
    {'id': 'STR-15', 'desc': 'Café em Páscoa (rotina em presente)', 'campanha': {
        'categoria': 'cafe', 'intensidade': 'desc5%',
        'desconto_pct': 5, 'dias_total': 3, 'data_inicio': '2027-04-01', 'data_fim': '2027-04-03',
        'eventos_comerciais_na_janela': ['Páscoa'],
    }},
    # ─── PRODUTO ALTA DEMANDA EM EVENTO ───
    {'id': 'STR-16', 'desc': 'Cerveja desc10% no Dia dos Pais (puxador em evento)', 'campanha': {
        'categoria': 'cerveja', 'intensidade': 'desc10%',
        'desconto_pct': 10, 'dias_total': 4, 'data_inicio': '2026-08-06', 'data_fim': '2026-08-09',
        'eventos_comerciais_na_janela': ['Dia dos Pais'],
    }},
    {'id': 'STR-17', 'desc': 'Gelo combo Carnaval (consumo intenso)', 'campanha': {
        'categoria': 'gelo', 'produto_complementar': 'cerveja', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 5, 'data_inicio': '2027-02-12', 'data_fim': '2027-02-16',
        'eventos_comerciais_na_janela': ['Carnaval'],
    }},
    # ─── COMBOS SUPER FORTES ───
    {'id': 'STR-18', 'desc': 'Cerveja + Gelo combo verão (par PERFEITO)', 'campanha': {
        'categoria': 'cerveja', 'produto_complementar': 'gelo', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 3, 'data_inicio': '2027-01-10', 'data_fim': '2027-01-12',
        'eventos_comerciais_na_janela': [],
    }},
    {'id': 'STR-19', 'desc': 'Destilados + Vinho combo (harmonia 2.5!)', 'campanha': {
        'categoria': 'destilados', 'produto_complementar': 'vinho', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 4, 'data_inicio': '2026-12-28', 'data_fim': '2026-12-31',
        'eventos_comerciais_na_janela': ['Réveillon'],
    }},
    # ─── PRODUTOS DE BAIXO VOLUME ───
    {'id': 'STR-20', 'desc': 'Vinho desc25% em Mães (volume baixo + evento certo)', 'campanha': {
        'categoria': 'vinho', 'intensidade': 'liq25%',
        'desconto_pct': 25, 'dias_total': 5, 'data_inicio': '2027-05-05', 'data_fim': '2027-05-09',
        'eventos_comerciais_na_janela': ['Dia das Mães'],
    }},
    # ─── EVENTO PERFEITO MAS PRODUTO ERRADO ───
    {'id': 'STR-21', 'desc': 'Cerveja em Dia das Mães (puxador_consumo em presente)', 'campanha': {
        'categoria': 'cerveja', 'intensidade': 'combo', 'produto_complementar': 'snack',
        'desconto_pct': 10, 'dias_total': 4, 'data_inicio': '2027-05-05', 'data_fim': '2027-05-09',
        'eventos_comerciais_na_janela': ['Dia das Mães'],
    }},
    # ─── DESC ALTO EM PRODUTO QUE NÃO PRECISA ───
    {'id': 'STR-22', 'desc': 'Sorvete liq25% em DEZ verão (já vende sozinho)', 'campanha': {
        'categoria': 'sorvete', 'intensidade': 'liq25%',
        'desconto_pct': 25, 'dias_total': 4, 'data_inicio': '2026-12-15', 'data_fim': '2026-12-18',
        'eventos_comerciais_na_janela': [],
        # Sem validade próxima — só desperdiço margem
    }},
    # ─── PRODUTO ROTINA SEM EVENTO ───
    {'id': 'STR-23', 'desc': 'Padaria desc5% manhã útil (rotina)', 'campanha': {
        'categoria': 'padaria', 'intensidade': 'desc5%',
        'desconto_pct': 5, 'dias_total': 5, 'data_inicio': '2026-06-15', 'data_fim': '2026-06-19',
        'eventos_comerciais_na_janela': [],
    }},
    # ─── PROIBIDA EM EVENTO ───
    {'id': 'STR-24', 'desc': 'Cigarro em Pais (proibida + evento — ainda assim proibida)', 'campanha': {
        'categoria': 'cigarro_philip_morris', 'intensidade': 'desc10%',
        'desconto_pct': 10, 'dias_total': 4, 'data_inicio': '2026-08-06', 'data_fim': '2026-08-09',
        'eventos_comerciais_na_janela': ['Dia dos Pais'],
    }},
    # ─── CATEGORIA DESCONHECIDA ───
    {'id': 'STR-25', 'desc': 'Categoria inexistente (robustez)', 'campanha': {
        'categoria': 'cosmeticos_premium', 'intensidade': 'combo',
        'desconto_pct': 10, 'dias_total': 3, 'data_inicio': '2026-05-15', 'data_fim': '2026-05-17',
        'eventos_comerciais_na_janela': [],
    }},
]


def main():
    agent = DemandAgent()

    print("=" * 110)
    print(f"STRESS TEST — {len(CENARIOS_STRESS)} cenários adversariais")
    print("=" * 110)

    resultados = []
    erros = []

    for cen in CENARIOS_STRESS:
        try:
            est = agent.estimar(cen['campanha'])

            # Validação básica de estabilidade
            checks = {
                'demanda_base_valida': est.demanda_base_dia >= 0 and not math.isnan(est.demanda_base_dia),
                'demanda_promo_valida': est.demanda_promocional_dia >= 0 and not math.isnan(est.demanda_promocional_dia),
                'uplift_realista': -50 <= est.uplift_pct <= 130,  # caps reais permitem até ~120%
                'canib_valida': 0 <= est.canibalizacao_estimada_pct <= 100,
                'classificacao_existe': est.qualidade_promocao in ['🟢 BOA', '🟡 MÉDIA', '🔴 RUIM', '🚫 PROIBIDA'],
                'motivo_existe': len(est.motivo) > 10,
                'componentes_completo': est.tipo_produto == 'proibida' or all(
                    k in est.componentes
                    for k in ['boost_preco', 'boost_combo', 'boost_evento', 'boost_clima']
                ),
            }
            falhas = [k for k, v in checks.items() if not v]

            resultados.append({
                'id': cen['id'],
                'desc': cen['desc'][:45],
                'qualidade': est.qualidade_promocao,
                'd_base': est.demanda_base_dia,
                'd_promo': est.demanda_promocional_dia,
                'uplift_%': est.uplift_pct,
                'canib_%': est.canibalizacao_estimada_pct,
                'cap': est.cap_aplicado,
                'falhas': ';'.join(falhas) if falhas else 'OK',
                'flags': ';'.join(est.flags),
            })

            ok = '✓' if not falhas else '⚠'
            print(f"  {ok} {cen['id']} {cen['desc'][:50]:50s} | {est.qualidade_promocao:10s} | "
                  f"d {est.demanda_base_dia:.1f}→{est.demanda_promocional_dia:.1f} | "
                  f"uplift {est.uplift_pct:+5.1f}% | canib {est.canibalizacao_estimada_pct:.0f}%")
            if falhas:
                print(f"     FALHAS: {falhas}")

        except Exception as e:
            erros.append({'id': cen['id'], 'desc': cen['desc'], 'erro': str(e)})
            print(f"  ✗ {cen['id']} CRASH: {e}")

    df = pd.DataFrame(resultados)
    df.to_csv(ROOT / 'stress_test_resultados.csv', index=False, encoding='utf-8')

    print()
    print("=" * 110)
    print("RESUMO DO STRESS TEST")
    print("=" * 110)
    print(f"  Cenários testados:       {len(CENARIOS_STRESS)}")
    print(f"  Crashes (excpetions):    {len(erros)}")
    sem_falha = sum(1 for r in resultados if r['falhas'] == 'OK')
    print(f"  Validação 100% (OK):     {sem_falha}/{len(resultados)}")
    com_falhas = [r for r in resultados if r['falhas'] != 'OK']
    if com_falhas:
        print("\nCASOS COM FALHAS:")
        for r in com_falhas:
            print(f"  - {r['id']}: {r['falhas']}")

    # Distribuição de classificações
    print("\nDISTRIBUIÇÃO DE CLASSIFICAÇÕES:")
    print(df['qualidade'].value_counts().to_string())

    if erros:
        print("\n⚠ CRASHES (precisam correção):")
        for e in erros:
            print(f"  - {e['id']}: {e['erro']}")
    else:
        print("\n🎉 Nenhum crash. Agente ESTÁVEL.")


if __name__ == '__main__':
    main()
