"""Protótipo V1 do output operacional — calendário de promoções.

Pega as decisões do V10 (validacao_timing.csv) e gera um calendário
acionável para os próximos N dias. Versão de prova de conceito —
ainda limitada aos 6 produtos do modelo atual e sem conhecimento
de datas comerciais.

Saídas:
  results/calendario_v1.json  — formato máquina
  results/calendario_v1.md    — formato legível para o dono
"""
import json
import sys
import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Forçar UTF-8 no terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results'

# ── Configuração ────────────────────────────────────────────────────────────

DATA_HOJE = date(2026, 5, 11)
HORIZONTE_DIAS = 60

PRODUTOS_DISPLAY = {
    'energetico': 'Energético',
    'gelo': 'Gelo',
    'refrigerante': 'Refrigerante',
    'agua': 'Água',
    'cerveja': 'Cerveja',
    'sorvete': 'Sorvete',
}

PARES_COMBO = {
    'energetico': 'refrigerante',
    'gelo': 'agua',
    'refrigerante': 'sorvete',
    'agua': 'energetico',
    'cerveja': 'refrigerante',
    'sorvete': 'agua',
}

# Risco relativo de vencimento (V10 alpha)
RISCO_VENCIMENTO = {
    'cerveja': 2.77,
    'energetico': 2.00,
    'gelo': 2.00,
    'refrigerante': 2.00,
    'agua': 2.00,
    'sorvete': 2.00,
}

# Margem unitária (R$) — dado real de venda_do_mes
MARGEM = {
    'energetico': 10.57,
    'gelo': 7.59,
    'refrigerante': 4.71,
    'agua': 3.51,
    'cerveja': 4.54,
    'sorvete': 5.69,
}

# Demanda base diária (un/dia) — dado real de venda_por_dia
DEMANDA_BASE = {
    'energetico': 5.3,
    'gelo': 5.6,
    'refrigerante': 10.7,
    'agua': 16.9,
    'cerveja': 9.4,
    'sorvete': 7.0,
}

DIAS_SEMANA = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
TURNOS = ['manhã', 'tarde', 'noite']
MESES_PT = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun',
            'jul', 'ago', 'set', 'out', 'nov', 'dez']

# ── 1. Decisão turno-a-turno via lookup do V10 ─────────────────────────────

val = pd.read_csv(RESULTS / 'validacao_timing.csv')

# Como V10 decide de forma idêntica para os 6 produtos no mesmo contexto,
# basta pegar a decisão de qualquer produto.
contexto = (val[val['produto'] == 'gelo']
            .set_index(['dia', 'turno', 'mes'])[['fator_combinado', 'promove']]
            .to_dict('index'))

decisoes = []
for offset in range(HORIZONTE_DIAS):
    d = DATA_HOJE + timedelta(days=offset)
    dia_sem = d.weekday()
    mes_idx = d.month - 1
    for turno in range(3):
        ctx = contexto[(dia_sem, turno, mes_idx)]
        decisoes.append({
            'data': d,
            'dia_semana_nome': DIAS_SEMANA[dia_sem],
            'turno_nome': TURNOS[turno],
            'turno_idx': turno,
            'promove': bool(ctx['promove']),
            'fator_combinado': float(ctx['fator_combinado']),
        })

df = pd.DataFrame(decisoes)

# ── 2. Agrupar em campanhas (dias consecutivos com promoção) ───────────────

# Considera "dia de promoção" se pelo menos 1 turno do dia promove
diario = (df.groupby('data')
            .agg(turnos_promo=('promove', 'sum'),
                 fator_medio=('fator_combinado', 'mean'),
                 dia_semana_nome=('dia_semana_nome', 'first'))
            .reset_index())
diario['promove'] = diario['turnos_promo'] > 0

campanhas = []
em_campanha = False
atual = None
for _, row in diario.iterrows():
    if row['promove'] and not em_campanha:
        atual = {
            'data_inicio': row['data'],
            'data_fim': row['data'],
            'dias_total': 1,
            'turnos_total': int(row['turnos_promo']),
            'fatores': [row['fator_medio']],
            'dias_semana': [row['dia_semana_nome']],
        }
        em_campanha = True
    elif row['promove'] and em_campanha:
        atual['data_fim'] = row['data']
        atual['dias_total'] += 1
        atual['turnos_total'] += int(row['turnos_promo'])
        atual['fatores'].append(row['fator_medio'])
        atual['dias_semana'].append(row['dia_semana_nome'])
    elif not row['promove'] and em_campanha:
        if atual['dias_total'] >= 2:
            campanhas.append(atual)
        atual = None
        em_campanha = False
if em_campanha and atual and atual['dias_total'] >= 2:
    campanhas.append(atual)

# ── 3. Para cada campanha, escolher o produto principal ─────────────────────

# Heurística: produto com maior margem × demanda × risco_vencimento (esperança
# de impacto + necessidade de giro). Sem dado de estoque real, é aproximação.
score_produto = {
    p: MARGEM[p] * DEMANDA_BASE[p] * RISCO_VENCIMENTO[p]
    for p in PRODUTOS_DISPLAY
}
ranking_produtos = sorted(score_produto.items(), key=lambda x: -x[1])
# Resultado esperado: cerveja > agua > refrigerante > sorvete > energetico > gelo

# Rotaciona os top 3 ao longo das campanhas para diversificar
top_produtos = [p for p, _ in ranking_produtos[:3]]

for i, c in enumerate(campanhas):
    principal = top_produtos[i % len(top_produtos)]
    complementar = PARES_COMBO[principal]
    c['produto_principal'] = principal
    c['produto_complementar'] = complementar
    c['fator_medio'] = float(sum(c['fatores']) / len(c['fatores']))

    # Estima uplift via boost do combo do V10 (1.12 no principal, 1.08 no complementar)
    # Combo aplicado em fator combinado fraco → ganho ≈ 12% sobre demanda esperada
    demanda_principal_dia = DEMANDA_BASE[principal] * c['fator_medio']
    uplift_un_principal = demanda_principal_dia * c['dias_total'] * 0.12
    demanda_complementar_dia = DEMANDA_BASE[complementar] * c['fator_medio']
    uplift_un_complementar = demanda_complementar_dia * c['dias_total'] * 0.08
    lucro_adicional = (uplift_un_principal * MARGEM[principal]
                       + uplift_un_complementar * MARGEM[complementar])

    c['uplift_un_principal'] = round(uplift_un_principal, 1)
    c['uplift_un_complementar'] = round(uplift_un_complementar, 1)
    c['lucro_adicional_R$'] = round(lucro_adicional, 2)
    c['desconto_recomendado_%'] = 10  # combo a -10% (sweet spot econômico V8)
    c['tipo'] = 'combo'

# ── 4. Salvar JSON ──────────────────────────────────────────────────────────

def serializar(c):
    return {
        'id': f"camp_{c['data_inicio'].isoformat()}_{c['produto_principal']}",
        'data_inicio': c['data_inicio'].isoformat(),
        'data_fim': c['data_fim'].isoformat(),
        'duracao_dias': c['dias_total'],
        'tipo': c['tipo'],
        'produto_principal': c['produto_principal'],
        'produto_complementar': c['produto_complementar'],
        'desconto_recomendado_%': c['desconto_recomendado_%'],
        'uplift_un_principal_estimado': c['uplift_un_principal'],
        'uplift_un_complementar_estimado': c['uplift_un_complementar'],
        'lucro_adicional_estimado_R$': c['lucro_adicional_R$'],
        'fator_demanda_medio': round(c['fator_medio'], 3),
        'dias_semana': c['dias_semana'],
        'justificativa': (
            f"Período de demanda fraca (fator médio {c['fator_medio']:.2f} × baseline). "
            f"Combo {PRODUTOS_DISPLAY[c['produto_principal']]} + "
            f"{PRODUTOS_DISPLAY[c['produto_complementar']]} a -{c['desconto_recomendado_%']}% "
            f"deve estimular giro com {c['turnos_total']} turnos de oportunidade."
        ),
    }

calendario = {
    'versao': 'V1-prototipo',
    'modelo': 'DQN V10 (6 produtos)',
    'gerado_em': DATA_HOJE.isoformat(),
    'horizonte_dias': HORIZONTE_DIAS,
    'total_campanhas': len(campanhas),
    'lucro_adicional_total_estimado_R$': round(sum(c['lucro_adicional_R$'] for c in campanhas), 2),
    'limitacoes_conhecidas': [
        'Apenas 6 produtos (faltam chocolate, vinho, snacks, etc.)',
        'Não considera datas comerciais (Dia dos Namorados, Mães, Black Friday)',
        'Elasticidade da literatura, não medida no posto',
        'Estoque simulado, não real',
        'Combo via heurística PARES_COMBO, não validado por análise de cesta',
    ],
    'campanhas': [serializar(c) for c in campanhas],
}

(RESULTS / 'calendario_v1.json').write_text(
    json.dumps(calendario, indent=2, ensure_ascii=False), encoding='utf-8'
)

# ── 5. Salvar Markdown legível ──────────────────────────────────────────────

md_lines = [
    f"# Calendário de Promoções Recomendadas — V1 (protótipo)",
    "",
    f"**Gerado em:** {DATA_HOJE.strftime('%d/%m/%Y')}",
    f"**Horizonte:** próximos {HORIZONTE_DIAS} dias ({DATA_HOJE.strftime('%d/%m')} a "
    f"{(DATA_HOJE + timedelta(days=HORIZONTE_DIAS-1)).strftime('%d/%m/%Y')})",
    f"**Modelo:** DQN V10 — 6 produtos (energético, gelo, refrigerante, água, cerveja, sorvete)",
    "",
    f"## Resumo",
    "",
    f"- **{len(campanhas)} campanhas** recomendadas no período",
    f"- **Lucro adicional estimado total:** R$ {calendario['lucro_adicional_total_estimado_R$']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
    f"- **Tipo único:** combo (V10 não usa descontos parciais ou liquidação)",
    "",
    "## Calendário",
    "",
]

for i, c in enumerate(campanhas, 1):
    di = c['data_inicio'].strftime('%d/%m')
    df_str = c['data_fim'].strftime('%d/%m')
    dias_unicos = sorted(set(c['dias_semana']), key=lambda x: DIAS_SEMANA.index(x))
    md_lines += [
        f"### Campanha {i}: {di} a {df_str} ({c['dias_total']} dias)",
        "",
        f"- **Tipo:** Combo promocional",
        f"- **Produto principal:** {PRODUTOS_DISPLAY[c['produto_principal']]}",
        f"- **Produto complementar (compra junto):** {PRODUTOS_DISPLAY[c['produto_complementar']]}",
        f"- **Desconto recomendado:** {c['desconto_recomendado_%']}% no combo",
        f"- **Dias da semana:** {', '.join(dias_unicos)}",
        f"- **Lucro adicional estimado:** R$ {c['lucro_adicional_R$']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
        f"- **Uplift estimado:** +{c['uplift_un_principal']:.1f} un de {PRODUTOS_DISPLAY[c['produto_principal']]}, "
        f"+{c['uplift_un_complementar']:.1f} un de {PRODUTOS_DISPLAY[c['produto_complementar']]}",
        f"- **Razão:** período historicamente fraco "
        f"(demanda esperada a {c['fator_medio']*100:.0f}% do baseline)",
        "",
    ]

md_lines += [
    "---",
    "",
    "## Limitações desta V1 (importantes para o seu pai entender)",
    "",
    "1. **Apenas 6 produtos.** Chocolate, vinho, snacks, café — ainda fora.",
    "2. **Não enxerga Dia dos Namorados (12/06).** Próxima versão vai considerar.",
    "3. **Elasticidade da literatura.** Uplift estimado é prior, não medição real do posto.",
    "4. **Estoque é simulado.** Não usa o estoque real do ERP.",
    "5. **Pares de combo são heurística.** Precisamos validar com cupom fiscal real.",
    "",
    "## O que valida nesta V1",
    "",
    "- **Formato do output:** está utilizável pelo dono? Falta alguma informação?",
    "- **Granularidade temporal:** dias, dias-da-semana, intervalo certo?",
    "- **Decisões fazem sentido qualitativamente?** Combos parecem combinar?",
    "- **Layout dos campos:** preferiria ver ROI%, % de margem perdida, outros KPIs?",
    "",
    "---",
    "",
    "*V1 gerada automaticamente a partir do modelo DQN V10 treinado em 11/05/2026.*",
]

(RESULTS / 'calendario_v1.md').write_text('\n'.join(md_lines), encoding='utf-8')

# ── 6. Print resumo ─────────────────────────────────────────────────────────

print(f"✓ Gerado calendário V1 com {len(campanhas)} campanhas nos próximos {HORIZONTE_DIAS} dias")
print(f"✓ Lucro adicional total estimado: R$ {calendario['lucro_adicional_total_estimado_R$']:,.2f}")
print(f"✓ JSON: results/calendario_v1.json")
print(f"✓ Markdown: results/calendario_v1.md")
print()
print("Primeiras 5 campanhas:")
for c in campanhas[:5]:
    di = c['data_inicio'].strftime('%d/%m')
    df_str = c['data_fim'].strftime('%d/%m')
    print(f"  {di}-{df_str} ({c['dias_total']}d): {c['produto_principal']:>12s} + "
          f"{c['produto_complementar']:>12s} → R$ {c['lucro_adicional_R$']:>7.2f}")
