"""V2 do output operacional — V10 cruzado com calendário comercial.

Diferenças da V1:
- Detecta datas comerciais e enriquece campanhas que caem em janela de evento
- Alerta quando V10 NÃO recomenda promoção em janela de evento (limitação
  conhecida: V10 não tem chocolate/vinho/espumante/snack no catálogo)
- Usa uplift medido do Google Trends quando disponível

Saída: results/calendario_v2.json + results/calendario_v2.md
"""
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

# ── Configuração ────────────────────────────────────────────────────────────

DATA_HOJE = date(2026, 5, 11)
HORIZONTE_DIAS = 60

PRODUTOS_DISPLAY = {
    'energetico': 'Energético', 'gelo': 'Gelo', 'refrigerante': 'Refrigerante',
    'agua': 'Água', 'cerveja': 'Cerveja', 'sorvete': 'Sorvete',
}
PARES_COMBO = {
    'energetico': 'refrigerante', 'gelo': 'agua', 'refrigerante': 'sorvete',
    'agua': 'energetico', 'cerveja': 'refrigerante', 'sorvete': 'agua',
}
MARGEM = {'energetico': 10.57, 'gelo': 7.59, 'refrigerante': 4.71,
          'agua': 3.51, 'cerveja': 4.54, 'sorvete': 5.69}
DEMANDA_BASE = {'energetico': 5.3, 'gelo': 5.6, 'refrigerante': 10.7,
                'agua': 16.9, 'cerveja': 9.4, 'sorvete': 7.0}

# Mapeamento: categoria do calendário → produto do catálogo V10
# Se a categoria não tiver match exato, marca como "fora do catálogo"
CATEGORIAS_NO_CATALOGO = {
    'cerveja': 'cerveja',
    'cerveja_premium': 'cerveja',
    'refrigerante': 'refrigerante',
    'energetico': 'energetico',
    'agua': 'agua',
    'gelo': 'gelo',
    'sorvete': 'sorvete',
}

DIAS_SEMANA = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']

# ── 1. Carregar dados base ──────────────────────────────────────────────────

val = pd.read_csv(RESULTS / 'validacao_timing.csv')
ctx_v10 = (val[val['produto'] == 'gelo']
           .set_index(['dia', 'turno', 'mes'])[['fator_combinado', 'promove']]
           .to_dict('index'))

cal = pd.read_csv(DATA / 'calendario_comercial.csv', parse_dates=['data'])
cal['data'] = pd.to_datetime(cal['data']).dt.date
cal_exp = pd.read_csv(DATA / 'calendario_comercial_expandido.csv', parse_dates=['data'])
cal_exp['data'] = pd.to_datetime(cal_exp['data']).dt.date

# Uplift medido do Google Trends (pode não cobrir todos eventos)
try:
    uplift_trends = pd.read_csv(DATA / 'priors_externos' / 'uplift_trends_agregado.csv')
    uplift_dict = {}
    for _, row in uplift_trends.iterrows():
        uplift_dict[(row['categoria_modelo'], row['evento'])] = row['uplift_medio']
except FileNotFoundError:
    uplift_dict = {}

# ── 2. Eventos relevantes no horizonte ──────────────────────────────────────

horizonte_fim = DATA_HOJE + timedelta(days=HORIZONTE_DIAS)
eventos_horizonte = []
for _, ev in cal.iterrows():
    # Considera evento se janela [data-pre, data+pos] intersecta o horizonte
    inicio_janela = ev['data'] - timedelta(days=int(ev['janela_pre_dias']))
    fim_janela = ev['data'] + timedelta(days=int(ev['janela_pos_dias']))
    if fim_janela < DATA_HOJE or inicio_janela > horizonte_fim:
        continue
    if ev['tipo_evento'] == 'feriado_oficial':
        continue
    eventos_horizonte.append({
        'data': ev['data'],
        'evento': ev['nome_evento'],
        'tipo': ev['tipo_evento'],
        'categorias': ev['categorias_afetadas'].split(';'),
        'uplift_prior': float(ev['uplift_prior']),
        'pre': int(ev['janela_pre_dias']),
        'pos': int(ev['janela_pos_dias']),
    })

# ── 3. Para cada evento, classificar a cobertura do V10 ────────────────────

def classificar_evento(ev):
    """Determina se V10 cobre o evento ou se é uma cegueira."""
    cats = ev['categorias']
    # Resolve cada categoria do calendário para o SKU correspondente no V10
    cobertos = []
    for c in cats:
        if c in CATEGORIAS_NO_CATALOGO:
            sku = CATEGORIAS_NO_CATALOGO[c]
            if sku not in cobertos:
                cobertos.append(sku)
    nao_cobertos = [c for c in cats if c not in CATEGORIAS_NO_CATALOGO
                    and c != 'todas']

    if 'todas' in cats:
        return 'parcial', cobertos, nao_cobertos
    if cobertos and not nao_cobertos:
        return 'total', cobertos, []
    if cobertos and nao_cobertos:
        return 'parcial', cobertos, nao_cobertos
    return 'fora_catalogo', [], nao_cobertos


eventos_anotados = []
for ev in eventos_horizonte:
    cobertura, cobertos, nao_cobertos = classificar_evento(ev)

    # V10 promoveria nessa janela?
    janela_inicio = ev['data'] - timedelta(days=ev['pre'])
    janela_fim = ev['data'] + timedelta(days=ev['pos'])
    v10_promo_dias = 0
    v10_total_dias = 0
    for offset in range((janela_fim - janela_inicio).days + 1):
        d = janela_inicio + timedelta(days=offset)
        if d < DATA_HOJE:
            continue
        v10_total_dias += 1
        for turno in range(3):
            if ctx_v10[(d.weekday(), turno, d.month - 1)]['promove']:
                v10_promo_dias += 1
                break

    # Uplift medido (se houver) ou prior do calendário
    uplifts_medidos = []
    for c in cobertos:
        chave = (c, ev['evento'])
        if chave in uplift_dict:
            uplifts_medidos.append((c, uplift_dict[chave]))

    eventos_anotados.append({
        **ev,
        'cobertura_v10': cobertura,
        'categorias_no_catalogo': cobertos,
        'categorias_fora_catalogo': nao_cobertos,
        'v10_dias_com_promo': v10_promo_dias,
        'v10_dias_total': v10_total_dias,
        'uplifts_medidos': uplifts_medidos,
    })

# ── 4. Construir recomendações ─────────────────────────────────────────────

recomendacoes = []

for ev in eventos_anotados:
    inicio = max(ev['data'] - timedelta(days=ev['pre']), DATA_HOJE)
    fim = min(ev['data'] + timedelta(days=ev['pos']), horizonte_fim)
    if inicio > fim:
        continue

    rec_base = {
        'data_inicio': inicio.isoformat(),
        'data_fim': fim.isoformat(),
        'data_evento': ev['data'].isoformat(),
        'evento': ev['evento'],
        'tipo_evento': ev['tipo'],
        'duracao_dias': (fim - inicio).days + 1,
    }

    if ev['cobertura_v10'] == 'total':
        rec_base['classificacao'] = 'cobertura_total_v10'
        rec_base['acao_recomendada'] = (
            f"V10 já cobre este evento — promover {', '.join(ev['categorias_no_catalogo'])} "
            f"durante a janela. Combo recomendado com produtos do catálogo."
        )
        # Usar primeiro produto coberto como principal
        principal = ev['categorias_no_catalogo'][0]
        rec_base['produto_principal'] = principal
        rec_base['produto_complementar'] = PARES_COMBO.get(principal, '')
        rec_base['desconto_recomendado_%'] = 10

    elif ev['cobertura_v10'] == 'parcial':
        rec_base['classificacao'] = 'cobertura_parcial'
        cobertos_str = ', '.join(ev['categorias_no_catalogo']) or 'nenhum específico'
        fora_str = ', '.join(ev['categorias_fora_catalogo'])
        rec_base['acao_recomendada'] = (
            f"V10 cobre parcialmente. Pode promover {cobertos_str} do catálogo atual, "
            f"mas o pico real é em {fora_str} (FORA do modelo). "
            f"ATENÇÃO: priorizar expandir catálogo para incluir essas categorias."
        )
        if ev['categorias_no_catalogo']:
            principal = ev['categorias_no_catalogo'][0]
            rec_base['produto_principal'] = principal
            rec_base['produto_complementar'] = PARES_COMBO.get(principal, '')
            rec_base['desconto_recomendado_%'] = 10
        rec_base['categorias_perdidas'] = ev['categorias_fora_catalogo']

    else:  # fora_catalogo
        rec_base['classificacao'] = 'cegueira_v10'
        rec_base['acao_recomendada'] = (
            f"V10 NÃO TEM produtos relacionados a este evento no catálogo. "
            f"Oportunidade perdida: {', '.join(ev['categorias_fora_catalogo'])}. "
            f"Solução: expandir catálogo na próxima versão."
        )
        rec_base['categorias_perdidas'] = ev['categorias_fora_catalogo']

    # Adicionar info de uplift
    if ev['uplifts_medidos']:
        rec_base['uplift_medido_trends'] = {
            cat: round(u, 2) for cat, u in ev['uplifts_medidos']
        }
    rec_base['uplift_prior'] = round(ev['uplift_prior'], 2)
    rec_base['v10_dias_com_promo'] = ev['v10_dias_com_promo']
    rec_base['v10_dias_total'] = ev['v10_dias_total']

    recomendacoes.append(rec_base)

# ── 5. Salvar JSON ──────────────────────────────────────────────────────────

resumo = {
    'versao': 'V2-calendario-comercial',
    'modelo_base': 'DQN V10 (6 produtos) + calendário comercial BR',
    'gerado_em': DATA_HOJE.isoformat(),
    'horizonte_dias': HORIZONTE_DIAS,
    'eventos_detectados': len(recomendacoes),
    'eventos_v10_cobre_total': sum(1 for r in recomendacoes
                                    if r['classificacao'] == 'cobertura_total_v10'),
    'eventos_v10_cobre_parcial': sum(1 for r in recomendacoes
                                      if r['classificacao'] == 'cobertura_parcial'),
    'eventos_v10_nao_cobre': sum(1 for r in recomendacoes
                                  if r['classificacao'] == 'cegueira_v10'),
    'recomendacoes': recomendacoes,
}

(RESULTS / 'calendario_v2.json').write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False, default=str),
    encoding='utf-8'
)

# ── 6. Markdown legível ─────────────────────────────────────────────────────

md = [
    f"# Calendário de Promoções V2 — com datas comerciais",
    "",
    f"**Gerado em:** {DATA_HOJE.strftime('%d/%m/%Y')}",
    f"**Horizonte:** {HORIZONTE_DIAS} dias",
    f"**Modelo base:** DQN V10 (6 produtos) + Calendário comercial BR + Google Trends",
    "",
    "## Eventos comerciais detectados no horizonte",
    "",
    f"- **Total:** {resumo['eventos_detectados']} eventos relevantes",
    f"- **V10 cobre totalmente:** {resumo['eventos_v10_cobre_total']}",
    f"- **V10 cobre parcialmente:** {resumo['eventos_v10_cobre_parcial']}",
    f"- **V10 NÃO enxerga:** {resumo['eventos_v10_nao_cobre']} ← oportunidades perdidas",
    "",
    "---",
    "",
]

# Ordenar por data
recomendacoes_ord = sorted(recomendacoes, key=lambda r: r['data_inicio'])

for i, r in enumerate(recomendacoes_ord, 1):
    di = pd.Timestamp(r['data_inicio']).strftime('%d/%m')
    df_s = pd.Timestamp(r['data_fim']).strftime('%d/%m')
    de = pd.Timestamp(r['data_evento']).strftime('%d/%m')
    classif = r['classificacao']

    if classif == 'cobertura_total_v10':
        emoji = '✓'
        cor = 'V10 cobre'
    elif classif == 'cobertura_parcial':
        emoji = '⚠'
        cor = 'parcial'
    else:
        emoji = '✗'
        cor = 'cegueira'

    md += [
        f"### {emoji} {r['evento']} (evento: {de}, campanha: {di}-{df_s})",
        "",
        f"- **Status do V10:** {cor}",
        f"- **Duração da campanha:** {r['duracao_dias']} dias",
        f"- **Uplift esperado (prior):** {r['uplift_prior']:.2f}×",
    ]
    if 'uplift_medido_trends' in r:
        for cat, u in r['uplift_medido_trends'].items():
            md.append(f"- **Uplift medido em buscas ({cat}):** {u:.2f}×")
    md.append(f"- **V10 promoveria:** {r['v10_dias_com_promo']}/{r['v10_dias_total']} dias da janela")
    md.append(f"- **Recomendação:** {r['acao_recomendada']}")
    if 'produto_principal' in r:
        md.append(f"- **Combo do catálogo:** "
                  f"{PRODUTOS_DISPLAY.get(r['produto_principal'], r['produto_principal'])} + "
                  f"{PRODUTOS_DISPLAY.get(r['produto_complementar'], r['produto_complementar'])} "
                  f"a -{r['desconto_recomendado_%']}%")
    if 'categorias_perdidas' in r:
        md.append(f"- **Categorias que faltam no catálogo:** "
                  f"{', '.join(r['categorias_perdidas'])}")
    md.append("")

md += [
    "---",
    "",
    "## Leitura dos sinais",
    "",
    "- **✓ V10 cobre** — campanha viável com catálogo atual.",
    "- **⚠ parcial** — V10 tem algo, mas o pico real é em produto que falta. Promover o que tem + planejar expansão.",
    "- **✗ cegueira** — V10 não tem nada útil. Oportunidade só se expandir o catálogo.",
    "",
    "## Próximo passo claro",
    "",
    "Cada **✗** ou **⚠** acima é uma justificativa direta para:",
    "1. Pedir ao posto inclusão de chocolate, vinho/espumante, salgadinho ao catálogo do modelo",
    "2. Coletar 6+ meses de venda detalhada por SKU dessas categorias",
    "3. Retreinar com o catálogo expandido (V11)",
    "",
    "---",
    "",
    f"*V2 gerada em {DATA_HOJE.isoformat()} cruzando V10 com calendário comercial brasileiro + Google Trends.*",
]

(RESULTS / 'calendario_v2.md').write_text('\n'.join(md), encoding='utf-8')

# ── 7. Print resumo ─────────────────────────────────────────────────────────

print(f"✓ Calendário V2 gerado")
print(f"  Eventos detectados:     {resumo['eventos_detectados']}")
print(f"  V10 cobre totalmente:   {resumo['eventos_v10_cobre_total']}")
print(f"  V10 cobre parcialmente: {resumo['eventos_v10_cobre_parcial']}")
print(f"  V10 NÃO enxerga:        {resumo['eventos_v10_nao_cobre']}  ← OPORTUNIDADES PERDIDAS")
print()
print("Próximos eventos no horizonte:")
for r in recomendacoes_ord[:8]:
    di = pd.Timestamp(r['data_inicio']).strftime('%d/%m')
    de = pd.Timestamp(r['data_evento']).strftime('%d/%m')
    status = {'cobertura_total_v10': 'OK     ',
              'cobertura_parcial': 'PARCIAL',
              'cegueira_v10': 'CEGUEIRA'}[r['classificacao']]
    print(f"  {status}  {de}  {r['evento']:<45s} "
          f"prior {r['uplift_prior']:.1f}x")
