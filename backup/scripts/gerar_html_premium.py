"""Gera HTML premium do calendário V3 — versão polida.

Foco em:
- Top 15 campanhas (não todas as 96)
- Combos visualmente claros (par lado a lado com ícones)
- Cards com info essencial + justificativa em linguagem natural
- Auto-refresh opcional (meta tag refresh)
- Lê arquivos novos automaticamente — basta rodar de novo

Output: results/v11/calendario_premium.html
"""
import io
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V11 = ROOT / 'results' / 'v11'

# ── Configuração ──────────────────────────────────────────────────────────

TOP_N = 15
AUTO_REFRESH_SEGUNDOS = 0  # 0 = desliga; >0 = recarrega a cada X segundos

# ── Carregar dados ─────────────────────────────────────────────────────────

# Carrega calendario_v3.json (saída padrão de gerar_calendario_v3.py) OU
# calendario_v3_anual.json (renomeado pelo atualizar_calendario.ps1 em runs anuais).
# Pega o MAIS RECENTE para nunca ficar stale.
candidatos = [V11 / 'calendario_v3.json', V11 / 'calendario_v3_anual.json']
candidatos = [p for p in candidatos if p.exists()]
if not candidatos:
    raise FileNotFoundError('Nenhum calendario_v3*.json encontrado em results/v11/')
json_path = max(candidatos, key=lambda p: p.stat().st_mtime)
with open(json_path, encoding='utf-8') as f:
    cal = json.load(f)
print(f"Lendo: {json_path.name}")

camp = cal['campanhas']
for c in camp:
    c['_dt_inicio'] = datetime.fromisoformat(c['data_inicio']).date()
    c['_dt_fim'] = datetime.fromisoformat(c['data_fim']).date()

# Ordenar por lucro adicional decrescente
camp_sorted = sorted(camp, key=lambda c: -c['lucro_adicional_estimado_R$'])

# DIVERSIFICAR top: max MAX_POR_GRUPO campanhas por (categoria, intensidade)
# para mostrar variedade (combos, descontos em diferentes categorias)
MAX_POR_GRUPO = 2
from collections import defaultdict
contagem = defaultdict(int)
top = []
# Primeiro pass: pegar até MAX_POR_GRUPO de cada (cat, intens)
for c in camp_sorted:
    chave = (c['categoria'], c['intensidade'])
    if contagem[chave] < MAX_POR_GRUPO:
        top.append(c)
        contagem[chave] += 1
    if len(top) >= TOP_N:
        break
# Se ainda não tem TOP_N, completar com as próximas melhores
if len(top) < TOP_N:
    ja_no_top = set(id(c) for c in top)
    for c in camp_sorted:
        if id(c) not in ja_no_top:
            top.append(c)
        if len(top) >= TOP_N:
            break
top = top[:TOP_N]

# Uplift real (se disponível) para validar
uplift_real = {}
try:
    df_up = pd.read_csv(V11 / 'uplift_real_posto_agregado.csv')
    for _, r in df_up.iterrows():
        uplift_real[(r['evento_base'], r['categoria'])] = r['uplift_medio']
except FileNotFoundError:
    pass

# ── Ícones por categoria ──────────────────────────────────────────────────

ICONES = {
    'cerveja': '🍺',
    'cigarro_souza_cruz': '🚬',
    'cigarro_philip_morris': '🚬',
    'cigarro_jti': '🚬',
    'agua': '💧',
    'refrigerante': '🥤',
    'energetico': '⚡',
    'isotonico': '💪',
    'gelo': '🧊',
    'sorvete': '🍦',
    'snack': '🥨',
    'biscoito': '🍪',
    'chocolate_premium': '🍫',
    'chocolate_impulso': '🍫',
    'doce': '🍬',
    'cafe': '☕',
    'padaria': '🥐',
    'suco': '🧃',
    'vinho': '🍷',
    'destilados': '🥃',
}

NOMES_LIMPOS = {
    'cigarro_souza_cruz': 'Cigarro Souza Cruz',
    'cigarro_philip_morris': 'Cigarro Philip Morris',
    'cigarro_jti': 'Cigarro JTI',
    'chocolate_premium': 'Chocolate Premium',
    'chocolate_impulso': 'Chocolate Impulso',
}

def nome_categoria(cat):
    if cat in NOMES_LIMPOS:
        return NOMES_LIMPOS[cat]
    return cat.replace('_', ' ').title()

# ── Cores por categoria ───────────────────────────────────────────────────

CORES = {
    'cerveja': '#fbbf24', 'cigarro_souza_cruz': '#dc2626',
    'cigarro_philip_morris': '#b91c1c', 'cigarro_jti': '#991b1b',
    'agua': '#3b82f6', 'refrigerante': '#ef4444', 'energetico': '#a855f7',
    'isotonico': '#22d3ee', 'gelo': '#06b6d4', 'sorvete': '#ec4899',
    'snack': '#f97316', 'biscoito': '#eab308', 'chocolate_premium': '#7c3aed',
    'chocolate_impulso': '#a855f7', 'doce': '#f472b6', 'cafe': '#92400e',
    'padaria': '#d97706', 'suco': '#84cc16', 'vinho': '#7c2d12',
    'destilados': '#581c87',
}

# ── Justificativa em linguagem natural ────────────────────────────────────

DIAS = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
MESES = ['', 'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
          'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

def justificativa(c):
    cat = c['categoria']
    dur = c['dias_total']
    d_ini = c['_dt_inicio']
    eventos = c.get('eventos_comerciais_na_janela', [])

    partes = []
    if eventos:
        partes.append(f"📌 Coincide com {eventos[0]}")
    if c.get('produto_complementar'):
        par = c['produto_complementar']
        partes.append(f"combo com {nome_categoria(par)} (par mais relevante no contexto)")
    if c['intensidade'] == 'desc5%':
        partes.append("desconto leve (5%) para estimular venda sem comprometer margem")
    elif c['intensidade'] == 'desc10%':
        partes.append("desconto moderado (10%)")
    elif c['intensidade'] == 'liq25%':
        partes.append("liquidação 25% — provavelmente urgência de vencimento")
    elif c['intensidade'] == 'combo':
        partes.append("combo de cross-sell para aumentar ticket médio")

    # Uplift real validado?
    for ev in eventos:
        ev_base = ev.split(' — ')[0]
        chave = (ev_base, cat)
        if chave in uplift_real and uplift_real[chave] > 1.10:
            partes.append(f"✅ uplift histórico medido: {uplift_real[chave]:.2f}× (validado)")
            break

    if not partes:
        partes.append(f"Período de demanda relativamente baixa para {nome_categoria(cat)}")
    return '. '.join(partes) + '.'

# ── Gerar HTML ────────────────────────────────────────────────────────────

# Metadata
total_camp = cal.get('n_campanhas', len(camp))
total_lucro = cal.get('lucro_adicional_total_R$', sum(c['lucro_adicional_estimado_R$']
                                                         for c in camp))
periodo_ini = cal.get('data_inicio', camp[0]['data_inicio'] if camp else '')
periodo_fim = cal.get('data_fim', camp[-1]['data_fim'] if camp else '')
gerado = cal.get('gerado_em', str(date.today()))

# Resumo por categoria (todas as campanhas, não só top)
from collections import Counter
contagem_cat = Counter(c['categoria'] for c in camp)
top_cats = contagem_cat.most_common(5)

refresh_meta = (f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SEGUNDOS}">'
                  if AUTO_REFRESH_SEGUNDOS > 0 else '')

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
{refresh_meta}
<title>Top {TOP_N} Promoções — Auto Posto Parque Viana</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  color: #e2e8f0;
  padding: 20px;
  min-height: 100vh;
  line-height: 1.6;
}}
.container {{
  max-width: 1280px;
  margin: 0 auto;
}}
.header {{
  background: rgba(255,255,255,0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 32px;
  margin-bottom: 28px;
}}
h1 {{
  font-size: 32px;
  font-weight: 800;
  color: #fff;
  margin-bottom: 8px;
  letter-spacing: -0.5px;
}}
.subtitle {{
  color: #94a3b8;
  font-size: 15px;
}}
.summary-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-top: 24px;
}}
.metric {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  padding: 18px;
  border-radius: 12px;
  text-align: center;
}}
.metric-value {{
  font-size: 28px;
  font-weight: 800;
  color: #3b82f6;
  margin-bottom: 4px;
}}
.metric-label {{
  color: #94a3b8;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.section-title {{
  color: #f1f5f9;
  font-size: 22px;
  font-weight: 700;
  margin: 32px 0 18px;
  padding-bottom: 12px;
  border-bottom: 2px solid #334155;
}}
.campaign-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
}}
.campaign {{
  background: rgba(255,255,255,0.04);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 16px;
  padding: 24px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}}
.campaign:hover {{
  background: rgba(255,255,255,0.07);
  transform: translateY(-2px);
  border-color: rgba(59, 130, 246, 0.4);
}}
.campaign::before {{
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 6px;
  background: var(--cor-cat, #3b82f6);
}}
.camp-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}}
.camp-rank {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f59e0b, #d97706);
  color: #fff;
  font-weight: 800;
  font-size: 18px;
  width: 38px;
  height: 38px;
  border-radius: 50%;
  margin-right: 14px;
  flex-shrink: 0;
}}
.camp-info {{
  flex: 1;
}}
.camp-dates {{
  font-size: 20px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 4px;
}}
.camp-period {{
  color: #94a3b8;
  font-size: 13px;
}}
.camp-lucro {{
  font-size: 24px;
  font-weight: 800;
  color: #10b981;
  text-align: right;
}}
.camp-lucro-label {{
  font-size: 11px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  text-align: right;
}}
.combo-display {{
  display: flex;
  align-items: center;
  gap: 16px;
  background: rgba(0,0,0,0.2);
  border-radius: 12px;
  padding: 18px;
  margin: 16px 0;
}}
.combo-product {{
  flex: 1;
  text-align: center;
  padding: 14px;
  background: rgba(255,255,255,0.03);
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.08);
}}
.combo-product .icon {{
  font-size: 40px;
  display: block;
  margin-bottom: 8px;
}}
.combo-product .name {{
  font-weight: 700;
  color: #fff;
  font-size: 15px;
  margin-bottom: 4px;
}}
.combo-product .role {{
  color: #94a3b8;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.combo-plus {{
  font-size: 32px;
  font-weight: 800;
  color: #f59e0b;
  flex-shrink: 0;
}}
.single-product {{
  display: flex;
  align-items: center;
  gap: 16px;
  background: rgba(0,0,0,0.2);
  border-radius: 12px;
  padding: 18px;
  margin: 16px 0;
}}
.single-product .icon {{
  font-size: 48px;
}}
.single-product .info .name {{
  font-weight: 700;
  color: #fff;
  font-size: 18px;
  margin-bottom: 4px;
}}
.single-product .info .meta {{
  color: #94a3b8;
  font-size: 13px;
}}
.tag-row {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 12px;
}}
.tag {{
  display: inline-flex;
  align-items: center;
  padding: 4px 12px;
  background: rgba(59, 130, 246, 0.15);
  color: #93c5fd;
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 16px;
  font-size: 12px;
  font-weight: 600;
}}
.tag.intensity {{
  background: rgba(245, 158, 11, 0.15);
  color: #fcd34d;
  border-color: rgba(245, 158, 11, 0.3);
}}
.tag.event {{
  background: rgba(236, 72, 153, 0.15);
  color: #f9a8d4;
  border-color: rgba(236, 72, 153, 0.3);
}}
.tag.validated {{
  background: rgba(16, 185, 129, 0.15);
  color: #6ee7b7;
  border-color: rgba(16, 185, 129, 0.3);
}}
.justification {{
  background: rgba(0,0,0,0.15);
  padding: 14px 16px;
  border-radius: 8px;
  font-size: 14px;
  color: #cbd5e1;
  margin-top: 14px;
  border-left: 3px solid #3b82f6;
}}
.categories-summary {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin-top: 16px;
}}
.cat-pill {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  padding: 14px;
  border-radius: 10px;
  text-align: center;
}}
.cat-pill .icon {{
  font-size: 28px;
  margin-bottom: 6px;
}}
.cat-pill .count {{
  font-size: 22px;
  font-weight: 800;
  color: #fff;
}}
.cat-pill .name {{
  font-size: 11px;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
footer {{
  text-align: center;
  color: #64748b;
  font-size: 12px;
  margin-top: 40px;
  padding: 20px 0;
  border-top: 1px solid #334155;
}}
@media (max-width: 768px) {{
  .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
  .categories-summary {{ grid-template-columns: repeat(3, 1fr); }}
  .combo-display {{ flex-direction: column; }}
  .combo-plus {{ transform: rotate(90deg); }}
}}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>🎯 Top {TOP_N} Promoções</h1>
    <div class="subtitle">
      Auto Posto Parque Viana · Modelo V11.3 com elasticidade empírica calibrada<br>
      Período: <strong>{periodo_ini[8:10]}/{periodo_ini[5:7]}/{periodo_ini[:4]}</strong> a
      <strong>{periodo_fim[8:10]}/{periodo_fim[5:7]}/{periodo_fim[:4]}</strong>
      · Atualizado em {gerado}
    </div>

    <div class="summary-grid">
      <div class="metric">
        <div class="metric-value">{total_camp}</div>
        <div class="metric-label">Campanhas no Ano</div>
      </div>
      <div class="metric">
        <div class="metric-value">R$ {total_lucro:,.0f}</div>
        <div class="metric-label">Lucro Adicional</div>
      </div>
      <div class="metric">
        <div class="metric-value">{len(set(c['categoria'] for c in camp))}</div>
        <div class="metric-label">Categorias</div>
      </div>
      <div class="metric">
        <div class="metric-value">{len(set(c['intensidade'] for c in camp))}</div>
        <div class="metric-label">Tipos Promo</div>
      </div>
    </div>
  </div>

  <h2 class="section-title">⭐ Top {TOP_N} Campanhas (ordenadas por impacto)</h2>

  <div class="campaign-grid">
"""

for rank, c in enumerate(top, 1):
    cat = c['categoria']
    cor = CORES.get(cat, '#3b82f6')
    icone = ICONES.get(cat, '📦')
    d_ini_str = c['_dt_inicio'].strftime('%d/%m')
    d_fim_str = c['_dt_fim'].strftime('%d/%m')
    dia_sem = DIAS[c['_dt_inicio'].weekday()]
    mes_nome = MESES[c['_dt_inicio'].month]

    eventos = c.get('eventos_comerciais_na_janela', [])
    eh_combo = c['intensidade'] == 'combo'
    par = c.get('produto_complementar', '')

    # Combo: 2 produtos lado a lado
    if eh_combo and par:
        produto_html = f'''
        <div class="combo-display">
          <div class="combo-product">
            <span class="icon">{icone}</span>
            <div class="name">{nome_categoria(cat)}</div>
            <div class="role">Principal</div>
          </div>
          <div class="combo-plus">+</div>
          <div class="combo-product">
            <span class="icon">{ICONES.get(par, '📦')}</span>
            <div class="name">{nome_categoria(par)}</div>
            <div class="role">Complementar</div>
          </div>
        </div>'''
    else:
        produto_html = f'''
        <div class="single-product">
          <span class="icon">{icone}</span>
          <div class="info">
            <div class="name">{nome_categoria(cat)}</div>
            <div class="meta">Demanda base: {c['demanda_base_dia']} un/dia · {c['dias_total']} dias de promoção</div>
          </div>
        </div>'''

    # Tags
    tags = [f'<span class="tag intensity">{c["intensidade"].upper()} · {c["desconto_pct"]}% desc</span>']
    if eventos:
        for ev in eventos[:2]:
            tags.append(f'<span class="tag event">🎯 {ev}</span>')

    # Verificar validação
    for ev in eventos:
        ev_base = ev.split(' — ')[0]
        if (ev_base, cat) in uplift_real and uplift_real[(ev_base, cat)] > 1.10:
            up = uplift_real[(ev_base, cat)]
            tags.append(f'<span class="tag validated">✅ uplift real {up:.2f}×</span>')
            break

    html += f'''
    <div class="campaign" style="--cor-cat: {cor}">
      <div class="camp-header">
        <div style="display:flex; align-items:center;">
          <div class="camp-rank">{rank}</div>
          <div class="camp-info">
            <div class="camp-dates">{d_ini_str} → {d_fim_str}</div>
            <div class="camp-period">{dia_sem}-feira de {mes_nome} · {c['dias_total']} dia{'s' if c['dias_total'] > 1 else ''}</div>
          </div>
        </div>
        <div>
          <div class="camp-lucro">+ R$ {c['lucro_adicional_estimado_R$']:,.2f}</div>
          <div class="camp-lucro-label">Lucro Adicional</div>
        </div>
      </div>
      {produto_html}
      <div class="tag-row">{''.join(tags)}</div>
      <div class="justification">{justificativa(c)}</div>
    </div>
'''

# Resumo por categoria
html += '\n  </div>\n\n  <h2 class="section-title">📊 Categorias Mais Promovidas (todas as campanhas)</h2>\n  <div class="categories-summary">\n'

for cat, n in top_cats:
    html += f'''
    <div class="cat-pill" style="border-color: {CORES.get(cat, "#475569")}">
      <div class="icon">{ICONES.get(cat, '📦')}</div>
      <div class="count">{n}</div>
      <div class="name">{nome_categoria(cat)}</div>
    </div>'''

html += f'''
  </div>

  <footer>
    <p><strong>Modelo V11.3</strong> — DQN treinado com elasticidade empírica calibrada (Dunnhumby + Iowa Liquor + Walmart Sales)</p>
    <p>3 fontes físicas independentes validaram elasticidade categórica de loja física: -0.5 (vs Bijmolt -3.0, 6× menor)</p>
    <p>Política implementada: proteger margem em alta demanda · combo cooperativo · desconto direto só em vencimento ou baixa sazonal</p>
    <p style="margin-top:12px; opacity:0.6;">Gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
  </footer>
</div>
</body>
</html>'''

(V11 / 'calendario_premium.html').write_text(html, encoding='utf-8')

print(f"✓ {V11 / 'calendario_premium.html'}")
print()
print(f"Top {TOP_N} campanhas:")
for rank, c in enumerate(top, 1):
    cat_nome = nome_categoria(c['categoria'])
    icone = ICONES.get(c['categoria'], '📦')
    par_str = f" + {nome_categoria(c['produto_complementar'])}" if c.get('produto_complementar') and c['intensidade'] == 'combo' else ''
    eventos = c.get('eventos_comerciais_na_janela', [])
    ev_str = f" [{eventos[0]}]" if eventos else ''
    print(f"  #{rank:>2d}  {c['_dt_inicio'].strftime('%d/%m')}→{c['_dt_fim'].strftime('%d/%m')}  "
          f"{icone} {cat_nome}{par_str}  R$ {c['lucro_adicional_estimado_R$']:,.2f}{ev_str}")
print()
print(f"Para abrir: start {V11 / 'calendario_premium.html'}")
