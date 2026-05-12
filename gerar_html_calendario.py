"""Gera HTML visual e interativo do calendário anual de promoções V3.

Output: results/v11/calendario_v3_anual.html
Para abrir: duplo clique no arquivo (abre no navegador padrão)
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

with open(V11 / 'calendario_v3_anual.json', encoding='utf-8') as f:
    cal = json.load(f)

camp = cal['campanhas']
for c in camp:
    c['data_inicio_dt'] = datetime.fromisoformat(c['data_inicio']).date()
    c['data_fim_dt'] = datetime.fromisoformat(c['data_fim']).date()

# Ordenar por data
camp = sorted(camp, key=lambda c: c['data_inicio_dt'])

# Carregar uplift real (se existir) para mostrar validação
try:
    uplift = pd.read_csv(V11 / 'uplift_real_posto_agregado.csv')
    uplift_dict = {(r['evento_base'], r['categoria']): r['uplift_medio']
                    for _, r in uplift.iterrows()}
except FileNotFoundError:
    uplift_dict = {}

# ── Cores por categoria ───────────────────────────────────────────────────

CORES = {
    'cerveja': '#fbbf24',
    'cigarro_souza_cruz': '#dc2626',
    'cigarro_philip_morris': '#b91c1c',
    'cigarro_jti': '#991b1b',
    'agua': '#3b82f6',
    'refrigerante': '#ef4444',
    'energetico': '#a855f7',
    'isotonico': '#22d3ee',
    'gelo': '#06b6d4',
    'sorvete': '#ec4899',
    'snack': '#f97316',
    'biscoito': '#eab308',
    'chocolate_premium': '#7c3aed',
    'chocolate_impulso': '#a855f7',
    'doce': '#f472b6',
    'cafe': '#92400e',
    'padaria': '#d97706',
    'suco': '#84cc16',
    'vinho': '#7c2d12',
    'destilados': '#581c87',
    'suco': '#84cc16',
}

NOMES_MES = ['', 'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
              'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
NOMES_DIA = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']

# ── Agrupar por mês ───────────────────────────────────────────────────────

por_mes = {}
for c in camp:
    chave = (c['data_inicio_dt'].year, c['data_inicio_dt'].month)
    if chave not in por_mes:
        por_mes[chave] = []
    por_mes[chave].append(c)

# ── HTML ──────────────────────────────────────────────────────────────────

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Calendário de Promoções V3 — Auto Posto Parque Viana</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: #f3f4f6;
  margin: 0;
  padding: 20px;
  color: #1f2937;
  line-height: 1.5;
}}
.container {{
  max-width: 1200px;
  margin: 0 auto;
  background: white;
  border-radius: 12px;
  padding: 32px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}}
h1 {{
  margin-top: 0;
  border-bottom: 3px solid #3b82f6;
  padding-bottom: 12px;
  color: #1e40af;
}}
.summary {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin: 24px 0;
}}
.metric {{
  background: #f9fafb;
  padding: 16px;
  border-radius: 8px;
  border-left: 4px solid #3b82f6;
}}
.metric-value {{
  font-size: 28px;
  font-weight: bold;
  color: #1e40af;
}}
.metric-label {{
  color: #6b7280;
  font-size: 13px;
}}
.month {{
  margin-top: 32px;
  padding-top: 16px;
  border-top: 2px solid #e5e7eb;
}}
.month h2 {{
  margin: 0 0 16px 0;
  color: #1f2937;
  font-size: 22px;
}}
.month-total {{
  display: inline-block;
  background: #dbeafe;
  color: #1e40af;
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 14px;
  margin-left: 12px;
}}
.campaign {{
  background: white;
  border: 1px solid #e5e7eb;
  border-left: 6px solid #3b82f6;
  padding: 14px 18px;
  margin: 10px 0;
  border-radius: 8px;
  display: grid;
  grid-template-columns: 110px 1fr 160px 120px;
  gap: 16px;
  align-items: center;
  transition: box-shadow 0.2s;
}}
.campaign:hover {{
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
.campaign .dates {{
  font-weight: bold;
  font-size: 15px;
  color: #1f2937;
}}
.campaign .duration {{
  color: #6b7280;
  font-size: 12px;
}}
.campaign .category {{
  font-weight: bold;
  text-transform: capitalize;
  font-size: 15px;
}}
.campaign .intensity {{
  display: inline-block;
  padding: 3px 10px;
  background: #fef3c7;
  color: #92400e;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}}
.campaign .lucro {{
  color: #059669;
  font-weight: bold;
  font-size: 16px;
  text-align: right;
}}
.event-tag {{
  display: inline-block;
  background: #fce7f3;
  color: #be185d;
  padding: 2px 8px;
  border-radius: 8px;
  font-size: 11px;
  margin-top: 4px;
}}
.uplift-real {{
  display: inline-block;
  background: #d1fae5;
  color: #065f46;
  padding: 2px 8px;
  border-radius: 8px;
  font-size: 11px;
  margin-left: 6px;
}}
.intro {{
  background: #fef3c7;
  border-left: 4px solid #d97706;
  padding: 16px;
  margin-bottom: 24px;
  border-radius: 4px;
}}
footer {{
  text-align: center;
  color: #6b7280;
  font-size: 12px;
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #e5e7eb;
}}
.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 16px 0;
}}
.legend-item {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 4px 10px;
  background: #f9fafb;
  border-radius: 12px;
}}
.legend-color {{
  width: 12px;
  height: 12px;
  border-radius: 3px;
}}
</style>
</head>
<body>
<div class="container">
  <h1>📅 Calendário de Promoções V3</h1>
  <p style="color: #6b7280; margin-top: 4px;">
    <strong>Auto Posto Parque Viana</strong> — gerado pelo modelo V11 (Double DQN com 20 categorias + calendário comercial brasileiro)
  </p>

  <div class="intro">
    <strong>Como ler este calendário:</strong> cada bloco abaixo é uma <strong>campanha de promoção</strong>
    recomendada pelo modelo. Mostra <strong>quando começar, quando terminar, qual categoria
    promover, qual desconto aplicar e o lucro adicional estimado</strong>. Tags rosa indicam
    coincidência com data comercial. Tags verdes mostram o uplift REAL medido nos 6 anos
    de dados do posto.
  </div>

  <div class="summary">
    <div class="metric">
      <div class="metric-value">{cal['n_campanhas']}</div>
      <div class="metric-label">Campanhas no ano</div>
    </div>
    <div class="metric">
      <div class="metric-value">R$ {cal['lucro_adicional_total_R$']:,.0f}</div>
      <div class="metric-label">Lucro adicional estimado</div>
    </div>
    <div class="metric">
      <div class="metric-value">{cal['horizonte_dias']}</div>
      <div class="metric-label">Dias de horizonte</div>
    </div>
    <div class="metric">
      <div class="metric-value">{cal['data_inicio'][8:10]}/{cal['data_inicio'][5:7]} a {cal['data_fim'][8:10]}/{cal['data_fim'][5:7]}</div>
      <div class="metric-label">Período</div>
    </div>
  </div>

  <h3>Legenda — categorias promovidas</h3>
  <div class="legend">
"""

cats_promovidas = sorted(set(c['categoria'] for c in camp))
for c in cats_promovidas:
    cor = CORES.get(c, '#9ca3af')
    html += f'    <span class="legend-item"><span class="legend-color" style="background: {cor}"></span>{c}</span>\n'

html += """  </div>
"""

# Por mês
for chave in sorted(por_mes.keys()):
    ano, mes = chave
    campanhas_mes = por_mes[chave]
    total_lucro = sum(c['lucro_adicional_estimado_R$'] for c in campanhas_mes)

    html += f"""
  <div class="month">
    <h2>{NOMES_MES[mes]} {ano}
      <span class="month-total">{len(campanhas_mes)} campanha{'s' if len(campanhas_mes) > 1 else ''} · R$ {total_lucro:,.2f}</span>
    </h2>
"""

    for c in campanhas_mes:
        d_ini = c['data_inicio_dt']
        d_fim = c['data_fim_dt']
        cor = CORES.get(c['categoria'], '#9ca3af')

        dias_str = d_ini.strftime('%d/%m')
        if d_ini != d_fim:
            dias_str += ' – ' + d_fim.strftime('%d/%m')

        weekdays = []
        d = d_ini
        while d <= d_fim:
            weekdays.append(NOMES_DIA[d.weekday()])
            d += timedelta(days=1)
        weekdays_str = ', '.join(weekdays[:3])
        if len(weekdays) > 3:
            weekdays_str += f' (+{len(weekdays) - 3})'

        # Evento na janela?
        eventos_str = ''
        if c.get('eventos_comerciais_na_janela'):
            eventos_str = '<br>'
            for ev in c['eventos_comerciais_na_janela'][:2]:
                eventos_str += f'<span class="event-tag">🎯 {ev}</span> '

        # Uplift real validado?
        uplift_str = ''
        for ev in c.get('eventos_comerciais_na_janela', []):
            ev_base = ev.split(' — ')[0]
            chave_u = (ev_base, c['categoria'])
            if chave_u in uplift_dict and uplift_dict[chave_u] > 1.10:
                uplift_str = f'<span class="uplift-real">✓ uplift real {uplift_dict[chave_u]:.2f}×</span>'
                break

        produto_compl = c.get('produto_complementar', '')
        combo_str = f' + {produto_compl}' if produto_compl else ''

        html += f"""    <div class="campaign" style="border-left-color: {cor}">
      <div>
        <div class="dates">{dias_str}</div>
        <div class="duration">{c['dias_total']} dia{'s' if c['dias_total'] > 1 else ''} · {weekdays_str}</div>
      </div>
      <div>
        <div class="category">{c['categoria'].replace('_', ' ').title()}{combo_str}</div>
        <div style="font-size: 13px; color: #6b7280; margin-top: 4px;">
          {c['demanda_base_dia']} un/dia base → uplift +{c['uplift_unidades_estimado']:.1f} un
          {eventos_str}
          {uplift_str}
        </div>
      </div>
      <div>
        <span class="intensity">{c['intensidade'].replace('%', '%').upper()}</span>
        <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">desconto {c['desconto_pct']}%</div>
      </div>
      <div class="lucro">+ R$ {c['lucro_adicional_estimado_R$']:,.2f}</div>
    </div>
"""
    html += "  </div>\n"

html += f"""
  <footer>
    <p>Modelo V11 (DQN com 20 categorias) + Calendário comercial BR + Validação cruzada com priors externos.</p>
    <p>Gerado em {cal['gerado_em']}. Próxima atualização recomendada: quando vendas detalhadas por SKU chegarem do ERP.</p>
  </footer>
</div>
</body>
</html>"""

(V11 / 'calendario_v3_anual.html').write_text(html, encoding='utf-8')

print(f"✓ Calendário HTML gerado em:")
print(f"  {V11 / 'calendario_v3_anual.html'}")
print()
print("Para abrir:")
print(f"  1. Abrir o Explorador de Arquivos do Windows")
print(f"  2. Navegar para C:\\Users\\vinin\\projeto-rl\\results\\v11\\")
print(f"  3. Duplo clique em 'calendario_v3_anual.html'")
print(f"  4. Vai abrir no seu navegador padrão (Chrome/Edge)")
print()
print(f"Ou pelo PowerShell:")
print(f"  start results\\v11\\calendario_v3_anual.html")
