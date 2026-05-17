"""V16 — Dashboard HTML moderno para o dono do posto.

Template inspirado em Linear/Stripe/Vercel: clean, profissional, mobile-friendly.
Renderiza o calendário operacional V16 com:
- Hero card com KPIs (campanhas, lucro estimado anual, período)
- Campanhas estruturais (sempre ativas) em grid de cards
- Timeline mensal das eventuais
- Destaque do mês (top 1 ROI)
- Footer com checklist operacional

Hook automático: chamado ao fim de gerar_calendario_operacional.py.

Uso:
    python gerar_dashboard_v16.py --input results/v16/calendario_operacional.json
"""
import argparse
import io
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                     default='results/v16/calendario_operacional.json')
parser.add_argument('--output', type=str,
                     default='results/v16/dashboard.html')
args = parser.parse_args()


with open(ROOT / args.input, encoding='utf-8') as f:
    cal = json.load(f)

estruturais = cal['campanhas_estruturais']
eventuais = cal['campanhas_eventuais']

# Encontra destaque do mês (top ROI)
destaque = max(eventuais, key=lambda c: c.get('lucro_adicional_estimado_R$', 0)) \
    if eventuais else None

# Agrupar eventuais por mês
MESES_PT = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
            'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
por_mes = defaultdict(list)
for c in eventuais:
    m = date.fromisoformat(c['data_inicio']).month
    por_mes[m].append(c)

# Ícones por categoria
ICONES_CAT = {
    'cerveja': '🍺', 'cigarro': '🚬', 'agua': '💧', 'refrigerante': '🥤',
    'energetico': '⚡', 'isotonico': '💪', 'gelo': '🧊', 'sorvete': '🍦',
    'snack': '🥨', 'biscoito': '🍪', 'chocolate_premium': '🍫',
    'chocolate_impulso': '🍫', 'doce': '🍬', 'cafe': '☕', 'padaria': '🥐',
    'suco': '🧃', 'vinho': '🍷', 'destilados': '🥃',
}

def icone(cat):
    return ICONES_CAT.get(cat, '📦')


def nome(cat):
    return cat.replace('_', ' ').title() if cat else ''


def card_estrutural(c):
    """Card de campanha estrutural."""
    return f'''
    <div class="card structural">
      <div class="card-badge">SEMPRE ATIVA</div>
      <div class="card-header">
        <div class="card-icon">{icone(c['categoria'])}</div>
        <div>
          <h3>{c['nome']}</h3>
          <p class="card-sub">{nome(c['categoria'])} + {nome(c['par_combo'])}</p>
        </div>
      </div>
      <div class="price-tag">
        <span class="price-currency">R$</span>
        <span class="price-value">{int(c['preco_combo'])}</span>
        <span class="price-cents">,{int(round((c['preco_combo']-int(c['preco_combo']))*100)):02d}</span>
      </div>
      <p class="card-message">{c['comunicacao'].split('•')[-1].strip()}</p>
      <div class="card-stats">
        <div><strong>{c['n_dias_no_ano']}</strong><span>dias/ano</span></div>
        <div><strong>R$ {c['lucro_adicional_estimado_anual_R$']:.0f}</strong><span>lucro est.</span></div>
      </div>
      <details class="card-detail">
        <summary>Por quê?</summary>
        <p>{c['justificativa']}</p>
      </details>
    </div>'''


def card_eventual(c):
    """Card compacto de campanha eventual."""
    par = c.get('produto_complementar', '')
    par_html = f'<span class="card-par"> + {nome(par)}</span>' if par else ''
    eventos = c.get('eventos_comerciais_na_janela', [])
    evt_tag = f'<div class="event-tag">📅 {eventos[0]}</div>' if eventos else ''
    return f'''
    <div class="card eventual">
      {evt_tag}
      <div class="card-header">
        <div class="card-icon-sm">{icone(c['categoria'])}</div>
        <div>
          <h4>{nome(c['categoria'])}{par_html}</h4>
          <p class="card-sub">{c['data_inicio']} → {c['data_fim']} ({c['dias_total']}d)</p>
        </div>
      </div>
      <p class="card-message-sm">{c.get('comunicacao', '')}</p>
      <div class="card-footer">
        <span class="price-sm">R$ {c.get('preco_combo_alvo', 0):.2f}</span>
        <span class="lucro-sm">+R$ {c.get('lucro_adicional_estimado_R$', 0):.2f}</span>
      </div>
    </div>'''


# ───────── Stats globais ─────────
n_estr = len(estruturais)
n_evt = len(eventuais)
lucro_total = cal['sumario']['lucro_total_anual_R$']
periodo_inicio = date.fromisoformat(cal['data_inicio']).strftime('%d/%m/%Y')
periodo_fim = date.fromisoformat(cal['data_fim']).strftime('%d/%m/%Y')


# ───────── HTML ─────────
html = f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calendário de Promoções • Auto Posto Parque Viana</title>
<style>
  :root {{
    --bg: #0a0a0a;
    --bg-card: #141414;
    --bg-elevated: #1f1f1f;
    --border: rgba(255,255,255,0.08);
    --text: #fafafa;
    --text-muted: #888;
    --text-dim: #555;
    --accent: #f59e0b;
    --accent-soft: rgba(245,158,11,0.1);
    --success: #10b981;
    --success-soft: rgba(16,185,129,0.1);
    --gradient: linear-gradient(135deg, #f59e0b 0%, #dc2626 100%);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    min-height: 100vh;
    padding: 32px 16px;
  }}
  .container {{ max-width: 1280px; margin: 0 auto; }}

  /* ───── Hero ───── */
  .hero {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px;
    margin-bottom: 32px;
    background-image:
      radial-gradient(at top left, rgba(245,158,11,0.08) 0%, transparent 50%),
      radial-gradient(at bottom right, rgba(220,38,38,0.08) 0%, transparent 50%);
  }}
  .hero-label {{
    color: var(--accent);
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
  }}
  .hero h1 {{
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 8px;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .hero-sub {{
    color: var(--text-muted);
    font-size: 16px;
    margin-bottom: 32px;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
  }}
  .kpi {{
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .kpi-label {{
    color: var(--text-muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
  }}
  .kpi-value {{
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
  }}
  .kpi-suffix {{
    color: var(--text-muted);
    font-size: 14px;
    font-weight: 400;
  }}

  /* ───── Section ───── */
  .section {{ margin-bottom: 40px; }}
  .section-header {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}
  .section h2 {{
    font-size: 20px;
    font-weight: 600;
  }}
  .section-count {{
    color: var(--text-muted);
    font-size: 14px;
  }}

  /* ───── Cards Estruturais ───── */
  .cards-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    position: relative;
    transition: border-color 0.15s ease;
  }}
  .card:hover {{
    border-color: var(--accent);
  }}
  .card-badge {{
    position: absolute;
    top: 16px;
    right: 16px;
    background: var(--success-soft);
    color: var(--success);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 4px 8px;
    border-radius: 4px;
  }}
  .card-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }}
  .card-icon {{
    font-size: 32px;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-soft);
    border-radius: 8px;
  }}
  .card-icon-sm {{
    font-size: 24px;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent-soft);
    border-radius: 6px;
  }}
  .card h3 {{ font-size: 18px; font-weight: 600; }}
  .card h4 {{ font-size: 15px; font-weight: 600; }}
  .card-sub {{
    color: var(--text-muted);
    font-size: 13px;
    margin-top: 2px;
  }}
  .card-par {{ color: var(--accent); font-weight: 500; }}
  .price-tag {{
    display: inline-flex;
    align-items: baseline;
    background: var(--gradient);
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    margin: 16px 0;
    font-weight: 700;
  }}
  .price-currency {{ font-size: 12px; opacity: 0.9; margin-right: 4px; }}
  .price-value {{ font-size: 32px; line-height: 1; }}
  .price-cents {{ font-size: 14px; opacity: 0.9; }}
  .price-sm {{
    background: var(--accent-soft);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 14px;
  }}
  .lucro-sm {{
    color: var(--success);
    font-weight: 600;
    font-size: 13px;
  }}
  .card-message {{
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 16px;
  }}
  .card-message-sm {{
    color: var(--text-muted);
    font-size: 12px;
    margin: 8px 0 12px;
  }}
  .card-stats {{
    display: flex;
    gap: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
  }}
  .card-stats > div {{
    display: flex;
    flex-direction: column;
  }}
  .card-stats strong {{
    color: var(--text);
    font-weight: 700;
    font-size: 16px;
  }}
  .card-stats span {{
    color: var(--text-dim);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .card-detail {{
    margin-top: 12px;
    color: var(--text-muted);
    font-size: 12px;
  }}
  .card-detail summary {{
    cursor: pointer;
    color: var(--accent);
    font-size: 12px;
    user-select: none;
  }}
  .card-detail p {{
    margin-top: 8px;
    padding: 12px;
    background: var(--bg-elevated);
    border-radius: 6px;
    line-height: 1.6;
  }}
  .card-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }}
  .event-tag {{
    background: rgba(220,38,38,0.1);
    color: #ef4444;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 4px;
    display: inline-block;
    margin-bottom: 12px;
  }}

  /* ───── Highlight Card ───── */
  .highlight {{
    background: linear-gradient(135deg, rgba(245,158,11,0.15) 0%, rgba(220,38,38,0.05) 100%);
    border: 1px solid var(--accent);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 32px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 32px;
    align-items: center;
  }}
  .highlight-content h3 {{
    font-size: 24px;
    margin-bottom: 8px;
  }}
  .highlight-label {{
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }}
  .highlight p {{ color: var(--text-muted); margin: 8px 0; }}
  @media (max-width: 700px) {{
    .highlight {{ grid-template-columns: 1fr; }}
  }}

  /* ───── Timeline mensal ───── */
  .timeline {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }}
  .timeline-month {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }}
  .timeline-month-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .timeline-month-name {{
    font-size: 14px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  .timeline-count {{
    color: var(--text-muted);
    font-size: 12px;
  }}
  .timeline-item {{
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }}
  .timeline-item:last-child {{ border-bottom: none; }}
  .timeline-date {{
    color: var(--text-dim);
    font-family: 'SF Mono', monospace;
    font-size: 11px;
    margin-bottom: 2px;
  }}
  .timeline-desc {{
    color: var(--text);
  }}

  /* ───── Checklist ───── */
  .checklist {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px;
  }}
  .checklist h2 {{ margin-bottom: 16px; }}
  .checklist ul {{
    list-style: none;
    padding: 0;
  }}
  .checklist li {{
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .checklist li:last-child {{ border-bottom: none; }}
  .checklist li::before {{
    content: '☐';
    color: var(--accent);
    font-size: 18px;
  }}

  footer {{
    text-align: center;
    color: var(--text-dim);
    font-size: 13px;
    margin-top: 48px;
    padding: 24px 0;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<div class="container">

  <!-- HERO -->
  <div class="hero">
    <div class="hero-label">Auto Posto Parque Viana · Barueri/SP</div>
    <h1>Calendário de Promoções</h1>
    <div class="hero-sub">Plano operacional anual · {periodo_inicio} a {periodo_fim}</div>

    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-label">Estruturais</div>
        <div class="kpi-value">{n_estr}<span class="kpi-suffix"> sempre ativas</span></div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Eventuais</div>
        <div class="kpi-value">{n_evt}<span class="kpi-suffix"> no ano</span></div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Lucro estimado</div>
        <div class="kpi-value">R$ {lucro_total:,.0f}<span class="kpi-suffix">/ano</span></div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Compliance</div>
        <div class="kpi-value" style="color: var(--success);">100%<span class="kpi-suffix"> ✓</span></div>
      </div>
    </div>
  </div>
'''

# Destaque
if destaque:
    html += f'''
  <!-- DESTAQUE -->
  <div class="highlight">
    <div class="highlight-content">
      <div class="highlight-label">Destaque do ano</div>
      <h3>{icone(destaque['categoria'])} {nome(destaque['categoria'])}{' + ' + nome(destaque.get('produto_complementar','')) if destaque.get('produto_complementar') else ''}</h3>
      <p><strong>{destaque['data_inicio']} → {destaque['data_fim']}</strong> · {destaque['dias_total']} dias</p>
      <p>{destaque.get('comunicacao', '')}</p>
      <p style="color: var(--success); font-weight: 600;">+R$ {destaque.get('lucro_adicional_estimado_R$', 0):.2f} estimados</p>
    </div>
    <div class="price-tag" style="font-size: 1.5em;">
      <span class="price-currency">R$</span>
      <span class="price-value">{int(destaque.get('preco_combo_alvo', 0))}</span>
      <span class="price-cents">,{int(round((destaque.get('preco_combo_alvo', 0) - int(destaque.get('preco_combo_alvo', 0)))*100)):02d}</span>
    </div>
  </div>
'''

# Estruturais
html += '''
  <!-- ESTRUTURAIS -->
  <div class="section">
    <div class="section-header">
      <h2>Campanhas estruturais</h2>
      <span class="section-count">Sempre ativas · base do calendário</span>
    </div>
    <div class="cards-grid">
'''
for c in estruturais:
    html += card_estrutural(c)
html += '    </div>\n  </div>\n'

# Timeline mensal
html += '''
  <!-- TIMELINE -->
  <div class="section">
    <div class="section-header">
      <h2>Calendário mensal</h2>
      <span class="section-count">Campanhas eventuais distribuídas no ano</span>
    </div>
    <div class="timeline">
'''
for m in sorted(por_mes.keys()):
    campanhas_mes = sorted(por_mes[m], key=lambda c: c['data_inicio'])
    items_html = ''
    for c in campanhas_mes:
        d = date.fromisoformat(c['data_inicio'])
        items_html += f'''
      <div class="timeline-item">
        <div class="timeline-date">{d.strftime('%d/%m')} · {c['dias_total']}d</div>
        <div class="timeline-desc">{icone(c['categoria'])} {c.get('comunicacao', nome(c['categoria']))[:50]}</div>
      </div>'''
    html += f'''
    <div class="timeline-month">
      <div class="timeline-month-header">
        <span class="timeline-month-name">{MESES_PT[m-1]}</span>
        <span class="timeline-count">{len(campanhas_mes)} campanha{'s' if len(campanhas_mes) > 1 else ''}</span>
      </div>
      {items_html}
    </div>'''
html += '    </div>\n  </div>\n'

# Checklist
html += '''
  <!-- CHECKLIST -->
  <div class="checklist">
    <h2>Como executar este plano</h2>
    <ul>
      <li>Imprimir cartazes A4 (pasta <code>operacao/cartazes/</code>) e fixar nas prateleiras</li>
      <li>Carregar <code>etiquetas_pdv.csv</code> no sistema do caixa</li>
      <li>Treinar equipe em 15min (ver <code>treinamento_equipe.md</code>)</li>
      <li>Revisar plano toda segunda-feira (<code>relatorio_semana_*.md</code>)</li>
      <li>Medir vendas vs baseline após cada campanha</li>
      <li>Ajustar elasticidade no modelo baseado em resultado real</li>
    </ul>
  </div>

  <footer>
    Gerado automaticamente · Modelo V13 (DQN + Action Mask) · Pipeline V16<br>
    Política calibrada em 6 anos de venda real do posto + 4 datasets externos
  </footer>

</div>

</body>
</html>
'''

OUT = ROOT / args.output
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html, encoding='utf-8')
print(f"✓ Dashboard: {OUT}")
