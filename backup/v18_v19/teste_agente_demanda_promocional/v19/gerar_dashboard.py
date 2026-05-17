"""Dashboard V19 — visualização moderna (dark theme) do pipeline de 3 agentes.

Mostra para cada campanha:
- Demanda base × demanda promocional (visualmente)
- Uplift e canibalização
- ROI e score de decisão
- Decisão final com cor

Inspirado em Linear/Stripe — dark, clean, monospace para números.
"""
import json
from pathlib import Path
from datetime import datetime


def gerar_dashboard(path_v19: str, path_saida: str):
    with open(path_v19, encoding='utf-8') as f:
        cal = json.load(f)

    # Separa por tipo
    aprovadas_estruturais = [r for r in cal['aprovadas'] if r['tipo'] == 'estrutural']
    aprovadas_eventuais = [r for r in cal['aprovadas'] if r['tipo'] == 'eventual']

    s = cal['sumario']

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>V19 Dashboard — Auto Posto Parque Viana</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a0f;
    color: #e1e1e6;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 32px;
    line-height: 1.5;
}}
.mono {{ font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace; }}
.container {{ max-width: 1400px; margin: 0 auto; }}

.header {{
    border-bottom: 1px solid #2a2a35;
    padding-bottom: 24px;
    margin-bottom: 32px;
}}
.header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
.header .subtitle {{ color: #8a8a98; font-size: 14px; }}
.header .badge {{
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: #fff;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}}
.kpi {{
    background: #14141f;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 20px;
}}
.kpi .label {{ color: #8a8a98; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
.kpi .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
.kpi .delta {{ font-size: 12px; margin-top: 4px; }}
.delta.up {{ color: #4ade80; }}
.delta.down {{ color: #f87171; }}

.section {{
    margin-bottom: 40px;
}}
.section h2 {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a35;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.section h2 .count {{
    font-size: 13px;
    color: #8a8a98;
    font-weight: 400;
}}

.card {{
    background: #14141f;
    border: 1px solid #2a2a35;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
    display: grid;
    grid-template-columns: 1fr 200px 200px 160px;
    gap: 20px;
    align-items: center;
    transition: border-color 0.2s;
}}
.card:hover {{ border-color: #4a4a55; }}
.card.prioritaria {{ border-left: 3px solid #fbbf24; }}
.card.aprovada {{ border-left: 3px solid #4ade80; }}
.card.condicional {{ border-left: 3px solid #f59e0b; }}
.card.rejeitada {{ border-left: 3px solid #f87171; opacity: 0.55; }}
.card.proibida {{ border-left: 3px solid #ef4444; opacity: 0.55; }}

.card-info .title {{ font-size: 15px; font-weight: 600; margin-bottom: 4px; }}
.card-info .subtitle {{ color: #8a8a98; font-size: 12px; margin-bottom: 8px; }}
.card-info .tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.tag {{
    background: #1f1f2e;
    color: #c1c1cc;
    padding: 2px 10px;
    border-radius: 8px;
    font-size: 11px;
    border: 1px solid #2a2a35;
}}
.tag.evento {{ background: linear-gradient(90deg, #ec4899, #8b5cf6); color: #fff; border: 0; }}
.tag.tipo-estrutural {{ background: #1e3a8a; color: #93c5fd; border: 0; }}
.tag.tipo-eventual {{ background: #14532d; color: #86efac; border: 0; }}

.demand-bar {{
    background: #1f1f2e;
    border-radius: 6px;
    padding: 8px;
    position: relative;
}}
.demand-bar .label {{ font-size: 11px; color: #8a8a98; }}
.demand-bar .nums {{ font-size: 13px; font-weight: 600; margin: 4px 0; }}
.demand-bar .bar-container {{
    height: 4px;
    background: #14141f;
    border-radius: 2px;
    position: relative;
    overflow: hidden;
}}
.demand-bar .bar-base {{
    position: absolute;
    height: 100%;
    background: #4f46e5;
    left: 0;
    top: 0;
}}
.demand-bar .bar-promo {{
    position: absolute;
    height: 100%;
    background: linear-gradient(90deg, #4ade80, #22d3ee);
    left: 0;
    top: 0;
}}
.demand-bar .uplift {{
    font-size: 11px;
    color: #4ade80;
    margin-top: 4px;
    font-weight: 600;
}}

.metric {{ text-align: right; }}
.metric .label {{ font-size: 11px; color: #8a8a98; text-transform: uppercase; }}
.metric .value {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}
.metric .delta {{ font-size: 11px; margin-top: 2px; }}

.decision {{
    text-align: right;
    padding: 8px 12px;
    border-radius: 8px;
}}
.decision .label {{ font-size: 11px; color: #8a8a98; }}
.decision .value {{ font-size: 14px; font-weight: 700; margin-top: 4px; }}
.decision.prioritaria .value {{ color: #fbbf24; }}
.decision.aprovada .value {{ color: #4ade80; }}
.decision.condicional .value {{ color: #f59e0b; }}
.decision.rejeitada .value {{ color: #f87171; }}
.decision .score {{
    font-size: 10px;
    color: #8a8a98;
    margin-top: 2px;
    font-family: 'SF Mono', monospace;
}}

.motivo {{
    grid-column: 1 / -1;
    font-size: 12px;
    color: #8a8a98;
    padding: 10px 12px;
    background: #0f0f18;
    border-radius: 8px;
    margin-top: 8px;
    border-left: 2px solid #2a2a35;
}}

.footer {{
    margin-top: 48px;
    padding-top: 24px;
    border-top: 1px solid #2a2a35;
    color: #6a6a78;
    font-size: 11px;
    text-align: center;
}}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>
        <span class="badge">V19</span>
        Pipeline de 3 Agentes
    </h1>
    <div class="subtitle">
        Demanda → Receita → Decisão · Auto Posto Parque Viana · Gerado em {cal.get('gerado_em', 'hoje')}
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi">
        <div class="label">Campanhas Aprovadas</div>
        <div class="value">{s['aprovadas']} <span style="color: #6a6a78; font-size: 18px;">/ {s['total_processadas']}</span></div>
        <div class="delta">{s['rejeitadas']} rejeitadas · {s['proibidas']} proibidas</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro V17 (estimativa antiga)</div>
        <div class="value mono">R$ {s['lucro_v17_total_R$']:,.0f}</div>
        <div class="delta">linha de base operacional</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro V19 (com 3 agentes)</div>
        <div class="value mono">R$ {s['lucro_v19_total_R$']:,.0f}</div>
        <div class="delta up">{s['delta_v19_v17_pct']:+.1f}% sobre V17</div>
    </div>
    <div class="kpi">
        <div class="label">Δ por campanha aprovada</div>
        <div class="value mono">R$ {s['lucro_v19_aprovadas_R$']/max(s['aprovadas'], 1):,.0f}</div>
        <div class="delta">média de retorno</div>
    </div>
</div>

<div class="section">
    <h2>
        Campanhas Estruturais
        <span class="count">{len(aprovadas_estruturais)} aprovadas · padrões recorrentes</span>
    </h2>
"""

    for c in aprovadas_estruturais:
        html += _render_estrutural(c)

    html += """
</div>

<div class="section">
    <h2>
        Campanhas Eventuais
        <span class="count">""" + str(len(aprovadas_eventuais)) + """ aprovadas · datas específicas</span>
    </h2>
"""

    # Eventuais ordenadas por data
    eventuais_sorted = sorted(aprovadas_eventuais, key=lambda x: x.get('data_inicio', ''))
    for c in eventuais_sorted:
        html += _render_eventual(c)

    html += f"""
</div>

<div class="footer">
    Pipeline V19 · Demand Agent + Revenue Agent + Decision Agent · {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

</div>
</body>
</html>
"""

    Path(path_saida).parent.mkdir(parents=True, exist_ok=True)
    with open(path_saida, 'w', encoding='utf-8') as f:
        f.write(html)


def _decisao_class(decisao: str) -> str:
    if 'PRIORITARIA' in decisao: return 'prioritaria'
    if 'APROVADA' in decisao: return 'aprovada'
    if 'CONDICIONAL' in decisao: return 'condicional'
    if 'PROIBIDA' in decisao: return 'proibida'
    return 'rejeitada'


def _render_estrutural(c: dict) -> str:
    cls = _decisao_class(c['decisao'])
    # Barras: demanda
    promo_pct = min((c['demanda_promo_dia'] / max(c['demanda_base_dia'], 0.1)) * 100, 200)
    promo_width = min(promo_pct / 2, 100)
    base_width = 50  # base é referência

    delta_v17 = c['lucro_v19_anual'] - c['lucro_v17_anual']
    delta_class = 'up' if delta_v17 > 0 else 'down'

    return f"""
    <div class="card {cls}">
        <div class="card-info">
            <div class="title">📅 {c.get('nome', c['categoria'])}</div>
            <div class="subtitle">{c.get('comunicacao', '')[:80]}</div>
            <div class="tags">
                <span class="tag tipo-estrutural">ESTRUTURAL</span>
                <span class="tag">{c['n_dias_no_ano']} dias/ano</span>
                <span class="tag">{c['categoria']} + {c.get('par_combo', '?')}</span>
                <span class="tag">canib {c['canibalizacao_pct']:.0f}%</span>
            </div>
        </div>
        <div class="demand-bar">
            <div class="label">Demanda base → promo (un/dia)</div>
            <div class="nums mono">{c['demanda_base_dia']:.1f} → {c['demanda_promo_dia']:.1f}</div>
            <div class="bar-container">
                <div class="bar-base" style="width: {base_width}%;"></div>
                <div class="bar-promo" style="width: {promo_width}%;"></div>
            </div>
            <div class="uplift">+{c['uplift_pct']:.1f}% uplift</div>
        </div>
        <div class="metric">
            <div class="label">Lucro Anual</div>
            <div class="value mono">R$ {c['lucro_v19_anual']:,.0f}</div>
            <div class="delta {delta_class}">{('+' if delta_v17 >= 0 else '')}{delta_v17:,.0f} vs V17</div>
        </div>
        <div class="decision {cls}">
            <div class="label">Decisão</div>
            <div class="value">{c['decisao']}</div>
            <div class="score">ROI {c['roi_pct']:.0f}% · Score {c['score']:.0f}</div>
        </div>
        <div class="motivo">{c['motivo']}</div>
    </div>
"""


def _render_eventual(c: dict) -> str:
    cls = _decisao_class(c['decisao'])
    # Barras: demanda
    promo_pct = min((c['demanda_promo_dia'] / max(c['demanda_base_dia'], 0.1)) * 100, 200)
    promo_width = min(promo_pct / 2, 100)
    base_width = 50

    delta_v17 = c['lucro_v19_estimado'] - c['lucro_v17_estimado']
    delta_class = 'up' if delta_v17 > 0 else 'down'

    evento_html = ''
    if c.get('evento') and c['evento'] != '-':
        evento_html = f'<span class="tag evento">🎉 {c["evento"]}</span>'

    return f"""
    <div class="card {cls}">
        <div class="card-info">
            <div class="title">📌 {c['categoria']} {('+ ' + c['produto_complementar']) if c.get('produto_complementar') else ''}</div>
            <div class="subtitle">{c['data_inicio']} → {c['data_fim']} · {c['dias_total']}d · {c['intensidade']} {c['desconto_pct']}%</div>
            <div class="tags">
                <span class="tag tipo-eventual">EVENTUAL</span>
                {evento_html}
                <span class="tag">canib {c['canibalizacao_pct']:.0f}%</span>
                <span class="tag">BE {c.get('razao_breakeven', 0):.1f}×</span>
            </div>
        </div>
        <div class="demand-bar">
            <div class="label">Demanda base → promo (un/dia)</div>
            <div class="nums mono">{c['demanda_base_dia']:.1f} → {c['demanda_promo_dia']:.1f}</div>
            <div class="bar-container">
                <div class="bar-base" style="width: {base_width}%;"></div>
                <div class="bar-promo" style="width: {promo_width}%;"></div>
            </div>
            <div class="uplift">+{c['uplift_pct']:.1f}% uplift</div>
        </div>
        <div class="metric">
            <div class="label">Lucro Campanha</div>
            <div class="value mono">R$ {c['lucro_v19_estimado']:,.0f}</div>
            <div class="delta {delta_class}">{('+' if delta_v17 >= 0 else '')}{delta_v17:,.0f} vs V17</div>
        </div>
        <div class="decision {cls}">
            <div class="label">Decisão</div>
            <div class="value">{c['decisao']}</div>
            <div class="score">ROI {c['roi_pct']:.0f}% · Score {c['score']:.0f}</div>
        </div>
        <div class="motivo">{c['motivo']}</div>
    </div>
"""


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    PROJETO = Path(__file__).parent.parent.parent
    path_v19 = PROJETO / 'results' / 'v19' / 'calendario_v19.json'
    path_html = PROJETO / 'results' / 'v19' / 'dashboard_v19.html'

    gerar_dashboard(str(path_v19), str(path_html))
    print(f"✓ Dashboard V19 salvo: {path_html}")
