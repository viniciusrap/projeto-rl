"""Dashboard visual V21 — calendário operacional formatado.

Lê o calendario_v21.json e gera HTML com:
- KPIs gerais
- Top campanhas por lucro
- Lista completa por data
- Categorias mais promovidas
- Eventos comerciais capturados
"""
import io
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


# Mapeamento de mês PT
MESES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
          'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
DIAS_SEM = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']


def categoria_emoji(cat: str) -> str:
    return {
        'cerveja': '🍻', 'gelo': '🧊', 'cafe': '☕', 'padaria': '🥐',
        'chocolate_caixa': '🍫', 'chocolate_unit': '🍫', 'doce_balcao': '🍬',
        'biscoito': '🍪', 'sorvete': '🍦', 'refrigerante': '🥤',
        'suco': '🧃', 'agua': '💧', 'energetico': '⚡', 'isotonico': '🧃',
        'sanduiche': '🥪', 'snack_salgado': '🥨', 'destilados': '🥃',
        'whisky': '🥃', 'vinho': '🍷', 'cha_pronto': '🍵',
    }.get(cat, '📦')


def gerar(path_v21: str, path_html: str):
    with open(path_v21, encoding='utf-8') as f:
        cal = json.load(f)

    campanhas = cal['campanhas']
    s = cal['sumario']

    # Ordena por data
    campanhas_por_data = sorted(campanhas, key=lambda x: x.get('data_inicio', ''))
    top15 = sorted(campanhas, key=lambda x: -x['lucro_total_acumulado'])[:15]

    # Categorias mais promovidas
    cat_count = Counter(c['categoria'] for c in campanhas)
    # Pares mais frequentes
    par_count = Counter()
    for c in campanhas:
        if c.get('par_combo'):
            par_count[(c['categoria'], c['par_combo'])] += 1
    # Eventos
    eventos = Counter(c.get('evento_proximo') for c in campanhas
                       if c.get('evento_proximo'))

    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>V21 PDV-Native — Calendário Operacional</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box;}}
body {{background:#0a0a0f;color:#e1e1e6;font-family:-apple-system,Segoe UI,sans-serif;
       padding:32px;line-height:1.5;}}
.container {{max-width:1500px;margin:0 auto;}}
.header {{border-bottom:1px solid #2a2a35;padding-bottom:24px;margin-bottom:32px;}}
.header h1 {{font-size:28px;font-weight:700;}}
.header .subtitle {{color:#8a8a98;margin-top:8px;font-size:14px;}}
.badge {{display:inline-block;padding:4px 12px;border-radius:12px;font-size:11px;
        font-weight:600;text-transform:uppercase;margin-right:8px;letter-spacing:0.5px;
        background:linear-gradient(90deg,#10b981,#06b6d4);color:#fff;}}
.kpi-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px;}}
.kpi {{background:#14141f;border:1px solid #2a2a35;border-radius:12px;padding:20px;}}
.kpi .label {{color:#8a8a98;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;}}
.kpi .value {{font-size:24px;font-weight:700;margin-top:8px;}}
.kpi .delta {{font-size:12px;margin-top:4px;color:#4ade80;}}
.h2 {{font-size:18px;font-weight:600;margin:32px 0 12px;padding-bottom:8px;
       border-bottom:1px solid #2a2a35;}}
.section-flex {{display:grid;grid-template-columns:2fr 1fr;gap:20px;margin-bottom:24px;}}
.card {{background:#14141f;border:1px solid #2a2a35;border-radius:12px;padding:18px;}}
.card.priori {{border-left:3px solid #fbbf24;}}
.card.normal {{border-left:3px solid #4ade80;}}
.card.combo-row {{display:grid;grid-template-columns:auto 1fr auto auto;gap:14px;
                  align-items:center;padding:14px;margin-bottom:8px;}}
.combo-icon {{font-size:28px;}}
.combo-info .titulo {{font-weight:600;font-size:14px;}}
.combo-info .sub {{color:#8a8a98;font-size:11px;margin-top:2px;}}
.combo-info .tags {{display:flex;gap:4px;margin-top:4px;flex-wrap:wrap;}}
.tag {{background:#1f1f2e;color:#c1c1cc;padding:2px 8px;border-radius:6px;
       font-size:10px;border:1px solid #2a2a35;}}
.tag.evt {{background:linear-gradient(90deg,#ec4899,#8b5cf6);color:#fff;border:0;}}
.combo-lucro {{font-family:'SF Mono',Monaco,monospace;font-weight:700;
                color:#4ade80;text-align:right;}}
.combo-int {{font-family:'SF Mono',Monaco,monospace;font-size:11px;color:#06b6d4;
              padding:3px 8px;background:#0a2a30;border-radius:6px;}}
.aside h3 {{font-size:14px;margin-bottom:12px;color:#c1c1cc;}}
.aside-row {{display:flex;justify-content:space-between;padding:6px 0;
              border-bottom:1px solid #1f1f2a;font-size:12px;}}
.aside-row:last-child {{border-bottom:0;}}
.aside-row .nome {{font-family:'SF Mono',Monaco,monospace;color:#c1c1cc;}}
.aside-row .freq {{font-family:'SF Mono',Monaco,monospace;color:#06b6d4;font-weight:600;}}
.mes-section {{margin-bottom:24px;}}
.mes-title {{font-size:14px;color:#8a8a98;text-transform:uppercase;
              letter-spacing:0.5px;margin-bottom:8px;}}
.linha {{display:grid;grid-template-columns:80px 30px 1fr auto auto;gap:12px;
         align-items:center;padding:8px 12px;background:#0f0f18;
         border-radius:6px;margin-bottom:4px;border-left:2px solid #2a2a35;font-size:12px;}}
.linha:hover {{border-left-color:#06b6d4;}}
.linha .data {{font-family:'SF Mono',Monaco,monospace;color:#8a8a98;}}
.linha .icon {{font-size:18px;}}
.linha .desc {{font-family:'SF Mono',Monaco,monospace;}}
.linha .lucro {{font-family:'SF Mono',Monaco,monospace;color:#4ade80;font-weight:600;}}
.linha .int {{font-family:'SF Mono',Monaco,monospace;font-size:10px;color:#06b6d4;
              padding:2px 6px;background:#0a2a30;border-radius:4px;}}
.evt-line {{display:flex;align-items:center;gap:8px;margin-top:2px;
             font-size:11px;color:#ec4899;}}
</style>
</head><body><div class="container">

<div class="header">
    <h1>
        <span class="badge">V21 PDV-Native</span>
        Calendário Operacional
    </h1>
    <div class="subtitle">
        Agente RL com calibração SEM Instacart · {s['total_campanhas']} campanhas em {cal['dias_simulados']} dias ·
        início: {cal['data_inicio_rollout']}
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi">
        <div class="label">Campanhas no Ano</div>
        <div class="value">{s['total_campanhas']}</div>
        <div class="delta">{s['combos']} combos · {s['categorias_distintas']} categorias</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro Anual Estimado</div>
        <div class="value">R$ {s['lucro_total_R$']:,.0f}</div>
        <div class="delta">média R$ {s['lucro_total_R$']/max(s['total_campanhas'],1):,.0f}/campanha</div>
    </div>
    <div class="kpi">
        <div class="label">PDV-inválidos</div>
        <div class="value" style="color:#4ade80;">{s['pdv_invalidos']}</div>
        <div class="delta">agente aprendeu sem hard rule</div>
    </div>
    <div class="kpi">
        <div class="label">Reward Total</div>
        <div class="value" style="font-family:'SF Mono',monospace;">{s['reward_total']:,.0f}</div>
        <div class="delta">após 5 iter de treino</div>
    </div>
</div>

<h2 class="h2">🏆 Top 15 campanhas por lucro</h2>
<div class="card">
"""

    for c in top15:
        cat = c['categoria']
        par = c.get('par_combo', '')
        evt = c.get('evento_proximo')
        priori_cls = 'priori' if evt else 'normal'
        emoji = categoria_emoji(cat)
        par_str = f' + {par}' if par else ''
        evt_html = f'<span class="tag evt">🎉 {evt}</span>' if evt else ''
        html += f"""
    <div class="card combo-row {priori_cls}">
        <div class="combo-icon">{emoji}</div>
        <div class="combo-info">
            <div class="titulo">{cat}{par_str}</div>
            <div class="sub">{c['data_inicio']} · {c['dias']} dia(s) · uplift +{c.get('uplift_pct', 0):.0f}%</div>
            <div class="tags">
                <span class="tag">demanda base {c.get('demanda_base_contextual', 0):.1f}/dia → promo {c.get('demanda_promocional', 0):.1f}</span>
                <span class="tag">canib {c.get('canib_pct', 0):.0f}%</span>
                {evt_html}
            </div>
        </div>
        <div class="combo-int">{c.get('intensidade', '?')}</div>
        <div class="combo-lucro">R$ {c['lucro_total_acumulado']:,.0f}</div>
    </div>
"""

    html += """
</div>

<div class="section-flex">
<div>

<h2 class="h2">📅 Calendário completo (por mês)</h2>
"""

    # Agrupa por mês
    por_mes = {}
    for c in campanhas_por_data:
        try:
            d = c['data_inicio']
            ym = d[:7]   # YYYY-MM
            por_mes.setdefault(ym, []).append(c)
        except Exception:
            pass

    for ym in sorted(por_mes.keys()):
        ano, mes = ym.split('-')
        nome_mes = MESES[int(mes) - 1]
        html += f'<div class="mes-section">'
        html += f'<div class="mes-title">{nome_mes} {ano} ({len(por_mes[ym])} campanhas)</div>'
        for c in por_mes[ym]:
            cat = c['categoria']
            par = c.get('par_combo', '')
            par_str = f' + {par}' if par else ''
            emoji = categoria_emoji(cat)
            try:
                dt = datetime.strptime(c['data_inicio'], '%Y-%m-%d')
                dia_sem = DIAS_SEM[dt.weekday()]
                data_str = f"{dt.day:02d}/{dt.month:02d} {dia_sem}"
            except Exception:
                data_str = c['data_inicio']
            evt = c.get('evento_proximo')
            evt_inline = f'<span style="color:#ec4899;font-size:10px;">🎉 {evt}</span>' if evt else ''
            html += f"""
            <div class="linha">
                <div class="data">{data_str}</div>
                <div class="icon">{emoji}</div>
                <div class="desc">{cat}{par_str} {evt_inline}</div>
                <div class="int">{c.get('intensidade', '?')}</div>
                <div class="lucro">R$ {c['lucro_total_acumulado']:,.0f}</div>
            </div>
"""
        html += '</div>'

    # Sidebar com agregados
    html += """
</div>
<div>

<h2 class="h2">📊 Agregados</h2>

<div class="card aside" style="margin-bottom:16px;">
    <h3>Categorias mais promovidas</h3>
"""
    for cat, freq in cat_count.most_common(12):
        emoji = categoria_emoji(cat)
        html += f'<div class="aside-row"><span class="nome">{emoji} {cat}</span><span class="freq">{freq}x</span></div>'

    html += """
</div>

<div class="card aside" style="margin-bottom:16px;">
    <h3>Top combos (par)</h3>
"""
    for (a, b), freq in par_count.most_common(12):
        html += f'<div class="aside-row"><span class="nome">{categoria_emoji(a)} {a}<br>+ {categoria_emoji(b)} {b}</span><span class="freq">{freq}x</span></div>'

    html += """
</div>

<div class="card aside" style="margin-bottom:16px;">
    <h3>Eventos comerciais capturados</h3>
"""
    for evt, freq in eventos.most_common(15):
        html += f'<div class="aside-row"><span class="nome">🎉 {evt}</span><span class="freq">{freq}x</span></div>'

    html += """
</div>

</div>
</div>

</div></body></html>
"""

    Path(path_html).parent.mkdir(parents=True, exist_ok=True)
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ Dashboard V21 salvo: {path_html}")


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    HERE = Path(__file__).parent
    PROJ = HERE.parent
    path_v21 = PROJ / 'results' / 'v21' / 'calendario_v21.json'
    path_html = PROJ / 'results' / 'v21' / 'calendario_v21.html'
    gerar(str(path_v21), str(path_html))
