"""Comparação V17 vs V19 — relatório side-by-side com diagnóstico.

Mostra:
- Tabela campanha-a-campanha: lucro V17 vs V19
- Razão V19/V17 por campanha (V17 superestimou? subestimou?)
- Eventos comerciais: V17 reconhecia uplift correto? V19 corrige?
- HTML side-by-side com gráfico comparativo
"""
import json
from pathlib import Path
from datetime import datetime


def gerar_comparacao(path_v17: str, path_v19: str, path_html: str):
    with open(path_v17, encoding='utf-8') as f:
        v17 = json.load(f)
    with open(path_v19, encoding='utf-8') as f:
        v19 = json.load(f)

    # Indexa V17 por identificador estável
    v17_estr = {c['nome']: c for c in v17.get('campanhas_estruturais', [])}
    v17_evt = {f"{c.get('data_inicio')}_{c.get('categoria')}": c
                for c in v17.get('campanhas_eventuais', [])}

    # Constrói pares para a tabela
    pares = []
    for c19 in v19['todos']:
        if c19['tipo'] == 'estrutural':
            c17 = v17_estr.get(c19['nome'])
            if not c17:
                continue
            par = {
                'id': c19['nome'],
                'tipo': 'estrutural',
                'categoria': c19['categoria'],
                'descricao': c17.get('comunicacao', ''),
                'janela': f"{c17.get('n_dias_no_ano', 0)} dias/ano",
                'lucro_v17': c17.get('lucro_adicional_estimado_anual_R$', 0),
                'lucro_v19': c19['lucro_v19_anual'],
                'decisao_v19': c19['decisao'],
                'evento': '',
                'demanda_base': c19['demanda_base_dia'],
                'demanda_promo': c19['demanda_promo_dia'],
                'uplift_pct': c19['uplift_pct'],
                'canib_pct': c19['canibalizacao_pct'],
                'roi_pct': c19['roi_pct'],
            }
        else:
            key = f"{c19.get('data_inicio')}_{c19['categoria']}"
            c17 = v17_evt.get(key)
            if not c17:
                continue
            evt = c19.get('evento', '-')
            par = {
                'id': f"{c19['data_inicio']} {c19['categoria']}",
                'tipo': 'eventual',
                'categoria': c19['categoria'],
                'descricao': c17.get('comunicacao', ''),
                'janela': f"{c19['data_inicio']} → {c19['data_fim']} ({c19['dias_total']}d)",
                'lucro_v17': c17.get('lucro_adicional_estimado_R$', 0),
                'lucro_v19': c19['lucro_v19_estimado'],
                'decisao_v19': c19['decisao'],
                'evento': evt,
                'demanda_base': c19['demanda_base_dia'],
                'demanda_promo': c19['demanda_promo_dia'],
                'uplift_pct': c19['uplift_pct'],
                'canib_pct': c19['canibalizacao_pct'],
                'roi_pct': c19['roi_pct'],
            }
        par['razao'] = par['lucro_v19'] / par['lucro_v17'] if par['lucro_v17'] > 0 else 0
        par['delta_R$'] = par['lucro_v19'] - par['lucro_v17']
        pares.append(par)

    # Ordena por delta (maior ganho primeiro)
    pares.sort(key=lambda x: -x['delta_R$'])

    # Estatísticas
    n_subestimadas = sum(1 for p in pares if p['razao'] > 1.5)
    n_superestimadas = sum(1 for p in pares if 0 < p['razao'] < 0.7)
    lucro_v17 = sum(p['lucro_v17'] for p in pares)
    lucro_v19 = sum(p['lucro_v19'] for p in pares)

    # Eventos vs sem-evento
    eventos = [p for p in pares if p['evento'] and p['evento'] != '-']
    sem_evento = [p for p in pares if not p['evento'] or p['evento'] == '-']

    print(f"\n{'='*100}")
    print(f"COMPARAÇÃO V17 vs V19")
    print(f"{'='*100}")
    print(f"  Campanhas comparadas:        {len(pares)}")
    print(f"  V17 subestimou (V19 > 1.5×): {n_subestimadas}")
    print(f"  V17 superestimou (V19 < 0.7×): {n_superestimadas}")
    print(f"  Lucro V17 total:             R$ {lucro_v17:>10.2f}")
    print(f"  Lucro V19 total:             R$ {lucro_v19:>10.2f}")
    print(f"  Δ V19 vs V17:                {(lucro_v19/lucro_v17 - 1)*100 if lucro_v17 > 0 else 0:+.1f}%")
    print(f"\n  Campanhas com evento:        {len(eventos)} (média razão V19/V17: {sum(p['razao'] for p in eventos)/max(len(eventos),1):.1f}×)")
    print(f"  Campanhas sem evento:        {len(sem_evento)} (média razão V19/V17: {sum(p['razao'] for p in sem_evento)/max(len(sem_evento),1):.1f}×)")
    print(f"{'='*100}\n")

    # HTML side-by-side
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>V17 vs V19 — Comparação</title>
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
.container {{ max-width: 1500px; margin: 0 auto; }}

.header {{ border-bottom: 1px solid #2a2a35; padding-bottom: 24px; margin-bottom: 32px; }}
.header h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
.header .subtitle {{ color: #8a8a98; font-size: 14px; }}

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
.kpi .value {{ font-size: 26px; font-weight: 700; margin-top: 8px; }}
.kpi .delta {{ font-size: 12px; margin-top: 4px; color: #4ade80; }}

table {{
    width: 100%;
    border-collapse: collapse;
    background: #14141f;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #2a2a35;
}}
th {{
    background: #0f0f18;
    padding: 12px;
    text-align: left;
    font-size: 11px;
    color: #8a8a98;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid #2a2a35;
}}
td {{
    padding: 14px 12px;
    border-bottom: 1px solid #1f1f2a;
    font-size: 13px;
    vertical-align: middle;
}}
tr:last-child td {{ border-bottom: 0; }}
tr:hover {{ background: #18182a; }}

.col-id {{ font-weight: 600; }}
.col-id .sub {{ color: #8a8a98; font-size: 11px; font-weight: 400; margin-top: 2px; }}
.tag {{
    display: inline-block;
    padding: 1px 8px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}}
.tag-estrutural {{ background: #1e3a8a; color: #93c5fd; }}
.tag-eventual {{ background: #14532d; color: #86efac; }}
.tag-evento {{ background: linear-gradient(90deg, #ec4899, #8b5cf6); color: #fff; }}

.lucro-cell {{
    font-family: 'SF Mono', monospace;
    font-weight: 600;
    text-align: right;
}}
.lucro-v17 {{ color: #8a8a98; }}
.lucro-v19 {{ color: #4ade80; }}
.razao {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    font-family: 'SF Mono', monospace;
    font-size: 12px;
    font-weight: 700;
}}
.razao.up-big {{ background: #16213e; color: #4ade80; }}
.razao.up {{ background: #1f1f2a; color: #a3e635; }}
.razao.flat {{ background: #1f1f2a; color: #94a3b8; }}
.razao.down {{ background: #2a1414; color: #f87171; }}

.decisao {{
    font-size: 11px;
    padding: 4px 8px;
    border-radius: 6px;
    font-weight: 600;
}}
.decisao.aprov {{ background: #14532d; color: #86efac; }}
.decisao.prio {{ background: #422006; color: #fbbf24; }}
.decisao.rej {{ background: #2a1414; color: #f87171; }}

.demanda-cell {{
    font-family: 'SF Mono', monospace;
    font-size: 12px;
}}
.demanda-cell .uplift {{ color: #4ade80; font-weight: 600; margin-left: 4px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>V17 vs V19 — Comparação Lado a Lado</h1>
    <div class="subtitle">
        Calendário operacional original (V17) vs filtrado pelos 3 agentes (V19) ·
        Gerado em {datetime.now().strftime('%Y-%m-%d')}
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi">
        <div class="label">Campanhas Comparadas</div>
        <div class="value">{len(pares)}</div>
        <div class="delta">{n_subestimadas} subestimadas pelo V17</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro V17 (anual)</div>
        <div class="value mono">R$ {lucro_v17:,.0f}</div>
        <div class="delta" style="color: #8a8a98;">cálculo simples uplift × dias × margem</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro V19 (3 agentes)</div>
        <div class="value mono">R$ {lucro_v19:,.0f}</div>
        <div class="delta">{(lucro_v19/lucro_v17 - 1)*100 if lucro_v17 > 0 else 0:+.0f}% sobre V17</div>
    </div>
    <div class="kpi">
        <div class="label">Razão V19/V17 média</div>
        <div class="value mono">{lucro_v19/lucro_v17 if lucro_v17 > 0 else 0:.1f}×</div>
        <div class="delta">V17 subestima sistematicamente</div>
    </div>
</div>

<table>
<thead>
<tr>
    <th style="width: 28%;">Campanha</th>
    <th style="width: 12%;">Demanda base→promo</th>
    <th style="width: 9%;">Canib</th>
    <th style="width: 10%; text-align: right;">Lucro V17</th>
    <th style="width: 10%; text-align: right;">Lucro V19</th>
    <th style="width: 10%; text-align: right;">Δ R$</th>
    <th style="width: 8%; text-align: center;">V19/V17</th>
    <th style="width: 13%; text-align: right;">Decisão V19</th>
</tr>
</thead>
<tbody>
"""

    for p in pares:
        tipo_tag = 'tag-estrutural' if p['tipo'] == 'estrutural' else 'tag-eventual'
        tipo_label = 'EST' if p['tipo'] == 'estrutural' else 'EVT'

        razao_class = 'up-big' if p['razao'] > 5 else 'up' if p['razao'] > 1.5 else 'flat' if p['razao'] > 0.7 else 'down'

        dec_class = 'prio' if 'PRIORITARIA' in p['decisao_v19'] else 'aprov' if 'APROVADA' in p['decisao_v19'] else 'rej'

        evento_html = f'<span class="tag tag-evento">🎉 {p["evento"]}</span>' if p['evento'] and p['evento'] != '-' else ''

        delta_color = '#4ade80' if p['delta_R$'] > 0 else '#f87171'
        delta_sign = '+' if p['delta_R$'] >= 0 else ''

        html += f"""
<tr>
    <td class="col-id">
        <div>
            <span class="tag {tipo_tag}">{tipo_label}</span>
            {p['id']}
        </div>
        <div class="sub">{p['janela']} {evento_html}</div>
    </td>
    <td class="demanda-cell">
        {p['demanda_base']:.1f} → {p['demanda_promo']:.1f}
        <span class="uplift">+{p['uplift_pct']:.0f}%</span>
    </td>
    <td class="mono" style="color: #f59e0b;">{p['canib_pct']:.0f}%</td>
    <td class="lucro-cell lucro-v17">R$ {p['lucro_v17']:,.2f}</td>
    <td class="lucro-cell lucro-v19">R$ {p['lucro_v19']:,.2f}</td>
    <td class="lucro-cell" style="color: {delta_color};">{delta_sign}R$ {p['delta_R$']:,.2f}</td>
    <td style="text-align: center;"><span class="razao {razao_class}">{p['razao']:.1f}×</span></td>
    <td style="text-align: right;"><span class="decisao {dec_class}">{p['decisao_v19']}</span></td>
</tr>
"""

    html += f"""
</tbody>
</table>

<div style="margin-top: 32px; padding: 20px; background: #14141f; border-radius: 12px; border: 1px solid #2a2a35;">
    <h3 style="margin-bottom: 12px; font-size: 14px; color: #c1c1cc;">Diagnóstico</h3>
    <div style="font-size: 13px; color: #c1c1cc; line-height: 1.8;">
        <p>• <strong style="color: #4ade80;">{n_subestimadas} campanhas</strong> que o V17 subestimava em 1.5×+ (V19 reconhece halo, uplift de evento e match produto×evento).</p>
        <p>• <strong style="color: #f59e0b;">Campanhas com evento</strong> têm razão média <strong>{sum(p['razao'] for p in eventos)/max(len(eventos),1):.1f}×</strong> entre V17 e V19. V17 não modelava o efeito de calendário comercial.</p>
        <p>• <strong style="color: #6366f1;">Campanhas sem evento</strong> têm razão média <strong>{sum(p['razao'] for p in sem_evento)/max(len(sem_evento),1):.1f}×</strong>. V19 ainda corrige por causa de halo e cap realista.</p>
        <p style="margin-top: 12px; color: #8a8a98; font-size: 12px;">
            <strong>Interpretação:</strong> o V17 estimava lucro a partir de "uplift × margem × dias", sem modelar canibalização, halo (cross-sell), uplift de evento ou cap por intensidade. V19 incorpora todos esses efeitos via os 3 agentes especializados, produzindo uma estimativa mais realista (e geralmente maior, exceto onde o V17 ignorava cap de intensidade).
        </p>
    </div>
</div>

</div>
</body>
</html>
"""

    Path(path_html).parent.mkdir(parents=True, exist_ok=True)
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Comparação V17 vs V19 salva: {path_html}")
    return pares


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    PROJETO = Path(__file__).parent.parent.parent
    path_v17 = PROJETO / 'results' / 'v17' / 'calendario_operacional.json'
    path_v19 = PROJETO / 'results' / 'v19' / 'calendario_v19.json'
    path_html = PROJETO / 'results' / 'v19' / 'comparacao_v17_v19.html'

    gerar_comparacao(str(path_v17), str(path_v19), str(path_html))
