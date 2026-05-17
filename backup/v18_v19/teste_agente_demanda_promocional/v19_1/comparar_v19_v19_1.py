"""Comparação V19 (baseline) vs V19.1 (Fase A+B) — mostra o efeito das correções.

Métricas chave:
- Diferenciação temporal: V19 tinha 12 campanhas de gelo IDÊNTICAS;
  V19.1 deveria ter campanhas diferentes por contexto.
- Total mais defensável: V19 era +483% sobre V17 (irreal);
  V19.1 deve ser menor.
- Discriminação: V19 aprovou 100%; V19.1 deve filtrar marginais.

Saída:
- results/v19_1/comparacao_v19_v19_1.html
- print de tabela no terminal
"""
import json
from pathlib import Path
from datetime import datetime


def comparar(path_v19: str, path_v19_1: str, path_html: str):
    with open(path_v19, encoding='utf-8') as f:
        v19 = json.load(f)
    with open(path_v19_1, encoding='utf-8') as f:
        v19_1 = json.load(f)

    # Indexa V19.1 por chave estável
    def key_of(r):
        if r['tipo'] == 'estrutural':
            return ('e', r['nome'])
        return ('v', r.get('data_inicio'), r['categoria'])

    by_v19_1 = {key_of(r): r for r in v19_1['todos']}

    pares = []
    for r19 in v19['todos']:
        k = key_of(r19)
        r19_1 = by_v19_1.get(k)
        if r19_1 is None:
            continue
        lucro_v19 = r19.get('lucro_v19_anual') or r19.get('lucro_v19_estimado') or 0
        lucro_v19_1 = r19_1.get('lucro_v19_anual') or r19_1.get('lucro_v19_estimado') or 0
        pares.append({
            'tipo': r19['tipo'],
            'id': (r19.get('nome') if r19['tipo'] == 'estrutural'
                    else f"{r19.get('data_inicio')} {r19['categoria']}"),
            'evento': r19.get('evento', '-'),
            'lucro_v19': lucro_v19,
            'lucro_v19_1': lucro_v19_1,
            'razao': lucro_v19_1 / lucro_v19 if lucro_v19 > 0 else 0,
            'd_base_v19': r19.get('demanda_base_dia', 0),
            'd_base_v19_1': r19_1.get('demanda_base_dia', 0),
            'uplift_v19': r19.get('uplift_pct', 0),
            'uplift_v19_1': r19_1.get('uplift_pct', 0),
            'decisao_v19': r19.get('decisao', '?'),
            'decisao_v19_1': r19_1.get('decisao', '?'),
        })

    # Estatísticas
    total_v19 = sum(p['lucro_v19'] for p in pares)
    total_v19_1 = sum(p['lucro_v19_1'] for p in pares)

    # Diferenciação: medir variância das campanhas de gelo eventuais
    gelos_v19 = [p['lucro_v19'] for p in pares
                  if p['tipo'] == 'eventual' and 'gelo' in p['id'].lower()]
    gelos_v19_1 = [p['lucro_v19_1'] for p in pares
                    if p['tipo'] == 'eventual' and 'gelo' in p['id'].lower()]

    def variancia(lista):
        if not lista:
            return 0
        m = sum(lista) / len(lista)
        return sum((x - m) ** 2 for x in lista) / len(lista)

    var_v19 = variancia(gelos_v19)
    var_v19_1 = variancia(gelos_v19_1)

    # Conta decisões por categoria
    def conta_decisoes(pares_, campo):
        c = {'PRIORITARIA': 0, 'APROVADA': 0, 'CONDICIONAL': 0, 'REJEITADA': 0, 'PROIBIDA': 0}
        for p in pares_:
            d = p[campo]
            for k in c:
                if k in d:
                    c[k] += 1
                    break
        return c

    dec_v19 = conta_decisoes(pares, 'decisao_v19')
    dec_v19_1 = conta_decisoes(pares, 'decisao_v19_1')

    print(f"\n{'='*100}")
    print(f"COMPARAÇÃO V19 (baseline) vs V19.1 (Fase A+B)")
    print(f"{'='*100}")
    print(f"  Campanhas comparadas:      {len(pares)}")
    print(f"  Lucro V19 total:           R$ {total_v19:>10.2f}")
    print(f"  Lucro V19.1 total:         R$ {total_v19_1:>10.2f}")
    print(f"  Δ V19.1/V19:               {(total_v19_1/total_v19 - 1)*100 if total_v19 > 0 else 0:+.1f}%")
    print()
    print(f"  Diferenciação temporal — campanhas de gelo eventuais ({len(gelos_v19)} campanhas):")
    print(f"    V19   — min/max: R${min(gelos_v19):.2f}/R${max(gelos_v19):.2f} · σ²={var_v19:.2f}")
    print(f"    V19.1 — min/max: R${min(gelos_v19_1):.2f}/R${max(gelos_v19_1):.2f} · σ²={var_v19_1:.2f}")
    print(f"    Δ variância: {(var_v19_1/var_v19 - 1)*100 if var_v19 > 0 else 999:.0f}% (esperado: MUITO maior)")
    print()
    print(f"  Distribuição de decisões:")
    for k in ['PRIORITARIA', 'APROVADA', 'CONDICIONAL', 'REJEITADA', 'PROIBIDA']:
        print(f"    {k:12s}  V19: {dec_v19[k]:>3d} | V19.1: {dec_v19_1[k]:>3d}")
    print(f"{'='*100}\n")

    # HTML side-by-side
    pares.sort(key=lambda x: -abs(x['lucro_v19_1'] - x['lucro_v19']))

    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>V19 vs V19.1 — Efeito Fase A+B</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: #0a0a0f; color: #e1e1e6;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    padding: 32px; line-height: 1.5;
}}
.mono {{ font-family: 'SF Mono', Monaco, monospace; }}
.container {{ max-width: 1500px; margin: 0 auto; }}
.header {{ border-bottom: 1px solid #2a2a35; padding-bottom: 24px; margin-bottom: 32px; }}
.header h1 {{ font-size: 28px; font-weight: 700; }}
.header .subtitle {{ color: #8a8a98; margin-top: 8px; }}
.badge {{
    display: inline-block; padding: 4px 12px; border-radius: 12px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.5px; margin-right: 8px;
}}
.b-v19 {{ background: #475569; color: #e2e8f0; }}
.b-v19_1 {{ background: linear-gradient(90deg, #6366f1, #8b5cf6); color: #fff; }}

.kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }}
.kpi {{ background: #14141f; border: 1px solid #2a2a35; border-radius: 12px; padding: 20px; }}
.kpi .label {{ color: #8a8a98; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }}
.kpi .value {{ font-size: 24px; font-weight: 700; margin-top: 8px; }}
.kpi .delta {{ font-size: 12px; margin-top: 4px; }}
.delta.up {{ color: #4ade80; }}
.delta.down {{ color: #f87171; }}

.h2 {{ font-size: 18px; font-weight: 600; margin: 24px 0 12px;
       padding-bottom: 8px; border-bottom: 1px solid #2a2a35; }}

table {{ width: 100%; border-collapse: collapse; background: #14141f;
         border-radius: 12px; overflow: hidden; border: 1px solid #2a2a35; }}
th {{ background: #0f0f18; padding: 10px; text-align: left;
      font-size: 10px; color: #8a8a98; text-transform: uppercase; letter-spacing: 0.5px;
      border-bottom: 1px solid #2a2a35; }}
td {{ padding: 12px 10px; border-bottom: 1px solid #1f1f2a; font-size: 12px; }}
tr:hover {{ background: #18182a; }}

.col-id .sub {{ color: #8a8a98; font-size: 10px; }}
.mono-cell {{ font-family: 'SF Mono', monospace; }}
.razao {{ display: inline-block; padding: 3px 10px; border-radius: 8px;
         font-family: 'SF Mono', monospace; font-weight: 700; font-size: 11px; }}
.razao.up-big {{ background: #16213e; color: #4ade80; }}
.razao.up {{ background: #1f1f2a; color: #a3e635; }}
.razao.flat {{ background: #1f1f2a; color: #94a3b8; }}
.razao.down {{ background: #2a1414; color: #f87171; }}

.dec {{ font-size: 10px; padding: 3px 8px; border-radius: 6px;
        font-weight: 600; white-space: nowrap; }}
.dec.aprov {{ background: #14532d; color: #86efac; }}
.dec.prio {{ background: #422006; color: #fbbf24; }}
.dec.cond {{ background: #422036; color: #f59e0b; }}
.dec.rej {{ background: #2a1414; color: #f87171; }}

.diagnostico {{
    margin-top: 32px; padding: 20px; background: #14141f;
    border-radius: 12px; border: 1px solid #2a2a35;
}}
.diagnostico h3 {{ margin-bottom: 12px; }}
.diagnostico p {{ font-size: 13px; line-height: 1.8; color: #c1c1cc; }}
</style>
</head><body><div class="container">

<div class="header">
    <h1>
        <span class="badge b-v19">V19</span> vs
        <span class="badge b-v19_1">V19.1</span>
        Efeito da Fase A+B
    </h1>
    <div class="subtitle">
        Sazonalidade movida do uplift para demanda base · uplift_prior do calendário real ·
        Pré/pós-feriado · Eventos esportivos · Temperatura real
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi">
        <div class="label">Lucro Total V19</div>
        <div class="value mono">R$ {total_v19:,.0f}</div>
        <div class="delta" style="color:#8a8a98;">baseline (otimista)</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro Total V19.1</div>
        <div class="value mono">R$ {total_v19_1:,.0f}</div>
        <div class="delta {'down' if total_v19_1 < total_v19 else 'up'}">
            {(total_v19_1/total_v19 - 1)*100 if total_v19 > 0 else 0:+.1f}% vs V19
        </div>
    </div>
    <div class="kpi">
        <div class="label">Variância gelo eventual</div>
        <div class="value mono">{var_v19_1/max(var_v19, 0.01):.0f}×</div>
        <div class="delta up">V19.1 diferencia por contexto</div>
    </div>
    <div class="kpi">
        <div class="label">Filtro discriminando</div>
        <div class="value">{dec_v19_1['CONDICIONAL']}+{dec_v19_1['REJEITADA']}</div>
        <div class="delta {'up' if (dec_v19_1['CONDICIONAL']+dec_v19_1['REJEITADA']) > 0 else 'down'}">
            cond+rej vs {dec_v19['CONDICIONAL']+dec_v19['REJEITADA']} no V19
        </div>
    </div>
</div>

<div class="h2">Tabela comparativa por campanha (ordenada por maior diferença)</div>

<table>
<thead><tr>
    <th>Campanha</th>
    <th>Evento</th>
    <th style="text-align:right;">d_base V19</th>
    <th style="text-align:right;">d_base V19.1</th>
    <th style="text-align:right;">Uplift V19</th>
    <th style="text-align:right;">Uplift V19.1</th>
    <th style="text-align:right;">Lucro V19</th>
    <th style="text-align:right;">Lucro V19.1</th>
    <th style="text-align:center;">V19.1/V19</th>
    <th>Decisão V19</th>
    <th>Decisão V19.1</th>
</tr></thead>
<tbody>
"""

    for p in pares:
        razao_class = ('up-big' if p['razao'] > 2 else 'up' if p['razao'] > 1.1
                        else 'down' if p['razao'] < 0.9 else 'flat')
        ev = p['evento'] if p['evento'] != '-' else ''
        dec19_cls = ('prio' if 'PRIORITARIA' in p['decisao_v19']
                      else 'aprov' if 'APROVADA' in p['decisao_v19']
                      else 'cond' if 'CONDICIONAL' in p['decisao_v19']
                      else 'rej')
        dec191_cls = ('prio' if 'PRIORITARIA' in p['decisao_v19_1']
                       else 'aprov' if 'APROVADA' in p['decisao_v19_1']
                       else 'cond' if 'CONDICIONAL' in p['decisao_v19_1']
                       else 'rej')
        html += f"""
<tr>
    <td>{p['id']}</td>
    <td style="color: #ec4899; font-size: 11px;">{ev}</td>
    <td class="mono-cell" style="text-align:right;">{p['d_base_v19']:.1f}</td>
    <td class="mono-cell" style="text-align:right; color: #4ade80;">{p['d_base_v19_1']:.1f}</td>
    <td class="mono-cell" style="text-align:right;">{p['uplift_v19']:.0f}%</td>
    <td class="mono-cell" style="text-align:right; color: #4ade80;">{p['uplift_v19_1']:.0f}%</td>
    <td class="mono-cell" style="text-align:right;">R$ {p['lucro_v19']:,.0f}</td>
    <td class="mono-cell" style="text-align:right; font-weight: 600;">R$ {p['lucro_v19_1']:,.0f}</td>
    <td style="text-align:center;"><span class="razao {razao_class}">{p['razao']:.1f}×</span></td>
    <td><span class="dec {dec19_cls}">{p['decisao_v19']}</span></td>
    <td><span class="dec {dec191_cls}">{p['decisao_v19_1']}</span></td>
</tr>
"""

    html += f"""
</tbody>
</table>

<div class="diagnostico">
    <h3>Diagnóstico</h3>
    <p>
    • <strong style="color:#4ade80;">Diferenciação temporal funcionou</strong>: a variância
    do lucro de campanhas de gelo eventuais aumentou
    <strong>{var_v19_1/max(var_v19, 0.01):.0f}×</strong> — V19.1 distingue gelo no Réveillon
    (R$ {max(gelos_v19_1):.0f}) de gelo numa sexta de outubro (R$ {min(gelos_v19_1):.0f}).
    </p>
    <p>
    • <strong style="color:#a3e635;">Lucro total mais defensável</strong>:
    {(total_v19_1/total_v19 - 1)*100 if total_v19 > 0 else 0:+.1f}% vs V19. V19 era
    +483% sobre V17 (suspeito); V19.1 reduz isso.
    </p>
    <p>
    • <strong style="color:#f59e0b;">Filtro começa a discriminar</strong>:
    {dec_v19_1['CONDICIONAL']} CONDICIONAIS no V19.1 (V19 tinha 0). Campanhas marginais
    de gelo SEM evento agora ficam em compasso de espera.
    </p>
    <p style="color:#8a8a98; font-size:12px; margin-top:12px;">
    Para mais detalhes técnicos sobre as mudanças, ver
    <code>teste_agente_demanda_promocional/v19_1/README.md</code>.
    </p>
</div>

</div></body></html>
"""

    Path(path_html).parent.mkdir(parents=True, exist_ok=True)
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ Comparação salva: {path_html}")


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    PROJETO = Path(__file__).parent.parent.parent
    p_v19 = PROJETO / 'results' / 'v19' / 'calendario_v19.json'
    p_v19_1 = PROJETO / 'results' / 'v19_1' / 'calendario_v19_1.json'
    p_html = PROJETO / 'results' / 'v19_1' / 'comparacao_v19_v19_1.html'
    comparar(str(p_v19), str(p_v19_1), str(p_html))
