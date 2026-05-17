"""Comparação V19.1 vs V19.2 — efeito da regra de combo PDV.

Mostra:
- Combos bloqueados (gelo + destilados/vinho/refri)
- Combos sobreviventes (gelo + cerveja, café + padaria)
- Impacto no lucro total e no filtro de decisão
"""
import json
from pathlib import Path
from datetime import datetime


def comparar(path_v19_1: str, path_v19_2: str, path_html: str):
    v1 = json.load(open(path_v19_1, encoding='utf-8'))
    v2 = json.load(open(path_v19_2, encoding='utf-8'))

    def key(r):
        if r['tipo'] == 'estrutural':
            return ('e', r['nome'])
        return ('v', r.get('data_inicio'), r['categoria'])

    by_v2 = {key(r): r for r in v2['todos']}
    pares = []
    for r1 in v1['todos']:
        r2 = by_v2.get(key(r1))
        if r2 is None:
            continue
        lucro_v1 = r1.get('lucro_v19_anual') or r1.get('lucro_v19_estimado') or 0
        lucro_v2 = r2.get('lucro_v19_anual') or r2.get('lucro_v19_estimado') or 0
        par_combo = r1.get('par_combo') or r1.get('produto_complementar') or ''
        pares.append({
            'id': r1.get('nome') if r1['tipo'] == 'estrutural'
                   else f"{r1.get('data_inicio')} {r1['categoria']}",
            'tipo': r1['tipo'],
            'cat': r1['categoria'],
            'par': par_combo,
            'evento': r1.get('evento', '-'),
            'lucro_v1': lucro_v1,
            'lucro_v2': lucro_v2,
            'dec_v1': r1['decisao'],
            'dec_v2': r2['decisao'],
            'mudou_decisao': r1['decisao'] != r2['decisao'],
        })

    # Estatísticas
    total_v1 = sum(p['lucro_v1'] for p in pares)
    total_v2 = sum(p['lucro_v2'] for p in pares)
    n_bloqueados = sum(1 for p in pares
                        if 'PROIBIDA' in p['dec_v2']
                        and 'PROIBIDA' not in p['dec_v1'])
    n_mudou = sum(1 for p in pares if p['mudou_decisao'])

    print(f"\n{'='*100}")
    print(f"COMPARAÇÃO V19.1 vs V19.2 — efeito da regra de combo PDV")
    print(f"{'='*100}")
    print(f"  Campanhas analisadas:        {len(pares)}")
    print(f"  Lucro V19.1:                 R$ {total_v1:>10,.2f}")
    print(f"  Lucro V19.2:                 R$ {total_v2:>10,.2f}")
    print(f"  Δ V19.2 vs V19.1:            {(total_v2/total_v1 - 1)*100 if total_v1>0 else 0:+.1f}%")
    print(f"  Combos bloqueados (PDV):     {n_bloqueados}")
    print(f"  Decisões que mudaram:        {n_mudou}")
    print(f"{'='*100}\n")

    # Categoriza
    bloqueados = [p for p in pares if 'PROIBIDA' in p['dec_v2'] and 'PROIBIDA' not in p['dec_v1']]
    sobreviventes = [p for p in pares if 'PROIBIDA' not in p['dec_v2']]

    print('COMBOS BLOQUEADOS PELA REGRA PDV:')
    for p in bloqueados:
        print(f"  ✗ {p['id']:<35s} ({p['cat']:>10s} + {p['par']:<15s}) "
                f"V19.1 R$ {p['lucro_v1']:>7.2f} → V19.2 R$ {p['lucro_v2']:>7.2f}")

    print('\nSOBREVIVENTES (combos válidos):')
    for p in sobreviventes:
        print(f"  ✓ {p['id']:<35s} ({p['cat']:>10s} + {p['par']:<15s}) "
                f"V19.1 R$ {p['lucro_v1']:>7.2f} → V19.2 R$ {p['lucro_v2']:>7.2f}  [{p['dec_v2']}]")

    # HTML
    pares.sort(key=lambda x: (0 if 'PROIBIDA' in x['dec_v2'] else 1, -x['lucro_v1']))

    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>V19.1 vs V19.2 — Regra de combo PDV</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a0f; color:#e1e1e6;
       font-family:-apple-system,Segoe UI,sans-serif; padding:32px; line-height:1.5; }}
.mono {{ font-family:'SF Mono',monospace; }}
.container {{ max-width:1500px; margin:0 auto; }}
.header {{ border-bottom:1px solid #2a2a35; padding-bottom:24px; margin-bottom:32px; }}
.header h1 {{ font-size:28px; font-weight:700; }}
.header .subtitle {{ color:#8a8a98; margin-top:8px; }}
.badge {{ display:inline-block; padding:4px 12px; border-radius:12px;
         font-size:11px; font-weight:600; text-transform:uppercase; margin-right:8px; }}
.b-v1 {{ background:#475569; color:#e2e8f0; }}
.b-v2 {{ background:linear-gradient(90deg,#ec4899,#8b5cf6); color:#fff; }}

.kpi-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:32px; }}
.kpi {{ background:#14141f; border:1px solid #2a2a35; border-radius:12px; padding:20px; }}
.kpi .label {{ color:#8a8a98; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; }}
.kpi .value {{ font-size:24px; font-weight:700; margin-top:8px; }}
.kpi .delta {{ font-size:12px; margin-top:4px; }}
.up {{ color:#4ade80; }} .down {{ color:#f87171; }} .neutral {{ color:#8a8a98; }}

.h2 {{ font-size:18px; font-weight:600; margin:24px 0 12px;
       padding-bottom:8px; border-bottom:1px solid #2a2a35; }}

table {{ width:100%; border-collapse:collapse; background:#14141f;
        border-radius:12px; overflow:hidden; border:1px solid #2a2a35; }}
th {{ background:#0f0f18; padding:10px; text-align:left;
     font-size:10px; color:#8a8a98; text-transform:uppercase; letter-spacing:0.5px;
     border-bottom:1px solid #2a2a35; }}
td {{ padding:12px 10px; border-bottom:1px solid #1f1f2a; font-size:12px; }}
tr:hover {{ background:#18182a; }}
tr.bloqueada td {{ opacity:0.7; }}
tr.bloqueada {{ background:#150d0d; }}

.mono-cell {{ font-family:'SF Mono',monospace; }}
.combo-pair {{ font-family:'SF Mono',monospace; font-size:11px; color:#c1c1cc; }}
.dec {{ font-size:10px; padding:3px 8px; border-radius:6px;
       font-weight:600; white-space:nowrap; }}
.dec.aprov {{ background:#14532d; color:#86efac; }}
.dec.prio {{ background:#422006; color:#fbbf24; }}
.dec.cond {{ background:#422036; color:#f59e0b; }}
.dec.rej {{ background:#2a1414; color:#f87171; }}
.dec.proib {{ background:#450a0a; color:#fecaca; border:1px solid #b91c1c; }}

.regra {{ margin-top:24px; padding:20px; background:#14141f;
         border-radius:12px; border:1px solid #2a2a35; }}
.regra h3 {{ margin-bottom:12px; }}
.regra p, .regra li {{ font-size:13px; line-height:1.8; color:#c1c1cc; }}
.regra ul {{ margin-left:20px; margin-top:8px; }}
.kkk {{ font-size:11px; color:#fbbf24; font-style:italic; }}
</style>
</head><body><div class="container">

<div class="header">
    <h1>
        <span class="badge b-v1">V19.1</span> vs
        <span class="badge b-v2">V19.2</span>
        Regra de combo no PDV
    </h1>
    <div class="subtitle">
        Saco de gelo 5kg + garrafa de whisky no balcão = combo absurdo · Insight do Vinicius (15/05/2026)
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi">
        <div class="label">Combos bloqueados</div>
        <div class="value down">{n_bloqueados}</div>
        <div class="delta">vs 0 no V19.1</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro V19.1</div>
        <div class="value mono">R$ {total_v1:,.0f}</div>
        <div class="delta neutral">aprovava tudo (filtro fraco)</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro V19.2</div>
        <div class="value mono">R$ {total_v2:,.0f}</div>
        <div class="delta down">{(total_v2/total_v1 - 1)*100 if total_v1>0 else 0:+.1f}% (menos otimismo)</div>
    </div>
    <div class="kpi">
        <div class="label">Decisões mudadas</div>
        <div class="value">{n_mudou} / {len(pares)}</div>
        <div class="delta up">filtro discriminando de verdade</div>
    </div>
</div>

<div class="regra">
    <h3>Regra de combo PDV — V19.2</h3>
    <p>
    O insight do Vinicius é direto: <span class="kkk">"Não faz sentido combo gelo + whisky.
    Vou levar um saco de gelo de 5kg e uma garrafa no balcão? kkkkk"</span>
    </p>
    <p>
    Cada produto tem um <strong>modo de consumo</strong>:
    </p>
    <ul>
        <li><strong>Individual/balcão</strong>: cliente leva 1 unidade já (long-neck, choc.impulso, refri lata, snack)</li>
        <li><strong>Cesta evento</strong>: cliente prepara festa em casa (saco gelo 5kg, garrafa premium)</li>
        <li><strong>Rotina manhã</strong>: café + padaria</li>
        <li><strong>Presente</strong>: chocolate premium para data comercial</li>
    </ul>
    <p>
    Combos só fazem sentido <strong>dentro da mesma ocasião</strong>. V19.2 implementa
    isso via <code>COMBOS_INVALIDOS_PDV</code> (no <code>demand_agent.py</code>) — flag
    automática que vira PROIBIDA na decisão final.
    </p>
</div>

<div class="h2">Comparação campanha-a-campanha</div>

<table>
<thead><tr>
    <th>Campanha</th>
    <th>Combo</th>
    <th>Evento</th>
    <th style="text-align:right;">Lucro V19.1</th>
    <th style="text-align:right;">Lucro V19.2</th>
    <th>Decisão V19.1</th>
    <th>Decisão V19.2</th>
</tr></thead>
<tbody>
"""

    for p in pares:
        bloqueada = 'PROIBIDA' in p['dec_v2']
        row_cls = 'bloqueada' if bloqueada else ''
        dec1_cls = ('prio' if 'PRIORITARIA' in p['dec_v1']
                     else 'aprov' if 'APROVADA' in p['dec_v1']
                     else 'cond' if 'CONDICIONAL' in p['dec_v1']
                     else 'proib' if 'PROIBIDA' in p['dec_v1']
                     else 'rej')
        dec2_cls = ('prio' if 'PRIORITARIA' in p['dec_v2']
                     else 'aprov' if 'APROVADA' in p['dec_v2']
                     else 'cond' if 'CONDICIONAL' in p['dec_v2']
                     else 'proib' if 'PROIBIDA' in p['dec_v2']
                     else 'rej')
        ev = p['evento'] if p['evento'] not in ('-', '') else ''
        ev_html = f'<span style="color:#ec4899; font-size:11px;">🎉 {ev}</span>' if ev else ''
        html += f"""
<tr class="{row_cls}">
    <td>{p['id']}</td>
    <td class="combo-pair">{p['cat']} + {p['par']}</td>
    <td>{ev_html}</td>
    <td class="mono-cell" style="text-align:right;">R$ {p['lucro_v1']:,.0f}</td>
    <td class="mono-cell" style="text-align:right; {'color:#f87171;' if bloqueada else 'font-weight:600;'}">R$ {p['lucro_v2']:,.0f}</td>
    <td><span class="dec {dec1_cls}">{p['dec_v1']}</span></td>
    <td><span class="dec {dec2_cls}">{p['dec_v2']}</span></td>
</tr>
"""

    html += """
</tbody>
</table>

</div></body></html>
"""

    Path(path_html).parent.mkdir(parents=True, exist_ok=True)
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✓ Comparação salva: {path_html}")


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    PROJETO = Path(__file__).parent.parent.parent
    p_v1 = PROJETO / 'results' / 'v19_1' / 'calendario_v19_1.json'
    p_v2 = PROJETO / 'results' / 'v19_2' / 'calendario_v19_2.json'
    p_html = PROJETO / 'results' / 'v19_2' / 'comparacao_v19_1_v19_2.html'
    comparar(str(p_v1), str(p_v2), str(p_html))
