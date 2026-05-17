"""Gera HTML visual estilo CALENDÁRIO para o dono do posto.

Grade mensal de 12 meses, cada dia colorido por tipo de promoção,
com legenda, resumo executivo e destaques (Esquenta de Sexta, Copa,
datas comerciais). Linguagem do dono, não técnica.
"""
import io
import json
import sys
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path

MESES = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
          'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
DIAS_SEM = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']

EMOJI = {
    'cerveja': '🍺', 'gelo': '🧊', 'cafe': '☕', 'padaria': '🥐',
    'chocolate_caixa': '🎁', 'chocolate_unit': '🍫', 'doce_balcao': '🍬',
    'biscoito': '🍪', 'sorvete': '🍦', 'refrigerante': '🥤', 'suco': '🧃',
    'agua': '💧', 'energetico': '⚡', 'isotonico': '🥤', 'sanduiche': '🥪',
    'snack_salgado': '🥨', 'destilados': '🥃', 'whisky': '🥃', 'vinho': '🍷',
    'cha_pronto': '🍵',
}

def nome_bonito(cat):
    return {
        'cerveja': 'Cerveja', 'gelo': 'Gelo', 'cafe': 'Café',
        'padaria': 'Padaria', 'chocolate_caixa': 'Chocolate (caixa)',
        'chocolate_unit': 'Chocolate', 'doce_balcao': 'Balas/Doces',
        'biscoito': 'Biscoito', 'sorvete': 'Sorvete',
        'refrigerante': 'Refrigerante', 'suco': 'Suco', 'agua': 'Água',
        'energetico': 'Energético', 'isotonico': 'Isotônico',
        'sanduiche': 'Sanduíche', 'snack_salgado': 'Salgadinho/Amendoim',
        'cha_pronto': 'Chá',
    }.get(cat, cat)


def intensidade_label(i):
    return {
        'combo': 'COMBO', 'desc3%': '-3%', 'desc5%': '-5%',
        'desc7%': '-7%', 'desc10%': '-10%', 'nada': '—',
    }.get(i, i)


def gerar(path_cal, path_html):
    cal = json.load(open(path_cal, encoding='utf-8'))
    campanhas = cal['campanhas']
    s = cal['sumario']

    # Indexa por data
    por_data = {}
    for c in campanhas:
        por_data.setdefault(c['data_inicio'], []).append(c)

    data_inicio = datetime.strptime(cal['data_inicio_rollout'], '%Y-%m-%d')

    # Stats para resumo
    n_combo = sum(1 for c in campanhas if c['intensidade'] == 'combo')
    n_desc = len(campanhas) - n_combo
    esquenta = [c for c in campanhas if c['categoria'] == 'cerveja'
                and c.get('par_combo') == 'snack_salgado']
    copa = [c for c in campanhas if 'Copa' in str(c.get('evento_proximo'))]
    eventos = Counter(c.get('evento_proximo') for c in campanhas
                       if c.get('evento_proximo'))

    # Top combos legíveis
    pares = Counter()
    for c in campanhas:
        if c.get('par_combo'):
            pares[(c['categoria'], c['par_combo'])] += 1

    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>Calendário de Promoções — Auto Posto Parque Viana</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#f4f6f9; color:#1a2230; font-family:-apple-system,'Segoe UI',Roboto,sans-serif; padding:28px; }}
.container {{ max-width:1400px; margin:0 auto; }}
.header {{ background:linear-gradient(120deg,#1e3a8a,#2563eb); color:#fff;
          padding:28px 32px; border-radius:16px; margin-bottom:24px;
          box-shadow:0 4px 20px rgba(37,99,235,.25); }}
.header h1 {{ font-size:26px; font-weight:800; }}
.header p {{ opacity:.9; margin-top:6px; font-size:14px; }}
.resumo {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:24px; }}
.card {{ background:#fff; border-radius:14px; padding:20px;
        box-shadow:0 2px 10px rgba(0,0,0,.05); border:1px solid #e6eaf0; }}
.card .num {{ font-size:30px; font-weight:800; color:#2563eb; }}
.card .lbl {{ font-size:12px; color:#64748b; text-transform:uppercase;
             letter-spacing:.5px; margin-top:4px; font-weight:600; }}
.card .sub {{ font-size:12px; color:#94a3b8; margin-top:6px; }}
.destaques {{ background:#fff; border-radius:14px; padding:22px; margin-bottom:24px;
             border:1px solid #e6eaf0; box-shadow:0 2px 10px rgba(0,0,0,.05); }}
.destaques h2 {{ font-size:17px; margin-bottom:14px; color:#1e3a8a; }}
.destaque-item {{ display:flex; align-items:center; gap:10px; padding:9px 0;
                 border-bottom:1px solid #f1f4f8; font-size:14px; }}
.destaque-item:last-child {{ border-bottom:0; }}
.destaque-item .ico {{ font-size:22px; }}
.destaque-item b {{ color:#1e3a8a; }}
.legenda {{ display:flex; gap:18px; flex-wrap:wrap; background:#fff;
           padding:14px 20px; border-radius:12px; margin-bottom:20px;
           border:1px solid #e6eaf0; font-size:13px; }}
.legenda span {{ display:flex; align-items:center; gap:6px; }}
.dot {{ width:14px; height:14px; border-radius:4px; display:inline-block; }}
.d-combo {{ background:#2563eb; }}
.d-desc {{ background:#f59e0b; }}
.d-evento {{ background:#ec4899; }}
.d-esquenta {{ background:#16a34a; }}
.d-copa {{ background:#eab308; }}
.meses {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }}
.mes {{ background:#fff; border-radius:14px; padding:16px;
       border:1px solid #e6eaf0; box-shadow:0 2px 8px rgba(0,0,0,.04); }}
.mes h3 {{ font-size:15px; color:#1e3a8a; margin-bottom:10px; font-weight:700; }}
.grade {{ display:grid; grid-template-columns:repeat(7,1fr); gap:3px; }}
.gd-head {{ font-size:10px; color:#94a3b8; text-align:center; font-weight:700;
           padding:3px 0; }}
.dia {{ aspect-ratio:1; border-radius:6px; background:#f1f4f8;
       display:flex; flex-direction:column; align-items:center; justify-content:center;
       font-size:10px; color:#94a3b8; position:relative; cursor:default; }}
.dia.tem {{ font-weight:700; color:#fff; }}
.dia.combo {{ background:#2563eb; }}
.dia.desc {{ background:#f59e0b; color:#fff; }}
.dia.evento {{ background:#ec4899; color:#fff; }}
.dia.esquenta {{ background:#16a34a; color:#fff; }}
.dia.copa {{ background:#eab308; color:#1a2230; }}
.dia .emoji {{ font-size:13px; line-height:1; }}
.dia .dnum {{ font-size:8px; position:absolute; top:2px; left:3px; opacity:.7; }}
.dia:hover .tip {{ display:block; }}
.tip {{ display:none; position:absolute; bottom:105%; left:50%;
       transform:translateX(-50%); background:#1a2230; color:#fff;
       padding:7px 10px; border-radius:7px; font-size:11px; white-space:nowrap;
       z-index:10; box-shadow:0 4px 14px rgba(0,0,0,.3); font-weight:500; }}
.footer {{ text-align:center; color:#94a3b8; font-size:12px; margin-top:28px; }}
.tabela {{ background:#fff; border-radius:14px; padding:20px; margin-top:24px;
          border:1px solid #e6eaf0; }}
.tabela h2 {{ font-size:17px; color:#1e3a8a; margin-bottom:14px; }}
.tabela table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.tabela th {{ text-align:left; color:#64748b; font-size:11px; text-transform:uppercase;
             padding:8px; border-bottom:2px solid #e6eaf0; }}
.tabela td {{ padding:9px 8px; border-bottom:1px solid #f1f4f8; }}
.tabela tr:hover {{ background:#f8fafc; }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:20px;
        font-size:11px; font-weight:700; }}
.pill.combo {{ background:#dbeafe; color:#1e40af; }}
.pill.desc {{ background:#fef3c7; color:#92400e; }}
</style>
</head><body><div class="container">

<div class="header">
  <h1>📅 Calendário de Promoções — Auto Posto Parque Viana</h1>
  <p>Plano de 12 meses gerado pelo agente · {len(campanhas)} promoções recomendadas ·
     início {data_inicio.strftime('%d/%m/%Y')}</p>
</div>

<div class="resumo">
  <div class="card">
    <div class="num">{len(campanhas)}</div>
    <div class="lbl">Promoções no ano</div>
    <div class="sub">{n_combo} combos · {n_desc} descontos diretos</div>
  </div>
  <div class="card">
    <div class="num">R$ {s['lucro_total_R$']:,.0f}</div>
    <div class="lbl">Lucro adicional estimado</div>
    <div class="sub">média R$ {s['lucro_total_R$']/max(len(campanhas),1):,.0f} por promoção</div>
  </div>
  <div class="card">
    <div class="num">{len(esquenta)}</div>
    <div class="lbl">Esquenta de Sexta</div>
    <div class="sub">Cerveja + Salgadinho nas sextas/sábados</div>
  </div>
  <div class="card">
    <div class="num">{len(copa)}</div>
    <div class="lbl">Promoções na Copa</div>
    <div class="sub">Cerveja + petisco nos jogos do Brasil</div>
  </div>
</div>

<div class="destaques">
  <h2>⭐ Destaques do plano</h2>
  <div class="destaque-item">
    <span class="ico">🍺</span>
    <div><b>Esquenta de Sexta</b> — toda sexta e sábado o agente recomenda
    Cerveja + Salgadinho. É o combo que mais aparece ({pares.most_common(1)[0][1]}x no ano).</div>
  </div>
  <div class="destaque-item">
    <span class="ico">⚽</span>
    <div><b>Copa do Mundo</b> — nos dias de jogo do Brasil, promoção de
    Cerveja + Salgadinho para a galera assistir.</div>
  </div>
  <div class="destaque-item">
    <span class="ico">🎁</span>
    <div><b>Datas comerciais</b> — Dia das Mães, Namorados, Pais e Crianças:
    combo de Chocolate (caixa) como presente de última hora.</div>
  </div>
  <div class="destaque-item">
    <span class="ico">☕</span>
    <div><b>Manhã do trabalhador</b> — Café + Biscoito/Pão para quem para
    no posto antes do trabalho.</div>
  </div>
</div>

<div class="legenda">
  <span><i class="dot d-esquenta"></i> Esquenta de Sexta (cerveja)</span>
  <span><i class="dot d-copa"></i> Jogo da Copa</span>
  <span><i class="dot d-evento"></i> Data comercial (Mães, Namorados...)</span>
  <span><i class="dot d-combo"></i> Combo normal</span>
  <span><i class="dot d-desc"></i> Desconto direto</span>
</div>

<div class="meses">
"""

    # Gera 12 meses a partir da data inicial
    cur = datetime(data_inicio.year, data_inicio.month, 1)
    for _ in range(12):
        ano, mes = cur.year, cur.month
        html += f'<div class="mes"><h3>{MESES[mes-1]} {ano}</h3><div class="grade">'
        for d in DIAS_SEM:
            html += f'<div class="gd-head">{d}</div>'
        # Espaço antes do dia 1
        primeiro = datetime(ano, mes, 1)
        offset = primeiro.weekday()
        for _ in range(offset):
            html += '<div class="dia"></div>'
        # Dias do mês
        dia = primeiro
        while dia.month == mes:
            ds = dia.strftime('%Y-%m-%d')
            promos = por_data.get(ds, [])
            if promos:
                p = max(promos, key=lambda x: x['lucro_total_acumulado'])
                cat = p['categoria']
                par = p.get('par_combo')
                ev = p.get('evento_proximo')
                # Classe (prioridade visual)
                if cat == 'cerveja' and par == 'snack_salgado' and dia.weekday() in (4,5):
                    cls = 'esquenta'
                elif ev and 'Copa' in str(ev):
                    cls = 'copa'
                elif ev:
                    cls = 'evento'
                elif p['intensidade'] == 'combo':
                    cls = 'combo'
                else:
                    cls = 'desc'
                emoji = EMOJI.get(cat, '🏷️')
                par_txt = f' + {nome_bonito(par)}' if par else ''
                ev_txt = f' · {ev}' if ev else ''
                tip = f"{nome_bonito(cat)}{par_txt} · {intensidade_label(p['intensidade'])}{ev_txt} · R$ {p['lucro_total_acumulado']:.0f}"
                html += (f'<div class="dia tem {cls}"><span class="dnum">{dia.day}</span>'
                          f'<span class="emoji">{emoji}</span>'
                          f'<span class="tip">{tip}</span></div>')
            else:
                html += f'<div class="dia"><span class="dnum">{dia.day}</span></div>'
            dia += timedelta(days=1)
        html += '</div></div>'
        # Próximo mês
        if mes == 12:
            cur = datetime(ano+1, 1, 1)
        else:
            cur = datetime(ano, mes+1, 1)

    html += "</div>"

    # Tabela: promoções por data comercial
    html += """
<div class="tabela">
  <h2>🎉 Promoções nas datas comerciais</h2>
  <table>
    <tr><th>Data</th><th>Evento</th><th>Promoção</th><th>Tipo</th><th>Lucro estimado</th></tr>
"""
    evt_camp = sorted([c for c in campanhas if c.get('evento_proximo')],
                       key=lambda x: x['data_inicio'])
    for c in evt_camp:
        dt = datetime.strptime(c['data_inicio'], '%Y-%m-%d')
        cat = nome_bonito(c['categoria'])
        par = f" + {nome_bonito(c['par_combo'])}" if c.get('par_combo') else ''
        tipo = c['intensidade']
        pill = 'combo' if tipo == 'combo' else 'desc'
        html += (f"<tr><td>{dt.strftime('%d/%m/%Y')} ({DIAS_SEM[dt.weekday()]})</td>"
                  f"<td>{c['evento_proximo']}</td>"
                  f"<td>{EMOJI.get(c['categoria'],'')} {cat}{par}</td>"
                  f"<td><span class='pill {pill}'>{intensidade_label(tipo)}</span></td>"
                  f"<td>R$ {c['lucro_total_acumulado']:.0f}</td></tr>")
    html += """
  </table>
</div>

<div class="footer">
  Gerado pelo agente de Reinforcement Learning · Auto Posto Parque Viana, Barueri/SP<br>
  Os valores são estimativas do modelo. Passe o mouse sobre os dias do calendário para ver o detalhe.
</div>

</div></body></html>
"""

    Path(path_html).parent.mkdir(parents=True, exist_ok=True)
    with open(path_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ Calendário visual salvo: {path_html}")


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    PROJ = Path(__file__).parent.parent
    gerar(PROJ / 'results' / 'v21' / 'calendario_v21.json',
          PROJ / 'results' / 'v21' / 'calendario_visual.html')
