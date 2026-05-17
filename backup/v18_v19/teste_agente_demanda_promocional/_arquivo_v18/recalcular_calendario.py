"""V18 — Aplica o agente de demanda promocional em calendário existente.

Pega calendário V17 (com business rules), passa cada campanha pelo
DemandPromotionAgent, recalcula lucro REALISTA, filtra campanhas com
ROI < 1.0 e gera novo calendário V18 + dashboard comparativo.

Uso:
    python recalcular_calendario.py
        --input input_calendarios/v17_business_rules.json
        --output output/v18_demanda_realista.json
"""
import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                     default='input_calendarios/v17_business_rules.json')
parser.add_argument('--output', type=str,
                     default='output/v18_demanda_realista.json')
parser.add_argument('--roi_minimo', type=float, default=1.0,
                     help='ROI mínimo para aprovar campanha (default 1.0×)')
args = parser.parse_args()


from demand_promotion_agent import DemandPromotionAgent

print(f"Carregando {args.input}…")
with open(ROOT / args.input, encoding='utf-8') as f:
    cal = json.load(f)

agent = DemandPromotionAgent()
eventuais_originais = cal['campanhas_eventuais']
print(f"  {len(eventuais_originais)} campanhas eventuais originais")

# ── Aplicar agente em CADA campanha ─────────────────────────────────

eventuais_avaliadas = []
rejeitadas_baixo_roi = []
flag_count = {'baixo_uplift': 0, 'canibalizacao_alta': 0,
              'roi_baixo': 0, 'prejuizo': 0}

print()
print("=" * 100)
print("AVALIANDO CAMPANHAS COM AGENTE DE DEMANDA PROMOCIONAL")
print("=" * 100)

for c in eventuais_originais:
    est = agent.estimar(c)

    # Enriquece campanha com estimativas
    c_novo = dict(c)
    c_novo['demanda_base_dia'] = est.demanda_base_dia
    c_novo['demanda_promocional_dia'] = est.demanda_promocional_dia
    c_novo['uplift_pct'] = est.uplift_pct
    c_novo['uplift_unidades_dia'] = est.uplift_unidades_dia
    c_novo['boost_elasticidade'] = est.boost_elasticidade
    c_novo['boost_combo'] = est.boost_combo
    c_novo['boost_evento'] = est.boost_evento
    c_novo['boost_clima'] = est.boost_clima
    c_novo['boost_dow'] = est.boost_dow
    c_novo['canibalizacao_pct'] = est.canibalizacao_pct
    c_novo['halo_pct'] = est.halo_pct

    # LUCRO REALISTA: substitui o antigo
    lucro_original = c.get('lucro_adicional_estimado_R$', 0)
    lucro_total_realista = est.lucro_liquido_dia * c.get('dias_total', 1)
    lucro_liquido_apos_custo = lucro_total_realista - est.custo_operacional

    c_novo['lucro_adicional_estimado_R$_original'] = lucro_original
    c_novo['lucro_bruto_promo_R$'] = round(lucro_total_realista, 2)
    c_novo['custo_operacional_R$'] = est.custo_operacional
    c_novo['lucro_adicional_estimado_R$'] = round(lucro_liquido_apos_custo, 2)
    c_novo['roi'] = est.roi
    c_novo['confianca_uplift'] = est.confianca
    c_novo['flags'] = est.flags
    c_novo['justificativa_agente'] = est.justificativa

    for f in est.flags:
        flag_count[f] = flag_count.get(f, 0) + 1

    # Aprovar se ROI ≥ mínimo
    if est.roi >= args.roi_minimo:
        eventuais_avaliadas.append(c_novo)
        status = '✓'
    else:
        rejeitadas_baixo_roi.append(c_novo)
        status = '✗'

    # Print resumido
    par = f" + {c.get('produto_complementar', '')}" if c.get('produto_complementar') else ''
    eventos = c.get('eventos_comerciais_na_janela', [])
    evt = f"[{eventos[0]}]" if eventos else ''
    print(f"  {status} {c['data_inicio']} {c['categoria']}{par:<20s} "
          f"  d_base={est.demanda_base_dia:.1f}→d_promo={est.demanda_promocional_dia:.1f} "
          f"({est.uplift_pct:+.1f}%)  ROI {est.roi:.2f}× "
          f"  lucro_real R$ {lucro_liquido_apos_custo:+.0f} {evt}")

# ── Sumário e estatísticas ───────────────────────────────────────────

print()
print("=" * 100)
print("SUMÁRIO DA AVALIAÇÃO")
print("=" * 100)
print(f"  Campanhas avaliadas:        {len(eventuais_originais)}")
print(f"  APROVADAS (ROI ≥ {args.roi_minimo:.2f}):    {len(eventuais_avaliadas)}")
print(f"  REJEITADAS (ROI < {args.roi_minimo:.2f}):   {len(rejeitadas_baixo_roi)}")
print()
print("Flags detectadas:")
for f, n in flag_count.items():
    print(f"  · {f}: {n}")

# Comparar lucro original vs realista
lucro_orig_total = sum(c.get('lucro_adicional_estimado_R$_original', 0)
                         for c in eventuais_originais if 'lucro_adicional_estimado_R$_original' not in c)
lucro_orig_total = sum(c.get('lucro_adicional_estimado_R$', 0)
                         for c in eventuais_originais)
lucro_realista_aprovadas = sum(c.get('lucro_adicional_estimado_R$', 0)
                                  for c in eventuais_avaliadas)
lucro_realista_total = sum(c['lucro_adicional_estimado_R$']
                              for c in eventuais_avaliadas + rejeitadas_baixo_roi)

print()
print("COMPARAÇÃO DE LUCRO:")
print(f"  Lucro V17 (otimista, demanda base fixa):  R$ {lucro_orig_total:,.2f}")
print(f"  Lucro V18 realista (todas avaliadas):     R$ {lucro_realista_total:,.2f}")
print(f"  Lucro V18 só aprovadas (ROI ≥ {args.roi_minimo}):       R$ {lucro_realista_aprovadas:,.2f}")
print(f"  Diferença (real vs otimista):             "
      f"{(lucro_realista_total - lucro_orig_total) / max(abs(lucro_orig_total), 1) * 100:+.1f}%")

# ── Salvar resultado ────────────────────────────────────────────────

cal_novo = dict(cal)
cal_novo['versao'] = 'V18 — Demanda promocional realista (agente novo)'
cal_novo['campanhas_eventuais'] = eventuais_avaliadas
cal_novo['campanhas_eventuais_rejeitadas'] = rejeitadas_baixo_roi
cal_novo['sumario'] = {
    **cal.get('sumario', {}),
    'campanhas_eventuais': len(eventuais_avaliadas),
    'campanhas_rejeitadas_baixo_roi': len(rejeitadas_baixo_roi),
    'lucro_eventuais_R$': round(lucro_realista_aprovadas, 2),
    'lucro_eventuais_R$_original': round(lucro_orig_total, 2),
    'flags': flag_count,
}
cal_novo['sumario']['lucro_total_anual_R$'] = round(
    cal_novo['sumario'].get('lucro_estruturais_anual_R$', 0)
    + lucro_realista_aprovadas, 2
)

OUT = ROOT / args.output
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(cal_novo, f, indent=2, ensure_ascii=False, default=str)

print()
print(f"✓ Saída: {OUT}")
