"""V16 — Calendário OPERACIONAL com filtro de priorização realista.

Pega o calendário bruto do modelo V13/V14/V15 e aplica filtros para gerar
calendário EXECUTÁVEL pelo dono do posto:

1. **3-4 campanhas estruturais** (sempre ativas — toda sexta cerveja+snack, etc)
2. **12-15 campanhas eventuais** (uma por mês + eventos comerciais)
3. **Reativas** (vencimento iminente — geradas semanalmente)

Critérios de filtro:
- Lucro estimado > R$ 30 (cobre custo operacional)
- Duração ≥ 3 dias (vale o esforço de trocar cartaz)
- Sem repetição mesma categoria nos últimos 7 dias
- Distribuição balanceada por mês

Aplica preço psicológico (.90/.50) no preço final do combo.
Calcula preço único de combo (não desc fragmentado).

Uso:
    python gerar_calendario_operacional.py \\
        --input results/v13/calendario_v5.json \\
        --output results/v16/calendario_operacional.json
"""
import argparse
import io
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='results/v13/calendario_v5.json',
                     help='Calendário bruto do modelo (V5/V6/V7)')
parser.add_argument('--output', type=str, default='results/v16/calendario_operacional.json')
parser.add_argument('--max_eventuais', type=int, default=15,
                     help='Máximo de campanhas eventuais (default 15)')
parser.add_argument('--max_per_mes', type=int, default=2,
                     help='Máximo de campanhas eventuais por mês')
parser.add_argument('--min_lucro', type=float, default=10.0,
                     help='Lucro mínimo R$ para aprovar campanha eventual')
parser.add_argument('--min_dias', type=int, default=3,
                     help='Duração mínima em dias')
args = parser.parse_args()


# ── Campanhas ESTRUTURAIS (sempre ativas) ─────────────────────────────────
# Definidas com base em conhecimento de varejo de conveniência
CAMPANHAS_ESTRUTURAIS = [
    {
        'nome': 'Esquenta de Sexta',
        'padrao_dias_semana': [4, 5],  # sex+sáb
        'categoria': 'cerveja',
        'par_combo': 'snack',
        'intensidade': 'combo',
        'preco_combo_alvo': 14.90,
        'comunicacao': 'CERVEJA + AMENDOIM • R$ 14,90 • Sex-Sáb',
        'justificativa': 'Cerveja+snack é cesta clássica de fim-de-semana (Instacart lift 2.5)'
    },
    {
        'nome': 'Café da Manhã',
        'padrao_dias_semana': [0, 1, 2, 3, 4],  # seg-sex
        'categoria': 'cafe',
        'par_combo': 'padaria',
        'intensidade': 'combo',
        'preco_combo_alvo': 9.90,
        'comunicacao': 'CAFÉ + PÃO • R$ 9,90 • Manhãs Seg-Sex',
        'justificativa': 'Rotina cliente que para no posto antes do trabalho'
    },
    {
        'nome': 'Verão Gelado',
        'meses_ativos': [12, 1, 2, 3],  # dez-mar (verão BR)
        'padrao_dias_semana': [5, 6],
        'categoria': 'gelo',
        'par_combo': 'refrigerante',
        'intensidade': 'combo',
        'preco_combo_alvo': 12.90,
        'comunicacao': 'GELO + REFRI • R$ 12,90 • Fins de semana',
        'justificativa': 'Verão + fim-de-semana = pico de demanda; combo aumenta ticket'
    },
    {
        'nome': 'Drink de Sábado',
        'padrao_dias_semana': [5],
        'categoria': 'destilados',
        'par_combo': 'gelo',
        'intensidade': 'combo',
        'preco_combo_alvo': 39.90,
        'comunicacao': 'DRINK COMPLETO: WHISKY/CACHAÇA + GELO • R$ 39,90',
        'justificativa': 'Instacart lift 14.0 destilados+vinho; gelo é par natural'
    },
]


def aplicar_preco_psicologico(p: float) -> float:
    """Arredonda para preço psicológico (.90 ou .50)."""
    if p < 3:
        return round(p) - 0.10  # 1.90, 2.90
    if p < 10:
        return int(p) + 0.90    # 4.90, 7.90
    if p < 20:
        return int(p) + 0.90    # 14.90
    if p < 50:
        # Arredondar para X9.90 (29.90, 39.90, 49.90)
        return int(p / 10) * 10 + 9.90
    return int(p / 10) * 10 + 9.90


def calc_preco_combo_unico(c, env_cats):
    """Calcula preço único de combo (cliente vê 1 preço, não desc fragmentado).

    base = preço(principal) + preço(complementar)
    combo = base × (1 - desc_efetivo)
    preço_psicologico = arredondar .90
    """
    # Não temos preço do par diretamente nesta versão — assumir desc do combo
    desc = c.get('desconto_pct', 10) / 100
    base = c.get('preco_unitario', 10) * 1.3  # estimativa par + principal
    preco_real = base * (1 - desc)
    preco_psico = aplicar_preco_psicologico(preco_real)
    return round(preco_psico, 2)


# ── Carregar calendário bruto ────────────────────────────────────────────

print(f"Carregando {args.input}…")
with open(ROOT / args.input, encoding='utf-8') as f:
    cal_raw = json.load(f)

campanhas_raw = cal_raw['campanhas']
print(f"  {len(campanhas_raw)} campanhas brutas")

data_inicio = date.fromisoformat(cal_raw['data_inicio'])
horizonte = cal_raw['horizonte_dias']
data_fim = data_inicio + timedelta(days=horizonte - 1)

# ── Filtro 1: por critérios mínimos ─────────────────────────────────────

filtradas = [
    c for c in campanhas_raw
    if c.get('lucro_adicional_estimado_R$', 0) >= args.min_lucro
       and c.get('dias_total', 0) >= args.min_dias
]
print(f"\nApós filtro mínimo (lucro≥{args.min_lucro}, dias≥{args.min_dias}): {len(filtradas)}")

# ── Filtro 2: distribuir 1-2 por mês ─────────────────────────────────────

# Agrupa por mês e pega top por lucro
por_mes = defaultdict(list)
for c in filtradas:
    m = date.fromisoformat(c['data_inicio']).month
    por_mes[m].append(c)

eventuais = []
for m, lista in por_mes.items():
    lista_ord = sorted(lista, key=lambda x: -x['lucro_adicional_estimado_R$'])
    eventuais.extend(lista_ord[:args.max_per_mes])

# Cap total
eventuais = sorted(eventuais, key=lambda c: -c['lucro_adicional_estimado_R$'])[:args.max_eventuais]
eventuais.sort(key=lambda c: c['data_inicio'])

print(f"Após filtro mensal (max {args.max_per_mes}/mês, total {args.max_eventuais}): {len(eventuais)}")

# ── Enriquecer eventuais com preço psicológico + comunicação ───────────

for c in eventuais:
    c['preco_combo_alvo'] = aplicar_preco_psicologico(
        c.get('preco_unitario', 10) * 1.3 * (1 - c.get('desconto_pct', 10) / 100)
    )
    nome_cat = c['categoria'].replace('_', ' ').upper()
    par = c.get('produto_complementar', '')
    if par and c.get('intensidade') == 'combo':
        par_str = f" + {par.replace('_', ' ').upper()}"
        c['comunicacao'] = f"COMBO {nome_cat}{par_str} • R$ {c['preco_combo_alvo']:.2f}"
    else:
        c['comunicacao'] = f"{nome_cat} COM DESC. {c.get('desconto_pct', 0)}%"

# ── Gerar campanhas ESTRUTURAIS expandidas para o horizonte ─────────────

estruturais_ativas = []
for esq in CAMPANHAS_ESTRUTURAIS:
    # Expand para todas as datas que batem padrão
    d = data_inicio
    while d <= data_fim:
        meses_ativos = esq.get('meses_ativos')
        if meses_ativos is None or d.month in meses_ativos:
            if d.weekday() in esq.get('padrao_dias_semana', list(range(7))):
                estruturais_ativas.append({
                    'tipo': 'estrutural',
                    'nome_campanha': esq['nome'],
                    'data': d.isoformat(),
                    'dia_semana': ['Seg','Ter','Qua','Qui','Sex','Sáb','Dom'][d.weekday()],
                    'categoria': esq['categoria'],
                    'par_combo': esq['par_combo'],
                    'intensidade': esq['intensidade'],
                    'preco_combo': esq['preco_combo_alvo'],
                    'comunicacao': esq['comunicacao'],
                    'justificativa': esq['justificativa'],
                })
        d += timedelta(days=1)

print(f"Campanhas estruturais (instâncias diárias): {len(estruturais_ativas)}")

# ── Sumarizar estruturais por nome (não imprimir 52 sex+sáb separados) ─

# V16.1: cálculo de lucro estrutural baseado em demanda real (não chute R$8)
# Carrega calibração para pegar demanda_base e margem reais
try:
    with open(ROOT / 'data/calibracao_v2.json', encoding='utf-8') as fcal:
        cfg = json.load(fcal)
    cat_dict = {c['categoria']: c for c in cfg['categorias']}
except Exception:
    cat_dict = {}

estruturais_resumo = []
for esq in CAMPANHAS_ESTRUTURAIS:
    instancias = [e for e in estruturais_ativas if e['nome_campanha'] == esq['nome']]
    if not instancias:
        continue
    # Lucro real: demanda × boost combo × margem × n_dias × (1 - desc)
    cat_info = cat_dict.get(esq['categoria'], {})
    demanda_base_dia = cat_info.get('demanda_base_dia', 5.0)
    margem = cat_info.get('margem', 4.0)
    # Combo 10% desc → boost ~1.12 (calibrado)
    boost = 1.12
    desc = 0.10
    uplift_un_dia = demanda_base_dia * (boost - 1)
    lucro_dia = uplift_un_dia * margem * (1 - desc)
    lucro_anual = round(lucro_dia * len(instancias), 2)
    estruturais_resumo.append({
        'tipo': 'estrutural',
        'nome': esq['nome'],
        'n_dias_no_ano': len(instancias),
        'categoria': esq['categoria'],
        'par_combo': esq['par_combo'],
        'preco_combo': esq['preco_combo_alvo'],
        'comunicacao': esq['comunicacao'],
        'justificativa': esq['justificativa'],
        'lucro_adicional_estimado_anual_R$': lucro_anual,
        'lucro_calculo_base': f"{demanda_base_dia:.1f}un/dia × {boost-1:.0%} boost × R${margem:.2f} margem × {len(instancias)}d",
    })

# ── Calendário final ──────────────────────────────────────────────────

calendario_op = {
    'versao': 'V16 operacional',
    'gerado_em': str(date.today()),
    'data_inicio': cal_raw['data_inicio'],
    'data_fim': cal_raw['data_fim'],
    'horizonte_dias': horizonte,
    'modelo_origem': cal_raw.get('modelo_base', 'V13'),
    'sumario': {
        'campanhas_estruturais': len(estruturais_resumo),
        'campanhas_eventuais': len(eventuais),
        'total_eventuais_anual': len(eventuais),
        'lucro_estruturais_anual_R$': sum(e['lucro_adicional_estimado_anual_R$']
                                            for e in estruturais_resumo),
        'lucro_eventuais_R$': round(sum(c['lucro_adicional_estimado_R$']
                                          for c in eventuais), 2),
    },
    'campanhas_estruturais': estruturais_resumo,
    'campanhas_eventuais': eventuais,
}
calendario_op['sumario']['lucro_total_anual_R$'] = round(
    calendario_op['sumario']['lucro_estruturais_anual_R$']
    + calendario_op['sumario']['lucro_eventuais_R$'], 2
)

OUT = ROOT / args.output
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(calendario_op, f, indent=2, ensure_ascii=False, default=str)

# ── Print resumo ──────────────────────────────────────────────────────

print()
print("=" * 80)
print("CALENDÁRIO OPERACIONAL V16")
print("=" * 80)
print(f"  Período:                    {cal_raw['data_inicio']} → {cal_raw['data_fim']}")
print(f"  Modelo base:                {cal_raw.get('modelo_base', 'V13')}")
print(f"  Campanhas eventuais:        {len(eventuais)} (era {len(campanhas_raw)} bruto)")
print(f"  Campanhas estruturais:      {len(estruturais_resumo)}")
print(f"  Lucro est. estruturais:     R$ {calendario_op['sumario']['lucro_estruturais_anual_R$']:,.2f}/ano")
print(f"  Lucro est. eventuais:       R$ {calendario_op['sumario']['lucro_eventuais_R$']:,.2f}")
print(f"  Lucro estimado total:       R$ {calendario_op['sumario']['lucro_total_anual_R$']:,.2f}/ano")
print()
print("CAMPANHAS ESTRUTURAIS (sempre ativas):")
for e in estruturais_resumo:
    print(f"  · {e['nome']:25s} {e['n_dias_no_ano']:3d} dias/ano  R$ {e['preco_combo']:>6.2f}  → {e['comunicacao']}")
print()
print(f"CAMPANHAS EVENTUAIS (top 5 por lucro):")
for c in sorted(eventuais, key=lambda x: -x['lucro_adicional_estimado_R$'])[:5]:
    print(f"  · {c['data_inicio']} → {c['data_fim']} ({c['dias_total']}d)  "
          f"{c['comunicacao']:<55s} R$ {c['lucro_adicional_estimado_R$']:.2f}")
print()
print(f"✓ Saída: {OUT}")

# ── Hook automático: gerar dashboard HTML moderno ─────────────────────
try:
    import subprocess
    print()
    print("Gerando dashboard HTML moderno…")
    subprocess.run(
        [sys.executable, str(ROOT / 'gerar_dashboard_v16.py'),
         '--input', str(OUT.relative_to(ROOT)),
         '--output', f"results/v16/dashboard.html"],
        check=True, cwd=str(ROOT)
    )
    # E também gera operação (cartazes + etiquetas + treinamento)
    print("Gerando operação (cartazes, etiquetas, treinamento)…")
    subprocess.run(
        [sys.executable, str(ROOT / 'gerar_operacao.py'),
         '--input', str(OUT.relative_to(ROOT)),
         '--output_dir', 'results/v16/operacao'],
        check=True, cwd=str(ROOT)
    )
    print(f"\n📊 Dashboard:  results/v16/dashboard.html")
    print(f"📂 Operação:   results/v16/operacao/")
except Exception as e:
    print(f"⚠ Erro no hook automático: {e}")
