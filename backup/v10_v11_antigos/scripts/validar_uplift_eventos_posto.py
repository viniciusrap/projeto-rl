"""Cruza vendas REAIS do posto (venda_por_dia.xlsx) com calendário comercial
brasileiro para validar se datas comerciais realmente movem vendas no posto.

Para cada (evento × categoria × ano):
- Janela: ±7 dias em torno da data do evento
- Baseline: resto do ano excluindo janela e buffer de 14 dias
- Uplift real = média_venda_diária(janela) / média_venda_diária(baseline)

Compara com:
- uplift_prior do calendário (heurística inicial)
- uplift_olist (medido em e-commerce BR)

Saídas:
  results/v11/uplift_real_posto_por_evento.csv  (1 linha por categoria × evento × ano)
  results/v11/uplift_real_posto_agregado.csv    (média entre anos)
  results/v11/comparacao_uplift_3fontes.csv     (posto vs olist vs heuristica)
  results/v11/uplift_real_posto.png             (visualização)
"""
import io
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results' / 'v11'

# ── Carrega dados ──────────────────────────────────────────────────────────

vendas = pd.read_csv(DATA / 'venda_por_dia_parseado.csv', parse_dates=['data'])
print(f"Vendas: {len(vendas):,} registros, {vendas['categoria'].nunique()} categorias")

calend = pd.read_csv(DATA / 'calendario_comercial.csv', parse_dates=['data'])
print(f"Calendário: {len(calend)} eventos")

# Filtrar apenas datas comerciais relevantes (descartar feriados oficiais)
eventos = calend[calend['tipo_evento'].isin(
    ['data_comercial', 'evento_esportivo', 'evento_local'])].copy()
print(f"Eventos a analisar: {len(eventos)}")

# Mapeamento de categorias do posto → categoria do modelo (alinhado com calibrar_v2)
MAPA_POSTO_MODELO = {
    'SOUZA CRUZ': 'cigarro_souza_cruz', 'ISQUEIROS': 'cigarro_souza_cruz',
    'PHILIP MORRIS': 'cigarro_philip_morris',
    'JTI': 'cigarro_jti', 'CIGARRILHAS': 'cigarro_jti',
    'ENERGÉTICO': 'energetico',
    'REFRIGERANTE': 'refrigerante',
    'AGUA': 'agua', 'AGUA SABORIZADA': 'agua', 'ÁGUA DE COCO': 'agua',
    'CERVEJA AMBEV': 'cerveja', 'CERVEJA FEMSA': 'cerveja',
    'CERVEJA ESPECIAIS': 'cerveja', 'CERVEJA ITAIPAVA': 'cerveja',
    'ISOTÔNICO': 'isotonico',
    'SUCO': 'suco', 'BEBIDA LÁCTEA': 'suco',
    'SORVETE KIBON': 'sorvete', 'SORVETE JUNDIÁ': 'sorvete',
    'SORVETE PERFETTO': 'sorvete',
    'GELO': 'gelo',
    'SNACK ELMA CHIPS': 'snack', 'SNACK DIVERSOS': 'snack',
    'SNACK TORCIDA': 'snack', 'SNACK PRINGLES': 'snack',
    'AMENDOIN/NOZES': 'snack', 'PIPOCA': 'snack', 'CEREAIS': 'snack',
    'BISCOITO': 'biscoito',
    'CHOCOLATE LACTA': 'chocolate_premium', 'CHOCOLATE NESTLE': 'chocolate_premium',
    'CHOCOLATE FERRERO': 'chocolate_premium',
    'CHOCOLATE GAROTO': 'chocolate_impulso',
    'CHOCOLATE M&M (MARS)': 'chocolate_impulso',
    'CHOCOLATE ARCOR': 'chocolate_impulso',
    'CHOCOLATE DIVERSOS': 'chocolate_impulso',
    'ACHOCOLATADO': 'chocolate_impulso',
    'CHICLETE': 'doce', 'MENTOS': 'doce', 'DROPS': 'doce',
    'BALA': 'doce', 'BALA FINI': 'doce', 'PASTILHAS': 'doce',
    'DOCES DIVERSOS': 'doce',
    'VINHO': 'vinho',
    'DESTILADOS DIVERSOS': 'destilados', 'WHISK': 'destilados',
    'VODKA': 'destilados', 'AGUARDENTES': 'destilados',
    'PADARIA': 'padaria', 'SANDUÍCHE': 'padaria',
    'SALGADO ASSADO/FRITO': 'padaria', 'BOLO': 'padaria',
    'IOGURTE': 'padaria', 'CONGELADOS': 'padaria',
    'MERCEARIA ALIMENTICIA': 'padaria',
    'CAFÉ': 'cafe', 'NESCAFÉ BEBIDAS': 'cafe', 'CHÁ': 'cafe',
}

vendas['categoria_modelo'] = vendas['categoria'].map(MAPA_POSTO_MODELO)
vendas_rel = vendas[vendas['categoria_modelo'].notna()].copy()
print(f"Vendas em categorias relevantes: {len(vendas_rel):,}")

# Mapeamento categorias_afetadas (eventos) → categoria_modelo do posto
# Para cada categoria no calendário, qual nossa categoria_modelo corresponde?
MAPA_CALENDARIO_MODELO = {
    'chocolate': ['chocolate_premium', 'chocolate_impulso'],
    'vinho': ['vinho'],
    'vinho_tinto': ['vinho'],
    'espumante': ['vinho', 'destilados'],
    'champagne': ['destilados'],
    'cerveja': ['cerveja'],
    'cerveja_premium': ['cerveja'],
    'whisky': ['destilados'],
    'cachaca': ['destilados'],
    'snack': ['snack', 'biscoito'],
    'salgadinho': ['snack'],
    'refrigerante': ['refrigerante'],
    'gelo': ['gelo'],
    'sorvete': ['sorvete'],
    'energetico': ['energetico'],
    'suco': ['suco'],
    'cafe': ['cafe'],
    'todas': list(set(MAPA_POSTO_MODELO.values())),
}

# ── Loop principal: para cada (evento × ano) calcular uplift real ─────────

resultados = []
for _, ev in eventos.iterrows():
    data_ev = ev['data']
    cats_evento = ev['categorias_afetadas'].split(';')
    # Categorias do modelo associadas a esse evento
    cats_modelo_ev = set()
    for c in cats_evento:
        for cm in MAPA_CALENDARIO_MODELO.get(c, []):
            cats_modelo_ev.add(cm)
    if not cats_modelo_ev:
        continue

    pre = max(int(ev['janela_pre_dias']), 7)
    pos = int(ev['janela_pos_dias'])
    inicio_janela = data_ev - timedelta(days=pre)
    fim_janela = data_ev + timedelta(days=pos)
    inicio_baseline_a = data_ev - timedelta(days=180)  # 6 meses antes
    fim_baseline_b = data_ev + timedelta(days=180)     # 6 meses depois

    for cat_m in cats_modelo_ev:
        sub = vendas_rel[vendas_rel['categoria_modelo'] == cat_m]
        if len(sub) == 0:
            continue
        # Receita diária dessa categoria
        diario = sub.groupby('data')['valor_venda'].sum().reset_index()

        # Janela do evento
        janela = diario[(diario['data'] >= inicio_janela)
                          & (diario['data'] <= fim_janela)]
        # Baseline: 6m antes/depois, excluindo a janela e buffer de 14d
        baseline = diario[
            ((diario['data'] >= inicio_baseline_a) &
             (diario['data'] < inicio_janela - timedelta(days=14)))
            | ((diario['data'] > fim_janela + timedelta(days=14)) &
               (diario['data'] <= fim_baseline_b))
        ]
        if len(janela) < 3 or len(baseline) < 14:
            continue

        media_janela = janela['valor_venda'].mean()
        media_baseline = baseline['valor_venda'].mean()
        if media_baseline < 0.01:
            continue
        uplift = media_janela / media_baseline

        resultados.append({
            'evento': ev['nome_evento'],
            'tipo_evento': ev['tipo_evento'],
            'data': data_ev.strftime('%Y-%m-%d'),
            'ano': data_ev.year,
            'categoria': cat_m,
            'media_janela_R$': round(media_janela, 2),
            'media_baseline_R$': round(media_baseline, 2),
            'uplift_real_posto': round(uplift, 3),
            'uplift_prior_calendario': float(ev['uplift_prior']),
            'n_dias_janela': len(janela),
            'n_dias_baseline': len(baseline),
        })

df = pd.DataFrame(resultados)
df.to_csv(RESULTS / 'uplift_real_posto_por_evento.csv',
           index=False, encoding='utf-8')
print(f"\n{len(df)} medições brutas (evento × ano × categoria)")

# ── Agregação por (evento, categoria): média entre anos ────────────────────

df['evento_base'] = df['evento'].str.split(' — ').str[0]
agg = (df.groupby(['evento_base', 'categoria'])
         .agg(uplift_medio=('uplift_real_posto', 'mean'),
              uplift_std=('uplift_real_posto', 'std'),
              n_anos=('uplift_real_posto', 'count'),
              uplift_prior_medio=('uplift_prior_calendario', 'mean'))
         .reset_index())
agg['uplift_medio'] = agg['uplift_medio'].round(3)
agg['uplift_std'] = agg['uplift_std'].round(3)
agg['uplift_prior_medio'] = agg['uplift_prior_medio'].round(3)
agg = agg.sort_values('uplift_medio', ascending=False)
agg.to_csv(RESULTS / 'uplift_real_posto_agregado.csv',
            index=False, encoding='utf-8')

# ── Comparação 3 fontes: posto vs prior calendario vs olist ────────────────

# Carregar Olist
try:
    olist = pd.read_csv(ROOT / 'data' / 'priors_externos' / 'olist' / 'uplift_agregado.csv')
    olist_dict = {(r['categoria'], r['evento_base']): float(r['uplift_medio'])
                   for _, r in olist.iterrows()}
except FileNotFoundError:
    olist_dict = {}

agg['uplift_olist'] = agg.apply(
    lambda r: olist_dict.get((r['categoria'], r['evento_base']), None), axis=1
)
agg.to_csv(RESULTS / 'comparacao_uplift_3fontes.csv',
            index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print("="*80)
print("UPLIFT REAL MEDIDO NO POSTO (6 anos de venda_por_dia)")
print("="*80)
print()

# Top eventos com uplift > 1.10 (confirmados)
confirmados = agg[agg['uplift_medio'] > 1.10].head(20)
print(f"Eventos × Categoria com UPLIFT REAL CONFIRMADO (>10% acima do baseline):")
print()
print(f"{'Evento':<35s} {'Categoria':<20s} {'Real':>6s} {'Prior':>6s} {'Olist':>6s} {'N':>3s}")
print('-'*85)
for _, r in confirmados.iterrows():
    olist_s = f"{r['uplift_olist']:.2f}" if pd.notna(r['uplift_olist']) else '  —  '
    print(f"  {r['evento_base'][:33]:<33s} {r['categoria']:<20s} "
          f"{r['uplift_medio']:>5.2f}× {r['uplift_prior_medio']:>5.2f}× "
          f"{olist_s:>6s} {int(r['n_anos']):>3d}")

print()
print(f"Eventos × Categoria SEM CONFIRMAÇÃO (<5% acima do baseline):")
print()
nao_conf = agg[agg['uplift_medio'] < 1.05].head(15)
for _, r in nao_conf.iterrows():
    olist_s = f"{r['uplift_olist']:.2f}" if pd.notna(r['uplift_olist']) else '  —  '
    print(f"  {r['evento_base'][:33]:<33s} {r['categoria']:<20s} "
          f"{r['uplift_medio']:>5.2f}× prior {r['uplift_prior_medio']:>5.2f}×")

# ── Visualização ──────────────────────────────────────────────────────────

# Para cada evento principal, mostrar uplift real × prior × olist (top 8 eventos)
eventos_focar = ['Dia das Mães', 'Dia dos Namorados', 'Dia dos Pais',
                  'Dia das Crianças', 'Véspera de Natal', 'Réveillon',
                  'Black Friday', 'Dia Internacional da Mulher']

fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
for i, ev_nome in enumerate(eventos_focar):
    ax = axes[i // 4, i % 4]
    sub = agg[agg['evento_base'] == ev_nome].sort_values('uplift_medio',
                                                            ascending=True)
    if len(sub) == 0:
        ax.set_title(f'{ev_nome}\n(sem dados)')
        continue
    y = range(len(sub))
    cores = ['#2ca02c' if u > 1.10 else '#ff7f0e' if u > 1.0 else '#d62728'
              for u in sub['uplift_medio']]
    ax.barh(y, sub['uplift_medio'], color=cores, alpha=0.7,
             edgecolor='black', linewidth=0.5)
    # Linha prior
    ax.scatter(sub['uplift_prior_medio'], y, marker='|', s=200, color='black',
                 label='prior calend')
    # Linha olist
    olist_vals = sub['uplift_olist']
    mask = olist_vals.notna()
    if mask.any():
        ax.scatter(olist_vals[mask], np.array(list(y))[mask],
                    marker='x', s=80, color='blue', label='olist')
    ax.set_yticks(y)
    ax.set_yticklabels(sub['categoria'], fontsize=8)
    ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('uplift × baseline')
    ax.set_title(ev_nome, fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    if i == 0:
        ax.legend(fontsize=7, loc='lower right')

plt.suptitle('Uplift REAL no posto (6 anos) vs PRIOR calendário vs OLIST por evento',
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RESULTS / 'uplift_real_posto.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ {RESULTS / 'uplift_real_posto_por_evento.csv'}")
print(f"✓ {RESULTS / 'uplift_real_posto_agregado.csv'}")
print(f"✓ {RESULTS / 'comparacao_uplift_3fontes.csv'}")
print(f"✓ {RESULTS / 'uplift_real_posto.png'}")

# ── Summary stats ──────────────────────────────────────────────────────────

print()
print("="*80)
print("ESTATÍSTICAS — quanto evento move venda no posto?")
print("="*80)
print()
print(f"  Total pares (evento × categoria): {len(agg)}")
print(f"  Com uplift CONFIRMADO (>10%):     {(agg['uplift_medio'] > 1.10).sum()} ({(agg['uplift_medio'] > 1.10).sum() / len(agg) * 100:.1f}%)")
print(f"  Com uplift POSITIVO (>0%):        {(agg['uplift_medio'] > 1.00).sum()} ({(agg['uplift_medio'] > 1.00).sum() / len(agg) * 100:.1f}%)")
print(f"  Sem mudança (~baseline):           {((agg['uplift_medio'] >= 0.95) & (agg['uplift_medio'] <= 1.05)).sum()}")
print(f"  NEGATIVO (<5% abaixo):             {(agg['uplift_medio'] < 0.95).sum()}")
print()
print(f"  Uplift médio em datas comerciais:  {agg['uplift_medio'].mean():.3f}×")
print(f"  Uplift máximo:                     {agg['uplift_medio'].max():.3f}× ({agg.loc[agg['uplift_medio'].idxmax(), 'evento_base']} × {agg.loc[agg['uplift_medio'].idxmax(), 'categoria']})")
