"""Processa Walmart Sales Forecasting (clássico Kaggle):
- 45 lojas físicas USA
- Vendas semanais agregadas (sem categoria) com Holiday_Flag
- 2010-02 a 2012-10 (~2.7 anos)

Walmart EUA marca como feriado APENAS 4 semanas/ano:
- Super Bowl: semana de 12/02 (1ª sex fev)
- Labor Day: semana de 09/09
- Thanksgiving: semana de 24/11
- Christmas: semana de 25/12

Combinamos a flag com a data para identificar qual feriado é.

Saída:
  data/priors_externos/walmart/uplift_feriados.csv
  results/v11/walmart_uplift.png
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
OUT = ROOT / 'data' / 'priors_externos' / 'walmart'
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / 'results' / 'v11'

# ── 1. Walmart Sales (real, com Holiday_Flag) ─────────────────────────────

df = pd.read_csv(ROOT / 'data' / 'raw_walmart_m5' / 'Walmart_Sales.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
print(f"Walmart Sales: {len(df):,} linhas")
print(f"  45 lojas × {df['Date'].nunique()} semanas")
print(f"  Período: {df['Date'].min().date()} a {df['Date'].max().date()}")
print(f"  Holiday_Flag=1: {(df['Holiday_Flag'] == 1).sum()} semanas-loja")

# Identificar qual feriado cada semana com flag=1 representa
# Walmart marca 4 feriados/ano: Super Bowl (~1ª sex fev), Labor Day (~1ª sex set),
# Thanksgiving (~4ª sex nov), Christmas (~última sex ano)
def identificar_feriado(data):
    m = data.month
    d = data.day
    if m == 2:  # fevereiro
        return 'Super Bowl'
    elif m == 9:
        return 'Labor Day'
    elif m == 11:
        return 'Thanksgiving'
    elif m == 12:
        return 'Christmas'
    return 'Outro'

df_hol = df[df['Holiday_Flag'] == 1].copy()
df_hol['feriado'] = df_hol['Date'].apply(identificar_feriado)
print(f"\nDistribuição dos feriados:")
print(df_hol['feriado'].value_counts())

# ── 2. Calcular uplift por feriado ─────────────────────────────────────────

# Para cada loja × feriado, comparar venda da semana do feriado com média do ano
resultados = []
for store in df['Store'].unique():
    sub = df[df['Store'] == store].copy()
    sub['ano'] = sub['Date'].dt.year
    for ano in sub['ano'].unique():
        sub_ano = sub[sub['ano'] == ano]
        if len(sub_ano) < 10:
            continue
        media_ano = sub_ano['Weekly_Sales'].mean()
        for _, row in sub_ano[sub_ano['Holiday_Flag'] == 1].iterrows():
            feriado = identificar_feriado(row['Date'])
            uplift = row['Weekly_Sales'] / media_ano if media_ano > 0 else 0
            resultados.append({
                'store': store,
                'ano': ano,
                'data': row['Date'].date().isoformat(),
                'feriado': feriado,
                'vendas': float(row['Weekly_Sales']),
                'media_ano': float(media_ano),
                'uplift': uplift,
            })

df_up = pd.DataFrame(resultados)
df_up.to_csv(OUT / 'uplift_loja_feriado.csv', index=False, encoding='utf-8')

agg = (df_up.groupby('feriado')
              .agg(uplift_medio=('uplift', 'mean'),
                   uplift_std=('uplift', 'std'),
                   n_obs=('uplift', 'count'))
              .reset_index())
agg['uplift_medio'] = agg['uplift_medio'].round(3)
agg['uplift_std'] = agg['uplift_std'].round(3)
agg = agg.sort_values('uplift_medio', ascending=False)
agg.to_csv(OUT / 'uplift_feriados.csv', index=False, encoding='utf-8')

print()
print("=" * 60)
print("UPLIFT POR FERIADO USA (Walmart, 45 lojas, 2010-2012)")
print("=" * 60)
print()
print(f"{'Feriado':<20s} {'Uplift':>8s} {'±':>5s} {'N':>5s}")
print('-' * 50)
for _, r in agg.iterrows():
    print(f"  {r['feriado']:<18s} {r['uplift_medio']:>6.2f}× ±{r['uplift_std']:>4.2f} "
          f"{int(r['n_obs']):>5d}")

# ── 3. Walmart sintético (com categorias + promotion) ─────────────────────

print()
print("=" * 60)
print("WALMART SINTÉTICO (com categorias e promoção)")
print("=" * 60)
print()
print("⚠ Dataset sintético/gerado, não é venda real. Tratar como prior fraco.")
print()

try:
    df_sint = pd.read_csv(ROOT / 'data' / 'raw_walmart_m5' / 'Walmart.csv',
                           parse_dates=['transaction_date'], dayfirst=False)
    df_sint['mes'] = df_sint['transaction_date'].dt.month

    # Promoção por categoria
    print("Promoção aplicada por categoria:")
    prom_cat = df_sint.groupby(['category', 'promotion_applied']).agg(
        n=('transaction_id', 'count'),
        vendas_medias=('actual_demand', 'mean'),
    ).reset_index()
    prom_cat.to_csv(OUT / 'sintetico_promo_por_categoria.csv',
                    index=False, encoding='utf-8')

    # Quanto promoção aumenta vendas?
    for cat in df_sint['category'].unique():
        sub_cat = df_sint[df_sint['category'] == cat]
        com_promo = sub_cat[sub_cat['promotion_applied'] == True]['actual_demand'].mean()
        sem_promo = sub_cat[sub_cat['promotion_applied'] == False]['actual_demand'].mean()
        if sem_promo > 0:
            uplift = com_promo / sem_promo
            print(f"  {cat[:20]:<20s}  com_promo={com_promo:>5.1f}  sem_promo={sem_promo:>5.1f}  "
                  f"uplift={uplift:.2f}×")
except Exception as e:
    print(f"Erro no sintético: {e}")

# ── Visualização ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
agg_sorted = agg.sort_values('uplift_medio')
cores = ['#16a34a' if u > 1.05 else '#ca8a04' if u > 0.98 else '#dc2626'
          for u in agg_sorted['uplift_medio']]
ax.barh(agg_sorted['feriado'], agg_sorted['uplift_medio'], color=cores)
ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Uplift × média anual')
ax.set_title('Walmart Sales — uplift por feriado USA\n(45 lojas físicas, 2010-2012, ~3 anos)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RES / 'walmart_uplift.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ Saídas:")
print(f"  {OUT / 'uplift_loja_feriado.csv'}")
print(f"  {OUT / 'uplift_feriados.csv'}")
print(f"  {OUT / 'sintetico_promo_por_categoria.csv'}")
print(f"  {RES / 'walmart_uplift.png'}")
