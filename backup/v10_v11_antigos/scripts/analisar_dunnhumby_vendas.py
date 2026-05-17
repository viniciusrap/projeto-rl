"""Aprofunda análise do Dunnhumby (loja física) — extrai uplift de VENDAS
por categoria × week_no, não só de promoção.

Substitui Olist (e-commerce) como prior de uplift sazonal.

Como Dunnhumby é anonimizado em weeks (1-102), mapeio week_no aproximado
para mês usando dataset USA padrão (sabemos que Thanksgiving cai semana ~47,
Christmas semana ~51-52, Easter primavera, Mother's Day semana 19, etc.).

Saída: data/priors_externos/dunnhumby/vendas_uplift_sazonal.csv
       results/v11/dunnhumby_vendas_sazonais.png
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RAW = ROOT / 'data' / 'raw_dunnhumby'
OUT = ROOT / 'data' / 'priors_externos' / 'dunnhumby'
RES = ROOT / 'results' / 'v11'

# ── Reusar product mapping do processar_dunnhumby ─────────────────────────

print("Carregando product.csv...")
prod = pd.read_csv(RAW / 'product.csv', usecols=['PRODUCT_ID', 'COMMODITY_DESC'])

MAPEAMENTO_CATEGORIAS = {
    'BEERS/ALES': 'cerveja',
    'DOMESTIC WINE': 'vinho', 'IMPORTED WINE': 'vinho', 'MISC WINE': 'vinho',
    'LIQUOR': 'destilados',
    'SOFT DRINKS': 'refrigerante',
    'WATER': 'agua', 'WATER - CARBONATED/FLVRD DRINK': 'agua',
    'COFFEE': 'cafe',
    'TEAS': 'cha',
    'CANDY - PACKAGED': 'chocolate_doce', 'CANDY - CHECKLANE': 'chocolate_doce',
    'COOKIES/CONES': 'biscoito', 'COOKIES': 'biscoito',
    'CRACKERS/MISC BKD FD': 'biscoito',
    'BAG SNACKS': 'snack', 'CHIPS&SNACKS': 'snack', 'SNACKS': 'snack',
    'SNACK NUTS': 'snack', 'NUTS': 'snack', 'POPCORN': 'snack',
    'ICE CREAM/MILK/SHERBTS': 'sorvete',
    'FRZN ICE': 'gelo',
    'CIGARETTES': 'cigarro', 'CIGARS': 'cigarro', 'TOBACCO OTHER': 'cigarro',
    'BAKED BREAD/BUNS/ROLLS': 'padaria', 'BAKED SWEET GOODS': 'padaria',
    'SANDWICHES': 'sanduiche', 'BREAKFAST SAUSAGE/SANDWICHES': 'sanduiche',
    'CANNED JUICES': 'suco', 'REFRGRATD JUICES/DRNKS': 'suco', 'JUICE': 'suco',
}
prod['categoria_modelo'] = prod['COMMODITY_DESC'].map(MAPEAMENTO_CATEGORIAS)
prod_rel = prod[prod['categoria_modelo'].notna()]
print(f"  Produtos relevantes: {len(prod_rel):,}")
cat_map = dict(zip(prod_rel['PRODUCT_ID'], prod_rel['categoria_modelo']))
ids = set(cat_map.keys())

# ── Carregar transactions em chunks ───────────────────────────────────────

print("\nLendo transactions e agregando por (week, categoria)...")
chunks = []
for i, chunk in enumerate(pd.read_csv(
    RAW / 'transaction_data.csv',
    usecols=['PRODUCT_ID', 'WEEK_NO', 'SALES_VALUE', 'QUANTITY'],
    chunksize=1_000_000)):
    if i % 5 == 0:
        print(f"  chunk {i+1}...")
    chunk = chunk[chunk['PRODUCT_ID'].isin(ids)]
    if len(chunk) == 0:
        continue
    chunk['categoria'] = chunk['PRODUCT_ID'].map(cat_map)
    agg = chunk.groupby(['WEEK_NO', 'categoria']).agg(
        receita=('SALES_VALUE', 'sum'),
        qtd=('QUANTITY', 'sum'),
        n_trans=('PRODUCT_ID', 'count'),
    ).reset_index()
    chunks.append(agg)

# Consolidar
vendas_sem = (pd.concat(chunks, ignore_index=True)
                .groupby(['WEEK_NO', 'categoria'])
                .agg(receita=('receita', 'sum'),
                     qtd=('qtd', 'sum'),
                     n_trans=('n_trans', 'sum'))
                .reset_index())
print(f"  {len(vendas_sem)} pontos (semana × categoria)")

# ── Mapear week_no → mês aproximado ───────────────────────────────────────

# Dunnhumby cobre 102 semanas (2 anos). Convenção comum: começa em junho/2014
# Vou usar mês cíclico (1-12) baseado em (week_no - 1) % 52 → semana_ano
vendas_sem['semana_ano'] = ((vendas_sem['WEEK_NO'] - 1) % 52) + 1
vendas_sem['mes_aprox'] = ((vendas_sem['semana_ano'] - 1) // 4.34 + 1).astype(int).clip(1, 12)

# ── Para cada categoria, calcular uplift por semana_ano ──────────────────

uplifts = []
for cat in vendas_sem['categoria'].unique():
    sub = vendas_sem[vendas_sem['categoria'] == cat]
    # Agregação por semana_ano (média entre os 2 anos)
    por_semana = sub.groupby('semana_ano')['receita'].mean().reset_index()
    media_anual = por_semana['receita'].mean()
    if media_anual < 0.01:
        continue
    por_semana['uplift'] = por_semana['receita'] / media_anual
    por_semana['categoria'] = cat
    uplifts.append(por_semana)

df_uplift = pd.concat(uplifts, ignore_index=True)
df_uplift = df_uplift[['categoria', 'semana_ano', 'receita', 'uplift']]
df_uplift['uplift'] = df_uplift['uplift'].round(3)
df_uplift.to_csv(OUT / 'vendas_uplift_sazonal.csv', index=False, encoding='utf-8')

# ── Agregação mensal ──────────────────────────────────────────────────────

uplift_mensal = []
for cat in vendas_sem['categoria'].unique():
    sub = vendas_sem[vendas_sem['categoria'] == cat]
    por_mes = sub.groupby('mes_aprox')['receita'].mean().reset_index()
    media = por_mes['receita'].mean()
    if media < 0.01:
        continue
    por_mes['uplift_mensal'] = (por_mes['receita'] / media).round(3)
    por_mes['categoria'] = cat
    uplift_mensal.append(por_mes)

df_mensal = pd.concat(uplift_mensal, ignore_index=True)
df_mensal = df_mensal[['categoria', 'mes_aprox', 'uplift_mensal']]
df_mensal.to_csv(OUT / 'vendas_uplift_mensal.csv', index=False, encoding='utf-8')

# ── Pico semanal por categoria ────────────────────────────────────────────

print()
print("=" * 70)
print("PICO DE VENDAS POR CATEGORIA (Dunnhumby, loja física USA)")
print("=" * 70)
print()
print(f"{'Categoria':<15s} {'Sem pico':>9s} {'Uplift':>7s} {'Mes equiv':>10s}")
print('-' * 60)
nomes_mes = ['', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago',
              'Set', 'Out', 'Nov', 'Dez']
for cat in sorted(df_uplift['categoria'].unique()):
    sub = df_uplift[df_uplift['categoria'] == cat]
    if len(sub) == 0:
        continue
    pico = sub.loc[sub['uplift'].idxmax()]
    mes = int(((pico['semana_ano'] - 1) // 4.34) + 1)
    print(f"  {cat:<13s}   {int(pico['semana_ano']):>6d}   {pico['uplift']:>5.2f}× "
          f"{nomes_mes[mes]:>10s}")

# ── Visualização ──────────────────────────────────────────────────────────

cats_foco = ['cerveja', 'vinho', 'chocolate_doce', 'snack', 'sorvete',
              'refrigerante', 'cigarro', 'gelo', 'cafe', 'biscoito',
              'destilados', 'sanduiche']

fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=False)
for i, cat in enumerate(cats_foco):
    ax = axes[i // 4, i % 4]
    sub = df_uplift[df_uplift['categoria'] == cat].sort_values('semana_ano')
    if len(sub) == 0:
        ax.set_title(f'{cat}\n(sem dados)')
        continue
    cores = ['#16a34a' if u > 1.20 else '#ca8a04' if u > 1.05 else '#9ca3af'
              for u in sub['uplift']]
    ax.bar(sub['semana_ano'], sub['uplift'], color=cores, alpha=0.7,
            edgecolor='black', linewidth=0.3)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'{cat}', fontsize=10)
    ax.set_xlabel('Semana do ano', fontsize=8)
    ax.set_ylabel('Uplift × média', fontsize=8)
    # Marcar Black Friday (~47), Natal (~52), Easter (~13), Mother (~19)
    for sem, nome, cor in [(47, 'BF', '#dc2626'), (52, 'Natal', '#1e40af'),
                              (13, 'Easter', '#7c3aed'),
                              (19, 'Mãe USA', '#ec4899')]:
        if 1 <= sem <= 52:
            ax.axvline(sem, color=cor, alpha=0.4, linewidth=1)
            ax.text(sem, ax.get_ylim()[1] * 0.95, nome, fontsize=7,
                     color=cor, rotation=90, va='top')

plt.suptitle('Dunnhumby — uplift de VENDAS por categoria × semana (loja física USA)',
              fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(RES / 'dunnhumby_vendas_sazonais.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print("✓ Saídas:")
print(f"  {OUT / 'vendas_uplift_sazonal.csv'}")
print(f"  {OUT / 'vendas_uplift_mensal.csv'}")
print(f"  {RES / 'dunnhumby_vendas_sazonais.png'}")
