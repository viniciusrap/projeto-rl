"""Processa Dunnhumby Complete Journey para extrair padrões temporais
de PROMOÇÃO por categoria.

Pré-requisito: arquivos em data/raw_dunnhumby/ (8 CSVs, ~800MB total)

O que extrai:
1. Sazonalidade de promoção por categoria (% de semanas no ano com display/mailer)
2. Padrão de desconto efetivo (RETAIL_DISC) por categoria × semana
3. "Antecedência" típica: quando empresas começam a promover antes do pico de venda

Saída em data/priors_externos/dunnhumby/:
  - promo_semanal_por_categoria.csv      (% display/mailer por week 1-52 × cat)
  - promo_mensal_por_categoria.csv       (agregado mensal)
  - desconto_efetivo_por_categoria.csv   (magnitude do desconto aplicado)
  - mapeamento_categorias.csv            (Dunnhumby → nosso modelo)

NOTA: Dunnhumby é dado americano (~2014). Padrões temporais são cíclicos
e válidos como prior, mas datas absolutas (Thanksgiving, Memorial Day)
não batem com Brasil. Usar com cautela.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RAW = ROOT / 'data' / 'raw_dunnhumby'
OUT = ROOT / 'data' / 'priors_externos' / 'dunnhumby'
OUT.mkdir(parents=True, exist_ok=True)

# ── Verificar arquivos ──────────────────────────────────────────────────────

arquivos_necessarios = ['product.csv', 'causal_data.csv', 'transaction_data.csv']
faltando = [f for f in arquivos_necessarios if not (RAW / f).exists()]
if faltando:
    print("✗ Faltando arquivos do Dunnhumby:")
    for f in faltando:
        print(f"  - {f}")
    sys.exit(1)

# ── Mapeamento Dunnhumby → nosso modelo ────────────────────────────────────

MAPEAMENTO_CATEGORIAS = {
    # Cerveja
    'BEERS/ALES': 'cerveja',
    # Vinho
    'DOMESTIC WINE': 'vinho',
    'IMPORTED WINE': 'vinho',
    'MISC WINE': 'vinho',
    # Destilados
    'LIQUOR': 'destilados',
    # Refrigerante
    'SOFT DRINKS': 'refrigerante',
    # Água
    'WATER': 'agua',
    'WATER - CARBONATED/FLVRD DRINK': 'agua',
    # Café
    'COFFEE': 'cafe',
    'COFFEE SHOP': 'cafe',
    'COFFEE SHOP SWEET GOODS&RETAIL': 'cafe',
    'DRY TEA/COFFEE/COCO MIX': 'cafe',
    # Chá
    'TEAS': 'cha',
    # Chocolate / doce
    'CANDY - PACKAGED': 'chocolate_doce',
    'CANDY - CHECKLANE': 'chocolate_doce',
    # Biscoito
    'COOKIES/CONES': 'biscoito',
    'COOKIES': 'biscoito',
    'CRACKERS/MISC BKD FD': 'biscoito',
    # Snacks
    'BAG SNACKS': 'snack',
    'CHIPS&SNACKS': 'snack',
    'SNACKS': 'snack',
    'SWEET GOODS & SNACKS': 'snack',
    'WAREHOUSE SNACKS': 'snack',
    'PACKAGED NATURAL SNACKS': 'snack',
    'SNACK NUTS': 'snack',
    'NUTS': 'snack',
    'POPCORN': 'snack',
    'CONVENIENT BRKFST/WHLSM SNACKS': 'snack',
    # Sorvete e gelo
    'ICE CREAM/MILK/SHERBTS': 'sorvete',
    'FRZN NOVELTIES/WTR ICE': 'sorvete',
    'FRZN ICE': 'gelo',
    # Cigarro
    'CIGARETTES': 'cigarro',
    'CIGARS': 'cigarro',
    'TOBACCO OTHER': 'cigarro',
    # Padaria / sanduíche
    'BAKED BREAD/BUNS/ROLLS': 'padaria',
    'BAKED SWEET GOODS': 'padaria',
    'BREAD': 'padaria',
    'SANDWICHES': 'sanduiche',
    'BREAKFAST SAUSAGE/SANDWICHES': 'sanduiche',
    # Sucos
    'CANNED JUICES': 'suco',
    'REFRGRATD JUICES/DRNKS': 'suco',
    'JUICE': 'suco',
}

# ── 1. Carrega produto e identifica SKUs relevantes ────────────────────────

print("Carregando product.csv...")
prod = pd.read_csv(RAW / 'product.csv',
                    usecols=['PRODUCT_ID', 'COMMODITY_DESC'])
prod['categoria_modelo'] = prod['COMMODITY_DESC'].map(MAPEAMENTO_CATEGORIAS)
prod_relevante = prod[prod['categoria_modelo'].notna()].copy()
print(f"  Produtos relevantes: {len(prod_relevante):,} de {len(prod):,}")
print(f"  Categorias mapeadas: {prod_relevante['categoria_modelo'].nunique()}")

# Salvar mapeamento usado
(pd.DataFrame(MAPEAMENTO_CATEGORIAS.items(),
              columns=['dunnhumby_commodity', 'nossa_categoria'])
   .to_csv(OUT / 'mapeamento_categorias.csv', index=False, encoding='utf-8'))

cat_map = dict(zip(prod_relevante['PRODUCT_ID'],
                    prod_relevante['categoria_modelo']))
ids_relevantes = set(cat_map.keys())

# ── 2. Causal data → sazonalidade de promoção (display + mailer) ──────────

print("\nProcessando causal_data em chunks...")
chunks_proc = []
chunksize = 1_000_000
total_chunks = 0
for chunk in pd.read_csv(RAW / 'causal_data.csv',
                          usecols=['PRODUCT_ID', 'WEEK_NO', 'display', 'mailer'],
                          chunksize=chunksize):
    total_chunks += 1
    if total_chunks % 5 == 0:
        print(f"  chunk {total_chunks}...")
    # Filtra produtos relevantes
    chunk = chunk[chunk['PRODUCT_ID'].isin(ids_relevantes)]
    if len(chunk) == 0:
        continue
    chunk['categoria'] = chunk['PRODUCT_ID'].map(cat_map)
    # display: string '0' a '9' ou 'A'. Tudo != '0' e não-NaN = teve display
    chunk['teve_display'] = ((chunk['display'].fillna('0').astype(str).str.strip())
                              .isin(['0', '', 'nan']) == False).astype(int)
    # mailer: string 'A', 'H', 'D', 'F' ou '0'. Tudo != '0' e não-NaN = teve mailer
    chunk['teve_mailer'] = ((chunk['mailer'].fillna('0').astype(str).str.strip())
                             .isin(['0', '', 'nan']) == False).astype(int)
    chunks_proc.append(chunk[['categoria', 'WEEK_NO',
                                'teve_display', 'teve_mailer']])

causal = pd.concat(chunks_proc, ignore_index=True)
print(f"  Total de registros causal relevantes: {len(causal):,}")

# Agrega: para cada (categoria, week_no), % de produto-loja-semana com display/mailer
agg_causal = (causal.groupby(['categoria', 'WEEK_NO'])
                     .agg(n_observacoes=('teve_display', 'count'),
                          pct_display=('teve_display', 'mean'),
                          pct_mailer=('teve_mailer', 'mean'))
                     .reset_index())
agg_causal['pct_display'] = (agg_causal['pct_display'] * 100).round(1)
agg_causal['pct_mailer'] = (agg_causal['pct_mailer'] * 100).round(1)
agg_causal['pct_promo_qualquer'] = (
    causal.groupby(['categoria', 'WEEK_NO'])
          .apply(lambda g: ((g['teve_display'] + g['teve_mailer']) > 0).mean() * 100,
                 include_groups=False)
          .round(1)
          .reset_index(drop=True)
)

# Converter WEEK_NO em semana_ciclo (1-52)
agg_causal['semana_ciclo'] = ((agg_causal['WEEK_NO'] - 1) % 52) + 1
# E mês aproximado (assumindo 52 sem/ano, 4.33 sem/mês)
agg_causal['mes_aprox'] = (((agg_causal['WEEK_NO'] - 1) % 52) // 4.34 + 1).astype(int).clip(1, 12)

agg_causal.to_csv(OUT / 'promo_semanal_por_categoria.csv',
                   index=False, encoding='utf-8')

# Agregação mensal
agg_mensal = (agg_causal.groupby(['categoria', 'mes_aprox'])
                          .agg(pct_display_medio=('pct_display', 'mean'),
                               pct_mailer_medio=('pct_mailer', 'mean'),
                               pct_promo_qualquer_medio=('pct_promo_qualquer', 'mean'),
                               n_semanas=('WEEK_NO', 'count'))
                          .reset_index())
for col in ['pct_display_medio', 'pct_mailer_medio', 'pct_promo_qualquer_medio']:
    agg_mensal[col] = agg_mensal[col].round(1)
agg_mensal.to_csv(OUT / 'promo_mensal_por_categoria.csv',
                   index=False, encoding='utf-8')

# ── 3. Transaction data → desconto real efetivo ─────────────────────────────

print("\nProcessando transaction_data em chunks...")
chunks_trans = []
total_chunks = 0
for chunk in pd.read_csv(RAW / 'transaction_data.csv',
                          usecols=['PRODUCT_ID', 'WEEK_NO', 'SALES_VALUE',
                                   'RETAIL_DISC'],
                          chunksize=chunksize):
    total_chunks += 1
    if total_chunks % 5 == 0:
        print(f"  chunk {total_chunks}...")
    chunk = chunk[chunk['PRODUCT_ID'].isin(ids_relevantes)]
    if len(chunk) == 0:
        continue
    chunk['categoria'] = chunk['PRODUCT_ID'].map(cat_map)
    chunk['teve_desconto'] = (chunk['RETAIL_DISC'] < -0.01).astype(int)
    chunk['preco_base'] = chunk['SALES_VALUE'] - chunk['RETAIL_DISC']  # RETAIL_DISC é negativo
    chunk['desconto_pct'] = np.where(
        chunk['preco_base'] > 0.01,
        -chunk['RETAIL_DISC'] / chunk['preco_base'] * 100,
        0
    )
    chunks_trans.append(chunk[['categoria', 'WEEK_NO',
                                 'teve_desconto', 'desconto_pct']])

trans = pd.concat(chunks_trans, ignore_index=True)
print(f"  Total transações relevantes: {len(trans):,}")

agg_trans = (trans.groupby(['categoria', 'WEEK_NO'])
                    .agg(n_transacoes=('teve_desconto', 'count'),
                         pct_com_desconto=('teve_desconto', 'mean'),
                         desconto_medio_pct=('desconto_pct',
                                              lambda x: x[x > 0].mean() if (x > 0).any() else 0))
                    .reset_index())
agg_trans['pct_com_desconto'] = (agg_trans['pct_com_desconto'] * 100).round(1)
agg_trans['desconto_medio_pct'] = agg_trans['desconto_medio_pct'].round(1)
agg_trans['semana_ciclo'] = ((agg_trans['WEEK_NO'] - 1) % 52) + 1
agg_trans['mes_aprox'] = (((agg_trans['WEEK_NO'] - 1) % 52) // 4.34 + 1).astype(int).clip(1, 12)

agg_trans.to_csv(OUT / 'desconto_efetivo_por_categoria.csv',
                  index=False, encoding='utf-8')

# Agregação mensal de desconto
desc_mensal = (agg_trans.groupby(['categoria', 'mes_aprox'])
                          .agg(pct_com_desconto_medio=('pct_com_desconto', 'mean'),
                               desconto_medio_pct_medio=('desconto_medio_pct', 'mean'))
                          .reset_index())
for col in ['pct_com_desconto_medio', 'desconto_medio_pct_medio']:
    desc_mensal[col] = desc_mensal[col].round(1)
desc_mensal.to_csv(OUT / 'desconto_mensal_por_categoria.csv',
                    index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print("="*75)
print("RESUMO: PADRÕES DE PROMOÇÃO POR CATEGORIA (Dunnhumby, EUA, 2014)")
print("="*75)
print()

print("Categorias mapeadas com volume:")
print('-' * 75)
for cat in sorted(agg_causal['categoria'].unique()):
    sub_c = agg_causal[agg_causal['categoria'] == cat]
    sub_t = agg_trans[agg_trans['categoria'] == cat]
    media_display = sub_c['pct_display'].mean()
    media_mailer = sub_c['pct_mailer'].mean()
    media_desc = sub_t['pct_com_desconto'].mean()
    desc_mag = sub_t['desconto_medio_pct'].mean()
    print(f"  {cat:<18s}  display {media_display:>5.1f}%  "
          f"mailer {media_mailer:>5.1f}%  desconto {media_desc:>5.1f}%  "
          f"mag {desc_mag:>5.1f}%")

print()
print("="*75)
print("SAZONALIDADE MENSAL — TOP MOMENTOS DE PROMOÇÃO POR CATEGORIA")
print("="*75)
print()
print("(mês aproximado, dado é americano - ajuste mental para BR pode ser necessário)")
print()
nomes_mes = ['', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
for cat in sorted(agg_mensal['categoria'].unique()):
    sub = agg_mensal[agg_mensal['categoria'] == cat].sort_values(
        'pct_promo_qualquer_medio', ascending=False)
    if len(sub) == 0:
        continue
    print(f"\n  {cat.upper()}:")
    for _, row in sub.head(3).iterrows():
        print(f"    {nomes_mes[int(row['mes_aprox'])]}: "
              f"promo qualquer {row['pct_promo_qualquer_medio']:.1f}% "
              f"(display {row['pct_display_medio']:.1f}% + "
              f"mailer {row['pct_mailer_medio']:.1f}%)")

print()
print("✓ Saídas em data/priors_externos/dunnhumby/:")
print("  - mapeamento_categorias.csv")
print("  - promo_semanal_por_categoria.csv")
print("  - promo_mensal_por_categoria.csv")
print("  - desconto_efetivo_por_categoria.csv")
print("  - desconto_mensal_por_categoria.csv")
