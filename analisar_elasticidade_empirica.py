"""Extrai curva DESCONTO × VOLUME empírica do Dunnhumby (loja física USA).

Para cada categoria-foco, calcula:
- Preço modal sem desconto (referência)
- Para cada faixa de desconto (0%, 1-5%, 5-10%, 10-15%, 15-25%, >25%):
    * Volume médio vendido por transação
    * Volume relativo ao baseline (sem desconto)
- Plota curva: x=desconto, y=uplift de volume

Isso é a CURVA DE ELASTICIDADE PROMOCIONAL EMPÍRICA — valida ou
refuta nosso parâmetro ELASTICIDADE_PROMOCAO (-2.5 a -3.8 da literatura).

Saída:
  data/priors_externos/dunnhumby/elasticidade_empirica.csv
  results/v11/elasticidade_empirica.png
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

# Reaproveitar mapeamento de categorias
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
    'CIGARETTES': 'cigarro',
    'CANNED JUICES': 'suco', 'REFRGRATD JUICES/DRNKS': 'suco', 'JUICE': 'suco',
}

# ── 1. Carregar produtos relevantes ───────────────────────────────────────

prod = pd.read_csv(RAW / 'product.csv',
                    usecols=['PRODUCT_ID', 'COMMODITY_DESC'])
prod['categoria'] = prod['COMMODITY_DESC'].map(MAPEAMENTO_CATEGORIAS)
prod_rel = prod[prod['categoria'].notna()]
cat_map = dict(zip(prod_rel['PRODUCT_ID'], prod_rel['categoria']))
print(f"Produtos relevantes: {len(prod_rel):,}")

# ── 2. Ler transactions e calcular preço pago + desconto ───────────────────

print("\nLendo transactions e agregando...")
chunks_proc = []
for i, chunk in enumerate(pd.read_csv(
    RAW / 'transaction_data.csv',
    usecols=['PRODUCT_ID', 'QUANTITY', 'SALES_VALUE', 'RETAIL_DISC',
             'COUPON_DISC', 'COUPON_MATCH_DISC'],
    chunksize=1_000_000)):
    if i % 5 == 0:
        print(f"  chunk {i+1}...")
    chunk = chunk[chunk['PRODUCT_ID'].isin(cat_map)]
    if len(chunk) == 0:
        continue
    chunk['categoria'] = chunk['PRODUCT_ID'].map(cat_map)
    # Preço pago por unidade
    chunk = chunk[chunk['QUANTITY'] > 0]
    chunk['preco_pago_unit'] = chunk['SALES_VALUE'] / chunk['QUANTITY']
    # Desconto total aplicado (R$): RETAIL_DISC + COUPON
    chunk['desconto_total'] = -(chunk['RETAIL_DISC']
                                 + chunk['COUPON_DISC'].fillna(0)
                                 + chunk['COUPON_MATCH_DISC'].fillna(0))
    # Preço de referência (sem desconto)
    chunk['preco_base'] = chunk['preco_pago_unit'] + (chunk['desconto_total']
                                                        / chunk['QUANTITY'])
    chunk = chunk[(chunk['preco_base'] > 0.01) & (chunk['desconto_total'] >= 0)]
    chunk['pct_desconto'] = (chunk['desconto_total']
                               / (chunk['preco_base'] * chunk['QUANTITY']) * 100)
    chunk['pct_desconto'] = chunk['pct_desconto'].clip(0, 100)
    chunks_proc.append(chunk[['categoria', 'PRODUCT_ID', 'QUANTITY',
                                'preco_pago_unit', 'preco_base',
                                'pct_desconto']])

trans = pd.concat(chunks_proc, ignore_index=True)
print(f"\nTotal transações relevantes: {len(trans):,}")

# ── 3. Para cada categoria: agrupar por faixa de desconto ─────────────────

# Faixas de desconto
FAIXAS = [
    (0, 1, '0%'),
    (1, 5, '1-5%'),
    (5, 10, '5-10%'),
    (10, 15, '10-15%'),
    (15, 25, '15-25%'),
    (25, 100, '>25%'),
]

def faixa_de(pct):
    for lo, hi, nome in FAIXAS:
        if lo <= pct < hi:
            return nome
    return '>25%'

trans['faixa_desc'] = trans['pct_desconto'].apply(faixa_de)

# Para cada (categoria, faixa): qtd média por transação e n
agg = (trans.groupby(['categoria', 'faixa_desc'])
              .agg(qtd_media_por_trans=('QUANTITY', 'mean'),
                   preco_pago_medio=('preco_pago_unit', 'mean'),
                   n_trans=('QUANTITY', 'count'))
              .reset_index())

# Ordenar faixas
faixa_order = {f[2]: i for i, f in enumerate(FAIXAS)}
agg['faixa_order'] = agg['faixa_desc'].map(faixa_order)
agg = agg.sort_values(['categoria', 'faixa_order'])

# Calcular uplift relativo ao baseline (faixa 0%)
def calc_uplift(grupo):
    base = grupo[grupo['faixa_desc'] == '0%']['qtd_media_por_trans']
    if len(base) == 0:
        return grupo
    base_val = base.iloc[0]
    if base_val < 0.01:
        return grupo
    grupo = grupo.copy()
    grupo['uplift_qtd'] = (grupo['qtd_media_por_trans'] / base_val).round(3)
    return grupo

agg = agg.groupby('categoria', group_keys=False).apply(calc_uplift,
                                                          include_groups=False)
# Re-merge categoria
agg_final = []
for cat in trans['categoria'].unique():
    sub = agg[agg['categoria'] == cat] if 'categoria' in agg.columns else None
    if sub is None or len(sub) == 0:
        continue
    agg_final.append(sub)

# Fallback simpler: refazer com merge manual
agg2 = (trans.groupby(['categoria', 'faixa_desc'])
               .agg(qtd_media_por_trans=('QUANTITY', 'mean'),
                    preco_pago_medio=('preco_pago_unit', 'mean'),
                    n_trans=('QUANTITY', 'count'))
               .reset_index())
agg2['faixa_order'] = agg2['faixa_desc'].map(faixa_order)
agg2 = agg2.sort_values(['categoria', 'faixa_order'])

# Calcular uplift por categoria
uplifts = []
for cat in agg2['categoria'].unique():
    sub = agg2[agg2['categoria'] == cat]
    base = sub[sub['faixa_desc'] == '0%']['qtd_media_por_trans']
    base_val = base.iloc[0] if len(base) > 0 else None
    for _, r in sub.iterrows():
        up = (r['qtd_media_por_trans'] / base_val if base_val and base_val > 0.01
                else None)
        uplifts.append({
            'categoria': cat,
            'faixa_desc': r['faixa_desc'],
            'pct_min': [f[0] for f in FAIXAS if f[2] == r['faixa_desc']][0],
            'pct_max': [f[1] for f in FAIXAS if f[2] == r['faixa_desc']][0],
            'qtd_media': round(r['qtd_media_por_trans'], 3),
            'uplift_qtd': round(up, 3) if up else None,
            'n_trans': int(r['n_trans']),
        })

df_up = pd.DataFrame(uplifts)
df_up.to_csv(OUT / 'elasticidade_empirica.csv', index=False, encoding='utf-8')

# ── 4. Calcular ELASTICIDADE EMPÍRICA por categoria ──────────────────────

# Elasticidade = % mudança em qtd / % mudança em preço
# Para cada categoria, ajustar reta em log-log:
#   log(uplift_qtd) = elasticidade × log(1 - desconto%/100)
print()
print("=" * 75)
print("ELASTICIDADE PROMOCIONAL EMPÍRICA (Dunnhumby — loja física USA)")
print("=" * 75)
print()
print(f"{'Categoria':<15s} {'Bijmolt':>9s} {'Empírica':>9s} {'Δ':>6s} {'Status':<25s}")
print('-' * 75)

# Valores Bijmolt usados na nossa calibração
BIJMOLT = {
    'cerveja': -3.0,
    'vinho': -3.4,
    'destilados': -3.4,
    'refrigerante': -3.2,
    'agua': -2.5,
    'suco': -3.1,
    'cafe': -2.8,
    'snack': -2.9,
    'biscoito': -3.2,
    'chocolate_doce': -2.9,
    'sorvete': -3.6,
    'gelo': -1.5,
    'cigarro': -3.0,
}

elasticidades_empiricas = []
for cat in sorted(df_up['categoria'].unique()):
    sub = df_up[df_up['categoria'] == cat].copy()
    sub = sub[sub['uplift_qtd'].notna() & (sub['uplift_qtd'] > 0)]
    if len(sub) < 3:
        continue
    # Centro de cada faixa
    sub['pct_centro'] = (sub['pct_min'] + sub['pct_max']) / 2
    # Filtrar só faixas com desconto > 0 (uplift = 1 no 0% por definição)
    sub_fit = sub[sub['pct_centro'] > 0]
    if len(sub_fit) < 2:
        continue
    # log-log fit
    x = np.log(1 - sub_fit['pct_centro'] / 100)
    y = np.log(sub_fit['uplift_qtd'].clip(0.1, 10))
    try:
        slope, _ = np.polyfit(x, y, 1)
        elast_emp = float(slope)
    except Exception:
        continue
    elasticidades_empiricas.append({
        'categoria': cat,
        'elasticidade_empirica': round(elast_emp, 3),
        'elasticidade_bijmolt': BIJMOLT.get(cat, -2.5),
    })
    bijmolt_val = BIJMOLT.get(cat, -2.5)
    diff = elast_emp - bijmolt_val
    if abs(diff) < 0.5:
        status = '~ alinhada'
    elif abs(elast_emp) < abs(bijmolt_val):
        status = 'Bijmolt SUPERESTIMA'
    else:
        status = 'Bijmolt SUBESTIMA'
    print(f"  {cat:<13s} {bijmolt_val:>7.2f}  {elast_emp:>7.2f}  {diff:>+5.2f}  {status:<25s}")

df_elast = pd.DataFrame(elasticidades_empiricas)
df_elast.to_csv(OUT / 'elasticidade_resumo.csv', index=False, encoding='utf-8')

# ── 5. Visualização ──────────────────────────────────────────────────────

cats_plot = sorted(df_up['categoria'].unique())[:12]
fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharey=False)
for i, cat in enumerate(cats_plot):
    ax = axes[i // 4, i % 4]
    sub = df_up[df_up['categoria'] == cat].copy()
    sub = sub[sub['uplift_qtd'].notna()]
    if len(sub) == 0:
        ax.set_title(f'{cat}\n(sem dados)')
        continue
    sub['pct_centro'] = (sub['pct_min'] + sub['pct_max']) / 2
    sub = sub.sort_values('pct_centro')

    # Curva empírica
    ax.plot(sub['pct_centro'], sub['uplift_qtd'], marker='o',
             linewidth=2, color='steelblue', label='medido (Dunnhumby)')

    # Curva teórica Bijmolt
    bijmolt_e = BIJMOLT.get(cat, -2.5)
    x_teor = np.linspace(0, 30, 30)
    # uplift = (1 - desc%)^elast (modelo log-log)
    y_teor = (1 - x_teor / 100) ** bijmolt_e
    ax.plot(x_teor, y_teor, '--', color='red', alpha=0.7,
             label=f'Bijmolt teórico (e={bijmolt_e})')

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('% Desconto')
    ax.set_ylabel('Uplift volume')
    ax.set_title(cat, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('Curva DESCONTO × UPLIFT DE VOLUME (Dunnhumby loja física USA)\n'
              'comparada com elasticidade da literatura (Bijmolt 2005)',
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RES / 'elasticidade_empirica.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ Saídas:")
print(f"  {OUT / 'elasticidade_empirica.csv'} ({len(df_up)} linhas)")
print(f"  {OUT / 'elasticidade_resumo.csv'} ({len(df_elast)} categorias)")
print(f"  {RES / 'elasticidade_empirica.png'}")
