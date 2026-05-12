"""Extrai elasticidade promocional empírica do Iowa Liquor Sales.

Iowa tem `State Bottle Retail` (preço estadual de tabela) e
`Sale (Dollars)/Bottles Sold` (preço efetivo praticado). Diferença =
desconto efetivo.

Como destilados é a única categoria do Iowa, foca nessa categoria
com 12 anos de dados (~9M transações).

Saída: data/priors_externos/iowa_liquor/elasticidade_empirica.csv
       results/v11/elasticidade_iowa.png
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
RAW = ROOT / 'data' / 'raw_iowa' / 'Iowa_Liquor_Sales.csv'
OUT = ROOT / 'data' / 'priors_externos' / 'iowa_liquor'
RES = ROOT / 'results' / 'v11'

print(f"Lendo Iowa Liquor ({RAW.name}, ~3.4 GB)")
print("Calculando desconto efetivo = (State Retail × Bottles - Sale Dollars) / (State Retail × Bottles)")
print()

chunks_proc = []
for i, chunk in enumerate(pd.read_csv(
    RAW,
    usecols=['Date', 'Category Name', 'State Bottle Retail',
             'Sale (Dollars)', 'Bottles Sold'],
    chunksize=500_000,
    low_memory=False,
)):
    if i % 5 == 0:
        print(f"  chunk {i+1}...")
    # Parse $ values
    for col in ['State Bottle Retail', 'Sale (Dollars)']:
        chunk[col] = pd.to_numeric(
            chunk[col].astype(str).str.replace('$', '', regex=False),
            errors='coerce'
        )
    chunk['Bottles Sold'] = pd.to_numeric(chunk['Bottles Sold'], errors='coerce')
    chunk = chunk.dropna(subset=['State Bottle Retail', 'Sale (Dollars)',
                                    'Bottles Sold'])
    chunk = chunk[(chunk['Bottles Sold'] > 0)
                   & (chunk['State Bottle Retail'] > 0)
                   & (chunk['Sale (Dollars)'] > 0)]
    # Preço teorico se cobrasse o State Retail
    chunk['preco_teorico'] = chunk['State Bottle Retail'] * chunk['Bottles Sold']
    chunk['pct_desconto'] = ((chunk['preco_teorico'] - chunk['Sale (Dollars)'])
                               / chunk['preco_teorico'] * 100).clip(0, 100)
    chunks_proc.append(chunk[['pct_desconto', 'Bottles Sold']])

dados = pd.concat(chunks_proc, ignore_index=True)
print(f"\nTotal transações: {len(dados):,}")
print(f"Distribuição de descontos:")
print(dados['pct_desconto'].describe().round(2).to_string())

# ── Agrupar por faixa de desconto ──────────────────────────────────────────

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

dados['faixa'] = dados['pct_desconto'].apply(faixa_de)

agg = (dados.groupby('faixa')
              .agg(qtd_media=('Bottles Sold', 'mean'),
                   n_trans=('Bottles Sold', 'count'),
                   pct_centro=('pct_desconto', 'mean'))
              .reset_index())
faixa_order = {f[2]: i for i, f in enumerate(FAIXAS)}
agg['ord'] = agg['faixa'].map(faixa_order)
agg = agg.sort_values('ord')

# Uplift relativo ao baseline
base = agg[agg['faixa'] == '0%']['qtd_media']
base_val = base.iloc[0] if len(base) > 0 else 1
agg['uplift'] = (agg['qtd_media'] / base_val).round(3)
agg['categoria'] = 'destilados'

print()
print("=" * 65)
print("ELASTICIDADE EMPÍRICA — IOWA LIQUOR (destilados, 12 anos)")
print("=" * 65)
print()
print(f"{'Faixa':<10s} {'centro':>7s} {'Garrafas':>9s} {'Uplift':>7s} {'N trans':>10s}")
print('-' * 65)
for _, r in agg.iterrows():
    print(f"  {r['faixa']:<8s} {r['pct_centro']:>6.1f}%  {r['qtd_media']:>7.2f}  "
          f"{r['uplift']:>5.2f}×  {int(r['n_trans']):>10,d}")

# Elasticidade via log-log fit (excluindo 0%)
sub_fit = agg[agg['pct_centro'] > 0.5].copy()
if len(sub_fit) >= 2:
    x = np.log(1 - sub_fit['pct_centro'] / 100)
    y = np.log(sub_fit['uplift'].clip(0.1, 10))
    slope, _ = np.polyfit(x, y, 1)
    elast = float(slope)
    print()
    print(f"Elasticidade promocional EMPÍRICA (Iowa, destilados): {elast:.3f}")
    print(f"  Bijmolt para destilados: -3.40")
    print(f"  Diferença: {elast - (-3.40):+.3f}")

agg.to_csv(OUT / 'elasticidade_empirica.csv', index=False, encoding='utf-8')
pd.DataFrame([{'categoria': 'destilados',
                 'elasticidade_empirica': round(elast, 3),
                 'elasticidade_bijmolt': -3.4,
                 'fonte': 'Iowa Liquor 12 anos'}]
              ).to_csv(OUT / 'elasticidade_resumo.csv', index=False, encoding='utf-8')

# ── Visualização ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(agg['pct_centro'], agg['uplift'], marker='o', linewidth=2,
         color='#7c3aed', markersize=10, label='medido (Iowa, destilados)')

# Comparar com Bijmolt teórico
x_teor = np.linspace(0, 30, 50)
y_teor = (1 - x_teor / 100) ** (-3.40)
ax.plot(x_teor, y_teor, '--', color='red', linewidth=2, alpha=0.7,
         label='Bijmolt teórico (elast=-3.40)')

# Empírico ajustado
y_emp = (1 - x_teor / 100) ** elast
ax.plot(x_teor, y_emp, '--', color='blue', linewidth=2, alpha=0.7,
         label=f'Empírica ajustada (elast={elast:.2f})')

ax.axhline(1.0, color='gray', linestyle=':')
ax.set_xlabel('% Desconto')
ax.set_ylabel('Uplift de volume (garrafas vendidas)')
ax.set_title(f'IOWA LIQUOR — curva desconto × volume\n'
              f'(destilados em lojas estaduais USA, 12 anos)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RES / 'elasticidade_iowa.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ Saídas:")
print(f"  {OUT / 'elasticidade_empirica.csv'}")
print(f"  {OUT / 'elasticidade_resumo.csv'}")
print(f"  {RES / 'elasticidade_iowa.png'}")
