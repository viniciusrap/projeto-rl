"""Walmart Sales: extrai magnitude de uplift em feriado (proxy de evento
promocional) vs nao-feriado.

Walmart NÃO tem coluna de preço/desconto por SKU. Tem Weekly_Sales,
Holiday_Flag, Temperature, Fuel_Price, CPI. Permite estimar:
- Quanto venda aumenta em semana de feriado (proxy de promo agressiva)
- Sensibilidade a CPI/inflação (proxy de elasticidade)

Limitação: Walmart é varejo amplo (não loja conveniência), e Holiday_Flag
não é desconto direto mas situação de varejo em alta demanda + promoções.

Saída: data/priors_externos/walmart/elasticidade_proxy.csv
       results/v11/elasticidade_walmart.png
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
RAW = ROOT / 'data' / 'raw_walmart_m5' / 'Walmart_Sales.csv'
OUT = ROOT / 'data' / 'priors_externos' / 'walmart'
RES = ROOT / 'results' / 'v11'

df = pd.read_csv(RAW)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
print(f"Walmart Sales: {len(df):,} linhas (45 lojas × 143 semanas)")

# ── Análise 1: feriado vs nao-feriado ─────────────────────────────────────

print()
print("=" * 65)
print("UPLIFT EM SEMANA DE FERIADO (proxy de evento promocional)")
print("=" * 65)
print()

por_loja = (df.groupby(['Store', 'Holiday_Flag'])['Weekly_Sales']
              .mean().unstack())
por_loja['uplift'] = por_loja[1] / por_loja[0]
print(f"Uplift médio em feriado vs nao-feriado: {por_loja['uplift'].mean():.3f}×")
print(f"  Mediana: {por_loja['uplift'].median():.3f}×")
print(f"  Desvio: {por_loja['uplift'].std():.3f}")
print(f"  Min: {por_loja['uplift'].min():.3f}×")
print(f"  Max: {por_loja['uplift'].max():.3f}×")

# ── Análise 2: sensibilidade de venda a CPI (proxy de preço relativo) ─────

print()
print("=" * 65)
print("SENSIBILIDADE DA VENDA A CPI (proxy macro de elasticidade)")
print("=" * 65)
print()

# Para cada loja, regredir log(Weekly_Sales) ~ log(CPI)
elasticidades = []
for store in df['Store'].unique():
    sub = df[df['Store'] == store]
    if len(sub) < 50:
        continue
    # log-log regression
    x = np.log(sub['CPI'])
    y = np.log(sub['Weekly_Sales'])
    slope, _ = np.polyfit(x, y, 1)
    elasticidades.append({
        'store': int(store),
        'elast_cpi': round(float(slope), 3),
    })

df_elast = pd.DataFrame(elasticidades)
df_elast.to_csv(OUT / 'elasticidade_cpi.csv', index=False, encoding='utf-8')

print(f"Elasticidade de venda x CPI ({len(df_elast)} lojas):")
print(f"  Média: {df_elast['elast_cpi'].mean():.3f}")
print(f"  Mediana: {df_elast['elast_cpi'].median():.3f}")
print(f"  Std: {df_elast['elast_cpi'].std():.3f}")

# Interpretação: elast_cpi negativo = venda cai quando CPI sobe (caro)
# Magnitude esperada: -0.5 a -1.5 (varejo amplo)

# ── Análise 3: efeito de fuel_price (proxy de poder de compra) ────────────

# Não muito útil, mas vou calcular
print()
print(f"Correlação venda × fuel_price (poder de compra):")
correl_fuel = df.groupby('Store').apply(
    lambda g: g['Weekly_Sales'].corr(g['Fuel_Price']),
    include_groups=False
).mean()
print(f"  Média entre lojas: {correl_fuel:.3f}")

# ── Resumo ────────────────────────────────────────────────────────────────

resumo = {
    'fonte': 'Walmart Sales 45 lojas USA 3 anos',
    'uplift_feriado_medio': round(por_loja['uplift'].mean(), 3),
    'uplift_feriado_mediana': round(por_loja['uplift'].median(), 3),
    'elast_cpi_media': round(df_elast['elast_cpi'].mean(), 3),
    'elast_cpi_mediana': round(df_elast['elast_cpi'].median(), 3),
    'correl_fuel': round(float(correl_fuel), 3),
    'n_lojas': len(df_elast),
}
pd.DataFrame([resumo]).to_csv(OUT / 'elasticidade_resumo.csv',
                                index=False, encoding='utf-8')

# ── Visualização ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Painel 1: distribuição de uplift por loja
ax = axes[0]
ax.hist(por_loja['uplift'], bins=20, color='#3b82f6', alpha=0.7,
         edgecolor='black')
ax.axvline(por_loja['uplift'].mean(), color='red', linewidth=2,
            label=f"Média {por_loja['uplift'].mean():.2f}×")
ax.axvline(1.0, color='gray', linestyle='--')
ax.set_xlabel('Uplift em semana de feriado (× nao-feriado)')
ax.set_ylabel('Nº lojas')
ax.set_title('Uplift de venda em feriado — 45 lojas Walmart')
ax.legend()
ax.grid(alpha=0.3)

# Painel 2: distribuição de elasticidade-CPI por loja
ax = axes[1]
ax.hist(df_elast['elast_cpi'], bins=20, color='#a855f7', alpha=0.7,
         edgecolor='black')
ax.axvline(df_elast['elast_cpi'].mean(), color='red', linewidth=2,
            label=f"Média {df_elast['elast_cpi'].mean():.2f}")
ax.axvline(0, color='gray', linestyle='--')
ax.set_xlabel('Elasticidade venda × CPI (regressão log-log)')
ax.set_ylabel('Nº lojas')
ax.set_title('Elasticidade-CPI por loja — proxy macro')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RES / 'elasticidade_walmart.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ Saídas:")
print(f"  {OUT / 'elasticidade_cpi.csv'}")
print(f"  {OUT / 'elasticidade_resumo.csv'}")
print(f"  {RES / 'elasticidade_walmart.png'}")
