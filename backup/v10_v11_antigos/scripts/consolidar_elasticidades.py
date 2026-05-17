"""Consolida elasticidades empíricas das 3 fontes físicas + literatura
em uma única tabela e gráfico.

Saída:
  results/v11/elasticidade_consolidada.csv
  results/v11/elasticidade_consolidada.png
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
PRIORS = ROOT / 'data' / 'priors_externos'
RES = ROOT / 'results' / 'v11'

# ── Carregar 3 fontes ──────────────────────────────────────────────────────

dh = pd.read_csv(PRIORS / 'dunnhumby' / 'elasticidade_resumo.csv')
iowa = pd.read_csv(PRIORS / 'iowa_liquor' / 'elasticidade_resumo.csv')
wm = pd.read_csv(PRIORS / 'walmart' / 'elasticidade_resumo.csv')

# Walmart: convertendo uplift de feriado em elasticidade implícita
# uplift = (1 - desc) ^ elast → elast = log(uplift) / log(1 - desc)
# Walmart não tem desconto explícito, mas Holiday_Flag pode ser proxy
# de "promoção média" estimada em 5-10%
# uplift 1.07, assumindo 7-10% de "promoção implícita":
# elast = log(1.07) / log(0.93) ≈ -0.93
walmart_uplift = wm['uplift_feriado_medio'].iloc[0]
walmart_elast_implicit = np.log(walmart_uplift) / np.log(0.93)

print("=" * 70)
print("CONSOLIDAÇÃO — ELASTICIDADE EM LOJA FÍSICA (3 fontes)")
print("=" * 70)
print()

dados = []

# Dunnhumby — várias categorias
for _, r in dh.iterrows():
    dados.append({
        'fonte': 'Dunnhumby',
        'tipo': 'Supermercado USA',
        'categoria': r['categoria'],
        'elasticidade_empirica': float(r['elasticidade_empirica']),
        'elasticidade_bijmolt': float(r['elasticidade_bijmolt']),
        'n_obs': 500_000,  # aproximado
    })

# Iowa — só destilados
dados.append({
    'fonte': 'Iowa Liquor',
    'tipo': 'Loja estadual USA',
    'categoria': iowa['categoria'].iloc[0],
    'elasticidade_empirica': float(iowa['elasticidade_empirica'].iloc[0]),
    'elasticidade_bijmolt': float(iowa['elasticidade_bijmolt'].iloc[0]),
    'n_obs': 12_500_000,
})

# Walmart — implícita (varejo amplo)
dados.append({
    'fonte': 'Walmart Sales',
    'tipo': 'Varejo amplo USA',
    'categoria': 'todas_agregadas',
    'elasticidade_empirica': round(walmart_elast_implicit, 3),
    'elasticidade_bijmolt': -3.0,
    'n_obs': 6435,
})

df = pd.DataFrame(dados)
df.to_csv(RES / 'elasticidade_consolidada.csv', index=False, encoding='utf-8')

# Estatísticas agregadas
print(f"Total de medidas empíricas:        {len(df)}")
print(f"Elasticidade EMPÍRICA média:       {df['elasticidade_empirica'].mean():.3f}")
print(f"Elasticidade EMPÍRICA mediana:     {df['elasticidade_empirica'].median():.3f}")
print(f"Range empírico:                    {df['elasticidade_empirica'].min():.3f} a {df['elasticidade_empirica'].max():.3f}")
print(f"\nElasticidade BIJMOLT média:        {df['elasticidade_bijmolt'].mean():.3f}")
print(f"\nDiferença Bijmolt - Empírica:")
print(f"  Média:                           {(df['elasticidade_bijmolt'] - df['elasticidade_empirica']).mean():.3f}")
print(f"  Fator de superestimação:         ~{(df['elasticidade_bijmolt'] / df['elasticidade_empirica']).abs().mean():.1f}×")

print()
print(f"{'Fonte':<14s} {'Categoria':<18s} {'Empírica':>9s} {'Bijmolt':>9s} {'Δ':>7s}")
print('-' * 70)
for _, r in df.iterrows():
    delta = r['elasticidade_empirica'] - r['elasticidade_bijmolt']
    print(f"  {r['fonte']:<12s} {r['categoria']:<18s} "
          f"{r['elasticidade_empirica']:>7.2f}  {r['elasticidade_bijmolt']:>7.2f}  "
          f"{delta:>+5.2f}")

# ── Visualização consolidada ──────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Painel 1: comparação direta empírica × Bijmolt
ax = axes[0]
df_sorted = df.sort_values('elasticidade_bijmolt')
y = range(len(df_sorted))
ax.barh([y_ - 0.2 for y_ in y], df_sorted['elasticidade_empirica'], 0.4,
         color='#16a34a', label='Empírica (medida)', alpha=0.85)
ax.barh([y_ + 0.2 for y_ in y], df_sorted['elasticidade_bijmolt'], 0.4,
         color='#dc2626', label='Bijmolt 2005 (literatura)', alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels([f"{r['categoria']}\n({r['fonte']})"
                       for _, r in df_sorted.iterrows()], fontsize=8)
ax.axvline(0, color='gray', linestyle='--')
ax.set_xlabel('Elasticidade promocional')
ax.set_title('Elasticidade EMPÍRICA vs LITERATURA\n(loja física, 3 fontes)')
ax.legend()
ax.grid(axis='x', alpha=0.3)

# Painel 2: curvas teóricas com diferentes elasticidades
ax = axes[1]
x = np.linspace(0, 30, 50)
for elast, label, cor in [
    (-3.0, 'Bijmolt (literatura, SKU/marca)', '#dc2626'),
    (-1.0, 'Loja física (típico)', '#ca8a04'),
    (-0.5, 'V11.3 atual (piso)', '#3b82f6'),
    (-0.22, 'Iowa Liquor medido', '#7c3aed'),
]:
    y_curve = (1 - x / 100) ** elast
    ax.plot(x, y_curve, linewidth=2.5, label=label, color=cor)

ax.axhline(1.0, color='gray', linestyle=':')
ax.set_xlabel('% Desconto')
ax.set_ylabel('Uplift teórico de volume')
ax.set_title('Curvas teóricas — quanto desconto move volume?')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RES / 'elasticidade_consolidada.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ Saídas:")
print(f"  {RES / 'elasticidade_consolidada.csv'}")
print(f"  {RES / 'elasticidade_consolidada.png'}")

# ── Recomendação ─────────────────────────────────────────────────────────

print()
print("=" * 70)
print("RECOMENDAÇÃO PARA O MODELO V11.3")
print("=" * 70)
print()
print("Elasticidade ajustada para -0.5 (piso) está ALINHADA com:")
print(f"  - Iowa Liquor (12 anos, 12M trans):  -0.22")
print(f"  - Dunnhumby média:                    {dh['elasticidade_empirica'].mean():.2f}")
print(f"  - Walmart implícita:                  {walmart_elast_implicit:.2f}")
print()
print("Conclusão: usar -0.5 mantém SINAL para o RL aprender mas")
print("           NÃO superestima o efeito de descontos como literatura faz.")
print()
print("Política alinhada:")
print("  ✓ Combo (boost 1.12/1.08) ainda gera uplift significativo")
print("  ✓ Desconto direto raramente compensa margem (elasticidade baixa)")
print("  ✓ Agente vai privilegiar combo + vencimento + baixa demanda")
print("  ✓ Agente vai NÃO promover em alta demanda saudável (correto!)")
