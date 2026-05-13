"""Processa Tesco Grocery 1.0 — vendas de supermercado físico UK por mês
e área geográfica. Tem **fração da cesta** dedicada a cada categoria:
f_chocolate, f_sweets, f_beer, f_wine, f_spirits, f_soft_drinks, etc.

Resposta direta à pergunta: "tem dataset de chocolate em loja física?"

Saída:
  data/priors_externos/tesco/sazonalidade_mensal.csv
  results/v11/tesco_sazonalidade.png
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
RAW = ROOT / 'data' / 'raw_tesco'
OUT = ROOT / 'data' / 'priors_externos' / 'tesco'
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / 'results' / 'v11'

# ── Cada mês tem 1 arquivo borough (33 áreas de Londres) ──────────────────

MESES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MES_NUM = {m: i + 1 for i, m in enumerate(MESES)}

# Colunas-foco — fração da cesta para cada categoria
# Tesco UK não tem chocolate separado — f_sweets é proxy (doces+chocolate)
CATS_FOCO = ['f_sweets', 'f_beer', 'f_wine', 'f_spirits',
              'f_soft_drinks', 'f_water', 'f_tea_coffee', 'f_readymade',
              'f_dairy']

# ── Ler 12 arquivos borough ───────────────────────────────────────────────

dados = []
for mes_abrev in MESES:
    f = RAW / f'{mes_abrev}_borough_grocery.csv'
    if not f.exists():
        print(f"⚠ {f.name} não existe")
        continue
    df = pd.read_csv(f)
    df['mes'] = MES_NUM[mes_abrev]
    df['mes_nome'] = mes_abrev
    # Pegar só colunas de fração que nos interessam
    cols_existem = [c for c in CATS_FOCO if c in df.columns]
    sub = df[['area_id', 'mes', 'mes_nome'] + cols_existem]
    dados.append(sub)
    print(f"  {mes_abrev}: {len(df)} áreas, {len(cols_existem)} categorias")

todos = pd.concat(dados, ignore_index=True)
print(f"\nTotal: {len(todos)} (mes × area)")

# ── Sazonalidade — média entre áreas por mês ──────────────────────────────

saz = (todos.groupby(['mes', 'mes_nome'])[CATS_FOCO]
              .mean()
              .reset_index()
              .sort_values('mes'))

# Calcular uplift_mensal = fração_mês / fração_média_anual
for cat in CATS_FOCO:
    if cat not in saz.columns:
        continue
    media_anual = saz[cat].mean()
    if media_anual > 0:
        saz[f'{cat}_uplift'] = (saz[cat] / media_anual).round(3)

saz.to_csv(OUT / 'sazonalidade_mensal.csv', index=False, encoding='utf-8')

# ── Imprimir ─────────────────────────────────────────────────────────────────

print()
print("=" * 80)
print("TESCO GROCERY UK — sazonalidade da CESTA por categoria (fração mensal)")
print("=" * 80)
print()
print("Uplift > 1.0 = mês em que essa categoria pesa mais na cesta vs média anual")
print()

linhas = []
for cat in CATS_FOCO:
    col_up = f'{cat}_uplift'
    if col_up not in saz.columns:
        continue
    # Top 3 meses dessa categoria
    top = saz.nlargest(3, col_up)
    bot = saz.nsmallest(2, col_up)
    cat_label = cat.replace('f_', '').replace('_', ' ').title()
    print(f"\n{cat_label}:")
    print(f"  Picos:  ", end='')
    print(', '.join([f"{r['mes_nome']} ({r[col_up]:.2f}×)" for _, r in top.iterrows()]))
    print(f"  Vales:  ", end='')
    print(', '.join([f"{r['mes_nome']} ({r[col_up]:.2f}×)" for _, r in bot.iterrows()]))

# ── Visualização ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(16, 11))
for i, cat in enumerate(CATS_FOCO):
    if i >= 9:
        break
    ax = axes[i // 3, i % 3]
    col_up = f'{cat}_uplift'
    if col_up not in saz.columns:
        ax.set_title(f'{cat}\n(sem dados)')
        continue
    cores = ['#16a34a' if u > 1.10 else '#ca8a04' if u > 1.0 else '#9ca3af'
              for u in saz[col_up]]
    ax.bar(saz['mes_nome'], saz[col_up], color=cores, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    titulo = cat.replace('f_', '').replace('_', ' ').title()
    ax.set_title(titulo, fontsize=11)
    ax.set_ylabel('Uplift × média anual', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    # Marcar Easter (Apr), Christmas (Dec), Mother's Day UK (Mar/Apr), Father's Day UK (Jun)
    for mes_idx, evento in [(3, 'Mãe UK'), (3, 'Easter'), (5, 'Pai UK'),
                              (11, 'Natal')]:
        # mes_idx é índice 0-11 no array
        pass  # simplificar — só marcar Easter (Apr) e Natal (Dec)
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)

plt.suptitle('Tesco UK — sazonalidade da cesta por categoria\n'
              '(supermercado físico, Londres, 1 ano)',
              fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(RES / 'tesco_sazonalidade.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ Saídas:")
print(f"  {OUT / 'sazonalidade_mensal.csv'}")
print(f"  {RES / 'tesco_sazonalidade.png'}")
