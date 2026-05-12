"""Análise temporal do estoque parado — 4 anos de 'produtos não vendidos'.

Identifica:
1. SKUs sempre parados (estoque morto crônico) — aparecem em quase todos snapshots
2. SKUs sazonais — aparecem mais em certos meses
3. SKUs com vida curta — descontinuados ou rotativos
4. Categorias com problema crônico (% SKUs sempre parados)
5. Evolução temporal do dinheiro imobilizado

Output: results/v11/analise_estoque_temporal_*.csv + .png

Pré-requisito: data/serie_estoque_parado.csv (gerado por analisar_produtos_nao_vendidos.py)
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
DATA = ROOT / 'data'
RESULTS = ROOT / 'results' / 'v11'
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Carrega série mensal ──────────────────────────────────────────────────

serie = pd.read_csv(DATA / 'serie_estoque_parado.csv', parse_dates=['ultima_venda'])
print(f"Total registros: {len(serie):,}")
print(f"SKUs únicos: {serie['sku'].nunique():,}")
print(f"Categorias: {serie['categoria'].nunique()}")
print(f"Snapshots: {len(serie.groupby(['ano_snapshot','mes_snapshot']))}")

# Filtrar só conveniência (usar tabela de classificação)
classif = pd.read_csv(DATA / 'categorias_classificadas.csv')
cats_conveniencia = set(classif[classif['tipo'].isin(['conveniencia', 'revisar'])]['categoria'])
serie_conv = serie[serie['categoria'].isin(cats_conveniencia)].copy()
print(f"  Filtrado para conveniência: {len(serie_conv):,} registros, "
      f"{serie_conv['sku'].nunique():,} SKUs")

# Período por snapshot
serie_conv['periodo'] = serie_conv['ano_snapshot'].astype(str) + '-' + \
                          serie_conv['mes_snapshot'].astype(str).str.zfill(2)
total_snapshots = serie_conv['periodo'].nunique()
print(f"  Total snapshots: {total_snapshots}")

# ── 1. SKUs sempre parados (cronicos) ────────────────────────────────────

freq_aparicao = (serie_conv.groupby('sku')
                            .agg(n_snapshots=('periodo', 'nunique'),
                                 categoria=('categoria', 'first'))
                            .reset_index())
freq_aparicao['pct_snapshots'] = (freq_aparicao['n_snapshots']
                                    / total_snapshots * 100).round(1)

# Cronicos = aparece em >80% dos snapshots
cronicos = freq_aparicao[freq_aparicao['pct_snapshots'] > 80].sort_values(
    'pct_snapshots', ascending=False)
print(f"\nSKUs cronicamente parados (>80% snapshots): {len(cronicos)}")

# Eventuais = aparece em <20% snapshots
eventuais = freq_aparicao[freq_aparicao['pct_snapshots'] < 20]
print(f"SKUs eventualmente parados (<20% snapshots): {len(eventuais)}")

# Salvar
freq_aparicao.to_csv(RESULTS / 'sku_frequencia_estoque_parado.csv',
                      index=False, encoding='utf-8')

# ── 2. Categorias com problema cronico ─────────────────────────────────────

cat_problemas = []
for cat in serie_conv['categoria'].unique():
    skus_cat = freq_aparicao[freq_aparicao['categoria'] == cat]
    n_total = len(skus_cat)
    n_cronicos = (skus_cat['pct_snapshots'] > 80).sum()
    n_eventuais = (skus_cat['pct_snapshots'] < 20).sum()
    pct_cronicos = n_cronicos / n_total * 100 if n_total > 0 else 0
    cat_problemas.append({
        'categoria': cat,
        'n_skus_total': n_total,
        'n_cronicos': int(n_cronicos),
        'n_eventuais': int(n_eventuais),
        'pct_cronicos': round(pct_cronicos, 1),
    })
df_cat = pd.DataFrame(cat_problemas).sort_values('pct_cronicos', ascending=False)
df_cat.to_csv(RESULTS / 'categorias_problema_cronico.csv',
                index=False, encoding='utf-8')

print()
print("TOP 15 categorias com mais SKUs CRONICAMENTE parados:")
for _, row in df_cat.head(15).iterrows():
    print(f"  {row['categoria']:<30s} {int(row['n_cronicos']):>3d}/{int(row['n_skus_total']):<3d} "
          f"({row['pct_cronicos']:>5.1f}%)")

# ── 3. Evolução temporal do dinheiro imobilizado ──────────────────────────

# Custo unitario × estoque por snapshot, sumarizado
# Ler catálogo para custo
cat = pd.read_csv(DATA / 'catalogo_inferido.csv')
custo_map = dict(zip(cat['sku'], cat['custo_medio']))
serie_conv['custo_unit_cat'] = serie_conv['sku'].map(custo_map).fillna(0)
serie_conv['valor_parado_R$'] = (pd.to_numeric(serie_conv['estoque'], errors='coerce').fillna(0).clip(lower=0)
                                    * serie_conv['custo_unit_cat'])

valor_por_periodo = serie_conv.groupby('periodo')['valor_parado_R$'].sum().reset_index()
valor_por_periodo = valor_por_periodo.sort_values('periodo')
valor_por_periodo.to_csv(RESULTS / 'evolucao_valor_estoque_parado.csv',
                          index=False, encoding='utf-8')

print()
print("Evolução do dinheiro parado em conveniência (primeiros e últimos 5):")
for _, row in valor_por_periodo.head(5).iterrows():
    print(f"  {row['periodo']}  R$ {row['valor_parado_R$']:>12,.2f}")
print("  ...")
for _, row in valor_por_periodo.tail(5).iterrows():
    print(f"  {row['periodo']}  R$ {row['valor_parado_R$']:>12,.2f}")

# ── 4. Padrão sazonal — produtos com pico em verão / inverno ──────────────

# Para cada SKU, % de aparição por MES (consolidando vários anos)
sazonalidade = []
for sku, grupo in serie_conv.groupby('sku'):
    if len(grupo) < 6:  # pelo menos 6 snapshots
        continue
    meses_aparece = grupo['mes_snapshot'].value_counts(normalize=True) * 100
    if len(meses_aparece) < 3:
        continue
    pico_mes = int(meses_aparece.idxmax())
    pico_pct = float(meses_aparece.max())
    vale_pct = float(meses_aparece.min()) if len(meses_aparece) > 0 else 0
    variacao = pico_pct - vale_pct
    if variacao > 30:  # SKU sazonal (varia muito entre meses)
        sazonalidade.append({
            'sku': sku,
            'categoria': grupo['categoria'].iloc[0],
            'pico_mes': pico_mes,
            'pico_pct_aparicao': round(pico_pct, 1),
            'vale_pct_aparicao': round(vale_pct, 1),
            'variacao_pct': round(variacao, 1),
        })

df_saz = pd.DataFrame(sazonalidade)
if len(df_saz) > 0:
    df_saz = df_saz.sort_values('variacao_pct', ascending=False)
df_saz.to_csv(RESULTS / 'skus_sazonais.csv', index=False, encoding='utf-8')

print()
print(f"SKUs com padrão sazonal (variação >30% entre pico/vale): {len(df_saz)}")
nomes_mes = ['', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
if len(df_saz) > 0:
    print("Top 10:")
    for _, row in df_saz.head(10).iterrows():
        print(f"  {row['sku'][:35]:<35s} {row['categoria']:<20s} "
              f"pico={nomes_mes[int(row['pico_mes'])]} ({row['pico_pct_aparicao']:.0f}%) "
              f"vale={row['vale_pct_aparicao']:.0f}% Δ={row['variacao_pct']:.0f}%")
else:
    print("  (nenhum SKU sazonal claro encontrado — granularidade mensal pode ser insuficiente)")

# ── 5. Gráfico de evolução ────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Painel 1: evolução do dinheiro parado
ax = axes[0]
valor_por_periodo['data'] = pd.to_datetime(valor_por_periodo['periodo'] + '-01',
                                              format='%Y-%m-%d')
ax.plot(valor_por_periodo['data'], valor_por_periodo['valor_parado_R$'] / 1000,
         marker='o', linewidth=2, color='steelblue')
ax.set_ylabel('R$ parado (milhares)')
ax.set_title('Evolução do dinheiro imobilizado em estoque parado (conveniência)')
ax.grid(alpha=0.3)
ax.fill_between(valor_por_periodo['data'],
                  valor_por_periodo['valor_parado_R$'] / 1000,
                  alpha=0.2, color='steelblue')

# Painel 2: top categorias com problema cronico
ax = axes[1]
top10 = df_cat.head(10)
ax.barh(top10['categoria'], top10['pct_cronicos'], color='coral')
ax.set_xlabel('% de SKUs com aparição crônica (>80% snapshots)')
ax.set_title('Top 10 categorias com mais estoque parado crônico')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / 'evolucao_estoque_parado.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print("="*70)
print("ANÁLISE TEMPORAL — DESCOBERTAS PRINCIPAIS")
print("="*70)
print()
print(f"1. {len(cronicos)} SKUs cronicamente parados (>80% snapshots)")
print(f"   = candidatos a DESCONTINUAR")
print()
print(f"2. {len(eventuais)} SKUs eventualmente parados (<20%)")
print(f"   = produtos rotativos / sazonais — POSSÍVEL alvo de promoção")
print()
print(f"3. {len(df_saz)} SKUs com padrão sazonal claro")
print(f"   = candidatos a CALENDÁRIO de promoção por temporada")
print()
print(f"4. Valor parado médio: R$ {valor_por_periodo['valor_parado_R$'].mean():,.2f}")
print(f"   Min: R$ {valor_por_periodo['valor_parado_R$'].min():,.2f}")
print(f"   Max: R$ {valor_por_periodo['valor_parado_R$'].max():,.2f}")
print(f"   Tendência (último vs primeiro): "
      f"{(valor_por_periodo['valor_parado_R$'].iloc[-1] / valor_por_periodo['valor_parado_R$'].iloc[0] - 1) * 100:+.1f}%")
print()
print(f"✓ Arquivos gerados:")
print(f"  results/v11/sku_frequencia_estoque_parado.csv")
print(f"  results/v11/categorias_problema_cronico.csv")
print(f"  results/v11/evolucao_valor_estoque_parado.csv")
print(f"  results/v11/skus_sazonais.csv")
print(f"  results/v11/evolucao_estoque_parado.png")
