"""Parseia os 52 arquivos de 'produtos não vendidos' (2022-2026) e extrai
o catálogo completo do posto + diagnóstico de estoque parado.

Estrutura do arquivo (descoberta na inspeção):
- Linhas 0-4: cabeçalho fixo (Auto Posto Parque Viana, período)
- Linha 5: cabeçalho das colunas (PRODUTO, QTD ESTOQUE, PREÇO CUSTO, PREÇO VENDA, DATA ÚLTIMA VENDA)
- Bloco por categoria:
    - Linha categoria: texto em col 0 SEM valores nas outras colunas
    - Linhas SKU: nome em col 0 + estoque (3) + custo (4) + venda (5) + ult_venda (6)

Saídas:
  data/catalogo_inferido.csv          (todos os SKUs com categoria, preço, custo)
  data/serie_estoque_parado.csv       (estoque mensal por SKU, 2022-2026)
  results/analise_estoque_parado.csv  (diagnóstico: quanto tempo parado por SKU)
  results/categorias_inferidas.csv    (estatísticas por categoria)
"""
import io
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
INPUT_DIR = ROOT / 'data' / 'produtos_nao_vendidos'
OUT_DATA = ROOT / 'data'
OUT_RESULTS = ROOT / 'results'
OUT_RESULTS.mkdir(exist_ok=True)

# ── Coleta arquivos ────────────────────────────────────────────────────────

arquivos = sorted(INPUT_DIR.glob('*/produtos_nao_vendidos-*.xlsx'))
print(f"Arquivos encontrados: {len(arquivos)}")

# ── Parser por arquivo ─────────────────────────────────────────────────────

def parsear_arquivo(caminho):
    """Retorna lista de dicts: cada SKU com categoria e métricas."""
    # Extrai ano e mês do nome
    m = re.match(r'produtos_nao_vendidos-(\d{4})-(\d+)\.xlsx', caminho.name)
    if not m:
        return []
    ano, mes = int(m.group(1)), int(m.group(2))

    df = pd.read_excel(caminho, header=None)
    registros = []
    categoria_atual = None

    for i, row in df.iterrows():
        col0 = row[0]
        if pd.isna(col0) or not isinstance(col0, str):
            continue
        col0_str = col0.strip()
        if not col0_str:
            continue

        # Pular cabeçalhos fixos
        if col0_str in ('PRODUTO', 'AUTO POSTO PARQUE VIANA LTDA',
                        'PRODUTOS NÃO VENDIDOS'):
            continue
        if col0_str.startswith('DE ') or col0_str.startswith('Emitido'):
            continue

        # Detectar linha de produto vs linha de categoria
        # Produto: tem valores em colunas 3, 4, 5 (qtd, custo, venda)
        valor_estoque = row[3] if len(row) > 3 else None
        valor_custo = row[4] if len(row) > 4 else None
        valor_venda = row[5] if len(row) > 5 else None
        valor_ult_venda = row[6] if len(row) > 6 else None

        eh_produto = (
            (pd.notna(valor_estoque) or pd.notna(valor_custo)
             or pd.notna(valor_venda))
        )

        if eh_produto:
            try:
                estoque = float(valor_estoque) if pd.notna(valor_estoque) else None
            except (ValueError, TypeError):
                estoque = None
            try:
                custo = float(valor_custo) if pd.notna(valor_custo) else None
            except (ValueError, TypeError):
                custo = None
            try:
                venda = float(valor_venda) if pd.notna(valor_venda) else None
            except (ValueError, TypeError):
                venda = None

            ult_venda_str = (str(valor_ult_venda).strip()
                             if pd.notna(valor_ult_venda) else '')
            try:
                ult_venda = pd.to_datetime(ult_venda_str, dayfirst=True,
                                            errors='coerce')
                if pd.isna(ult_venda):
                    ult_venda = None
            except Exception:
                ult_venda = None

            registros.append({
                'ano_snapshot': ano,
                'mes_snapshot': mes,
                'categoria': categoria_atual,
                'sku': col0_str,
                'estoque': estoque,
                'custo_unit': custo,
                'preco_venda': venda,
                'ultima_venda': ult_venda,
            })
        else:
            # Linha de categoria: texto em col 0, vazias nas demais
            categoria_atual = col0_str

    return registros


# ── Processa todos os arquivos ─────────────────────────────────────────────

todos = []
for i, arq in enumerate(arquivos):
    if i % 10 == 0:
        print(f"  Processando {i+1}/{len(arquivos)}: {arq.name}")
    todos.extend(parsear_arquivo(arq))

df = pd.DataFrame(todos)
print(f"  Total de registros: {len(df):,}")
print()

# Limpeza
df = df.dropna(subset=['sku', 'categoria'])
df['sku'] = df['sku'].str.strip().str.upper()
df['categoria'] = df['categoria'].str.strip().str.upper()

# ── 1. Catálogo inferido ──────────────────────────────────────────────────

# Para cada SKU, agrega: categoria mais frequente, preço médio, custo médio
catalogo = (df.groupby('sku')
              .agg(categoria=('categoria', lambda x: x.mode().iloc[0]
                                if len(x.mode()) > 0 else None),
                   preco_venda_medio=('preco_venda', 'mean'),
                   custo_medio=('custo_unit', 'mean'),
                   n_snapshots=('ano_snapshot', 'count'),
                   primeiro_visto=('ano_snapshot',
                                   lambda x: f"{int(x.min())}-{int(df.loc[x.index, 'mes_snapshot'].min())}"),
                   ultimo_visto=('ano_snapshot',
                                  lambda x: f"{int(x.max())}-{int(df.loc[x.index, 'mes_snapshot'].max())}"),
                   ultima_venda_max=('ultima_venda', 'max'))
              .reset_index())
catalogo['margem_R$'] = (catalogo['preco_venda_medio']
                          - catalogo['custo_medio']).round(2)
catalogo['margem_pct'] = ((catalogo['margem_R$']
                            / catalogo['preco_venda_medio'])
                            * 100).round(1)
catalogo['preco_venda_medio'] = catalogo['preco_venda_medio'].round(2)
catalogo['custo_medio'] = catalogo['custo_medio'].round(4)
catalogo = catalogo.sort_values(['categoria', 'sku']).reset_index(drop=True)
catalogo.to_csv(OUT_DATA / 'catalogo_inferido.csv',
                 index=False, encoding='utf-8')

# ── 2. Série mensal de estoque parado por SKU ─────────────────────────────

serie = df[['ano_snapshot', 'mes_snapshot', 'categoria', 'sku',
            'estoque', 'ultima_venda']].copy()
serie.to_csv(OUT_DATA / 'serie_estoque_parado.csv',
              index=False, encoding='utf-8')

# ── 3. Diagnóstico por SKU: tempo parado, estoque atual ────────────────────

# Última visão (snapshot mais recente)
ultimo_snap = (df.sort_values(['ano_snapshot', 'mes_snapshot'])
                  .drop_duplicates('sku', keep='last'))

hoje = datetime(2026, 5, 12)
ultimo_snap = ultimo_snap.copy()
ultimo_snap['dias_sem_vender'] = ultimo_snap['ultima_venda'].apply(
    lambda d: (hoje - d).days if pd.notna(d) else 9999
)
ultimo_snap['valor_estoque_parado_R$'] = (
    ultimo_snap['estoque'].fillna(0).clip(lower=0)
    * ultimo_snap['custo_unit'].fillna(0)
).round(2)

diagnostico = ultimo_snap[['categoria', 'sku', 'estoque', 'custo_unit',
                             'preco_venda', 'ultima_venda',
                             'dias_sem_vender', 'valor_estoque_parado_R$']]
diagnostico = diagnostico.sort_values('valor_estoque_parado_R$',
                                        ascending=False)
diagnostico.to_csv(OUT_RESULTS / 'analise_estoque_parado.csv',
                    index=False, encoding='utf-8')

# ── 4. Estatísticas por categoria ─────────────────────────────────────────

cat_stats = (catalogo.groupby('categoria')
                       .agg(n_skus=('sku', 'count'),
                            preco_medio=('preco_venda_medio', 'mean'),
                            margem_pct_media=('margem_pct', 'mean'))
                       .reset_index()
                       .sort_values('n_skus', ascending=False))
cat_stats['preco_medio'] = cat_stats['preco_medio'].round(2)
cat_stats['margem_pct_media'] = cat_stats['margem_pct_media'].round(1)
cat_stats.to_csv(OUT_RESULTS / 'categorias_inferidas.csv',
                  index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print("="*70)
print("ANÁLISE: PRODUTOS NÃO VENDIDOS (2022-2026)")
print("="*70)
print()
print(f"Arquivos processados:       {len(arquivos)}")
print(f"Registros totais:           {len(df):,}")
print(f"SKUs únicos no catálogo:    {df['sku'].nunique():,}")
print(f"Categorias únicas:          {df['categoria'].nunique()}")
print()

# Datas cobertas
periodos = df.apply(lambda r: f"{int(r['ano_snapshot'])}-{int(r['mes_snapshot']):02d}", axis=1)
print(f"Período coberto:            {periodos.min()} a {periodos.max()}")
print(f"Snapshots únicos:           {periodos.nunique()}")
print()

print("Top 15 categorias por n° de SKUs:")
print('-'*60)
for _, row in cat_stats.head(15).iterrows():
    print(f"  {row['categoria']:<35s} {row['n_skus']:>4d} SKUs  "
          f"preço médio R$ {row['preco_medio']:>7.2f}  "
          f"margem {row['margem_pct_media']:>5.1f}%")

print()
print("="*70)
print("DIAGNÓSTICO DE ESTOQUE PARADO (snapshot mais recente)")
print("="*70)
print()

total_valor = diagnostico['valor_estoque_parado_R$'].sum()
print(f"Valor TOTAL em estoque parado:  R$ {total_valor:,.2f}")
print(f"SKUs em estoque parado:         {len(diagnostico):,}")
print(f"  - Com venda nos últimos 30d:  {(diagnostico['dias_sem_vender'] <= 30).sum():,}")
print(f"  - Sem venda há > 6 meses:     {(diagnostico['dias_sem_vender'] > 180).sum():,}")
print(f"  - Nunca vendido (sem data):   {(diagnostico['dias_sem_vender'] == 9999).sum():,}")
print()

print("TOP 15 SKUs com mais dinheiro parado:")
print('-'*80)
for _, row in diagnostico.head(15).iterrows():
    sku_short = row['sku'][:40]
    ult = (row['ultima_venda'].strftime('%d/%m/%Y')
           if pd.notna(row['ultima_venda']) else 'NUNCA')
    print(f"  R$ {row['valor_estoque_parado_R$']:>10,.2f}  "
          f"{sku_short:<42s} qtd {int(row['estoque']) if pd.notna(row['estoque']) else '?':>6}  "
          f"últ. {ult}")
print()

print("Categorias com mais SKUs em estoque parado:")
print('-'*60)
cat_parado = (diagnostico.groupby('categoria')
                          .agg(n_skus=('sku', 'count'),
                                valor_parado=('valor_estoque_parado_R$', 'sum'))
                          .sort_values('valor_parado', ascending=False))
for cat, row in cat_parado.head(15).iterrows():
    print(f"  {cat:<35s} {int(row['n_skus']):>4d} SKUs  R$ {row['valor_parado']:>10,.2f}")
print()

print(f"✓ Saídas:")
print(f"  data/catalogo_inferido.csv")
print(f"  data/serie_estoque_parado.csv")
print(f"  results/analise_estoque_parado.csv")
print(f"  results/categorias_inferidas.csv")
