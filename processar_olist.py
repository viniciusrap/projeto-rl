"""Processa dataset Olist (Kaggle) para extrair uplift por categoria × data comercial.

Pré-requisito: arquivos do Olist em data/raw_olist/
(ver instruções de download em data/priors_externos/README.md)

O que faz:
1. Junta orders + items + products → tabela (timestamp, categoria, qtd, valor)
2. Agrega por (dia, categoria_modelo) — mapeia categorias Olist → categorias úteis
3. Para cada data comercial brasileira:
   - Vendas na janela do evento (até 7 dias antes)
   - Vendas em janela equivalente fora do evento (baseline)
   - Uplift = janela / baseline
4. Gera tabela final por (categoria, evento, ano) com uplift medido

Saída:
  data/priors_externos/olist/uplift_por_evento.csv      (1 linha por categoria × evento × ano)
  data/priors_externos/olist/uplift_agregado.csv         (média entre anos)
  data/priors_externos/olist/serie_diaria_por_categoria.csv  (série diária bruta)
"""
import io
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RAW = ROOT / 'data' / 'raw_olist'
OUT = ROOT / 'data' / 'priors_externos' / 'olist'
OUT.mkdir(parents=True, exist_ok=True)

# ── Verificar pré-requisitos ────────────────────────────────────────────────

arquivos_necessarios = [
    'olist_orders_dataset.csv',
    'olist_order_items_dataset.csv',
    'olist_products_dataset.csv',
    'product_category_name_translation.csv',
]
faltando = [f for f in arquivos_necessarios if not (RAW / f).exists()]
if faltando:
    print("✗ Arquivos do Olist não encontrados em data/raw_olist/")
    print()
    print("Faltando:")
    for f in faltando:
        print(f"  - {f}")
    print()
    print("Instruções de download:")
    print("  1. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
    print("  2. Clica em 'Download' (~120MB)")
    print("  3. Extrai o zip em data/raw_olist/")
    print()
    print("Ou via API (ver data/priors_externos/README.md)")
    sys.exit(1)

# ── 1. Carregar e juntar ────────────────────────────────────────────────────

print("Carregando arquivos Olist...")
orders = pd.read_csv(RAW / 'olist_orders_dataset.csv',
                     parse_dates=['order_purchase_timestamp'])
items = pd.read_csv(RAW / 'olist_order_items_dataset.csv')
products = pd.read_csv(RAW / 'olist_products_dataset.csv')
translation = pd.read_csv(RAW / 'product_category_name_translation.csv')

print(f"  orders:      {len(orders):,}")
print(f"  items:       {len(items):,}")
print(f"  products:    {len(products):,}")

# Filtrar orders entregues
orders = orders[orders['order_status'] == 'delivered'].copy()
print(f"  orders entregues: {len(orders):,}")

# Join completo
df = (items
      .merge(orders[['order_id', 'order_purchase_timestamp']], on='order_id')
      .merge(products[['product_id', 'product_category_name']], on='product_id')
      .merge(translation, on='product_category_name', how='left'))

df['data'] = df['order_purchase_timestamp'].dt.date
df['ano'] = df['order_purchase_timestamp'].dt.year
df['mes'] = df['order_purchase_timestamp'].dt.month
df['dia_semana'] = df['order_purchase_timestamp'].dt.dayofweek

print(f"  Linhas após join: {len(df):,}")
print(f"  Período: {df['data'].min()} a {df['data'].max()}")

# ── 2. Mapear categorias Olist → categorias do nosso modelo ────────────────

# Olist é e-commerce, não conveniência de posto. Mapeamento parcial:
# categorias úteis como proxy para datas comerciais brasileiras.

MAPA_CATEGORIA = {
    # Alimentos/bebidas
    'food': 'alimentos',
    'food_drink': 'alimentos_bebidas',
    'drinks': 'bebidas',
    'la_cuisine': 'alimentos',
    # Doces e chocolates (proxy fraco mas existe)
    # Olist não tem categoria específica de chocolate. Usar 'food' como proxy.
    # Presentes (relevante para Mães, Pais, Namorados)
    'perfumery': 'presente_perfume',
    'flowers': 'presente_flores',
    'watches_gifts': 'presente_geral',
    'fashion_bags_accessories': 'presente_acessorio',
    'cool_stuff': 'presente_geral',
    'jewelry': 'presente_joia',
    # Categorias com sazonalidade forte
    'toys': 'brinquedos',
    'baby': 'baby',
    'pet_shop': 'pet',
    'health_beauty': 'beleza',
    # Eventos esportivos (proxy)
    'sports_leisure': 'esportes',
}

df['categoria_modelo'] = df['product_category_name_english'].map(MAPA_CATEGORIA)
df_rel = df[df['categoria_modelo'].notna()].copy()
print(f"  Linhas em categorias relevantes: {len(df_rel):,}")
print(f"  Categorias mapeadas: {df_rel['categoria_modelo'].nunique()}")

# ── 3. Série diária por categoria ──────────────────────────────────────────

serie = (df_rel.groupby(['data', 'categoria_modelo'])
                .agg(qtd=('order_item_id', 'count'),
                     receita=('price', 'sum'))
                .reset_index())
serie.to_csv(OUT / 'serie_diaria_por_categoria.csv',
             index=False, encoding='utf-8')

# ── 4. Cruzar com calendário comercial ──────────────────────────────────────

cal_path = ROOT / 'data' / 'calendario_comercial.csv'
if not cal_path.exists():
    print("✗ data/calendario_comercial.csv não existe. Rode gerar_calendario_comercial.py primeiro.")
    sys.exit(1)

cal = pd.read_csv(cal_path, parse_dates=['data'])
cal['data'] = pd.to_datetime(cal['data']).dt.date
cal_eventos = cal[cal['tipo_evento'].isin(['data_comercial', 'evento_esportivo'])].copy()

# Mapear: cada categoria_modelo → quais eventos ela "responde"
# (lista heurística baseada em significado dos nomes)
RESPONDE_A = {
    'presente_perfume':   ['Dia das Mães', 'Dia dos Namorados', 'Dia das Crianças'],
    'presente_flores':    ['Dia das Mães', 'Dia dos Namorados', 'Dia Internacional da Mulher'],
    'presente_geral':     ['Dia das Mães', 'Dia dos Pais', 'Dia dos Namorados',
                           'Dia das Crianças', 'Véspera de Natal'],
    'presente_acessorio': ['Dia das Mães', 'Dia dos Namorados'],
    'presente_joia':      ['Dia das Mães', 'Dia dos Namorados'],
    'brinquedos':         ['Dia das Crianças', 'Véspera de Natal'],
    'baby':               ['Dia das Mães', 'Dia das Crianças'],
    'beleza':             ['Dia das Mães', 'Dia Internacional da Mulher'],
    'esportes':           ['Dia dos Pais', 'Copa 2022'],
    'bebidas':            ['Véspera de Natal', 'Réveillon', 'Black Friday'],
    'alimentos':          ['Véspera de Natal', 'Réveillon'],
    'pet':                ['Black Friday'],
}

# Black Friday e Réveillon afetam tudo
PARA_TODOS = ['Black Friday', 'Cyber Monday', 'Réveillon', 'Véspera de Natal']

resultados = []
for cat in serie['categoria_modelo'].unique():
    serie_cat = serie[serie['categoria_modelo'] == cat].set_index('data')
    eventos_cat = RESPONDE_A.get(cat, []) + PARA_TODOS

    for _, ev in cal_eventos.iterrows():
        # Só processar eventos que essa categoria responde
        # match parcial (Copa 2022 — Brasil x ... bate com 'Copa 2022')
        if not any(e.lower() in ev['nome_evento'].lower() for e in eventos_cat):
            continue

        data_ev = ev['data']
        pre = max(int(ev['janela_pre_dias']), 7)
        pos = int(ev['janela_pos_dias'])

        # Janela do evento
        inicio_jan = data_ev - timedelta(days=pre)
        fim_jan = data_ev + timedelta(days=pos)
        janela = serie_cat[(serie_cat.index >= inicio_jan) & (serie_cat.index <= fim_jan)]

        # Baseline: mesmo ano, fora da janela (com buffer de 14 dias)
        ano_serie = serie_cat[
            (pd.to_datetime(serie_cat.index).year == data_ev.year)
        ]
        baseline = ano_serie[(ano_serie.index < inicio_jan - timedelta(days=14)) |
                              (ano_serie.index > fim_jan + timedelta(days=14))]

        if len(janela) == 0 or len(baseline) == 0:
            continue

        qtd_janela_dia = janela['qtd'].sum() / max(len(janela), 1)
        qtd_baseline_dia = baseline['qtd'].sum() / max(len(baseline), 1)
        if qtd_baseline_dia < 1:
            continue

        uplift = qtd_janela_dia / qtd_baseline_dia
        resultados.append({
            'categoria': cat,
            'evento': ev['nome_evento'],
            'ano': data_ev.year,
            'data_evento': data_ev,
            'qtd_dia_janela': round(qtd_janela_dia, 2),
            'qtd_dia_baseline': round(qtd_baseline_dia, 2),
            'uplift_medido': round(uplift, 3),
            'uplift_prior_calendario': float(ev['uplift_prior']),
        })

if not resultados:
    print()
    print("⚠ Nenhum evento com dados suficientes encontrado.")
    print("  Olist cobre 2016-2018. Eventos fora dessa janela não foram processados.")
    sys.exit(0)

df_res = pd.DataFrame(resultados)
df_res.to_csv(OUT / 'uplift_por_evento.csv', index=False, encoding='utf-8')

# Agregar por (categoria, evento): média entre anos
ev_normalizado = df_res.copy()
# Para "Copa 2022 — Brasil x ..." colapsar para "Copa 2022"
ev_normalizado['evento_base'] = (ev_normalizado['evento']
                                  .str.split(' — ').str[0])
agg = (ev_normalizado.groupby(['categoria', 'evento_base'])
                       .agg(uplift_medio=('uplift_medido', 'mean'),
                            uplift_std=('uplift_medido', 'std'),
                            n_anos=('uplift_medido', 'count'),
                            uplift_prior=('uplift_prior_calendario', 'mean'))
                       .reset_index()
                       .sort_values('uplift_medio', ascending=False))
agg['uplift_medio'] = agg['uplift_medio'].round(3)
agg['uplift_std'] = agg['uplift_std'].round(3)
agg['uplift_prior'] = agg['uplift_prior'].round(3)
agg.to_csv(OUT / 'uplift_agregado.csv', index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print(f"✓ {len(df_res)} medições brutas, {len(agg)} agregadas por (categoria, evento)")
print()
print("Top 20 uplifts medidos (Olist, BR, 2016-2018):")
print()
print(f"{'Categoria':<22s} {'Evento':<30s} {'Medido':>7s} {'Prior':>7s} {'N':>3s}")
print('-' * 80)
for _, row in agg.head(20).iterrows():
    print(f"{row['categoria']:<22s} {row['evento_base']:<30s} "
          f"{row['uplift_medio']:>6.2f}× "
          f"{row['uplift_prior']:>6.2f}× "
          f"{row['n_anos']:>3d}")

print()
print("Comparação com prior do calendário (medido vs prior):")
agg['diff'] = (agg['uplift_medio'] - agg['uplift_prior']).abs()
divergentes = agg[agg['diff'] > 0.5].head(10)
if len(divergentes) > 0:
    print("Divergências relevantes:")
    for _, row in divergentes.iterrows():
        marca = '↑' if row['uplift_medio'] > row['uplift_prior'] else '↓'
        print(f"  {marca} {row['categoria']:<20s} {row['evento_base']:<25s} "
              f"medido {row['uplift_medio']:.2f} vs prior {row['uplift_prior']:.2f}")
