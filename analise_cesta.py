"""Market Basket Analysis — análise de cesta de compras.

Pega cupom fiscal (transação × itens) e identifica produtos vendidos
juntos com frequência superior ao acaso. Substitui os PARES_COMBO
heurísticos do V10 por combos validados pela realidade do posto.

Métricas:
  - Support: % de transações que contêm o par
  - Confidence: P(B | A) = quem compra A também compra B
  - Lift: razão entre confidence e P(B) base.
          Lift > 1.5 = associação significativa.

Schema esperado do CSV de entrada (cupom fiscal):

    transacao_id, data, hora, sku, quantidade, valor_unitario, valor_total

    Onde transacao_id identifica unicamente cada cupom (= conjunto de itens
    comprados juntos). Outras colunas opcionais.

Uso:
    python analise_cesta.py data/cupom_fiscal.csv

Se o arquivo não existir, gera dados mock para validar que o script funciona.

Saídas:
  results/combos_validados.csv         (top combos por lift)
  results/combos_heatmap.png           (mapa de calor das associações)
  results/comparacao_pares_v10.csv     (combos atuais vs sugeridos)
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from mlxtend.frequent_patterns import apriori, association_rules

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

# ── Configuração ────────────────────────────────────────────────────────────

ARQUIVO_CUPOM = sys.argv[1] if len(sys.argv) > 1 else 'data/cupom_fiscal.csv'
MIN_SUPPORT = 0.005   # par precisa aparecer em ≥0.5% das transações
MIN_LIFT = 1.3        # associação ≥30% acima do acaso
TOP_N = 30

# PARES_COMBO atuais do V10 (heurística)
PARES_COMBO_V10 = {
    'energetico': 'refrigerante',
    'gelo': 'agua',
    'refrigerante': 'sorvete',
    'agua': 'energetico',
    'cerveja': 'refrigerante',
    'sorvete': 'agua',
}

# ── Carrega cupom ou gera mock ──────────────────────────────────────────────

caminho = ROOT / ARQUIVO_CUPOM if not Path(ARQUIVO_CUPOM).is_absolute() else Path(ARQUIVO_CUPOM)

if caminho.exists():
    print(f"Carregando cupom fiscal de {caminho}")
    df = pd.read_csv(caminho)
    USANDO_MOCK = False
else:
    print(f"⚠ Arquivo {caminho} não existe ainda — gerando dados MOCK para validar pipeline")
    print(f"  Quando o cupom fiscal real chegar, salve em {caminho} e rode de novo.")
    USANDO_MOCK = True
    # Mock: simula 2000 transações com padrões plausíveis
    rng = np.random.default_rng(42)
    skus = ['cerveja_brahma_350ml', 'cerveja_heineken_330ml',
            'refrigerante_coca_2l', 'refrigerante_guarana_2l',
            'agua_mineral_500ml', 'agua_mineral_1l',
            'energetico_redbull_250ml', 'energetico_monster_473ml',
            'gelo_5kg', 'sorvete_kibon_individual',
            'chocolate_lacta_90g', 'chocolate_kitkat_45g',
            'salgadinho_doritos_92g', 'salgadinho_ruffles_84g',
            'cafe_pilao_500g', 'cigarro_marlboro']
    # Pares com afinidade alta (lift esperado > 2)
    afinidades = {
        ('cerveja_brahma_350ml', 'salgadinho_doritos_92g'): 0.6,
        ('cerveja_heineken_330ml', 'salgadinho_ruffles_84g'): 0.55,
        ('cerveja_brahma_350ml', 'gelo_5kg'): 0.45,
        ('refrigerante_coca_2l', 'salgadinho_doritos_92g'): 0.35,
        ('energetico_redbull_250ml', 'chocolate_kitkat_45g'): 0.4,
        ('cafe_pilao_500g', 'chocolate_lacta_90g'): 0.3,
        ('sorvete_kibon_individual', 'refrigerante_coca_2l'): 0.4,
    }
    transacoes = []
    for i in range(2000):
        cesta = set()
        # 1-4 itens base aleatórios
        base = rng.choice(skus, size=rng.integers(1, 5), replace=False)
        cesta.update(base)
        # Aplicar afinidades
        for (a, b), prob in afinidades.items():
            if a in cesta and rng.random() < prob:
                cesta.add(b)
        for sku in cesta:
            transacoes.append({
                'transacao_id': f'T{i:04d}',
                'sku': sku,
                'quantidade': int(rng.integers(1, 3)),
            })
    df = pd.DataFrame(transacoes)

# ── Pivotear: 1 linha por transação, 1 coluna por SKU, valor binário ───────

if 'transacao_id' not in df.columns or 'sku' not in df.columns:
    print(f"✗ CSV precisa ter colunas 'transacao_id' e 'sku'. Encontradas: {list(df.columns)}")
    sys.exit(1)

print(f"  Transações: {df['transacao_id'].nunique():,}")
print(f"  SKUs distintos: {df['sku'].nunique():,}")
print(f"  Linhas (transação × item): {len(df):,}")

# Filtrar SKUs com volume mínimo (presentes em ≥1% das transações)
n_trans = df['transacao_id'].nunique()
sku_counts = df.groupby('sku')['transacao_id'].nunique()
skus_relevantes = sku_counts[sku_counts >= n_trans * 0.01].index.tolist()
print(f"  SKUs com support ≥1% (entrarão na análise): {len(skus_relevantes)}")

df_f = df[df['sku'].isin(skus_relevantes)].copy()
# Encodar SKU como int para contornar bug do mlxtend novo com numpy strings
sku_to_id = {sku: i for i, sku in enumerate(sorted(df_f['sku'].unique()))}
id_to_sku = {i: sku for sku, i in sku_to_id.items()}
df_f['sku_id'] = df_f['sku'].map(sku_to_id)
matriz = df_f.groupby(['transacao_id', 'sku_id']).size().unstack(fill_value=0)
matriz = (matriz > 0).astype(bool)

# ── Apriori para itemsets frequentes ───────────────────────────────────────

print()
print(f"Rodando Apriori (min_support={MIN_SUPPORT})...")
itemsets = apriori(matriz, min_support=MIN_SUPPORT,
                    use_colnames=True, max_len=3)
print(f"  Itemsets frequentes encontrados: {len(itemsets)}")

if len(itemsets) == 0:
    print("✗ Nenhum itemset frequente. Reduzir MIN_SUPPORT ou aumentar dados.")
    sys.exit(1)

# Filtrar para itemsets de 2 itens (pares)
itemsets_pares = itemsets[itemsets['itemsets'].apply(len) == 2].copy()
print(f"  Pares frequentes: {len(itemsets_pares)}")

# ── Gerar regras de associação ─────────────────────────────────────────────

try:
    regras = association_rules(itemsets, metric='lift',
                                min_threshold=MIN_LIFT, num_itemsets=len(itemsets))
except TypeError:
    # Versão antiga do mlxtend
    regras = association_rules(itemsets, metric='lift', min_threshold=MIN_LIFT)

# Filtrar regras com 1→1 (pares simples)
regras = regras[(regras['antecedents'].apply(len) == 1) &
                (regras['consequents'].apply(len) == 1)].copy()
regras['antecedent_sku'] = regras['antecedents'].apply(
    lambda x: id_to_sku.get(list(x)[0], str(list(x)[0])))
regras['consequent_sku'] = regras['consequents'].apply(
    lambda x: id_to_sku.get(list(x)[0], str(list(x)[0])))
regras = regras[['antecedent_sku', 'consequent_sku', 'support',
                  'confidence', 'lift']]
regras = regras.sort_values('lift', ascending=False).reset_index(drop=True)
regras['support'] = regras['support'].round(4)
regras['confidence'] = regras['confidence'].round(3)
regras['lift'] = regras['lift'].round(3)

print(f"  Regras com lift ≥ {MIN_LIFT}: {len(regras)}")

# Deduplicar pares (A→B e B→A são o mesmo combo na prática — fica o de maior lift)
pares_unicos = set()
mask = []
for _, row in regras.iterrows():
    par = tuple(sorted([row['antecedent_sku'], row['consequent_sku']]))
    if par in pares_unicos:
        mask.append(False)
    else:
        pares_unicos.add(par)
        mask.append(True)
regras_unicas = regras[mask].head(TOP_N).reset_index(drop=True)

regras_unicas.to_csv(RESULTS / 'combos_validados.csv', index=False, encoding='utf-8')

# ── Comparar com PARES_COMBO_V10 ───────────────────────────────────────────

if not USANDO_MOCK:
    comparacao = []
    for principal, complementar in PARES_COMBO_V10.items():
        # Procurar regra no dataset que envolva esse par
        regra_match = regras[
            ((regras['antecedent_sku'].str.contains(principal, case=False)) &
             (regras['consequent_sku'].str.contains(complementar, case=False))) |
            ((regras['antecedent_sku'].str.contains(complementar, case=False)) &
             (regras['consequent_sku'].str.contains(principal, case=False)))
        ]
        if len(regra_match) > 0:
            r = regra_match.iloc[0]
            comparacao.append({
                'par_v10_principal': principal,
                'par_v10_complementar': complementar,
                'encontrado_nos_dados': True,
                'lift_medido': r['lift'],
                'support_medido': r['support'],
                'recomendacao': 'manter' if r['lift'] >= 1.5 else 'considerar_substituir',
            })
        else:
            comparacao.append({
                'par_v10_principal': principal,
                'par_v10_complementar': complementar,
                'encontrado_nos_dados': False,
                'lift_medido': None,
                'support_medido': None,
                'recomendacao': 'substituir_par_nao_validado',
            })
    pd.DataFrame(comparacao).to_csv(
        RESULTS / 'comparacao_pares_v10.csv', index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
if USANDO_MOCK:
    print("=" * 60)
    print("RESULTADO COM DADOS MOCK (validando pipeline)")
    print("=" * 60)
else:
    print("=" * 60)
    print(f"RESULTADO COM CUPOM FISCAL REAL")
    print("=" * 60)
print()
print(f"Top {min(TOP_N, len(regras_unicas))} combos com maior lift:")
print()
print(f"{'Produto A':<35s} {'Produto B':<35s} {'Lift':>6s} {'Sup':>6s}")
print('-' * 90)
for _, row in regras_unicas.head(TOP_N).iterrows():
    print(f"{row['antecedent_sku']:<35s} {row['consequent_sku']:<35s} "
          f"{row['lift']:>6.2f} {row['support']*100:>5.1f}%")

print()
print(f"✓ Salvo em results/combos_validados.csv")
if not USANDO_MOCK:
    print(f"✓ Comparação com V10 em results/comparacao_pares_v10.csv")
    print()
    n_validados = sum(1 for c in comparacao if c['recomendacao'] == 'manter')
    print(f"  Pares V10 validados:    {n_validados}/{len(PARES_COMBO_V10)}")
    n_substituir = sum(1 for c in comparacao
                        if c['recomendacao'] != 'manter')
    print(f"  Pares V10 a substituir: {n_substituir}/{len(PARES_COMBO_V10)}")
else:
    print()
    print("Para rodar com dados reais:")
    print(f"  1. Salvar cupom fiscal em data/cupom_fiscal.csv com colunas")
    print(f"     transacao_id, sku (+ outras opcionais)")
    print(f"  2. Rodar: python analise_cesta.py")
