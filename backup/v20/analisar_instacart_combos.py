"""Analisa dataset Instacart (archive 8) para extrair combos reais via Apriori.

Mapeia categorias Instacart (aisles + departments) → nossas 20 categorias do posto.
Roda Apriori para descobrir top combinações reais com lift > 1.5.

Saída: data/priors_externos/instacart/combos_validados.csv
"""
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RAW = ROOT / 'data' / 'raw_novos' / 'archive(8)'
OUT = ROOT / 'data' / 'priors_externos' / 'instacart'
OUT.mkdir(parents=True, exist_ok=True)

# ── Mapeamento aisle_id Instacart → nossa categoria ──────────────────────

# Inspeção manual de aisles relevantes → nossas 20 categorias
# Pode não ser 1-1 (Instacart tem chocolate em "candy chocolate" + "energy granola bars")
MAPA_INSTACART = {
    # cigarro: Instacart não vende, skipping
    'energy granola bars': 'snack',
    'crackers': 'snack',
    'chips pretzels': 'snack',
    'cookies cakes': 'biscoito',
    'candy chocolate': 'chocolate_premium',  # premium
    'ice cream ice': 'sorvete',
    'frozen meals': 'padaria',  # pão congelado, semelhança
    'bread': 'padaria',
    'breakfast bakery': 'padaria',
    'soft drinks': 'refrigerante',
    'juice nectars': 'suco',
    'water seltzer sparkling water': 'agua',
    'energy sports drinks': 'isotonico',
    'beer wines spirits': 'cerveja',  # agrupado
    'red wines': 'vinho',
    'white wines': 'vinho',
    'specialty wines champagnes': 'vinho',
    'spirits': 'destilados',
    'coffee': 'cafe',
    'tea': 'cafe',  # bebida quente
    'milk': 'suco',  # similar bebida fria
}


def main():
    print("Carregando Instacart…")
    aisles = pd.read_csv(RAW / 'aisles.csv')
    products = pd.read_csv(RAW / 'products.csv')
    departments = pd.read_csv(RAW / 'departments.csv')

    print(f"  aisles: {len(aisles)} | products: {len(products):,} | departments: {len(departments)}")

    # Merge product → aisle name
    products = products.merge(aisles, on='aisle_id').merge(departments, on='department_id')

    # Mapear products → nossas categorias
    products['cat_modelo'] = products['aisle'].map(MAPA_INSTACART)
    n_mapeados = products['cat_modelo'].notna().sum()
    print(f"  Produtos mapeados para nossas 20 categorias: {n_mapeados:,} de {len(products):,}")
    print(f"  Distribuição:")
    print(products['cat_modelo'].value_counts().to_string())

    # Map product_id → cat_modelo
    map_prod_cat = dict(zip(products['product_id'], products['cat_modelo']))

    # Carregar uma AMOSTRA de orders (32M é demais)
    print("\nCarregando order_products__prior (sample 500k orders)…")
    op = pd.read_csv(RAW / 'order_products__prior.csv',
                      nrows=2_000_000)  # 2M linhas ~ 60k orders
    print(f"  {len(op):,} linhas carregadas")

    # Filtrar apenas produtos mapeados
    op['cat'] = op['product_id'].map(map_prod_cat)
    op = op.dropna(subset=['cat'])
    print(f"  {len(op):,} linhas em categorias do nosso modelo")

    # Agrupar por order_id em conjuntos de categorias (transações)
    transacoes = op.groupby('order_id')['cat'].apply(lambda x: tuple(sorted(set(x))))
    transacoes = transacoes[transacoes.apply(len) >= 2]
    print(f"  {len(transacoes):,} transações com ≥2 categorias diferentes")

    # ── Apriori manual (rápido para nosso N=20 categorias) ─────────────

    # Contagens de pares
    print("\nContando pares de categorias…")
    pares = Counter()
    singles = Counter()
    n_total = len(transacoes)
    for t in transacoes:
        for c in t:
            singles[c] += 1
        for i, c1 in enumerate(t):
            for c2 in t[i+1:]:
                if c1 != c2:
                    pares[tuple(sorted([c1, c2]))] += 1

    # Calcular support, confidence, lift
    resultados = []
    for (a, b), cnt_ab in pares.items():
        support_ab = cnt_ab / n_total
        support_a = singles[a] / n_total
        support_b = singles[b] / n_total
        confidence_a_b = cnt_ab / singles[a]
        confidence_b_a = cnt_ab / singles[b]
        lift = support_ab / (support_a * support_b)
        resultados.append({
            'cat_a': a, 'cat_b': b,
            'n_transacoes': cnt_ab,
            'support': round(support_ab, 4),
            'confidence_a_b': round(confidence_a_b, 3),
            'confidence_b_a': round(confidence_b_a, 3),
            'lift': round(lift, 3),
        })

    df = pd.DataFrame(resultados)
    df = df.sort_values('lift', ascending=False)
    df.to_csv(OUT / 'combos_instacart.csv', index=False, encoding='utf-8')

    print()
    print("=" * 80)
    print(f"TOP 20 COMBINAÇÕES (lift mais alto)")
    print("=" * 80)
    print(df.head(20).to_string(index=False))

    print()
    print("=" * 80)
    print(f"COMBOS COM LIFT > 1.5 (clientes compram juntos significativamente)")
    print("=" * 80)
    fortes = df[df['lift'] > 1.5]
    print(fortes.to_string(index=False))

    print()
    print(f"✓ Salvo em {OUT / 'combos_instacart.csv'}")

    # ── Sugestão de update na harmonia_combo ──────────────────────────────

    print()
    print("=" * 80)
    print("ATUALIZAÇÕES SUGERIDAS PARA harmonia_combo no calibrar_v2.py")
    print("=" * 80)
    print("# Lift > 2.0 → harmonia = 2.5 (forte)")
    print("# Lift 1.5-2.0 → harmonia = 1.8 (média-forte)")
    print("# Lift 1.2-1.5 → harmonia = 1.3 (média)")
    print("# Lift < 1.0 → harmonia = 0.8 (antagônica)")
    print()
    for _, r in fortes.head(15).iterrows():
        if r['lift'] > 2.0:
            h = 2.5
        elif r['lift'] > 1.5:
            h = 1.8
        else:
            h = 1.3
        print(f"    ('{r['cat_a']}', '{r['cat_b']}'): {h},  # lift {r['lift']} (Instacart)")


if __name__ == '__main__':
    main()
