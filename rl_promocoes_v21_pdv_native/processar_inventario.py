"""Processa inventario2022-26.xlsx do posto real.

Extrai por categoria agregada:
- preço custo médio (dos produtos do inventário)
- taxa de perda real (Qtd contábil vs física)
- frequência de aparição (proxy de movimento)

Saída: data_sintetica/inventario_processado.csv
"""
import io
import sys
from pathlib import Path

import pandas as pd


# Mapeamento de 87 categorias do posto → 20 categorias agregadas PDV
MAPEAMENTO_CATEGORIA = {
    # Bebidas frias
    'AGUA': 'agua', 'AGUA SABORIZADA': 'agua', 'ÁGUA DE COCO': 'agua',
    'REFRIGERANTE': 'refrigerante',
    'SUCO': 'suco',
    'ISOTÔNICO': 'isotonico',
    'ENERGÉTICO': 'energetico',
    'CHÁ': 'cha_pronto',
    # Bebidas alcoólicas (cerveja agregada)
    'CERVEJA AMBEV': 'cerveja', 'CERVEJA FEMSA': 'cerveja',
    'CERVEJA ITAIPAVA': 'cerveja', 'CERVEJA ESPECIAIS': 'cerveja',
    'DESTILADOS DIVERSOS': 'destilados', 'AGUARDENTES': 'destilados',
    'VODKA': 'destilados',
    'WHISK': 'whisky',
    'VINHO': 'vinho',
    # Comida/refeição quick
    'PADARIA': 'padaria', 'BOLO': 'padaria',
    'SANDUÍCHE': 'sanduiche', 'SANDUICHES SELECT': 'sanduiche',
    'SALGADO ASSADO/FRITO': 'sanduiche',
    'CONGELADOS': 'congelados',
    'CAFÉ': 'cafe',
    'ACHOCOLATADO': 'cafe', 'INSTANTANEO': 'cafe',
    'CEREAIS': 'biscoito',
    # Chocolate caixa (premium, ferrero, lacta, garoto, nestle, m&m)
    'CHOCOLATE FERRERO': 'chocolate_caixa',
    'CHOCOLATE LACTA': 'chocolate_caixa',
    'CHOCOLATE GAROTO': 'chocolate_caixa',
    'CHOCOLATE NESTLE': 'chocolate_caixa',
    'CHOCOLATE M&M (MARS)': 'chocolate_caixa',
    # Chocolate impulso (unidade pequena)
    'CHOCOLATE ARCOR': 'chocolate_unit',
    'CHOCOLATE DIVERSOS': 'chocolate_unit',
    # Doces de balcão
    'BALA': 'doce_balcao', 'BALA FINI': 'doce_balcao',
    'MENTOS': 'doce_balcao', 'DROPS': 'doce_balcao',
    'PASTILHAS': 'doce_balcao', 'CHICLETE': 'doce_balcao',
    'DOCES DIVERSOS': 'doce_balcao',
    # Biscoito
    'BISCOITO': 'biscoito',
    # Snack salgado
    'SNACK ELMA CHIPS': 'snack_salgado',
    'SNACK PRINGLES': 'snack_salgado',
    'SNACK TORCIDA': 'snack_salgado',
    'SNACK DIVERSOS': 'snack_salgado',
    'AMENDOIN/NOZES': 'snack_salgado',
    'PIPOCA': 'snack_salgado',
    # Sorvete
    'SORVETE KIBON': 'sorvete',
    'SORVETE JUNDIÁ': 'sorvete',
    'SORVETE PERFETTO': 'sorvete',
    # Lácteo
    'BEBIDA LÁCTEA': 'lacteo', 'IOGURTE': 'lacteo',
    # Utilitários
    'GELO': 'gelo',
    'CARVÃO': 'carvao',
    # PROIBIDOS (cigarros)
    'JTI': 'cigarro_jti',
    'PHILIP MORRIS': 'cigarro_philip_morris',
    'SOUZA CRUZ': 'cigarro_souza_cruz',
    'CIGARRILHAS': 'cigarrilha',
    'SEDA': 'cigarrilha',
    # Resto = não conveniência (loja automotiva) → ignorar
}


def processar(inv_path, out_path):
    print(f"Lendo inventário: {inv_path}")
    df = pd.read_excel(inv_path, header=None)

    # Coluna 1 tem "Classificação produto: X", coluna 3 tem nome do produto,
    # 4 = qtd física, 5 = qtd contábil, 7 = preço custo
    registros = []
    categoria_atual = None
    for _, row in df.iterrows():
        v1 = str(row[1]) if pd.notna(row[1]) else ''
        if 'Classificação produto:' in v1:
            categoria_atual = v1.replace('Classificação produto:', '').strip()
            continue
        # Linha de produto
        if categoria_atual and pd.notna(row[3]) and pd.notna(row[7]):
            try:
                produto = str(row[3]).strip()
                qtd_fis = float(row[4]) if pd.notna(row[4]) else 0
                qtd_cont = float(row[5]) if pd.notna(row[5]) else 0
                custo = float(row[7])
                registros.append({
                    'categoria_orig': categoria_atual,
                    'categoria': MAPEAMENTO_CATEGORIA.get(categoria_atual, '_outro'),
                    'produto': produto,
                    'qtd_fis': qtd_fis,
                    'qtd_cont': qtd_cont,
                    'diff': qtd_fis - qtd_cont,
                    'custo': custo,
                })
            except (ValueError, TypeError):
                pass

    df_inv = pd.DataFrame(registros)
    print(f"Total de registros: {len(df_inv)}")

    # Agrega por categoria
    agg = df_inv[df_inv['categoria'] != '_outro'].groupby('categoria').agg(
        n_skus=('produto', 'count'),
        custo_medio=('custo', 'median'),
        qtd_fis_total=('qtd_fis', 'sum'),
        qtd_cont_total=('qtd_cont', 'sum'),
        diff_total=('diff', 'sum'),
    ).reset_index()

    # Taxa de perda (sobra - perda)
    agg['taxa_diff_pct'] = (agg['diff_total'] / agg['qtd_cont_total'].replace(0, 1)) * 100

    print(f"\nCategorias agregadas:")
    print(agg.to_string(index=False))

    agg.to_csv(out_path, index=False, encoding='utf-8')
    print(f"\n✓ Salvo: {out_path}")
    return agg


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    HERE = Path(__file__).parent
    PROJ = HERE.parent
    processar(
        PROJ / 'data' / 'inventario2022-26.xlsx',
        HERE / 'data_sintetica' / 'inventario_processado.csv',
    )
