"""Filtra catálogo removendo categorias não-conveniência.

Posto vende dois tipos de produto:
- CONVENIÊNCIA: bebida, snack, doce, chocolate, sorvete, cigarro, café
- AUTOMOTIVO: óleo, filtro, aditivo, palheta, limpa-pneu

O modelo de RL para promoção de loja de conveniência deve focar só
no primeiro grupo — automotivo tem dinâmica completamente diferente
(compra por reposição, baixa frequência, sem sazonalidade comercial,
sem validade curta).

Saídas:
  data/categorias_classificadas.csv      (tabela de classificação manual — pode revisar)
  data/catalogo_conveniencia.csv         (só SKUs de conveniência)
  results/categorias_conveniencia.csv    (estatísticas)
  results/analise_estoque_parado_conveniencia.csv  (diagnóstico filtrado)
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

# ── Classificação manual das 77 categorias ─────────────────────────────────
# Tipo: 'conveniencia' | 'automotivo' | 'revisar'
# 'revisar' = caso cinza — incluído por padrão mas marcado para Vinicius validar

CLASSIFICACAO = {
    # ── Bebidas alcoólicas ─────────────────────────────────────────────
    'CERVEJA AMBEV':            ('conveniencia', 'Cerveja mainstream'),
    'CERVEJA ESPECIAIS':        ('conveniencia', 'Cerveja premium/artesanal'),
    'CERVEJA FEMSA':            ('conveniencia', 'Cerveja Heineken/Sol/Amstel'),
    'CERVEJA ITAIPAVA':         ('conveniencia', 'Cerveja popular'),
    'DESTILADOS DIVERSOS':      ('conveniencia', 'Bebida alcoólica'),
    'AGUARDENTES':              ('conveniencia', 'Cachaça, pinga'),
    'WHISK':                    ('conveniencia', 'Whisky'),
    'VODKA':                    ('conveniencia', 'Vodka'),
    'VINHO':                    ('conveniencia', 'Vinho — produto chave para Dia dos Namorados/Mães'),

    # ── Bebidas não alcoólicas ─────────────────────────────────────────
    'REFRIGERANTE':             ('conveniencia', 'Bebida'),
    'SUCO':                     ('conveniencia', 'Bebida'),
    'AGUA':                     ('conveniencia', 'Bebida'),
    'AGUA SABORIZADA':          ('conveniencia', 'Bebida'),
    'ÁGUA DE COCO':             ('conveniencia', 'Bebida'),
    'ISOTÔNICO':                ('conveniencia', 'Bebida esportiva'),
    'ENERGÉTICO':               ('conveniencia', 'Bebida'),
    'BEBIDA LÁCTEA':            ('conveniencia', 'Bebida'),
    'CAFÉ':                     ('conveniencia', 'Bebida quente'),
    'ACHOCOLATADO':             ('conveniencia', 'Bebida (chocolate)'),
    'NESCAFÉ BEBIDAS':          ('conveniencia', 'Bebida quente pronta'),
    'CHÁ':                      ('conveniencia', 'Bebida'),

    # ── Doces e chocolates ─────────────────────────────────────────────
    'CHOCOLATE LACTA':          ('conveniencia', 'Chocolate — Dia dos Namorados/Páscoa'),
    'CHOCOLATE NESTLE':         ('conveniencia', 'Chocolate'),
    'CHOCOLATE GAROTO':         ('conveniencia', 'Chocolate'),
    'CHOCOLATE FERRERO':        ('conveniencia', 'Chocolate premium'),
    'CHOCOLATE M&M (MARS)':     ('conveniencia', 'Chocolate'),
    'CHOCOLATE ARCOR':          ('conveniencia', 'Chocolate'),
    'CHOCOLATE DIVERSOS':       ('conveniencia', 'Chocolate'),
    'BALA':                     ('conveniencia', 'Doce'),
    'BALA FINI':                ('conveniencia', 'Doce'),
    'CHICLETE':                 ('conveniencia', 'Doce'),
    'MENTOS':                   ('conveniencia', 'Doce/balas'),
    'PASTILHAS':                ('conveniencia', 'Doce'),
    'DOCES DIVERSOS':           ('conveniencia', 'Doce'),

    # ── Snacks ─────────────────────────────────────────────────────────
    'BISCOITO':                 ('conveniencia', 'Snack'),
    'SNACK ELMA CHIPS':         ('conveniencia', 'Snack'),
    'SNACK PRINGLES':           ('conveniencia', 'Snack'),
    'SNACK TORCIDA':            ('conveniencia', 'Snack'),
    'SNACK DIVERSOS':           ('conveniencia', 'Snack'),
    'AMENDOIN/NOZES':           ('conveniencia', 'Snack'),
    'CEREAIS':                  ('conveniencia', 'Cereal/granola'),
    'PIPOCA':                   ('conveniencia', 'Snack'),

    # ── Sorvetes ───────────────────────────────────────────────────────
    'SORVETE KIBON':            ('conveniencia', 'Sorvete — alta sazonalidade verão'),
    'SORVETE JUNDIÁ':           ('conveniencia', 'Sorvete'),
    'SORVETE PERFETTO':         ('conveniencia', 'Sorvete premium'),

    # ── Padaria / alimentos prontos ───────────────────────────────────
    'PADARIA':                  ('conveniencia', 'Pão, doce de balcão'),
    'SANDUÍCHE':                ('conveniencia', 'Comida pronta'),
    'SALGADO ASSADO/FRITO':     ('conveniencia', 'Comida pronta'),
    'BOLO':                     ('conveniencia', 'Comida pronta'),
    'IOGURTE':                  ('conveniencia', 'Laticínio'),
    'CONGELADOS':               ('conveniencia', 'Comida congelada'),
    'MERCEARIA ALIMENTICIA':    ('conveniencia', 'Alimento básico'),

    # ── Tabacaria ──────────────────────────────────────────────────────
    'SOUZA CRUZ':               ('conveniencia', 'Cigarro — margem regulada baixa (~11%)'),
    'JTI':                      ('conveniencia', 'Cigarro Japan Tobacco'),
    'PHILIP MORRIS':            ('conveniencia', 'Cigarro Marlboro'),
    'CIGARRILHAS':              ('conveniencia', 'Tabaco'),
    'ISQUEIROS':                ('conveniencia', 'Acessório fumante'),

    # ── Acessórios e impulso de balcão ────────────────────────────────
    'HIGIENE PESSOAL':          ('conveniencia', 'Compra de urgência'),
    'PILHAS':                   ('conveniencia', 'Compra de impulso'),
    'CARVÃO':                   ('conveniencia', 'Provavelmente churrasco'),

    # ── AUTOMOTIVO (excluir do modelo) ─────────────────────────────────
    'LUBRIFICANTE LUBRAX':      ('automotivo', 'Óleo motor — dinâmica diferente'),
    'FILTRO DE ÓLEO':           ('automotivo', 'Filtro motor'),
    'FILTRO DE AR':             ('automotivo', 'Filtro motor'),
    'FILTRO DE AR - ANTIGO':    ('automotivo', 'Filtro motor (descontinuado)'),
    'ADITIVOS STP':             ('automotivo', 'Aditivo combustível/sistema'),
    'PALHETA':                  ('automotivo', 'Palheta limpador para-brisa'),
    'LIMPA PNEU':               ('automotivo', 'Produto para carro'),
    'LIMPA VIDROS':             ('automotivo', 'Produto para carro'),
    'ARRUELA':                  ('automotivo', 'Peça mecânica'),
    'ODORIZANTE':               ('automotivo', 'Provavelmente odorizante de carro'),

    # ── REVISAR (casos cinza, incluídos por padrão) ───────────────────
    'ACESSORIOS CELULAR':       ('revisar', 'Cabo, carregador, suporte veicular — incerto se promove'),
    'UTILIDADES DOMESTICAS':    ('revisar', 'Vago — pode ser sabão, faxina'),
    'ACESSÓRIOS DIVERSOS':      ('revisar', '1 SKU, R$ 6.99 — vago'),
    'BATERIA':                  ('revisar', '1 SKU, R$ 3.99 — provavelmente pilha/relógio'),
    'PEGBOARD':                 ('automotivo', 'Display/gancho do posto, NÃO produto à venda'),
    'EXTINTOR':                 ('automotivo', 'Equipamento do posto, não para venda'),
    'MERCEARIA MAT. LIMPEZA':   ('revisar', 'Não é conveniência clássica'),
}

# ── Salva tabela de classificação ──────────────────────────────────────────

tabela_class = pd.DataFrame([
    {'categoria': cat, 'tipo': tipo, 'motivo': motivo}
    for cat, (tipo, motivo) in CLASSIFICACAO.items()
]).sort_values(['tipo', 'categoria']).reset_index(drop=True)
tabela_class.to_csv(DATA / 'categorias_classificadas.csv',
                     index=False, encoding='utf-8')

# ── Carrega catálogo completo ──────────────────────────────────────────────

catalogo = pd.read_csv(DATA / 'catalogo_inferido.csv')
print(f"Catálogo total: {len(catalogo):,} SKUs em {catalogo['categoria'].nunique()} categorias")

# ── Verificar se há categorias não classificadas ───────────────────────────

cats_catalogo = set(catalogo['categoria'].unique())
cats_classificadas = set(CLASSIFICACAO.keys())
nao_classificadas = cats_catalogo - cats_classificadas
extras_na_classificacao = cats_classificadas - cats_catalogo

if nao_classificadas:
    print(f"\n⚠ {len(nao_classificadas)} categorias do catálogo NÃO classificadas:")
    for c in sorted(nao_classificadas):
        print(f"  - {c}")
    print("  (vão ser tratadas como 'revisar' por segurança)")
    for c in nao_classificadas:
        CLASSIFICACAO[c] = ('revisar', 'Não classificada explicitamente')

if extras_na_classificacao:
    print(f"\nℹ {len(extras_na_classificacao)} categorias na classificação que NÃO aparecem no catálogo:")
    for c in sorted(extras_na_classificacao):
        print(f"  - {c}")

# ── Aplica filtro ───────────────────────────────────────────────────────────

catalogo['tipo'] = catalogo['categoria'].map(
    lambda c: CLASSIFICACAO.get(c, ('revisar', ''))[0])

# Mantém conveniência + revisar (revisar = cinza, deixa pra Vinicius validar)
catalogo_conv = catalogo[catalogo['tipo'].isin(['conveniencia', 'revisar'])].copy()
catalogo_conv.to_csv(DATA / 'catalogo_conveniencia.csv',
                      index=False, encoding='utf-8')

# Estatísticas por categoria filtrada
cat_stats_conv = (catalogo_conv.groupby('categoria')
                                .agg(n_skus=('sku', 'count'),
                                     preco_medio=('preco_venda_medio', 'mean'),
                                     margem_pct_media=('margem_pct', 'mean'),
                                     tipo=('tipo', 'first'))
                                .reset_index()
                                .sort_values('n_skus', ascending=False))
cat_stats_conv['preco_medio'] = cat_stats_conv['preco_medio'].round(2)
cat_stats_conv['margem_pct_media'] = cat_stats_conv['margem_pct_media'].round(1)
cat_stats_conv.to_csv(RESULTS / 'categorias_conveniencia.csv',
                       index=False, encoding='utf-8')

# Versão filtrada da análise de estoque parado
diag = pd.read_csv(RESULTS / 'analise_estoque_parado.csv')
diag_conv = diag[diag['categoria'].map(
    lambda c: CLASSIFICACAO.get(c, ('revisar', ''))[0]
) != 'automotivo'].copy()
diag_conv.to_csv(RESULTS / 'analise_estoque_parado_conveniencia.csv',
                  index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

contagem = pd.Series([t for t, _ in CLASSIFICACAO.values()]).value_counts()
skus_por_tipo = catalogo.groupby('tipo').agg(
    n_categorias=('categoria', 'nunique'),
    n_skus=('sku', 'count'),
).reset_index()

print()
print("="*70)
print("CLASSIFICAÇÃO DAS CATEGORIAS")
print("="*70)
print()
for tipo in ['conveniencia', 'revisar', 'automotivo']:
    cats = [c for c, (t, _) in CLASSIFICACAO.items() if t == tipo and c in cats_catalogo]
    n_skus = catalogo[catalogo['categoria'].isin(cats)]['sku'].count()
    pct = n_skus / len(catalogo) * 100
    emoji = {'conveniencia': '✓', 'revisar': '?', 'automotivo': '✗'}[tipo]
    print(f"  {emoji} {tipo.upper():<14s} {len(cats):>3d} categorias  "
          f"{n_skus:>4d} SKUs ({pct:.1f}%)")

print()
print("Categorias REVISAR (verificar com Vinicius):")
for cat, (tipo, motivo) in sorted(CLASSIFICACAO.items()):
    if tipo == 'revisar' and cat in cats_catalogo:
        n = (catalogo['categoria'] == cat).sum()
        print(f"  - {cat:<30s} {n:>3d} SKUs — {motivo}")

print()
print("Categorias AUTOMOTIVO removidas:")
for cat, (tipo, motivo) in sorted(CLASSIFICACAO.items()):
    if tipo == 'automotivo' and cat in cats_catalogo:
        n = (catalogo['categoria'] == cat).sum()
        print(f"  - {cat:<30s} {n:>3d} SKUs — {motivo}")

print()
print("="*70)
print("IMPACTO DO FILTRO")
print("="*70)
print()

# Estatísticas antes/depois
n_skus_antes = len(catalogo)
n_skus_depois = len(catalogo_conv)
n_cats_antes = catalogo['categoria'].nunique()
n_cats_depois = catalogo_conv['categoria'].nunique()

valor_total_antes = diag['valor_estoque_parado_R$'].sum()
valor_total_depois = diag_conv['valor_estoque_parado_R$'].sum()
valor_automotivo = valor_total_antes - valor_total_depois

print(f"  SKUs:        {n_skus_antes:>4d} → {n_skus_depois:>4d}  ({n_skus_depois/n_skus_antes*100:.0f}% mantido)")
print(f"  Categorias:  {n_cats_antes:>4d} → {n_cats_depois:>4d}")
print()
print(f"  Estoque parado:")
print(f"    Total:      R$ {valor_total_antes:>9,.2f}")
print(f"    Conveniência: R$ {valor_total_depois:>9,.2f}  ({valor_total_depois/valor_total_antes*100:.0f}%)")
print(f"    Automotivo:   R$ {valor_automotivo:>9,.2f}  ({valor_automotivo/valor_total_antes*100:.0f}%)")
print()
print("="*70)
print("TOP 20 CATEGORIAS DE CONVENIÊNCIA (após filtro)")
print("="*70)
print()
print(f"{'Categoria':<35s} {'SKUs':>5s} {'Preço':>8s} {'Margem':>7s} {'Status':>10s}")
print('-'*70)
for _, row in cat_stats_conv.head(20).iterrows():
    marca = '?' if row['tipo'] == 'revisar' else ' '
    print(f"  {row['categoria']:<33s} {int(row['n_skus']):>5d} "
          f"R$ {row['preco_medio']:>6.2f} {row['margem_pct_media']:>6.1f}% "
          f"{marca:>8s}")

print()
print("✓ Arquivos gerados:")
print(f"  data/categorias_classificadas.csv     ← tabela de classificação (pode revisar)")
print(f"  data/catalogo_conveniencia.csv        ← {n_skus_depois} SKUs filtrados")
print(f"  results/categorias_conveniencia.csv   ← stats por categoria filtrada")
print(f"  results/analise_estoque_parado_conveniencia.csv ← diagnóstico só conveniência")
print()
print("Pra ajustar: edita data/categorias_classificadas.csv e me avisa as mudanças")
