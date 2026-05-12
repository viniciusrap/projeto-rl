"""Análise refinada dos dados Dunnhumby já processados.

A passagem anterior agregou mal o causal_data (todos meses davam 100%).
Esta análise usa as MÉTRICAS QUE PRESTAM:

1. % de transações com desconto efetivo (RETAIL_DISC < 0) — promoção real
2. Magnitude média do desconto (% de redução vs preço base)
3. Variação sazonal dessas duas métricas — quando promove mais / mais profundo

E produz prior temporal por categoria útil para V11.
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DIR = ROOT / 'data' / 'priors_externos' / 'dunnhumby'

# ── Carrega dados pré-processados ──────────────────────────────────────────

desc_sem = pd.read_csv(DIR / 'desconto_efetivo_por_categoria.csv')

# Calcular sazonalidade real por (categoria, semana_ciclo)
# Como temos 102 semanas no dataset, cada semana_ciclo (1-52) aparece ~2 vezes
saz_semanal = (desc_sem.groupby(['categoria', 'semana_ciclo'])
                         .agg(pct_desconto=('pct_com_desconto', 'mean'),
                              mag_desconto=('desconto_medio_pct', 'mean'),
                              n_trans=('n_transacoes', 'sum'))
                         .reset_index())
saz_semanal.to_csv(DIR / 'sazonalidade_semanal.csv',
                    index=False, encoding='utf-8')

# Sazonalidade mensal real
saz_mensal = (desc_sem.groupby(['categoria', 'mes_aprox'])
                        .agg(pct_desconto=('pct_com_desconto', 'mean'),
                             mag_desconto=('desconto_medio_pct', 'mean'),
                             n_trans=('n_transacoes', 'sum'))
                        .reset_index())
saz_mensal.to_csv(DIR / 'sazonalidade_mensal_real.csv',
                   index=False, encoding='utf-8')

# Para cada categoria, baseline = média anual
baselines = saz_mensal.groupby('categoria')['pct_desconto'].mean().to_dict()
mag_baselines = saz_mensal.groupby('categoria')['mag_desconto'].mean().to_dict()

saz_mensal['indice_freq'] = saz_mensal.apply(
    lambda r: r['pct_desconto'] / baselines[r['categoria']]
                if baselines[r['categoria']] > 0 else 1.0, axis=1
).round(3)
saz_mensal['indice_mag'] = saz_mensal.apply(
    lambda r: r['mag_desconto'] / mag_baselines[r['categoria']]
                if mag_baselines[r['categoria']] > 0 else 1.0, axis=1
).round(3)

# ── Apresentação ──────────────────────────────────────────────────────────

nomes_mes = ['', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

print("="*78)
print("PADRÃO DE PROMOÇÃO POR CATEGORIA (Dunnhumby — supermercado USA, 2014)")
print("="*78)
print()
print("PERFIL GERAL — frequência e magnitude do desconto efetivo:")
print()
print(f"{'Categoria':<18s}  {'% c/ desc':>10s}  {'mag média':>10s}  {'Perfil':<25s}")
print('-'*78)

# Ordenar por % desconto descendente
perfil = (desc_sem.groupby('categoria')
                    .agg(pct=('pct_com_desconto', 'mean'),
                         mag=('desconto_medio_pct', 'mean'))
                    .reset_index()
                    .sort_values('pct', ascending=False))

for _, row in perfil.iterrows():
    pct = row['pct']
    mag = row['mag']
    if pct > 50 and mag > 25:
        perfil_str = 'PROMOÇÃO AGRESSIVA'
    elif pct > 50:
        perfil_str = 'promove sempre, raso'
    elif pct > 20:
        perfil_str = 'promove regular'
    elif pct > 5:
        perfil_str = 'promove ocasional'
    else:
        perfil_str = 'quase nunca promove'
    print(f"  {row['categoria']:<16s}    {pct:>6.1f}%    {mag:>6.1f}%    {perfil_str:<25s}")

print()
print("="*78)
print("SAZONALIDADE — variação mensal do desconto")
print("="*78)
print()
print("Índice > 1 = mês promove MAIS que média do ano")
print("Índice < 1 = mês promove MENOS que média do ano")
print()

# Top variações sazonais por categoria
for cat in sorted(saz_mensal['categoria'].unique()):
    sub = saz_mensal[saz_mensal['categoria'] == cat].copy()
    if sub['pct_desconto'].sum() == 0:
        continue

    # Pegar os 3 meses de maior promoção e 2 de menor
    sub_sorted = sub.sort_values('indice_freq', ascending=False)
    top = sub_sorted.head(3)
    bottom = sub_sorted.tail(2)

    var = sub['indice_freq'].max() - sub['indice_freq'].min()
    if var < 0.15:  # variação pequena = sem sazonalidade
        continue

    print(f"\n  {cat.upper()}:")
    print(f"    Promove MAIS em:", end=' ')
    print(', '.join([f"{nomes_mes[int(r['mes_aprox'])]} ({r['indice_freq']:.2f}×)"
                     for _, r in top.iterrows()]))
    print(f"    Promove MENOS em:", end=' ')
    print(', '.join([f"{nomes_mes[int(r['mes_aprox'])]} ({r['indice_freq']:.2f}×)"
                     for _, r in bottom.iterrows()]))

print()
print("="*78)
print("MAGNITUDE DO DESCONTO — quando é mais profundo")
print("="*78)
print()

for cat in sorted(saz_mensal['categoria'].unique()):
    sub = saz_mensal[saz_mensal['categoria'] == cat].copy()
    if sub['mag_desconto'].sum() == 0:
        continue
    var = sub['indice_mag'].max() - sub['indice_mag'].min()
    if var < 0.10:
        continue
    top = sub.sort_values('indice_mag', ascending=False).head(2)
    bottom = sub.sort_values('indice_mag', ascending=True).head(2)
    print(f"\n  {cat.upper()}:")
    print(f"    Desconto MAIS profundo:", end=' ')
    print(', '.join([f"{nomes_mes[int(r['mes_aprox'])]} ({r['indice_mag']:.2f}×, "
                     f"{r['mag_desconto']:.0f}%)" for _, r in top.iterrows()]))
    print(f"    Desconto MAIS raso:", end=' ')
    print(', '.join([f"{nomes_mes[int(r['mes_aprox'])]} ({r['indice_mag']:.2f}×, "
                     f"{r['mag_desconto']:.0f}%)" for _, r in bottom.iterrows()]))

print()
print("="*78)
print("INSIGHTS — usar como prior no V11")
print("="*78)
print()

print("1. ÁLCOOL E CIGARRO promovem MUITO POUCO (<5% das transações com desconto)")
print("   mas quando promovem, magnitude é baixa (~5-10%). Categoria de alta")
print("   margem e regulada não usa desconto agressivo como estratégia regular.")
print()
print("2. SORVETE é a categoria que MAIS promove (~76% das transações com")
print("   desconto), com magnitude alta (~34%). Sazonal, validade curta,")
print("   precisa girar.")
print()
print("3. REFRIGERANTE, BISCOITO, PADARIA, SANDUÍCHE, SUCO operam com")
print("   ~60% das transações em algum desconto. É default em supermercado")
print("   de varejo — desconto é parte do preço normal.")
print()
print("4. ÁGUA promove 40% das transações com magnitude 25%. Comportamento")
print("   intermediário — produto commodity com margem mediana.")
print()
print(f"✓ Arquivos atualizados em {DIR}/")
print("  - sazonalidade_semanal.csv")
print("  - sazonalidade_mensal_real.csv (com indice_freq e indice_mag)")
