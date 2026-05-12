"""Analisa impacto real da Copa do Mundo 2022 nas vendas do posto.

Copa 2022 (Qatar, nov-dez/2022) — Brasil jogou:
- 24/11/2022 (qui) — Brasil 2 × 0 Sérvia
- 28/11/2022 (seg) — Brasil 1 × 0 Suíça
- 02/12/2022 (sex) — Brasil 1 × 1 Camarões
- 05/12/2022 (seg) — Brasil 4 × 1 Coréia do Sul (oitavas)
- 09/12/2022 (sex) — Brasil 1 × 1 Croácia (quartas, eliminação nos pênaltis)

Para cada jogo, calcula uplift em:
- Categoria-alvo: cerveja, snack, refrigerante, gelo
- Categorias controle: água, padaria (não esperamos uplift forte)

Método: comparar venda do dia do jogo com média do mesmo dia da semana
em janelas equivalentes de 2020, 2021, 2023, 2024 (controlando dia da
semana + mês).

Saída: results/v11/uplift_copa_2022_posto.csv + .png
"""
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
V11 = ROOT / 'results' / 'v11'

# ── Carrega vendas ──────────────────────────────────────────────────────────

vendas = pd.read_csv(DATA / 'venda_por_dia_parseado.csv', parse_dates=['data'])
print(f"Vendas: {len(vendas):,} registros 2020-2026")

# Mesmo mapeamento dos outros scripts
MAPA = {
    'CERVEJA AMBEV': 'cerveja', 'CERVEJA FEMSA': 'cerveja',
    'CERVEJA ESPECIAIS': 'cerveja', 'CERVEJA ITAIPAVA': 'cerveja',
    'REFRIGERANTE': 'refrigerante',
    'SNACK ELMA CHIPS': 'snack', 'SNACK DIVERSOS': 'snack',
    'SNACK TORCIDA': 'snack', 'SNACK PRINGLES': 'snack',
    'AMENDOIN/NOZES': 'snack',
    'GELO': 'gelo',
    'AGUA': 'agua', 'AGUA SABORIZADA': 'agua', 'ÁGUA DE COCO': 'agua',
    'ENERGÉTICO': 'energetico',
    'ISOTÔNICO': 'isotonico',
    'SUCO': 'suco',
    'PADARIA': 'padaria', 'SANDUÍCHE': 'padaria',
}
vendas['cat_modelo'] = vendas['categoria'].map(MAPA)
vendas_rel = vendas[vendas['cat_modelo'].notna()].copy()

# Receita diária por categoria
diario = (vendas_rel.groupby(['data', 'cat_modelo'])['valor_venda']
                     .sum().reset_index())
diario['dia_semana'] = diario['data'].dt.dayofweek
diario['mes'] = diario['data'].dt.month
diario['ano'] = diario['data'].dt.year

# ── Jogos do Brasil na Copa 2022 ───────────────────────────────────────────

JOGOS_COPA = [
    (date(2022, 11, 24), 'Quinta', 'Brasil 2×0 Sérvia (estreia)'),
    (date(2022, 11, 28), 'Segunda', 'Brasil 1×0 Suíça'),
    (date(2022, 12, 2),  'Sexta',   'Brasil 1×1 Camarões'),
    (date(2022, 12, 5),  'Segunda', 'Brasil 4×1 Coréia (oitavas)'),
    (date(2022, 12, 9),  'Sexta',   'Brasil 1×1 Croácia (eliminação)'),
]

# Período da Copa em geral (todo)
COPA_INICIO = date(2022, 11, 20)
COPA_FIM = date(2022, 12, 18)

# ── 1. Para cada jogo, calcular uplift ─────────────────────────────────────

print("\n=== UPLIFT DURANTE JOGOS DO BRASIL ===")
print()

CATS_FOCO = ['cerveja', 'snack', 'refrigerante', 'gelo', 'energetico',
              'isotonico', 'agua', 'padaria']

resultados = []
for data_jogo, dia_nome, descricao in JOGOS_COPA:
    print(f"\n📅 {data_jogo.strftime('%d/%m/%Y')} ({dia_nome}) — {descricao}")
    print('-' * 70)
    dia_sem = data_jogo.weekday()
    mes_jogo = data_jogo.month

    # Janela: dia do jogo (jogos do dia podem influenciar +1 dia se for à noite)
    janela = [data_jogo, data_jogo + timedelta(days=1)]

    for cat in CATS_FOCO:
        sub = diario[diario['cat_modelo'] == cat]
        # Venda média no dia do jogo
        vendas_jogo = sub[sub['data'].isin([pd.Timestamp(d) for d in janela])]
        if len(vendas_jogo) == 0:
            continue
        venda_jogo = vendas_jogo['valor_venda'].mean()

        # Baseline: mesmo dia da semana, mesmo mês, outros anos NÃO copa
        anos_baseline = [2020, 2021, 2023, 2024]  # 2022 é Copa, excluir
        baseline_data = sub[(sub['ano'].isin(anos_baseline))
                              & (sub['dia_semana'] == dia_sem)
                              & (sub['mes'] == mes_jogo)]
        if len(baseline_data) < 3:
            continue
        venda_baseline = baseline_data['valor_venda'].mean()
        if venda_baseline < 0.01:
            continue

        uplift = venda_jogo / venda_baseline
        resultados.append({
            'data_jogo': data_jogo.isoformat(),
            'descricao': descricao,
            'categoria': cat,
            'venda_jogo_R$': round(venda_jogo, 2),
            'venda_baseline_R$': round(venda_baseline, 2),
            'uplift': round(uplift, 3),
            'n_baseline': len(baseline_data),
        })
        marca = '🔥' if uplift > 1.20 else '↑' if uplift > 1.05 else '·' if uplift > 0.95 else '↓'
        print(f"  {marca} {cat:<14s} R$ {venda_jogo:>7.2f} jogo vs R$ {venda_baseline:>7.2f} baseline  "
              f"= {uplift:>5.2f}× ({len(baseline_data)} pontos baseline)")

df = pd.DataFrame(resultados)
df.to_csv(V11 / 'uplift_copa_2022_jogos_posto.csv', index=False, encoding='utf-8')

# ── 2. Período inteiro da Copa (20/11 a 18/12/2022) ────────────────────────

print()
print("=" * 70)
print("UPLIFT MÉDIO DURANTE TODO O PERÍODO DA COPA (20/11 a 18/12/2022)")
print("=" * 70)
print()

resultados_periodo = []
for cat in CATS_FOCO:
    sub = diario[diario['cat_modelo'] == cat]
    # Período Copa
    copa_data = sub[(sub['data'] >= pd.Timestamp(COPA_INICIO))
                      & (sub['data'] <= pd.Timestamp(COPA_FIM))]
    if len(copa_data) == 0:
        continue
    media_copa = copa_data['valor_venda'].mean()

    # Baseline: nov-dez de outros anos (2020, 2021, 2023, 2024)
    base_data = sub[((sub['ano'].isin([2020, 2021, 2023, 2024]))
                       & ((sub['mes'] == 11) | (sub['mes'] == 12))
                       & ~((sub['mes'] == 12) & (sub['data'].dt.day > 18))
                       & ~((sub['mes'] == 11) & (sub['data'].dt.day < 20)))]
    if len(base_data) < 30:
        continue
    media_base = base_data['valor_venda'].mean()
    if media_base < 0.01:
        continue
    uplift = media_copa / media_base
    resultados_periodo.append({
        'categoria': cat,
        'media_copa_R$': round(media_copa, 2),
        'media_baseline_nov_dez_outros_anos': round(media_base, 2),
        'uplift_periodo_copa': round(uplift, 3),
        'n_dias_copa': len(copa_data),
        'n_dias_baseline': len(base_data),
    })
    marca = '🔥' if uplift > 1.20 else '↑' if uplift > 1.05 else '·' if uplift > 0.95 else '↓'
    print(f"  {marca} {cat:<14s} R$ {media_copa:>7.2f}/dia vs R$ {media_base:>7.2f}/dia  "
          f"= {uplift:>5.2f}× ({len(copa_data)}d vs {len(base_data)}d)")

df_per = pd.DataFrame(resultados_periodo)
df_per.to_csv(V11 / 'uplift_copa_2022_periodo_posto.csv',
                index=False, encoding='utf-8')

# ── 3. Comparar dia a dia do mês de novembro 2022 vs outros novembros ─────

print()
print("=" * 70)
print("EVOLUÇÃO TEMPORAL — Nov/2022 (com Copa) vs Nov dos outros anos")
print("=" * 70)
print()

# Para cerveja: tendência diária
sub_cerveja = diario[(diario['cat_modelo'] == 'cerveja')
                       & (diario['mes'] == 11)]
sub_cerveja['dia_mes'] = sub_cerveja['data'].dt.day
agg_diario = sub_cerveja.groupby(['dia_mes', 'ano'])['valor_venda'].sum().unstack()

# ── Visualização ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Painel 1: uplift por categoria nos jogos
ax = axes[0, 0]
agg_jogos = df.groupby('categoria')['uplift'].mean().sort_values()
cores = ['#16a34a' if u > 1.15 else '#ca8a04' if u > 1.0 else '#dc2626'
          for u in agg_jogos]
ax.barh(agg_jogos.index, agg_jogos.values, color=cores)
ax.axvline(1.0, color='gray', linestyle='--')
ax.set_xlabel('Uplift médio nos jogos do Brasil')
ax.set_title('Uplift médio durante jogos do Brasil 2022\n(5 jogos)')
ax.grid(axis='x', alpha=0.3)

# Painel 2: uplift por categoria no período inteiro da Copa
ax = axes[0, 1]
df_per_sorted = df_per.sort_values('uplift_periodo_copa')
cores = ['#16a34a' if u > 1.15 else '#ca8a04' if u > 1.0 else '#dc2626'
          for u in df_per_sorted['uplift_periodo_copa']]
ax.barh(df_per_sorted['categoria'], df_per_sorted['uplift_periodo_copa'], color=cores)
ax.axvline(1.0, color='gray', linestyle='--')
ax.set_xlabel('Uplift médio no período da Copa')
ax.set_title('Uplift no período inteiro da Copa\n(20/11 a 18/12/2022)')
ax.grid(axis='x', alpha=0.3)

# Painel 3: linha do tempo cerveja em nov/dez 2022
ax = axes[1, 0]
nov_dez_2022 = diario[(diario['cat_modelo'] == 'cerveja')
                        & ((diario['mes'] == 11) | (diario['mes'] == 12))
                        & (diario['ano'] == 2022)].sort_values('data')
nov_dez_outros = (diario[(diario['cat_modelo'] == 'cerveja')
                            & ((diario['mes'] == 11) | (diario['mes'] == 12))
                            & (diario['ano'].isin([2020, 2021, 2023, 2024]))]
                    .groupby(diario['data'].dt.dayofyear)['valor_venda']
                    .mean()
                    .reset_index())
ax.plot(nov_dez_2022['data'], nov_dez_2022['valor_venda'],
         label='2022 (com Copa)', linewidth=2, color='#dc2626')
# Mapear dia do ano de 2022 para a mesma data dos outros anos
for ano in [2020, 2021, 2023, 2024]:
    sub_ano = diario[(diario['cat_modelo'] == 'cerveja')
                       & ((diario['mes'] == 11) | (diario['mes'] == 12))
                       & (diario['ano'] == ano)].sort_values('data')
    if len(sub_ano) > 0:
        # Recompor data de 2022 para sobreposição visual
        datas_2022 = pd.to_datetime([f'2022-{d.month:02d}-{d.day:02d}'
                                        for d in sub_ano['data']])
        ax.plot(datas_2022, sub_ano['valor_venda'], alpha=0.3,
                 label=f'{ano}', linewidth=1)
# Marcar jogos do Brasil
for d, _, _ in JOGOS_COPA:
    ax.axvline(pd.Timestamp(d), color='gold', alpha=0.7, linestyle='--', linewidth=1)
ax.set_xlabel('Data')
ax.set_ylabel('Receita cerveja (R$)')
ax.set_title('Vendas de cerveja Nov-Dez/2022 (Copa)\nvs mesma janela em outros anos')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45)

# Painel 4: idem para snack
ax = axes[1, 1]
nov_dez_2022_s = diario[(diario['cat_modelo'] == 'snack')
                          & ((diario['mes'] == 11) | (diario['mes'] == 12))
                          & (diario['ano'] == 2022)].sort_values('data')
ax.plot(nov_dez_2022_s['data'], nov_dez_2022_s['valor_venda'],
         label='2022 (com Copa)', linewidth=2, color='#dc2626')
for ano in [2020, 2021, 2023, 2024]:
    sub_ano = diario[(diario['cat_modelo'] == 'snack')
                       & ((diario['mes'] == 11) | (diario['mes'] == 12))
                       & (diario['ano'] == ano)].sort_values('data')
    if len(sub_ano) > 0:
        datas_2022 = pd.to_datetime([f'2022-{d.month:02d}-{d.day:02d}'
                                        for d in sub_ano['data']])
        ax.plot(datas_2022, sub_ano['valor_venda'], alpha=0.3,
                 label=f'{ano}', linewidth=1)
for d, _, _ in JOGOS_COPA:
    ax.axvline(pd.Timestamp(d), color='gold', alpha=0.7, linestyle='--', linewidth=1)
ax.set_xlabel('Data')
ax.set_ylabel('Receita snack (R$)')
ax.set_title('Vendas de snack Nov-Dez/2022 (Copa)\nvs mesma janela em outros anos')
ax.legend(fontsize=8, loc='upper right')
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(V11 / 'copa_2022_uplift.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Conclusões ──────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("CONCLUSÕES — COPA 2022 NO POSTO")
print("=" * 70)
print()
print(f"Uplift médio durante jogos do Brasil (todas categorias):")
mean_jogo = df['uplift'].mean()
print(f"  {mean_jogo:.2f}× (n={len(df)} medições)")
print()
print(f"Uplift médio no período inteiro da Copa:")
mean_per = df_per['uplift_periodo_copa'].mean()
print(f"  {mean_per:.2f}× (categoria média)")
print()
print(f"Maior uplift em jogo individual:")
top_jogo = df.nlargest(3, 'uplift')
for _, r in top_jogo.iterrows():
    print(f"  {r['uplift']:.2f}× — {r['data_jogo']} {r['categoria']} ({r['descricao']})")
print()
print(f"✓ Saídas:")
print(f"  results/v11/uplift_copa_2022_jogos_posto.csv")
print(f"  results/v11/uplift_copa_2022_periodo_posto.csv")
print(f"  results/v11/copa_2022_uplift.png")
