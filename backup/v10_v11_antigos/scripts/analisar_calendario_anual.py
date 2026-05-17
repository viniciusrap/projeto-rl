"""Análise visual do calendário anual de promoções V3.

Lê results/v11/calendario_v3_anual.json e gera:
- Distribuição de campanhas por mês × categoria (heatmap)
- Gráfico Gantt das 30 maiores campanhas
- Sobreposição com eventos comerciais do ano
- Resumo estatístico
"""
import io
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V11 = ROOT / 'results' / 'v11'

# ── Carregar dados ─────────────────────────────────────────────────────────

with open(V11 / 'calendario_v3_anual.json', encoding='utf-8') as f:
    cal = json.load(f)

camp = pd.DataFrame(cal['campanhas'])
camp['data_inicio'] = pd.to_datetime(camp['data_inicio'])
camp['data_fim'] = pd.to_datetime(camp['data_fim'])
camp['mes'] = camp['data_inicio'].dt.month
print(f"Total campanhas: {len(camp)}")
print(f"Lucro total estimado: R$ {cal['lucro_adicional_total_R$']:,.2f}")
print(f"Período: {cal['data_inicio']} a {cal['data_fim']}")

# ── 1. Distribuição por mês × categoria ───────────────────────────────────

heat = (camp.groupby(['mes', 'categoria'])
              .agg(n=('dias_total', 'count'),
                   dias=('dias_total', 'sum'),
                   lucro=('lucro_adicional_estimado_R$', 'sum'))
              .reset_index())

# Pivot para heatmap
piv = heat.pivot_table(index='categoria', columns='mes', values='lucro',
                         fill_value=0)
# Completar meses faltantes
for m in range(1, 13):
    if m not in piv.columns:
        piv[m] = 0
piv = piv[sorted(piv.columns)]

print(f"\n{len(piv)} categorias promovidas, {len(piv.columns)} meses cobertos")

# Distribuição por categoria
print("\nDistribuição por categoria (campanhas):")
cat_cnt = camp.groupby('categoria').agg(
    n=('dias_total', 'count'),
    dias=('dias_total', 'sum'),
    lucro=('lucro_adicional_estimado_R$', 'sum')
).sort_values('lucro', ascending=False)
print(cat_cnt.to_string())

# Distribuição por mês
print("\nDistribuição por mês:")
mes_cnt = camp.groupby('mes').agg(
    n_camp=('dias_total', 'count'),
    dias_total=('dias_total', 'sum'),
    lucro=('lucro_adicional_estimado_R$', 'sum')
)
nomes_mes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
              'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
for m, row in mes_cnt.iterrows():
    print(f"  {nomes_mes[int(m) - 1]:<4s} {int(row['n_camp']):>3d} campanhas  "
          f"{int(row['dias_total']):>4d} dias  R$ {row['lucro']:>8.2f}")

# ── 2. Visualizações ──────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

# Painel 1: Heatmap categoria × mês
ax1 = fig.add_subplot(gs[0, :])
piv_sorted = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
im = ax1.imshow(piv_sorted.values, aspect='auto', cmap='YlOrRd')
ax1.set_yticks(range(len(piv_sorted.index)))
ax1.set_yticklabels(piv_sorted.index)
ax1.set_xticks(range(12))
ax1.set_xticklabels(nomes_mes)
ax1.set_title(f'Lucro adicional (R$) por categoria × mês — calendário anual V11\n'
                f'Total: {len(camp)} campanhas, R$ {cal["lucro_adicional_total_R$"]:,.2f}',
                fontsize=12)
# Anotar valores não-zero
for i in range(piv_sorted.shape[0]):
    for j in range(piv_sorted.shape[1]):
        v = piv_sorted.values[i, j]
        if v > 10:
            ax1.text(j, i, f'{v:.0f}', ha='center', va='center',
                       fontsize=8, color='black' if v < piv_sorted.values.max()*0.6 else 'white')
plt.colorbar(im, ax=ax1, label='R$ lucro adicional')

# Painel 2: Barras por mês
ax2 = fig.add_subplot(gs[1, 0])
mes_cnt_sorted = mes_cnt.sort_index()
ax2.bar([nomes_mes[int(m) - 1] for m in mes_cnt_sorted.index],
         mes_cnt_sorted['lucro'], color='steelblue')
ax2.set_ylabel('R$ lucro adicional')
ax2.set_title('Lucro adicional por mês')
ax2.grid(axis='y', alpha=0.3)
plt.setp(ax2.get_xticklabels(), rotation=45)

# Painel 3: Barras por categoria
ax3 = fig.add_subplot(gs[1, 1])
top_cat = cat_cnt.head(8)
ax3.barh(top_cat.index, top_cat['lucro'], color='coral')
ax3.set_xlabel('R$ lucro adicional')
ax3.set_title('Top 8 categorias por lucro adicional')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(V11 / 'calendario_anual_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"\n✓ {V11 / 'calendario_anual_heatmap.png'}")

# ── 3. Gantt das top 30 campanhas ────────────────────────────────────────

top30 = camp.nlargest(30, 'lucro_adicional_estimado_R$').sort_values('data_inicio')

fig, ax = plt.subplots(figsize=(14, 10))
cores_cat = plt.cm.tab20(np.linspace(0, 1, camp['categoria'].nunique()))
mapa_cor = {cat: cores_cat[i]
              for i, cat in enumerate(sorted(camp['categoria'].unique()))}

for i, (idx, row) in enumerate(top30.iterrows()):
    cor = mapa_cor[row['categoria']]
    ini = row['data_inicio']
    dur = (row['data_fim'] - row['data_inicio']).days + 1
    ax.barh(i, dur, left=ini.toordinal(), color=cor, edgecolor='black',
             linewidth=0.5)
    ax.text(ini.toordinal() + dur + 1, i,
              f"R${row['lucro_adicional_estimado_R$']:.0f}",
              va='center', fontsize=8)

ax.set_yticks(range(len(top30)))
ax.set_yticklabels([f"{r['data_inicio'].strftime('%d/%m')} "
                       f"{r['categoria'][:15]} ({r['intensidade']})"
                       for _, r in top30.iterrows()],
                      fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Data')

# Converter xticks de ordinal para data
inicio = date(2026, 5, 12)
fim = date(2027, 5, 11)
months = pd.date_range(inicio, fim, freq='MS')
ax.set_xticks([d.toordinal() for d in months])
ax.set_xticklabels([d.strftime('%b/%y') for d in months], rotation=45)

ax.set_title('Top 30 campanhas anuais (modelo V11 20 cat + penalidade)',
              fontsize=12)
ax.grid(axis='x', alpha=0.3)

# Legenda
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=mapa_cor[c])
                    for c in sorted(camp['categoria'].unique())]
ax.legend(legend_handles, sorted(camp['categoria'].unique()),
            loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=8)

plt.tight_layout()
plt.savefig(V11 / 'gantt_top30_campanhas.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ {V11 / 'gantt_top30_campanhas.png'}")

# ── 4. Cobertura de eventos comerciais ────────────────────────────────────

# Para cada evento comercial relevante no horizonte, quantas campanhas
# caem na janela?
eventos_horizonte = []
for ev in cal['campanhas']:
    if ev.get('eventos_comerciais_na_janela'):
        for nome in ev['eventos_comerciais_na_janela']:
            eventos_horizonte.append({
                'evento': nome,
                'categoria_promovida': ev['categoria'],
                'data_inicio': ev['data_inicio'],
                'lucro': ev['lucro_adicional_estimado_R$'],
            })

if eventos_horizonte:
    df_ev = pd.DataFrame(eventos_horizonte)
    df_ev_agg = df_ev.groupby('evento').agg(
        n_campanhas=('categoria_promovida', 'count'),
        categorias=('categoria_promovida', lambda x: ';'.join(set(x))),
        lucro_total=('lucro', 'sum'),
    ).reset_index().sort_values('lucro_total', ascending=False)
    df_ev_agg.to_csv(V11 / 'eventos_cobertos_pelo_calendario.csv',
                       index=False, encoding='utf-8')
    print(f"\nEventos comerciais cobertos por campanhas ({len(df_ev_agg)}):")
    for _, r in df_ev_agg.iterrows():
        print(f"  {r['evento']:<35s} {int(r['n_campanhas']):>2d} campanhas  "
              f"R$ {r['lucro_total']:>7.2f}  cats: {r['categorias']}")
else:
    print("\n(Sem eventos cobertos por campanhas)")

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print("="*70)
print("RESUMO DO CALENDÁRIO ANUAL")
print("="*70)
print()
print(f"  Total de campanhas:        {len(camp)}")
print(f"  Lucro adicional anual:     R$ {cal['lucro_adicional_total_R$']:,.2f}")
print(f"  Categorias promovidas:     {camp['categoria'].nunique()}/20")
print(f"  Intensidades usadas:       {camp['intensidade'].nunique()}/4")
print(f"  Maior campanha (R$):       {camp.loc[camp['lucro_adicional_estimado_R$'].idxmax(), 'data_inicio'].strftime('%d/%m')} "
      f"{camp.loc[camp['lucro_adicional_estimado_R$'].idxmax(), 'categoria']} "
      f"({camp['lucro_adicional_estimado_R$'].max():.2f})")
print(f"  Eventos comerciais cobertos: {len(set(e['evento'] for e in eventos_horizonte)) if eventos_horizonte else 0}")

# Estatística importante: % de promoção por bucket de evento
todos_dias = pd.date_range(cal['data_inicio'], cal['data_fim'])
total_dias = len(todos_dias)
dias_com_camp = sum(c['dias_total'] for c in cal['campanhas'])
print(f"\n  % do ano em campanha:      {dias_com_camp / total_dias * 100:.1f}%")
print(f"  R$/dia em campanha:         R$ {cal['lucro_adicional_total_R$'] / dias_com_camp:.2f}")
print(f"  R$/dia anualizado:          R$ {cal['lucro_adicional_total_R$'] / total_dias:.2f}")
