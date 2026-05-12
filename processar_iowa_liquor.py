"""Processa Iowa Liquor Sales — vendas de álcool em LOJA FÍSICA estadual,
2012-2024 (12 anos). Gold standard para validar uplift de álcool em datas
comerciais (Mães, Pais, Namorados, Réveillon, Natal, Halloween).

Dataset: 3.4GB, ~28M transações. Vai ser lido em chunks.

Saída:
  data/priors_externos/iowa_liquor/uplift_por_evento.csv
  data/priors_externos/iowa_liquor/categorias_mapeadas.csv
  results/v11/iowa_liquor_uplift.png
"""
import io
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RAW = ROOT / 'data' / 'raw_iowa' / 'Iowa_Liquor_Sales.csv'
OUT = ROOT / 'data' / 'priors_externos' / 'iowa_liquor'
OUT.mkdir(parents=True, exist_ok=True)
RES = ROOT / 'results' / 'v11'

# ── Datas comerciais americanas relevantes (alinhadas com Brasil) ────────

EVENTOS_USA = [
    # Eventos universais (mesma data Brasil/USA)
    ('valentines', 'Dia dos Namorados USA (14/02)', '%Y-02-14', 14, 0),
    ('thanksgiving', 'Thanksgiving (4ª quinta nov)', 'thanksgiving', 7, 0),
    ('black_friday', 'Black Friday (dia após Thanksgiving)', 'black_friday', 3, 3),
    ('christmas_eve', 'Véspera de Natal (24/12)', '%Y-12-24', 10, 0),
    ('new_year_eve', 'Réveillon (31/12)', '%Y-12-31', 5, 0),
    ('halloween', 'Halloween (31/10)', '%Y-10-31', 7, 0),
    # USA-specific mas relacionados a feriados gerais
    ('mothers_day', 'Dia das Mães USA (2º dom maio)', 'mothers_day', 10, 0),
    ('fathers_day', 'Dia dos Pais USA (3º dom junho)', 'fathers_day', 10, 0),
    ('memorial_day', 'Memorial Day (último dom maio)', 'memorial_day', 3, 1),
    ('july_4', 'Independência USA (04/07)', '%Y-07-04', 5, 0),
    ('labor_day', 'Labor Day USA (1º seg setembro)', 'labor_day', 3, 0),
    ('super_bowl', 'Super Bowl (1º dom fev)', 'super_bowl', 3, 0),
]


def get_data_evento(evento_key, ano):
    """evento_key aqui é o 'formato' (3º elemento da tupla)."""
    if '%Y' in evento_key:
        return datetime.strptime(evento_key.replace('%Y', str(ano)), '%Y-%m-%d')
    elif evento_key == 'thanksgiving':
        d = datetime(ano, 11, 1)
        d += timedelta(days=(3 - d.weekday()) % 7)  # primeira quinta
        d += timedelta(days=21)  # quarta quinta
        return d
    elif evento_key == 'black_friday':
        d = datetime(ano, 11, 1)
        d += timedelta(days=(3 - d.weekday()) % 7)
        d += timedelta(days=22)
        return d
    elif evento_key == 'mothers_day':
        d = datetime(ano, 5, 1)
        d += timedelta(days=(6 - d.weekday()) % 7)  # 1º domingo
        d += timedelta(days=7)  # 2º domingo
        return d
    elif evento_key == 'fathers_day':
        d = datetime(ano, 6, 1)
        d += timedelta(days=(6 - d.weekday()) % 7)
        d += timedelta(days=14)  # 3º domingo
        return d
    elif evento_key == 'memorial_day':
        d = datetime(ano, 5, 31)
        d -= timedelta(days=(d.weekday() - 6) % 7)  # último dom de maio
        return d
    elif evento_key == 'labor_day':
        d = datetime(ano, 9, 1)
        while d.weekday() != 0:
            d += timedelta(days=1)
        return d
    elif evento_key == 'super_bowl':
        d = datetime(ano, 2, 1)
        d += timedelta(days=(6 - d.weekday()) % 7)
        return d
    return None


# ── Mapear categorias Iowa → nossas categorias ────────────────────────────

# Iowa usa "Category Name" e podem ser específicas (Vodkas, etc)
def mapear_categoria_iowa(cat_name):
    if pd.isna(cat_name):
        return None
    c = str(cat_name).upper()
    if 'VODKA' in c:
        return 'destilados'
    if 'WHISKY' in c or 'WHISKEY' in c or 'BOURBON' in c or 'SCOTCH' in c:
        return 'destilados'
    if 'RUM' in c:
        return 'destilados'
    if 'TEQUILA' in c or 'MEZCAL' in c:
        return 'destilados'
    if 'GIN' in c:
        return 'destilados'
    if 'BRANDY' in c or 'COGNAC' in c:
        return 'destilados'
    if 'LIQUEUR' in c or 'CORDIAL' in c or 'SCHNAPPS' in c:
        return 'destilados'
    if 'WINE' in c:
        return 'vinho'
    if 'CHAMPAGNE' in c or 'SPARKLING' in c:
        return 'vinho'
    if 'BEER' in c or 'ALE' in c:
        return 'cerveja'
    return None


# ── Leitura em chunks ──────────────────────────────────────────────────────

print(f"Carregando Iowa Liquor ({RAW.name}, 3.4 GB)...")
print("Lendo em chunks de 500k linhas — pode levar 3-5 minutos\n")

chunks_proc = []
total_linhas = 0
for i, chunk in enumerate(pd.read_csv(
    RAW,
    usecols=['Date', 'Category Name', 'Bottles Sold', 'Sale (Dollars)'],
    chunksize=500_000,
    parse_dates=False,
    low_memory=False,
)):
    total_linhas += len(chunk)
    if i % 5 == 0:
        print(f"  chunk {i+1} — {total_linhas:>12,} linhas lidas...")
    # Filtrar só categorias relevantes
    chunk['cat_modelo'] = chunk['Category Name'].apply(mapear_categoria_iowa)
    chunk = chunk[chunk['cat_modelo'].notna()].copy()
    if len(chunk) == 0:
        continue
    chunk['Date'] = pd.to_datetime(chunk['Date'], format='%m/%d/%Y',
                                     errors='coerce')
    chunk = chunk.dropna(subset=['Date'])
    # Iowa vem com $ na frente — remover
    chunk['Sale (Dollars)'] = pd.to_numeric(
        chunk['Sale (Dollars)'].astype(str).str.replace('$', '', regex=False),
        errors='coerce'
    ).fillna(0)
    chunk['Bottles Sold'] = pd.to_numeric(chunk['Bottles Sold'],
                                            errors='coerce').fillna(0)
    chunks_proc.append(chunk[['Date', 'cat_modelo', 'Bottles Sold', 'Sale (Dollars)']])

print(f"\nTotal lido: {total_linhas:,} linhas")
print(f"Linhas relevantes (álcool): {sum(len(c) for c in chunks_proc):,}")

vendas = pd.concat(chunks_proc, ignore_index=True)
print(f"Categorias mapeadas: {vendas['cat_modelo'].value_counts().to_dict()}")

# ── Agregação diária por categoria ────────────────────────────────────────

print("\nAgregando vendas diárias...")
diario = (vendas.groupby([vendas['Date'].dt.date, 'cat_modelo'])
                 .agg(receita=('Sale (Dollars)', 'sum'),
                      garrafas=('Bottles Sold', 'sum'),
                      n_trans=('Date', 'count'))
                 .reset_index())
diario['Date'] = pd.to_datetime(diario['Date'])
diario['ano'] = diario['Date'].dt.year
diario['mes'] = diario['Date'].dt.month
print(f"  {len(diario):,} (data × categoria)")
print(f"  Período: {diario['Date'].min().date()} a {diario['Date'].max().date()}")

# ── Calcular uplift por evento × categoria × ano ──────────────────────────

print("\nCalculando uplift por evento...")
resultados = []
anos = sorted(diario['ano'].unique())

for evento_key, nome_evento, formato, pre, pos in EVENTOS_USA:
    for ano in anos:
        d_ev = get_data_evento(formato, ano)
        if d_ev is None:
            continue

        inicio_jan = d_ev - timedelta(days=pre)
        fim_jan = d_ev + timedelta(days=pos)
        inicio_base = d_ev - timedelta(days=180)
        fim_base = d_ev + timedelta(days=180)

        # Iowa Liquor Stores são estaduais e vendem APENAS destilados.
        # Cerveja e vinho ficam para Dunnhumby/Tesco.
        for cat in ['destilados']:
            sub = diario[(diario['cat_modelo'] == cat)
                          & (diario['Date'] >= pd.Timestamp(inicio_base))
                          & (diario['Date'] <= pd.Timestamp(fim_base))]
            if len(sub) == 0:
                continue
            janela = sub[(sub['Date'] >= pd.Timestamp(inicio_jan))
                          & (sub['Date'] <= pd.Timestamp(fim_jan))]
            baseline = sub[((sub['Date'] < pd.Timestamp(inicio_jan - timedelta(days=14))))
                            | ((sub['Date'] > pd.Timestamp(fim_jan + timedelta(days=14))))]
            if len(janela) < 3 or len(baseline) < 14:
                continue
            mj = janela['receita'].mean()
            mb = baseline['receita'].mean()
            if mb < 0.01:
                continue
            uplift = mj / mb
            resultados.append({
                'evento': nome_evento,
                'evento_key': evento_key,
                'ano': ano,
                'data_evento': d_ev.date().isoformat(),
                'categoria': cat,
                'receita_janela': round(mj, 2),
                'receita_baseline': round(mb, 2),
                'uplift': round(uplift, 3),
                'n_dias_janela': len(janela),
            })

df = pd.DataFrame(resultados)
if len(df) == 0:
    print("\n✗ Nenhum uplift calculado. Veja se mapeamento de categorias bate.")
    sys.exit(0)
df.to_csv(OUT / 'uplift_por_evento.csv', index=False, encoding='utf-8')
print(f"  {len(df)} medições brutas")

# Agregação por (evento, categoria): média entre anos
agg = (df.groupby(['evento', 'evento_key', 'categoria'])
         .agg(uplift_medio=('uplift', 'mean'),
              uplift_std=('uplift', 'std'),
              n_anos=('uplift', 'count'))
         .reset_index()
         .sort_values('uplift_medio', ascending=False))
agg['uplift_medio'] = agg['uplift_medio'].round(3)
agg['uplift_std'] = agg['uplift_std'].round(3)
agg.to_csv(OUT / 'uplift_agregado.csv', index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print("=" * 75)
print(f"IOWA LIQUOR — UPLIFT POR EVENTO × CATEGORIA (média de {agg['n_anos'].max()} anos)")
print("=" * 75)
print()
print(f"{'Evento':<40s} {'Cat':<12s} {'Uplift':>7s} {'±':>5s} {'N':>3s}")
print('-' * 75)
for _, r in agg.iterrows():
    print(f"  {r['evento'][:38]:<38s} {r['categoria']:<12s} "
          f"{r['uplift_medio']:>5.2f}× ±{r['uplift_std']:>4.2f} "
          f"{int(r['n_anos']):>3d}")

# ── Visualização ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 8))
piv = agg.pivot_table(index='evento', columns='categoria', values='uplift_medio')
piv = piv.fillna(0).sort_values(by='destilados', ascending=True, na_position='first')
piv.plot(kind='barh', ax=ax, color=['#7c3aed', '#7c2d12', '#fbbf24'])
ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Uplift × baseline')
ax.set_title(f'Iowa Liquor Sales — uplift de álcool por evento\n'
              f'(loja física estadual USA, 12 anos)')
ax.legend(title='Categoria')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RES / 'iowa_liquor_uplift.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ Saídas:")
print(f"  {OUT / 'uplift_por_evento.csv'}")
print(f"  {OUT / 'uplift_agregado.csv'}")
print(f"  {RES / 'iowa_liquor_uplift.png'}")
