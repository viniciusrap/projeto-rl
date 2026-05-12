"""Avalia quão bem o forecaster manual atual prevê demanda real.

Modelo atual = demanda_base × fator_dia × fator_turno × fator_mes
              × fator_clima (de temperatura)

Compara com:
- Naive: média móvel 28 dias
- ML: regressão Ridge com features manuais

Métricas:
- RMSE
- MAPE (mean absolute percentage error)
- R²

Se ML reduz RMSE em >20% sobre o manual, vale fazer V12 com Forecaster.

Saída: results/v11/comparacao_forecasters.csv + .png
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results' / 'v11'

# ── Carregar dados de venda ───────────────────────────────────────────────

print("Carregando dados...")
vendas = pd.read_csv(DATA / 'venda_por_dia_parseado.csv', parse_dates=['data'])
temp = pd.read_csv(DATA / 'temperatura_historica.csv', parse_dates=['data'])
with open(DATA / 'calibracao_v2.json', encoding='utf-8') as f:
    cal = json.load(f)

# Mapeamento posto -> modelo (mesmo do calibrar_v2)
MAPA = cal['mapa_categoria_posto_para_modelo']
vendas['cat_modelo'] = vendas['categoria'].map(MAPA)
vendas = vendas[vendas['cat_modelo'].notna()].copy()

# Agregar por dia × categoria_modelo (não por turno para simplificar)
diario = vendas.groupby([vendas['data'].dt.date, 'cat_modelo']).agg(
    receita=('valor_venda', 'sum')
).reset_index()
diario['data'] = pd.to_datetime(diario['data'])
diario['dia_sem'] = diario['data'].dt.dayofweek
diario['mes'] = diario['data'].dt.month - 1  # 0-11
diario['ano'] = diario['data'].dt.year

# Merge temperatura
tmin = cal['clima_params']['temp_min']
tmax = cal['clima_params']['temp_max']
temp['temp_norm'] = (temp['temp_max'] - tmin) / (tmax - tmin)
diario = diario.merge(temp[['data', 'temp_norm']], on='data', how='left')
diario = diario.dropna(subset=['temp_norm'])

print(f"Registros: {len(diario):,}")
print(f"Período: {diario['data'].min().date()} a {diario['data'].max().date()}")
print(f"Categorias: {diario['cat_modelo'].nunique()}")

# ── Forecaster ATUAL (manual): demanda_base × fatores ────────────────────

# Mapeamento categoria → params
cat_params = {c['categoria']: c for c in cal['categorias']}

def previsao_manual(row, cat):
    p = cat_params.get(cat)
    if p is None:
        return np.nan
    demanda_base_diaria = p['demanda_base_dia']
    preco = p['preco_venda']
    fd = p['fator_dia'][row['dia_sem']]
    fm = p['fator_mes'][row['mes']]
    # turno: somando 3 turnos
    ft_total = sum(p['fator_turno'])
    # clima
    clima = p['clima_slope'] * row['temp_norm'] + p['clima_intercept']
    clima = max(0.5, min(2.0, clima))
    demanda_un = demanda_base_diaria * fd * fm * clima
    return demanda_un * preco  # receita esperada

print("\nCalculando previsao manual...")
diario['previsao_manual'] = diario.apply(
    lambda r: previsao_manual(r, r['cat_modelo']), axis=1)

# Filtrar valores válidos
df_eval = diario.dropna(subset=['previsao_manual', 'receita'])
df_eval = df_eval[(df_eval['receita'] > 0) & (df_eval['previsao_manual'] > 0)]

# ── Métricas por categoria ────────────────────────────────────────────────

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mape(real, pred):
    mask = real > 0.01
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask]) * 100))

def r2(real, pred):
    real_mean = real.mean()
    ss_res = ((real - pred) ** 2).sum()
    ss_tot = ((real - real_mean) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0

resultados = []
for cat in sorted(df_eval['cat_modelo'].unique()):
    sub = df_eval[df_eval['cat_modelo'] == cat]
    if len(sub) < 30:
        continue
    real = sub['receita'].values
    pred_manual = sub['previsao_manual'].values
    resultados.append({
        'categoria': cat,
        'n_dias': len(sub),
        'receita_media_real': round(real.mean(), 2),
        'receita_media_pred': round(pred_manual.mean(), 2),
        'rmse_manual': round(rmse(real, pred_manual), 2),
        'mape_manual_%': round(mape(real, pred_manual), 1),
        'r2_manual': round(r2(real, pred_manual), 3),
    })

df_metr = pd.DataFrame(resultados)
df_metr = df_metr.sort_values('mape_manual_%')

print()
print("=" * 80)
print("DESEMPENHO DO FORECASTER MANUAL (fator_dia × turno × mês × clima)")
print("=" * 80)
print()
print(f"{'Categoria':<22s} {'N':>5s} {'média real':>11s} {'média pred':>11s} {'MAPE':>7s} {'R²':>7s}")
print('-' * 80)
for _, r in df_metr.iterrows():
    print(f"  {r['categoria']:<20s} {int(r['n_dias']):>5d} "
          f"{r['receita_media_real']:>10.1f}  {r['receita_media_pred']:>10.1f}  "
          f"{r['mape_manual_%']:>6.1f}% {r['r2_manual']:>7.3f}")

mape_medio = df_metr['mape_manual_%'].mean()
r2_medio = df_metr['r2_manual'].mean()
print()
print(f"MAPE médio:  {mape_medio:.1f}%")
print(f"R² médio:    {r2_medio:.3f}")

# ── Naive baseline (média móvel 28 dias) ──────────────────────────────────

print()
print("=" * 80)
print("BASELINE: previsão = média dos últimos 28 dias por categoria")
print("=" * 80)
print()

resultados_naive = []
for cat in df_eval['cat_modelo'].unique():
    sub = df_eval[df_eval['cat_modelo'] == cat].sort_values('data').copy()
    sub['ma28'] = sub['receita'].shift(1).rolling(28, min_periods=1).mean()
    sub = sub.dropna(subset=['ma28'])
    if len(sub) < 30:
        continue
    real = sub['receita'].values
    pred = sub['ma28'].values
    resultados_naive.append({
        'categoria': cat,
        'mape_naive_%': round(mape(real, pred), 1),
        'r2_naive': round(r2(real, pred), 3),
    })

df_naive = pd.DataFrame(resultados_naive)
print(f"MAPE médio:  {df_naive['mape_naive_%'].mean():.1f}%")
print(f"R² médio:    {df_naive['r2_naive'].mean():.3f}")

# ── ML simples (Ridge com features manuais) ──────────────────────────────

print()
print("=" * 80)
print("ML SIMPLES: Ridge Regression com features [dia, mês, temp, lag-1, lag-7]")
print("=" * 80)
print()

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    SK_OK = True
except ImportError:
    SK_OK = False
    print("⚠ scikit-learn não disponível, pulando ML")

if SK_OK:
    resultados_ml = []
    for cat in df_eval['cat_modelo'].unique():
        sub = df_eval[df_eval['cat_modelo'] == cat].sort_values('data').copy()
        sub['lag1'] = sub['receita'].shift(1)
        sub['lag7'] = sub['receita'].shift(7)
        sub['lag28'] = sub['receita'].shift(28)
        sub = sub.dropna()
        if len(sub) < 60:
            continue
        # Features: dia_sem one-hot, mes one-hot, temp_norm, lag1, lag7, lag28
        X = pd.concat([
            pd.get_dummies(sub['dia_sem'], prefix='dia').astype(float),
            pd.get_dummies(sub['mes'], prefix='mes').astype(float),
            sub[['temp_norm', 'lag1', 'lag7', 'lag28']],
        ], axis=1).values
        y = sub['receita'].values

        # Split temporal 80/20
        n_train = int(len(sub) * 0.8)
        X_tr, X_ts = X[:n_train], X[n_train:]
        y_tr, y_ts = y[:n_train], y[n_train:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_ts_s = scaler.transform(X_ts)

        model = Ridge(alpha=1.0)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_ts_s).clip(0, None)

        resultados_ml.append({
            'categoria': cat,
            'mape_ml_%': round(mape(y_ts, pred), 1),
            'r2_ml': round(r2(y_ts, pred), 3),
            'n_test': len(y_ts),
        })

    df_ml = pd.DataFrame(resultados_ml)
    print(f"MAPE médio:  {df_ml['mape_ml_%'].mean():.1f}%")
    print(f"R² médio:    {df_ml['r2_ml'].mean():.3f}")

# ── Comparação final ─────────────────────────────────────────────────────

if SK_OK:
    final = df_metr[['categoria', 'mape_manual_%', 'r2_manual']].merge(
        df_naive[['categoria', 'mape_naive_%', 'r2_naive']], on='categoria'
    ).merge(
        df_ml[['categoria', 'mape_ml_%', 'r2_ml']], on='categoria'
    )
    final['melhoria_ml_vs_manual_%'] = (
        (final['mape_manual_%'] - final['mape_ml_%'])
         / final['mape_manual_%'].clip(0.1) * 100
    ).round(1)
    final.to_csv(RESULTS / 'comparacao_forecasters.csv',
                  index=False, encoding='utf-8')

    print()
    print("=" * 80)
    print("COMPARAÇÃO FINAL — MAPE por modelo, por categoria")
    print("=" * 80)
    print()
    print(f"{'Categoria':<22s} {'Manual':>8s} {'Naive':>8s} {'ML Ridge':>9s} {'Δ ML':>7s}")
    print('-' * 60)
    for _, r in final.sort_values('mape_manual_%').iterrows():
        print(f"  {r['categoria']:<20s} {r['mape_manual_%']:>6.1f}%  "
              f"{r['mape_naive_%']:>6.1f}%  {r['mape_ml_%']:>7.1f}%  "
              f"{r['melhoria_ml_vs_manual_%']:>+5.1f}%")

    print()
    print(f"Médias gerais:")
    print(f"  Manual atual:  {df_metr['mape_manual_%'].mean():.1f}% MAPE  R²={df_metr['r2_manual'].mean():.3f}")
    print(f"  Naive (MA28):  {df_naive['mape_naive_%'].mean():.1f}% MAPE  R²={df_naive['r2_naive'].mean():.3f}")
    print(f"  ML Ridge:      {df_ml['mape_ml_%'].mean():.1f}% MAPE  R²={df_ml['r2_ml'].mean():.3f}")

    melhoria_pct = (df_metr['mape_manual_%'].mean()
                     - df_ml['mape_ml_%'].mean()) / df_metr['mape_manual_%'].mean() * 100
    print()
    print(f"ML reduz MAPE em {melhoria_pct:.1f}% vs Manual")
    if melhoria_pct > 20:
        print("✓ Vale a pena implementar V12 com Forecaster ML")
    else:
        print("✗ Forecaster Manual atual já está bom — V12 traria ganho marginal")

print()
print(f"✓ Salvo em {RESULTS / 'comparacao_forecasters.csv'}")
