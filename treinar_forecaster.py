"""V12 — Forecaster ML por categoria.

Treina HistGradientBoostingRegressor (sklearn) por categoria modelo. Features:
- Calendário: dia_sem, mes, dia_mes
- Clima: temp_norm
- Lags: receita lag1, lag7, lag28
- Eventos: is_event, days_to_event, tipo_pico_pre, tipo_pico_no_dia
- Categoria: idx (one-hot já é por modelo independente, não precisa)

Train/val split temporal 80/20.

Saída: results/v12/forecasters.pkl — dict por categoria com {model, scaler?, feature_names, mape_val, r2_val}
       results/v12/forecaster_metricas.csv — relatório
"""
import io
import json
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
V12 = ROOT / 'results' / 'v12'
V12.mkdir(parents=True, exist_ok=True)


def mape(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = real > 0.5
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask]) * 100))


def r2_score(real, pred):
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = float(((real - pred) ** 2).sum())
    ss_tot = float(((real - real.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── Carregar tudo ──────────────────────────────────────────────────────────

print("Carregando dados…")
with open(DATA / 'calibracao_v2.json', encoding='utf-8') as f:
    cal = json.load(f)

vendas = pd.read_csv(DATA / 'venda_por_dia_parseado.csv', parse_dates=['data'])
temp = pd.read_csv(DATA / 'temperatura_historica.csv', parse_dates=['data'])

MAPA = cal['mapa_categoria_posto_para_modelo']
vendas['cat_modelo'] = vendas['categoria'].map(MAPA)
vendas = vendas[vendas['cat_modelo'].notna()].copy()

# Agregar dia × categoria (somando 3 turnos)
diario = (vendas.groupby([vendas['data'].dt.date, 'cat_modelo'])
                 .agg(receita=('valor_venda', 'sum'))
                 .reset_index())
diario['data'] = pd.to_datetime(diario['data'])

# Temperatura
tmin = cal['clima_params']['temp_min']
tmax = cal['clima_params']['temp_max']
temp['temp_norm'] = (temp['temp_max'] - tmin) / (tmax - tmin)
diario = diario.merge(temp[['data', 'temp_norm']], on='data', how='left')
diario = diario.dropna(subset=['temp_norm'])

# Calendar features
diario['dia_sem'] = diario['data'].dt.dayofweek
diario['mes'] = diario['data'].dt.month - 1
diario['dia_mes'] = diario['data'].dt.day

# ── Features de evento comercial ───────────────────────────────────────────

# Para cada data, calcula:
# - is_event: 1 se está dentro de alguma janela de evento
# - days_to_event: dias até o próximo evento (clipped 30)
# - tipo_pico_pre, tipo_pico_no_dia: one-hot

eventos = cal['calendario_comercial']
ev_datas = sorted({ev['data'] for ev in eventos})
ev_lookup = {ev['data']: ev for ev in eventos}

def features_evento(d):
    dd = d.date() if hasattr(d, 'date') else d
    em_janela = False
    tipo_pico_pre = False
    tipo_pico_no_dia = False
    for ev in eventos:
        ev_d = date.fromisoformat(ev['data'])
        pre = int(ev.get('janela_pre_dias', 0))
        pos = int(ev.get('janela_pos_dias', 0))
        jini = ev_d - timedelta(days=pre)
        jfim = ev_d + timedelta(days=pos)
        if jini <= dd <= jfim:
            em_janela = True
            tp = ev.get('tipo_pico', '')
            if tp in ('pre', 'ambos'):
                tipo_pico_pre = True
            if tp in ('no_dia', 'ambos'):
                tipo_pico_no_dia = True
            break
    # days to next event
    future = [date.fromisoformat(ev['data']) for ev in eventos
              if date.fromisoformat(ev['data']) >= dd]
    days_to = min((e - dd).days for e in future) if future else 30
    return em_janela, min(days_to, 30), tipo_pico_pre, tipo_pico_no_dia

print("Calculando features de evento (cache em RAM)…")
# Vetorize por data única
datas_unicas = diario['data'].drop_duplicates().sort_values()
feat_ev = {d: features_evento(d) for d in datas_unicas}
diario['is_event'] = diario['data'].map(lambda d: feat_ev[d][0]).astype(float)
diario['days_to_event'] = diario['data'].map(lambda d: feat_ev[d][1]).astype(float)
diario['tipo_pico_pre'] = diario['data'].map(lambda d: feat_ev[d][2]).astype(float)
diario['tipo_pico_no_dia'] = diario['data'].map(lambda d: feat_ev[d][3]).astype(float)

# ── Train forecaster por categoria ─────────────────────────────────────────

CATEGORIAS_MODELO = [c['categoria'] for c in cal['categorias']]
FEATURES = [
    'dia_sem', 'mes', 'dia_mes', 'temp_norm',
    'lag1', 'lag7', 'lag28',
    'is_event', 'days_to_event',
    'tipo_pico_pre', 'tipo_pico_no_dia',
]

forecasters = {}
metricas = []

for cat in CATEGORIAS_MODELO:
    sub = diario[diario['cat_modelo'] == cat].sort_values('data').copy()
    if len(sub) < 100:
        print(f"  {cat:25s}: SKIPPED ({len(sub)} registros)")
        continue
    sub['lag1'] = sub['receita'].shift(1)
    sub['lag7'] = sub['receita'].shift(7)
    sub['lag28'] = sub['receita'].shift(28)
    sub = sub.dropna(subset=['lag1', 'lag7', 'lag28'])
    if len(sub) < 60:
        print(f"  {cat:25s}: SKIPPED após lags ({len(sub)} registros)")
        continue

    X = sub[FEATURES].values
    y = sub['receita'].values

    # Split temporal 80/20
    n_train = int(len(sub) * 0.8)
    X_tr, X_val = X[:n_train], X[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_tr_s, y_tr)
    pred_tr = model.predict(X_tr_s).clip(0, None)
    pred_val = model.predict(X_val_s).clip(0, None)

    m_tr = mape(y_tr, pred_tr)
    m_val = mape(y_val, pred_val)
    r2_val = r2_score(y_val, pred_val)

    forecasters[cat] = {
        'model': model,
        'scaler': scaler,
        'feature_names': FEATURES,
        'media_receita_train': float(y_tr.mean()),
        'desvio_receita_train': float(y_tr.std()),
        'mape_train': m_tr,
        'mape_val': m_val,
        'r2_val': r2_val,
        'n_train': len(X_tr),
        'n_val': len(X_val),
    }
    metricas.append({
        'categoria': cat,
        'n_train': len(X_tr),
        'n_val': len(X_val),
        'mape_train_%': round(m_tr, 1),
        'mape_val_%': round(m_val, 1),
        'r2_val': round(r2_val, 3),
        'receita_media': round(y_tr.mean(), 2),
    })

    print(f"  {cat:25s}: MAPE train={m_tr:5.1f}% val={m_val:5.1f}% R²={r2_val:+.3f}")

# ── Persistir ─────────────────────────────────────────────────────────────

with open(V12 / 'forecasters.pkl', 'wb') as f:
    pickle.dump({
        'forecasters': forecasters,
        'features': FEATURES,
        'categorias': list(forecasters.keys()),
        'periodo_treino': (cal.get('periodos', {}).get('data_inicio_treino'),
                            cal.get('periodos', {}).get('data_fim_treino')),
    }, f)

df_met = pd.DataFrame(metricas).sort_values('mape_val_%')
df_met.to_csv(V12 / 'forecaster_metricas.csv', index=False, encoding='utf-8')

# ── Resumo ────────────────────────────────────────────────────────────────

print()
print("=" * 70)
print("FORECASTER V12 — sumário")
print("=" * 70)
print(f"  N categorias treinadas: {len(forecasters)}")
print(f"  MAPE val médio: {df_met['mape_val_%'].mean():.1f}%")
print(f"  R² val médio:   {df_met['r2_val'].mean():+.3f}")
print()
print("Top 5 melhores (menor MAPE val):")
for _, r in df_met.head(5).iterrows():
    print(f"  {r['categoria']:25s} MAPE val={r['mape_val_%']:5.1f}% R²={r['r2_val']:+.3f}")
print()
print("Top 5 piores:")
for _, r in df_met.tail(5).iterrows():
    print(f"  {r['categoria']:25s} MAPE val={r['mape_val_%']:5.1f}% R²={r['r2_val']:+.3f}")
print()
print(f"✓ Forecaster salvo em {V12 / 'forecasters.pkl'}")
print(f"✓ Métricas em {V12 / 'forecaster_metricas.csv'}")
