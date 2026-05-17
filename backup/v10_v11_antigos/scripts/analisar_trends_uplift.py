"""Cruza Google Trends × calendário comercial para extrair uplift real
(de busca, não vendas — mas é o melhor prior público disponível).

Para cada par (termo, evento comercial):
- Pega a média do índice na janela [-pre_dias, +pos_dias] em torno da data
- Compara com baseline (resto do ano, mesmo mês excluindo a janela)
- Uplift = janela / baseline

Saída: data/priors_externos/uplift_trends_por_evento.csv
"""
import io
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
TRENDS_DIR = ROOT / 'data' / 'priors_externos' / 'google_trends'
OUT = ROOT / 'data' / 'priors_externos'

# Carregar índice de termos disponíveis
indice = pd.read_csv(TRENDS_DIR / '_indice.csv')
termos_ok = indice['termo'].tolist()
print(f"Termos disponíveis: {termos_ok}")
print()

# Carregar calendário comercial (apenas eventos não-rotineiros)
cal = pd.read_csv(ROOT / 'data' / 'calendario_comercial.csv',
                  parse_dates=['data'])
cal_eventos = cal[cal['tipo_evento'].isin(
    ['data_comercial', 'evento_local', 'evento_esportivo'])].copy()

# Mapa termo → categorias modelo (para matching)
mapa_termo = dict(zip(indice['termo'], indice['categoria']))

# Quais termos batem em quais eventos
# Para cada evento, identificar termos relacionados via categorias_afetadas
def termos_para_evento(categorias_str):
    cats = set(categorias_str.split(';'))
    matched = []
    for termo, cat in mapa_termo.items():
        if cat in cats or 'todas' in cats:
            matched.append(termo)
    return matched


# ── Calcular uplift por (termo, evento, ano) ────────────────────────────────

resultados = []

for termo in termos_ok:
    nome_arq = (termo.replace(' ', '_').replace('ã', 'a')
                .replace('é', 'e').replace('ê', 'e'))
    serie = pd.read_csv(TRENDS_DIR / f'{nome_arq}.csv', parse_dates=['data'])
    serie['data'] = pd.to_datetime(serie['data'])

    for _, ev in cal_eventos.iterrows():
        cats_evento = set(ev['categorias_afetadas'].split(';'))
        if mapa_termo[termo] not in cats_evento and 'todas' not in cats_evento:
            continue

        # Janela em torno da data
        data_ev = ev['data']
        if not isinstance(data_ev, pd.Timestamp):
            data_ev = pd.Timestamp(data_ev)

        pre = int(ev['janela_pre_dias'])
        pos = int(ev['janela_pos_dias'])
        # Mínimo: 7 dias antes (Trends é semanal)
        pre_efetivo = max(pre, 7)

        inicio_janela = data_ev - timedelta(days=pre_efetivo)
        fim_janela = data_ev + timedelta(days=pos)

        janela = serie[(serie['data'] >= inicio_janela) &
                       (serie['data'] <= fim_janela)]

        # Baseline: mesmo ano, fora da janela
        ano_serie = serie[serie['data'].dt.year == data_ev.year]
        baseline = ano_serie[(ano_serie['data'] < inicio_janela - timedelta(days=14)) |
                             (ano_serie['data'] > fim_janela + timedelta(days=14))]

        if len(janela) == 0 or len(baseline) == 0:
            continue

        media_janela = janela['indice'].mean()
        media_baseline = baseline['indice'].mean()
        if media_baseline < 1:
            continue

        uplift = media_janela / media_baseline

        resultados.append({
            'termo': termo,
            'categoria_modelo': mapa_termo[termo],
            'evento': ev['nome_evento'],
            'ano': data_ev.year,
            'data': data_ev.date(),
            'indice_janela': round(media_janela, 1),
            'indice_baseline': round(media_baseline, 1),
            'uplift_medido': round(uplift, 3),
            'uplift_prior_calendario': ev['uplift_prior'],
        })

df = pd.DataFrame(resultados)
df = df.sort_values(['evento', 'termo', 'ano']).reset_index(drop=True)
df.to_csv(OUT / 'uplift_trends_por_evento.csv', index=False, encoding='utf-8')

# ── Agregação por (categoria, evento): média entre anos ─────────────────────

agg = (df.groupby(['categoria_modelo', 'evento'])
         .agg(uplift_medio=('uplift_medido', 'mean'),
              uplift_std=('uplift_medido', 'std'),
              n_anos=('uplift_medido', 'count'),
              prior_calendario=('uplift_prior_calendario', 'first'))
         .reset_index())
agg['uplift_medio'] = agg['uplift_medio'].round(3)
agg['uplift_std'] = agg['uplift_std'].round(3)
agg = agg.sort_values('uplift_medio', ascending=False)
agg.to_csv(OUT / 'uplift_trends_agregado.csv', index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print(f"✓ Análise concluída")
print(f"  data/priors_externos/uplift_trends_por_evento.csv: {len(df)} linhas")
print(f"  data/priors_externos/uplift_trends_agregado.csv:   {len(agg)} linhas")
print()
print("TOP 15 maiores uplifts MEDIDOS (Google Trends, BR):")
print()
print(f"{'Categoria':<14s} {'Evento':<35s} {'Medido':>7s} {'Prior':>7s} {'N anos':>7s}")
print('-' * 80)
for _, row in agg.head(15).iterrows():
    print(f"{row['categoria_modelo']:<14s} {row['evento']:<35s} "
          f"{row['uplift_medio']:>6.2f}× "
          f"{row['prior_calendario']:>6.2f}× "
          f"{row['n_anos']:>5d}")

print()
print("Casos onde MEDIDO discorda do PRIOR (|diff| > 0.5):")
agg['diff'] = (agg['uplift_medio'] - agg['prior_calendario']).abs()
divergentes = agg[agg['diff'] > 0.5].sort_values('diff', ascending=False)
if len(divergentes) > 0:
    for _, row in divergentes.head(10).iterrows():
        print(f"  {row['categoria_modelo']:<14s} {row['evento']:<35s} "
              f"medido {row['uplift_medio']:.2f} vs prior {row['prior_calendario']:.2f}")
else:
    print("  (nenhum)")
