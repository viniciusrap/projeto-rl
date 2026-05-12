"""Coleta Google Trends Brasil como prior de sazonalidade.

Para cada termo, baixa a série semanal dos últimos 5 anos (relativa,
0-100). Os índices alimentam priors de uplift por categoria em torno
de datas comerciais.

API não-oficial — sujeita a rate-limit. Se falhar em um termo, segue.

Saída: data/priors_externos/google_trends/<termo>.csv
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pytrends.request import TrendReq

ROOT = Path(__file__).parent
OUT = ROOT / 'data' / 'priors_externos' / 'google_trends'
OUT.mkdir(parents=True, exist_ok=True)

# ── Termos a coletar ────────────────────────────────────────────────────────
# Cada entry: (termo_busca, categoria_modelo, datas_alvo)

TERMOS = [
    # Bebidas
    ('cerveja', 'cerveja', 'fim_ano;copa;carnaval'),
    ('vinho', 'vinho', 'maes;namorados;natal'),
    ('espumante', 'espumante', 'reveillon;namorados'),
    ('whisky', 'whisky', 'pais;natal'),
    ('energético', 'energetico', 'cotidiano'),
    ('refrigerante', 'refrigerante', 'verao;copa'),
    # Doces e snacks
    ('chocolate', 'chocolate', 'pascoa;namorados;maes'),
    ('panetone', 'panettone', 'natal'),
    ('sorvete', 'sorvete', 'verao'),
    ('salgadinho', 'snack', 'copa;fim_semana'),
    # Termos de presente (intenção)
    ('presente dia dos namorados', 'intencao_namorados', 'namorados'),
    ('presente dia das mães', 'intencao_maes', 'maes'),
    ('presente dia dos pais', 'intencao_pais', 'pais'),
    # Eventos
    ('black friday', 'evento_bf', 'black_friday'),
    ('copa do mundo', 'evento_copa', 'copa'),
]

# ── Coleta ──────────────────────────────────────────────────────────────────

# Timeframe: últimos 5 anos
TIMEFRAME = 'today 5-y'  # 5 anos rolantes
GEO = 'BR'

pytrends = TrendReq(hl='pt-BR', tz=180, timeout=(10, 25))

resultados = []
falhas = []

for termo, categoria, alvos in TERMOS:
    nome_arquivo_check = (termo.replace(' ', '_').replace('ã', 'a')
                          .replace('é', 'e').replace('ê', 'e'))
    if (OUT / f'{nome_arquivo_check}.csv').exists():
        print(f"  '{termo}': já coletado, pulando")
        continue
    print(f"  Buscando '{termo}'... ", end='', flush=True)
    try:
        pytrends.build_payload([termo], cat=0, timeframe=TIMEFRAME, geo=GEO, gprop='')
        df = pytrends.interest_over_time()
        if df.empty:
            print('vazio')
            falhas.append((termo, 'vazio'))
            continue
        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])
        df = df.rename(columns={termo: 'indice'})
        df['termo'] = termo
        df['categoria'] = categoria
        df['datas_alvo'] = alvos
        df = df.reset_index().rename(columns={'date': 'data'})

        nome_arquivo = (termo.replace(' ', '_').replace('ã', 'a')
                        .replace('é', 'e').replace('ê', 'e'))
        df.to_csv(OUT / f'{nome_arquivo}.csv', index=False, encoding='utf-8')

        resultados.append({
            'termo': termo,
            'categoria': categoria,
            'datas_alvo': alvos,
            'linhas': len(df),
            'indice_max': float(df['indice'].max()),
            'indice_min': float(df['indice'].min()),
            'indice_medio': round(float(df['indice'].mean()), 1),
            'arquivo': f'{nome_arquivo}.csv',
        })
        print(f'OK ({len(df)} pontos, range {df["indice"].min()}-{df["indice"].max()})')

        # Respeitar rate-limit (mais conservador após 429)
        time.sleep(8)
    except Exception as e:
        msg = str(e)[:80]
        print(f'FALHA: {msg}')
        falhas.append((termo, msg))
        # Em caso de 429, espera longa antes de tentar próximo
        if '429' in msg:
            print('  (rate-limit detectado, aguardando 60s)')
            time.sleep(60)
        else:
            time.sleep(10)

# ── Índice mestre ───────────────────────────────────────────────────────────

if resultados:
    indice = pd.DataFrame(resultados)
    indice.to_csv(OUT / '_indice.csv', index=False, encoding='utf-8')
    print()
    print(f"✓ {len(resultados)} séries coletadas com sucesso")
    if falhas:
        print(f"⚠ {len(falhas)} falharam:")
        for termo, motivo in falhas:
            print(f"  - {termo}: {motivo}")
    print()
    print("Resumo (termo → arquivo, range do índice):")
    for r in resultados:
        print(f"  {r['termo']:<30s} → {r['arquivo']:<35s} "
              f"({r['indice_min']:.0f}-{r['indice_max']:.0f}, "
              f"média {r['indice_medio']:.0f})")
else:
    print()
    print("✗ Nenhuma série coletada. Provável bloqueio de rate-limit do Google Trends.")
    print("  Tentar de novo em alguns minutos ou usar VPN.")
