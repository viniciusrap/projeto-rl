"""Baixa série mensal do IBGE PMC (Pesquisa Mensal do Comércio) via SIDRA.

PMC = índice mensal de volume de vendas no varejo brasileiro.
Útil como prior de sazonalidade macro — qual mês historicamente vende mais
no Brasil inteiro.

Tabela 8881 (PMC varejo ampliado) — variáveis 7169 (bruto) e 7170 (ajustado
sazonal). A razão entre bruto e ajustado revela o componente sazonal.

Saída: data/priors_externos/ibge_pmc/
  - pmc_varejo_ampliado.csv      (série mensal completa 2003-2025)
  - sazonalidade_mensal.csv      (fator_mes por mês)
"""
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
OUT = ROOT / 'data' / 'priors_externos' / 'ibge_pmc'
OUT.mkdir(parents=True, exist_ok=True)

# ── Baixa duas variáveis da tabela 8881 ────────────────────────────────────

# Variável 7169 = PMC Número-Índice (sem ajuste sazonal)
# Variável 7170 = PMC Número-Índice (com ajuste sazonal)
# c11046 56736 = Índice de VOLUME de vendas (não nominal, já descontando inflação)

VARIAVEIS = {
    '7169': 'bruto',     # com efeito sazonal
    '7170': 'ajustado',  # sem efeito sazonal (suavizado)
}

dfs = {}
for cod, nome in VARIAVEIS.items():
    arquivo = OUT / f'pmc_varejo_{nome}.csv'
    if arquivo.exists():
        print(f"  pmc_varejo_{nome}: já existe, pulando")
        dfs[nome] = pd.read_csv(arquivo)
        continue
    print(f"  Baixando PMC varejo {nome}... ", end='', flush=True)
    url = (f'https://apisidra.ibge.gov.br/values/t/8881/n1/all/'
           f'v/{cod}/p/all/c11046/56736')
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dados = r.json()
        rows = []
        for item in dados[1:]:
            periodo = item.get('D3C', '')
            valor = item.get('V', '...')
            if valor in ('...', '-') or not periodo:
                continue
            try:
                rows.append({
                    'periodo': periodo,
                    'ano': int(periodo[:4]),
                    'mes': int(periodo[4:6]),
                    'indice': float(valor),
                })
            except (ValueError, TypeError):
                continue
        df = pd.DataFrame(rows).sort_values('periodo').reset_index(drop=True)
        df.to_csv(arquivo, index=False, encoding='utf-8')
        dfs[nome] = df
        print(f'OK ({len(df)} meses)')
        time.sleep(0.5)
    except Exception as e:
        print(f'FALHA: {str(e)[:80]}')
        dfs[nome] = None

if dfs.get('bruto') is None or dfs.get('ajustado') is None:
    print("✗ Não foi possível baixar PMC")
    sys.exit(1)

# ── Cruzar bruto vs ajustado → fator sazonal mensal ────────────────────────

bruto = dfs['bruto'].rename(columns={'indice': 'indice_bruto'})
ajustado = dfs['ajustado'].rename(columns={'indice': 'indice_ajustado'})
joined = bruto.merge(ajustado, on=['periodo', 'ano', 'mes'])
joined['fator_sazonal'] = joined['indice_bruto'] / joined['indice_ajustado']
joined.to_csv(OUT / 'pmc_varejo_ampliado.csv', index=False, encoding='utf-8')

# Sazonalidade: média do fator_sazonal por mês (entre anos)
saz = (joined.groupby('mes')['fator_sazonal']
              .agg(['mean', 'std', 'count'])
              .reset_index()
              .rename(columns={'mean': 'fator_medio',
                              'std': 'desvio_padrao',
                              'count': 'n_anos'}))
saz['fator_medio'] = saz['fator_medio'].round(3)
saz['desvio_padrao'] = saz['desvio_padrao'].round(3)
saz['variacao_pct'] = ((saz['fator_medio'] - 1) * 100).round(1)
saz.to_csv(OUT / 'sazonalidade_mensal.csv', index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print(f"✓ Série PMC baixada: {len(joined)} meses ({joined['periodo'].min()} → {joined['periodo'].max()})")
print()
print("Sazonalidade mensal do varejo ampliado brasileiro (média histórica):")
print()
nomes_mes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
for _, row in saz.iterrows():
    fator = row['fator_medio']
    var = row['variacao_pct']
    barra = '█' * int(abs(var))
    sinal = '+' if var > 0 else ''
    print(f"  {nomes_mes[int(row['mes']) - 1]}: {fator:.3f} "
          f"({sinal}{var:5.1f}%) {barra}")

print()
print("Insights:")
maior = saz.loc[saz['fator_medio'].idxmax()]
menor = saz.loc[saz['fator_medio'].idxmin()]
print(f"  - Pico do ano:  {nomes_mes[int(maior['mes']) - 1]} (+{maior['variacao_pct']}% vs média)")
print(f"  - Vale do ano:  {nomes_mes[int(menor['mes']) - 1]} ({menor['variacao_pct']}% vs média)")
