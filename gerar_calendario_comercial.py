"""Gera calendário comercial brasileiro 2020-2027.

Inclui:
- Feriados oficiais (pacote holidays)
- Datas comerciais nacionais (Namorados, Mães, Pais, Crianças, Black Friday,
  Páscoa, Carnaval, Natal, Ano Novo, Dia do Consumidor, Dia da Mulher)
- Eventos esportivos (Copa do Mundo, Eurocopa, Olimpíadas, finais Brasileirão)
- Datas locais (Aniversário de Barueri, Aniversário de São Paulo)

Para cada data: tipo, janela de impacto (dias antes/depois), categorias
afetadas e prior de uplift baseado em literatura/heurística.

Saída: data/calendario_comercial.csv
"""
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import holidays

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

ANO_INICIO = 2020
ANO_FIM = 2027

# ── Feriados nacionais (pacote holidays) ────────────────────────────────────

br = holidays.Brazil(years=range(ANO_INICIO, ANO_FIM + 1), subdiv='SP')
feriados_oficiais = []
for d, nome in sorted(br.items()):
    feriados_oficiais.append({
        'data': d,
        'tipo_evento': 'feriado_oficial',
        'nome_evento': nome,
        'janela_pre_dias': 0,
        'janela_pos_dias': 1,
        'intensidade': 'media',
        'categorias_afetadas': 'todas',
        'uplift_prior': 1.1,
        'fonte': 'holidays-br',
    })

# ── Datas comerciais brasileiras ────────────────────────────────────────────

def segundo_domingo(ano, mes):
    """Retorna o segundo domingo do mês."""
    d = date(ano, mes, 1)
    primeiro_dom = d + timedelta(days=(6 - d.weekday()) % 7)
    return primeiro_dom + timedelta(days=7)


def ultima_sexta(ano, mes):
    """Última sexta-feira do mês."""
    if mes == 12:
        prox = date(ano + 1, 1, 1)
    else:
        prox = date(ano, mes + 1, 1)
    ultimo_dia = prox - timedelta(days=1)
    return ultimo_dia - timedelta(days=(ultimo_dia.weekday() - 4) % 7)


datas_comerciais = []
for ano in range(ANO_INICIO, ANO_FIM + 1):
    # Dia da Mulher — 8/03
    datas_comerciais.append({
        'data': date(ano, 3, 8),
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Dia Internacional da Mulher',
        'janela_pre_dias': 5,
        'janela_pos_dias': 0,
        'intensidade': 'media',
        'categorias_afetadas': 'chocolate;vinho;espumante;flores',
        'uplift_prior': 1.6,
        'fonte': 'manual',
    })

    # Dia do Consumidor — 15/03 ("Black Friday brasileira de março")
    datas_comerciais.append({
        'data': date(ano, 3, 15),
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Dia do Consumidor',
        'janela_pre_dias': 3,
        'janela_pos_dias': 1,
        'intensidade': 'baixa',
        'categorias_afetadas': 'todas',
        'uplift_prior': 1.2,
        'fonte': 'manual',
    })

    # Dia das Mães — segundo domingo de maio
    dia_maes = segundo_domingo(ano, 5)
    datas_comerciais.append({
        'data': dia_maes,
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Dia das Mães',
        'janela_pre_dias': 10,
        'janela_pos_dias': 0,
        'intensidade': 'muito_alta',
        'categorias_afetadas': 'chocolate;vinho;espumante;perfume',
        'uplift_prior': 2.2,
        'fonte': 'manual',
    })

    # Dia dos Namorados — 12/06
    datas_comerciais.append({
        'data': date(ano, 6, 12),
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Dia dos Namorados',
        'janela_pre_dias': 7,
        'janela_pos_dias': 0,
        'intensidade': 'muito_alta',
        'categorias_afetadas': 'chocolate;vinho;espumante;cerveja_premium',
        'uplift_prior': 2.5,
        'fonte': 'manual',
    })

    # Dia dos Pais — segundo domingo de agosto
    dia_pais = segundo_domingo(ano, 8)
    datas_comerciais.append({
        'data': dia_pais,
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Dia dos Pais',
        'janela_pre_dias': 10,
        'janela_pos_dias': 0,
        'intensidade': 'alta',
        'categorias_afetadas': 'cerveja;whisky;cachaca;vinho_tinto;snack',
        'uplift_prior': 1.9,
        'fonte': 'manual',
    })

    # Dia das Crianças — 12/10
    datas_comerciais.append({
        'data': date(ano, 10, 12),
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Dia das Crianças',
        'janela_pre_dias': 7,
        'janela_pos_dias': 0,
        'intensidade': 'alta',
        'categorias_afetadas': 'chocolate;salgadinho;refrigerante;sorvete;suco',
        'uplift_prior': 1.7,
        'fonte': 'manual',
    })

    # Black Friday — última sexta de novembro
    bf = ultima_sexta(ano, 11)
    datas_comerciais.append({
        'data': bf,
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Black Friday',
        'janela_pre_dias': 3,
        'janela_pos_dias': 3,
        'intensidade': 'muito_alta',
        'categorias_afetadas': 'todas',
        'uplift_prior': 1.5,
        'fonte': 'manual',
    })
    # Cyber Monday — segunda após Black Friday
    datas_comerciais.append({
        'data': bf + timedelta(days=3),
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Cyber Monday',
        'janela_pre_dias': 0,
        'janela_pos_dias': 1,
        'intensidade': 'media',
        'categorias_afetadas': 'todas',
        'uplift_prior': 1.3,
        'fonte': 'manual',
    })

    # Natal — pico de consumo nas véspera (vinho, espumante, panettone)
    datas_comerciais.append({
        'data': date(ano, 12, 24),
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Véspera de Natal',
        'janela_pre_dias': 10,
        'janela_pos_dias': 0,
        'intensidade': 'muito_alta',
        'categorias_afetadas': 'vinho;espumante;chocolate;cerveja;refrigerante;panettone',
        'uplift_prior': 2.8,
        'fonte': 'manual',
    })

    # Ano Novo — réveillon
    datas_comerciais.append({
        'data': date(ano, 12, 31),
        'tipo_evento': 'data_comercial',
        'nome_evento': 'Réveillon',
        'janela_pre_dias': 5,
        'janela_pos_dias': 0,
        'intensidade': 'muito_alta',
        'categorias_afetadas': 'champagne;espumante;cerveja;gelo;refrigerante;energetico',
        'uplift_prior': 3.0,
        'fonte': 'manual',
    })

    # Aniversário de Barueri — 26/03
    datas_comerciais.append({
        'data': date(ano, 3, 26),
        'tipo_evento': 'evento_local',
        'nome_evento': 'Aniversário de Barueri',
        'janela_pre_dias': 0,
        'janela_pos_dias': 0,
        'intensidade': 'baixa',
        'categorias_afetadas': 'cerveja;refrigerante;snack',
        'uplift_prior': 1.15,
        'fonte': 'manual',
    })

    # Aniversário de São Paulo — 25/01
    datas_comerciais.append({
        'data': date(ano, 1, 25),
        'tipo_evento': 'evento_local',
        'nome_evento': 'Aniversário de São Paulo',
        'janela_pre_dias': 0,
        'janela_pos_dias': 0,
        'intensidade': 'baixa',
        'categorias_afetadas': 'cerveja;refrigerante',
        'uplift_prior': 1.1,
        'fonte': 'manual',
    })

# ── Eventos esportivos (datas conhecidas) ───────────────────────────────────

eventos_esportivos = [
    # Copa do Mundo 2022 (Qatar — Brasil foi nas oitavas a 09/12, eliminado nas quartas 09/12)
    {'data': date(2022, 11, 24), 'nome': 'Copa 2022 — Brasil x Sérvia',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.8},
    {'data': date(2022, 11, 28), 'nome': 'Copa 2022 — Brasil x Suíça',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.5},
    {'data': date(2022, 12, 2), 'nome': 'Copa 2022 — Brasil x Camarões',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.3},
    {'data': date(2022, 12, 5), 'nome': 'Copa 2022 — Brasil x Coréia',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.8},
    {'data': date(2022, 12, 9), 'nome': 'Copa 2022 — Brasil x Croácia (eliminação)',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.9},

    # Copa do Mundo 2026 (USA/MEX/CAN — 11/06 a 19/07/2026)
    {'data': date(2026, 6, 11), 'nome': 'Copa 2026 — Abertura',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 1.8},
    # Datas de jogos do Brasil ainda não confirmadas — marcar período como elevado
    {'data': date(2026, 6, 15), 'nome': 'Copa 2026 — Estreia provável Brasil',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.5},
    {'data': date(2026, 6, 20), 'nome': 'Copa 2026 — Fase de grupos Brasil',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.3},
    {'data': date(2026, 6, 25), 'nome': 'Copa 2026 — Fase de grupos Brasil',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.5},
    {'data': date(2026, 7, 19), 'nome': 'Copa 2026 — Final',
     'categorias': 'cerveja;snack;refrigerante;gelo', 'uplift': 2.0},
]

eventos_normalizados = [{
    'data': e['data'],
    'tipo_evento': 'evento_esportivo',
    'nome_evento': e['nome'],
    'janela_pre_dias': 0,
    'janela_pos_dias': 0,
    'intensidade': 'alta' if e['uplift'] >= 2.5 else 'media',
    'categorias_afetadas': e['categorias'],
    'uplift_prior': e['uplift'],
    'fonte': 'manual',
} for e in eventos_esportivos]

# ── Consolidar ──────────────────────────────────────────────────────────────

df = pd.DataFrame(feriados_oficiais + datas_comerciais + eventos_normalizados)
df = df.sort_values('data').reset_index(drop=True)
df['ano'] = pd.to_datetime(df['data']).dt.year
df['mes'] = pd.to_datetime(df['data']).dt.month
df['dia'] = pd.to_datetime(df['data']).dt.day
df['dia_semana'] = pd.to_datetime(df['data']).dt.dayofweek

# Reorganizar colunas
cols = ['data', 'ano', 'mes', 'dia', 'dia_semana',
        'tipo_evento', 'nome_evento',
        'janela_pre_dias', 'janela_pos_dias',
        'intensidade', 'categorias_afetadas', 'uplift_prior', 'fonte']
df = df[cols]

df.to_csv(DATA / 'calendario_comercial.csv', index=False, encoding='utf-8')

# ── Versão expandida: 1 linha por dia × evento (com janela) ────────────────

linhas_expandidas = []
for _, row in df.iterrows():
    base = pd.to_datetime(row['data']).date()
    for offset in range(-row['janela_pre_dias'], row['janela_pos_dias'] + 1):
        d = base + timedelta(days=offset)
        # Fator de proximidade: 1.0 no dia, decai linearmente
        if row['janela_pre_dias'] + row['janela_pos_dias'] > 0:
            dist = abs(offset)
            max_janela = max(row['janela_pre_dias'], row['janela_pos_dias'])
            proximidade = max(0.3, 1.0 - dist / (max_janela + 1))
        else:
            proximidade = 1.0
        uplift_dia = 1 + (row['uplift_prior'] - 1) * proximidade
        linhas_expandidas.append({
            'data': d,
            'evento': row['nome_evento'],
            'tipo': row['tipo_evento'],
            'dias_para_evento': -offset,
            'categorias': row['categorias_afetadas'],
            'uplift_dia': round(uplift_dia, 3),
        })

df_exp = pd.DataFrame(linhas_expandidas)
df_exp = df_exp.sort_values(['data', 'uplift_dia'], ascending=[True, False])
df_exp.to_csv(DATA / 'calendario_comercial_expandido.csv',
              index=False, encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print(f"✓ Calendário comercial gerado ({ANO_INICIO}-{ANO_FIM})")
print(f"  data/calendario_comercial.csv:           {len(df):4d} eventos")
print(f"  data/calendario_comercial_expandido.csv: {len(df_exp):4d} linhas (janelas)")
print()
print("Distribuição por tipo:")
for tipo, count in df['tipo_evento'].value_counts().items():
    print(f"  {tipo:25s} {count:4d}")
print()
print("Próximos 10 eventos a partir de hoje (2026-05-11):")
hoje = pd.Timestamp(2026, 5, 11)
proximos = df[pd.to_datetime(df['data']) >= hoje].head(10)
for _, row in proximos.iterrows():
    d = pd.to_datetime(row['data']).strftime('%d/%m/%Y')
    print(f"  {d}  {row['nome_evento']:<45s} uplift {row['uplift_prior']:.2f}")
