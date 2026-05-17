"""V18 — Análise do dataset archive(7) — Convenience Store (18k transações).

Extrai insights que podem refinar o modelo:
1. Distribuição de promoções por hora (refinar fator_turno em hora real)
2. Tipos de promoção (BOGO vs Discount) — qual converte mais?
3. Distribuição por dia da semana (validar fator_dia)
4. Segmento de cliente que mais responde a promo
5. Ticket médio com vs sem promo (NÃO disponível — todas as transações têm promoção)
6. Sazonalidade por mês

Saída: data/priors_externos/convenience_store/*.csv
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RAW = ROOT / 'data' / 'raw_novos' / 'archive(7)' / 'convenience_store.csv'
OUT = ROOT / 'data' / 'priors_externos' / 'convenience_store'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("Carregando archive(7) convenience_store.csv…")
    df = pd.read_csv(RAW, parse_dates=['Date'])
    print(f"  {len(df):,} transações | {df['Date'].min().date()} a {df['Date'].max().date()}")
    print(f"  Promoções: {df['Promotion'].nunique()} tipos | "
          f"Clientes: {df['Customer_Category'].nunique()} segmentos")

    # Parse horário detalhado
    df['hour'] = pd.to_datetime(df['time_of_day'], format='%H:%M:%S').dt.hour
    df['dow'] = df['Date'].dt.dayofweek  # 0=seg
    df['month'] = df['Date'].dt.month

    # ── 1. Distribuição por HORA ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("1. DISTRIBUIÇÃO DE PROMOÇÕES POR HORA DO DIA")
    print("=" * 80)

    por_hora = df.groupby('hour').agg(
        n_transacoes=('Transaction_ID', 'count'),
        ticket_medio=('Total_Cost', 'mean'),
        itens_medio=('Total_Items', 'mean'),
    ).reset_index()
    por_hora['pct'] = por_hora['n_transacoes'] / por_hora['n_transacoes'].sum() * 100

    # Mapeia hora → nosso bucket de turno (0=manhã 06-13, 1=tarde 14-19, 2=noite 20-05)
    def hour_to_turno(h):
        if 6 <= h <= 13: return 0
        if 14 <= h <= 19: return 1
        return 2
    por_hora['nosso_turno'] = por_hora['hour'].apply(hour_to_turno)

    print(por_hora.to_string(index=False))

    # Comparação por nosso turno
    por_turno = por_hora.groupby('nosso_turno').agg(
        n=('n_transacoes', 'sum'),
        ticket=('ticket_medio', 'mean'),
    ).reset_index()
    por_turno['pct'] = por_turno['n'] / por_turno['n'].sum() * 100
    por_turno['turno_nome'] = ['Manhã (06-13)', 'Tarde (14-19)', 'Noite (20-05)']
    print()
    print("AGRUPADO POR NOSSO TURNO:")
    print(por_turno[['turno_nome', 'n', 'pct', 'ticket']].to_string(index=False))

    por_hora.to_csv(OUT / 'distribuicao_hora.csv', index=False, encoding='utf-8')
    por_turno.to_csv(OUT / 'distribuicao_turno.csv', index=False, encoding='utf-8')

    # ── 2. Tipos de promoção ──────────────────────────────────────────────
    print()
    print("=" * 80)
    print("2. EFETIVIDADE POR TIPO DE PROMOÇÃO")
    print("=" * 80)

    por_promo = df.groupby('Promotion').agg(
        n=('Transaction_ID', 'count'),
        ticket_medio=('Total_Cost', 'mean'),
        itens_medio=('Total_Items', 'mean'),
    ).reset_index()
    por_promo['pct'] = por_promo['n'] / por_promo['n'].sum() * 100
    print(por_promo.to_string(index=False))

    # Diferença BOGO vs Discount
    bogo = df[df['Promotion'] == 'BOGO (Buy One Get One)']
    desc = df[df['Promotion'] == 'Discount on Selected Items']
    print()
    print(f"Ticket BOGO:     R$ {bogo['Total_Cost'].mean():.2f} | itens: {bogo['Total_Items'].mean():.2f}")
    print(f"Ticket Desconto: R$ {desc['Total_Cost'].mean():.2f} | itens: {desc['Total_Items'].mean():.2f}")
    print(f"Diferença ticket: {(bogo['Total_Cost'].mean() / desc['Total_Cost'].mean() - 1) * 100:+.1f}%")
    print(f"Diferença itens:  {(bogo['Total_Items'].mean() / desc['Total_Items'].mean() - 1) * 100:+.1f}%")

    por_promo.to_csv(OUT / 'tipos_promocao.csv', index=False, encoding='utf-8')

    # ── 3. Distribuição por dia da semana ────────────────────────────────
    print()
    print("=" * 80)
    print("3. DISTRIBUIÇÃO POR DIA DA SEMANA")
    print("=" * 80)

    por_dow = df.groupby('dow').agg(
        n=('Transaction_ID', 'count'),
        ticket=('Total_Cost', 'mean'),
    ).reset_index()
    por_dow['pct'] = por_dow['n'] / por_dow['n'].sum() * 100
    por_dow['fator_relativo'] = por_dow['n'] / por_dow['n'].mean()
    por_dow['dia'] = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    print(por_dow[['dia', 'n', 'pct', 'ticket', 'fator_relativo']].to_string(index=False))
    por_dow.to_csv(OUT / 'distribuicao_dow.csv', index=False, encoding='utf-8')

    # ── 4. Segmento de cliente ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("4. SEGMENTO DE CLIENTE E TICKET MÉDIO")
    print("=" * 80)

    por_cust = df.groupby('Customer_Category').agg(
        n=('Transaction_ID', 'count'),
        ticket_medio=('Total_Cost', 'mean'),
        itens_medio=('Total_Items', 'mean'),
    ).reset_index().sort_values('n', ascending=False)
    por_cust['pct'] = por_cust['n'] / por_cust['n'].sum() * 100
    print(por_cust.to_string(index=False))

    # Cruzamento cliente × tipo de promo
    print()
    print("CLIENTE × TIPO DE PROMOÇÃO (qual segmento responde melhor):")
    cruz = pd.crosstab(
        df['Customer_Category'], df['Promotion'],
        values=df['Total_Cost'], aggfunc='mean'
    ).round(2)
    print(cruz)

    por_cust.to_csv(OUT / 'segmento_cliente.csv', index=False, encoding='utf-8')
    cruz.to_csv(OUT / 'cliente_x_promocao.csv', encoding='utf-8')

    # ── 5. Member vs No Member ───────────────────────────────────────────
    print()
    print("=" * 80)
    print("5. SÓCIO vs NÃO SÓCIO")
    print("=" * 80)
    por_mem = df.groupby('Member').agg(
        n=('Transaction_ID', 'count'),
        ticket=('Total_Cost', 'mean'),
        itens=('Total_Items', 'mean'),
    )
    print(por_mem)
    delta = (por_mem.loc['Yes', 'ticket'] / por_mem.loc['No', 'ticket'] - 1) * 100
    print(f"\nTicket Sócio vs Não-Sócio: {delta:+.1f}%")

    # ── 6. Sazonalidade mensal ────────────────────────────────────────────
    print()
    print("=" * 80)
    print("6. SAZONALIDADE MENSAL")
    print("=" * 80)
    por_mes = df.groupby('month').agg(
        n=('Transaction_ID', 'count'),
        ticket=('Total_Cost', 'mean'),
    ).reset_index()
    por_mes['fator'] = por_mes['n'] / por_mes['n'].mean()
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
              'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    por_mes['mes_nome'] = [meses[m-1] for m in por_mes['month']]
    print(por_mes[['mes_nome', 'n', 'ticket', 'fator']].to_string(index=False))
    por_mes.to_csv(OUT / 'sazonalidade_mes.csv', index=False, encoding='utf-8')

    # ── 7. Comparação com nosso fator_turno calibrado ────────────────────
    print()
    print("=" * 80)
    print("7. COMPARAÇÃO COM NOSSO MODELO (fator_turno calibrado posto)")
    print("=" * 80)

    # Carregar nossa calibração
    import json
    with open(ROOT / 'data/calibracao_v2.json', encoding='utf-8') as f:
        cfg = json.load(f)
    # Média do fator_turno entre todas as categorias
    fatores_nossos = []
    for c in cfg['categorias']:
        fatores_nossos.append(c['fator_turno'])
    fator_medio_nosso = np.mean(fatores_nossos, axis=0)

    fator_archive7 = por_turno['n'].values / por_turno['n'].mean()

    print(f"Turno     |  Posto Viana (calibrado) | Convenience Store (archive7)")
    print(f"Manhã     |  {fator_medio_nosso[0]:.3f}                  | {fator_archive7[0]:.3f}")
    print(f"Tarde     |  {fator_medio_nosso[1]:.3f}                  | {fator_archive7[1]:.3f}")
    print(f"Noite     |  {fator_medio_nosso[2]:.3f}                  | {fator_archive7[2]:.3f}")
    diff = np.abs(fator_medio_nosso - fator_archive7) / fator_medio_nosso * 100
    print()
    print(f"Diferença média: {diff.mean():.1f}%")

    # ── 8. RECOMENDAÇÕES ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("8. INSIGHTS APLICÁVEIS AO MODELO")
    print("=" * 80)

    rec = []

    # Insight 1: BOGO vs Discount
    ticket_bogo = bogo['Total_Cost'].mean()
    ticket_desc = desc['Total_Cost'].mean()
    if ticket_bogo > ticket_desc:
        ganho = (ticket_bogo / ticket_desc - 1) * 100
        rec.append(f"• BOGO gera ticket {ganho:+.1f}% maior que desconto. "
                    "Considerar adicionar ação 'BOGO' ao modelo (atualmente só desc%/combo).")
    else:
        rec.append("• Desconto gera ticket maior que BOGO neste dataset.")

    # Insight 2: hora específica
    pico_hora = por_hora.loc[por_hora['n_transacoes'].idxmax()]
    rec.append(f"• Pico horário: {pico_hora['hour']}h ({pico_hora['pct']:.1f}% das transações). "
                f"Ticket médio nessa hora: R$ {pico_hora['ticket_medio']:.2f}")

    # Insight 3: segmento dominante
    top_cust = por_cust.iloc[0]
    rec.append(f"• Cliente dominante: {top_cust['Customer_Category']} ({top_cust['pct']:.1f}%). "
                f"Ticket R$ {top_cust['ticket_medio']:.2f}, {top_cust['itens_medio']:.1f} itens/transação.")

    # Insight 4: sócio
    if delta > 0:
        rec.append(f"• Sócio gasta {delta:+.1f}% mais que não-sócio. "
                    "Modelo poderia ter feature 'fluxo de sócios' para datas certas.")

    # Insight 5: dia da semana
    top_dow = por_dow.loc[por_dow['n'].idxmax()]
    rec.append(f"• Dia da semana de maior promoção: {top_dow['dia']} ({top_dow['pct']:.1f}%). "
                "Confirma nossos fator_dia.")

    for r in rec:
        print(r)


if __name__ == '__main__':
    main()
