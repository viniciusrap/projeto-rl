"""Pipeline V19 — integração dos 3 agentes aplicada ao calendário V17.

Fluxo:
  1. Lê results/v17/calendario_operacional.json
  2. Para cada campanha (estrutural + eventual):
     - Adapta para input do DemandAgent
     - DemandAgent → EstimativaDemanda (demanda promocional realista)
     - RevenueAgent → EstimativaReceita (lucro real com canibalização/halo)
     - DecisionAgent → DecisaoCampanha (aprovação/rejeição)
  3. Filtra campanhas rejeitadas, ordena por score
  4. Salva results/v19/calendario_v19.json + comparação V17 vs V19

Saída inclui DIAGNÓSTICO: quantas campanhas o V17 superestimou ou subestimou.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from demand_agent import DemandAgent
from revenue_agent import RevenueAgent
from decision_agent import DecisionAgent


# Mapeia intensidade do V17 (texto) → intensidade dos agentes
MAP_INTENSIDADE = {
    'desc5%': 'desc5%',
    'desc10%': 'desc10%',
    'combo': 'combo',
    'liq25%': 'liq25%',
    # V17 às vezes vem como "combo" sem categoria — assumimos combo
}


def adaptar_campanha_v17(camp: dict, tipo: str = 'eventual') -> Optional[dict]:
    """Converte campanha V17 para input do DemandAgent.

    Retorna None se a campanha não puder ser interpretada.
    """
    # Estrutural — dia repete ao longo do ano
    # Adapta para uma janela típica de 2 dias para calcular custo operacional
    # correto; o lucro/dia é o que importa, depois multiplicamos por n_dias_no_ano.
    if tipo == 'estrutural':
        n_dias = camp.get('n_dias_no_ano', 50)
        return {
            'categoria': camp['categoria'],
            'produto_complementar': camp.get('par_combo'),
            'intensidade': 'combo',
            'desconto_pct': 10,
            'dias_total': 2,
            'data_inicio': '2026-08-21',
            'data_fim': '2026-08-22',
            'eventos_comerciais_na_janela': [],
            'estrutural_n_dias_ano': n_dias,
            'estrutural_flag': True,
        }

    # Eventual — vem com tudo do V17
    if 'data_inicio' not in camp or 'data_fim' not in camp:
        return None

    intensidade = camp.get('intensidade', 'combo')
    desconto = camp.get('desconto_pct', 10)
    eventos = camp.get('eventos_comerciais_na_janela', [])

    return {
        'categoria': camp['categoria'],
        'produto_complementar': camp.get('produto_complementar'),
        'intensidade': MAP_INTENSIDADE.get(intensidade, intensidade),
        'desconto_pct': desconto,
        'dias_total': camp.get('dias_total', 2),
        'data_inicio': camp['data_inicio'],
        'data_fim': camp['data_fim'],
        'eventos_comerciais_na_janela': eventos,
    }


def processar_calendario(path_v17: str, path_saida: str):
    """Aplica os 3 agentes ao calendário V17 e gera calendário V19."""

    demand = DemandAgent()
    revenue = RevenueAgent()
    decision = DecisionAgent()

    with open(path_v17, encoding='utf-8') as f:
        cal_v17 = json.load(f)

    print(f"\n{'='*100}")
    print(f"PIPELINE V19 — processando {path_v17}")
    print(f"{'='*100}\n")

    resultados = []
    aprovadas = []
    rejeitadas = []
    proibidas = []

    # ─── Processa estruturais ───
    # Estruturais avaliadas pelo lucro ANUAL (n_dias_no_ano × lucro/dia),
    # não pela janela de 2 dias (que serve só para calibrar o agente).
    print(f"📋 ESTRUTURAIS ({len(cal_v17.get('campanhas_estruturais', []))} campanhas)\n")
    for camp in cal_v17.get('campanhas_estruturais', []):
        camp_input = adaptar_campanha_v17(camp, tipo='estrutural')
        if camp_input is None:
            continue

        try:
            d = demand.estimar(camp_input)
            r = revenue.calcular(d, camp_input)
        except Exception as e:
            print(f"  ✗ {camp.get('nome', '?')}: ERRO {e}")
            continue

        # Anualiza para julgar
        n_dias_ano = camp.get('n_dias_no_ano', 1)
        lucro_anual = r.lucro_liquido_dia * n_dias_ano - revenue.CUSTO_OP_BASE
        # Custo op anualizado = base + (CUSTO_OP_POR_DIA × n_dias_ano)
        custo_op_anual = revenue.CUSTO_OP_BASE + revenue.CUSTO_OP_POR_DIA * n_dias_ano
        roi_anual = (lucro_anual / custo_op_anual * 100 if custo_op_anual > 0 else 0)

        # Decisão própria para estruturais (usa anualizado)
        if 'PROIBIDA' in d.qualidade_promocao:
            decisao_str = '🚫 PROIBIDA'
            prioridade = 99
            criterios_ap = []
            criterios_rej = ['Categoria proibida por lei']
        elif r.lucro_liquido_dia < 1.0:
            decisao_str = '🔴 REJEITADA'
            prioridade = 99
            criterios_ap = []
            criterios_rej = [f'Lucro/dia R${r.lucro_liquido_dia:.2f} < R$1 (estrutural inviável)']
        elif roi_anual >= 500 and lucro_anual >= 500:
            decisao_str = '🌟 APROVADA_PRIORITARIA'
            prioridade = 1
            criterios_ap = [f'ROI anual {roi_anual:.0f}%', f'R${lucro_anual:.0f}/ano']
            criterios_rej = []
        elif roi_anual >= 100 and lucro_anual >= 100:
            decisao_str = '🟢 APROVADA'
            prioridade = 2
            criterios_ap = [f'ROI anual {roi_anual:.0f}%', f'R${lucro_anual:.0f}/ano']
            criterios_rej = []
        elif roi_anual >= 50:
            decisao_str = '🟡 CONDICIONAL'
            prioridade = 3
            criterios_ap = [f'ROI anual {roi_anual:.0f}%']
            criterios_rej = [f'Lucro anual R${lucro_anual:.0f} marginal']
        else:
            decisao_str = '🔴 REJEITADA'
            prioridade = 99
            criterios_ap = []
            criterios_rej = [f'ROI anual {roi_anual:.0f}% < 50%']

        rec = {
            'tipo': 'estrutural',
            'nome': camp.get('nome'),
            'categoria': d.categoria,
            'par_combo': camp.get('par_combo'),
            'n_dias_no_ano': n_dias_ano,
            'lucro_v17_anual': camp.get('lucro_adicional_estimado_anual_R$', 0),
            'lucro_v19_anual': round(lucro_anual, 2),
            'lucro_v19_por_dia': r.lucro_liquido_dia,
            'custo_op_anual': round(custo_op_anual, 2),
            'demanda_base_dia': d.demanda_base_dia,
            'demanda_promo_dia': d.demanda_promocional_dia,
            'uplift_pct': d.uplift_pct,
            'canibalizacao_pct': d.canibalizacao_estimada_pct,
            'qualidade': d.qualidade_promocao,
            'roi_pct': round(roi_anual, 2),
            'decisao': decisao_str,
            'score': round(min(roi_anual / 10, 100), 1),
            'prioridade': prioridade,
            'criterios_aprovacao': criterios_ap,
            'criterios_rejeicao': criterios_rej,
            'motivo': f"Estrutural {n_dias_ano}d/ano. Lucro R${r.lucro_liquido_dia:.2f}/dia × {n_dias_ano} = R${lucro_anual:.2f}. ROI anual {roi_anual:.0f}%.",
            'comunicacao': camp.get('comunicacao', ''),
        }
        resultados.append(rec)

        marker = '✓' if prioridade < 99 else '✗'
        print(f"  {marker} {camp.get('nome'):28s} "
                f"V17 R${camp.get('lucro_adicional_estimado_anual_R$', 0):>8.2f}/ano | "
                f"V19 R${lucro_anual:>8.2f}/ano | {decisao_str}")

        if decisao_str.startswith('🚫'):
            proibidas.append(rec)
        elif prioridade < 99:
            aprovadas.append(rec)
        else:
            rejeitadas.append(rec)

    # ─── Processa eventuais ───
    print(f"\n📅 EVENTUAIS ({len(cal_v17.get('campanhas_eventuais', []))} campanhas)\n")
    for camp in cal_v17.get('campanhas_eventuais', []):
        camp_input = adaptar_campanha_v17(camp, tipo='eventual')
        if camp_input is None:
            continue

        try:
            d = demand.estimar(camp_input)
            r = revenue.calcular(d, camp_input)
            dec = decision.decidir(d, r)
        except Exception as e:
            print(f"  ✗ {camp['data_inicio']} {camp['categoria']}: ERRO {e}")
            continue

        eventos = camp.get('eventos_comerciais_na_janela', [])
        evento_str = eventos[0] if eventos else '-'

        rec = {
            'tipo': 'eventual',
            'data_inicio': camp['data_inicio'],
            'data_fim': camp['data_fim'],
            'dias_total': camp.get('dias_total', 0),
            'categoria': d.categoria,
            'produto_complementar': camp.get('produto_complementar'),
            'intensidade': d.intensidade,
            'desconto_pct': camp.get('desconto_pct', 10),
            'evento': evento_str,
            'origem': camp.get('origem', 'modelo_v15'),
            # Métricas V17 vs V19
            'lucro_v17_estimado': camp.get('lucro_adicional_estimado_R$', 0),
            'lucro_v19_estimado': r.lucro_liquido_campanha,
            'lucro_v19_por_dia': r.lucro_liquido_dia,
            'demanda_base_dia': d.demanda_base_dia,
            'demanda_promo_dia': d.demanda_promocional_dia,
            'uplift_pct': d.uplift_pct,
            'canibalizacao_pct': d.canibalizacao_estimada_pct,
            'qualidade': d.qualidade_promocao,
            'roi_pct': dec.roi_pct,
            'razao_breakeven': dec.razao_uplift_breakeven,
            'decisao': dec.decisao,
            'score': dec.score,
            'prioridade': dec.prioridade,
            'motivo': dec.motivo,
            'criterios_aprovacao': dec.criterios_aprovacao,
            'criterios_rejeicao': dec.criterios_rejeicao,
            'comunicacao': camp.get('comunicacao', ''),
        }
        resultados.append(rec)

        marker = '✓' if dec.prioridade < 99 else '✗'
        evento_tag = f"[{evento_str}]" if evento_str != '-' else ''
        print(f"  {marker} {camp['data_inicio']} ({camp.get('dias_total', '?')}d) "
                f"{d.categoria:20s} {evento_tag:25s} "
                f"V17 R${camp.get('lucro_adicional_estimado_R$', 0):>6.2f} → "
                f"V19 R${r.lucro_liquido_campanha:>7.2f} | {dec.decisao}")

        if dec.decisao.startswith('🚫'):
            proibidas.append(rec)
        elif dec.prioridade < 99:
            aprovadas.append(rec)
        else:
            rejeitadas.append(rec)

    # ─── Estatísticas ───
    total = len(resultados)
    lucro_v17_total = sum(
        (r.get('lucro_v17_anual') or r.get('lucro_v17_estimado') or 0)
        for r in resultados
    )
    lucro_v19_total = sum(
        (r.get('lucro_v19_anual') or r.get('lucro_v19_estimado') or 0)
        for r in resultados
    )
    lucro_v19_aprovado = sum(
        (r.get('lucro_v19_anual') or r.get('lucro_v19_estimado') or 0)
        for r in aprovadas
    )

    print(f"\n{'='*100}")
    print(f"RESUMO V17 → V19")
    print(f"{'='*100}")
    print(f"  Campanhas processadas:    {total}")
    print(f"  ✓ Aprovadas:              {len(aprovadas)}")
    print(f"  ✗ Rejeitadas pelo V19:    {len(rejeitadas)}")
    print(f"  🚫 Proibidas:              {len(proibidas)}")
    print(f"\n  Lucro V17 total estimado: R$ {lucro_v17_total:>10.2f}")
    print(f"  Lucro V19 total estimado: R$ {lucro_v19_total:>10.2f}")
    print(f"  Lucro V19 só aprovadas:   R$ {lucro_v19_aprovado:>10.2f}")
    print(f"  Δ V19/V17:                {(lucro_v19_total/lucro_v17_total - 1) * 100 if lucro_v17_total > 0 else 0:+.1f}%")

    # ─── Salva ───
    out = {
        'versao': 'V19 (V17 + 3 agentes: Demand → Revenue → Decision)',
        'gerado_em': datetime.now().strftime('%Y-%m-%d'),
        'origem': path_v17,
        'sumario': {
            'total_processadas': total,
            'aprovadas': len(aprovadas),
            'rejeitadas': len(rejeitadas),
            'proibidas': len(proibidas),
            'lucro_v17_total_R$': round(lucro_v17_total, 2),
            'lucro_v19_total_R$': round(lucro_v19_total, 2),
            'lucro_v19_aprovadas_R$': round(lucro_v19_aprovado, 2),
            'delta_v19_v17_pct': round(
                (lucro_v19_total/lucro_v17_total - 1) * 100 if lucro_v17_total > 0 else 0, 2),
        },
        'aprovadas': sorted(aprovadas, key=lambda x: (x['prioridade'], -x['score'])),
        'rejeitadas': rejeitadas,
        'proibidas': proibidas,
        'todos': resultados,
    }

    Path(path_saida).parent.mkdir(parents=True, exist_ok=True)
    with open(path_saida, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Calendário V19 salvo: {path_saida}")

    return out


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    PROJETO = Path(__file__).parent.parent.parent
    # Input padrão: calendário V17 (comum a todas versões)
    path_v17 = PROJETO / 'teste_agente_demanda_promocional' / '_common' / 'input_calendarios' / 'v17_business_rules.json'
    if not path_v17.exists():
        path_v17 = PROJETO / 'results' / 'v17' / 'calendario_operacional.json'
    # Output exclusivo da V19.1
    path_saida = PROJETO / 'results' / 'v19_1' / 'calendario_v19_1.json'

    processar_calendario(str(path_v17), str(path_saida))
