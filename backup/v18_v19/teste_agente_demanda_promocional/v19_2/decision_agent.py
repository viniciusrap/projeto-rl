"""Agente de Decisão — agrega Demand + Revenue e decide se campanha entra no calendário.

RESPONSABILIDADE ÚNICA: dado uma EstimativaDemanda e EstimativaReceita,
classificar a campanha em uma das 5 categorias operacionais:

- 🟢 APROVADA          → entra no calendário, prioridade normal
- 🌟 APROVADA_PRIORITARIA → entra no calendário, destaque (ROI excepcional)
- 🟡 CONDICIONAL       → entra só se sobrar slot (ROI marginal)
- 🔴 REJEITADA         → não entra (ROI negativo ou breakeven inviável)
- 🚫 PROIBIDA          → bloqueada por lei/risco (cigarros, combo antagônico)

NÃO calcula receita NEM demanda — apenas decide sobre o que veio dos outros agentes.
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from demand_agent import EstimativaDemanda
from revenue_agent import EstimativaReceita


@dataclass
class DecisaoCampanha:
    """Saída padronizada do agente de decisão."""

    # Identificação
    categoria: str
    intensidade: str

    # Classificação final
    decisao: str                # APROVADA / APROVADA_PRIORITARIA / CONDICIONAL / REJEITADA / PROIBIDA
    score: float                # 0-100 (score composto de viabilidade)
    prioridade: int             # 1=alta, 2=normal, 3=baixa, 99=rejeitada

    # Métricas de decisão
    roi_pct: float              # lucro_liquido / custo_operacional × 100
    lucro_por_dia: float
    lucro_total: float
    razao_uplift_breakeven: float  # uplift_real / breakeven (>1 viável)

    # Critérios atendidos
    criterios_aprovacao: list
    criterios_rejeicao: list

    # Justificativa
    motivo: str

    def to_dict(self):
        return asdict(self)


class DecisionAgent:
    """Agente de Decisão. Recebe Demand + Revenue, retorna DecisaoCampanha."""

    # ─── Thresholds de aprovação ───
    ROI_MIN_APROVADO = 50.0           # ROI mínimo para aprovar (%)
    ROI_PRIORITARIO = 200.0           # ROI excepcional → prioridade alta
    LUCRO_MIN_DIA = 10.0              # Lucro mínimo R$/dia para considerar
    LUCRO_MIN_TOTAL = 50.0            # Lucro mínimo total da campanha
    RAZAO_UPLIFT_MIN = 1.2            # uplift real precisa ser ≥ 1.2× breakeven

    # Score weights (somam 100)
    W_ROI = 35
    W_LUCRO_ABS = 25
    W_RAZAO_BREAKEVEN = 20
    W_QUALIDADE_DEMANDA = 15
    W_RISCO = 5

    def decidir(self, demanda: EstimativaDemanda,
                receita: EstimativaReceita) -> DecisaoCampanha:
        """Decide se a campanha entra no calendário."""

        criterios_ap = []
        criterios_rej = []

        # ─── 1. CHECAGEM DE BLOQUEIO ABSOLUTO ───
        if 'PROIBIDA' in demanda.qualidade_promocao:
            return self._construir_decisao(
                demanda, receita,
                decisao='🚫 PROIBIDA',
                score=0.0,
                prioridade=99,
                criterios_ap=[],
                criterios_rej=['Categoria proibida por lei'],
                motivo=f"Bloqueada: {demanda.qualidade_promocao}. {demanda.motivo}",
            )

        # Combo antagônico (harmonia < 1.0) — bloqueio defensivo
        if (demanda.intensidade == 'combo'
                and 'combo_antagonico' in demanda.flags):
            return self._construir_decisao(
                demanda, receita,
                decisao='🚫 PROIBIDA',
                score=0.0,
                prioridade=99,
                criterios_ap=[],
                criterios_rej=['Combo com harmonia antagônica'],
                motivo=f"Combo antagônico. {demanda.motivo}",
            )

        # V19.2: combo inválido no balcão de posto
        # (gelo 5kg + garrafa whisky, etc — não casa no PDV)
        if (demanda.intensidade == 'combo'
                and 'combo_invalido_pdv' in demanda.flags):
            return self._construir_decisao(
                demanda, receita,
                decisao='🚫 PROIBIDA',
                score=0.0,
                prioridade=99,
                criterios_ap=[],
                criterios_rej=['Combo cross-class no balcão de PDV (não faz sentido de cesta)'],
                motivo=f"Combo PDV-inválido. {demanda.motivo}",
            )

        # ─── 2. MÉTRICAS DE DECISÃO ───
        roi = (receita.lucro_liquido_campanha / receita.custo_operacional * 100
                if receita.custo_operacional > 0 else 0)
        razao_breakeven = (demanda.uplift_pct / receita.breakeven_uplift_pct
                            if receita.breakeven_uplift_pct > 0 else 99)

        # ─── 3. CHECAGENS DE REJEIÇÃO ───
        if receita.lucro_liquido_campanha < 0:
            criterios_rej.append(f"Lucro líquido negativo (R${receita.lucro_liquido_campanha:.2f})")
        if roi < self.ROI_MIN_APROVADO:
            criterios_rej.append(f"ROI {roi:.1f}% < mínimo {self.ROI_MIN_APROVADO}%")
        if receita.lucro_liquido_dia < self.LUCRO_MIN_DIA:
            criterios_rej.append(f"Lucro/dia R${receita.lucro_liquido_dia:.2f} < mínimo R${self.LUCRO_MIN_DIA}")
        if razao_breakeven < self.RAZAO_UPLIFT_MIN:
            criterios_rej.append(f"Uplift {demanda.uplift_pct:.1f}% < {self.RAZAO_UPLIFT_MIN:.1f}× breakeven ({receita.breakeven_uplift_pct:.1f}%)")
        if 'RUIM' in demanda.qualidade_promocao:
            criterios_rej.append("Demanda classificada RUIM pelo DemandAgent")

        # ─── 4. CHECAGENS DE APROVAÇÃO ───
        if roi >= self.ROI_MIN_APROVADO:
            criterios_ap.append(f"ROI {roi:.1f}% ≥ {self.ROI_MIN_APROVADO}%")
        if receita.lucro_liquido_campanha >= self.LUCRO_MIN_TOTAL:
            criterios_ap.append(f"Lucro total R${receita.lucro_liquido_campanha:.2f}")
        if razao_breakeven >= self.RAZAO_UPLIFT_MIN:
            criterios_ap.append(f"Uplift {razao_breakeven:.1f}× breakeven")
        if 'BOA' in demanda.qualidade_promocao:
            criterios_ap.append("Demanda BOA")
        if receita.lucro_defensivo_dia > 0:
            criterios_ap.append(f"Lucro defensivo +R${receita.lucro_defensivo_dia:.2f}/dia")

        # ─── 5. SCORE COMPOSTO (0-100) ───
        # ROI score: 0% = 0, 200%+ = full
        roi_score = min(roi / self.ROI_PRIORITARIO * self.W_ROI, self.W_ROI)
        # Lucro absoluto: cap em R$500 = full
        lucro_score = min(receita.lucro_liquido_campanha / 500 * self.W_LUCRO_ABS,
                            self.W_LUCRO_ABS)
        lucro_score = max(lucro_score, 0)  # não negativo
        # Razão breakeven: cap em 5×
        razao_score = min(razao_breakeven / 5 * self.W_RAZAO_BREAKEVEN,
                            self.W_RAZAO_BREAKEVEN)
        # Qualidade demanda
        if 'BOA' in demanda.qualidade_promocao:
            qual_score = self.W_QUALIDADE_DEMANDA
        elif 'MÉDIA' in demanda.qualidade_promocao:
            qual_score = self.W_QUALIDADE_DEMANDA * 0.5
        else:
            qual_score = 0
        # Risco (canibalização)
        if demanda.nivel_risco_canibalizacao == 'baixo':
            risco_score = self.W_RISCO
        elif demanda.nivel_risco_canibalizacao == 'medio':
            risco_score = self.W_RISCO * 0.5
        else:
            risco_score = 0

        score = roi_score + lucro_score + razao_score + qual_score + risco_score
        score = round(max(min(score, 100), 0), 1)

        # ─── 6. DECISÃO FINAL ───
        if len(criterios_rej) > 0:
            decisao = '🔴 REJEITADA'
            prioridade = 99
        elif roi >= self.ROI_PRIORITARIO and score >= 75:
            decisao = '🌟 APROVADA_PRIORITARIA'
            prioridade = 1
        elif roi >= self.ROI_MIN_APROVADO and score >= 50:
            decisao = '🟢 APROVADA'
            prioridade = 2
        elif roi >= self.ROI_MIN_APROVADO * 0.7:
            decisao = '🟡 CONDICIONAL'
            prioridade = 3
        else:
            decisao = '🔴 REJEITADA'
            prioridade = 99
            criterios_rej.append(f"Score composto {score} baixo")

        # ─── 7. MOTIVO TEXTUAL ───
        partes = []
        partes.append(
            f"ROI {roi:.1f}%, lucro R${receita.lucro_liquido_campanha:.2f} em {receita.dias}d "
            f"(R${receita.lucro_liquido_dia:.2f}/dia)."
        )
        partes.append(
            f"Uplift {demanda.uplift_pct:.1f}% vs breakeven {receita.breakeven_uplift_pct:.1f}% "
            f"({razao_breakeven:.1f}× margem)."
        )
        if receita.lucro_defensivo_dia > 0:
            partes.append(
                f"Defensivo R${receita.lucro_defensivo_dia:.2f}/dia (validade próxima)."
            )
        partes.append(f"Score {score}/100.")
        motivo = ' '.join(partes)

        return self._construir_decisao(
            demanda, receita,
            decisao=decisao,
            score=score,
            prioridade=prioridade,
            criterios_ap=criterios_ap,
            criterios_rej=criterios_rej,
            motivo=motivo,
            roi=roi,
            razao_breakeven=razao_breakeven,
        )

    def _construir_decisao(self, demanda, receita, decisao, score, prioridade,
                            criterios_ap, criterios_rej, motivo,
                            roi=0.0, razao_breakeven=0.0) -> DecisaoCampanha:
        return DecisaoCampanha(
            categoria=demanda.categoria,
            intensidade=demanda.intensidade,
            decisao=decisao,
            score=round(score, 1),
            prioridade=prioridade,
            roi_pct=round(roi, 2),
            lucro_por_dia=round(receita.lucro_liquido_dia, 2),
            lucro_total=round(receita.lucro_liquido_campanha, 2),
            razao_uplift_breakeven=round(razao_breakeven, 2),
            criterios_aprovacao=criterios_ap,
            criterios_rejeicao=criterios_rej,
            motivo=motivo,
        )


# ───────── Teste integrado: Demand → Revenue → Decision ─────────

if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from demand_agent import DemandAgent
    from revenue_agent import RevenueAgent

    demand = DemandAgent()
    revenue = RevenueAgent()
    decision = DecisionAgent()

    print("=" * 95)
    print("TESTE INTEGRADO: DemandAgent → RevenueAgent → DecisionAgent")
    print("=" * 95)

    campanhas_teste = [
        {
            'nome': 'Chocolate+Vinho Namorados (esperado: APROVADA/PRIORITARIA)',
            'campanha': {
                'categoria': 'chocolate_premium', 'produto_complementar': 'vinho',
                'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 4,
                'data_inicio': '2026-06-08', 'data_fim': '2026-06-11',
                'eventos_comerciais_na_janela': ['Dia dos Namorados'],
            }
        },
        {
            'nome': 'Gelo+Cerveja sábado verão (esperado: APROVADA)',
            'campanha': {
                'categoria': 'gelo', 'produto_complementar': 'cerveja',
                'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 3,
                'data_inicio': '2027-01-09', 'data_fim': '2027-01-11',
                'eventos_comerciais_na_janela': [],
            }
        },
        {
            'nome': 'Sorvete liquidação vencendo (esperado: APROVADA_PRIORITARIA)',
            'campanha': {
                'categoria': 'sorvete', 'intensidade': 'liq25%',
                'desconto_pct': 25, 'dias_total': 3,
                'data_inicio': '2026-05-18', 'data_fim': '2026-05-20',
                'eventos_comerciais_na_janela': [],
                'validade_restante_pct': 0.15, 'estoque_pct_normal': 1.4,
            }
        },
        {
            'nome': 'Isotônico desc5% comum (esperado: REJEITADA)',
            'campanha': {
                'categoria': 'isotonico', 'intensidade': 'desc5%',
                'desconto_pct': 5, 'dias_total': 5,
                'data_inicio': '2026-07-15', 'data_fim': '2026-07-19',
                'eventos_comerciais_na_janela': [],
            }
        },
        {
            'nome': 'Cigarro com desc (esperado: PROIBIDA)',
            'campanha': {
                'categoria': 'cigarro_souza_cruz', 'intensidade': 'desc5%',
                'desconto_pct': 5, 'dias_total': 3,
                'data_inicio': '2026-05-13', 'data_fim': '2026-05-15',
                'eventos_comerciais_na_janela': [],
            }
        },
    ]

    aprovadas = []
    for t in campanhas_teste:
        print()
        print(f"📌 {t['nome']}")
        d = demand.estimar(t['campanha'])
        r = revenue.calcular(d, t['campanha'])
        dec = decision.decidir(d, r)

        print(f"   Demanda:     {d.qualidade_promocao} ({d.uplift_pct:+.1f}%, canib {d.canibalizacao_estimada_pct}%)")
        print(f"   Receita:     R${r.lucro_liquido_campanha:.2f} ({r.dias}d) | breakeven {r.breakeven_uplift_pct}%")
        print(f"   ─────────────────────────────────────────────")
        print(f"   DECISÃO:     {dec.decisao}")
        print(f"   Score:       {dec.score}/100")
        print(f"   Prioridade:  {dec.prioridade}")
        print(f"   ROI:         {dec.roi_pct}%")
        print(f"   Razão BE:    {dec.razao_uplift_breakeven}×")
        if dec.criterios_aprovacao:
            print(f"   ✓ Aprovação: {' | '.join(dec.criterios_aprovacao)}")
        if dec.criterios_rejeicao:
            print(f"   ✗ Rejeição:  {' | '.join(dec.criterios_rejeicao)}")
        print(f"   💬 {dec.motivo}")

        if dec.prioridade < 99:
            aprovadas.append((dec.prioridade, dec.score, t['nome'], dec.lucro_total))

    print()
    print("=" * 95)
    print(f"RESUMO: {len(aprovadas)} campanhas aprovadas (de {len(campanhas_teste)})")
    print("=" * 95)
    aprovadas.sort(key=lambda x: (x[0], -x[1]))
    for prio, score, nome, lucro in aprovadas:
        print(f"  P{prio} (score {score}): {nome} → R${lucro:.2f}")
