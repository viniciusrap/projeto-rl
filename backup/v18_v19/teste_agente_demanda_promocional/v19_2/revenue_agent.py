"""Agente de Receita — calcula lucro REAL a partir da estimativa de demanda.

RESPONSABILIDADE ÚNICA: dado uma EstimativaDemanda (do DemandAgent), calcular
todos os componentes financeiros realistas da campanha:

- lucro_uplift: ganho de vendas EXTRAS (uplift × margem × (1 - desc))
- lucro_canibalizacao: PERDA de margem em clientes que comprariam sem promo
- lucro_halo: cross-sell de outros itens
- custo_operacional: cartaz + tempo + risco PDV (escalável por volume)
- lucro_liquido_total: soma final - custo

NÃO classifica (BOA/MÉDIA/RUIM) — isso é responsabilidade do DecisionAgent.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from demand_agent import EstimativaDemanda


@dataclass
class EstimativaReceita:
    """Saída padronizada do agente de receita."""

    # Identificação
    categoria: str

    # Componentes do lucro (por dia)
    lucro_uplift_dia: float
    lucro_canibalizacao_dia: float  # negativo
    lucro_halo_dia: float
    lucro_defensivo_dia: float       # liquidação de produto vencendo
    lucro_liquido_dia: float

    # Totais da campanha
    dias: int
    lucro_bruto_campanha: float       # antes do custo operacional
    custo_operacional: float
    lucro_liquido_campanha: float     # final

    # Métricas auxiliares
    margem_unitaria: float
    receita_total_estimada: float      # demanda_promo × dias × preço_efetivo
    margem_efetiva_pct: float          # margem_liquida / receita
    breakeven_uplift_pct: float        # uplift mínimo para empatar custo op

    # Componentes (transparência)
    componentes: dict
    motivo: str

    def to_dict(self):
        return asdict(self)


class RevenueAgent:
    """Agente de Receita. Recebe EstimativaDemanda, retorna EstimativaReceita."""

    # Custos operacionais
    CUSTO_OP_BASE = 15.0
    CUSTO_OP_POR_DIA = 2.0
    CUSTO_OP_BAIXO_VOLUME = 10.0  # se d_base < 3 un/dia

    # Margem média de produtos vendidos juntos (cross-sell)
    MARGEM_CROSS_SELL = 3.50

    def __init__(self, calibracao_path: str = 'calibracao_v2.json'):
        path = Path(__file__).parent / calibracao_path
        with open(path, encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.cats = {c['categoria']: c for c in self.cfg['categorias']}

    def calcular(self, demanda: EstimativaDemanda,
                  campanha: dict) -> EstimativaReceita:
        """Calcula receita realista a partir da estimativa de demanda.

        campanha precisa ter: validade_restante_pct (opcional).
        """
        cat_info = self.cats.get(demanda.categoria, {})
        margem = float(cat_info.get('margem', 4.0))
        custo_unit = float(cat_info.get('custo', margem * 0.6))
        preco = float(cat_info.get('preco_venda', margem + custo_unit))

        desc_pct = demanda.desconto_pct / 100.0
        intensidade = demanda.intensidade

        # ─── 1. LUCRO UPLIFT (vendas extras) ───
        # Vendas extras × margem × (1 - desc)
        uplift_un = demanda.uplift_unidades_dia
        lucro_uplift = uplift_un * margem * (1 - desc_pct)

        # ─── 2. LUCRO CANIBALIZAÇÃO (perda em clientes orgânicos) ───
        # Clientes que comprariam SEM promo agora pagam menos
        d_promo = demanda.demanda_promocional_dia
        canib_pct = demanda.canibalizacao_estimada_pct / 100.0
        vendas_canib = d_promo * canib_pct
        lucro_canib = -vendas_canib * margem * desc_pct

        # ─── 3. LUCRO HALO (cross-sell) ───
        # Cliente entra pela promo, leva outros itens (5-10% das visitas)
        if intensidade == 'combo':
            halo_pct = 0.10
        elif intensidade == 'liq25%':
            halo_pct = 0.03
        else:
            halo_pct = 0.05
        lucro_halo = d_promo * halo_pct * self.MARGEM_CROSS_SELL

        # ─── 4. LUCRO DEFENSIVO (liquidação evita perda total) ───
        lucro_defensivo = 0.0
        validade_restante = campanha.get('validade_restante_pct')
        if (intensidade == 'liq25%'
                and validade_restante is not None
                and validade_restante < 0.30):
            # Estima estoque que venceria sem promo
            d_base = demanda.demanda_base_dia
            estoque_em_risco_dia = d_base * 1.5
            prob_vencer = min((1 - validade_restante) * 1.5, 0.95)
            estoque_salvo_dia = estoque_em_risco_dia * prob_vencer
            # Valor evitado = custo do estoque salvo × 0.7 (capturamos 70%)
            lucro_defensivo = estoque_salvo_dia * custo_unit * 0.7

        # ─── 5. LUCRO LÍQUIDO DIA ───
        lucro_dia = lucro_uplift + lucro_canib + lucro_halo + lucro_defensivo

        # ─── 6. CUSTO OPERACIONAL ───
        d_base = demanda.demanda_base_dia
        custo_op_base = (self.CUSTO_OP_BAIXO_VOLUME if d_base < 3.0
                          else self.CUSTO_OP_BASE)
        dias = demanda.dias
        custo_op = custo_op_base + self.CUSTO_OP_POR_DIA * dias

        # ─── 7. TOTAIS DA CAMPANHA ───
        lucro_bruto = lucro_dia * dias
        lucro_liquido = lucro_bruto - custo_op

        # ─── 8. MÉTRICAS AUXILIARES ───
        preco_efetivo = preco * (1 - desc_pct)
        receita_total = d_promo * dias * preco_efetivo
        margem_efetiva_pct = (lucro_liquido / receita_total * 100
                                if receita_total > 0 else 0)

        # Breakeven: qual uplift mínimo cobre o custo operacional?
        # custo_op = uplift_un × margem × (1-desc) × dias
        # uplift_un_min = custo_op / (margem × (1-desc) × dias)
        denom = margem * (1 - desc_pct) * dias
        if denom > 0:
            uplift_un_breakeven = custo_op / denom
            breakeven_pct = (uplift_un_breakeven / d_base * 100
                              if d_base > 0 else 0)
        else:
            breakeven_pct = 0

        # ─── 9. MOTIVO TEXTUAL ───
        partes = []
        partes.append(
            f"Demanda {d_base:.1f}→{d_promo:.1f}/dia × margem R${margem:.2f}."
        )
        partes.append(
            f"Uplift gera R${lucro_uplift:.2f}/dia, canibalização -R${-lucro_canib:.2f}/dia, "
            f"halo +R${lucro_halo:.2f}/dia."
        )
        if lucro_defensivo > 0:
            partes.append(
                f"Defensivo evita perda de R${lucro_defensivo:.2f}/dia (validade próxima)."
            )
        partes.append(
            f"Total {dias}d: R${lucro_bruto:.2f} bruto - R${custo_op:.2f} custo "
            f"= R${lucro_liquido:.2f} líquido."
        )
        if breakeven_pct > demanda.uplift_pct:
            partes.append(
                f"⚠ Uplift {demanda.uplift_pct:.1f}% < breakeven {breakeven_pct:.1f}%."
            )
        motivo = ' '.join(partes)

        return EstimativaReceita(
            categoria=demanda.categoria,
            lucro_uplift_dia=round(lucro_uplift, 2),
            lucro_canibalizacao_dia=round(lucro_canib, 2),
            lucro_halo_dia=round(lucro_halo, 2),
            lucro_defensivo_dia=round(lucro_defensivo, 2),
            lucro_liquido_dia=round(lucro_dia, 2),
            dias=dias,
            lucro_bruto_campanha=round(lucro_bruto, 2),
            custo_operacional=round(custo_op, 2),
            lucro_liquido_campanha=round(lucro_liquido, 2),
            margem_unitaria=round(margem, 2),
            receita_total_estimada=round(receita_total, 2),
            margem_efetiva_pct=round(margem_efetiva_pct, 2),
            breakeven_uplift_pct=round(breakeven_pct, 2),
            componentes={
                'preco_venda': round(preco, 2),
                'preco_efetivo': round(preco_efetivo, 2),
                'custo_unit': round(custo_unit, 2),
                'desc_pct': round(desc_pct * 100, 1),
                'canib_pct': round(canib_pct * 100, 1),
                'halo_pct': round(halo_pct * 100, 1),
                'custo_op_base_usado': custo_op_base,
            },
            motivo=motivo,
        )


# ───────── Teste integrado: Demand → Revenue ─────────

if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from demand_agent import DemandAgent

    demand = DemandAgent()
    revenue = RevenueAgent()

    print("=" * 95)
    print("TESTE INTEGRADO: DemandAgent → RevenueAgent")
    print("=" * 95)

    campanhas_teste = [
        {
            'nome': 'Chocolate+Vinho Namorados',
            'campanha': {
                'categoria': 'chocolate_premium', 'produto_complementar': 'vinho',
                'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 4,
                'data_inicio': '2026-06-08', 'data_fim': '2026-06-11',
                'eventos_comerciais_na_janela': ['Dia dos Namorados'],
            }
        },
        {
            'nome': 'Gelo+Cerveja sábado verão',
            'campanha': {
                'categoria': 'gelo', 'produto_complementar': 'cerveja',
                'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 3,
                'data_inicio': '2027-01-09', 'data_fim': '2027-01-11',
                'eventos_comerciais_na_janela': [],
            }
        },
        {
            'nome': 'Sorvete liquidação vencendo',
            'campanha': {
                'categoria': 'sorvete', 'intensidade': 'liq25%',
                'desconto_pct': 25, 'dias_total': 3,
                'data_inicio': '2026-05-18', 'data_fim': '2026-05-20',
                'eventos_comerciais_na_janela': [],
                'validade_restante_pct': 0.15, 'estoque_pct_normal': 1.4,
            }
        },
        {
            'nome': 'Isotônico desc5% comum (RUIM)',
            'campanha': {
                'categoria': 'isotonico', 'intensidade': 'desc5%',
                'desconto_pct': 5, 'dias_total': 5,
                'data_inicio': '2026-07-15', 'data_fim': '2026-07-19',
                'eventos_comerciais_na_janela': [],
            }
        },
    ]

    for t in campanhas_teste:
        print()
        print(f"📌 {t['nome']}")
        d = demand.estimar(t['campanha'])
        r = revenue.calcular(d, t['campanha'])

        print(f"   DEMANDA: {d.qualidade_promocao}  ({d.demanda_base_dia}→{d.demanda_promocional_dia} un/dia, "
              f"+{d.uplift_pct:.1f}%, canib {d.canibalizacao_estimada_pct}%)")
        print(f"   RECEITA:")
        print(f"     • Lucro uplift:        R$ {r.lucro_uplift_dia:>7.2f}/dia")
        print(f"     • Lucro canibal:       R$ {r.lucro_canibalizacao_dia:>7.2f}/dia")
        print(f"     • Lucro halo:          R$ {r.lucro_halo_dia:>7.2f}/dia")
        if r.lucro_defensivo_dia > 0:
            print(f"     • Lucro defensivo:     R$ {r.lucro_defensivo_dia:>7.2f}/dia")
        print(f"     • Lucro líquido/dia:   R$ {r.lucro_liquido_dia:>7.2f}")
        print(f"     • Custo operacional:   R$ {r.custo_operacional:>7.2f} ({r.dias} dias)")
        print(f"     • LUCRO TOTAL:         R$ {r.lucro_liquido_campanha:>7.2f}")
        print(f"     • Receita total:       R$ {r.receita_total_estimada:>7.2f}")
        print(f"     • Margem efetiva:      {r.margem_efetiva_pct:>5.2f}%")
        print(f"     • Breakeven uplift:    {r.breakeven_uplift_pct:>5.2f}% (real: {d.uplift_pct:.1f}%)")
        print(f"   💬 {r.motivo}")
