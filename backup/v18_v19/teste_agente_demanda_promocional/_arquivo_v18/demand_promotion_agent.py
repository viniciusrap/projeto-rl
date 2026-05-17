"""V18 — Agente Estimador de Demanda Promocional.

NOVA ABORDAGEM CONCEITUAL:
Hoje o modelo trata promoção como "filtro" sobre demanda base estática.
Errado. Promoção REDEFINE a demanda esperada do período. Esta classe
implementa essa lógica como agente especializado.

Fórmula da demanda promocional:

    d_promo = d_base × M_elasticidade × M_combo × M_evento × M_clima × M_dow

    d_realista = d_promo × (1 - canibalizacao) × (1 + halo)

Onde:
- M_elasticidade = 1 + elast × desc% (efeito puro do desconto)
- M_combo = 1 + (harmonia - 1) × 0.15 (combo amplifica harmonia)
- M_evento = 1 + (uplift_evento - 1) × captura_evento (50% do uplift teórico)
- M_clima = bonus se clima favorece categoria
- M_dow = bonus se dia da semana é o "natural" da categoria
- canibalizacao = 0.20 + desc × 1.5 (Walmart docs: 30-50% promo é canibalização)
- halo = +10% se combo, +5% se desc direto (cliente leva outras coisas)

Saída: dicionário com d_base, d_promo, uplift_pct, lucro_ajustado, justificativa.

Cases reais que validam essa lógica:
- Alibaba 2019 DRCR: reward absoluto colapsa, usar delta
- Freshippo KDD 2021: demanda contrafactual estocástica (não fixa)
- Walmart 2021: canibalização documentada empiricamente
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EstimativaDemanda:
    """Output do agente de demanda promocional."""
    demanda_base_dia: float
    demanda_promocional_dia: float
    uplift_pct: float
    uplift_unidades_dia: float
    boost_elasticidade: float
    boost_combo: float
    boost_evento: float
    boost_clima: float
    boost_dow: float
    canibalizacao_pct: float
    halo_pct: float
    lucro_adicional_dia: float
    lucro_canibalizacao: float  # negativo
    lucro_halo: float            # positivo
    lucro_liquido_dia: float
    custo_operacional: float
    roi: float
    confianca: str               # 'alta', 'media', 'baixa'
    justificativa: str = ''
    flags: list = field(default_factory=list)  # alertas: 'baixo_uplift', 'canibalizacao_alta', etc


class DemandPromotionAgent:
    """Agente que estima demanda promocional realista a partir de campanha."""

    # Captura efetiva do uplift teórico de evento comercial
    # Em produção real, captamos só 50% do uplift teórico (cliente decide
    # comprar antes ou depois). Calibrado em Alibaba Freshippo.
    CAPTURA_EVENTO = 0.5

    # Captura efetiva do clima — depende da categoria
    CAPTURA_CLIMA = 0.7

    # Custo operacional por campanha (R$)
    CUSTO_OPERACIONAL_BASE = 25.0  # cartaz + tempo equipe + risco PDV
    CUSTO_OPERACIONAL_POR_DIA = 3.0  # custo marginal de manter ativo

    # Canibalização base (Walmart 2021)
    CANIBALIZACAO_BASE = 0.20    # 20% das vendas eram orgânicas mesmo sem promo
    CANIBALIZACAO_POR_DESC = 1.5  # cada 10% desc → +15% canibalização
    CANIBALIZACAO_MAX = 0.50

    # Halo effect (cross-sell)
    HALO_COMBO = 0.10            # combo: cliente leva 10% mais outras categorias
    HALO_DESC = 0.05             # desc direto: 5%
    MARGEM_MEDIA_OUTROS = 3.50   # R$ margem média de itens vendidos junto

    def __init__(self, calibracao_path: str = 'calibracao_v2.json'):
        with open(Path(__file__).parent / calibracao_path, encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.cats_dict = {c['categoria']: c for c in self.cfg['categorias']}
        self.harmonia = self.cfg.get('harmonia_combo', [])
        self.cats_idx = {c['categoria']: i for i, c in enumerate(self.cfg['categorias'])}

    def estimar(self, campanha: dict) -> EstimativaDemanda:
        """Estima demanda promocional realista para uma campanha.

        campanha deve conter:
            categoria, intensidade (combo/desc5%/desc10%/liq25%),
            produto_complementar (opcional), data_inicio, data_fim,
            dias_total, desconto_pct, eventos_comerciais_na_janela (opcional)
        """
        cat = campanha['categoria']
        cat_info = self.cats_dict.get(cat, {})
        d_base = float(cat_info.get('demanda_base_dia', 5.0))
        margem = float(cat_info.get('margem', 4.0))
        elast = abs(float(cat_info.get('elasticidade_promo', -0.5)))
        intensidade = campanha.get('intensidade', 'combo')
        desc_pct = float(campanha.get('desconto_pct', 10)) / 100.0
        dias = int(campanha.get('dias_total', 3))
        par = campanha.get('produto_complementar', '')
        eventos = campanha.get('eventos_comerciais_na_janela', [])
        flags = []

        # ── 1. Elasticidade (efeito puro do desconto) ─────────────────
        # Para combo, desc é menor mas valor percebido maior
        if intensidade == 'combo':
            boost_elasticidade = 1.0 + elast * desc_pct * 1.3  # combo: +30% sinalização
        elif intensidade == 'liq25%':
            # Liquidação 25% atrai cliente, mas só pra produto perto vencer
            boost_elasticidade = 1.0 + elast * desc_pct * 0.9  # captura 90%
        else:
            boost_elasticidade = 1.0 + elast * desc_pct

        # ── 2. Combo amplificado por HARMONIA categoria↔par ───────────
        boost_combo = 1.0
        if intensidade == 'combo' and par and par in self.cats_idx:
            i_cat = self.cats_idx.get(cat)
            i_par = self.cats_idx.get(par)
            if i_cat is not None and i_par is not None and self.harmonia:
                h = float(self.harmonia[i_cat][i_par])
                # Harmonia 1.0 = neutro. 2.5 = forte. Combo amplifica 15% da harmonia.
                boost_combo = 1.0 + (h - 1.0) * 0.15
            else:
                boost_combo = 1.05  # combo genérico

        # ── 3. Evento comercial ──────────────────────────────────────
        boost_evento = 1.0
        if eventos:
            # Em vez de buscar uplift_prior no calendário, usar heurística:
            # eventos de presente: +50%, eventos consumo: +30%
            evento_str = (eventos[0] if eventos else '').lower()
            if any(k in evento_str for k in ['mães', 'maes', 'namorados', 'pais',
                                                'crianças', 'criancas', 'páscoa',
                                                'pascoa', 'natal', 'mulher', 'reveillon']):
                uplift_teorico = 1.7  # presente: 70%
            elif any(k in evento_str for k in ['copa', 'carnaval', 'mundo']):
                uplift_teorico = 1.5  # consumo intenso: 50%
            else:
                uplift_teorico = 1.2  # genérico: 20%
            boost_evento = 1.0 + (uplift_teorico - 1.0) * self.CAPTURA_EVENTO

        # ── 4. Clima ──────────────────────────────────────────────────
        # Categorias sensíveis ao clima
        boost_clima = 1.0
        cats_quentes = ['gelo', 'sorvete', 'isotonico', 'refrigerante', 'suco', 'agua']
        cats_frias = ['cafe', 'padaria', 'chocolate_premium', 'vinho']
        # Aproximação: se data está em verão (dez-mar) e cat quente → boost
        if campanha.get('data_inicio'):
            mes = int(campanha['data_inicio'][5:7])
            if mes in [12, 1, 2, 3] and cat in cats_quentes:
                boost_clima = 1.15
            elif mes in [6, 7, 8] and cat in cats_frias:
                boost_clima = 1.10

        # ── 5. Dia da semana ──────────────────────────────────────────
        # Captura o "dia natural" da categoria
        boost_dow = 1.0
        try:
            from datetime import date
            d_ini = date.fromisoformat(campanha['data_inicio'])
            dow = d_ini.weekday()
            # Pega fator_dia da calibração — média do range da campanha
            fator_dia = cat_info.get('fator_dia', [1]*7)
            d_fim = date.fromisoformat(campanha['data_fim'])
            dias_campanha = (d_fim - d_ini).days + 1
            soma = sum(fator_dia[(dow + i) % 7] for i in range(dias_campanha))
            media_fator_dow = soma / dias_campanha
            # Normaliza pelo fator médio anual (1.0)
            boost_dow = max(0.8, min(1.3, media_fator_dow))
        except Exception:
            boost_dow = 1.0

        # ── 6. DEMANDA PROMOCIONAL (vendas REAIS com promo) ───────────
        # Demanda promocional é diretamente o resultado dos boosts.
        # Canibalização afeta LUCRO, não quantidade.
        d_promocional = d_base * boost_elasticidade * boost_combo * boost_evento * boost_clima * boost_dow

        # ── 7. CANIBALIZAÇÃO (impacto no LUCRO, não na demanda) ───────
        # % das vendas COM PROMO que aconteceriam mesmo sem promo.
        # Walmart 2021: 30-50% das vendas promocionais são canibalização.
        # Com desc% maior, MAIS clientes-naturais aproveitam o desc.
        canibalizacao_pct = min(
            self.CANIBALIZACAO_BASE + desc_pct * self.CANIBALIZACAO_POR_DESC,
            self.CANIBALIZACAO_MAX
        )

        # ── 8. HALO (cross-sell) ──────────────────────────────────────
        halo_pct = self.HALO_COMBO if intensidade == 'combo' else self.HALO_DESC

        # ── 9. Uplift líquido (apenas vendas EXTRAS) ──────────────────
        uplift_un_dia = max(d_promocional - d_base, 0)
        uplift_pct = (d_promocional / d_base - 1) * 100 if d_base > 0 else 0

        # ── 10. LUCRO REALISTA ────────────────────────────────────────
        # 10a. Ganho bruto das vendas EXTRAS (uplift)
        lucro_uplift = uplift_un_dia * margem * (1 - desc_pct)

        # 10b. PERDA DE MARGEM em clientes que comprariam sem promo
        # Esses clientes (canibalizacao_pct × d_promocional) pagam menos agora
        vendas_canibalizadas_dia = d_promocional * canibalizacao_pct
        lucro_canibalizacao = -vendas_canibalizadas_dia * margem * desc_pct

        # 10c. Halo: cross-sell de outros itens (cliente entra, leva mais)
        lucro_halo = d_promocional * halo_pct * self.MARGEM_MEDIA_OUTROS

        # 10d. Líquido por dia
        lucro_dia = lucro_uplift + lucro_canibalizacao + lucro_halo

        # ── 12. Custo operacional ─────────────────────────────────────
        custo_operacional = self.CUSTO_OPERACIONAL_BASE + self.CUSTO_OPERACIONAL_POR_DIA * dias
        # Lucro total da campanha
        lucro_total = lucro_dia * dias - custo_operacional

        # ── 13. ROI e confiança ───────────────────────────────────────
        roi = lucro_total / custo_operacional if custo_operacional > 0 else 0

        # Confiança: baixa se uplift < 10%, alta se > 30%, média senão
        if uplift_pct < 10:
            confianca = 'baixa'
            flags.append('baixo_uplift')
        elif uplift_pct > 30:
            confianca = 'alta'
        else:
            confianca = 'media'

        if canibalizacao_pct > 0.40:
            flags.append('canibalizacao_alta')
        if roi < 1.0:
            flags.append('roi_baixo')
        if lucro_total < 0:
            flags.append('prejuizo')

        # ── 14. Justificativa textual ─────────────────────────────────
        partes = [
            f"Demanda base {d_base:.1f} un/dia → estimada {d_promocional:.1f} un/dia "
            f"(uplift {uplift_pct:+.1f}%)."
        ]
        if boost_combo > 1.05:
            partes.append(f"Combo com {par} tem harmonia {boost_combo:.2f}×.")
        if boost_evento > 1.1:
            partes.append(f"Evento '{eventos[0] if eventos else ''}' "
                            f"agrega {(boost_evento-1)*100:.0f}% via captura.")
        if boost_clima > 1.05:
            partes.append(f"Clima favorece (+{(boost_clima-1)*100:.0f}%).")
        if canibalizacao_pct > 0.35:
            partes.append(f"⚠ Canibalização alta ({canibalizacao_pct*100:.0f}%)")
        if roi < 1.0:
            partes.append(f"⚠ ROI {roi:.2f}× — campanha questionável.")
        justificativa = ' '.join(partes)

        return EstimativaDemanda(
            demanda_base_dia=round(d_base, 2),
            demanda_promocional_dia=round(d_promocional, 2),
            uplift_pct=round(uplift_pct, 1),
            uplift_unidades_dia=round(uplift_un_dia, 2),
            boost_elasticidade=round(boost_elasticidade, 3),
            boost_combo=round(boost_combo, 3),
            boost_evento=round(boost_evento, 3),
            boost_clima=round(boost_clima, 3),
            boost_dow=round(boost_dow, 3),
            canibalizacao_pct=round(canibalizacao_pct * 100, 1),
            halo_pct=round(halo_pct * 100, 1),
            lucro_adicional_dia=round(lucro_dia, 2),
            lucro_canibalizacao=round(lucro_canibalizacao, 2),
            lucro_halo=round(lucro_halo, 2),
            lucro_liquido_dia=round(lucro_dia, 2),
            custo_operacional=round(custo_operacional, 2),
            roi=round(roi, 2),
            confianca=confianca,
            justificativa=justificativa,
            flags=flags,
        )


# ─────────── Smoke test ───────────

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    from dataclasses import asdict

    agent = DemandPromotionAgent()

    # 5 campanhas teste pedidas pelo Vinicius
    campanhas_teste = [
        {
            'categoria': 'gelo', 'produto_complementar': 'cerveja',
            'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 2,
            'data_inicio': '2027-01-09', 'data_fim': '2027-01-10',
            'eventos_comerciais_na_janela': [],
        },
        {
            'categoria': 'gelo', 'produto_complementar': 'destilados',
            'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 2,
            'data_inicio': '2026-12-25', 'data_fim': '2026-12-31',
            'eventos_comerciais_na_janela': ['Réveillon'],
        },
        {
            'categoria': 'chocolate_premium', 'produto_complementar': 'vinho',
            'intensidade': 'combo', 'desconto_pct': 10, 'dias_total': 4,
            'data_inicio': '2026-06-08', 'data_fim': '2026-06-11',
            'eventos_comerciais_na_janela': ['Dia dos Namorados'],
        },
        {
            'categoria': 'isotonico', 'produto_complementar': '',
            'intensidade': 'desc5%', 'desconto_pct': 5, 'dias_total': 5,
            'data_inicio': '2026-07-15', 'data_fim': '2026-07-19',
            'eventos_comerciais_na_janela': [],
        },
        {
            'categoria': 'chocolate_impulso', 'produto_complementar': '',
            'intensidade': 'desc10%', 'desconto_pct': 10, 'dias_total': 3,
            'data_inicio': '2026-09-15', 'data_fim': '2026-09-17',
            'eventos_comerciais_na_janela': [],
        },
    ]

    print("=" * 100)
    print("TESTE DO AGENTE DE DEMANDA PROMOCIONAL — 5 CAMPANHAS")
    print("=" * 100)
    for c in campanhas_teste:
        print()
        print(f"📌 {c['categoria']}", end='')
        if c.get('produto_complementar'):
            print(f" + {c['produto_complementar']}", end='')
        print(f"  ({c['intensidade']}, {c['dias_total']}d, desc {c['desconto_pct']}%)")
        evt = c.get('eventos_comerciais_na_janela', [])
        if evt:
            print(f"   Evento: {evt[0]}")

        est = agent.estimar(c)
        print(f"  Demanda base:        {est.demanda_base_dia:.2f} un/dia")
        print(f"  Demanda promocional: {est.demanda_promocional_dia:.2f} un/dia")
        print(f"  Uplift:              {est.uplift_pct:+.1f}%")
        print(f"  Componentes:         elast×{est.boost_elasticidade}  "
              f"combo×{est.boost_combo}  evento×{est.boost_evento}  "
              f"clima×{est.boost_clima}  dow×{est.boost_dow}")
        print(f"  Canibalização:       {est.canibalizacao_pct}%")
        print(f"  Halo (cross-sell):   +{est.halo_pct}%")
        print(f"  Lucro/dia:           R$ {est.lucro_adicional_dia:.2f} "
              f"(uplift R$ {est.lucro_adicional_dia - est.lucro_canibalizacao - est.lucro_halo:.2f} "
              f"+ halo R$ {est.lucro_halo:.2f} − canibal R$ {-est.lucro_canibalizacao:.2f})")
        print(f"  Custo operacional:   R$ {est.custo_operacional:.2f}")
        print(f"  ROI:                 {est.roi}× ({est.confianca} confiança)")
        if est.flags:
            print(f"  ⚠ Flags:             {', '.join(est.flags)}")
        print(f"  💬 {est.justificativa}")
