"""V2 — Agente de Demanda Promocional refinado.

DIFERENÇAS-CHAVE vs v1:

1. CLASSIFICAÇÃO DE PRODUTOS em 6 tipos:
   - puxador_consumo (cerveja, gelo) — gera fluxo
   - puxador_premium (destilados, vinho) — margem alta, baixo volume
   - puxador_presente (chocolate_premium) — explosivo em datas
   - impulso (chocolate_impulso, snack, biscoito) — sensível a desconto
   - rotina (cafe, padaria) — pouco mudaria com promo
   - commodity (agua, refrigerante, isotonico) — preço pouco influencia

2. LÓGICA DE COMBO REVISADA:
   - Desconto é aplicado no COMPLEMENTAR (protege margem do principal)
   - Boost no PRINCIPAL via efeito "ancoragem" (cliente vê valor)
   - Penaliza combos com harmonia < 1.2 (fraco)

3. CAPS DE UPLIFT por intensidade (impede números absurdos):
   - desc5%: max +25%
   - desc10%: max +50%
   - combo: max +90%
   - liq25%: max +120%

4. CONSIDERA ESTOQUE/VALIDADE/GIRO:
   - Estoque alto → desc tem MAIS impacto (queima estoque parado)
   - Validade próxima → liquidação é VENCEDORA (defensiva)
   - Giro baixo → desc é arriscado (pode não converter)

5. EVENTOS DIFERENCIADOS:
   - Presente (Mães, Namorados): foco em produto puxador_presente, captura 60%
   - Consumo intenso (Copa, Carnaval): foco em commodity+impulso, captura 70%
   - Comercial genérico (Black Friday): captura 30%

6. CLASSIFICAÇÃO FINAL DA CAMPANHA:
   - 🟢 BOA: ROI ≥ 2.0× e uplift ≥ 30%
   - 🟡 MÉDIA: ROI ≥ 1.0× e uplift ≥ 15%
   - 🔴 RUIM: ROI < 1.0× ou uplift < 15%
"""
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional


# ─────────── Classificação de produtos (conhecimento de varejo) ────────

TIPO_PRODUTO = {
    # Puxador de consumo (gera fluxo, alta demanda fim-de-semana/evento)
    'cerveja': 'puxador_consumo',
    'gelo': 'puxador_consumo',
    # Puxador premium (margem alta, baixo volume, datas específicas)
    'destilados': 'puxador_premium',
    'vinho': 'puxador_premium',
    # Puxador presente (chocolate em datas comerciais explode)
    'chocolate_premium': 'puxador_presente',
    # Impulso (cliente decide na hora, sensível a desc)
    'chocolate_impulso': 'impulso',
    'snack': 'impulso',
    'biscoito': 'impulso',
    'doce': 'impulso',
    'sorvete': 'impulso',
    # Rotina (cliente já compra, promo pouco muda)
    'cafe': 'rotina',
    'padaria': 'rotina',
    # Commodity (preço pouco influencia)
    'agua': 'commodity',
    'refrigerante': 'commodity',
    'suco': 'commodity',
    'isotonico': 'commodity',
    'energetico': 'commodity',
    # Proibidas
    'cigarro_souza_cruz': 'proibida',
    'cigarro_philip_morris': 'proibida',
    'cigarro_jti': 'proibida',
}


# Elasticidade por tipo (sobrescreve elast do catálogo)
ELASTICIDADE_TIPO = {
    'commodity':         0.3,   # baixíssima (água vende sempre)
    'rotina':            0.4,   # baixa (cliente compra mesmo sem promo)
    'puxador_premium':   0.5,   # baixa (cliente premium não é sensível)
    'impulso':           0.9,   # média-alta (desc move impulso)
    'puxador_consumo':   1.1,   # alta (cerveja em desc gera fluxo)
    'puxador_presente':  1.5,   # explosiva em datas comerciais
    'proibida':          0.0,
}


# Caps de uplift máximo por intensidade (impede números absurdos)
CAP_UPLIFT = {
    'nada':   0.0,
    'desc5%':  0.25,
    'desc10%': 0.50,
    'combo':   0.90,
    'liq25%':  1.20,
}


# Classificação de eventos
TIPO_EVENTO = {
    # PRESENTE (planejado, foco em puxador_presente)
    'dia das mães': 'presente',
    'dia das maes': 'presente',
    'dia dos namorados': 'presente',
    'dia internacional da mulher': 'presente',
    'dia da mulher': 'presente',
    'dia dos pais': 'presente',
    'páscoa': 'presente',
    'pascoa': 'presente',
    'véspera de natal': 'presente',
    'vespera de natal': 'presente',
    'natal': 'presente',
    'dia das crianças': 'presente',
    'dia das criancas': 'presente',
    # CONSUMO INTENSO (curto, foco em commodity/impulso)
    'réveillon': 'consumo_intenso',
    'reveillon': 'consumo_intenso',
    'carnaval': 'consumo_intenso',
    'copa': 'consumo_intenso',
    # COMERCIAL (genérico, multiplicador médio)
    'black friday': 'comercial',
    'cyber monday': 'comercial',
    'dia do consumidor': 'comercial',
}


@dataclass
class EstimativaDemanda:
    """Saída padronizada do agente."""
    # Inputs ecoados
    categoria: str
    intensidade: str
    par_combo: str
    dias: int
    desconto_pct: float
    # Classificação
    tipo_produto: str
    tipo_evento: str  # 'nenhum', 'presente', 'consumo_intenso', 'comercial'
    # Demanda
    demanda_base_dia: float
    demanda_promocional_dia: float
    uplift_pct: float
    uplift_unidades_dia: float
    # Multiplicadores aplicados (transparência)
    boost_elasticidade: float
    boost_combo_harmonia: float
    boost_evento: float
    boost_clima: float
    boost_dow: float
    boost_estoque_giro: float
    cap_aplicado: bool  # True se uplift bateu no teto
    # Risco
    canibalizacao_pct: float
    halo_pct: float
    # Lucro
    lucro_uplift: float
    lucro_canibalizacao: float
    lucro_halo: float
    lucro_liquido_dia: float
    custo_operacional: float
    lucro_total_campanha: float
    roi: float
    # Classificação final
    classificacao: str   # '🟢 BOA' | '🟡 MÉDIA' | '🔴 RUIM' | '🚫 PROIBIDA'
    confianca: str       # 'alta' | 'media' | 'baixa'
    justificativa: str
    flags: list = field(default_factory=list)


class DemandPromotionAgentV2:
    """Agente refinado. Lê data/calibracao_v2.json (ou parametrizável)."""

    # V3: custo operacional reduzido (mais realista) + escalável por volume
    CUSTO_OP_BASE = 15.0           # cartaz simples + treino caixa
    CUSTO_OP_POR_DIA = 2.0          # custo marginal manter ativo
    # Categorias baixo volume (vinho 0.6 un/dia) custo operacional MENOR
    CUSTO_OP_BAIXO_VOLUME = 10.0    # se demanda < 3 un/dia
    MARGEM_MEDIA_CROSS_SELL = 3.50  # margem média de cross-sell

    # Captura efetiva do uplift teórico de eventos
    CAPTURA_EVENTO_PRESENTE = 0.6
    CAPTURA_EVENTO_CONSUMO = 0.7
    CAPTURA_EVENTO_COMERCIAL = 0.3

    # Canibalização base (Walmart 2021)
    CANIBAL_BASE = 0.20
    CANIBAL_POR_DESC = 1.0  # cada 10% desc → +10% canib (mais conservador que v1)
    CANIBAL_MAX = 0.45

    # Halo
    HALO_COMBO = 0.10
    HALO_DESC = 0.05
    HALO_LIQ = 0.03  # liquidação não chama cross-sell tão bem

    def __init__(self, calibracao_path: str = 'calibracao_v2.json'):
        path = Path(__file__).parent / calibracao_path
        with open(path, encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.cats = {c['categoria']: c for c in self.cfg['categorias']}
        self.cat_idx = {c['categoria']: i for i, c in enumerate(self.cfg['categorias'])}
        self.harmonia = self.cfg.get('harmonia_combo', [])

    # ─────────────── Helpers internos ────────────────

    def _classifica_evento(self, eventos_str_list):
        """Classifica evento como presente/consumo_intenso/comercial/nenhum."""
        if not eventos_str_list:
            return 'nenhum'
        for ev in eventos_str_list:
            ev_norm = ev.lower()
            for key, tipo in TIPO_EVENTO.items():
                if key in ev_norm:
                    return tipo
        return 'comercial'  # default se não bate

    def _harmonia(self, cat_a: str, cat_b: str) -> float:
        ia = self.cat_idx.get(cat_a)
        ib = self.cat_idx.get(cat_b)
        if ia is None or ib is None or not self.harmonia:
            return 1.0
        return float(self.harmonia[ia][ib])

    def _boost_evento(self, tipo_produto: str, tipo_evento: str) -> float:
        """Boost de evento depende do MATCH entre produto e evento."""
        if tipo_evento == 'nenhum':
            return 1.0
        # Match perfeito: produto-presente em evento-presente
        if tipo_evento == 'presente' and tipo_produto == 'puxador_presente':
            return 1.0 + 0.7 * self.CAPTURA_EVENTO_PRESENTE  # +42%
        if tipo_evento == 'consumo_intenso' and tipo_produto in ('puxador_consumo', 'commodity', 'impulso'):
            return 1.0 + 0.5 * self.CAPTURA_EVENTO_CONSUMO   # +35%
        if tipo_evento == 'presente' and tipo_produto == 'puxador_premium':
            return 1.0 + 0.5 * self.CAPTURA_EVENTO_PRESENTE  # +30% (vinho em Mães)
        if tipo_evento == 'presente' and tipo_produto == 'impulso':
            return 1.0 + 0.3 * self.CAPTURA_EVENTO_PRESENTE  # +18% (chocolate_impulso em Mães)
        if tipo_evento == 'comercial':
            return 1.0 + 0.2 * self.CAPTURA_EVENTO_COMERCIAL  # +6% genérico
        # Mismatch (Réveillon promovendo chocolate premium etc): boost pequeno
        return 1.05

    def _boost_clima(self, tipo_produto: str, mes: int) -> float:
        """Clima sazonal (mes 1-12)."""
        if mes in [12, 1, 2, 3]:  # verão BR
            if tipo_produto == 'puxador_consumo':  # gelo, cerveja
                return 1.20
            if 'gelo' in tipo_produto or 'sorvete' in tipo_produto:
                return 1.25
            return 1.0
        if mes in [6, 7, 8]:  # inverno BR
            if tipo_produto == 'rotina':  # café, padaria
                return 1.10
            if tipo_produto in ('puxador_consumo', 'commodity'):
                return 0.85  # cerveja/gelo cai no inverno
        return 1.0

    def _boost_dow(self, categoria: str, data_inicio_str: str, dias: int,
                    intensidade: str) -> float:
        """Captura fator_dia da calibração — média do range da campanha.

        V3 — REGRA NOVA: se intensidade é desc% (não combo) E dow é baixo,
        boost_dow é NEUTRO (1.0) em vez de < 1. Razão: desc em dia
        naturalmente fraco MOVE estoque parado (não amplifica demanda
        existente). Combo em dia fraco continua menos efetivo.
        """
        try:
            cat_info = self.cats.get(categoria, {})
            fator_dia = cat_info.get('fator_dia', [1]*7)
            d_ini = date.fromisoformat(data_inicio_str)
            dow_ini = d_ini.weekday()
            soma = sum(fator_dia[(dow_ini + i) % 7] for i in range(dias))
            media = soma / dias
            media_cap = max(0.7, min(1.4, media))
            # V3: desc% em dia fraco → boost neutro (não penaliza)
            if intensidade in ('desc5%', 'desc10%') and media_cap < 1.0:
                return max(media_cap, 1.0)  # piso 1.0 para desc em dia fraco
            return media_cap
        except Exception:
            return 1.0

    def _multiplicador_estoque_giro(self, campanha: dict, tipo_produto: str) -> float:
        """Considera estoque alto, validade próxima, giro baixo.

        - estoque_alto (>120% normal): +15% desc converte (queima estoque)
        - validade_proxima (<30% restante): +25% desc é defensivo
        - giro_baixo: -10% desc menos efetivo
        """
        boost = 1.0
        estoque_norm = campanha.get('estoque_pct_normal')  # 0.5 = metade, 2.0 = dobro
        validade_restante = campanha.get('validade_restante_pct')  # 0.3 = 30%
        giro_alto = campanha.get('giro_alto', True)

        if estoque_norm is not None and estoque_norm > 1.2:
            boost *= 1.15  # estoque parado → desc move
        if validade_restante is not None and validade_restante < 0.30:
            boost *= 1.25  # liquidação é estratégia certa
        if not giro_alto:
            boost *= 0.90  # giro baixo → menos resposta a desc
        return boost

    def _penaliza_combo_fraco(self, cat: str, par: str, intensidade: str) -> float:
        """Combo de produtos sem harmonia tem boost menor."""
        if intensidade != 'combo' or not par:
            return 1.0
        h = self._harmonia(cat, par)
        if h < 1.0:
            return 0.85  # combo antagônico
        if h < 1.2:
            return 0.95  # combo neutro
        return 1.0  # harmonia >= 1.2 é OK

    def _aplicar_cap(self, uplift_bruto: float, intensidade: str) -> tuple:
        """Aplica cap de uplift por intensidade. Retorna (uplift_final, cap_aplicado)."""
        cap = CAP_UPLIFT.get(intensidade, 0.30)
        if uplift_bruto > cap:
            return cap, True
        return uplift_bruto, False

    # ─────────────── ESTIMAR (público) ────────────────

    def estimar(self, campanha: dict) -> EstimativaDemanda:
        """Estima demanda promocional realista. Retorna EstimativaDemanda."""
        cat = campanha['categoria']
        intensidade = campanha.get('intensidade', 'combo')
        par = campanha.get('produto_complementar', '')
        dias = int(campanha.get('dias_total', 3))
        desc_pct = float(campanha.get('desconto_pct', 10)) / 100.0
        eventos = campanha.get('eventos_comerciais_na_janela', [])
        data_ini = campanha.get('data_inicio', '2026-01-01')

        # ─── Classificações ───
        tipo_produto = TIPO_PRODUTO.get(cat, 'commodity')
        tipo_evento = self._classifica_evento(eventos)

        cat_info = self.cats.get(cat, {})
        d_base = float(cat_info.get('demanda_base_dia', 5.0))
        margem = float(cat_info.get('margem', 4.0))

        # ─── Proibida → bloqueia ───
        if tipo_produto == 'proibida':
            return self._resultado_proibido(campanha, d_base, margem)

        # ─── Elasticidade efetiva (por tipo, não literatura) ───
        elast = ELASTICIDADE_TIPO.get(tipo_produto, 0.5)

        # ─── 1. BOOST DE PREÇO (elasticidade × desconto) ───
        # Atenção: combo, desc é no complementar — boost no principal vem de
        # "valor percebido", aplicado parcialmente
        if intensidade == 'combo':
            # Cliente vê combo com desc no complementar → boost no principal +30% do efeito do desc
            boost_elast = 1.0 + elast * desc_pct * 0.30
        elif intensidade == 'liq25%':
            boost_elast = 1.0 + elast * desc_pct * 0.85  # liq é forte mas urgência mata margem
        else:
            boost_elast = 1.0 + elast * desc_pct

        # ─── 2. HARMONIA DE COMBO ───
        boost_combo = 1.0
        if intensidade == 'combo' and par:
            h = self._harmonia(cat, par)
            # Harmonia 1.0 = neutro, 2.5 = clássico. Combo amplifica 20% da harmonia.
            boost_combo = 1.0 + max(0, h - 1.0) * 0.20

        # ─── 3. EVENTO ───
        boost_evento = self._boost_evento(tipo_produto, tipo_evento)

        # ─── 4. CLIMA SAZONAL ───
        mes = int(data_ini[5:7])
        boost_clima = self._boost_clima(tipo_produto, mes)

        # ─── 5. DIA DA SEMANA ───
        boost_dow = self._boost_dow(cat, data_ini, dias, intensidade)

        # ─── 6. ESTOQUE/VALIDADE/GIRO ───
        boost_eg = self._multiplicador_estoque_giro(campanha, tipo_produto)

        # ─── 7. PENALIDADE de combo fraco ───
        penalidade_combo = self._penaliza_combo_fraco(cat, par, intensidade)

        # ─── 8. UPLIFT BRUTO ───
        boost_total = (boost_elast * boost_combo * boost_evento
                        * boost_clima * boost_dow * boost_eg * penalidade_combo)
        uplift_bruto = boost_total - 1.0

        # ─── 9. APLICAR CAP ───
        uplift_final, cap_aplicado = self._aplicar_cap(uplift_bruto, intensidade)
        d_promocional = d_base * (1.0 + uplift_final)

        # ─── 10. CANIBALIZAÇÃO (afeta lucro, não quantidade) ───
        # Eventos PRESENTE têm menos canibalização (cliente compra ESPECÍFICO p/ data)
        canib_ajuste = 0.7 if tipo_evento == 'presente' else 1.0
        canibalizacao = min(
            (self.CANIBAL_BASE + desc_pct * self.CANIBAL_POR_DESC) * canib_ajuste,
            self.CANIBAL_MAX
        )

        # ─── 11. HALO (cross-sell) ───
        if intensidade == 'combo':
            halo = self.HALO_COMBO
        elif intensidade == 'liq25%':
            halo = self.HALO_LIQ
        else:
            halo = self.HALO_DESC

        # ─── 12. LUCROS ───
        uplift_un = max(d_promocional - d_base, 0)
        lucro_uplift = uplift_un * margem * (1 - desc_pct)
        vendas_canib = d_promocional * canibalizacao
        lucro_canib = -vendas_canib * margem * desc_pct
        lucro_halo = d_promocional * halo * self.MARGEM_MEDIA_CROSS_SELL

        # V3 — BONUS DEFENSIVO (validade próxima)
        # Se sem promo o produto venceria, perda = custo total. Promo evita isso.
        lucro_defensivo_dia = 0.0
        validade_restante = campanha.get('validade_restante_pct')
        if validade_restante is not None and validade_restante < 0.30 and intensidade == 'liq25%':
            # Estima estoque que venceria: demanda_base × dias_até_vencimento (em turnos)
            # Aproximação: prob_vencer ~ (1 - validade_restante) × 1.5
            custo_unit = cat_info.get('custo', margem * 0.6)
            estoque_em_risco_dia = d_base * 1.5  # estoque que venceria sem ação
            prob_vencer_sem_promo = (1 - validade_restante) * 1.5
            prob_vencer_sem_promo = min(prob_vencer_sem_promo, 0.95)
            # Promo "salva" parte do estoque
            estoque_salvo_dia = estoque_em_risco_dia * prob_vencer_sem_promo
            # Lucro defensivo: custo evitado = custo × estoque_salvo
            lucro_defensivo_dia = estoque_salvo_dia * custo_unit * 0.7

        lucro_dia = lucro_uplift + lucro_canib + lucro_halo + lucro_defensivo_dia

        # V3: custo operacional escalável por volume
        if d_base < 3.0:
            custo_op_base = self.CUSTO_OP_BAIXO_VOLUME
        else:
            custo_op_base = self.CUSTO_OP_BASE
        custo_op = custo_op_base + self.CUSTO_OP_POR_DIA * dias
        lucro_total = lucro_dia * dias - custo_op
        roi = lucro_total / custo_op if custo_op > 0 else 0

        # ─── 13. CLASSIFICAÇÃO FINAL ───
        uplift_pct = uplift_final * 100
        flags = []
        if cap_aplicado:
            flags.append('cap_atingido')
        if canibalizacao > 0.35:
            flags.append('canibalizacao_alta')
        if uplift_pct < 15:
            flags.append('baixo_uplift')
        if roi < 1.0:
            flags.append('roi_baixo')
        if lucro_total < 0:
            flags.append('prejuizo')

        # Classificação textual
        if roi >= 2.0 and uplift_pct >= 30:
            classificacao = '🟢 BOA'
        elif roi >= 1.0 and uplift_pct >= 15:
            classificacao = '🟡 MÉDIA'
        else:
            classificacao = '🔴 RUIM'

        # Confiança
        if cap_aplicado or uplift_pct < 10:
            confianca = 'baixa'
        elif tipo_evento != 'nenhum' or boost_combo > 1.2:
            confianca = 'alta'
        else:
            confianca = 'media'

        # ─── 14. JUSTIFICATIVA ───
        partes = [
            f"Produto tipo {tipo_produto}, evento {tipo_evento}.",
            f"Demanda base {d_base:.1f}→{d_promocional:.1f}/dia ({uplift_pct:+.1f}%).",
        ]
        if intensidade == 'combo' and par:
            h = self._harmonia(cat, par)
            partes.append(f"Combo {cat}+{par} (harmonia {h:.2f}).")
            if h < 1.2:
                partes.append(f"⚠ Combo fraco (harmonia<1.2)")
        if boost_evento > 1.1:
            partes.append(f"Match com evento {tipo_evento}: +{(boost_evento-1)*100:.0f}%.")
        if boost_clima > 1.1:
            partes.append(f"Clima favorece (+{(boost_clima-1)*100:.0f}%).")
        if boost_eg > 1.05:
            partes.append("Estoque/validade reforça desc.")
        if cap_aplicado:
            partes.append(f"⚠ Cap de uplift {CAP_UPLIFT[intensidade]*100:.0f}% atingido.")
        if canibalizacao > 0.35:
            partes.append(f"⚠ Canibalização {canibalizacao*100:.0f}%.")
        if roi < 1.0:
            partes.append(f"⚠ ROI {roi:.2f}× (campanha não cobre custo operacional).")

        return EstimativaDemanda(
            categoria=cat,
            intensidade=intensidade,
            par_combo=par,
            dias=dias,
            desconto_pct=desc_pct * 100,
            tipo_produto=tipo_produto,
            tipo_evento=tipo_evento,
            demanda_base_dia=round(d_base, 2),
            demanda_promocional_dia=round(d_promocional, 2),
            uplift_pct=round(uplift_pct, 1),
            uplift_unidades_dia=round(uplift_un, 2),
            boost_elasticidade=round(boost_elast, 3),
            boost_combo_harmonia=round(boost_combo, 3),
            boost_evento=round(boost_evento, 3),
            boost_clima=round(boost_clima, 3),
            boost_dow=round(boost_dow, 3),
            boost_estoque_giro=round(boost_eg, 3),
            cap_aplicado=cap_aplicado,
            canibalizacao_pct=round(canibalizacao * 100, 1),
            halo_pct=round(halo * 100, 1),
            lucro_uplift=round(lucro_uplift, 2),
            lucro_canibalizacao=round(lucro_canib, 2),
            lucro_halo=round(lucro_halo, 2),
            lucro_liquido_dia=round(lucro_dia, 2),
            custo_operacional=round(custo_op, 2),
            lucro_total_campanha=round(lucro_total, 2),
            roi=round(roi, 2),
            classificacao=classificacao,
            confianca=confianca,
            justificativa=' '.join(partes),
            flags=flags,
        )

    def _resultado_proibido(self, campanha, d_base, margem):
        """Campanha de produto proibido (cigarro)."""
        return EstimativaDemanda(
            categoria=campanha['categoria'],
            intensidade=campanha.get('intensidade', '?'),
            par_combo=campanha.get('produto_complementar', ''),
            dias=campanha.get('dias_total', 0),
            desconto_pct=campanha.get('desconto_pct', 0),
            tipo_produto='proibida',
            tipo_evento='nenhum',
            demanda_base_dia=d_base,
            demanda_promocional_dia=d_base,
            uplift_pct=0, uplift_unidades_dia=0,
            boost_elasticidade=1.0, boost_combo_harmonia=1.0,
            boost_evento=1.0, boost_clima=1.0, boost_dow=1.0, boost_estoque_giro=1.0,
            cap_aplicado=False,
            canibalizacao_pct=0, halo_pct=0,
            lucro_uplift=0, lucro_canibalizacao=0, lucro_halo=0,
            lucro_liquido_dia=0, custo_operacional=0,
            lucro_total_campanha=0, roi=0,
            classificacao='🚫 PROIBIDA',
            confianca='alta',
            justificativa='Produto proibido por Lei 9.294/96 (cigarros).',
            flags=['proibida_lei'],
        )
