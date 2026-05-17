"""Agente de Demanda Promocional — versão V3 final.

RESPONSABILIDADE ÚNICA: estimar quanto cada produto vai vender DURANTE
uma promoção, considerando contexto realista de varejo de conveniência.

NÃO faz:
- Cálculo de lucro (responsabilidade do agente de receita)
- Cálculo de ROI (responsabilidade do agente de decisão)
- Custo operacional (responsabilidade do agente de operação)
- Ranking entre campanhas (responsabilidade do agente de priorização)

OUTPUT: EstimativaDemanda com:
- demanda_base_dia
- demanda_promocional_dia
- uplift_pct, uplift_unidades_dia
- canibalizacao_estimada_pct
- qualidade_promocao (BOA/MÉDIA/RUIM/PROIBIDA)
- motivo (texto)
- componentes (cada boost separado, transparente)

LÓGICA EM 7 ETAPAS:

1. CLASSIFICAR produto em 6 tipos (puxador_consumo, puxador_premium,
   puxador_presente, impulso, rotina, commodity) — afeta elasticidade
2. CLASSIFICAR evento em 4 tipos (presente, consumo_intenso, comercial, nenhum)
3. APLICAR REGRAS DE NEGÓCIO (cigarros proibidos, desc direto em puxador
   sem evento é ruim, combo sem harmonia é ruim)
4. CALCULAR boosts individuais (preço, combo, evento, clima, dia, estoque/giro)
5. AGREGAR boosts → uplift bruto → aplicar cap por intensidade
6. ESTIMAR canibalização (% das vendas que aconteceriam sem promo)
7. CLASSIFICAR qualidade da promoção + gerar justificativa
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# CONHECIMENTO DE DOMÍNIO (varejo de conveniência)
# ═══════════════════════════════════════════════════════════════════════

TIPO_PRODUTO = {
    'cerveja': 'puxador_consumo',
    'gelo': 'puxador_consumo',
    'destilados': 'puxador_premium',
    'vinho': 'puxador_premium',
    'chocolate_premium': 'puxador_presente',
    'chocolate_impulso': 'impulso',
    'snack': 'impulso',
    'biscoito': 'impulso',
    'doce': 'impulso',
    'sorvete': 'impulso',
    'cafe': 'rotina',
    'padaria': 'rotina',
    'agua': 'commodity',
    'refrigerante': 'commodity',
    'suco': 'commodity',
    'isotonico': 'commodity',
    'energetico': 'commodity',
    'cigarro_souza_cruz': 'proibida',
    'cigarro_philip_morris': 'proibida',
    'cigarro_jti': 'proibida',
}

# Elasticidade efetiva por tipo de produto (sobrescreve catálogo)
# Valores: quanto a quantidade vendida aumenta para 1% de desconto
ELASTICIDADE = {
    'commodity':         0.3,
    'rotina':            0.4,
    'puxador_premium':   0.5,
    'impulso':           0.9,
    'puxador_consumo':   1.1,
    'puxador_presente':  1.5,
    'proibida':          0.0,
}

# Cap de uplift por intensidade (impede números absurdos)
CAP_UPLIFT = {
    'nada':    0.00,
    'desc5%':  0.25,   # max +25% (desc 5% raramente passa disso)
    'desc10%': 0.50,
    'combo':   0.90,   # combo pode ir mais alto (cross-sell)
    'liq25%':  1.20,
}

# Tipos de evento
TIPO_EVENTO = {
    'dia das mães': 'presente', 'dia das maes': 'presente',
    'dia dos namorados': 'presente',
    'dia internacional da mulher': 'presente', 'dia da mulher': 'presente',
    'dia dos pais': 'presente',
    'páscoa': 'presente', 'pascoa': 'presente',
    'véspera de natal': 'presente', 'vespera de natal': 'presente',
    'natal': 'presente',
    'dia das crianças': 'presente', 'dia das criancas': 'presente',
    'réveillon': 'consumo_intenso', 'reveillon': 'consumo_intenso',
    'carnaval': 'consumo_intenso',
    'copa': 'consumo_intenso',
    'black friday': 'comercial',
    'cyber monday': 'comercial',
    'dia do consumidor': 'comercial',
}


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT PADRONIZADO
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EstimativaDemanda:
    """Saída padronizada do agente. Pronta para serializar em JSON."""

    # Identificação da campanha
    categoria: str
    intensidade: str
    par_combo: Optional[str]
    dias: int
    desconto_pct: float

    # Classificações internas
    tipo_produto: str
    tipo_evento: str

    # SAÍDA PRINCIPAL — o que outros agentes vão usar
    demanda_base_dia: float
    demanda_promocional_dia: float
    uplift_pct: float
    uplift_unidades_dia: float

    # Risco
    canibalizacao_estimada_pct: float
    nivel_risco_canibalizacao: str  # 'baixo', 'medio', 'alto'

    # Componentes (transparência — outros agentes podem auditar)
    componentes: dict
    cap_aplicado: bool

    # Classificação heurística (sem cálculo de lucro)
    qualidade_promocao: str  # 'BOA', 'MÉDIA', 'RUIM', 'PROIBIDA'
    confianca: str           # 'alta', 'media', 'baixa'

    # Justificativa
    motivo: str
    flags: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════
# AGENTE
# ═══════════════════════════════════════════════════════════════════════

class DemandAgent:
    """Estima demanda promocional. Responsabilidade única."""

    # Captura realista do uplift teórico de eventos
    CAPTURA_EVENTO_PRESENTE = 0.6   # 60% do uplift teórico se materializa
    CAPTURA_EVENTO_CONSUMO  = 0.7   # consumo intenso é mais imediato
    CAPTURA_EVENTO_COMERCIAL = 0.3  # BF em posto pouco aplicável

    # Canibalização (Walmart 2021 — 30-50% das vendas promocionais)
    CANIB_BASE = 0.20            # 20% base sem desc
    CANIB_POR_DESC = 1.0          # cada 10% desc → +10% canibalização
    CANIB_MAX = 0.50

    # Limiares de qualidade
    UPLIFT_MIN_BOA = 30.0         # >= 30% uplift
    UPLIFT_MIN_MEDIA = 15.0
    CANIB_MAX_BOA = 35.0          # <= 35% canibalização
    CANIB_MAX_MEDIA = 45.0

    def __init__(self, calibracao_path: str = 'calibracao_v2.json'):
        path = Path(__file__).parent / calibracao_path
        with open(path, encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.cats = {c['categoria']: c for c in self.cfg['categorias']}
        self.cat_idx = {c['categoria']: i for i, c in enumerate(self.cfg['categorias'])}
        self.harmonia = self.cfg.get('harmonia_combo', [])

    # ────────────────── INTERFACE PÚBLICA ──────────────────

    def estimar(self, campanha: dict) -> EstimativaDemanda:
        """Estima demanda promocional. Único método público.

        campanha (dict) precisa de:
            categoria (str) - obrigatório
            intensidade (str) - obrigatório ('nada', 'desc5%', 'desc10%', 'combo', 'liq25%')
            produto_complementar (str) - opcional (para combo)
            dias_total (int) - obrigatório
            desconto_pct (float) - obrigatório
            data_inicio (str) - obrigatório (YYYY-MM-DD)
            eventos_comerciais_na_janela (list[str]) - opcional
            estoque_pct_normal (float) - opcional (1.0=normal, 2.0=dobro)
            validade_restante_pct (float) - opcional (0.15=15% restante)
            giro_alto (bool) - opcional (default True)
        """
        # ─── 1. CLASSIFICAR ───
        cat = campanha['categoria']
        tipo_produto = TIPO_PRODUTO.get(cat, 'commodity')
        eventos = campanha.get('eventos_comerciais_na_janela', [])
        tipo_evento = self._classifica_evento(eventos)

        # ─── 2. PRODUTO PROIBIDO → RETORNO ANTECIPADO ───
        if tipo_produto == 'proibida':
            return self._resultado_proibido(campanha)

        # ─── 3. DADOS DA CATEGORIA ───
        cat_info = self.cats.get(cat, {})
        d_base = float(cat_info.get('demanda_base_dia', 5.0))

        intensidade = campanha.get('intensidade', 'combo')
        par = campanha.get('produto_complementar', '') or ''
        dias = int(campanha.get('dias_total', 3))
        desc_pct = float(campanha.get('desconto_pct', 10)) / 100.0
        data_ini = campanha.get('data_inicio', '2026-01-01')

        # ─── 4. CALCULAR BOOSTS INDIVIDUAIS ───
        elast = ELASTICIDADE[tipo_produto]

        # Boost preço (elasticidade × desconto)
        # Combo: desc é "valor percebido" no produto, captura 40% do efeito direto
        if intensidade == 'combo':
            boost_preco = 1.0 + elast * desc_pct * 0.40
        elif intensidade == 'liq25%':
            boost_preco = 1.0 + elast * desc_pct * 0.90
        elif intensidade in ('desc5%', 'desc10%'):
            boost_preco = 1.0 + elast * desc_pct
        else:
            boost_preco = 1.0

        # Boost combo (harmonia categoria × par)
        boost_combo = 1.0
        if intensidade == 'combo' and par:
            h = self._harmonia(cat, par)
            # Harmonia 1.0 = neutro. Combo amplifica 25% do excesso de harmonia.
            boost_combo = 1.0 + max(0, h - 1.0) * 0.25

        # Boost evento (depende do match produto × evento)
        boost_evento = self._boost_evento(tipo_produto, tipo_evento)

        # Boost clima (mês × tipo de produto)
        mes = int(data_ini[5:7])
        boost_clima = self._boost_clima(tipo_produto, mes)

        # Boost dia-semana
        boost_dow = self._boost_dow(cat, data_ini, dias, intensidade)

        # Boost estoque/giro (estoque alto + giro baixo = desc converte mais)
        boost_eg = self._boost_estoque_giro(campanha, intensidade)

        # Penalidade combo fraco
        penalidade_combo = self._penalidade_combo_fraco(cat, par, intensidade)

        # ─── 5. UPLIFT BRUTO + CAP ───
        uplift_bruto = (boost_preco * boost_combo * boost_evento
                         * boost_clima * boost_dow * boost_eg
                         * penalidade_combo - 1.0)
        cap = CAP_UPLIFT.get(intensidade, 0.30)
        cap_aplicado = uplift_bruto > cap
        uplift_final = min(uplift_bruto, cap)
        d_promo = d_base * (1.0 + uplift_final)

        # ─── 6. CANIBALIZAÇÃO ───
        canib_pct = self._canibalizacao(desc_pct, tipo_evento)

        # ─── 7. CLASSIFICAR + JUSTIFICAR ───
        qualidade, flags = self._classificar_qualidade(
            uplift_final * 100, canib_pct * 100,
            tipo_produto, tipo_evento, intensidade, campanha, cat, par
        )

        confianca = self._confianca(cap_aplicado, uplift_final, tipo_evento)
        motivo = self._gerar_motivo(
            tipo_produto, tipo_evento, d_base, d_promo, uplift_final * 100,
            boost_combo, boost_evento, boost_clima, boost_eg, cap_aplicado,
            canib_pct * 100, cat, par, intensidade, flags
        )

        return EstimativaDemanda(
            categoria=cat,
            intensidade=intensidade,
            par_combo=par if par else None,
            dias=dias,
            desconto_pct=round(desc_pct * 100, 1),
            tipo_produto=tipo_produto,
            tipo_evento=tipo_evento,
            demanda_base_dia=round(d_base, 2),
            demanda_promocional_dia=round(d_promo, 2),
            uplift_pct=round(uplift_final * 100, 1),
            uplift_unidades_dia=round(max(d_promo - d_base, 0), 2),
            canibalizacao_estimada_pct=round(canib_pct * 100, 1),
            nivel_risco_canibalizacao=self._nivel_canib(canib_pct),
            componentes={
                'boost_preco': round(boost_preco, 3),
                'boost_combo': round(boost_combo, 3),
                'boost_evento': round(boost_evento, 3),
                'boost_clima': round(boost_clima, 3),
                'boost_dow': round(boost_dow, 3),
                'boost_estoque_giro': round(boost_eg, 3),
                'penalidade_combo_fraco': round(penalidade_combo, 3),
                'uplift_bruto_pre_cap': round(uplift_bruto * 100, 1),
            },
            cap_aplicado=cap_aplicado,
            qualidade_promocao=qualidade,
            confianca=confianca,
            motivo=motivo,
            flags=flags,
        )

    # ────────────────── HELPERS INTERNOS ──────────────────

    def _classifica_evento(self, eventos: list) -> str:
        if not eventos:
            return 'nenhum'
        for ev in eventos:
            ev_norm = ev.lower()
            for chave, tipo in TIPO_EVENTO.items():
                if chave in ev_norm:
                    return tipo
        return 'comercial'

    def _harmonia(self, a: str, b: str) -> float:
        ia, ib = self.cat_idx.get(a), self.cat_idx.get(b)
        if ia is None or ib is None or not self.harmonia:
            return 1.0
        return float(self.harmonia[ia][ib])

    def _boost_evento(self, tipo_produto: str, tipo_evento: str) -> float:
        """Boost só ativa quando há match produto × evento."""
        if tipo_evento == 'nenhum':
            return 1.0

        # Tabela de match (boost teórico) — calibrada com Olist/Walmart
        match = {
            ('presente', 'puxador_presente'):  0.70,
            ('presente', 'puxador_premium'):    0.50,
            ('presente', 'impulso'):            0.30,
            ('consumo_intenso', 'puxador_consumo'): 0.50,
            ('consumo_intenso', 'commodity'):    0.40,
            ('consumo_intenso', 'impulso'):      0.30,
            ('comercial', 'impulso'):            0.20,
            ('comercial', 'puxador_presente'):   0.15,
        }
        uplift_teorico = match.get((tipo_evento, tipo_produto), 0.05)

        captura = {
            'presente': self.CAPTURA_EVENTO_PRESENTE,
            'consumo_intenso': self.CAPTURA_EVENTO_CONSUMO,
            'comercial': self.CAPTURA_EVENTO_COMERCIAL,
        }.get(tipo_evento, 0.3)

        return 1.0 + uplift_teorico * captura

    def _boost_clima(self, tipo_produto: str, mes: int) -> float:
        """Verão BR (dez-mar) impulsiona consumo. Inverno (jun-ago) penaliza."""
        if mes in [12, 1, 2, 3]:  # verão
            if tipo_produto == 'puxador_consumo':
                return 1.20
            if tipo_produto == 'commodity':
                return 1.10
            if tipo_produto == 'impulso':
                return 1.05
        elif mes in [6, 7, 8]:  # inverno
            if tipo_produto == 'puxador_consumo':
                return 0.88
            if tipo_produto == 'rotina':
                return 1.10  # café/padaria sobe
        return 1.0

    def _boost_dow(self, cat: str, data_ini: str, dias: int, intensidade: str) -> float:
        """Média do fator_dia da calibração no range da campanha.

        REGRA IMPORTANTE: para desc% direto, se dow é fraco (<1.0), boost vira
        1.0 (neutro). Razão: desc em dia fraco MOVE estoque parado, não
        amplifica demanda existente.
        """
        try:
            cat_info = self.cats.get(cat, {})
            fator = cat_info.get('fator_dia', [1] * 7)
            d_ini = date.fromisoformat(data_ini)
            soma = sum(fator[(d_ini.weekday() + i) % 7] for i in range(dias))
            media = soma / dias
            media_cap = max(0.7, min(1.4, media))
            # Desc% em dia fraco → neutro (não penaliza)
            if intensidade in ('desc5%', 'desc10%') and media_cap < 1.0:
                return 1.0
            return media_cap
        except Exception:
            return 1.0

    def _boost_estoque_giro(self, campanha: dict, intensidade: str) -> float:
        """Estoque alto + validade próxima reforça desc."""
        boost = 1.0
        estoque = campanha.get('estoque_pct_normal')
        validade = campanha.get('validade_restante_pct')
        giro_alto = campanha.get('giro_alto', True)

        if estoque is not None and estoque > 1.3:
            boost *= 1.15
        if validade is not None and validade < 0.30:
            # Liquidação em produto perto de vencer é forte
            if intensidade == 'liq25%':
                boost *= 1.30
            else:
                boost *= 1.10
        if not giro_alto and intensidade in ('desc5%', 'desc10%'):
            boost *= 0.90
        return boost

    def _penalidade_combo_fraco(self, cat: str, par: str, intensidade: str) -> float:
        """Combo de produtos sem afinidade tem boost reduzido."""
        if intensidade != 'combo' or not par:
            return 1.0
        h = self._harmonia(cat, par)
        if h < 1.0:
            return 0.85
        if h < 1.2:
            return 0.95
        return 1.0

    def _canibalizacao(self, desc_pct: float, tipo_evento: str) -> float:
        """% das vendas promocionais que aconteceriam mesmo sem promo.

        Walmart 2021: 30-50% típico. Eventos PRESENTE têm menos
        canibalização (cliente compra ESPECÍFICO para a data).
        """
        ajuste_evento = 0.7 if tipo_evento == 'presente' else 1.0
        canib = (self.CANIB_BASE + desc_pct * self.CANIB_POR_DESC) * ajuste_evento
        return min(canib, self.CANIB_MAX)

    def _nivel_canib(self, canib_pct: float) -> str:
        if canib_pct < 0.30: return 'baixo'
        if canib_pct < 0.42: return 'medio'
        return 'alto'

    def _classificar_qualidade(self, uplift_pct: float, canib_pct: float,
                                tipo_produto: str, tipo_evento: str,
                                intensidade: str, campanha: dict,
                                cat: str, par: str) -> tuple:
        """Classifica BOA/MÉDIA/RUIM SEM cálculo de lucro.

        Regras:
        - PROIBIDA: cigarro (já tratado antes)
        - REGRA DO DONO: desc direto (5/10/25%) em puxador (consumo ou premium)
          SEM evento E giro_alto → no MÁXIMO MÉDIA, com bandeira de alerta
        - Combo com harmonia < 1.0 → no MÁXIMO MÉDIA
        - Liquidação em validade < 30% → no MÍNIMO MÉDIA (defensiva sempre vale)
        - Padrão: uplift ≥ 30% E canib < 35% → BOA
        - Senão: uplift ≥ 15% E canib < 45% → MÉDIA
        - Senão: RUIM
        """
        flags = []

        # Regra do dono: desc direto em puxador sem evento
        desc_direto = intensidade in ('desc5%', 'desc10%', 'liq25%')
        eh_puxador = tipo_produto in ('puxador_consumo', 'puxador_premium')
        giro_alto = campanha.get('giro_alto', True)
        validade = campanha.get('validade_restante_pct')
        eh_liquidacao_defensiva = (intensidade == 'liq25%'
                                     and validade is not None
                                     and validade < 0.30)

        if desc_direto and eh_puxador and tipo_evento == 'nenhum' and giro_alto:
            if not eh_liquidacao_defensiva:
                flags.append('desc_direto_em_puxador_destrói_margem')
                return ('🔴 RUIM', flags)

        # Combo absurdo
        if intensidade == 'combo' and par:
            h = self._harmonia(cat, par)
            if h < 1.0:
                flags.append('combo_antagonico')
                return ('🔴 RUIM', flags)

        # Cap atingido
        if uplift_pct >= CAP_UPLIFT.get(intensidade, 0.5) * 100 - 0.5:
            flags.append('cap_uplift_atingido')

        # Canibalização alta
        if canib_pct >= 40:
            flags.append('canibalizacao_alta')

        # Liquidação defensiva: mínimo MÉDIA
        if eh_liquidacao_defensiva:
            if uplift_pct >= 25:
                return ('🟢 BOA', flags + ['estrategia_defensiva_vencimento'])
            return ('🟡 MÉDIA', flags + ['estrategia_defensiva_vencimento'])

        # Classificação padrão
        if uplift_pct >= self.UPLIFT_MIN_BOA and canib_pct <= self.CANIB_MAX_BOA:
            return ('🟢 BOA', flags)
        if uplift_pct >= self.UPLIFT_MIN_MEDIA and canib_pct <= self.CANIB_MAX_MEDIA:
            return ('🟡 MÉDIA', flags)
        return ('🔴 RUIM', flags)

    def _confianca(self, cap: bool, uplift: float, tipo_evento: str) -> str:
        if cap or uplift < 0.10:
            return 'baixa'
        if tipo_evento != 'nenhum':
            return 'alta'
        return 'media'

    def _gerar_motivo(self, tipo_produto, tipo_evento, d_base, d_promo, uplift_pct,
                       boost_combo, boost_evento, boost_clima, boost_eg, cap,
                       canib_pct, cat, par, intensidade, flags) -> str:
        """Motivo textual em português."""
        partes = []
        partes.append(f"Produto {tipo_produto}, evento {tipo_evento}.")
        partes.append(f"Demanda {d_base}→{d_promo} un/dia ({uplift_pct:+.1f}%).")

        if intensidade == 'combo' and par:
            h = self._harmonia(cat, par)
            partes.append(f"Combo com {par} (harmonia {h:.2f}).")

        if boost_evento > 1.1:
            partes.append(
                f"Evento {tipo_evento} bate com produto: +{(boost_evento-1)*100:.0f}%."
            )
        if boost_clima > 1.05:
            partes.append(f"Clima sazonal favorece (+{(boost_clima-1)*100:.0f}%).")
        if boost_eg > 1.05:
            partes.append("Estoque alto/validade próxima reforça desc.")
        if cap:
            partes.append(f"⚠ Atingiu cap máximo de uplift para {intensidade}.")

        if 'estrategia_defensiva_vencimento' in flags:
            partes.append("Liquidação defensiva evita perda total por vencimento.")
        if 'desc_direto_em_puxador_destrói_margem' in flags:
            partes.append("⚠ Desc direto em produto-puxador sem evento: destrói margem.")
        if 'combo_antagonico' in flags:
            partes.append(f"⚠ Combo {cat}+{par} sem afinidade (harmonia < 1.0).")
        if canib_pct >= 40:
            partes.append(f"⚠ Canibalização alta ({canib_pct:.0f}%).")

        return ' '.join(partes)

    def _resultado_proibido(self, campanha: dict) -> EstimativaDemanda:
        cat = campanha['categoria']
        cat_info = self.cats.get(cat, {})
        d_base = float(cat_info.get('demanda_base_dia', 0))
        return EstimativaDemanda(
            categoria=cat,
            intensidade=campanha.get('intensidade', 'nada'),
            par_combo=None,
            dias=campanha.get('dias_total', 0),
            desconto_pct=campanha.get('desconto_pct', 0),
            tipo_produto='proibida',
            tipo_evento='nenhum',
            demanda_base_dia=d_base,
            demanda_promocional_dia=d_base,
            uplift_pct=0,
            uplift_unidades_dia=0,
            canibalizacao_estimada_pct=0,
            nivel_risco_canibalizacao='baixo',
            componentes={'proibido': True},
            cap_aplicado=False,
            qualidade_promocao='🚫 PROIBIDA',
            confianca='alta',
            motivo='Produto proibido pela Lei 9.294/96 (cigarros).',
            flags=['proibida_por_lei'],
        )
