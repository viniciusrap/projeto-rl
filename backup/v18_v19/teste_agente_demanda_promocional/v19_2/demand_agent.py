"""Agente de Demanda Promocional — V19.1 (Fase A + B).

EVOLUÇÃO sobre V19:
  Fase A:
    1. Separação sazonalidade × promoção: demanda_base_contextual ≠ uplift
    2. Lê calendário comercial real (uplift_prior, categorias_afetadas, tipo_pico)
    3. Match SKU × evento via categorias_afetadas (não tipo de produto genérico)
    4. Respeita janela_pre_dias e tipo_pico (pre/no_dia/ambos)
  Fase B:
    5. Pré-feriado prolongado (ponte) modula demanda BASE
    6. Pós-feriado reduz demanda
    7. Eventos esportivos com uplift próprio
    8. Eventos locais (Aniv. Barueri/SP)
    9. Temperatura REAL do CSV (não hardcoded)

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

# V19.2 — TIPO DE UNIDADE DE CONSUMO
# Posto de gasolina vende em duas modalidades:
#   - 'individual': cliente leva 1 unidade no balcão (long-neck, choc.unitário,
#     refri lata, snack, doce, sorvete pote, água, café). Compra de IMPULSO.
#   - 'evento_casa': cliente prepara festa/churrasco em casa. Volume alto.
#     Tipicamente: saco de gelo 5kg, fardo de cerveja (não modelado), garrafa
#     premium (destilados/vinho).
#
# Regra de combo válido: itens devem fazer sentido NA MESMA TRANSAÇÃO no balcão.
# Saco gelo 5kg + 1 garrafa de whisky? Não — quem vai fazer drink em casa compra
# em mercado, não no posto. Quem para no posto leva uma cerveja já gelada.
TIPO_UNIDADE = {
    'cerveja':           'evento_casa_ou_individual',  # ambíguo (long-neck vs fardo)
    'gelo':              'evento_casa',                # saco 5kg
    'destilados':        'individual_premium',         # 1 garrafa, raro no posto
    'vinho':             'individual_premium',
    'chocolate_premium': 'presente',
    'chocolate_impulso': 'individual',
    'snack':             'individual',
    'biscoito':          'individual',
    'doce':              'individual',
    'sorvete':           'individual',
    'cafe':              'rotina_manha',
    'padaria':           'rotina_manha',
    'agua':              'individual',
    'refrigerante':      'individual',
    'suco':              'individual',
    'isotonico':         'individual',
    'energetico':        'individual',
}

# Combos PROIBIDOS POR REGRA DE PDV (override de harmonia)
# Mesmo que a harmonia calibrada na Instacart diga 2.0, no balcão de posto
# esses pares não fazem sentido. Cliente que vai usar o gelo em casa COMPRA
# os destilados em mercado, não no posto.
COMBOS_INVALIDOS_PDV = {
    frozenset(['gelo', 'destilados']),
    frozenset(['gelo', 'vinho']),
    frozenset(['gelo', 'sorvete']),       # ou um ou outro
    frozenset(['cafe', 'cerveja']),
    frozenset(['cafe', 'destilados']),
    frozenset(['cafe', 'vinho']),
    frozenset(['sorvete', 'cerveja']),    # estranho
    frozenset(['padaria', 'cerveja']),    # estranho
    frozenset(['padaria', 'destilados']),
}

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

    # Constantes V19.1
    BOOST_PRE_FERIADO_PROLONGADO = 1.35   # sex/sáb antes de feriado seg (ponte)
    BOOST_PRE_FERIADO_NORMAL = 1.15        # véspera de feriado isolado
    BOOST_POS_FERIADO = 0.85               # seg pós-feriadão (gente viajou)
    BOOST_VESPERA_EVENTO_PRESENTE = 1.20   # demanda BASE sobe na semana pré-Mães
    BOOST_DIA_PAGAMENTO = 1.10             # 5/15/20 do mês

    # Limites de boost contextual (para não explodir)
    DEMANDA_CTX_MIN = 0.5
    DEMANDA_CTX_MAX = 2.5

    def __init__(self,
                  calibracao_path: str = 'calibracao_v2.json',
                  calendario_path: Optional[str] = None,
                  temperatura_path: Optional[str] = None):
        here = Path(__file__).parent
        # Caminhos default: calibracao na pasta da versão; data/ na raiz do projeto
        proj_root = here.parent.parent  # .../projeto-rl
        cal_path = here / calibracao_path
        with open(cal_path, encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.cats = {c['categoria']: c for c in self.cfg['categorias']}
        self.cat_idx = {c['categoria']: i for i, c in enumerate(self.cfg['categorias'])}
        self.harmonia = self.cfg.get('harmonia_combo', [])

        # Carrega calendário comercial (eventos com uplift_prior, janela, tipo_pico)
        if calendario_path is None:
            calendario_path = proj_root / 'data' / 'calendario_comercial.csv'
        self.eventos_calendario = self._carregar_calendario(calendario_path)

        # Carrega temperatura histórica
        if temperatura_path is None:
            temperatura_path = proj_root / 'data' / 'temperatura_historica.csv'
        self.temperatura = self._carregar_temperatura(temperatura_path)

    def _carregar_calendario(self, path) -> list:
        """Lê calendario_comercial.csv → lista de dicts indexados por data."""
        import csv
        eventos = []
        try:
            with open(path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eventos.append({
                        'data': row['data'],
                        'tipo_evento': row['tipo_evento'],
                        'nome': row['nome_evento'],
                        'janela_pre_dias': int(row.get('janela_pre_dias') or 0),
                        'janela_pos_dias': int(row.get('janela_pos_dias') or 0),
                        'intensidade': row.get('intensidade', 'media'),
                        'categorias_afetadas': [
                            c.strip() for c in (row.get('categorias_afetadas') or '').split(';')
                            if c.strip()
                        ],
                        'uplift_prior': float(row.get('uplift_prior') or 1.0),
                        'tipo_pico': row.get('tipo_pico', 'no_dia'),
                    })
        except FileNotFoundError:
            pass
        return eventos

    def _carregar_temperatura(self, path) -> dict:
        """Lê temperatura_historica.csv → dict {data: temp_norm}."""
        import csv
        temps = {}
        try:
            with open(path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                vals = []
                for row in reader:
                    try:
                        vals.append((row['data'], float(row['temp_max'])))
                    except (KeyError, ValueError):
                        continue
                if vals:
                    temps_raw = [v for _, v in vals]
                    t_min, t_max = min(temps_raw), max(temps_raw)
                    rng = max(t_max - t_min, 1.0)
                    temps = {d: (t - t_min) / rng for d, t in vals}
        except FileNotFoundError:
            pass
        return temps

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

        # ═══════════════════════════════════════════════════════════════════
        # V19.1 — SEPARAÇÃO SAZONALIDADE × PROMOÇÃO
        # ═══════════════════════════════════════════════════════════════════

        # ─── 4a. BOOSTS DE SAZONALIDADE (modulam DEMANDA BASE) ───
        # Dia-semana (FATOR_DIA da calibração)
        boost_dow = self._sazonal_dow(cat, data_ini, dias)
        # Mês (FATOR_MES da calibração — sazonalidade ampla)
        boost_mes = self._sazonal_mes(cat, data_ini)
        # Clima REAL via temperatura histórica
        boost_clima_real = self._sazonal_clima(cat, data_ini, tipo_produto)
        # Pré-feriado prolongado (ponte): sex/sáb antes de feriado seg
        boost_pre_feriado = self._sazonal_pre_feriado(data_ini, dias, tipo_produto)
        # Pós-feriado: queda 15% na segunda pós-feriadão (gente viajou)
        boost_pos_feriado = self._sazonal_pos_feriado(data_ini, dias, tipo_produto)
        # Véspera de data-presente: chocolate sobe NATURALMENTE na semana pré-Mães
        boost_vespera_presente = self._sazonal_vespera_presente(
            cat, data_ini, dias, tipo_produto
        )
        # Dia de pagamento (5/15/20/30)
        boost_pagamento = self._sazonal_pagamento(data_ini, dias)

        # Multiplica sazonalidade na demanda base, com cap
        fator_sazonal = (boost_dow * boost_mes * boost_clima_real
                         * boost_pre_feriado * boost_pos_feriado
                         * boost_vespera_presente * boost_pagamento)
        fator_sazonal = max(self.DEMANDA_CTX_MIN,
                            min(self.DEMANDA_CTX_MAX, fator_sazonal))
        d_base_ctx = d_base * fator_sazonal

        # ─── 4b. BOOSTS DE PROMOÇÃO (modulam UPLIFT sobre d_base_ctx) ───
        elast = ELASTICIDADE[tipo_produto]
        # Boost preço
        if intensidade == 'combo':
            boost_preco = 1.0 + elast * desc_pct * 0.40
        elif intensidade == 'liq25%':
            boost_preco = 1.0 + elast * desc_pct * 0.90
        elif intensidade in ('desc5%', 'desc10%'):
            boost_preco = 1.0 + elast * desc_pct
        else:
            boost_preco = 1.0
        # Boost combo (harmonia)
        boost_combo = 1.0
        if intensidade == 'combo' and par:
            h = self._harmonia(cat, par)
            boost_combo = 1.0 + max(0, h - 1.0) * 0.25
        # Boost evento — agora CALIBRADO pelo calendário comercial
        boost_evento, evento_match = self._boost_evento_calibrado(
            cat, tipo_produto, data_ini, dias, eventos, tipo_evento
        )
        # Boost estoque/giro
        boost_eg = self._boost_estoque_giro(campanha, intensidade)
        # Penalidade combo fraco
        penalidade_combo = self._penalidade_combo_fraco(cat, par, intensidade)

        # ─── 5. UPLIFT BRUTO + CAP ───
        # NOTE: clima/dow/mes NÃO entram aqui — eles foram para a demanda base.
        uplift_bruto = (boost_preco * boost_combo * boost_evento
                         * boost_eg * penalidade_combo - 1.0)
        cap = CAP_UPLIFT.get(intensidade, 0.30)
        cap_aplicado = uplift_bruto > cap
        uplift_final = min(uplift_bruto, cap)
        d_promo = d_base_ctx * (1.0 + uplift_final)

        # Compatibilidade com motivo/qualidade que usavam variáveis antigas
        boost_clima = boost_clima_real  # alias

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
            # V19.1: reportamos a DEMANDA BASE CONTEXTUAL (com sazonalidade),
            # não a média anual. Isso bate com o que o RevenueAgent precisa.
            demanda_base_dia=round(d_base_ctx, 2),
            demanda_promocional_dia=round(d_promo, 2),
            uplift_pct=round(uplift_final * 100, 1),
            uplift_unidades_dia=round(max(d_promo - d_base_ctx, 0), 2),
            canibalizacao_estimada_pct=round(canib_pct * 100, 1),
            nivel_risco_canibalizacao=self._nivel_canib(canib_pct),
            componentes={
                # SAZONALIDADE (modula demanda base) — V19.1
                'd_base_anual': round(d_base, 2),
                'd_base_contextual': round(d_base_ctx, 2),
                'fator_sazonal': round(fator_sazonal, 3),
                'sazonal_dow': round(boost_dow, 3),
                'sazonal_mes': round(boost_mes, 3),
                'sazonal_clima_real': round(boost_clima_real, 3),
                'sazonal_pre_feriado': round(boost_pre_feriado, 3),
                'sazonal_pos_feriado': round(boost_pos_feriado, 3),
                'sazonal_vespera_presente': round(boost_vespera_presente, 3),
                'sazonal_pagamento': round(boost_pagamento, 3),
                # PROMOÇÃO (modula uplift sobre d_base_contextual)
                'boost_preco': round(boost_preco, 3),
                'boost_combo': round(boost_combo, 3),
                'boost_evento': round(boost_evento, 3),
                'evento_match': evento_match,
                'boost_estoque_giro': round(boost_eg, 3),
                'penalidade_combo_fraco': round(penalidade_combo, 3),
                'uplift_bruto_pre_cap': round(uplift_bruto * 100, 1),
                # Aliases retro-compatíveis (testes/consumidores antigos)
                'boost_dow': round(boost_dow, 3),
                'boost_clima': round(boost_clima_real, 3),
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
        """Combo de produtos sem afinidade tem boost reduzido.

        V19.2: combos PROIBIDOS POR REGRA DE PDV (saco gelo + garrafa whisky etc)
        recebem penalidade dura (0.6). Isso vai bloqueá-los na classificação.
        """
        if intensidade != 'combo' or not par:
            return 1.0
        # V19.2: regra dura de PDV
        if frozenset([cat, par]) in COMBOS_INVALIDOS_PDV:
            return 0.60   # combo cross-class no balcão
        h = self._harmonia(cat, par)
        if h < 1.0:
            return 0.85
        if h < 1.2:
            return 0.95
        return 1.0

    def _eh_combo_invalido_pdv(self, cat: str, par: str, intensidade: str) -> bool:
        """V19.2: detecta combo que não faz sentido no balcão de posto."""
        if intensidade != 'combo' or not par:
            return False
        return frozenset([cat, par]) in COMBOS_INVALIDOS_PDV

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

    # ═══════════════════════════════════════════════════════════════════
    # V19.1 — SAZONALIDADE (modula DEMANDA BASE)
    # ═══════════════════════════════════════════════════════════════════

    def _sazonal_dow(self, cat: str, data_ini: str, dias: int) -> float:
        """Média do FATOR_DIA da calibração na janela da campanha.

        NÃO aplica a regra V19 de 'neutralizar em dia fraco' — isso fazia
        sentido para UPLIFT, mas para SAZONALIDADE da demanda base, dia
        fraco É dia fraco mesmo.
        """
        try:
            cat_info = self.cats.get(cat, {})
            fator = cat_info.get('fator_dia', [1] * 7)
            d_ini = date.fromisoformat(data_ini)
            soma = sum(fator[(d_ini.weekday() + i) % 7] for i in range(dias))
            return max(0.4, min(2.5, soma / dias))
        except Exception:
            return 1.0

    def _sazonal_mes(self, cat: str, data_ini: str) -> float:
        """FATOR_MES da calibração. V19 IGNORAVA isso completamente."""
        try:
            cat_info = self.cats.get(cat, {})
            fator = cat_info.get('fator_mes', [1] * 12)
            mes = int(data_ini[5:7]) - 1
            return max(0.4, min(2.5, float(fator[mes])))
        except Exception:
            return 1.0

    def _sazonal_clima(self, cat: str, data_ini: str, tipo_produto: str) -> float:
        """Clima via temperatura REAL do CSV histórico (não hardcoded por mês).

        Usa coeficientes simples por tipo de produto:
        - puxador_consumo/commodity (gelo, cerveja, refri): +30% no calor
        - rotina/padaria: -10% no calor (sobe no frio)
        - resto: pequeno efeito
        """
        try:
            # Tenta achar a data exata; se não tem, usa média da janela
            temp = self.temperatura.get(data_ini)
            if temp is None:
                # Fallback: temperatura média histórica para o mês da data
                mes = int(data_ini[5:7])
                # Verão BR aproximado
                temp = 0.75 if mes in (12, 1, 2, 3) else 0.30 if mes in (6, 7, 8) else 0.50
            # Mapeia temperatura normalizada → boost
            if tipo_produto in ('puxador_consumo', 'commodity'):
                # Gelo, cerveja, refri, água: sobem com calor
                return 1.0 + (temp - 0.5) * 0.40   # 0.8 a 1.2
            if tipo_produto == 'impulso':
                return 1.0 + (temp - 0.5) * 0.20   # sorvete, mas menor amp.
            if tipo_produto == 'rotina':
                return 1.0 - (temp - 0.5) * 0.15   # café/padaria sobem no frio
            return 1.0
        except Exception:
            return 1.0

    def _sazonal_pre_feriado(self, data_ini: str, dias: int,
                              tipo_produto: str) -> float:
        """Boost se campanha está em PRÉ-feriado prolongado (ponte sex/sáb).

        Procura no calendário comercial feriados oficiais nos 1-3 dias APÓS
        a campanha. Boost maior para cerveja/snack/gelo (churrasco).
        """
        if not self.eventos_calendario:
            return 1.0
        try:
            d_ini = date.fromisoformat(data_ini)
            from datetime import timedelta
            datas_campanha = [d_ini + timedelta(days=i) for i in range(dias)]

            # Cada dia da campanha: é véspera de feriado prolongado?
            boost_max = 1.0
            for dc in datas_campanha:
                # Olha 1-3 dias para frente
                for offset in (1, 2, 3):
                    feriado_dt = dc + timedelta(days=offset)
                    fer_str = feriado_dt.isoformat()
                    feriado = next((e for e in self.eventos_calendario
                                     if e['data'] == fer_str
                                     and e['tipo_evento'] == 'feriado_oficial'), None)
                    if feriado is None:
                        continue
                    # Detecta "ponte": feriado em seg/ter/qui/sex deixa fim de
                    # semana prolongado. Sex de campanha pra feriado de seg → ponte.
                    weekday_feriado = feriado_dt.weekday()
                    weekday_campanha = dc.weekday()
                    eh_ponte = (
                        (weekday_feriado == 0 and weekday_campanha == 4) or  # sex→seg
                        (weekday_feriado == 4 and weekday_campanha in (1, 2)) or  # ter/qua→sex
                        (weekday_feriado in (1, 3))  # ter ou qui = ponte natural
                    )
                    if eh_ponte:
                        b = self.BOOST_PRE_FERIADO_PROLONGADO
                    else:
                        b = self.BOOST_PRE_FERIADO_NORMAL
                    # Categorias que mais ganham: cerveja, snack, gelo
                    if tipo_produto in ('puxador_consumo', 'impulso'):
                        boost_max = max(boost_max, b)
                    else:
                        # Outros ganham menos
                        boost_max = max(boost_max, 1.0 + (b - 1.0) * 0.5)
            return boost_max
        except Exception:
            return 1.0

    def _sazonal_pos_feriado(self, data_ini: str, dias: int,
                              tipo_produto: str) -> float:
        """Queda na segunda/terça pós-feriadão (gente viajou)."""
        if not self.eventos_calendario:
            return 1.0
        try:
            d_ini = date.fromisoformat(data_ini)
            from datetime import timedelta
            datas_campanha = [d_ini + timedelta(days=i) for i in range(dias)]
            penaliza = False
            for dc in datas_campanha:
                # Foi feriado 1-2 dias antes?
                for offset in (1, 2):
                    pre_dt = dc - timedelta(days=offset)
                    feriado = next((e for e in self.eventos_calendario
                                     if e['data'] == pre_dt.isoformat()
                                     and e['tipo_evento'] == 'feriado_oficial'), None)
                    if feriado:
                        # Penaliza só seg/ter pós-feriadão
                        if dc.weekday() in (0, 1):
                            penaliza = True
            return self.BOOST_POS_FERIADO if penaliza else 1.0
        except Exception:
            return 1.0

    def _sazonal_vespera_presente(self, cat: str, data_ini: str, dias: int,
                                    tipo_produto: str) -> float:
        """Demanda BASE sobe na semana pré-data-presente (Mães, Pais, etc).

        Diferente do boost_evento (que é da promoção): aqui é o efeito
        NATURAL — chocolate vende mais em Mães mesmo SEM promoção.
        """
        if not self.eventos_calendario:
            return 1.0
        try:
            d_ini = date.fromisoformat(data_ini)
            from datetime import timedelta

            # Procura datas comerciais 'presente' nos próximos 14 dias
            for offset in range(0, 14):
                check_dt = d_ini + timedelta(days=offset)
                check_str = check_dt.isoformat()
                ev = next((e for e in self.eventos_calendario
                            if e['data'] == check_str
                            and e['tipo_evento'] == 'data_comercial'
                            and e['tipo_pico'] in ('pre', 'ambos')), None)
                if ev is None:
                    continue
                # Estamos dentro da janela pré-evento?
                if offset > ev['janela_pre_dias']:
                    continue
                # A categoria está afetada?
                if not self._categoria_afetada(cat, ev['categorias_afetadas']):
                    continue
                # Boost decai linearmente com distância (forte mais perto)
                proximidade = 1.0 - (offset / max(ev['janela_pre_dias'], 1))
                # Intensidade do evento amplifica
                intens_mult = {
                    'muito_alta': 1.0, 'alta': 0.75,
                    'media': 0.50, 'baixa': 0.25
                }.get(ev['intensidade'], 0.50)
                boost = 1.0 + (self.BOOST_VESPERA_EVENTO_PRESENTE - 1.0) \
                          * proximidade * intens_mult
                return min(boost, 1.50)
            return 1.0
        except Exception:
            return 1.0

    def _sazonal_pagamento(self, data_ini: str, dias: int) -> float:
        """Dias de pagamento (5/15/20/30) têm ticket médio maior.

        Heurística simples: se algum dia da campanha cai em 5/15/20/30,
        boost leve. Posto de bairro é especialmente sensível.
        """
        try:
            d_ini = date.fromisoformat(data_ini)
            from datetime import timedelta
            datas_campanha = [d_ini + timedelta(days=i) for i in range(dias)]
            dias_pgto = {5, 15, 20, 30}
            pct_pgto = sum(1 for dc in datas_campanha if dc.day in dias_pgto) / dias
            return 1.0 + (self.BOOST_DIA_PAGAMENTO - 1.0) * pct_pgto
        except Exception:
            return 1.0

    # ═══════════════════════════════════════════════════════════════════
    # V19.1 — BOOST EVENTO CALIBRADO (uplift_prior + categorias_afetadas)
    # ═══════════════════════════════════════════════════════════════════

    def _categoria_afetada(self, cat: str, lista: list) -> bool:
        """Cat está nas categorias afetadas do evento?

        Faz match flexível: 'chocolate' bate em 'chocolate_premium' e
        'chocolate_impulso'; 'vinho' bate em 'vinho'.
        """
        if not lista or 'todas' in lista:
            return True
        cat_lower = cat.lower()
        for c in lista:
            cl = c.lower().strip()
            if cl in cat_lower or cat_lower in cl:
                return True
            # Casos especiais
            if cl == 'cerveja' and cat_lower in ('cerveja',):
                return True
            if cl == 'salgadinho' and cat_lower == 'snack':
                return True
            if cl == 'cerveja_premium' and 'cerveja' in cat_lower:
                return True
        return False

    def _boost_evento_calibrado(self, cat: str, tipo_produto: str,
                                 data_ini: str, dias: int,
                                 eventos_campanha: list,
                                 tipo_evento: str) -> tuple:
        """Boost promocional do evento — usa uplift_prior do calendário.

        Retorna (boost, descricao_match).
        """
        if not eventos_campanha or tipo_evento == 'nenhum':
            return 1.0, 'sem_evento'

        # Procura no calendário comercial os eventos que batem com o nome
        eventos_encontrados = []
        for nome in eventos_campanha:
            nome_lower = nome.lower().strip()
            for e in self.eventos_calendario:
                if nome_lower in e['nome'].lower():
                    eventos_encontrados.append(e)
                    break

        if not eventos_encontrados:
            # Fallback: usa lógica V19 antiga
            return self._boost_evento_v19_fallback(tipo_produto, tipo_evento), 'fallback_v19'

        # Pega o evento com maior uplift_prior entre os candidatos
        ev_principal = max(eventos_encontrados, key=lambda e: e['uplift_prior'])
        afeta_cat = self._categoria_afetada(cat, ev_principal['categorias_afetadas'])

        if not afeta_cat:
            # Evento existe mas não afeta esta categoria (ex: cerveja em Mães)
            return 1.0, f"evento_{ev_principal['nome']}_mas_nao_afeta_{cat}"

        # uplift_prior é multiplicativo (2.5 = +150%)
        # Para PROMOÇÃO, capturamos uma FRAÇÃO desse uplift natural — a promoção
        # AMPLIFICA o que o evento já causa naturalmente.
        # Boost promocional = 1 + (uplift_prior - 1) × captura
        captura = {
            'muito_alta': 0.45,   # Mães=2.2 × 0.45 = +54% boost da promo
            'alta':       0.40,
            'media':      0.30,
            'baixa':      0.20,
        }.get(ev_principal['intensidade'], 0.30)

        boost = 1.0 + (ev_principal['uplift_prior'] - 1.0) * captura
        return boost, f"{ev_principal['nome']}_uplift{ev_principal['uplift_prior']}_cat={cat}"

    def _boost_evento_v19_fallback(self, tipo_produto: str, tipo_evento: str) -> float:
        """Fallback para eventos sem calibração no calendário."""
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

        # V19.2: combo cross-class no balcão de posto
        # Ex: saco gelo 5kg + garrafa whisky — cliente não combina no balcão
        if self._eh_combo_invalido_pdv(cat, par, intensidade):
            flags.append('combo_invalido_pdv')
            return ('🔴 RUIM', flags)

        # Combo absurdo (harmonia < 1.0)
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
