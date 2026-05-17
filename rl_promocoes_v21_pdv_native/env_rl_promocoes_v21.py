"""V20 — Ambiente RL para decisão de promoção em conveniência de posto.

Arquitetura: Gymnasium env com ação MultiDiscrete([6, N+1, 2]):
  - Cabeça INTENSIDADE (Agente de Desconto): 0=nada, 1=desc3%, 2=desc5%, 3=desc7%, 4=desc10%, 5=combo
  - Cabeça COMPLEMENTAR (Agente de Combo): 0=nenhum, 1..N=índice categoria
  - Cabeça ALVO (Agente de Margem): 0=principal, 1=complementar

A FUNÇÃO DE TRANSIÇÃO (step) usa a calibração V19.1 (Fase A+B) para estimar:
  - demanda base contextual (sazonalidade)
  - demanda promocional (uplift do desconto + combo + evento)
  - lucro real (uplift + halo - canibalização - custo op + defensivo)

A RECOMPENSA codifica todas as regras V19/V19.1/V19.2 como shaping (NÃO hard rules):
  - + bonus combo em alta demanda
  - + bonus desconto no complementar (não no principal)
  - + bonus defensivo (liquidação em validade <30%)
  - - penalidade desc direto em produto em alta natural
  - - penalidade combo PDV-inválido (gelo+destilados etc) — agente APRENDE a evitar
  - - penalidade canibalização alta
  - - penalidade repetir categoria nos últimos 7 dias
  - - penalidade baixo uplift (< breakeven)

Episódio: 30 turnos (1 mês). A cada turno o env apresenta UM produto candidato
e o agente decide intensidade × complementar × alvo. Estado evolui (estoque
consumido, histórico atualiza).
"""
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES DE DOMÍNIO (vindas da V19 — mantidas como conhecimento)
# ═══════════════════════════════════════════════════════════════════════

# V21 — TIPO_PRODUTO vem agora da CALIBRAÇÃO (não hardcoded)
# A calibração V21 PDV-native define 20 categorias + 3 cigarros proibidos
# Tipos: commodity, puxador_consumo, puxador_premium, puxador_presente,
#        impulso, rotina, utilitario_evento, proibida
TIPO_PRODUTO = {}  # preenchido em __init__ a partir da calibração

ELASTICIDADE = {
    'commodity': 0.4, 'rotina': 0.4, 'puxador_premium': 0.5,
    'impulso': 0.9, 'puxador_consumo': 1.1, 'puxador_presente': 1.5,
    'utilitario_evento': 0.7,  # NOVO: gelo, carvão
    'proibida': 0.0,
}

# Cap de uplift por intensidade
CAP_UPLIFT = {0: 0.00, 1: 0.20, 2: 0.30, 3: 0.40, 4: 0.50, 5: 0.85}
# Idx → intensidade label
INTENSIDADE_LABEL = {0: 'nada', 1: 'desc3%', 2: 'desc5%', 3: 'desc7%',
                      4: 'desc10%', 5: 'combo'}
INTENSIDADE_DESCONTO_PCT = {0: 0.0, 1: 3.0, 2: 5.0, 3: 7.0, 4: 10.0, 5: 10.0}

# Constantes de recompensa (reward shaping) — calibradas para serem MATERIAIS
# vs. lucro per turno (~R$ 30-50/dia)
# ITER 16 — equilibrar 9/9 com top pares V19.2
K_COMBO_EM_ALTA = 180.0              # ↑ 100→180
K_DESC_COMPLEMENTAR = 50.0
K_DEFENSIVO_VENCIMENTO = 350.0
K_DESC_EM_ALTA_NATURAL = -200.0
K_COMBO_PDV_INVALIDO = -500.0
K_CANIBALIZACAO_ALTA = -8.0
K_REPETICAO_CATEGORIA = -130.0
K_UPLIFT_ABAIXO_BREAKEVEN = -500.0   # V21 iter17: ↑ -350→-500 (combos marginais)
K_DEMANDA_MUITO_BAIXA = -600.0       # V21 iter16: ↑ -400→-600 (isotonico tb)
K_ESQUENTA_SEXTA = 2000.0            # V21 iter12: ↑ 1200→2000 (Esquenta TEM que emergir)
K_CERVEJA_NOITURNA = 500.0           # V21 iter12: ↑ 300→500
K_COMBO_IDEAL_EVENTO = 1000.0
K_PROMOVER_CERVEJA_SEX = 600.0       # V21 iter12: bonus FORTE pra cerveja em sex/sáb com QUALQUER int>0
# V21 iter21 — incentivos para PROMOÇÃO INDIVIDUAL (desc%, sem combo)
# antes 100% combo; desc% individual nunca compensava
K_DESC_IND_VENCIMENTO = 500.0        # desc 7/10% em produto vencendo (liquidação)
K_DESC_IND_BAIXA_GIRO = 180.0        # V21 iter22: 350→180
K_DESC_IND_COMMODITY = 90.0          # V21 iter22: 200→90 (desc leve commodity)

# COMBO IDEAL POR EVENTO (categoria_principal, par) → válido para esse evento
# Conhecimento de domínio: o que cliente leva no posto perto de cada data
# V21 iter18 — combos ideais REVISADOS criticamente para PDV de posto.
# Cada combo: o que o cliente REALMENTE leva no balcão perto da data.
_NAMORADOS = [
    ('chocolate_caixa', 'doce_balcao'),   # caixa bombom + balinha (presente)
    ('chocolate_caixa', 'chocolate_unit'), # kit chocolate
    ('cerveja', 'snack_salgado'),          # casal curtindo em casa
]
_MAES = [
    ('chocolate_caixa', 'doce_balcao'),    # presente
    ('chocolate_caixa', 'biscoito'),       # cesta café da manhã da mãe
    ('chocolate_caixa', 'cafe'),           # kit café da manhã
]
_PAIS = [
    ('cerveja', 'snack_salgado'),          # clássico do pai (cerveja+amendoim)
    ('snack_salgado', 'refrigerante'),     # galera junta vendo jogo
    ('chocolate_caixa', 'cafe'),           # pai que gosta de doce
]
_CRIANCAS = [
    ('chocolate_unit', 'doce_balcao'),
    ('chocolate_unit', 'refrigerante'),
    ('sorvete', 'biscoito'),
    ('refrigerante', 'biscoito'),
]
_PASCOA = [
    ('chocolate_caixa', 'chocolate_unit'),  # ovo + bombom
    ('chocolate_caixa', 'biscoito'),
    ('chocolate_caixa', 'doce_balcao'),
]
_REVEILLON = [
    ('cerveja', 'snack_salgado'),           # festa (gelo é mascarado, fora)
    ('snack_salgado', 'refrigerante'),
    ('refrigerante', 'snack_salgado'),
]
_NATAL = [
    ('chocolate_caixa', 'biscoito'),        # panettone-substituto
    ('chocolate_caixa', 'doce_balcao'),
    ('cerveja', 'snack_salgado'),           # ceia
]
_MULHER = [
    ('chocolate_caixa', 'doce_balcao'),
    ('chocolate_caixa', 'biscoito'),
]
_BLACKFRIDAY = [
    ('cerveja', 'snack_salgado'),
    ('chocolate_caixa', 'doce_balcao'),
]
# V21 iter21: Copa do Mundo — cerveja+petisco para ver o jogo (consumo no dia)
_COPA = [
    ('cerveja', 'snack_salgado'),       # combo clássico de jogo
    ('snack_salgado', 'cerveja'),
    ('snack_salgado', 'refrigerante'),  # quem não bebe
    ('refrigerante', 'snack_salgado'),
]
COMBOS_IDEAIS_EVENTO = {
    'dia dos namorados': _NAMORADOS,
    'dia das maes': _MAES, 'dia das mães': _MAES,
    'dia dos pais': _PAIS,
    'dia das criancas': _CRIANCAS, 'dia das crianças': _CRIANCAS,
    'pascoa': _PASCOA, 'páscoa': _PASCOA,
    'reveillon': _REVEILLON, 'réveillon': _REVEILLON,
    'vespera de natal': _NATAL, 'véspera de natal': _NATAL,
    'dia da mulher': _MULHER, 'dia internacional da mulher': _MULHER,
    'black friday': _BLACKFRIDAY,
}
K_BONUS_EVENTO_MATCH = 120.0          # V21 iter20: 250→120 (campanhas fracas pré-evento)
K_BONUS_EVENTO_PRESENTE_PUXADOR = 350.0  # V21 iter20: 500→350

# V21 iter23: combo DEFLACIONADO para desc% individual competir naturalmente
K_BONUS_HARMONIA_FORTE = 300.0        # ↓ 600→300
K_BONUS_HARMONIA_MEDIA = 30.0
K_PENALIDADE_HARMONIA_LEVE = -100.0
K_BONUS_HARMONIA_MUITO_FORTE = 150.0  # ↓ 300→150
K_PENALIDADE_HARMONIA_FRACA = -200.0
K_HARMONIA_MULTIPLICATIVO = 200.0     # ↓ 400→200

# Combos PDV-inválidos — DEFINIDOS aqui só para reward shaping, NÃO bloqueiam
PARES_INVALIDOS_PDV = {
    frozenset(['gelo', 'destilados']),
    frozenset(['gelo', 'vinho']),
    frozenset(['gelo', 'sorvete']),
    frozenset(['cafe', 'cerveja']),
    frozenset(['cafe', 'destilados']),
    frozenset(['cafe', 'vinho']),
    frozenset(['sorvete', 'cerveja']),
    frozenset(['padaria', 'cerveja']),
    frozenset(['padaria', 'destilados']),
}

# Eventos do calendário comercial → tipo
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
    'carnaval': 'consumo_intenso', 'copa': 'consumo_intenso',
    'black friday': 'comercial',
    'cyber monday': 'comercial',
}

# Custos operacionais (do RevenueAgent)
CUSTO_OP_BASE = 28.0       # ITER 19: intermediário (22 = 305 camp, 35 = 230 perdeu C1)
CUSTO_OP_POR_DIA = 2.5     # ITER 19: 2.2 → 2.5
MARGEM_CROSS_SELL = 3.50


# ═══════════════════════════════════════════════════════════════════════
# ESTADO E AÇÃO
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StepInfo:
    """Informações detalhadas de cada step (para debug/logs/validação)."""
    categoria: str
    intensidade: str
    par_combo: Optional[str]
    alvo_desconto: str
    data: str
    demanda_base_anual: float
    demanda_base_contextual: float
    demanda_promocional: float
    uplift_pct: float
    canib_pct: float
    lucro_uplift_dia: float
    lucro_halo_dia: float
    lucro_canib_dia: float
    lucro_defensivo_dia: float
    custo_op: float
    lucro_total: float
    reward: float
    reward_breakdown: dict
    eh_combo_invalido_pdv: bool
    eh_alta_natural: bool
    evento_proximo: Optional[str]


class EnvRLPromocoes(gym.Env):
    """Ambiente RL para decisão promocional em conveniência de posto."""

    metadata = {'render_modes': []}

    # Espaço temporal: 30 turnos = 30 decisões num episódio
    EPISODIO_TURNOS = 30
    # Janela histórico
    HIST_DIAS = 7

    def __init__(self,
                  calibracao_path: Optional[str] = None,
                  calendario_path: Optional[str] = None,
                  temperatura_path: Optional[str] = None,
                  seed: Optional[int] = None):
        super().__init__()
        proj_root = Path(__file__).parent.parent

        # V21 PDV-native: calibração SEM Instacart, baseada em inventário
        # real do posto + conhecimento de domínio de conveniência
        if calibracao_path is None:
            calibracao_path = (Path(__file__).parent
                                / 'data_sintetica' / 'calibracao_v21_pdv.json')
        with open(calibracao_path, encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.categorias = [c['categoria'] for c in self.cfg['categorias']]
        self.cats = {c['categoria']: c for c in self.cfg['categorias']}
        self.cat_idx = {c: i for i, c in enumerate(self.categorias)}
        self.harmonia = self.cfg.get('harmonia_combo', [])
        self.N_CATEGORIAS = len(self.categorias)

        # V21: preenche TIPO_PRODUTO global a partir da calibração
        for c in self.cfg['categorias']:
            TIPO_PRODUTO[c['categoria']] = c['tipo']

        # V21: PARES_INVALIDOS_PDV gerado dinamicamente (harmonia < 0.85)
        # NÃO é hard rule no espaço de decisão — só sinaliza reward shaping
        global PARES_INVALIDOS_PDV
        PARES_INVALIDOS_PDV = set()
        if self.harmonia:
            promov = [c['categoria'] for c in self.cfg['categorias']
                       if c.get('promovivel', True)]
            for a in promov:
                for b in promov:
                    if a >= b: continue
                    ia, ib = self.cat_idx[a], self.cat_idx[b]
                    if self.harmonia[ia][ib] < 0.85:
                        PARES_INVALIDOS_PDV.add(frozenset([a, b]))

        # Carrega calendário comercial
        if calendario_path is None:
            calendario_path = proj_root / 'data' / 'calendario_comercial.csv'
        self.eventos = self._carregar_calendario(calendario_path)

        # Carrega temperatura
        if temperatura_path is None:
            temperatura_path = proj_root / 'data' / 'temperatura_historica.csv'
        self.temperatura = self._carregar_temperatura(temperatura_path)

        # ═════ ESPAÇOS GYM ═════
        # Action: 3 cabeças (MultiDiscrete)
        self.action_space = spaces.MultiDiscrete([
            6,                       # intensidade
            self.N_CATEGORIAS + 1,   # complementar (0=nenhum)
            2,                       # alvo desconto (0=principal, 1=complementar)
        ])

        # State: vetor float
        self.state_dim = self._calcular_state_dim()
        self.observation_space = spaces.Box(
            low=-2.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )

        # Random state
        self._rng = np.random.default_rng(seed)

        # Estado do episódio (inicializado em reset)
        self._reset_estado_interno()

    def _reset_estado_interno(self):
        self.turno = 0
        self.data_atual = None
        self.produto_atual_idx = 0
        # Histórico (últimos HIST_DIAS turnos): categorias promovidas
        self.hist_categorias = []
        # V21 iter18: conta uso de cada PAR de combo no episódio (anti-repetição)
        self.par_uso_count = {}
        # Receita acumulada
        self.receita_acum = 0.0
        # Estoque relativo por categoria (1.0 = normal)
        self.estoque_rel = np.ones(self.N_CATEGORIAS, dtype=np.float32)
        # Validade relativa por categoria (1.0 = fresco)
        self.validade_rel = np.ones(self.N_CATEGORIAS, dtype=np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # GYM API
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_estado_interno()

        # Data inicial: sorteia entre datas para cobrir sazonalidades
        anos = [2026, 2027]
        ano = self._rng.choice(anos)
        # Evita janeiro 1 e 31 dez (bordas)
        dia_do_ano = self._rng.integers(15, 350)
        self.data_atual = date(ano, 1, 1) + timedelta(days=int(dia_do_ano))

        # Sorteia produto inicial (excluindo proibidos)
        self.produto_atual_idx = self._sortear_produto()

        # Ruído inicial: alguns produtos com estoque alto/validade baixa
        # (simula situações realistas que o agente precisa lidar)
        for i in range(self.N_CATEGORIAS):
            if self._rng.random() < 0.25:
                self.estoque_rel[i] = self._rng.uniform(1.3, 2.0)
            if self._rng.random() < 0.15:
                self.validade_rel[i] = self._rng.uniform(0.10, 0.30)

        return self._observar(), {}

    def step(self, action):
        intensidade_idx = int(action[0])
        complementar_idx = int(action[1])
        alvo_idx = int(action[2])

        info = self._executar_decisao(intensidade_idx, complementar_idx, alvo_idx)
        reward = info.reward

        # Atualiza estado interno
        self._aplicar_efeito_no_estoque(info)
        self._atualizar_historico(info.categoria)
        # V21 iter18: registra uso do par (anti-repetição)
        if info.intensidade == 'combo' and info.par_combo:
            pk = tuple(sorted([info.categoria, info.par_combo]))
            self.par_uso_count[pk] = self.par_uso_count.get(pk, 0) + 1
        self.turno += 1
        # Avança no tempo (1 dia)
        self.data_atual += timedelta(days=1)
        # Sorteia próximo produto candidato
        self.produto_atual_idx = self._sortear_produto()

        terminated = self.turno >= self.EPISODIO_TURNOS
        truncated = False

        return self._observar(), float(reward), terminated, truncated, info.__dict__

    # ═══════════════════════════════════════════════════════════════════
    # LÓGICA CENTRAL: estimar demanda, lucro e reward
    # ═══════════════════════════════════════════════════════════════════

    def _executar_decisao(self, intensidade_idx, complementar_idx,
                            alvo_idx) -> StepInfo:
        """Função de transição: dada a ação, simula a campanha e devolve info+reward.

        ESTA É A "FUNÇÃO DE TRANSIÇÃO" DO MDP — usa lógica calibrada V19.1
        para estimar demanda e lucro REALISTA.
        """
        cat = self.categorias[self.produto_atual_idx]
        cat_info = self.cats[cat]
        tipo_produto = TIPO_PRODUTO.get(cat, 'commodity')

        # Tipo de evento próximo
        evento, dias_evento, evt_tipo = self._evento_proximo(self.data_atual)
        evt_label = evento['nome'] if evento else None
        afeta_categoria = (evento is not None
                            and self._categoria_afetada(cat, evento['categorias_afetadas']))

        # Produto complementar
        par_combo = None
        if intensidade_idx == 5 and 1 <= complementar_idx <= self.N_CATEGORIAS:
            par_combo = self.categorias[complementar_idx - 1]
            if par_combo == cat:
                par_combo = None  # complementar igual ao principal não conta

        # Alvo do desconto (só faz sentido se for combo)
        alvo_str = 'principal' if alvo_idx == 0 else 'complementar'
        if intensidade_idx != 5:
            alvo_str = 'principal'

        # ─── 1. CALCULAR DEMANDA BASE CONTEXTUAL ───
        d_base_anual = float(cat_info.get('demanda_base_dia', 5.0))
        fator_sazonal = self._fator_sazonal(cat, self.data_atual, tipo_produto)
        d_base_ctx = d_base_anual * fator_sazonal

        # ─── 2. CALCULAR UPLIFT PROMOCIONAL ───
        elast = ELASTICIDADE.get(tipo_produto, 0.5)
        desc_pct = INTENSIDADE_DESCONTO_PCT[intensidade_idx] / 100.0

        # Boost preço
        if intensidade_idx == 5:  # combo
            # Combo: 5% no produto (alvo) — desc reduzido se for no complementar
            boost_preco = 1.0 + elast * 0.05 * 0.40  # combo captura 40% do efeito direto
        elif intensidade_idx == 0:
            boost_preco = 1.0
        else:
            boost_preco = 1.0 + elast * desc_pct

        # Boost combo (harmonia)
        boost_combo = 1.0
        if intensidade_idx == 5 and par_combo:
            harm = self._harmonia(cat, par_combo)
            boost_combo = 1.0 + max(0, harm - 1.0) * 0.25

        # Boost evento
        boost_evento = 1.0
        if evento and afeta_categoria:
            captura = {'muito_alta': 0.45, 'alta': 0.40,
                        'media': 0.30, 'baixa': 0.20}.get(evento['intensidade'], 0.30)
            boost_evento = 1.0 + (evento['uplift_prior'] - 1.0) * captura

        # Boost estoque alto (estoque alto + desc ajuda a girar)
        boost_estoque = 1.0
        if self.estoque_rel[self.produto_atual_idx] > 1.3 and intensidade_idx > 0:
            boost_estoque = 1.10
        # Validade próxima + liquidação
        if self.validade_rel[self.produto_atual_idx] < 0.3 and intensidade_idx >= 4:
            boost_estoque *= 1.20

        # Uplift agregado (com cap)
        uplift_bruto = (boost_preco * boost_combo * boost_evento
                         * boost_estoque - 1.0)
        cap = CAP_UPLIFT[intensidade_idx]
        uplift = min(uplift_bruto, cap)
        uplift = max(uplift, 0.0)
        d_promo = d_base_ctx * (1.0 + uplift)

        # ─── 3. CANIBALIZAÇÃO ───
        canib_base = 0.20 + desc_pct * 1.0
        if evt_tipo == 'presente':
            canib_base *= 0.7
        canib_pct = min(canib_base, 0.50)

        # ─── 4. COMPONENTES DE LUCRO (em R$/dia) ───
        margem = float(cat_info.get('margem', 4.0))
        custo_unit = float(cat_info.get('custo', margem * 0.6))

        uplift_un = max(d_promo - d_base_ctx, 0)
        lucro_uplift_dia = uplift_un * margem * (1 - desc_pct)
        vendas_canib = d_promo * canib_pct
        lucro_canib_dia = -vendas_canib * margem * desc_pct

        # Halo: cross-sell de outros itens
        if intensidade_idx == 5:
            halo_pct = 0.10
        else:
            halo_pct = 0.05
        lucro_halo_dia = d_promo * halo_pct * MARGEM_CROSS_SELL

        # Defensivo (liquidação que evita perda)
        lucro_defensivo_dia = 0.0
        validade = self.validade_rel[self.produto_atual_idx]
        if intensidade_idx >= 4 and validade < 0.30:
            estoque_em_risco_dia = d_base_ctx * 1.5
            prob_vencer = min((1 - validade) * 1.5, 0.95)
            estoque_salvo_dia = estoque_em_risco_dia * prob_vencer
            lucro_defensivo_dia = estoque_salvo_dia * custo_unit * 0.7

        # Lucro líquido/dia
        lucro_dia = (lucro_uplift_dia + lucro_canib_dia
                     + lucro_halo_dia + lucro_defensivo_dia)

        # Campanha dura 3 dias (padrão)
        dias = 3 if intensidade_idx > 0 else 1
        lucro_bruto = lucro_dia * dias
        if intensidade_idx == 0:
            custo_op = 0.0  # não fazer promo = sem custo extra
        else:
            custo_op = CUSTO_OP_BASE + CUSTO_OP_POR_DIA * dias

        lucro_total = lucro_bruto - custo_op

        # ─── 5. REWARD SHAPING (codifica regras como aprendizado) ───
        reward = lucro_total
        breakdown = {'lucro_base': lucro_total}

        # Bonus: combo em alta demanda (estratégia ideal V19)
        em_alta = fator_sazonal > 1.15
        if intensidade_idx == 5 and em_alta:
            reward += K_COMBO_EM_ALTA
            breakdown['bonus_combo_em_alta'] = K_COMBO_EM_ALTA

        # Bonus: desconto no complementar, não no principal
        if intensidade_idx == 5 and alvo_idx == 1 and par_combo:
            reward += K_DESC_COMPLEMENTAR
            breakdown['bonus_desc_complementar'] = K_DESC_COMPLEMENTAR

        # V21 iter23: DEFENSIVO exclusivo desc% individual (3,4), vencimento <0.25
        if intensidade_idx in (3, 4) and validade < 0.25:
            reward += K_DEFENSIVO_VENCIMENTO
            breakdown['bonus_defensivo'] = K_DEFENSIVO_VENCIMENTO

        # V21 iter24: incentivos desc% individual — TODOS exigem lucro real >0
        # Liquidação em vencimento (com lucro defensivo positivo)
        if intensidade_idx in (3, 4) and validade < 0.25 and lucro_total > 0:
            reward += K_DESC_IND_VENCIMENTO
            breakdown['bonus_desc_ind_vencimento'] = K_DESC_IND_VENCIMENTO
        # Girar parado: estoque alto + lucro real positivo
        if (intensidade_idx in (1, 2, 3, 4) and not par_combo
                and self.estoque_rel[self.produto_atual_idx] > 1.3
                and lucro_total > 10):
            reward += K_DESC_IND_BAIXA_GIRO
            breakdown['bonus_desc_ind_baixa'] = K_DESC_IND_BAIXA_GIRO
        # Commodity desc leve lucrativa (água/refri/suco — venda de volume)
        if (intensidade_idx in (1, 2) and tipo_produto == 'commodity'
                and not par_combo and lucro_total > 10):
            reward += K_DESC_IND_COMMODITY
            breakdown['bonus_desc_ind_commodity'] = K_DESC_IND_COMMODITY
        # V21 iter24: PENALIDADE desc% individual com prejuízo (limpa negativas)
        if intensidade_idx in (1, 2, 3, 4) and not par_combo and lucro_total < 0:
            reward -= 250.0
            breakdown['pen_desc_ind_prejuizo'] = -250.0

        # V21 iter15: bonus match produto × evento ESCALADO POR PROXIMIDADE
        # dias_evento=0 → 100% bonus; dias_evento=7 → 30% bonus
        # Evita bonus em campanhas longe da data
        if evento and afeta_categoria and intensidade_idx > 0:
            janela = evento.get('janela_pre_dias', 7)
            proximidade = max(0.3, 1.0 - (dias_evento or 0) / max(janela, 1))
            bonus_ev = K_BONUS_EVENTO_MATCH * proximidade
            reward += bonus_ev
            breakdown['bonus_evento_match'] = round(bonus_ev, 1)

        # PENALIDADE: desc direto em produto em alta natural (regra do dono)
        eh_alta_natural = em_alta and tipo_produto in ('puxador_consumo',
                                                          'puxador_premium')
        if intensidade_idx in (1, 2, 3, 4) and eh_alta_natural:
            reward += K_DESC_EM_ALTA_NATURAL
            breakdown['pen_desc_em_alta'] = K_DESC_EM_ALTA_NATURAL

        # V21 iter16: penalidade COMBO SEM PAR ↑↑ (isotônico/energetico viram loophole)
        if intensidade_idx == 5 and not par_combo:
            reward -= 2500.0
            breakdown['pen_combo_sem_par'] = -2500.0

        # V21 ITER 6: penalidade DEMANDA MUITO BAIXA (whisky/vinho c/ d<1.5)
        # Evita promover categorias com demanda quase zero (lucro nunca cobre custo)
        if intensidade_idx > 0 and d_base_ctx < 1.5:
            reward += K_DEMANDA_MUITO_BAIXA
            breakdown['pen_demanda_muito_baixa'] = K_DEMANDA_MUITO_BAIXA

        # V21 ITER 12: ESQUENTA DE SEXTA — bonus em 4 níveis (cabeça INTENSIDADE foca)
        dow = self.data_atual.weekday()
        eh_sex_sab = dow in (4, 5)
        # Nível 0 (NOVO): cerveja promovida com QUALQUER intensidade > 0 em sex/sáb
        # Esse é o sinal mais forte para a cabeça INTENSIDADE decidir não-nada quando vê cerveja+sex
        if cat == 'cerveja' and eh_sex_sab and intensidade_idx > 0:
            reward += K_PROMOVER_CERVEJA_SEX
            breakdown['bonus_promover_cerveja_sex'] = K_PROMOVER_CERVEJA_SEX
        # Nível 1: cerveja promovida em sex/sáb (independente do par)
        if cat == 'cerveja' and eh_sex_sab and intensidade_idx == 5:
            reward += K_ESQUENTA_SEXTA * 0.5
            breakdown['bonus_cerveja_sex_sab'] = K_ESQUENTA_SEXTA * 0.5
        # Nível 2: bidirecional cerveja↔snack (par perfeito da Esquenta)
        eh_esquenta_a = (cat == 'cerveja' and par_combo == 'snack_salgado'
                         and intensidade_idx == 5 and eh_sex_sab)
        eh_esquenta_b = (cat == 'snack_salgado' and par_combo == 'cerveja'
                         and intensidade_idx == 5 and eh_sex_sab)
        if eh_esquenta_a or eh_esquenta_b:
            reward += K_ESQUENTA_SEXTA  # 800
            breakdown['bonus_esquenta_sexta'] = K_ESQUENTA_SEXTA

        # V21 ITER 7: cerveja em sex/sáb com qualquer combo válido (h>=2.0)
        if (cat == 'cerveja' and eh_sex_sab and intensidade_idx == 5 and par_combo
                and self._harmonia(cat, par_combo) >= 2.0):
            reward += K_CERVEJA_NOITURNA
            breakdown['bonus_cerveja_noturna'] = K_CERVEJA_NOITURNA

        # V21 iter15: COMBO IDEAL POR EVENTO escalado por proximidade
        # Só dispara forte nos últimos dias antes da data (não 14 dias antes)
        if evento and par_combo and intensidade_idx == 5:
            ev_nome = evento['nome'].lower()
            # V21 iter21: match por prefixo "copa" (nome é "Copa 2026 — Abertura")
            if 'copa' in ev_nome:
                ideais = _COPA
            else:
                ideais = COMBOS_IDEAIS_EVENTO.get(ev_nome, [])
            if (cat, par_combo) in ideais:
                janela = evento.get('janela_pre_dias', 7)
                proximidade = max(0.3, 1.0 - (dias_evento or 0) / max(janela, 1))
                bonus = K_COMBO_IDEAL_EVENTO * proximidade
                reward += bonus
                breakdown['bonus_combo_ideal_evento'] = round(bonus, 1)

        # ITER 13: bonus por "não promover quando NÃO há contexto favorável"
        em_baixa = fator_sazonal < 1.0
        vencimento_proximo = validade < 0.35
        evento_relevante = evento is not None and afeta_categoria
        if intensidade_idx == 0 and em_baixa and not vencimento_proximo and not evento_relevante:
            reward += 120.0   # ITER 18: 50→120 (forte incentivo conservadorismo)
            breakdown['bonus_nao_promover_justificado'] = 120.0

        # ITER 17 testada e RESTAURADA (puxou agente para padaria em produtos
        # errados, regredindo cenários). Mantém iter 16 como final.
        # if rotina+rotina: reward += 150 — REMOVIDO.

        # PENALIDADE: combo PDV-inválido (substitui hard rule V19.2)
        eh_combo_invalido = False
        if intensidade_idx == 5 and par_combo:
            if frozenset([cat, par_combo]) in PARES_INVALIDOS_PDV:
                eh_combo_invalido = True
                reward += K_COMBO_PDV_INVALIDO
                breakdown['pen_combo_pdv_invalido'] = K_COMBO_PDV_INVALIDO

        # ITER 4: harmonia DOMINANTE — bonus aditivo (níveis) + multiplicativo
        if intensidade_idx == 5 and par_combo:
            harm = self._harmonia(cat, par_combo)
            # V21 iter3: extra bonus pra h>=2.5 (combo IDEAL)
            if harm >= 2.5:
                reward += K_BONUS_HARMONIA_MUITO_FORTE
                breakdown['bonus_harmonia_muito_forte'] = K_BONUS_HARMONIA_MUITO_FORTE
            # Aditivo (níveis discretos)
            if harm >= 2.0:
                reward += K_BONUS_HARMONIA_FORTE
                breakdown['bonus_harmonia_forte'] = K_BONUS_HARMONIA_FORTE
            elif harm >= 1.5:
                reward += K_BONUS_HARMONIA_MEDIA
                breakdown['bonus_harmonia_media'] = K_BONUS_HARMONIA_MEDIA
            elif harm < 1.0:
                reward += K_PENALIDADE_HARMONIA_FRACA
                breakdown['pen_harmonia_fraca'] = K_PENALIDADE_HARMONIA_FRACA
            elif harm < 1.3:
                reward += K_PENALIDADE_HARMONIA_LEVE
                breakdown['pen_harmonia_leve'] = K_PENALIDADE_HARMONIA_LEVE
            # ITER 11: multiplicativo PROPORCIONAL ao lucro real (não constante)
            # combo bom em produto rentável → grande bonus
            # combo em produto não-rentável → bonus pequeno (agente prefere nada)
            lucro_abs = abs(lucro_bruto)  # escala
            harm_factor = (harm / 1.5 - 1.0)  # -0.6 a +0.67
            harm_mult = harm_factor * min(lucro_abs * 0.5, K_HARMONIA_MULTIPLICATIVO)
            reward += harm_mult
            breakdown['harm_multiplicativo'] = round(harm_mult, 1)

        # V21 iter15: bonus puxador_presente em evento ESCALADO por proximidade
        if (evt_tipo == 'presente' and tipo_produto == 'puxador_presente'
                and afeta_categoria and intensidade_idx > 0):
            janela = evento.get('janela_pre_dias', 7) if evento else 7
            proximidade = max(0.3, 1.0 - (dias_evento or 0) / max(janela, 1))
            bonus = K_BONUS_EVENTO_PRESENTE_PUXADOR * proximidade
            reward += bonus
            breakdown['bonus_evento_presente_puxador'] = round(bonus, 1)

        # PENALIDADE: canibalização alta
        if canib_pct > 0.40:
            pen = K_CANIBALIZACAO_ALTA * (canib_pct - 0.40) * 100
            reward += pen
            breakdown['pen_canib_alta'] = pen

        # PENALIDADE: repetição de categoria (semana)
        if cat in self.hist_categorias:
            reward += K_REPETICAO_CATEGORIA
            breakdown['pen_repeticao'] = K_REPETICAO_CATEGORIA

        # V21 iter18: penalidade CRESCENTE por repetir o MESMO PAR de combo
        # Força diversidade — evita 41x cerveja+isotonico dominando o calendário
        if intensidade_idx == 5 and par_combo:
            par_key = tuple(sorted([cat, par_combo]))
            uso = self.par_uso_count.get(par_key, 0)
            if uso >= 2:
                # 3ª vez: -100, 4ª: -200, 5ª: -300... satura em -800
                pen_rep_par = -min(100 * (uso - 1), 800)
                reward += pen_rep_par
                breakdown['pen_repetir_par'] = pen_rep_par

        # PENALIDADE: uplift abaixo do breakeven (campanha que não compensa custo)
        if intensidade_idx > 0:
            denom = margem * (1 - desc_pct) * dias
            if denom > 0:
                breakeven_un_dia = custo_op / denom
                if uplift_un < breakeven_un_dia:
                    reward += K_UPLIFT_ABAIXO_BREAKEVEN
                    breakdown['pen_uplift_baixo'] = K_UPLIFT_ABAIXO_BREAKEVEN

        # PROIBIDA (cigarro) — penalidade muito forte mas via reward (não bloqueio)
        if tipo_produto == 'proibida' and intensidade_idx > 0:
            reward -= 1000.0
            breakdown['pen_categoria_proibida'] = -1000.0

        return StepInfo(
            categoria=cat,
            intensidade=INTENSIDADE_LABEL[intensidade_idx],
            par_combo=par_combo,
            alvo_desconto=alvo_str,
            data=self.data_atual.isoformat(),
            demanda_base_anual=round(d_base_anual, 2),
            demanda_base_contextual=round(d_base_ctx, 2),
            demanda_promocional=round(d_promo, 2),
            uplift_pct=round(uplift * 100, 1),
            canib_pct=round(canib_pct * 100, 1),
            lucro_uplift_dia=round(lucro_uplift_dia, 2),
            lucro_halo_dia=round(lucro_halo_dia, 2),
            lucro_canib_dia=round(lucro_canib_dia, 2),
            lucro_defensivo_dia=round(lucro_defensivo_dia, 2),
            custo_op=round(custo_op, 2),
            lucro_total=round(lucro_total, 2),
            reward=round(reward, 2),
            reward_breakdown={k: round(v, 2) for k, v in breakdown.items()},
            eh_combo_invalido_pdv=eh_combo_invalido,
            eh_alta_natural=eh_alta_natural,
            evento_proximo=evt_label,
        )

    # ═══════════════════════════════════════════════════════════════════
    # ESTADO (observação)
    # ═══════════════════════════════════════════════════════════════════

    def _calcular_state_dim(self):
        N = self.N_CATEGORIAS
        return (
            7    # dia da semana (1-hot)
            + 12  # mês (1-hot)
            + 1   # dia do mês norm
            + 4   # tipo de evento próximo (1-hot)
            + 1   # dias até próximo evento norm
            + N   # categoria do produto atual (1-hot)
            + 6   # tipo_produto (1-hot)
            + 1   # demanda base anual norm
            + 1   # demanda base contextual norm
            + 1   # fator sazonal atual
            + 1   # margem norm
            + 1   # estoque rel
            + 1   # validade rel
            + 1   # em_alta_flag
            + 1   # em_baixa_flag
            + 1   # temp_norm
            + 1   # uplift_prior do evento próximo
            + 1   # afeta_categoria_flag
            + N   # historico_promos_7d (1 se categoria foi promovida)
            + 1   # turnos restantes norm
        )

    def _observar(self):
        cat_idx = self.produto_atual_idx
        cat = self.categorias[cat_idx]
        cat_info = self.cats[cat]
        tipo_produto = TIPO_PRODUTO.get(cat, 'commodity')

        # Dia da semana e mês
        dow = self.data_atual.weekday()
        mes = self.data_atual.month - 1
        dia_norm = self.data_atual.day / 31.0

        dow_oh = np.zeros(7); dow_oh[dow] = 1.0
        mes_oh = np.zeros(12); mes_oh[mes] = 1.0

        # Evento próximo
        evento, dias_evt, evt_tipo = self._evento_proximo(self.data_atual)
        tipo_evento_oh = np.zeros(4)
        if evt_tipo == 'presente': tipo_evento_oh[0] = 1.0
        elif evt_tipo == 'consumo_intenso': tipo_evento_oh[1] = 1.0
        elif evt_tipo == 'comercial': tipo_evento_oh[2] = 1.0
        else: tipo_evento_oh[3] = 1.0
        dias_evt_norm = min(dias_evt / 30.0, 1.0) if dias_evt else 1.0
        uplift_prior = evento['uplift_prior'] / 3.5 if evento else 0.0
        afeta_cat_flag = float(
            evento is not None and self._categoria_afetada(cat, evento['categorias_afetadas'])
        )

        # Categoria atual
        cat_oh = np.zeros(self.N_CATEGORIAS)
        cat_oh[cat_idx] = 1.0

        # Tipo produto
        tipos = ['puxador_consumo', 'puxador_premium', 'puxador_presente',
                  'impulso', 'rotina', 'commodity']
        tipo_oh = np.zeros(6)
        if tipo_produto in tipos:
            tipo_oh[tipos.index(tipo_produto)] = 1.0

        # Demanda e fatores
        d_base_anual = float(cat_info.get('demanda_base_dia', 5.0))
        fator_sazonal = self._fator_sazonal(cat, self.data_atual, tipo_produto)
        d_base_ctx = d_base_anual * fator_sazonal
        em_alta = float(fator_sazonal > 1.15)
        em_baixa = float(fator_sazonal < 0.85)
        margem = float(cat_info.get('margem', 4.0))
        temp = self._temp_norm(self.data_atual)

        # Histórico
        hist_oh = np.zeros(self.N_CATEGORIAS)
        for h in self.hist_categorias:
            if h in self.cat_idx:
                hist_oh[self.cat_idx[h]] = 1.0

        turnos_restantes = (self.EPISODIO_TURNOS - self.turno) / self.EPISODIO_TURNOS

        obs = np.concatenate([
            dow_oh, mes_oh, [dia_norm],
            tipo_evento_oh, [dias_evt_norm],
            cat_oh, tipo_oh,
            [d_base_anual / 30.0],   # norm
            [d_base_ctx / 50.0],
            [fator_sazonal / 2.5],
            [margem / 50.0],
            [min(self.estoque_rel[cat_idx], 3.0) / 3.0],
            [self.validade_rel[cat_idx]],
            [em_alta], [em_baixa],
            [temp],
            [uplift_prior],
            [afeta_cat_flag],
            hist_oh,
            [turnos_restantes],
        ])
        return obs.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # HELPERS DE TRANSIÇÃO E ESTADO INTERNO
    # ═══════════════════════════════════════════════════════════════════

    def _sortear_produto(self) -> int:
        """Sorteia próximo produto candidato — exclui proibidos.

        V21 iter13: enviesa sorteio para dar EXPOSURE a categorias-chave
        em seu contexto natural (cerveja em sex/sáb, café em manhã).
        Isto é curriculum learning — o agente ainda decide sozinho.
        """
        validos = [i for i, c in enumerate(self.categorias)
                    if TIPO_PRODUTO.get(c, 'commodity') != 'proibida']
        if self.data_atual is not None:
            dow = self.data_atual.weekday()
            # Sex/sáb: 30% das vezes apresenta cerveja (para Esquenta emergir)
            if dow in (4, 5):
                cerv_idx = self.cat_idx.get('cerveja')
                if cerv_idx is not None and self._rng.random() < 0.30:
                    return cerv_idx
            # Seg-qui: leve viés para rotina (café/padaria)
            elif dow < 4:
                cafe_idx = self.cat_idx.get('cafe')
                pad_idx = self.cat_idx.get('padaria')
                r = self._rng.random()
                if r < 0.12 and cafe_idx is not None: return cafe_idx
                if r < 0.20 and pad_idx is not None: return pad_idx
        return int(self._rng.choice(validos))

    def _aplicar_efeito_no_estoque(self, info: StepInfo):
        """Promoção consome estoque, validade decresce no tempo."""
        if info.intensidade != 'nada':
            cat_idx = self.cat_idx.get(info.categoria)
            if cat_idx is not None:
                # Reduz estoque (3 dias de venda promocional)
                consumo = info.demanda_promocional * 3 / 30.0  # frac do mês
                self.estoque_rel[cat_idx] = max(0.1,
                                                  self.estoque_rel[cat_idx] - consumo)
        # Validade decresce em todos (1/30 por turno)
        self.validade_rel = np.clip(self.validade_rel - 0.033, 0.05, 1.0)

    def _atualizar_historico(self, cat: str):
        """Mantém janela de últimas HIST_DIAS categorias promovidas."""
        if cat:
            self.hist_categorias.append(cat)
            if len(self.hist_categorias) > self.HIST_DIAS:
                self.hist_categorias.pop(0)

    # ═══════════════════════════════════════════════════════════════════
    # FATORES (sazonalidade etc) — pega da calibração V19.1
    # ═══════════════════════════════════════════════════════════════════

    def _fator_sazonal(self, cat: str, dt: date, tipo_produto: str) -> float:
        """Fator combinado: dia × mês × clima × pre-feriado."""
        cat_info = self.cats.get(cat, {})
        fator_dow = float(cat_info.get('fator_dia', [1]*7)[dt.weekday()])
        fator_mes = float(cat_info.get('fator_mes', [1]*12)[dt.month - 1])
        # Clima
        temp = self._temp_norm(dt)
        if tipo_produto in ('puxador_consumo', 'commodity'):
            fator_clima = 1.0 + (temp - 0.5) * 0.40
        elif tipo_produto == 'impulso':
            fator_clima = 1.0 + (temp - 0.5) * 0.20
        elif tipo_produto == 'rotina':
            fator_clima = 1.0 - (temp - 0.5) * 0.15
        else:
            fator_clima = 1.0
        # Pré-feriado (boost para puxador_consumo/impulso)
        boost_pre = self._boost_pre_feriado(dt, tipo_produto)
        combinado = fator_dow * fator_mes * fator_clima * boost_pre
        return max(0.4, min(2.5, combinado))

    def _boost_pre_feriado(self, dt: date, tipo_produto: str) -> float:
        for offset in (1, 2, 3):
            check = dt + timedelta(days=offset)
            ev = next((e for e in self.eventos
                        if e['data'] == check.isoformat()
                        and e['tipo_evento'] == 'feriado_oficial'), None)
            if ev:
                if tipo_produto in ('puxador_consumo', 'impulso'):
                    return 1.20
                return 1.08
        return 1.0

    def _evento_proximo(self, dt: date):
        """Retorna (evento, dias_até, tipo).

        V21 iter15: respeita janela_pre_dias do calendário. Se evento tem
        janela_pre=7, agente só "vê" o evento nos 7 dias antes (não 14).
        Isso evita bonus disparando em datas distantes.
        """
        for offset in range(0, 14):
            check = dt + timedelta(days=offset)
            # V21 iter21: inclui evento_esportivo (Copa) além de data_comercial
            ev = next((e for e in self.eventos
                        if e['data'] == check.isoformat()
                        and e['tipo_evento'] in ('data_comercial', 'evento_esportivo')),
                       None)
            if ev:
                # Evento esportivo (Copa) = consumo NO DIA, janela curta (0-2 dias)
                eh_esportivo = ev['tipo_evento'] == 'evento_esportivo'
                janela = 2 if eh_esportivo else ev.get('janela_pre_dias', 7)
                if offset > janela:
                    continue
                tipo = TIPO_EVENTO.get(ev['nome'].lower(), 'comercial')
                for k, v in TIPO_EVENTO.items():
                    if k in ev['nome'].lower():
                        tipo = v
                        break
                if eh_esportivo:
                    tipo = 'consumo_intenso'  # Copa = cerveja+snack imediato
                return ev, offset, tipo
        return None, None, 'nenhum'

    def _categoria_afetada(self, cat: str, lista: list) -> bool:
        """V21 iter15: match MAIS RIGOROSO para evitar bonus em categorias
        tangencialmente relacionadas (ex: 'cerveja' não casa com
        'cerveja_premium' do Dia dos Namorados).
        """
        if not lista or 'todas' in lista:
            return True
        cat_l = cat.lower()
        for c in lista:
            cl = c.lower().strip()
            # Match exato
            if cl == cat_l:
                return True
            # Match de prefixo de chocolate (chocolate_caixa, chocolate_unit)
            if cl == 'chocolate' and cat_l.startswith('chocolate_'):
                return True
            # Match snack (Copa afeta "snack", catálogo tem "snack_salgado")
            if cl in ('salgadinho', 'snack') and cat_l == 'snack_salgado':
                return True
            # NÃO faz match parcial (cerveja vs cerveja_premium)
        return False

    def _harmonia(self, a: str, b: str) -> float:
        ia, ib = self.cat_idx.get(a), self.cat_idx.get(b)
        if ia is None or ib is None or not self.harmonia:
            return 1.0
        # V21: matriz harmonia é só pra promovíveis (20×20)
        # Se a ou b for proibido (cigarro idx 20-22), retorna 0
        N = len(self.harmonia)
        if ia >= N or ib >= N:
            return 0.0
        return float(self.harmonia[ia][ib])

    def mask_complementar_valida(self) -> np.ndarray:
        """V20 ITER 7: máscara dura — pares com harmonia <1.0 são bloqueados.

        Isso não é hard rule no espaço de DECISÃO de promoção; é restrição
        no espaço de COMBINAÇÕES que fazem sentido em PDV. O agente ainda
        pode escolher "nenhum" (idx=0) ou par com h>=1.0.
        """
        cat = self.categorias[self.produto_atual_idx]
        mask = np.zeros(self.N_CATEGORIAS + 1, dtype=bool)
        mask[0] = True   # "nenhum" sempre válido (= não combo)
        if self.harmonia:
            for j, b in enumerate(self.categorias):
                if b == cat:
                    continue
                h = self._harmonia(cat, b)
                # ITER 10: mask em h>=1.4 (V19.2 valida combos com h>=1.4)
                if h >= 1.4 and TIPO_PRODUTO.get(b, 'commodity') != 'proibida':
                    mask[j + 1] = True
        return mask

    def prior_complementar(self) -> np.ndarray:
        """Retorna vetor (N+1,) de probabilidades a priori para a cabeça
        COMPLEMENTAR, baseado na harmonia do produto ATUAL.

        Useado por envieded exploration — agente fica mais propenso a explorar
        pares com harmonia alta em vez de uniforme.
        """
        cat = self.categorias[self.produto_atual_idx]
        ia = self.cat_idx[cat]
        N = self.N_CATEGORIAS
        prior = np.zeros(N + 1, dtype=np.float64)
        prior[0] = 0.3
        if self.harmonia:
            harm_N = len(self.harmonia)
            for j, b in enumerate(self.categorias):
                if b == cat:
                    prior[j + 1] = 0.001
                elif ia >= harm_N or j >= harm_N:
                    # Cigarros etc — fora da matriz harmonia
                    prior[j + 1] = 0.001
                else:
                    h = float(self.harmonia[ia][j])
                    prior[j + 1] = max(0.001, h ** 4)
        return prior

    def _temp_norm(self, dt: date) -> float:
        key = dt.isoformat()
        if key in self.temperatura:
            return self.temperatura[key]
        mes = dt.month
        return 0.75 if mes in (12, 1, 2, 3) else 0.30 if mes in (6, 7, 8) else 0.50

    # ═══════════════════════════════════════════════════════════════════
    # LOADERS
    # ═══════════════════════════════════════════════════════════════════

    def _carregar_calendario(self, path) -> list:
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
                    raw = [v for _, v in vals]
                    t_min, t_max = min(raw), max(raw)
                    rng = max(t_max - t_min, 1.0)
                    temps = {d: (t - t_min) / rng for d, t in vals}
        except FileNotFoundError:
            pass
        return temps


# ═══════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    env = EnvRLPromocoes(seed=42)
    print(f"State dim: {env.state_dim}")
    print(f"Action space: {env.action_space}")
    print(f"N categorias: {env.N_CATEGORIAS}")

    obs, _ = env.reset(seed=42)
    print(f"Obs shape: {obs.shape}, min/max: {obs.min():.2f}/{obs.max():.2f}")

    print("\nRollout aleatório de 10 turnos:")
    total = 0
    for t in range(10):
        action = env.action_space.sample()
        obs, r, done, _, info = env.step(action)
        total += r
        print(f"  T{t}: {info['categoria']:<22s} {info['intensidade']:<8s} "
                f"par={str(info['par_combo'])[:12]:<12s} alvo={info['alvo_desconto']:<12s} "
                f"R$ {info['lucro_total']:>7.2f} | reward {r:>7.1f}")
    print(f"\nTotal reward (random): {total:.1f}")
