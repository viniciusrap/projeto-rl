"""ConvenienceStoreEnvV12 — ambiente final V12 para promoção em conveniência.

Versão CONSOLIDADA (V12.1+V12.2): integra calendário comercial BR, harmonia
categorial, harmonia evento→puxador, forecaster Ridge ML por categoria.

Carrega:
- data/calibracao_v2.json (gerado por calibrar_v2.py)
- results/v12/forecasters.pkl (gerado por treinar_forecaster.py)

Estado: 50 + 5N features (N = nº categorias = 20)
  [0:50]   calendário + clima + contexto + padding
  [50:70]  estoque normalizado por categoria
  [70:90]  validade restante por categoria
  [90:110] fraco_flag (bottom 30% sazonal) por categoria
  [110:130] promo_anterior por categoria
  [130:150] FORECAST receita normalizada por categoria (V12)

Ação: MultiDiscrete([N+1, 5]) = (qual produto, intensidade)
  Produto: 0 = sem-promo, 1..N = índice categoria
  Intensidade: 0 nada, 1 -5%, 2 -10%, 3 combo -5%, 4 liquidação -25%

Episódio: 1095 turnos = 365 dias × 3 turnos = 1 ano calendário real

Recompensa (13 termos):
  + lucro                                           V11
  - alpha × vencimento × custo                       V11
  - beta × ruptura × margem                          V11
  - pen_desconto (tier 5/10/25%)                     V11
  + bonus_giro                                       V11
  + K_TIMING_BONUS / -K_TIMING_PENALTY              V10
  + bonus_evento × harmonia_evento[evento][cat]     V11.7 + V12.2
  + bonus_padrao_dunnhumby                           V11
  - pen_instabilidade                                V11
  - pen_nao_promovivel (cigarros)                    V12.1
  - K_DESC_ALTA_SAUDAVEL                             V11.7
  + K_COMBO_ALTA + K_COMBO_DATA_PICO                 V11.5/V11.7
  + K_DESC_VENCIMENTO + K_DESC_BAIXA                 V11.7
  + bonus_dia_semana_categoria                       V11.6

Combo cooperativo (V12.2):
  Agente escolhe principal, env escolhe par via:
  score_par = fator_contextual × harmonia_combo[principal]
"""
from __future__ import annotations

import json
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

ROOT = Path(__file__).parent
DATA = ROOT / 'data'

# ── Tipos de evento (one-hot 10 buckets) ───────────────────────────────────

TIPOS_EVENTO = ['chocolate', 'vinho', 'espumante', 'cerveja', 'snack',
                'whisky', 'todas', 'refrigerante', 'gelo', 'sorvete']


def _evento_para_idx(categorias_str: str) -> int:
    """Mapeia categorias_afetadas do calendário para índice 0-9."""
    cats = set(categorias_str.split(';'))
    if 'chocolate' in cats: return 0
    if 'vinho' in cats: return 1
    if 'espumante' in cats: return 2
    if 'cerveja' in cats or 'cerveja_premium' in cats: return 3
    if 'snack' in cats: return 4
    if 'whisky' in cats: return 5
    if 'todas' in cats: return 6
    if 'refrigerante' in cats: return 7
    if 'gelo' in cats: return 8
    if 'sorvete' in cats: return 9
    return 6  # default = todas


# ── Classe principal ───────────────────────────────────────────────────────

class ConvenienceStoreEnvV12(gym.Env):
    """Ambiente V12 consolidado para Auto Posto Parque Viana."""

    metadata = {'render_modes': []}

    def __init__(self, calibracao_path: str = 'data/calibracao_v2.json',
                 modo: str = 'treino',
                 forecaster_path: str = 'results/v12/forecasters.pkl'):
        super().__init__()

        with open(ROOT / calibracao_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)

        self.modo = modo
        self.cats = self.cfg['categorias']
        self.N = len(self.cats)
        self.k = self.cfg['constantes']

        # Vetores por categoria
        self.preco = np.array([c['preco_venda'] for c in self.cats], dtype=np.float32)
        self.custo = np.array([c['custo'] for c in self.cats], dtype=np.float32)
        self.margem = self.preco - self.custo
        self.demanda_base = np.array([c['demanda_base_dia'] for c in self.cats],
                                       dtype=np.float32) / 3.0  # por turno
        self.validade_tipica = np.array([c['validade_tipica_turnos'] for c in self.cats],
                                          dtype=np.float32)
        self.elasticidade = np.array([c['elasticidade_promo'] for c in self.cats],
                                       dtype=np.float32)
        self.alpha = np.array([c['alpha_venc'] for c in self.cats], dtype=np.float32)
        self.estoque_inicial = np.array([c['estoque_inicial'] for c in self.cats],
                                          dtype=np.float32)

        self.fator_dia = np.array([c['fator_dia'] for c in self.cats], dtype=np.float32)
        self.fator_turno = np.array([c['fator_turno'] for c in self.cats], dtype=np.float32)
        self.fator_mes = np.array([c['fator_mes'] for c in self.cats], dtype=np.float32)
        self.clima_slope = np.array([c['clima_slope'] for c in self.cats], dtype=np.float32)
        self.clima_intercept = np.array([c['clima_intercept'] for c in self.cats],
                                          dtype=np.float32)

        # Categorias NÃO-promovíveis (água + cigarros por Lei 9.294/96)
        self.promovivel = np.array([c.get('promovivel', True) for c in self.cats],
                                     dtype=bool)
        self.bonus_promo_dia = np.array([
            c.get('bonus_promo_dia_semana', [0]*7) for c in self.cats
        ], dtype=np.float32)
        n_nao_prom = (~self.promovivel).sum()
        if n_nao_prom > 0:
            nomes_np = [c['categoria'] for c in self.cats if not c.get('promovivel', True)]
            print(f"  {n_nao_prom} categorias NÃO-promovíveis: {nomes_np}")

        # Mapping nome → índice
        self.nome_para_idx = {c['categoria']: i for i, c in enumerate(self.cats)}

        # Par estático (fallback) — par dinâmico calculado em runtime
        self.par_idx = np.zeros(self.N, dtype=np.int32)
        for i, c in enumerate(self.cats):
            par_nome = c.get('par_combo', 'snack')
            self.par_idx[i] = self.nome_para_idx.get(par_nome, i)

        # V12.2: Matriz de harmonia categorial (N×N)
        harm = self.cfg.get('harmonia_combo')
        if harm:
            self.harmonia_combo = np.array(harm, dtype=np.float32)
            for i, c in enumerate(self.cats):
                if not c.get('promovivel', True):
                    self.harmonia_combo[:, i] = 0.0
        else:
            self.harmonia_combo = np.ones((self.N, self.N), dtype=np.float32)
            np.fill_diagonal(self.harmonia_combo, 0.0)

        # V12.2: Harmonia evento → categoria-puxadora
        self.harmonia_evento_cat = self.cfg.get('harmonia_evento_categoria', {})

        # Prior Dunnhumby por categoria (índice freq mensal)
        self.prior_freq_mes = np.zeros((self.N, 12), dtype=np.float32)
        for i, c in enumerate(self.cats):
            for m in range(1, 13):
                self.prior_freq_mes[i, m - 1] = c['prior_dunnhumby_indice_freq_mes'].get(
                    str(m), c['prior_dunnhumby_indice_freq_mes'].get(m, 1.0))

        # Pré-computa fator combinado para fraco/forte_flag
        self._fator_combinado = np.zeros((self.N, 7, 3, 12), dtype=np.float32)
        for d in range(7):
            for t in range(3):
                for m in range(12):
                    self._fator_combinado[:, d, t, m] = (
                        self.fator_dia[:, d] * self.fator_turno[:, t]
                        * self.fator_mes[:, m]
                    )
        self._limiar_fraco = np.quantile(
            self._fator_combinado.reshape(self.N, -1),
            self.k['PCT_FRACO'], axis=1
        )
        self._limiar_forte = np.quantile(
            self._fator_combinado.reshape(self.N, -1),
            self.k.get('PCT_FORTE', 0.70), axis=1
        )

        # Temperatura histórica
        df_temp = pd.read_csv(DATA / 'temperatura_historica.csv',
                                parse_dates=['data'])
        tmin = self.cfg['clima_params']['temp_min']
        tmax = self.cfg['clima_params']['temp_max']
        df_temp['temp_norm'] = (df_temp['temp_max'] - tmin) / (tmax - tmin)
        self.temp_lookup = dict(zip(df_temp['data'].dt.date, df_temp['temp_norm']))

        # Calendário comercial
        self._eventos_por_data = {}
        for ev in self.cfg['calendario_comercial']:
            data_ev = date.fromisoformat(ev['data'])
            pre = int(ev['janela_pre_dias'])
            tipo_pico = ev.get('tipo_pico', 'no_dia')
            for offset in range(-pre, int(ev['janela_pos_dias']) + 1):
                d = data_ev + timedelta(days=offset)
                if d not in self._eventos_por_data:
                    self._eventos_por_data[d] = []
                prox = 1.0 - abs(offset) / max(pre + 1, 1)
                prox = max(0.3, prox)
                uplift_dia = 1 + (float(ev['uplift_prior']) - 1) * prox
                em_pico = False
                if tipo_pico == 'pre' and offset < 0:
                    em_pico = True
                elif tipo_pico == 'no_dia' and offset == 0:
                    em_pico = True
                elif tipo_pico == 'ambos':
                    em_pico = True
                self._eventos_por_data[d].append({
                    'evento': ev['nome_evento'],
                    'categorias': set(ev['categorias_afetadas'].split(';')),
                    'tipo_idx': _evento_para_idx(ev['categorias_afetadas']),
                    'uplift_dia': uplift_dia,
                    'offset': offset,
                    'tipo_pico': tipo_pico,
                    'em_pico': em_pico,
                })

        # ── V12 FORECASTER ML (Ridge por categoria) ──────────────────────
        with open(ROOT / forecaster_path, 'rb') as f:
            fc_data = pickle.load(f)
        self.forecasters = fc_data['forecasters']
        self.fc_features = fc_data['features']

        # Receita média histórica por categoria (normalização do forecast)
        self.receita_media_cat = np.zeros(self.N, dtype=np.float32)
        for i, c in enumerate(self.cats):
            fc = self.forecasters.get(c['categoria'])
            if fc:
                self.receita_media_cat[i] = float(fc['media_receita_train'])
            else:
                self.receita_media_cat[i] = float(c['demanda_base_dia'] * c['preco_venda'])

        # Cache de forecast (1× por dia)
        self._fc_cache_date = None
        self._fc_cache_values = None
        # Buffer rolling de 28 dias de receita por categoria
        self.demanda_buffer = None
        self.receita_dia_atual = None

        # Períodos
        per = self.cfg['periodos']
        self.data_inicio = date.fromisoformat(
            per['data_inicio_treino' if modo == 'treino' else 'data_inicio_validacao']
        )
        self.data_fim = date.fromisoformat(
            per['data_fim_treino' if modo == 'treino' else 'data_fim_validacao']
        )

        # Spaces
        n_obs = 50 + 5 * self.N  # V12: +N features de forecast
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.N + 1, 5])

        # Estado interno
        self.estoque = None
        self.idade = None
        self.promo_ant = None
        self.acao_ant = (0, 0)
        self.data_atual = None
        self.turno = 0
        self.passo = 0

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        # Inicializa buffer ANTES do _get_obs (que precisa do buffer)
        self.demanda_buffer = np.tile(
            self.receita_media_cat, (28, 1)
        ).astype(np.float32)
        self.receita_dia_atual = np.zeros(self.N, dtype=np.float32)
        self._fc_cache_date = None
        self._fc_cache_values = None

        dias_disp = (self.data_fim - self.data_inicio).days - 365
        if dias_disp <= 0:
            self.data_atual = self.data_inicio
        else:
            offset = int(self.np_random.integers(0, dias_disp))
            self.data_atual = self.data_inicio + timedelta(days=offset)

        self.estoque = self.estoque_inicial.copy()
        self.idade = np.zeros(self.N, dtype=np.float32)
        self.promo_ant = np.zeros(self.N, dtype=np.float32)
        self.acao_ant = (0, 0)
        self.turno = 0
        self.passo = 0
        return self._get_obs(), self._get_info()

    # ── step ───────────────────────────────────────────────────────────

    def step(self, action):
        prod_idx, intensidade = int(action[0]), int(action[1])
        turno_pre = self.turno

        # Anular promoção em categoria não-promovível
        tentou_promover_nao_promovivel = False
        if prod_idx > 0 and intensidade > 0:
            if not self.promovivel[prod_idx - 1]:
                tentou_promover_nao_promovivel = True
                prod_idx = 0
                intensidade = 0

        # 1. Aplicar promoção
        fator_promo = np.ones(self.N, dtype=np.float32)
        preco_efetivo = self.preco.copy()
        desconto_aplicado = 0.0

        if prod_idx > 0 and intensidade > 0:
            p = prod_idx - 1
            elast = abs(self.elasticidade[p])
            if intensidade == 1:
                fator_promo[p] = 1 + elast * 0.05
                preco_efetivo[p] *= 0.95
                desconto_aplicado = 0.05
            elif intensidade == 2:
                fator_promo[p] = 1 + elast * 0.10
                preco_efetivo[p] *= 0.90
                desconto_aplicado = 0.10
            elif intensidade == 3:  # combo cooperativo + harmonia
                principal_din = p
                dia_sem_tmp = self.data_atual.weekday()
                mes_tmp = self.data_atual.month - 1
                fator_ctx = (self.fator_dia[:, dia_sem_tmp]
                              * self.fator_turno[:, self.turno]
                              * self.fator_mes[:, mes_tmp])
                score_par = fator_ctx * self.harmonia_combo[principal_din]
                score_masked = np.where(self.promovivel, score_par, -np.inf)
                score_masked[principal_din] = -np.inf
                par_din = int(np.argmax(score_masked))
                fator_promo[principal_din] = self.k.get('BOOST_COMBO_PRINCIPAL', 1.15)
                fator_promo[par_din] = self.k.get('BOOST_COMBO_PAR', 1.10)
                desc_combo = self.k.get('DESC_COMBO_MAX', 0.05)
                preco_efetivo[principal_din] *= (1 - desc_combo)
                desconto_aplicado = desc_combo
                produtos_promovidos_combo = (principal_din, par_din)
            elif intensidade == 4:
                fator_promo[p] = 1 + elast * 0.25
                preco_efetivo[p] *= 0.75
                desconto_aplicado = 0.25

        # 2. Demanda contextual
        dia_sem = self.data_atual.weekday()
        mes = self.data_atual.month - 1
        f_dia = self.fator_dia[:, dia_sem]
        f_turno = self.fator_turno[:, self.turno]
        f_mes = self.fator_mes[:, mes]
        temp_norm = self._temperatura_norm()
        f_clima = np.clip(self.clima_slope * temp_norm + self.clima_intercept,
                           0.5, 2.0)
        f_evento = self._fator_evento_dia(self.data_atual)
        demanda_esperada = (self.demanda_base * f_dia * f_turno * f_mes
                             * f_clima * f_evento * fator_promo)
        demanda_real = self.np_random.poisson(np.maximum(demanda_esperada, 0.01))
        demanda_real = demanda_real.astype(np.float32)

        vendas = np.minimum(demanda_real, self.estoque)
        rupturas = np.maximum(demanda_real - vendas, 0)

        # 3. Lucro
        lucro = float(np.sum(vendas * (preco_efetivo - self.custo)))

        # 4. Vencimentos
        idade_pre = self.idade.copy()
        self.idade += 1
        venceu = self.idade > self.validade_tipica
        perdas = np.where(venceu, self.estoque, 0).astype(np.float32)
        pen_venc = float(np.sum(self.alpha * perdas * self.custo))
        self.estoque = np.where(venceu, 0, self.estoque)
        self.idade = np.where(venceu, 0, self.idade)

        # 5. Vendas → estoque
        self.estoque = np.maximum(self.estoque - vendas, 0)

        # 6. Reposição implícita
        cobertura_alvo = self.demanda_base * self.k['TURNOS_POR_DIA'] * self.k['COBERTURA_ALVO_DIAS']
        precisa_repor = self.estoque < cobertura_alvo * 0.3
        if np.any(precisa_repor):
            qtd_repor = np.where(precisa_repor,
                                   cobertura_alvo - self.estoque,
                                   0).astype(np.float32)
            soma = self.estoque + qtd_repor
            self.idade = np.where(
                soma > 0,
                self.estoque * self.idade / np.maximum(soma, 1e-6),
                self.idade
            )
            self.estoque = soma

        # 7. Penalidade ruptura
        pen_ruptura = self.k['BETA_RUPTURA'] * float(np.sum(rupturas * self.margem * 0.5))

        # 8. Penalidade desconto em saudável (tiered)
        pen_desconto = 0.0
        if desconto_aplicado > 0 and prod_idx > 0:
            p = prod_idx - 1
            risco = idade_pre[p] / max(self.validade_tipica[p], 1)
            if risco < 0.4:
                if intensidade == 1:
                    pen_desconto = self.k['GAMMA_DESC_5']
                elif intensidade == 2:
                    pen_desconto = self.k['GAMMA_DESC_10']
                elif intensidade == 4:
                    pen_desconto = self.k['GAMMA_DESC_25']
                else:
                    pen_desconto = self.k['GAMMA_DESC_10']

        # 9. Bonus giro
        validade_critica = (idade_pre / self.validade_tipica > 0.7).astype(np.float32)
        bonus_giro = self.k['DELTA_GIRO'] * float(np.sum(vendas * validade_critica
                                                            * self.margem * 0.3))

        # 10. Bonus timing
        bonus_timing = 0.0
        if prod_idx > 0 and intensidade > 0:
            p = prod_idx - 1
            fator_p = f_dia[p] * f_turno[p] * f_mes[p]
            if fator_p < self._limiar_fraco[p]:
                bonus_timing = self.k['K_TIMING_BONUS']
            else:
                bonus_timing = -self.k['K_TIMING_PENALTY']

        # 11. Bonus evento comercial (V11 + V12.1 + V12.2)
        bonus_evento = 0.0
        evs = self._eventos_por_data.get(self.data_atual, [])
        evs_relevantes = [ev for ev in evs if 'todas' not in ev['categorias']]
        if evs_relevantes:
            ev_main = max(evs_relevantes, key=lambda e: e['uplift_dia'])
            eh_presente = ev_main.get('tipo_pico', '') == 'pre'
            k_bonus = (self.k.get('K_EVENTO_PRESENTE', self.k['K_EVENTO'])
                       if eh_presente else self.k['K_EVENTO'])
            k_perdido = (self.k.get('K_EVENTO_PERDIDO_PRESENTE',
                                     self.k.get('K_EVENTO_PERDIDO', 0))
                         if eh_presente else self.k.get('K_EVENTO_PERDIDO', 0))
            if prod_idx > 0 and intensidade > 0:
                p = prod_idx - 1
                cat_p = self.cats[p]['categoria']
                if self._categoria_bate_com_evento(cat_p, ev_main['categorias']):
                    harm_score = self._harmonia_evento_score(
                        ev_main['evento'], cat_p
                    )
                    bonus_evento = k_bonus * (ev_main['uplift_dia'] - 1) * harm_score
                else:
                    bonus_evento = -k_perdido * 0.3 * (ev_main['uplift_dia'] - 1)
            else:
                bonus_evento = -k_perdido * (ev_main['uplift_dia'] - 1)

        # 12. Bonus padrão Dunnhumby
        bonus_padrao = 0.0
        if prod_idx > 0 and intensidade > 0:
            p = prod_idx - 1
            idx_freq = self.prior_freq_mes[p, mes]
            if idx_freq > 1.0:
                bonus_padrao = self.k['THETA_PADRAO'] * (idx_freq - 1.0)
            else:
                bonus_padrao = -self.k['THETA_PADRAO'] * (1.0 - idx_freq) * 0.5

        # 13. Penalidade instabilidade
        pen_instabilidade = 0.0
        if (prod_idx > 0 and self.acao_ant[0] > 0
                and (prod_idx != self.acao_ant[0]
                      or intensidade != self.acao_ant[1])):
            pen_instabilidade = self.k['LAMBDA_INSTABILIDADE']

        pen_nao_promovivel = 30.0 if tentou_promover_nao_promovivel else 0.0

        # ── Regras de política (Vinicius) ─────────────────────────
        pen_desc_alta_saudavel = 0.0
        bonus_combo_alta = 0.0
        bonus_combo_data_pico = 0.0
        bonus_desc_vencimento = 0.0
        bonus_desc_baixa = 0.0

        if prod_idx > 0 and intensidade > 0:
            p = prod_idx - 1
            fator_p = f_dia[p] * f_turno[p] * f_mes[p]
            em_alta_demanda = fator_p >= self._limiar_forte[p]
            em_baixa_demanda = fator_p < self._limiar_fraco[p]
            risco_venc = idade_pre[p] / max(self.validade_tipica[p], 1)
            perto_de_vencer = risco_venc >= self.k.get('LIMIAR_VENCIMENTO', 0.70)
            produto_saudavel = risco_venc < self.k.get('LIMIAR_SAUDAVEL', 0.30)
            eh_desconto_direto = intensidade in (1, 2, 4)
            eh_combo = intensidade == 3

            if eh_desconto_direto and em_alta_demanda and produto_saudavel:
                pen_desc_alta_saudavel = self.k.get('K_DESC_ALTA_SAUDAVEL', 200.0)
            if eh_combo and em_alta_demanda:
                bonus_combo_alta = self.k.get('K_COMBO_ALTA', 150.0)
            if eh_combo:
                evs_pico = self._eventos_por_data.get(self.data_atual, [])
                em_pico_relevante = [e for e in evs_pico if e.get('em_pico', False)]
                if em_pico_relevante:
                    ev_max = max(em_pico_relevante, key=lambda e: e['uplift_dia'])
                    bonus_combo_data_pico = (self.k.get('K_COMBO_DATA_PICO', 250.0)
                                                * (ev_max['uplift_dia'] - 1))
            if eh_desconto_direto and perto_de_vencer:
                escala_venc = (risco_venc - self.k.get('LIMIAR_VENCIMENTO', 0.70)) \
                                / (1 - self.k.get('LIMIAR_VENCIMENTO', 0.70))
                bonus_desc_vencimento = self.k.get('K_DESC_VENCIMENTO', 120.0) * escala_venc
            if eh_desconto_direto and em_baixa_demanda:
                bonus_desc_baixa = self.k.get('K_DESC_BAIXA', 100.0)

        # Bonus dia-semana por categoria
        bonus_dia_semana_categoria = 0.0
        if prod_idx > 0 and intensidade > 0:
            p = prod_idx - 1
            bonus_dia_semana_categoria = float(self.bonus_promo_dia[p, dia_sem])

        reward = (lucro - pen_venc - pen_ruptura - pen_desconto + bonus_giro
                   + bonus_timing + bonus_evento + bonus_padrao
                   - pen_instabilidade - pen_nao_promovivel
                   - pen_desc_alta_saudavel
                   + bonus_combo_alta + bonus_combo_data_pico
                   + bonus_desc_vencimento + bonus_desc_baixa
                   + bonus_dia_semana_categoria)

        # 14. Atualizar estado
        self.promo_ant = np.zeros(self.N, dtype=np.float32)
        if intensidade == 3 and prod_idx > 0:
            if 'produtos_promovidos_combo' in locals():
                p1, p2 = produtos_promovidos_combo
                self.promo_ant[p1] = 1.0
                self.promo_ant[p2] = 1.0
        elif prod_idx > 0 and intensidade > 0:
            self.promo_ant[prod_idx - 1] = 1.0
        self.acao_ant = (prod_idx, intensidade)
        self.turno += 1
        if self.turno >= 3:
            self.turno = 0
            self.data_atual = self.data_atual + timedelta(days=1)
        self.passo += 1

        # V12: Atualizar buffer de receita diária (para forecaster lags)
        self.receita_dia_atual += vendas.astype(np.float32) * self.preco
        if self.turno == 0 and turno_pre == 2:
            self.demanda_buffer = np.roll(self.demanda_buffer, -1, axis=0)
            self.demanda_buffer[-1] = self.receita_dia_atual.copy()
            self.receita_dia_atual = np.zeros(self.N, dtype=np.float32)
            self._fc_cache_date = None

        terminated = self.passo >= 3 * 365
        truncated = False

        info = self._get_info()
        combo_principal = None
        combo_par = None
        if intensidade == 3 and 'produtos_promovidos_combo' in locals():
            combo_principal = self.cats[produtos_promovidos_combo[0]]['categoria']
            combo_par = self.cats[produtos_promovidos_combo[1]]['categoria']

        info.update({
            'lucro': lucro,
            'pen_venc': pen_venc,
            'combo_principal': combo_principal,
            'combo_par': combo_par,
            'pen_ruptura': pen_ruptura,
            'pen_desconto': pen_desconto,
            'bonus_giro': bonus_giro,
            'bonus_timing': bonus_timing,
            'bonus_evento': bonus_evento,
            'bonus_padrao': bonus_padrao,
            'pen_instabilidade': pen_instabilidade,
            'pen_nao_promovivel': pen_nao_promovivel,
            'pen_desc_alta_saudavel': pen_desc_alta_saudavel,
            'bonus_combo_alta': bonus_combo_alta,
            'bonus_combo_data_pico': bonus_combo_data_pico,
            'bonus_dia_semana_categoria': bonus_dia_semana_categoria,
            'bonus_desc_vencimento': bonus_desc_vencimento,
            'bonus_desc_baixa': bonus_desc_baixa,
            'vendas': vendas.copy(),
            'rupturas': rupturas.copy(),
            'perdas': perdas.copy(),
            'reward_total': reward,
        })

        return self._get_obs(), reward, terminated, truncated, info

    # ── Observação ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        dia_sem = self.data_atual.weekday()
        mes = self.data_atual.month - 1
        dia_oh = np.zeros(7, dtype=np.float32); dia_oh[dia_sem] = 1.0
        turno_oh = np.zeros(3, dtype=np.float32); turno_oh[self.turno] = 1.0
        mes_oh = np.zeros(12, dtype=np.float32); mes_oh[mes] = 1.0
        dia_mes_norm = (self.data_atual.day - 1) / 30.0

        # Evento próximo
        evento_oh = np.zeros(10, dtype=np.float32)
        dias_evento_norm = 1.0
        for delta in range(31):
            d_check = self.data_atual + timedelta(days=delta)
            if d_check in self._eventos_por_data:
                ev = self._eventos_por_data[d_check][0]
                evento_oh[ev['tipo_idx']] = 1.0
                dias_evento_norm = min(delta / 30, 1.0)
                break

        temp_norm = self._temperatura_norm()
        delta_temp = 0.0  # TODO: usar histórico

        passo_norm = self.passo / (3 * 365)
        ibge_fator = self.cfg['ibge_fator_mes'].get(str(mes + 1),
                                                       self.cfg['ibge_fator_mes'].get(mes + 1, 1.0))
        ibge_fator_norm = (ibge_fator - 0.85) / 0.35

        prior_pct_medio = float(np.mean([self.cats[i]['prior_dunnhumby_pct_promo']
                                            for i in range(self.N)]))
        prior_freq_mes = float(np.mean(self.prior_freq_mes[:, mes]))
        em_alta_promo = 1.0 if prior_freq_mes > 1.1 else 0.0

        estoque_norm = np.clip(self.estoque / (self.estoque_inicial * 1.5 + 1e-6),
                                0, 1).astype(np.float32)
        validade_rest = np.clip(1 - self.idade / (self.validade_tipica + 1e-6),
                                 0, 1).astype(np.float32)
        fraco = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            fator_i = (self.fator_dia[i, dia_sem]
                        * self.fator_turno[i, self.turno]
                        * self.fator_mes[i, mes])
            if fator_i < self._limiar_fraco[i]:
                fraco[i] = 1.0

        padding = np.zeros(9, dtype=np.float32)

        # V12: Features de forecast (1 por categoria)
        fc_norm = self._compute_forecast_norm()

        obs = np.concatenate([
            dia_oh, turno_oh, mes_oh, [dia_mes_norm],
            evento_oh, [dias_evento_norm],
            [temp_norm, delta_temp],
            [passo_norm, np.clip(ibge_fator_norm, 0, 1)],
            [prior_pct_medio, prior_freq_mes / 2, em_alta_promo],
            padding,
            estoque_norm, validade_rest, fraco, self.promo_ant,
            fc_norm,
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        return {
            'data': self.data_atual.isoformat() if self.data_atual else None,
            'turno': self.turno,
            'passo': self.passo,
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _compute_forecast_norm(self) -> np.ndarray:
        """V12: forecast Ridge por categoria, cache por dia."""
        if self._fc_cache_date == self.data_atual and self._fc_cache_values is not None:
            return self._fc_cache_values

        d = self.data_atual
        dia_sem = d.weekday()
        mes = d.month - 1
        dia_mes = d.day
        temp_norm = self._temperatura_norm()

        is_event = 1.0 if d in self._eventos_por_data else 0.0
        days_to_event = 30.0
        for delta in range(31):
            if (d + timedelta(days=delta)) in self._eventos_por_data:
                days_to_event = float(delta)
                break
        tipo_pico_pre = 0.0
        tipo_pico_no_dia = 0.0
        if d in self._eventos_por_data:
            for ev in self._eventos_por_data[d]:
                tp = ev.get('tipo_pico', '')
                if tp in ('pre', 'ambos'):
                    tipo_pico_pre = 1.0
                if tp in ('no_dia', 'ambos'):
                    tipo_pico_no_dia = 1.0

        forecast = np.zeros(self.N, dtype=np.float32)
        for i, c in enumerate(self.cats):
            cat_nome = c['categoria']
            fc = self.forecasters.get(cat_nome)
            if fc is None:
                expected = (self.demanda_base[i] * 3 * self.preco[i]
                             * self.fator_dia[i, dia_sem]
                             * self.fator_mes[i, mes])
                forecast[i] = float(np.clip(
                    expected / max(self.receita_media_cat[i] * 3, 1e-6),
                    0, 1))
                continue
            lag1 = float(self.demanda_buffer[-1, i])
            lag7 = float(self.demanda_buffer[-7, i])
            lag28 = float(self.demanda_buffer[-28, i])
            X = np.array([[
                dia_sem, mes, dia_mes, temp_norm,
                lag1, lag7, lag28,
                is_event, days_to_event,
                tipo_pico_pre, tipo_pico_no_dia,
            ]], dtype=np.float32)
            scaler = fc.get('scaler')
            if scaler is not None:
                X = scaler.transform(X)
            pred = float(fc['model'].predict(X)[0])
            pred = max(0.0, pred)
            forecast[i] = float(np.clip(
                pred / max(self.receita_media_cat[i] * 3, 1e-6),
                0, 1))

        self._fc_cache_date = self.data_atual
        self._fc_cache_values = forecast
        return forecast

    def _temperatura_norm(self) -> float:
        return float(self.temp_lookup.get(self.data_atual,
                       0.5 + 0.4 * np.sin(
                           (self.data_atual.timetuple().tm_yday - 80) / 365 * 2 * np.pi
                       )))

    def _fator_evento_dia(self, d: date) -> np.ndarray:
        if d not in self._eventos_por_data:
            return np.ones(self.N, dtype=np.float32)
        fator = np.ones(self.N, dtype=np.float32)
        for ev in self._eventos_por_data[d]:
            for i, cat in enumerate(self.cats):
                cat_nome = cat['categoria']
                if self._categoria_bate_com_evento(cat_nome, ev['categorias']):
                    fator[i] = max(fator[i], ev['uplift_dia'])
        return fator

    def _harmonia_evento_score(self, nome_evento: str, cat: str) -> float:
        """V12.2: score harmonia evento→categoria via substring match."""
        if not self.harmonia_evento_cat:
            return 1.0
        nome_norm = nome_evento.lower()
        for orig, repl in [('á', 'a'), ('â', 'a'), ('ã', 'a'),
                            ('é', 'e'), ('ê', 'e'),
                            ('í', 'i'), ('ó', 'o'), ('ô', 'o'),
                            ('ú', 'u'), ('ç', 'c'),
                            ('Á', 'a'), ('É', 'e'), ('Í', 'i'),
                            ('Ó', 'o'), ('Ú', 'u'), ('Ç', 'c')]:
            nome_norm = nome_norm.replace(orig, repl)
        for key, cat_dict in self.harmonia_evento_cat.items():
            if key in nome_norm:
                return float(cat_dict.get(cat, 1.0))
        return 1.0

    @staticmethod
    def _categoria_bate_com_evento(cat_nome: str, cats_evento: set) -> bool:
        """Matching flexível: chocolate_premium bate com 'chocolate', etc."""
        if cat_nome in cats_evento or 'todas' in cats_evento:
            return True
        base = cat_nome.split('_')[0]
        if base in cats_evento:
            return True
        if cat_nome == 'cerveja' and 'cerveja_premium' in cats_evento:
            return True
        if cat_nome == 'vinho' and any(c.startswith('vinho') for c in cats_evento):
            return True
        if cat_nome == 'destilados' and any(c in cats_evento
                                              for c in ['whisky', 'cachaca', 'vodka']):
            return True
        return False


# ── Factory ────────────────────────────────────────────────────────────────

def construir_env_v12(modo: str = 'treino') -> ConvenienceStoreEnvV12:
    """Constrói env V12 consolidado a partir de data/calibracao_v2.json."""
    return ConvenienceStoreEnvV12(
        'data/calibracao_v2.json',
        modo=modo,
        forecaster_path='results/v12/forecasters.pkl',
    )


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("Construindo env V12 consolidado...")
    env = construir_env_v12(modo='treino')
    print(f"  N categorias: {env.N}")
    print(f"  obs space: {env.observation_space}")
    print(f"  action space: {env.action_space}")
    print(f"  N forecasters: {len(env.forecasters)}")
    print(f"  harmonia_combo: {env.harmonia_combo.shape}")
    print(f"  harmonia_evento_cat: {len(env.harmonia_evento_cat)} eventos")

    print("\nSmoke test (30 turnos)...")
    obs, info = env.reset(seed=42)
    print(f"  obs shape: {obs.shape}")
    rewards = []
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if step in (0, 5, 29):
            print(f"  step {step:>2d}  ação=({action[0]:>2d},{action[1]})  "
                  f"reward={reward:>7.2f}  lucro={info['lucro']:>6.2f}")
        if terminated or truncated:
            break

    print(f"\n✓ Smoke test OK. Reward total: R$ {sum(rewards):,.2f}")
