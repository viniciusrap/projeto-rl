"""ConvenienceStoreEnvV2 — esqueleto do ambiente expandido (V11).

ATENÇÃO: esqueleto para discussão. Não treina ainda. Espera:
1. Catálogo completo do posto (data/catalogo_completo.xlsx) — 20-30 SKUs
2. Calibração por SKU (data/calibracao_v2.json) — preço, custo, margem,
   demanda_base, fator_dia, fator_turno, fator_mes, validade, alpha
3. Pares de combo validados (results/combos_validados.csv) — saída de
   analise_cesta.py com cupom fiscal real

Diferenças principais vs V10:
- Estado: 47 → ~120-150 features (1 grupo por produto)
- Episódio: 90 turnos → 1095 turnos (1 ano calendário real)
- Datas comerciais: lidas de data/calendario_comercial.csv
- Ação: 5 discretas → multi-discreta (produto × intensidade) ou
  Box contínuo (vetor de descontos)
- Recompensa: mantém estrutura V10 + termo de estabilidade da promoção

Uso:
    env = ConvenienceStoreEnvV2(
        catalogo='data/catalogo_completo.xlsx',
        calibracao='data/calibracao_v2.json',
        calendario='data/calendario_comercial.csv',
        combos='results/combos_validados.csv',
    )
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# ── Configuração do ambiente ────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """Configuração que vem dos dados calibrados do posto."""

    n_produtos: int                                # tamanho do catálogo
    skus: list[str]                                # ['cerveja_brahma', ...]
    categorias: list[str]                          # ['bebida', 'snack', ...]

    # Por produto (vetores de tamanho n_produtos)
    preco_venda: np.ndarray
    custo: np.ndarray
    margem: np.ndarray                             # preco - custo
    demanda_base: np.ndarray                       # un/dia
    validade_tipica: np.ndarray                    # turnos
    elasticidade_promo: np.ndarray                 # neg, ~-2 a -4
    alpha: np.ndarray                              # peso da penalidade venc

    # Sazonalidade por produto (n_produtos × 7), (× 3), (× 12)
    fator_dia: np.ndarray                          # shape (n, 7)
    fator_turno: np.ndarray                        # shape (n, 3)
    fator_mes: np.ndarray                          # shape (n, 12)

    # Sensibilidade climática (n_produtos × 2: slope, intercept)
    clima_coef: np.ndarray                         # shape (n, 2)

    # Pares de combo validados (sku → sku complementar)
    pares_combo: dict[str, str]                    # ex: {'cerveja_brahma': 'doritos'}

    # Estoque inicial e capacidade
    estoque_inicial: np.ndarray                    # un
    capacidade_max: np.ndarray                     # un

    # Calendário comercial expandido (1 linha por dia × evento × prior)
    calendario: pd.DataFrame                       # cols: data, categorias, uplift

    # Configuração temporal
    data_inicio_treino: date = date(2020, 6, 22)
    data_fim_treino: date = date(2024, 6, 30)
    data_inicio_val: date = date(2024, 7, 1)
    data_fim_val: date = date(2026, 4, 30)

    # Reward shaping
    k_timing_bonus: float = 250.0
    k_timing_penalty: float = 250.0
    pct_fraco: float = 0.30                         # bottom 30% = "período fraco"
    estabilidade_bonus: float = 5.0                 # bonus por manter mesma ação ≥2 dias


# ── Ambiente ────────────────────────────────────────────────────────────────

class ConvenienceStoreEnvV2(gym.Env):
    """Ambiente expandido — N produtos + calendário comercial real.

    OBSERVATION SPACE (~120-150 features dependendo de n_produtos):
        [0:7]      dia da semana one-hot
        [7:10]     turno one-hot
        [10:22]    mês one-hot
        [22]       dia do mês normalizado (0-1)
        [23]       temperatura normalizada
        [24]       variação de temperatura últimos 7 dias
        [25:25+N_EVENTOS]   one-hot do evento comercial próximo (até 10 tipos)
        [...:]              dias até evento mais próximo / 30
        Por produto P (4 features cada):
            estoque_norm
            validade_rest
            fraco_flag       (1 se fator combinado nesse contexto < pct_fraco)
            promo_ant        (binário: 1 se P foi promovido turno anterior)

    ACTION SPACE — Opção A (multi-discrete, recomendada):
        spaces.MultiDiscrete([n_produtos + 1, 5])
        - dim 0: qual produto promover (0 = nenhum, 1..N = produto)
        - dim 1: intensidade (0 = nada, 1 = -5%, 2 = -10%, 3 = combo, 4 = -25%)

    REWARD: igual V10 — lucro - pen_venc - pen_ruptura - pen_desconto
            + bonus_giro + bonus_timing - bonus_estabilidade
    """

    metadata = {'render_modes': []}

    def __init__(self, cfg: EnvConfig, modo: str = 'treino'):
        super().__init__()
        self.cfg = cfg
        self.modo = modo  # 'treino' ou 'validacao'

        # ── Spaces ──────────────────────────────────────────────────
        n = cfg.n_produtos
        # 7+3+12+1+1+1 (calendário) + 10 (eventos one-hot) + 1 (dias até evento)
        # + 4*n (por produto)
        n_obs = 7 + 3 + 12 + 1 + 1 + 1 + 10 + 1 + 4 * n
        self.observation_space = spaces.Box(0., 1., shape=(n_obs,),
                                             dtype=np.float32)
        # Multi-discrete: (produto a promover, intensidade)
        self.action_space = spaces.MultiDiscrete([n + 1, 5])

        # ── Estado ──────────────────────────────────────────────────
        self.estoque: Optional[np.ndarray] = None
        self.idade_estoque: Optional[np.ndarray] = None
        self.promo_ant: Optional[np.ndarray] = None
        self.acao_ant = (0, 0)
        self.data_atual: Optional[date] = None
        self.turno: int = 0
        self.passo: int = 0

        # Pré-computar para acesso rápido durante step
        self._datas_eventos = self._indexar_calendario()
        self._fator_combinado_lookup = self._precomputar_fator_combinado()

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Amostra data de início aleatória dentro da janela do modo
        if self.modo == 'treino':
            ini, fim = self.cfg.data_inicio_treino, self.cfg.data_fim_treino
        else:
            ini, fim = self.cfg.data_inicio_val, self.cfg.data_fim_val
        # 1 ano de horizonte por episódio
        dias_disponiveis = (fim - ini).days - 365
        if dias_disponiveis <= 0:
            self.data_atual = ini
        else:
            offset = self.np_random.integers(0, dias_disponiveis)
            self.data_atual = ini + timedelta(days=int(offset))

        self.estoque = self.cfg.estoque_inicial.astype(np.float32).copy()
        self.idade_estoque = np.zeros(self.cfg.n_produtos, dtype=np.float32)
        self.promo_ant = np.zeros(self.cfg.n_produtos, dtype=np.float32)
        self.acao_ant = (0, 0)
        self.turno = 0
        self.passo = 0

        return self._get_obs(), self._get_info()

    # ── Step ───────────────────────────────────────────────────────────

    def step(self, action):
        prod_idx, intensidade = int(action[0]), int(action[1])

        # 1. Aplicar promoção
        fator_promo = np.ones(self.cfg.n_produtos, dtype=np.float32)
        preco_efetivo = self.cfg.preco_venda.astype(np.float32).copy()
        desconto_aplicado = 0.0

        if prod_idx > 0 and intensidade > 0:
            p = prod_idx - 1  # produto real (0..N-1)
            elast = abs(self.cfg.elasticidade_promo[p])
            if intensidade == 1:    # -5%
                fator_promo[p] = 1 + elast * 0.05
                preco_efetivo[p] *= 0.95
                desconto_aplicado = 0.05
            elif intensidade == 2:  # -10%
                fator_promo[p] = 1 + elast * 0.10
                preco_efetivo[p] *= 0.90
                desconto_aplicado = 0.10
            elif intensidade == 3:  # combo
                fator_promo[p] = 1.12
                # Encontrar par
                sku_p = self.cfg.skus[p]
                par = self.cfg.pares_combo.get(sku_p)
                if par and par in self.cfg.skus:
                    p_par = self.cfg.skus.index(par)
                    fator_promo[p_par] = 1.08
                # Combo sai com 10% off no V10. Replicar.
                preco_efetivo[p] *= 0.90
                desconto_aplicado = 0.10
            elif intensidade == 4:  # liquidação -25%
                fator_promo[p] = 1 + elast * 0.25
                preco_efetivo[p] *= 0.75
                desconto_aplicado = 0.25

        # 2. Calcular demanda no contexto
        dia_sem = self.data_atual.weekday()
        mes = self.data_atual.month - 1
        f_dia = self.cfg.fator_dia[:, dia_sem]
        f_turno = self.cfg.fator_turno[:, self.turno]
        f_mes = self.cfg.fator_mes[:, mes]
        temp_norm = self._temperatura_norm(self.data_atual)
        f_clima = self._fator_clima(temp_norm)
        f_evento = self._fator_evento_dia(self.data_atual)

        demanda_esperada = (self.cfg.demanda_base / 3  # por turno
                            * f_dia * f_turno * f_mes
                            * f_clima * f_evento * fator_promo)
        demanda_real = self.np_random.poisson(np.maximum(demanda_esperada, 0.01))
        vendas = np.minimum(demanda_real, self.estoque)
        rupturas = np.maximum(demanda_real - vendas, 0)

        # 3. Lucro
        lucro = float(np.sum(vendas * (preco_efetivo - self.cfg.custo)))

        # 4. Vencimentos
        self.idade_estoque += 1
        venceu = self.idade_estoque > self.cfg.validade_tipica
        perdas = np.where(venceu, self.estoque, 0)
        pen_venc = float(np.sum(self.cfg.alpha * perdas * self.cfg.custo))
        self.estoque = np.where(venceu, 0, self.estoque)
        self.idade_estoque = np.where(venceu, 0, self.idade_estoque)

        # 5. Atualizar estoque
        self.estoque = self.estoque - vendas
        # Reposição implícita: mantém ~7 dias cobertura
        cobertura_alvo = self.cfg.demanda_base * 7 / 3  # turnos
        precisa_repor = self.estoque < cobertura_alvo * 0.3
        if np.any(precisa_repor):
            qtd_repor = np.where(precisa_repor,
                                  cobertura_alvo - self.estoque, 0)
            # Idade do novo lote = 0; idade média ponderada do estoque atual
            self.idade_estoque = np.where(
                qtd_repor > 0,
                self.estoque * self.idade_estoque / (self.estoque + qtd_repor + 1e-6),
                self.idade_estoque
            )
            self.estoque = self.estoque + qtd_repor

        # 6. Penalidade de ruptura
        pen_ruptura = 1.5 * float(np.sum(rupturas * self.cfg.margem * 0.5))

        # 7. Penalidade de desconto em produto saudável
        pen_desconto = 0.0
        if desconto_aplicado > 0 and prod_idx > 0:
            p = prod_idx - 1
            risco = self.idade_estoque[p] / max(self.cfg.validade_tipica[p], 1)
            if risco < 0.4:  # produto saudável
                pen_desconto = 5.0 * (desconto_aplicado / 0.05)  # escala com desconto

        # 8. Bonus giro (produto perto de vencer + venda)
        bonus_giro = float(np.sum(vendas
                                   * (self.idade_estoque / self.cfg.validade_tipica > 0.7)
                                   * self.cfg.margem * 0.3))

        # 9. Bonus de timing (V10): fraco_flag × promoção
        bonus_timing = 0.0
        if prod_idx > 0 and intensidade > 0:
            p = prod_idx - 1
            fator_combinado_p = f_dia[p] * f_turno[p] * f_mes[p]
            limiar_fraco = self._limiar_fraco_produto(p)
            if fator_combinado_p < limiar_fraco:
                bonus_timing = self.cfg.k_timing_bonus
            else:
                bonus_timing = -self.cfg.k_timing_penalty

        # 10. Estabilidade (NOVO): bonus por manter mesma decisão
        bonus_estabilidade = 0.0
        if action[0] == self.acao_ant[0] and action[1] == self.acao_ant[1] \
                and prod_idx > 0:
            bonus_estabilidade = self.cfg.estabilidade_bonus

        reward = (lucro - pen_venc - pen_ruptura - pen_desconto
                  + bonus_giro + bonus_timing + bonus_estabilidade)

        # 11. Atualizar estado temporal
        self.promo_ant = np.zeros(self.cfg.n_produtos, dtype=np.float32)
        if prod_idx > 0 and intensidade > 0:
            self.promo_ant[prod_idx - 1] = 1.0
        self.acao_ant = (action[0], action[1])
        self.turno += 1
        if self.turno >= 3:
            self.turno = 0
            self.data_atual = self.data_atual + timedelta(days=1)
        self.passo += 1

        # 12. Episódio termina após 1 ano (1095 turnos)
        terminated = self.passo >= 1095
        truncated = False

        info = self._get_info()
        info.update({
            'lucro': lucro,
            'pen_venc': pen_venc,
            'pen_ruptura': pen_ruptura,
            'pen_desconto': pen_desconto,
            'bonus_giro': bonus_giro,
            'bonus_timing': bonus_timing,
            'bonus_estabilidade': bonus_estabilidade,
            'vendas': vendas.copy(),
            'rupturas': rupturas.copy(),
            'perdas': perdas.copy(),
        })

        return self._get_obs(), reward, terminated, truncated, info

    # ── Observação ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        n = self.cfg.n_produtos
        dia_sem = self.data_atual.weekday()
        mes = self.data_atual.month - 1

        # Calendário
        dia_oh = np.zeros(7, dtype=np.float32); dia_oh[dia_sem] = 1.
        turno_oh = np.zeros(3, dtype=np.float32); turno_oh[self.turno] = 1.
        mes_oh = np.zeros(12, dtype=np.float32); mes_oh[mes] = 1.
        dia_mes_norm = (self.data_atual.day - 1) / 30.

        # Clima
        temp_norm = self._temperatura_norm(self.data_atual)
        delta_temp = 0.0  # TODO: usar histórico

        # Evento comercial (próximo)
        evento_oh, dias_evento_norm = self._features_evento(self.data_atual)

        # Por produto
        estoque_norm = np.clip(self.estoque / (self.cfg.estoque_inicial * 2 + 1e-6),
                                0, 1).astype(np.float32)
        validade_rest = np.clip(1 - self.idade_estoque / (self.cfg.validade_tipica + 1e-6),
                                 0, 1).astype(np.float32)
        fraco_flags = self._fraco_flags(dia_sem, self.turno, mes).astype(np.float32)

        obs = np.concatenate([
            dia_oh, turno_oh, mes_oh, [dia_mes_norm],
            [temp_norm], [delta_temp],
            evento_oh, [dias_evento_norm],
            estoque_norm, validade_rest, fraco_flags, self.promo_ant,
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        return {
            'data': self.data_atual.isoformat() if self.data_atual else None,
            'turno': self.turno,
            'passo': self.passo,
        }

    # ── Utilitários ────────────────────────────────────────────────────

    def _temperatura_norm(self, d: date) -> float:
        # TODO: carregar de temperatura_historica.csv + futuro forecast
        # Aproximação: senoide anual + ruído
        dia_ano = d.timetuple().tm_yday
        return 0.5 + 0.4 * np.sin((dia_ano - 80) / 365 * 2 * np.pi)

    def _fator_clima(self, temp_norm: float) -> np.ndarray:
        # slope * temp + intercept, clipped
        slope = self.cfg.clima_coef[:, 0]
        intercept = self.cfg.clima_coef[:, 1]
        return np.clip(slope * temp_norm + intercept, 0.5, 2.0).astype(np.float32)

    def _fator_evento_dia(self, d: date) -> np.ndarray:
        """Para a data, retorna fator de uplift por produto baseado em
        evento comercial próximo. Vetor de tamanho n_produtos."""
        if d not in self._datas_eventos:
            return np.ones(self.cfg.n_produtos, dtype=np.float32)
        info_evento = self._datas_eventos[d]
        fator = np.ones(self.cfg.n_produtos, dtype=np.float32)
        for cat_afetada, uplift in info_evento:
            for i, cat_sku in enumerate(self.cfg.categorias):
                if cat_sku == cat_afetada or cat_afetada == 'todas':
                    fator[i] = max(fator[i], uplift)
        return fator

    def _features_evento(self, d: date) -> tuple[np.ndarray, float]:
        """One-hot do tipo de evento mais próximo (até 30 dias à frente)
        + dias até esse evento normalizado."""
        oh = np.zeros(10, dtype=np.float32)
        for delta in range(31):
            d_check = d + timedelta(days=delta)
            if d_check in self._datas_eventos:
                info = self._datas_eventos[d_check]
                # primeiro tipo de evento (simplificação)
                tipo = info[0][0] if info else 'outro'
                tipo_idx = self._tipo_evento_idx(tipo)
                oh[tipo_idx] = 1.0
                return oh, min(delta / 30, 1.0)
        return oh, 1.0  # nenhum evento próximo

    def _tipo_evento_idx(self, tipo: str) -> int:
        # 10 buckets de tipos de evento — mapping fixo
        mapping = {
            'chocolate': 0, 'vinho': 1, 'espumante': 2,
            'cerveja': 3, 'cerveja_premium': 3,
            'snack': 4, 'whisky': 5, 'todas': 6,
            'refrigerante': 7, 'gelo': 8, 'sorvete': 9,
        }
        return mapping.get(tipo, 6)

    def _indexar_calendario(self) -> dict:
        """data → list of (categoria, uplift_dia)"""
        if 'calendario_comercial_expandido' in str(self.cfg.calendario):
            cal_path = ROOT / 'data' / 'calendario_comercial_expandido.csv'
            if cal_path.exists():
                df = pd.read_csv(cal_path)
            else:
                return {}
        else:
            df = self.cfg.calendario
        idx = {}
        for _, row in df.iterrows():
            d = pd.to_datetime(row['data']).date()
            for cat in str(row.get('categorias', '')).split(';'):
                idx.setdefault(d, []).append((cat, float(row.get('uplift_dia', 1.0))))
        return idx

    def _precomputar_fator_combinado(self) -> np.ndarray:
        """Tabela (n_produtos, 7, 3, 12) com fator combinado por contexto."""
        n = self.cfg.n_produtos
        tab = np.zeros((n, 7, 3, 12), dtype=np.float32)
        for d in range(7):
            for t in range(3):
                for m in range(12):
                    tab[:, d, t, m] = (self.cfg.fator_dia[:, d]
                                        * self.cfg.fator_turno[:, t]
                                        * self.cfg.fator_mes[:, m])
        return tab

    def _limiar_fraco_produto(self, p: int) -> float:
        """Percentil pct_fraco da distribuição de fatores combinados do produto p."""
        valores = self._fator_combinado_lookup[p].flatten()
        return float(np.quantile(valores, self.cfg.pct_fraco))

    def _fraco_flags(self, dia: int, turno: int, mes: int) -> np.ndarray:
        """Para cada produto, 1 se contexto atual é fraco."""
        n = self.cfg.n_produtos
        flags = np.zeros(n, dtype=np.float32)
        for p in range(n):
            fator_atual = self._fator_combinado_lookup[p, dia, turno, mes]
            if fator_atual < self._limiar_fraco_produto(p):
                flags[p] = 1.0
        return flags


# ── Factory para construir env a partir de arquivos ────────────────────────

ROOT = Path(__file__).parent


def construir_env_v2(
    catalogo_path: str = 'data/catalogo_completo.xlsx',
    calibracao_path: str = 'data/calibracao_v2.json',
    calendario_path: str = 'data/calendario_comercial_expandido.csv',
    combos_path: str = 'results/combos_validados.csv',
    modo: str = 'treino',
) -> ConvenienceStoreEnvV2:
    """Constrói env V2 a partir dos arquivos calibrados.

    TODO: implementar quando dados do posto chegarem.
    Esta função é o entry point único que conecta os dados ao env.
    """
    raise NotImplementedError(
        "construir_env_v2 espera os seguintes arquivos:\n"
        f"  - {catalogo_path} (catálogo completo de SKUs)\n"
        f"  - {calibracao_path} (parâmetros calibrados por SKU)\n"
        f"  - {calendario_path} (calendário comercial) ← ✓ já existe\n"
        f"  - {combos_path} (combos validados por market basket)\n"
        "\n"
        "Aguardando: catálogo, vendas detalhadas e cupom fiscal do posto.\n"
        "Calibração será feita por script separado: calibrar_v2.py\n"
    )


if __name__ == '__main__':
    print(__doc__)
    print()
    print("Este é o esqueleto. Para virar treinável precisa de:")
    print("  1. Catálogo completo do posto (Fase 1.1)")
    print("  2. Vendas detalhadas por SKU (Fase 1.2)")
    print("  3. Cupom fiscal para combos validados (Fase 1.3)")
    print("  4. Descarte ampliado para alpha por SKU (Fase 1.4)")
    print()
    print("Calendário comercial já está pronto em data/calendario_comercial.csv")
