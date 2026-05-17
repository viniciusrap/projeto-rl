"""V14 — DRCR Reward Wrapper (Alibaba 2019, arXiv:1912.02572).

Difference of Revenue Conversion Rate: substitui o componente de LUCRO
absoluto da reward por delta relativo a baseline calibrado.

Motivação (do paper): reward absoluto faz agente colapsar em produtos
de alto volume (no nosso caso seria cigarro se não tivesse mask, ou gelo
em fins de semana de verão). DRCR normaliza pela magnitude do baseline,
forçando o agente a otimizar GANHO MARGINAL, não receita absoluta.

Fórmula:
    drcr_lucro = (lucro_turno - baseline_lucro_turno) / baseline_lucro_turno × K
    novo_reward = drcr_lucro + outros_termos (mantidos)

K = escala (default 5000 — calibrado para drcr ter ordem de grandeza
similar a outros termos do reward V13).

Baseline calibrado em rollout "sem-promo" no mesmo env (calculado uma vez).
"""
import numpy as np
import gymnasium as gym


class DRCRWrapper(gym.Wrapper):
    """Aplica DRCR no componente de lucro do reward."""

    def __init__(self, env, baseline_lucro_turno: float = None, k_drcr: float = 5000.0):
        super().__init__(env)
        self.k_drcr = k_drcr
        if baseline_lucro_turno is None:
            # Calibrado em rollouts sem-promo (V13: ~592 R$/turno)
            baseline_lucro_turno = 592.0
        self.baseline = baseline_lucro_turno

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        lucro = info.get('lucro', 0.0)
        # DRCR: substitui lucro absoluto por delta relativo
        drcr_lucro = (lucro - self.baseline) / max(abs(self.baseline), 1.0) * self.k_drcr
        # Soma outros termos preservados (reward - lucro original)
        outros = reward - lucro
        new_reward = drcr_lucro + outros
        info['drcr_lucro'] = drcr_lucro
        info['reward_original'] = reward
        return obs, new_reward, term, trunc, info
