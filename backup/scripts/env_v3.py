"""V12 — env_v3.py = env_v2 + features de FORECAST ML no estado.

Herda ConvenienceStoreEnvV2. Único acréscimo:
- Carrega forecasters de results/v12/forecasters.pkl
- Mantém buffer rolling de 28 dias de receita por categoria
- Adiciona 1 feature por categoria ao estado: forecast_norm para o dia atual

Estado V11.7: 50 + 4N
Estado V12:   50 + 5N  (+ N features de forecast)

A demanda real do env continua sendo gerada pelos fatores manuais (sem
substituição). O forecaster apenas INFORMA o agente da expectativa, e a
política aprendida pode usar essa info para decidir.
"""
from __future__ import annotations

import pickle
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from gymnasium import spaces

from env_v2 import ConvenienceStoreEnvV2

ROOT = Path(__file__).parent
V12 = ROOT / 'results' / 'v12'


class ConvenienceStoreEnvV3(ConvenienceStoreEnvV2):
    """V12: env_v2 + forecast features no estado."""

    def __init__(self, calibracao_path: str = 'data/calibracao_v2.json',
                 modo: str = 'treino',
                 forecaster_path: str = 'results/v12/forecasters.pkl'):
        super().__init__(calibracao_path=calibracao_path, modo=modo)

        with open(ROOT / forecaster_path, 'rb') as f:
            fc_data = pickle.load(f)
        self.forecasters = fc_data['forecasters']
        self.fc_features = fc_data['features']

        # Receita média histórica por categoria (para normalização)
        # Para categorias sem forecaster, usa demanda_base * preco diário como proxy
        self.receita_media_cat = np.zeros(self.N, dtype=np.float32)
        for i, c in enumerate(self.cats):
            fc = self.forecasters.get(c['categoria'])
            if fc:
                self.receita_media_cat[i] = float(fc['media_receita_train'])
            else:
                # Fallback: usa demanda_base × preco
                self.receita_media_cat[i] = float(c['demanda_base_dia'] * c['preco_venda'])

        # Cache de forecast por dia (calcula 1×/dia, válido para 3 turnos)
        self._fc_cache_date = None
        self._fc_cache_values = None

        # Expande obs space: +N features de forecast
        n_obs = 50 + 5 * self.N
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n_obs,), dtype=np.float32)

        # Buffer rolling de receita diária (28 dias × N) — preenchido no reset()
        self.demanda_buffer = None
        self.receita_dia_atual = None

    # ── reset ──────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        # Inicializa buffer ANTES de super().reset(), porque o reset do super
        # chama _get_obs() (polimórfico → V3._get_obs → forecaster precisa buffer)
        self.demanda_buffer = np.tile(
            self.receita_media_cat, (28, 1)
        ).astype(np.float32)
        self.receita_dia_atual = np.zeros(self.N, dtype=np.float32)
        self._fc_cache_date = None
        self._fc_cache_values = None
        return super().reset(seed=seed, options=options)

    # ── step ───────────────────────────────────────────────────────────

    def step(self, action):
        turno_pre = self.turno
        obs_base, reward, term, trunc, info = super().step(action)

        # Acumula receita do dia (vendas * preco)
        vendas = info.get('vendas')
        if vendas is not None and self.receita_dia_atual is not None:
            self.receita_dia_atual += vendas.astype(np.float32) * self.preco

        # Detectar virada de dia (turno wrap 2 → 0)
        if self.turno == 0 and turno_pre == 2:
            self.demanda_buffer = np.roll(self.demanda_buffer, -1, axis=0)
            self.demanda_buffer[-1] = self.receita_dia_atual.copy()
            self.receita_dia_atual = np.zeros(self.N, dtype=np.float32)
            self._fc_cache_date = None  # invalida cache

        return obs_base, reward, term, trunc, info

    # ── Observação ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        # Acrescenta features de forecast (1 por categoria)
        fc_norm = self._compute_forecast_norm()
        return np.concatenate([base, fc_norm]).astype(np.float32)

    def _compute_forecast_norm(self) -> np.ndarray:
        # Cache: só recalcula 1× por dia
        if self._fc_cache_date == self.data_atual and self._fc_cache_values is not None:
            return self._fc_cache_values

        d = self.data_atual
        dia_sem = d.weekday()
        mes = d.month - 1
        dia_mes = d.day
        temp_norm = self._temperatura_norm()

        # Event features (alinhado com treinar_forecaster.py)
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
                # Fallback: demanda_base × fator_dia × fator_mes × preco
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


# ── Factory ────────────────────────────────────────────────────────────────

def construir_env_v3(modo: str = 'treino') -> ConvenienceStoreEnvV3:
    """Constrói env V3 (V12) a partir de data/calibracao_v2.json."""
    return ConvenienceStoreEnvV3(
        'data/calibracao_v2.json',
        modo=modo,
        forecaster_path='results/v12/forecasters.pkl',
    )


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("Construindo env V12 (V3)...")
    env = construir_env_v3(modo='treino')
    print(f"  N categorias: {env.N}")
    print(f"  obs space: {env.observation_space}")
    print(f"  action space: {env.action_space}")
    print(f"  N forecasters carregados: {len(env.forecasters)}")

    print("\nRodando smoke test (30 turnos)...")
    obs, info = env.reset(seed=42)
    print(f"  obs shape: {obs.shape}")
    print(f"  obs[-N:] (forecast features): min={obs[-env.N:].min():.3f} max={obs[-env.N:].max():.3f}")

    rewards = []
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if step in (0, 1, 5, 29):
            fc = obs[-env.N:]
            print(f"  step {step:>3d}  ação=({action[0]:>2d},{action[1]})  "
                  f"reward={reward:>8.2f}  forecast[3:6]={fc[3:6]}")
        if terminated or truncated:
            break

    print(f"\n✓ Smoke test OK. Reward total: R$ {sum(rewards):,.2f}")
