"""Avalia se 150 episódios são suficientes para o V11 convergir.

Gera 6 painéis de diagnóstico:
1. Reward por episódio + média móvel 10 + IC ± 2σ
2. Lucro por episódio + média móvel
3. Loss por episódio (HuberLoss)
4. ε (exploração) por episódio
5. % de turnos com promoção (medida de "qual fração da política colapsou")
6. Variância de reward em janelas deslizantes (medida de estabilidade)

Métricas de convergência calculadas:
- Coeficiente de variação dos últimos 30 episódios
- Slope da regressão linear sobre últimos 50 episódios (deve ser ~0)
- Comparação primeiros 50 vs últimos 50 (teste t implícito)

Output: results/v11/convergencia_v11.png + convergencia_v11_metricas.csv
"""
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V11 = ROOT / 'results' / 'v11'

log = pd.read_csv(V11 / 'training_log_v11.csv')
n_eps = len(log)
print(f"Treino: {n_eps} episódios")
print(f"  reward range: R$ {log['reward_total'].min():,.0f} a R$ {log['reward_total'].max():,.0f}")
print(f"  ε inicial: {log['epsilon'].iloc[0]:.3f}, final: {log['epsilon'].iloc[-1]:.3f}")
print()

# ── Métricas de convergência ─────────────────────────────────────────────

def calc_convergencia(serie, n_ultimos=30):
    """Métricas de estabilidade dos últimos N episódios."""
    ultimos = serie.iloc[-n_ultimos:]
    media = ultimos.mean()
    std = ultimos.std()
    cv = std / abs(media) if abs(media) > 1e-6 else float('inf')

    # Slope da regressão linear (estabilidade direcional)
    x = np.arange(len(ultimos))
    slope, intercept = np.polyfit(x, ultimos, 1)
    # Slope relativo (% por episódio)
    slope_rel = (slope / abs(media) * 100) if abs(media) > 1e-6 else 0

    # Comparar primeiros 50 com últimos 50
    if len(serie) >= 100:
        primeiros = serie.iloc[:50].mean()
        ultimos50 = serie.iloc[-50:].mean()
        delta_50 = ultimos50 - primeiros
        delta_pct = (delta_50 / abs(primeiros) * 100) if abs(primeiros) > 1e-6 else 0
    else:
        delta_pct = None

    return {
        'media_ultimos_30': round(media, 2),
        'std_ultimos_30': round(std, 2),
        'cv_ultimos_30': round(cv, 4),
        'slope_pct_por_episodio': round(slope_rel, 4),
        'delta_pct_primeiros50_vs_ultimos50': round(delta_pct, 2) if delta_pct else None,
    }

metricas = {
    'reward_total': calc_convergencia(log['reward_total']),
    'lucro_total': calc_convergencia(log['lucro_total']),
    'perdas_total': calc_convergencia(log['perdas_total']),
    'pct_promove': calc_convergencia(log['pct_promove']),
    'loss_media': calc_convergencia(log['loss_media']),
}

df_metricas = pd.DataFrame(metricas).T
df_metricas.index.name = 'serie'
df_metricas.to_csv(V11 / 'convergencia_v11_metricas.csv', encoding='utf-8')

print("CONVERGÊNCIA (últimos 30 episódios):")
print(df_metricas.to_string())
print()

# Critério de convergência
reward_cv = metricas['reward_total']['cv_ultimos_30']
reward_slope = abs(metricas['reward_total']['slope_pct_por_episodio'])
print("DIAGNÓSTICO:")
if reward_cv < 0.05 and reward_slope < 0.1:
    print(f"  ✓ CONVERGIDO (cv {reward_cv:.4f} < 0.05, slope {reward_slope:.3f}% < 0.1%/ep)")
elif reward_cv < 0.10 and reward_slope < 0.2:
    print(f"  ↗ QUASE CONVERGIDO (cv {reward_cv:.4f}, slope {reward_slope:.3f}%/ep)")
else:
    print(f"  ✗ NÃO CONVERGIDO (cv {reward_cv:.4f}, slope {reward_slope:.3f}%/ep)")
    print(f"    Recomenda mais episódios.")

# ── Visualização ──────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(15, 11))

# Painel 1: Reward com média móvel + IC
ax = axes[0, 0]
ax.plot(log['episodio'], log['reward_total'] / 1000, alpha=0.3, color='steelblue',
         linewidth=1, label='por episódio')
roll = log['reward_total'].rolling(10, min_periods=1).mean() / 1000
roll_std = log['reward_total'].rolling(10, min_periods=1).std() / 1000
ax.plot(log['episodio'], roll, color='navy', linewidth=2, label='média móvel (10 eps)')
ax.fill_between(log['episodio'], roll - 2 * roll_std, roll + 2 * roll_std,
                  alpha=0.15, color='navy', label='IC ± 2σ')
# Marcar últimos 30
ax.axvspan(n_eps - 30, n_eps, alpha=0.1, color='green',
            label='últimos 30 (convergência)')
ax.set_xlabel('Episódio')
ax.set_ylabel('Reward total (R$ k)')
ax.set_title(f'Reward — CV={reward_cv:.3f}, slope={reward_slope:.3f}%/ep')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

# Painel 2: Lucro + média móvel
ax = axes[0, 1]
ax.plot(log['episodio'], log['lucro_total'] / 1000, alpha=0.3, color='green',
         linewidth=1)
roll_l = log['lucro_total'].rolling(10, min_periods=1).mean() / 1000
ax.plot(log['episodio'], roll_l, color='darkgreen', linewidth=2,
         label='média móvel')
ax.set_xlabel('Episódio')
ax.set_ylabel('Lucro (R$ k)')
ax.set_title(f"Lucro — média últimos 30: R$ {metricas['lucro_total']['media_ultimos_30']:,.0f}")
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

# Painel 3: Loss
ax = axes[1, 0]
ax.plot(log['episodio'], log['loss_media'], alpha=0.4, color='coral')
roll_loss = log['loss_media'].rolling(10, min_periods=1).mean()
ax.plot(log['episodio'], roll_loss, color='darkred', linewidth=2)
ax.set_yscale('log')
ax.set_xlabel('Episódio')
ax.set_ylabel('Loss média (HuberLoss, log scale)')
ax.set_title('Loss do otimizador — deve descer')
ax.grid(alpha=0.3, which='both')

# Painel 4: ε exploração
ax = axes[1, 1]
ax.plot(log['episodio'], log['epsilon'], color='orange', linewidth=2)
ax.axhline(0.1, color='red', linestyle='--', alpha=0.5,
            label='ε=0.10 (limite explore-exploit)')
ax.axhline(0.05, color='darkred', linestyle='--', alpha=0.5,
            label='ε=0.05 (limite mínimo)')
ax.set_xlabel('Episódio')
ax.set_ylabel('ε (probabilidade exploração)')
ax.set_title(f"ε final = {log['epsilon'].iloc[-1]:.3f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Painel 5: % promove
ax = axes[2, 0]
ax.plot(log['episodio'], log['pct_promove'], color='purple', alpha=0.4)
roll_p = log['pct_promove'].rolling(10, min_periods=1).mean()
ax.plot(log['episodio'], roll_p, color='indigo', linewidth=2,
         label='média móvel')
ax.set_xlabel('Episódio')
ax.set_ylabel('% turnos com promoção')
ax.set_title(f"Política — promove {roll_p.iloc[-1]:.1f}% dos turnos")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Painel 6: Variância em janelas deslizantes
ax = axes[2, 1]
window = 20
var_reward = log['reward_total'].rolling(window).std()
ax.plot(log['episodio'], var_reward / 1000, color='teal', linewidth=2)
ax.set_xlabel('Episódio')
ax.set_ylabel(f'σ do reward (janela {window} eps, R$ k)')
ax.set_title('Variância — queda indica estabilização')
ax.grid(alpha=0.3)

plt.suptitle(f'Convergência V11 — {n_eps} episódios de treino',
              fontsize=14, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(V11 / 'convergencia_v11.png', dpi=120, bbox_inches='tight')
plt.close()

print()
print(f"✓ {V11 / 'convergencia_v11.png'}")
print(f"✓ {V11 / 'convergencia_v11_metricas.csv'}")

# ── Recomendação final ────────────────────────────────────────────────────

print()
print("=" * 70)
print("RECOMENDAÇÃO")
print("=" * 70)
print()

# Avaliar se ε ainda alto
eps_final = log['epsilon'].iloc[-1]
if eps_final > 0.1:
    print(f"⚠ ε final {eps_final:.3f} > 0.1 — agente ainda explorando demais.")
    n_extra = int(np.log(0.05 / eps_final) / np.log(0.99))
    print(f"  Para ε < 0.05: precisa ~{n_extra} episódios a mais.")
else:
    print(f"✓ ε final {eps_final:.3f} ≤ 0.10 — exploração estabilizada.")

# Avaliar se reward ainda subindo
if metricas['reward_total']['delta_pct_primeiros50_vs_ultimos50']:
    delta = metricas['reward_total']['delta_pct_primeiros50_vs_ultimos50']
    print(f"\nReward primeiros 50 vs últimos 50: {delta:+.1f}%")
    if abs(delta) < 5:
        print("  ✓ Estável (variação < 5%)")
    elif delta > 5:
        print(f"  ↗ Ainda subindo ({delta:.1f}%) — mais episódios podem ajudar")
    else:
        print(f"  ↘ Descendo ({delta:.1f}%) — investigar instabilidade")

print()
if reward_cv < 0.05 and reward_slope < 0.1 and eps_final < 0.10:
    print("VEREDICTO: 150 ep é SUFICIENTE para este modelo.")
else:
    deficit = []
    if reward_cv >= 0.05:
        deficit.append(f"CV alto ({reward_cv:.3f})")
    if reward_slope >= 0.1:
        deficit.append(f"slope alto ({reward_slope:.3f}%/ep)")
    if eps_final >= 0.10:
        deficit.append(f"ε alto ({eps_final:.3f})")
    print(f"VEREDICTO: 150 ep NÃO basta para convergência total.")
    print(f"  Problemas: {', '.join(deficit)}")
    print(f"  Sugestão: 250-300 episódios + eps_decay mais agressivo (0.985)")
