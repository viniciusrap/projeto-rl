"""Gera visualizações de aprendizado do V11 e comparação direta com V10.

Saídas em results/v11/:
- curvas_aprendizado_v11.png (4 painéis: reward, lucro, perdas, % promove)
- comparacao_v10_v11.png (4 painéis comparativos)
- comparacao_v10_v11.csv (tabela)
- diagnostico_eventos_perdidos.csv (porque F1 = 0 em Mães/Namorados)
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
RESULTS = ROOT / 'results'
V11 = RESULTS / 'v11'

# ── 1. Curvas de aprendizado V11 ──────────────────────────────────────────

log = pd.read_csv(V11 / 'training_log_v11.csv')
print(f"V11 training log: {len(log)} episódios")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Reward
ax = axes[0, 0]
ax.plot(log['episodio'], log['reward_total'] / 1000, alpha=0.4, color='steelblue')
roll = log['reward_total'].rolling(10, min_periods=1).mean() / 1000
ax.plot(log['episodio'], roll, linewidth=2, color='navy', label='média móvel (10 eps)')
ax.set_xlabel('Episódio')
ax.set_ylabel('Reward total (R$ k)')
ax.set_title('Reward — V11 (150 ep × 1095 turnos)')
ax.legend()
ax.grid(alpha=0.3)

# Lucro
ax = axes[0, 1]
ax.plot(log['episodio'], log['lucro_total'] / 1000, alpha=0.4, color='green')
roll = log['lucro_total'].rolling(10, min_periods=1).mean() / 1000
ax.plot(log['episodio'], roll, linewidth=2, color='darkgreen', label='média móvel')
ax.set_xlabel('Episódio')
ax.set_ylabel('Lucro (R$ k)')
ax.set_title('Lucro')
ax.legend()
ax.grid(alpha=0.3)

# Perdas
ax = axes[1, 0]
ax.plot(log['episodio'], log['perdas_total'], alpha=0.4, color='coral')
roll = log['perdas_total'].rolling(10, min_periods=1).mean()
ax.plot(log['episodio'], roll, linewidth=2, color='darkred', label='média móvel')
ax.set_xlabel('Episódio')
ax.set_ylabel('Perdas (un)')
ax.set_title('Perdas por vencimento')
ax.legend()
ax.grid(alpha=0.3)

# % promove + ε
ax = axes[1, 1]
ax.plot(log['episodio'], log['pct_promove'], alpha=0.4, color='purple',
         label='% promove')
roll = log['pct_promove'].rolling(10, min_periods=1).mean()
ax.plot(log['episodio'], roll, linewidth=2, color='indigo')
ax.set_xlabel('Episódio')
ax.set_ylabel('% turnos com promoção', color='purple')
ax.tick_params(axis='y', labelcolor='purple')
ax2 = ax.twinx()
ax2.plot(log['episodio'], log['epsilon'], color='orange', linewidth=2,
          label='ε (exploração)')
ax2.set_ylabel('ε', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax.set_title('Política e exploração')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(V11 / 'curvas_aprendizado_v11.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ {V11 / 'curvas_aprendizado_v11.png'}")

# ── 2. Comparação V10 vs V11 ───────────────────────────────────────────────

# Carrega comparação V10 (já gerada)
v10_cmp = pd.read_csv(RESULTS / 'comparacao_politicas.csv')
v11_cmp = pd.read_csv(V11 / 'comparacao_politicas_v11.csv')

# Tabela combinada
combo = []
for _, r in v10_cmp.iterrows():
    combo.append({
        'modelo': 'V10 (6 produtos)',
        'politica': r['politica'],
        'reward_medio': r['reward_medio'],
        'lucro_medio': r['lucro_medio'],
        'perdas_medio': r['perdas_medio'],
    })
for _, r in v11_cmp.iterrows():
    combo.append({
        'modelo': 'V11 (18 categorias)',
        'politica': r['politica'],
        'reward_medio': r['reward_medio'],
        'lucro_medio': r['lucro_medio'],
        'perdas_medio': r['perdas_medio'],
    })
df_combo = pd.DataFrame(combo)
df_combo.to_csv(V11 / 'comparacao_v10_v11.csv', index=False, encoding='utf-8')

# Gráfico comparativo
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Reward por política, separado V10 vs V11
ax = axes[0, 0]
v10_dqn_r = float(v10_cmp[v10_cmp['politica'].str.contains('DQN')]['reward_medio'].iloc[0])
v10_sp_r = float(v10_cmp[v10_cmp['politica'].str.contains('Sem')]['reward_medio'].iloc[0])
v11_dqn_r = float(v11_cmp[v11_cmp['politica'].str.contains('DQN')]['reward_medio'].iloc[0])
v11_sp_r = float(v11_cmp[v11_cmp['politica'].str.contains('Sem')]['reward_medio'].iloc[0])
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [v10_dqn_r / 1000, v10_sp_r / 1000], width,
        label='V10 (6 produtos)', color='steelblue')
ax.bar(x + width/2, [v11_dqn_r / 1000, v11_sp_r / 1000], width,
        label='V11 (18 cat)', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(['DQN', 'Sem promoção'])
ax.set_ylabel('Reward médio (R$ k)')
ax.set_title('Reward médio por política')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Lucro
ax = axes[0, 1]
v10_dqn_l = float(v10_cmp[v10_cmp['politica'].str.contains('DQN')]['lucro_medio'].iloc[0])
v10_sp_l = float(v10_cmp[v10_cmp['politica'].str.contains('Sem')]['lucro_medio'].iloc[0])
v11_dqn_l = float(v11_cmp[v11_cmp['politica'].str.contains('DQN')]['lucro_medio'].iloc[0])
v11_sp_l = float(v11_cmp[v11_cmp['politica'].str.contains('Sem')]['lucro_medio'].iloc[0])
ax.bar(x - width/2, [v10_dqn_l / 1000, v10_sp_l / 1000], width,
        label='V10', color='steelblue')
ax.bar(x + width/2, [v11_dqn_l / 1000, v11_sp_l / 1000], width,
        label='V11', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(['DQN', 'Sem promoção'])
ax.set_ylabel('Lucro médio (R$ k)')
ax.set_title('Lucro absoluto')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Perdas
ax = axes[1, 0]
v10_dqn_p = float(v10_cmp[v10_cmp['politica'].str.contains('DQN')]['perdas_medio'].iloc[0])
v10_sp_p = float(v10_cmp[v10_cmp['politica'].str.contains('Sem')]['perdas_medio'].iloc[0])
v11_dqn_p = float(v11_cmp[v11_cmp['politica'].str.contains('DQN')]['perdas_medio'].iloc[0])
v11_sp_p = float(v11_cmp[v11_cmp['politica'].str.contains('Sem')]['perdas_medio'].iloc[0])
ax.bar(x - width/2, [v10_dqn_p, v10_sp_p], width, label='V10', color='steelblue')
ax.bar(x + width/2, [v11_dqn_p, v11_sp_p], width, label='V11', color='coral')
ax.set_xticks(x)
ax.set_xticklabels(['DQN', 'Sem promoção'])
ax.set_ylabel('Perdas (un)')
ax.set_title('Perdas por vencimento')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# F1 por evento (só V11 tem)
ax = axes[1, 1]
ev = pd.read_csv(V11 / 'validacao_eventos_v11.csv')
ev_top = ev.sort_values('f1', ascending=True).tail(7)
colors = ['#2ca02c' if f > 0.3 else '#ff7f0e' if f > 0.1 else '#d62728'
           for f in ev_top['f1']]
ax.barh(range(len(ev_top)), ev_top['f1'], color=colors)
ax.set_yticks(range(len(ev_top)))
ax.set_yticklabels([e[:25] for e in ev_top['evento']], fontsize=9)
ax.set_xlabel('F1')
ax.set_title('F1 por evento comercial — V11 aprende?')
ax.axvline(0.3, color='gray', linestyle='--', alpha=0.5,
            label='F1=0.3 (bom)')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(V11 / 'comparacao_v10_v11.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"✓ {V11 / 'comparacao_v10_v11.png'}")

# ── 3. Diagnóstico — por que F1 do Dia das Mães/Namorados deu 0? ──────────

print()
print("="*70)
print("DIAGNÓSTICO — eventos com F1 = 0")
print("="*70)
ev_zero = ev[ev['f1'] == 0].copy()
print()
print(f"Eventos com F1 = 0: {len(ev_zero)}")

# Mapear: cada evento tem categorias-alvo. Quantas DESSAS estão no nosso catálogo?
import json
with open(ROOT / 'data' / 'calibracao_v2.json', encoding='utf-8') as f:
    cal = json.load(f)
cats_modelo = {c['categoria'] for c in cal['categorias']}

diag_eventos = []
for _, r in ev.iterrows():
    cats_alvo = set(r['categorias_alvo'].split(';'))
    cobertas = cats_alvo & cats_modelo
    nao_cobertas = cats_alvo - cats_modelo
    diag_eventos.append({
        'evento': r['evento'],
        'f1': r['f1'],
        'precision': r['precision'],
        'recall': r['recall'],
        'n_categorias_alvo': len(cats_alvo),
        'n_no_catalogo_v11': len(cobertas),
        'pct_cobertura_alvo_v11': round(len(cobertas) / len(cats_alvo) * 100, 1),
        'categorias_cobertas': ';'.join(sorted(cobertas)),
        'categorias_faltantes': ';'.join(sorted(nao_cobertas)),
    })

df_diag = pd.DataFrame(diag_eventos).sort_values('f1', ascending=False)
df_diag.to_csv(V11 / 'diagnostico_eventos_perdidos.csv', index=False,
                 encoding='utf-8')

print()
print("Cobertura do catálogo V11 vs categorias-alvo de cada evento:")
print()
print(f"{'Evento':<35s} {'F1':>5s} {'Cobertura':>10s} {'Faltantes':<40s}")
print('-' * 95)
for _, r in df_diag.iterrows():
    cob_pct = r['pct_cobertura_alvo_v11']
    falt = r['categorias_faltantes'][:40] if r['categorias_faltantes'] else '—'
    print(f"  {r['evento'][:33]:<33s} {r['f1']:>5.2f} {cob_pct:>9.0f}% {falt:<40s}")

print()
print("DIAGNÓSTICO:")
print(f"  Correlação cobertura × F1:")
correl = df_diag[['pct_cobertura_alvo_v11', 'f1']].corr().iloc[0, 1]
print(f"    Pearson r = {correl:.3f}")
print()
print(f"  Eventos com cobertura > 60%: F1 médio = "
      f"{df_diag[df_diag['pct_cobertura_alvo_v11'] > 60]['f1'].mean():.3f}")
print(f"  Eventos com cobertura ≤ 60%: F1 médio = "
      f"{df_diag[df_diag['pct_cobertura_alvo_v11'] <= 60]['f1'].mean():.3f}")
print()
print("CONCLUSÃO: o agente acerta eventos onde catálogo cobre alvos.")
print("Para Dia das Mães/Namorados (alvos: chocolate, vinho, espumante, perfume),")
print("V11 só tem 'chocolate' parcialmente — daí F1 = 0 nesses eventos.")
print()
print(f"✓ {V11 / 'diagnostico_eventos_perdidos.csv'}")
