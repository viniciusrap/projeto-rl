"""Valida o modelo V12 no ambiente consolidado.

Métricas:
- Reward, lucro, perdas, rupturas em hold-out (2024-2026)
- Comparação com baselines (Aleatória, Sem promoção, Sempre combo)
- Distribuição de ações por categoria
- F1 timing por categoria
- F1 evento comercial

Saídas em results/v12/:
  - validacao_metricas_v12.csv
  - validacao_acoes_por_categoria_v12.csv
  - validacao_timing_v12.csv
  - validacao_eventos_v12.csv
  - comparacao_v11_vs_v12.csv  (mantido nome legacy — agora só baselines)

Uso: python validar_v12.py [--n_episodios N] [--seed N]
"""
import argparse
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env_v12 import construir_env_v12
from dqn import BranchingDQN

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
V12 = ROOT / 'results' / 'v12'
V12.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_episodios', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_steps', type=int, default=1095)
parser.add_argument('--modelo_v12', type=str, default='results/v12/dqn_v12.pt')
args = parser.parse_args()


def load_dqn(path: str) -> BranchingDQN:
    p = ROOT / path
    if not p.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {p}")
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    c = ckpt['config']
    m = BranchingDQN(c['obs_dim'], c['n_produtos'], c['n_intensidades'])
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    return m, c


def rollout(env, politica, n_ep: int, max_steps: int, seed_base: int):
    rewards, lucros, perdas, rupturas = [], [], [], []
    decisoes_completas = []
    for ep in range(n_ep):
        obs, info = env.reset(seed=seed_base + ep * 100)
        ep_r, ep_l, ep_p, ep_rup = 0.0, 0.0, 0.0, 0.0
        for step in range(max_steps):
            ap, ai = politica(obs)
            obs2, r, term, trunc, info = env.step((ap, ai))
            ep_r += r
            ep_l += info['lucro']
            ep_p += float(info['perdas'].sum())
            ep_rup += float(info['rupturas'].sum())
            decisoes_completas.append({
                'ep': ep,
                'step': step,
                'data': info['data'],
                'turno': info['turno'],
                'produto': ap,
                'intensidade': ai,
            })
            obs = obs2
            if term or trunc:
                break
        rewards.append(ep_r)
        lucros.append(ep_l)
        perdas.append(ep_p)
        rupturas.append(ep_rup)
    return {
        'reward_medio': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'lucro_medio': float(np.mean(lucros)),
        'perdas_medio': float(np.mean(perdas)),
        'rupturas_medio': float(np.mean(rupturas)),
        'decisoes': decisoes_completas,
    }


# ── Carregar modelo V12 ─────────────────────────────────────────────

print("Carregando modelo V12…")
modelo_v12, cfg_v12 = load_dqn(args.modelo_v12)
print(f"  V12: obs_dim={cfg_v12['obs_dim']}, n_p={cfg_v12['n_produtos']}, n_i={cfg_v12['n_intensidades']}")

# ── Env ─────────────────────────────────────────────────────────────────

env = construir_env_v12(modo='validacao')

# ── Políticas ──────────────────────────────────────────────────────────────

def pol_v12(obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return modelo_v12.select_action(obs_t, eps=0.0,
                                      rng=np.random.default_rng(0))

def pol_aleatoria(obs):
    return (int(np.random.randint(0, env.N + 1)),
            int(np.random.randint(0, 5)))

def pol_sempre_combo(obs):
    return (1, 3)

def pol_sem_promo(obs):
    return (0, 0)

# ── 1. Comparação entre políticas ─────────────────────────────────────────

print(f"\n=== COMPARAÇÃO ({args.n_episodios} ep × {args.max_steps} steps) ===")

resultados = {}
decisoes_v12 = None

print(f"  Rodando DQN V12 (nosso)…")
out = rollout(env, pol_v12, args.n_episodios, args.max_steps, args.seed)
resultados['DQN V12 (nosso)'] = out
decisoes_v12 = out['decisoes']

print(f"  Rodando Sempre combo…")
resultados['Sempre combo'] = rollout(env, pol_sempre_combo,
                                       args.n_episodios, args.max_steps, args.seed)
print(f"  Rodando Aleatória…")
resultados['Aleatória'] = rollout(env, pol_aleatoria,
                                    args.n_episodios, args.max_steps, args.seed)
print(f"  Rodando Sem promoção…")
resultados['Sem promoção'] = rollout(env, pol_sem_promo,
                                      args.n_episodios, args.max_steps, args.seed)

comparacao = []
for nome, r in resultados.items():
    comparacao.append({
        'politica': nome,
        'reward_medio': round(r['reward_medio'], 2),
        'reward_std': round(r['reward_std'], 2),
        'lucro_medio': round(r['lucro_medio'], 2),
        'perdas_medio': round(r['perdas_medio'], 2),
        'rupturas_medio': round(r['rupturas_medio'], 2),
    })
df_cmp = pd.DataFrame(comparacao)
df_cmp.to_csv(V12 / 'comparacao_v11_vs_v12.csv', index=False, encoding='utf-8')

print()
print("Resultado:")
print(df_cmp.to_string(index=False))

# Δ V12 vs Sem promoção (baseline operacional)
v12 = resultados['DQN V12 (nosso)']
sp = resultados['Sem promoção']
delta_lucro = (v12['lucro_medio'] - sp['lucro_medio']) / sp['lucro_medio'] * 100
delta_perdas = (v12['perdas_medio'] - sp['perdas_medio']) / max(sp['perdas_medio'], 1) * 100
print()
print(f"V12 vs Sem promoção:")
print(f"  Δ lucro:    {delta_lucro:+.2f}%")
print(f"  Δ perdas:   {delta_perdas:+.2f}%")

# ── 2. Distribuição de ações DQN V12 ─────────────────────────────────────

df_dec = pd.DataFrame(decisoes_v12)
total_steps = len(df_dec)
n_p = env.N + 1
n_i = 5

acao_dist = []
for p in range(n_p):
    for i in range(n_i):
        n = int(((df_dec['produto'] == p) & (df_dec['intensidade'] == i)).sum())
        cat_nome = env.cats[p - 1]['categoria'] if p > 0 else '_sem_promo'
        intens_nome = ['nada', 'desc5%', 'desc10%', 'combo', 'liq25%'][i]
        acao_dist.append({
            'produto_idx': p,
            'categoria': cat_nome,
            'intensidade': intens_nome,
            'n': n,
            'pct': round(n / total_steps * 100, 2),
        })

df_acao = pd.DataFrame(acao_dist)
df_acao = df_acao[df_acao['n'] > 0].sort_values('pct', ascending=False)
df_acao.to_csv(V12 / 'validacao_acoes_por_categoria_v12.csv',
                index=False, encoding='utf-8')

print()
print("=== DISTRIBUIÇÃO DE AÇÕES V12 (top 15) ===")
print(df_acao.head(15).to_string(index=False))

# ── 3. F1 timing por categoria ────────────────────────────────────────────

fraco_por_cat = env._limiar_fraco
fator_combinado = env._fator_combinado

def contexto_fraco_para_produto(produto_idx, data_str, turno):
    if produto_idx == 0:
        return None
    d = date.fromisoformat(data_str)
    dia = d.weekday()
    mes = d.month - 1
    p = produto_idx - 1
    fator = fator_combinado[p, dia, turno, mes]
    return fator < fraco_por_cat[p]

# Mapeamento (data,turno) -> fraco_flag pré-computado para vetorizar
df_dec['data_dt'] = pd.to_datetime(df_dec['data']).dt.date
df_dec['dia_sem'] = df_dec['data_dt'].apply(lambda d: d.weekday())
df_dec['mes'] = df_dec['data_dt'].apply(lambda d: d.month - 1)

metricas_timing = []
for cat_idx in range(env.N):
    cat_nome = env.cats[cat_idx]['categoria']
    fator_arr = fator_combinado[cat_idx]  # (7,3,12)
    df_dec['fraco_cat'] = df_dec.apply(
        lambda r: fator_arr[r['dia_sem'], r['turno'], r['mes']] < fraco_por_cat[cat_idx],
        axis=1
    )
    df_cat = df_dec[df_dec['produto'] == cat_idx + 1]
    if len(df_cat) == 0:
        continue
    promovidos = len(df_cat)
    promovidos_fracos = int(df_cat['fraco_cat'].sum())
    todos_fracos = int(df_dec['fraco_cat'].sum())

    precision = promovidos_fracos / promovidos if promovidos else 0
    recall = promovidos_fracos / todos_fracos if todos_fracos else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    metricas_timing.append({
        'categoria': cat_nome,
        'n_promovidos': promovidos,
        'n_promovidos_em_fraco': promovidos_fracos,
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3),
    })

df_timing = pd.DataFrame(metricas_timing)
if len(df_timing) > 0:
    df_timing = df_timing.sort_values('f1', ascending=False)
df_timing.to_csv(V12 / 'validacao_timing_v12.csv',
                  index=False, encoding='utf-8')

print()
print("=== F1 TIMING POR CATEGORIA (V12) ===")
if len(df_timing) > 0:
    print(df_timing.to_string(index=False))

# ── 4. F1 evento comercial ────────────────────────────────────────────────

eventos_por_data = env._eventos_por_data
nomes_eventos = set()
for evs in eventos_por_data.values():
    for ev in evs:
        nomes_eventos.add(ev['evento'])

metricas_evento = []
for nome_ev in sorted(nomes_eventos):
    datas_janela = []
    cats_alvo = set()
    for d, evs in eventos_por_data.items():
        for ev in evs:
            if ev['evento'] == nome_ev:
                datas_janela.append(d.isoformat())
                cats_alvo.update(ev['categorias'])
    if not datas_janela:
        continue
    cats_alvo_idx = [env.nome_para_idx[c] for c in cats_alvo
                      if c in env.nome_para_idx]
    if not cats_alvo_idx:
        continue
    df_jan = df_dec[df_dec['data'].isin(datas_janela)]
    if len(df_jan) == 0:
        continue
    n_total = len(df_jan)
    n_acertos = int(df_jan['produto'].apply(
        lambda p: (p - 1) in cats_alvo_idx if p > 0 else False
    ).sum())
    n_promovidos = int((df_jan['produto'] > 0).sum())
    precision_evento = n_acertos / n_promovidos if n_promovidos else 0
    recall_evento = n_acertos / n_total if n_total else 0
    f1_evento = (2 * precision_evento * recall_evento
                  / (precision_evento + recall_evento + 1e-9))
    metricas_evento.append({
        'evento': nome_ev,
        'n_turnos_janela': n_total,
        'n_promovidos': n_promovidos,
        'n_acertos': n_acertos,
        'precision': round(precision_evento, 3),
        'recall': round(recall_evento, 3),
        'f1': round(f1_evento, 3),
        'categorias_alvo': ';'.join(sorted(cats_alvo)),
    })

df_evento = pd.DataFrame(metricas_evento)
if len(df_evento) > 0:
    df_evento = df_evento.sort_values('f1', ascending=False)
df_evento.to_csv(V12 / 'validacao_eventos_v12.csv', index=False,
                   encoding='utf-8')

print()
print("=== F1 EVENTOS COMERCIAIS V12 (top 10) ===")
if len(df_evento) > 0:
    print(df_evento.head(10).to_string(index=False))

# ── 5. Métricas finais ────────────────────────────────────────────────────

dqn_v12 = resultados['DQN V12 (nosso)']
sem_promo = resultados['Sem promoção']

metricas_finais = {
    'modelo': 'V12',
    'data_validacao': date.today().isoformat(),
    'n_episodios': args.n_episodios,
    'steps_por_episodio': args.max_steps,
    'reward_medio_dqn': round(dqn_v12['reward_medio'], 2),
    'lucro_medio_dqn': round(dqn_v12['lucro_medio'], 2),
    'lucro_medio_sem_promo': round(sem_promo['lucro_medio'], 2),
    'delta_lucro_pct': round((dqn_v12['lucro_medio'] - sem_promo['lucro_medio'])
                                / sem_promo['lucro_medio'] * 100, 2),
    'perdas_dqn': round(dqn_v12['perdas_medio'], 2),
    'perdas_sem_promo': round(sem_promo['perdas_medio'], 2),
    'delta_perdas_pct': round((dqn_v12['perdas_medio'] - sem_promo['perdas_medio'])
                                 / max(sem_promo['perdas_medio'], 1) * 100, 2),
    'f1_timing_medio': round(df_timing['f1'].mean(), 3) if len(df_timing) > 0 else 0.0,
    'f1_evento_medio': round(df_evento['f1'].mean(), 3) if len(df_evento) > 0 else 0.0,
    'pct_categorias_promovidas': round(
        len(df_acao[df_acao['produto_idx'] > 0]['produto_idx'].unique())
        / env.N * 100, 1
    ),
    'pct_intensidades_usadas': round(
        len(df_acao[df_acao['produto_idx'] > 0]['intensidade'].unique())
        / 4 * 100, 1
    ),
}
pd.DataFrame([metricas_finais]).to_csv(
    V12 / 'validacao_metricas_v12.csv', index=False, encoding='utf-8'
)

print()
print("=" * 60)
print("MÉTRICAS FINAIS V12")
print("=" * 60)
for k, v in metricas_finais.items():
    print(f"  {k:<32s} {v}")

print()
print(f"✓ Saídas em {V12}/")
