"""Valida o modelo V11 treinado em múltiplas dimensões.

Métricas:
1. Reward, lucro, perdas, rupturas em hold-out (2024-07 a 2026-04)
2. Comparação com baselines (Aleatória, Sem promoção)
3. Distribuição de ações por categoria — agente discrimina ou colapsa?
4. F1 de timing por categoria (igual V10 7.4)
5. F1 de evento comercial — agente promove na janela certa?
6. Análise por dia da semana, turno, mês

Saídas em results/v11/:
  - validacao_metricas.csv
  - validacao_acoes_por_categoria.csv
  - validacao_timing_v11.csv
  - validacao_eventos_v11.csv
  - comparacao_politicas_v11.csv

Uso: python validar_v11.py [--n_episodios N] [--seed N]
"""
import argparse
import io
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env_v2 import construir_env_v2
from treinar_v11 import BranchingDQN

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results' / 'v11'
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Argumentos ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--n_episodios', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_steps', type=int, default=1095)
parser.add_argument('--modelo', type=str, default='results/v11/dqn_v11.pt')
args = parser.parse_args()

# ── Carrega modelo ──────────────────────────────────────────────────────────

modelo_path = ROOT / args.modelo
if not modelo_path.exists():
    print(f"✗ Modelo não encontrado em {modelo_path}")
    print(f"  Rode primeiro: python treinar_v11.py")
    sys.exit(1)

checkpoint = torch.load(modelo_path, map_location='cpu', weights_only=False)
cfg = checkpoint['config']
modelo = BranchingDQN(cfg['obs_dim'], cfg['n_produtos'], cfg['n_intensidades'])
modelo.load_state_dict(checkpoint['state_dict'])
modelo.eval()
print(f"✓ Modelo carregado: obs_dim={cfg['obs_dim']}, "
      f"n_produtos={cfg['n_produtos']}, n_intensidades={cfg['n_intensidades']}")

# ── Função de rollout ──────────────────────────────────────────────────────

def rollout(env, politica, n_ep: int, max_steps: int, seed_base: int):
    """politica: callable(obs) -> (prod, intensidade)."""
    rewards, lucros, perdas, rupturas = [], [], [], []
    decisoes_completas = []  # uma linha por (ep, step)

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
        'reward_medio': np.mean(rewards),
        'reward_std': np.std(rewards),
        'lucro_medio': np.mean(lucros),
        'perdas_medio': np.mean(perdas),
        'rupturas_medio': np.mean(rupturas),
        'decisoes': decisoes_completas,
    }

# ── Políticas ──────────────────────────────────────────────────────────────

env = construir_env_v2(modo='validacao')

def politica_dqn(obs):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return modelo.select_action(obs_t, eps=0.0,
                                  rng=np.random.default_rng(0))

def politica_aleatoria(obs):
    return (int(np.random.randint(0, env.N + 1)),
            int(np.random.randint(0, 5)))

def politica_sempre_combo(obs):
    return (1, 3)  # combo no produto 0

def politica_sem_promo(obs):
    return (0, 0)

# ── 1. Comparação entre políticas ─────────────────────────────────────────

print("\n=== COMPARAÇÃO DE POLÍTICAS (hold-out 2024-2026) ===")
print(f"  {args.n_episodios} episódios × {args.max_steps} steps...")
print()

politicas = {
    'DQN V11 (nosso)': politica_dqn,
    'Sempre combo': politica_sempre_combo,
    'Aleatória': politica_aleatoria,
    'Sem promoção': politica_sem_promo,
}

resultados = {}
decisoes_dqn = None
for nome, pol in politicas.items():
    print(f"  Rodando {nome}...")
    out = rollout(env, pol, args.n_episodios, args.max_steps, args.seed)
    resultados[nome] = out
    if nome == 'DQN V11 (nosso)':
        decisoes_dqn = out['decisoes']

# Tabela comparativa
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
df_cmp.to_csv(RESULTS / 'comparacao_politicas_v11.csv', index=False,
                encoding='utf-8')

print()
print("Resultado:")
print(df_cmp.to_string(index=False))

# ── 2. Distribuição de ações do DQN por categoria ─────────────────────────

df_dec = pd.DataFrame(decisoes_dqn)
total_steps = len(df_dec)
n_p = env.N + 1
n_i = 5

# Conta combinações (prod, intensidade)
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
df_acao.to_csv(RESULTS / 'validacao_acoes_por_categoria.csv',
                 index=False, encoding='utf-8')

print()
print("=== DISTRIBUIÇÃO DE AÇÕES DO DQN (top 15) ===")
print(df_acao.head(15).to_string(index=False))

# ── 3. F1 de timing por categoria (versão V11) ────────────────────────────

# Para cada categoria, "contexto fraco" = bottom 30% da distribuição
# do fator combinado (já calculado no env)
fraco_por_cat = env._limiar_fraco  # array de N
fator_combinado = env._fator_combinado  # (N, 7, 3, 12)

# Para cada decisão DQN, classificar contexto como fraco/forte
def contexto_fraco_para_produto(produto_idx, data_str, turno):
    if produto_idx == 0:
        return None
    d = date.fromisoformat(data_str)
    dia = d.weekday()
    mes = d.month - 1
    p = produto_idx - 1
    fator = fator_combinado[p, dia, turno, mes]
    return fator < fraco_por_cat[p]

# Calcular F1 por categoria
metricas_timing = []
for cat_idx in range(env.N):
    cat_nome = env.cats[cat_idx]['categoria']
    # Verdadeiros: quando o agente promoveu cat (prod_idx == cat_idx + 1)
    # E o contexto naquele momento era fraco para essa categoria
    df_cat = df_dec[df_dec['produto'] == cat_idx + 1].copy()
    if len(df_cat) == 0:
        continue
    # Para cada turno, contexto fraco daquela categoria?
    df_cat['fraco'] = df_cat.apply(
        lambda r: contexto_fraco_para_produto(r['produto'], r['data'], r['turno']),
        axis=1
    )
    # Agora, o agente PROMOVEU sempre que escolheu essa categoria
    # Precisamos comparar com universo: todas as decisões onde categoria poderia ter sido promovida
    # mas não foi (incluindo decisoes onde escolheu OUTRA categoria ou nenhuma)

    # Simplificação: calcular precision/recall de promover_essa_cat × contexto_fraco_dessa_cat
    # em TODO o conjunto de decisões
    todos_fracos = 0
    for _, row in df_dec.iterrows():
        if contexto_fraco_para_produto(cat_idx + 1, row['data'], row['turno']):
            todos_fracos += 1

    promovidos = len(df_cat)
    promovidos_fracos = int(df_cat['fraco'].sum())

    if promovidos > 0:
        precision = promovidos_fracos / promovidos
    else:
        precision = 0
    if todos_fracos > 0:
        recall = promovidos_fracos / todos_fracos
    else:
        recall = 0
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
df_timing.to_csv(RESULTS / 'validacao_timing_v11.csv', index=False,
                   encoding='utf-8')

print()
print("=== F1 DE TIMING POR CATEGORIA (V11) ===")
if len(df_timing) > 0:
    print(df_timing.to_string(index=False))
else:
    print("  (agente nunca promoveu nenhuma categoria - modelo subtreinado)")

# ── 4. F1 de evento comercial ──────────────────────────────────────────────

# Para cada decisão, está em janela de evento comercial?
# Se sim, qual categoria deveria ser promovida (categorias_afetadas)?

eventos_por_data = env._eventos_por_data
metricas_evento = []

# Agrupa por nome de evento (sem repetir por ano)
nomes_eventos = set()
for evs in eventos_por_data.values():
    for ev in evs:
        nomes_eventos.add(ev['evento'])

for nome_ev in sorted(nomes_eventos):
    # Datas que pertencem à janela desse evento
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

    # Decisões dentro da janela
    df_jan = df_dec[df_dec['data'].isin(datas_janela)]
    if len(df_jan) == 0:
        continue

    # Acertos: produto promovido bate com categoria alvo
    n_total = len(df_jan)
    n_acertos = int(df_jan['produto'].apply(
        lambda p: (p - 1) in cats_alvo_idx if p > 0 else False
    ).sum())
    n_promovidos = int((df_jan['produto'] > 0).sum())

    if n_promovidos > 0:
        precision_evento = n_acertos / n_promovidos
    else:
        precision_evento = 0
    recall_evento = n_acertos / n_total
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
df_evento.to_csv(RESULTS / 'validacao_eventos_v11.csv', index=False,
                   encoding='utf-8')

print()
print("=== F1 DE EVENTOS COMERCIAIS (top 10) ===")
if len(df_evento) > 0:
    print(df_evento.head(10).to_string(index=False))
else:
    print("  (sem eventos no horizonte avaliado)")

# ── 5. Métricas consolidadas ──────────────────────────────────────────────

dqn = resultados['DQN V11 (nosso)']
sem_promo = resultados['Sem promoção']

metricas_finais = {
    'modelo': 'V11',
    'data_validacao': date.today().isoformat(),
    'n_episodios': args.n_episodios,
    'steps_por_episodio': args.max_steps,
    'reward_medio_dqn': round(dqn['reward_medio'], 2),
    'lucro_medio_dqn': round(dqn['lucro_medio'], 2),
    'lucro_medio_sem_promo': round(sem_promo['lucro_medio'], 2),
    'delta_lucro_pct': round((dqn['lucro_medio'] - sem_promo['lucro_medio'])
                                / sem_promo['lucro_medio'] * 100, 2),
    'perdas_dqn': round(dqn['perdas_medio'], 2),
    'perdas_sem_promo': round(sem_promo['perdas_medio'], 2),
    'delta_perdas_pct': round((dqn['perdas_medio'] - sem_promo['perdas_medio'])
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
    RESULTS / 'validacao_metricas.csv', index=False, encoding='utf-8'
)

print()
print("="*60)
print("MÉTRICAS FINAIS V11")
print("="*60)
for k, v in metricas_finais.items():
    print(f"  {k:<30s} {v}")

print()
print(f"✓ Saídas em {RESULTS}/")
