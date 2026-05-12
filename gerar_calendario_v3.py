"""V3 do output operacional — usa o modelo V11 treinado.

Diferenças vs V1/V2:
- N categorias (não 6 fixos como V1/V2)
- Decisão por categoria + intensidade (não só combo)
- Considera datas comerciais embutidas no estado V11
- Considera prior de padrão promocional do Dunnhumby

Pipeline:
1. Carrega V11 treinado (dqn_v11.pt)
2. Constrói env iniciando na data atual + horizonte 60 dias
3. Rollout determinístico (ε=0)
4. Agrupa decisões consecutivas em CAMPANHAS (≥2 dias, ≤7 dias)
5. Calcula uplift esperado de cada campanha
6. Salva JSON + Markdown legível pro dono

Uso: python gerar_calendario_v3.py [--horizonte 60] [--data 2026-05-12]
"""
import argparse
import io
import json
import sys
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
parser.add_argument('--horizonte', type=int, default=60,
                     help='Dias a frente do output')
parser.add_argument('--data', type=str, default='2026-05-12',
                     help='Data inicial (YYYY-MM-DD)')
parser.add_argument('--modelo', type=str, default='results/v11/dqn_v11.pt')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--min_dias_campanha', type=int, default=2)
parser.add_argument('--max_dias_campanha', type=int, default=7)
args = parser.parse_args()

DATA_HOJE = date.fromisoformat(args.data)
HORIZONTE = args.horizonte

# ── Carrega modelo ──────────────────────────────────────────────────────────

ckpt = torch.load(ROOT / args.modelo, map_location='cpu', weights_only=False)
cfg = ckpt['config']
modelo = BranchingDQN(cfg['obs_dim'], cfg['n_produtos'], cfg['n_intensidades'])
modelo.load_state_dict(ckpt['state_dict'])
modelo.eval()

env = construir_env_v2(modo='validacao')

# Forçar data inicial específica (override do reset aleatório)
env.estoque = env.estoque_inicial.copy()
env.idade = np.zeros(env.N, dtype=np.float32)
env.promo_ant = np.zeros(env.N, dtype=np.float32)
env.acao_ant = (0, 0)
env.turno = 0
env.passo = 0
env.data_atual = DATA_HOJE
env._np_random = np.random.default_rng(args.seed)

# ── Rollout ──────────────────────────────────────────────────────────────

print(f"Rodando rollout determinístico de {HORIZONTE} dias a partir de {DATA_HOJE}...")
decisoes = []
n_steps = HORIZONTE * 3
obs = env._get_obs()
for step in range(n_steps):
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    ap, ai = modelo.select_action(obs_t, eps=0.0,
                                    rng=np.random.default_rng(0))
    decisoes.append({
        'step': step,
        'data': env.data_atual.isoformat(),
        'turno': env.turno,
        'produto_idx': ap,
        'intensidade_idx': ai,
        'categoria': env.cats[ap - 1]['categoria'] if ap > 0 else None,
    })
    obs, _, term, trunc, _ = env.step((ap, ai))
    if term or trunc:
        break

df_dec = pd.DataFrame(decisoes)
print(f"  {len(df_dec)} decisões coletadas ({df_dec['data'].nunique()} dias)")

# ── Agrupar em campanhas ──────────────────────────────────────────────────

# Por dia, qual é a categoria/intensidade predominante?
INTENS_NOMES = ['nada', 'desc5%', 'desc10%', 'combo', 'liq25%']
INTENS_DESCONTO = [0, 5, 10, 10, 25]  # combo conta como 10% desconto

decisoes_por_dia = (df_dec.groupby('data')
                            .apply(lambda g: (g[g['produto_idx'] > 0]
                                                .groupby(['categoria', 'intensidade_idx'])
                                                .size().idxmax()
                                              if (g['produto_idx'] > 0).any()
                                              else (None, 0)),
                                    include_groups=False)
                            .to_dict())

# Sequência ordenada
dias_ordenados = sorted(decisoes_por_dia.keys())
campanhas = []
em_campanha = False
atual = None

for d in dias_ordenados:
    cat, intens = decisoes_por_dia[d]
    if cat is None:
        # Sem promoção neste dia
        if em_campanha and atual['dias_total'] >= args.min_dias_campanha:
            campanhas.append(atual)
        em_campanha = False
        atual = None
        continue

    # Tem promoção
    if not em_campanha:
        # Inicia nova campanha
        em_campanha = True
        atual = {
            'data_inicio': d,
            'data_fim': d,
            'categoria': cat,
            'intensidade_idx': intens,
            'dias_total': 1,
        }
    elif cat == atual['categoria'] and intens == atual['intensidade_idx']:
        # Continua campanha mesma categoria/intensidade
        atual['data_fim'] = d
        atual['dias_total'] += 1
        if atual['dias_total'] >= args.max_dias_campanha:
            campanhas.append(atual)
            em_campanha = False
            atual = None
    else:
        # Categoria/intensidade mudou — fecha anterior e abre nova
        if atual['dias_total'] >= args.min_dias_campanha:
            campanhas.append(atual)
        atual = {
            'data_inicio': d,
            'data_fim': d,
            'categoria': cat,
            'intensidade_idx': intens,
            'dias_total': 1,
        }

if em_campanha and atual and atual['dias_total'] >= args.min_dias_campanha:
    campanhas.append(atual)

print(f"  {len(campanhas)} campanhas identificadas")

# ── Enriquece campanhas com uplift estimado ───────────────────────────────

cal_dict = {ev['data']: ev for ev in env.cfg['calendario_comercial']}

for c in campanhas:
    cat_nome = c['categoria']
    cat_idx = env.nome_para_idx[cat_nome]
    cat_cfg = env.cats[cat_idx]

    # Demanda esperada no período (categoria × dias)
    demanda_dia = cat_cfg['demanda_base_dia']
    intens = c['intensidade_idx']
    intens_nome = INTENS_NOMES[intens]
    desc = INTENS_DESCONTO[intens]

    elast = abs(cat_cfg['elasticidade_promo'])
    if intens == 1:
        boost = 1 + elast * 0.05
    elif intens == 2:
        boost = 1 + elast * 0.10
    elif intens == 3:
        boost = 1.12  # combo
    elif intens == 4:
        boost = 1 + elast * 0.25
    else:
        boost = 1.0

    uplift_un = demanda_dia * c['dias_total'] * (boost - 1)
    lucro_adicional = uplift_un * cat_cfg['margem'] * (1 - desc / 100)

    # Detectar se cai em data comercial
    eventos_na_janela = []
    d_ini = date.fromisoformat(c['data_inicio'])
    d_fim = date.fromisoformat(c['data_fim'])
    for ev_data, ev in cal_dict.items():
        ev_d = date.fromisoformat(ev_data)
        pre = int(ev['janela_pre_dias'])
        pos = int(ev['janela_pos_dias'])
        janela_ini = ev_d - timedelta(days=pre)
        janela_fim = ev_d + timedelta(days=pos)
        if not (d_fim < janela_ini or d_ini > janela_fim):
            eventos_na_janela.append(ev['nome_evento'])

    c['intensidade'] = intens_nome
    c['desconto_pct'] = desc
    c['demanda_base_dia'] = round(demanda_dia, 1)
    c['uplift_unidades_estimado'] = round(uplift_un, 1)
    c['lucro_adicional_estimado_R$'] = round(lucro_adicional, 2)
    c['preco_unitario'] = round(cat_cfg['preco_venda'], 2)
    c['margem_unitaria'] = round(cat_cfg['margem'], 2)
    c['eventos_comerciais_na_janela'] = eventos_na_janela
    if cat_cfg.get('par_combo') and intens == 3:
        c['produto_complementar'] = cat_cfg['par_combo']

# ── Salva JSON ──────────────────────────────────────────────────────────────

calendario = {
    'versao': 'V3-modelo-V11',
    'modelo_base': 'DQN V11 — 18 categorias + calendário comercial BR + Dunnhumby',
    'gerado_em': str(date.today()),
    'data_inicio': DATA_HOJE.isoformat(),
    'data_fim': (DATA_HOJE + timedelta(days=HORIZONTE - 1)).isoformat(),
    'horizonte_dias': HORIZONTE,
    'n_campanhas': len(campanhas),
    'lucro_adicional_total_R$': round(sum(c['lucro_adicional_estimado_R$']
                                            for c in campanhas), 2),
    'campanhas': campanhas,
    'limitacoes': [
        'Demanda calibrada por CATEGORIA (vendas detalhadas por SKU ainda não chegaram do ERP)',
        'Validade típica heurística (esperando ERP)',
        'Elasticidade da literatura Bijmolt + ajuste Dunnhumby — validação real só virá com teste A/B',
        'Combos via heurística (esperando cupom fiscal para análise de cesta)',
    ],
}

with open(RESULTS / 'calendario_v3.json', 'w', encoding='utf-8') as f:
    json.dump(calendario, f, indent=2, ensure_ascii=False, default=str)

# ── Markdown legível ────────────────────────────────────────────────────────

DIAS_SEM = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']

md = [
    f"# Calendário de Promoções V3 — Modelo V11",
    "",
    f"**Gerado em:** {date.today().strftime('%d/%m/%Y')}",
    f"**Horizonte:** {HORIZONTE} dias ({DATA_HOJE.strftime('%d/%m/%Y')} a "
    f"{(DATA_HOJE + timedelta(days=HORIZONTE-1)).strftime('%d/%m/%Y')})",
    f"**Modelo:** DQN V11 com 18 categorias + calendário comercial BR",
    "",
    "## Resumo",
    "",
    f"- **{len(campanhas)} campanhas** recomendadas",
    f"- **Lucro adicional estimado total:** R$ {calendario['lucro_adicional_total_R$']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
    "",
]

if not campanhas:
    md += [
        "## ⚠ Nenhuma campanha recomendada",
        "",
        "O modelo V11 não identificou janela de promoção lucrativa no horizonte.",
        "Provavelmente o modelo está subtreinado. Para uso real, treinar com:",
        "",
        "```",
        "python treinar_v11.py --episodios 200 --seeds 3 --max_steps_per_ep 1095",
        "```",
        "",
    ]
else:
    md.append("## Campanhas")
    md.append("")
    for i, c in enumerate(campanhas, 1):
        d_ini = date.fromisoformat(c['data_inicio'])
        d_fim = date.fromisoformat(c['data_fim'])
        md += [
            f"### Campanha {i}: {d_ini.strftime('%d/%m')} a {d_fim.strftime('%d/%m')} ({c['dias_total']} dias)",
            "",
            f"- **Categoria:** {c['categoria']}",
            f"- **Tipo:** {c['intensidade']}  (desconto {c['desconto_pct']}%)",
            f"- **Preço médio da categoria:** R$ {c['preco_unitario']}",
            f"- **Margem unitária:** R$ {c['margem_unitaria']}",
            f"- **Uplift estimado:** +{c['uplift_unidades_estimado']:.1f} unidades",
            f"- **Lucro adicional estimado:** R$ {c['lucro_adicional_estimado_R$']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
        ]
        if c.get('produto_complementar'):
            md.append(f"- **Combo com:** {c['produto_complementar']}")
        if c.get('eventos_comerciais_na_janela'):
            md.append(f"- **🎯 Coincide com:** {', '.join(c['eventos_comerciais_na_janela'])}")
        md.append("")

md += [
    "---",
    "",
    "## Limitações desta V3",
    "",
    "1. **Demanda calibrada por categoria, não por SKU** — esperando dados detalhados do ERP",
    "2. **Validade típica heurística** — esperando dado do posto",
    "3. **Elasticidade da literatura** — validação real só com teste A/B",
    "4. **Combos via heurística** — esperando cupom fiscal para Apriori",
    "",
    "## Para refinar",
    "",
    "Quando o ERP exportar vendas detalhadas por SKU e cupom fiscal, rodar:",
    "",
    "```powershell",
    "python calibrar_v2.py        # re-calibra com dados completos",
    "python treinar_v11.py        # re-treina V11",
    "python validar_v11.py        # valida metricas",
    "python gerar_calendario_v3.py # gera novo calendario",
    "```",
    "",
    "---",
    "",
    f"*V3 gerada em {date.today().isoformat()} pelo modelo DQN V11 treinado.*",
]

(RESULTS / 'calendario_v3.md').write_text('\n'.join(md), encoding='utf-8')

# ── Resumo ──────────────────────────────────────────────────────────────────

print()
print("="*70)
print("CALENDÁRIO V3 GERADO")
print("="*70)
print(f"  Período: {DATA_HOJE.strftime('%d/%m/%Y')} a "
      f"{(DATA_HOJE + timedelta(days=HORIZONTE-1)).strftime('%d/%m/%Y')}")
print(f"  Campanhas: {len(campanhas)}")
print(f"  Lucro adicional estimado: R$ {calendario['lucro_adicional_total_R$']:,.2f}")
print()
if campanhas:
    print("Top 5 campanhas:")
    top = sorted(campanhas, key=lambda c: -c['lucro_adicional_estimado_R$'])[:5]
    for c in top:
        d_ini = date.fromisoformat(c['data_inicio']).strftime('%d/%m')
        d_fim = date.fromisoformat(c['data_fim']).strftime('%d/%m')
        print(f"  {d_ini}-{d_fim} ({c['dias_total']}d)  {c['categoria']:<25s} "
              f"{c['intensidade']:<8s}  R$ {c['lucro_adicional_estimado_R$']:>8.2f}")
else:
    print("  ⚠ Modelo não recomendou nenhuma campanha — provável subtreino")
print()
print(f"✓ Saídas:")
print(f"  results/v11/calendario_v3.json")
print(f"  results/v11/calendario_v3.md")
