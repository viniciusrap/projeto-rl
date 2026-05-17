# Otimização de Promoções com Reinforcement Learning — Auto Posto Parque Viana

**Disciplina:** Reinforcement Learning — Insper
**Professor:** Fabrício Barth
**Integrantes:** Vinicius Rocha Pereira · Luigi Zema Matizonkas
**Empresa parceira:** Auto Posto Parque Viana Ltda — Barueri/SP

**Vídeo de apresentação:** _(link a inserir antes da entrega — YouTube/Vimeo)_
**Relatório final:** [`relatorio/latex/main.pdf`](relatorio/latex/main.pdf)

---

## 1. Problema e objetivo

A loja de conveniência do posto decide promoções por intuição. Isso custa
dinheiro: desconto em produto que já vende sozinho destrói margem; produto
sem giro vence e vira prejuízo; oportunidades sazonais (Copa, Dia dos
Namorados, fim de semana) são perdidas.

A decisão é **sequencial e sob incerteza** (o estado da loja evolui, cada
decisão afeta o próximo estado, há trade-off entre lucro imediato e
prevenção de vencimento) — exatamente a estrutura de um **MDP**. Por isso
usamos **Reinforcement Learning**: o agente aprende, por tentativa e erro
em milhares de dias simulados, **quando** e **o que** promover, e entrega
um **calendário operacional anual** de campanhas.

> Modelo final: **V21 PDV-native** (`rl_promocoes_v21_pdv_native/`).
> Calibrado com inventário **real** do posto (2022–2026) + conhecimento de
> domínio de balcão de conveniência — **sem dados de supermercado**
> (Instacart induzia combos inválidos, ex.: saco de gelo 5 kg + whisky).

---

## 2. Organização do repositório

```
projeto-rl/
├── README.md                         ← este arquivo
├── requirements.txt                  ← dependências Python
├── relatorio/latex/                  ← relatório final (LaTeX + main.pdf)
│
├── rl_promocoes_v21_pdv_native/      ← MODELO FINAL (tudo que importa)
│   ├── env_rl_promocoes_v21.py        · ambiente Gymnasium (MDP completo)
│   ├── branching_dqn.py               · Branching Double DQN (3 cabeças)
│   ├── processar_inventario.py        · inventário real → CSV
│   ├── construir_calibracao_pdv.py    · CSV + domínio → calibração JSON
│   ├── treinar_v21.py / iterar_v21.py · treino
│   ├── gerar_calendario_v21.py        · rollout → calendário + HTML
│   ├── comparar_baselines_v21.py      · agente vs não-promover/aleatório/combo
│   ├── gerar_graficos_treino.py       · curvas de aprendizado
│   ├── gerar_dashboard_v21.py         · dashboard operacional
│   ├── data_sintetica/                · calibração versionada (roda sem dados confidenciais)
│   ├── models/v21_final.pt            · modelo treinado (1 seed × 5000 ep)
│   └── logs/                          · logs de treino por iteração
│
├── results/v21/                      ← saídas (calendário, gráficos, dashboards)
├── data/                             ← dados crus do posto (confidenciais, gitignored)
└── backup/                           ← versões históricas V10–V20 (contexto da evolução)
```

`data/` contém planilhas confidenciais do posto e **não vai para o
GitHub**. O pipeline roda mesmo assim: a calibração derivada
(`data_sintetica/calibracao_v21_pdv.json`) está versionada e é o que o
ambiente carrega. Só `processar_inventario.py` precisa do `.xlsx`
confidencial — os demais passos são reproduzíveis.

---

## 3. O ambiente (MDP)

Ambiente Gymnasium próprio (`EnvRLPromocoes`), 23 categorias de
conveniência derivadas do inventário real.

- **Estado** — vetor contínuo de **89 features**: turno, dia da semana,
  mês, proximidade/tipo do próximo evento comercial, temperatura, e — por
  categoria — estoque relativo, validade restante, flag de demanda fraca e
  histórico de promoção recente.
- **Ação** — `MultiDiscrete([6, 24, 2])`, três decisões por turno:
  | Cabeça | Papel | Opções |
  |---|---|---|
  | Intensidade | Agente de Desconto | nada · 3% · 5% · 7% · 10% · combo |
  | Complementar | Agente de Combo | nenhum + 23 categorias |
  | Alvo | Agente de Margem | desconto no principal ou no par |
- **Transição** — estocástica: demanda promocional sobre a demanda
  contextual (sazonalidade), com canibalização, *halo* (venda casada) e
  ganho defensivo (liquidar item perto de vencer). Demanda `~ Poisson(λ)`.
- **Recompensa** — lucro econômico real + ~38 termos de *reward shaping*
  de domínio (harmonia de combo válida no posto, acerto de evento
  comercial, prevenção de vencimento; penaliza desconto em produto em alta
  natural e combos inválidos para posto). Cigarros são não-promovíveis
  (Lei 9.294/96); água é commodity inelástica.

Detalhes formais no relatório (`relatorio/latex/main.pdf`).

---

## 4. O método

**Branching Double DQN** (Tavakoli et al. 2018 sobre DQN/Double DQN):
*encoder* compartilhado `89→128→128` + **3 cabeças independentes**
`128→64→n` (Desconto, Combo, Margem). Implementação **própria** em
PyTorch, sem biblioteca de RL. Treino: Double DQN, replay 50k,
ε-greedy → 0,10, HuberLoss, *target network*.

Três técnicas de domínio aceleram a convergência **sem dar a resposta
pronta**: *action masking* (combo com harmonia < 1,4 bloqueado),
pré-treino supervisionado da cabeça Combo com a matriz de afinidade, e
*curriculum* (cerveja aparece mais em sex/sáb para a "Esquenta de Sexta"
emergir via recompensa).

---

## 5. Como rodar

### Setup

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
cd rl_promocoes_v21_pdv_native
```

### Reproduzir o calendário com o modelo treinado (rápido, ~1 min)

O modelo final (`models/v21_final.pt`) e a calibração já estão no repo:

```powershell
python gerar_calendario_v21.py        # → results/v21/calendario_v21.json + calendario_visual.html
python comparar_baselines_v21.py      # → results/v21/comparacao_baselines.json
```

### Pipeline completo (re-treinar do zero, ~15–40 min em CPU)

```powershell
# 1. (opcional, precisa do .xlsx confidencial) inventário real → CSV
python processar_inventario.py
# 2. calibração PDV-native (usa o CSV já versionado)
python construir_calibracao_pdv.py
# 3. treino — modelo final = 1 seed × 5000 episódios
python iterar_v21.py --iter_nome final_5000 --episodios 5000 --n_seeds 1
#    (copie models/iter_final_5000_seed_0.pt → models/v21_final.pt)
# 4. gráficos de aprendizado
python gerar_graficos_treino.py --iter_nome final_5000
# 5. calendário + comparação
python gerar_calendario_v21.py
python comparar_baselines_v21.py
```

---

## 6. Resultados (resumo)

Modelo final — **1 seed × 5000 episódios**. Curva converge (volatilidade
local −88%, CV 3,52 → 0,44). Validação determinística (365 dias): 11/12
cenários canônicos corretos, 208 campanhas, **0 combos inválidos para
posto**, R$ 16.918 de lucro anual estimado. Padrões emergentes coerentes:
"Esquenta de Sexta" (cerveja + salgadinho) como combo nº 1, café + padaria
de manhã, captura de Dia dos Namorados e jogos da Copa.

**Comparação com baselines** (mesmo ambiente, 365 dias, 8 seeds):

| Política | Lucro/ano (R$) | Campanhas |
|---|---:|---:|
| Não-promover | 815 ± 124 | 0 |
| Aleatório | −155 ± 1.559 | 304 |
| Sempre-combo | 22.235 ± 5.735 | 270 |
| **Agente V21 (RL)** | **21.640 ± 4.728** | **176** |

O agente supera folgadamente "não-promover" e "aleatório" (que destrói
valor). Empata em lucro bruto com a heurística trivial "sempre-combo",
mas com **muito menos campanhas** (176 vs 270): o valor é a parcimônia, a
validade (0 combos inválidos) e o *timing* — não o R$ bruto.

---

## 7. Evolução do modelo (V10 → V21)

Cada versão nasceu de uma falha **medida** da anterior — o histórico é
parte da contribuição metodológica:

| Versão | Mudança-chave | Aprendizado |
|---|---|---|
| V10 | 6 produtos, reward só lucro | Política colapsou numa ação só; o gargalo era a modelagem do MDP, não o algoritmo |
| V11 | Catálogo + datas comerciais no estado | F1 por evento correlaciona com cobertura do catálogo (r≈0,75): *feature engineering* pesa tanto quanto o algoritmo |
| V19 | Economia explícita (canibalização, halo, defensivo) | Recompensa tem de modelar o ecossistema econômico, não só "demanda × desconto" |
| V20 | Regras viram *reward shaping* + pré-treino da cabeça Combo | *Reward shaping* puro cai em ótimo local; domínio como inicialização destrava |
| **V21** | Rejeitar dados de supermercado | Afinidade de mercearia ≠ afinidade de PDV; calibração nativa de balcão |

Histórico detalhado das ~20 iterações em `backup/` (versões anteriores).

---

## 8. Limitações honestas

1. **Off-policy evaluation**: o simulador foi calibrado em 6 anos de
   vendas **sem promoção real** e a elasticidade vem da literatura, não
   medida no posto. Os valores em R$ são **estimativas de simulação**, não
   medições. A comparação com baselines é **relativa**.
2. **Decisões induzidas**: o agente é fortemente guiado por *reward
   shaping* + pré-treino + *curriculum*. O que genuinamente emerge é o
   *timing*, a combinação e a quantidade — não a noção de harmonia em si.
3. **1 seed**: o modelo final é single-seed (escolha justificada: o
   ensemble de 5 seeds "lavava" decisões marginais). Daí o desvio alto na
   comparação.
4. **Próximo passo essencial**: teste A/B in loco (semanas com/sem o
   agente) para calibrar elasticidade real e fechar o ciclo de validação.

---

## 9. Trabalho próprio vs. externo

| Componente | Origem |
|---|---|
| `EnvRLPromocoes` (Gymnasium) | Implementação própria |
| `BranchingDQN` / Double DQN | Implementação própria (Tavakoli 2018; van Hasselt 2016) |
| Matriz de harmonia de combo | Conhecimento de domínio de PDV (sem Instacart) |
| Calibração temporal/custos | Inventário real do posto 2022–2026 |
| Calendário comercial BR | `holidays` (pip) + curadoria manual |
| *Reward shaping* | Fundamentação teórica: Ng, Harada & Russell (1999) |

**Referências**: Mnih et al. (2013); van Hasselt, Guez & Silver (2016);
Tavakoli, Pardo & Kormushev (2018); Ng, Harada & Russell (1999); Bijmolt,
van Heerde & Pieters (2005). Lista completa no relatório.
