# V20 — RL de verdade

Versão isolada onde a lógica da V19 vira **modelo treinável por RL**.

## Por que existe

Resposta ao briefing do Vinicius (15/05/2026):

> "Muita coisa que fizemos na V19 ainda está mais parecida com regra de negócio
> e melhoria de saída do que com um modelo realmente treinado. Como RL é a base
> do nosso projeto, quero que essa lógica seja aplicada dentro de um modelo de
> Reinforcement Learning de verdade."

V20 transforma:
- ⛔ Regras hard (`COMBOS_INVALIDOS_PDV`) → penalidades na recompensa
- ⛔ Score composto com pesos fixos → política aprendida pelo DQN
- ⛔ Thresholds calibrados na mão → Q-values aprendidos
- ⛔ DecisionAgent if/else → cabeça RL que decide aprovação

## Arquitetura

```
                   Estado (83 features)
                            │
                  Encoder compartilhado (128)
                  │           │           │
        Cabeça INTENS    Cabeça COMP   Cabeça ALVO
        (Agente Desc)   (Agente Combo) (Agente Marg)
         6 ações         21 ações       2 ações
```

**Branching DQN** (Tavakoli 2018): 3 cabeças independentes, encoder comum.
Cada cabeça é um "sub-agente RL" — observação compartilhada, decisões coordenadas.

## Função de recompensa (codifica as regras V19 como aprendizado)

```python
r = lucro_real_dia × dias_campanha          # núcleo econômico

# BONUS (estratégia ideal)
+ K_COMBO_EM_ALTA       (150)  if em alta + combo
+ K_DESC_COMPLEMENTAR    (50)  if combo com desconto no complementar
+ K_DEFENSIVO_VENCIMENTO (280) if liq% em validade <30%
+ K_BONUS_EVENTO_MATCH  (100)  if evento + categoria afetada
+ K_BONUS_HARMONIA_FORTE (80)  if combo harmonia >= 2.0

# PENALIDADES (regras codificadas como aprendizado)
- K_DESC_EM_ALTA_NATURAL (200) if desc direto em alta natural
- K_COMBO_PDV_INVALIDO   (300) if (cat,par) em COMBOS_INVALIDOS_PDV  ← era hard rule
- K_CANIBALIZACAO_ALTA   (8/%) por % canib > 40%
- K_REPETICAO_CATEGORIA   (50) se categoria já promovida em últimos 7d
- K_UPLIFT_ABAIXO_BE      (80) se uplift < breakeven
- K_HARMONIA_FRACA        (40) se combo harmonia < 1.0
```

## Resultados do treino (400 ep × 1 seed)

```
Reward inicial (eps 1-10):    -746.3
Reward final (eps 370-400):  +3062.9  (4× melhora)
Best avg30:                  +3440.9

Combos PDV-inválidos:
  Eps 0-50:   0.32/ep      (agente aleatório)
  Eps 250-300: 0.12/ep
  Eps 350-400:   0/ep      ✓ agente aprendeu perfeitamente

Distribuição de ações aprendida:
  combo:   41.8%  ← descoberto como dominante
  nada:    25.0%  ← aprendeu quando NÃO promover
  desc10%: 11.8%
  desc7%:   7.2%
  desc5%:   7.2%
  desc3%:   7.0%
```

## Validação nos 9 cenários do briefing — **7/9 corretos**

| # | Cenário | Decisão V20 | OK? |
|---|---|---|:---:|
| 1 | Gelo+Cerveja FDS quente | combo + energético | ⚠️ par errado |
| 2 | Gelo+Destilados Réveillon | combo + **doce** (NÃO destilados) | ✅ |
| 3 | Chocolate+Vinho Mães | nada | ❌ |
| 4 | Isotônico dia comum | nada | ✅ |
| 5 | Chocolate impulso baixa | nada | ✅ |
| 6 | Cerveja sex alta natural | combo (não desc direto!) | ✅ |
| 7 | Sorvete parado verão | combo+energético | ✅ |
| 8 | Sorvete vencimento | combo desc10% + defensivo +280 | ✅ |
| 9 | Café+cerveja | nada | ✅ |

## Comparação V19.1 vs V20

| Métrica | V19.1 (regras) | V20 (RL) | Δ |
|---|---:|---:|---|
| Lucro anual estimado | R$ 8.111 | R$ 10.511 | **+29.6%** |
| Campanhas | 24 (dadas) | 237 (decididas) | RL escolhe |
| Combos | 20 | 196 | maior amplitude |
| Categorias distintas | 6 | 16 | +267% |
| Combos PDV-inválidos | 0 (hard rule) | 0 (aprendido!) | igual mas via RL |

**Diferença filosófica**: V19.1 tem `if (cat,par) in INVALIDOS: rejeita`. V20
recebe penalidade -300 quando escolhe, e o DQN APRENDE sozinho a evitar.
Generaliza para combos não-listados similares.

## Como rodar

```powershell
cd rl_promocoes_v20_teste

# 1. Smoke test do ambiente
..\.venv\Scripts\python.exe env_rl_promocoes.py

# 2. Treinar (~2 min com 400 episódios em CPU)
..\.venv\Scripts\python.exe treinar_v20.py --seeds 1 --episodios 400

# 3. Validar nos 9 cenários
..\.venv\Scripts\python.exe validar_v20.py

# 4. Gerar calendário operacional (365 dias)
..\.venv\Scripts\python.exe gerar_calendario_v20.py

# 5. Comparar com V19.1
..\.venv\Scripts\python.exe comparar_v19_v20.py
```

Outputs em `../results/v20/`:
- `calendario_v20.json` — 237 campanhas decididas pelo agente
- `validacao_v20.json` — Q-values dos 9 cenários
- `comparacao_v19_v20.html` — dashboard side-by-side

## Arquivos

```
rl_promocoes_v20_teste/
├── README.md                  ← este arquivo
├── ANALISE_V19.md             ← mapeamento V19 → V20
├── env_rl_promocoes.py        ← Gymnasium env (state 83, action MultiDiscrete[6,21,2])
├── branching_dqn.py           ← Branching DQN (3 cabeças, encoder compartilhado)
├── treinar_v20.py             ← script de treino com logs CSV
├── validar_v20.py             ← teste em 9 cenários
├── gerar_calendario_v20.py    ← rollout determinístico 365 dias
├── comparar_v19_v20.py        ← side-by-side V19.1 vs V20
├── logs/
│   ├── training_seed_0.csv    ← log episódio-a-episódio
│   └── treino_completo.log
└── models/
    └── best_seed_0.pt         ← melhor checkpoint (best avg30=3441)
```

## Próximos passos (para V20.1+)

1. **Ensemble multi-seed**: rodar 3-5 seeds e Q-mean para reduzir variância.
2. **Corrigir viés "energético"**: aumentar bonus de harmonia ou usar action masking
   na cabeça COMPLEMENTAR (apenas pares com harmonia ≥ 1.5).
3. **Multi-agent puro**: separar otimizadores das 3 cabeças → "MARL" de verdade.
4. **Estado expandido**: incluir saturação por categoria, ticket médio histórico.
5. **Test A/B in loco**: validar política aprendida em campo (depende do posto).

## Lições

1. **Reward shaping > hard rule**: o agente aprendeu a evitar combos PDV-inválidos
   apenas com penalidade na recompensa. Generaliza para casos não-listados.
2. **Single-seed converge mas tem viés**: a cabeça COMPLEMENTAR caiu num ótimo
   local. Ensemble resolveria.
3. **400 episódios são suficientes para o problema**: reward subiu 4×.
4. **"Não promover" é estratégia válida**: 25% das ações são nada — agente
   aprendeu que nem todo turno merece promoção.
