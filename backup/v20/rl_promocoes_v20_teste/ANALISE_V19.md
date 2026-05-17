# Análise V19 — o que vira RL na V20

Mapeamento honesto do que a V19 (V19/V19.1/V19.2) faz, separando:
- ✅ **Lógica forte** que deve ser preservada (vira parte da função reward ou da
  função de transição do ambiente)
- ⚠️ **Heurística** que pode virar decisão aprendida (vira ação)
- 🔴 **Regra hard** que deve virar penalidade na reward (não bloqueio)
- 📊 **Cálculo puro** que vira input do estado ou do reward

## Componente por componente

### DemandAgent (v19_1/demand_agent.py)

| Linha | O que faz | Classificação | Destino V20 |
|---|---|---|---|
| `TIPO_PRODUTO` map | Classifica em puxador/impulso/commodity | 📊 cálculo | **Estado**: `tipo_produto` one-hot |
| `ELASTICIDADE` por tipo | -0.3 a -1.5 | 📊 calibração | **Reward**: usa no cálculo de demanda promo |
| `CAP_UPLIFT` por intensidade | 25%/50%/90%/120% | ⚠️ heurística | **Reward**: limita ganho de uplift (anti-otimismo) |
| `_sazonal_dow` (V19.1) | FATOR_DIA × demanda base | 📊 cálculo | **Estado/Reward**: modula demanda base contextual |
| `_sazonal_mes` (V19.1) | FATOR_MES × demanda base | 📊 cálculo | **Estado**: contexto temporal |
| `_sazonal_clima` (V19.1) | Temperatura real × tipo produto | 📊 cálculo | **Estado**: temperatura norm |
| `_sazonal_pre_feriado` (V19.1) | Ponte sex/sáb antes feriado seg | 📊 cálculo | **Estado**: dias até próximo feriado |
| `_sazonal_vespera_presente` (V19.1) | Semana pré-Mães puxa chocolate | 📊 cálculo | **Estado**: dias até próx evento |
| `_boost_evento_calibrado` (V19.1) | uplift_prior do calendário comercial | 📊 cálculo | **Reward**: amplifica reward em evento |
| `_canibalizacao` | 20% base + desc × 1.0 | ⚠️ heurística | **Reward**: penaliza alta canib |
| `_classificar_qualidade` | BOA/MÉDIA/RUIM | 🔴 hard rule | **REMOVER** — agente aprende isso |

### RevenueAgent (v19_1/revenue_agent.py)

| Componente | Classificação | Destino V20 |
|---|---|---|
| `lucro_uplift_dia` | 📊 cálculo essencial | **Reward**: termo principal |
| `lucro_canibalizacao_dia` (negativo) | 📊 cálculo essencial | **Reward**: subtrai |
| `lucro_halo_dia` (cross-sell) | 📊 cálculo essencial | **Reward**: adiciona |
| `lucro_defensivo_dia` (liquidação) | 📊 cálculo essencial | **Reward**: bonus em validade<30% |
| `custo_operacional` | 📊 cálculo | **Reward**: subtrai (custo fixo) |
| `breakeven_uplift_pct` | 📊 derivado | **Estado**: pode entrar como feature |

### DecisionAgent (v19_1/decision_agent.py) — TODO ELE VIRA RL

| Componente | Classificação | Destino V20 |
|---|---|---|
| Score composto (35% ROI + 25% lucro + 20% BE + 15% qual + 5% risco) | 🔴 hard rule | **REMOVER** — RL aprende ponderação |
| Thresholds ROI_MIN/PRIORITARIO | 🔴 hard rule | **REMOVER** — RL decide aprovar |
| Decisão APROVADA/CONDICIONAL/REJEITADA | 🔴 hard rule | **AÇÃO** do RL |

### COMBOS_INVALIDOS_PDV (V19.2)

| Linha | Classificação | Destino V20 |
|---|---|---|
| `frozenset(['gelo', 'destilados'])` etc | 🔴 hard rule | **Reward**: penalidade -300 quando agente escolhe |

A regra continua viva — mas como **shaping de reward**, não como bloqueio.
O agente APRENDE a evitar (e generaliza para combos não-listados similares).

## O que V19 faz bem (preservar conceitualmente)

1. ✅ Modela canibalização (Walmart 30-50%)
2. ✅ Modela halo (cross-sell 5-10% das visitas)
3. ✅ Modela lucro defensivo (liquidação evita perda total)
4. ✅ Modela elasticidade por tipo de produto
5. ✅ Usa `uplift_prior` real do calendário comercial
6. ✅ Separa sazonalidade do uplift (V19.1)
7. ✅ Reconhece compra individual vs cesta evento (V19.2)

## O que V19 NÃO é RL

1. ❌ DecisionAgent é puro if/else
2. ❌ Score composto tem pesos fixos não-aprendidos
3. ❌ Combos PDV-inválidos bloqueados por hard rule
4. ❌ Thresholds calibrados na mão
5. ❌ Não há treino, não há recompensa, não há iteração
6. ❌ Não generaliza para situações fora do hardcoded

## Mapeamento V20

### Estado (~80 features)

```
TEMPORAL (35):
  data: dia_semana (7), mês (12), turno (3)
  dia_do_mês_norm (1)
  evento_proximo: tipo (4) + dias_até (1) + intensidade (4)
  pre_feriado_flag (1), pos_feriado_flag (1)

PRODUTO PRINCIPAL (15):
  categoria (one-hot 20)  — N categorias
  tipo_produto (one-hot 6)
  demanda_base_anual_norm (1)
  demanda_contextual_norm (1) ← com sazonalidade aplicada
  margem_pct_norm (1), preço_norm (1)
  estoque_norm (1), validade_pct (1), giro_baixo_flag (1)
  promovido_há_X_dias (vetor 7)
  
CONTEXTO PROMO (10):
  fator_sazonal_atual (1)
  esta_em_alta_demanda_flag (1)
  esta_em_baixa_demanda_flag (1)
  uplift_prior_evento_proximo (1)
  temperatura_norm (1)
  
HISTÓRICO (10):
  promoçoes_últimos_7d (1)
  receita_últimos_7d_norm (1)
  campanhas_repetidas_flag (1)
```

### Ações (MultiDiscrete)

```
Cabeça INTENSIDADE (Agente de Desconto, 6 opções):
  0 = não promover
  1 = desc 3%
  2 = desc 5%
  3 = desc 7%
  4 = desc 10%
  5 = combo (desc 5% no complementar)

Cabeça COMPLEMENTAR (Agente de Combo, N+1 opções):
  0 = nenhum
  1..N = índice da categoria do par

Cabeça ALVO (Agente de Margem, 2 opções):
  0 = desconto no principal
  1 = desconto no complementar
```

Total Q-values: 6 × 21 × 2 = 252 — Branching DQN com 3 cabeças independentes.

### Reward (a soma do bom senso V19, codificada)

```python
r = 0
# Termos POSITIVOS (Revenue calculado pelo simulador)
r += lucro_uplift_dia × dias              # ganho de vendas extras
r += lucro_halo_dia × dias                # cross-sell
r += lucro_defensivo_dia × dias           # liq evita perda

# Termos NEGATIVOS
r -= |lucro_canibalizacao_dia| × dias     # perda de margem em quem já compraria
r -= custo_operacional                    # cartaz + tempo

# REWARD SHAPING — codifica regras como aprendizado
r += K_combo_alta × 1[combo em alta demanda]
r += K_complementar × 1[desconto no complementar, não principal]
r += K_defensivo × 1[liq25% em validade <30%]

r -= K_desc_alta × 1[desc direto em produto em alta natural]
r -= K_combo_pdv_invalido × 1[(cat,par) em pares inválidos PDV]
r -= K_canib_alta × max(0, canib_pct - 0.40)
r -= K_repetir × 1[mesma categoria promovida nos últimos 7 dias]
r -= K_baixo_uplift × 1[uplift < breakeven]
```

Constantes a calibrar (ordem de grandeza):
- `K_combo_alta = +150`
- `K_complementar = +50`
- `K_defensivo = +120`
- `K_desc_alta = -200`
- `K_combo_pdv_invalido = -300` ← **substitui o hard rule do V19.2**
- `K_canib_alta = -300` (por unidade de canib > 40%)
- `K_repetir = -50`
- `K_baixo_uplift = -100`

## Decisão arquitetural: single-agent com 3 sub-cabeças = MARL conceitual

Justificativa para começar com single-agent Branching DQN:
1. Briefing permite ("você pode implementar primeiro versão com um único agente")
2. Branching DQN coordena decisões compostas (Tavakoli 2018) — cada cabeça pode
   ser interpretada como "agente especializado" com observação compartilhada
3. MARL puro (3 agentes independentes treinando paralelo) com poucas amostras
   tende a divergir
4. Arquitetura preparada para escalar: cada cabeça pode virar agente próprio se
   necessário (basta separar otimizadores)

Mapeamento "agentes V20":
- Cabeça **INTENSIDADE** = Agente de Desconto
- Cabeça **COMPLEMENTAR** = Agente de Combo
- Cabeça **ALVO** = Agente de Margem
- Agente de Demanda = **função de transição do ambiente** (estimativa)
- Agente de Priorização = **emerge da política** (Q-values mostram qual produto/intensidade priorizar)

## Episódio

- 30 turnos (1 turno = 1 decisão de campanha, a cada ~1 dia)
- A cada turno, o ambiente apresenta o estado de **um produto candidato**
- Agente decide ação MultiDiscrete
- Recompensa imediata calculada
- Estado evolui: estoque consumido, validade decresce, histórico atualiza
- 30 turnos depois, episódio termina

Múltiplos episódios reset com diferentes datas iniciais (cobre sazonalidades).

## Critério de sucesso

Política do agente deve mostrar (após treino):
1. Q-value de "desc direto em alta natural" < Q-value de "combo em alta natural" (aprendeu regra do dono)
2. Q-value de combo `(gelo, destilados)` < Q-value de combo `(gelo, cerveja)` (aprendeu regra PDV)
3. Q-value de "promover na semana pré-Mães em chocolate" alto (aprendeu calendário)
4. Q-value de "liq25%" alto quando validade < 30% (aprendeu defensivo)
5. Comparável ou superior ao calendário V19.1 em lucro real estimado

---

*Documento finalizado: análise sistemática de V19 → V20. Tudo que era hard rule
vira reward shaping. Tudo que era cálculo bom vira função de transição.*
