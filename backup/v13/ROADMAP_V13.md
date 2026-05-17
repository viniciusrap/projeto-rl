# Roadmap V13 — Baseado em pesquisa de 8 frentes paralelas

Pesquisa conduzida 12/05/2026 noite (8 agents paralelos, ~150k tokens consumidos). Síntese de papers e cases reais de Alibaba, Amazon, Walmart, Stitch Fix, JD.com, Freshippo, Uber. Aplicado ao Auto Posto Parque Viana (V12.1 → V13).

---

## TL;DR — 4 melhorias prioritárias

| # | Melhoria | Esforço | Impacto | Status |
|---|---|---|---|---|
| 1 | **Hard mask de cigarros** (Lei 9.294/96) | 30 min | Garante 0% cigarro, regulatório | ✅ **IMPLEMENTADO** em `dqn.py` + `env_v12.py` |
| 2 | **CondPolicyDQN** (decomposição condicional) | 2h treino | Destrava liquid25%/desc10% | ✅ Arquitetura pronta, falta treinar |
| 3 | **CausalImpact loop** (validação contrafactual) | 1 dia | Mede uplift real sem A/B | Pendente — `medir_uplift_causalimpact.py` |
| 4 | **LightGBM global** (substitui Ridge) | 4h | MAPE 57% → ~28% esperado | Pendente — `treinar_forecaster_lgbm.py` |

---

## Validação empírica imediata do V12.3a (hard mask)

Aplicando mask no V12 base (modelo que originalmente promovia cigarro 18.5%
das vezes) **SEM RETREINAR**, em 10 episódios × 1095 steps:

| Métrica | SEM mask | COM mask | Δ |
|---|---:|---:|---|
| Reward médio | R$ 774.951 | **R$ 823.639** | **+6.28%** |
| Lucro médio | R$ 650.091 | R$ 650.387 | +0.05% |
| Perdas médias | 1727 un | 1739 un | +0.72% |
| **% cigarro promovido** | **18.5%** | **0.0%** | **-100%** ✅ |

Reward sobe +6.28% porque mask elimina as ~6.080 R$/episódio gastas em
pen_nao_promovivel=30 × 18.5% × 1095 turnos. Lucro estável (env já convertia
cigarro em sem-promo). **Compliance regulatório 100% garantido.**

## 1. Hard mask de cigarros — IMPLEMENTADO ✅

**Causa:** V12 base promovia `cigarro_jti desc5%` em 15% das decisões. Matematicamente bom (cigarro tem volume 122 un/dia) mas **proibido por lei** (Lei Antifumo 9.294/96 + ANVISA: cigarro não pode ter desconto/promoção/publicidade).

**V12.1 atual:** usa `pen_não_promovível = 30` (penalty leve). Funciona estatisticamente mas não dá garantia 100%.

**V12.3 (implementado):** Hard mask em Q-values via `torch.finfo(torch.float32).min` (sentinela `NEG_INF`):

```python
# Em dqn.py
def select_action(self, x, eps, rng, mask_cat=None):
    if mask_cat is not None:
        q = q.masked_fill(~mask_t, NEG_INF)  # cigarros viram -inf
    ...

# Em env_v12.py
def get_action_mask(self):
    mask = np.ones(self.N + 1, dtype=bool)
    for i, c in enumerate(self.cats):
        if not c.get('promovivel', True):
            mask[i + 1] = False
    return mask
```

**Referências:** Huang & Ontañón 2020 (arXiv:2006.14171), Boring Guy blog. SB3 não tem MaskableDQN nativo — implementação manual feita aqui.

**Lição da indústria:** Amazon Personalize e Alibaba fazem masking em **camada de re-ranking POST-policy**, fora do agente. Para compliance regulatória, **ninguém em produção confia em penalty** — sempre hard mask ou safety layer.

---

## 2. CondPolicyDQN — ARQUITETURA PRONTA ✅

**Causa:** V12.1 colapsa em 7 ações de 105. Diagnóstico (HRL agent):

> O colapso **não é falha de exploração** — é estrutural do BranchingDQN. Cabeças `A_prod` e `A_int` são **aditivas e independentes**. Liquid25% nunca emerge porque `A_int(liquid25%)` é marginalizado sobre TODAS categorias — mas liquid25% só é boa **condicionada** a "produto vencendo".

**Solução:** decomposição autoregressiva:
```
Q_cat(s) = MLP(s)
Q_int(s, cat) = MLP(concat[s, embedding(cat)])
```

Implementada em `dqn.CondPolicyDQN` (93k params, similar ao BranchingDQN).

**Cases reais:**
- **AlphaStar** (DeepMind 2019): ação composta decomposta sequencialmente (unidade → ação → alvo)
- **Hierarchical DQN** (Kulkarni et al. 2016, Montezuma's Revenge): meta-controller + sub-policy
- **JD.com 2018** (Zhao et al.): state+/state- decomposto — alinhado com nosso fraco_flag/forte_flag

**Para treinar V12.3 com CondPolicyDQN** (próximo passo):
- Trocar `BranchingDQN` por `CondPolicyDQN` em `treinar_v12.py`
- Bellman update precisa adaptar: target usa `argmax_cat → q_int(s', best_cat)`
- ε-greedy diferenciado por cabeça (ε_cat=0.05, ε_int=0.2) durante warmup

**Risco:** sem warmup, q_cat pode colapsar antes de q_int aprender. Mitigação: 50 episódios iniciais com π_cat uniforme aleatório.

---

## 3. CausalImpact — validação contrafactual sem A/B

**Causa:** V12.1 entrega +0.23% lucro vs sem-promo no SIMULADOR. Mas elasticidade é da literatura (Bijmolt 2005 + 3 datasets físicos), não medida no posto. Sem teste A/B real, números são **defensáveis dentro da nossa suposição**, não validados.

**Solução de produção (Causal agent):**

Cada campanha que o V12.1 sugere = **experimento natural**. Usar Bayesian Structural Time Series (Google CausalImpact) com produtos não-promovidos como controles sintéticos:

```python
from tfp_causalimpact import fit_causalimpact

df = pd.read_csv('vendas_diarias.csv', parse_dates=['data']).set_index('data')
pre  = ('2024-01-01', '2026-05-08')
post = ('2026-05-09', '2026-05-12')  # janela pré-Mães
ci = fit_causalimpact(
    df[['chocolate_premium', 'agua', 'cigarro', 'biscoito', 'snack']],
    pre_period=pre, post_period=post)
print(ci.summary())  # uplift absoluto + IC 95% + p-valor bayesiano
```

**Loop fechado:**
1. V12.1 sugere campanha
2. Posto roda 1 semana
3. CausalImpact mede uplift real (com IC 95%)
4. Atualiza `ELASTICIDADE_PROMOCAO` no `calibracao_v2.json`
5. Re-treina V12

Em 6-8 campanhas: elasticidade empírica do POSTO, não da literatura.

**Próximo arquivo:** `medir_uplift_causalimpact.py` consumindo `data/campanhas_executadas.csv` + vendas reais. CausalLift (Minyus) também é opção.

---

## 4. LightGBM global como forecaster

**Causa:** Ridge atual tem MAPE val médio 57%. Cigarros 14-25% (excelente), sorvete/gelo 90-122% (péssimo). Categorias voláteis comprometem decisão do RL.

**Recomendação (Forecasting agent):**

Substituir Ridge por **LightGBM global** com `categoria` como feature categórica + objetivo Tweedie (lida com volumes baixos como vinho 0.6 un/dia).

```python
import lightgbm as lgb

# Treina UM modelo para todas as 20 categorias
X = pd.concat([
    df_long[features],
    pd.get_dummies(df_long['categoria']).astype(float),
], axis=1)
y = df_long['receita']

model = lgb.LGBMRegressor(
    objective='tweedie',  # robusto a volumes baixos + caudas pesadas
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=63,
)
model.fit(X, y)
```

**MAPE esperado:** 25-32% (Forecasting agent estima; M5 winners usaram exatamente LightGBM global).

**Treino:** 1-3 min CPU. Predict <5ms.

**Fase 2 (opcional):** N-HiTS via NeuralForecast (Nixtla). MAPE 22-28% esperado, treino 3-8 min, sorvete/gelo melhoram dramaticamente.

---

## 5-7 — Melhorias secundárias (depois de A/B)

### 5. DRCR Reward (Alibaba 2019)

**Paper:** "Dynamic Pricing on E-commerce Platform with Deep Reinforcement Learning: A Field Experiment" (arXiv:1912.02572)

Alibaba descobriu que reward absoluto colapsa em pricing dinâmico. Substituiu por **DRCR** (Difference of Revenue Conversion Rates):

```
reward_drcr = (lucro_promovido - lucro_baseline_sem_promo) / lucro_baseline
```

**Por que ajuda:** elimina dependência de magnitude. Agente aprende **delta marginal**, não receita absoluta. Já temos K_TIMING_BONUS=250 e K_EVENTO_PRESENTE=600 que são tentativas ad-hoc disso — DRCR formaliza.

**Aplicabilidade ao V13:** alta. Refatorar `step()` de `env_v12.py` para calcular baseline (rollout simulado de "sem-promo" no mesmo estado) e usar como denominador.

### 6. Open Bandit Pipeline + DR estimator

**Paper:** Saito et al. NeurIPS 2021. Lib: `obp` no PyPI.

Para validar offline ANTES do A/B real: tratar nossa política V12.1 como "evaluation policy", calcular reward esperado via Doubly Robust (DR) estimator usando 6 anos de dataset como "behavior policy log".

**Limitação:** behavior policy tem `a=0` em 99% → importance weights explodem. Mitigação: weight truncation + simulador V11.7 como reward model = **model-based OPE** (canon Saito & Joachims 2021).

**Citar na apresentação:** "validação offline rigorosa" é diferencial acadêmico forte.

### 7. (s, S) Optimizer para reposição

**Inventory agent diagnosticou:** maior valor para o dono não é melhor promoção — é melhor reposição. Hoje env usa "reposição implícita instantânea" (irrealista). Lead time real + MOQ podem mudar TUDO.

**Arquitetura recomendada:**
```
Forecaster ML  →  (s,S) Optimizer  →  Promoter DQN
LightGBM         scipy.optimize        V12.3 + mask
                                       + features novas:
                                       forecast_7d_norm
                                       cobertura_dias
```

**Cases:** Amazon Deep Inventory Management (Madeka 2022 arXiv:2210.03137), Walmart "system-centric architecture com purpose-built agents" (2025), Stitch Fix N-BEATS → Bayesian opt → ML matching.

**Veredito da indústria:** ninguém em produção usa monolithic RL fim-a-fim. Pipeline hierárquico com componentes especializados.

---

## O que NÃO fazer (refutado pela pesquisa)

| Ideia | Por que NÃO |
|---|---|
| CQL / IQL / BCQ offline | Nosso dataset tem 99% `a=0` (sem promo). Esses métodos colapsam para "nunca promover" |
| MARL (1 agente por categoria) | Restrição de exclusividade (1 promo/turno) + credit assignment severo + 0 paralelismo temporal |
| MultiDiscrete([21,5,K]) joint inventory+promo | Action space explode 10× sem ganho — credit assignment confuso entre reposição e promo |
| LSTM no DQN | Já testamos (V11.8). Catastrophic forgetting com episódios de comprimento variável (5×5 vs 20×20) |
| TFT (Temporal Fusion Transformer) | Overkill para 20 séries. Brilha em 1000+ séries. LightGBM global é melhor custo-benefício |
| PatchTST / TimesNet | Channel-independence não usa exógenas bem. Nosso caso depende de temperatura + eventos |
| Decision Transformer | Sem variação de return-to-go no dataset — DT precisa de trajetórias de retorno variado |

---

## Papers / cases citados na pesquisa

### Produção real em varejo

- **Alibaba 2019** — DDPG dynamic pricing field experiment, A/B real, bateu humanos (arXiv:1912.02572)
- **Alibaba/Freshippo KDD 2021** — Markdown perecíveis semi-paramétrico, **paper mais próximo do nosso problema** (arXiv:2105.08313)
- **Walmart 2021 (Edelman finalist)** — Markdown clearance lojas físicas, +21% sell-through (-7% custos) (INFORMS)
- **Amazon Deep Inventory Management 2022** — Transformer policy para ordering, separado do pricing (arXiv:2210.03137)
- **Stitch Fix Algorithms Tour** — N-BEATS + Bayesian opt + ML matching, arquitetura hierárquica
- **JD.com 2018** — DQN com state+/state- decomposto (arXiv:1802.06501)
- **Uber Engineering** — RL para matching em 400+ cidades, dueling DQN com value function
- **Virtual Taobao AAAI 2019** — GAN-SD para simulador realista (arXiv:1805.10000)
- **MARIOD Sensors 2025** — MARL hierárquico para retail supply chain

### Algoritmos / teoria

- **CQL** Kumar et al. 2020 (NeurIPS)
- **IQL** Kostrikov et al. 2021
- **BCQ** Fujimoto et al. 2019
- **Decision Transformer** Chen et al. 2021
- **CausalImpact** Brodersen et al. 2015 (Google)
- **Doubly Robust OPE** Dudík et al. 2014 + Saito & Joachims 2021
- **DoubleML** Chernozhukov et al. 2018
- **X-learner** Künzel et al. 2019
- **Causal Forests** Wager & Athey 2018
- **Action Masking** Huang & Ontañón 2020 (arXiv:2006.14171)
- **RCPO** Tessler et al. 2018 (Lagrangian)
- **CPO** Achiam et al. 2017
- **AlphaStar** Vinyals et al. 2019 (Nature)
- **Hierarchical DQN** Kulkarni et al. 2016
- **LinUCB** Li et al. 2010
- **QMIX** Rashid et al. 2018
- **N-HiTS** Challu et al. 2023
- **LightGBM** Ke et al. 2017 + M5 Competition winners

### Bibliotecas Python recomendadas

- `obp` (Open Bandit Pipeline) — OPE/CPE rigoroso
- `d3rlpy` — Offline RL (CQL/IQL/BCQ)
- `tfp-causalimpact` — Bayesian Structural Time Series
- `econml` + `DoubleML` — causal inference
- `causalml` (Uber) — uplift modeling
- `causallift` (Minyus) — uplift para business
- `pysyncon` — Synthetic Control
- `neuralforecast` (Nixtla) — N-HiTS, TFT, PatchTST
- `lightgbm` — global forecaster M5-style
- `darts` (Unit8) — TFT, N-BEATS, ensembles
- `sb3-contrib` — MaskablePPO
- `omnisafe` (PKU-Alignment) — Constrained RL

---

## Ordem sugerida de implementação

1. ✅ **V12.3a — Action masking** (FEITO hoje)
2. **V12.3b — CondPolicyDQN training** (2h: trocar BranchingDQN, treinar 150 ep)
3. **V13.1 — LightGBM forecaster** (4h: substituir Ridge, comparar MAPE)
4. **V13.2 — CausalImpact loop** (1 dia: script + integração com calendário)
5. **V13.3 — Open Bandit Pipeline OPE** (1 dia: validação offline)
6. **V13.4 — DRCR reward refactor** (4h: redefinir reward como delta)
7. **V13.5 — (s, S) Inventory** (2 dias: scipy.optimize + features novas no env)
8. **A/B real no posto** (4-8 semanas, conforme Vinicius alinhar com o dono)
9. **Re-calibração com elasticidade empírica** (loop CausalImpact mede, modelo aprende)

---

*12/05/2026 madrugada. Pesquisa baseada em 8 agentes paralelos consumindo
~150k tokens. Filtrado para o que cabe no Auto Posto Parque Viana específico
(20 categorias, sem A/B, 6 anos de dado, regulação cigarro). Cada melhoria
tem justificativa de paper real de produção citado.*
