# Otimização de Promoções com RL — Auto Posto Parque Viana

**Disciplina:** Reinforcement Learning — Insper
**Autores:** Vinicius Rocha Pereira · Luigi Zema Matizonkas
**Empresa:** Auto Posto Parque Viana Ltda — Barueri/SP

Um agente RL que decide qual promoção aplicar a cada turno da loja de conveniência do posto para maximizar lucro e minimizar perdas por vencimento. O modelo final (V12.1) é treinado em 6 anos de vendas reais e produz um calendário operacional anual de campanhas.

---

## 1. Problema e objetivo

O Auto Posto Parque Viana opera uma loja de conveniência com mais de 700 SKUs. O dono decide promoções por intuição: quando estoque sobra, dá desconto; quando algo está perto de vencer, faz combo. Não há método sistemático para:

- **Maximizar lucro em datas comerciais** (Réveillon, Copa, Dia dos Namorados)
- **Prevenir vencimento** de cerveja, gelo, sorvete e padaria
- **Decidir qual produto promover quando** — segundas vs sábados, manhã vs noite, verão vs inverno

A meta é entregar um **calendário anual de promoções** que o dono possa imprimir e seguir, com justificativa clara para cada campanha.

### Não é só academia

O projeto é um produto. A entrega para o Insper é o checkpoint, mas o que importa é o calendário V12.1 funcional, validado em hold-out, pronto para teste A/B no posto.

---

## 2. Os dados

| Arquivo | Conteúdo | Período |
|---|---|---|
| `data/venda_por_dia.xlsx` | 6 anos de venda por categoria × turno (147.873 registros) | 22/06/2020 → 30/04/2026 |
| `data/venda_do_mes.xlsx` | Preços, custos, margens por SKU | Março/2026 |
| `data/descarte_produto.xlsx` | Descarte por vencimento | Março/2026 (19 registros) |
| `data/produtos_nao_vendidos/` | 52 snapshots mensais de estoque parado | 2022–2026 |
| `data/temperatura_historica.csv` | Temperatura máxima diária Barueri/SP (Open-Meteo) | 2020–2026 |
| `data/calendario_comercial.csv` | 278 eventos comerciais BR (feriados + datas comerciais + Copa + locais) | 2016–2027 |

Priors externos calibrados:
- **Dunnhumby** (cesta supermercado USA): elasticidade promocional empírica
- **Iowa Liquor** (vendas álcool USA): uplift por feriado para destilados
- **Walmart Sales** (varejo USA): sazonalidade macro
- **IBGE PMC** (varejo BR): sazonalidade Dez +20%, Fev −11%

---

## 3. Formulação do MDP

### Estado — 150 features

| Slice | Conteúdo | Dim |
|---|---|---:|
| `[0:23]` | Calendário: dia_sem (7) + turno (3) + mês (12) + dia_do_mês (1) | 23 |
| `[23:34]` | Evento próximo (10 tipos one-hot) + dias_até_evento | 11 |
| `[34:36]` | Temperatura norm. + Δ temperatura 7d | 2 |
| `[36:39]` | Passo temporal + fator IBGE-PMC + contexto promo | 3 |
| `[39:50]` | Padding + sinal Dunnhumby | 11 |
| `[50:70]` | Estoque norm. por categoria (N=20) | 20 |
| `[70:90]` | Validade restante por categoria | 20 |
| `[90:110]` | `fraco_flag` (contexto sazonal bottom 30%) | 20 |
| `[110:130]` | Promoção do turno anterior por categoria | 20 |
| `[130:150]` | **Forecast Ridge** receita normalizada (V12) | 20 |

### Ação — MultiDiscrete `[N+1, 5]`

Decisão decomposta em duas dimensões:

| Dimensão 1: produto | Dimensão 2: intensidade |
|---|---|
| 0 = sem promoção | 0 = nada |
| 1..20 = categoria | 1 = desconto 5% |
|  | 2 = desconto 10% |
|  | 3 = **combo cooperativo** (env escolhe par via harmonia) |
|  | 4 = liquidação 25% |

Espaço total: 21 × 5 = 105 ações.

### Recompensa — 13 termos

```
r = lucro
  - α × vencimento × custo                    (V11)
  - β × ruptura × margem                       (V11)
  - pen_desconto (tier 5/10/25%)               (V11)
  + bonus_giro (validade < 30%)                (V11)
  + K_TIMING_BONUS / -K_TIMING_PENALTY         (V10)
  + bonus_evento × harmonia_evento[ev][cat]    (V11.7 + V12.2)
  + bonus_padrão_dunnhumby                     (V11)
  - pen_instabilidade                          (V11)
  - pen_não_promovível (cigarros)              (V12.1)
  - K_DESC_ALTA_SAUDÁVEL                       (V11.7)
  + K_COMBO_ALTA + K_COMBO_DATA_PICO           (V11.5+)
  + K_DESC_VENCIMENTO + K_DESC_BAIXA           (V11.7)
  + bonus_dia_semana_categoria                 (V11.6)
```

Cada termo nasceu de uma falha específica do modelo anterior. Detalhe em §5.

### Episódio

1095 turnos = 365 dias × 3 turnos = **1 ano calendário real**. Data inicial sorteada uniformemente no range de treino (2020–2024). Avaliação em hold-out 2024–2026.

---

## 4. As 20 categorias

Cobertura de ~80% do faturamento da loja, agrupando 717 SKUs em 20 categorias:

| Categoria | Demanda/dia | Margem | Promovível |
|---|---:|---:|:---:|
| cerveja | 20.6 un | 55.7% | ✅ |
| cigarro_souza_cruz | 122.3 un | 22.0% | ❌ Lei 9.294/96 |
| cigarro_philip_morris | 65.3 un | 14.7% | ❌ Lei 9.294/96 |
| cigarro_jti | 22.0 un | 9.9% | ❌ Lei 9.294/96 |
| chocolate_premium | 8.2 un | 56.0% | ✅ |
| chocolate_impulso | 6.9 un | 58.6% | ✅ |
| refrigerante | 10.7 un | 56.9% | ✅ |
| gelo | 5.6 un | 50.6% | ✅ |
| sorvete | 12.1 un | 42.3% | ✅ |
| água | 16.9 un | 70.2% | ❌ commodity |
| snack | 10.6 un | 47.2% | ✅ |
| isotonico | 2.5 un | 61.0% | ✅ |
| energetico | 5.3 un | 59.2% | ✅ |
| destilados | 0.5 un | 92.9% | ✅ |
| vinho | 0.6 un | 100% | ✅ |
| doce | 22.8 un | 60.1% | ✅ |
| biscoito | 3.2 un | 49.5% | ✅ |
| café | 1.3 un | 44.4% | ✅ |
| padaria | 2.7 un | 44.7% | ✅ |
| suco | 2.8 un | 54.6% | ✅ |

**Cigarros não-promovíveis** por causa da Lei Antifumo (9.294/96 e regulamentações ANVISA): promoção/desconto/publicidade de cigarro é proibida no Brasil. O modelo V12 base "descobriu" promover cigarro_jti em 15% das decisões — matematicamente bom, regulatoriamente inviável. Marcado como NPM a partir da V12.1.

**Água** é commodity utilitária: cliente compra porque tem sede, preço não estimula compra.

---

## 5. Evolução do modelo — por que cada upgrade existe

| Versão | Mudança principal | Reward médio | F1 evento | Por quê |
|---|---|---:|---:|---|
| V10 (entrega Insper) | 47 features, 6 produtos fixos, episódio 21 passos | R$ 33.926 | 0 | Bug crítico: mês fixo em janeiro durante treino |
| V11 base | 41 features, 5 ações, fator mensal correto | R$ 5.977 | — | Ação 3 (combo) nunca selecionada |
| V11.3 | Elasticidade empírica (-0.5 vs Bijmolt -3.0, **6× menor**) | R$ 17.000 | — | Bijmolt superestima 52× elasticidade em loja física |
| V11.5 | Combos reforçados + bonus por timing de data comercial | — | — | Combos eram dominados por descontos diretos |
| V11.6 | Bonus dia-semana data-driven por categoria | — | — | Agente promovia gelo nas segundas (péssimo timing) |
| V11.7 (operacional) | K_COMBO_DATA_PICO 400 | R$ 873.000 | 0.106 | Réveillon e Natal funcionaram, Mães/Namorados não |
| V12 base | + Forecaster Ridge no estado (130 → 150 features) | R$ 846.000 | 0.053 | F1 timing +51%, mas descobriu promover cigarro 15% |
| **V12.1** | + Cigarros NPM + K_EVENTO_PRESENTE=600 (3× base) | **R$ 859.000** | **0.208** | **Destravou Dia das Mães + Namorados (F1 0→0.22)** |
| V12.2 | + Harmonia categoria-categoria + janela_pre reduzida 2d | R$ 857.000 | 0.106 | Janela curta prejudicou aprendizado; harmonia funciona |

### Por que V12.1 e não V12.2

O V12.2 tem mais sofisticação técnica (harmonia categorial faz chocolate em Dia dos Namorados escolher vinho como par de combo, não gelo). Mas a redução da janela_pre_dias de 5-7d para 2d (alinhar com realidade do posto onde cliente compra presente de última hora) tirou sinal de treino do agente — episódios com evento ficaram curtos demais.

Trade-off identificado:
- **Janela larga durante treino** = melhor aprendizado (mais turnos com sinal de evento)
- **Janela curta = realidade do posto** (cliente de última hora)

Solução para V12.3 (não implementada): treinar com janela larga, deploy com janela curta. OU usar decay agressivo de uplift (pico em D-2 a D, fraco antes).

### Detalhe técnico — descobertas críticas

**Bug do mês fixo (V10):**
```python
m = (self._step // 63) % 12   # com episódio de 21 passos, divisor 63 nunca alcançado
```
Toda a calibração de `FATOR_MES` (gelo em dezembro = 2.19×) não era usada no treino. O agente sempre via janeiro. Fix V11: mês aleatório por episódio.

**Elasticidade-preço vs elasticidade-promocional (V11.3):**
A literatura inicial (Bijmolt 2005) usa elasticidade promocional de e-commerce (-3.0 a -3.8). Testamos com 3 fontes físicas (Dunnhumby, Iowa Liquor, Walmart) e elas convergem em **e = -0.5** — 6× menor que a literatura. Em loja física conveniência, desconto 10% gera ~5% mais venda, não 30%.

**Combo cooperativo (V11.5 → V12.2):**
A ação 3 (combo) decompõe a decisão: agente escolhe **principal**, env escolhe **par**. Em V11.7 o par é o produto com maior `fator_dia × fator_turno × fator_mes`. Em V12.2:
```python
score_par = fator_contextual × harmonia_combo[principal]
par_dinamico = argmax(score_par)
```
Resultado: chocolate_premium em **manhã de dia útil de dezembro** → par = vinho (harmonia 2.5 × fator 3.75 = score 9.38, vence gelo 4.68).

---

## 6. Estratégia de treinamento

### Arquitetura DQN — BranchingDQN

Dueling DQN com cabeças decompostas para `MultiDiscrete([21, 5])`:

```
              ┌──── Linear(150 → 256 → 128) ────┐
                                                 │
                                       ┌─── Value(128 → 1) ──────┐
                                       │                          │
                                       ├─── A_prod(128 → 64 → 21)─┼──▶ Q(s, p, i)
                                       │                          │
                                       └─── A_int(128 → 64 → 5)───┘
```

`Q(s, p, i) = V(s) + A_prod(s, p) + A_int(s, i)` — reduz saída de 21×5=105 para 21+5=26 unidades.

### Forecaster ML — Ridge por categoria

Cada categoria tem seu próprio modelo Ridge. Features: dia_sem, mês, dia_mês, temp_norm, lag1/7/28 de receita, is_event, days_to_event, tipo_pico (pre/no_dia). MAPE val médio 57%, R² médio +0.045.

Decisão de design: Ridge em vez de HistGradientBoosting (testado, 18× mais lento). Loss similar, inferência ~50× mais rápida — essencial porque o env chama predict 19 vezes por dia simulado.

### Hiperparâmetros V12.1

| Param | Valor |
|---|---|
| Episódios | 150 × 1 seed |
| Steps por episódio | 1095 (1 ano) |
| `lr` | 3e-4 |
| `gamma` | 0.99 |
| `eps_decay` | 0.985 (ε final ~0.10) |
| Batch size | 64 |
| Replay buffer | 50.000 |
| Loss | HuberLoss (SmoothL1Loss) |
| Target update | a cada 5 episódios |
| Threads CPU | 8 (`torch.set_num_threads(8)`) |
| Tempo treino | ~30 min em CPU 8 cores |

### Reward shaping consciente

A literatura (Ng, Harada & Russell 1999) aceita reward shaping desde que potencial. Aqui usamos shaping **não-potencial** intencionalmente — codifica diretamente regras de varejo no MDP:

- `K_EVENTO_PRESENTE=600` (3× base) → datas de presente têm peso maior porque o sinal de demanda é "ruidoso" no histórico (Mães/Namorados acontece 1 vez por ano)
- `K_DESC_ALTA_SAUDAVEL=-200` → proibir desconto direto em produto de alta demanda saudável (regra do dono: "não dou desconto onde já vendo bem")
- `K_COMBO_ALTA=+200` → combo é estratégia válida em alta demanda (aumenta ticket médio)

A justificativa é operacional, não algorítmica.

---

## 7. Resultados

### Validação hold-out — 20 ep × 1095 steps em 2024-2026

| Modelo | Reward médio | Δ Lucro vs sem-promo | Δ Perdas vs sem-promo | F1 evento médio |
|---|---:|---:|---:|---:|
| Sem promoção (baseline) | R$ 563.700 | 0% | 0% | — |
| Aleatória | R$ 465.000 | -0.92% | -0.71% | 0.04 |
| Sempre combo | R$ 541.350 | +0.10% | -0.05% | 0.18 |
| V12 base | R$ 773.659 | +0.36% | -1.53% | 0.053 |
| **V12.1** | **R$ 859.248** | **+0.23%** | **-1.81%** | **0.208** ⭐ |
| V12.2 | R$ 857.000 | +0.06% | -1.85% | 0.106 |

### F1 por evento comercial — V12.1

| Evento | V12 base | V12.1 | Variação |
|---|---:|---:|---|
| **Réveillon** | 0.24 | **0.96** | quase perfeito |
| **Véspera de Natal** | 0.00 | 0.46 | destravado |
| **Dia das Mães** | 0.00 | **0.24** | destravado |
| **Dia dos Namorados** | 0.00 | **0.22** | destravado |
| Dia das Crianças | 0.02 | 0.04 | marginal |
| Dia dos Pais | 0.10 | 0.00 | regressão |
| Dia da Mulher | 0.00 | 0.00 | continua falhando |

### Política aprendida — top 7 ações

| % decisões | Ação |
|---:|---|
| 31.3% | gelo + combo (par dinâmico = destilados ou cerveja) |
| 12.9% | chocolate_premium + desc5% |
| 11.1% | chocolate_impulso + desc5% |
| 10.5% | vinho + desc5% |
| 9.1% | sem promoção (combo flag) |
| 6.4% | snack + combo |
| 4.7% | refrigerante + desc5% |

Notar: chocolate_premium, chocolate_impulso e vinho juntos representam **34.5%** das decisões — exatamente as categorias-alvo de Mães/Namorados/Páscoa/Natal/Pais. Em V12 base esses 3 produtos somavam ~8%.

### Distribuição de intensidades — V12.1 usa todas

| Intensidade | % uso em V11.7 | % uso em V12.1 |
|---|---:|---:|
| desc5% | 0% | **34%** ⭐ |
| desc10% | 0% | 2% |
| combo | 33% | 47% |
| liquidação 25% | 0% | 0% |
| sem promoção | 67% | 17% |

V12.1 descobriu `desc5%` como ação útil — V11.7 nunca usava.

---

## 8. Análise das campanhas — por que cada uma faz sentido

Top 15 do calendário anual V12.1 (12/05/2026 a 11/05/2027):

| # | Período | Campanha | Lucro adic | Evento | Justificativa |
|---:|---|---|---:|---|---|
| 1 | 25/12 → 31/12 (7d) | 🧊 Gelo + Destilados combo | R$ 32,08 | Réveillon | Pico anual de drinks/festa. Olist confirma +3-5× em bebidas no Réveillon. Combo aumenta ticket médio sem matar margem. |
| 2 | 11/03 → 14/03 (4d) | 🧊 Gelo + Destilados combo | R$ 18,33 | Dia do Consumidor | Final de verão + data comercial. Gelo ainda demanda alta em março (sáb 3-4×). |
| **3** | **11/07 → 12/07 (2d)** | **🍫 Chocolate Premium + Vinho combo** | **R$ 8,26** | — | **Harmonia categorial em ação:** sexta-sábado de inverno, chocolate em combo escolhe vinho como par (harmonia 2.5 × fator_ctx 4.10 = 10.26, vence gelo). Cesta de presente clássica fim-de-semana. |
| 4 | 07/09 → 11/09 (5d) | 🍫 Chocolate Premium desc5% | R$ 4,54 | Pré-Dia das Crianças | Semana antes de 12/10 (data não-pico em setembro mas pré-evento). Desconto direto leve para movimentar estoque de chocolate antes da campanha grande. |
| 5 | 14/09 → 18/09 (5d) | 🍫 Chocolate Premium desc5% | R$ 4,54 | Pré-Dia das Crianças | Continuação. |
| 6 | 29/06 → 03/07 (5d) | 🍫 Chocolate Impulso desc5% | R$ 2,84 | — | Inverno seco, chocolate impulso (Snickers, KitKat) é compra emocional em posto. Volume sobe quando dá desconto leve. |
| 7 | 06/07 → 10/07 (5d) | 🍫 Chocolate Impulso desc5% | R$ 2,84 | — | Continuação. |
| 8 | 04/06 → 05/06 (2d) | 💪 Isotonico desc5% | R$ 0,80 | Pré-Copa | Atletismo + pré-jogo. Isotônico tem demanda dirigida por evento esportivo. |
| 9 | 11/06 → 12/06 (2d) | 💪 Isotonico desc5% | R$ 0,80 | Copa 2026 Abertura | Jogo de abertura = pico de movimento no posto. Isotônico para esporte + cervejada. |
| 10 | 12/02 → 14/02 (3d) | 🧊 Gelo + Cerveja combo | R$ 13,75 | Pré-Carnaval | Carnaval 14-18/02/2026. Cerveja + gelo é cesta clássica de bloco/festa. Harmonia 2.4 (gelo↔cerveja). |
| 11 | 19/02 → 21/02 (3d) | 🧊 Gelo + Cerveja combo | R$ 13,75 | Pós-Carnaval | Continuação. |
| 12 | 26/02 → 28/02 (3d) | 🧊 Gelo + Cerveja combo | R$ 13,75 | — | Verão alto, fim de semana. |
| 13 | 05/03 → 07/03 (3d) | 🧊 Gelo + Destilados combo | R$ 13,75 | Dia Internacional da Mulher | F1 deste evento é 0 — agente promove gelo+destilados (que tem alta demanda) em vez de chocolate/vinho. **Limitação identificada:** chocolate/vinho têm volume baixo demais para o agente priorizar via lucro. Próximo passo: aumentar K_EVENTO_PRESENTE para 1000+. |
| 14 | 19/03 → 21/03 (3d) | 🧊 Gelo + Destilados combo | R$ 13,75 | — | Final do verão. |
| 15 | 26/03 → 28/03 (3d) | 🧊 Gelo + Destilados combo | R$ 13,75 | — | Idem. |

### Padrões emergentes da política

**1. Combo gelo + destilados / cerveja é o "default" de fim-de-semana**
Aparece em 9 das 15 top campanhas. Por que: gelo tem o maior fator_dia em sábado (2.24×) e dezembro (2.19×). Destilados harmonia 2.2 com gelo, cerveja harmonia 2.4. Em qualquer fim de semana de verão/festa, é a campanha que mais gera lucro adicional por turno.

**2. Chocolate em todas as semanas-presente** (Mães, Namorados, Pais, Crianças, Páscoa, Natal)
Apareceu 5 vezes no top 15, com `desc5%` (não combo). O agente aprendeu que chocolate em data de presente tem uplift suficiente para justificar o desconto direto. Confirmação visual: a campanha #3 (julho) usa combo com vinho — harmonia ativa.

**3. Isotônico em jogos da Copa**
Vinicius identificou esse padrão na análise externa: dia de jogo do Brasil = +30-50% no fluxo do posto. Isotônico bate com churrasco/cerveja (consumo conjunto). Agente acertou.

### Campanhas que parecem fracas mas têm razão

**Lucro adicional de R$ 4,54 por campanha de chocolate parece baixo.** Por que isso é OK:

- Chocolate_premium no posto tem demanda diária baixa (8.2 un/dia, R$ 8,29 cada)
- 5 dias × desc5% × elasticidade -0.5 = ~5% mais venda
- Uplift esperado: 8.2 × 5 × 0.05 = ~2 unidades extras
- Margem unitária R$ 4,65 → 2 × 4,65 = ~R$ 9,30, descontando perda de margem ≈ R$ 4,50

O número bate. **A magnitude absoluta é pequena porque o volume base é pequeno** — não porque a decisão está errada. Em SKUs com volume maior (cigarro 122 un/dia que NÃO podemos promover), o uplift seria 15× maior.

### O que NÃO foi destravado — limitações honestas

**Dia da Mulher (F1 = 0):** Categorias-alvo são chocolate, vinho, espumante, **perfume**, **flores**. Perfume e flores não existem no catálogo do posto. Mesmo com K_EVENTO_PRESENTE alto, o agente não tem onde "acertar" — as únicas alvos que ele tem (chocolate + vinho) têm volume baixíssimo.

**Dia dos Pais (F1 = 0 no V12.1):** A regressão vs V12 base (F1 0.10) é instável. Categorias-alvo (cerveja, whisky/cachaça/destilados, snack, vinho_tinto) bate parcialmente com o catálogo, mas o agente prefere gelo+destilados (que dá mais lucro absoluto).

**Estatisticamente:** F1 evento médio 0.208 é baixo em termos absolutos. Significa que em ~80% dos turnos de janela de evento, o agente NÃO promove a categoria-alvo. Mas isso é esperado dada a estrutura: existem 365 dias e ~30 dias de "janela de evento" — o restante é decisão de rotina (gelo no fim de semana, chocolate desc em semanas mortas).

---

## 9. Limitações honestas

### 1. Validação off-policy

O simulador foi calibrado em 6 anos de vendas **sem promoção real** do posto. A elasticidade usada (e = -0.5) vem de 3 datasets externos (Dunnhumby, Iowa, Walmart), não medida no Auto Posto Viana. Os números "+0.23% lucro" e "-1.81% perdas" são válidos **dentro da nossa suposição de elasticidade** — não validados empiricamente.

Próximo passo essencial: teste A/B real (semanas alternadas com/sem agente) para medir elasticidade in loco e recalibrar.

### 2. Calibração por categoria, não por SKU

Vendas por SKU detalhadas ainda não chegaram do ERP. O agente trata chocolate_premium como "categoria" (8.2 un/dia) sem distinguir Lacta vs Nestlé vs Garoto. Quando dados chegarem:

```powershell
python calibrar_v2.py     # re-calibra com SKU
python treinar_v12.py     # re-treina
```

### 3. Janela_pre_dias é trade-off não resolvido

V12.1 mantém janela 5-7d (boa para treino mas otimista para varejo de posto). V12.2 reduziu para 2d (realidade do posto) mas perdeu sinal de aprendizado. V12.3 deveria treinar com janela larga e fazer rollout/avaliação com janela curta.

### 4. Categorias-alvo de Dia da Mulher fora do catálogo

Perfume, flores, espumante — não estão no posto. F1 = 0 estrutural, não falha do algoritmo.

### 5. Política colapsa em ~7 ações de 105 possíveis

Liquidação 25% nunca emerge. Desc 10% raramente. Diversidade limitada pelo espaço de ação + dado. Sugestão: penalidade explícita para colapso, ou hierarquia (decisão de categoria → decisão de intensidade).

---

## 10. Como rodar

### Setup

```powershell
cd C:\Users\vinin\projeto-rl
.venv\Scripts\activate
```

### Pipeline completo (regenerar tudo)

```powershell
# 1. Calibrar (5s)
python calibrar_v2.py

# 2. Treinar forecaster Ridge (5s)
python treinar_forecaster.py

# 3. Treinar DQN V12.1 (~30 min em CPU 8 cores)
python treinar_v12.py --episodios 150 --seeds 1 --max_steps_per_ep 1095

# 4. Validar V12.1 (~3 min)
python validar_v12.py --n_episodios 20 --max_steps 1095

# 5. Comparar V12 base × V12.1 × V12.2 (~10 min)
python comparar_v12_versoes.py

# 6. Gerar calendário operacional (10s)
python gerar_calendario_v4.py --horizonte 365

# 7. HTML visual
python gerar_html_premium_v12.py

# Abrir HTML
start results\v12\calendario_premium_v12.html
```

### Estrutura final do repositório

```
projeto-rl/
├── README.md                       ← este arquivo
├── CLAUDE.md                       ← memória interna de desenvolvimento
│
├── calibrar_v2.py                  ← calibração do MDP (priors + dados reais)
├── env_v12.py                      ← ambiente Gymnasium consolidado
├── dqn.py                          ← BranchingDQN + ReplayBuffer
├── treinar_forecaster.py           ← Ridge por categoria
├── treinar_v12.py                  ← treino DQN V12
├── validar_v12.py                  ← validação hold-out
├── comparar_v12_versoes.py         ← comparação V12 base/V12.1/V12.2
├── gerar_calendario_v4.py          ← rollout determinístico → calendário
├── gerar_html_premium_v12.py       ← HTML operacional
│
├── data/                           ← dados crus + calibração + priors
├── notebooks/                      ← V10 (entrega Insper original)
├── relatorio/                      ← estrutura para relatório final
├── results/v12/                    ← deploy V12
└── backup/                         ← scripts e resultados V10/V11 históricos
    ├── scripts/   (35 scripts)
    └── results/   (V11.7 + EDA + análises antigas)
```

---

## 11. Trabalho original vs externo

| Componente | Origem |
|---|---|
| `ConvenienceStoreEnvV12` | Implementação própria — padrão Gymnasium |
| `BranchingDQN` (dueling decomposto) | Wang et al. 2016 + adaptação MultiDiscrete |
| Forecaster Ridge por categoria | Implementação própria, features hand-crafted |
| `HARMONIA_PARES` (62 pares categoria↔categoria) | Conhecimento de varejo + verificação parcial via Dunnhumby market basket |
| `HARMONIA_EVENTO_CATEGORIA` (19 eventos) | Conhecimento de varejo brasileiro do dono do posto |
| Calibração temporal (FATOR_DIA/TURNO/MES/CLIMA) | Calculado em 6 anos de vendas reais do posto |
| Elasticidade empírica (e = -0.5) | 3 datasets físicos: Dunnhumby USA, Iowa Liquor, Walmart Sales |
| Calendário comercial BR (278 eventos) | `holidays` pip + curadoria manual |
| `IBGE PMC` (sazonalidade macro varejo BR) | API SIDRA pública IBGE |
| `Stable-Baselines3` | Open-source, citado |
| `gym_custom_env` (referência inicial) | Professor F. Barth |

---

## 12. Referências

- Bijmolt, van Heerde & Pieters (2005). *New Empirical Generalizations on the Determinants of Price Elasticity*. JMR — meta-análise de 1851 elasticidades promocionais
- Ng, Harada & Russell (1999). *Policy Invariance Under Reward Transformations*. ICML 1999 — fundamentação teórica do reward shaping
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML 2016 — base do BranchingDQN
- Stable-Baselines3: Raffin et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- Dunnhumby Complete Journey (Kaggle dataset)
- Iowa Liquor Sales (Iowa Open Data)
- Walmart Sales Forecasting (Kaggle)
- Tesco Grocery (UK Data Service)
- IBGE Pesquisa Mensal do Comércio (SIDRA)
- Open-Meteo Historical Weather API
- Lei 9.294/96 — Lei Antifumo (Brasil)
