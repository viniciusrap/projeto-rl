# PROMPT — Apresentação Final RL Conveniência Viana

Copie tudo abaixo e cole no Claude do PowerPoint.

---

# Crie uma apresentação em PowerPoint para minha entrega final de Reinforcement Learning no Insper

## Contexto do projeto

**Título:** Otimização de Promoções com RL — Auto Posto Parque Viana
**Autores:** Vinicius Rocha Pereira e Luigi Zema Matizonkas
**Disciplina:** Reinforcement Learning — Insper, 2026
**Empresa real:** Auto Posto Parque Viana Ltda — Barueri/SP. Loja de conveniência com ~700 SKUs e ~20 categorias.
**Problema:** O dono decide promoções por intuição. Queremos um agente RL que sugira calendário operacional (quando e qual desconto/combo aplicar) maximizando lucro e minimizando vencimento, com dados reais.

**Versão final em deploy:** V13 (= V12.1 + Action Mask para compliance Lei 9.294/96 sobre cigarros).
**Tentativa rigorosa V14:** ensemble de 5 seeds com 5 técnicas SOTA (PER, curriculum, mask dinâmico, DRCR, ensemble).

## Requisitos da apresentação

- **Duração:** 10 minutos máximo (10 slides + capa = 11 slides total)
- **Tom:** técnico mas acessível, honesto sobre limitações. Banca é composta por professores de RL — não precisa fingir que tudo funcionou.
- **Visual:** clean, profissional. Use tabelas quando tiver números, gráficos quando tiver evolução temporal, código mínimo (só fórmulas-chave).
- **Speaker notes:** Cada slide deve ter notas detalhadas (3-4 frases) para eu falar durante a apresentação.
- **Slides:** mínimo de texto no slide, máximo de informação nas notas.

## Estrutura dos 11 slides

---

### Slide 1 — Capa

**Título:** Otimização de Promoções com Reinforcement Learning
**Subtítulo:** Auto Posto Parque Viana · Estudo de caso real com 6 anos de dados
**Rodapé:** Vinicius Rocha Pereira · Luigi Zema Matizonkas · Insper 2026

**Visual:** apenas o título grande, subtítulo, autores. Fundo limpo.

**Speaker notes:** "Estudo de caso real com 6 anos de dados de vendas de uma loja de conveniência em Barueri. Vou mostrar a evolução do nosso agente RL desde V10 até V13, o que funcionou, o que não funcionou, e por quê."

---

### Slide 2 — O Problema

**Título:** O dono do posto promove por intuição

**Conteúdo (3 caixas/bullets):**
- **Loja de conveniência** com ~700 SKUs em 20 categorias agregadas
- **Decisões diárias:** quando dar desconto? Qual combo? Liquidar para evitar vencimento?
- **Dor real medida:** 19 produtos descartados em mar/2026, R$ XX em prejuízo de cerveja vencida

**Visual:** Foto/ilustração simples de um posto, ou apenas tipografia limpa.

**Speaker notes:** "O Auto Posto Viana fica em Barueri, São Paulo. O dono toma decisões de promoção por intuição. Quando estoque sobra, dá desconto. Quando algo está perto de vencer, faz combo. Não tem método sistemático. Cerveja AMBEV teve 7.7% de taxa de perda em março. Queremos automatizar essa decisão com RL."

---

### Slide 3 — Os Dados

**Título:** 6 anos de venda real + priors externos

**Tabela:**
| Fonte | Conteúdo | Período |
|---|---|---|
| venda_por_dia.xlsx | 147.873 registros por categoria × turno | 2020-2026 |
| venda_do_mes.xlsx | Preços, custos, margens por SKU | Mar/2026 |
| descarte_produto.xlsx | Produtos vencidos por turno | Mar/2026 |
| temperatura_historica.csv | Temp diária Barueri (Open-Meteo) | 2020-2026 |
| calendario_comercial.csv | 278 eventos comerciais BR | 2016-2027 |

**Priors externos calibrados:**
- Dunnhumby (supermercado USA): elasticidade promocional
- Iowa Liquor + Walmart Sales: uplift por feriado
- IBGE PMC: sazonalidade macro do varejo BR (Dez +20%, Fev -11%)

**Speaker notes:** "Calibramos tudo em vendas reais do posto, mas como não temos promoções aplicadas no histórico, complementamos com 3 datasets físicos públicos. A elasticidade que usamos é -0.5, calibrada em três fontes independentes que convergem — não em literatura de e-commerce que dava -3, muito otimista para conveniência."

---

### Slide 4 — Formulação MDP

**Título:** O ambiente como Processo de Decisão Markoviano

**Conteúdo em 3 colunas:**

**Estado (150 features):**
- Calendário (dia/turno/mês + evento próximo)
- Estoque + validade por categoria
- Forecast Ridge ML
- Fraco/forte flags

**Ação (MultiDiscrete [21,5]):**
- Qual categoria (0=sem-promo, 1-20)
- Qual intensidade: nada / -5% / -10% / combo / liquidação -25%
- **105 ações totais, máscara hard para cigarros**

**Recompensa (13 termos):**
- Lucro − vencimento − ruptura − pen_desconto
- + bonus_giro + bonus_timing + bonus_evento
- + harmonia categoria + harmonia evento-puxador
- − pen_instabilidade − pen_não-promovível

**Episódio:** 1095 turnos = 1 ano calendário real

**Speaker notes:** "Estado de 150 features porque o agente precisa ver muita coisa: calendário, clima, estoque, validade, evento próximo, previsão. Ação multi-discreta porque queremos decompor 'qual produto' de 'qual intensidade'. Recompensa tem 13 termos porque cada termo nasceu corrigindo uma falha de uma versão anterior — não é arbitrário."

---

### Slide 5 — Evolução V10 → V13

**Título:** Aprendizado iterativo das versões

**Tabela:**
| Versão | Mudança principal | F1 evento | Status |
|---|---|---:|---|
| V10 (entrega Insper) | 47 features, 6 produtos, 21 turnos | 0 | Bug: mês fixo em janeiro |
| V11 | 20 categorias, 1095 turnos | 0.106 | Combo nunca emergia |
| V11.5 | Combos reforçados + bonus data | 0.157 | F1 ainda baixo |
| V11.7 | K_COMBO_DATA_PICO=400 | 0.106 | Mães/Namorados zerados |
| V12 base | + Forecaster Ridge (150 features) | 0.053 | Descobriu cigarro 15% (proibido!) |
| **V12.1** | + cigarros NPM + K_EVENTO_PRESENTE=600 | **0.208** | Destravou Mães e Namorados |
| **V13** | V12.1 + Hard Mask cigarros | **0.208** | DEPLOY |

**Insight crítico (caixa destacada):**
> V12.1 deu **sorte estatística** — outras seeds não reproduzem. Variance natural do RL com 1 seed.

**Speaker notes:** "Cada versão corrige uma falha específica. V10 tinha bug crítico: mês fixo em janeiro durante todo o treino. V11 trouxe sazonalidade. V11.7 destravou eventos. V12 base descobriu uma estratégia matematicamente boa MAS ilegal: promover cigarro 15% das vezes — Lei Antifumo 9.294/96 proíbe. V13 corrige isso com hard mask."

---

### Slide 6 — Modelo V13 em Deploy

**Título:** O agente final em produção

**Arquitetura (diagrama simples):**
```
Estado [150] → BranchingDQN dueling decomposto
                ├─ V(s)
                ├─ A_prod(s,p) [21 produtos]
                └─ A_int(s,i)  [5 intensidades]

Q(s,p,i) = V + A_prod + A_int
```

**Decisões-chave:**
- **Double DQN** (Hasselt 2016) — evita overestimation
- **HuberLoss** — robusto a outliers
- **ε-greedy decay** 1.0 → 0.05 em 150 episódios
- **Hard mask** Q[cigarros] = −∞ (compliance Lei 9.294/96)
- **Replay buffer** 50.000 transições

**Treino:** 150 ep × 1095 steps × 1 seed = ~30 min em CPU 8 cores

**Speaker notes:** "BranchingDQN porque a ação é composta — separamos em duas cabeças decompostas, reduz parâmetros de saída de 105 para 26. Double DQN porque DQN clássico superestima. Hard mask para cigarros porque é compliance regulatória — não pode ser penalty leve, tem que ser proibição categórica."

---

### Slide 7 — Resultados V13

**Título:** Calendário operacional anual

**Métricas (hold-out 2024-2026):**
| Métrica | V13 | vs Sem-promo |
|---|---:|---:|
| Lucro | R$ 649k | +0.21% |
| Perdas | 1747 un | -0.87% |
| F1 evento médio | 0.208 | — |
| F1 Réveillon | **0.96** | — |
| F1 Mães | **0.31** | — |
| F1 Namorados | **0.22** | — |
| % cigarro | **0%** | ✓ compliance |

**Calendário 365 dias:**
- 109 campanhas anuais
- R$ 688 lucro adicional/ano

**Top 3 campanhas:**
1. 25/12-31/12: Gelo + Destilados combo — Réveillon
2. 11/03-14/03: Gelo + Destilados combo — Dia do Consumidor
3. 11/07-12/07: Chocolate + Vinho combo — fim de semana

**Speaker notes:** "Lucro só +0.21% sobre não fazer nada — número honesto, pequeno. Mas redução de perdas -0.87% é real. F1 Réveillon 0.96 mostra que o agente DOMINOU essa data. F1 Mães 0.31 — destravou comparado a versões anteriores que zeravam. E 0% de cigarro garantido por mask. O calendário é o entregável real para o dono do posto."

---

### Slide 8 — V14 (Pesquisa) — 5 Técnicas SOTA

**Título:** Tentativa rigorosa de superar V13

**Implementamos 5 técnicas state-of-the-art em paralelo:**

| Técnica | Origem |
|---|---|
| **PrioritizedReplayBuffer** | Schaul et al. 2016 (DeepMind) |
| **Curriculum Learning** | Bengio et al. 2009 |
| **Action Mask Dinâmico** | Huang & Ontañón 2020 |
| **DRCR Reward** | Alibaba 2019 (arXiv:1912.02572) |
| **Multi-seed Ensemble** | Wiering & van Hasselt 2008 |

**Treino:** 5 seeds em paralelo × 150 ep = ~1h wall time

**Resultado-chave (caixa destacada):**
> **Variance entre 5 seeds: R$ 183 (0.03%)** — 38× menor que single-seed. Ensemble eliminou dependência de sorte estatística.

**Trade-off real:**
- V14 GANHA: Réveillon 0.20 (vs 0.03 nesta avaliação)
- V14 PERDE: Mães 0 (vs 0.31), Namorados 0 (vs 0.22)
- Curriculum learning Fase 1 sem K_EVENTO "desensinou" datas-presente

**Speaker notes:** "V14 é nossa contribuição metodológica. Implementamos 5 técnicas publicadas em papers reais. O resultado mais importante: a variance entre seeds caiu de R$ 7k para R$ 183. Isso significa que se vocês me pedirem para retreinar amanhã, dá o mesmo número. V13 não dá essa garantia — deu sorte estatística com uma seed específica."

---

### Slide 9 — Limitações Honestas

**Título:** O que ainda limita o modelo

**4 limitações estruturais:**

1. **Sem teste A/B real** — Elasticidade vem de literatura + 3 datasets físicos. Real só com piloto no posto.

2. **Dados por categoria, não por SKU** — Não distingue Chocolate Lacta de Nestlé. ERP do posto ainda não exportou.

3. **Catálogo incompleto** — Espumante, perfume, flores não estão no env. F1 estrutural=0 para Dia da Mulher.

4. **Variance estocástica RL** — V13 deu sorte com seed 42 em Mães/Namorados. V14 ensemble corrige.

**Tentativas que NÃO funcionaram (refutadas pela pesquisa):**
- ❌ CondPolicyDQN (decomposição condicional) — destrava diversidade mas perde eventos
- ❌ Curriculum learning — mata datas-presente
- ❌ K_EVENTO_PRESENTE=1000 — desbalanceia outros 13 termos
- ❌ Offline RL (CQL/IQL) — sem dataset de promoção histórica
- ❌ LSTM — catastrophic forgetting

**Speaker notes:** "Estamos sendo bem honestos aqui. Tentamos várias coisas que não funcionaram. Cada falha está documentada com hipótese, diagnóstico e veredito. O gargalo real do projeto não é mais algoritmo — é dados. Quando o ERP exportar SKU detalhado, a próxima iteração pode destravar essas datas."

---

### Slide 10 — Próximos Passos

**Título:** Roadmap baseado em pesquisa real

**3 caminhos validados por papers de produção:**

1. **CausalImpact loop** (Google) — cada campanha real vira experimento natural. Calibra elasticidade empírica do posto sem precisar A/B clássico.

2. **Open Bandit Pipeline + Doubly Robust** (Saito et al. 2021) — validar política offline antes do A/B real. Defensável academicamente.

3. **V14 ensemble com dados SKU** — quando ERP exportar, retreinar V14 com chocolate Lacta vs Nestlé separados. Próxima banca: F1 Mães/Namorados devem subir.

**Cases reais de produção citados:**
- **Alibaba 2019** (DDPG dynamic pricing): A/B real, bateu humanos
- **Freshippo KDD 2021** (markdown perecíveis): paper mais próximo do nosso problema
- **Walmart 2021** (Edelman finalist): +21% sell-through em clearance

**Speaker notes:** "Pesquisamos o que Alibaba, Amazon, Walmart e Stitch Fix fazem em produção. CausalImpact do Google é a próxima ferramenta crítica — permite medir uplift real sem A/B clássico, usando produtos não-promovidos como controles sintéticos."

---

### Slide 11 — Conclusão

**Título:** O que entregamos

**3 entregáveis:**

1. **Modelo V13 em deploy** com 0% cigarro garantido por hard mask, calendário anual de 109 campanhas

2. **Metodologia V14** com 5 técnicas SOTA validadas, ensemble robusto (variance 38× menor), documentação científica honesta

3. **Pipeline reproduzível** publicado em [github.com/viniciusrap/projeto-rl](https://github.com/viniciusrap/projeto-rl) + [insper-classroom/capstone-lv-project](https://github.com/insper-classroom/capstone-lv-project)

**Mensagem final (caixa destacada):**
> "RL otimizou corretamente dado o ambiente. O próximo gargalo é DADOS — vendas detalhadas por SKU e catálogo expandido."

**Speaker notes:** "Não estamos prometendo +X% lucro porque sem A/B real não dá. Estamos entregando um pipeline calibrado, validado em hold-out, com diagnóstico rigoroso do que destrava e o que não destrava. A banca pode confiar nos números porque a metodologia é defensável."

---

## Formato dos slides

- **Fonte:** sans-serif moderna (Inter, Roboto, ou similar)
- **Cores:** azul corporate (#2563eb) + cinzas neutros + verde (#059669) para ✓ e vermelho (#dc2626) para ✗
- **Layout:** títulos à esquerda, conteúdo abaixo
- **Tabelas:** bordas finas, alinhamento à direita para números
- **Sem clip-art ou emojis decorativos** — só ✓/✗ em tabelas
- **Logo do Insper** no canto inferior esquerdo de cada slide (se possível)

## Munição para arguição (incluir nas notas finais)

### "Por que não usaram LSTM/Transformer?"
> LSTM testamos em V11.8 — catastrophic forgetting com episódios de tamanho variável. Decision Transformer requer dataset com return-to-go variado, e nosso histórico tem só `a=0` (sem promo) em 99% dos turnos. Refutados pela pesquisa.

### "Por que não fizeram A/B real?"
> O dono do posto ainda não autorizou piloto. CausalImpact é nossa proposta para calibrar elasticidade empírica usando produtos não-promovidos como controles sintéticos, sem precisar de A/B clássico.

### "Por que action mask em vez de penalty?"
> Para compliance regulatória (cigarros pela Lei 9.294/96), penalty=30 não é categórico. Em 18% dos turnos do V12 base, agente ainda tentava promover cigarro. Hard mask = garantia 100%, e validamos empiricamente +6.28% reward por economia de penalty desperdiçada.

### "Por que ensemble e não single-seed?"
> V12.1 deu F1 Mães 0.31 por sorte de seed específica. Outras 4 seeds retreinadas com mesmos hiperparâmetros não reproduziram. Ensemble de 5 seeds entrega resultado estável (std R$ 183) — defensável academicamente.

### "Como vocês validam que a política é boa?"
> Hold-out temporal 2024-2026 (período não visto no treino). 4 métricas: lucro vs sem-promo, perdas vs sem-promo, F1 timing por categoria, F1 evento por data comercial. Comparamos com 4 baselines: sem promoção, promoção aleatória, sempre combo, V12 base.

### "Qual a contribuição original?"
> 1) Pipeline calibrado em 6 anos de dado real do posto + 3 datasets físicos para elasticidade. 2) Diagnóstico rigoroso da "política tosca" do V10. 3) Implementação de 5 técnicas SOTA com ensemble robusto. 4) Roadmap fundamentado em papers de produção (Alibaba, Walmart, Stitch Fix).

### "Por que vocês acham que vai funcionar em produção?"
> Não afirmamos isso. Afirmamos que o pipeline está calibrado, validado em hold-out e que o gargalo NÃO é mais algorítmico (V14 prova isso com 5 técnicas SOTA testadas). O próximo passo obrigatório é piloto no posto para medir elasticidade real.

---

## Output desejado

Crie a apresentação em PowerPoint (.pptx) com:
- 11 slides totais
- Speaker notes detalhadas em cada slide
- Visual limpo e profissional
- Tabelas formatadas
- Mínimo de texto no slide visível
- Times: ajustar para que apresentação caiba em 10 minutos (~55 segundos por slide útil)
