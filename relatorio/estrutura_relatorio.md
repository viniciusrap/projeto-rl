# Relatório do Projeto de RL — Estrutura Sugerida (entrega 17/05/2026)

> Este é o esqueleto do relatório. Cada seção tem objetivo, conteúdo
> sugerido e fontes diretas dos artefatos já gerados no projeto.

---

## 0. Resumo executivo (1 página)

**Conteúdo:**
- Problema: posto de conveniência perde dinheiro com vencimento, ruptura e oportunidade perdida
- Solução proposta: agente de RL (Double DQN) que decide promoções
- Resultado: V10 reduz perdas -39% sem ganhar lucro absoluto; V11 expande catálogo para 18-20 categorias e introduz datas comerciais
- Contribuição metodológica: validação por timing (F1 por contexto e por evento) — descobre limitações invisíveis ao reward
- Limitação: validação final exige teste A/B in loco (próximo passo)

**Métricas-chave:**
- V10: R$ 26.743 lucro vs R$ 26.637 sem-promo (+0,4%); 3,0 vs 4,9 perdas (-39%)
- V11: 16 campanhas em 60 dias com R$ 571 de lucro adicional
- Correlação cobertura × F1 = 0,75 (limitação é do catálogo, não do algoritmo)

---

## 1. Introdução (2 páginas)

### 1.1 Contexto do negócio
- Auto Posto Parque Viana Ltda — Barueri/SP
- Mercado de conveniência: trade-off venda × vencimento × ruptura
- Decisão do dono é intuitiva, sem otimização sistemática

### 1.2 Justificativa para RL
- Decisão sequencial sob incerteza → MDP natural
- Estado evolui (estoque, validade, calendário, clima)
- Trade-off curto/longo prazo
- Métodos clássicos (regressão de elasticidade, OR) não capturam dinâmica
- Escolha: Double DQN do zero + PPO (SB3) como baseline

### 1.3 Objetivos
- Construir agente que aprende quando promover qual produto
- Validar a capacidade do agente em identificar timing (não só magnitude)
- Diagnosticar limitações e propor caminho para deployment real

### 1.4 Contribuições principais
1. **Validação por timing** com ground truth nos dados (Seção 7.4 do V10)
2. **Iteração documentada V1→V10** (10 versões, cada uma resolvendo um diagnóstico)
3. **Expansão para V11** com calendário comercial brasileiro embutido
4. **Diagnóstico quantitativo** da limitação V11 (r=0,75 entre cobertura e F1)

---

## 2. Revisão da literatura (1-2 páginas)

### 2.1 RL em otimização de preços/promoções
- Bertsimas & Perakis (2006) — dynamic pricing
- Aplicações de DQN/PPO em varejo (revisão breve)

### 2.2 Elasticidade promocional vs elasticidade de preço
- Bijmolt, van Heerde & Pieters (2005) — meta-análise de 1.851 elasticidades
- Por que a distinção importa: literatura inicial usava -1,2 a -1,8 (elast preço); promocional é -2,5 a -4,5

### 2.3 Reward shaping
- Ng, Harada & Russell (1999) — Policy Invariance Under Reward Transformations
- Justifica `bonus_timing` e `bonus_evento_comercial` como aceitos academicamente

---

## 3. Coleta e tratamento de dados (2 páginas)

### 3.1 Dados internos do posto
- `venda_por_dia.xlsx` — 6 anos por categoria × turno (147k registros)
- `venda_do_mes.xlsx` — preços/custos/margens (mar/26)
- `descarte_produto.xlsx` — descarte (apenas mar/26, 19 registros)
- `produtos_nao_vendidos/` — 52 xlsx mensais, 850 SKUs em 77 categorias (4 anos)

### 3.2 Priors externos coletados
- **Calendário comercial BR**: 278 eventos 2016-2027 (`gerar_calendario_comercial.py`)
- **Google Trends BR**: 7 séries de termos relevantes (5 anos)
- **IBGE PMC**: sazonalidade macro do varejo (Dez +20%, Fev -11%)
- **Olist (Kaggle)**: uplift de presente por categoria em datas comerciais
- **Dunnhumby Complete Journey**: padrão temporal de PROMOÇÃO no varejo americano

### 3.3 Tratamento e filtragem
- Filtro automotivo: 12 categorias / 131 SKUs removidos (`filtrar_conveniencia.py`)
- Catálogo final: 65 categorias / 719 SKUs de conveniência
- Mapeamento para 18 → 20 categorias agregadas do modelo

---

## 4. Formulação do problema como MDP (2 páginas)

### 4.1 Estado $s_t \in \mathbb{R}^{122}$
- Calendário (35): turno, dia, mês, dia do mês, evento próximo (10 buckets), dias até evento
- Clima (2): temperatura normalizada, Δ 7d
- Padding contextual (9)
- Por categoria (4 × N): estoque_norm, validade_rest, fraco_flag, promo_ant

### 4.2 Ação $a_t \in \{0..N\} \times \{0..4\}$
- MultiDiscrete: (qual produto, qual intensidade)
- Total: 95 ações para N=18 ou 105 ações para N=20

### 4.3 Transição
- $\lambda_i = D_i \cdot F^{dia} \cdot F^{turno} \cdot F^{mês} \cdot F^{clima}(T) \cdot F^{evento}(d) \cdot F^{promo}(a)$
- $q_i \sim \text{Poisson}(\lambda_i)$
- Estoque, validade evoluem deterministicamente
- Reposição implícita: mantém 7 dias de cobertura

### 4.4 Recompensa (8 termos)
$r_t = L_t - V_t - R_t - D_t + G_t + B^{tim}_t + B^{evt}_t + B^{pad}_t - I_t$

### 4.5 Episódio
- V10: 90 turnos (1 mês abstrato)
- V11: 1095 turnos (1 ano calendário real)
- Train: 2020-06 a 2024-06; Validação hold-out: 2024-07 a 2026-04

---

## 5. Implementação (2 páginas)

### 5.1 Ambiente Gymnasium
- `env_v2.py` — `ConvenienceStoreEnvV2`
- Calibração via `data/calibracao_v2.json` (gerado por `calibrar_v2.py`)
- Carrega temperatura histórica + calendário comercial + priors

### 5.2 Agente — Branching DQN
- Decomposição aditiva: $Q(s,(p,i)) = V(s) + A_p(s,p) + A_i(s,i)$
- Reduz parâmetros de $Q(s, a_{flat})$ com 95 saídas para $V + A_p (N+1) + A_i (5) = N+7$ saídas
- Double DQN, HuberLoss, replay buffer 50k, ε-decay
- Arquitetura: `Linear(122 → 256 → 128) → [V(1), A_p(N+1), A_i(5)]`

### 5.3 Treinamento
- 150 episódios × 1 seed (CPU, ~14 min) para V11 (versão estável)
- 50 episódios × 1 seed (~2 min) para validação rápida do pipeline
- Versões futuras: 200ep × 3 seeds

---

## 6. Resultados V10 (3 páginas)

### 6.1 Iteração V1→V10 — diagnósticos rigorosos
Tabela com 10 versões e o problema corrigido em cada uma.

### 6.2 Resultados quantitativos (V10)
- Reward, lucro, perdas vs baselines (Aleatória, Sem-promoção, PPO)
- Curvas de aprendizado (5 seeds × 500 episódios)
- Distribuição de ações: 67% sem-promo / 33% combo

### 6.3 Validação por timing (Seção 7.4 — contribuição principal)
- Ground truth: 30% percentil do fator combinado por produto
- F1 por produto:
  - Gelo 98,7% (sazonalidade extrema → aprende perfeitamente)
  - Refrigerante 29,9% (sazonalidade fraca → quase aleatório)
- **Descoberta crítica**: política "colapsa" em regra simples (segunda/quarta/quinta à noite)
- Diagnóstico: não é falha do algoritmo, é simplicidade do MDP

### 6.4 Robustez à elasticidade
- Variação de 0,5× a 2,0× da elasticidade Bijmolt
- Política do agente: idêntica em todos os cenários (66/34)
- Reward varia apenas ±2% → política robusta

---

## 7. Resultados V11 (3 páginas)

### 7.1 Mudanças vs V10
Tabela comparativa em 12 dimensões.

### 7.2 Política aprendida (150 ep)
- Promove 77,8% das categorias (não colapsa)
- Usa 75% das intensidades
- F1 evento médio: 0,224

### 7.3 F1 por evento comercial
- Réveillon: 0,61
- Véspera de Natal: 0,46
- Dia das Crianças: 0,26
- Dia dos Pais: 0,24
- **Dia das Mães / Mulher / Namorados: 0** ← falha sistemática

### 7.4 Diagnóstico quantitativo da falha
- **Correlação Pearson cobertura × F1 = 0,751**
- Eventos com cobertura > 60%: F1 médio 0,435
- Eventos com cobertura ≤ 60%: F1 médio 0,140
- **Conclusão estatística**: a falha do V11 nesses eventos é matemática da modelagem do MDP (catálogo limitado), não limitação algorítmica

### 7.5 Calendário operacional V3
- 16 campanhas em 60 dias
- R$ 571 lucro adicional
- 9 cerveja, 6 cigarro Philip Morris, 1 água
- Cobertura inteligente da Copa 2026 (3 semanas seguidas em jogos do Brasil)

---

## 8. Análise temporal e operacional (1 página)

### 8.1 Estoque parado — visão dinâmica
- Posto reduziu valor parado em -47,6% em 4 anos (R$ 20k → R$ 10k)
- 253 SKUs cronicamente parados (>80% snapshots) → candidatos a descontinuar
- 118 SKUs eventuais → alvos potenciais de promoção
- Categorias com 100% problema crônico: destilados, vinho, whisky, vodka

### 8.2 Diagnóstico do potencial real
- Modelo identifica padrões válidos (cerveja seg-qui, eventos esportivos)
- Tamanho do impacto: pequeno em R$ absoluto (consistente com tese)
- Ganho real virá de **expansão do catálogo** + **calibração com dados por SKU**

---

## 9. Limitações e próximos passos (1 página)

### 9.1 Limitações reconhecidas
1. **Off-policy evaluation problem**: simulador calibrado em vendas SEM promoção, elasticidade da literatura
2. **Catálogo agregado por categoria**: vendas detalhadas por SKU ainda não disponíveis (Fase 2.5)
3. **Cupom fiscal não disponível**: combos são heurísticos, sem Apriori validado
4. **Descarte limitado**: 19 registros (1 mês) → α por SKU não calibrável
5. **Validação final exige A/B in loco**: até lá, números são "dentro da suposição"

### 9.2 Roadmap pós-entrega
- Fase 1 (dados): obter exports do ERP do posto
- Fase 2 (calibração detalhada): re-rodar `calibrar_v2.py` com dados completos
- Fase 3 (treino refinado): 200ep × 3 seeds × catálogo expandido
- Fase 4 (output operacional): dashboard Streamlit consumindo `calendario_v3.json`
- Fase 5 (teste A/B): 8-10 campanhas pré-registradas, medir uplift real
- Fase 6 (re-calibração): substituir elasticidade da literatura por medida

---

## 10. Conclusão (1 página)

### 10.1 O que o trabalho contribui
1. **Pipeline completo e reprodutível** (5 scripts + 1 notebook + ~150KB JSON de calibração)
2. **Validação metodológica** que vai além de métricas de reward (F1 por timing e por evento)
3. **Diagnóstico quantitativo das limitações** (r=0,75 entre cobertura e F1)
4. **Honestidade científica**: ganho real é -39% perdas, não "+28% reward"
5. **Integração de 5 fontes externas** (calendário BR + Trends + IBGE + Olist + Dunnhumby)

### 10.2 O que aprendemos
- RL **pode** otimizar promoções em varejo de conveniência
- Mas a **modelagem do MDP** é mais crítica que o algoritmo
- Sem **dados detalhados por SKU** e **teste A/B real**, qualquer número absoluto é prior
- Iteração documentada (V1→V10 e V10→V11) é o caminho honesto para diagnóstico

---

## Apêndices

### A. Bibliografia (Bijmolt 2005, Ng 1999, Mnih 2015, etc.)

### B. Tabelas detalhadas (calibração por categoria, F1 por evento, etc.)

### C. Estrutura do código (organização do repositório)

### D. Reproduzibilidade (como rodar o pipeline)

---

# Artefatos prontos para usar no relatório

## Tabelas
- `results/comparacao_politicas.csv` (V10 vs baselines)
- `results/v11/comparacao_politicas_v11.csv` (V11 vs baselines)
- `results/v11/comparacao_v10_v11.csv` (V10 vs V11)
- `results/v11/diagnostico_eventos_perdidos.csv` (cobertura × F1)
- `results/v11/validacao_eventos_v11.csv`
- `results/v11/categorias_problema_cronico.csv`
- `results/validacao_metricas.csv` (V10 timing por produto)

## Gráficos
- `results/curvas_aprendizado.png` (V10)
- `results/comparacao_politicas.png` (V10)
- `results/validacao_heatmap.png` (V10 timing)
- `results/analise_robustez.png` (V10 robustez à elasticidade)
- `results/v11/curvas_aprendizado_v11.png`
- `results/v11/comparacao_v10_v11.png`
- `results/v11/evolucao_estoque_parado.png`

## Texto pronto (parágrafos a copiar)
- CLAUDE.md seção "POLÍTICA TOSCA" — diagnóstico V10 honesto
- CLAUDE.md seção "DIAGNÓSTICO V11" — correlação r=0,75
- CLAUDE.md seção "MDP FORMAL DO V11" — formulação matemática

---

*Última atualização: 12/05/2026 — esqueleto para o relatório de entrega 17/05.*
