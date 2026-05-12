# Próximos datasets para baixar (loja física)

Decisão de 12/05/2026: Olist removido como prior (e-commerce ≠ loja física).
Substituído por 3 datasets de loja física com cobertura específica para
diferentes categorias do posto.

## Status atual

| Dataset | Tipo | Cobertura | Status |
|---|---|---|---|
| ~~Olist~~ | ~~e-commerce BR~~ | ~~todas~~ | **❌ descontinuado** |
| Dunnhumby Complete Journey | Supermercado USA | bebidas, snacks, chocolate, café | ✓ baixado |
| **Iowa Liquor Sales** | Lojas físicas USA álcool | bebidas alcoólicas (gold standard) | ⏳ baixar |
| **Walmart M5 Forecasting** | Supermercado USA físico | todas (3000 SKUs) | ⏳ baixar |
| **Tesco Grocery 1.0** | Supermercado UK físico | chocolate, doces, alimentos | ⏳ baixar |

---

## 1. 🍷 Iowa Liquor Sales (PRIORITÁRIO)

**Por que:** O melhor dataset público de vendas de **álcool em loja física**
com 12+ anos de granularidade diária. Vai validar nossas medições do posto
em datas comerciais (Dia das Mães, Pais, Namorados, Réveillon).

**Conteúdo:**
- ~28 milhões de transações
- Lojas físicas estaduais do Iowa, EUA (state-run liquor stores)
- 2012-2024 (12 anos!)
- Granularidade: data, loja, produto, volume, preço
- Categorias: vodka, whisky, vinho, cerveja, espumante, etc.

**URL Kaggle:** https://www.kaggle.com/datasets/residentmario/iowa-liquor-sales

**Tamanho:** ~1.7 GB

**Como baixar:** mesmo procedimento do Olist:
1. Login Kaggle (já tem conta)
2. Acessar URL acima
3. Clicar "Download" (zip ~1.7GB)
4. Extrair em `data/raw_iowa/`
5. Me avisar "iowa baixado"

**Processamento (já planejado):**
- Script: `processar_iowa_liquor.py` (criar depois)
- Extrai uplift por (categoria álcool × data comercial × ano)
- Cobre 12 anos = ~12 amostras por evento (estatísticamente robusto)
- Cruza com `uplift_real_posto_agregado.csv` para confirmar/refutar
- Substitui o prior Olist no `calibrar_v2.py`

**O que vai destravar:**
- Validar uplift de destilados em Mães/Namorados (1.49× e 3.34× medido com N=1 ano no posto)
- Confirmar uplift de vinho em outubro/Halloween (Dunnhumby mostrou 2.17×)
- Padrão pré-feriado de bebidas alcoólicas (planejamento 1-2 semanas antes)

---

## 2. 🛒 Walmart M5 Forecasting

**Por que:** Supermercado físico com 3000 SKUs reais × 10 lojas × 5 anos
+ flags de evento/feriado. Cobertura mais ampla de varejo físico geral.

**Conteúdo:**
- 30.490 séries temporais (SKU × loja)
- 1.913 dias de venda diária
- Categorias: FOODS, HOUSEHOLD, HOBBIES (cobre tudo, incluindo doces e chocolate)
- Flags: `event_name`, `event_type`, `snap_TX/CA/WI` (food assistance)
- 2011-2016 (5 anos)

**URL Kaggle:** https://www.kaggle.com/datasets/aremoto/m5-forecasting-accuracy

**Tamanho:** ~600 MB

**Como baixar:** idem Iowa, extrair em `data/raw_walmart_m5/`

**Processamento planejado:**
- Script: `processar_walmart_m5.py` (criar)
- Foco nas séries de FOODS_3 (alimentos prontos), HOBBIES_1 (livro/jogos/chocolate)
- Cruzar com event_name para validar uplift por categoria × evento
- Quantificar magnitude em loja física americana

**O que vai destravar:**
- Validar uplift de snack/chocolate em datas como SuperBowl, Halloween,
  Thanksgiving — proxies de Copa, Halloween, Dia da Criança no Brasil
- Quantificar "efeito SNAP" (cupons de comida) — análogo ao "dia da folha"
  no posto, que pode importar

---

## 3. 🍫 Tesco Grocery 1.0 (CHOCOLATE/DOCES)

**Por que:** Cobertura específica de **chocolate e doces em loja física**
(supermercado UK). Resposta à sua pergunta: "tem dataset de chocolate?"

**Conteúdo:**
- 4 anos de transações reais (anonimizadas)
- Loja física UK (rede Tesco)
- 411 áreas (granularidade geográfica)
- Inclui flag "1_year_purchases" com chocolate e doces específicos

**URL:** https://figshare.com/articles/dataset/Tesco_grocery_1_0/7796666
(Figshare, **acesso gratuito sem login**)

**Tamanho:** ~2 GB

**Como baixar:**
1. Acessar URL acima
2. Não precisa login — clicar "Download all"
3. Vai baixar `Tesco_grocery_1_0.zip`
4. Extrair em `data/raw_tesco/`
5. Me avisar "tesco baixado"

**Processamento planejado:**
- Script: `processar_tesco_grocery.py`
- Foco em categorias `Chocolate confectionery`, `Sweets`, `Biscuits`
- Cruzar com calendário UK (Easter, Mother's Day, Christmas)
- Inferir padrão de chocolate em datas comerciais físicas

**O que vai destravar:**
- Confirmar uplift de chocolate em Páscoa/Mães em LOJA FÍSICA
- Calibrar magnitude correta para nosso prior (em vez de Olist 2.5×, talvez 1.5×)
- Validar separação chocolate_premium vs chocolate_impulso que fiz no V11

---

## 4. 🔍 Reforço do Google Trends (intenção de busca)

Vinicius destacou: **intenção de busca importa também para loja física**.
Concordo. Para o Dia das Mães, a pessoa busca "presente dia das mães" no
Google, depois pode comprar tanto online quanto em loja física.

**Plano para Google Trends:**
1. Já temos 7 séries (cerveja, vinho, espumante, whisky, chocolate, refrigerante, energético)
2. Adicionei termos novos: "promoção cerveja", "promoção chocolate", "cerveja em oferta", "chocolate em oferta"
3. Tentar coletar **panetone, sorvete, salgadinho** (verão/copa)
4. **CRÍTICOS:** "presente dia das mães", "presente dia dos namorados", "presente dia dos pais" (intenção)

**Como funcionar:** o coletor é idempotente (pula termos já coletados).
Quando o rate-limit do Google liberar (geralmente 24h+), rodar:

```powershell
.venv\Scripts\python.exe coletar_google_trends.py
```

**Importância para V11:** os termos de "intenção de presente" são os que mais
preveem aumento real em loja física antes das datas. Por isso são prioritários.

---

## 5. 🏆 Copa do Mundo — análise dos próprios dados do posto

Vinicius perguntou: "como pegar dados de Copa?"

**Boa notícia:** já temos **Copa 2022 inteira nos dados do posto** (nov-dez/2022).
Não precisa dataset externo para isso!

**Já feito (script `analisar_copa_2022_posto.py`):**

| Dia | Categoria | Uplift medido no POSTO |
|---|---|---:|
| 24/11/2022 (estreia BR×Sérvia) | Gelo | **4.50×** |
| 24/11/2022 (estreia BR×Sérvia) | Cerveja | **2.77×** |
| 09/12/2022 (eliminação) | Isotônico | **2.12×** |
| 09/12/2022 (eliminação) | Água | **1.56×** |
| 09/12/2022 (eliminação) | Energético | **1.35×** |
| 09/12/2022 (eliminação) | Cerveja | **1.29×** |
| Período inteiro Copa | Média | **0.97×** (sem efeito!) |

**Conclusão:** Copa NÃO move o posto durante todo o período. Move APENAS
nos **dias exatos de jogo**. Isso confirma que o calendário atual está
correto em focar nas datas específicas dos jogos.

Para Copa 2026 (que ainda vai acontecer):
- O calendário comercial já tem datas (abertura 11/06, jogos prováveis)
- O agente V11 já promove cerveja na semana da Copa (vimos no calendário V3)
- Quando 2026 acontecer, coletar dados em tempo real do posto para recalibrar

---

## Ordem sugerida de download

Por importância × tempo de baixar:

1. **Iowa Liquor Sales** (1.7GB) — gold standard para álcool
2. **Tesco Grocery** (2GB) — cobre chocolate específico
3. **Walmart M5** (600MB) — varejo físico geral, último

Total: ~4.3 GB. Recomendo baixar 1 por vez para testar cada um.

**Quando você baixar qualquer um deles, me avisa e eu rodo o
processamento + cruzamento com dados do posto.**

---

*Atualizado em 12/05/2026 após decisão de descontinuar Olist e priorizar
datasets de loja física para validação correta do prior.*
