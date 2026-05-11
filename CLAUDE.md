# CONTEXTO COMPLETO — Projeto RL Conveniência Viana
## Documento de briefing para nova sessão de trabalho

---

## IDENTIDADE DO PROJETO

**Nome:** Otimização de Promoções com Reinforcement Learning  
**Alunos:** Vinicius Rocha e Luigi Zema  
**Curso:** Reinforcement Learning — Insper (Brasil)  
**Empresa real:** Auto Posto Parque Viana Ltda — Viana, ES  
**Objetivo:** Agente RL que decide qual promoção aplicar a cada turno da loja de conveniência do posto para maximizar lucro e minimizar desperdício por vencimento  
**Status atual:** Planejamento concluído, parâmetros calibrados, aguardando planilhas completas para montar o notebook final  

---

## O QUE O AGENTE FAZ

A cada turno (manhã / tarde / noite), o agente observa o estado atual da loja e escolhe uma das 5 ações promocionais. O ciclo de um episódio é 1 semana = 7 dias × 3 turnos = 21 passos.

**Entrada (estado):** 41 features numéricas  
**Saída (ação):** 1 inteiro de 0 a 4  
**Objetivo:** maximizar lucro acumulado no episódio  

---

## ARQUITETURA DO MODELO

### Algoritmo
- **Principal:** Double DQN implementado do zero em PyTorch
- **Comparação:** PPO via Stable-Baselines3 (já conhecem do curso)
- **Por que DQN:** 5 ações discretas, episódios curtos (21 passos), implementação própria demonstra domínio do algoritmo
- **Melhorias já implementadas:** HuberLoss (SmoothL1Loss), target network, replay buffer, ε-greedy decrescente, 5 seeds independentes, CSV logging

### Hiperparâmetros do DQN
```python
lr = 1e-3
gamma = 0.95
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.990  # calibrado para eps≈0.007 ao fim de 500 episódios
batch_size = 64
target_update = 10  # passos entre sincronizações
buffer_size = 10_000
hidden1 = 128
hidden2 = 64
N_EPISODIOS = 500  # aumentar para 2000+ no treino final
N_RUNS = 5  # seeds independentes para IC confiável
```

### Estado — 41 features
```
[0:3]   turno one-hot (manhã, tarde, noite)
[3:10]  dia da semana one-hot (seg=0 ... dom=6)
[10:22] mês one-hot (jan=0 ... dez=11)  ← NOVO, crítico para sazonalidade
[22]    temperatura normalizada [0,1]    ← dado real Open-Meteo
[23:29] estoque normalizado por produto (÷ ESTOQUE_INICIAL × 1.2)
[29:35] validade normalizada por produto (÷ VALIDADE_TIPICA)
[35:41] promoção ativa no turno anterior (binary por produto)
```

### Espaço de ações — 5 discretas
```
0 → sem promoção
1 → desconto 5%  no produto com maior estoque
2 → desconto 10% no produto com maior estoque
3 → combo: dois produtos complementares (ver PARES_COMBO)
4 → promoção de giro: produto com validade mais crítica
```

### PARES_COMBO (ação 3 — definitivo)
```python
# produto principal → produto complementar
PARES_COMBO = {
    0: 2,  # energético + refrigerante
    1: 3,  # gelo + água
    2: 5,  # refrigerante + sorvete
    3: 0,  # água + energético
    4: 2,  # cerveja + refrigerante
    5: 3,  # sorvete + água
}
fator_cross = 1.12  # efeito cross-sell em 2 produtos simultaneamente
```
**IMPORTANTE:** A ação 3 estava com bug — selecionava o mesmo produto que as ações 1 e 2 (argmax de estoque) e era dominada. Foi redesenhada para operar em par de produtos complementares.

---

## OS 6 PRODUTOS DO MODELO

### Decisão final (substituições em relação ao notebook original)
| # | Produto | Status | Motivo |
|---|---------|--------|--------|
| 0 | Energético | MANTER | 96% presença, padrão sexta/sábado forte |
| 1 | Gelo | MANTER | Padrão mais extremo: sáb 2.24×, dez 2.19× |
| 2 | Refrigerante | MANTER | 99% presença, mais consistente |
| 3 | Água | MANTER | 99% presença, maior margem (70%) |
| 4 | Cerveja Ambev | NOVO (era Sanduíche) | 98% presença, dom 1.46×, padrão fim de semana |
| 5 | Sorvete Kibon | NOVO (era Chocolate Nestlé) | 93% presença, maior receita/dia, sazonalidade verão |

### Por que removemos Sanduíche
Apenas 43% de presença nos dados (metade dos dias sem venda). Padrão não confiável para estimar parâmetros. Impossível calibrar DEMANDA_BASE com dados esparsos.

### Por que removemos Chocolate Nestlé
Volume de R$25k em 6 anos vs Sorvete Kibon com R$199k no mesmo período (4× mais). Sorvete tem padrão de fim de semana (1.60× no domingo) e sazonalidade de verão (1.46× em dezembro) — muito mais rico para o agente aprender.

---

## PARÂMETROS CALIBRADOS — TODOS COM FONTE REAL

**Ordem dos vetores:** [energético, gelo, refrigerante, água, cerveja, sorvete]

### Preços e margens (fonte: venda_do_mes.xlsx × venda_por_dia.xlsx)
```python
PRODUTOS    = ['energetico','gelo','refrigerante','agua','cerveja','sorvete']

PRECO_VENDA = np.array([17.86, 14.99, 8.29,  5.00,  7.55, 14.47])
CUSTO       = np.array([ 7.29,  7.40, 3.58,  1.49,  3.01,  8.78])
MARGEM      = np.array([10.57,  7.59, 4.72,  3.51,  4.54,  5.69])
# Margem %: [59.2%, 50.6%, 56.9%, 70.2%, 60.1%, 39.3%]
# Nota: água tem menor preço mas maior margem % — insight importante
# Nota: sorvete tem maior receita/dia mas menor lucro/dia por custo alto (R$8.78/un)
```

### Demanda base (fonte: cruzamento venda_por_dia ÷ preço médio)
```python
DEMANDA_BASE = np.array([5.3, 5.6, 10.7, 16.9, 9.4, 7.0])  # unidades/dia
# Notebook original tinha: [4, 3, 8, 10, 3, 5] — muito diferente
# Água: 16.9 un/dia (o dobro do que estava estimado)
```

### Lucro diário real (referência para avaliar o reward)
```python
# R$/dia: energético=55.85, gelo=42.40, refri=50.38, agua=59.43, cerveja=42.47, sorvete=39.56
# Água é o produto mais lucrativo por dia apesar de ser o mais barato por unidade
```

### Fator por dia da semana (fonte: venda_por_dia.xlsx — calculado)
```python
FATOR_DIA = np.array([
    # seg    ter    qua    qui    sex    sab    dom
    [0.761, 0.804, 0.847, 0.940, 1.164, 1.387, 1.096],  # energetico
    [0.406, 0.649, 0.704, 0.547, 0.744, 2.240, 1.710],  # gelo (EXTREMO no fim de semana)
    [0.883, 0.833, 0.896, 0.910, 0.999, 1.112, 1.366],  # refrigerante
    [0.949, 0.879, 0.933, 0.945, 1.028, 1.191, 1.074],  # agua
    [0.702, 0.707, 0.779, 0.825, 1.137, 1.392, 1.459],  # cerveja
    [0.794, 0.815, 0.867, 0.950, 0.872, 1.098, 1.604],  # sorvete
])
```

### Fator por turno (fonte: venda_por_dia.xlsx — em quantidade real)
```python
FATOR_TURNO = np.array([
    # manha   tarde   noite
    [1.105,  0.874,  1.021],  # energetico (pico manhã)
    [1.062,  0.951,  0.987],  # gelo
    [0.801,  1.090,  1.109],  # refrigerante (cresce à tarde/noite)
    [1.082,  0.997,  0.922],  # agua (pico manhã)
    [0.834,  1.253,  0.913],  # cerveja (pico tarde)
    [0.802,  1.185,  1.013],  # sorvete (pico tarde)
])
# NOTA: turnos 1-2=manhã, 3-4=tarde, 5-6=noite (colapsados de 6 para 3)
```

### Sazonalidade mensal (fonte: venda_por_dia.xlsx — calculado)
```python
FATOR_MES = np.array([
    # jan    fev    mar    abr    mai    jun    jul    ago    set    out    nov    dez
    [0.998, 1.100, 1.173, 1.089, 0.933, 0.811, 0.776, 0.848, 0.893, 0.967, 1.057, 1.355],  # energ
    [0.945, 1.032, 1.009, 0.950, 0.783, 0.895, 0.740, 0.852, 0.803, 0.835, 0.966, 2.191],  # gelo
    [1.086, 1.170, 1.181, 1.011, 0.892, 0.837, 0.800, 0.843, 0.999, 0.875, 1.032, 1.274],  # refri
    [1.027, 1.184, 1.214, 1.029, 0.852, 0.793, 0.751, 0.844, 1.034, 0.909, 1.083, 1.281],  # agua
    [0.921, 0.970, 1.013, 0.980, 0.908, 0.832, 0.903, 0.899, 1.133, 1.046, 1.160, 1.237],  # cerv
    [1.254, 1.246, 1.129, 0.864, 0.741, 0.628, 0.614, 0.751, 1.115, 1.032, 1.165, 1.461],  # sorv
])
# Gelo em dezembro = 2.19× a média anual (verão brasileiro)
# Sorvete em dezembro = 1.46× (correlacionado com calor)
```

### Estoque inicial (fórmula baseada em giro real)
```python
# CV real por categoria (coeficiente de variação das vendas diárias)
CV = np.array([0.645, 1.520, 0.514, 0.519, 0.690, 0.860])
# Gelo tem CV muito alto (1.52) por causa do padrão extremo de fim de semana

# Fórmula: demanda × (3 + CV × 4) dias de cobertura
ESTOQUE_INICIAL = np.array([21, 24, 53, 67, 56, 49])
# Mais variável = mais buffer de segurança
```

### Validade típica em turnos (preliminar — aguarda descarte histórico)
```python
VALIDADE_TIPICA = np.array([270, 18, 270, 270, 90, 30])
# energ(90dias×3), gelo(6dias×3), refri, agua, cerveja(30d×3), sorvete(10d×3)
# A calibrar com descarte histórico de 1 ano quando disponível
```

### Elasticidade-preço (substitui FATOR_PROMO — fonte: literatura econômica)
```python
# FATOR_PROMO foi removido e substituído por elasticidade-preço
# Bebidas em conveniências: literatura indica -1.2 a -1.8
ELASTICIDADE = np.array([-1.5, -1.3, -1.6, -1.2, -1.4, -1.5])
# energ, gelo, refri, agua, cerveja, sorvete

# Uso no step():
# desc_5pct  → ΔQ = Q × 1.5 × 0.05 = +7.5% demanda para energético
# desc_10pct → ΔQ = Q × 1.5 × 0.10 = +15.0% demanda para energético
# IMPORTANTE: o teste A/B real vai calibrar esses valores empiricamente

# O FATOR_PROMO original [1.0, 1.15, 1.30, 1.45, 1.40] foi descartado
# porque não tinha base teórica nem empírica
```

---

## FUNÇÃO DE RECOMPENSA

```python
# Componentes da recompensa por turno:
reward = lucro - pen_vencimento - pen_ruptura - pen_desconto + bonus_giro

# Detalhes:
lucro        = sum(vendas × (preco_efetivo - CUSTO))
pen_venc     = alpha × sum(perdas × CUSTO)           # alpha calibrar pelo descarte
pen_ruptura  = beta  × sum(rupturas × MARGEM × 0.5)  # beta = 1.5
pen_desconto = gamma × desconto_excessivo × 5.0       # desconto em produto sem estoque
bonus_giro   = delta × sum(vendas × (validade_pre_reset < 3) × MARGEM × 0.3)

# Pesos atuais (a calibrar com descarte histórico):
alpha = 2.0  # penalidade vencimento
beta  = 1.5  # penalidade ruptura
gamma = 0.5  # penalidade desconto excessivo
delta = 1.0  # bônus de giro de validade
```

---

## BUGS JÁ CORRIGIDOS NO CÓDIGO

### Bug 1 — Temperatura dupla (crítico)
**Problema:** `step()` gerava `temp_norm` para calcular demanda, mas `_get_obs()` gerava outro valor independente. O agente via temperatura diferente da que influenciou as vendas.

**Correção:**
```python
# Em __init__ e reset():
self._temp_norm = 0.5  # valor compartilhado

# Em step():
self._temp_norm = self.np_random.uniform(0.2, 0.9)  # armazena

# Em _get_obs():
temp = np.array([self._temp_norm])  # usa o mesmo valor
```

### Bug 2 — Cálculo errado de rupturas (crítico)
**Problema:** rupturas calculadas APÓS atualizar o estoque, superestimando a penalidade.

**Correção:**
```python
vendas   = np.minimum(demanda_real, self.estoque)
rupturas = np.maximum(demanda_real - vendas, 0)  # ANTES de atualizar estoque
self.estoque = np.maximum(self.estoque - vendas - perdas, 0)
```

### Bug 3 — Bonus de giro após reset de validade (crítico)
**Problema:** validade era resetada para produtos vencidos ANTES de calcular o bonus_giro, então produtos vendidos perto do vencimento nunca recebiam o bônus.

**Correção:**
```python
validade_pre_reset = self.validade.copy()  # salva antes do reset
self.validade = np.where(self.validade <= 0, VALIDADE_TIPICA.astype(float), self.validade)
bonus_giro = delta × sum(vendas × (validade_pre_reset < 3) × MARGEM × 0.3)
```

### Bug 4 — Ação 3 (combo) nunca aprendida (crítico)
**Problema:** FATOR_PROMO[3] = 1.20 < FATOR_PROMO[2] = 1.30, e ambas selecionavam o mesmo produto (argmax de estoque). Ação 3 era estritamente dominada.

**Correção:**
- FATOR_PROMO substituído por elasticidade
- Ação 3 redesenhada para operar em DOIS produtos usando PARES_COMBO
- Agora é diferenciada das ações 1 e 2

### Bug 5 — MSELoss instável (melhoria)
```python
# Antes:
self.criterion = nn.MSELoss()
# Depois:
self.criterion = nn.SmoothL1Loss()  # HuberLoss — mais estável para rewards em R$
```

### Bug 6 — Double DQN não implementado (melhoria)
```python
# Antes (vanilla DQN):
q_next = self.target_net(next_states).max(1)[0]

# Depois (Double DQN):
best_actions = self.q_net(next_states).argmax(1)
q_next = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
```

### Bug 7 — Epsilon decay mal calibrado (melhoria)
```python
# Antes: eps_decay=0.997 → ε≈22% após 500 episódios (muito exploração ainda)
# Depois: eps_decay=0.990 → ε≈0.7% após 500 episódios (calibrado)
```

---

## PLANILHAS DE DADOS — ESTRUTURA E USO

### 1. venda_por_dia.xlsx (PRINCIPAL — já disponível)
- **Período:** 22/06/2020 a 30/04/2026 (2.139 dias, 147.873 registros)
- **Estrutura:** Cada bloco começa com "Data: DD/MM/AAAA", seguido de linhas com [categoria, turno1_R$, turno2_R$, ..., turno6_R$]
- **Particularidade:** Até 6 turnos por dia (varia: alguns dias têm 3, outros 5)
- **Colapsamento necessário:** turnos 1-2→manhã, 3-4→tarde, 5-6→noite

**Código de parsing:**
```python
import pandas as pd
import numpy as np

df = pd.read_excel('venda_por_dia.xlsx', header=None)
records = []
current_date = None

for idx, row in df.iterrows():
    if pd.notna(row[1]) and 'Data:' in str(row[1]):
        date_str = str(row[1]).replace('Data:', '').strip()
        current_date = pd.to_datetime(date_str, dayfirst=True)
    elif current_date is not None and pd.notna(row[2]) and pd.notna(row[3]):
        categoria = str(row[2]).strip()
        for turno_idx, col in enumerate([3,4,5,6,7,8], start=1):
            val = row[col]
            if pd.notna(val) and float(val) != 0:
                records.append({
                    'data': current_date,
                    'categoria': categoria,
                    'turno': turno_idx,
                    'valor_venda': float(val)
                })

dfp = pd.DataFrame(records)
dfp['turno3'] = dfp['turno'].apply(lambda t: 'manha' if t<=2 else ('tarde' if t<=4 else 'noite'))
dfp['dia_semana'] = dfp['data'].dt.dayofweek  # 0=seg, 6=dom
dfp['mes'] = dfp['data'].dt.month
```

**O que calcular a partir daqui:**
```python
# FATOR_DIA por categoria:
daily = sub.groupby('data')['valor_venda'].sum().reset_index()
daily['dia_semana'] = pd.to_datetime(daily['data']).dt.dayofweek
fator_dia = (daily.groupby('dia_semana')['valor_venda'].mean() / 
             daily['valor_venda'].mean()).values

# FATOR_TURNO (em quantidade — precisa do preço médio do venda_do_mes):
sub_t = sub[sub['turno'] <= 3].copy()
sub_t['qtd_est'] = sub_t['valor_venda'] / preco_medio[cat]
fator_turno = (sub_t.groupby('turno')['qtd_est'].mean() / 
               sub_t['qtd_est'].mean()).values

# FATOR_MES:
daily['mes'] = pd.to_datetime(daily['data']).dt.month
fator_mes = (daily.groupby('mes')['valor_venda'].mean() / 
             daily['valor_venda'].mean()).values

# DEMANDA_BASE (em qtd/dia):
demanda_base = (sub.groupby('data')['valor_venda'].sum().mean() / preco_medio[cat])
```

### 2. venda_do_mes.xlsx (PREÇOS E MARGENS — já disponível)
- **Período:** Março/2026 (pode pedir outros meses para comparação)
- **Estrutura:** Blocos por categoria "Classificação produto: NOME", linhas com [produto, qtd, valor_venda, custo, margem_R$]
- **Uso principal:** calcular preço médio por unidade por categoria para converter venda_por_dia de R$ para quantidade

**Código de parsing:**
```python
df_mes = pd.read_excel('venda_do_mes.xlsx', header=None)
cats_stats = {}
current_cat = None
cat_items = []

for _, row in df_mes.iterrows():
    if pd.notna(row[0]) and 'Classificação produto:' in str(row[0]):
        if current_cat and cat_items:
            cats_stats[current_cat] = cat_items
        current_cat = str(row[0]).replace('Classificação produto: ', '').strip()
        cat_items = []
    elif current_cat and pd.isna(row[0]) and pd.notna(row[1]) and pd.notna(row[2]):
        try:
            qtd = float(row[2]); venda = float(row[3])
            custo = float(row[4]); margem = float(row[5])
            if qtd > 0 and venda > 0:
                cat_items.append({'qtd':qtd,'venda':venda,'custo':custo,'margem':margem})
        except: pass

# Preço médio por categoria:
for cat, items in cats_stats.items():
    df_c = pd.DataFrame(items)
    preco_medio[cat] = df_c['venda'].sum() / df_c['qtd'].sum()
    custo_medio[cat] = df_c['custo'].sum() / df_c['qtd'].sum()
    margem_pct[cat]  = df_c['margem'].sum() / df_c['venda'].sum()
```

**Mapeamento de nomes (venda_do_mes → venda_por_dia):**
```python
CAT_MAP = {
    'ENERGÉTICO':    'ENERGÉTICO',
    'GELO':          'GELO',
    'REFRIGERANTE':  'REFRIGERANTE',
    'AGUA':          'AGUA',
    'CERVEJA AMBEV': 'CERVEJA AMBEV',
    'SORVETE KIBON': 'SORVETE KIBON',
}
```

### 3. descarte_produto.xlsx (VENCIMENTOS — disponível 1 mês, pedir 1 ANO)
- **Disponível:** Março/2026 (2 eventos, 71 unidades, R$1.349 custo perdido)
- **Necessário:** 12 meses completos (mesmo período que as outras planilhas)
- **Estrutura:** [data, turno, categoria, produto, quantidade, custo_unitário, valor_venda, motivo, NF]
- **Motivo:** sempre "PERDA PRODUTO VENCIDO"

**Código de parsing:**
```python
df_d = pd.read_excel('descarte_produto.xlsx', header=None, skiprows=1)
df_d.columns = ['data','turno','categoria','produto','quantidade',
                'custo_unit','valor_venda','plano','obs']
df_d = df_d[df_d['categoria'].notna()].copy()
df_d['custo_total'] = pd.to_numeric(df_d['custo_unit'], errors='coerce') * \
                      pd.to_numeric(df_d['quantidade'], errors='coerce')
```

**O que calcular:**
```python
# Taxa de perda por categoria:
taxa_perda[cat] = custo_descarte[cat] / (receita_vendas[cat] + custo_descarte[cat])
# Março/2026: isotônico=12.9%, snack=9.0%, cerveja=7.7%, gelo/energ=0%

# Calibrar alpha (peso penalidade vencimento):
# Categorias com alta taxa → alpha maior
# VALIDADE_TIPICA → calibrar pela frequência real de descartes
```

**Descoberta importante:** Isotônico tem 12.9% de taxa de perda — considerar trocar Água por Isotônico no modelo se quiser que o agente aprenda gestão de vencimento de forma mais intensa.

### 4. produtos_nao_vendido.pdf (ESTOQUE PARADO — uso indireto)
- **Conteúdo:** SKUs com venda < 1 unidade no período, estoque atual, custo e preço
- **Uso:** identificar quais SKUs têm giro zero dentro de cada categoria; confirmar que as categorias escolhidas têm produtos parados que justificam a ação de giro
- **Descoberta:** Isotônico aparece TANTO no não vendidos (Gatorade Frutas Cítricas, 20 unidades) QUANTO no descarte — candidato a substituir Água no modelo

### 5. Temperatura histórica (BUSCAR — Open-Meteo API gratuita)
```python
# Uma chamada só, sem autenticação
import requests

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": -20.38,
    "longitude": -40.37,
    "start_date": "2020-06-22",
    "end_date": "2026-04-30",
    "daily": "temperature_2m_max",
    "timezone": "America/Sao_Paulo"
}
response = requests.get(url, params=params)
data = response.json()
df_temp = pd.DataFrame({
    'data': pd.to_datetime(data['daily']['time']),
    'temp_max': data['daily']['temperature_2m_max']
})
# Normalizar: temp_norm = (temp - temp.min()) / (temp.max() - temp.min())
```

**Depois de buscar:** cruzar por data com venda_por_dia para calibrar fator_clima() real em vez do valor aleatório atual.

---

## O CRUZAMENTO MAIS IMPORTANTE

**Problema:** venda_por_dia tem tudo em R$ de receita. O modelo precisa de QUANTIDADE de unidades.

**Solução:** dividir receita diária por categoria pelo preço médio da categoria (calculado do venda_do_mes).

```python
# Passo 1: preço médio por categoria (do venda_do_mes)
preco_medio = {cat: venda_total/qtd_total for cat, (venda_total, qtd_total) in ...}

# Passo 2: demanda diária em quantidade
demanda_qtd = receita_diaria / preco_medio[cat]

# Passo 3: FATOR_TURNO em quantidade (mais preciso que em R$)
# (em R$ o fator poderia ser distorcido se preços variam entre turnos)
fator_turno_qtd = qtd_por_turno / qtd_media

# Resultados reais obtidos:
# DEMANDA_BASE = [5.3, 5.6, 10.7, 16.9, 9.4, 7.0] un/dia
# Notebook original tinha: [4, 3, 8, 10, 3, 5] — muito diferente
# Água: 16.9 un/dia (69% maior que o estimado)
```

---

## PLANILHA DE CRUZAMENTO MENSAL RECOMENDADA

Gerar mensalmente a partir das 3 fontes para monitorar saúde do inventário:

```
gestao_categorias_MMAAAA.xlsx:
- categoria
- qtd_vendida           ← venda_do_mes
- receita_vendas        ← venda_do_mes
- margem_pct            ← venda_do_mes
- qtd_descartada        ← descarte_produto
- custo_descarte        ← descarte_produto
- taxa_perda_pct        ← calculado: custo_desc/(receita+custo_desc)
- skus_parados          ← produtos_nao_vendido (contagem)
- margem_liquida        ← calculado: receita×margem_pct - custo_descarte
```

---

## ESTRUTURA DO NOTEBOOK FINAL

```
Seção 0: Contexto e Motivação
  - Auto Posto Parque Viana, Viana/ES
  - Por que RL é a abordagem certa
  - Impacto financeiro potencial

Seção 1: Imports e Configuração

Seção 2: Análise Exploratória dos Dados (EDA) ← SEÇÃO MAIS IMPORTANTE
  - Parsing de cada planilha
  - Visualizações dos padrões encontrados
  - Justificativa dos 6 produtos com dados
  - Tabela de parâmetros × origem (cada vetor com sua fonte)

Seção 3: Formulação do MDP (seção formal antes do código)
  - Estado: 41 features com definição formal
  - Ações: 5 com elasticidade-preço
  - Função de recompensa com pesos calibrados
  - Dinâmica do episódio

Seção 4: Ambiente Gymnasium (ConvenienceStoreEnv)
  - Código com todos os parâmetros reais
  - Bugs 1-4 já corrigidos

Seção 5: Agente Double DQN
  - Rede neural, replay buffer, target net
  - Bugs 5-7 corrigidos

Seção 6: Treino com múltiplas seeds
  - 5 runs × 500 episódios
  - CSV logging em results/

Seção 7: Avaliação e Análise de Comportamento
  - DQN vs PPO (SB3) vs aleatório vs sem promoção
  - Curvas de aprendizado com IC (seaborn errorbar='sd')
  - Análise qualitativa: o que o agente aprendeu?
  - Quais ações prefere em cada contexto?

Seção 8: Conclusões e Próximos Passos
  - Limitações honestas
  - Proposta de teste A/B
  - Função get_recommendation() para uso real
```

---

## MUDANÇAS NECESSÁRIAS — 9 ITENS EM 3 NÍVEIS

### Nível 1 — Sem isso o agente não aprende nada útil
1. **Adicionar mês ao estado** (29→41 features): sem mês, o agente não aprende sazonalidade. Gelo em dezembro é 2.2× a média.
2. **Trocar FATOR_PROMO por elasticidade-preço**: único parâmetro sem base. Elasticidade da literatura é defensável.
3. **Corrigir ação 3 (combo)**: redesenhar para dois produtos com PARES_COMBO. Hoje é ação dominada.
4. **Atualizar 6 produtos e todos os parâmetros reais**: substituir sanduíche/chocolate, inserir vetores calibrados.

### Nível 2 — Aprende, mas com qualidade limitada
5. **Temperatura real (Open-Meteo)**: uma chamada de API, impacto em gelo e sorvete.
6. **VALIDADE_TIPICA e alpha pelo descarte histórico**: quando disponível, calibrar pela taxa real.
7. **ESTOQUE_INICIAL pela fórmula de giro**: demanda × (3 + CV × 4).

### Nível 3 — Necessário para ser funcional (não só acadêmico)
8. **Função de inferência**: `get_recommendation(dia, turno, mes, temp, estoque_atual)` → string em português.
9. **Protocolo de teste A/B**: semanas seguindo vs não seguindo o agente. Mede lucro, perdas, rupturas. Calibra elasticidade empiricamente.

---

## DESCOBERTAS-CHAVE DOS DADOS

1. **Gelo é o produto com padrão mais extremo:** sábado 2.24× média, dezembro 2.19×
2. **Água é o mais lucrativo:** 70% de margem, R$59/dia de lucro (maior do grupo)
3. **Sorvete engana:** maior receita bruta (R$100/dia) mas menor lucro (R$39/dia) — custo de R$8.78/un
4. **Isotônico tem 12.9% de taxa de perda** — candidato a substituir Água se quiser gestão de vencimento mais intensa
5. **Energético no notebook original custava R$9.90** — real é R$17.86 (quase o dobro)
6. **Água no notebook original era 10 un/dia** — real é 16.9 un/dia (69% maior)
7. **Cigarro (Souza Cruz + Philip Morris + JTI) = 62% do faturamento** mas só 10% de margem — não entra no modelo de promoções
8. **6 anos de dados:** 2.139 dias, 87 categorias, dados por turno — raro em projetos de RL aplicado

---

## AMBIENTE GYMNASIUM — CLASSE COMPLETA (versão com bugs corrigidos)

```python
N_PRODUTOS      = 6
PRODUTOS        = ['energetico','gelo','refrigerante','agua','cerveja','sorvete']
PRECO_VENDA     = np.array([17.86, 14.99, 8.29, 5.00, 7.55, 14.47])
CUSTO           = np.array([ 7.29,  7.40, 3.58, 1.49, 3.01,  8.78])
MARGEM          = PRECO_VENDA - CUSTO
VALIDADE_TIPICA = np.array([270, 18, 270, 270, 90, 30])  # em turnos — calibrar com descarte
ESTOQUE_INICIAL = np.array([21, 24, 53, 67, 56, 49])

ELASTICIDADE = np.array([-1.5, -1.3, -1.6, -1.2, -1.4, -1.5])

PARES_COMBO = {0:2, 1:3, 2:5, 3:0, 4:2, 5:3}

FATOR_TURNO = np.array([
    [1.105, 0.874, 1.021],
    [1.062, 0.951, 0.987],
    [0.801, 1.090, 1.109],
    [1.082, 0.997, 0.922],
    [0.834, 1.253, 0.913],
    [0.802, 1.185, 1.013],
])

FATOR_DIA = np.array([
    [0.761, 0.804, 0.847, 0.940, 1.164, 1.387, 1.096],
    [0.406, 0.649, 0.704, 0.547, 0.744, 2.240, 1.710],
    [0.883, 0.833, 0.896, 0.910, 0.999, 1.112, 1.366],
    [0.949, 0.879, 0.933, 0.945, 1.028, 1.191, 1.074],
    [0.702, 0.707, 0.779, 0.825, 1.137, 1.392, 1.459],
    [0.794, 0.815, 0.867, 0.950, 0.872, 1.098, 1.604],
])

FATOR_MES = np.array([
    [0.998,1.100,1.173,1.089,0.933,0.811,0.776,0.848,0.893,0.967,1.057,1.355],
    [0.945,1.032,1.009,0.950,0.783,0.895,0.740,0.852,0.803,0.835,0.966,2.191],
    [1.086,1.170,1.181,1.011,0.892,0.837,0.800,0.843,0.999,0.875,1.032,1.274],
    [1.027,1.184,1.214,1.029,0.852,0.793,0.751,0.844,1.034,0.909,1.083,1.281],
    [0.921,0.970,1.013,0.980,0.908,0.832,0.903,0.899,1.133,1.046,1.160,1.237],
    [1.254,1.246,1.129,0.864,0.741,0.628,0.614,0.751,1.115,1.032,1.165,1.461],
])

DEMANDA_BASE = np.array([5.3, 5.6, 10.7, 16.9, 9.4, 7.0])


def fator_clima(temp_norm):
    fc = np.ones(N_PRODUTOS)
    fc[1] = 0.5 + 1.0 * temp_norm   # gelo: mais no calor
    fc[2] = 0.7 + 0.6 * temp_norm   # refrigerante
    fc[3] = 0.8 + 0.4 * temp_norm   # agua
    fc[5] = 1.3 - 0.5 * temp_norm   # sorvete: mais no calor também
    # calibrar coeficientes com Open-Meteo quando disponível
    return fc


class ConvenienceStoreEnv(gym.Env):
    def __init__(self, alpha=2.0, beta=1.5, gamma_pen=0.5, delta=1.0, max_steps=21):
        super().__init__()
        self.alpha = alpha; self.beta = beta
        self.gamma_pen = gamma_pen; self.delta = delta
        self.max_steps = max_steps
        # Estado: 3+7+12+1+6+6+6 = 41 features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(41,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self._step = 0
        self._temp_norm = 0.5  # FIX Bug 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        self._temp_norm = 0.5  # FIX Bug 1
        noise = self.np_random.uniform(0.8, 1.2, size=N_PRODUTOS)
        self.estoque   = (ESTOQUE_INICIAL * noise).astype(float)
        self.validade  = VALIDADE_TIPICA.copy().astype(float)
        self.promo_ant = np.zeros(N_PRODUTOS)
        return self._get_obs(), {}

    def step(self, action):
        turno_idx = self._step % 3
        dia_idx   = (self._step // 3) % 7
        mes_idx   = (self._step // 63) % 12  # aproximação — usar mês real quando possível

        self._temp_norm = self.np_random.uniform(0.2, 0.9)  # FIX Bug 1 — TODO: usar real

        # Efeito da promoção via elasticidade-preço
        produto_promo = self._produto_alvo(action)
        fp = np.ones(N_PRODUTOS)
        desc_pct = 0.0
        if action == 1 and produto_promo >= 0:
            desc_pct = 0.05
            fp[produto_promo] = 1 + abs(ELASTICIDADE[produto_promo]) * desc_pct
        elif action == 2 and produto_promo >= 0:
            desc_pct = 0.10
            fp[produto_promo] = 1 + abs(ELASTICIDADE[produto_promo]) * desc_pct
        elif action == 3 and produto_promo >= 0:
            comp = PARES_COMBO[produto_promo]
            fp[produto_promo] = 1.12
            fp[comp] = 1.08  # efeito cross-sell no complementar
        elif action == 4 and produto_promo >= 0:
            fp[produto_promo] = 1 + abs(ELASTICIDADE[produto_promo]) * 0.08

        demanda = (DEMANDA_BASE
                   * FATOR_TURNO[:, turno_idx]
                   * FATOR_DIA[:, dia_idx]
                   * FATOR_MES[:, mes_idx]
                   * fator_clima(self._temp_norm)
                   * fp)
        demanda_real = self.np_random.poisson(np.maximum(demanda, 0))
        vendas = np.minimum(demanda_real, self.estoque)

        # FIX Bug 2: rupturas antes de atualizar estoque
        rupturas = np.maximum(demanda_real - vendas, 0)

        self.validade -= 1
        perdas = np.where(self.validade <= 0, self.estoque - vendas, 0.0)

        # FIX Bug 3: validade antes do reset para bonus_giro
        validade_pre_reset = self.validade.copy()
        self.estoque  = np.maximum(self.estoque - vendas - perdas, 0)
        self.validade = np.where(self.validade <= 0, VALIDADE_TIPICA.astype(float), self.validade)

        desconto_excessivo = 0.0
        if action in [1,2] and produto_promo >= 0 and self.estoque[produto_promo] < 5:
            desconto_excessivo = 1.0

        # Reposição simplificada
        for i in range(N_PRODUTOS):
            if self.estoque[i] < 0.3 * ESTOQUE_INICIAL[i]:
                self.estoque[i] = ESTOQUE_INICIAL[i] * self.np_random.uniform(0.8, 1.0)

        preco_efetivo = PRECO_VENDA.copy()
        if action == 1 and produto_promo >= 0: preco_efetivo[produto_promo] *= (1 - desc_pct)
        elif action == 2 and produto_promo >= 0: preco_efetivo[produto_promo] *= (1 - desc_pct)

        lucro        = np.sum(vendas * (preco_efetivo - CUSTO))
        pen_venc     = self.alpha     * np.sum(perdas   * CUSTO)
        pen_ruptura  = self.beta      * np.sum(rupturas * MARGEM * 0.5)
        pen_desconto = self.gamma_pen * desconto_excessivo * 5.0
        bonus_giro   = self.delta     * np.sum(vendas * (validade_pre_reset < 3) * MARGEM * 0.3)
        reward = lucro - pen_venc - pen_ruptura - pen_desconto + bonus_giro

        self.promo_ant = np.zeros(N_PRODUTOS)
        if produto_promo >= 0: self.promo_ant[produto_promo] = 1.0

        self._step += 1
        info = {'lucro':lucro,'vendas':vendas.copy(),'perdas':perdas.copy(),
                'rupturas':rupturas.copy(),'pen_venc':pen_venc,'pen_ruptura':pen_ruptura}
        return self._get_obs(), float(reward), self._step >= self.max_steps, False, info

    def _produto_alvo(self, action):
        if action == 0:          return -1
        if action in [1, 2, 3]: return int(np.argmax(self.estoque))
        if action == 4:          return int(np.argmin(self.validade))
        return -1

    def _get_obs(self):
        turno_idx = self._step % 3
        dia_idx   = (self._step // 3) % 7
        mes_idx   = (self._step // 63) % 12
        turno_oh  = np.zeros(3);  turno_oh[turno_idx] = 1.0
        dia_oh    = np.zeros(7);  dia_oh[dia_idx]     = 1.0
        mes_oh    = np.zeros(12); mes_oh[mes_idx]     = 1.0  # NOVO
        temp          = np.array([self._temp_norm])  # FIX Bug 1
        estoque_norm  = np.clip(self.estoque  / (ESTOQUE_INICIAL * 1.2), 0, 1)
        validade_norm = np.clip(self.validade /  VALIDADE_TIPICA,        0, 1)
        return np.concatenate([turno_oh, dia_oh, mes_oh, temp,
                               estoque_norm, validade_norm, self.promo_ant]).astype(np.float32)
```

---

## FUNÇÃO DE INFERÊNCIA (a implementar — Nível 3)

```python
NOMES_ACOES = {
    0: "Nenhuma promoção neste turno",
    1: "Desconto de 5% em {}",
    2: "Desconto de 10% em {}",
    3: "Combo: {} + {} (preço cheio, efeito combinado)",
    4: "Promoção de giro em {} (validade crítica)",
}

def get_recommendation(dia_semana: int, turno: str, mes: int,
                        temperatura: float, estoque_atual: dict,
                        agent_path: str = 'results/dqn_model.pt') -> str:
    """
    dia_semana: 0=seg, 6=dom
    turno: 'manha', 'tarde' ou 'noite'
    mes: 1-12
    temperatura: graus Celsius (ex: 32.5)
    estoque_atual: {'energetico': 18, 'gelo': 5, ...}
    """
    # Monta o estado
    turno_map = {'manha': 0, 'tarde': 1, 'noite': 2}
    env = ConvenienceStoreEnv()
    env.reset()
    env._step = turno_map[turno] + dia_semana * 3
    env._temp_norm = (temperatura - 15) / (40 - 15)  # normaliza 15-40°C
    env.estoque = np.array([estoque_atual[p] for p in PRODUTOS], dtype=float)

    obs = env._get_obs()

    # Carrega modelo e faz inferência
    model = DQN(input_dim=41, output_dim=5)
    model.load_state_dict(torch.load(agent_path))
    model.eval()

    with torch.no_grad():
        q_vals = model(torch.tensor(obs).unsqueeze(0))
        action = int(q_vals.argmax().item())

    # Interpreta a ação
    produto_idx = env._produto_alvo(action)
    produto_nome = PRODUTOS[produto_idx].upper() if produto_idx >= 0 else ''

    if action == 3 and produto_idx >= 0:
        comp_idx = PARES_COMBO[produto_idx]
        return NOMES_ACOES[action].format(produto_nome, PRODUTOS[comp_idx].upper())
    elif action > 0:
        return NOMES_ACOES[action].format(produto_nome)
    else:
        return NOMES_ACOES[0]
```

---

## PROTOCOLO DE TESTE A/B (a implementar após treino)

```
Semanas 1-2: operação normal (sem seguir o agente) — coleta baseline
Semanas 3-4: seguir recomendações do agente — mede resultado
Semanas 5-6: normal novamente
Semanas 7-8: agente novamente

Métricas a comparar:
- Lucro total por semana (R$)
- Total de produtos descartados (unidades e R$)
- Total de rupturas de estoque registradas
- Promoções aplicadas vs não aplicadas

O teste A/B também vai calibrar ELASTICIDADE empiricamente:
- Se o agente sugeriu 10% de desconto no energético e as vendas subiram X%,
  elasticidade_real = X / 10% → substitui o valor da literatura
```

---

## TECNOLOGIAS E AMBIENTE

- **OS:** Windows, PowerShell, VS Code
- **Python:** 3.13 / 3.14 via launcher `py`
- **Hardware:** 8 CPUs, 32GB RAM
- **Bibliotecas:** gymnasium, stable-baselines3, torch, pandas, numpy, seaborn, matplotlib
- **Versionamento:** GitHub — fork do professor, PRs com feature branches

**Atenção para Windows:**
- Usar `pygame-ce` em vez de `pygame`
- TensorFlow/Keras: usar `int()` para tamanhos de camadas
- `ProcessPoolExecutor` falha no Jupyter — usar `subprocess.Popen` com scripts `.py`

---

## O QUE FALTA PARA COMEÇAR O NOTEBOOK FINAL

1. **Planilhas pendentes a receber:**
   - `descarte_produto.xlsx` com 1 ano completo (mesmo período das outras)
   - Confirmar se `venda_do_mes.xlsx` pode ser exportado para outros meses também

2. **Buscar temperatura:** rodar a chamada Open-Meteo (código acima) e salvar como CSV

3. **Montar o notebook final** com:
   - Nova estrutura (Seções 0-8)
   - EDA integrada mostrando origem de cada parâmetro
   - Estado de 41 features
   - Elasticidade no lugar de FATOR_PROMO
   - Ação 3 redesenhada com PARES_COMBO
   - Função get_recommendation() na seção de conclusões

---

*Documento gerado em 04/05/2026 — captura completa da sessão de planejamento.*  
*Próxima sessão: inserir planilhas completas e montar notebook final.*
