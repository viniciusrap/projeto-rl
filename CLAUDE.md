# CLAUDE.md — Projeto RL Conveniência Viana
## Guia completo para o Claude Code trabalhar neste projeto

---

## IDENTIDADE DO PROJETO

**Nome:** Otimização de Promoções com Reinforcement Learning
**Alunos:** Vinicius Rocha e Luigi Zema
**Curso:** Reinforcement Learning — Insper (Brasil)
**Empresa:** Auto Posto Parque Viana Ltda — Barueri, SP
**Nota:** "Viana" é sobrenome do dono, não a cidade. O posto fica em Barueri/SP.
**Objetivo:** Agente RL que decide qual promoção aplicar a cada turno da loja de conveniência para maximizar lucro e minimizar desperdício por vencimento.
**Repositório:** https://github.com/viniciusrap/projeto-rl
**Branch principal:** main

---

## ESTRUTURA DE PASTAS

```
projeto-rl/
├── CLAUDE.md                          ← este arquivo
├── .gitignore                         ← exclui .venv/, data/, results/, *.pt
├── data/                              ← planilhas reais (não sobem pro GitHub)
│   ├── venda_por_dia.xlsx             ← 6 anos de vendas por turno e categoria
│   ├── venda_do_mes.xlsx              ← preços, custos e margens por SKU
│   ├── descarte_produto.xlsx          ← perdas reais por vencimento
│   ├── produtos_nao_vendido.pdf       ← estoque parado
│   └── temperatura_historica.csv     ← temp. diária Barueri/SP (Open-Meteo)
├── notebooks/
│   └── rl_conveniencia_viana_FINAL.ipynb  ← notebook principal
├── results/                           ← gerado automaticamente ao rodar
│   ├── training_log.csv
│   ├── comparacao_politicas.csv
│   ├── dqn_model.pt
│   ├── eda_fator_dia.png
│   ├── eda_sazonalidade.png
│   ├── eda_lucro_margem.png
│   ├── eda_fator_clima.png
│   ├── curvas_aprendizado.png
│   ├── comparacao_politicas.png
│   └── distribuicao_acoes.png
└── .venv/                             ← Python 3.11.9 virtual environment
```

---

## AMBIENTE PYTHON

```powershell
# Ativar sempre antes de qualquer comando Python
.venv\Scripts\activate

# Python e versão
py -3.11 --version  # deve ser 3.11.9

# Pacotes instalados
gymnasium, stable-baselines3, torch, pandas, numpy,
seaborn, matplotlib, openpyxl, requests, jupyter
```

---

## OS 6 PRODUTOS DO MODELO

Ordem fixa em todos os vetores: [energético, gelo, refrigerante, água, cerveja, sorvete]

| Índice | Produto | Categoria no sistema | Por que foi escolhido |
|--------|---------|---------------------|----------------------|
| 0 | energético | ENERGÉTICO | 96% presença, padrão sex/sáb forte |
| 1 | gelo | GELO | Padrão mais extremo: sáb 2.24×, dez 2.19× |
| 2 | refrigerante | REFRIGERANTE | 99% presença, mais consistente |
| 3 | água | AGUA | 99% presença, maior margem (70%) |
| 4 | cerveja | CERVEJA AMBEV | 98% presença, pico noite (1.44×) |
| 5 | sorvete | SORVETE KIBON | 93% presença, sazonalidade verão |

**Removidos:** Sanduíche (43% presença — muito esparso) e Chocolate Nestlé (volume 4× menor que sorvete).

---

## PLANILHAS — ESTRUTURA E PARSING

### venda_por_dia.xlsx
- **O que é:** 6 anos de vendas (22/06/2020 a 30/04/2026), 2.139 dias, 147.873 registros
- **Estrutura:** blocos por data, cada bloco tem linhas com [categoria, t1_R$, t2_R$, t3_R$, t4_R$, t5_R$, t6_R$]
- **Atenção:** até 6 turnos por dia. Colapsar: turnos 1-2 → manhã (0), 3-4 → tarde (1), 5-6 → noite (2)
- **Valores em R$** — precisam ser divididos pelo preço médio para virar quantidade
- **Parsing:**
```python
df_dia = pd.read_excel('data/venda_por_dia.xlsx', header=None)
records = []; current_date = None
for _, row in df_dia.iterrows():
    if pd.notna(row[1]) and 'Data:' in str(row[1]):
        current_date = pd.to_datetime(str(row[1]).replace('Data:','').strip(), dayfirst=True)
    elif current_date is not None and pd.notna(row[2]) and pd.notna(row[3]):
        cat = str(row[2]).strip()
        for t, col in enumerate([3,4,5,6,7,8], start=1):
            val = row[col]
            if pd.notna(val) and float(val) != 0:
                records.append({'data':current_date,'categoria':cat,'turno':t,'valor_venda':float(val)})
dfp = pd.DataFrame(records)
dfp['turno3'] = dfp['turno'].apply(lambda t: 0 if t<=2 else (1 if t<=4 else 2))
dfp['dia_semana'] = dfp['data'].dt.dayofweek  # 0=seg, 6=dom
dfp['mes'] = dfp['data'].dt.month             # 1=jan, 12=dez
```
- **Gera:** FATOR_DIA, FATOR_TURNO, FATOR_MES, DEMANDA_BASE (em R$, depois converte)

---

### venda_do_mes.xlsx
- **O que é:** Vendas de março/2026 por SKU individual com quantidade, valor, custo e margem
- **Estrutura:** blocos por categoria "Classificação produto: NOME", linhas com [produto, qtd, venda, custo, margem]
- **Parsing:**
```python
df_mes = pd.read_excel('data/venda_do_mes.xlsx', header=None)
cats = {}; current_cat = None; items = []
for _, row in df_mes.iterrows():
    if pd.notna(row[0]) and 'Classificação produto:' in str(row[0]):
        if current_cat and items: cats[current_cat] = pd.DataFrame(items)
        current_cat = str(row[0]).replace('Classificação produto: ','').strip(); items=[]
    elif current_cat and pd.isna(row[0]) and pd.notna(row[1]) and pd.notna(row[2]):
        try:
            q,v,c,m = float(row[2]),float(row[3]),float(row[4]),float(row[5])
            if q>0 and v>0: items.append({'qtd':q,'venda':v,'custo':c,'margem':m})
        except: pass
```
- **Gera:** PRECO_VENDA, CUSTO, MARGEM por categoria

---

### descarte_produto.xlsx
- **O que é:** Produtos descartados por vencimento — apenas março/2026 (19 registros)
- **Estrutura:** cabeçalho na linha 0, dados a partir da linha 1. Colunas: data, turno, categoria, produto, quantidade, custo_unit, valor_venda, plano, obs
- **Parsing:**
```python
df_d = pd.read_excel('data/descarte_produto.xlsx', header=None, skiprows=1)
df_d.columns = ['data','turno','categoria','produto','quantidade',
                'custo_unit','valor_venda','plano','obs']
df_d = df_d[df_d['categoria'].notna()].copy()
df_d['custo_total'] = pd.to_numeric(df_d['custo_unit'],errors='coerce') * \
                      pd.to_numeric(df_d['quantidade'],errors='coerce')
```
- **Categorias com descarte:** ISOTÔNICO (12.9%), SNACK ELMA CHIPS (9.0%), CERVEJA AMBEV (7.7%)
- **Nossas 6 categorias:** gelo=0%, energético=0%, refrigerante=0%, água=0%, cerveja=7.7%
- **Gera:** alpha por produto (peso da penalidade de vencimento)

---

### temperatura_historica.csv
- **O que é:** Temperatura máxima diária de Barueri/SP — Open-Meteo API
- **Coordenadas:** lat=-23.5057, lon=-46.879
- **Período:** 22/06/2020 a 30/04/2026 (2.139 dias)
- **Colunas:** data (YYYY-MM-DD), temp_max (°C)
- **Temperatura:** mín ~16°C (julho), máx ~33°C (janeiro/fevereiro)
- **Script para regenerar se necessário:**
```python
import requests, pandas as pd
r = requests.get('https://archive-api.open-meteo.com/v1/archive', params={
    'latitude': -23.5057, 'longitude': -46.879,
    'start_date': '2020-06-22', 'end_date': '2026-04-30',
    'daily': 'temperature_2m_max', 'timezone': 'America/Sao_Paulo'
})
df = pd.DataFrame({'data': r.json()['daily']['time'],
                   'temp_max': r.json()['daily']['temperature_2m_max']})
df.to_csv('data/temperatura_historica.csv', index=False)
```
- **Gera:** fator_clima() calibrado com coeficientes reais por produto

---

## CRUZAMENTO DE DADOS — A PARTE MAIS IMPORTANTE

### Cruzamento 1: venda_do_mes → preço médio por categoria
```python
PRECO_MEDIO = {}
for cat, df_c in cats.items():
    PRECO_MEDIO[cat] = df_c['venda'].sum() / df_c['qtd'].sum()
# Resultados reais:
# ENERGÉTICO: R$17.86/un | GELO: R$14.99/un | REFRIGERANTE: R$8.29/un
# AGUA: R$5.00/un | CERVEJA AMBEV: R$7.55/un | SORVETE KIBON: R$14.47/un
```

### Cruzamento 2: venda_por_dia ÷ preço → DEMANDA_BASE em quantidade
```python
# Converte receita diária (R$) em quantidade (unidades)
daily_rev = dfp[dfp['categoria']==cat].groupby('data')['valor_venda'].sum()
DEMANDA_BASE[idx] = (daily_rev / PRECO_MEDIO[cat]).mean()
# Resultados reais: [5.3, 5.6, 10.7, 16.9, 9.4, 7.0] un/dia
```

### Cruzamento 3: temperatura × vendas → fator_clima()
```python
# Correlação linear entre temp_norm e qtd_norm por data
# Gera coeficientes slope e intercept para cada produto
# Impacto real: gelo e sorvete sobem com calor, chocolate cai
# Correlações calculadas: gelo=0.130, refri=0.259, agua=0.278, sorvete=0.264
```

### Cruzamento 4: descarte × venda_do_mes → alpha calibrado
```python
# taxa_perda = custo_descarte / (receita_vendas + custo_descarte)
# alpha = 2.0 * (1 + taxa_perda * 5)
# Cerveja: taxa=7.7% → alpha=2.77 | Gelo/Energético: taxa=0% → alpha=2.0
```

---

## PARÂMETROS CALIBRADOS — VETORES COMPLETOS

Ordem: [energético, gelo, refrigerante, água, cerveja, sorvete]

```python
PRECO_VENDA = np.array([17.86, 14.99,  8.29, 5.00,  7.55, 14.47])
CUSTO       = np.array([ 7.29,  7.40,  3.58, 1.49,  3.01,  8.78])
MARGEM      = PRECO_VENDA - CUSTO
# MARGEM %: [59.2%, 50.6%, 56.9%, 70.2%, 60.1%, 39.3%]
# Água tem maior margem %. Sorvete tem menor margem apesar do maior preço.

DEMANDA_BASE = np.array([5.3, 5.6, 10.7, 16.9, 9.4, 7.0])  # un/dia

# Fator por turno (manhã=0, tarde=1, noite=2) — DADOS REAIS
FATOR_TURNO = np.array([
    [0.940, 0.995, 1.064],  # energetico (pico noite)
    [1.103, 1.138, 0.758],  # gelo (pico manha/tarde)
    [0.795, 0.995, 1.210],  # refrigerante (pico noite)
    [1.054, 0.964, 0.982],  # agua (pico manha)
    [0.691, 0.866, 1.443],  # cerveja (FORTE pico noite — posto de gasolina)
    [0.856, 0.988, 1.155],  # sorvete (pico noite)
])

# Fator por dia da semana (seg=0..dom=6) — DADOS REAIS
FATOR_DIA = np.array([
    [0.761, 0.804, 0.847, 0.940, 1.164, 1.387, 1.096],
    [0.406, 0.649, 0.704, 0.547, 0.744, 2.240, 1.710],  # gelo: sab 2.24x!
    [0.883, 0.833, 0.896, 0.910, 0.999, 1.112, 1.366],
    [0.949, 0.879, 0.933, 0.945, 1.028, 1.191, 1.074],
    [0.702, 0.707, 0.779, 0.825, 1.137, 1.392, 1.459],
    [0.794, 0.815, 0.867, 0.950, 0.872, 1.098, 1.604],
])

# Fator por mês (jan=0..dez=11) — DADOS REAIS
FATOR_MES = np.array([
    [0.998,1.100,1.173,1.089,0.933,0.811,0.776,0.848,0.893,0.967,1.057,1.355],
    [0.945,1.032,1.009,0.950,0.783,0.895,0.740,0.852,0.803,0.835,0.966,2.191],
    [1.086,1.170,1.181,1.011,0.892,0.837,0.800,0.843,0.999,0.875,1.032,1.274],
    [1.027,1.184,1.214,1.029,0.852,0.793,0.751,0.844,1.034,0.909,1.083,1.281],
    [0.921,0.970,1.013,0.980,0.908,0.832,0.903,0.899,1.133,1.046,1.160,1.237],
    [1.254,1.246,1.129,0.864,0.741,0.628,0.614,0.751,1.115,1.032,1.165,1.461],
])

# Estoque inicial: demanda × (3 + CV × 4)
CV              = np.array([0.645, 1.521, 0.514, 0.519, 0.692, 0.855])
ESTOQUE_INICIAL = np.array([29, 51, 54, 86, 54, 45])

# Validade em turnos
VALIDADE_TIPICA = np.array([270, 18, 270, 270, 90, 30])

# Elasticidade-preço (substitui FATOR_PROMO)
ELASTICIDADE = np.array([-1.5, -1.3, -1.6, -1.2, -1.4, -1.5])

# Pares de produtos para combo (ação 3)
PARES_COMBO = {0:2, 1:3, 2:5, 3:0, 4:2, 5:3}
# energético+refrigerante, gelo+água, refrigerante+sorvete,
# água+energético, cerveja+refrigerante, sorvete+água

# Alpha calibrado por produto (penalidade vencimento)
ALPHA_CAT = {0:2.0, 1:2.0, 2:2.0, 3:2.0, 4:2.77, 5:2.0}
# Cerveja tem alpha maior (7.7% taxa de perda real)
```

---

## NOTEBOOK — O QUE CADA SEÇÃO FAZ

### Seção 1 — Imports e Configuração
- Importa todas as bibliotecas
- Define DATA_DIR='data', RESULTS_DIR='results'
- Cria as pastas se não existirem
- Define SEED=42, device (CPU/GPU)
- Define PRODUTOS, N_PRODUTOS, CORES, CAT_MAP

### Seção 2A — Parse venda_por_dia.xlsx
- Lê o Excel linha por linha detectando blocos de data
- Cria DataFrame com colunas: data, categoria, turno (1-6), valor_venda
- Adiciona colunas derivadas: turno3 (0/1/2), dia_semana (0-6), mes (1-12)
- SAÍDA: DataFrame `dfp` com 147.873 registros

### Seção 2B — Parse venda_do_mes.xlsx → Preços e Margens
- Lê blocos por categoria
- Calcula preço médio = venda_total / qtd_total por categoria
- Preenche vetores PRECO_VENDA, CUSTO, MARGEM_PCT
- SAÍDA: vetores reais de preço e margem

### Seção 2C — Cruzamento A×B → Demanda e Fatores
- Divide receita diária pelo preço médio → quantidade em unidades
- Calcula DEMANDA_BASE (média de unidades/dia)
- Calcula CV (coeficiente de variação das vendas diárias)
- Calcula ESTOQUE_INICIAL = demanda × (3 + CV × 4)
- Calcula FATOR_TURNO agrupando por turno3
- Calcula FATOR_DIA agrupando por dia_semana
- Calcula FATOR_MES agrupando por mês
- SAÍDA: todos os vetores de demanda calibrados

### Seção 2D — Temperatura Histórica
- Verifica se 'data/temperatura_historica.csv' existe
- Se sim: carrega direto (já foi baixado)
- Se não: chama API Open-Meteo (lat=-23.5057, lon=-46.879, Barueri/SP)
- Normaliza: temp_norm = (temp - temp_min) / (temp_max - temp_min)
- SAÍDA: DataFrame df_temp com colunas data, temp_max, temp_norm

### Seção 2E — Cruzamento Temperatura × Vendas → fator_clima()
- Para cada produto sensível ao clima (gelo, refri, água, sorvete):
  - Junta vendas diárias com temperatura por data
  - Ajusta regressão linear: qtd_norm = slope × temp_norm + intercept
  - Calcula correlação de Pearson
- Monta matriz CLIMA_COEF com slope e intercept por produto
- Define função fator_clima(temp_norm) → vetor de fatores
- SAÍDA: fator_clima() calibrado com dados reais de Barueri

### Seção 2F — Parse descarte_produto.xlsx → Alpha calibrado
- Lê o arquivo (skiprows=1, sem cabeçalho)
- Calcula taxa_perda = custo_descarte / (receita + custo_descarte)
- Calcula alpha = 2.0 × (1 + taxa_perda × 5) por categoria
- Define VALIDADE_TIPICA em turnos (gelo=18 turnos ≈ 6 dias)
- SAÍDA: ALPHA_CAT dicionário com alpha por índice de produto

### Seção 2G — Visualizações EDA
- Gráfico 1: FATOR_DIA — barra por produto, 7 dias
- Gráfico 2: FATOR_MES — linha todos os produtos, 12 meses
- Gráfico 3: Lucro real vs Receita bruta + Margem % por produto
- Salva em results/

### Seção 3 — Formulação do MDP (markdown)
- Tabela formal: estado, ações, recompensa, episódio
- Explica elasticidade-preço (substitui FATOR_PROMO)
- Explica combo com PARES_COMBO

### Seção 4 — Ambiente Gymnasium (ConvenienceStoreEnv)
- Estado: 41 features (3+7+12+1+6+6+6)
- Ações: 5 discretas
- fator_clima() usa CLIMA_COEF calibrado
- alpha usa ALPHA_CAT calibrado por produto
- Bugs corrigidos (ver seção de bugs abaixo)
- Reposição automática quando estoque < 30% do inicial

### Seção 5 — Agente Double DQN
- DQN: Linear(41→128→ReLU→64→ReLU→5)
- ReplayBuffer: deque com capacidade 10.000
- Double DQN: online seleciona ação, target avalia
- HuberLoss (SmoothL1Loss) — mais estável que MSELoss para rewards em R$
- eps_decay=0.990 → ε≈0.007 após 500 episódios

### Seção 6 — Treino
- N_RUNS=5 seeds independentes (SEED+run×1000)
- N_EPISODIOS=500 por run
- Loga reward, lucro, perdas, rupturas, epsilon, loss por episódio
- Salva training_log.csv ao final
- Salva dqn_model.pt do último agente treinado

### Seção 7 — Avaliação
- Curvas de aprendizado com IC (seaborn errorbar='sd')
- Comparação: DQN vs Aleatória vs Sem promoção (50 episódios cada)
- Distribuição de ações da política aprendida (200 episódios)
- Salva todos os gráficos e comparacao_politicas.csv

### Seção 8 — Função de Inferência
- get_recommendation(dia, turno, mes, temperatura, estoque_atual)
- Carrega dqn_model.pt
- Retorna dicionário com ação, produto, descrição e Q-values

---

## BUGS CORRIGIDOS NO CÓDIGO

### Bug 1 — Temperatura dupla (CRÍTICO)
- **Problema:** step() gerava temp_norm para demanda, _get_obs() gerava outro valor diferente
- **Fix:** self._temp_norm armazenado em step(), usado em _get_obs()

### Bug 2 — Rupturas erradas (CRÍTICO)
- **Problema:** rupturas calculadas APÓS atualizar estoque → superestimadas
- **Fix:** `rupturas = np.maximum(demanda_real - vendas, 0)` ANTES de atualizar estoque

### Bug 3 — Bonus giro após reset validade (CRÍTICO)
- **Problema:** validade resetada antes de calcular bonus_giro → bônus nunca aplicado
- **Fix:** `validade_pre_reset = self.validade.copy()` antes do reset; usa vp no cálculo

### Bug 4 — Ação 3 (combo) nunca aprendida (CRÍTICO)
- **Problema:** combo selecionava mesmo produto que ações 1 e 2; era dominada
- **Fix:** combo usa PARES_COMBO → dois produtos complementares; fator cross-sell 1.12/1.08

### Bug 5 — MSELoss instável
- **Fix:** `nn.SmoothL1Loss()` (HuberLoss)

### Bug 6 — Vanilla DQN
- **Fix:** Double DQN — `best_actions = q_net(ns).argmax(1)` + target avalia

### Bug 7 — Epsilon mal calibrado
- **Fix:** eps_decay=0.990 → ε≈0.007 após 500 eps (antes 0.997 → ε≈22%)

---

## ESTADO DO AMBIENTE (41 FEATURES)

```
[0:3]   turno one-hot      (manhã, tarde, noite)
[3:10]  dia semana one-hot (seg=0 ... dom=6)
[10:22] mês one-hot        (jan=0 ... dez=11) ← CHAVE para sazonalidade
[22]    temperatura norm    (0=mais frio, 1=mais quente de Barueri)
[23:29] estoque norm        (÷ ESTOQUE_INICIAL × 1.2, clip 0-1)
[29:35] validade norm       (÷ VALIDADE_TIPICA, clip 0-1)
[35:41] promo anterior      (binary: 1 se produto i foi promovido no turno anterior)
```

---

## AÇÕES E EFEITOS

```
Ação 0 — Sem promoção
  → sem alteração na demanda

Ação 1 — Desconto 5% no produto com maior estoque
  → fp[prod] = 1 + |ELASTICIDADE[prod]| × 0.05
  → preco_efetivo[prod] × 0.95

Ação 2 — Desconto 10% no produto com maior estoque
  → fp[prod] = 1 + |ELASTICIDADE[prod]| × 0.10
  → preco_efetivo[prod] × 0.90

Ação 3 — Combo: produto maior estoque + complementar
  → fp[prod_principal] = 1.12
  → fp[PARES_COMBO[prod_principal]] = 1.08
  → sem desconto no preço

Ação 4 — Giro: produto com validade mais crítica
  → prod = argmin(self.validade)
  → fp[prod] = 1 + |ELASTICIDADE[prod]| × 0.08
```

---

## FUNÇÃO DE RECOMPENSA

```python
lucro        = sum(vendas × (preco_efetivo - CUSTO))
pen_venc     = sum(ALPHA_CAT[i] × perdas × CUSTO)     # alpha calibrado
pen_ruptura  = 1.5 × sum(rupturas × MARGEM × 0.5)
pen_desconto = 0.5 × 5.0   # se desconto em produto com estoque < 5
bonus_giro   = 1.0 × sum(vendas × (validade_pre_reset < 3) × MARGEM × 0.3)

reward = lucro - pen_venc - pen_ruptura - pen_desconto + bonus_giro
```

---

## COMO RODAR O PROJETO

```powershell
# 1. Entrar na pasta e ativar o ambiente
cd C:\Users\vinin\projeto-rl
.venv\Scripts\activate

# 2. Abrir o Jupyter
jupyter notebook notebooks\rl_conveniencia_viana_FINAL.ipynb

# 3. No Jupyter: Cell → Run All
# Tempo estimado: 15-25 minutos (treino de 5 seeds × 500 episódios)
```

---

## COMO O CLAUDE CODE DEVE TRABALHAR NESTE PROJETO

**Início de cada sessão:**
1. Ler este CLAUDE.md completo
2. Confirmar que .venv está ativo antes de rodar qualquer Python
3. Verificar que os arquivos de data/ existem antes de rodar o notebook

**Para corrigir erros no notebook:**
- Erros de import → checar se .venv está ativo
- FileNotFoundError → checar se arquivo está em data/ com underscore (não espaço)
- CUDA out of memory → mudar device para 'cpu'
- Convergência lenta → aumentar N_EPISODIOS ou ajustar eps_decay

**Para atualizar parâmetros:**
- Novos dados de descarte → recalcular ALPHA_CAT e VALIDADE_TIPICA
- Nova temperatura → deletar data/temperatura_historica.csv e rodar Seção 2D
- Novos dados de vendas → atualizar data/venda_por_dia.xlsx e rodar a partir da Seção 2A

**Para fazer commit:**
```powershell
git add notebooks/ results/*.png results/*.csv
git commit -m "descrição do que foi feito"
git push origin main
```

**Nunca commitar:**
- data/ (planilhas confidenciais da loja)
- .venv/ (ambiente virtual)
- results/dqn_model.pt (arquivo grande)

---

## PRÓXIMOS PASSOS DO PROJETO

1. ✅ Setup completo (Python, VS Code, Claude Code, Git)
2. ✅ Dados coletados e organizados em data/
3. ✅ Temperatura real de Barueri/SP obtida
4. ✅ Notebook FINAL criado com todos os cruzamentos
5. 🔄 Rodar o notebook completo e gerar os resultados
6. ⬜ Análise dos resultados — o agente aprendeu algo sensato?
7. ⬜ Relatório final com EDA + MDP + resultados + comparação
8. ⬜ Teste A/B real na loja para calibrar elasticidade
9. ⬜ get_recommendation() em produção

---

## CHECKPOINTS DO CURSO

- **12/05/2026** — Implementação do agente e ambiente + Coleta e análise dos resultados
- Para o checkpoint: notebook rodando + curvas de aprendizado + comparação de políticas

---

## INFORMAÇÕES TÉCNICAS

- **OS:** Windows, PowerShell
- **Python:** 3.11.9 (via .venv — NUNCA usar o Python 3.14 do sistema)
- **Hardware:** 8 CPUs, 32GB RAM
- **Repositório:** https://github.com/viniciusrap/projeto-rl (branch: main)
- **Coordenadas do posto:** lat=-23.5057, lon=-46.879 (Barueri/SP)
- **Período dos dados:** 22/06/2020 a 30/04/2026
