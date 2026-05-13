"""Combina TODOS os priors + dados disponíveis do posto em um único
calibracao_v2.json que alimenta o env V11.

15 categorias agregadas do modelo (top conveniência por volume).

Quando vendas detalhadas por SKU chegarem, basta re-rodar este script —
o env vai consumir o JSON novo sem mudança de código.

Saída: data/calibracao_v2.json
"""
import io
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent
DATA = ROOT / 'data'
PRIORS = DATA / 'priors_externos'

# ── Mapeamento: categoria_posto → categoria_modelo ─────────────────────────

MAPA_CATEGORIA_MODELO = {
    # Cigarros (3 categorias separadas — alto volume e dinâmica diferente)
    'SOUZA CRUZ': 'cigarro_souza_cruz',
    'PHILIP MORRIS': 'cigarro_philip_morris',
    'JTI': 'cigarro_jti',
    'CIGARRILHAS': 'cigarro_jti',  # agregado
    'ISQUEIROS': 'cigarro_souza_cruz',  # acessório fumante

    # Bebidas
    'ENERGÉTICO': 'energetico',
    'REFRIGERANTE': 'refrigerante',
    'AGUA': 'agua',
    'AGUA SABORIZADA': 'agua',
    'ÁGUA DE COCO': 'agua',
    'CERVEJA AMBEV': 'cerveja',
    'CERVEJA FEMSA': 'cerveja',
    'CERVEJA ESPECIAIS': 'cerveja',
    'CERVEJA ITAIPAVA': 'cerveja',
    'ISOTÔNICO': 'isotonico',
    'SUCO': 'suco',
    'BEBIDA LÁCTEA': 'suco',

    # Frios / sorvete / gelo
    'SORVETE KIBON': 'sorvete',
    'SORVETE JUNDIÁ': 'sorvete',
    'SORVETE PERFETTO': 'sorvete',
    'GELO': 'gelo',

    # Snacks
    'SNACK ELMA CHIPS': 'snack',
    'SNACK DIVERSOS': 'snack',
    'SNACK TORCIDA': 'snack',
    'SNACK PRINGLES': 'snack',
    'AMENDOIN/NOZES': 'snack',
    'PIPOCA': 'snack',
    'CEREAIS': 'snack',
    'BISCOITO': 'biscoito',

    # Chocolate PREMIUM (target presente Mães/Namorados/Páscoa)
    'CHOCOLATE LACTA': 'chocolate_premium',
    'CHOCOLATE NESTLE': 'chocolate_premium',
    'CHOCOLATE FERRERO': 'chocolate_premium',
    # Chocolate IMPULSO (compra de balcão, doce comum)
    'CHOCOLATE GAROTO': 'chocolate_impulso',
    'CHOCOLATE M&M (MARS)': 'chocolate_impulso',
    'CHOCOLATE ARCOR': 'chocolate_impulso',
    'CHOCOLATE DIVERSOS': 'chocolate_impulso',
    'ACHOCOLATADO': 'chocolate_impulso',

    # Doces (compra de impulso de balcão)
    'CHICLETE': 'doce',
    'MENTOS': 'doce',
    'DROPS': 'doce',
    'BALA': 'doce',
    'BALA FINI': 'doce',
    'PASTILHAS': 'doce',
    'DOCES DIVERSOS': 'doce',

    # Vinho separado (target Dia dos Namorados/Mães/Natal)
    'VINHO': 'vinho',
    # Outras bebidas destiladas (whisky, vodka, cachaça)
    'DESTILADOS DIVERSOS': 'destilados',
    'WHISK': 'destilados',
    'VODKA': 'destilados',
    'AGUARDENTES': 'destilados',

    # Comida pronta
    'PADARIA': 'padaria',
    'SANDUÍCHE': 'padaria',
    'SALGADO ASSADO/FRITO': 'padaria',
    'BOLO': 'padaria',
    'IOGURTE': 'padaria',
    'CONGELADOS': 'padaria',
    'MERCEARIA ALIMENTICIA': 'padaria',

    # Café / bebida quente
    'CAFÉ': 'cafe',
    'NESCAFÉ BEBIDAS': 'cafe',
    'CHÁ': 'cafe',
}

# Categorias finais do modelo (15)
CATEGORIAS_MODELO = sorted(set(MAPA_CATEGORIA_MODELO.values()))
N_CATEGORIAS = len(CATEGORIAS_MODELO)
print(f"Categorias do modelo V11: {N_CATEGORIAS}")
for c in CATEGORIAS_MODELO:
    print(f"  - {c}")

# Mapeamento categoria_modelo → categoria_dunnhumby (para prior comportamental)
MAPA_PARA_DUNNHUMBY = {
    'cerveja': 'cerveja',
    'vinho': 'vinho',
    'destilados': 'vinho',  # proxy
    'refrigerante': 'refrigerante',
    'agua': 'agua',
    'suco': 'suco',
    'energetico': 'refrigerante',
    'isotonico': 'refrigerante',
    'snack': 'snack',
    'biscoito': 'biscoito',
    'chocolate_premium': 'chocolate_doce',
    'chocolate_impulso': 'chocolate_doce',
    'doce': 'chocolate_doce',
    'sorvete': 'sorvete',
    'gelo': 'gelo',
    'cafe': 'cafe',
    'padaria': 'padaria',
    'cigarro_souza_cruz': 'cigarro',
    'cigarro_philip_morris': 'cigarro',
    'cigarro_jti': 'cigarro',
}

# ── 1. Carrega vendas parseadas ────────────────────────────────────────────

dfp = pd.read_csv(DATA / 'venda_por_dia_parseado.csv', parse_dates=['data'])
dfp['categoria_modelo'] = dfp['categoria'].map(MAPA_CATEGORIA_MODELO)
dfp_rel = dfp[dfp['categoria_modelo'].notna()].copy()
print(f"\nVendas relevantes: {len(dfp_rel):,} registros (de {len(dfp):,})")

# Agregação por categoria_modelo
dfp_rel['turno3'] = dfp_rel['turno'].apply(lambda t: 0 if t <= 2 else (1 if t <= 4 else 2))
dfp_rel['dia_semana'] = dfp_rel['data'].dt.dayofweek
dfp_rel['mes'] = dfp_rel['data'].dt.month

# ── 2. Preço médio por categoria (de venda_do_mes.xlsx) ────────────────────

print("\nCarregando venda_do_mes.xlsx...")
df_mes = pd.read_excel(DATA / 'venda_do_mes.xlsx', header=None)
cats_precos = {}
current_cat = None
items = []
for _, row in df_mes.iterrows():
    if pd.notna(row[0]) and 'Classificação produto:' in str(row[0]):
        if current_cat and items:
            cats_precos[current_cat] = pd.DataFrame(items)
        current_cat = str(row[0]).replace('Classificação produto: ', '').strip()
        items = []
    elif current_cat and pd.isna(row[0]) and pd.notna(row[1]) and pd.notna(row[2]):
        try:
            q, v, c, m = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            if q > 0 and v > 0:
                items.append({'qtd': q, 'venda': v, 'custo': c, 'margem': m})
        except (ValueError, TypeError):
            pass
if current_cat and items:
    cats_precos[current_cat] = pd.DataFrame(items)

PRECO_MEDIO_POSTO = {}
CUSTO_MEDIO_POSTO = {}
for cat_posto, df_c in cats_precos.items():
    if df_c['qtd'].sum() > 0:
        PRECO_MEDIO_POSTO[cat_posto] = float(df_c['venda'].sum() / df_c['qtd'].sum())
        CUSTO_MEDIO_POSTO[cat_posto] = float(df_c['custo'].sum() / df_c['qtd'].sum())

# Agrega por categoria_modelo
preco_modelo = {}
custo_modelo = {}
for cat_posto, cat_modelo in MAPA_CATEGORIA_MODELO.items():
    if cat_posto in PRECO_MEDIO_POSTO:
        preco_modelo.setdefault(cat_modelo, []).append(PRECO_MEDIO_POSTO[cat_posto])
        custo_modelo.setdefault(cat_modelo, []).append(CUSTO_MEDIO_POSTO[cat_posto])

# ── 3. Carrega temperatura ─────────────────────────────────────────────────

df_temp = pd.read_csv(DATA / 'temperatura_historica.csv', parse_dates=['data'])
temp_min = df_temp['temp_max'].min()
temp_max = df_temp['temp_max'].max()
df_temp['temp_norm'] = (df_temp['temp_max'] - temp_min) / (temp_max - temp_min)

# ── 4. Carrega prior Dunnhumby ─────────────────────────────────────────────

dh = pd.read_csv(PRIORS / 'dunnhumby' / 'sazonalidade_mensal_real.csv')
prior_dh = {}
for cat in dh['categoria'].unique():
    sub = dh[dh['categoria'] == cat].sort_values('mes_aprox')
    media_freq = float(sub['pct_desconto'].mean())
    media_mag = float(sub['mag_desconto'].mean())
    prior_dh[cat] = {
        'pct_desconto_medio': media_freq / 100 if media_freq > 1 else media_freq,
        'mag_desconto_medio': media_mag / 100 if media_mag > 1 else media_mag,
        'indice_freq_por_mes': {
            int(r['mes_aprox']): round(float(r['pct_desconto'] / media_freq), 3)
                                  if media_freq > 0 else 1.0
            for _, r in sub.iterrows()
        },
        'indice_mag_por_mes': {
            int(r['mes_aprox']): round(float(r['mag_desconto'] / media_mag), 3)
                                  if media_mag > 0 else 1.0
            for _, r in sub.iterrows()
        },
    }

# ── 5. Olist DESCONTINUADO. Substituído por priors de LOJA FÍSICA:
olist_dict = {}  # vazio = não usar Olist

# Carregar Iowa Liquor (álcool destilado físico USA, 6 anos)
try:
    iowa = pd.read_csv(PRIORS / 'iowa_liquor' / 'uplift_agregado.csv')
    iowa_dict = {(r['evento'], r['categoria']): float(r['uplift_medio'])
                  for _, r in iowa.iterrows()}
except FileNotFoundError:
    iowa_dict = {}

# Carregar Walmart Sales (loja física USA, 3 anos)
try:
    walmart = pd.read_csv(PRIORS / 'walmart' / 'uplift_feriados.csv')
    walmart_dict = {r['feriado']: float(r['uplift_medio']) for _, r in walmart.iterrows()}
except FileNotFoundError:
    walmart_dict = {}

# Carregar Tesco (loja física UK, 1 ano de fração de cesta mensal)
try:
    tesco = pd.read_csv(PRIORS / 'tesco' / 'sazonalidade_mensal.csv')
    # Mapeamento Tesco → nossas categorias
    MAPA_TESCO = {
        'f_sweets_uplift': 'doce',        # tambem servir para chocolate
        'f_beer_uplift': 'cerveja',
        'f_wine_uplift': 'vinho',
        'f_spirits_uplift': 'destilados',
        'f_soft_drinks_uplift': 'refrigerante',
        'f_water_uplift': 'agua',
        'f_tea_coffee_uplift': 'cafe',
        'f_readymade_uplift': 'padaria',
    }
    tesco_dict = {}  # {(categoria_modelo, mes): uplift}
    for col_uplift, cat_modelo in MAPA_TESCO.items():
        if col_uplift in tesco.columns:
            for _, r in tesco.iterrows():
                tesco_dict[(cat_modelo, int(r['mes']))] = float(r[col_uplift])
except FileNotFoundError:
    tesco_dict = {}

print(f"Priors físicos carregados:")
print(f"  Iowa Liquor: {len(iowa_dict)} pares (evento × cat)")
print(f"  Walmart: {len(walmart_dict)} feriados")
print(f"  Tesco UK: {len(tesco_dict)} pares (cat × mês)")

# ── 6. Carrega calendário comercial ────────────────────────────────────────

cal = pd.read_csv(DATA / 'calendario_comercial.csv', parse_dates=['data'])
cal['data'] = pd.to_datetime(cal['data']).dt.date.astype(str)
cal_eventos = cal[cal['tipo_evento'].isin(
    ['data_comercial', 'evento_esportivo'])].copy()
cols_calendario = ['data', 'nome_evento', 'categorias_afetadas',
                     'uplift_prior', 'janela_pre_dias', 'janela_pos_dias']
if 'tipo_pico' in cal_eventos.columns:
    cols_calendario.append('tipo_pico')
calendario_para_env = cal_eventos[cols_calendario].to_dict('records')

# ── 7. Carrega descarte (mar/26) para alpha por categoria ──────────────────

try:
    df_d = pd.read_excel(DATA / 'descarte_produto.xlsx', header=None, skiprows=1)
    df_d.columns = ['data', 'turno', 'categoria', 'produto', 'quantidade',
                    'custo_unit', 'valor_venda', 'plano', 'obs']
    df_d = df_d[df_d['categoria'].notna()].copy()
    df_d['custo_total'] = (pd.to_numeric(df_d['custo_unit'], errors='coerce')
                            * pd.to_numeric(df_d['quantidade'], errors='coerce'))
    descarte_por_categoria = df_d.groupby('categoria')['custo_total'].sum().to_dict()
except Exception:
    descarte_por_categoria = {}

# Vendas totais de mar/26 por categoria (denominador para taxa de perda)
mar26_vendas = (dfp_rel[(dfp_rel['data'] >= '2026-03-01') &
                         (dfp_rel['data'] < '2026-04-01')]
                 .groupby('categoria')['valor_venda'].sum().to_dict())

# ── 8. Calibração POR CATEGORIA DO MODELO ──────────────────────────────────

print("\nCalibrando categorias do modelo...")
config_categorias = []

for i, cat_m in enumerate(CATEGORIAS_MODELO):
    cats_posto = [c for c, cm in MAPA_CATEGORIA_MODELO.items() if cm == cat_m]

    # Subconjunto de vendas dessa categoria_modelo
    sub = dfp_rel[dfp_rel['categoria_modelo'] == cat_m]
    if len(sub) == 0:
        print(f"  ⚠ {cat_m}: sem dados de venda")
        continue

    # Preço médio agregado (média ponderada simples)
    precos = preco_modelo.get(cat_m, [10.0])
    custos = custo_modelo.get(cat_m, [5.0])
    preco_med = float(np.mean(precos))
    custo_med = float(np.mean(custos))
    margem_med = preco_med - custo_med

    # Demanda base em unidades/dia (receita_diaria ÷ preço médio)
    daily_rev = sub.groupby('data')['valor_venda'].sum()
    demanda_base_dia = float((daily_rev / preco_med).mean()) if preco_med > 0 else 5.0

    # Fator dia da semana
    fator_dia = []
    for d in range(7):
        sub_d = sub[sub['dia_semana'] == d]
        if len(sub_d) > 0:
            rev_d = sub_d.groupby('data')['valor_venda'].sum().mean()
        else:
            rev_d = daily_rev.mean()
        media = daily_rev.mean()
        fator_dia.append(round(float(rev_d / media) if media > 0 else 1.0, 3))

    # Fator turno (manhã = 0, tarde = 1, noite = 2)
    fator_turno = []
    for t in range(3):
        sub_t = sub[sub['turno3'] == t]
        if len(sub_t) > 0:
            rev_t = sub_t.groupby('data')['valor_venda'].sum().mean()
        else:
            rev_t = daily_rev.mean() / 3
        media_turno = daily_rev.mean() / 3
        fator_turno.append(round(float(rev_t / media_turno) if media_turno > 0 else 1.0, 3))

    # Fator mês (jan=0 ... dez=11)
    fator_mes = []
    for m in range(12):
        sub_m = sub[sub['mes'] == m + 1]
        if len(sub_m) > 0:
            rev_m = sub_m.groupby('data')['valor_venda'].sum().mean()
        else:
            rev_m = daily_rev.mean()
        media = daily_rev.mean()
        fator_mes.append(round(float(rev_m / media) if media > 0 else 1.0, 3))

    # Coeficiente clima — regressão linear simples
    sub_daily = sub.groupby('data')['valor_venda'].sum().reset_index()
    sub_daily = sub_daily.merge(df_temp[['data', 'temp_norm']], on='data', how='left').dropna()
    if len(sub_daily) > 30:
        qtd_norm = sub_daily['valor_venda'] / sub_daily['valor_venda'].mean()
        x = sub_daily['temp_norm'].values
        y = qtd_norm.values
        # Regressão linear manual: y = slope*x + intercept
        slope, intercept = np.polyfit(x, y, 1)
        clima_slope = float(slope)
        clima_intercept = float(intercept)
    else:
        clima_slope = 0.0
        clima_intercept = 1.0

    # Coeficiente de variação para estoque inicial
    cv = float(daily_rev.std() / daily_rev.mean()) if daily_rev.mean() > 0 else 0.5
    estoque_inicial = max(10, int(demanda_base_dia * (3 + cv * 4)))

    # Validade típica (heurística por categoria — refinar quando ERP enviar)
    VALIDADE_HEURISTICA = {
        'cerveja': 90, 'agua': 270, 'refrigerante': 270, 'energetico': 270,
        'isotonico': 180, 'suco': 180,
        'gelo': 18,  # turnos (~6 dias)
        'sorvete': 30,
        'snack': 60, 'biscoito': 180,
        'chocolate_premium': 270, 'chocolate_impulso': 270, 'doce': 270,
        'cafe': 365, 'padaria': 6,  # padaria vence rápido
        'cigarro_souza_cruz': 365, 'cigarro_philip_morris': 365,
        'cigarro_jti': 365, 'vinho': 720, 'destilados': 720,
    }
    validade = VALIDADE_HEURISTICA.get(cat_m, 180)

    # Elasticidade promocional — VERSÃO V11.3 (12/05/2026)
    # Decisão Vinicius: usar elasticidade EMPÍRICA medida em Dunnhumby
    # (loja física USA, 500k transações) com PISO de -0.5 para que o
    # agente ainda tenha sinal para aprender.
    #
    # Bijmolt 2005 superestima ~10× porque mede SUBSTITUIÇÃO entre marcas,
    # não volume total de categoria. Em loja física, descontos categoricos
    # têm elasticidade real próxima de zero (-0.0 a -0.4).
    cat_dh = MAPA_PARA_DUNNHUMBY.get(cat_m)
    elasticidade_empirica = None
    if cat_dh:
        try:
            df_emp = pd.read_csv(PRIORS / 'dunnhumby' / 'elasticidade_resumo.csv')
            sub_emp = df_emp[df_emp['categoria'] == cat_dh]
            if len(sub_emp) > 0:
                elasticidade_empirica = float(sub_emp['elasticidade_empirica'].iloc[0])
        except FileNotFoundError:
            pass

    if elasticidade_empirica is not None:
        # Piso de magnitude (-0.5): mantém sinal pro RL aprender
        # Se medido é positivo (ruído), usa default -0.5
        if elasticidade_empirica > 0:
            elasticidade = -0.5
        else:
            elasticidade = min(elasticidade_empirica, -0.5)
    else:
        # Sem medida empírica → default conservador
        elasticidade = -0.5

    # Alpha (penalidade vencimento)
    taxa_perda = 0.0
    for cat_posto in cats_posto:
        descarte = descarte_por_categoria.get(cat_posto, 0)
        venda = mar26_vendas.get(cat_posto, 0)
        if venda > 0:
            taxa_perda = max(taxa_perda, descarte / (venda + descarte))
    alpha = 2.0 * (1 + taxa_perda * 5)

    # Pares de combo (heurística — vai ser substituído por Apriori quando cupom chegar)
    PARES_COMBO_HEURISTICA = {
        'cerveja': 'snack', 'snack': 'cerveja',
        'refrigerante': 'snack', 'biscoito': 'cafe',
        'energetico': 'snack', 'agua': 'sorvete',
        'sorvete': 'biscoito',
        'chocolate_premium': 'vinho',    # presente
        'chocolate_impulso': 'cafe',
        'doce': 'refrigerante', 'cafe': 'biscoito',
        'gelo': 'cerveja', 'isotonico': 'snack',
        'padaria': 'cafe', 'suco': 'biscoito',
        'vinho': 'chocolate_premium',    # presente
        'destilados': 'snack',
        'cigarro_souza_cruz': 'cafe', 'cigarro_philip_morris': 'cafe',
        'cigarro_jti': 'cafe',
    }
    par_combo = PARES_COMBO_HEURISTICA.get(cat_m, 'snack')

    # Prior Dunnhumby para esta categoria
    if cat_dh and cat_dh in prior_dh:
        prior_d = prior_dh[cat_dh]
        prior_pct_promo = prior_d['pct_desconto_medio']
        prior_mag_promo = prior_d['mag_desconto_medio']
        prior_indice_freq_mes = prior_d['indice_freq_por_mes']
    else:
        prior_pct_promo = 0.2
        prior_mag_promo = 0.10
        prior_indice_freq_mes = {m: 1.0 for m in range(1, 13)}

    # Categorias NÃO promovíveis
    # - 'agua' (Vinicius 12/05/2026): commodity inelástica
    # - cigarros (Vinicius 12/05/2026 noite): proibido pela Lei 9.294/96 e
    #   ANVISA. Cigarro não pode ter promoção/desconto/publicidade no Brasil.
    #   V12 inicial promoveu cigarro_jti em 15% das decisões — decisão
    #   matematicamente boa mas LEGALMENTE INVIÁVEL. Marca como não-promovível.
    CATEGORIAS_NAO_PROMOVIVEIS = {
        'agua',
        'cigarro_souza_cruz',
        'cigarro_philip_morris',
        'cigarro_jti',
    }
    promovivel = cat_m not in CATEGORIAS_NAO_PROMOVIVEIS

    # Bonus/penalidade por dia da semana DATA-DRIVEN (Vinicius 12/05/26)
    # Para CADA categoria, calcula o desvio relativo do fator_dia em cada dia
    # da semana e converte em bonus/penalidade proporcional.
    #
    # Fórmula: bonus_dia[d] = K_BONUS_DIA × (fator_dia[d] / média - 1)
    # - Dia que vende MAIS que a média → bonus positivo (promover faz sentido)
    # - Dia que vende MENOS que a média → penalidade (promover não converte)
    #
    # K_BONUS_DIA = 400 dá escala parecida com manual antigo (cerveja Sab ~+220)
    # mas com diferenciação por magnitude real de cada categoria.
    K_BONUS_DIA = 400.0
    media_fator_dia = sum(fator_dia) / 7
    if media_fator_dia > 0.01:
        bonus_promo_dia_semana = [
            round(K_BONUS_DIA * (fator_dia[d] / media_fator_dia - 1), 1)
            for d in range(7)
        ]
    else:
        bonus_promo_dia_semana = [0.0] * 7

    config = {
        'indice': i,
        'categoria': cat_m,
        'categorias_posto_agregadas': cats_posto,
        'promovivel': promovivel,
        'bonus_promo_dia_semana': bonus_promo_dia_semana,
        'preco_venda': round(preco_med, 2),
        'custo': round(custo_med, 4),
        'margem': round(margem_med, 2),
        'demanda_base_dia': round(demanda_base_dia, 2),
        'cv_demanda': round(cv, 3),
        'estoque_inicial': estoque_inicial,
        'validade_tipica_turnos': validade,
        'elasticidade_promo': round(elasticidade, 3),
        'alpha_venc': round(alpha, 3),
        'taxa_perda_observada': round(taxa_perda, 4),
        'par_combo': par_combo,
        'fator_dia': fator_dia,
        'fator_turno': fator_turno,
        'fator_mes': fator_mes,
        'clima_slope': round(clima_slope, 4),
        'clima_intercept': round(clima_intercept, 4),
        'prior_dunnhumby_pct_promo': round(prior_pct_promo, 3),
        'prior_dunnhumby_mag_promo': round(prior_mag_promo, 3),
        'prior_dunnhumby_indice_freq_mes': prior_indice_freq_mes,
    }
    config_categorias.append(config)
    print(f"  ✓ {cat_m:<25s} demanda {demanda_base_dia:>6.1f} un/d  "
          f"preço R$ {preco_med:>6.2f}  margem {margem_med/preco_med*100:>5.1f}%  "
          f"e={elasticidade:.2f}")

# ── 9. IBGE PMC para fator macro mensal ────────────────────────────────────

try:
    ibge = pd.read_csv(PRIORS / 'ibge_pmc' / 'sazonalidade_mensal.csv')
    ibge_fator_mes = {int(r['mes']): float(r['fator_medio']) for _, r in ibge.iterrows()}
except FileNotFoundError:
    ibge_fator_mes = {m: 1.0 for m in range(1, 13)}

# ── 10. Constantes do MDP ──────────────────────────────────────────────────

constantes = {
    'K_TIMING_BONUS': 250.0,
    'K_TIMING_PENALTY': 250.0,
    'K_EVENTO': 200.0,
    'K_EVENTO_PERDIDO': 150.0,
    # ── V12.1 (Vinicius 12/05/2026 noite): bonus DOBRADO para eventos de
    # PRESENTE (Mães, Namorados, Mulher, Pais, Crianças, Páscoa). Estes
    # eventos têm pico de venda em chocolate/vinho/espumante — categorias
    # de baixo volume que são "abafadas" pelo lucro de volume das outras.
    # K_EVENTO_PRESENTE = 600 (3× K_EVENTO base) e K_EVENTO_PERDIDO_PRESENTE
    # = 400 forçam o agente a explorar essas categorias durante a janela
    # do evento. tipo_pico == 'pre' identifica eventos de presente.
    'K_EVENTO_PRESENTE': 600.0,
    'K_EVENTO_PERDIDO_PRESENTE': 400.0,
    'THETA_PADRAO': 80.0,
    'LAMBDA_INSTABILIDADE': 50.0,
    'GAMMA_DESC_5': 2.0,
    'GAMMA_DESC_10': 5.0,
    'GAMMA_DESC_25': 12.0,
    'BETA_RUPTURA': 1.5,
    'DELTA_GIRO': 1.0,
    'PCT_FRACO': 0.30,
    'PCT_FORTE': 0.70,  # NOVO — top 30% do fator combinado = ALTA DEMANDA
    'TURNOS_POR_DIA': 3,
    'EPISODIO_DIAS': 365,
    'COBERTURA_ALVO_DIAS': 7,
    'CV_FACTOR_ESTOQUE_INICIAL': 4.0,
    # ── Nova política (Vinicius 12/05/2026) ─────────────────────
    # V11.6 final: voltar para 200 (Vinicius nao quer proibir 100%)
    'K_DESC_ALTA_SAUDAVEL': 200.0,
    # Regra: combo eh a estrategia ideal quando produto principal eh alta demanda
    'K_COMBO_ALTA': 200.0,           # AUMENTADO 150->200 (V11.5)
    # V11.7: bonus EXTRA combo em data certa (250 -> 400 para reforçar)
    'K_COMBO_DATA_PICO': 400.0,
    # Regra: desconto eh OK quando produto perto de vencer (>= 70% validade)
    'K_DESC_VENCIMENTO': 120.0,      # bonus
    # Regra: desconto eh OK quando produto em baixa demanda sazonal
    'K_DESC_BAIXA': 100.0,            # bonus
    # Combo: desconto maximo 5% (era 10%)
    'DESC_COMBO_MAX': 0.05,
    # Boost de combo aumentados (V11.5)
    'BOOST_COMBO_PRINCIPAL': 1.15,    # era 1.12
    'BOOST_COMBO_PAR': 1.10,           # era 1.08
    # Risco de vencimento que justifica desconto (idade / validade_tipica)
    'LIMIAR_VENCIMENTO': 0.70,
    'LIMIAR_SAUDAVEL': 0.30,           # validade > 30% = produto saudavel
}

# ── 11. Períodos de treino/validação ───────────────────────────────────────

periodos = {
    'data_inicio_treino': '2020-06-22',
    'data_fim_treino': '2024-06-30',
    'data_inicio_validacao': '2024-07-01',
    'data_fim_validacao': '2026-04-30',
}

# ── 12. Temperatura: parâmetros para normalização ─────────────────────────

clima_params = {
    'temp_min': float(temp_min),
    'temp_max': float(temp_max),
}

# ── Salva JSON ──────────────────────────────────────────────────────────────

# Para cada categoria do modelo, agregar uplift sazonal Tesco (loja física UK)
prior_loja_fisica_mes = {}
for cat_m in CATEGORIAS_MODELO:
    fatores = {}
    for mes in range(1, 13):
        if (cat_m, mes) in tesco_dict:
            fatores[mes] = tesco_dict[(cat_m, mes)]
        else:
            fatores[mes] = 1.0
    prior_loja_fisica_mes[cat_m] = fatores

# ── V12.2 (Vinicius 12/05/2026 noite): Matrizes de HARMONIA ──────────────────
#
# DUAS matrizes complementares:
#
# 1. HARMONIA_PARES (categoria↔categoria): afinidade entre 2 produtos quando
#    formam combo. 1.0 = neutro. >1 = cesta clássica. Aplicada no env quando
#    agente escolhe principal, env completa par via fator_ctx × harmonia.
#
# 2. HARMONIA_EVENTO_CATEGORIA (evento→puxador): identifica QUAL categoria é
#    o produto-puxador de cada evento. Quando agente promove a categoria
#    puxadora no evento, bonus_evento é multiplicado por esse score.
#    Ex: chocolate em Namorados ganha 1.8×, vinho em Réveillon ganha 1.6×.

HARMONIA_PARES = {
    # ─── Cestas de presente / fim de ano ───
    ('chocolate_premium', 'vinho'):         2.5,
    ('chocolate_premium', 'padaria'):       1.8,
    ('chocolate_premium', 'cafe'):          1.6,
    ('chocolate_premium', 'doce'):          1.4,
    ('chocolate_premium', 'chocolate_impulso'): 1.2,
    ('chocolate_premium', 'biscoito'):      1.3,
    ('vinho', 'padaria'):                   1.7,
    ('vinho', 'doce'):                      1.4,
    ('vinho', 'cafe'):                      1.2,
    # ─── Impulso / lanche compra rápida ───
    ('chocolate_impulso', 'refrigerante'):  2.0,
    ('chocolate_impulso', 'suco'):          1.7,
    ('chocolate_impulso', 'cafe'):          1.5,
    ('chocolate_impulso', 'agua'):          1.3,
    ('chocolate_impulso', 'padaria'):       1.4,
    ('chocolate_impulso', 'sorvete'):       1.7,
    ('chocolate_impulso', 'biscoito'):      1.4,
    ('chocolate_impulso', 'energetico'):    1.2,
    # ─── Churrasco / Copa / fim de semana ───
    ('cerveja', 'snack'):                   2.5,
    ('cerveja', 'gelo'):                    2.4,
    ('cerveja', 'isotonico'):               1.4,
    ('cerveja', 'refrigerante'):            1.2,
    ('cerveja', 'biscoito'):                1.1,
    ('cerveja', 'sorvete'):                 1.1,
    ('snack', 'refrigerante'):              2.0,
    ('snack', 'isotonico'):                 1.6,
    ('snack', 'energetico'):                1.6,
    ('snack', 'suco'):                      1.3,
    ('snack', 'gelo'):                      1.4,
    ('gelo', 'refrigerante'):               1.8,
    ('gelo', 'suco'):                       1.5,
    ('gelo', 'isotonico'):                  1.5,
    ('gelo', 'sorvete'):                    1.4,
    # ─── Drinks (bar em casa) ───
    ('destilados', 'gelo'):                 2.2,
    ('destilados', 'snack'):                1.6,
    ('destilados', 'refrigerante'):         1.5,
    ('destilados', 'isotonico'):            1.2,
    ('destilados', 'suco'):                 1.4,
    # ─── Café da manhã / tarde ───
    ('biscoito', 'cafe'):                   2.3,
    ('biscoito', 'suco'):                   1.7,
    ('biscoito', 'padaria'):                1.6,
    ('biscoito', 'agua'):                   1.2,
    ('cafe', 'padaria'):                    2.2,
    ('cafe', 'doce'):                       1.4,
    ('cafe', 'suco'):                       1.2,
    ('padaria', 'suco'):                    1.6,
    ('padaria', 'doce'):                    1.4,
    # ─── Sobremesa / doce ───
    ('sorvete', 'biscoito'):                1.8,
    ('sorvete', 'doce'):                    1.5,
    ('sorvete', 'suco'):                    1.4,
    ('sorvete', 'refrigerante'):            1.3,
    ('doce', 'refrigerante'):               1.6,
    ('doce', 'cafe'):                       1.3,
    # ─── Energético / Isotônico ───
    ('energetico', 'agua'):                 1.4,
    ('energetico', 'isotonico'):            1.3,
    ('isotonico', 'agua'):                  1.7,
    ('isotonico', 'snack'):                 1.6,
    # ─── Refrigerante (versátil) ───
    ('refrigerante', 'biscoito'):           1.4,
    ('refrigerante', 'agua'):               1.0,
    # ─── Antagonias suaves (cesta não-natural) ───
    ('energetico', 'cerveja'):              0.8,
    ('vinho', 'isotonico'):                 0.7,
    ('destilados', 'agua'):                 0.9,
    ('cigarro_souza_cruz', 'agua'):         0.5,  # cigarro não-promov, mas se fosse
}

# Harmonia EVENTO → CATEGORIA PUXADORA
# Substring match (case-insensitive, sem acentos): chave em minúsculo simples.
# Valor: dict categoria → multiplicador de bonus_evento quando agente acerta.
# Default = 1.0 para categorias-alvo não listadas.
HARMONIA_EVENTO_CATEGORIA = {
    'dia dos namorados': {
        'chocolate_premium': 1.8,
        'chocolate_impulso': 1.3,
        'vinho': 1.6,
        'padaria': 1.2,
        'cerveja': 1.3,  # cerveja_premium na verdade, mas mapeia cerveja
    },
    'dia das maes': {
        'chocolate_premium': 1.8,
        'chocolate_impulso': 1.2,
        'vinho': 1.5,
        'padaria': 1.5,
        'cafe': 1.3,
        'doce': 1.3,
    },
    'dia internacional da mulher': {
        'chocolate_premium': 1.7,
        'chocolate_impulso': 1.2,
        'vinho': 1.5,
        'padaria': 1.3,
        'cafe': 1.2,
    },
    'dia dos pais': {
        'cerveja': 1.8,
        'destilados': 1.7,  # whisky/cachaca mapeia destilados
        'snack': 1.4,
        'chocolate_premium': 1.2,
        'vinho': 1.4,
        'gelo': 1.3,
    },
    'dia das criancas': {
        'chocolate_impulso': 1.7,
        'sorvete': 1.6,
        'refrigerante': 1.5,
        'suco': 1.5,
        'biscoito': 1.4,
        'doce': 1.5,
        'snack': 1.4,
        'chocolate_premium': 1.3,
    },
    'pascoa': {
        'chocolate_premium': 2.0,  # PUXADOR MASSIVO
        'chocolate_impulso': 1.5,
        'doce': 1.3,
        'vinho': 1.2,
        'padaria': 1.4,
    },
    'vespera de natal': {
        'chocolate_premium': 1.8,
        'vinho': 1.7,
        'padaria': 1.6,  # panettone
        'cerveja': 1.3,
        'cafe': 1.4,
        'doce': 1.4,
    },
    'natal': {
        'chocolate_premium': 1.8,
        'vinho': 1.7,
        'padaria': 1.6,
        'cerveja': 1.3,
        'cafe': 1.4,
    },
    'reveillon': {
        'vinho': 1.6,
        'cerveja': 1.5,
        'gelo': 1.7,
        'refrigerante': 1.4,
        'energetico': 1.3,
        'snack': 1.4,
        'chocolate_premium': 1.3,
    },
    'copa': {  # qualquer jogo da Copa do Mundo
        'cerveja': 2.0,
        'snack': 1.8,
        'gelo': 1.6,
        'refrigerante': 1.5,
        'isotonico': 1.2,
        'energetico': 1.2,
    },
    'carnaval': {
        'cerveja': 1.8,
        'isotonico': 1.5,
        'gelo': 1.5,
        'refrigerante': 1.4,
        'snack': 1.4,
        'agua': 1.3,
        'energetico': 1.3,
    },
    'black friday': {
        'chocolate_premium': 1.4,
        'chocolate_impulso': 1.3,
        'doce': 1.3,
        'snack': 1.2,
        'refrigerante': 1.2,
    },
    'cyber monday': {
        'chocolate_premium': 1.3,
        'chocolate_impulso': 1.2,
        'doce': 1.2,
        'snack': 1.2,
    },
    'dia do consumidor': {
        'snack': 1.3,
        'cerveja': 1.3,
        'refrigerante': 1.2,
        'chocolate_impulso': 1.2,
    },
    'independencia': {
        'cerveja': 1.5,
        'snack': 1.4,
        'gelo': 1.4,
        'refrigerante': 1.3,
    },
    'tiradentes': {
        'cerveja': 1.4,
        'snack': 1.3,
        'gelo': 1.3,
    },
    'corpus christi': {
        'cerveja': 1.4,
        'padaria': 1.3,
        'snack': 1.3,
    },
    'aniversario': {  # SP, Barueri, etc.
        'cerveja': 1.4,
        'snack': 1.3,
        'refrigerante': 1.3,
    },
    'festa junina': {
        'cerveja': 1.5,
        'snack': 1.4,
        'refrigerante': 1.4,
        'doce': 1.6,  # paçoca, pé de moleque
    },
}

# Matriz simétrica N×N (lista de listas para JSON)
nomes_cats = [c['categoria'] for c in config_categorias]
N = len(nomes_cats)
harmonia_matriz = [[1.0] * N for _ in range(N)]
for (a, b), score in HARMONIA_PARES.items():
    if a in nomes_cats and b in nomes_cats:
        i = nomes_cats.index(a)
        j = nomes_cats.index(b)
        harmonia_matriz[i][j] = score
        harmonia_matriz[j][i] = score
# Zera a diagonal (não pode combo consigo mesmo)
for i in range(N):
    harmonia_matriz[i][i] = 0.0

# ── V12.2: REDUZIR janela_pre_dias para eventos de PRESENTE ───────────────
# Em posto de conveniência, cliente compra presente de última hora (D-1 a D).
# Janela padrão (5-7 dias do varejo geral) é otimista demais. Reduz para 2.
for ev in calendario_para_env:
    if ev.get('tipo_pico') == 'pre' and int(ev.get('janela_pre_dias', 0)) > 2:
        ev['janela_pre_dias_original'] = ev['janela_pre_dias']
        ev['janela_pre_dias'] = 2
print(f"  Janela_pre_dias reduzida para 2 em {sum(1 for ev in calendario_para_env if ev.get('janela_pre_dias_original')) } eventos de presente")

calibracao = {
    'versao': 'v2.2',
    'gerado_em': str(date.today()),
    'n_categorias': N_CATEGORIAS,
    'categorias': config_categorias,
    'calendario_comercial': calendario_para_env,
    'constantes': constantes,
    'periodos': periodos,
    'clima_params': clima_params,
    'ibge_fator_mes': ibge_fator_mes,
    'mapa_categoria_posto_para_modelo': MAPA_CATEGORIA_MODELO,
    'harmonia_combo': harmonia_matriz,
    'harmonia_pares_definidos': {f"{a}__{b}": s
                                   for (a, b), s in HARMONIA_PARES.items()},
    'harmonia_evento_categoria': HARMONIA_EVENTO_CATEGORIA,
    # NOVOS PRIORS DE LOJA FÍSICA (12/05/2026):
    'prior_loja_fisica_mes': prior_loja_fisica_mes,  # Tesco UK
    # Iowa: tuple keys (evento, cat) → string para JSON
    'prior_iowa_alcool': {f"{ev}__{cat}": v
                            for (ev, cat), v in iowa_dict.items()},
    'prior_walmart_feriados': walmart_dict,             # Walmart USA feriados
    'fonte_dados': {
        'vendas': 'data/venda_por_dia.xlsx (6 anos)',
        'precos_custos': 'data/venda_do_mes.xlsx (mar/26)',
        'descarte': 'data/descarte_produto.xlsx (mar/26 — pouco)',
        'temperatura': 'Open-Meteo Barueri/SP',
        'calendario_comercial': 'gerar_calendario_comercial.py',
        'prior_dunnhumby': 'data/priors_externos/dunnhumby/',
        'prior_olist': 'data/priors_externos/olist/',
        'prior_ibge': 'data/priors_externos/ibge_pmc/',
    },
    'limitacoes': [
        'Demanda calibrada por CATEGORIA, não por SKU (dados detalhados ainda nao chegaram do ERP)',
        'Validade tipica por heuristica (esperando ERP)',
        'Descarte so de mar/26 — alpha calibrado em amostra pequena',
        'Combos via heuristica PARES_COMBO_HEURISTICA — aguardando cupom fiscal para Apriori',
        'Elasticidade da literatura Bijmolt ajustada por Dunnhumby — aguardando teste A/B in loco',
    ],
}

with open(DATA / 'calibracao_v2.json', 'w', encoding='utf-8') as f:
    json.dump(calibracao, f, indent=2, ensure_ascii=False)

print()
print(f"✓ data/calibracao_v2.json gerado")
print(f"  {len(config_categorias)} categorias calibradas")
print(f"  {len(calendario_para_env)} eventos comerciais carregados")
print(f"  IBGE PMC: fator dezembro = {ibge_fator_mes.get(12, '?')}")
print(f"  Constantes do MDP: K_TIMING={constantes['K_TIMING_BONUS']}, "
      f"K_EVENTO={constantes['K_EVENTO']}")
