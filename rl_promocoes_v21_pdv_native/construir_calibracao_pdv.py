"""Constrói calibração V21 PDV-native a partir de:
- Inventário REAL do posto (custos, volumes, taxas de perda)
- Conhecimento de domínio de PDV de posto de gasolina

NÃO USA Instacart, NÃO USA matriz de cesta de supermercado.

A matriz de harmonia é CRIADA por raciocínio de domínio:
- O que cliente leva no MESMO TICKET no balcão (não em cesta de festa em casa)
- Distinção CRÍTICA entre cesta de supermercado e impulso de PDV

Output: data_sintetica/calibracao_v21_pdv.json
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# CATÁLOGO PDV-NATIVE (20 categorias relevantes)
# Baseado em inventário 2022-26 do posto Viana, custos medianos REAIS.
# Demanda estimada por volume de inventário (giro proxy) + benchmarks.
# ═══════════════════════════════════════════════════════════════════════

# Definição: 20 categorias de conveniência (cigarros = separados, NÃO promovíveis)
CATEGORIAS_PDV = [
    # Bebidas (consumo individual no carro)
    {'nome': 'agua',          'tipo': 'commodity',       'promovivel': True},
    {'nome': 'refrigerante',  'tipo': 'commodity',       'promovivel': True},
    {'nome': 'suco',          'tipo': 'commodity',       'promovivel': True},
    {'nome': 'isotonico',     'tipo': 'commodity',       'promovivel': True},
    {'nome': 'energetico',    'tipo': 'commodity',       'promovivel': True},
    {'nome': 'cha_pronto',    'tipo': 'commodity',       'promovivel': True},
    # Bebidas alcoólicas
    {'nome': 'cerveja',       'tipo': 'puxador_consumo', 'promovivel': True},
    {'nome': 'destilados',    'tipo': 'puxador_premium', 'promovivel': True},
    {'nome': 'whisky',        'tipo': 'puxador_premium', 'promovivel': True},
    {'nome': 'vinho',         'tipo': 'puxador_premium', 'promovivel': True},
    # Comida quick
    {'nome': 'cafe',          'tipo': 'rotina',          'promovivel': True},
    {'nome': 'padaria',       'tipo': 'rotina',          'promovivel': True},
    {'nome': 'sanduiche',     'tipo': 'rotina',          'promovivel': True},
    # Doces/snacks
    {'nome': 'chocolate_caixa', 'tipo': 'puxador_presente', 'promovivel': True},
    {'nome': 'chocolate_unit', 'tipo': 'impulso',         'promovivel': True},
    {'nome': 'doce_balcao',   'tipo': 'impulso',         'promovivel': True},
    {'nome': 'biscoito',      'tipo': 'impulso',         'promovivel': True},
    {'nome': 'snack_salgado', 'tipo': 'impulso',         'promovivel': True},
    {'nome': 'sorvete',       'tipo': 'impulso',         'promovivel': True},
    # Utilitário (compra de ocasião)
    {'nome': 'gelo',          'tipo': 'utilitario_evento','promovivel': True},
]

# Cigarros (sempre PROIBIDOS — Lei 9.294/96)
CIGARROS = ['cigarro_jti', 'cigarro_philip_morris', 'cigarro_souza_cruz']

# ═══════════════════════════════════════════════════════════════════════
# DADOS DE DEMANDA — calibrados em REGRAS REAIS de PDV de posto
# Demanda diária por categoria, baseada em conhecimento de varejo de
# conveniência (não em supermercado).
# ═══════════════════════════════════════════════════════════════════════

DEMANDA_BASE_DIA = {
    # Bebidas — alta rotatividade
    'agua': 12.0,         # 12 garrafas/dia (compra rápida no carro)
    'refrigerante': 18.0, # impulso de lanche
    'suco': 4.0,          # menor (mais nicho)
    'isotonico': 3.0,     # nicho recovery
    'energetico': 5.0,    # motoristas/madrugada
    'cha_pronto': 1.5,
    # Alcoólicas — perfil NOITE/FDS
    'cerveja': 22.0,      # long-neck/lata sex-sáb à noite
    'destilados': 0.4,    # raro (raríssimo no posto)
    'whisky': 0.1,        # baixíssimo (cliente vai loja própria)
    'vinho': 0.2,         # quase zero
    # Comida — perfil MANHÃ
    'cafe': 25.0,         # commuter pré-trabalho (cafeteira)
    'padaria': 14.0,      # pão de queijo/pão na manhã
    'sanduiche': 3.5,     # almoço quick
    # Chocolates
    'chocolate_caixa': 8.0,   # presente/saidão
    'chocolate_unit': 7.0,    # impulso doce
    'doce_balcao': 15.0,      # bala/chiclete/menthos no caixa
    'biscoito': 6.0,
    'snack_salgado': 10.0,    # amendoim/Doritos/Pringles
    'sorvete': 6.0,
    # Utilitário
    'gelo': 5.5,          # sex/sáb fim de tarde (FDS)
}

# Margem (preço venda - custo) baseada em mercados típicos
# Custos vieram do inventário real; preço venda = custo × markup típico
MARKUP_CATEGORIA = {
    'agua': 2.5, 'refrigerante': 1.8, 'suco': 1.8, 'isotonico': 1.7,
    'energetico': 1.6, 'cha_pronto': 1.6,
    'cerveja': 2.0,
    'destilados': 1.4, 'whisky': 1.3, 'vinho': 1.5,
    'cafe': 4.5, 'padaria': 2.5, 'sanduiche': 1.8,
    'chocolate_caixa': 1.6, 'chocolate_unit': 1.8,
    'doce_balcao': 2.2, 'biscoito': 1.9,
    'snack_salgado': 1.9, 'sorvete': 1.7,
    'gelo': 2.8,
}

# Validade típica em DIAS (proxy para alpha de vencimento)
VALIDADE_DIAS = {
    'agua': 365, 'refrigerante': 180, 'suco': 90,
    'isotonico': 180, 'energetico': 365, 'cha_pronto': 180,
    'cerveja': 180, 'destilados': 720, 'whisky': 720, 'vinho': 720,
    'cafe': 270, 'padaria': 7, 'sanduiche': 5,
    'chocolate_caixa': 365, 'chocolate_unit': 180,
    'doce_balcao': 180, 'biscoito': 120, 'snack_salgado': 90, 'sorvete': 365,
    'gelo': 60,  # saco intacto, no congelador
}

# Elasticidade promocional por tipo
ELASTICIDADE_TIPO = {
    'commodity': 0.4,           # água/refri reagem POUCO a desconto
    'puxador_consumo': 1.1,     # cerveja reage bem
    'puxador_premium': 0.5,     # destilados/vinho — preço alto, sensível pra menos
    'puxador_presente': 1.5,    # chocolate caixa em data presente
    'impulso': 0.9,             # snack/doce — reage médio
    'rotina': 0.4,              # café da manhã pouco sensível
    'utilitario_evento': 0.7,   # gelo em contexto de festa
}

# ═══════════════════════════════════════════════════════════════════════
# MATRIZ DE HARMONIA PDV-NATIVE (criada por DOMÍNIO, não Instacart)
# ═══════════════════════════════════════════════════════════════════════
#
# Filosofia: em um posto, cliente compra IMPULSO ou ROTINA, não cesta de festa.
# Combos VÁLIDOS são pares que cliente leva NO MESMO TICKET no balcão.
#
# Padrões reais observados:
# - Manhã (commuter): café+padaria, café+biscoito, água+padaria
# - Tarde (lanche): refri+snack, refri+biscoito, água+sanduíche
# - Noite (saída): cerveja+snack, cerveja+amendoim, energético+snack
# - FDS (família): refri+biscoito, refri+sorvete
# - Saída balada: cerveja+snack, energético+isotônico (recovery)
# - Presente: chocolate_caixa+doce_balcao (kit), chocolate+biscoito
# - Churrasco (raro no posto): gelo+cerveja (1 saco pequeno)
#
# NÃO faz sentido:
# - gelo+destilados/whisky/vinho (compra em mercado, não posto)
# - chocolate_caixa+vinho (presente caro, cliente vai loja própria)
# - café+cerveja, café+vinho (manhã vs noite, ocasiões opostas)
# - sanduíche+cerveja (mais raro em posto)

CATS_NOME = [c['nome'] for c in CATEGORIAS_PDV]
N = len(CATS_NOME)


def harmonia_pdv_native():
    """Matriz N×N criada por conhecimento de domínio de PDV.

    Valores:
    - >= 2.5: combo clássico de PDV (cerveja+snack, café+padaria)
    - 1.5-2.0: combo razoável (refri+snack, agua+sanduiche)
    - 1.0-1.4: par neutro (cliente pode levar mas não típico)
    - < 1.0: combo absurdo (gelo+whisky, café+cerveja)
    """
    H = [[1.0] * N for _ in range(N)]

    # Combos CLÁSSICOS de PDV (h >= 2.5)
    pares_classicos = {
        # MANHÃ — combo café-pão clássico
        ('cafe', 'padaria'): 2.8,
        ('cafe', 'biscoito'): 2.2,
        ('cafe', 'doce_balcao'): 1.8,
        # TARDE — combo refri-snack
        ('refrigerante', 'snack_salgado'): 2.6,
        ('refrigerante', 'biscoito'): 2.0,
        ('refrigerante', 'doce_balcao'): 1.6,
        ('refrigerante', 'chocolate_unit'): 1.9,
        ('refrigerante', 'sorvete'): 1.8,
        # NOITE — cerveja é rei. V21 iter18: cerveja casa SÓ com snack salgado
        # (amendoim/Doritos é o combo de balcão real). Isotônico/doce/biscoito
        # NÃO são combo de cerveja em posto — Vinicius confirmou.
        ('cerveja', 'snack_salgado'): 2.8,      # ÚNICO combo forte da cerveja
        ('cerveja', 'doce_balcao'): 0.9,         # ↓ 1.5 (não casa)
        ('cerveja', 'isotonico'): 0.8,           # ↓ 1.6 (recovery não é PDV)
        ('cerveja', 'biscoito'): 0.9,            # ↓ 1.3
        # ENERGÉTICO — motorista/madrugada. V21 iter19: energético+isotônico
        # NÃO casa (2 bebidas funcionais — cliente leva UMA, não as duas).
        ('energetico', 'snack_salgado'): 1.8,    # motorista + petisco
        ('energetico', 'doce_balcao'): 1.4,
        ('energetico', 'biscoito'): 1.3,
        ('energetico', 'isotonico'): 0.8,        # ↓ 1.5 (não é combo PDV)
        # ÁGUA — lanche rápido
        ('agua', 'padaria'): 1.8,
        ('agua', 'sanduiche'): 2.0,
        ('agua', 'biscoito'): 1.4,
        ('agua', 'snack_salgado'): 1.4,
        ('agua', 'doce_balcao'): 1.2,
        # SUCO — meio termo
        ('suco', 'padaria'): 1.8,
        ('suco', 'biscoito'): 1.6,
        ('suco', 'sanduiche'): 1.7,
        # CHOCOLATE CAIXA (presente)
        ('chocolate_caixa', 'doce_balcao'): 1.5,
        ('chocolate_caixa', 'biscoito'): 1.4,
        ('chocolate_caixa', 'cafe'): 1.5,
        ('chocolate_caixa', 'chocolate_unit'): 1.6,
        # CHOCOLATE UNIT (impulso)
        ('chocolate_unit', 'doce_balcao'): 1.5,
        ('chocolate_unit', 'biscoito'): 1.4,
        ('chocolate_unit', 'refrigerante'): 1.9,
        # SORVETE (impulso doce)
        ('sorvete', 'biscoito'): 1.4,
        ('sorvete', 'refrigerante'): 1.8,
        # GELO — V21 ITER 11: NO POSTO, gelo é compra de OCASIÃO (festa em casa)
        # Cliente que para pra abastecer e leva 1 cerveja não vai sair com saco 5kg
        # Combos GELO+X = comprado em mercado, não posto. Harmonia BAIXA em todos
        # exceto carvão (churrasco). Insight Vinicius (16/05 e 17/05).
        ('gelo', 'cerveja'): 0.7,        # ↓ 1.6 → 0.7
        ('gelo', 'refrigerante'): 0.8,    # ↓ 1.2 → 0.8
        ('gelo', 'snack_salgado'): 0.8,   # ↓ 1.3 → 0.8
        # PADARIA
        ('padaria', 'sanduiche'): 1.3,
        ('padaria', 'cha_pronto'): 1.7,
        # SANDUÍCHE
        ('sanduiche', 'refrigerante'): 1.6,
        ('sanduiche', 'cha_pronto'): 1.3,
        # CHÁ
        ('cha_pronto', 'biscoito'): 1.6,
        ('cha_pronto', 'doce_balcao'): 1.2,
    }

    # Combos ABSURDOS em PDV (h < 1.0) — saco gelo+garrafa = ir ao mercado
    pares_invalidos = {
        # V21 ITER 11: GELO no posto é sempre compra ocasional (festa em casa).
        # Nenhum combo de gelo+X faz sentido no balcão (exceto gelo+carvão).
        ('gelo', 'destilados'): 0.5,
        ('gelo', 'whisky'): 0.5,
        ('gelo', 'vinho'): 0.5,
        ('gelo', 'sorvete'): 0.6,
        ('gelo', 'doce_balcao'): 0.7,
        ('gelo', 'cha_pronto'): 0.6,
        ('gelo', 'agua'): 0.5,
        ('gelo', 'cafe'): 0.4,
        ('gelo', 'padaria'): 0.4,
        ('gelo', 'sanduiche'): 0.5,
        ('gelo', 'biscoito'): 0.6,
        ('gelo', 'chocolate_caixa'): 0.5,
        ('gelo', 'chocolate_unit'): 0.6,
        ('gelo', 'suco'): 0.6,
        ('gelo', 'isotonico'): 0.5,
        ('gelo', 'energetico'): 0.5,
        # Café (manhã) + bebida alcoólica (noite) = ocasiões opostas
        ('cafe', 'cerveja'): 0.6,
        ('cafe', 'destilados'): 0.5,
        ('cafe', 'whisky'): 0.5,
        ('cafe', 'vinho'): 0.6,
        ('cafe', 'sorvete'): 0.7,
        # Padaria (manhã) + álcool
        ('padaria', 'cerveja'): 0.6,
        ('padaria', 'destilados'): 0.5,
        ('padaria', 'whisky'): 0.5,
        ('padaria', 'vinho'): 0.5,
        # Sanduíche + álcool (almoço com bebida = raro no posto)
        ('sanduiche', 'destilados'): 0.7,
        ('sanduiche', 'whisky'): 0.6,
        ('sanduiche', 'vinho'): 0.6,
        # Chocolate caixa (presente) + álcool (não casa)
        ('chocolate_caixa', 'cerveja'): 0.8,
        ('chocolate_caixa', 'destilados'): 0.8,
        # Vinho + tudo (vinho é loja, não posto)
        ('vinho', 'cerveja'): 0.7,
        ('vinho', 'refrigerante'): 0.7,
        ('vinho', 'snack_salgado'): 0.7,
        # Whisky + tudo (whisky é loja)
        ('whisky', 'cerveja'): 0.7,
        ('whisky', 'refrigerante'): 0.6,
        ('whisky', 'snack_salgado'): 0.8,
        # Mais absurdos
        ('isotonico', 'destilados'): 0.7,
        ('isotonico', 'whisky'): 0.7,
        ('isotonico', 'vinho'): 0.7,
        ('sorvete', 'cerveja'): 0.8,
        ('sorvete', 'destilados'): 0.6,
        ('sorvete', 'whisky'): 0.6,
        ('sorvete', 'vinho'): 0.6,
    }

    idx = {c: i for i, c in enumerate(CATS_NOME)}

    def set_pair(a, b, val):
        i, j = idx[a], idx[b]
        H[i][j] = val
        H[j][i] = val

    for (a, b), v in pares_classicos.items():
        set_pair(a, b, v)
    for (a, b), v in pares_invalidos.items():
        set_pair(a, b, v)

    # Diagonal: mesmo produto não combina consigo
    for i in range(N):
        H[i][i] = 0.0

    return H


# ═══════════════════════════════════════════════════════════════════════
# FATORES SAZONAIS (dow, mês) — calibração realista para PDV de posto
# ═══════════════════════════════════════════════════════════════════════
# Padrões observados em postos:
# - Cerveja: pico sex/sáb noite, dezembro
# - Café/padaria: pico segunda-sexta manhã (commuter)
# - Sorvete/gelo: pico verão + sábado tarde
# - Isotônico: pico sábado tarde (pós-academia)

FATOR_DIA_PADRAO = {
    # seg ter qua qui sex sáb dom
    'agua':           [0.85, 0.90, 0.95, 1.00, 1.10, 1.20, 1.10],
    'refrigerante':   [0.85, 0.90, 0.95, 1.00, 1.15, 1.20, 1.10],
    'suco':           [0.90, 0.95, 1.00, 1.00, 1.10, 1.15, 1.05],
    'isotonico':      [0.80, 0.85, 0.90, 0.95, 1.10, 1.40, 1.25],
    'energetico':     [1.00, 1.00, 1.05, 1.05, 1.20, 1.00, 0.85],
    'cha_pronto':     [1.10, 1.10, 1.05, 1.05, 0.95, 0.85, 0.80],
    'cerveja':        [0.65, 0.70, 0.75, 0.85, 1.40, 1.60, 1.30],
    'destilados':     [0.70, 0.75, 0.80, 0.85, 1.30, 1.50, 1.20],
    'whisky':         [0.70, 0.75, 0.80, 0.85, 1.30, 1.50, 1.20],
    'vinho':          [0.80, 0.85, 0.90, 0.95, 1.20, 1.30, 1.10],
    'cafe':           [1.30, 1.30, 1.25, 1.25, 1.20, 0.70, 0.60],
    'padaria':        [1.35, 1.30, 1.25, 1.20, 1.15, 0.65, 0.55],
    'sanduiche':      [1.20, 1.20, 1.20, 1.20, 1.10, 0.70, 0.60],
    'chocolate_caixa':[0.90, 0.95, 1.00, 1.05, 1.15, 1.15, 0.95],
    'chocolate_unit': [0.95, 0.95, 1.00, 1.00, 1.10, 1.15, 1.05],
    'doce_balcao':    [0.95, 0.95, 1.00, 1.00, 1.10, 1.10, 1.00],
    'biscoito':       [0.95, 0.95, 1.00, 1.00, 1.10, 1.10, 1.00],
    'snack_salgado':  [0.85, 0.90, 0.95, 1.00, 1.20, 1.25, 1.10],
    'sorvete':        [0.80, 0.85, 0.90, 1.00, 1.20, 1.40, 1.30],
    'gelo':           [0.40, 0.50, 0.60, 0.75, 1.35, 1.95, 1.60],
}

FATOR_MES_PADRAO = {
    # jan fev mar abr mai jun jul ago set out nov dez
    'agua':            [1.30, 1.25, 1.10, 0.95, 0.85, 0.75, 0.75, 0.85, 0.95, 1.05, 1.15, 1.30],
    'refrigerante':    [1.25, 1.20, 1.10, 1.00, 0.90, 0.85, 0.85, 0.90, 1.00, 1.05, 1.10, 1.25],
    'suco':            [1.15, 1.15, 1.10, 1.00, 0.95, 0.85, 0.85, 0.95, 1.00, 1.05, 1.05, 1.20],
    'isotonico':       [1.30, 1.30, 1.20, 1.00, 0.85, 0.70, 0.70, 0.80, 0.95, 1.05, 1.15, 1.25],
    'energetico':      [1.05, 1.00, 1.00, 1.00, 1.00, 0.95, 0.95, 1.00, 1.00, 1.00, 1.05, 1.10],
    'cha_pronto':      [0.80, 0.85, 0.95, 1.00, 1.15, 1.30, 1.30, 1.20, 1.05, 0.95, 0.85, 0.80],
    'cerveja':         [1.30, 1.30, 1.15, 1.00, 0.85, 0.80, 0.80, 0.85, 0.95, 1.05, 1.15, 1.50],
    'destilados':      [1.05, 1.00, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1.00, 1.05, 1.10, 1.50],
    'whisky':          [1.05, 1.00, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1.00, 1.05, 1.10, 1.50],
    'vinho':           [0.90, 0.85, 0.90, 0.95, 1.00, 1.05, 1.05, 1.00, 0.95, 1.00, 1.20, 1.45],
    'cafe':            [0.85, 0.90, 1.00, 1.05, 1.10, 1.20, 1.20, 1.15, 1.05, 1.00, 0.95, 0.85],
    'padaria':         [0.85, 0.90, 1.00, 1.05, 1.10, 1.15, 1.15, 1.10, 1.05, 1.00, 0.95, 0.85],
    'sanduiche':       [0.90, 0.95, 1.00, 1.05, 1.10, 1.10, 1.10, 1.05, 1.00, 1.00, 0.95, 0.90],
    'chocolate_caixa': [0.85, 0.95, 1.05, 1.40, 1.30, 1.40, 0.90, 1.30, 0.90, 1.20, 0.95, 1.40],
    'chocolate_unit':  [0.95, 1.00, 1.00, 1.05, 1.00, 1.05, 1.00, 1.05, 1.05, 1.20, 1.00, 1.10],
    'doce_balcao':     [1.00, 1.00, 1.00, 1.05, 1.00, 1.00, 1.00, 1.00, 1.05, 1.15, 1.05, 1.10],
    'biscoito':        [0.95, 0.95, 1.00, 1.00, 1.05, 1.10, 1.10, 1.05, 1.00, 1.00, 0.95, 1.05],
    'snack_salgado':   [1.10, 1.05, 1.05, 1.00, 0.95, 0.90, 0.90, 0.95, 1.00, 1.00, 1.05, 1.20],
    'sorvete':         [1.55, 1.45, 1.20, 0.95, 0.75, 0.55, 0.55, 0.70, 0.95, 1.10, 1.30, 1.55],
    'gelo':            [1.55, 1.40, 1.15, 0.85, 0.65, 0.50, 0.50, 0.60, 0.85, 1.10, 1.30, 1.60],
}


def construir(inv_path, out_path):
    inv = pd.read_csv(inv_path)
    inv_dict = {row['categoria']: row for _, row in inv.iterrows()}

    categorias = []
    for idx, c in enumerate(CATEGORIAS_PDV):
        nome = c['nome']
        custo_real = float(inv_dict[nome]['custo_medio']) if nome in inv_dict else 1.0
        markup = MARKUP_CATEGORIA[nome]
        preco_venda = round(custo_real * markup, 2)
        margem = round(preco_venda - custo_real, 2)
        validade = VALIDADE_DIAS[nome]
        elast = ELASTICIDADE_TIPO[c['tipo']]

        categorias.append({
            'indice': idx,
            'categoria': nome,
            'tipo': c['tipo'],
            'promovivel': c['promovivel'],
            'preco_venda': preco_venda,
            'custo': round(custo_real, 2),
            'margem': margem,
            'demanda_base_dia': DEMANDA_BASE_DIA[nome],
            'validade_dias': validade,
            'validade_tipica_turnos': validade * 3,  # 3 turnos/dia
            'elasticidade_promo': elast,
            'fator_dia': FATOR_DIA_PADRAO[nome],
            'fator_mes': FATOR_MES_PADRAO[nome],
            'taxa_perda_observada_pct': float(inv_dict[nome]['taxa_diff_pct']) if nome in inv_dict else 0,
        })

    # Cigarros (PROIBIDOS) — adicionados separadamente
    for cig in CIGARROS:
        custo_real = float(inv_dict[cig]['custo_medio']) if cig in inv_dict else 8.0
        categorias.append({
            'indice': len(categorias),
            'categoria': cig,
            'tipo': 'proibida',
            'promovivel': False,
            'preco_venda': round(custo_real * 1.3, 2),
            'custo': round(custo_real, 2),
            'margem': round(custo_real * 0.3, 2),
            'demanda_base_dia': 0,  # nunca promovido
            'validade_dias': 365,
            'validade_tipica_turnos': 1095,
            'elasticidade_promo': 0,
            'fator_dia': [1] * 7,
            'fator_mes': [1] * 12,
            'taxa_perda_observada_pct': 0,
        })

    H = harmonia_pdv_native()

    cal = {
        'versao': 'V21 PDV-native',
        'fonte_dados': 'inventario_real_posto_viana_2022-26 + dominio_pdv_conveniencia',
        'filosofia': 'Sem dependência Instacart. Harmonia criada por conhecimento de PDV.',
        'n_categorias': len(categorias),
        'categorias': categorias,
        'harmonia_combo': H,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    print(f"✓ Calibração V21 salva: {out_path}")
    print(f"  Total categorias: {len(categorias)} ({len([c for c in categorias if c['promovivel']])} promovíveis)")
    print(f"  Matriz harmonia: {len(H)}×{len(H[0])}")
    return cal


if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    HERE = Path(__file__).parent
    construir(
        HERE / 'data_sintetica' / 'inventario_processado.csv',
        HERE / 'data_sintetica' / 'calibracao_v21_pdv.json',
    )
