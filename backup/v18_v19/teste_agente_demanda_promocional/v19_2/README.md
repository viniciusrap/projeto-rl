# V19.2 — Semântica de combo em PDV

Evolução do V19.1. Conserta um problema conceitual: a matriz de harmonia
herdou números da Instacart (cesta de supermercado), mas no PDV de posto
de gasolina certos combos não fazem sentido.

## Insight (Vinicius, 15/05/2026)

> "Não faz sentido fazer combo com gelo porque gelo no posto é saco de 5kg.
> Vou levar saco de gelo + garrafa de whisky no balcão? kkkk"

Conceito: posto vende em dois modos de consumo:
- **Individual / balcão**: long-neck, snack, choc.impulso, refri lata (cliente leva 1 unidade já)
- **Cesta evento**: saco gelo 5kg, garrafa premium (cliente prepara festa em casa)

Combos só funcionam dentro da MESMA OCASIÃO de consumo. Saco de gelo 5kg
+ garrafa de whisky no balcão é absurdo — quem vai fazer drink em casa
compra no mercado.

## Mudanças vs V19.1

### 1. Matriz `harmonia_combo` ajustada (em `calibracao_v2.json`)

| Par | V19.1 | V19.2 | Motivo |
|---|---:|---:|---|
| gelo + cerveja | 2.40 | **2.40** | MANTIDO — churrasco/festa clássico |
| gelo + destilados | 2.20 | **0.80** | Saco 5kg + 1 garrafa não casa no balcão |
| gelo + vinho | 1.00 | **0.80** | Idem |
| gelo + refrigerante | 1.80 | **0.95** | Raro família pegar gelo + refri grande |
| gelo + suco | 1.50 | **0.90** | Não faz sentido |
| gelo + isotonico | 1.50 | **0.90** | Idem |
| gelo + sorvete | 1.40 | **0.90** | Ou um ou outro |
| cafe + cerveja | 1.00 | **0.70** | Combo absurdo |
| cafe + destilados | 1.00 | **0.70** | Idem |

### 2. Set `COMBOS_INVALIDOS_PDV` no `demand_agent.py`

Override DURO: mesmo se harmonia ficar ≥1.0 por engano, esses pares são
rejeitados pela regra de negócio.

```python
COMBOS_INVALIDOS_PDV = {
    frozenset(['gelo', 'destilados']),
    frozenset(['gelo', 'vinho']),
    frozenset(['gelo', 'sorvete']),
    frozenset(['cafe', 'cerveja']),
    frozenset(['cafe', 'destilados']),
    frozenset(['cafe', 'vinho']),
    frozenset(['sorvete', 'cerveja']),
    frozenset(['padaria', 'cerveja']),
    frozenset(['padaria', 'destilados']),
}
```

### 3. DecisionAgent bloqueia combo_invalido_pdv → PROIBIDA

Igual ao bloqueio de cigarros (Lei 9.294/96) e combo_antagonico.

## Resultado: V19.1 vs V19.2

Aplicado ao mesmo calendário V17 (24 campanhas):

| Métrica | V19.1 | V19.2 |
|---|---:|---:|
| Aprovadas | 24/24 (filtro fraco) | **13/24** (filtro forte) |
| CONDICIONAIS | 7 | 0 |
| REJEITADAS | 0 | **2** (estruturais ruins) |
| **PROIBIDAS** | 0 | **9** (todos combos gelo+destilados) |
| Lucro total | R$ 8.111 | R$ 6.621 |
| % sobre V17 | +384% | +295% |

**Drink de Sábado** (destilados+gelo, 52 dias/ano) virou **lucro NEGATIVO**
(-R$32) — totalmente bloqueado.

**Réveillon gelo+destilados** foi para PROIBIDA. Se Vinicius achar que faz
sentido (Réveillon tem gente comprando ambos pra ceia), basta remover
o par do `COMBOS_INVALIDOS_PDV`.

## Combos que sobreviveram (faz sentido em PDV)

- ✅ Esquenta de Sexta: cerveja + snack (clássico)
- ✅ Café da Manhã: café + padaria (rotina)
- ✅ Chocolate Premium + Vinho (presente)
- ✅ Chocolate Impulso + Refrigerante (impulso unitário)
- ✅ Gelo + Cerveja em sex/sáb fim de semana (4 campanhas)

## Como rodar

```powershell
cd v19_2
..\..\.venv\Scripts\python.exe pipeline.py
..\..\.venv\Scripts\python.exe gerar_dashboard.py
..\..\.venv\Scripts\python.exe comparar_v19_1_v19_2.py
```

Outputs em `../../results/v19_2/`.

## Para reverter um combo específico

Se discordar de algum bloqueio, basta editar `demand_agent.py`:

```python
COMBOS_INVALIDOS_PDV = {
    # frozenset(['gelo', 'destilados']),  # <- comentar para liberar
    ...
}
```

E re-rodar `pipeline.py`.
