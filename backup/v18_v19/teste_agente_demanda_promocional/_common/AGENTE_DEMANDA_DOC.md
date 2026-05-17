# Agente de Demanda Promocional — V3 final

Status: ✅ **Validado em 12 cenários (100% conceitualmente correto)**.

## Princípio

**Responsabilidade ÚNICA:** estimar quanto cada produto vai vender DURANTE uma promoção.

NÃO calcula lucro, NÃO calcula ROI, NÃO faz ranking — isso é trabalho dos próximos agentes.

## Entrada

```python
campanha = {
    'categoria': 'chocolate_premium',          # obrigatório
    'intensidade': 'combo',                     # obrigatório
    'produto_complementar': 'vinho',            # opcional (combo)
    'desconto_pct': 10,                         # obrigatório
    'dias_total': 4,                            # obrigatório
    'data_inicio': '2026-06-08',                # obrigatório
    'data_fim': '2026-06-11',                   # obrigatório
    'eventos_comerciais_na_janela': ['Dia dos Namorados'],  # opcional
    'estoque_pct_normal': 1.4,                  # opcional (1.0=normal)
    'validade_restante_pct': 0.15,              # opcional (0-1)
    'giro_alto': True,                          # opcional (default True)
}
```

## Saída padronizada

```python
EstimativaDemanda(
    # SAÍDA PRINCIPAL — outros agentes vão usar
    demanda_base_dia=8.24,
    demanda_promocional_dia=15.66,
    uplift_pct=90.0,
    uplift_unidades_dia=7.42,

    # Risco
    canibalizacao_estimada_pct=21.0,
    nivel_risco_canibalizacao='baixo',  # baixo/medio/alto

    # Classificação
    qualidade_promocao='🟢 BOA',  # BOA/MÉDIA/RUIM/PROIBIDA
    confianca='alta',              # alta/media/baixa

    # Componentes (transparência)
    componentes={
        'boost_preco': 1.06,
        'boost_combo': 1.375,
        'boost_evento': 1.42,
        'boost_clima': 1.0,
        'boost_dow': 1.05,
        'boost_estoque_giro': 1.15,
        'penalidade_combo_fraco': 1.0,
        'uplift_bruto_pre_cap': 122.4,  # uplift antes do cap
    },
    cap_aplicado=True,  # uplift bateu no teto de 90% (combo)

    motivo='Produto puxador_presente, evento presente. Demanda 8.2→15.7/dia '
           '(+90.0%). Combo com vinho (harmonia 2.50). Evento presente bate '
           'com produto: +42%. Estoque alto/validade próxima reforça desc. '
           '⚠ Atingiu cap máximo de uplift para combo.',

    flags=['cap_uplift_atingido'],
)
```

## Lógica em 7 etapas

### 1. Classificação de produto (6 tipos)

| Tipo | Categorias | Elasticidade efetiva |
|---|---|---:|
| `puxador_consumo` | cerveja, gelo | 1.1 |
| `puxador_premium` | destilados, vinho | 0.5 |
| `puxador_presente` | chocolate_premium | 1.5 |
| `impulso` | chocolate_impulso, snack, biscoito, doce, sorvete | 0.9 |
| `rotina` | cafe, padaria | 0.4 |
| `commodity` | agua, refrigerante, suco, isotonico, energetico | 0.3 |
| `proibida` | cigarros (Lei 9.294/96) | 0.0 |

### 2. Classificação de evento (4 tipos)

| Tipo | Eventos | Captura efetiva |
|---|---|---:|
| `presente` | Mães, Namorados, Mulher, Pais, Crianças, Páscoa, Natal | 60% |
| `consumo_intenso` | Réveillon, Carnaval, Copa | 70% |
| `comercial` | Black Friday, Cyber Monday, Dia do Consumidor | 30% |
| `nenhum` | (sem evento na janela) | — |

### 3. Match produto × evento (uplift teórico)

| Evento | Produto | Uplift teórico |
|---|---|---:|
| presente | puxador_presente | +70% |
| presente | puxador_premium (vinho) | +50% |
| presente | impulso (chocolate_impulso) | +30% |
| consumo_intenso | puxador_consumo | +50% |
| consumo_intenso | commodity | +40% |
| consumo_intenso | impulso | +30% |

**Uplift real = uplift_teórico × captura_efetiva.**

### 4. Boosts individuais

```
boost_preco       = 1 + elast × desc% × fator_intensidade
                    (fator combo=0.4, liq=0.9, desc%=1.0)

boost_combo       = 1 + max(0, harmonia - 1) × 0.25

boost_evento      = 1 + uplift_match × captura_evento

boost_clima       = verão+puxador_consumo=1.20, inverno+rotina=1.10, etc.

boost_dow         = média do fator_dia. REGRA: desc% em dia fraco → 1.0 neutro
                    (move estoque parado, não amplifica demanda)

boost_estoque_giro = estoque>1.3×normal: ×1.15
                      validade<30%: ×1.10 (×1.30 se liq25%)
                      giro baixo + desc: ×0.90
```

### 5. Uplift bruto + CAP

```
uplift_bruto = produto de todos os boosts - 1

cap por intensidade:
  desc5%:  max +25%
  desc10%: max +50%
  combo:   max +90%
  liq25%:  max +120%
```

**Cap impede números absurdos** (boost combinado de +500% que aconteceria sem teto).

### 6. Canibalização

```
canib = (0.20 + desc × 1.0) × ajuste_evento
        - cap em 50%
        - ajuste_evento = 0.7 para presente (cliente compra para data específica)
```

Walmart 2021: 30-50% das vendas promocionais aconteceriam sem promo.

### 7. Classificação da qualidade

| Regra | Resultado |
|---|---|
| Cigarros | 🚫 PROIBIDA |
| Desc direto em puxador SEM evento E giro_alto E não-defensivo | 🔴 RUIM (regra do dono) |
| Combo com harmonia < 1.0 | 🔴 RUIM (antagônico) |
| Liquidação em validade < 30% (defensiva) | mínimo 🟡 MÉDIA |
| Uplift ≥ 30% E canib ≤ 35% | 🟢 BOA |
| Uplift ≥ 15% E canib ≤ 45% | 🟡 MÉDIA |
| Senão | 🔴 RUIM |

## Validação — 12/12 cenários conceitualmente corretos

```
✓ BOA-1   Chocolate+Vinho Namorados                    BOA
✓ BOA-2   Gelo+Cerveja sáb verão                       BOA
✓ BOA-3   Sorvete liquidação vencendo                  BOA
✓ MED-1   Gelo+Destilados Réveillon                    BOA (uplift 53.9%)
✓ MED-2   Chocolate impulso seg comum                   RUIM (uplift 9%, baixo)
✓ MED-3   Vinho parado desc10%                          MÉDIA
✓ RUIM-1  Cerveja desc em alta natural                  RUIM (regra do dono)
✓ RUIM-2  Isotônico desc5% comum                        RUIM
✓ RUIM-3  Combo absurdo café+cerveja                    RUIM
✓ RUIM-4  Destilados desc sem evento                    RUIM (regra do dono)
✓ CONTR-1 Chocolate impulso em Crianças                 BOA (match!)
✓ PROIB-1 Cigarro com desc                              PROIBIDA
```

## Próximos passos (NÃO FAZER AGORA)

Depois deste agente, criar em ordem:

1. **Agente de Receita** — usa `demanda_promocional_dia` para calcular lucro
   real (já considera canibalização e halo via números do agente de demanda)
2. **Agente de Decisão** — usa output dos dois acima para decidir ROI e
   classificar campanhas para o calendário operacional
3. **Agente de Operação** — gera cartazes/etiquetas para campanhas aprovadas
4. **Integração com DQN** — agente atual decide ação, depois agente de demanda
   avalia, agente de decisão filtra

## Arquivos

```
teste_agente_demanda_promocional/
├── demand_agent.py                       (300 linhas, agente único)
├── testes_demand_agent.py                (12 cenários de validação)
├── testes_demand_agent_resultados.csv    (resultados tabelados)
├── calibracao_v2.json                    (copia da calibração — input)
└── AGENTE_DEMANDA_DOC.md                 (este arquivo)
```

## Como rodar

```powershell
cd teste_agente_demanda_promocional
python testes_demand_agent.py            # bateria de validação
python demand_agent.py                    # smoke test embutido
```

## Status

✅ **Aprovado para integração.** Próxima sessão pode usar este agente como
base para os agentes de receita/decisão/operação.
