# Agentes de Demanda Promocional — organização por versão

Cada versão é **isolada**. Só usa:
- arquivos da própria pasta `vXX/`
- `_common/` (docs + inputs padrão)
- `data/` da raiz do projeto

```
teste_agente_demanda_promocional/
├── _common/                # docs + input_calendarios/ comuns
├── _arquivo_v18/           # versões antigas (não rodar)
├── v19/                    # BASELINE preservado
│   ├── calibracao_v2.json
│   ├── demand_agent.py     # estima demanda promocional
│   ├── revenue_agent.py    # calcula lucro real
│   ├── decision_agent.py   # decide viabilidade
│   ├── pipeline.py         # orquestra os 3 + lê V17 + grava em results/v19/
│   ├── gerar_dashboard.py
│   ├── comparar_baseline.py
│   ├── testes_demand_agent.py
│   ├── stress_test_agent.py
│   └── README.md
└── v19_1/                  # Fase A + B (sazonalidade + calendário rico)
    ├── calibracao_v2.json
    ├── ... (mesmos arquivos, modificados)
    └── README.md
```

## Como rodar

```powershell
cd C:\Users\vinin\projeto-rl\teste_agente_demanda_promocional\v19      # ou v19_1
..\..\.venv\Scripts\python.exe pipeline.py
..\..\.venv\Scripts\python.exe gerar_dashboard.py
```

Outputs em `results/vXX/` (uma pasta por versão).

## Regras de versionamento

1. **NÃO importar de outra pasta vXX/**. Cada versão é self-contained.
2. **Sempre criar nova pasta** ao mudar fórmula crítica (ex: V19 → V19.1 quando
   separei sazonalidade do uplift).
3. **`_common/`** só recebe coisas que NÃO mudam por versão:
   - docs gerais
   - input_calendarios/ (saídas do V15/V17 que servem de entrada)
4. **`data/` da raiz** é fonte canônica de:
   - calendario_comercial.csv
   - temperatura_historica.csv
   - venda_por_dia.xlsx
   - venda_do_mes.xlsx

## Histórico

- **V19** (baseline): primeiro pipeline 3-agentes. Demand → Revenue → Decision.
  Bug conhecido: `boost_dow`/`boost_clima` no lugar errado (multiplicam uplift
  em vez de modular demanda base contextual).
- **V19.1**: Fase A (separar sazonalidade do uplift, usar `uplift_prior`,
  `categorias_afetadas`, `tipo_pico` do calendário) + Fase B (pré-feriado,
  pós-feriado, eventos esportivos, eventos locais, temperatura real).
