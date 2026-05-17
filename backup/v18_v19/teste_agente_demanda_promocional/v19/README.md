# V19 — Baseline (preservado)

**Status**: congelado. Não modificar. Serve como ponto de comparação para V19.1+.

## Resultado conhecido (14/05/2026)

Aplicado ao calendário V17 (4 estruturais + 20 eventuais):
- 24/24 aprovadas (0 rejeições — filtro fraco)
- Lucro estimado: R$ 9.768/ano (+483% vs V17, **suspeito**)
- ROI campanha "Esquenta de Sexta": 2874% (irreal)

## Bug conhecido

Fórmula do uplift mistura sazonalidade com efeito promocional:

```python
# ERRADO (V19):
uplift_bruto = boost_preco × boost_combo × boost_evento × boost_clima × boost_dow × ...
d_promo = d_base × (1 + uplift_final)

# CORRETO (V19.1):
d_base_contextual = d_base × boost_dow × boost_clima × ...   # sazonalidade
uplift_promo = boost_preco × boost_combo × boost_evento - 1   # efeito promo
d_promo = d_base_contextual × (1 + uplift_promo)
```

Sintoma: 12 campanhas de gelo eventuais mostram exatamente os mesmos números
(R$52.32) independente da data — porque o agente não modulou demanda base.

## Como rodar

```powershell
cd v19
..\..\.venv\Scripts\python.exe pipeline.py
..\..\.venv\Scripts\python.exe gerar_dashboard.py
..\..\.venv\Scripts\python.exe comparar_baseline.py
```

Outputs em `../../results/v19/`.
