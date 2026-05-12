# Calendário de Promoções V3 — Modelo V11

**Gerado em:** 12/05/2026
**Horizonte:** 60 dias (12/05/2026 a 10/07/2026)
**Modelo:** DQN V11 com 18 categorias + calendário comercial BR

## Resumo

- **12 campanhas** recomendadas
- **Lucro adicional estimado total:** R$ 118,76

## Campanhas

### Campanha 1: 12/05 a 17/05 (6 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +2.6 unidades
- **Lucro adicional estimado:** R$ 12,06

### Campanha 2: 19/05 a 21/05 (3 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +1.3 unidades
- **Lucro adicional estimado:** R$ 6,03

### Campanha 3: 23/05 a 28/05 (6 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +2.6 unidades
- **Lucro adicional estimado:** R$ 12,06

### Campanha 4: 30/05 a 31/05 (2 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +0.9 unidades
- **Lucro adicional estimado:** R$ 4,02

### Campanha 5: 01/06 a 02/06 (2 dias)

- **Categoria:** cigarro_philip_morris
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 11.2
- **Margem unitária:** R$ 1.65
- **Uplift estimado:** +19.3 unidades
- **Lucro adicional estimado:** R$ 30,30

### Campanha 6: 03/06 a 07/06 (5 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +2.2 unidades
- **Lucro adicional estimado:** R$ 10,05
- **🎯 Coincide com:** Dia dos Namorados

### Campanha 7: 09/06 a 15/06 (7 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +3.1 unidades
- **Lucro adicional estimado:** R$ 14,08
- **🎯 Coincide com:** Copa 2026 — Abertura, Dia dos Namorados, Copa 2026 — Estreia provável Brasil

### Campanha 8: 16/06 a 22/06 (7 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +3.1 unidades
- **Lucro adicional estimado:** R$ 14,08
- **🎯 Coincide com:** Copa 2026 — Fase de grupos Brasil

### Campanha 9: 23/06 a 28/06 (6 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +2.6 unidades
- **Lucro adicional estimado:** R$ 12,06
- **🎯 Coincide com:** Copa 2026 — Fase de grupos Brasil

### Campanha 10: 01/07 a 02/07 (2 dias)

- **Categoria:** suco
- **Tipo:** nada  (desconto 0%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +0.0 unidades
- **Lucro adicional estimado:** R$ 0,00

### Campanha 11: 03/07 a 04/07 (2 dias)

- **Categoria:** suco
- **Tipo:** desc5%  (desconto 5%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +0.9 unidades
- **Lucro adicional estimado:** R$ 4,02

### Campanha 12: 05/07 a 09/07 (5 dias)

- **Categoria:** suco
- **Tipo:** nada  (desconto 0%)
- **Preço médio da categoria:** R$ 8.85
- **Margem unitária:** R$ 4.83
- **Uplift estimado:** +0.0 unidades
- **Lucro adicional estimado:** R$ 0,00

---

## Limitações desta V3

1. **Demanda calibrada por categoria, não por SKU** — esperando dados detalhados do ERP
2. **Validade típica heurística** — esperando dado do posto
3. **Elasticidade da literatura** — validação real só com teste A/B
4. **Combos via heurística** — esperando cupom fiscal para Apriori

## Para refinar

Quando o ERP exportar vendas detalhadas por SKU e cupom fiscal, rodar:

```powershell
python calibrar_v2.py        # re-calibra com dados completos
python treinar_v11.py        # re-treina V11
python validar_v11.py        # valida metricas
python gerar_calendario_v3.py # gera novo calendario
```

---

*V3 gerada em 2026-05-12 pelo modelo DQN V11 treinado.*