# Calendário de Promoções Recomendadas — V1 (protótipo)

**Gerado em:** 11/05/2026
**Horizonte:** próximos 60 dias (11/05 a 09/07/2026)
**Modelo:** DQN V10 — 6 produtos (energético, gelo, refrigerante, água, cerveja, sorvete)

## Resumo

- **9 campanhas** recomendadas no período
- **Lucro adicional estimado total:** R$ 233,45
- **Tipo único:** combo (V10 não usa descontos parciais ou liquidação)

## Calendário

### Campanha 1: 11/05 a 15/05 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Água
- **Produto complementar (compra junto):** Energético
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 27,70
- **Uplift estimado:** +4.8 un de Água, +1.0 un de Energético
- **Razão:** período historicamente fraco (demanda esperada a 48% do baseline)

### Campanha 2: 18/05 a 22/05 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Cerveja
- **Produto complementar (compra junto):** Refrigerante
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 21,86
- **Uplift estimado:** +2.7 un de Cerveja, +2.0 un de Refrigerante
- **Razão:** período historicamente fraco (demanda esperada a 48% do baseline)

### Campanha 3: 25/05 a 29/05 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Energético
- **Produto complementar (compra junto):** Refrigerante
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 25,68
- **Uplift estimado:** +1.5 un de Energético, +2.0 un de Refrigerante
- **Razão:** período historicamente fraco (demanda esperada a 48% do baseline)

### Campanha 4: 01/06 a 05/06 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Água
- **Produto complementar (compra junto):** Energético
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 31,68
- **Uplift estimado:** +5.5 un de Água, +1.2 un de Energético
- **Razão:** período historicamente fraco (demanda esperada a 55% do baseline)

### Campanha 5: 08/06 a 12/06 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Cerveja
- **Produto complementar (compra junto):** Refrigerante
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 24,99
- **Uplift estimado:** +3.1 un de Cerveja, +2.3 un de Refrigerante
- **Razão:** período historicamente fraco (demanda esperada a 55% do baseline)

### Campanha 6: 15/06 a 19/06 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Energético
- **Produto complementar (compra junto):** Refrigerante
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 29,37
- **Uplift estimado:** +1.7 un de Energético, +2.3 un de Refrigerante
- **Razão:** período historicamente fraco (demanda esperada a 55% do baseline)

### Campanha 7: 22/06 a 26/06 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Água
- **Produto complementar (compra junto):** Energético
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 31,68
- **Uplift estimado:** +5.5 un de Água, +1.2 un de Energético
- **Razão:** período historicamente fraco (demanda esperada a 55% do baseline)

### Campanha 8: 29/06 a 03/07 (5 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Cerveja
- **Produto complementar (compra junto):** Refrigerante
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta, Sexta
- **Lucro adicional estimado:** R$ 22,15
- **Uplift estimado:** +2.7 un de Cerveja, +2.1 un de Refrigerante
- **Razão:** período historicamente fraco (demanda esperada a 48% do baseline)

### Campanha 9: 06/07 a 09/07 (4 dias)

- **Tipo:** Combo promocional
- **Produto principal:** Energético
- **Produto complementar (compra junto):** Refrigerante
- **Desconto recomendado:** 10% no combo
- **Dias da semana:** Segunda, Terça, Quarta, Quinta
- **Lucro adicional estimado:** R$ 18,34
- **Uplift estimado:** +1.1 un de Energético, +1.5 un de Refrigerante
- **Razão:** período historicamente fraco (demanda esperada a 43% do baseline)

---

## Limitações desta V1 (importantes para o seu pai entender)

1. **Apenas 6 produtos.** Chocolate, vinho, snacks, café — ainda fora.
2. **Não enxerga Dia dos Namorados (12/06).** Próxima versão vai considerar.
3. **Elasticidade da literatura.** Uplift estimado é prior, não medição real do posto.
4. **Estoque é simulado.** Não usa o estoque real do ERP.
5. **Pares de combo são heurística.** Precisamos validar com cupom fiscal real.

## O que valida nesta V1

- **Formato do output:** está utilizável pelo dono? Falta alguma informação?
- **Granularidade temporal:** dias, dias-da-semana, intervalo certo?
- **Decisões fazem sentido qualitativamente?** Combos parecem combinar?
- **Layout dos campos:** preferiria ver ROI%, % de margem perdida, outros KPIs?

---

*V1 gerada automaticamente a partir do modelo DQN V10 treinado em 11/05/2026.*