# V19.1 — Fase A + B

Evolução crítica do V19. Conserta bug de sazonalidade no lugar errado e expande
uso do calendário comercial. **Em desenvolvimento.**

## Fase A — Correções (consertam o V19)

1. **Separar sazonalidade do uplift promocional**
   - V19: `d_promo = d_base × (boost_preco × boost_dow × boost_clima × ...)`
   - V19.1: `d_base_ctx = d_base × boost_dow × boost_clima × boost_pre_feriado`
            `d_promo = d_base_ctx × (1 + boost_preco × boost_combo × boost_evento - 1)`

2. **Usar `uplift_prior` direto do calendário comercial**
   - V19: hardcoded `presente × puxador_presente = +70%`
   - V19.1: lê `data/calendario_comercial.csv` → Mães=2.2×, Namorados=2.5×, Réveillon=3.0×

3. **Usar `categorias_afetadas` reais do calendário**
   - V19: tipo de produto genérico (`puxador_presente`)
   - V19.1: match SKU específico (cerveja em Mães não recebe boost)

4. **Usar `janela_pre_dias` e `tipo_pico`**
   - Mães bate 10 dias antes (`pre`), Black Friday bate no dia (`ambos`)

## Fase B — Novos fatores

5. **Pré-feriado prolongado** — ponte de sex/sáb antes de feriado seg = pico
6. **Pós-feriado** — penalidade (gente viajou)
7. **Eventos esportivos** — Copa Brasil, Brasileirão (uplift 2.5-2.9× cerveja+snack+gelo)
8. **Eventos locais** — Aniv. Barueri, Aniv. SP
9. **Temperatura real** do `data/temperatura_historica.csv` (não mais hardcoded)

## Inputs

- `_common/input_calendarios/v17_business_rules.json` — calendário V17
- `data/calendario_comercial.csv` — 278 eventos BR
- `data/temperatura_historica.csv` — temp diária Barueri 2020-2026
- `calibracao_v2.json` local (mesmo do V19, não recalibrado)

## Como rodar

```powershell
cd v19_1
..\..\.venv\Scripts\python.exe pipeline.py
..\..\.venv\Scripts\python.exe gerar_dashboard.py
```

Outputs em `../../results/v19_1/`.
