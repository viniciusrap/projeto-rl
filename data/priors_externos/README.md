# Priors externos — fontes de calibração além dos dados do posto

Este diretório contém séries e estatísticas de fontes públicas usadas como
**prior bayesiano** para o modelo. Não substituem dados do posto — calibram
expectativa quando o histórico interno é insuficiente (ex: chocolate ainda
fora do catálogo, ou poucas amostras de Black Friday).

## Estrutura

```
priors_externos/
├── README.md                          (este arquivo)
├── google_trends/                     (séries semanais, índice 0-100, BR, 5y)
│   ├── _indice.csv                    (metadados das séries coletadas)
│   ├── cerveja.csv
│   ├── vinho.csv
│   ├── espumante.csv
│   ├── whisky.csv
│   ├── chocolate.csv
│   ├── refrigerante.csv
│   └── energetico.csv
├── uplift_trends_por_evento.csv       (uplift medido por (termo, evento, ano))
└── uplift_trends_agregado.csv         (média entre anos)
```

## O que foi coletado (2026-05-11)

| Fonte                | Status      | Cobre o quê                              |
|----------------------|-------------|------------------------------------------|
| **Google Trends BR** | ✓ parcial  | 7 de 15 termos (rate-limit 429)         |
| **Calendário BR**    | ✓ pronto   | 190 eventos 2020-2027                   |
| **Olist (Kaggle)**   | ⏳ pendente | Requer login Kaggle manual              |
| **IBGE PMC**         | ⏳ pendente | API SIDRA, índice mensal varejo         |

## Limitações importantes (lições do dia)

### Google Trends ≠ vendas

Trends mede **buscas**, não compras. Vimos que:

- **Capta bem compra planejada (presente):**
  espumante no Réveillon medido **6.32×** vs prior 3.0×
  vinho no Natal medido **1.76×**
- **Subestima compra rotineira/impulso:**
  cerveja no Réveillon medido só 1.57× (na vida real é muito maior)
  chocolate no Dia dos Namorados medido só 1.16×

### Uso recomendado dos priors

| Tipo de produto              | Fonte do prior |
|------------------------------|----------------|
| Bebidas presente (espumante, vinho premium, whisky) | **Google Trends ✓** |
| Bebidas rotina (cerveja, refrigerante, água)         | Literatura + dado do posto |
| Doces/chocolate em datas comerciais                  | **Olist (quando vier)** + dado do posto |
| Snacks em eventos esportivos                         | Literatura + dado do posto |

## Como coletar os termos do Google Trends que faltaram

Os 8 termos abaixo falharam por rate-limit (429). Esperar 30+ min e rodar:

```powershell
.venv\Scripts\python.exe coletar_google_trends.py
```

O script é idempotente — só repete os termos que ainda não tem arquivo. (TODO:
adicionar essa lógica no script. Hoje ele tenta tudo de novo. Workaround: editar
a lista `TERMOS` removendo os que já vieram.)

Termos pendentes:
- `panetone` (Natal)
- `sorvete` (verão)
- `salgadinho` (eventos esportivos)
- `presente dia dos namorados` (intenção)
- `presente dia das mães` (intenção)
- `presente dia dos pais` (intenção)
- `black friday` (evento)
- `copa do mundo` (evento)

## Como baixar Olist (TODO manual)

Olist é um dataset público de 100k pedidos de e-commerce brasileiro
(2016-2018). Útil para extrair uplift de chocolate/vinho/snack em datas
comerciais.

**Por que precisa intervenção manual:** Kaggle exige login. Não dá para
baixar via URL anônima.

**Passos:**
1. Criar conta em kaggle.com (gratuita)
2. Em Account → API → Create New API Token → baixa `kaggle.json`
3. Colocar em `~/.kaggle/kaggle.json` (Linux/Mac) ou `%USERPROFILE%\.kaggle\kaggle.json` (Win)
4. Rodar:
   ```powershell
   .venv\Scripts\pip install kaggle
   .venv\Scripts\kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw_olist/ --unzip
   ```
5. Rodar script de processamento (a criar):
   `processar_olist.py` — extrai uplift por categoria × data comercial

**Alternativa sem Kaggle:** baixar manualmente do site, extrair na pasta
`data/raw_olist/`, rodar processamento.

## Como baixar IBGE PMC (futuro)

API SIDRA gratuita, sem login:

```python
import requests
# Tabela 8881 — índice de volume de vendas no comércio varejista
url = 'https://apisidra.ibge.gov.br/values/t/8881/n1/all/v/all/p/all/c11046/all'
df = pd.DataFrame(requests.get(url).json())
```

Atividades relevantes (código c11046):
- `40312` — Hipermercados e supermercados
- `40313` — Produtos alimentícios, bebidas e fumo
- `40316` — Outros artigos de uso pessoal e doméstico

Devolve índice mensal nacional. Útil para validar sazonalidade macro,
não suficiente para granularidade diária/semanal.
