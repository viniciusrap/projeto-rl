"""V16 — Gerador de operação: cartaz HTML imprimível + etiqueta PDV + relatório.

Pega o calendário operacional e produz:
1. **Cartazes A4** (HTML imprimível) — 1 por campanha eventual + 1 por estrutural
2. **Etiquetas PDV** (CSV) — código_PDV + preço promocional + dias válidos
3. **Relatório semanal** (Markdown) — o que rodar essa semana + treinamento da equipe

Uso:
    python gerar_operacao.py \\
        --input results/v16/calendario_operacional.json \\
        --output_dir results/v16/operacao/

Saídas:
    results/v16/operacao/cartazes/cartaz_NNN.html
    results/v16/operacao/etiquetas_pdv.csv
    results/v16/operacao/treinamento_equipe.md
    results/v16/operacao/relatorio_semana_YYYY-MM-DD.md
"""
import argparse
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='results/v16/calendario_operacional.json')
parser.add_argument('--output_dir', type=str, default='results/v16/operacao')
parser.add_argument('--semana', type=str, default=None,
                     help='Data início da semana atual (default: hoje)')
args = parser.parse_args()

OUT = ROOT / args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'cartazes').mkdir(exist_ok=True)

# ── Carregar calendário operacional ──────────────────────────────────────

print(f"Carregando {args.input}…")
with open(ROOT / args.input, encoding='utf-8') as f:
    cal = json.load(f)

print(f"  {len(cal['campanhas_estruturais'])} estruturais + {len(cal['campanhas_eventuais'])} eventuais")


# ── 1. Gerar cartazes HTML (A4 imprimível) ───────────────────────────────

CARTAZ_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Cartaz — {nome}</title>
<style>
  @page {{ size: A4 portrait; margin: 1cm; }}
  body {{
    font-family: 'Impact', 'Arial Black', sans-serif;
    margin: 0; padding: 40px;
    text-align: center;
    color: #1a1a1a;
  }}
  .banner {{
    background: linear-gradient(135deg, #dc2626 0%, #fbbf24 100%);
    color: white;
    padding: 20px 40px;
    border-radius: 12px;
    font-size: 32px;
    font-weight: 900;
    letter-spacing: 4px;
    margin-bottom: 40px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  }}
  .produto {{
    font-size: 56px;
    font-weight: 900;
    line-height: 1.1;
    margin: 60px 0 30px;
    text-transform: uppercase;
  }}
  .par {{
    font-size: 36px;
    margin: 20px 0;
    color: #6b7280;
  }}
  .preco-de {{
    font-size: 24px;
    color: #9ca3af;
    text-decoration: line-through;
    margin: 30px 0 10px;
  }}
  .preco-por {{
    font-size: 96px;
    font-weight: 900;
    color: #dc2626;
    line-height: 1;
  }}
  .preco-por .reais {{ font-size: 64px; vertical-align: top; }}
  .periodo {{
    font-size: 28px;
    font-weight: 700;
    margin-top: 50px;
    padding: 16px;
    background: #fbbf24;
    border-radius: 8px;
    display: inline-block;
  }}
  .footer {{
    margin-top: 80px;
    color: #6b7280;
    font-size: 16px;
  }}
</style>
</head>
<body>
  <div class="banner">{titulo_banner}</div>
  <div class="produto">{produto_principal}</div>
  {par_html}
  {preco_de_html}
  <div class="preco-por">
    <span class="reais">R$</span> {preco_inteiro}<span class="reais">,{preco_decimal}</span>
  </div>
  <div class="periodo">{periodo}</div>
  <div class="footer">Auto Posto Parque Viana · Promoção válida no estoque vigente</div>
</body>
</html>
"""


def fmt_preco(p):
    inteiro = int(p)
    decimal = int(round((p - inteiro) * 100))
    return inteiro, f"{decimal:02d}"


def gerar_cartaz(camp, idx, tipo='eventual'):
    cat = camp['categoria'].replace('_', ' ').title()
    par_str = ''
    titulo = 'PROMOÇÃO'
    periodo = ''

    if tipo == 'estrutural':
        nome = camp['nome']
        par = camp['par_combo'].replace('_', ' ').title()
        preco = camp['preco_combo']
        titulo = nome.upper()
        par_str = f'<div class="par">+ {par}</div>'
        periodo = camp['comunicacao'].split('•')[-1].strip()
    else:
        preco = camp.get('preco_combo_alvo', 9.90)
        if camp.get('produto_complementar') and camp.get('intensidade') == 'combo':
            par = camp['produto_complementar'].replace('_', ' ').title()
            par_str = f'<div class="par">+ {par}</div>'
            titulo = 'COMBO ESPECIAL'
        else:
            titulo = f"DESCONTO {camp.get('desconto_pct', 5)}%"
        d_ini = camp['data_inicio'].replace('-', '/')[8:10] + '/' + camp['data_inicio'][5:7]
        d_fim = camp['data_fim'].replace('-', '/')[8:10] + '/' + camp['data_fim'][5:7]
        periodo = f"{d_ini} a {d_fim}"

    # Preço "de" (estimado)
    preco_de_estim = preco / 0.88  # assumindo ~12% off
    preco_de_html = f'<div class="preco-de">De R$ {preco_de_estim:.2f}</div>'

    preco_int, preco_dec = fmt_preco(preco)

    html = CARTAZ_TEMPLATE.format(
        nome=cat,
        titulo_banner=titulo,
        produto_principal=cat,
        par_html=par_str,
        preco_de_html=preco_de_html,
        preco_inteiro=preco_int,
        preco_decimal=preco_dec,
        periodo=periodo,
    )
    fname = f"cartaz_{idx:03d}_{tipo}_{camp.get('categoria', 'X')}.html"
    (OUT / 'cartazes' / fname).write_text(html, encoding='utf-8')
    return fname


# V16.1: AGRUPAR cartazes de eventuais repetitivas (1 cartaz por categoria+
# intensidade+par, com lista de datas embutida). Evita 8 cartazes idênticos.
from collections import defaultdict
cartazes_gerados = []
idx = 0
for c in cal['campanhas_estruturais']:
    idx += 1
    fname = gerar_cartaz(c, idx, 'estrutural')
    cartazes_gerados.append({'tipo': 'estrutural', 'arquivo': fname, 'nome': c['nome']})

# Agrupar eventuais por (categoria, intensidade, par)
grupos = defaultdict(list)
for c in cal['campanhas_eventuais']:
    chave = (c.get('categoria'), c.get('intensidade'),
             c.get('produto_complementar', ''))
    grupos[chave].append(c)

for chave, lista in grupos.items():
    idx += 1
    # Usa primeiro como template, adiciona datas como lista no cartaz
    c_template = dict(lista[0])
    datas_str = ', '.join(f"{c['data_inicio'][8:10]}/{c['data_inicio'][5:7]}"
                            for c in lista)
    if len(lista) > 1:
        c_template['data_inicio'] = lista[0]['data_inicio']
        c_template['data_fim'] = lista[-1]['data_fim']
        c_template['lista_datas'] = datas_str
    fname = gerar_cartaz(c_template, idx, 'eventual')
    cartazes_gerados.append({
        'tipo': 'eventual', 'arquivo': fname,
        'n_campanhas_agrupadas': len(lista),
        'categoria': chave[0],
        'datas': datas_str,
    })

print(f"\n✓ {len(cartazes_gerados)} cartazes A4 gerados em {OUT/'cartazes'}")


# ── 2. Etiquetas PDV (CSV para upload no sistema) ────────────────────────

linhas_pdv = []
for c in cal['campanhas_estruturais']:
    linhas_pdv.append({
        'tipo': 'estrutural',
        'campanha': c['nome'],
        'categoria_principal': c['categoria'],
        'par_combo': c.get('par_combo', ''),
        'preco_combo': c['preco_combo'],
        'desconto_efetivo_pct': 10,
        'data_inicio': cal['data_inicio'],
        'data_fim': cal['data_fim'],
        'dias_semana_validos': 'ver_padrao',  # vide JSON
        'observacao': c['comunicacao'],
    })
for c in cal['campanhas_eventuais']:
    linhas_pdv.append({
        'tipo': 'eventual',
        'campanha': c.get('comunicacao', '')[:40],
        'categoria_principal': c['categoria'],
        'par_combo': c.get('produto_complementar', ''),
        'preco_combo': c.get('preco_combo_alvo', ''),
        'desconto_efetivo_pct': c.get('desconto_pct', 5),
        'data_inicio': c['data_inicio'],
        'data_fim': c['data_fim'],
        'dias_semana_validos': 'todos',
        'observacao': c.get('comunicacao', ''),
    })

df_pdv = pd.DataFrame(linhas_pdv)
df_pdv.to_csv(OUT / 'etiquetas_pdv.csv', index=False, encoding='utf-8')
print(f"✓ Etiquetas PDV em {OUT/'etiquetas_pdv.csv'} ({len(df_pdv)} linhas)")


# ── 3. Treinamento da equipe ─────────────────────────────────────────────

treinamento = ["""# Treinamento da Equipe — Promoções do Mês

## Como reconhecer combos

Quando o cliente trouxer dois produtos no balcão, o caixa precisa verificar:

1. É uma combinação de combo ativo?
2. Qual o preço promocional?
3. Aplicar manualmente no PDV usando código de campanha (ver tabela)

## Comunicação com o cliente

- "Aproveita o nosso combo de [nome]: leva os 2 por R$ X,XX"
- Apontar para o cartaz ao lado da prateleira
- Se cliente recusar, NÃO insistir — só sugerir uma vez

## Erros comuns

- ⚠️ Aplicar promoção em produto fora da lista — VERIFICAR sempre os 2 produtos
- ⚠️ Cliente quer combo mas só tem 1 produto em estoque → vender separado pelo preço normal
- ⚠️ Promoção vencida — verificar data_fim no PDV antes de aplicar

---

## Campanhas ESTRUTURAIS (sempre ativas)
"""]

for c in cal['campanhas_estruturais']:
    treinamento.append(f"""
### {c['nome']}
- **O que oferecer:** {c['comunicacao']}
- **Categoria principal:** {c['categoria']}
- **Par de combo:** {c['par_combo']}
- **Preço final:** R$ {c['preco_combo']:.2f}
- **Justificativa:** {c['justificativa']}
""")

treinamento.append("\n## Campanhas EVENTUAIS deste período\n")
for c in cal['campanhas_eventuais'][:8]:
    treinamento.append(f"""
### {c['data_inicio']} → {c['data_fim']}
- **Categoria:** {c['categoria']}
- **Tipo:** {c.get('intensidade', '?')}
- **Comunicação:** {c.get('comunicacao', '')}
- **Preço alvo:** R$ {c.get('preco_combo_alvo', 0):.2f}
""")

treinamento.append("""
---

## Dúvidas frequentes do cliente

**Cliente:** "Vocês cobram caro né?"
**Resposta:** "Temos vários combos promocionais. Hoje, por exemplo, [apontar para o cartaz mais relevante]."

**Cliente:** "Posso pegar só o de menor preço do combo?"
**Resposta:** "Pode sim. Mas se levar os 2 por R$ X,XX, sai mais barato que comprar separado."

**Cliente:** "Promoção é só para sócio/cartão?"
**Resposta:** "Não, é para todo mundo. Válida até [data_fim]."
""")

(OUT / 'treinamento_equipe.md').write_text(''.join(treinamento), encoding='utf-8')
print(f"✓ Treinamento em {OUT/'treinamento_equipe.md'}")


# ── 4. Relatório da semana ───────────────────────────────────────────────

semana_inicio = date.fromisoformat(args.semana) if args.semana else date.today()
semana_fim = semana_inicio + timedelta(days=6)

relatorio = [f"""# Relatório Semanal — {semana_inicio.strftime('%d/%m/%Y')} a {semana_fim.strftime('%d/%m/%Y')}

## Campanhas a rodar esta semana

### Estruturais (sempre ativas)
"""]

for c in cal['campanhas_estruturais']:
    relatorio.append(f"- ✓ **{c['nome']}** — {c['comunicacao']}\n")

# Eventuais que caem na semana
eventuais_semana = [
    c for c in cal['campanhas_eventuais']
    if not (date.fromisoformat(c['data_fim']) < semana_inicio
             or date.fromisoformat(c['data_inicio']) > semana_fim)
]

relatorio.append(f"\n### Eventuais ativas ({len(eventuais_semana)})\n")
for c in eventuais_semana:
    relatorio.append(f"""
**{c['data_inicio']} → {c['data_fim']}** ({c.get('dias_total', '?')}d)
- {c.get('comunicacao', '')}
- Preço alvo: R$ {c.get('preco_combo_alvo', 0):.2f}
- Lucro estimado: R$ {c.get('lucro_adicional_estimado_R$', 0):.2f}
""")

relatorio.append(f"""
---

## Ações operacionais desta semana

- [ ] Imprimir cartazes (ver pasta `cartazes/`)
- [ ] Atualizar etiquetas no PDV (ver `etiquetas_pdv.csv`)
- [ ] Treinar equipe (ver `treinamento_equipe.md`)
- [ ] Verificar estoque das categorias-alvo
- [ ] Encerrar campanhas vencidas

## Métricas a acompanhar

- Vendas totais por categoria promovida
- Comparação com semana anterior (sem promo)
- Reclamações ou dúvidas registradas no caixa

## Próximos passos (semana que vem)

- Coletar vendas reais das campanhas que terminaram
- Rodar CausalImpact para medir uplift verdadeiro
- Atualizar elasticidade de cada categoria no modelo
""")

fname_rel = f"relatorio_semana_{semana_inicio.isoformat()}.md"
(OUT / fname_rel).write_text(''.join(relatorio), encoding='utf-8')
print(f"✓ Relatório em {OUT/fname_rel}")


# ── Sumário final ────────────────────────────────────────────────────────

print()
print("=" * 80)
print("OPERAÇÃO V16 GERADA")
print("=" * 80)
print(f"  📂 Cartazes A4 imprimíveis:  {OUT/'cartazes'}/  ({len(cartazes_gerados)} arquivos)")
print(f"  📊 Etiquetas PDV (CSV):       {OUT/'etiquetas_pdv.csv'}")
print(f"  📋 Treinamento equipe:        {OUT/'treinamento_equipe.md'}")
print(f"  📅 Relatório semanal:         {OUT/fname_rel}")
print()
print("Como usar:")
print("  1. Imprimir cartazes da pasta cartazes/ (A4)")
print("  2. Carregar etiquetas_pdv.csv no sistema PDV")
print("  3. Fazer reunião de 15min com equipe usando treinamento_equipe.md")
print("  4. Acompanhar relatório semanal toda segunda-feira")
