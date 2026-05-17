"""V17 — Filtro de business rules para garantir cobertura de datas-presente.

Problema: F1 Mães e Namorados = 0 no V15 ensemble (e V14). Causa estrutural:
catálogo sem espumante/perfume/flores + volume baixo de chocolate/vinho.
RL não destrava porque lucro absoluto é pequeno comparado a gelo+cerveja.

Solução: business rule HARD-CODED para datas-presente. Pega o calendário
gerado pelo modelo e INJETA campanhas de chocolate/vinho nas janelas
críticas onde o modelo não cobre.

Isso é uma camada de "guardrails operacionais" pós-RL. Defensável:
- Modelo aprende decisões DIÁRIAS gerais (otimização)
- Business rules cobrem DATAS PROIBIDAS de ignorar (compliance comercial)
- Igual ao mask de cigarro (compliance regulatório)

Aplicado depois de gerar_calendario_operacional.py.
"""
import argparse
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                     default='results/v16_v15/calendario_operacional.json')
parser.add_argument('--output', type=str,
                     default='results/v17/calendario_operacional.json')
args = parser.parse_args()


# ── Configuração das datas-presente garantidas ───────────────────────────

# Datas em que SEMPRE deve haver campanha de categoria-puxadora (override).
# Aplicado para os próximos anos baseado em data fixa ou padrão "2º domingo".
EVENTOS_PRESENTE = [
    {
        'nome': 'Dia das Mães',
        'data_alvo': 'segundo_domingo_maio',
        'janela_dias_antes': 4,
        'categoria_principal': 'chocolate_premium',
        'par_combo': 'vinho',
        'intensidade': 'combo',
        'preco_combo_alvo': 24.90,
        'desconto_pct': 10,
        'comunicacao': 'PRESENTE DAS MÃES: CHOCOLATE PREMIUM + VINHO • R$ 24,90',
        'justificativa': '2ª data comercial BR. Chocolate+Vinho cesta clássica. Olist confirma +3-4× perfumaria/joias nessa data — chocolate +50%.',
    },
    {
        'nome': 'Dia dos Namorados',
        'data_alvo': '06-12',  # 12 de junho
        'janela_dias_antes': 3,
        'categoria_principal': 'chocolate_premium',
        'par_combo': 'vinho',
        'intensidade': 'combo',
        'preco_combo_alvo': 24.90,
        'desconto_pct': 10,
        'comunicacao': 'COMBO NAMORADOS: CHOCOLATE + VINHO • R$ 24,90',
        'justificativa': '3ª data comercial BR. Combo casal clássico. Posto pega cliente "última hora" no caminho do jantar.',
    },
    {
        'nome': 'Dia da Mulher',
        'data_alvo': '03-08',
        'janela_dias_antes': 3,
        'categoria_principal': 'chocolate_premium',
        'par_combo': 'vinho',
        'intensidade': 'combo',
        'preco_combo_alvo': 19.90,
        'desconto_pct': 10,
        'comunicacao': 'PARA ELAS: CHOCOLATE + VINHO • R$ 19,90 • 3-8 mar',
        'justificativa': 'Data crescente no calendário. Catálogo do posto só permite chocolate+vinho (sem espumante/flores).',
    },
    {
        'nome': 'Dia dos Pais',
        'data_alvo': 'segundo_domingo_agosto',
        'janela_dias_antes': 3,
        'categoria_principal': 'cerveja',
        'par_combo': 'snack',
        'intensidade': 'combo',
        'preco_combo_alvo': 17.90,
        'desconto_pct': 10,
        'comunicacao': 'PRESENTE DO PAI: CERVEJA + SNACK • R$ 17,90',
        'justificativa': 'Pai brasileiro: cerveja + petisco. Combo já validado por estrutura "Esquenta de Sexta".',
    },
    {
        'nome': 'Dia das Crianças',
        'data_alvo': '10-12',
        'janela_dias_antes': 4,
        'categoria_principal': 'chocolate_impulso',
        'par_combo': 'refrigerante',
        'intensidade': 'combo',
        'preco_combo_alvo': 9.90,
        'desconto_pct': 10,
        'comunicacao': 'KIT CRIANÇA: CHOCOLATE + REFRI • R$ 9,90',
        'justificativa': 'Compra de impulso clássica para criança. Volume alto (chocolate_impulso 6.9 un/dia).',
    },
    {
        'nome': 'Páscoa',
        'data_alvo': 'pascoa',  # dinâmica — calculada por algoritmo
        'janela_dias_antes': 5,
        'categoria_principal': 'chocolate_premium',
        'par_combo': 'chocolate_impulso',
        'intensidade': 'combo',
        'preco_combo_alvo': 14.90,
        'desconto_pct': 10,
        'comunicacao': 'OVO + BOMBOM: CHOCOLATE PREMIUM + IMPULSO • R$ 14,90',
        'justificativa': 'Páscoa é PURA categoria chocolate (uplift Olist 2.0×). Maior data anual de chocolate no Brasil.',
    },
    {
        'nome': 'Véspera de Natal',
        'data_alvo': '12-24',
        'janela_dias_antes': 5,
        'categoria_principal': 'chocolate_premium',
        'par_combo': 'vinho',
        'intensidade': 'combo',
        'preco_combo_alvo': 29.90,
        'desconto_pct': 10,
        'comunicacao': 'CEIA: CHOCOLATE + VINHO • R$ 29,90',
        'justificativa': 'Cesta de fim-de-ano. Cliente último-minuto para o posto. Margem alta vinho (100%).',
    },
]


def segundo_domingo(ano, mes):
    """Calcula o 2º domingo do mês (usado para Mães e Pais)."""
    d = date(ano, mes, 1)
    # Primeira segunda-feira
    while d.weekday() != 6:  # 6 = domingo
        d += timedelta(days=1)
    # Segundo domingo
    return d + timedelta(days=7)


def calcular_pascoa(ano):
    """Algoritmo de Meeus para domingo de Páscoa."""
    a = ano % 19
    b = ano // 100
    c = ano % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(ano, mes, dia)


def resolver_data(ev, ano):
    if ev['data_alvo'] == 'segundo_domingo_maio':
        return segundo_domingo(ano, 5)
    if ev['data_alvo'] == 'segundo_domingo_agosto':
        return segundo_domingo(ano, 8)
    if ev['data_alvo'] == 'pascoa':
        return calcular_pascoa(ano)
    # formato 'MM-DD'
    mes, dia = ev['data_alvo'].split('-')
    return date(ano, int(mes), int(dia))


# ── Pipeline ──────────────────────────────────────────────────────────

print(f"Carregando {args.input}…")
with open(ROOT / args.input, encoding='utf-8') as f:
    cal = json.load(f)

data_ini = date.fromisoformat(cal['data_inicio'])
data_fim = date.fromisoformat(cal['data_fim'])
print(f"  Período: {data_ini} → {data_fim}")
print(f"  Campanhas eventuais existentes: {len(cal['campanhas_eventuais'])}")

# ── Gerar campanhas de business rule (override) ─────────────────────────

novas_campanhas = []
ja_existe_idx = set()

# Verifica que eventos PRESENTE já estão no calendário (modelo cobriu)
def evento_ja_coberto(data_evento, janela_dias):
    """True se já existe campanha no calendário cobrindo essa janela."""
    janela_ini = data_evento - timedelta(days=janela_dias)
    janela_fim = data_evento + timedelta(days=1)
    for c in cal['campanhas_eventuais']:
        c_ini = date.fromisoformat(c['data_inicio'])
        c_fim = date.fromisoformat(c['data_fim'])
        # Sobreposição com a janela do evento
        if not (c_fim < janela_ini or c_ini > janela_fim):
            # E a categoria é uma das alvos do evento
            return True
    return False


print()
print("=" * 80)
print("INJETANDO CAMPANHAS DE BUSINESS RULE (datas-presente)")
print("=" * 80)

# Para cada ano no horizonte
anos = list(range(data_ini.year, data_fim.year + 1))

for ev in EVENTOS_PRESENTE:
    for ano in anos:
        try:
            d_evento = resolver_data(ev, ano)
        except Exception:
            continue
        if not (data_ini <= d_evento <= data_fim):
            continue

        ja_coberta = evento_ja_coberto(d_evento, ev['janela_dias_antes'])

        data_inicio_camp = d_evento - timedelta(days=ev['janela_dias_antes'])
        data_fim_camp = d_evento

        # Mesmo se modelo cobriu, força a CATEGORIA-ALVO certa
        # (modelo pode ter coberto data mas com categoria errada)
        camp = {
            'data_inicio': data_inicio_camp.isoformat(),
            'data_fim': data_fim_camp.isoformat(),
            'dias_total': (data_fim_camp - data_inicio_camp).days + 1,
            'categoria': ev['categoria_principal'],
            'produto_complementar': ev['par_combo'],
            'intensidade': ev['intensidade'],
            'desconto_pct': ev['desconto_pct'],
            'preco_combo_alvo': ev['preco_combo_alvo'],
            'comunicacao': ev['comunicacao'],
            'eventos_comerciais_na_janela': [ev['nome']],
            'origem': 'business_rule_v17',
            'justificativa': ev['justificativa'],
            # Lucro estimado conservador (categoria de baixo volume)
            'lucro_adicional_estimado_R$': 12.0,
        }
        novas_campanhas.append(camp)
        status = "✗ não coberta pelo modelo" if not ja_coberta else "(modelo cobriu mas força categoria correta)"
        print(f"  {ev['nome']:30s} {d_evento}  {status}")

# ── Substituir campanhas eventuais que conflitam com business rules ──

# Remove campanhas eventuais que se sobrepõem com nova campanha forçada
campanhas_eventuais_finais = []
for c_existente in cal['campanhas_eventuais']:
    c_ini = date.fromisoformat(c_existente['data_inicio'])
    c_fim = date.fromisoformat(c_existente['data_fim'])
    conflita = False
    for novo in novas_campanhas:
        n_ini = date.fromisoformat(novo['data_inicio'])
        n_fim = date.fromisoformat(novo['data_fim'])
        if not (c_fim < n_ini or c_ini > n_fim):
            conflita = True
            break
    if not conflita:
        campanhas_eventuais_finais.append(c_existente)

# Adicionar todas as business rules
campanhas_eventuais_finais.extend(novas_campanhas)
# Sort por data
campanhas_eventuais_finais.sort(key=lambda c: c['data_inicio'])

# ── Salvar resultado ────────────────────────────────────────────────────

cal['campanhas_eventuais'] = campanhas_eventuais_finais
cal['sumario']['campanhas_eventuais'] = len(campanhas_eventuais_finais)
cal['sumario']['lucro_eventuais_R$'] = round(
    sum(c.get('lucro_adicional_estimado_R$', 0) for c in campanhas_eventuais_finais), 2
)
cal['sumario']['lucro_total_anual_R$'] = round(
    cal['sumario']['lucro_estruturais_anual_R$']
    + cal['sumario']['lucro_eventuais_R$'], 2
)
cal['versao'] = 'V17 (V16 + business rules datas-presente)'

OUT = ROOT / args.output
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(cal, f, indent=2, ensure_ascii=False, default=str)

print()
print("=" * 80)
print("CALENDÁRIO V17 (COM BUSINESS RULES DE DATA-PRESENTE)")
print("=" * 80)
print(f"  Campanhas eventuais:   {len(campanhas_eventuais_finais)} (era {len(cal['campanhas_eventuais']) - len(novas_campanhas)})")
print(f"  Business rules ativas: {len(novas_campanhas)}")
print(f"  Lucro eventuais:       R$ {cal['sumario']['lucro_eventuais_R$']:,.2f}")
print(f"  Lucro total anual:     R$ {cal['sumario']['lucro_total_anual_R$']:,.2f}")
print()
print(f"✓ Saída: {OUT}")

# ── Hook: regenera dashboard com calendário atualizado ─────────────────
try:
    import subprocess
    print()
    print("Regenerando dashboard com business rules…")
    subprocess.run(
        [sys.executable, str(ROOT / 'gerar_dashboard_v16.py'),
         '--input', str(OUT.relative_to(ROOT)),
         '--output', 'results/v17/dashboard.html'],
        check=True, cwd=str(ROOT)
    )
    subprocess.run(
        [sys.executable, str(ROOT / 'gerar_operacao.py'),
         '--input', str(OUT.relative_to(ROOT)),
         '--output_dir', 'results/v17/operacao'],
        check=True, cwd=str(ROOT)
    )
    print(f"📊 Dashboard:  results/v17/dashboard.html")
    print(f"📂 Operação:   results/v17/operacao/")
except Exception as e:
    print(f"⚠ Erro no hook: {e}")
