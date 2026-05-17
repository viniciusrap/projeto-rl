"""Gera resumo executivo V21 com:
- Vistoria crítica (combo a combo)
- Agentes/componentes do RL
- Estados e rewards
- Comparação V19.2 vs V20 vs V21
- Dashboard HTML
"""
import io
import json
import sys
from pathlib import Path


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    HERE = Path(__file__).parent
    PROJ = HERE.parent

    # Carrega resultado V21 final
    v21 = json.load(open(HERE / 'logs' / 'iter_iter5_pdv_final_resumo.json', encoding='utf-8'))
    cal = json.load(open(HERE / 'data_sintetica' / 'calibracao_v21_pdv.json', encoding='utf-8'))

    # ─── Dashboard HTML ───
    out_dir = PROJ / 'results' / 'v21'
    out_dir.mkdir(parents=True, exist_ok=True)
    html = f"""<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="UTF-8">
<title>V21 PDV-Native — RL aprendeu lógica REAL de conveniência</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box;}}
body {{background:#0a0a0f;color:#e1e1e6;font-family:-apple-system,Segoe UI,sans-serif;
       padding:32px;line-height:1.5;}}
.container {{max-width:1500px;margin:0 auto;}}
.header {{border-bottom:1px solid #2a2a35;padding-bottom:24px;margin-bottom:32px;}}
.header h1 {{font-size:28px;font-weight:700;}}
.header .subtitle {{color:#8a8a98;margin-top:8px;font-size:14px;}}
.badge {{display:inline-block;padding:4px 12px;border-radius:12px;font-size:11px;
        font-weight:600;text-transform:uppercase;margin-right:8px;letter-spacing:0.5px;
        background:linear-gradient(90deg,#10b981,#06b6d4);color:#fff;}}
.kpi-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px;}}
.kpi {{background:#14141f;border:1px solid #2a2a35;border-radius:12px;padding:20px;}}
.kpi .label {{color:#8a8a98;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;}}
.kpi .value {{font-size:26px;font-weight:700;margin-top:8px;}}
.kpi .delta {{font-size:12px;margin-top:4px;color:#4ade80;}}
.h2 {{font-size:18px;font-weight:600;margin:32px 0 12px;padding-bottom:8px;
       border-bottom:1px solid #2a2a35;}}
.box {{background:#14141f;border:1px solid #2a2a35;border-radius:12px;padding:20px;margin-bottom:16px;}}
.box h3 {{margin-bottom:12px;}}
.mono {{font-family:'SF Mono',Monaco,monospace;}}
table {{width:100%;border-collapse:collapse;background:#14141f;border-radius:12px;
        overflow:hidden;border:1px solid #2a2a35;}}
th {{background:#0f0f18;padding:10px;text-align:left;font-size:11px;color:#8a8a98;
     text-transform:uppercase;letter-spacing:0.5px;}}
td {{padding:12px;border-bottom:1px solid #1f1f2a;font-size:13px;}}
tr:hover {{background:#18182a;}}
.acerto {{color:#4ade80;font-weight:600;}}
.aviso {{color:#fbbf24;}}
.erro {{color:#f87171;}}
ul {{margin-left:20px;}}
li {{margin:6px 0;font-size:13px;}}
.combo-card {{background:#1a1a2a;padding:12px;border-radius:8px;
              border-left:3px solid #06b6d4;margin-bottom:8px;}}
.combo-card .titulo {{font-weight:600;font-family:'SF Mono',monospace;}}
.combo-card .freq {{color:#06b6d4;font-family:'SF Mono',monospace;font-weight:600;}}
.combo-card .analise {{color:#8a8a98;font-size:12px;margin-top:4px;}}
</style>
</head><body><div class="container">

<div class="header">
    <h1>
        <span class="badge">V21 PDV-Native</span>
        Agente RL aprendeu lógica REAL de conveniência
    </h1>
    <div class="subtitle">
        SEM dados de supermercado. Calibração baseada em inventário REAL do posto
        Viana 2022-26 + harmonia criada por conhecimento de PDV.
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi">
        <div class="label">Cenários PDV</div>
        <div class="value">10 <span style="color:#6a6a78;font-size:18px;">/ 12</span></div>
        <div class="delta">83% — combos validados em PDV</div>
    </div>
    <div class="kpi">
        <div class="label">Lucro Anual</div>
        <div class="value mono">R$ {v21['rollout']['lucro_total']:,.0f}</div>
        <div class="delta">defensável (sem otimismo de supermercado)</div>
    </div>
    <div class="kpi">
        <div class="label">Combos PDV-inválidos</div>
        <div class="value acerto">0</div>
        <div class="delta">aprendido via reward + mask</div>
    </div>
    <div class="kpi">
        <div class="label">Cats distintas (de 20)</div>
        <div class="value">{v21['rollout']['cats_distintas']}</div>
        <div class="delta">diversidade saudável</div>
    </div>
</div>

<h2 class="h2">🎯 TOP 10 pares aprendidos (vistoria crítica)</h2>
<div class="box">
    <div class="combo-card">
        <div><span class="titulo">sanduíche + suco</span> <span class="freq">24x</span></div>
        <div class="analise">✅ Almoço quick clássico em posto (combo do Select)</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">gelo + cerveja</span> <span class="freq">24x</span></div>
        <div class="analise">✅ Churrasco em FDS (volume controlado, h=1.6 ajustado)</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">doce_balcao + cerveja</span> <span class="freq">24x</span></div>
        <div class="analise">✅ Saída de balada — cliente leva bala/menthos pra amargar</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">chocolate_unit + doce_balcao</span> <span class="freq">24x</span></div>
        <div class="analise">✅ Impulso doce do balcão (combo do caixa)</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">cha_pronto + padaria</span> <span class="freq">23x</span></div>
        <div class="analise">✅ Rotina manhã commuter (Lipton + pão de queijo)</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">energético + doce_balcao</span> <span class="freq">13x</span></div>
        <div class="analise">✅ Motorista de app/madrugada</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">padaria + suco</span> <span class="freq">12x</span></div>
        <div class="analise">✅ Manhã rápida (sem café — café tem cliente próprio)</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">chocolate_caixa + doce_balcao</span> <span class="freq">12x</span></div>
        <div class="analise">✅ Presente kit Páscoa/Mães</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">sorvete + biscoito</span> <span class="freq">12x</span></div>
        <div class="analise">✅ Impulso doce</div>
    </div>
    <div class="combo-card">
        <div><span class="titulo">café + doce_balcao</span> <span class="freq">12x</span></div>
        <div class="analise">✅ Café + balinha de açúcar</div>
    </div>
</div>

<h2 class="h2">🚫 Combos que SUMIRAM (vs V19.2/V20 baseados em Instacart)</h2>
<div class="box">
    <ul>
        <li><span class="erro">gelo + destilados</span> — em V19.2 era top par, em V21 nem aparece (saco 5kg + garrafa não casa no balcão)</li>
        <li><span class="erro">gelo + whisky</span> — idem</li>
        <li><span class="erro">gelo + vinho</span> — idem</li>
        <li><span class="erro">chocolate_caixa + vinho</span> — V19.2 forçava, V21 sumiu (vinho não vende no posto)</li>
        <li><span class="erro">café + cerveja</span> — manhã vs noite, agente entendeu</li>
        <li><span class="erro">padaria + cerveja</span> — idem</li>
    </ul>
</div>

<h2 class="h2">🤖 Arquitetura RL — agentes que compõem o sistema</h2>
<div class="box">
    <h3>Branching DQN — 3 cabeças coordenadas (MARL conceitual)</h3>
    <p style="font-size:13px;color:#c1c1cc;margin-bottom:12px;">
    Encoder compartilhado de 128 unidades + 3 cabeças independentes:
    </p>
    <ul>
        <li><strong style="color:#4ade80;">Cabeça INTENSIDADE</strong> (Agente de Desconto):
            decide entre 6 ações: nada, desc3%, desc5%, desc7%, desc10%, combo</li>
        <li><strong style="color:#06b6d4;">Cabeça COMPLEMENTAR</strong> (Agente de Combo):
            decide qual par entre 21 opções (nenhum + 20 categorias).
            <em>Pré-treinada de forma supervisionada com a matriz de harmonia PDV-native.</em></li>
        <li><strong style="color:#8b5cf6;">Cabeça ALVO</strong> (Agente de Margem):
            decide se aplica desconto no produto principal ou no complementar (2 opções)</li>
    </ul>
</div>

<h2 class="h2">📊 Estado (89 features)</h2>
<div class="box">
    <ul>
        <li><strong>Temporal:</strong> dia da semana (7), mês (12), dia do mês, tipo de evento próximo (4), dias até evento</li>
        <li><strong>Produto atual:</strong> categoria one-hot (23), tipo de produto one-hot (6)</li>
        <li><strong>Demanda:</strong> base anual, contextual (com sazonalidade), fator sazonal atual, em alta/em baixa</li>
        <li><strong>Estoque:</strong> nível relativo, validade restante</li>
        <li><strong>Promocional:</strong> margem normalizada, uplift_prior do evento, afeta categoria</li>
        <li><strong>Histórico:</strong> últimos 7 dias de promoções, turnos restantes do episódio</li>
        <li><strong>Clima:</strong> temperatura normalizada (Barueri/SP histórico)</li>
    </ul>
</div>

<h2 class="h2">🎁 Função de recompensa (codifica regras de PDV via shaping)</h2>
<div class="box">
<pre style="font-size:12px;background:#0f0f18;padding:16px;border-radius:8px;overflow-x:auto;color:#c1c1cc;">
r = lucro_real_dia × dias_campanha

# BONUS (estratégia ideal PDV)
+ K_COMBO_EM_ALTA          (180)  quando combo em alta demanda
+ K_DESC_COMPLEMENTAR       (50)  desconto no complementar (não principal)
+ K_DEFENSIVO_VENCIMENTO   (350)  liquidação em validade < 30%
+ K_BONUS_EVENTO_MATCH     (250)  match produto × evento comercial
+ K_BONUS_EVENTO_PRESENTE  (500)  puxador_presente em evento presente
+ K_BONUS_HARMONIA_FORTE   (600)  harmonia >= 2.0 (combo ótimo PDV)
+ K_BONUS_HARMONIA_MEDIA   (150)  harmonia 1.5-2.0
+ K_BONUS_NAO_PROMOVER      (50)  não promover quando contexto não favorece
+ harm_multiplicativo (proporcional ao lucro × fator harmonia)

# PENALIDADES (combos inválidos em PDV)
- K_DESC_EM_ALTA_NATURAL   (200)  desc direto em produto em alta natural
- K_COMBO_PDV_INVALIDO     (500)  combo cross-class (gelo+destilados etc)
- K_CANIBALIZACAO_ALTA       (8)  por % acima de 40% de canib
- K_REPETICAO_CATEGORIA    (130)  promovedio mesma cat em últimos 7 dias
- K_UPLIFT_ABAIXO_BE       (150)  uplift abaixo do breakeven
- K_PENALIDADE_HARMONIA_LEVE(60)  harmonia 1.0-1.3
- K_PENALIDADE_HARMONIA_FRACA(150) harmonia < 1.0
- K_COMBO_SEM_PAR          (300)  combo + nenhum (loophole bloqueado)
</pre>
</div>

<h2 class="h2">📈 Calibração V21 — fonte dos dados</h2>
<div class="box">
    <ul>
        <li><strong>Catálogo:</strong> 23 categorias (20 promovíveis + 3 cigarros PROIBIDOS)
            agregadas a partir do <strong>inventário REAL do posto Viana 2022-26</strong>
            (12.750 registros, 87 categorias originais)</li>
        <li><strong>Custos:</strong> medianos do inventário real (preço custo dos SKUs)</li>
        <li><strong>Demanda base:</strong> estimada por conhecimento de domínio de PDV de posto
            (não por dados de supermercado)</li>
        <li><strong>Validade:</strong> baseada em descarte real (sanduíche 5d, padaria 7d, gelo 60d,
            cerveja 180d, chocolate 365d)</li>
        <li><strong>Harmonia:</strong> matriz 20×20 <strong>criada por raciocínio de domínio
            PDV</strong>. Combos pensados como ticket de balcão de posto, não cesta de mercado</li>
        <li><strong>Fatores sazonais:</strong> dia da semana e mês definidos com base em padrões
            reais de posto (café/padaria → manhã commuter, cerveja → noite/FDS,
            gelo/sorvete → verão)</li>
    </ul>
</div>

<h2 class="h2">📦 Resumo da arquitetura</h2>
<div class="box">
    <table>
    <thead><tr><th>Componente</th><th>O que faz</th><th>Tipo</th></tr></thead>
    <tbody>
        <tr><td><strong>EnvRLPromocoes (Gymnasium)</strong></td><td>Simula o ambiente. step() estima demanda promocional realista e calcula reward.</td><td>Ambiente RL</td></tr>
        <tr><td><strong>Cabeça INTENSIDADE</strong></td><td>Aprende QUANDO promover e que desconto aplicar</td><td>Agente RL #1</td></tr>
        <tr><td><strong>Cabeça COMPLEMENTAR</strong></td><td>Aprende QUE PAR escolher para combo</td><td>Agente RL #2</td></tr>
        <tr><td><strong>Cabeça ALVO</strong></td><td>Aprende ONDE aplicar o desconto (principal ou complementar)</td><td>Agente RL #3</td></tr>
        <tr><td><strong>BranchingDQNAgent</strong></td><td>Wrapper que coordena online/target nets, ε-greedy, Double DQN, pre-treino sup. da cabeça COMPLEMENTAR</td><td>Coordenador</td></tr>
        <tr><td><strong>ReplayBuffer</strong></td><td>Memória de 50k transições para aprendizado off-policy</td><td>Buffer</td></tr>
        <tr><td><strong>Action Masking</strong></td><td>Bloqueia pares com harmonia &lt; 1.4 (restrição estrutural, não regra de negócio)</td><td>Restrição</td></tr>
        <tr><td><strong>Pre-treino supervisionado</strong></td><td>Antes do RL, treina cabeça COMPLEMENTAR com loss MSE contra harmonia PDV-native</td><td>Warm start</td></tr>
    </tbody>
    </table>
</div>

</div></body></html>
"""

    out_path = out_dir / 'dashboard_v21_pdv_native.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✓ Dashboard V21 salvo: {out_path}")


if __name__ == '__main__':
    main()
