# atualizar_calendario.ps1 — regenera todo o calendário V3
#
# Quando rodar:
#   - Quando dados do posto mudarem (novo venda_por_dia, novo descarte, etc.)
#   - Quando quiser ver projeção com data inicial diferente
#   - Quando o modelo treinado mudar (após novo treino)
#
# Uso:
#   .\atualizar_calendario.ps1            # regenera com modelo atual
#   .\atualizar_calendario.ps1 -Treinar   # treina V11 novamente antes
#
# Saída: results/v11/calendario_v3_anual.html (abre no navegador)

param(
    [switch]$Treinar = $false,
    [int]$EpisodiosTreino = 150,
    [int]$Horizonte = 365
)

$ErrorActionPreference = "Stop"
$Inicio = Get-Date

Write-Host ""
Write-Host "===================================================="
Write-Host "  ATUALIZAÇÃO DO CALENDÁRIO V3"
Write-Host "===================================================="
Write-Host ""

# Passo 1: re-calibrar
Write-Host "1/4 Re-calibrando V11..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe calibrar_v2.py
if ($LASTEXITCODE -ne 0) { throw "Falha na calibração" }

# Passo 2 (opcional): re-treinar
if ($Treinar) {
    Write-Host ""
    Write-Host "2/4 Re-treinando V11 ($EpisodiosTreino episódios, ~14 min)..." -ForegroundColor Cyan
    & .\.venv\Scripts\python.exe treinar_v11.py --episodios $EpisodiosTreino --seeds 1 --max_steps_per_ep 1095
    if ($LASTEXITCODE -ne 0) { throw "Falha no treino" }
} else {
    Write-Host ""
    Write-Host "2/4 Pulando treino (usando modelo atual results/v11/dqn_v11.pt)" -ForegroundColor Yellow
    Write-Host "    Para re-treinar: .\atualizar_calendario.ps1 -Treinar" -ForegroundColor DarkGray
}

# Passo 3: gerar calendário
Write-Host ""
Write-Host "3/4 Gerando calendário de $Horizonte dias..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe gerar_calendario_v3.py --horizonte $Horizonte
if ($LASTEXITCODE -ne 0) { throw "Falha na geração do calendário" }

# Mover o output gerado para nome _anual se for 365 dias
if ($Horizonte -ge 300) {
    if (Test-Path "results\v11\calendario_v3.json") {
        Move-Item -Force "results\v11\calendario_v3.json" "results\v11\calendario_v3_anual.json"
        Move-Item -Force "results\v11\calendario_v3.md" "results\v11\calendario_v3_anual.md"
    }
}

# Passo 4: gerar HTML
Write-Host ""
Write-Host "4/4 Gerando HTML visual..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe gerar_html_calendario.py
if ($LASTEXITCODE -ne 0) { throw "Falha na geração do HTML" }

$Duracao = (Get-Date) - $Inicio

Write-Host ""
Write-Host "===================================================="
Write-Host "  ✓ ATUALIZAÇÃO CONCLUÍDA em $($Duracao.TotalSeconds.ToString('0.0'))s"
Write-Host "===================================================="
Write-Host ""
Write-Host "Abrindo calendário no navegador..."
Start-Process "results\v11\calendario_v3_anual.html"
