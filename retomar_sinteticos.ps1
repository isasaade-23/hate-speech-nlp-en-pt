# Retoma a geracao do lote sintetico a partir do checkpoint (data/processed/synthetic_batch.jsonl).
# Chamado por retomar_sinteticos.bat. Abre o Ollama se nao estiver aberto e inicia o gerador
# como processo independente do Windows (sobrevive ao fechamento do terminal e da sessao).
$ErrorActionPreference = "Stop"
$h = $PSScriptRoot
$meta = 1500
$jsonl = Join-Path $h "data\processed\synthetic_batch.jsonl"
New-Item -ItemType Directory -Force (Join-Path $h "logs") | Out-Null

function Contar { if (Test-Path $jsonl) { @(Get-Content $jsonl).Count } else { 0 } }

if ((Contar) -ge $meta) { Write-Host "Lote completo: $(Contar)/$meta. Nada a fazer."; exit 0 }

$gerador = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'generate_synthetic.py' }
if ($gerador) { Write-Host "O gerador ja esta rodando ($(Contar)/$meta). Nada a fazer."; exit 0 }

if (-not (Get-Process ollama -ErrorAction SilentlyContinue)) {
    $ol = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    $cmd = "cmd.exe /c `"set OLLAMA_MODELS=C:\ollama_models&& `"$ol`" serve > `"$h\logs\ollama.log`" 2>&1`""
    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $h } | Out-Null
    Write-Host "Ollama iniciado."
}
$ok = $false
foreach ($i in 1..30) {
    try { Invoke-WebRequest "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 2 | Out-Null; $ok = $true; break } catch { Start-Sleep 2 }
}
if (-not $ok) { Write-Host "O Ollama nao respondeu em 60 s. Veja logs\ollama.log."; exit 1 }

$cmd = "cmd.exe /c `"set PYTHONUTF8=1&& .venv\Scripts\python.exe -u scripts\generate_synthetic.py gen --n 0 --n-odio 900 --n-nao 600 --offset 2 --exclude data/processed/synthetic_pilot.jsonl data/processed/synthetic_pilot2.jsonl --out data/processed/synthetic_batch.jsonl >> logs\synthetic_batch.log 2>&1`""
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $h } | Out-Null
Write-Host "Gerador retomado de $(Contar)/$meta. Progresso: Get-Content logs\synthetic_batch.log -Tail 3"
