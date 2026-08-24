# Gera o pacote de codigo para o INPI e-Software e imprime o SHA-512.
#
# Metodo: git archive de HEAD dos tres repositorios (so codigo versionado, sem
# venv, dados ou binarios nao versionados) mais um LEIA-ME com autora, data,
# commits e licenca. O INPI guarda so o hash, entao o ZIP gerado tem que ser
# preservado imutavel: numa disputa e preciso apresentar o arquivo que gera
# aquele hash.
#
# Uso:  .\gerar_pacote.ps1 -Versao v2
param([string]$Versao = "v2")
$ErrorActionPreference = "Stop"

$repos = [ordered]@{
  "hate-speech-project" = "estudo e pipeline"
  "hate-speech-space"   = "demo Streamlit"
  "luciola-extension"   = "extensao de navegador"
}
$destDir = "C:\Users\Renato\luciola_inpi"
$stage = Join-Path $env:TEMP "inpi_stage_$Versao"

if (Test-Path $stage) { [IO.Directory]::Delete($stage, $true) }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
New-Item -ItemType Directory -Path $destDir -Force | Out-Null

$linhas = @()
foreach ($r in $repos.Keys) {
  $src = "C:\Users\Renato\$r"
  $sha = (git -C $src rev-parse --short HEAD)
  $tmpZip = Join-Path $env:TEMP "$r.zip"
  git -C $src archive --format=zip -o $tmpZip HEAD
  Expand-Archive -Path $tmpZip -DestinationPath (Join-Path $stage $r) -Force
  [IO.File]::Delete($tmpZip)
  $linhas += "  {0,-20} ({1,-22}) commit {2}" -f $r, $repos[$r], $sha
}

$leiame = @"
Luciola - pacote de codigo para Registro de Programa de Computador (INPI e-Software)
Autora: Isabela Venancio da Silva
Data do pacote: $(Get-Date -Format 'yyyy-MM-dd')
Conteudo: codigo-fonte versionado (git archive de HEAD) dos tres repositorios:
$($linhas -join "`n")
Linguagens: Python, JavaScript.
Licenca do codigo: GNU Affero General Public License v3.0 (AGPL-3.0).
Os artefatos de modelo treinado NAO estao sob a AGPL: uso em pesquisa e educacao
apenas, conforme LICENSE-MODEL.md, porque as licencas dos dados de treino nao
permitem conceder uso comercial adiante.
"@
Set-Content -Path (Join-Path $stage "LEIA-ME.txt") -Value $leiame -Encoding utf8

$out = Join-Path $destDir "luciola_codigo_$Versao.zip"
if (Test-Path $out) { [IO.File]::Delete($out) }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $out
[IO.Directory]::Delete($stage, $true)

$h = (Get-FileHash $out -Algorithm SHA512).Hash
"pacote: $out"
"tamanho: $([math]::Round((Get-Item $out).Length / 1MB, 2)) MB"
"SHA-512: $h"
"`nGuardar copia imutavel tambem no Drive. Nao re-compactar: muda o hash."
