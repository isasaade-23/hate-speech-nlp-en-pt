@echo off
rem Traduz os 500 ids do piloto com o Gemini, retomando de onde parou. Ver scripts/translate_gemini.py.
rem Chaves: variavel GEMINI_KEYS_FILE (arquivo fora do repositorio). Nada de chave aqui.
rem Log em logs\traducao_gemini_<data>.log; Ctrl+C para parar, rode de novo para retomar.
cd /d "%~dp0"
if "%GEMINI_KEYS_FILE%%GEMINI_API_KEY%"=="" (
  echo Defina GEMINI_KEYS_FILE ou GEMINI_API_KEY antes de rodar.
  pause
  exit /b 1
)
powershell -NoProfile -Command "$log = 'logs\traducao_gemini_' + (Get-Date -Format 'yyyy-MM-dd') + '.log'; $env:PYTHONIOENCODING = 'utf-8'; & '.venv\Scripts\python.exe' -u scripts\translate_gemini.py traduzir --completo --modelo gemini-3-flash-preview --rotacionar 2>&1 | Tee-Object -FilePath $log -Append"
pause
