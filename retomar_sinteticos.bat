@echo off
rem Retoma o lote sintetico do checkpoint. Ver retomar_sinteticos.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0retomar_sinteticos.ps1"
pause
