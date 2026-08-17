@echo off
setlocal
TITLE "HAYAI v2 - Ingestion, Training, Verify & Backtest (100 Assets, 5 Years)"
color 0b

echo ========================================================
echo  HAYAI v2: Download dati (5y), Training, Verify e Backtest
echo ========================================================
cd /d %~dp0

REM Attiva l'ambiente virtuale e imposta il PYTHONPATH su hayai-new
call venv\Scripts\activate
set PYTHONPATH=hayai-new

echo.
echo === [1/3] Ingestion e Training v2 ===
python -m app.jobs.train_universe_pipeline
if errorlevel 1 goto :err

echo.
echo === [2/3] Verifica modello v2 ===
python -m app.cli verify --version v2
if errorlevel 1 goto :err

echo.
echo === [3/3] Backtest selezione v2 ===
python -m app.cli backtest --version v2
if errorlevel 1 goto :err

echo.
echo Pipeline completata con successo.
pause
exit /b 0

:err
echo.
echo ERRORE: Pipeline fallita.
pause
exit /b 1
