@echo off
TITLE "HAYAI v2 - Ingestion & Training Pipeline (100 Assets, 5 Years)"
color 0b

echo ========================================================
echo  HAYAI v2: Download dati storici (5 anni) e Training MLP
echo ========================================================
cd /d %~dp0

REM Attiva l'ambiente virtuale e imposta il PYTHONPATH su hayai-new
call venv\Scripts\activate
set PYTHONPATH=hayai-new

echo Esecuzione dello script di ingestion e training...
python -m app.jobs.train_universe_pipeline

echo Pipeline completata.
pause
