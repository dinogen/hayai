@echo off
TITLE HAYAI v2 - Batch Pipeline Runner
color 0b

echo ========================================================
echo   HAYAI v2 - Esecuzione Pipeline Batch Giornaliera
echo ========================================================
cd /d %~dp0

call venv\Scripts\activate
set PYTHONPATH=hayai-new

echo [1/7] Aggiornamento prezzi di mercato (yfinance)...
python -m app.cli data --portfolio main
if errorlevel 1 goto error

echo [2/7] Download notizie finanziarie...
python -m app.cli news --portfolio main
if errorlevel 1 goto error

echo [3/7] Analisi sentiment notizie con DeepSeek AI...
python -m app.cli sentiment --portfolio main
if errorlevel 1 goto error

echo [4/7] Inferenza modello Quant (ONNX)...
python -m app.cli predict --portfolio main
if errorlevel 1 goto error

echo [5/7] Calcolo segnale ibrido (Quant + LLM)...
python -m app.cli signal --portfolio main
if errorlevel 1 goto error

echo [6/7] Ottimizzazione portafoglio long/short (5000 EUR)...
python -m app.cli recommend --portfolio main
if errorlevel 1 goto error

echo [7/7] Generazione riassunto Markdown...
python -m app.cli summaries --portfolio main
if errorlevel 1 goto error

echo.
echo ========================================================
echo   Pipeline completata con successo!
echo ========================================================
goto end

:error
echo.
echo [ERRORE] Si e' verificato un errore durante l'esecuzione della pipeline.

:end
pause
