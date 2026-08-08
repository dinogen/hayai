@echo off
TITLE "HAYAI v2 - Ciclo Notturno Completo"
color 0b

echo ========================================================
echo  HAYAI v2: Ciclo Notturno Completo
echo  data - news - sentiment - predict - signal - recommend
echo ========================================================
cd /d %~dp0

REM Attiva l'ambiente virtuale e imposta il PYTHONPATH su hayai-new
call venv\Scripts\activate
set PYTHONPATH=hayai-new

echo [1/7] Job data - aggiornamento prezzi giornalieri (yfinance)...
python -m app.cli data
if errorlevel 1 goto :error

echo [2/7] Job news - scarico notizie per gli strumenti monitorati...
python -m app.cli news
if errorlevel 1 goto :error

echo [3/7] Job sentiment - analisi notizie con DeepSeek...
python -m app.cli sentiment
if errorlevel 1 goto :error

echo [4/7] Job predict - quant_score dal modello ONNX...
python -m app.cli predict
if errorlevel 1 goto :error

echo [5/7] Job signal - segnale ibrido (Quant + LLM sentiment)...
python -m app.cli signal
if errorlevel 1 goto :error

echo [6/7] Job recommend - composizione long/short su 5.000 EUR...
python -m app.cli recommend
if errorlevel 1 goto :error

echo [7/7] Job summaries - riepilogo markdown notizie e sentiment...
python -m app.cli summaries
if errorlevel 1 goto :error

echo.
echo Ciclo completo terminato. Apri la webapp su http://localhost:4200
goto :eof

:error
echo.
echo ERRORE: uno dei job non e' andato a buon fine.
pause
