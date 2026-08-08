@echo off
TITLE "HAYAI v2 - Predict, Signal & Recommend"
color 0d

echo ========================================================
echo  HAYAI v2: Predict (ONNX) - Signal (Hybrid) - Recommend
echo ========================================================
cd /d %~dp0

REM Attiva l'ambiente virtuale e imposta il PYTHONPATH su hayai-new
call venv\Scripts\activate
set PYTHONPATH=hayai-new

echo [1/3] Job predict - calcolo dei quant_score da modello ONNX...
python -m app.cli predict
if errorlevel 1 goto :error

echo [2/3] Job signal - calcolo del segnale ibrido (Quant + DeepSeek)...
python -m app.cli signal
if errorlevel 1 goto :error

echo [3/3] Job recommend - composizione long/short e allocazione su 5.000 EUR...
python -m app.cli recommend
if errorlevel 1 goto :error

echo.
echo Pipeline completata. I risultati sono salvati nel database:
echo   - model_prediction (quant_score)
echo   - portfolio_signal (segnale ibrido)
echo   - portfolio_recommendation (long/short, importi, quantita)
goto :eof

:error
echo.
echo ERRORE: uno dei job non e' andato a buon fine.

:eof
pause
