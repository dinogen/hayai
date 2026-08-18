@echo off
TITLE HAYAI v2 - FastAPI Backend
color 0a

echo Avvio del backend FastAPI di HAYAI v2...
cd /d %~dp0

REM Attiva l'ambiente virtuale e imposta il PYTHONPATH su hayai-new
rem call venv\Scripts\activate
set PYTHONPATH=hayai-new

echo Backend in ascolto su http://127.0.0.1:8000 (Documentazione Swagger: http://127.0.0.1:8000/docs)
venv\Scripts\python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

pause
