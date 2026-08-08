@echo off
TITLE HAYAI v2 - Angular Frontend
color 0e

echo Avvio del frontend Angular di HAYAI v2...
cd /d %~dp0\hayai-new\web

echo Frontend in ascolto su http://localhost:4200 (Assicurati che il backend sia avviato con avvia_backend.bat)
call npm start

pause
