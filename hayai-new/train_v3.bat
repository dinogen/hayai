@echo off
setlocal
cd /d "%~dp0"

echo === [1/3] Seed universe + download 5y history + train v3 ===
..\venv\Scripts\python.exe -c "from app.jobs.train_universe_pipeline import seed_universe, download_historical_data, build_dataset_and_train; seed_universe(); download_historical_data('5y'); build_dataset_and_train(split='time', version='v3', make_active=True)"
if errorlevel 1 goto :err

echo === [2/3] Verify v3 ===
..\venv\Scripts\python.exe -m app.cli verify --version v3
if errorlevel 1 goto :err

echo === [3/3] Backtest v3 ===
..\venv\Scripts\python.exe -m app.cli backtest --version v3
if errorlevel 1 goto :err

echo.
echo Training v3 completed. Check verify/backtest results above.
exit /b 0

:err
echo.
echo ERROR: step failed, see output above.
exit /b 1
