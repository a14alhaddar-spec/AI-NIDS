@echo off
REM Start CNN-LSTM Dashboard
title CNN-LSTM Dashboard
echo ============================================================
echo CNN-LSTM NIDS Dashboard
echo ============================================================
echo.
echo Starting on http://localhost:8082
echo.

cd /d "%~dp0.."
.venv\Scripts\python.exe dashboards\dashboard_cnn_lstm.py

pause
