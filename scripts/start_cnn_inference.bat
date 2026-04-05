@echo off
REM Start CNN-LSTM Inference Service
title CNN-LSTM Inference Service
echo ============================================================
echo CNN-LSTM Inference Service
echo ============================================================
echo.
echo Running on http://localhost:5003
echo.

cd /d "%~dp0.."
.venv\Scripts\python.exe services\inference_cnn_lstm\app.py

pause
