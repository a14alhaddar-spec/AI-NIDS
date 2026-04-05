@echo off
REM Start Random Forest Inference Service
title RF Inference Service
echo ============================================================
echo Random Forest Inference Service
echo ============================================================
echo.
echo Running on http://localhost:5002
echo.

cd /d "%~dp0.."
.venv\Scripts\python.exe services\inference\app.py

pause
