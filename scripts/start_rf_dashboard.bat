@echo off
REM Start Random Forest Dashboard
title Random Forest Dashboard
echo ============================================================
echo Random Forest NIDS Dashboard
echo ============================================================
echo.
echo Starting on http://localhost:8081
echo.

cd /d "%~dp0.."
.venv\Scripts\python.exe dashboards\dashboard_rf.py

pause
