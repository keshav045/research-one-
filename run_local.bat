@echo off
title ResearchLens Local
cd /d "%~dp0"
echo Starting ResearchLens locally...
venv\Scripts\python.exe run_local.py
pause
