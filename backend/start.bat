@echo off
REM ResearchLens Backend — Windows startup script
REM Run from the project root: backend\start.bat

echo === ResearchLens Backend Setup ===
echo.

REM Navigate to backend directory
cd /d "%~dp0"

REM Create virtual environment if it doesn't exist
IF NOT EXIST "venv\" (
    echo Creating Python virtual environment...
    python -m venv venv
    echo Virtual environment created.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install / upgrade dependencies
echo Installing dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo Dependencies installed.

REM Copy .env if missing
IF NOT EXIST ".env" (
    echo Copying .env.example to .env — please fill in your API keys!
    copy .env.example .env
)

echo.
echo === Starting ResearchLens Backend on http://localhost:8000 ===
echo.

REM Start the server from the project root so Python package imports work
cd ..
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
