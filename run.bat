@echo off
rem SFEDUCASTLING - install and run (Windows).
rem Usage:  run.bat
setlocal
cd /d "%~dp0"

echo === SFEDUCASTLING - install and run ===

rem 1. Virtual environment
if not exist .venv (
    echo Creating virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 python -m venv .venv
)
call .venv\Scripts\activate.bat

rem 2. Dependencies (install only if something is missing)
python -c "import fastapi, alembic, sqlalchemy, aiosqlite, dotenv" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)

rem 3. Configuration
if not exist .env (
    copy .env.example .env >nul
    echo Created .env from .env.example
)

rem 4. Database (migrations)
echo Updating database...
python -m scripts.init_db
if errorlevel 1 (
    echo ERROR: failed to update the database.
    pause
    exit /b 1
)

rem 5. Server: http://127.0.0.1:8005
rem Run via -m so that the project root lands in sys.path and the
rem `backend` package resolves (python backend\app.py would not).
echo Starting server: http://127.0.0.1:8005
python -m backend.app

pause
