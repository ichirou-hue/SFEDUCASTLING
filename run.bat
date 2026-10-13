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
rem Maia3 is a git submodule: init before pip install, otherwise
rem the -e ./maia3 line in requirements.txt fails on an empty dir.
if exist .git if not exist maia3\pyproject.toml (
    echo Initializing maia3 submodule...
    git submodule update --init maia3 >nul 2>&1
)
python -c "import fastapi, alembic, sqlalchemy, aiosqlite, dotenv" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
)
rem Maia3 is installed from the submodule: add it separately when the
rem main deps are already present but the engine is missing from venv.
if exist maia3\pyproject.toml (
    python -c "import maia3.uci" >nul 2>&1
    if errorlevel 1 (
        echo Installing maia3 engine from submodule...
        python -m pip install -e ./maia3
    )
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
