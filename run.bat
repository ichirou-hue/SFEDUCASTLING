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
    >.env echo DATABASE_URL=sqlite+aiosqlite:///./backend.db
    echo Created .env with the local SQLite database
)

rem 4. Avoid starting a duplicate backend on the same port.
python -m scripts.check_running_server
if not errorlevel 1 (
    echo SFEDUCASTLING is already running at http://127.0.0.1:8005
    exit /b 0
)
if errorlevel 2 (
    echo ERROR: Port 8005 is occupied, but the process is not SFEDUCASTLING.
    pause
    exit /b 1
)

rem 5. Start the local PostgreSQL 18 cluster only for a local PostgreSQL URL.
python -c "import os; from dotenv import dotenv_values; from sqlalchemy.engine import make_url; u=make_url(os.getenv('DATABASE_URL') or dotenv_values('.env').get('DATABASE_URL') or 'sqlite+aiosqlite:///'); raise SystemExit(0 if u.get_backend_name() == 'postgresql' and u.host in ('127.0.0.1', 'localhost', '::1') else 1)"
if not errorlevel 1 (
    if not exist "%ProgramFiles%\PostgreSQL\18\bin\pg_ctl.exe" (
        echo ERROR: Local PostgreSQL 18 was selected but pg_ctl.exe was not found.
        pause
        exit /b 1
    )
    "%ProgramFiles%\PostgreSQL\18\bin\pg_ctl.exe" status -D "%ProgramFiles%\PostgreSQL\18\data" >nul 2>&1
    if errorlevel 1 (
        echo Starting local PostgreSQL 18...
        "%ProgramFiles%\PostgreSQL\18\bin\pg_ctl.exe" start -D "%ProgramFiles%\PostgreSQL\18\data" -w -t 30
        if errorlevel 1 (
            echo ERROR: Could not start the local PostgreSQL 18 cluster.
            pause
            exit /b 1
        )
    )
)

rem 6. Database (migrations)
echo Updating database...
python -m scripts.init_db
if errorlevel 1 (
    echo ERROR: failed to update the database.
    pause
    exit /b 1
)

rem 7. Server: http://127.0.0.1:8005
rem Run via -m so that the project root lands in sys.path and the
rem `backend` package resolves (python backend\app.py would not).
echo Starting server: http://127.0.0.1:8005
python -m backend.app

pause
