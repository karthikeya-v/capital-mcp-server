@echo off
echo.
echo Capital.com MCP Server Setup
echo ================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.10 or higher.
    pause
    exit /b 1
)

echo Python detected
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created
echo.

REM Activate virtual environment and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo Failed to install dependencies
    pause
    exit /b 1
)

echo Dependencies installed
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo .env file created - PLEASE EDIT IT WITH YOUR API CREDENTIALS
    echo.
)

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your Capital.com API credentials
echo 2. Test the server: python server.py
echo 3. Add to Claude Desktop config (see README.md)
echo 4. Restart Claude Desktop
echo.
echo IMPORTANT: Start with demo account (CAPITAL_USE_DEMO=true)
echo.
pause
