@echo off
REM Federated Learning Dashboard - Quick Start Script for Windows

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Federated Learning Dashboard Setup
echo ==========================================
echo.

REM Check if Node.js is installed
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js v14 or higher.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('node -v') do set NODE_VERSION=%%i
for /f "tokens=*" %%i in ('npm -v') do set NPM_VERSION=%%i

echo ✓ Node.js %NODE_VERSION% detected
echo ✓ npm %NPM_VERSION% detected
echo.

REM Install dashboard dependencies
echo 📦 Installing dashboard dependencies...
cd dashboard
call npm install

if %errorlevel% neq 0 (
    echo ❌ Failed to install dashboard dependencies
    pause
    exit /b 1
)

echo ✓ Dashboard dependencies installed
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo 📋 Next Steps:
echo.
echo 1. Install Python dependencies:
echo    pip install flask flask-cors
echo.
echo 2. Integrate the API files:
echo    - Copy server/server_api.py to your server
echo    - Copy client/client_api.py to your client
echo.
echo 3. Update server_config.json to set num_clients
echo.
echo 4. Start the servers in separate terminals:
echo    Terminal 1: python server\server.py
echo    Terminal 2: python client\train_client.py
echo.
echo 5. Start the dashboard:
echo    cd dashboard ^&^& npm start
echo.
echo 6. Open browser to http://localhost:3000
echo.
echo 📚 For detailed setup instructions, see: DASHBOARD_SETUP.md
echo.
pause
