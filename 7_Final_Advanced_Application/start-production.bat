@echo off
REM Fossil AI Hub - Windows Deployment Script with Access Information

echo 🚀 Starting Fossil AI Hub Production Deployment...
echo ==================================================

REM Start services in detached mode
docker compose -f docker-compose.production.yml up -d

REM Wait a moment for services to initialize
timeout /t 3 /nobreak > nul

REM Get the local IP address (Windows method)
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    for /f "tokens=1" %%b in ("%%a") do set LOCAL_IP=%%b
)
if "%LOCAL_IP%"=="" set LOCAL_IP=localhost

echo.
echo ✅ Services started successfully!
echo ==================================
echo.
echo 📱 Fossil AI Hub Access URLs:
echo    Frontend (Web UI):
echo      • Local:    http://localhost:8080
echo      • Network:  http://%LOCAL_IP%:8080
echo.
echo 🔧 Backend API:
echo      • Local:    http://localhost:5000
echo      • Network:  http://%LOCAL_IP%:5000
echo.
echo 📊 Health Check:
echo      • http://localhost:5000/api/health
echo.
echo 📋 Management Commands:
echo      • View logs:  docker compose -f docker-compose.production.yml logs -f
echo      • Stop:       docker compose -f docker-compose.production.yml down
echo      • Update:     docker compose -f docker-compose.production.yml pull ^&^& docker compose -f docker-compose.production.yml up -d
echo.
echo ⏳ Waiting for services to be ready...

REM Wait for backend health check
echo    Backend: Checking...
:backend_check
curl -sf http://localhost:5000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo    Backend: ✅ Ready!
    goto frontend_check
)
timeout /t 2 /nobreak > nul
goto backend_check

:frontend_check
echo    Frontend: Checking...
curl -sf http://localhost:8080 >nul 2>&1
if %errorlevel% equ 0 (
    echo    Frontend: ✅ Ready!
    goto done
)
timeout /t 1 /nobreak > nul
goto frontend_check

:done
echo.
echo 🎉 Fossil AI Hub is now running!
echo    Open your browser and go to: http://%LOCAL_IP%:8080
echo ==================================

pause