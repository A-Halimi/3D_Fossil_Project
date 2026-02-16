@echo off
setlocal enabledelayedexpansion

REM ====== settings ======
REM Change this if your compose filename is different
set COMPOSE_FILE=docker-compose.production.yml

echo Starting Fossil AI Hub deployment in detached mode...
echo.

if not exist "%COMPOSE_FILE%" (
  echo Compose file "%COMPOSE_FILE%" not found.
  echo Edit this script and set COMPOSE_FILE to your compose filename.
  goto :end
)

echo Shutting down any existing stack from %COMPOSE_FILE%...
docker compose -f "%COMPOSE_FILE%" down --remove-orphans

echo.
echo Pulling latest images from Docker Hub...
docker compose -f "%COMPOSE_FILE%" pull

echo.
echo Starting containers in background...
docker compose -f "%COMPOSE_FILE%" up -d

echo.
echo Deployment started. Access the application at:
echo   Frontend: http://localhost:8080
echo   Backend API: http://localhost:5000
echo   Health: http://localhost:5000/api/health
echo.
echo Useful commands:
echo   docker compose -f "%COMPOSE_FILE%" ps
echo   docker compose -f "%COMPOSE_FILE%" logs -f backend
echo   docker compose -f "%COMPOSE_FILE%" logs -f frontend
echo.

echo Press any key to exit...
pause >nul

:end
endlocal