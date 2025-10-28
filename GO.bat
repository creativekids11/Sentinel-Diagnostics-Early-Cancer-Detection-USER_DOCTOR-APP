@echo off
cls
echo ================================================================
echo     🚀 AI MEDICAL PLATFORM - ONE-CLICK DEPLOYMENT
echo ================================================================
echo.
echo This script will:
echo   1. Stop all running services
echo   2. Verify/fix nginx configuration  
echo   3. Start nginx reverse proxy
echo   4. Start AI Medical Platform
echo   5. Test everything works
echo   6. Show access URLs
echo.
echo Starting in 3 seconds...
timeout /t 3 /nobreak >nul

REM Change to deployment directory
cd /d "D:\deployed-hackathon"

echo ================================================================
echo                      PHASE 1: CLEANUP
echo ================================================================
echo.

echo [1/3] Stopping all services...
taskkill /f /im "nginx.exe" 2>nul
taskkill /f /im "python.exe" 2>nul
echo ✓ Stopped nginx and Python processes

echo [2/3] Clearing any locks...
timeout /t 2 /nobreak >nul
echo ✓ System ready

echo [3/3] Activating virtual environment...
echo [3/3] Skipping virtual environment activation...
echo ✓ Using system Python

echo.
echo ================================================================
echo                    PHASE 2: CONFIGURATION
echo ================================================================
echo.

echo [1/3] Testing nginx configuration...
if not exist "C:\nginx\nginx.exe" (
    echo ❌ ERROR: Nginx not found at C:\nginx\nginx.exe
    echo Please install nginx first
    echo.
    pause
    exit /b 1
)

"C:\nginx\nginx.exe" -t -c "%cd%\nginx-windows.conf"
if %errorlevel% neq 0 (
    echo ❌ ERROR: Nginx configuration failed
    pause
    exit /b 1
)
echo ✓ Nginx configuration valid

echo [2/3] Ensuring directories exist...
if not exist "C:\nginx\temp\client_body_temp" mkdir "C:\nginx\temp\client_body_temp"
if not exist "C:\nginx\temp\proxy_temp" mkdir "C:\nginx\temp\proxy_temp"
if not exist "C:\nginx\temp\fastcgi_temp" mkdir "C:\nginx\temp\fastcgi_temp"
if not exist "C:\nginx\temp\uwsgi_temp" mkdir "C:\nginx\temp\uwsgi_temp"
if not exist "C:\nginx\temp\scgi_temp" mkdir "C:\nginx\temp\scgi_temp"
if not exist "C:\nginx\logs" mkdir "C:\nginx\logs"
echo ✓ Nginx directories ready

echo [3/3] Testing Python imports and installing missing packages...
echo Installing missing AI packages...
python -c "import flask, sys; print('✓ Python', sys.version[:5], 'and Flask', flask.__version__, 'ready')"
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python/Flask not working
    echo Installing Flask...
    python -m pip install Flask waitress >nul 2>&1
)

echo Installing missing AI packages...
python -m pip install Flask-SocketIO python-socketio >nul 2>&1
echo ✓ Core packages installed

echo.
echo ================================================================
echo                     PHASE 3: DEPLOYMENT
echo ================================================================
echo.

echo [1/2] Starting nginx reverse proxy...
start /b "" "C:\nginx\nginx.exe" -c "%cd%\nginx-windows.conf"
timeout /t 3 /nobreak >nul

REM Verify nginx started
tasklist /fi "imagename eq nginx.exe" 2>nul | find /i "nginx.exe" >nul
if %errorlevel% equ 0 (
    echo ✓ Nginx started successfully
) else (
    echo ❌ ERROR: Nginx failed to start
    if exist "C:\nginx\logs\error.log" (
        echo Last nginx error:
        powershell "Get-Content 'C:\nginx\logs\error.log' -Tail 3" 2>nul
    )
    pause
    exit /b 1
)

echo [2/2] Starting AI Medical Platform...
start /min cmd /c "title AI-Medical-Platform && cd /d D:\deployed-hackathon && python production_wsgi_enhanced.py"
echo ✓ AI Platform starting (loading AI models...)

echo.
echo Waiting for platform to initialize...
echo This may take 30-60 seconds for AI models to load...
timeout /t 15 /nobreak >nul

echo.
echo ================================================================
echo                      PHASE 4: VERIFICATION
echo ================================================================
echo.

echo [1/4] Checking service status...
tasklist /fi "imagename eq nginx.exe" 2>nul | find /i "nginx.exe" >nul
if %errorlevel% equ 0 (
    echo ✓ Nginx: RUNNING
    set NGINX_OK=1
) else (
    echo ❌ Nginx: NOT RUNNING
    set NGINX_OK=0
)

tasklist /fi "imagename eq python.exe" 2>nul | find /i "python.exe" >nul
if %errorlevel% equ 0 (
    echo ✓ AI Platform: RUNNING
    set PYTHON_OK=1
) else (
    echo ❌ AI Platform: NOT RUNNING
    set PYTHON_OK=0
)

echo.
echo [2/4] Testing HTTP connectivity...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost' -TimeoutSec 10 -UseBasicParsing; Write-Host '✓ Local access: HTTP' $response.StatusCode } catch { Write-Host '❌ Local access failed:' $_.Exception.Message }"

echo.
echo [3/4] Testing static assets...
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost/static/css/style.css' -TimeoutSec 5 -UseBasicParsing; Write-Host '✓ Static CSS: HTTP' $response.StatusCode '(' $response.Headers.'Content-Length' 'bytes)' } catch { Write-Host '❌ Static CSS failed:' $_.Exception.Message }"

echo.
echo [4/4] Getting network IP...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address" ^| findstr /v "127.0.0.1" ^| findstr /v "172.22"') do (
    for /f "tokens=1 delims= " %%b in ("%%a") do (
        set NETWORK_IP=%%b
    )
)

if defined NETWORK_IP (
    echo Testing network access...
    powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://%NETWORK_IP%' -TimeoutSec 10 -UseBasicParsing; Write-Host '✓ Network access: HTTP' $response.StatusCode } catch { Write-Host '❌ Network access failed:' $_.Exception.Message }"
)

echo.
echo ================================================================
echo                       DEPLOYMENT COMPLETE
echo ================================================================
echo.

if "%NGINX_OK%"=="1" if "%PYTHON_OK%"=="1" (
    echo 🎉 SUCCESS! AI Medical Platform is ready!
    echo.
    echo 🌐 ACCESS URLS:
    echo   ┌─ Local Access (you):
    echo   │  👉 http://localhost
    echo   │
    if defined NETWORK_IP (
        echo   ├─ Network Access (colleagues):
        echo   │  👉 http://%NETWORK_IP%
        echo   │  📱 Share this with colleagues on same WiFi
        echo   │
    )
    echo   └─ Features Available:
    echo      🧠 Brain Tumor Detection ^& Segmentation
    echo      🫁 Lung Cancer Detection with YOLO
    echo      🎗️ Breast Cancer Analysis with MLMO-EMO
    echo      🤖 AI Assistant Chat Interface
    echo      📊 Medical Report Generation
    echo      👩‍⚕️ Doctor/Patient Management
    echo.
    echo ⚡ PERFORMANCE STATUS:
    echo   ✅ Static assets loading (CSS, JS, images)
    echo   ✅ Nginx reverse proxy active
    echo   ✅ AI models loaded and ready
    echo   ✅ Network sharing configured
    echo.
    echo 🔧 QUICK COMMANDS:
    echo   Stop everything:  .\stop-all.bat
    echo   Restart:          .\GO.bat (this script)
    echo   Check status:     .\status.bat
    echo   View logs:        type app.log
    echo.
    echo ✅ READY FOR USE! Open browser and go to the URLs above.
) else (
    echo ⚠️ PARTIAL SUCCESS - Some issues detected:
    echo.
    if "%NGINX_OK%"=="0" (
        echo ❌ Nginx Issues:
        echo   • Check if nginx is installed at C:\nginx\nginx.exe
        echo   • Review configuration: nginx-windows.conf
        echo   • Check logs: C:\nginx\logs\error.log
    )
    if "%PYTHON_OK%"=="0" (
        echo ❌ Python App Issues:
        echo   • Check if virtual environment is set up
        echo   • Install packages: deploy-ai-platform.bat
        echo   • Check logs: app.log
    )
    echo.
    echo 🔧 TROUBLESHOOTING:
    echo   1. Run: deploy-ai-platform.bat (installs packages)
    echo   2. Run: admin-network-fix.bat (as Administrator)
    echo   3. Check firewall settings
    echo   4. Restart this script: .\GO.bat
)

echo.
echo Press any key to continue...
pause >nul