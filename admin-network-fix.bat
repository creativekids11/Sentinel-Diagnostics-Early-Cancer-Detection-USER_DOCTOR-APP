@echo off
REM ================================================
REM     ADMIN NETWORK FIX - Run as Administrator
REM ================================================

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script must be run as Administrator
    echo Right-click on this file and select "Run as administrator"
    pause
    exit /b 1
)

echo ================================================
echo    Fixing Network Sharing (Administrator Mode)
echo ================================================
echo.

echo [1/4] Adding comprehensive firewall rules...
REM Remove any existing rules
netsh advfirewall firewall delete rule name="AI Medical Platform" >nul 2>&1
netsh advfirewall firewall delete rule name="AI Medical Platform HTTP" >nul 2>&1
netsh advfirewall firewall delete rule name="AI Medical Platform Flask" >nul 2>&1

REM Add new comprehensive rules
echo Adding HTTP (port 80) inbound rule...
netsh advfirewall firewall add rule name="AI Medical Platform HTTP" dir=in action=allow protocol=TCP localport=80 profile=any enable=yes

echo Adding Flask (port 5000) inbound rule...
netsh advfirewall firewall add rule name="AI Medical Platform Flask" dir=in action=allow protocol=TCP localport=5000 profile=any enable=yes

echo Adding HTTP outbound rule...
netsh advfirewall firewall add rule name="AI Medical Platform HTTP Out" dir=out action=allow protocol=TCP localport=80 profile=any enable=yes

echo [2/4] Enabling network discovery and file sharing...
netsh advfirewall firewall set rule group="Network Discovery" new enable=Yes
netsh advfirewall firewall set rule group="File and Printer Sharing" new enable=Yes
netsh advfirewall firewall set rule group="Core Networking" new enable=Yes

echo [3/4] Setting network profile to Private...
powershell -Command "Get-NetConnectionProfile | Where-Object {$_.NetworkCategory -eq 'Public'} | Set-NetConnectionProfile -NetworkCategory Private"

echo [4/4] Configuring additional network settings...
REM Enable ICMP (ping) - helps with network diagnostics
netsh advfirewall firewall add rule name="AI Medical Platform ICMP" dir=in action=allow protocol=icmpv4:8,any enable=yes

echo.
echo ================================================
echo          ADMIN CONFIGURATION COMPLETE
echo ================================================
echo.
echo ✅ Changes Applied Successfully:
echo   • Firewall rules added for ports 80 and 5000
echo   • Network discovery enabled
echo   • File and printer sharing enabled  
echo   • Network profile set to Private
echo   • ICMP (ping) enabled
echo.
echo 🌐 Correct URLs to share:
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address" ^| findstr /v "127.0.0.1"') do (
    for /f "tokens=1 delims= " %%b in ("%%a") do (
        echo   http://%%b
    )
)
echo.
echo ✅ Network sharing should now work properly!
echo.
pause