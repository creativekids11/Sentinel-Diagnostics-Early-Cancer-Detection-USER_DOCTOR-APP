@echo off
REM Sage AI Assistant - Quick Start Script for Windows
REM This script helps you set up and test Sage AI Assistant

echo 🤖 SAGE AI ASSISTANT - QUICK START
echo =================================

REM Check if we're in the right directory
if not exist "AI_Assistant\functions.py" (
    echo ❌ Please run this script from the WEBPAGE directory
    echo    cd WEBPAGE && AI_Assistant\quickstart.bat
    pause
    exit /b 1
)

echo 📂 Working directory: %cd%

REM Check Python version
echo 🐍 Checking Python...
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python --version') do echo ✅ Python found: %%i
) else (
    echo ❌ Python not found. Please install Python 3.7+
    pause
    exit /b 1
)

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import google.generativeai" >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ google-generativeai installed
) else (
    echo ⚠️ Installing missing dependencies...
    pip install -r AI_Assistant\requirements.txt
)

REM Check for API key
echo 🔑 Checking API key configuration...
if exist "AI_Assistant\.env" (
    findstr /c:"your_actual_gemini_api_key_here" AI_Assistant\.env >nul
    if %errorlevel% equ 0 (
        echo ⚠️ Please set your actual Gemini API key in AI_Assistant\.env
        echo    Get your key from: https://makersuite.google.com/app/apikey
        echo    Then edit AI_Assistant\.env and replace 'your_actual_gemini_api_key_here'
        echo.
        pause
    ) else (
        echo ✅ API key configuration found
    )
) else (
    echo ❌ .env file not found in AI_Assistant\
    pause
    exit /b 1
)

REM Test the AI Assistant
echo.
echo 🧪 Testing Sage AI Assistant...
cd AI_Assistant
python demo.py --test >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Sage AI Assistant is working!
) else (
    echo ⚠️ Testing with demo script...
    python demo.py
)

cd ..

REM Final instructions
echo.
echo 🎉 SETUP COMPLETE!
echo ==================
echo.
echo Next steps:
echo 1. Start your Flask app: python app.py
echo 2. Open your browser to: http://localhost:5601
echo 3. Login as a patient or doctor
echo 4. Look for the 🧠 Sage AI Assistant button
echo 5. Click it and start chatting!
echo.
echo 💡 Tips:
echo - Sage works on both patient and doctor pages
echo - Try asking: 'How do I book an appointment?'
echo - Use the suggestion buttons for quick help
echo - Sage remembers your conversation context
echo.
echo 📚 Documentation:
echo - README: AI_Assistant\README.md
echo - Integration Guide: AI_Assistant\INTEGRATION_GUIDE.md
echo - Implementation Details: AI_Assistant\IMPLEMENTATION_SUMMARY.md
echo.
echo 🆘 Need help? Run: python AI_Assistant\demo.py

pause