#!/bin/bash

# Sage AI Assistant - Quick Start Script
# This script helps you set up and test Sage AI Assistant

echo "🤖 SAGE AI ASSISTANT - QUICK START"
echo "================================="

# Check if we're in the right directory
if [ ! -f "AI_Assistant/functions.py" ]; then
    echo "❌ Please run this script from the WEBPAGE directory"
    echo "   cd WEBPAGE && bash AI_Assistant/quickstart.sh"
    exit 1
fi

echo "📂 Working directory: $(pwd)"

# Check Python version
echo "🐍 Checking Python..."
python_version=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ Python found: $python_version"
else
    echo "❌ Python not found. Please install Python 3.7+"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
if python3 -c "import google.generativeai" 2>/dev/null; then
    echo "✅ google-generativeai installed"
else
    echo "⚠️  Installing missing dependencies..."
    pip3 install -r AI_Assistant/requirements.txt || pip install -r AI_Assistant/requirements.txt
fi

# Check for API key
echo "🔑 Checking API key configuration..."
if [ -f "AI_Assistant/.env" ]; then
    if grep -q "your_actual_gemini_api_key_here" AI_Assistant/.env; then
        echo "⚠️  Please set your actual Gemini API key in AI_Assistant/.env"
        echo "   Get your key from: https://makersuite.google.com/app/apikey"
        echo "   Then edit AI_Assistant/.env and replace 'your_actual_gemini_api_key_here'"
        echo ""
        read -p "Press Enter after setting your API key, or Ctrl+C to exit..."
    else
        echo "✅ API key configuration found"
    fi
else
    echo "❌ .env file not found in AI_Assistant/"
    exit 1
fi

# Test the AI Assistant
echo ""
echo "🧪 Testing Sage AI Assistant..."
cd AI_Assistant
if python3 demo.py --test 2>/dev/null || python demo.py --test 2>/dev/null; then
    echo "✅ Sage AI Assistant is working!"
else
    echo "⚠️  Testing with demo script..."
    python3 demo.py 2>/dev/null || python demo.py
fi

cd ..

# Final instructions
echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Start your Flask app: python app.py"
echo "2. Open your browser to: http://localhost:5601" 
echo "3. Login as a patient or doctor"
echo "4. Look for the 🧠 Sage AI Assistant button"
echo "5. Click it and start chatting!"
echo ""
echo "💡 Tips:"
echo "- Sage works on both patient and doctor pages"
echo "- Try asking: 'How do I book an appointment?'"
echo "- Use the suggestion buttons for quick help"
echo "- Sage remembers your conversation context"
echo ""
echo "📚 Documentation:"
echo "- README: AI_Assistant/README.md"
echo "- Integration Guide: AI_Assistant/INTEGRATION_GUIDE.md"
echo "- Implementation Details: AI_Assistant/IMPLEMENTATION_SUMMARY.md"
echo ""
echo "🆘 Need help? Run: python AI_Assistant/demo.py"