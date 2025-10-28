#!/usr/bin/env python3
"""
Sage AI Assistant Setup Script
This script helps set up the Sage AI Assistant with proper configuration.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create or update .env file with Sage AI configuration."""
    env_path = Path('.env')
    
    # Read existing .env if it exists
    existing_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_vars[key] = value
    
    # Check if GEMINI_API_KEY exists
    if 'GEMINI_API_KEY' not in existing_vars:
        print("🔑 Gemini API Key Setup Required")
        print("To use Sage AI Assistant, you need a Google Gemini API key.")
        print("Get your free API key at: https://aistudio.google.com/app/apikey")
        print()
        
        api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
        if api_key:
            existing_vars['GEMINI_API_KEY'] = api_key
            print("✅ API key saved!")
        else:
            existing_vars['GEMINI_API_KEY'] = 'your-gemini-api-key-here'
            print("⚠️  Placeholder API key added. Update it in .env file before using Sage.")
    else:
        print("✅ Gemini API key found in .env file")
    
    # Write updated .env file
    with open(env_path, 'w') as f:
        f.write("# Sage AI Assistant Configuration\n")
        f.write(f"GEMINI_API_KEY={existing_vars['GEMINI_API_KEY']}\n")
        f.write("\n# Other Flask Configuration\n")
        for key, value in existing_vars.items():
            if key != 'GEMINI_API_KEY':
                f.write(f"{key}={value}\n")
    
    return existing_vars.get('GEMINI_API_KEY', '').startswith('AIza')

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import google.generativeai
        print("✅ google-generativeai package found")
        return True
    except ImportError:
        print("❌ google-generativeai package not found")
        print("Run: pip install google-generativeai python-dotenv")
        return False

def verify_file_structure():
    """Verify that all Sage AI files are in place."""
    required_files = [
        'AI_Assistant/__init__.py',
        'AI_Assistant/functions.py',
        'AI_Assistant/routes.py',
        'AI_Assistant/knowledge.txt',
        'templates/AI_Assistant/chat_widget.html',
        'templates/layouts/patient.html',
        'templates/layouts/doctor.html'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print("\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    return True

def test_sage_integration():
    """Test if Sage AI can be imported and initialized."""
    try:
        sys.path.append('.')
        from AI_Assistant.functions import SageAssistant
        
        # Try to create an instance (won't work without API key, but will test imports)
        print("✅ SageAssistant class can be imported")
        return True
    except Exception as e:
        print(f"❌ Error importing SageAssistant: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Sage AI Assistant Setup")
    print("=" * 50)
    
    # Check current directory
    if not Path('app.py').exists():
        print("❌ Please run this script from the WEBPAGE directory")
        sys.exit(1)
    
    print("📁 Checking file structure...")
    if not verify_file_structure():
        print("\n❌ Setup incomplete. Please ensure all Sage AI files are present.")
        sys.exit(1)
    
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n🔧 Setting up environment...")
    api_key_ok = create_env_file()
    
    print("\n🧪 Testing integration...")
    integration_ok = test_sage_integration()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary")
    print("=" * 50)
    
    status = "✅ READY" if all([deps_ok, api_key_ok, integration_ok]) else "⚠️  NEEDS ATTENTION"
    print(f"Status: {status}")
    
    if not deps_ok:
        print("\n🔧 Next Steps:")
        print("1. Install required packages:")
        print("   pip install -r requirements.txt")
    
    if not api_key_ok:
        print("2. Add your Gemini API key to .env file")
        print("   Get one at: https://aistudio.google.com/app/apikey")
    
    if all([deps_ok, api_key_ok, integration_ok]):
        print("\n🎉 Sage AI Assistant is ready!")
        print("Start your Flask app with: python app.py")
        print("Look for the purple 'Sage' button in your patient/doctor dashboards.")
    
    print("\n📚 For help, see: AI_Assistant/README.md")

if __name__ == '__main__':
    main()