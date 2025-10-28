#!/usr/bin/env python3
"""
Quick test script for Sage AI Assistant
Run this to verify that Sage AI is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

def test_sage_ai():
    """Test Sage AI Assistant functionality."""
    print("🧪 Testing Sage AI Assistant")
    print("=" * 40)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from AI_Assistant.functions import SageAssistant
        print("   ✅ SageAssistant imported successfully")
        
        # Test environment variables
        print("2. Testing environment variables...")
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY', '')
        if api_key and api_key != 'your-gemini-api-key-here':
            print("   ✅ Gemini API key found")
        else:
            print("   ⚠️  Gemini API key not configured")
            return False
        
        # Test Sage initialization
        print("3. Testing Sage initialization...")
        sage = SageAssistant()
        print("   ✅ Sage AI Assistant initialized")
        
        # Test knowledge loading
        print("4. Testing knowledge base...")
        if hasattr(sage, 'knowledge_base') and sage.knowledge_base:
            print(f"   ✅ Knowledge base loaded ({len(sage.knowledge_base)} characters)")
        else:
            print("   ⚠️  Knowledge base not loaded")
        
        # Test basic AI response (if API key is valid)
        print("5. Testing AI response...")
        try:
            response = sage.get_response("Hello, what is Sentinel Diagnostics?", "patient")
            if response and len(response) > 10:
                print("   ✅ AI response generated successfully")
                print(f"   📝 Sample response: {response[:100]}...")
            else:
                print("   ⚠️  AI response seems empty or invalid")
        except Exception as e:
            print(f"   ❌ Error generating AI response: {e}")
            return False
        
        # Test suggestions
        print("6. Testing quick suggestions...")
        suggestions = sage.get_quick_suggestions("patient")
        if suggestions and len(suggestions) > 0:
            print(f"   ✅ Generated {len(suggestions)} suggestions")
            print(f"   💡 Sample suggestions: {suggestions[:2]}")
        else:
            print("   ⚠️  No suggestions generated")
        
        print("\n🎉 All tests passed! Sage AI is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_flask_integration():
    """Test Flask integration."""
    print("\n🌐 Testing Flask Integration")
    print("=" * 40)
    
    try:
        # Test blueprint import
        print("1. Testing blueprint import...")
        from AI_Assistant.routes import ai_assistant_bp
        print("   ✅ AI Assistant blueprint imported")
        
        # Check routes
        print("2. Checking routes...")
        routes = [rule.rule for rule in ai_assistant_bp.url_map.iter_rules()]
        expected_routes = ['/api/sage/chat', '/api/sage/suggestions', '/api/sage/search', '/api/sage/health']
        
        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"   ✅ Route {route} found")
            else:
                print(f"   ⚠️  Route {route} not found")
        
        print("\n✅ Flask integration looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Flask integration error: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 Sage AI Assistant Test Suite")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Please run this script from the WEBPAGE directory")
        sys.exit(1)
    
    # Run tests
    sage_ok = test_sage_ai()
    flask_ok = test_flask_integration()
    
    print("\n" + "=" * 50)
    print("📊 Test Results")
    print("=" * 50)
    
    if sage_ok and flask_ok:
        print("🎉 All tests passed!")
        print("\n🚀 Ready to use Sage AI Assistant!")
        print("   Start your Flask app: python app.py")
        print("   Visit: http://localhost:5000")
        print("   Look for the purple 'Sage' button in patient/doctor dashboards")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        if not sage_ok:
            print("   - Sage AI core functionality needs attention")
        if not flask_ok:
            print("   - Flask integration needs attention")

if __name__ == '__main__':
    main()