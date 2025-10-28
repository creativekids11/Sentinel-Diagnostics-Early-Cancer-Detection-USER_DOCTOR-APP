"""
Firebase Google OAuth Integration Test Script
Tests the new Google authentication features
"""

import os
import sys
import sqlite3
import requests
from datetime import datetime

# Add the WEBPAGE directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_database_schema():
    """Test that the google_id column exists in users table"""
    print("🔍 Testing database schema...")
    
    db_path = "database.db"  # Main database file
    if not os.path.exists(db_path):
        print("❌ Database not found. Please run the app first to initialize the database.")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if google_id column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'google_id' in columns:
            print("✅ google_id column exists in users table")
            return True
        else:
            print("❌ google_id column missing from users table")
            return False
            
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_firebase_config():
    """Test that Firebase configuration file exists and is properly formatted"""
    print("\n🔍 Testing Firebase configuration...")
    
    config_path = "static/js/firebase-config.js"
    if not os.path.exists(config_path):
        print("❌ Firebase config file not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Check for required Firebase config elements
        required_elements = [
            'firebaseConfig',
            'apiKey',
            'authDomain',
            'projectId',
            'initializeApp',
            'signInWithGoogle',
            'GoogleAuthProvider'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing Firebase config elements: {missing_elements}")
            return False
        else:
            print("✅ Firebase configuration file is properly structured")
            return True
            
    except Exception as e:
        print(f"❌ Error reading Firebase config: {str(e)}")
        return False

def test_templates():
    """Test that authentication templates include Google OAuth buttons"""
    print("\n🔍 Testing template modifications...")
    
    templates_to_check = [
        ("templates/auth/login.html", ["googleSignInBtn", "Continue with Google"]),
        ("templates/auth/signup.html", ["googleSignUpBtn", "Continue with Google", "googleRoleModal"]),
        ("templates/auth/doctor_verification.html", ["qualification_pdf", "Complete Your Verification"])
    ]
    
    all_passed = True
    
    for template_path, required_elements in templates_to_check:
        if not os.path.exists(template_path):
            print(f"❌ Template not found: {template_path}")
            all_passed = False
            continue
            
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            missing_elements = []
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"❌ {template_path} missing elements: {missing_elements}")
                all_passed = False
            else:
                print(f"✅ {template_path} has required Google OAuth elements")
                
        except Exception as e:
            print(f"❌ Error reading {template_path}: {str(e)}")
            all_passed = False
    
    return all_passed

def test_flask_routes():
    """Test that Google OAuth routes are defined in app.py"""
    print("\n🔍 Testing Flask routes...")
    
    if not os.path.exists("app.py"):
        print("❌ app.py not found")
        return False
    
    try:
        with open("app.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_routes = [
            "@app.route(\"/auth/google/login\"",
            "@app.route(\"/auth/google/signup\"",
            "def google_login():",
            "def google_signup():",
            "def doctor_verification_page(",
            "def download_google_photo("
        ]
        
        missing_routes = []
        for route in required_routes:
            if route not in content:
                missing_routes.append(route)
        
        if missing_routes:
            print(f"❌ Missing Flask routes/functions: {missing_routes}")
            return False
        else:
            print("✅ All required Google OAuth routes are defined")
            return True
            
    except Exception as e:
        print(f"❌ Error reading app.py: {str(e)}")
        return False

def test_upload_directories():
    """Test that upload directories exist and are writable"""
    print("\n🔍 Testing upload directories...")
    
    directories = [
        "static/uploads/photos",
        "static/uploads/pancards", 
        "static/uploads/qualifications"
    ]
    
    all_passed = True
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created directory: {directory}")
            except Exception as e:
                print(f"❌ Cannot create directory {directory}: {str(e)}")
                all_passed = False
        else:
            print(f"✅ Directory exists: {directory}")
        
        # Test write permissions
        try:
            test_file = os.path.join(directory, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Directory writable: {directory}")
        except Exception as e:
            print(f"❌ Directory not writable {directory}: {str(e)}")
            all_passed = False
    
    return all_passed

def test_requirements():
    """Test that requirements.txt includes necessary packages"""
    print("\n🔍 Testing requirements...")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open("requirements.txt", 'r') as f:
            content = f.read()
        
        required_packages = ["requests", "Flask", "Pillow"]
        missing_packages = []
        
        for package in required_packages:
            if package.lower() not in content.lower():
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {missing_packages}")
            return False
        else:
            print("✅ All required packages are in requirements.txt")
            return True
            
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Firebase Google OAuth Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_firebase_config,
        test_templates,
        test_flask_routes,
        test_upload_directories,
        test_requirements,
        test_database_schema,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Firebase integration is ready.")
        print("\n📋 Next steps:")
        print("1. Update firebase-config.js with your actual Firebase project settings")
        print("2. Start the Flask application")
        print("3. Test Google OAuth login/signup in browser")
        print("4. Verify doctor verification flow")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    main()