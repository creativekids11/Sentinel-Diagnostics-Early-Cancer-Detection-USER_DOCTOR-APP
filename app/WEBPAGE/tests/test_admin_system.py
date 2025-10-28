import requests
import time

BASE_URL = "http://127.0.0.1:5602"

def test_admin_system():
    print("=== Testing Admin Payment Verification System ===\n")
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    try:
        # Test 1: Check if test app is running
        print("1. Testing if test app is accessible...")
        response = session.get(BASE_URL)
        print(f"   Status: {response.status_code}")
        print(f"   Content: {response.text.strip()}")
        
        # Test 2: Login as test doctor
        print("\n2. Logging in as test doctor...")
        response = session.post(f"{BASE_URL}/login")
        print(f"   Login status: {response.status_code}")
        if response.status_code == 200:
            print("   Successfully logged in!")
        
        # Test 3: Access admin payments page
        print("\n3. Accessing admin payments page...")
        response = session.get(f"{BASE_URL}/admin/payments")
        print(f"   Admin payments status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.text
            print(f"   Content length: {len(content)} characters")
            
            # Check if pending payments are displayed
            if "pending" in content.lower():
                print("   ✓ Pending payments section found")
            if "test@example.com" in content:
                print("   ✓ Test pending payment found in UI")
                
        # Test 4: Check database for pending payments
        print("\n4. Current database state:")
        import sqlite3
        conn = sqlite3.connect('instance/database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, user_email, plan_type, status, created_at FROM pending_payments ORDER BY id')
        rows = cursor.fetchall()
        
        for row in rows:
            print(f"   Payment ID {row[0]}: {row[1]} - {row[2]} - {row[3]} - {row[4]}")
        conn.close()
        
        print(f"\n   Total payments: {len(rows)}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_admin_system()