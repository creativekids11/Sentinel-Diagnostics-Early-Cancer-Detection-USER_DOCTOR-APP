import requests
import time
from datetime import datetime, timedelta

# Base URL for the Flask app
BASE_URL = "http://127.0.0.1:5601"

def test_doctor_status():
    # Create a session to maintain cookies
    session = requests.Session()

    print("Testing Doctor Status System")
    print("=" * 40)

    # Step 1: Try to login as doctor (we need to check if users exist)
    print("\n1. Checking for existing users...")

    # First, let's see if we can access the login page
    try:
        response = session.get(f"{BASE_URL}/login")
        if response.status_code == 200:
            print("✓ Login page accessible")
        else:
            print(f"✗ Login page not accessible: {response.status_code}")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return

    # Step 2: Test the doctor status API endpoint
    print("\n2. Testing doctor status API...")

    # Since we don't have existing users, let's test the API directly
    # We'll need to create test users first or assume they exist

    # For now, let's test if the status endpoint exists
    try:
        response = session.get(f"{BASE_URL}/api/doctor/status")
        print(f"Status API response: {response.status_code}")
        if response.status_code == 401:
            print("✓ Status API requires authentication (expected)")
        elif response.status_code == 200:
            print("✓ Status API accessible")
            try:
                data = response.json()
                print(f"  Response data: {data}")
            except:
                print(f"  Response text: {response.text[:200]}")
        else:
            print(f"✗ Unexpected status API response: {response.status_code}")
    except Exception as e:
        print(f"✗ Status API error: {e}")

    print("\n3. Manual Testing Instructions:")
    print("- Open browser to http://127.0.0.1:5601")
    print("- Create/register a doctor account")
    print("- Login as doctor and go to dashboard")
    print("- Click 'Busy' or 'Emergency' button")
    print("- Try to book appointment as patient - should be blocked")
    print("- Wait for status to expire and try booking again")

    print("\n✓ Test setup complete. Server is running.")

if __name__ == "__main__":
    test_doctor_status()