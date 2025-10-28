#!/usr/bin/env python3
"""
Test script to verify the Clear All notifications functionality
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5601"

def test_clear_all_notifications():
    """Test the clear all notifications endpoint"""

    print("Testing Clear All Notifications functionality...")

    # Test the clear all endpoint (will fail without authentication, but we can check the response)
    try:
        response = requests.delete(f"{BASE_URL}/api/notifications/clear-all")
        print(f"Clear All Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        # Try to get the response content
        try:
            data = response.json()
            print(f"Response JSON: {data}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")

        if response.status_code == 401:
            print("✅ Endpoint exists and requires authentication (expected)")
        elif response.status_code == 200:
            print("✅ Clear All notifications endpoint working!")
        else:
            print(f"❌ Unexpected status code: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"❌ Error testing clear all: {e}")

if __name__ == "__main__":
    test_clear_all_notifications()