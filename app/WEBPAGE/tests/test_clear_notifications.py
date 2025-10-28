import requests
import json

# Test the clear all notifications endpoint
base_url = "http://localhost:5601"

# First, let's try to get notifications (this will fail without authentication, but we can check if the route exists)
try:
    response = requests.get(f"{base_url}/api/notifications")
    print(f"GET /api/notifications status: {response.status_code}")
    if response.status_code == 401:
        print("Route exists but requires authentication (expected)")
    else:
        print(f"Unexpected response: {response.text}")
except Exception as e:
    print(f"Error testing GET /api/notifications: {e}")

# Test the clear all endpoint
try:
    response = requests.delete(f"{base_url}/api/notifications/clear-all")
    print(f"DELETE /api/notifications/clear-all status: {response.status_code}")
    if response.status_code == 401:
        print("Route exists but requires authentication (expected)")
    else:
        print(f"Unexpected response: {response.text}")
except Exception as e:
    print(f"Error testing DELETE /api/notifications/clear-all: {e}")

print("Test completed.")