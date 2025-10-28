import requests
import sys

# Test if admin payments route works
try:
    response = requests.get('http://127.0.0.1:5601/admin/payments')
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    if response.status_code == 302:
        print("Redirected (likely to login page)")
        print(f"Redirect location: {response.headers.get('Location', 'Not specified')}")
    elif response.status_code == 200:
        print("Success! Admin payments page is accessible")
        print(f"Content length: {len(response.text)} characters")
    else:
        print(f"Unexpected response: {response.status_code}")
        
except Exception as e:
    print(f"Error connecting to admin payments: {e}")
    
# Also test the main application
try:
    response = requests.get('http://127.0.0.1:5601/')
    print(f"\nMain app status: {response.status_code}")
except Exception as e:
    print(f"Main app error: {e}")