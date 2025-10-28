#!/usr/bin/env python3
"""
Test Admin Payment Rejection
Simulates an admin rejecting a payment
"""

import requests
import json

def test_admin_rejection():
    """Test the admin payment rejection endpoint"""
    
    # Test data
    payment_id = 3  # The pending payment ID we found
    admin_notes = "Test rejection - insufficient payment proof"
    
    # This would normally require a logged-in session
    # For now, just test that the endpoint is accessible
    url = f"http://127.0.0.1:5601/admin/payments/{payment_id}/reject"
    
    data = {
        'admin_notes': admin_notes
    }
    
    try:
        response = requests.post(url, data=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        
        if response.status_code == 302:
            print("Redirected (likely to login) - this is expected without authentication")
        elif response.status_code == 200:
            print("Request successful!")
        else:
            print(f"Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing Admin Payment Rejection")
    print("===============================")
    test_admin_rejection()
    print("\nNote: This test expects a 302 redirect to login without authentication.")
    print("The real test would be to log in as an admin and then try the rejection.")