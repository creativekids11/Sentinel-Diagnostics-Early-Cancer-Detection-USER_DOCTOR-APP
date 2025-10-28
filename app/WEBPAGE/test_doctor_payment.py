#!/usr/bin/env python3
"""
Test script to verify doctor payment submission works correctly
"""
import requests
import json

def test_payment_submission():
    """Test the doctor payment submission endpoint"""

    # Test data
    test_data = {
        "plan_type": "premium",
        "amount": "5",
        "upi_transaction_id": "TEST123456789",
        "payment_method": "UPI",
        "user_message": "Test payment submission",
        "email_receipt": True
    }

    print("Testing doctor payment submission...")
    print(f"Test data: {json.dumps(test_data, indent=2)}")

    try:
        # Make the request
        response = requests.post(
            'http://127.0.0.1:5000/doctor/submit-payment',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )

        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")

        try:
            result = response.json()
            print(f"Response JSON: {json.dumps(result, indent=2)}")
        except:
            print(f"Response text: {response.text}")

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server. Make sure the Flask app is running.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_payment_submission()