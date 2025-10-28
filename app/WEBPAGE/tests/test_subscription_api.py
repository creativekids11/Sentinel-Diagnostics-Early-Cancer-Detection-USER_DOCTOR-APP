import requests
import json

def test_subscription_api():
    base_url = 'http://127.0.0.1:5000'

    print("Testing subscription verification API...")

    # Test with patient ID 1 (assuming it exists)
    try:
        response = requests.get(f'{base_url}/api/get_patient_subscription/1')
        print(f"Patient ID 1 - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error testing patient 1: {e}")

    # Test with invalid patient ID
    try:
        response = requests.get(f'{base_url}/api/get_patient_subscription/999')
        print(f"Patient ID 999 - Status: {response.status_code}")
        if response.status_code != 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"Unexpected success: {response.text}")
    except Exception as e:
        print(f"Error testing patient 999: {e}")

if __name__ == "__main__":
    test_subscription_api()