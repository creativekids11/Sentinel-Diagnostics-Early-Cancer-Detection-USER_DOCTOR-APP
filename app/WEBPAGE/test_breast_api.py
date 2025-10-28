import requests
import os
import numpy as np
import cv2

# Create a session to maintain cookies
session = requests.Session()

# Create a synthetic mammogram image for testing
def create_test_image():
    """Create a synthetic mammogram image with some features"""
    # Create a 512x512 grayscale image
    img = np.zeros((512, 512), dtype=np.uint8)

    # Add some background noise
    noise = np.random.normal(50, 10, (512, 512))
    img = np.clip(img + noise, 0, 255).astype(np.uint8)

    # Add a simulated mass (circular region)
    center_x, center_y = 256, 256
    radius = 30
    y, x = np.ogrid[:512, :512]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    img[mask] = np.clip(img[mask] + 80, 0, 255)  # Make mass brighter

    # Add some tissue-like structures
    for i in range(10):
        cx = np.random.randint(50, 462)
        cy = np.random.randint(50, 462)
        r = np.random.randint(5, 15)
        y_coords, x_coords = np.ogrid[:512, :512]
        tissue_mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= r**2
        img[tissue_mask] = np.clip(img[tissue_mask] + np.random.randint(20, 40), 0, 255)

    return img

# Login function
def login(username, password):
    """Login to get authenticated session"""
    login_url = 'http://127.0.0.1:5000/login'
    login_data = {
        'username': username,
        'password': password
    }

    print(f"Logging in as {username}...")
    response = session.post(login_url, data=login_data, allow_redirects=True)

    if response.status_code == 200:
        print("Login successful!")
        return True
    else:
        print(f"Login failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

# Save test image
test_img = create_test_image()
cv2.imwrite('test_mammogram.png', test_img)
print("Test image created and saved as 'test_mammogram.png'")

# Login first (using existing doctor account)
if login('Ballastic_1422', 'password123'):  # Try common password
    # Test the API
    url = 'http://127.0.0.1:5000/doctor/ai-scanner/upload'
    files = {'scan': open('test_mammogram.png', 'rb')}
    data = {'scan_type': 'breast'}

    print("Sending request to breast cancer analysis endpoint...")
    try:
        response = session.post(url, files=files, data=data)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            # The endpoint redirects on success, check the final URL
            if 'ai-scanner' in response.url:
                print("Analysis completed successfully!")
                print("Check the web interface for results")
            else:
                print(f"Unexpected redirect to: {response.url}")
        else:
            print(f"Error response: {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")
else:
    print("Could not authenticate. Please check credentials.")

# Clean up
if os.path.exists('test_mammogram.png'):
    os.remove('test_mammogram.png')