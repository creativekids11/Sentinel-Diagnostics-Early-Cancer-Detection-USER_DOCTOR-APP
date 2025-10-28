import sys
import os
import numpy as np
import cv2

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the analysis function directly
from app import analyze_with_ai_model

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

# Create and save test image
test_img = create_test_image()
test_image_path = 'test_mammogram.png'
cv2.imwrite(test_image_path, test_img)
print("Test image created and saved as 'test_mammogram.png'")

# Test the enhanced breast cancer analysis
print("Testing enhanced breast cancer analysis...")
try:
    ai_results = analyze_with_ai_model(test_image_path, 'breast')

    print("Analysis completed!")
    print(f"Results: {ai_results}")

    if ai_results.get('error'):
        print(f"Error: {ai_results['error']}")
    else:
        print(f"Confidence: {ai_results.get('confidence', 'N/A')}%")
        print(f"Risk Level: {ai_results.get('risk_level', 'N/A')}")
        print(f"Tumor Area: {ai_results.get('tumor_percentage', 'N/A')}%")
        print(f"Algorithm: {ai_results.get('algorithm', 'N/A')}")
        print(f"Segmentation Available: {ai_results.get('segmentation_available', False)}")

except Exception as e:
    print(f"Analysis failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up
if os.path.exists(test_image_path):
    os.remove(test_image_path)
    print("Test image cleaned up.")