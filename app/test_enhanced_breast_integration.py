#!/usr/bin/env python3
"""
Test script for enhanced breast cancer detection using MLMO-EMO algorithm
"""

import os
import sys
import cv2
import numpy as np

# Add the WEBPAGE directory to the path so we can import the functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'WEBPAGE'))

try:
    from app import perform_enhanced_breast_inference, initialize_enhanced_breast_model
    print("Successfully imported enhanced breast cancer functions")
except ImportError as e:
    print(f"Error importing functions: {e}")
    sys.exit(1)

def test_enhanced_breast_detection():
    """Test the enhanced breast cancer detection with a sample image"""

    print("Testing Enhanced Breast Cancer Detection with MLMO-EMO Algorithm")
    print("=" * 60)

    # Initialize the model
    print("Initializing enhanced breast model...")
    success = initialize_enhanced_breast_model()
    if not success:
        print("Failed to initialize enhanced breast model")
        return False

    print("Enhanced breast model initialized successfully")

    # Look for test images in the uploads directory
    uploads_dir = os.path.join(os.path.dirname(__file__), 'WEBPAGE', 'static', 'uploads')

    # Try to find a suitable test image
    test_image_path = None

    # Look for any image file in uploads
    for root, dirs, files in os.walk(uploads_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                test_image_path = os.path.join(root, file)
                break
        if test_image_path:
            break

    if not test_image_path:
        print("No test image found in uploads directory. Creating a synthetic test image...")

        # Create a synthetic mammogram-like image for testing
        test_image_path = os.path.join(os.path.dirname(__file__), 'test_breast_image.png')

        # Create a synthetic mammogram image (512x512 grayscale)
        height, width = 512, 512

        # Create base image with some tissue-like texture
        base_image = np.random.normal(128, 20, (height, width)).astype(np.uint8)

        # Add some circular structures to simulate breast tissue
        for _ in range(10):
            center_x = np.random.randint(50, width-50)
            center_y = np.random.randint(50, height-50)
            radius = np.random.randint(10, 30)
            intensity = np.random.randint(100, 200)

            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            base_image[mask] = intensity

        # Add a simulated tumor (darker region)
        tumor_x = width // 2
        tumor_y = height // 2
        tumor_radius = 25
        tumor_intensity = 80  # Darker than surrounding tissue

        y, x = np.ogrid[:height, :width]
        tumor_mask = (x - tumor_x)**2 + (y - tumor_y)**2 <= tumor_radius**2
        base_image[tumor_mask] = tumor_intensity

        # Save the synthetic image
        cv2.imwrite(test_image_path, base_image)
        print(f"Created synthetic test image: {test_image_path}")

    print(f"Using test image: {test_image_path}")

    # Test the enhanced breast cancer detection
    print("\nRunning enhanced breast cancer detection...")
    try:
        result = perform_enhanced_breast_inference(test_image_path)

        if "error" in result:
            print(f"Error in breast cancer detection: {result['error']}")
            return False

        print("Enhanced Breast Cancer Detection Results:")
        print("-" * 40)
        print(f"Confidence: {result.get('confidence', 'N/A')}%")
        print(f"Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"Tumor Percentage: {result.get('tumor_percentage', 'N/A')}%")
        print(f"Algorithm: {result.get('algorithm', 'N/A')}")
        print(f"Optimization Iterations: {result.get('optimization_iterations', 'N/A')}")

        print("\nFindings:")
        findings = result.get('findings', [])
        if findings:
            for finding in findings:
                print(f"- {finding}")
        else:
            print("No specific findings")

        # Check if segmentation mask was generated
        mask = result.get('_internal_mask')
        if mask is not None:
            print(f"\nSegmentation: Available (mask shape: {mask.shape})")
            unique_vals = np.unique(mask)
            print(f"Mask values: {unique_vals}")
        else:
            print("\nSegmentation: Not available")

        print("\nTest completed successfully!")
        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_breast_detection()
    sys.exit(0 if success else 1)