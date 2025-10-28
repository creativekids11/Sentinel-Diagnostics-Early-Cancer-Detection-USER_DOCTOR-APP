#!/usr/bin/env python3
"""
Test the brain tumor inference function integration
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the function from app.py
from app import perform_roboflow_brain_inference, initialize_models

def test_brain_tumor_integration():
    """Test the integrated brain tumor inference function"""
    # Initialize models first
    print("Initializing models...")
    initialize_models()

    # Use a sample brain scan image
    image_path = "static/uploads/scans/10_brain_20251021_191109_brain_tumor_scan.jpeg"

    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return False

    print(f"Testing brain tumor inference with: {image_path}")

    # Call the integrated function
    result = perform_roboflow_brain_inference(image_path)

    if "error" in result:
        print(f"❌ Test failed: {result['error']}")
        return False

    # Print results
    print("✅ Brain tumor inference successful!")
    print(f"Confidence: {result['confidence']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Tumor Percentage: {result['tumor_percentage']}%")
    print(f"Segmentation Available: {result['segmentation_available']}")
    print(f"Findings: {len(result['findings'])} items")

    for i, finding in enumerate(result['findings'], 1):
        print(f"  {i}. {finding}")

    # Check mask
    if '_internal_mask' in result and result['_internal_mask'] is not None:
        mask = result['_internal_mask']
        print(f"Mask shape: {mask.shape}")
        print(f"Mask type: {mask.dtype}")
        print(f"Mask unique values: {set(mask.flatten())}")

    return True

if __name__ == "__main__":
    success = test_brain_tumor_integration()
    if success:
        print("\n🎉 Brain tumor detection integration test passed!")
    else:
        print("\n💥 Brain tumor detection integration test failed!")
        sys.exit(1)