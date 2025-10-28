#!/usr/bin/env python3
"""
Brain Tumor Inference Test

This script demonstrates how to use the Roboflow API for brain tumor segmentation.
Run this to test the brain tumor inference functionality.
"""

import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables
load_dotenv()

# Initialize the Roboflow client with environment variables
roboflow_api_key = os.getenv('ROBOFLOW_API_KEY')
roboflow_api_url = os.getenv('ROBOFLOW_API_URL', 'https://serverless.roboflow.com/')

if not roboflow_api_key:
    print("❌ ERROR: ROBOFLOW_API_KEY not found in environment variables")
    print("Please set ROBOFLOW_API_KEY in your .env file")
    exit(1)

CLIENT = InferenceHTTPClient(
    api_url=roboflow_api_url,
    api_key=roboflow_api_key
)

def test_brain_tumor_inference(image_path):
    """
    Test brain tumor inference using Roboflow API
    
    Args:
        image_path (str): Path to the brain scan image
    
    Returns:
        dict: Inference results from Roboflow API
    """
    try:
        print(f"Running brain tumor inference on: {image_path}")
        
        # Perform inference using Roboflow
        result = CLIENT.infer(image_path, model_id="brain-tumor-segmentation-dwapu-xxwhv/3")
        
        print("Inference successful!")
        print(f"Inference ID: {result.get('inference_id')}")
        print(f"Processing time: {result.get('time')} ms")
        print(f"Number of predictions: {len(result.get('predictions', []))}")
        
        # Print prediction details
        for i, prediction in enumerate(result.get('predictions', [])):
            print(f"Prediction {i+1}:")
            print(f"  Confidence: {prediction.get('confidence', 'N/A')}")
            print(f"  Class: {prediction.get('class', 'N/A')}")
            if 'points' in prediction:
                print(f"  Points: {len(prediction['points'])} segmentation points")
        
        return result
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return {"error": str(e)}

def main():
    """Main function to test brain tumor inference"""
    import sys
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python brain_tumor_inference_test.py <image_path>")
        print("Example: python brain_tumor_inference_test.py path/to/brain_scan.jpg")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Test the inference
    result = test_brain_tumor_inference(image_path)
    
    if "error" not in result:
        print("\n=== Test Completed Successfully! ===")
        print("The Roboflow brain tumor inference is working correctly.")
    else:
        print(f"\n=== Test Failed ===")
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()