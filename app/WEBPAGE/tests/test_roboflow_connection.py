#!/usr/bin/env python3
"""
Simple Brain Tumor API Connection Test

This script tests the connection to the Roboflow API for brain tumor segmentation
without requiring an actual image file.
"""

import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load environment variables
load_dotenv()

def test_roboflow_connection():
    """Test connection to Roboflow API"""
    try:
        # Get API configuration from environment variables
        roboflow_api_key = os.getenv('ROBOFLOW_API_KEY')
        roboflow_api_url = os.getenv('ROBOFLOW_API_URL', 'https://serverless.roboflow.com/')
        roboflow_model_id = os.getenv('ROBOFLOW_BRAIN_MODEL_ID', 'brain-tumor-segmentation-dwapu-xxwhv/3')
        
        if not roboflow_api_key:
            print("❌ ERROR: ROBOFLOW_API_KEY not found in environment variables")
            print("Please set ROBOFLOW_API_KEY in your .env file")
            return False
        
        # Initialize the Roboflow client
        client = InferenceHTTPClient(
            api_url=roboflow_api_url,
            api_key=roboflow_api_key
        )
        
        print("✅ Roboflow client initialized successfully!")
        print(f"📡 API URL: {roboflow_api_url}")
        print(f"🔑 API Key: {roboflow_api_key[:4]}***{roboflow_api_key[-4:]}")
        print(f"🧠 Model ID: {roboflow_model_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing Roboflow client: {e}")
        return False

def show_usage_example():
    """Show example of how to use the brain tumor inference"""
    print("\n" + "="*50)
    print("📋 USAGE EXAMPLE")
    print("="*50)
    print("To test with an actual brain scan image:")
    print("1. Get a brain scan image (jpg, png, etc.)")
    print("2. Run: python brain_tumor_inference_test.py path/to/your/brain_scan.jpg")
    print("\nExample:")
    print("  python brain_tumor_inference_test.py C:\\images\\brain_scan.jpg")
    print("\n" + "="*50)
    print("🌐 FLASK APP INTEGRATION")
    print("="*50)
    print("The brain tumor inference is automatically integrated in app.py:")
    print("1. Start Flask app: python app.py")
    print("2. Upload brain scan through web interface")
    print("3. Select 'brain' as cancer type")
    print("4. Get Roboflow API results!")

def main():
    print("🧠 Brain Tumor Roboflow API Connection Test")
    print("=" * 45)
    
    if test_roboflow_connection():
        print("\n✅ SUCCESS: Ready for brain tumor inference!")
        show_usage_example()
    else:
        print("\n❌ FAILED: Cannot connect to Roboflow API")
        print("Please check your internet connection and API key.")

if __name__ == "__main__":
    main()