import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def test_model(model_path, test_images_dir):
    """
    Test the trained YOLO model on test images
    """
    # Load the trained model
    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return
    
    model = YOLO(model_path)
    print(f"Loaded model: {model_path}")
    
    # Get test images
    test_dir = Path(test_images_dir)
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_dir.glob(f"*{ext}")))
    
    if not test_images:
        print(f"No test images found in {test_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    # Process first 10 images for demonstration
    for i, img_path in enumerate(test_images[:10]):
        print(f"Processing {img_path.name}...")
        
        # Run inference
        results = model(str(img_path))
        
        # Get the result
        result = results[0]
        
        # Draw bounding boxes and save
        annotated_img = result.plot()
        
        # Save result
        output_path = results_dir / f"result_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
        
        # Print detections
        if result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                class_name = result.names[int(cls)]
                print(f"  Detected: {class_name} (confidence: {conf:.2f})")
        else:
            print("  No detections")
    
    print(f"\nResults saved in: {results_dir}")
    
    # Run validation on full test set
    print("\nRunning validation on test set...")
    val_results = model.val(data="dataset/data.yaml", split="test")
    
    print("\nTest Results:")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Test trained YOLO model')
    parser.add_argument('--model', type=str, default='yolo11m_lung_cancer_best.pt', 
                      help='Path to trained model')
    parser.add_argument('--test-dir', type=str, default='dataset/test/images', 
                      help='Path to test images directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YOLOv11m Lung Cancer Detection - Model Testing")
    print("="*60)
    
    test_model(args.model, args.test_dir)

if __name__ == "__main__":
    main()