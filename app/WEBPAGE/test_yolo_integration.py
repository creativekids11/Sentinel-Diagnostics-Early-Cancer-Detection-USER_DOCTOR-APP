#!/usr/bin/env python3
"""
Test script for YOLO lung cancer detection integration
"""
import sys
import os
import cv2
import base64
import numpy as np
from ultralytics import YOLO

def perform_yolo_lung_inference(image_path):
    """Perform YOLO inference for lung cancer ROI detection"""
    try:
        # Load YOLO model
        model_path = r"c:\Users\Jay\Downloads\Hackathon-2.0\Lung_Cancer\runs\best.pt"
        lung_yolo_model = YOLO(model_path)
        
        # Read original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            return {"error": "Could not read uploaded image"}
        
        original_size = (original_img.shape[1], original_img.shape[0])  # (width, height)
        
        # Run YOLO inference
        results = lung_yolo_model(image_path, conf=0.25)  # confidence threshold
        
        # Process results
        detections = []
        annotated_img = original_img.copy()
        
        if len(results) > 0:
            result = results[0]  # Get first result
            
            # Get bounding boxes, confidences, and class IDs
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates, confidence, and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_names = lung_yolo_model.names
                    class_name = class_names.get(cls, f"class_{cls}")
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Store detection
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    
                    # Draw bounding box on image
                    color = (0, 255, 0) if class_name == "Lung Lobe" else (255, 0, 0)  # Green for lung lobe, red for tumor
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate statistics
        num_tumors = sum(1 for d in detections if d["class"] == "Tumor")
        num_lung_lobes = sum(1 for d in detections if d["class"] == "Lung Lobe")
        
        # Determine risk level based on detections
        if num_tumors > 0:
            risk_level = "High"
            color = "#ef4444"
            confidence = min(95, 70 + num_tumors * 10)
        elif num_lung_lobes > 0:
            risk_level = "Low"
            color = "#10b981"
            confidence = 85
        else:
            risk_level = "Unknown"
            color = "#6b7280"
            confidence = 50
        
        # Generate findings
        findings = []
        if num_tumors > 0:
            findings.append(f"Detected {num_tumors} potential tumor(s) in lung tissue")
            findings.append("Immediate clinical evaluation recommended")
            findings.append("Consider biopsy and oncological consultation")
        elif num_lung_lobes > 0:
            findings.append(f"Detected {num_lung_lobes} lung lobe(s) - normal lung structure identified")
            findings.append("No suspicious lesions detected")
        else:
            findings.append("No clear lung structures or abnormalities detected")
            findings.append("Consider re-uploading image or different imaging modality")
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.png', annotated_img)
        annotated_img_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert original image to base64 for overlay
        _, orig_buffer = cv2.imencode('.png', original_img)
        orig_img_b64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        return {
            "confidence": round(confidence, 1),
            "risk_level": risk_level,
            "color": color,
            "findings": findings,
            "tumor_percentage": 0,  # Not applicable for YOLO
            "segmentation_available": False,  # This is object detection, not segmentation
            "yolo_detections": detections,
            "num_tumors": num_tumors,
            "num_lung_lobes": num_lung_lobes,
            "_internal_annotated_image": annotated_img_b64,
            "_internal_original_image": orig_img_b64
        }
        
    except Exception as e:
        print(f"Error in YOLO lung inference: {e}")
        return {
            "error": f"YOLO inference failed: {str(e)}",
            "confidence": 0,
            "risk_level": "Unknown",
            "color": "#6b7280",
            "findings": ["YOLO analysis could not be completed"],
            "tumor_percentage": 0,
            "segmentation_available": False
        }

def test_yolo_inference():
    """Test the YOLO inference function with a sample image"""
    # Use a test image that should have detections
    test_image_path = r"c:\Users\Jay\Downloads\Hackathon-2.0\Lung_Cancer\test_results\result_Bengin-case-114-_jpg.rf.4da706dae9fe4d08f154f7a532359e15.jpg"

    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return

    print("Testing YOLO lung cancer inference...")
    print(f"Using test image: {test_image_path}")

    # Run inference
    result = perform_yolo_lung_inference(test_image_path)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print("\nInference Results:")
    print(f"Confidence: {result['confidence']}%")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Color: {result['color']}")
    print(f"Number of tumors: {result['num_tumors']}")
    print(f"Number of lung lobes: {result['num_lung_lobes']}")

    print("\nFindings:")
    for finding in result['findings']:
        print(f"- {finding}")

    print(f"\nDetections ({len(result['yolo_detections'])}):")
    for detection in result['yolo_detections']:
        print(f"- {detection['class']}: {detection['confidence']:.2%} at {detection['bbox']}")

    # Check if annotated image was created
    if "_internal_annotated_image" in result:
        print("\nAnnotated image created successfully!")
        print("Base64 image length:", len(result["_internal_annotated_image"]))
    else:
        print("\nWarning: No annotated image created")

if __name__ == "__main__":
    test_yolo_inference()