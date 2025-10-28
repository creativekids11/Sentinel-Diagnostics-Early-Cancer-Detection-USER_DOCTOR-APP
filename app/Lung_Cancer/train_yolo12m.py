import torch
from ultralytics import YOLO
import os
from pathlib import Path
import argparse

def train_yolo12m():
    """
    Train YOLOv12m on lung cancer dataset
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataset path
    data_path = Path("dataset/data.yaml").resolve()
    
    if not data_path.exists():
        print(f"Error: Dataset config file not found at {data_path}")
        return
    
    print(f"Dataset config: {data_path}")
    
    # Initialize YOLOv11n model (nano - much faster, using as YOLOv12n equivalent)
    model = YOLO('yolo11n.pt')  # YOLOv11n as base (fastest version, equivalent to YOLOv12n)
    
    print("Starting YOLOv12n (nano) training...")
    print("Dataset classes: Lung Lobe, Tumor")
    
    # Training parameters
    results = model.train(
        data=str(data_path),
        epochs=100,  # Full training epochs for better performance
        imgsz=416,  # Reduced image size for faster training
        batch=8,  # Reduced batch size for faster training
        device=device,
        project='runs/train',
        name='yolo12n_lung_cancer',
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        patience=10,  # Reduced patience for faster convergence
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=2.0,
        label_smoothing=0.0,
        nbs=64,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0,
        mixup=0.0,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    print(f"Training completed!")
    print(f"Results saved to: runs/train/yolo11m_lung_cancer")
    
    # Validate the model
    print("\nRunning validation...")
    val_results = model.val(data=str(data_path))
    
    print("\nValidation Results:")
    print(f"mAP50: {val_results.box.map50}")
    print(f"mAP50-95: {val_results.box.map}")
    
    # Save the trained model
    model_save_path = "yolo12n_lung_cancer_best.pt"
    best_model_path = Path("runs/train/yolo12n_lung_cancer/weights/best.pt")
    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, model_save_path)
        print(f"Best model saved as: {model_save_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv12n on Lung Cancer Dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=416, help='Image size')
    parser.add_argument('--device', type=str, default='', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("YOLOv12n Lung Cancer Detection Training")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print("Classes: Lung Lobe, Tumor")
    print("="*60)
    
    try:
        results = train_yolo12m()
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        print("Make sure you have:")
        print("1. Installed ultralytics: pip install ultralytics")
        print("2. Dataset in the correct structure")
        print("3. Sufficient GPU memory (if using CUDA)")

if __name__ == "__main__":
    main()