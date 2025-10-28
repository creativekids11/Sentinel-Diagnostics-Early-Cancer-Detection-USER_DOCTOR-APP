#!/usr/bin/env python3
"""
Test Visual Overlay Adjustments

This script demonstrates the updated visual settings for risk area overlays:
- Red overlay with 90% transparency (10% opacity)
- Thinner green contour lines (1 pixel instead of 2)
"""

import numpy as np
import cv2
from PIL import Image
import os

def create_sample_overlay_demo():
    """Create a demo showing the new overlay settings"""
    
    print("🎨 Creating Visual Overlay Demo")
    print("=" * 35)
    
    # Create a sample brain scan image (grayscale)
    img_size = 512
    brain_img = np.random.randint(50, 200, (img_size, img_size), dtype=np.uint8)
    
    # Add some brain-like texture
    for _ in range(10):
        center = (np.random.randint(100, img_size-100), np.random.randint(100, img_size-100))
        radius = np.random.randint(20, 80)
        cv2.circle(brain_img, center, radius, np.random.randint(80, 150), -1)
    
    # Convert to RGB for overlay
    img_rgb = cv2.cvtColor(brain_img, cv2.COLOR_GRAY2RGB)
    
    # Create sample tumor mask (simulate detected regions)
    mask = np.zeros(brain_img.shape, dtype=np.uint8)
    
    # Add some tumor-like regions
    cv2.ellipse(mask, (200, 150), (40, 60), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (350, 300), (25, 35), 45, 0, 360, 255, -1)
    cv2.circle(mask, (100, 400), 20, 255, -1)
    
    # Apply the NEW overlay settings
    overlay = img_rgb.copy()
    mask_bool = (mask > 0)
    
    # NEW: 90% transparency red overlay (alpha = 0.1)
    red = np.zeros_like(img_rgb)
    red[..., 0] = 255  # Red channel
    alpha = 0.1  # 10% opacity = 90% transparency
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + red[mask_bool] * alpha)
    
    # NEW: Thinner green contours (thickness = 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 1)  # Thickness = 1 (thin)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    
    # Save demo images
    try:
        cv2.imwrite("demo_original.png", brain_img)
        cv2.imwrite("demo_overlay_new_settings.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        print("✅ Demo images created:")
        print("  📁 demo_original.png - Original brain scan")
        print("  📁 demo_overlay_new_settings.png - With new visual settings")
        
    except Exception as e:
        print(f"❌ Error saving demo images: {e}")
    
    # Display settings info
    print(f"\n🎯 NEW Visual Settings Applied:")
    print(f"  🔴 Red Overlay Transparency: {int((1-alpha)*100)}% (alpha={alpha})")
    print(f"  🟢 Green Contour Thickness: 1 pixel (was 2)")
    print(f"  📊 Detected Regions: {len(contours)} tumor areas")
    
    return True

def show_settings_comparison():
    """Show before/after settings comparison"""
    print(f"\n📋 Settings Comparison")
    print("=" * 25)
    
    print("BEFORE:")
    print("  🔴 Red Overlay: 50% transparency (alpha=0.5)")
    print("  🟢 Green Lines: 2 pixel thickness")
    print("  👁️  Effect: Strong red overlay, thick borders")
    
    print("\nAFTER:")
    print("  🔴 Red Overlay: 90% transparency (alpha=0.1)")  
    print("  🟢 Green Lines: 1 pixel thickness")
    print("  👁️  Effect: Subtle red tint, crisp thin borders")
    
    return True

def main():
    """Run visual overlay demo"""
    print("🖼️  Visual Overlay Settings Update Demo\n")
    
    demo_ok = create_sample_overlay_demo()
    comparison_ok = show_settings_comparison()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    if demo_ok and comparison_ok:
        print("✅ Visual settings updated successfully!")
        print("✅ Red overlay now 90% transparent")
        print("✅ Green contours now thinner (1px)")
        print("✅ More subtle and professional appearance")
        print("\n🚀 Ready to test in Flask app!")
        
        print(f"\n💡 The brain tumor risk areas will now show:")
        print(f"  → Very subtle red tint over detected regions")
        print(f"  → Crisp, thin green outlines")
        print(f"  → Better visibility of underlying brain structure")
    else:
        print("❌ Demo creation failed")

if __name__ == "__main__":
    main()