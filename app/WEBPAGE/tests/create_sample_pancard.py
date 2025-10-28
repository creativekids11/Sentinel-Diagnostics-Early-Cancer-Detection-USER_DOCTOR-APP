"""
Script to create a sample PAN card image for testing OCR functionality
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_pancard():
    """Create a realistic sample PAN card image"""
    
    # Create image dimensions similar to actual PAN card
    width, height = 850, 540
    
    # Create background with light blue gradient
    img = Image.new('RGB', (width, height), color='#E6F3FF')
    draw = ImageDraw.Draw(img)
    
    # Try to load different fonts, fallback to default
    try:
        # Try common Windows fonts
        font_large = ImageFont.truetype("arial.ttf", 32)
        font_medium = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 18)
        font_tiny = ImageFont.truetype("arial.ttf", 14)
    except:
        try:
            # Try loading from Windows system fonts
            font_large = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
            font_medium = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 24)
            font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            font_tiny = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
        except:
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()
    
    # Add border
    draw.rectangle([10, 10, width-10, height-10], outline='#003366', width=3)
    
    # Header section
    draw.text((50, 30), "INCOME TAX DEPARTMENT", fill='#003366', font=font_large)
    draw.text((50, 70), "GOVT. OF INDIA", fill='#003366', font=font_medium)
    
    # PAN card title
    draw.text((50, 120), "PERMANENT ACCOUNT NUMBER CARD", fill='#000080', font=font_medium)
    
    # Add a sample photo placeholder
    photo_rect = [600, 150, 750, 300]
    draw.rectangle(photo_rect, fill='#CCCCCC', outline='#666666', width=2)
    draw.text((620, 215), "PHOTO", fill='#666666', font=font_small)
    
    # Personal details section
    y_start = 200
    line_height = 35
    
    # Name
    draw.text((50, y_start), "Name", fill='#000000', font=font_small)
    draw.text((200, y_start), "JOHN DOE", fill='#000000', font=font_medium)
    
    # Father's name
    draw.text((50, y_start + line_height), "Father's Name", fill='#000000', font=font_small)
    draw.text((200, y_start + line_height), "ROBERT DOE", fill='#000000', font=font_small)
    
    # Date of birth
    draw.text((50, y_start + 2*line_height), "Date of Birth", fill='#000000', font=font_small)
    draw.text((200, y_start + 2*line_height), "01/01/1990", fill='#000000', font=font_small)
    
    # PAN number (large and prominent)
    pan_y = y_start + 3*line_height + 20
    draw.text((50, pan_y), "PAN", fill='#000000', font=font_small)
    draw.text((200, pan_y), "ABCDE1234F", fill='#000000', font=font_large)
    
    # Signature section
    sig_y = pan_y + 60
    draw.text((50, sig_y), "Signature", fill='#000000', font=font_tiny)
    draw.rectangle([200, sig_y-5, 400, sig_y+25], outline='#666666', width=1)
    
    # Add some security features (hologram effect)
    draw.ellipse([650, 350, 780, 480], outline='#FFD700', width=2)
    draw.text((680, 405), "SECURITY", fill='#FFD700', font=font_tiny)
    
    return img

if __name__ == "__main__":
    # Create the sample PAN card
    sample_img = create_sample_pancard()
    
    # Save in the workspace root
    output_path = "sample-pan-card.jpg"
    sample_img.save(output_path, "JPEG", quality=95)
    
    print(f"Sample PAN card created: {output_path}")
    print("This image contains:")
    print("- Name: JOHN DOE")
    print("- INCOME TAX DEPARTMENT text")
    print("- PAN: ABCDE1234F")
    print("Use this for testing the OCR verification functionality.")