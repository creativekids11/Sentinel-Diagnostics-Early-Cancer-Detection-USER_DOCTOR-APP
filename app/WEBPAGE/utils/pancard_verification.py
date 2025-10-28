"""
PAN Card Verification Utility using OCR
Extracts name and validates INCOME TAX DEPARTMENT text from PAN card images
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
from difflib import SequenceMatcher

# Configuration for debugging
ENABLE_PANCARD_VERIFICATION = True  # Set to False to disable verification

def set_verification_enabled(enabled: bool):
    """Set global verification enabled state for debugging"""
    global ENABLE_PANCARD_VERIFICATION
    ENABLE_PANCARD_VERIFICATION = enabled

def is_verification_enabled():
    """Check if verification is enabled"""
    return ENABLE_PANCARD_VERIFICATION

def preprocess_image(image_path):
    """
    Preprocess the PAN card image for better OCR accuracy
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(filtered)

        # Apply threshold to get better contrast
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Optional: Resize if image is too small
        height, width = cleaned.shape
        if height < 300 or width < 400:
            scale_factor = max(300 / height, 400 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        return cleaned
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def extract_text_from_pancard(image_path):
    """
    Extract text from PAN card image using OCR
    """
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        # Convert to PIL Image for pytesseract
        pil_img = Image.fromarray(processed_img)

        # Use pytesseract to extract text with better config for PAN cards
        # PAN cards typically have uppercase text, so we'll optimize for that
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,:()-/'
        extracted_text = pytesseract.image_to_string(pil_img, config=custom_config, lang='eng')

        return extracted_text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PAN card: {str(e)}")

def validate_income_tax_department(extracted_text):
    """
    Check if the extracted text contains 'INCOME TAX DEPARTMENT' or similar variations
    """
    # Clean the text and make it uppercase
    clean_text = re.sub(r'[^A-Za-z\s]', '', extracted_text).upper()
    
    # Possible variations of "INCOME TAX DEPARTMENT"
    variations = [
        "INCOME TAX DEPARTMENT",
        "INCOMETAX DEPARTMENT", 
        "INCOME TAX DEPT",
        "INCOMETAX DEPT",
        "GOVT OF INDIA",
        "GOVERNMENT OF INDIA"
    ]
    
    # Check for exact matches
    for variation in variations:
        if variation in clean_text:
            return True
    
    # Check for fuzzy matches (in case of OCR errors)
    words = clean_text.split()
    for variation in variations:
        variation_words = variation.split()
        if len(variation_words) <= len(words):
            # Check if we can find a sequence with high similarity
            for i in range(len(words) - len(variation_words) + 1):
                sequence = ' '.join(words[i:i+len(variation_words)])
                similarity = SequenceMatcher(None, variation, sequence).ratio()
                if similarity > 0.7:  # 70% similarity threshold
                    return True
    
    return False

def extract_name_from_pancard(extracted_text):
    """
    Extract the name from PAN card text with improved OCR error handling
    """
    lines = extracted_text.strip().split('\n')

    extracted_names = []

    # Look for patterns after "Name" or "NAME"
    name_indicators = ['name', 'nara', 'nama', 'nam']  # OCR might misread "name"

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Check if this line contains a name indicator
        has_name_indicator = any(indicator in line_lower for indicator in name_indicators)

        if has_name_indicator:
            # Extract text after the name indicator
            for indicator in name_indicators:
                if indicator in line_lower:
                    # Split by the indicator and take everything after it
                    parts = line_lower.split(indicator, 1)
                    if len(parts) > 1:
                        name_part = parts[1].strip()
                        # Clean up the name
                        name_part = re.sub(r'[^a-zA-Z\s]', '', name_part).strip()
                        if len(name_part) > 2 and name_part.replace(' ', '').isalpha():
                            extracted_names.append(name_part)
                    break

        # Also check the next line if current line has name indicator
        if has_name_indicator and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            next_line_clean = re.sub(r'[^a-zA-Z\s]', '', next_line).strip()
            if len(next_line_clean) > 2 and next_line_clean.replace(' ', '').isalpha():
                # Skip if it looks like other PAN card fields
                skip_keywords = ['father', 'birth', 'pan', 'date', 'govt', 'income', 'tax', 'department']
                if not any(keyword in next_line_clean.lower() for keyword in skip_keywords):
                    extracted_names.append(next_line_clean)

    # Fallback: Look for any line that looks like a name (2+ words, alphabetic only)
    for line in lines:
        line_clean = re.sub(r'[^a-zA-Z\s]', '', line).strip()
        if (len(line_clean) > 5 and
            line_clean.replace(' ', '').isalpha() and
            len(line_clean.split()) >= 2):

            # Skip common PAN card text
            skip_keywords = ['income', 'tax', 'department', 'govt', 'india', 'permanent', 'account', 'number', 'father', 'birth', 'pan', 'date']
            if not any(keyword in line_clean.lower() for keyword in skip_keywords):
                extracted_names.append(line_clean)

    # Remove duplicates and clean names
    cleaned_names = []
    for name in extracted_names:
        name = ' '.join(name.split())  # Normalize spaces
        name = name.title()  # Title case for consistency
        if name not in cleaned_names and len(name) > 2:
            cleaned_names.append(name)

    return cleaned_names

def normalize_name(name):
    """
    Normalize name for comparison (remove extra spaces, convert to lowercase)
    """
    return ' '.join(name.lower().split())

def _token_similarity(a, b):
    """Return fuzzy similarity between two tokens (0..1)."""
    return SequenceMatcher(None, a, b).ratio()


def _token_match_score(pancard_tokens, form_tokens, token_threshold=0.75):
    """
    Given lists of tokens from pancard and form, return fraction of form tokens
    that have a good match in pancard tokens.
    """
    if not form_tokens:
        return 0.0
    match_count = 0
    for ft in form_tokens:
        ft_clean = ft.strip()
        if not ft_clean:
            continue
        best = 0.0
        for pt in pancard_tokens:
            s = _token_similarity(ft_clean, pt)
            if s > best:
                best = s
            # also accept prefix/substring matches as strong signals
            if pt.startswith(ft_clean) or ft_clean in pt:
                best = max(best, 0.9)
                break
        if best >= token_threshold:
            match_count += 1

    return match_count / max(1, len(form_tokens))


def compare_names(pancard_name, form_name, threshold=0.7):
    """
    Compare extracted PAN card name with form name using a mix of strategies:
    - normalized full-string fuzzy matching
    - token-level fuzzy matching (preferred)
    Returns True if names match with given similarity threshold
    """
    normalized_pancard = re.sub(r'[^a-z0-9\s]', ' ', normalize_name(pancard_name))
    normalized_form = re.sub(r'[^a-z0-9\s]', ' ', normalize_name(form_name))

    # Direct match
    if normalized_pancard == normalized_form:
        return True

    # Tokenized matching - more robust to OCR concatenation and ordering
    pancard_tokens = [t for t in normalized_pancard.split() if len(t) > 1]
    form_tokens = [t for t in normalized_form.split() if len(t) > 1]

    # If form tokens are a subset (token-wise) of pancard tokens
    token_score = _token_match_score(pancard_tokens, form_tokens)
    if token_score >= 0.66:  # two out of three tokens matched or ~66%
        return True

    # If pancard contains form as substring or vice versa (after cleanup)
    if normalized_form in normalized_pancard or normalized_pancard in normalized_form:
        return True

    # Fallback: overall fuzzy similarity
    similarity = SequenceMatcher(None, normalized_pancard, normalized_form).ratio()
    return similarity >= threshold

def verify_pancard(image_path, full_name):
    """
    Main function to verify PAN card
    Returns (is_valid, message, extracted_data)
    """
    if not ENABLE_PANCARD_VERIFICATION:
        return True, "PAN card verification disabled for debugging", {}
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "PAN card image file not found", {}
        
        # Extract text from image
        extracted_text = extract_text_from_pancard(image_path)
        
        if not extracted_text:
            return False, "Could not extract text from PAN card image", {}
        
        # Validate INCOME TAX DEPARTMENT
        has_income_tax_dept = validate_income_tax_department(extracted_text)
        if not has_income_tax_dept:
            return False, "PAN card does not appear to be genuine - 'INCOME TAX DEPARTMENT' text not found", {
                'extracted_text': extracted_text
            }
        
        # Extract names from PAN card
        extracted_names = extract_name_from_pancard(extracted_text)
        
        if not extracted_names:
            # Try fallback: look for candidate name-like sequences in full text
            # e.g. uppercase sequences separated by spaces and with 2+ words
            fallback_candidates = []
            for line in extracted_text.split('\n'):
                clean = re.sub(r'[^A-Za-z\s]', ' ', line).strip()
                if len(clean) > 5 and clean.replace(' ', '').isalpha() and len(clean.split()) >= 2:
                    fallback_candidates.append(clean)
            # If fallback candidates exist, proceed with them
            if fallback_candidates:
                extracted_names = fallback_candidates
            else:
                return False, "Could not extract name from PAN card", {
                    'extracted_text': extracted_text
                }
        
        # Check if any extracted name matches the form name
        name_match_found = False
        best_match = ""
        best_similarity = 0
        
        for extracted_name in extracted_names:
            if compare_names(extracted_name, full_name):
                name_match_found = True
                best_match = extracted_name
                best_similarity = SequenceMatcher(None, normalize_name(extracted_name), normalize_name(full_name)).ratio()
                break
            else:
                similarity = SequenceMatcher(None, normalize_name(extracted_name), normalize_name(full_name)).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = extracted_name
        
        if not name_match_found:
            return False, f"Name mismatch: PAN card name '{best_match}' does not match provided name '{full_name}'", {
                'extracted_text': extracted_text,
                'extracted_names': extracted_names,
                'best_match': best_match,
                'similarity': best_similarity
            }
        
        return True, "PAN card verification successful", {
            'extracted_text': extracted_text,
            'extracted_names': extracted_names,
            'matched_name': best_match,
            'similarity': best_similarity
        }
        
    except Exception as e:
        return False, f"Error verifying PAN card: {str(e)}", {}

# Utility function to create a sample PAN card for testing
def create_sample_pancard():
    """
    Create a sample PAN card image for testing purposes
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple sample PAN card image
        width, height = 800, 500
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_medium = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Add sample text with better OCR readability
        draw.text((50, 50), "INCOME TAX DEPARTMENT", fill='black', font=font_large)
        draw.text((50, 80), "GOVT. OF INDIA", fill='black', font=font_medium)
        draw.text((50, 120), "PERMANENT ACCOUNT NUMBER CARD", fill='black', font=font_medium)

        draw.text((50, 180), "Name:", fill='black', font=font_medium)
        draw.text((150, 180), "JOHN DOE", fill='black', font=font_medium)

        draw.text((50, 220), "Father's Name:", fill='black', font=font_small)
        draw.text((200, 220), "JANE DOE", fill='black', font=font_small)

        draw.text((50, 250), "Date of Birth:", fill='black', font=font_small)
        draw.text((180, 250), "01/01/1990", fill='black', font=font_small)

        draw.text((50, 290), "PAN:", fill='black', font=font_large)
        draw.text((120, 290), "ABCDE1234F", fill='black', font=font_large)
        
        # Save the sample image
        sample_path = "sample-pan-card.jpg"
        img.save(sample_path)
        return sample_path
        
    except Exception as e:
        print(f"Error creating sample PAN card: {str(e)}")
        return None