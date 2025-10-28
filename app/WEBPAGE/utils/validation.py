"""
Validation utilities for user signup
Includes password strength validation and username character restrictions
"""

import re
from typing import Tuple, List

# Restricted characters for username
RESTRICTED_USERNAME_CHARS = [',', "'", '"', '/', '\\', '*']

def validate_password_strength(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password strength according to security requirements
    
    Requirements:
    - At least 8 characters long
    - At least one uppercase letter
    - At least one lowercase letter  
    - At least one digit
    - At least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)
    - No common passwords
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check length
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    
    # Check for uppercase letter
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    # Check for lowercase letter
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    # Check for digit
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one digit")
    
    # Check for special character
    if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
        errors.append("Password must contain at least one special character (!@#$%^&*()_+-=[]{}|;:,.<>?)")
    
    # Check against common passwords
    common_passwords = [
        'password', '123456', '12345678', 'qwerty', 'abc123', 
        'password123', 'admin', 'letmein', 'welcome', 'monkey',
        '1234567890', 'password1', '123456789', 'qwerty123'
    ]
    
    if password.lower() in common_passwords:
        errors.append("Password is too common, please choose a more secure password")
    
    # Check for sequential patterns
    if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
        errors.append("Password should not contain sequential numbers")
    
    if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
        errors.append("Password should not contain sequential letters")
    
    # Check for repetitive patterns
    if re.search(r'(.)\1{2,}', password):
        errors.append("Password should not contain repetitive characters (e.g., aaa, 111)")
    
    return len(errors) == 0, errors

def validate_username(username: str) -> Tuple[bool, List[str]]:
    """
    Validate username according to character restrictions
    
    Restrictions:
    - Cannot contain: , ' " / \\ *
    - Must be at least 3 characters long
    - Must start with a letter
    - Can only contain letters, numbers, underscore, and hyphen
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check length
    if len(username) < 3:
        errors.append("Username must be at least 3 characters long")
    
    if len(username) > 30:
        errors.append("Username must be no more than 30 characters long")
    
    # Check for restricted characters
    for char in RESTRICTED_USERNAME_CHARS:
        if char in username:
            errors.append(f"Username cannot contain the character: {char}")
    
    # Check if starts with letter
    if username and not username[0].isalpha():
        errors.append("Username must start with a letter")
    
    # Check for valid characters only (letters, numbers, underscore, hyphen)
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', username):
        errors.append("Username can only contain letters, numbers, underscore (_), and hyphen (-)")
    
    # Check for reserved usernames
    reserved_usernames = [
        'admin', 'administrator', 'root', 'user', 'test', 'guest', 
        'doctor', 'patient', 'null', 'undefined', 'api', 'www',
        'mail', 'email', 'support', 'help', 'info', 'contact'
    ]
    
    if username.lower() in reserved_usernames:
        errors.append("This username is reserved and cannot be used")
    
    return len(errors) == 0, errors

def validate_full_name(full_name: str) -> Tuple[bool, List[str]]:
    """
    Validate full name format
    
    Requirements:
    - At least 2 words (first and last name)
    - Each word at least 2 characters
    - Only letters and spaces allowed
    - No leading/trailing spaces
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    if not full_name or not full_name.strip():
        errors.append("Full name is required")
        return False, errors
    
    # Remove extra spaces
    full_name = ' '.join(full_name.split())
    
    # Check for minimum words
    words = full_name.split()
    if len(words) < 2:
        errors.append("Full name must contain at least first and last name")
    
    # Check each word
    for word in words:
        if len(word) < 2:
            errors.append("Each part of the name must be at least 2 characters long")
            break
        
        if not word.isalpha():
            errors.append("Full name can only contain letters and spaces")
            break
    
    # Check length
    if len(full_name) > 100:
        errors.append("Full name is too long (maximum 100 characters)")
    
    return len(errors) == 0, errors

def validate_phone(phone: str) -> Tuple[bool, List[str]]:
    """
    Validate phone number format
    
    Requirements:
    - 10 digits for Indian phone numbers
    - Can start with +91 country code
    - Only digits, +, -, (, ), and spaces allowed
    
    Returns:
        tuple: (is_valid, list_of_errors)  
    """
    errors = []
    
    if not phone or not phone.strip():
        errors.append("Phone number is required")
        return False, errors
    
    # Clean phone number
    clean_phone = re.sub(r'[^\d+]', '', phone)
    
    # Check for Indian phone number patterns
    if clean_phone.startswith('+91'):
        # Remove country code
        clean_phone = clean_phone[3:]
    elif clean_phone.startswith('91') and len(clean_phone) == 12:
        # Remove country code without +
        clean_phone = clean_phone[2:]
    
    # Check if it's a valid 10-digit number
    if not re.match(r'^\d{10}$', clean_phone):
        errors.append("Phone number must be a valid 10-digit Indian mobile number")
    
    # Check if it starts with valid mobile prefixes
    if clean_phone and not clean_phone[0] in '6789':
        errors.append("Phone number must start with 6, 7, 8, or 9")
    
    return len(errors) == 0, errors

def validate_email(email: str) -> Tuple[bool, List[str]]:
    """
    Validate email format
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    if not email or not email.strip():
        errors.append("Email address is required")
        return False, errors
    
    # Basic email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        errors.append("Please enter a valid email address")
    
    # Check length
    if len(email) > 254:
        errors.append("Email address is too long")
    
    return len(errors) == 0, errors

def get_password_strength_score(password: str) -> int:
    """
    Calculate password strength score (0-100)
    Used for frontend password strength indicator
    
    Returns:
        int: Score between 0-100
    """
    score = 0
    
    # Length score (0-25 points)
    if len(password) >= 8:
        score += 10
    if len(password) >= 12:
        score += 10
    if len(password) >= 16:
        score += 5
    
    # Character variety (0-40 points)
    if re.search(r'[a-z]', password):
        score += 10
    if re.search(r'[A-Z]', password):
        score += 10
    if re.search(r'\d', password):
        score += 10
    if re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password):
        score += 10
    
    # Complexity bonus (0-35 points)
    # No common patterns
    if not re.search(r'(012|123|234|345|456|567|678|789|890)', password):
        score += 10
    if not re.search(r'(abc|bcd|cde|def|efg|fgh)', password.lower()):
        score += 10
    if not re.search(r'(.)\1{2,}', password):
        score += 10
    if len(set(password)) / len(password) > 0.7:  # Character diversity
        score += 5
    
    return min(score, 100)