#!/usr/bin/env python3
"""
Test script for PAN card verification
"""

from utils.pancard_verification import verify_pancard, create_sample_pancard
import os

def test_pan_verification():
    print("Testing PAN card verification...")

    # Create sample PAN card
    sample_path = create_sample_pancard()
    if sample_path and os.path.exists(sample_path):
        print(f'✓ Sample PAN card created at: {sample_path}')

        # First, let's see what text is extracted
        from utils.pancard_verification import extract_text_from_pancard
        extracted_text = extract_text_from_pancard(sample_path)
        print(f"\nExtracted text:\n'{extracted_text}'")

        # Test verification with matching name
        print("\nTesting with matching name 'JOHN DOE':")
        result = verify_pancard(sample_path, 'JOHN DOE')
        print(f'Result: {result[0]}')
        print(f'Message: {result[1]}')

        # Test verification with non-matching name
        print("\nTesting with non-matching name 'JANE SMITH':")
        result = verify_pancard(sample_path, 'JANE SMITH')
        print(f'Result: {result[0]}')
        print(f'Message: {result[1]}')

        # Clean up
        os.remove(sample_path)
        print(f'\n✓ Cleaned up sample file: {sample_path}')
    else:
        print('✗ Failed to create sample PAN card')

if __name__ == "__main__":
    test_pan_verification()