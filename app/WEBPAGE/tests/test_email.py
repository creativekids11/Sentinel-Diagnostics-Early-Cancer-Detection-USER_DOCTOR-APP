#!/usr/bin/env python3
"""
Email Configuration Test Script
Tests the Gmail SMTP configuration for Sentinel Diagnostics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Test the email sending functionality
if __name__ == "__main__":
    try:
        from app import send_email, send_otp_email
        
        print("🧪 Testing Gmail SMTP Configuration...")
        print("📧 SMTP Settings:")
        print("   Host: smtp.gmail.com")
        print("   Port: 587 (TLS)")
        print("   User: hackathonproject.victoriors@gmail.com")
        print()
        
        # Test basic email sending
        test_email = "hackathonproject.victoriors@gmail.com"  # Send to self for testing
        
        print(f"📤 Testing email send to: {test_email}")
        
        # Test OTP email
        result = send_otp_email(test_email, "123456", "Test User")
        
        if result:
            print("✅ Email sent successfully!")
            print("🔍 Check the inbox for the verification email.")
            print()
            print("📋 What to verify:")
            print("   • Email delivered to inbox (check spam folder too)")
            print("   • Subject line contains verification info")
            print("   • OTP code '123456' is clearly displayed")
            print("   • Professional formatting and branding")
        else:
            print("❌ Email sending failed!")
            print("🔧 Possible issues:")
            print("   • Gmail credentials might be incorrect")
            print("   • Gmail might require App Password instead of regular password")
            print("   • Network connectivity issues")
            print("   • Gmail security settings blocking the app")
        
    except Exception as e:
        print(f"❌ Error testing email configuration: {e}")
        print()
        print("🔧 Troubleshooting steps:")
        print("   1. Ensure Gmail account has 'Less secure app access' enabled")
        print("   2. Or use Gmail App Password instead of regular password")
        print("   3. Check if 2-factor authentication is enabled (requires App Password)")
        print("   4. Verify network connectivity")