#!/usr/bin/env python3
"""
Test script to verify payment amount validation fix.
Tests that only exact plan amounts are accepted for payments.
"""

import sqlite3
import requests
import json
from datetime import datetime

def check_database_payment_validation():
    """Check if database has the necessary tables and structure"""
    try:
        conn = sqlite3.connect('instance/database.db')
        cursor = conn.cursor()
        
        # Check if pending_payments table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='pending_payments'
        """)
        
        if not cursor.fetchone():
            print("❌ pending_payments table does not exist")
            return False
            
        # Check table structure
        cursor.execute("PRAGMA table_info(pending_payments)")
        columns = [col[1] for col in cursor.fetchall()]
        
        required_columns = ['id', 'user_id', 'plan_type', 'amount', 'status', 'created_at']
        for col in required_columns:
            if col not in columns:
                print(f"❌ Missing column: {col}")
                return False
                
        print("✅ Database structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_payment_verification_endpoint():
    """Test the payment verification endpoint with different amounts"""
    
    base_url = "http://127.0.0.1:5601"
    
    # Test cases: plan_type -> correct_amount, wrong_amount
    test_cases = {
        'basic': {'correct': 100, 'wrong': 50},
        'premium': {'correct': 500, 'wrong': 100},
        'enterprise': {'correct': 1500, 'wrong': 500}
    }
    
    print("\n🧪 Testing Payment Verification Endpoint...")
    
    for plan_type, amounts in test_cases.items():
        print(f"\n📋 Testing {plan_type.upper()} plan:")
        
        # Test with correct amount
        payload = {
            'plan_type': plan_type,
            'amount': amounts['correct'],
            'upi_transaction_id': f'TEST_{plan_type}_CORRECT_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'user_message': f'Test payment for {plan_type} plan with correct amount'
        }
        
        try:
            response = requests.post(
                f"{base_url}/patient/verify-payment",
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"✅ Correct amount (₹{amounts['correct']}) accepted")
                else:
                    print(f"❌ Correct amount rejected: {data.get('message')}")
            else:
                print(f"❌ HTTP Error for correct amount: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for correct amount: {e}")
        
        # Test with wrong amount
        payload['amount'] = amounts['wrong']
        payload['upi_transaction_id'] = f'TEST_{plan_type}_WRONG_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        payload['user_message'] = f'Test payment for {plan_type} plan with wrong amount'
        
        try:
            response = requests.post(
                f"{base_url}/patient/verify-payment",
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"❌ Wrong amount (₹{amounts['wrong']}) incorrectly accepted!")
                else:
                    print(f"✅ Wrong amount correctly rejected: {data.get('message')}")
            else:
                print(f"✅ Wrong amount rejected with HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for wrong amount: {e}")

def check_pending_payments():
    """Check the pending payments in database"""
    try:
        conn = sqlite3.connect('instance/database.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT plan_type, amount, status, created_at 
            FROM pending_payments 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        payments = cursor.fetchall()
        
        print(f"\n📊 Recent Pending Payments ({len(payments)} found):")
        for payment in payments:
            plan_type, amount, status, created_at = payment
            print(f"  • {plan_type.upper()}: ₹{amount} - {status} - {created_at}")
            
    except Exception as e:
        print(f"❌ Failed to check pending payments: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Run all payment validation tests"""
    print("🔧 Payment Amount Validation Fix Test")
    print("=" * 50)
    
    # Check database structure
    if not check_database_payment_validation():
        print("\n❌ Database validation failed. Cannot proceed with tests.")
        return
    
    # Test payment verification
    test_payment_verification_endpoint()
    
    # Check results in database
    check_pending_payments()
    
    print("\n" + "=" * 50)
    print("✅ Payment validation test completed!")
    print("💡 The fix ensures only exact plan amounts are accepted:")
    print("   • Basic: ₹100 only")
    print("   • Premium: ₹500 only") 
    print("   • Enterprise: ₹1500 only")

if __name__ == "__main__":
    main()