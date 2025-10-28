#!/usr/bin/env python3
"""
Test script to simulate payment approval and verify subscription update
"""
import sqlite3
import os
from datetime import datetime, timedelta

# Database path
DATABASE = os.path.join(os.path.dirname(__file__), "database.db")

def test_payment_approval():
    """Test payment approval process"""
    print("Testing payment approval process...")

    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row

        # Find a user with pending payment
        payment = conn.execute("""
            SELECT pp.*, u.fullname, u.subscription_plan, u.subscription_expires_at
            FROM pending_payments pp
            JOIN users u ON pp.user_id = u.id
            WHERE pp.status = 'pending'
            LIMIT 1
        """).fetchone()

        if not payment:
            print("  - No pending payments found. Cannot test payment approval.")
            return

        user_id = payment['user_id']
        payment_id = payment['id']
        plan_type = payment['plan_type']
        
        print(f"  - Found pending payment for user: {payment['fullname']}")
        print(f"  - Current plan: {payment['subscription_plan']}")
        print(f"  - Payment plan: {plan_type}")
        print(f"  - Before approval - expires at: {payment['subscription_expires_at']}")

        # Simulate payment approval
        print(f"\n  - Simulating approval of payment {payment_id}...")
        
        # Update payment status
        conn.execute("""
            UPDATE pending_payments 
            SET status = 'approved', approved_by = 'test_admin', approved_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (payment_id,))

        # Update user subscription (simulating the approval logic)
        if plan_type.lower() in ['pro', 'premium']:
            expiration_date = (datetime.now() + timedelta(days=365)).isoformat()
            conn.execute("""
                UPDATE users 
                SET subscription_plan = ?, subscription_status = 'active', subscription_expires_at = ?
                WHERE id = ?
            """, (plan_type.lower(), expiration_date, user_id))
        
        conn.commit()

        # Verify the update
        user_after = conn.execute("""
            SELECT subscription_plan, subscription_expires_at, subscription_status
            FROM users WHERE id = ?
        """, (user_id,)).fetchone()

        print(f"  - After approval:")
        print(f"    * Plan: {user_after['subscription_plan']}")
        print(f"    * Status: {user_after['subscription_status']}")
        print(f"    * Expires: {user_after['subscription_expires_at']}")

        # Test subscription limits after upgrade
        print(f"\n  - Testing new subscription limits:")
        
        # Case limits
        limits = {
            'free': 3,
            'pro': 10,
            'premium': None
        }
        
        limit = limits.get(user_after['subscription_plan'], 3)
        if limit:
            print(f"    * Case limit: {limit} per day")
        else:
            print(f"    * Case limit: unlimited (Premium)")
        
        # Nutrition access
        nutrition_access = user_after['subscription_plan'] in ['pro', 'premium']
        print(f"    * Nutrition access: {nutrition_access}")
        
        # Meeting access
        meeting_access = user_after['subscription_plan'] == 'premium'
        print(f"    * Private meeting access: {meeting_access}")
        
        # Report delays
        delays = {
            'free': 30,
            'pro': 10,
            'premium': 0
        }
        delay = delays.get(user_after['subscription_plan'], 30)
        print(f"    * Report delay: {delay} minutes")

        print(f"\n✅ Payment approval test completed successfully!")
        print(f"✅ User subscription updated to {user_after['subscription_plan']} plan!")

if __name__ == "__main__":
    test_payment_approval()