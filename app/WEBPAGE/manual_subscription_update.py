#!/usr/bin/env python3
"""
Manually update user subscription to test if the system recognizes it
"""
import sqlite3
import os
from datetime import datetime, timedelta

# Database path
DATABASE = os.path.join(os.path.dirname(__file__), "database.db")

def update_user_subscription():
    """Manually update user subscription"""
    print("Manually updating user subscription...")

    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row

        # Find user with approved payment
        user = conn.execute("""
            SELECT u.*, pp.plan_type
            FROM users u
            JOIN pending_payments pp ON u.id = pp.user_id
            WHERE pp.status = 'approved' AND u.subscription_plan = 'free'
            LIMIT 1
        """).fetchone()

        if not user:
            print("  - No user found with approved payment and free plan")
            return

        user_id = user['id']
        plan_type = user['plan_type']
        
        print(f"  - Updating user: {user['fullname']}")
        print(f"  - Current plan: {user['subscription_plan']}")
        print(f"  - New plan: {plan_type}")

        # Update subscription
        expiration_date = (datetime.now() + timedelta(days=365)).isoformat()
        conn.execute("""
            UPDATE users 
            SET subscription_plan = ?, subscription_status = 'active', subscription_expires_at = ?
            WHERE id = ?
        """, (plan_type.lower(), expiration_date, user_id))
        
        conn.commit()

        # Verify update
        updated_user = conn.execute("""
            SELECT subscription_plan, subscription_expires_at, subscription_status
            FROM users WHERE id = ?
        """, (user_id,)).fetchone()

        print(f"  - Updated successfully!")
        print(f"    * Plan: {updated_user['subscription_plan']}")
        print(f"    * Status: {updated_user['subscription_status']}")
        print(f"    * Expires: {updated_user['subscription_expires_at']}")

        print(f"\n✅ User subscription updated successfully!")

if __name__ == "__main__":
    update_user_subscription()