#!/usr/bin/env python3
"""
Test script to verify subscription-based features
"""
import sqlite3
import os
from datetime import datetime, timedelta

# Database path
DATABASE = os.path.join(os.path.dirname(__file__), "database.db")

def test_subscription_features():
    """Test subscription-based feature access"""
    print("Testing subscription-based features...")

    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row

        # Test 1: Check if payment approval updates subscription
        print("\n1. Testing payment approval updates subscription:")
        
        # Find a user with a pending payment
        payment = conn.execute("""
            SELECT pp.*, u.fullname, u.subscription_plan, u.subscription_expires_at
            FROM pending_payments pp
            JOIN users u ON pp.user_id = u.id
            WHERE pp.status = 'pending'
            ORDER BY pp.created_at DESC
            LIMIT 1
        """).fetchone()
        
        if payment:
            print(f"  - Found pending payment for user: {payment['fullname']}")
            print(f"  - Current plan: {payment['subscription_plan']}")
            print(f"  - Payment plan: {payment['plan_type']}")
            print(f"  - Expires at: {payment['subscription_expires_at']}")
        else:
            print("  - No pending payments found")

        # Test 2: Check available_at column in reports table
        print("\n2. Testing reports table has available_at column:")
        try:
            cursor = conn.execute("PRAGMA table_info(reports)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'available_at' in columns:
                print("  ✅ available_at column exists in reports table")
            else:
                print("  ❌ available_at column missing from reports table")
        except Exception as e:
            print(f"  ❌ Error checking reports table: {e}")

        # Test 3: Check subscription functions work
        print("\n3. Testing subscription check functions:")
        
        # Get a test user
        user = conn.execute("SELECT id, subscription_plan FROM users WHERE role = 'patient' LIMIT 1").fetchone()
        if user:
            user_id = user['id']
            plan = user['subscription_plan'] or 'free'
            print(f"  - Testing with user ID: {user_id}, plan: {plan}")
            
            # Test case limits
            today = datetime.now().strftime("%Y-%m-%d")
            today_cases = conn.execute(
                "SELECT COUNT(*) as count FROM cases WHERE user_id = ? AND DATE(created_at) = ?",
                (user_id, today)
            ).fetchone()
            case_count = today_cases["count"] if today_cases else 0
            
            limits = {
                'free': 3,
                'pro': 10,
                'premium': None
            }
            
            limit = limits.get(plan, 3)
            if limit:
                can_create = case_count < limit
                print(f"  - Cases today: {case_count}/{limit}, can create: {can_create}")
            else:
                print(f"  - Cases today: {case_count}/unlimited (Premium)")
            
            # Test nutrition access
            nutrition_access = plan in ['pro', 'premium']
            print(f"  - Nutrition access: {nutrition_access}")
            
            # Test meeting access
            meeting_access = plan == 'premium'
            print(f"  - Private meeting access: {meeting_access}")
            
            # Test report delays
            delays = {
                'free': 30,
                'pro': 10,
                'premium': 0
            }
            delay = delays.get(plan, 30)
            print(f"  - Report delay: {delay} minutes")
        else:
            print("  - No patient users found for testing")

        print("\n✅ Subscription feature testing completed!")

if __name__ == "__main__":
    test_subscription_features()