#!/usr/bin/env python3
"""
Check current user subscription statuses
"""
import sqlite3
import os

# Database path
DATABASE = os.path.join(os.path.dirname(__file__), "database.db")

def check_user_subscriptions():
    """Check current user subscription statuses"""
    print("Current user subscription statuses:")

    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row

        users = conn.execute("""
            SELECT id, fullname, email, subscription_plan, subscription_expires_at, subscription_status
            FROM users WHERE role = 'patient'
        """).fetchall()

        for user in users:
            print(f"  - User: {user['fullname']} ({user['email']})")
            print(f"    Plan: {user['subscription_plan']}")
            print(f"    Status: {user['subscription_status']}")
            print(f"    Expires: {user['subscription_expires_at']}")
            print()

        # Check payment statuses
        payments = conn.execute("""
            SELECT pp.*, u.fullname
            FROM pending_payments pp
            JOIN users u ON pp.user_id = u.id
            ORDER BY pp.created_at DESC
            LIMIT 5
        """).fetchall()

        print("Recent payments:")
        for payment in payments:
            print(f"  - Payment ID {payment['id']}: {payment['fullname']} - {payment['plan_type']} - {payment['status']}")

if __name__ == "__main__":
    check_user_subscriptions()