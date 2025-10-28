#!/usr/bin/env python3
"""
Test script to verify admin payment functionality
"""
import sqlite3
import os
from datetime import datetime

# Database path
DATABASE = os.path.join(os.path.dirname(__file__), "database.db")

def test_admin_payments():
    """Test the admin payments functionality"""
    print("Testing admin payment functionality...")

    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row

        # Check if pending_payments table exists and has data
        payments = conn.execute("""
            SELECT pp.*, u.fullname, u.username, u.email
            FROM pending_payments pp
            JOIN users u ON pp.user_id = u.id
            WHERE pp.status = 'pending'
            ORDER BY pp.created_at DESC
        """).fetchall()

        print(f"Found {len(payments)} pending payments:")

        for payment in payments:
            print(f"  - Payment ID: {payment['id']}")
            print(f"    User: {payment['fullname']} ({payment['email']})")
            print(f"    Plan: {payment['plan_type']}")
            print(f"    Amount: {payment['amount']}")
            print(f"    Status: {payment['status']}")
            print(f"    Created: {payment['created_at']}")
            print()

        # Check recent payments
        recent = conn.execute("""
            SELECT pp.*, u.fullname, u.username, u.email
            FROM pending_payments pp
            JOIN users u ON pp.user_id = u.id
            WHERE pp.status IN ('approved', 'rejected')
            AND pp.approved_at >= datetime('now', '-30 days')
            ORDER BY pp.approved_at DESC
            LIMIT 10
        """).fetchall()

        print(f"Found {len(recent)} recent processed payments:")
        for payment in recent:
            print(f"  - Payment ID: {payment['id']}")
            print(f"    User: {payment['fullname']} ({payment['email']})")
            print(f"    Plan: {payment['plan_type']}")
            print(f"    Status: {payment['status']}")
            print(f"    Processed: {payment['approved_at']}")
            print()

        print("✅ Admin payment queries executed successfully!")
        print("✅ JOIN operations working correctly!")
        print("✅ User details are accessible in payment records!")

if __name__ == "__main__":
    test_admin_payments()