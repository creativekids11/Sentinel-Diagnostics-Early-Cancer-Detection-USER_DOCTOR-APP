#!/usr/bin/env python3
"""
Create a test patient account bypassing email verification for testing
"""

import sys
import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime

# Test patient data
test_patients = [
    {
        'fullname': 'John Patient Smith',
        'username': 'patient_john',
        'password': 'Patient123!',
        'email': 'john.patient@example.com',
        'phone': '9876543210',
        'role': 'patient'
    },
    {
        'fullname': 'Sarah Patient Wilson',
        'username': 'patient_sarah',
        'password': 'Sarah123!',
        'email': 'sarah.patient@example.com',
        'phone': '9876543211',
        'role': 'patient'
    }
]

def create_test_patients():
    """Create test patient accounts directly in the database"""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    created_count = 0
    
    for patient_data in test_patients:
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE username=? OR email=?', 
                      (patient_data['username'], patient_data['email']))
        existing = cursor.fetchone()
        
        if existing:
            print(f"User {patient_data['username']} already exists, skipping...")
            continue
        
        # Hash password
        hashed_password = generate_password_hash(patient_data['password'])
        
        # Create user with verified email and active status
        cursor.execute('''
            INSERT INTO users (
                fullname, username, password, role, phone, email,
                status, email_verified, is_active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_data['fullname'],
            patient_data['username'], 
            hashed_password,
            patient_data['role'],
            patient_data['phone'],
            patient_data['email'],
            'approved',  # status
            1,           # email_verified
            1,           # is_active
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        
        created_count += 1
        print(f"✅ Created patient account: {patient_data['username']} / {patient_data['password']}")
    
    conn.commit()
    conn.close()
    
    print(f"\n🎉 Successfully created {created_count} test patient accounts!")
    print("\nYou can now log in with:")
    for patient in test_patients:
        print(f"  Username: {patient['username']}")
        print(f"  Password: {patient['password']}")
        print()

if __name__ == "__main__":
    create_test_patients()