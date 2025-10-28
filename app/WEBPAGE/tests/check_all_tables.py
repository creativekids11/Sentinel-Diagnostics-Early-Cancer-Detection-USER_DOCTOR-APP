#!/usr/bin/env python3
"""
Check Database Tables
"""

import sqlite3

def check_all_tables():
    """Check what tables exist in the database"""
    db = sqlite3.connect('instance/users.db')
    cursor = db.cursor()
    
    print("=== All Tables in Database ===")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        # Get table structure
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")
    
    db.close()

if __name__ == "__main__":
    check_all_tables()