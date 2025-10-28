import sys
import os
sys.path.append('.')

# Import the Flask app and functions
from app import app, clear_all_notifications
import sqlite3

# Test the clear_all_notifications function
print("Testing clear_all_notifications function...")

with app.app_context():
    # Check notifications before clearing
    conn = sqlite3.connect('instance/app.db')
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) as count FROM notifications')
    result = cur.fetchone()
    print(f'Notifications before clearing: {result[0]}')

    # Test clearing for user_id = 1
    success = clear_all_notifications(1)
    print(f'Clear function returned: {success}')

    # Check notifications after clearing
    cur.execute('SELECT COUNT(*) as count FROM notifications WHERE user_id = 1')
    result = cur.fetchone()
    print(f'Notifications for user 1 after clearing: {result[0]}')

    # Check all notifications
    cur.execute('SELECT COUNT(*) as count FROM notifications')
    result = cur.fetchone()
    print(f'Total notifications after clearing: {result[0]}')

    conn.close()

print("Test completed!")