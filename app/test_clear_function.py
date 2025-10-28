import sqlite3
import sys
import os
sys.path.append('WEBPAGE')

from app import app, clear_all_notifications

# Check current notifications
with app.app_context():
    # Check current notifications - use the same database as the app
    db_path = os.path.join('WEBPAGE', 'database.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('SELECT id, user_id, title, message FROM notifications')
    notifications = cur.fetchall()
    print(f'Notifications before clear: {len(notifications)}')
    for n in notifications:
        print(f'  ID: {n["id"]}, User: {n["user_id"]}, Title: {n["title"]}')

    # Test clear function
    if notifications:
        user_id = notifications[0]['user_id']
        result = clear_all_notifications(user_id)
        print(f'Clear result: {result}')

        # Check notifications after
        cur.execute('SELECT id, user_id, title, message FROM notifications WHERE user_id = ?', (user_id,))
        notifications_after = cur.fetchall()
        print(f'Notifications after clear for user {user_id}: {len(notifications_after)}')
    else:
        print('No notifications to test with')

    conn.close()