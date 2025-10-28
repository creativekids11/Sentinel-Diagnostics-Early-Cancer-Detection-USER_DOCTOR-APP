import sqlite3
import os

DATABASE = 'instance/app.db'
if os.path.exists(DATABASE):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    # Check all tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    print('Tables in database:')
    for table in tables:
        print(' ', table[0])

    # Check if notifications table exists
    if any('notifications' in table[0] for table in tables):
        print('\nNotifications table exists, checking contents...')
        cur.execute('SELECT COUNT(*) as count FROM notifications')
        result = cur.fetchone()
        print('Total notifications in database:', result[0])

        # Check recent notifications
        cur.execute('SELECT id, user_id, title, message, read_status, created_at FROM notifications ORDER BY created_at DESC LIMIT 5')
        notifications = cur.fetchall()
        print('Recent notifications:')
        for n in notifications:
            print(f'  ID: {n[0]}, User: {n[1]}, Title: {n[2]}, Read: {n[3]}, Created: {n[4]}')
    else:
        print('\nNotifications table does not exist!')

    conn.close()
else:
    print('Database not found')