import sqlite3
import os
from datetime import datetime

DATABASE = 'instance/app.db'
if os.path.exists(DATABASE):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()

    # Try to create the notifications table manually
    try:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT NOT NULL,
                read_status INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                related_id INTEGER,
                related_type TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()
        print('Notifications table created successfully')

        # Check if table exists now
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'")
        if cur.fetchone():
            print('Notifications table confirmed to exist')

            # Insert a test notification
            cur.execute('''
                INSERT INTO notifications (user_id, title, message, type, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (1, 'Test Notification', 'This is a test notification', 'test', datetime.now().isoformat()))
            conn.commit()
            print('Test notification inserted')

            # Check notifications
            cur.execute('SELECT COUNT(*) as count FROM notifications')
            result = cur.fetchone()
            print('Total notifications after test insert:', result[0])

            # Check recent notifications
            cur.execute('SELECT id, user_id, title, message, read_status, created_at FROM notifications ORDER BY created_at DESC LIMIT 5')
            notifications = cur.fetchall()
            print('Recent notifications:')
            for n in notifications:
                print(f'  ID: {n[0]}, User: {n[1]}, Title: {n[2]}, Read: {n[3]}, Created: {n[4]}')

        else:
            print('Notifications table still does not exist')

    except Exception as e:
        print('Error creating notifications table:', str(e))

    conn.close()
else:
    print('Database not found')