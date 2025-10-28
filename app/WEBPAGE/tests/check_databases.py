import sqlite3
import os

# Check database.db
print("=== Checking database.db ===")
if os.path.exists('database.db'):
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    print('Tables in database.db:')
    for t in tables:
        print(f'  {t[0]}')

    has_notifications = any('notifications' in t[0] for t in tables)
    print('Has notifications table:', has_notifications)

    if has_notifications:
        cur.execute('SELECT COUNT(*) FROM notifications')
        result = cur.fetchone()
        print('Notifications count:', result[0])

        cur.execute('SELECT id, user_id, title, type, read_status, created_at FROM notifications ORDER BY created_at DESC LIMIT 5')
        notifications = cur.fetchall()
        print('Recent notifications:')
        for n in notifications:
            print(f'  ID: {n[0]}, User: {n[1]}, Title: {n[2]}, Type: {n[3]}, Read: {n[4]}')
    conn.close()
else:
    print('database.db not found')

print("\n=== Checking instance/app.db ===")
if os.path.exists('instance/app.db'):
    conn = sqlite3.connect('instance/app.db')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cur.fetchall()
    print('Tables in instance/app.db:')
    for t in tables:
        print(f'  {t[0]}')

    has_notifications = any('notifications' in t[0] for t in tables)
    print('Has notifications table:', has_notifications)

    if has_notifications:
        cur.execute('SELECT COUNT(*) FROM notifications')
        result = cur.fetchone()
        print('Notifications count:', result[0])

        cur.execute('SELECT id, user_id, title, type, read_status, created_at FROM notifications ORDER BY created_at DESC LIMIT 5')
        notifications = cur.fetchall()
        print('Recent notifications:')
        for n in notifications:
            print(f'  ID: {n[0]}, User: {n[1]}, Title: {n[2]}, Type: {n[3]}, Read: {n[4]}')
    conn.close()
else:
    print('instance/app.db not found')