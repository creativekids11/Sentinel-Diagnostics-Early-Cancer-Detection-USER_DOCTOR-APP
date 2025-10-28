import sqlite3
import os

print('Current directory:', os.getcwd())
print('Database file exists:', os.path.exists('database.db'))

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

try:
    cursor.execute('SELECT id, username, fullname, role, status, email_verified, is_active FROM users')
    rows = cursor.fetchall()
    print('Users in database:')
    for row in rows:
        print(row)
except Exception as e:
    print('Error:', e)
finally:
    conn.close()