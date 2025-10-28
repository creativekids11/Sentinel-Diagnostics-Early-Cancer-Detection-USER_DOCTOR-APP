import sqlite3
import json
from datetime import datetime

# Connect to database
conn = sqlite3.connect('instance/app.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Check notifications table
cursor.execute('SELECT * FROM notifications ORDER BY created_at DESC LIMIT 10')
notifications = cursor.fetchall()

print('Recent notifications:')
for n in notifications:
    print('ID:', n['id'], 'User:', n['user_id'], 'Title:', n['title'], 'Type:', n['type'], 'Read:', n['read_status'], 'Created:', n['created_at'])

# Check if there are any doctor users
cursor.execute('SELECT id, fullname, role FROM users WHERE role = "doctor"')
doctors = cursor.fetchall()
print('')
print('Doctors in system:', len(doctors))
for d in doctors:
    print('ID:', d['id'], 'Name:', d['fullname'])

conn.close()