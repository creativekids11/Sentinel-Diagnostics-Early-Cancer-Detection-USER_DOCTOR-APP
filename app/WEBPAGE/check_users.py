import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Check for doctor users
cursor.execute('SELECT username, role, status FROM users WHERE role="doctor" LIMIT 5')
doctors = cursor.fetchall()
print('Doctor users:', doctors)

# Check all users
cursor.execute('SELECT username, role FROM users LIMIT 10')
all_users = cursor.fetchall()
print('All users:', all_users)

conn.close()