import sqlite3

db = sqlite3.connect('database.db')

print("Doctor users:")
users = db.execute("SELECT id, username, role FROM users WHERE role = 'doctor' ORDER BY id LIMIT 5").fetchall()
for row in users:
    print(f"ID: {row[0]}, Username: {row[1]}, Role: {row[2]}")

print("\nNotifications for doctor users:")
notifications = db.execute("SELECT id, user_id, title, read_status FROM notifications WHERE user_id IN (SELECT id FROM users WHERE role = 'doctor') ORDER BY created_at DESC LIMIT 5").fetchall()
for row in notifications:
    print(f"ID: {row[0]}, User: {row[1]}, Title: {row[2]}, Read Status: {row[3]}")

db.close()