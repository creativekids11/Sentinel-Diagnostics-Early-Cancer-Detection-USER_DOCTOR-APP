import sqlite3

conn = sqlite3.connect('database.db')
cur = conn.cursor()

# Check doctors
cur.execute('SELECT id, fullname, role FROM users WHERE role="doctor"')
doctors = cur.fetchall()
print('Doctors:')
for d in doctors:
    print(f'  ID: {d[0]}, Name: {d[1]}, Role: {d[2]}')

# Check notifications for doctors
doctor_ids = [d[0] for d in doctors]
if doctor_ids:
    placeholders = ','.join('?' * len(doctor_ids))
    cur.execute(f'SELECT COUNT(*) FROM notifications WHERE user_id IN ({placeholders})', doctor_ids)
    result = cur.fetchone()
    print('Notifications for doctors:', result[0])

    # Show recent doctor notifications
    cur.execute(f'SELECT id, user_id, title, type, read_status FROM notifications WHERE user_id IN ({placeholders}) ORDER BY created_at DESC LIMIT 5', doctor_ids)
    notifications = cur.fetchall()
    print('Recent doctor notifications:')
    for n in notifications:
        print(f'  ID: {n[0]}, User: {n[1]}, Title: {n[2]}, Type: {n[3]}, Read: {n[4]}')

# Check all appointments and their notifications
cur.execute('SELECT id, user_id, doctor_id, status FROM appointments ORDER BY created_at DESC LIMIT 10')
appointments = cur.fetchall()
print('\nRecent appointments:')
for a in appointments:
    print(f'  ID: {a[0]}, Patient: {a[1]}, Doctor: {a[2]}, Status: {a[3]}')

conn.close()