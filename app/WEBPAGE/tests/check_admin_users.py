import sqlite3

# Connect to database
conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()

# Check admin users
cursor.execute('SELECT email, role FROM users WHERE role = "admin"')
admins = cursor.fetchall()
print('Admin users:')
for admin in admins:
    print(admin)

# Also check doctors (who might have admin access)  
cursor.execute('SELECT email, first_name, last_name FROM doctors')
doctors = cursor.fetchall()
print('\nDoctors (may have admin access):')
for doctor in doctors:
    print(doctor)

conn.close()