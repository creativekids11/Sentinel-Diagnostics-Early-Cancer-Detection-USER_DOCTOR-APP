import sqlite3import sqlite3import sqlite3



# Check the main database.db

conn = sqlite3.connect('database.db')

cursor = conn.cursor()# Check the main database.db# Connect to database.db



# Check what tables exist in database.dbconn = sqlite3.connect('database.db')conn = sqlite3.connect('instance/database.db')

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

tables = cursor.fetchall()cursor = conn.cursor()cursor = conn.cursor()

print('Tables in database.db:')

for table in tables:

    print(f'  {table[0]}')

# Check what tables exist in database.db# Get all table names

# Check users table

try:cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    cursor.execute('SELECT id, fullname, username, role, subscription_plan FROM users LIMIT 10')

    users = cursor.fetchall()tables = cursor.fetchall()tables = cursor.fetchall()

    print('\nUsers in database.db:')

    for user in users:print('Tables in database.db:')

        print(f'ID: {user[0]}, Name: {user[1]}, Username: {user[2]}, Role: {user[3]}, Plan: {user[4]}')

for table in tables:print('Tables in database.db:')

    print('\nPatient users:')

    cursor.execute('SELECT id, fullname, username, role, subscription_plan FROM users WHERE role="patient" LIMIT 5')    print(f'  {table[0]}')for table in tables:

    patients = cursor.fetchall()

    for user in patients:    print(table[0])

        print(f'ID: {user[0]}, Name: {user[1]}, Username: {user[2]}, Role: {user[3]}, Plan: {user[4]}')

except Exception as e:# Check users table

    print(f'Error querying users: {e}')

try:# Check if users table exists

conn.close()
    cursor.execute('SELECT id, fullname, username, role, subscription_plan FROM users LIMIT 10')if ('users',) in tables:

    users = cursor.fetchall()    print('')

    print('\nUsers in database.db:')    print('Users table exists, checking doctors:')

    for user in users:    cursor.execute('SELECT id, fullname, role FROM users WHERE role = "doctor"')

        print(f'ID: {user[0]}, Name: {user[1]}, Username: {user[2]}, Role: {user[3]}, Plan: {user[4]}')    doctors = cursor.fetchall()

    print('Doctors in system:', len(doctors))

    print('\nPatient users:')    for d in doctors:

    cursor.execute('SELECT id, fullname, username, role, subscription_plan FROM users WHERE role="patient" LIMIT 5')        print('ID:', d[0], 'Name:', d[1])

    patients = cursor.fetchall()

    for user in patients:conn.close()
        print(f'ID: {user[0]}, Name: {user[1]}, Username: {user[2]}, Role: {user[3]}, Plan: {user[4]}')
except Exception as e:
    print(f'Error querying users: {e}')

conn.close()