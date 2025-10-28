import sqlite3

# Connect to database
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Check all tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables in database:')
for table in tables:
    print(f'  {table[0]}')

# Check if doctor_status table exists
cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="doctor_status"')
result = cursor.fetchone()
print(f'\ndoctor_status table exists: {result is not None}')

# If it exists, check its schema
if result:
    cursor.execute('PRAGMA table_info(doctor_status)')
    columns = cursor.fetchall()
    print('doctor_status table schema:')
    for col in columns:
        print(f'  {col[1]}: {col[2]}')

conn.close()