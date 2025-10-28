import sqlite3

# Connect to database
conn = sqlite3.connect('instance/app.db')
cursor = conn.cursor()

# Get all table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print('Tables in database:')
for table in tables:
    print(table[0])

# Check notifications table structure
print('')
print('Notifications table structure:')
cursor.execute('PRAGMA table_info(notifications)')
columns = cursor.fetchall()
for col in columns:
    print(col)

conn.close()