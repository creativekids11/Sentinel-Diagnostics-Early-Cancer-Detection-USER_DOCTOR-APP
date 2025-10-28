import sqlite3

# Connect to database
conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('Tables in database:')
for table in tables:
    print(f"- {table[0]}")

# Check structure of one table
if tables:
    table_name = tables[0][0]
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f'\nStructure of {table_name}:')
    for column in columns:
        print(f"- {column[1]} ({column[2]})")

conn.close()