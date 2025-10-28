import sqlite3

conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()

# Check pending_payments table structure
cursor.execute('PRAGMA table_info(pending_payments)')
columns = cursor.fetchall()
print('pending_payments columns:')
for col in columns:
    print(f'- {col[1]} ({col[2]})')

# Check current data
cursor.execute('SELECT * FROM pending_payments')
rows = cursor.fetchall()
print(f'\nCurrent data ({len(rows)} rows):')
for row in rows:
    print(row)

conn.close()