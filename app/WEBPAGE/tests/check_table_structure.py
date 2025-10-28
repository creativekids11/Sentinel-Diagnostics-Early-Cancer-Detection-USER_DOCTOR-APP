import sqlite3

conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(pending_payments)')
columns = cursor.fetchall()
print("Current pending_payments table structure:")
for col in columns:
    print(f"  {col[1]} {col[2]} (nullable: {not col[3]})")
conn.close()