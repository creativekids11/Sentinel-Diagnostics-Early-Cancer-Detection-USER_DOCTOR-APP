import sqlite3

# Connect to database
conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()

# Create pending_payments table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS pending_payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        plan_type TEXT NOT NULL,
        amount DECIMAL(10,2) NOT NULL,
        transaction_id TEXT,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT "pending",
        processed_by TEXT,
        processed_at TIMESTAMP
    )
''')

conn.commit()
conn.close()
print("pending_payments table created successfully!")