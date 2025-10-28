import sqlite3

# Connect to database
conn = sqlite3.connect('instance/database.db')
cursor = conn.cursor()

# Insert test pending payment
cursor.execute('''
    INSERT INTO pending_payments (user_email, plan_type, amount, transaction_id, message) 
    VALUES (?, ?, ?, ?, ?)
''', ('test@example.com', 'premium', 999.00, 'TEST123456', 'Test payment via UPI'))

conn.commit()

# Check if it was inserted
cursor.execute('SELECT * FROM pending_payments')
rows = cursor.fetchall()
print("Pending payments in database:")
for row in rows:
    print(row)

conn.close()