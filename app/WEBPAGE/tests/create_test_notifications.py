import sqlite3
from datetime import datetime

# Connect to database
db = sqlite3.connect('database.db')

# Create test notifications
current_time = datetime.now().isoformat()

# Test notification for user ID 1 (assuming doctor)
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status) 
              VALUES (?, ?, ?, ?, ?, ?)''', 
              (1, 'Test Doctor Notification', 'This is a test notification for doctor', 'message', current_time, 0))

# Test notification for user ID 2 (assuming patient)  
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status) 
              VALUES (?, ?, ?, ?, ?, ?)''', 
              (2, 'Test Patient Notification', 'This is a test notification for patient', 'appointment', current_time, 0))

db.commit()
db.close()
print(f'Test notifications created at {current_time}')