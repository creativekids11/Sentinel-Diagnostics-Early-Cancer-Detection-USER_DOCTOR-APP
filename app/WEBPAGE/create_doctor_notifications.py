import sqlite3
from datetime import datetime
import os

# Connect to database - use absolute path
db_path = os.path.join(os.path.dirname(__file__), 'database.db')
db = sqlite3.connect(db_path)

# Create test notifications for the actual doctor user (ID 10)
current_time = datetime.now().isoformat()

# Test notification for doctor user ID 10
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status) 
              VALUES (?, ?, ?, ?, ?, ?)''', 
              (10, 'Test Doctor Notification', 'This is a test notification for the doctor portal', 'message', current_time, 0))

# Create a few more test notifications
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status) 
              VALUES (?, ?, ?, ?, ?, ?)''', 
              (10, 'New Appointment Booked', 'A new appointment has been scheduled', 'appointment', current_time, 0))

db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status) 
              VALUES (?, ?, ?, ?, ?, ?)''', 
              (10, 'New Patient Message', 'You have received a new message from a patient', 'message', current_time, 0))

db.commit()
db.close()
print(f'Test notifications created for doctor (user ID 10) at {current_time}')