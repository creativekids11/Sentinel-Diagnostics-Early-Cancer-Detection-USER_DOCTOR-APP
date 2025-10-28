import sqlite3
from datetime import datetime, timedelta
import os

# Connect to database
db_path = os.path.join(os.path.dirname(__file__), 'database.db')
db = sqlite3.connect(db_path)

# Get doctor ID (user ID 10)
doctor_id = 10

# Get a patient ID for testing
patient_result = db.execute("SELECT id, fullname FROM users WHERE role = 'patient' LIMIT 1").fetchone()
if not patient_result:
    print("No patient found, creating test patient...")
    # Create a test patient if none exists
    test_time = datetime.now().isoformat()
    db.execute("""INSERT INTO users (username, fullname, email, password_hash, role, created_at) 
                  VALUES (?, ?, ?, ?, ?, ?)""", 
                  ('test_patient', 'John Patient', 'patient@test.com', 'dummy_hash', 'patient', test_time))
    patient_id = db.lastrowid
    patient_name = 'John Patient'
else:
    patient_id = patient_result[0]
    patient_name = patient_result[1]

print(f"Using patient: {patient_name} (ID: {patient_id})")

# Create realistic notifications for doctor from patient activities
current_time = datetime.now()

# 1. Appointment booking notification
appointment_time = current_time - timedelta(minutes=30)
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status, related_id, related_type) 
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
              (doctor_id, 
               'New Appointment Request', 
               f'New appointment request from {patient_name} for tomorrow at 2:00 PM',
               'appointment', 
               appointment_time.isoformat(), 
               0, 
               1, 
               'appointment'))

# 2. Message notification
message_time = current_time - timedelta(minutes=15)
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status, related_id, related_type) 
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
              (doctor_id, 
               'New Message', 
               f'You have a new message from {patient_name}: "Can we reschedule my appointment?"',
               'message', 
               message_time.isoformat(), 
               0, 
               patient_id, 
               'user'))

# 3. Case update notification
case_time = current_time - timedelta(minutes=5)
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status, related_id, related_type) 
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
              (doctor_id, 
               'Patient Case Update', 
               f'{patient_name} has uploaded new medical documents for review',
               'case', 
               case_time.isoformat(), 
               0, 
               1, 
               'case'))

# 4. Urgent message notification
urgent_time = current_time - timedelta(minutes=2)
db.execute('''INSERT INTO notifications (user_id, title, message, type, created_at, read_status, related_id, related_type) 
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
              (doctor_id, 
               'Urgent Message', 
               f'URGENT: {patient_name} reports severe symptoms and needs immediate consultation',
               'message', 
               urgent_time.isoformat(), 
               0, 
               patient_id, 
               'user'))

db.commit()
db.close()
print(f'Created 4 patient activity notifications for doctor at {current_time}')