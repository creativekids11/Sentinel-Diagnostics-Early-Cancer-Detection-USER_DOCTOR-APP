#!/usr/bin/env python3
"""
Test script to run Flask app without the heavy AI model loading for quick testing
"""
import sys
import os

# Add the directory containing the Flask app to the Python path
sys.path.insert(0, os.path.dirname(__file__))

# Set environment variable to disable model loading for testing
os.environ['DISABLE_AI_MODELS'] = 'true'

# Create a minimal Flask app for testing subscription features
from flask import Flask, render_template, session, redirect, url_for, flash
import sqlite3
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = "test-secret-key"

DATABASE = os.path.join(os.path.dirname(__file__), "database.db")

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def get_user_subscription_plan(user_id):
    """Get user's current subscription plan and status"""
    db = get_db()
    user = db.execute(
        "SELECT subscription_plan, subscription_expires_at, subscription_status FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    
    if not user:
        return {"plan": "free", "status": "active", "expires_at": None}
    
    # Check if subscription is expired
    if user["subscription_expires_at"]:
        expires_datetime = datetime.fromisoformat(user["subscription_expires_at"])
        if expires_datetime < datetime.now():
            # Update expired subscription back to free
            db.execute(
                "UPDATE users SET subscription_plan = 'free', subscription_status = 'expired' WHERE id = ?",
                (user_id,)
            )
            db.commit()
            return {"plan": "free", "status": "expired", "expires_at": user["subscription_expires_at"]}
    
    return {
        "plan": user["subscription_plan"] or "free",
        "status": user["subscription_status"] or "active",
        "expires_at": user["subscription_expires_at"]
    }

def check_case_limit(user_id):
    """Check daily case creation limit for user"""
    subscription = get_user_subscription_plan(user_id)
    plan = subscription["plan"]
    
    # Get today's case count
    today = datetime.now().strftime("%Y-%m-%d")
    db = get_db()
    today_cases = db.execute(
        "SELECT COUNT(*) as count FROM cases WHERE user_id = ? AND DATE(created_at) = ?",
        (user_id, today)
    ).fetchone()
    
    current_count = today_cases["count"] if today_cases else 0
    
    if plan == "free":
        return {"allowed": current_count < 3, "limit": 3, "used": current_count}
    elif plan == "pro":
        return {"allowed": current_count < 10, "limit": 10, "used": current_count}
    elif plan == "premium":
        return {"allowed": True, "limit": None, "used": current_count}
    
    return {"allowed": False, "limit": 3, "used": current_count}

def check_nutrition_access(user_id):
    """Check if user has access to nutrition planner"""
    subscription = get_user_subscription_plan(user_id)
    plan = subscription["plan"]
    return {"allowed": plan in ["pro", "premium"]}

def check_meeting_access(user_id):
    """Check if user has access to private meetings"""
    subscription = get_user_subscription_plan(user_id)
    plan = subscription["plan"]
    return {"allowed": plan == "premium"}

def check_report_access(user_id):
    """Check if user can generate reports and get time delay"""
    subscription = get_user_subscription_plan(user_id)
    plan = subscription["plan"]
    
    if plan == "free":
        return {"allowed": True, "delay_minutes": 30}
    elif plan == "pro":
        return {"allowed": True, "delay_minutes": 10}
    elif plan == "premium":
        return {"allowed": True, "delay_minutes": 0}
    
    return {"allowed": False, "delay_minutes": 30}

@app.route("/")
def home():
    return f"""
    <h1>Subscription Testing</h1>
    <p><a href="/test-subscription/15">Test Premium User (ID: 15)</a></p>
    <p><a href="/test-subscription/1">Test Free User (ID: 1 if exists)</a></p>
    """

@app.route("/test-subscription/<int:user_id>")
def test_subscription(user_id):
    """Test subscription features for a specific user"""
    
    # Get current subscription info
    subscription = get_user_subscription_plan(user_id)
    
    # Get usage statistics
    db = get_db()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Count today's cases
    today_cases = db.execute(
        "SELECT COUNT(*) as count FROM cases WHERE user_id = ? AND DATE(created_at) = ?",
        (user_id, today)
    ).fetchone()
    cases_used = today_cases["count"] if today_cases else 0
    
    # Get case limit info
    case_limit_info = check_case_limit(user_id)
    
    # Get access info
    nutrition_access = check_nutrition_access(user_id)
    meeting_access = check_meeting_access(user_id)
    report_access = check_report_access(user_id)
    
    # Count pending reports (not yet available)
    current_time = datetime.now().isoformat()
    pending_reports = db.execute("""
        SELECT COUNT(*) as count FROM reports 
        WHERE patient_id = ? AND available_at > ?
    """, (user_id, current_time)).fetchone()
    pending_count = pending_reports["count"] if pending_reports else 0
    
    subscription_details = {
        "current_plan": subscription["plan"],
        "status": subscription["status"],
        "expires_at": subscription["expires_at"],
        "cases_used": cases_used,
        "case_limit": case_limit_info["limit"],
        "can_create_case": case_limit_info["allowed"],
        "nutrition_access": nutrition_access["allowed"],
        "meeting_access": meeting_access["allowed"],
        "report_delay_minutes": report_access["delay_minutes"],
        "pending_reports": pending_count
    }
    
    result = f"""
    <h1>Subscription Test for User ID: {user_id}</h1>
    <h2>Subscription Details:</h2>
    <ul>
        <li><strong>Plan:</strong> {subscription_details['current_plan']}</li>
        <li><strong>Status:</strong> {subscription_details['status']}</li>
        <li><strong>Expires:</strong> {subscription_details['expires_at'] or 'N/A'}</li>
    </ul>
    
    <h2>Feature Access:</h2>
    <ul>
        <li><strong>Cases Today:</strong> {subscription_details['cases_used']}/{subscription_details['case_limit'] or 'Unlimited'}</li>
        <li><strong>Can Create Case:</strong> {'✅ Yes' if subscription_details['can_create_case'] else '❌ No'}</li>
        <li><strong>Nutrition Access:</strong> {'✅ Yes' if subscription_details['nutrition_access'] else '❌ No'}</li>
        <li><strong>Private Meetings:</strong> {'✅ Yes' if subscription_details['meeting_access'] else '❌ No'}</li>
        <li><strong>Report Delay:</strong> {subscription_details['report_delay_minutes']} minutes</li>
        <li><strong>Pending Reports:</strong> {subscription_details['pending_reports']}</li>
    </ul>
    
    <p><a href="/">Back to Home</a></p>
    """
    
    return result

if __name__ == "__main__":
    print("Starting test Flask app on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)