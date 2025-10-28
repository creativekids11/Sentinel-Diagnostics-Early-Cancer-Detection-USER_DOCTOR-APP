"""
Minimal Flask app for testing admin payment verification
"""
from flask import Flask, render_template, request, flash, redirect, url_for, session
import sqlite3
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = "test_key_for_admin_verification"

def get_db():
    conn = sqlite3.connect('instance/database.db')
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # For testing, assume all logged in users are doctors
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route("/")
def index():
    return "Test Admin Payment Verification System"

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Simple test login
        session['user_email'] = 'test@doctor.com'
        session['user_role'] = 'doctor'
        flash('Logged in as test doctor', 'success')
        return redirect(url_for('admin_payments'))
    
    return """
    <form method="POST">
        <button type="submit">Login as Test Doctor</button>
    </form>
    """

@app.route("/admin/payments")
@login_required
@role_required("doctor")
def admin_payments():
    """Display pending payment verifications for admin approval"""
    db = get_db()
    
    # Get all pending payments
    pending_payments = db.execute("""
        SELECT * FROM pending_payments
        WHERE status = 'pending'
        ORDER BY created_at DESC
    """).fetchall()
    
    # Get recently processed payments (last 30 days)  
    recent_payments = db.execute("""
        SELECT * FROM pending_payments
        WHERE status IN ('approved', 'rejected') 
        AND processed_at >= datetime('now', '-30 days')
        ORDER BY processed_at DESC
        LIMIT 50
    """).fetchall()
    
    return render_template("admin/payments.html", 
                         pending_payments=pending_payments,
                         recent_payments=recent_payments,
                         pending_count=len(pending_payments))

@app.route("/admin/payments/<int:payment_id>/approve", methods=["POST"])
@login_required
@role_required("doctor")
def admin_approve_payment(payment_id):
    """Approve a pending payment and activate user subscription"""
    admin_notes = request.form.get("admin_notes", "").strip()
    
    db = get_db()
    
    try:
        # Get payment details
        payment = db.execute("""
            SELECT * FROM pending_payments 
            WHERE id = ? AND status = 'pending'
        """, (payment_id,)).fetchone()
        
        if not payment:
            flash("Payment not found or already processed.", "error")
            return redirect(url_for("admin_payments"))
        
        # Get current user for processed_by field  
        current_doctor = session.get('user_email', 'unknown')
        
        # Update payment record to approved
        db.execute("""
            UPDATE pending_payments 
            SET status = 'approved', processed_by = ?, processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (current_doctor, payment_id))
        
        # For this demo, we'll just mark the payment as approved
        # In a real system, you'd update the user's subscription status
        
        db.commit()
        
        # Log the approval
        app.logger.info(f"Payment approved by doctor {current_doctor} for user {payment['user_email']} - {payment['plan_type']} plan")
        
        flash(f"Payment approved! User {payment['user_email']}'s {payment['plan_type']} subscription is now active.", "success")
        
        # TODO: Send email notification to user
        
    except Exception as e:
        db.rollback()
        flash(f"Error approving payment: {str(e)}", "error")
    
    return redirect(url_for("admin_payments"))

@app.route("/admin/payments/<int:payment_id>/reject", methods=["POST"])
@login_required
@role_required("doctor")
def admin_reject_payment(payment_id):
    """Reject a pending payment"""
    admin_notes = request.form.get("admin_notes", "").strip()
    
    if not admin_notes:
        flash("Please provide a reason for rejection.", "error")
        return redirect(url_for("admin_payments"))
    
    db = get_db()
    
    try:
        # Get payment details
        payment = db.execute("""
            SELECT * FROM pending_payments 
            WHERE id = ? AND status = 'pending'
        """, (payment_id,)).fetchone()
        
        if not payment:
            flash("Payment not found or already processed.", "error")
            return redirect(url_for("admin_payments"))
        
        # Get current user for processed_by field  
        current_doctor = session.get('user_email', 'unknown')
        
        # Update payment record to rejected
        db.execute("""
            UPDATE pending_payments 
            SET status = 'rejected', processed_by = ?, processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (current_doctor, payment_id))
        
        db.commit()
        
        # Log the rejection
        app.logger.info(f"Payment rejected by doctor {current_doctor} for user {payment['user_email']} - {payment['plan_type']} plan. Reason: {admin_notes}")
        
        flash(f"Payment rejected. User {payment['user_email']} has been notified.", "info")
        
        # TODO: Send email notification to user with rejection reason
        
    except Exception as e:
        db.rollback()
        flash(f"Error rejecting payment: {str(e)}", "error")
    
    return redirect(url_for("admin_payments"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5602, host='127.0.0.1')