import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Load SMTP configuration from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email(recipient, subject, body):
    """
    Sends an email using the configured SMTP server.

    Args:
        recipient (str): The recipient's email address.
        subject (str): The subject of the email.
        body (str): The body content of the email.

    Returns:
        None
    """
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("Error: Email credentials are not set in environment variables.")
        return

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

            # Create the email
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Send the email
            server.sendmail(EMAIL_ADDRESS, recipient, msg.as_string())
            print(f"Email sent successfully to {recipient}!")
    except smtplib.SMTPException as smtp_error:
        print(f"SMTP error occurred: {smtp_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with actual recipient email for testing
    # Test email - replace with actual recipient for testing
    recipient_email = "test-recipient@example.com"
    subject = "Test Subject"
    body = "This is a test email."

    send_email(recipient_email, subject, body)