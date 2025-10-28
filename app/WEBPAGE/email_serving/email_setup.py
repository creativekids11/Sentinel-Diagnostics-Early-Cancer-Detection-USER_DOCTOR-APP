import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import logging

load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_email(recipient_email, subject, body):
    """Send an email and log the process for debugging."""
    try:
        logger.debug("Preparing email to send.")
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        logger.debug("Connecting to SMTP server.")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())

        logger.info("Email sent successfully to %s", recipient_email)
        return True
    except (smtplib.SMTPException, ConnectionError) as e:
        logger.error("Error sending email: %s", e)
        return False

# Example usage for testing
if __name__ == "__main__":
    recipient = "test@example.com"
    subject = "Test Email"
    body = "This is a test email."
    success = send_email(recipient, subject, body)
    if success:
        print("Test email sent successfully.")
    else:
        print("Failed to send test email.")
