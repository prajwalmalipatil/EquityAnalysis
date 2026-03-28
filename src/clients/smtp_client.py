"""
smtp_client.py
Infrastructure client for sending emails via SMTP.
Handles SSL/TLS connection and message dispatching.
"""

import smtplib
import ssl
from email.message import EmailMessage
from typing import Optional

from src.utils.observability import get_tenant_logger
from src.utils.http_client import with_retry
from src.constants import email_constants as const

logger = get_tenant_logger("smtp-client")

class SMTPClient:
    """
    Client for secure email transmission.
    Decoupled from HTML content generation.
    """
    
    def __init__(self, smtp_server: str = const.SMTP_SERVER, port: int = const.DEFAULT_PORT):
        self.smtp_server = smtp_server
        self.port = port
        self.context = ssl.create_default_context()

    @with_retry(max_attempts=3, base_delay=2.0)
    def send_email(self, sender_email: str, sender_password: str, 
                   recipient_email: str, subject: str, html_body: str) -> bool:
        """Sends an HTML email message using SSL."""
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg.set_content("Please view this email in an HTML-compatible client.")
        msg.add_alternative(html_body, subtype='html')

        try:
            logger.info("SENDING_EMAIL", extra={"recipient": recipient_email, "subject": subject})
            with smtplib.SMTP_SSL(self.smtp_server, self.port, context=self.context) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
            logger.info("EMAIL_SENT_SUCCESSFULLY")
            return True
        except Exception as e:
            logger.error("EMAIL_SEND_FAILED", extra={"error": str(e)})
            raise e
