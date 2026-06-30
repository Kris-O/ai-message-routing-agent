import smtplib
from email.message import EmailMessage

from app.config import get_settings
from app.models import TargetDepartment


class EmailDeliveryError(RuntimeError):
    """Raised when SMTP delivery fails."""


def send_to_department(
    *,
    department: TargetDepartment,
    to_email: str,
    sender: str,
    message: str,
) -> None:
    """Send a message to the chosen department via SMTP."""
    settings = get_settings()
    email = EmailMessage()
    email["From"] = settings.mail_from
    email["To"] = to_email
    email["Reply-To"] = sender
    email["Subject"] = f"[routing] Nowa wiadomość do działu {department.value}"
    email.set_content(message)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            if settings.smtp_starttls:
                smtp.starttls()
            if settings.smtp_user and settings.smtp_password:
                smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(email)
    except (OSError, smtplib.SMTPException) as exc:
        raise EmailDeliveryError(str(exc)) from exc
