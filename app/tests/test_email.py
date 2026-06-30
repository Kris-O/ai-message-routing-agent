import smtplib

import httpx
import pytest

import app.classifier as classifier
import app.email_sender as email_sender
import app.main as main
from app.config import Settings, get_settings
from app.models import DEPARTMENT_EMAILS, TargetDepartment


pytestmark = pytest.mark.anyio
app = main.app


class _CapturingSMTP:
    last_message = None
    last_instance = None

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.starttls_calls = 0
        self.login_calls = []
        type(self).last_instance = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self) -> None:
        self.starttls_calls += 1

    def login(self, user: str, password: str) -> None:
        self.login_calls.append((user, password))

    def send_message(self, msg) -> None:
        type(self).last_message = msg

    @classmethod
    def reset(cls) -> None:
        cls.last_message = None
        cls.last_instance = None


async def test_message_is_delivered_to_department_with_reply_to(
    monkeypatch, tool_calling_model
) -> None:
    # End-to-end through the endpoint: the agent calls the send_email tool, which delivers via SMTP.
    _CapturingSMTP.reset()
    monkeypatch.setattr(email_sender.smtplib, "SMTP", _CapturingSMTP)

    transport = httpx.ASGITransport(app=app)
    with classifier.agent.override(model=tool_calling_model("KADRY")):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/route-message",
                json={"email": "jan@firma.pl", "message": "Chcę iść na urlop"},
            )

    assert response.status_code == 200
    assert response.json()["sent"] is True

    message = _CapturingSMTP.last_message
    assert message is not None
    assert message["From"] == get_settings().mail_from
    assert message["To"] == DEPARTMENT_EMAILS[TargetDepartment.KADRY]
    assert message["Reply-To"] == "jan@firma.pl"
    assert TargetDepartment.KADRY.value in message["Subject"]
    # loop / auto-reply prevention headers (RFC 3834 + X-Loop)
    assert message["Auto-Submitted"] == "auto-generated"
    assert message["X-Loop"] == get_settings().mail_from
    assert _CapturingSMTP.last_instance is not None
    assert _CapturingSMTP.last_instance.starttls_calls == 0
    assert _CapturingSMTP.last_instance.login_calls == []


def test_send_to_department_defaults_skip_starttls_and_login(monkeypatch) -> None:
    _CapturingSMTP.reset()
    monkeypatch.setattr(email_sender.smtplib, "SMTP", _CapturingSMTP)
    monkeypatch.setattr(email_sender, "get_settings", lambda: Settings())

    email_sender.send_to_department(
        department=TargetDepartment.HR,
        to_email=DEPARTMENT_EMAILS[TargetDepartment.HR],
        sender="ewa@firma.pl",
        message="Proszę o szkolenie",
    )

    assert _CapturingSMTP.last_message is not None
    assert _CapturingSMTP.last_instance is not None
    assert _CapturingSMTP.last_instance.starttls_calls == 0
    assert _CapturingSMTP.last_instance.login_calls == []


def test_send_to_department_uses_starttls_and_login_when_configured(
    monkeypatch,
) -> None:
    _CapturingSMTP.reset()
    monkeypatch.setattr(email_sender.smtplib, "SMTP", _CapturingSMTP)
    monkeypatch.setattr(
        email_sender,
        "get_settings",
        lambda: Settings(
            smtp_starttls=True,
            smtp_user="mailer",
            smtp_password="secret",
        ),
    )

    email_sender.send_to_department(
        department=TargetDepartment.IT,
        to_email=DEPARTMENT_EMAILS[TargetDepartment.IT],
        sender="adam@firma.pl",
        message="VPN nie działa",
    )

    assert _CapturingSMTP.last_instance is not None
    assert _CapturingSMTP.last_instance.starttls_calls == 1
    assert _CapturingSMTP.last_instance.login_calls == [("mailer", "secret")]


async def test_delivery_failure_returns_503(monkeypatch, tool_calling_model) -> None:
    def boom(*args, **kwargs):
        _ = (args, kwargs)
        raise smtplib.SMTPException("mailhog down")

    monkeypatch.setattr(email_sender.smtplib, "SMTP", boom)

    transport = httpx.ASGITransport(app=app)
    with classifier.agent.override(model=tool_calling_model("IT")):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/route-message",
                json={"email": "ola@firma.pl", "message": "VPN nie działa"},
            )

    assert response.status_code == 503
    assert response.json() == {"detail": "message delivery is temporarily unavailable"}
