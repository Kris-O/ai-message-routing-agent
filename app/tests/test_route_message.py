import httpx
import pytest

import app.main as main
from app.config import Settings
from app.models import DEPARTMENT_EMAILS, TargetDepartment


app = main.app
pytestmark = pytest.mark.anyio


async def test_route_message_returns_mapped_department_email(monkeypatch) -> None:
    async def _route(*, sender: str, message: str) -> TargetDepartment:
        return TargetDepartment.KADRY

    monkeypatch.setattr(main, "route_and_send", _route)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": "Chcę iść na urlop"},
        )

    assert response.status_code == 200
    body = response.json()
    department = TargetDepartment(body["department"])
    assert body["target_email"] == DEPARTMENT_EMAILS[department]
    assert body["sent"] is True


async def test_route_message_upstream_failure_returns_503(monkeypatch) -> None:
    from app.classifier import ClassifierError

    async def _route(*, sender: str, message: str) -> TargetDepartment:
        raise ClassifierError("ollama unreachable")

    monkeypatch.setattr(main, "route_and_send", _route)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": "Potrzebuję pomocy"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "message classifier is temporarily unavailable"}


async def test_route_message_rejects_invalid_email() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/route-message",
            json={"email": "not-an-email", "message": "Proszę o pomoc"},
        )

    assert response.status_code == 422


async def test_route_message_rejects_empty_message() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": ""},
        )

    assert response.status_code == 422


async def test_route_message_rejects_whitespace_only_message() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": "   "},
        )

    assert response.status_code == 422


async def test_route_message_rejects_too_long_message() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": "a" * 4001},
        )

    assert response.status_code == 422


async def test_route_message_requires_message_field() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/route-message", json={"email": "jan@firma.pl"}
        )

    assert response.status_code == 422


async def test_health_endpoint() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_openapi_is_served_under_api_v1() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/openapi.json")

    assert response.status_code == 200


async def test_route_message_rate_limit_returns_429_on_third_request(
    monkeypatch,
) -> None:
    limited_app = main.create_app(Settings(rate_limit="2/minute"))

    async def _route(*, sender: str, message: str) -> TargetDepartment:
        return TargetDepartment.KADRY

    monkeypatch.setattr(main, "route_and_send", _route)

    transport = httpx.ASGITransport(app=limited_app, client=("198.51.100.7", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response_1 = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": "Chcę iść na urlop"},
        )
        response_2 = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": "Chcę iść na urlop"},
        )
        response_3 = await client.post(
            "/api/v1/route-message",
            json={"email": "jan@firma.pl", "message": "Chcę iść na urlop"},
        )

    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert response_3.status_code == 429
    # slowapi's default 429 handler returns {"error": "Rate limit exceeded: <limit>"}.
    assert "Rate limit exceeded" in response_3.json()["error"]
