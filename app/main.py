import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException, Request

from app.classifier import (
    ClassifierError,
    aclose as close_classifier,
    route_and_send,
    warmup,
)
from app.config import Settings, get_settings
from app.email_sender import EmailDeliveryError
from app.models import DEPARTMENT_EMAILS, RouteRequest, RouteResponse

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model in the background so startup/health isn't blocked by the cold model load.
    warmup_task = asyncio.create_task(warmup())
    yield
    warmup_task.cancel()
    await close_classifier()  # release the agent's HTTP client on shutdown


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    limiter = Limiter(
        key_func=get_remote_address,
        enabled=settings.rate_limit_enabled,
    )
    router = APIRouter(prefix="/api/v1")

    @router.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/route-message", response_model=RouteResponse)
    @limiter.limit(settings.rate_limit)
    async def route_message(
        request: Request,
        payload: RouteRequest,
    ) -> RouteResponse:
        _ = request
        # The agent both decides the department and delivers the mail via its send_email tool
        # (tool/function calling). We only map the two upstream failure modes onto HTTP 503.
        try:
            department = await route_and_send(
                sender=str(payload.email),
                message=payload.message,
            )
        except ClassifierError as exc:
            raise HTTPException(
                status_code=503,
                detail="message classifier is temporarily unavailable",
            ) from exc
        except EmailDeliveryError as exc:
            raise HTTPException(
                status_code=503,
                detail="message delivery is temporarily unavailable",
            ) from exc

        return RouteResponse(
            department=department,
            target_email=DEPARTMENT_EMAILS[department],
            sent=True,
            detail="classified and delivered",
        )

    app = FastAPI(
        title="AI Message Routing",
        docs_url="/api/v1/docs",
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router)
    return app


app = create_app()
