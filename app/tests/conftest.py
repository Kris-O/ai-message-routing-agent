"""Shared test helpers — fake LLM models driven through pydantic-ai's FunctionModel.

These let the unit tests exercise the *real* agent + tool wiring (including the enum guardrail and SMTP
delivery) without a live Ollama: we script exactly what the "model" does — call the send_email tool with a
given department, call it with a bad value, or refuse to call it at all. Swapped in via
`agent.override(model=...)`."""
import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def _tool_has_returned(messages) -> bool:
    """True once the send_email tool has run and returned — i.e. delivery already happened."""
    return any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in getattr(message, "parts", [])
    )


@pytest.fixture
def tool_calling_model():
    """Factory: a fake model that routes by calling send_email(department=...) once, then finishes.

    Pass an invalid department (e.g. "PRAWNICY") to exercise the enum guardrail: pydantic-ai rejects the
    arg and keeps retrying this same bad call until it gives up."""

    def make(department: str) -> FunctionModel:
        def respond(messages, info: AgentInfo) -> ModelResponse:
            if _tool_has_returned(messages):
                return ModelResponse(parts=[TextPart("Gotowe.")])
            return ModelResponse(
                parts=[ToolCallPart(tool_name="send_email", args={"department": department})]
            )

        return FunctionModel(respond)

    return make


@pytest.fixture
def text_only_model():
    """Factory: a fake model that never calls the tool — replies with plain text (the 3B failure mode)."""

    def make(text: str = "KADRY") -> FunctionModel:
        def respond(messages, info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(text)])

        return FunctionModel(respond)

    return make


@pytest.fixture
def exploding_model():
    """A fake model that fails if the agent ever calls it — proves a code path bypassed the LLM."""

    def respond(messages, info: AgentInfo) -> ModelResponse:
        raise AssertionError("the model must not be invoked on this path")

    return FunctionModel(respond)
