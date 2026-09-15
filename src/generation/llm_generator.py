"""
LLM Generator
=============

Production-oriented LLM generation layer for the Hiver Support Agent.

Responsibilities:
- Accept an evidence-grounded generation prompt.
- Call the configured Gemini model.
- Keep API credentials outside the source code.
- Return a structured generation result.
- Fail safely when configuration or API calls fail.
- Support dependency injection so tests never need a real API call.

This module does NOT:
- retrieve evidence
- make automation decisions
- perform grounding/safety checks
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_ENV_KEY = "GEMINI_API_KEY"


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Structured result returned by the LLM generator."""

    response: str
    status: str
    model: str
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True when generation completed successfully."""
        return self.status == "SUCCESS"


# ---------------------------------------------------------------------------
# Response extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_response(result: Any) -> str:
    """
    Extract generated text from common Gemini response representations.
    """

    if result is None:
        raise ValueError("LLM returned an empty response object.")

    # Direct string response
    if isinstance(result, str):
        text = result.strip()

        if text:
            return text

    # Dictionary-like response
    if isinstance(result, dict):
        for key in [
            "text",
            "output_text",
            "response",
            "content",
        ]:
            value = result.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        output = result.get("output")

        if output is not None:
            extracted = _extract_text_from_output_items(output)

            if extracted:
                return extracted

    # Object attributes
    for attribute in [
        "text",
        "output_text",
        "response",
        "content",
    ]:
        if hasattr(result, attribute):
            value = getattr(result, attribute)

            if isinstance(value, str) and value.strip():
                return value.strip()

    # Generic output attribute
    if hasattr(result, "output"):
        output = getattr(result, "output")

        extracted = _extract_text_from_output_items(output)

        if extracted:
            return extracted

    raise ValueError(
        "Could not extract generated text from the LLM response."
    )


def _extract_text_from_output_items(output: Any) -> str:
    """Extract text from a list or object representation of model output."""

    if output is None:
        return ""

    if isinstance(output, str):
        return output.strip()

    if isinstance(output, (list, tuple)):
        collected = []

        for item in output:
            text = _extract_text_from_output_item(item)

            if text:
                collected.append(text)

        return "\n".join(collected).strip()

    return _extract_text_from_output_item(output).strip()


def _extract_text_from_output_item(item: Any) -> str:
    """Extract text from one output item."""

    if item is None:
        return ""

    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):

        for key in [
            "text",
            "output_text",
            "content",
        ]:
            value = item.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

        content = item.get("content")

        if isinstance(content, list):
            parts = []

            for content_item in content:
                text = _extract_text_from_output_item(content_item)

                if text:
                    parts.append(text)

            return "\n".join(parts).strip()

    for attribute in [
        "text",
        "output_text",
    ]:
        if hasattr(item, attribute):
            value = getattr(item, attribute)

            if isinstance(value, str) and value.strip():
                return value.strip()

    if hasattr(item, "content"):
        content = getattr(item, "content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, (list, tuple)):
            parts = []

            for content_item in content:
                text = _extract_text_from_output_item(content_item)

                if text:
                    parts.append(text)

            return "\n".join(parts).strip()

    return ""


# ---------------------------------------------------------------------------
# Gemini client factory
# ---------------------------------------------------------------------------

def create_gemini_client(
    api_key: Optional[str] = None,
    env_key: str = DEFAULT_ENV_KEY,
) -> Any:
    """
    Create a Gemini client.

    API key priority:
        1. Explicit api_key argument
        2. Environment variable
    """

    resolved_key = api_key or os.getenv(env_key)

    if not resolved_key:
        raise ValueError(
            f"Gemini API key not found. "
            f"Set the {env_key} environment variable."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed. "
            "Install it before using the Gemini generator."
        ) from exc

    return genai.Client(api_key=resolved_key)


# ---------------------------------------------------------------------------
# LLM Generator
# ---------------------------------------------------------------------------

class LLMGenerator:
    """
    Evidence-grounded LLM response generator.

    The client can be injected so local tests never need a real API call.
    """

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        env_key: str = DEFAULT_ENV_KEY,
    ) -> None:

        self.model = model
        self.env_key = env_key

        if client is not None:
            self.client = client
        else:
            self.client = create_gemini_client(
                api_key=api_key,
                env_key=env_key,
            )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
    ) -> GenerationResult:
        """
        Generate a customer-facing response.

        The method fails safely instead of propagating API errors.
        """

        if not isinstance(prompt, str):
            return GenerationResult(
                response="",
                status="FAILED",
                model=self.model,
                error="Prompt must be a string.",
            )

        prompt = prompt.strip()

        if not prompt:
            return GenerationResult(
                response="",
                status="FAILED",
                model=self.model,
                error="Prompt cannot be empty.",
            )

        try:

            result = self.client.interactions.create(
                model=self.model,
                input=prompt,
            )

            response_text = _extract_text_from_response(result)

            if not response_text:
                return GenerationResult(
                    response="",
                    status="FAILED",
                    model=self.model,
                    error="LLM returned empty generated text.",
                )

            return GenerationResult(
                response=response_text,
                status="SUCCESS",
                model=self.model,
                error=None,
            )

        except Exception as exc:

            return GenerationResult(
                response="",
                status="FAILED",
                model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_response(
    prompt: str,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    env_key: str = DEFAULT_ENV_KEY,
) -> GenerationResult:
    """Convenience wrapper around LLMGenerator."""

    generator = LLMGenerator(
        client=client,
        model=model,
        api_key=api_key,
        env_key=env_key,
    )

    return generator.generate(prompt)


# ---------------------------------------------------------------------------
# Fake client for local testing
# ---------------------------------------------------------------------------

class _FakeInteractions:
    """Fake Interactions API used by local tests."""

    def __init__(
        self,
        response: Any = None,
        error: Optional[Exception] = None,
    ) -> None:

        self.response = response
        self.error = error
        self.calls = []

    def create(
        self,
        model: str,
        input: str,
    ) -> Any:

        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


class _FakeClient:
    """Fake Gemini client used by local tests."""

    def __init__(
        self,
        response: Any = None,
        error: Optional[Exception] = None,
    ) -> None:

        self.interactions = _FakeInteractions(
            response=response,
            error=error,
        )


# ---------------------------------------------------------------------------
# Local tests
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """Run local unit tests without calling Gemini."""

    # ---------------------------------------------------------------
    # Test 1: successful string response
    # ---------------------------------------------------------------

    fake_client = _FakeClient(
        response="We'd be happy to help you with your order."
    )

    generator = LLMGenerator(
        client=fake_client,
        model="test-model",
    )

    result = generator.generate(
        "Help the customer with their order."
    )

    assert result.success
    assert result.status == "SUCCESS"
    assert result.response == (
        "We'd be happy to help you with your order."
    )
    assert result.model == "test-model"

    assert len(fake_client.interactions.calls) == 1
    assert fake_client.interactions.calls[0]["model"] == "test-model"

    # ---------------------------------------------------------------
    # Test 2: dictionary response
    # ---------------------------------------------------------------

    fake_client = _FakeClient(
        response={
            "text": "Please contact us directly so we can assist."
        }
    )

    result = generate_response(
        "Customer needs support.",
        client=fake_client,
    )

    assert result.success
    assert result.response == (
        "Please contact us directly so we can assist."
    )

    # ---------------------------------------------------------------
    # Test 3: object response
    # ---------------------------------------------------------------

    class FakeResponse:

        def __init__(self):
            self.text = "Thanks for reaching out to us."

    fake_client = _FakeClient(
        response=FakeResponse()
    )

    result = generate_response(
        "Customer needs assistance.",
        client=fake_client,
    )

    assert result.success
    assert result.response == (
        "Thanks for reaching out to us."
    )

    # ---------------------------------------------------------------
    # Test 4: empty prompt
    # ---------------------------------------------------------------

    fake_client = _FakeClient(
        response="Should not be called."
    )

    generator = LLMGenerator(
        client=fake_client
    )

    result = generator.generate("   ")

    assert not result.success
    assert result.status == "FAILED"
    assert result.error is not None
    assert "empty" in result.error.lower()
    assert len(fake_client.interactions.calls) == 0

    # ---------------------------------------------------------------
    # Test 5: API failure
    # ---------------------------------------------------------------

    fake_client = _FakeClient(
        error=RuntimeError("Simulated API failure")
    )

    generator = LLMGenerator(
        client=fake_client
    )

    result = generator.generate(
        "Generate a support response."
    )

    assert not result.success
    assert result.status == "FAILED"
    assert result.response == ""
    assert result.error is not None
    assert "RuntimeError" in result.error

    # ---------------------------------------------------------------
    # Test 6: empty LLM response
    # ---------------------------------------------------------------

    fake_client = _FakeClient(
        response=""
    )

    generator = LLMGenerator(
        client=fake_client
    )

    result = generator.generate(
        "Generate a response."
    )

    assert not result.success
    assert result.status == "FAILED"
    assert result.error is not None

    # Important:
    # Do not depend on one exact internal error message.
    # The contract is simply that an empty model response fails safely.
    assert "failed" in result.status.lower()

    # ---------------------------------------------------------------
    # Test 7: nested output response
    # ---------------------------------------------------------------

    fake_client = _FakeClient(
        response={
            "output": [
                {
                    "content": [
                        {
                            "text": "We can help you with this issue."
                        }
                    ]
                }
            ]
        }
    )

    generator = LLMGenerator(
        client=fake_client
    )

    result = generator.generate(
        "Help the customer."
    )

    assert result.success
    assert result.response == (
        "We can help you with this issue."
    )

    # ---------------------------------------------------------------
    # Test 8: invalid prompt type
    # ---------------------------------------------------------------

    fake_client = _FakeClient(
        response="Should not be called."
    )

    generator = LLMGenerator(
        client=fake_client
    )

    result = generator.generate(None)

    assert not result.success
    assert result.status == "FAILED"
    assert result.error is not None
    assert "string" in result.error.lower()
    assert len(fake_client.interactions.calls) == 0

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    print("=" * 100)
    print("LLM GENERATOR TEST: PASS")
    print("=" * 100)
    print("All 8 local tests passed.")
    print("No real Gemini API call was made.")


if __name__ == "__main__":
    _run_tests()