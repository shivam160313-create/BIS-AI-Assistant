import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

# ---------------------------------------------------------------------------
# Chat model selection.
#
# Free-tier Gemini model availability/naming shifts fairly often. Instead of
# hard-failing when GEMINI_MODEL points at a model that's been renamed or
# retired, we keep a small ordered list of fallbacks and walk through them
# automatically the first time a request fails with a "model not found"
# style error. Whichever one first succeeds is cached and reused.
# ---------------------------------------------------------------------------
_CANDIDATE_MODELS = list(dict.fromkeys(
    model
    for model in [
        os.getenv("GEMINI_MODEL"),
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-flash-latest",
        "gemini-1.5-flash",
    ]
    if model
))

DEFAULT_MODEL = _CANDIDATE_MODELS[0]

_client = None
_working_model = None


class _Response:
    def __init__(self, text):
        self.output_text = text


class _Responses:
    """
    Thin shim so ai_assistant.py / document_qa.py can keep calling

        client.responses.create(model=..., input=..., max_output_tokens=...)

    exactly as they did against the OpenAI SDK, without needing to know the
    underlying provider is now Gemini. Only .output_text is used downstream,
    so that's all this shim needs to provide.
    """

    def create(self, model=None, input=None, max_output_tokens=300, **kwargs):
        global _working_model

        raw_client = _get_raw_client()

        models_to_try = [model] if model else []
        models_to_try += [m for m in _CANDIDATE_MODELS if m not in models_to_try]

        if _working_model and _working_model not in models_to_try:
            models_to_try.insert(0, _working_model)

        last_error = None

        for candidate in models_to_try:
            try:
                response = raw_client.models.generate_content(
                    model=candidate,
                    contents=input,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_output_tokens
                    ),
                )

                text = getattr(response, "text", None)

                if not text:
                    # Response came back empty (e.g. safety filtering) -
                    # still a "success" from the API's point of view, so
                    # don't fall through to another model for this.
                    text = (
                        "I could not generate a response for this question."
                    )

                _working_model = candidate
                return _Response(text)

            except Exception as error:
                last_error = error
                message = str(error).lower()

                # Rate-limited: brief backoff, then try the next candidate
                # model rather than failing the request outright.
                if "429" in message or "quota" in message or "resource_exhausted" in message:
                    time.sleep(2)
                    continue

                # Model not found / not available on this key: try the next.
                if "404" in message or "not found" in message or "not supported" in message:
                    continue

                raise

        raise RuntimeError(f"All Gemini chat models failed: {last_error}")


class GeminiClient:
    def __init__(self):
        self.responses = _Responses()


def _get_raw_client():
    global _client

    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key (no credit card "
                "required) at https://aistudio.google.com/apikey and "
                "configure it as an environment variable "
                '(fastapi cloud env set --secret GOOGLE_API_KEY "...") or '
                "in a local .env file for development."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def get_client() -> GeminiClient:
    """
    Lazily create and cache a Gemini-backed client that mimics the small
    slice of the OpenAI client interface (`client.responses.create(...)
    .output_text`) that ai_assistant.py and document_qa.py rely on.

    Raises RuntimeError with a clear, actionable message if no API key is
    configured, instead of crashing at import time (which used to take
    down the entire FastAPI app before it could even start).
    """

    _get_raw_client()  # validates the key eagerly, same as before
    return GeminiClient()
