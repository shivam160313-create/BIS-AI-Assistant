import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configurable via env var so a wrong/placeholder model name can never take
# the whole app down again - just set OPENAI_MODEL in FastAPI Cloud if you
# want to change it later.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """
    Lazily create and cache the OpenAI client.

    Raises RuntimeError with a clear, actionable message if OPENAI_API_KEY
    is missing, instead of crashing at import time (which used to take down
    the entire FastAPI app before it could even start).
    """

    global _client

    if _client is None:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Configure it as an environment "
                "variable in FastAPI Cloud "
                "(fastapi cloud env set --secret OPENAI_API_KEY \"sk-...\") "
                "or in a local .env file for development."
            )

        _client = OpenAI(api_key=api_key, timeout=30.0)

    return _client
