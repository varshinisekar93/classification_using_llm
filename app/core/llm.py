from langchain_openai import ChatOpenAI

from app.core.config import settings


class LLMConfigurationError(RuntimeError):
    pass


def get_llm() -> ChatOpenAI:
    if not settings.OPENAI_API_KEY.strip():
        raise LLMConfigurationError("Set OPENAI_API_KEY to enable classification.")

    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        timeout=20,
        max_retries=1,
    )
