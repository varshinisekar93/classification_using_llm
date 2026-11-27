from fastapi import FastAPI, HTTPException
from langchain_core.exceptions import OutputParserException
from openai import APIError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import ValidationError

from app.core.llm import LLMConfigurationError
from app.schemas.classification_schema import ClassificationRequest, ClassificationResponse
from app.services.classifier import classify_message


app = FastAPI(
    title="Customer Support Classification API",
    description="Classify customer messages using an OpenAI LLM",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "Customer Support Classification API is running"}


@app.post("/classify", response_model=ClassificationResponse)
def classify(request: ClassificationRequest) -> ClassificationResponse:
    try:
        return classify_message(request.message)
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=503, detail="OpenAI rejected the configured credentials."
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(
            status_code=503, detail="OpenAI is rate limited or has insufficient quota."
        ) from exc
    except APITimeoutError as exc:
        raise HTTPException(status_code=504, detail="OpenAI request timed out.") from exc
    except (OutputParserException, ValidationError) as exc:
        raise HTTPException(
            status_code=502, detail="OpenAI returned an invalid classification."
        ) from exc
    except APIError as exc:
        raise HTTPException(status_code=502, detail="OpenAI request failed.") from exc
