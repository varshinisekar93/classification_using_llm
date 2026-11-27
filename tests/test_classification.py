import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

from app.core.llm import settings
from app.main import app
from app.services import classifier


client = TestClient(app)


@pytest.mark.parametrize(
    ("message", "category"),
    [
        ("I was charged twice for the same order.", "Billing"),
        ("The app crashes whenever I open the dashboard.", "Technical Support"),
        ("I lost access to my account after changing phones.", "Account Issues"),
        ("Do you ship orders on Saturdays?", "General Inquiry"),
    ],
)
def test_classification_through_openai_adapter(monkeypatch, message, category):
    expected = {"category": category, "probability": 0.93, "reason": "Main issue identified."}

    def mock_openai(request):
        payload = json.loads(request.content)
        assert payload["messages"][-1] == {"role": "user", "content": message}
        assert "Billing:" in payload["messages"][0]["content"]
        function = payload["tools"][0]["function"]
        assert set(function["parameters"]["properties"]["category"]["enum"]) == {
            "Billing", "Technical Support", "Account Issues", "General Inquiry"
        }
        return httpx.Response(200, json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_test",
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "arguments": json.dumps(expected),
                        },
                    }],
                },
            }], 
        })

    with httpx.Client(transport=httpx.MockTransport(mock_openai)) as transport:
        monkeypatch.setattr(
            classifier, "get_llm",
            lambda: ChatOpenAI(model="gpt-4o-mini", api_key="test", http_client=transport),
        )
        response = client.post("/classify", json={"message": message})

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize("payload", [{}, {"message": "   "}, {"message": 123}])
def test_invalid_input(monkeypatch, payload):
    def unexpected_provider_call():
        pytest.fail("Invalid input reached the model")

    monkeypatch.setattr(classifier, "get_llm", unexpected_provider_call)
    assert client.post("/classify", json=payload).status_code == 422


def test_missing_credentials(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    assert client.get("/").status_code == 200
    assert client.post("/classify", json={"message": "Where is my invoice?"}).status_code == 503


def test_invalid_model_output(monkeypatch):
    class InvalidClassifier:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            return {"category": "Billing", "probability": 1.2, "reason": "Duplicate charge."}

    monkeypatch.setattr(classifier, "get_llm", InvalidClassifier)
    response = client.post("/classify", json={"message": "I was charged twice."})
    assert response.status_code == 502
    assert response.json() == {"detail": "OpenAI returned an invalid classification."}
