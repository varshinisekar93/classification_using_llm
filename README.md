# Customer Support Classification API

From the project folder, set `OPENAI_API_KEY` in the existing `.env` file to a valid OpenAI key. `OPENAI_MODEL` defaults to `gpt-4o-mini`.

Start the API using the project's Windows virtual environment.

```powershell
.\myenv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open [Swagger UI](http://127.0.0.1:8000/docs) and try `POST /classify` with this request.

```json
{"message": "I was charged twice for the same order."}
```

A typical response looks like this.

```json
{
  "category": "Billing",
  "probability": 0.95,
  "reason": "The customer reports a duplicate charge."
}
```

The probability is the model's confidence estimate. Its value and the reason can vary by request.

Run the automated tests with the following command. The suite uses simulated OpenAI responses and requires no credentials.

```powershell
.\myenv\Scripts\python.exe -m pytest
```
