# pvz-ai

FastAPI + Gradio chat MVP for `openai/gpt-oss-120b`, with SQLAlchemy dialog history,
PostgreSQL storage, Docker, pytest, Playwright smoke coverage, logging, and Vercel deploy
configuration.

## Stack

- FastAPI backend with `/health`, `/api/chat`, and mounted Gradio UI at `/chat`
- Groq OpenAI-compatible API as the default `openai/gpt-oss-120b` provider
- Optional Hugging Face Inference Providers router fallback
- SQLAlchemy async models and Alembic migrations for PostgreSQL
- Docker Compose for local app + Postgres
- GitHub Actions for lint, tests, Playwright smoke, and Vercel deploy

## Environment

Copy `.env.example` to `.env` and fill the values:

```powershell
Copy-Item .env.example .env
```

Minimum real-model setup:

```text
DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST.neon.tech/DBNAME?ssl=require
LLM_PROVIDER=groq
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=...
```

For local UI smoke without a model key, use `LLM_PROVIDER=echo`.

## Local Run

With the virtual environment activated:

```powershell
python -m pip install -r requirements-dev.txt
python -m alembic upgrade head
python -m uvicorn pvz_ai.main:app --reload
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/chat`

If the virtual environment is not activated, use:

```powershell
.\.venv\Scripts\python.exe -m uvicorn pvz_ai.main:app --reload
```

## Docker

The local Windows machine did not have Docker CLI installed during this implementation.
Once Docker Desktop is installed:

```powershell
docker compose up --build
```

The app will be available at `http://127.0.0.1:8000/chat`.

## Tests

```powershell
python -m pytest
python -m playwright install chromium
$env:E2E_BASE_URL="http://127.0.0.1:8000"
python -m pytest e2e
```

## Deploy

Vercel needs these environment variables:

- `DATABASE_URL`
- `LLM_PROVIDER`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `GROQ_API_KEY` or `HF_TOKEN`

GitHub Actions deploy also needs:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Vercel runs the ASGI app through `api/index.py`, while local dev can use
`pvz_ai.main:app` or the top-level `app.py` entrypoint.
