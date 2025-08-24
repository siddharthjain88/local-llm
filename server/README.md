# GPT-OSS-20B API Server

FastAPI server providing OpenAI-compatible REST API endpoints for GPT-OSS-20B model via Ollama.

## Prerequisites

1. **Ollama** must be installed and running:
   ```bash
   ollama serve
   ```

2. **GPT-OSS-20B model** must be downloaded:
   ```bash
   ollama pull gpt-oss:20b
   ```

## Installation

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Server

```bash
source venv/bin/activate
python app.py
```

Server will start at `http://localhost:8000`

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion (OpenAI-compatible)
- `POST /v1/chat/completions` - Chat completion (OpenAI-compatible)

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## Example Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```