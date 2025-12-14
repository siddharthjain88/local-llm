# CLAUDE.md - Project Context for Claude Code

## Project Overview

Local LLM API is an OpenAI-compatible REST API server for running GPT-OSS-20B locally via Ollama. It provides drop-in compatible endpoints with the OpenAI API, allowing you to use local LLMs with existing OpenAI client libraries.

### GPT-OSS-20B Model

OpenAI's open-weight model ([Hugging Face](https://huggingface.co/openai/gpt-oss-20b)):
- 21B total parameters, 3.6B active (Mixture of Experts)
- Apache 2.0 license (commercial-friendly)
- MXFP4 quantization (fits in 16GB RAM)
- Chain-of-thought reasoning with configurable effort levels
- Agentic capabilities (function calling, tool use)

## Architecture

### Components

1. **Server** (`/server`)
   - FastAPI application with async request handling
   - OpenAI-compatible REST endpoints
   - Streaming support via Server-Sent Events
   - Health monitoring with Ollama status
   - Uses httpx for async Ollama API calls

2. **Client** (`/client`)
   - Python client library
   - Compatible with official OpenAI SDK
   - Interactive CLI with streaming
   - Fallback to httpx when OpenAI SDK not installed

3. **Setup Utilities** (`/setup`)
   - Model setup scripts
   - Testing utilities
   - Direct Ollama/Transformers interfaces

### Technology Stack

- **Model Runtime**: Ollama
- **API Framework**: FastAPI + Uvicorn
- **HTTP Client**: httpx (async)
- **Client SDK**: OpenAI Python SDK compatible
- **Model**: GPT-OSS-20B

## Key Files

```
.
├── server/
│   ├── app.py              # FastAPI application (main server)
│   └── requirements.txt    # Server dependencies
├── client/
│   ├── client.py           # Python client library
│   └── requirements.txt    # Client dependencies (includes openai)
├── tests/                  # Test suites
└── setup/                  # Setup utilities
```

## Development Guidelines

### Code Standards
- Python 3.10+ with type hints
- PEP 8 style (enforced by ruff/black)
- Async/await for all I/O operations
- Environment variables for configuration

### API Design
- Maintain strict OpenAI API compatibility
- Use Pydantic models for validation
- Support both streaming and non-streaming
- Include proper error responses

## Quick Commands

```bash
# Start Ollama
ollama serve

# Start API server
cd server && python app.py

# Run client examples
cd client && python client.py

# Run tests
make test

# Health check
curl http://localhost:8000/health
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_MODEL` | `gpt-oss:20b` | Default model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama URL |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |

## Common Tasks

### Add New Endpoint
1. Define Pydantic request/response models in `server/app.py`
2. Add async endpoint function
3. Update client with corresponding function
4. Add tests

### Change Default Model
1. Set `LOCAL_LLM_MODEL` environment variable
2. Ensure model is pulled: `ollama pull model:tag`

### Debug Issues
1. Check Ollama: `curl http://localhost:11434/api/tags`
2. Check API: `curl http://localhost:8000/health`
3. Check server logs for errors
