# Local LLM API

OpenAI-compatible REST API for running GPT-OSS-20B locally via Ollama.

## About GPT-OSS-20B

[GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) is OpenAI's open-weight model designed for local and specialized use cases:

- **Parameters**: 21B total, 3.6B active (Mixture of Experts architecture)
- **License**: Apache 2.0 (commercial-friendly)
- **Memory**: Fits in 16GB RAM with MXFP4 quantization
- **Features**: Chain-of-thought reasoning, function calling, tool use

## Project Structure

```
.
├── server/          # FastAPI server with OpenAI-compatible endpoints
├── client/          # Python client library (works with OpenAI SDK)
└── setup/           # Setup utilities and test scripts
```

## Prerequisites

1. **System Requirements**
   - 16GB+ RAM (24GB recommended)
   - macOS, Linux, or Windows
   - Python 3.10+

2. **Install Ollama**
   ```bash
   # macOS
   brew install ollama

   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   ```

3. **Download the Model**
   ```bash
   ollama pull gpt-oss:20b
   ```

## Quick Start

### Terminal 1: Start Ollama
```bash
ollama serve
```

### Terminal 2: Start API Server
```bash
cd server
pip install -r requirements.txt
python app.py
```

### Terminal 3: Use the API
```bash
# With curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# With Python client
cd client
pip install -r requirements.txt
python client.py
```

## Using with OpenAI SDK

The API is fully compatible with the official OpenAI Python SDK:

```python
from openai import OpenAI

# Point to local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Local server doesn't require auth
)

# Use exactly like OpenAI's API
response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing briefly."}
    ]
)
print(response.choices[0].message.content)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion (with streaming) |
| `/v1/completions` | POST | Text completion (with streaming) |

All endpoints are OpenAI-compatible for easy integration with existing tools.

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCAL_LLM_MODEL` | `gpt-oss:20b` | Default model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |

## Features

- OpenAI-compatible API endpoints
- Streaming support (Server-Sent Events)
- Health monitoring for Ollama and model status
- Interactive API documentation at `/docs`
- Python client with OpenAI SDK compatibility
- Environment-based configuration

## Development

### Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### Run Tests
```bash
make test
```

### Code Quality
```bash
make lint      # Run linters
make format    # Auto-format code
make check     # Run all checks
```

## License

Apache 2.0 (same as GPT-OSS-20B model)

## Resources

- [GPT-OSS-20B on Hugging Face](https://huggingface.co/openai/gpt-oss-20b)
- [OpenAI GPT-OSS Cookbook](https://cookbook.openai.com/topic/gpt-oss)
- [Ollama Documentation](https://ollama.com/docs)
