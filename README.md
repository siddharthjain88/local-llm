# GPT-OSS-20B Local API

A production-ready API server and client for running OpenAI's GPT-OSS-20B model locally via Ollama.

## Project Structure

```
.
├── server/          # FastAPI server with OpenAI-compatible endpoints
├── client/          # Python client library and examples
└── setup/           # Setup utilities and test scripts
```

## Prerequisites

1. **System Requirements**
   - 16GB+ RAM (24GB recommended)
   - macOS, Linux, or Windows
   - Python 3.8+

2. **Ollama Setup**
   ```bash
   # Install Ollama
   brew install ollama  # macOS
   
   # Start Ollama server
   ollama serve
   
   # Download GPT-OSS-20B model (13GB)
   ollama pull gpt-oss:20b
   ```

## Quick Start

### Easy Method (Recommended)

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start API Server
./start_server.sh

# Terminal 3: Run Client
./start_client.sh
```

### Manual Method

#### 1. Start the API Server

```bash
cd server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Server runs at `http://localhost:8000` with docs at `/docs`

#### 2. Use the Client

```bash
cd client
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python client.py
```

## API Endpoints

- `GET /health` - Server health check
- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion

All endpoints are OpenAI-compatible for easy integration.

## Example Usage

### Curl
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Python
```python
from client.client import chat_completion

response = chat_completion([
    {"role": "user", "content": "What is 2+2?"}
])
print(response)
```

## Features

- ✅ OpenAI-compatible API endpoints
- ✅ Health monitoring for Ollama and model status
- ✅ Interactive API documentation (FastAPI)
- ✅ Python client library with examples
- ✅ Token usage tracking
- ✅ Error handling with helpful messages

## Architecture

- **Server**: FastAPI application providing REST API
- **Client**: Python library for API interaction
- **Model**: GPT-OSS-20B (21B parameters, 3.6B active) via Ollama

## Testing & Quality Assurance

### Quick Test Commands

```bash
# Install development dependencies
make setup-dev

# Run all tests
make test

# Run specific test suites
make test-server      # Server tests only
make test-client      # Client tests only
make test-integration # Integration tests (requires running server)

# Run with coverage
make test-coverage

# Code quality checks
make lint            # Run all linters
make format          # Auto-format code
make security        # Security scan
make check           # Run all checks (format, lint, test)
```

### Manual Testing

1. **Health Check**:
   ```bash
   make health
   ```

2. **Full Test Suite**:
   ```bash
   make all  # Clean, install, and run all checks
   ```

3. **CI Pipeline Locally**:
   ```bash
   make ci  # Run the full CI pipeline
   ```

### Code Quality Tools

- **Black** - Code formatting
- **isort** - Import sorting
- **flake8** - Style guide enforcement
- **ruff** - Fast Python linter
- **mypy** - Static type checking
- **pylint** - Code analysis
- **bandit** - Security linting
- **pytest** - Testing framework
- **coverage** - Code coverage

### Test Categories

- **Unit Tests**: Test individual functions
- **Integration Tests**: Test with running server (marked with `@pytest.mark.integration`)
- **Health Checks**: Verify system components are running
- **LLM Intelligence Tests**: Validate model responses are coherent

## Development

### Project Structure

```
.
├── server/          # API server
├── client/          # Python client
├── tests/           # Test suites
│   ├── server/     # Server tests
│   └── client/     # Client tests
├── setup/           # Setup utilities
├── Makefile        # Development commands
├── pytest.ini      # Test configuration
├── pyproject.toml  # Python tooling config
└── .github/        # CI/CD workflows
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `make check`
4. Ensure code quality: `make format && make lint`
5. Submit a pull request

## License

Apache 2.0 (GPT-OSS-20B model license)