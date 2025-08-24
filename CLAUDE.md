# CLAUDE.md - Project Context for Claude Code

## Project Overview
This project provides a production-ready API server and client for running OpenAI's GPT-OSS-20B model locally. It uses Ollama as the model runtime and provides OpenAI-compatible REST API endpoints through FastAPI.

## Architecture

### Components
1. **Server** (`/server`)
   - FastAPI application
   - OpenAI-compatible endpoints
   - Health monitoring
   - Async request handling

2. **Client** (`/client`)
   - Python client library
   - Interactive CLI
   - Example scripts

3. **Setup Utilities** (`/setup`)
   - Model setup scripts
   - Testing utilities
   - Direct Ollama/Transformers interfaces

### Technology Stack
- **Model Runtime**: Ollama
- **API Framework**: FastAPI + Uvicorn
- **Client**: Python requests/httpx
- **Model**: GPT-OSS-20B (21B params, 3.6B active, MoE architecture)

## Key Files and Directories

```
.
├── server/
│   ├── app.py              # Main FastAPI application
│   ├── requirements.txt    # Server dependencies
│   └── README.md           # Server documentation
├── client/
│   ├── client.py           # Python client library
│   ├── examples.sh         # Curl examples
│   ├── requirements.txt    # Client dependencies
│   └── README.md           # Client documentation
└── setup/
    ├── run_with_ollama.py  # Quick Ollama test
    ├── test_gpt_oss.py     # Transformers test
    └── setup_gpt_oss.py    # Model downloader
```

## Development Guidelines

### Code Standards
- Use type hints for all function signatures
- Follow PEP 8 style guidelines
- Add docstrings to all public functions
- Handle errors gracefully with informative messages

### API Design
- Maintain OpenAI API compatibility
- Use pydantic models for request/response validation
- Implement proper error codes and messages
- Include health checks for dependencies

## Testing

### Quick Testing Commands
```bash
# Run health check
./health_check.py
# or
make health

# Run all tests
./run_tests.sh
# or
make test

# Run specific test types
./run_tests.sh quick      # Fast unit tests only
./run_tests.sh server     # Server tests
./run_tests.sh client     # Client tests
./run_tests.sh coverage   # With coverage report
./run_tests.sh lint       # Linting checks
./run_tests.sh all        # Everything
```

### Test Structure
```
tests/
├── server/
│   └── test_server.py    # Server tests (health, LLM responses)
└── client/
    └── test_client.py    # Client tests (connectivity, API calls)
```

### Test Categories
1. **Unit Tests** - Test individual functions in isolation
2. **Integration Tests** - Test with running server (marked with `@pytest.mark.integration`)
3. **Health Checks** - Verify system components
4. **LLM Intelligence Tests** - Validate response coherence and reasoning

### Code Quality Tools
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **black** - Code formatting
- **isort** - Import sorting
- **flake8** - Style guide enforcement
- **ruff** - Fast linting
- **mypy** - Type checking
- **pylint** - Code analysis
- **bandit** - Security scanning

## Deployment

### Local Development
1. Start Ollama: `ollama serve`
2. Start API server: `cd server && python app.py`
3. Use client: `cd client && python client.py`

### Production Considerations
- Use environment variables for configuration
- Implement rate limiting
- Add authentication if exposed publicly
- Use proper logging instead of print statements
- Consider containerization with Docker

## Common Tasks

### Add New Endpoint
1. Define pydantic models in `server/app.py`
2. Add endpoint function with proper decorators
3. Update client with corresponding function
4. Document in README

### Update Model
1. Pull new model: `ollama pull model:tag`
2. Update model ID in server and client code
3. Test all endpoints

### Debug Issues
1. Check Ollama server: `ollama list`
2. Check API health: `curl localhost:8000/health`
3. Review server logs for errors
4. Verify model is downloaded and available

## Known Issues
- Ollama server must be started manually before API server
- Model responses include "thinking" process (feature of GPT-OSS)
- First request after server start may be slow (model loading)
- Deprecation warnings in datetime usage (should update to timezone-aware)

## Additional Context
- GPT-OSS-20B uses MXFP4 quantization to fit in 16GB memory
- The model has characteristic "thinking" output before responses
- API is designed to be drop-in compatible with OpenAI clients
- Project uses virtual environments to avoid system package conflicts