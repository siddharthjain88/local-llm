# GPT-OSS-20B API Client

Python client for interacting with the GPT-OSS-20B API server.

## Installation

```bash
cd client
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Prerequisites

The API server must be running at `http://localhost:8000`. See the server README for setup instructions.

## Usage

### Running the Interactive Client

```bash
source venv/bin/activate
python client.py
```

This will:
1. Check server health
2. List available models
3. Run example completions
4. Optionally start an interactive chat session

### Using as a Library

```python
from client import chat_completion, text_completion

# Text completion
response = text_completion("The capital of France is")
print(response)

# Chat completion
messages = [
    {"role": "user", "content": "What is 2+2?"}
]
response = chat_completion(messages)
print(response)
```

## Features

- Health checks
- Model listing
- Text completions
- Chat completions
- Interactive chat mode
- Token usage tracking