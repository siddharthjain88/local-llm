#!/usr/bin/env python3
"""
Local LLM Client
Python client library for interacting with the Local LLM API server.

This client uses the official OpenAI Python SDK, demonstrating that our
local API is fully compatible with OpenAI's client libraries.

Usage:
    # With official OpenAI SDK (recommended)
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

    # Or use this module's convenience functions
    from client import chat, complete
"""

import os
from typing import Generator, List, Dict

# Try to import OpenAI SDK, fall back to httpx if not available
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

import httpx

# Configuration
API_BASE_URL = os.getenv("LOCAL_LLM_API_URL", "http://localhost:8000")
DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "gpt-oss:20b")


def get_client() -> "OpenAI":
    """Get an OpenAI client configured for the local API."""
    if not HAS_OPENAI:
        raise ImportError(
            "OpenAI SDK not installed. Install with: pip install openai\n"
            "Or use the httpx-based functions instead."
        )
    return OpenAI(
        base_url=f"{API_BASE_URL}/v1",
        api_key="not-needed",  # Local server doesn't require auth
    )


def check_health() -> dict:
    """Check API server health."""
    with httpx.Client(timeout=5.0) as client:
        try:
            response = client.get(f"{API_BASE_URL}/health")
            data = response.json()
            print("API Health Check:")
            print(f"  Status: {data['status']}")
            print(f"  Ollama Server: {'Connected' if data['ollama_server'] else 'Not running'}")
            print(f"  Model Available: {'Yes' if data['model_available'] else 'No'}")
            print(f"  Configured Model: {data.get('configured_model', 'unknown')}")
            if data.get('available_models'):
                print(f"  Available Models: {', '.join(data['available_models'])}")
            return data
        except httpx.ConnectError:
            print("API server is not running!")
            print(f"Start it with: cd server && python app.py")
            print(f"Expected URL: {API_BASE_URL}")
            return {"status": "unreachable"}


def list_models() -> List[str]:
    """List available models."""
    if HAS_OPENAI:
        client = get_client()
        models = client.models.list()
        model_ids = [m.id for m in models.data]
    else:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_BASE_URL}/v1/models")
            data = response.json()
            model_ids = [m["id"] for m in data.get("data", [])]

    print("\nAvailable Models:")
    for model_id in model_ids:
        print(f"  - {model_id}")
    return model_ids


def chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Send a chat completion request.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model to use (default: gpt-oss:20b)
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response

    Returns:
        Response text (or generator if streaming)

    Example:
        response = chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ])
        print(response)
    """
    if HAS_OPENAI:
        client = get_client()
        if stream:
            def stream_response():
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return stream_response()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
    else:
        # Fallback to httpx
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_BASE_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,  # httpx fallback doesn't support streaming
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


def complete(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """
    Send a text completion request.

    Args:
        prompt: Input prompt text
        model: Model to use (default: gpt-oss:20b)
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response

    Returns:
        Response text (or generator if streaming)

    Example:
        response = complete("The capital of France is")
        print(response)
    """
    if HAS_OPENAI:
        client = get_client()
        if stream:
            def stream_response():
                response = client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices[0].text:
                        yield chunk.choices[0].text
            return stream_response()
        else:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].text
    else:
        # Fallback to httpx
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{API_BASE_URL}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"]


def interactive_chat(model: str = DEFAULT_MODEL):
    """Interactive chat session."""
    print("\n" + "=" * 60)
    print(f"Interactive Chat with {model}")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'clear' to start a new conversation")
    print("=" * 60)

    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = []
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)

        try:
            # Use streaming for interactive chat
            response_text = ""
            for chunk in chat(messages, model=model, stream=True):
                print(chunk, end="", flush=True)
                response_text += chunk
            print()  # Newline after response

            messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            print(f"\nError: {e}")
            messages.pop()  # Remove failed user message


def main():
    """Main function demonstrating client usage."""
    print("=" * 60)
    print("Local LLM Client")
    print("=" * 60)

    # Check health
    health = check_health()
    if health.get("status") == "unreachable":
        print("\nPlease ensure:")
        print("1. Ollama server is running: ollama serve")
        print("2. API server is running: cd server && python app.py")
        print(f"3. Model is downloaded: ollama pull {DEFAULT_MODEL}")
        return

    if health.get("status") == "unhealthy":
        print("\nWarning: Server is running but Ollama may not be available.")

    # List models
    models = list_models()

    if not models:
        print(f"\nNo models available. Run: ollama pull {DEFAULT_MODEL}")
        return

    # Example 1: Chat Completion
    print("\n" + "=" * 60)
    print("Example 1: Chat Completion")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is GPT-OSS-20B?"},
    ]

    print("Messages:")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content']}")

    print("\nResponse: ", end="", flush=True)
    response = chat(messages, max_tokens=200)
    print(response)

    # Example 2: Text Completion
    print("\n" + "=" * 60)
    print("Example 2: Text Completion")
    print("=" * 60)

    prompt = "The three main features of GPT-OSS-20B are:"
    print(f"Prompt: {prompt}")
    print("\nResponse: ", end="", flush=True)

    response = complete(prompt, max_tokens=150)
    print(response)

    # Example 3: Streaming
    print("\n" + "=" * 60)
    print("Example 3: Streaming Response")
    print("=" * 60)

    print("Prompt: Write a haiku about local AI models\n")
    print("Response: ", end="", flush=True)

    for chunk in chat(
        [{"role": "user", "content": "Write a haiku about local AI models"}],
        stream=True,
        max_tokens=100,
    ):
        print(chunk, end="", flush=True)
    print()

    # Interactive chat option
    print("\n" + "=" * 60)
    response = input("\nWould you like to start an interactive chat? (y/n): ").lower()
    if response in ("y", "yes"):
        interactive_chat()


if __name__ == "__main__":
    main()
