#!/usr/bin/env python3
"""
GPT-OSS-20B API Client
Python client library for interacting with the GPT-OSS-20B API server
"""

import json
from typing import Any, Dict, List

import requests

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_health():
    """Check API server health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        data = response.json()
        print("API Health Check:")
        print(f"  Status: {data['status']}")
        print(f"  Ollama Server: {'✓' if data['ollama_server'] else '✗'}")
        print(f"  Model Available: {'✓' if data['model_available'] else '✗'}")
        return data["status"] == "healthy"
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running!")
        print("Start it with: cd server && python app.py")
        return False


def list_models():
    """List available models"""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/models")
        data = response.json()
        print("\nAvailable Models:")
        for model in data["data"]:
            print(f"  - {model['id']}")
        return data["data"]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []


def text_completion(prompt: str, max_tokens: int = 256, temperature: float = 0.7):
    """Send a text completion request"""
    endpoint = f"{API_BASE_URL}/v1/completions"

    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": "gpt-oss:20b",
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"]
    except Exception as e:
        print(f"Error: {e}")
        return None


def chat_completion(
    messages: List[Dict[str, str]], max_tokens: int = 256, temperature: float = 0.7
):
    """Send a chat completion request"""
    endpoint = f"{API_BASE_URL}/v1/chat/completions"

    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": "gpt-oss:20b",
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error: {e}")
        return None


def interactive_chat():
    """Interactive chat session using the API"""
    print("\n" + "=" * 50)
    print("Interactive Chat (via API)")
    print("Type 'exit' to quit")
    print("=" * 50)

    messages = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Get response from API
        print("GPT-OSS: ", end="", flush=True)
        response = chat_completion(messages)

        if response:
            print(response)
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
        else:
            print("Failed to get response from API")
            messages.pop()  # Remove the user message if we failed


def main():
    print("GPT-OSS-20B API Client Example")
    print("=" * 50)

    # Check health
    if not check_health():
        print("\n⚠️  Please ensure:")
        print("1. Ollama server is running: ollama serve")
        print("2. API server is running: python api_server.py")
        print("3. Model is downloaded: ollama pull gpt-oss:20b")
        return

    # List models
    models = list_models()

    if not models:
        print("\n⚠️  No models available. Run: ollama pull gpt-oss:20b")
        return

    # Example 1: Text Completion
    print("\n" + "=" * 50)
    print("Example 1: Text Completion")
    print("=" * 50)

    prompt = "The capital of France is"
    print(f"Prompt: {prompt}")
    print("Response: ", end="", flush=True)

    result = text_completion(prompt, max_tokens=50)
    if result:
        print(result)

    # Example 2: Chat Completion
    print("\n" + "=" * 50)
    print("Example 2: Chat Completion")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about programming"},
    ]

    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")

    print("\nResponse: ", end="", flush=True)
    result = chat_completion(messages, max_tokens=100)
    if result:
        print(result)

    # Example 3: Interactive Chat
    print("\n" + "=" * 50)
    print("Example 3: Interactive Chat")

    response = input("\nWould you like to start an interactive chat? (y/n): ").lower()
    if response == "y":
        interactive_chat()


if __name__ == "__main__":
    main()
