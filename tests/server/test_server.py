"""
Tests for GPT-OSS-20B API Server
Tests connectivity, health checks, and LLM response intelligence
"""

import json
import re
import time
from typing import Any, Dict

import pytest
import requests

# Test configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30


class TestServerHealth:
    """Test server health and connectivity"""

    def test_server_reachable(self):
        """Test that the server is running and reachable"""
        try:
            response = requests.get(f"{API_BASE_URL}/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "GPT-OSS-20B" in data["message"]
        except requests.exceptions.ConnectionError:
            pytest.fail("Server is not running. Please start with: ./start_server.sh")

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "ollama_server" in data
        assert "model_available" in data
        assert "timestamp" in data

        # Check if system is healthy
        if not data["ollama_server"]:
            pytest.skip("Ollama server not running - skipping")
        if not data["model_available"]:
            pytest.skip("Model not available - please run: ollama pull gpt-oss:20b")

        assert data["status"] == "healthy"

    def test_models_endpoint(self):
        """Test the models listing endpoint"""
        response = requests.get(f"{API_BASE_URL}/v1/models", timeout=5)
        assert response.status_code == 200

        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

        # Check if model is available
        model_ids = [model["id"] for model in data["data"]]
        if "gpt-oss:20b" not in model_ids:
            pytest.skip("GPT-OSS-20B model not found")


class TestLLMResponses:
    """Test LLM response quality and intelligence"""

    def test_completion_endpoint_basic(self):
        """Test basic text completion functionality"""
        payload = {
            "prompt": "The capital of France is",
            "max_tokens": 50,
            "temperature": 0.1,  # Low temperature for more deterministic output
        }

        response = requests.post(f"{API_BASE_URL}/v1/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "id" in data
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]

        # Check for intelligent response (should mention Paris)
        response_text = data["choices"][0]["text"].lower()
        assert any(word in response_text for word in ["paris", "capitale"])

    def test_chat_completion_endpoint_basic(self):
        """Test basic chat completion functionality"""
        payload = {
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "id" in data
        assert "model" in data
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        # Check for intelligent response (should mention 4)
        response_text = data["choices"][0]["message"]["content"].lower()
        assert "4" in response_text or "four" in response_text

    def test_llm_reasoning_capability(self):
        """Test that LLM can perform basic reasoning"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "If I have 3 apples and buy 2 more, how many do I have?",
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1,
        }

        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        response_text = data["choices"][0]["message"]["content"].lower()
        # Should contain 5 or five
        assert "5" in response_text or "five" in response_text

    def test_llm_context_understanding(self):
        """Test that LLM understands context from conversation"""
        payload = {
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 50,
            "temperature": 0.1,
        }

        response = requests.post(
            f"{API_BASE_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT
        )

        assert response.status_code == 200
        data = response.json()

        response_text = data["choices"][0]["message"]["content"].lower()
        # Should mention Alice
        assert "alice" in response_text

    def test_response_coherence(self):
        """Test that responses are coherent and well-formed"""
        payload = {
            "prompt": "Write a one-sentence description of Python programming language:",
            "max_tokens": 100,
            "temperature": 0.3,
        }

        response = requests.post(f"{API_BASE_URL}/v1/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        response_text = data["choices"][0]["text"].strip()

        # Check for coherent response characteristics
        assert len(response_text) > 10  # Not empty or too short
        assert len(response_text.split()) > 3  # Multiple words

        # Should mention Python-related terms
        response_lower = response_text.lower()
        python_terms = ["python", "programming", "language", "code", "script", "high-level"]
        assert any(term in response_lower for term in python_terms)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_endpoint(self):
        """Test that invalid endpoints return proper error"""
        response = requests.get(f"{API_BASE_URL}/invalid/endpoint", timeout=5)
        assert response.status_code == 404

    def test_missing_required_fields(self):
        """Test that missing required fields return proper error"""
        # Missing prompt/messages
        payload = {"max_tokens": 50}

        response = requests.post(f"{API_BASE_URL}/v1/completions", json=payload, timeout=5)

        assert response.status_code == 422  # Unprocessable Entity

    def test_invalid_temperature(self):
        """Test that invalid temperature values are handled"""
        payload = {"prompt": "Test", "temperature": 3.0, "max_tokens": 10}  # Invalid (>2.0)

        response = requests.post(f"{API_BASE_URL}/v1/completions", json=payload, timeout=5)

        assert response.status_code == 422  # Validation error


class TestPerformance:
    """Test performance and response times"""

    def test_response_time_health(self):
        """Test that health check responds quickly"""
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond in less than 1 second

    def test_response_includes_usage(self):
        """Test that responses include token usage information"""
        payload = {"prompt": "Hello", "max_tokens": 10}

        response = requests.post(f"{API_BASE_URL}/v1/completions", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]
        assert data["usage"]["total_tokens"] > 0
