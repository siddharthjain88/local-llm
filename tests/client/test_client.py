"""
Tests for GPT-OSS-20B API Client
Tests API connectivity, client functions, and error handling
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

# Add client directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../client")))

import client


class TestClientConnectivity:
    """Test client connectivity to API server"""

    def test_check_health_success(self):
        """Test successful health check"""
        with patch("client.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "ollama_server": True,
                "model_available": True,
                "timestamp": "2024-01-01T00:00:00",
            }
            mock_get.return_value = mock_response

            result = client.check_health()
            assert result is True
            mock_get.assert_called_once_with(f"{client.API_BASE_URL}/health")

    def test_check_health_server_down(self):
        """Test health check when server is down"""
        with patch("client.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError()

            result = client.check_health()
            assert result is False

    def test_check_health_unhealthy(self):
        """Test health check when server is unhealthy"""
        with patch("client.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "unhealthy",
                "ollama_server": False,
                "model_available": False,
                "timestamp": "2024-01-01T00:00:00",
            }
            mock_get.return_value = mock_response

            result = client.check_health()
            assert result is False


class TestClientModelFunctions:
    """Test client model interaction functions"""

    def test_list_models_success(self):
        """Test successful model listing"""
        with patch("client.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "gpt-oss:20b", "object": "model"}]}
            mock_get.return_value = mock_response

            result = client.list_models()
            assert len(result) == 1
            assert result[0]["id"] == "gpt-oss:20b"

    def test_list_models_error(self):
        """Test model listing with error"""
        with patch("client.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")

            result = client.list_models()
            assert result == []


class TestClientCompletions:
    """Test client completion functions"""

    def test_text_completion_success(self):
        """Test successful text completion"""
        with patch("client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"text": "Paris"}],
                "usage": {"total_tokens": 10},
            }
            mock_post.return_value = mock_response

            result = client.text_completion("The capital of France is")
            assert result == "Paris"

            # Verify the request was made correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["prompt"] == "The capital of France is"
            assert "max_tokens" in call_args[1]["json"]
            assert "temperature" in call_args[1]["json"]

    def test_text_completion_error(self):
        """Test text completion with error"""
        with patch("client.requests.post") as mock_post:
            mock_post.side_effect = Exception("API error")

            result = client.text_completion("Test prompt")
            assert result is None

    def test_chat_completion_success(self):
        """Test successful chat completion"""
        messages = [{"role": "user", "content": "Hello"}]

        with patch("client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hello! How can I help?"}}],
                "usage": {"total_tokens": 15},
            }
            mock_post.return_value = mock_response

            result = client.chat_completion(messages)
            assert result == "Hello! How can I help?"

            # Verify the request was made correctly
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]["json"]["messages"] == messages

    def test_chat_completion_with_parameters(self):
        """Test chat completion with custom parameters"""
        messages = [{"role": "user", "content": "Test"}]

        with patch("client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
            mock_post.return_value = mock_response

            result = client.chat_completion(messages, max_tokens=100, temperature=0.5)

            # Verify parameters were passed
            call_args = mock_post.call_args
            assert call_args[1]["json"]["max_tokens"] == 100
            assert call_args[1]["json"]["temperature"] == 0.5

    def test_chat_completion_error_handling(self):
        """Test chat completion error handling"""
        with patch("client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
            mock_post.return_value = mock_response

            result = client.chat_completion([{"role": "user", "content": "Test"}])
            assert result is None


class TestClientIntegration:
    """Integration tests for client (requires running server)"""

    @pytest.mark.integration
    def test_real_server_connectivity(self):
        """Test actual connectivity to running server"""
        try:
            # This test requires the server to be running
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                result = client.check_health()
                assert isinstance(result, bool)
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running - skipping integration test")

    @pytest.mark.integration
    def test_real_completion_request(self):
        """Test actual completion request to running server"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            data = response.json()
            if data.get("status") != "healthy":
                pytest.skip("Server not healthy - skipping integration test")

            # Try a simple completion
            result = client.text_completion("Hello", max_tokens=10)
            if result:
                assert isinstance(result, str)
                assert len(result) > 0
        except requests.exceptions.ConnectionError:
            pytest.skip("Server not running - skipping integration test")


class TestClientEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_messages(self):
        """Test chat completion with empty messages"""
        with patch("client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 422  # Validation error
            mock_post.return_value = mock_response

            result = client.chat_completion([])
            assert result is None

    def test_invalid_api_url(self):
        """Test with invalid API URL"""
        original_url = client.API_BASE_URL
        try:
            client.API_BASE_URL = "http://invalid-url-12345"

            with patch("client.requests.get") as mock_get:
                mock_get.side_effect = requests.exceptions.ConnectionError()

                result = client.check_health()
                assert result is False
        finally:
            client.API_BASE_URL = original_url

    def test_large_max_tokens(self):
        """Test with large max_tokens value"""
        with patch("client.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"text": "Response text"}]}
            mock_post.return_value = mock_response

            result = client.text_completion("Test", max_tokens=4096)
            assert result == "Response text"

            # Verify max_tokens was passed
            call_args = mock_post.call_args
            assert call_args[1]["json"]["max_tokens"] == 4096
