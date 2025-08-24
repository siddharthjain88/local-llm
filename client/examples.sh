#!/bin/bash

# GPT-OSS-20B API Examples using curl
# Make sure the API server is running: python api_server.py

API_URL="http://localhost:8000"

echo "GPT-OSS-20B API Examples"
echo "========================"

# Health check
echo -e "\n1. Health Check:"
curl -s "$API_URL/health" | python -m json.tool

# List models
echo -e "\n2. List Available Models:"
curl -s "$API_URL/v1/models" | python -m json.tool

# Text completion
echo -e "\n3. Text Completion:"
curl -s -X POST "$API_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.7
  }' | python -m json.tool

# Chat completion
echo -e "\n4. Chat Completion:"
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Python?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool

# Chat with context
echo -e "\n5. Multi-turn Chat:"
curl -s -X POST "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "2+2 equals 4."},
      {"role": "user", "content": "What about 3+3?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | python -m json.tool