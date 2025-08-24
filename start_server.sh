#!/bin/bash

# Start script for GPT-OSS-20B API Server

echo "GPT-OSS-20B API Server Launcher"
echo "================================"

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama server is not running!"
    echo "Please start it in another terminal with: ollama serve"
    echo ""
    read -p "Press Enter when Ollama is running, or Ctrl+C to exit..."
fi

# Check if model is available
if ! ollama list | grep -q "gpt-oss:20b"; then
    echo "⚠️  GPT-OSS-20B model not found!"
    echo "Downloading model (this will take a few minutes)..."
    ollama pull gpt-oss:20b
fi

# Start the API server
echo ""
echo "Starting API server..."
echo "Server will be available at: http://localhost:8000"
echo "API docs will be at: http://localhost:8000/docs"
echo ""

cd server
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python app.py