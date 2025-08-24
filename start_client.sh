#!/bin/bash

# Start script for GPT-OSS-20B API Client

echo "GPT-OSS-20B API Client"
echo "======================"

# Check if API server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  API server is not running!"
    echo "Please start it first with: ./start_server.sh"
    exit 1
fi

echo "✅ API server is running"
echo ""

# Start the client
cd client
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python client.py