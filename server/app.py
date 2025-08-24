#!/usr/bin/env python3
"""
FastAPI server for GPT-OSS-20B model
Provides REST API endpoints for interacting with the model via Ollama
"""

import asyncio
import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="GPT-OSS-20B API",
    description="REST API for interacting with GPT-OSS-20B model through Ollama",
    version="1.0.0",
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(default="gpt-oss:20b", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        default=256, ge=1, le=4096, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Stream the response")


class ChatResponse(BaseModel):
    id: str = Field(..., description="Unique response ID")
    model: str = Field(..., description="Model used")
    created: int = Field(..., description="Unix timestamp")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    model: str = Field(default="gpt-oss:20b", description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        default=256, ge=1, le=4096, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Stream the response")


class CompletionResponse(BaseModel):
    id: str = Field(..., description="Unique response ID")
    model: str = Field(..., description="Model used")
    created: int = Field(..., description="Unix timestamp")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    owned_by: str = Field(default="openai", description="Model owner")


def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=2)
        return result.returncode == 0
    except:
        return False


def check_model_available():
    """Check if GPT-OSS-20B model is available"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return "gpt-oss:20b" in result.stdout
    except:
        return False


async def run_ollama_generate(
    prompt: str, model: str = "gpt-oss:20b", temperature: float = 0.7, max_tokens: int = 256
):
    """Run Ollama generate command asynchronously"""
    cmd = ["ollama", "run", model, "--verbose", prompt]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Model error: {stderr.decode()}")

        return stdout.decode().strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Check requirements on startup"""
    if not check_ollama_server():
        print("WARNING: Ollama server is not running!")
        print("Please run 'ollama serve' in a separate terminal")

    if not check_model_available():
        print("WARNING: GPT-OSS-20B model not found!")
        print("Please run 'ollama pull gpt-oss:20b' to download the model")


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "GPT-OSS-20B API Server",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completion": "/v1/completions",
            "models": "/v1/models",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_running = check_ollama_server()
    model_available = check_model_available()

    status = "healthy" if (ollama_running and model_available) else "unhealthy"

    return {
        "status": status,
        "ollama_server": ollama_running,
        "model_available": model_available,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)"""
    models = []

    if check_model_available():
        models.append(ModelInfo(id="gpt-oss:20b", created=int(datetime.utcnow().timestamp())))

    return {"data": models, "object": "list"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint (OpenAI-compatible)"""
    if not check_ollama_server():
        raise HTTPException(status_code=503, detail="Ollama server is not running")

    if not check_model_available():
        raise HTTPException(
            status_code=404, detail="Model not found. Run 'ollama pull gpt-oss:20b'"
        )

    # Build conversation prompt from messages
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n"

    prompt += "Assistant: "

    # Generate response
    response_text = await run_ollama_generate(
        prompt=prompt,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens or 256,
    )

    # Format response in OpenAI-compatible format
    response = ChatResponse(
        id=f"chatcmpl-{int(datetime.utcnow().timestamp())}",
        model=request.model,
        created=int(datetime.utcnow().timestamp()),
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(prompt.split()) + len(response_text.split()),
        },
    )

    return response


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI-compatible)"""
    if not check_ollama_server():
        raise HTTPException(status_code=503, detail="Ollama server is not running")

    if not check_model_available():
        raise HTTPException(
            status_code=404, detail="Model not found. Run 'ollama pull gpt-oss:20b'"
        )

    # Generate response
    response_text = await run_ollama_generate(
        prompt=request.prompt,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens or 256,
    )

    # Format response in OpenAI-compatible format
    response = CompletionResponse(
        id=f"cmpl-{int(datetime.utcnow().timestamp())}",
        model=request.model,
        created=int(datetime.utcnow().timestamp()),
        choices=[{"index": 0, "text": response_text, "finish_reason": "stop"}],
        usage={
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(request.prompt.split()) + len(response_text.split()),
        },
    )

    return response


def main():
    """Main function to run the API server"""
    print("Starting GPT-OSS-20B API Server...")
    print("=" * 50)

    # Check requirements
    if not check_ollama_server():
        print("\n⚠️  WARNING: Ollama server is not running!")
        print("Please run 'ollama serve' in a separate terminal\n")

    if not check_model_available():
        print("\n⚠️  WARNING: GPT-OSS-20B model not found!")
        print("Please run 'ollama pull gpt-oss:20b' to download the model\n")

    print("API Server starting at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("=" * 50)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
