#!/usr/bin/env python3
"""
Local LLM API Server
OpenAI-compatible REST API for running GPT-OSS-20B locally via Ollama.

This server provides drop-in compatible endpoints with the OpenAI API,
allowing you to use local LLMs with existing OpenAI client libraries.

GPT-OSS-20B: OpenAI's open-weight model (21B params, 3.6B active, MoE architecture)
- Apache 2.0 license
- MXFP4 quantization (fits in 16GB RAM)
- Full chain-of-thought reasoning
- Agentic capabilities (function calling, tool use)

See: https://huggingface.co/openai/gpt-oss-20b
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configuration via environment variables
DEFAULT_MODEL = os.getenv("LOCAL_LLM_MODEL", "gpt-oss:20b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# HTTP client for Ollama API
http_client: Optional[httpx.AsyncClient] = None


def check_ollama_server() -> bool:
    """Check if Ollama server is running via HTTP."""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{OLLAMA_HOST}/api/tags")
            return response.status_code == 200
    except Exception:
        return False


def check_model_available() -> bool:
    """Check if the configured model is available in Ollama."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(DEFAULT_MODEL in name for name in model_names)
    except Exception:
        pass
    return False


def get_available_models() -> List[str]:
    """Get list of available models from Ollama."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{OLLAMA_HOST}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
    except Exception:
        pass
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    global http_client

    # Startup
    http_client = httpx.AsyncClient(timeout=120.0)

    print("=" * 60)
    print("Local LLM API Server Starting...")
    print("=" * 60)

    if not check_ollama_server():
        print(f"\nWARNING: Ollama server is not running at {OLLAMA_HOST}")
        print("Please run 'ollama serve' or set OLLAMA_HOST environment variable\n")
    else:
        print(f"Ollama server: Connected at {OLLAMA_HOST}")

    if not check_model_available():
        print(f"\nWARNING: Model '{DEFAULT_MODEL}' not found!")
        print(f"Please run 'ollama pull {DEFAULT_MODEL}' to download the model")
        print(f"Available models: {get_available_models()}\n")
    else:
        print(f"Model: {DEFAULT_MODEL} (ready)")

    print(f"\nAPI Server: http://{API_HOST}:{API_PORT}")
    print(f"API Docs: http://{API_HOST}:{API_PORT}/docs")
    print("=" * 60)

    yield

    # Shutdown
    if http_client:
        await http_client.aclose()


app = FastAPI(
    title="Local LLM API",
    description="""
OpenAI-compatible REST API for running GPT-OSS-20B locally via Ollama.

## Features
- Drop-in replacement for OpenAI API
- Chat completions with streaming support
- Text completions
- Health monitoring

## Model Info
GPT-OSS-20B is OpenAI's open-weight model:
- 21B total parameters, 3.6B active (MoE)
- Apache 2.0 license
- MXFP4 quantization (16GB RAM)
- Chain-of-thought reasoning
    """,
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models (OpenAI-compatible)
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        default=2048, ge=1, le=32768, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Stream the response")
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")


class ChatResponse(BaseModel):
    id: str = Field(..., description="Unique response ID")
    object: str = Field(default="chat.completion", description="Object type")
    model: str = Field(..., description="Model used")
    created: int = Field(..., description="Unix timestamp")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        default=2048, ge=1, le=32768, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=False, description="Stream the response")


class CompletionResponse(BaseModel):
    id: str = Field(..., description="Unique response ID")
    object: str = Field(default="text_completion", description="Object type")
    model: str = Field(..., description="Model used")
    created: int = Field(..., description="Unix timestamp")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    owned_by: str = Field(default="openai", description="Model owner")


async def generate_ollama_response(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
) -> AsyncGenerator[str, None] | str:
    """Generate response from Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if stream:
        async def stream_response():
            async with http_client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Ollama API error"
                    )
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
        return stream_response()
    else:
        response = await http_client.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ollama API error: {response.text}"
            )
        data = response.json()
        return data.get("response", "")


async def generate_chat_response(
    messages: List[ChatMessage],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = False,
) -> AsyncGenerator[str, None] | str:
    """Generate chat response from Ollama API using chat endpoint."""
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if stream:
        async def stream_response():
            async with http_client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
            ) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Ollama API error"
                    )
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
        return stream_response()
    else:
        response = await http_client.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Ollama API error: {response.text}"
            )
        data = response.json()
        return data.get("message", {}).get("content", "")


@app.get("/")
async def root():
    """API root endpoint with service information."""
    return {
        "service": "Local LLM API",
        "version": "2.0.0",
        "model": DEFAULT_MODEL,
        "description": "OpenAI-compatible API for GPT-OSS-20B via Ollama",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
            "health": "/health",
        },
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    ollama_running = check_ollama_server()
    model_available = check_model_available()
    available_models = get_available_models() if ollama_running else []

    status = "healthy" if (ollama_running and model_available) else "degraded"
    if not ollama_running:
        status = "unhealthy"

    return {
        "status": status,
        "ollama_server": ollama_running,
        "ollama_host": OLLAMA_HOST,
        "model_available": model_available,
        "configured_model": DEFAULT_MODEL,
        "available_models": available_models,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    models = []
    available = get_available_models()

    for model_name in available:
        models.append(
            ModelInfo(
                id=model_name,
                created=int(datetime.now(timezone.utc).timestamp()),
                owned_by="local" if "gpt-oss" not in model_name else "openai",
            )
        )

    return {"data": models, "object": "list"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completions endpoint (OpenAI-compatible)."""
    if not check_ollama_server():
        raise HTTPException(
            status_code=503,
            detail=f"Ollama server is not running at {OLLAMA_HOST}"
        )

    now = datetime.now(timezone.utc)
    response_id = f"chatcmpl-{int(now.timestamp())}"

    if request.stream:
        async def stream_sse():
            response_gen = await generate_chat_response(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 2048,
                stream=True,
            )
            async for chunk in response_gen:
                data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(now.timestamp()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Send final chunk
            final_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(now.timestamp()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming response
    response_text = await generate_chat_response(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens or 2048,
        stream=False,
    )

    # Estimate token counts (approximate)
    prompt_text = " ".join(m.content for m in request.messages)
    prompt_tokens = len(prompt_text.split())
    completion_tokens = len(response_text.split())

    return ChatResponse(
        id=response_id,
        model=request.model,
        created=int(now.timestamp()),
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI-compatible)."""
    if not check_ollama_server():
        raise HTTPException(
            status_code=503,
            detail=f"Ollama server is not running at {OLLAMA_HOST}"
        )

    now = datetime.now(timezone.utc)
    response_id = f"cmpl-{int(now.timestamp())}"

    if request.stream:
        async def stream_sse():
            response_gen = await generate_ollama_response(
                prompt=request.prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 2048,
                stream=True,
            )
            async for chunk in response_gen:
                data = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": int(now.timestamp()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "text": chunk,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming response
    response_text = await generate_ollama_response(
        prompt=request.prompt,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens or 2048,
        stream=False,
    )

    prompt_tokens = len(request.prompt.split())
    completion_tokens = len(response_text.split())

    return CompletionResponse(
        id=response_id,
        model=request.model,
        created=int(now.timestamp()),
        choices=[{
            "index": 0,
            "text": response_text,
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def main():
    """Main function to run the API server."""
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
