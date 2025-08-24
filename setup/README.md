# Setup Utilities

This directory contains utility scripts for setting up and testing GPT-OSS-20B.

## Scripts

### `run_with_ollama.py`
Quick test script to verify Ollama and GPT-OSS-20B are working:
```bash
python run_with_ollama.py "Your prompt here"
```

### `test_gpt_oss.py`
Direct testing with Transformers library (downloads model from HuggingFace):
```bash
python test_gpt_oss.py
```

### `setup_gpt_oss.py`
Downloads GPT-OSS-20B model from HuggingFace for local use:
```bash
python setup_gpt_oss.py
```

## Prerequisites

For Ollama-based scripts:
1. Install Ollama: `brew install ollama`
2. Start server: `ollama serve`
3. Pull model: `ollama pull gpt-oss:20b`

For Transformers-based scripts:
```bash
pip install transformers torch accelerate huggingface-hub
```