#!/usr/bin/env python3
"""
Test script for GPT-OSS-20B model
Provides a simple interface to interact with the model
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path=None):
    """Load the GPT-OSS-20B model and tokenizer"""
    if model_path and Path(model_path).exists():
        model_id = model_path
        print(f"Loading model from local path: {model_id}")
    else:
        model_id = "openai/gpt-oss-20b"
        print(f"Loading model from HuggingFace: {model_id}")
    
    print("Note: This may take a few moments on first load...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Load model with automatic device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✓ Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response[len(prompt):].strip()
    
    return response

def interactive_mode(model, tokenizer):
    """Run the model in interactive mode"""
    print("\nInteractive Mode - GPT-OSS-20B")
    print("=" * 50)
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'clear' to clear the screen")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n> ").strip()
            
            if prompt.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            elif prompt.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif not prompt:
                continue
            
            print("\nGenerating response...")
            response = generate_response(model, tokenizer, prompt)
            print(f"\n{response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("GPT-OSS-20B Test Script")
    print("=" * 50)
    
    # Check for local model
    local_model_path = "./models/gpt-oss-20b"
    if Path(local_model_path).exists():
        print(f"Found local model at {local_model_path}")
        model_path = local_model_path
    else:
        print("No local model found. Will download from HuggingFace on first use.")
        model_path = None
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Check if we have command line arguments for a single prompt
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...")
        response = generate_response(model, tokenizer, prompt)
        print(f"\nResponse: {response}")
    else:
        # Run in interactive mode
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()