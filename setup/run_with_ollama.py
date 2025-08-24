#!/usr/bin/env python3
"""
Alternative script to run GPT-OSS-20B using Ollama
Ollama provides an easier way to run the model with optimized performance
"""

import subprocess
import sys
import time

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("Ollama is not installed.")
    print("\nTo install Ollama on macOS, run:")
    print("  brew install ollama")
    print("\nOr download from: https://ollama.com/download")
    return False

def check_ollama_server():
    """Check if Ollama server is running"""
    try:
        # Try to list models - this will fail if server is not running
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print("✓ Ollama server is running")
            return True
        elif "Error" in result.stderr and "connect" in result.stderr:
            return False
    except subprocess.TimeoutExpired:
        return False
    except:
        return False
    return False

def start_ollama_server():
    """Attempt to start Ollama server"""
    print("Ollama server is not running.")
    print("\nPlease start the Ollama server in a separate terminal with:")
    print("  ollama serve")
    print("\nWaiting for server to start...")
    
    # Give user time to start the server
    for i in range(30):  # Wait up to 30 seconds
        time.sleep(1)
        if check_ollama_server():
            return True
        if i == 10:
            print("Still waiting for Ollama server... (run 'ollama serve' in another terminal)")
    
    print("\nTimeout: Ollama server is not running.")
    print("Please start it with 'ollama serve' and try again.")
    return False

def check_model_exists():
    """Check if GPT-OSS-20B model is already downloaded in Ollama"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'gpt-oss:20b' in result.stdout:
            print("✓ GPT-OSS-20B model is already available")
            return True
    except:
        pass
    return False

def pull_model():
    """Pull the GPT-OSS-20B model using Ollama"""
    print("Downloading GPT-OSS-20B model...")
    print("Note: This is approximately 13GB and may take some time")
    
    try:
        subprocess.run(['ollama', 'pull', 'gpt-oss:20b'], check=True)
        print("✓ Model downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        return False

def run_interactive():
    """Run the model in interactive mode using Ollama"""
    print("\nStarting GPT-OSS-20B with Ollama...")
    print("Type /bye to exit\n")
    
    try:
        subprocess.run(['ollama', 'run', 'gpt-oss:20b'])
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error running model: {e}")

def run_single_prompt(prompt):
    """Run a single prompt through the model"""
    try:
        result = subprocess.run(
            ['ollama', 'run', 'gpt-oss:20b', prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def main():
    print("GPT-OSS-20B with Ollama")
    print("=" * 50)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        sys.exit(1)
    
    # Check if Ollama server is running
    if not check_ollama_server():
        if not start_ollama_server():
            sys.exit(1)
    
    # Check if model exists, if not, pull it
    if not check_model_exists():
        print("\nModel not found locally.")
        response = input("Download GPT-OSS-20B model? (y/n): ").lower()
        if response == 'y':
            if not pull_model():
                sys.exit(1)
        else:
            print("Cannot proceed without the model.")
            sys.exit(1)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        print(f"\nPrompt: {prompt}")
        print("\nGenerating response...")
        response = run_single_prompt(prompt)
        if response:
            print(f"\nResponse: {response}")
    else:
        # Run in interactive mode
        run_interactive()

if __name__ == "__main__":
    main()