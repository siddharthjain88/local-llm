#!/usr/bin/env python3
"""
System health check script for GPT-OSS-20B API
Verifies all components are properly configured and running
"""

import subprocess
import requests
import sys
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

def check_command_exists(command):
    """Check if a command exists in the system"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False

def check_ollama_installed():
    """Check if Ollama is installed"""
    return check_command_exists("ollama")

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        result = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True)
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

def check_api_server():
    """Check if API server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def check_api_health():
    """Get detailed API health status"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def print_status(name, status, details=""):
    """Print formatted status message"""
    if status:
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} {name}: {Fore.GREEN}OK{Style.RESET_ALL} {details}")
    else:
        print(f"{Fore.RED}✗{Style.RESET_ALL} {name}: {Fore.RED}FAILED{Style.RESET_ALL} {details}")

def main():
    """Run health checks"""
    print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}GPT-OSS-20B System Health Check{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*50}{Style.RESET_ALL}\n")
    
    all_good = True
    
    # 1. Check Python version
    python_version = sys.version.split()[0]
    python_ok = sys.version_info >= (3, 8)
    print_status("Python Version", python_ok, f"({python_version})")
    all_good = all_good and python_ok
    
    # 2. Check Ollama installation
    ollama_installed = check_ollama_installed()
    print_status("Ollama Installed", ollama_installed)
    if not ollama_installed:
        print(f"  {Fore.YELLOW}→ Install with: brew install ollama{Style.RESET_ALL}")
    all_good = all_good and ollama_installed
    
    # 3. Check Ollama server
    ollama_running = check_ollama_running()
    print_status("Ollama Server", ollama_running)
    if not ollama_running:
        print(f"  {Fore.YELLOW}→ Start with: ollama serve{Style.RESET_ALL}")
    
    # 4. Check model availability
    model_available = check_model_available()
    print_status("GPT-OSS-20B Model", model_available)
    if not model_available:
        print(f"  {Fore.YELLOW}→ Download with: ollama pull gpt-oss:20b{Style.RESET_ALL}")
    
    # 5. Check API server
    api_running = check_api_server()
    print_status("API Server", api_running)
    if not api_running:
        print(f"  {Fore.YELLOW}→ Start with: ./start_server.sh{Style.RESET_ALL}")
    
    # 6. Detailed API health (if running)
    if api_running:
        health = check_api_health()
        if health:
            print(f"\n{Fore.CYAN}API Server Details:{Style.RESET_ALL}")
            print(f"  • Status: {health.get('status', 'unknown')}")
            print(f"  • Ollama Connected: {health.get('ollama_server', False)}")
            print(f"  • Model Loaded: {health.get('model_available', False)}")
    
    # Summary
    print(f"\n{Fore.YELLOW}{'='*50}{Style.RESET_ALL}")
    if all_good and ollama_running and model_available and api_running:
        print(f"{Fore.GREEN}✓ System is fully operational!{Style.RESET_ALL}")
        print(f"  You can now use: {Fore.CYAN}./start_client.sh{Style.RESET_ALL}")
        return 0
    else:
        print(f"{Fore.YELLOW}⚠ System needs configuration{Style.RESET_ALL}")
        print(f"  Follow the steps above to complete setup")
        return 1

if __name__ == "__main__":
    sys.exit(main())