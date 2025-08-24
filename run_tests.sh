#!/bin/bash

# Test runner script for GPT-OSS-20B API

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}GPT-OSS-20B Test Runner${NC}"
echo "========================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Setting up test environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements-dev.txt
else
    source venv/bin/activate
fi

# Parse command line arguments
TEST_TYPE=${1:-"all"}

case $TEST_TYPE in
    quick)
        echo -e "\n${YELLOW}Running quick tests (unit tests only)...${NC}"
        pytest tests/ -v -m "not integration" --tb=short
        ;;
    
    server)
        echo -e "\n${YELLOW}Running server tests...${NC}"
        pytest tests/server/ -v --tb=short
        ;;
    
    client)
        echo -e "\n${YELLOW}Running client tests...${NC}"
        pytest tests/client/ -v --tb=short
        ;;
    
    integration)
        echo -e "\n${YELLOW}Running integration tests...${NC}"
        echo -e "${YELLOW}Checking if server is running...${NC}"
        if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${RED}API server is not running!${NC}"
            echo "Please start it with: ./start_server.sh"
            exit 1
        fi
        pytest tests/ -v -m integration
        ;;
    
    coverage)
        echo -e "\n${YELLOW}Running tests with coverage...${NC}"
        pytest tests/ --cov=server --cov=client --cov-report=term --cov-report=html
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    lint)
        echo -e "\n${YELLOW}Running linting checks...${NC}"
        echo "Running ruff..."
        ruff check server/ client/ tests/ || true
        echo "Running flake8..."
        flake8 server/ client/ || true
        echo "Running mypy..."
        mypy server/ client/ || true
        ;;
    
    format)
        echo -e "\n${YELLOW}Checking code formatting...${NC}"
        black --check server/ client/ tests/
        isort --check-only server/ client/ tests/
        ;;
    
    all)
        echo -e "\n${YELLOW}Running all tests and checks...${NC}"
        
        # Format check
        echo -e "\n${YELLOW}1. Format check${NC}"
        black --check server/ client/ tests/ || true
        isort --check-only server/ client/ tests/ || true
        
        # Linting
        echo -e "\n${YELLOW}2. Linting${NC}"
        ruff check server/ client/ tests/ || true
        
        # Unit tests
        echo -e "\n${YELLOW}3. Unit tests${NC}"
        pytest tests/ -v -m "not integration" --tb=short
        
        # Coverage
        echo -e "\n${YELLOW}4. Coverage${NC}"
        pytest tests/ --cov=server --cov=client --cov-report=term -m "not integration"
        ;;
    
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Usage: $0 [quick|server|client|integration|coverage|lint|format|all]"
        echo "  quick       - Run unit tests only (fast)"
        echo "  server      - Run server tests"
        echo "  client      - Run client tests"
        echo "  integration - Run integration tests (requires running server)"
        echo "  coverage    - Run tests with coverage report"
        echo "  lint        - Run linting checks"
        echo "  format      - Check code formatting"
        echo "  all         - Run all tests and checks (default)"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✓ Tests passed successfully!${NC}"
else
    echo -e "\n${RED}✗ Tests failed!${NC}"
    exit 1
fi