.PHONY: help install test lint format check clean server client all

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
VENV_SERVER := server/venv
VENV_CLIENT := client/venv
VENV_TEST := venv

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Install all dependencies (server, client, and dev tools)
	@echo "$(YELLOW)Installing server dependencies...$(NC)"
	cd server && $(PYTHON) -m venv venv && . venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(YELLOW)Installing client dependencies...$(NC)"
	cd client && $(PYTHON) -m venv venv && . venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "$(YELLOW)Installing development dependencies...$(NC)"
	$(PYTHON) -m venv $(VENV_TEST)
	. $(VENV_TEST)/bin/activate && $(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(NC)"
	@if [ ! -d "$(VENV_TEST)" ]; then \
		echo "$(RED)Virtual environment not found. Run 'make install' first.$(NC)"; \
		exit 1; \
	fi
	. $(VENV_TEST)/bin/activate && pytest tests/ -v
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-server: ## Run server tests only
	@echo "$(YELLOW)Running server tests...$(NC)"
	. $(VENV_TEST)/bin/activate && pytest tests/server/ -v

test-client: ## Run client tests only
	@echo "$(YELLOW)Running client tests...$(NC)"
	. $(VENV_TEST)/bin/activate && pytest tests/client/ -v

test-integration: ## Run integration tests (requires running server)
	@echo "$(YELLOW)Running integration tests...$(NC)"
	. $(VENV_TEST)/bin/activate && pytest tests/ -v -m integration

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	. $(VENV_TEST)/bin/activate && pytest tests/ --cov=server --cov=client --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

lint: ## Run all linting tools
	@echo "$(YELLOW)Running linting checks...$(NC)"
	@if [ ! -d "$(VENV_TEST)" ]; then \
		echo "$(RED)Virtual environment not found. Run 'make install' first.$(NC)"; \
		exit 1; \
	fi
	@echo "  Running flake8..."
	. $(VENV_TEST)/bin/activate && flake8 server/ client/ || true
	@echo "  Running pylint..."
	. $(VENV_TEST)/bin/activate && pylint server/ client/ || true
	@echo "  Running mypy..."
	. $(VENV_TEST)/bin/activate && mypy server/ client/ || true
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-ruff: ## Run ruff linter (fast)
	@echo "$(YELLOW)Running ruff...$(NC)"
	. $(VENV_TEST)/bin/activate && ruff check server/ client/ tests/
	@echo "$(GREEN)✓ Ruff check complete$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	. $(VENV_TEST)/bin/activate && black server/ client/ tests/
	. $(VENV_TEST)/bin/activate && isort server/ client/ tests/
	@echo "$(GREEN)✓ Code formatted$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(YELLOW)Checking code format...$(NC)"
	. $(VENV_TEST)/bin/activate && black --check server/ client/ tests/
	. $(VENV_TEST)/bin/activate && isort --check-only server/ client/ tests/
	@echo "$(GREEN)✓ Code format check complete$(NC)"

security: ## Run security checks with bandit
	@echo "$(YELLOW)Running security checks...$(NC)"
	. $(VENV_TEST)/bin/activate && bandit -r server/ client/ -ll
	@echo "$(GREEN)✓ Security check complete$(NC)"

check: format-check lint-ruff test ## Run all checks (format, lint, test)
	@echo "$(GREEN)✓ All checks passed$(NC)"

check-all: format-check lint security test-coverage ## Run ALL checks including security and coverage
	@echo "$(GREEN)✓ All comprehensive checks passed$(NC)"

clean: ## Clean up cache files and directories
	@echo "$(YELLOW)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .ruff_cache/
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

server: ## Start the API server
	@echo "$(YELLOW)Starting API server...$(NC)"
	@if ! pgrep -x "ollama" > /dev/null; then \
		echo "$(RED)⚠️  Ollama is not running! Start it with: ollama serve$(NC)"; \
	fi
	cd server && . venv/bin/activate && python app.py

client: ## Start the client
	@echo "$(YELLOW)Starting client...$(NC)"
	cd client && . venv/bin/activate && python client.py

health: ## Check system health
	@echo "$(YELLOW)Checking system health...$(NC)"
	@echo -n "Ollama server: "
	@if pgrep -x "ollama" > /dev/null; then \
		echo "$(GREEN)✓ Running$(NC)"; \
	else \
		echo "$(RED)✗ Not running$(NC)"; \
	fi
	@echo -n "API server: "
	@if curl -s http://localhost:8000/health > /dev/null 2>&1; then \
		echo "$(GREEN)✓ Running$(NC)"; \
	else \
		echo "$(RED)✗ Not running$(NC)"; \
	fi
	@echo -n "Model available: "
	@if ollama list 2>/dev/null | grep -q "gpt-oss:20b"; then \
		echo "$(GREEN)✓ Installed$(NC)"; \
	else \
		echo "$(RED)✗ Not installed$(NC)"; \
	fi

setup-dev: ## Set up development environment
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	$(PYTHON) -m venv $(VENV_TEST)
	. $(VENV_TEST)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV_TEST)/bin/activate && $(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)✓ Development environment ready$(NC)"
	@echo "Activate with: source $(VENV_TEST)/bin/activate"

ci: ## Run CI pipeline locally
	@echo "$(YELLOW)Running CI pipeline...$(NC)"
	make clean
	make install
	make format-check
	make lint-ruff
	make security
	make test-coverage
	@echo "$(GREEN)✓ CI pipeline complete$(NC)"

all: clean install check-all ## Run everything (clean, install, all checks)