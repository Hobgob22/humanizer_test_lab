# Makefile for Humanizer Test-Bench

.PHONY: help dev prod build clean logs shell test lint format

# Default target
help:
	@echo "Humanizer Test-Bench - Available commands:"
	@echo "  make dev        - Start development server with hot reload"
	@echo "  make prod       - Start production server"
	@echo "  make build      - Build Docker images"
	@echo "  make clean      - Stop and remove containers"
	@echo "  make logs       - View container logs"
	@echo "  make shell      - Open shell in container"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make backup     - Backup data and results"

# Development server with hot reload
dev:
	@echo "Starting development server with hot reload..."
	docker-compose --profile dev up

# Development server in background
dev-d:
	@echo "Starting development server in background..."
	docker-compose --profile dev up -d
	@echo "Server running at http://localhost:8501"

# Production server
prod:
	@echo "Starting production server..."
	docker-compose --profile prod up -d
	@echo "Server running at http://localhost:8501"

# Production with nginx
prod-nginx:
	@echo "Starting production server with nginx..."
	docker-compose --profile prod-nginx --profile prod up -d
	@echo "Server running at http://localhost"

# Build images
build:
	@echo "Building Docker images..."
	docker-compose build --no-cache

# Build development image only
build-dev:
	@echo "Building development image..."
	docker-compose build humanizer-dev

# Build production image only
build-prod:
	@echo "Building production image..."
	docker-compose build humanizer-prod

# Stop and remove containers
clean:
	@echo "Stopping and removing containers..."
	docker-compose down -v

# View logs
logs:
	docker-compose logs -f

# View specific service logs
logs-dev:
	docker-compose logs -f humanizer-dev

logs-prod:
	docker-compose logs -f humanizer-prod

# Open shell in container
shell:
	docker-compose exec humanizer-dev /bin/bash

shell-prod:
	docker-compose exec humanizer-prod /bin/bash

# Run CLI in container
cli:
	docker-compose exec humanizer-dev python -m src.cli $(ARGS)

# Run tests
test:
	@echo "Running tests..."
	docker-compose exec humanizer-dev python -m pytest tests/

# Lint code
lint:
	@echo "Running linters..."
	docker-compose exec humanizer-dev python -m flake8 src/
	docker-compose exec humanizer-dev python -m mypy src/

# Format code
format:
	@echo "Formatting code..."
	docker-compose exec humanizer-dev python -m black src/
	docker-compose exec humanizer-dev python -m isort src/

# Create backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ results/ cache/ logs/
	@echo "Backup created in backups/"

# Restore from backup
restore:
	@echo "Available backups:"
	@ls -la backups/
	@echo "To restore, run: tar -xzf backups/backup-TIMESTAMP.tar.gz"

# Check environment
check-env:
	@echo "Checking environment variables..."
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found!"; \
		echo "Creating from example..."; \
		cp .env.example .env; \
		echo "Please edit .env and add your API keys"; \
	else \
		echo ".env file found"; \
		echo "Checking required variables..."; \
		grep -E "OPENAI_API_KEY|GPTZERO_API_KEY|SAPLING_API_KEY" .env | sed 's/=.*/=***/' || echo "Warning: Some API keys may be missing"; \
	fi

# Initialize project
init: check-env
	@echo "Initializing project..."
	@mkdir -p data/ai_texts data/human_texts data/ai_paras data/human_paras
	@mkdir -p cache logs results
	@echo "Project initialized!"

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	docker-compose exec humanizer-dev pip install --upgrade -r requirements.txt

# Docker system prune
docker-clean:
	@echo "Cleaning Docker system..."
	docker system prune -af --volumes

# Show resource usage
stats:
	docker stats --no-stream

# Monitor resources
monitor:
	docker stats