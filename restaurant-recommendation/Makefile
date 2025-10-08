# Makefile for Restaurant Recommendation App Docker Operations

.PHONY: help build run stop clean test lint format dev prod logs shell

# Default target
help:
	@echo "🐳 Restaurant App Docker Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make prod         - Start production environment"
	@echo "  make logs         - View application logs"
	@echo "  make shell        - Open shell in running container"
	@echo ""
	@echo "Building:"
	@echo "  make build        - Build Docker image"
	@echo "  make clean        - Clean up Docker resources"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Utilities:"
	@echo "  make stop         - Stop all containers"
	@echo "  make restart      - Restart containers"

# Development commands
dev:
	@echo "🚀 Starting development environment..."
	docker-compose up --build

prod:
	@echo "🚀 Starting production environment..."
	docker-compose -f docker-compose.yml up --build -d

# Build commands
build:
	@echo "🔨 Building Docker image..."
	docker build -t restaurant-app .

# Container management
run:
	@echo "🏃 Running container..."
	docker run -p 5000:5000 restaurant-app

stop:
	@echo "🛑 Stopping containers..."
	docker-compose down

restart: stop dev

# Development utilities
logs:
	@echo "📋 Viewing logs..."
	docker-compose logs -f

shell:
	@echo "🐚 Opening shell in container..."
	docker-compose exec restaurant-app bash

# Testing and quality
test:
	@echo "🧪 Running tests..."
	docker-compose exec restaurant-app pytest

lint:
	@echo "🔍 Running linting..."
	docker-compose exec restaurant-app flake8 .

format:
	@echo "✨ Formatting code..."
	docker-compose exec restaurant-app black .

# Cleanup
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v
	docker system prune -f
	docker image prune -f

# Health check
health:
	@echo "🏥 Checking application health..."
	curl -f http://localhost:5000/ || echo "❌ Application not responding"

# Install dependencies (for local development)
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install pytest flake8 black

# Run all quality checks
check: lint format test
	@echo "✅ All quality checks passed!"
