# 🐳 Docker Setup for Restaurant Recommendation App

This document explains how to run the Restaurant Recommendation App using Docker for cross-platform compatibility.

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed on your system
- Git (to clone the repository)

### Running with Docker Compose (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd Restaurant-recommendation/restaurant-recommendation
   ```

2. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Open your browser and go to `http://localhost:5000`
   - The app will be running with hot-reload for development

### Running with Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t restaurant-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 restaurant-app
   ```

## 🔧 Development Setup

### Using Docker Compose for Development

The `docker-compose.yml` file is configured for development with:
- Volume mounting for live code changes
- Hot-reload enabled
- Health checks
- Proper port mapping

### Environment Variables

You can customize the app behavior using environment variables:

```bash
# Development mode
docker-compose up -e FLASK_ENV=development

# Production mode
docker-compose up -e FLASK_ENV=production
```

## 🏗️ CI/CD Pipeline

The project includes a GitHub Actions CI/CD pipeline that:

### Automated Testing
- **Code Linting**: Flake8 for Python code quality
- **Format Checking**: Black for code formatting
- **Unit Tests**: Pytest for test execution
- **Coverage Reports**: Code coverage analysis

### Automated Building
- **Docker Image Building**: Multi-platform Docker images
- **Security Scanning**: Trivy vulnerability scanning
- **Container Registry**: Automatic pushing to GitHub Container Registry

### Automated Deployment
- **Staging Deployment**: Automatic deployment on `develop` branch
- **Production Deployment**: Automatic deployment on `main` branch
- **Environment Management**: Separate staging and production environments

## 📁 Project Structure

```
restaurant-recommendation/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD pipeline
├── Dockerfile                 # Docker container definition
├── docker-compose.yml           # Local development setup
├── .dockerignore             # Docker build optimization
├── requirements.txt          # Python dependencies
├── test_app.py              # Basic test suite
├── app.py                   # Main Flask application
└── DOCKER_README.md         # This file
```

## 🛠️ Available Commands

### Docker Commands

```bash
# Build the image
docker build -t restaurant-app .

# Run the container
docker run -p 5000:5000 restaurant-app

# Run in background
docker run -d -p 5000:5000 --name restaurant-app restaurant-app

# Stop the container
docker stop restaurant-app

# Remove the container
docker rm restaurant-app
```

### Docker Compose Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Execute commands in running container
docker-compose exec restaurant-app bash
```

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change the port in docker-compose.yml
   ports:
     - "5001:5000"  # Use port 5001 instead
   ```

2. **Permission issues on Linux:**
   ```bash
   # Add your user to the docker group
   sudo usermod -aG docker $USER
   # Log out and log back in
   ```

3. **Build failures:**
   ```bash
   # Clean Docker cache
   docker system prune -a
   # Rebuild without cache
   docker-compose build --no-cache
   ```

### Health Checks

The application includes health checks to ensure it's running properly:

```bash
# Check container health
docker ps

# View health check logs
docker inspect <container-id> | grep -A 10 Health
```

## 🌐 Cross-Platform Compatibility

This Docker setup works on:
- ✅ **Windows** (Windows 10/11 with Docker Desktop)
- ✅ **macOS** (Intel and Apple Silicon)
- ✅ **Linux** (Ubuntu, CentOS, RHEL, etc.)

## 📊 Monitoring and Logs

### View Application Logs
```bash
# Docker Compose
docker-compose logs -f restaurant-app

# Docker
docker logs -f <container-name>
```

### Monitor Resource Usage
```bash
# View container stats
docker stats restaurant-app
```

## 🔒 Security Features

The Docker setup includes several security measures:
- Non-root user execution
- Minimal base image (Python slim)
- Security scanning in CI/CD
- Health checks for monitoring
- Proper secret management

## 🚀 Production Deployment

For production deployment:

1. **Use the production Dockerfile** (already configured)
2. **Set environment variables** for production
3. **Use a reverse proxy** (nginx) for better performance
4. **Enable HTTPS** with SSL certificates
5. **Monitor with logging** and health checks

## 📝 Contributing

When contributing to this project:

1. **Run tests locally:**
   ```bash
   docker-compose exec restaurant-app pytest
   ```

2. **Check code formatting:**
   ```bash
   docker-compose exec restaurant-app black --check .
   ```

3. **Run linting:**
   ```bash
   docker-compose exec restaurant-app flake8 .
   ```

## 📞 Support

If you encounter issues:
1. Check the logs: `docker-compose logs`
2. Verify Docker is running: `docker --version`
3. Check port availability: `netstat -tulpn | grep 5000`
4. Review the CI/CD pipeline for automated testing results
