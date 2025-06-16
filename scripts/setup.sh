#!/bin/bash
# PHAL Platform Setup Script

echo "🌱 PHAL Platform Setup"
echo "====================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then 
    echo "❌ Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION"

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker installed"
else
    echo "❌ Docker not found. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Create directories
echo "Creating directory structure..."
mkdir -p data logs certs config grafana/provisioning

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your configuration"
fi

# Generate self-signed certificates for development
if [ ! -f certs/server.crt ]; then
    echo "Generating self-signed certificates..."
    ./scripts/generate-certs.sh
fi

# Install Python dependencies
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt

# Build Docker images
echo "Building Docker images..."
docker-compose build

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start PHAL:"
echo "  1. Edit .env with your configuration"
echo "  2. Run: docker-compose up"
echo "  3. Visit: http://localhost:8080"
echo ""
echo "For development mode:"
echo "  source venv/bin/activate"
echo "  cd backend && python -m phal.api"
echo ""
echo "Happy growing! 🌱"
