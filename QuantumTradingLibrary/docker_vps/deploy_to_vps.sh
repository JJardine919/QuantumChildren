#!/bin/bash
# Deploy MT5 MCP Docker to Hostinger VPS
# Run this script ON THE VPS after copying files

set -e

echo "=========================================="
echo "  MT5 MCP Docker Deployment"
echo "  VPS: $(hostname)"
echo "=========================================="

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    zypper refresh
    zypper install -y docker docker-compose
    systemctl enable docker
    systemctl start docker
fi

# Create deployment directory
DEPLOY_DIR="/opt/mt5-mcp"
mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR

echo "Deployment directory: $DEPLOY_DIR"

# Check if files exist
if [ ! -f "docker-compose.yml" ]; then
    echo "ERROR: docker-compose.yml not found!"
    echo "Copy docker_vps/* files to /opt/mt5-mcp first"
    exit 1
fi

# Build and start containers
echo "Building Docker containers..."
docker-compose build

echo "Starting containers..."
docker-compose up -d

# Wait for startup
echo "Waiting for services to start..."
sleep 30

# Check health
echo ""
echo "=========================================="
echo "  Health Check"
echo "=========================================="
curl -s http://localhost:8080/health | python3 -m json.tool || echo "Gateway not responding yet"

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "Gateway URL: http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "Test commands:"
echo "  curl http://localhost:8080/health"
echo "  curl http://localhost:8080/api/accounts"
echo ""
echo "Docker commands:"
echo "  docker-compose logs -f           # View logs"
echo "  docker-compose restart           # Restart all"
echo "  docker-compose down              # Stop all"
echo ""
