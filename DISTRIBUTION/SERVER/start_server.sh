#!/bin/bash
# QuantumChildren Collection Server Startup

echo "========================================"
echo "  QUANTUM CHILDREN - Collection Server"
echo "========================================"

# Install requirements
pip3 install -r requirements.txt

# Run with gunicorn for production
echo "Starting server on port 8888..."
gunicorn -w 4 -b 0.0.0.0:8888 collection_server:app --access-logfile access.log --error-logfile error.log --daemon

echo "Server started. Check logs:"
echo "  tail -f access.log"
echo "  tail -f error.log"
