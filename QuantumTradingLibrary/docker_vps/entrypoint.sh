#!/bin/bash
# MT5 Docker Entrypoint
# Starts Xvfb (virtual display), Wine MT5, mt5linux bridge, and MCP HTTP server

set -e

echo "=== MT5 Docker Container Starting ==="
echo "Account: ${MT5_ACCOUNT_KEY:-default}"
echo "Port: ${PORT:-8080}"

# Start virtual display
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x16 &
sleep 2

# MT5 terminal path (installed by mt5setup.exe)
MT5_PATH="/root/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"

# Start MT5 terminal
if [ -f "$MT5_PATH" ]; then
    echo "Starting MT5 terminal: $MT5_PATH"
    wine "$MT5_PATH" /portable &
    sleep 15
else
    echo "WARNING: MT5 terminal not found at $MT5_PATH"
    echo "Checking alternative paths..."
    # Try alternative paths
    ALT_PATH="/root/.wine/drive_c/Program Files (x86)/MetaTrader 5/terminal.exe"
    if [ -f "$ALT_PATH" ]; then
        echo "Found at: $ALT_PATH"
        wine "$ALT_PATH" /portable &
        sleep 15
    else
        echo "MT5 terminal not found - running in mock mode"
    fi
fi

# Start mt5linux bridge server (connects Linux Python to Wine Python/MT5)
echo "Starting mt5linux bridge server..."
python3 -m mt5linux "C:\\Python310\\python.exe" &
sleep 5

# Start MCP HTTP server
echo "Starting MCP HTTP server on port ${PORT:-8080}..."
exec python3 /app/mcp_mt5_http_server.py
