#!/bin/bash
# QuantumChildren Collection Server Startup

echo "========================================"
echo "  QUANTUM CHILDREN - Collection Server"
echo "========================================"

# Install requirements
pip3 install -r requirements.txt

# ============================================================
# BASE44 WEBHOOK FORWARDING
# Set these to enable forwarding to your Base44 dashboard.
# Get the webhook URL from Base44: Dashboard > Code > Functions > ingestSignal
# Strip the /ingestSignal off the end — the code appends endpoint names automatically.
# The key must match what you stored in Base44 Dashboard > Secrets as QC_WEBHOOK_KEY.
# ============================================================
export BASE44_WEBHOOK_URL="${BASE44_WEBHOOK_URL:-}"
export BASE44_WEBHOOK_KEY="${BASE44_WEBHOOK_KEY:-}"

# Admin API key for protected endpoints (/backtest, /compile)
export QC_ADMIN_KEY="${QC_ADMIN_KEY:-}"

# Run with gunicorn for production
echo "Starting server on port 8888..."
echo "BASE44 forwarding: $([ -n \"$BASE44_WEBHOOK_URL\" ] && echo 'ENABLED' || echo 'DISABLED (set BASE44_WEBHOOK_URL and BASE44_WEBHOOK_KEY)')"
gunicorn -w 4 -b 0.0.0.0:8888 collection_server:app --access-logfile access.log --error-logfile error.log --daemon

echo "Server started. Check logs:"
echo "  tail -f access.log"
echo "  tail -f error.log"
