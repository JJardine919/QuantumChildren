# MT5 Docker MCP Setup for Hostinger VPS

## Overview

This setup runs MT5 trading accounts on your Hostinger VPS (72.62.170.153) using Docker.
Claude Code connects via HTTP MCP protocol instead of local stdio.

## Architecture

```
Your PC (Claude Code)
        │
        │ HTTP (port 8080)
        ▼
┌─────────────────────────────────────┐
│  Hostinger VPS (72.62.170.153)      │
│                                     │
│  ┌─────────────────────────────┐    │
│  │ MCP Gateway (:8080)         │    │
│  │ Routes to correct account   │    │
│  └─────────────────────────────┘    │
│       │       │       │             │
│       ▼       ▼       ▼             │
│  ┌───────┐ ┌───────┐ ┌───────┐     │
│  │BG_INST│ │BG_CHAL│ │ ATLAS │     │
│  │:8081  │ │:8082  │ │:8083  │     │
│  └───────┘ └───────┘ └───────┘     │
│  (Docker containers with Wine+MT5)  │
└─────────────────────────────────────┘
```

## Quick Deploy

### Step 1: Copy Files to VPS

```powershell
cd C:\Users\jimjj\Music\QuantumChildren\QuantumTradingLibrary\docker_vps
python copy_to_hostinger.py
```

Enter your VPS password when prompted.

### Step 2: SSH to VPS and Deploy

```bash
ssh root@72.62.170.153
cd /opt/mt5-mcp
./deploy_to_vps.sh
```

### Step 3: Configure Claude Code

Add this to your Claude Code MCP configuration:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mt5": {
      "transport": "http",
      "url": "http://72.62.170.153:8080/mcp"
    }
  }
}
```

**Or for Claude Code CLI:** `~/.claude/mcp_config.json`

```json
{
  "servers": {
    "mt5-vps": {
      "transport": {
        "type": "http",
        "url": "http://72.62.170.153:8080/mcp"
      }
    }
  }
}
```

### Step 4: Restart Claude Code

After configuring, restart Claude Code to load the new MCP server.

## Available Commands

Once connected, Claude has access to:

| Tool | Description |
|------|-------------|
| `mt5_connect` | Connect to account (ATLAS, BG_INSTANT, BG_CHALLENGE) |
| `mt5_positions` | Get all open positions |
| `mt5_summary` | Full account summary |
| `mt5_close` | Close position by ticket |
| `mt5_close_losers` | Close positions exceeding loss limit |
| `mt5_modify_sl` | Modify stop loss |
| `mt5_history` | Get trade history |
| `mt5_all_summaries` | Get ALL accounts at once |

## Management

### View Logs
```bash
cd /opt/mt5-mcp
docker-compose logs -f
```

### Restart All Containers
```bash
docker-compose restart
```

### Stop Everything
```bash
docker-compose down
```

### Check Health
```bash
curl http://localhost:8080/health
```

## Firewall

Make sure port 8080 is open:

```bash
# openSUSE
firewall-cmd --add-port=8080/tcp --permanent
firewall-cmd --reload
```

Or via Hostinger panel > Firewall rules.

## Troubleshooting

### MT5 Not Connecting
- Check Wine logs: `docker-compose logs mt5-bg-instant`
- MT5 may need broker-specific DLLs

### Gateway Timeout
- Individual containers may still be starting
- Check: `docker ps` to see container status

### Connection Refused
- Ensure firewall allows port 8080
- Check if services are running: `docker-compose ps`

## Files

| File | Purpose |
|------|---------|
| `Dockerfile.mt5` | MT5 container with Wine |
| `Dockerfile.gateway` | MCP gateway container |
| `docker-compose.yml` | Orchestration |
| `mcp_mt5_http_server.py` | HTTP MCP server (per account) |
| `mcp_gateway.py` | Aggregates all accounts |
| `config.json` | Account credentials |
| `entrypoint.sh` | Container startup script |
| `deploy_to_vps.sh` | VPS deployment script |

## Security Notes

- Port 8080 is exposed without authentication
- For production, add API key authentication
- Consider using VPN or SSH tunnel instead of open port
