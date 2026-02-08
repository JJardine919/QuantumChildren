"""
MCP Gateway - Aggregates Multiple MT5 Accounts
===============================================
Single entry point that routes to the correct MT5 container.
Claude Code connects to this gateway on port 8080.
"""

import os
import json
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn

# Account to container URL mapping
ACCOUNT_URLS = {
    "BG_INSTANT": os.environ.get("BG_INSTANT_URL", "http://mt5-bg-instant:8081"),
    "BG_CHALLENGE": os.environ.get("BG_CHALLENGE_URL", "http://mt5-bg-challenge:8082"),
    "ATLAS": os.environ.get("ATLAS_URL", "http://mt5-atlas:8083"),
    # Add more as needed
}

# Current active account
current_account = None
http_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=30.0)
    print("MCP Gateway starting...")
    print(f"Available accounts: {list(ACCOUNT_URLS.keys())}")
    yield
    await http_client.aclose()
    print("MCP Gateway stopped")


app = FastAPI(title="MT5 MCP Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def proxy_to_account(account_key: str, method: str, params: dict, req_id) -> dict:
    """Proxy request to specific MT5 container"""
    if account_key not in ACCOUNT_URLS:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32602, "message": f"Unknown account: {account_key}"}
        }

    url = f"{ACCOUNT_URLS[account_key]}/mcp"
    try:
        resp = await http_client.post(url, json={
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params
        })
        return resp.json()
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": f"Connection failed: {str(e)}"}
        }


# MCP Tools Definition (Gateway level)
MCP_TOOLS = [
    {
        "name": "mt5_connect",
        "description": "Connect to MT5 account. Keys: ATLAS, BG_INSTANT, BG_CHALLENGE",
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_key": {"type": "string", "description": "Account key: ATLAS, BG_INSTANT, BG_CHALLENGE"}
            },
            "required": ["account_key"]
        }
    },
    {
        "name": "mt5_positions",
        "description": "Get all open positions with P/L, SL, TP",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "mt5_summary",
        "description": "Get full account summary with balance, equity, positions",
        "inputSchema": {"type": "object", "properties": {}}
    },
    {
        "name": "mt5_close",
        "description": "Close a position by ticket number",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "integer", "description": "Position ticket to close"}
            },
            "required": ["ticket"]
        }
    },
    {
        "name": "mt5_close_losers",
        "description": "Close all positions exceeding loss limit",
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_loss": {"type": "number", "description": "Max loss in dollars (default 1.00)"}
            }
        }
    },
    {
        "name": "mt5_modify_sl",
        "description": "Modify stop loss of a position",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "integer", "description": "Position ticket"},
                "sl": {"type": "number", "description": "New stop loss price"}
            },
            "required": ["ticket", "sl"]
        }
    },
    {
        "name": "mt5_history",
        "description": "Get closed trades history",
        "inputSchema": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days (default 1)"}
            }
        }
    },
    {
        "name": "mt5_all_summaries",
        "description": "Get summaries from ALL connected accounts at once",
        "inputSchema": {"type": "object", "properties": {}}
    }
]


@app.post("/mcp")
async def mcp_handler(request: Request):
    """Handle MCP JSON-RPC requests"""
    global current_account

    try:
        body = await request.json()
    except:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}

    method = body.get("method", "")
    params = body.get("params", {})
    req_id = body.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mt5-mcp-gateway", "version": "1.0.0"}
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS}
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        # Handle mt5_connect specially - set current account
        if tool_name == "mt5_connect":
            account_key = args.get("account_key", "").upper()
            if account_key not in ACCOUNT_URLS:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "error": f"Unknown account: {account_key}",
                                "available": list(ACCOUNT_URLS.keys())
                            }, indent=2)
                        }]
                    }
                }

            current_account = account_key
            result = await proxy_to_account(account_key, method, params, req_id)
            return result

        # Handle mt5_all_summaries - query all accounts
        elif tool_name == "mt5_all_summaries":
            summaries = {}
            for acc_key in ACCOUNT_URLS:
                try:
                    resp = await http_client.get(f"{ACCOUNT_URLS[acc_key]}/api/summary")
                    summaries[acc_key] = resp.json()
                except Exception as e:
                    summaries[acc_key] = {"error": str(e)}

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(summaries, indent=2)
                    }]
                }
            }

        # All other tools require current account to be set
        else:
            if not current_account:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "error": "No account connected. Use mt5_connect first.",
                                "available_accounts": list(ACCOUNT_URLS.keys())
                            }, indent=2)
                        }]
                    }
                }

            return await proxy_to_account(current_account, method, params, req_id)

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"}
    }


# ============================================================
# SSE Endpoint for Claude Code
# ============================================================

@app.get("/sse")
async def sse_endpoint(request: Request):
    """Server-Sent Events endpoint"""
    async def event_generator():
        init_msg = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mt5-mcp-gateway", "version": "1.0.0"}
            }
        }
        yield f"data: {json.dumps(init_msg)}\n\n"

        while True:
            if await request.is_disconnected():
                break
            yield f": keepalive\n\n"
            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ============================================================
# REST API Endpoints
# ============================================================

@app.get("/health")
async def health():
    """Health check all containers"""
    statuses = {}
    for acc_key, url in ACCOUNT_URLS.items():
        try:
            resp = await http_client.get(f"{url}/health", timeout=5.0)
            statuses[acc_key] = resp.json()
        except Exception as e:
            statuses[acc_key] = {"status": "error", "error": str(e)}

    return {
        "gateway": "ok",
        "current_account": current_account,
        "accounts": statuses,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/accounts")
async def list_accounts():
    """List available accounts"""
    return {"accounts": list(ACCOUNT_URLS.keys()), "current": current_account}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
