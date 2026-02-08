"""
MT5 MCP HTTP Server - Remote MetaTrader5 Control
=================================================
HTTP-based MCP server for remote access from Claude Code.
Runs on Linux VPS with Wine + MT5.

Transport: HTTP with SSE (Server-Sent Events)
Port: 8080 (configurable)
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# MT5 import - works under Wine on Linux
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNING: MetaTrader5 not available - running in mock mode")


# Load config
CONFIG_PATH = os.environ.get('MT5_CONFIG', '/app/config.json')
try:
    with open(CONFIG_PATH) as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {
        "ACCOUNTS": {},
        "TRADING_SETTINGS": {"AGENT_SL_MAX": 1.00}
    }

ACCOUNTS = CONFIG.get('ACCOUNTS', {})
MAX_LOSS_DOLLARS = CONFIG.get('TRADING_SETTINGS', {}).get('AGENT_SL_MAX', 1.00)


class MT5Manager:
    """Direct MT5 management"""

    def __init__(self):
        self.connected_account = None
        self.initialized = False

    def connect(self, account_key: str = None) -> dict:
        """Connect to MT5 account"""
        if not MT5_AVAILABLE:
            return {"error": "MT5 not available", "mock": True}

        if account_key and account_key in ACCOUNTS:
            acc = ACCOUNTS[account_key]
            terminal_path = acc.get('terminal_path_linux', '/app/mt5/terminal64.exe')

            # Shutdown existing connection
            mt5.shutdown()

            if terminal_path:
                if not mt5.initialize(path=terminal_path):
                    return {"error": f"Init failed: {mt5.last_error()}"}
            else:
                if not mt5.initialize():
                    return {"error": f"Init failed: {mt5.last_error()}"}

            # Get password from environment variable
            password = None
            if acc.get('env_password_key'):
                password = os.environ.get(acc['env_password_key'])
            elif acc.get('password'):
                password = acc['password']

            if password:
                if not mt5.login(acc['account'], password=password, server=acc['server']):
                    return {"error": f"Login failed: {mt5.last_error()}"}
        else:
            if not mt5.initialize():
                return {"error": f"Init failed: {mt5.last_error()}"}

        info = mt5.account_info()
        if info:
            self.connected_account = info.login
            self.initialized = True
            return {
                "account": info.login,
                "balance": info.balance,
                "equity": info.equity,
                "profit": info.profit,
                "margin_free": info.margin_free,
                "server": info.server
            }
        return {"error": "Could not get account info"}

    def get_positions(self) -> list:
        """Get all open positions"""
        if not MT5_AVAILABLE or not self.initialized:
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == 0 else "SELL",
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "magic": pos.magic,
                "comment": pos.comment,
                "time": datetime.fromtimestamp(pos.time).isoformat()
            })
        return result

    def close_position(self, ticket: int) -> dict:
        """Close a position by ticket"""
        if not MT5_AVAILABLE:
            return {"error": "MT5 not available"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"error": f"Position {ticket} not found"}

        pos = positions[0]
        symbol = pos.symbol
        volume = pos.volume
        pos_type = pos.type

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"error": f"Cannot get tick for {symbol}"}

        if pos_type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        symbol_info = mt5.symbol_info(symbol)
        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info and symbol_info.filling_mode & mt5.ORDER_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 50,
            "magic": 888888,
            "comment": "MCP_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "ticket": ticket, "closed_profit": pos.profit}
        else:
            return {"error": f"Close failed: {result.comment if result else 'None'} ({result.retcode if result else 'N/A'})"}

    def modify_sl_tp(self, ticket: int, sl: float = None, tp: float = None) -> dict:
        """Modify SL/TP of a position"""
        if not MT5_AVAILABLE:
            return {"error": "MT5 not available"}

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"error": f"Position {ticket} not found"}

        pos = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": sl if sl is not None else pos.sl,
            "tp": tp if tp is not None else pos.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {"success": True, "ticket": ticket, "new_sl": request["sl"], "new_tp": request["tp"]}
        else:
            return {"error": f"Modify failed: {result.comment if result else 'None'}"}

    def close_losing_positions(self, max_loss: float = None) -> dict:
        """Close all positions exceeding loss limit"""
        if max_loss is None:
            max_loss = MAX_LOSS_DOLLARS

        positions = mt5.positions_get() if MT5_AVAILABLE else []
        if not positions:
            return {"message": "No positions", "closed": []}

        closed = []
        for pos in positions:
            if pos.profit < -max_loss:
                result = self.close_position(pos.ticket)
                closed.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "profit": pos.profit,
                    "result": result
                })

        return {"closed": closed, "count": len(closed)}

    def get_account_summary(self) -> dict:
        """Get full account summary"""
        if not MT5_AVAILABLE or not self.initialized:
            return {"error": "Not connected"}

        info = mt5.account_info()
        if not info:
            return {"error": "Not connected"}

        positions = self.get_positions()
        total_profit = sum(p["profit"] for p in positions)
        losing = [p for p in positions if p["profit"] < 0]
        winning = [p for p in positions if p["profit"] > 0]

        return {
            "account": info.login,
            "server": info.server,
            "balance": info.balance,
            "equity": info.equity,
            "floating_pl": total_profit,
            "margin_used": info.margin,
            "margin_free": info.margin_free,
            "positions_count": len(positions),
            "winning_count": len(winning),
            "losing_count": len(losing),
            "worst_position": min(positions, key=lambda x: x["profit"]) if positions else None,
            "best_position": max(positions, key=lambda x: x["profit"]) if positions else None,
            "positions": positions
        }

    def get_history(self, days: int = 1) -> list:
        """Get closed trades history"""
        if not MT5_AVAILABLE:
            return []

        now = datetime.now()
        from_date = now - timedelta(days=days)

        deals = mt5.history_deals_get(from_date, now)
        if deals is None:
            return []

        result = []
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT:
                result.append({
                    "ticket": deal.ticket,
                    "order": deal.order,
                    "symbol": deal.symbol,
                    "type": "BUY" if deal.type == 0 else "SELL",
                    "volume": deal.volume,
                    "price": deal.price,
                    "profit": deal.profit,
                    "commission": deal.commission,
                    "swap": deal.swap,
                    "magic": deal.magic,
                    "comment": deal.comment,
                    "time": datetime.fromtimestamp(deal.time).isoformat()
                })

        return sorted(result, key=lambda x: x["time"], reverse=True)


# Global manager
manager = MT5Manager()

# MCP Tools Definition
MCP_TOOLS = [
    {
        "name": "mt5_connect",
        "description": "Connect to MT5 account. Keys: ATLAS, BG_INSTANT, BG_CHALLENGE, GL_1, GL_2, GL_3",
        "inputSchema": {
            "type": "object",
            "properties": {
                "account_key": {"type": "string", "description": "Account key from config"}
            }
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
                "max_loss": {"type": "number", "description": "Max loss in dollars (default 1.50)"}
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
    }
]


# FastAPI App
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("MT5 HTTP MCP Server starting...")
    print(f"Available accounts: {list(ACCOUNTS.keys())}")
    yield
    if MT5_AVAILABLE:
        mt5.shutdown()
    print("MT5 HTTP MCP Server stopped")

app = FastAPI(title="MT5 MCP Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def handle_tool_call(tool_name: str, args: dict) -> Any:
    """Execute a tool call"""
    if tool_name == "mt5_connect":
        return manager.connect(args.get("account_key"))
    elif tool_name == "mt5_positions":
        return manager.get_positions()
    elif tool_name == "mt5_summary":
        return manager.get_account_summary()
    elif tool_name == "mt5_close":
        return manager.close_position(args["ticket"])
    elif tool_name == "mt5_close_losers":
        return manager.close_losing_positions(args.get("max_loss"))
    elif tool_name == "mt5_modify_sl":
        return manager.modify_sl_tp(args["ticket"], sl=args["sl"])
    elif tool_name == "mt5_history":
        return manager.get_history(args.get("days", 1))
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ============================================================
# MCP Protocol Endpoints (HTTP Transport)
# ============================================================

@app.post("/mcp")
async def mcp_handler(request: Request):
    """Handle MCP JSON-RPC requests via HTTP POST"""
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
                "serverInfo": {"name": "mt5-mcp-http", "version": "1.0.0"}
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

        try:
            result = handle_tool_call(tool_name, args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)}
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown method: {method}"}
    }


# ============================================================
# MCP SSE Endpoint (for Claude Code)
# ============================================================

@app.get("/sse")
async def sse_endpoint(request: Request):
    """Server-Sent Events endpoint for MCP"""
    async def event_generator():
        # Send initial capabilities
        init_msg = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mt5-mcp-http", "version": "1.0.0"}
            }
        }
        yield f"data: {json.dumps(init_msg)}\n\n"

        # Keep connection alive
        while True:
            if await request.is_disconnected():
                break
            yield f": keepalive\n\n"
            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================
# Simple REST API (alternative to MCP)
# ============================================================

@app.get("/api/accounts")
async def list_accounts():
    """List available account keys"""
    return {"accounts": list(ACCOUNTS.keys())}

@app.post("/api/connect/{account_key}")
async def connect_account(account_key: str):
    """Connect to account"""
    return manager.connect(account_key)

@app.get("/api/positions")
async def get_positions():
    """Get positions"""
    return manager.get_positions()

@app.get("/api/summary")
async def get_summary():
    """Get account summary"""
    return manager.get_account_summary()

@app.post("/api/close/{ticket}")
async def close_position(ticket: int):
    """Close position"""
    return manager.close_position(ticket)

@app.get("/api/history")
async def get_history(days: int = 1):
    """Get trade history"""
    return manager.get_history(days)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "mt5_available": MT5_AVAILABLE,
        "connected_account": manager.connected_account,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
