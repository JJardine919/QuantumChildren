"""
MT5 MCP Server - Direct MetaTrader5 Control for Claude
=======================================================
Gives Claude direct access to manage trading accounts.

Run: python mcp_mt5_server.py
"""

import json
import sys
import asyncio
from typing import Any
from datetime import datetime

import MetaTrader5 as mt5

# Load account configs
try:
    from config_loader import ACCOUNTS, MAX_LOSS_DOLLARS, AGENT_SL_MAX
except ImportError:
    ACCOUNTS = {}
    MAX_LOSS_DOLLARS = 1.50
    AGENT_SL_MAX = 1.00


class MT5Manager:
    """Direct MT5 management"""

    # Known magic numbers from the trading system
    KNOWN_MAGIC = {
        212001, 366001, 365001, 113001, 113002, 107001,
        888888, 999999, 20251222, 20251227,
    }

    def __init__(self):
        self.connected_account = None

    def connect(self, account_key: str = None) -> dict:
        """Connect to MT5 account"""
        # ALWAYS shutdown first to prevent terminal mixing
        mt5.shutdown()

        if account_key and account_key in ACCOUNTS:
            acc = ACCOUNTS[account_key]
            terminal_path = acc.get('terminal_path')

            if terminal_path:
                if not mt5.initialize(path=terminal_path):
                    return {"error": f"Init failed: {mt5.last_error()}"}
            else:
                if not mt5.initialize():
                    return {"error": f"Init failed: {mt5.last_error()}"}

            if acc.get('password'):
                if not mt5.login(acc['account'], password=acc['password'], server=acc['server']):
                    return {"error": f"Login failed: {mt5.last_error()}"}
        else:
            if not mt5.initialize():
                return {"error": f"Init failed: {mt5.last_error()}"}

        info = mt5.account_info()
        if info:
            self.connected_account = info.login
            return {
                "account": info.login,
                "balance": info.balance,
                "equity": info.equity,
                "profit": info.profit,
                "margin_free": info.margin_free
            }
        return {"error": "Could not get account info"}

    def get_positions(self) -> list:
        """Get all open positions"""
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

        # Close in opposite direction
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
            max_loss = AGENT_SL_MAX

        positions = mt5.positions_get()
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
        info = mt5.account_info()
        if not info:
            return {"error": "Not connected"}

        positions = self.get_positions()
        total_profit = sum(p["profit"] for p in positions)
        losing = [p for p in positions if p["profit"] < 0]
        winning = [p for p in positions if p["profit"] > 0]

        return {
            "account": info.login,
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

    def scan_no_sl(self) -> dict:
        """Scan all positions for missing stop losses and rogue magic numbers"""
        positions = mt5.positions_get()
        if positions is None:
            return {"error": "Could not get positions"}

        no_sl = []
        no_tp = []
        rogue = []

        for pos in positions:
            direction = "BUY" if pos.type == 0 else "SELL"
            entry = {
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": direction,
                "volume": pos.volume,
                "open_price": pos.price_open,
                "current_price": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "magic": pos.magic,
                "comment": pos.comment,
                "time": datetime.fromtimestamp(pos.time).isoformat(),
            }

            if pos.sl == 0.0:
                no_sl.append(entry)
            if pos.tp == 0.0:
                no_tp.append(entry)
            if pos.magic not in self.KNOWN_MAGIC:
                rogue.append(entry)

        return {
            "total_positions": len(positions),
            "missing_sl": len(no_sl),
            "missing_tp": len(no_tp),
            "rogue_magic": len(rogue),
            "positions_without_sl": no_sl,
            "positions_without_tp": no_tp,
            "rogue_trades": rogue,
        }

    def force_emergency_sl(self, ticket: int, max_loss_dollars: float = 2.0) -> dict:
        """Apply emergency SL to a position based on max dollar loss from CURRENT price"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"error": f"Position {ticket} not found"}

        pos = positions[0]
        if pos.sl != 0.0:
            return {"info": f"Position {ticket} already has SL={pos.sl}", "current_sl": pos.sl}

        symbol_info = mt5.symbol_info(pos.symbol)
        tick = mt5.symbol_info_tick(pos.symbol)
        if symbol_info is None or tick is None:
            return {"error": f"Cannot get symbol info/tick for {pos.symbol}"}

        tick_value = symbol_info.trade_tick_value
        tick_size = symbol_info.trade_tick_size
        point = symbol_info.point
        digits = symbol_info.digits
        stops_level = symbol_info.trade_stops_level
        spread = symbol_info.spread

        # Calculate SL distance for max_loss_dollars
        if tick_value > 0 and pos.volume > 0:
            sl_ticks = max_loss_dollars / (tick_value * pos.volume)
            sl_distance = sl_ticks * tick_size
        else:
            sl_distance = 200 * point

        # Respect broker minimum stop distance
        min_sl_distance = (stops_level + spread + 20) * point
        if sl_distance < min_sl_distance:
            sl_distance = min_sl_distance

        # Calculate SL from CURRENT price (not open price)
        if pos.type == mt5.POSITION_TYPE_BUY:
            current = tick.bid
            sl_price = round(current - sl_distance, digits)
        else:
            current = tick.ask
            sl_price = round(current + sl_distance, digits)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": sl_price,
            "tp": pos.tp,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                "success": True,
                "ticket": ticket,
                "new_sl": sl_price,
                "current_price": current,
                "sl_distance": round(sl_distance, digits),
                "max_further_loss": max_loss_dollars,
            }
        else:
            return {
                "error": f"SL modify failed: {result.comment if result else 'None'} ({result.retcode if result else 'N/A'})",
                "attempted_sl": sl_price,
            }

    def get_history(self, days: int = 1) -> list:
        """Get closed trades history"""
        from datetime import timedelta

        now = datetime.now()
        from_date = now - timedelta(days=days)

        deals = mt5.history_deals_get(from_date, now)
        if deals is None:
            return []

        result = []
        for deal in deals:
            if deal.entry == mt5.DEAL_ENTRY_OUT:  # Closing deals only
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


# MCP Protocol Implementation
manager = MT5Manager()


def handle_request(request: dict) -> dict:
    """Handle MCP request"""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mt5-mcp", "version": "1.0.0"}
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
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
                    },
                    {
                        "name": "mt5_scan_no_sl",
                        "description": "Scan all positions for missing stop losses and rogue trades (unknown magic numbers)",
                        "inputSchema": {"type": "object", "properties": {}}
                    },
                    {
                        "name": "mt5_force_sl",
                        "description": "Apply emergency stop loss to a position missing SL. Sets SL based on max dollar loss from CURRENT price.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "ticket": {"type": "integer", "description": "Position ticket"},
                                "max_loss_dollars": {"type": "number", "description": "Max further loss in dollars from current price (default 2.00)"}
                            },
                            "required": ["ticket"]
                        }
                    }
                ]
            }
        }

    elif method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        try:
            if tool_name == "mt5_connect":
                result = manager.connect(args.get("account_key"))
            elif tool_name == "mt5_positions":
                result = manager.get_positions()
            elif tool_name == "mt5_summary":
                result = manager.get_account_summary()
            elif tool_name == "mt5_close":
                result = manager.close_position(args["ticket"])
            elif tool_name == "mt5_close_losers":
                result = manager.close_losing_positions(args.get("max_loss"))
            elif tool_name == "mt5_modify_sl":
                result = manager.modify_sl_tp(args["ticket"], sl=args["sl"])
            elif tool_name == "mt5_history":
                result = manager.get_history(args.get("days", 1))
            elif tool_name == "mt5_scan_no_sl":
                result = manager.scan_no_sl()
            elif tool_name == "mt5_force_sl":
                result = manager.force_emergency_sl(args["ticket"], args.get("max_loss_dollars", 2.0))
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

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


def main():
    """Run MCP server over stdio"""
    print("MT5 MCP Server starting...", file=sys.stderr)

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(e)}
            }), flush=True)


if __name__ == "__main__":
    main()
