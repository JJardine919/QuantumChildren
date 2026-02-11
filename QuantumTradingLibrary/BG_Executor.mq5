//+------------------------------------------------------------------+
//|                                              BG_Executor.mq5    |
//|                                  Copyright 2026, Quantum Library |
//+------------------------------------------------------------------+
#property copyright "Quantum Library"
#property link      "https://www.mql5.com"
#property version   "2.00"
#property description "Blue Guardian Sniper Executor - Fixed v2"

//--- Input parameters
input string SignalFile = "etare_signals.json";      // Signal file from Python
input int    MagicNumber = 365060;                   // Blue Guardian Magic
input double LotSize = 0.03;                         // Fixed Lot Size
input int    Slippage = 50;                          // Max slippage
input bool   TradeEnabled = true;                    // ENABLED FOR LIVE TRADING

input group "=== STEALTH SETTINGS ==="
input bool   StealthMode = false;                    // Stealth Mode (hide EA identifiers)

input group "=== WEEKEND PROTECTION ==="
input bool   WeekendCloseEnabled = true;             // Close positions before weekend

//--- Global variables
datetime lastSignalCheck = 0;
int checkInterval = 10;  // Check signals every 10 seconds

//--- Virtual position tracking for hidden SL/TP (array-based)
struct VirtualPosition
{
   ulong  ticket;
   double entryPrice;
   double virtualSL;
   double virtualTP;
   bool   active;
};

#define MAX_VIRTUAL_POSITIONS 20
VirtualPosition g_virtualPositions[MAX_VIRTUAL_POSITIONS];
int g_virtualCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("BLUE GUARDIAN EXECUTOR v2.0 LOADED");
    Print("Account: ", AccountInfoInteger(ACCOUNT_LOGIN));
    Print("Lot Size: ", LotSize);
    Print("Trading Enabled: ", TradeEnabled);
    Print("Stealth Mode: ", StealthMode);
    Print("========================================");

    // Initialize virtual position array
    for(int i = 0; i < MAX_VIRTUAL_POSITIONS; i++)
        g_virtualPositions[i].active = false;
    g_virtualCount = 0;

    // CRITICAL: Recover existing positions on startup/restart
    RecoverExistingPositions();

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Recover existing positions that match our magic number           |
//| Rebuilds virtual SL/TP so positions aren't orphaned on restart   |
//+------------------------------------------------------------------+
void RecoverExistingPositions()
{
    int recovered = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;

        if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
           PositionGetString(POSITION_SYMBOL) == _Symbol)
        {
            double entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            double volume = PositionGetDouble(POSITION_VOLUME);
            ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

            // Recalculate virtual SL/TP from entry price
            double sl_dist = CalculateSLDistance(volume);
            double tp_dist = sl_dist * 3.0;

            double sl = (posType == POSITION_TYPE_BUY) ? entryPrice - sl_dist : entryPrice + sl_dist;
            double tp = (posType == POSITION_TYPE_BUY) ? entryPrice + tp_dist : entryPrice - tp_dist;

            if(AddVirtualPosition(ticket, entryPrice, sl, tp))
            {
                recovered++;
                Print("RECOVERED position #", ticket, " SL=", DoubleToString(sl, _Digits),
                      " TP=", DoubleToString(tp, _Digits));
            }
        }
    }
    if(recovered > 0)
        Print("Recovered ", recovered, " existing position(s) with virtual SL/TP");
}

//+------------------------------------------------------------------+
//| Calculate SL distance for given volume                           |
//+------------------------------------------------------------------+
double CalculateSLDistance(double volume)
{
    double MAX_LOSS_DOLLARS = 1.00;
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    double sl_dist = 0;
    if(tickValue > 0 && volume > 0)
        sl_dist = (MAX_LOSS_DOLLARS / (tickValue * volume)) * tickSize;

    // Fallback if calculation fails
    if(sl_dist <= 0) sl_dist = 500.0 * SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    return sl_dist;
}

//+------------------------------------------------------------------+
//| Add a virtual position to the tracking array                     |
//+------------------------------------------------------------------+
bool AddVirtualPosition(ulong ticket, double entryPrice, double sl, double tp)
{
    // Check if already tracked
    for(int i = 0; i < MAX_VIRTUAL_POSITIONS; i++)
    {
        if(g_virtualPositions[i].active && g_virtualPositions[i].ticket == ticket)
            return false;  // Already tracked
    }

    // Find empty slot
    for(int i = 0; i < MAX_VIRTUAL_POSITIONS; i++)
    {
        if(!g_virtualPositions[i].active)
        {
            g_virtualPositions[i].ticket = ticket;
            g_virtualPositions[i].entryPrice = entryPrice;
            g_virtualPositions[i].virtualSL = sl;
            g_virtualPositions[i].virtualTP = tp;
            g_virtualPositions[i].active = true;
            g_virtualCount++;
            return true;
        }
    }

    Print("WARNING: Virtual position array full! Cannot track ticket #", ticket);
    return false;
}

//+------------------------------------------------------------------+
//| Remove a virtual position from tracking                          |
//+------------------------------------------------------------------+
void RemoveVirtualPosition(int index)
{
    if(index >= 0 && index < MAX_VIRTUAL_POSITIONS)
    {
        g_virtualPositions[index].active = false;
        g_virtualCount--;
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Weekend close check - BEFORE any trade logic
    CheckWeekendClose();

    // ALWAYS manage virtual SL/TP on every tick
    ManageVirtualSLTP();

    if(TimeCurrent() - lastSignalCheck < checkInterval) return;
    lastSignalCheck = TimeCurrent();

    ReadAndExecuteSignals();
}

//+------------------------------------------------------------------+
//| Manage virtual (hidden) SL/TP - iterates ALL tracked positions   |
//+------------------------------------------------------------------+
void ManageVirtualSLTP()
{
    for(int idx = 0; idx < MAX_VIRTUAL_POSITIONS; idx++)
    {
        if(!g_virtualPositions[idx].active) continue;

        ulong ticket = g_virtualPositions[idx].ticket;

        // Check if position still exists
        if(!PositionSelectByTicket(ticket))
        {
            // Position closed externally (manual, watchdog, etc.)
            Print("Position #", ticket, " no longer exists - removing from tracking");
            RemoveVirtualPosition(idx);
            continue;
        }

        double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
        ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
        double volume = PositionGetDouble(POSITION_VOLUME);

        bool hitSL = false;
        bool hitTP = false;

        if(posType == POSITION_TYPE_BUY)
        {
            hitSL = (currentPrice <= g_virtualPositions[idx].virtualSL);
            hitTP = (currentPrice >= g_virtualPositions[idx].virtualTP);
        }
        else
        {
            hitSL = (currentPrice >= g_virtualPositions[idx].virtualSL);
            hitTP = (currentPrice <= g_virtualPositions[idx].virtualTP);
        }

        if(hitSL || hitTP)
        {
            Print(hitSL ? "HIDDEN SL HIT" : "HIDDEN TP HIT",
                  " - Closing position #", ticket, " at ", currentPrice);

            MqlTradeRequest request;
            MqlTradeResult result;
            ZeroMemory(request);
            ZeroMemory(result);

            request.action = TRADE_ACTION_DEAL;
            request.position = ticket;
            request.symbol = _Symbol;
            request.volume = volume;
            request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
            request.price = (posType == POSITION_TYPE_BUY) ?
                            SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                            SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            request.deviation = Slippage;
            request.magic = StealthMode ? 0 : MagicNumber;
            request.type_filling = GetFillingMode();

            if(OrderSend(request, result))
            {
                if(result.retcode == TRADE_RETCODE_DONE)
                {
                    Print("Position #", ticket, " closed successfully");
                    RemoveVirtualPosition(idx);
                }
            }
            else
            {
                Print("ERROR closing position #", ticket, ": ", GetLastError());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Read signals and execute                                         |
//+------------------------------------------------------------------+
void ReadAndExecuteSignals()
{
    int fileHandle = FileOpen(SignalFile, FILE_READ|FILE_TXT|FILE_ANSI);
    if(fileHandle == INVALID_HANDLE) return;

    string json = "";
    while(!FileIsEnding(fileHandle)) json += FileReadString(fileHandle);
    FileClose(fileHandle);

    if(json == "") return;

    // Check for our symbol
    string currentSymbol = _Symbol;
    if(StringFind(json, "\"" + currentSymbol + "\"") < 0) return;

    // Parse Action
    string action = ExtractStringValue(json, "action");
    double confidence = ExtractDoubleValue(json, "confidence");

    if(!TradeEnabled)
    {
        Print("DRY RUN >> Signal: ", action, " (", DoubleToString(confidence*100, 1), "%)");
        return;
    }

    // Execution Logic
    if(action == "BUY")
    {
        ExecuteTrade(ORDER_TYPE_BUY);
    }
    else if(action == "SELL")
    {
        ExecuteTrade(ORDER_TYPE_SELL);
    }
}

//+------------------------------------------------------------------+
//| Execute Trade                                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE type)
{
    // Check if position already exists for this magic
    if(PositionExists()) return;

    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    double price = (type == ORDER_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);

    // Dollar-based SL calculation: $1.00 max loss
    double sl_dist = CalculateSLDistance(LotSize);

    // TP = SL * 3 (3:1 R:R)
    double tp_dist = sl_dist * 3.0;

    // Calculate virtual SL/TP levels (hidden from broker)
    double sl = (type == ORDER_TYPE_BUY) ? price - sl_dist : price + sl_dist;
    double tp = (type == ORDER_TYPE_BUY) ? price + tp_dist : price - tp_dist;

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = LotSize;
    request.type = type;
    request.price = price;
    request.sl = 0;  // HIDDEN - managed internally
    request.tp = 0;  // HIDDEN - managed internally
    request.deviation = Slippage;
    request.magic = StealthMode ? 0 : MagicNumber;
    request.comment = StealthMode ? "" : "BG_Redux";
    request.type_time = ORDER_TIME_GTC;
    request.type_filling = GetFillingMode();

    if(!OrderSend(request, result))
    {
        Print("ERROR: OrderSend failed: ", GetLastError());
    }
    else
    {
        if(result.retcode == TRADE_RETCODE_DONE)
        {
            // Track virtual SL/TP in array
            if(!AddVirtualPosition(result.order, price, sl, tp))
                Print("WARNING: Could not track virtual SL/TP for order #", result.order);

            Print("SUCCESS: ", (type==ORDER_TYPE_BUY ? "BUY" : "SELL"), " executed.");
            Print("  Hidden SL: ", DoubleToString(sl, _Digits), " | Hidden TP: ", DoubleToString(tp, _Digits));
            Print("  SL dist: ", DoubleToString(sl_dist, 2), " | TP dist: ", DoubleToString(tp_dist, 2));
            Print("  Max loss: $1.00");

            // === EMERGENCY BROKER-SIDE SL (catastrophic backstop - 5x virtual SL) ===
            // If MT5 crashes or EA is removed, this wide SL protects the position
            {
                double emergency_sl_dist = sl_dist * 5.0;  // 5x normal = ~$5.00 max loss
                double emergencySL = (type == ORDER_TYPE_BUY) ?
                    price - emergency_sl_dist :
                    price + emergency_sl_dist;
                emergencySL = NormalizeDouble(emergencySL, _Digits);

                MqlTradeRequest slReq;
                MqlTradeResult slRes;
                ZeroMemory(slReq);
                ZeroMemory(slRes);
                slReq.action = TRADE_ACTION_SLTP;
                slReq.position = result.order;
                slReq.symbol = _Symbol;
                slReq.sl = emergencySL;
                slReq.tp = 0;
                if(!OrderSend(slReq, slRes))
                    Print("WARNING: Could not set emergency SL: ", GetLastError());
                else if(slRes.retcode == TRADE_RETCODE_DONE)
                    Print("  Emergency backstop SL set at ", DoubleToString(emergencySL, _Digits),
                          " (", DoubleToString(emergency_sl_dist, 2), " dist, ~$5.00 max loss)");
                else
                    Print("WARNING: Emergency SL rejected: ", slRes.retcode, " - ", slRes.comment);
            }
        }
        else
        {
            Print("Order rejected: ", result.comment);
        }
    }
}

bool PositionExists()
{
    for(int i=PositionsTotal()-1; i>=0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber) return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Weekend close - close all positions before Friday market close   |
//+------------------------------------------------------------------+
void CheckWeekendClose()
{
    if(!WeekendCloseEnabled) return;

    // Skip for crypto symbols (24/7 markets)
    string sym = _Symbol;
    if(StringFind(sym, "BTC") >= 0 || StringFind(sym, "ETH") >= 0 ||
       StringFind(sym, "LTC") >= 0 || StringFind(sym, "XRP") >= 0) return;

    MqlDateTime dt;
    TimeCurrent(dt);

    // Friday = day_of_week 5, close by 16:45 server time
    if(dt.day_of_week == 5 && (dt.hour > 16 || (dt.hour == 16 && dt.min >= 45)))
    {
        CloseAllPositionsForWeekend();
    }
}

//+------------------------------------------------------------------+
//| Close all positions matching our MagicNumber for weekend         |
//+------------------------------------------------------------------+
void CloseAllPositionsForWeekend()
{
    int closedCount = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(!PositionSelectByTicket(ticket)) continue;
        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

        double volume = PositionGetDouble(POSITION_VOLUME);
        ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

        MqlTick tick;
        if(!SymbolInfoTick(_Symbol, tick)) continue;

        MqlTradeRequest request;
        MqlTradeResult result;
        ZeroMemory(request);
        ZeroMemory(result);

        request.action = TRADE_ACTION_DEAL;
        request.position = ticket;
        request.symbol = _Symbol;
        request.volume = volume;
        request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
        request.price = (posType == POSITION_TYPE_BUY) ? tick.bid : tick.ask;
        request.deviation = Slippage;
        request.magic = StealthMode ? 0 : MagicNumber;
        request.comment = StealthMode ? "" : "WeekendClose";
        request.type_filling = GetFillingMode();

        if(OrderSend(request, result) && result.retcode == TRADE_RETCODE_DONE)
        {
            Print("WEEKEND: Closed ticket #", ticket);
            closedCount++;

            // Remove from virtual tracking
            for(int v = 0; v < MAX_VIRTUAL_POSITIONS; v++)
            {
                if(g_virtualPositions[v].active && g_virtualPositions[v].ticket == ticket)
                {
                    RemoveVirtualPosition(v);
                    break;
                }
            }
        }
        else
        {
            Print("WEEKEND: Failed to close ticket #", ticket, " error=", GetLastError());
        }
    }

    if(closedCount > 0)
        Print("WEEKEND CLOSE: Closed ", closedCount, " position(s) before market close");
}

//+------------------------------------------------------------------+
//| Get broker-supported filling mode (fixes hard-coded IOC bug)     |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
    uint filling = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);

    if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
        return ORDER_FILLING_FOK;

    if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
        return ORDER_FILLING_IOC;

    return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| JSON Helpers                                                     |
//+------------------------------------------------------------------+
string ExtractStringValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return "";
    int valueStart = StringFind(json, "\"", keyPos + StringLen(key) + 3);
    if(valueStart < 0) return "";
    int valueEnd = StringFind(json, "\"", valueStart + 1);
    return StringSubstr(json, valueStart + 1, valueEnd - valueStart - 1);
}

double ExtractDoubleValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return 0;
    int colonPos = StringFind(json, ":", keyPos);
    string remaining = StringSubstr(json, colonPos + 1);
    int endPos = StringFind(remaining, ",");
    if(endPos < 0) endPos = StringFind(remaining, "}");
    return StringToDouble(remaining);
}
//+------------------------------------------------------------------+
