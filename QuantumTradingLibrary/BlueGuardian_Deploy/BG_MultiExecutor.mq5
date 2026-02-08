//+------------------------------------------------------------------+
//|                                          BG_MultiExecutor.mq5    |
//|                                  Copyright 2026, Quantum Library |
//|                           Blue Guardian Multi-Account Executor   |
//|                                   v3.0 - Dynamic SL/TP Edition   |
//+------------------------------------------------------------------+
#property copyright "Quantum Library"
#property link      "https://www.mql5.com"
#property version   "3.00"
#property description "Blue Guardian Multi-Account Signal Executor"
#property description "Dynamic Trailing SL + Scaling TP - Maintains 1:3 R:R"

//--- Input parameters (SET PER ACCOUNT)
input string   AccountName      = "BG_INSTANT_1";     // Account identifier
input int      MagicNumber      = 100001;             // Unique magic per account
input double   MaxLotSize       = 0.04;               // Maximum lot size
input double   DailyDDLimit     = 4.0;                // Daily drawdown limit %
input double   MaxDDLimit       = 8.0;                // Max drawdown limit %
input int      Slippage         = 50;                 // Max slippage points
input bool     TradeEnabled     = true;               // ENABLED FOR LIVE TRADING
input int      CheckInterval    = 10;                 // Signal check interval (seconds)
input double   MaxSpreadPoints  = 100;                // Max spread to allow trade

//=============================================================================
// DYNAMIC SL/TP SETTINGS - Adjustable via inputs
//=============================================================================
input group "=== Dynamic SL/TP Settings ==="
input double   SL_ATR_MULT         = 1.5;       // SL ATR Multiplier (SL = ATR * this)
input double   RR_RATIO            = 3.0;        // Risk:Reward Ratio (1:N)
input bool     USE_TRAILING_SL     = true;       // Trailing Stop Enabled
input bool     USE_DYNAMIC_TP      = true;       // Dynamic TP Enabled
input double   BREAKEVEN_TRIGGER   = 0.5;        // Break-Even Trigger (% of SL dist)
input double   BREAKEVEN_BUFFER    = 5.0;        // Break-Even Buffer (points)
input double   TRAIL_START_MULT    = 1.0;        // Trail Start (% of SL profit)
input double   TRAIL_DISTANCE      = 25.0;       // Trail Distance (points)
input bool     USE_PARTIAL_TP      = true;       // Partial TP Enabled
input double   PARTIAL_TP_PCT      = 50.0;       // Partial TP % to close
input double   PARTIAL_TP_TRIGGER  = 0.5;        // Partial TP Trigger (% of full TP)
//=============================================================================

//--- Global variables
datetime       g_lastCheck = 0;
double         g_startBalance = 0;
double         g_highWaterMark = 0;
double         g_dailyStartBalance = 0;
datetime       g_lastDayReset = 0;
int            g_tradesToday = 0;
bool           g_blocked = false;
string         g_blockReason = "";
string         g_lastSignalTimestamp = "";

//--- Position tracking for dynamic management (HIDDEN SL/TP)
double         g_entryPrice = 0;
double         g_initialSL = 0;     // Virtual SL (not sent to broker)
double         g_initialTP = 0;     // Virtual TP (not sent to broker)
double         g_currentVirtualSL = 0;  // Current virtual SL after trailing
double         g_currentVirtualTP = 0;  // Current virtual TP after dynamic updates
bool           g_breakevenHit = false;
bool           g_partialClosed = false;
bool           g_trailingActive = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("========================================");
    Print("BLUE GUARDIAN MULTI-EXECUTOR v3.0");
    Print("   DYNAMIC SL/TP EDITION (HARDCODED)");
    Print("========================================");
    Print("Account Name: ", AccountName);
    Print("Account ID: ", AccountInfoInteger(ACCOUNT_LOGIN));
    Print("Magic Number: ", MagicNumber);
    Print("Max Lot: ", MaxLotSize);
    Print("Daily DD Limit: ", DailyDDLimit, "%");
    Print("Max DD Limit: ", MaxDDLimit, "%");
    Print("Max Spread: ", MaxSpreadPoints, " points");
    Print("--- HARDCODED SL/TP Settings ---");
    Print("SL: ATR x ", SL_ATR_MULT);
    Print("R:R Ratio: 1:", RR_RATIO);
    Print("--- Dynamic Features (LOCKED) ---");
    Print("Trailing SL: ENABLED (hardcoded)");
    Print("Dynamic TP: ENABLED (hardcoded)");
    Print("Partial TP: ", PARTIAL_TP_PCT, "% (hardcoded)");
    Print("Breakeven @ ", BREAKEVEN_TRIGGER * 100, "% of SL");
    Print("Trail Start @ ", TRAIL_START_MULT * 100, "% of SL");
    Print("Trail Distance: ", TRAIL_DISTANCE);
    Print("Trading Enabled: ", TradeEnabled);
    Print("========================================");

    g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    g_highWaterMark = g_startBalance;
    g_dailyStartBalance = g_startBalance;
    g_lastDayReset = TimeCurrent();

    // Safety check
    if(g_startBalance <= 0)
    {
        Print("ERROR: Invalid starting balance. Cannot initialize.");
        return INIT_FAILED;
    }

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("BG Executor shutting down. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // ALWAYS manage open positions (trailing/dynamic) on every tick
    ManageOpenPositions();

    // Check interval for signal processing
    if(TimeCurrent() - g_lastCheck < CheckInterval) return;
    g_lastCheck = TimeCurrent();

    // Daily reset
    CheckDailyReset();

    // Update high water mark
    UpdateHighWaterMark();

    // Check drawdown limits
    if(!CheckDrawdownLimits())
    {
        if(!g_blocked)
        {
            g_blocked = true;
            Print("BLOCKED: ", g_blockReason);
        }
        return;
    }

    // Read and execute signals
    ReadAndExecuteSignals();
}

//+------------------------------------------------------------------+
//| Daily reset check                                                |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
    MqlDateTime now, last;
    TimeToStruct(TimeCurrent(), now);
    TimeToStruct(g_lastDayReset, last);

    if(now.day != last.day || now.mon != last.mon || now.year != last.year)
    {
        g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
        g_tradesToday = 0;
        g_lastDayReset = TimeCurrent();
        g_blocked = false;
        g_blockReason = "";
        Print("Daily reset. New balance baseline: $", DoubleToString(g_dailyStartBalance, 2));
    }
}

//+------------------------------------------------------------------+
//| Update high water mark for accurate drawdown calculation         |
//+------------------------------------------------------------------+
void UpdateHighWaterMark()
{
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    if(currentBalance > g_highWaterMark)
    {
        g_highWaterMark = currentBalance;
    }
}

//+------------------------------------------------------------------+
//| Check drawdown limits                                            |
//+------------------------------------------------------------------+
bool CheckDrawdownLimits()
{
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);

    // Use minimum of balance and equity for safety
    double current = MathMin(currentBalance, equity);

    // Guard against division by zero
    if(g_dailyStartBalance <= 0 || g_highWaterMark <= 0)
    {
        g_blockReason = "Invalid balance baseline - cannot calculate drawdown";
        return false;
    }

    // Daily drawdown check (from day's starting balance)
    double dailyDD = ((g_dailyStartBalance - current) / g_dailyStartBalance) * 100;
    if(dailyDD >= DailyDDLimit * 0.9)  // 90% of limit = warning zone
    {
        g_blockReason = StringFormat("Daily DD %.2f%% approaching limit %.2f%%", dailyDD, DailyDDLimit);
        if(dailyDD >= DailyDDLimit)
        {
            return false;
        }
    }

    // Max drawdown check (from high water mark)
    double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
    if(maxDD >= MaxDDLimit * 0.9)
    {
        g_blockReason = StringFormat("Max DD %.2f%% approaching limit %.2f%%", maxDD, MaxDDLimit);
        if(maxDD >= MaxDDLimit)
        {
            return false;
        }
    }

    return true;
}

//+------------------------------------------------------------------+
//| Manage open positions - Trailing SL, Dynamic TP, Partial Close   |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
    // Find our position
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(!PositionSelectByTicket(ticket)) continue;

        if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

        // Get position details
        double posOpenPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double posVolume = PositionGetDouble(POSITION_VOLUME);
        ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

        // Use virtual SL/TP (hidden from broker)
        double posSL = g_currentVirtualSL;
        double posTP = g_currentVirtualTP;

        // Get current price
        MqlTick tick;
        if(!SymbolInfoTick(_Symbol, tick)) return;

        double currentPrice = (posType == POSITION_TYPE_BUY) ? tick.bid : tick.ask;

        // Check if virtual SL or TP hit - close immediately
        bool hitSL = false;
        bool hitTP = false;
        if(posSL > 0)
        {
            if(posType == POSITION_TYPE_BUY)
            {
                hitSL = (currentPrice <= posSL);
                hitTP = (posTP > 0 && currentPrice >= posTP);
            }
            else
            {
                hitSL = (currentPrice >= posSL);
                hitTP = (posTP > 0 && currentPrice <= posTP);
            }
        }

        if(hitSL || hitTP)
        {
            Print(hitSL ? "[HIDDEN SL HIT]" : "[HIDDEN TP HIT]", " Closing at ", DoubleToString(currentPrice, _Digits));
            ClosePartialPosition(ticket, posVolume, posType);
            g_breakevenHit = false;
            g_partialClosed = false;
            g_trailingActive = false;
            g_entryPrice = 0;
            g_initialSL = 0;
            g_initialTP = 0;
            g_currentVirtualSL = 0;
            g_currentVirtualTP = 0;
            return;
        }

        // Calculate profit distance
        double profitDist = (posType == POSITION_TYPE_BUY) ?
                           (currentPrice - posOpenPrice) :
                           (posOpenPrice - currentPrice);

        // Calculate initial risk (SL distance from entry)
        double riskDist = (posType == POSITION_TYPE_BUY) ?
                         (posOpenPrice - posSL) :
                         (posSL - posOpenPrice);

        // Ensure we have valid risk distance
        if(riskDist <= 0)
        {
            riskDist = CalculateATR(14) * SL_ATR_MULT;
        }

        // ===== 1. BREAKEVEN LOGIC (virtual) =====
        if(USE_TRAILING_SL && !g_breakevenHit && BREAKEVEN_TRIGGER > 0)
        {
            double beThreshold = riskDist * BREAKEVEN_TRIGGER;

            if(profitDist >= beThreshold)
            {
                double newSL;
                if(posType == POSITION_TYPE_BUY)
                {
                    newSL = posOpenPrice + BREAKEVEN_BUFFER;
                    if(newSL > g_currentVirtualSL)
                    {
                        g_currentVirtualSL = newSL;
                        g_breakevenHit = true;
                        Print("[BREAKEVEN] Virtual SL moved to ", DoubleToString(newSL, _Digits), " (+", BREAKEVEN_BUFFER, " buffer)");
                    }
                }
                else // SELL
                {
                    newSL = posOpenPrice - BREAKEVEN_BUFFER;
                    if(newSL < g_currentVirtualSL)
                    {
                        g_currentVirtualSL = newSL;
                        g_breakevenHit = true;
                        Print("[BREAKEVEN] Virtual SL moved to ", DoubleToString(newSL, _Digits), " (-", BREAKEVEN_BUFFER, " buffer)");
                    }
                }
            }
        }

        // ===== 2. TRAILING STOP LOGIC (virtual) =====
        if(USE_TRAILING_SL && g_breakevenHit)
        {
            double trailThreshold = riskDist * TRAIL_START_MULT;

            if(profitDist >= trailThreshold)
            {
                g_trailingActive = true;
                double newSL;

                if(posType == POSITION_TYPE_BUY)
                {
                    newSL = currentPrice - TRAIL_DISTANCE;
                    // Only move SL up, never down
                    if(newSL > g_currentVirtualSL)
                    {
                        g_currentVirtualSL = newSL;
                        Print("[TRAILING] Virtual SL trailed to ", DoubleToString(newSL, _Digits),
                              " | Profit: ", DoubleToString(profitDist, 2));
                    }
                }
                else // SELL
                {
                    newSL = currentPrice + TRAIL_DISTANCE;
                    // Only move SL down, never up
                    if(newSL < g_currentVirtualSL)
                    {
                        g_currentVirtualSL = newSL;
                        Print("[TRAILING] Virtual SL trailed to ", DoubleToString(newSL, _Digits),
                              " | Profit: ", DoubleToString(profitDist, 2));
                    }
                }
            }
        }

        // ===== 3. DYNAMIC TP - Scale TP while maintaining R:R (virtual) =====
        if(USE_DYNAMIC_TP && g_trailingActive)
        {
            // Recalculate TP based on current virtual SL to maintain R:R ratio
            double currentRisk = (posType == POSITION_TYPE_BUY) ?
                                (currentPrice - g_currentVirtualSL) :
                                (g_currentVirtualSL - currentPrice);

            if(currentRisk > 0)
            {
                double targetProfit = currentRisk * RR_RATIO;
                double newTP;

                if(posType == POSITION_TYPE_BUY)
                {
                    newTP = currentPrice + targetProfit;
                    // Only move TP up
                    if(newTP > g_currentVirtualTP)
                    {
                        g_currentVirtualTP = newTP;
                        Print("[DYNAMIC TP] Virtual TP scaled to ", DoubleToString(newTP, _Digits),
                              " | Maintaining 1:", RR_RATIO, " R:R");
                    }
                }
                else // SELL
                {
                    newTP = currentPrice - targetProfit;
                    // Only move TP down (lower is better for sells)
                    if(newTP < g_currentVirtualTP || g_currentVirtualTP == 0)
                    {
                        g_currentVirtualTP = newTP;
                        Print("[DYNAMIC TP] Virtual TP scaled to ", DoubleToString(newTP, _Digits),
                              " | Maintaining 1:", RR_RATIO, " R:R");
                    }
                }
            }
        }

        // ===== 4. PARTIAL TAKE PROFIT (using virtual TP) =====
        if(USE_PARTIAL_TP && !g_partialClosed)
        {
            double tpDist = (posType == POSITION_TYPE_BUY) ?
                           (g_currentVirtualTP - posOpenPrice) :
                           (posOpenPrice - g_currentVirtualTP);

            double partialTrigger = tpDist * PARTIAL_TP_TRIGGER;

            if(profitDist >= partialTrigger && posVolume > SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN))
            {
                double closeVolume = NormalizeVolume(posVolume * (PARTIAL_TP_PCT / 100.0));

                if(closeVolume >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN))
                {
                    if(ClosePartialPosition(ticket, closeVolume, posType))
                    {
                        g_partialClosed = true;
                        Print("[PARTIAL TP] Closed ", PARTIAL_TP_PCT, "% (",
                              DoubleToString(closeVolume, 2), " lots) at ",
                              DoubleToString(profitDist, 2), " profit distance");
                    }
                }
            }
        }

        return;  // Only manage one position at a time
    }

    // No position found - reset tracking variables
    if(g_breakevenHit || g_partialClosed || g_trailingActive)
    {
        g_breakevenHit = false;
        g_partialClosed = false;
        g_trailingActive = false;
        g_entryPrice = 0;
        g_initialSL = 0;
        g_initialTP = 0;
    }
}

//+------------------------------------------------------------------+
//| Modify position stop loss (Virtual Only - NOT sent to broker)    |
//+------------------------------------------------------------------+
bool ModifyPositionSL(ulong ticket, double newSL)
{
    // Virtual SL management only - no broker modification
    // The caller updates g_currentVirtualSL after this returns
    // SL hit detection happens in ManageOpenPositions()
    return PositionSelectByTicket(ticket);
}

//+------------------------------------------------------------------+
//| Modify position take profit (Virtual Only - NOT sent to broker)  |
//+------------------------------------------------------------------+
bool ModifyPositionTP(ulong ticket, double newTP)
{
    // Virtual TP management only - no broker modification
    // The caller updates g_currentVirtualTP after this returns
    // TP hit detection happens in ManageOpenPositions()
    return PositionSelectByTicket(ticket);
}

//+------------------------------------------------------------------+
//| Close partial position                                           |
//+------------------------------------------------------------------+
bool ClosePartialPosition(ulong ticket, double volume, ENUM_POSITION_TYPE posType)
{
    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick)) return false;

    request.action = TRADE_ACTION_DEAL;
    request.position = ticket;
    request.symbol = _Symbol;
    request.volume = volume;
    request.type = (posType == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
    request.price = (posType == POSITION_TYPE_BUY) ? tick.bid : tick.ask;
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "BG_PartialTP";
    request.type_filling = GetFillingMode(_Symbol);

    if(!OrderSend(request, result))
    {
        Print("ERROR: ClosePartialPosition failed - ", GetLastError());
        return false;
    }

    return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Normalize volume to valid lot step                               |
//+------------------------------------------------------------------+
double NormalizeVolume(double volume)
{
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    volume = MathMax(volume, minLot);
    volume = MathMin(volume, maxLot);
    volume = MathFloor(volume / lotStep) * lotStep;

    return NormalizeDouble(volume, 2);
}

//+------------------------------------------------------------------+
//| Read signals and execute                                         |
//+------------------------------------------------------------------+
void ReadAndExecuteSignals()
{
    // Build signal file path (use common folder for VPS compatibility)
    string signalFile = "signal_" + AccountName + ".json";

    int fileHandle = FileOpen(signalFile, FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);
    if(fileHandle == INVALID_HANDLE)
    {
        // Try standard files folder as fallback
        fileHandle = FileOpen(signalFile, FILE_READ|FILE_TXT|FILE_ANSI);
        if(fileHandle == INVALID_HANDLE) return;
    }

    string json = "";
    while(!FileIsEnding(fileHandle))
    {
        json += FileReadString(fileHandle);
    }
    FileClose(fileHandle);

    if(json == "") return;

    // Check timestamp to avoid duplicate execution
    string signalTimestamp = ExtractStringValue(json, "timestamp");
    if(signalTimestamp == g_lastSignalTimestamp && signalTimestamp != "")
    {
        return;  // Already processed this signal
    }

    // Check for our symbol
    string currentSymbol = _Symbol;
    if(StringFind(json, "\"" + currentSymbol + "\"") < 0) return;

    // Verify magic number matches
    int jsonMagic = (int)ExtractDoubleValue(json, "magic_number");
    if(jsonMagic != 0 && jsonMagic != MagicNumber)
    {
        Print("WARNING: Signal magic ", jsonMagic, " doesn't match our magic ", MagicNumber);
        return;
    }

    // Parse action and confidence
    string action = ExtractStringValue(json, "action");
    double confidence = ExtractDoubleValue(json, "confidence");
    double signalLotSize = ExtractDoubleValue(json, "max_lot_size");

    // Use smaller of signal lot size and our max
    double lotSize = MathMin(signalLotSize > 0 ? signalLotSize : MaxLotSize, MaxLotSize);

    // Check if blocked by brain
    string status = ExtractStringValue(json, "status");
    if(status == "BLOCKED")
    {
        string reason = ExtractStringValue(json, "block_reason");
        Print("Brain blocked: ", reason);
        return;
    }

    // Update last processed timestamp
    g_lastSignalTimestamp = signalTimestamp;

    if(!TradeEnabled)
    {
        Print("DRY RUN [", AccountName, "] >> ", action, " (", DoubleToString(confidence*100, 1), "%)");
        return;
    }

    // Execute based on action
    if(action == "BUY")
    {
        ExecuteTrade(ORDER_TYPE_BUY, lotSize);
    }
    else if(action == "SELL")
    {
        ExecuteTrade(ORDER_TYPE_SELL, lotSize);
    }
}

//+------------------------------------------------------------------+
//| Execute Trade with full safety checks                            |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE type, double lots)
{
    // Check if position already exists for this magic
    if(PositionExists()) return;

    // Symbol trade mode check
    ENUM_SYMBOL_TRADE_MODE tradeMode = (ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);
    if(tradeMode == SYMBOL_TRADE_MODE_DISABLED)
    {
        Print("ERROR: Trading disabled for ", _Symbol);
        return;
    }
    if(tradeMode == SYMBOL_TRADE_MODE_CLOSEONLY)
    {
        Print("ERROR: Symbol ", _Symbol, " is close-only");
        return;
    }

    // Spread check
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    if(spread > MaxSpreadPoints)
    {
        Print("SKIP: Spread ", spread, " exceeds max ", MaxSpreadPoints);
        return;
    }

    // Refresh prices
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick))
    {
        Print("ERROR: Could not get current tick");
        return;
    }

    double price = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;

    // Calculate SL/TP - ATR-based with HARDCODED multipliers
    // TP is ALWAYS calculated from SL * R:R ratio to maintain consistency
    double atr = CalculateATR(14);
    double sl_dist = atr * SL_ATR_MULT;                    // SL = ATR x 1.5
    double tp_dist = sl_dist * RR_RATIO;                   // TP = SL x 3 (1:3 R:R)

    Print("[SL/TP] ATR: ", DoubleToString(atr, 2),
          " | SL: ", DoubleToString(sl_dist, 2), " (ATR x ", SL_ATR_MULT, ")",
          " | TP: ", DoubleToString(tp_dist, 2), " (1:", RR_RATIO, " R:R)");

    // Get stop level requirement
    int stopLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double minDistance = stopLevel * point;

    // Ensure SL/TP meet minimum requirements
    if(sl_dist < minDistance) sl_dist = minDistance * 1.5;
    if(tp_dist < minDistance) tp_dist = minDistance * 1.5;

    double sl, tp;
    if(type == ORDER_TYPE_BUY)
    {
        sl = price - sl_dist;
        tp = price + tp_dist;
    }
    else
    {
        sl = price + sl_dist;
        tp = price - tp_dist;
    }

    // Validate and normalize lot size
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    lots = MathMax(lots, minLot);
    lots = MathMin(lots, maxLot);
    lots = MathFloor(lots / lotStep) * lotStep;
    lots = NormalizeDouble(lots, 2);

    // Margin check
    double marginRequired;
    if(!OrderCalcMargin(type, _Symbol, lots, price, marginRequired))
    {
        Print("ERROR: Could not calculate margin");
        return;
    }

    double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
    if(marginRequired > freeMargin * 0.8)  // Use max 80% of free margin
    {
        Print("ERROR: Insufficient margin. Required: ", marginRequired, " Free: ", freeMargin);
        return;
    }

    // Get proper filling mode
    ENUM_ORDER_TYPE_FILLING filling = GetFillingMode(_Symbol);

    // Build order request
    MqlTradeRequest request;
    MqlTradeResult result;
    ZeroMemory(request);
    ZeroMemory(result);

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lots;
    request.type = type;
    request.price = price;
    request.sl = 0;  // HIDDEN - managed internally via ManageOpenPositions()
    request.tp = 0;  // HIDDEN - managed internally via ManageOpenPositions()
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "BG_" + AccountName;
    request.type_time = ORDER_TIME_GTC;
    request.type_filling = filling;

    if(!OrderSend(request, result))
    {
        Print("ERROR [", AccountName, "]: OrderSend failed - ", GetLastError());
        return;
    }

    if(result.retcode == TRADE_RETCODE_DONE)
    {
        string typeStr = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
        Print("SUCCESS [", AccountName, "]: ", typeStr, " ", lots, " lots @ ", price);
        Print("  Hidden SL: ", sl, " | Hidden TP: ", tp, " | R:R = 1:", RR_RATIO);
        g_tradesToday++;

        // Store initial values for hidden dynamic management
        g_entryPrice = price;
        g_initialSL = sl;
        g_initialTP = tp;
        g_currentVirtualSL = sl;
        g_currentVirtualTP = tp;
        g_breakevenHit = false;
        g_partialClosed = false;
        g_trailingActive = false;

        Print("  [DYNAMIC] Trailing SL - BE @ ", BREAKEVEN_TRIGGER * 100, "% | Trail @ ", TRAIL_START_MULT * 100, "%");
        Print("  [DYNAMIC] Dynamic TP - Maintaining 1:", RR_RATIO, " R:R");
        Print("  [DYNAMIC] Partial TP - ", PARTIAL_TP_PCT, "% @ ", PARTIAL_TP_TRIGGER * 100, "% of TP");
    }
    else
    {
        Print("ERROR [", AccountName, "]: ", result.comment, " (", result.retcode, ")");
    }
}

//+------------------------------------------------------------------+
//| Get appropriate filling mode for symbol                          |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode(string symbol)
{
    uint filling = (uint)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);

    if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
        return ORDER_FILLING_FOK;

    if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
        return ORDER_FILLING_IOC;

    return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Check if position exists for our magic                          |
//+------------------------------------------------------------------+
bool PositionExists()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionSelectByTicket(PositionGetTicket(i)))
        {
            if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
               PositionGetString(POSITION_SYMBOL) == _Symbol)
            {
                return true;
            }
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| Calculate ATR with proper fallback                               |
//+------------------------------------------------------------------+
double CalculateATR(int period)
{
    MqlRates rates[];
    ArraySetAsSeries(rates, true);

    int copied = CopyRates(_Symbol, PERIOD_M5, 0, period + 1, rates);
    if(copied < period + 1)
    {
        // Fallback: use spread-based estimate
        double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
        return spread * 10;  // Conservative fallback
    }

    double trSum = 0;
    for(int i = 0; i < period; i++)
    {
        double high = rates[i].high;
        double low = rates[i].low;
        double prevClose = rates[i+1].close;

        double tr = MathMax(high - low,
                   MathMax(MathAbs(high - prevClose),
                          MathAbs(low - prevClose)));
        trSum += tr;
    }

    double atr = trSum / period;

    // Sanity check
    if(atr <= 0)
    {
        double spread = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - SymbolInfoDouble(_Symbol, SYMBOL_BID);
        return spread * 10;
    }

    return atr;
}

//+------------------------------------------------------------------+
//| JSON Helpers                                                     |
//+------------------------------------------------------------------+
string ExtractStringValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return "";

    int colonPos = StringFind(json, ":", keyPos);
    if(colonPos < 0) return "";

    int valueStart = StringFind(json, "\"", colonPos);
    if(valueStart < 0) return "";

    int valueEnd = StringFind(json, "\"", valueStart + 1);
    if(valueEnd < 0) return "";

    return StringSubstr(json, valueStart + 1, valueEnd - valueStart - 1);
}

double ExtractDoubleValue(string json, string key)
{
    int keyPos = StringFind(json, "\"" + key + "\"");
    if(keyPos < 0) return 0;

    int colonPos = StringFind(json, ":", keyPos);
    if(colonPos < 0) return 0;

    string remaining = StringSubstr(json, colonPos + 1, 50);
    StringTrimLeft(remaining);

    // Find end of number
    int endPos = 0;
    for(int i = 0; i < StringLen(remaining); i++)
    {
        ushort c = StringGetCharacter(remaining, i);
        if((c >= '0' && c <= '9') || c == '.' || c == '-')
        {
            endPos = i + 1;
        }
        else if(endPos > 0)
        {
            break;
        }
    }

    if(endPos > 0)
    {
        return StringToDouble(StringSubstr(remaining, 0, endPos));
    }

    return 0;
}
//+------------------------------------------------------------------+
