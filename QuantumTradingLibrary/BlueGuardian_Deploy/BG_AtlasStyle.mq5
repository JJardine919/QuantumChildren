//+------------------------------------------------------------------+
//|                                              BG_AtlasStyle.mq5   |
//|                           Blue Guardian - Atlas Trading Style    |
//|                           Matching Atlas Funded EA Configuration |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property description "Blue Guardian EA matching Atlas Funded trading style"
#property description "BTCUSD M1 | Scaling BUY Positions | Hidden SL/TP"
#property description "Lot: 0.06-0.07 | TP: ~450 points | Grid Style"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS - ATLAS STYLE CONFIGURATION                      |
//+------------------------------------------------------------------+
input group "=== Account Settings ==="
input string   InpAccountName    = "BG_100K";           // Account Name
input int      InpMagicNumber    = 365001;              // Magic Number (matches Atlas 212001 style)

input group "=== Trading Configuration - ATLAS STYLE ==="
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M1;         // Timeframe (M1 like Atlas)
input string   InpSymbol          = "BTCUSD";           // Symbol
input double   InpBaseLot         = 0.06;               // Base Lot Size
input double   InpMaxLot          = 0.07;               // Max Lot Size
input int      InpTPPoints        = 450;                // Take Profit (points) ~400-500 like Atlas
input bool     InpHiddenSLTP      = true;               // Hidden SL/TP (no visible SL like Atlas)

input group "=== Grid Configuration ==="
input int      InpMaxPositions    = 5;                  // Max Scaling Positions
input int      InpGridSpacing     = 100;                // Grid Spacing (points between entries)
input bool     InpOnlyBuy         = true;               // Only BUY (Atlas style bullish bias)

input group "=== Risk Management ==="
input double   InpDailyDDLimit    = 4.5;                // Daily Drawdown Limit %
input double   InpMaxDDLimit      = 9.0;                // Max Drawdown Limit %
input bool     InpUseTrailing     = true;               // Use Trailing Stop (internal)
input double   InpTrailStartPct   = 0.5;                // Trail Start (% of TP reached)
input double   InpTrailDistance   = 50.0;               // Trail Distance (points)
input bool     InpBreakevenMove   = true;               // Move to Breakeven
input double   InpBreakevenTrigger= 0.3;                // Breakeven Trigger (% of TP)

input group "=== Signal Filters ==="
input int      InpEmaFast         = 8;                  // Fast EMA
input int      InpEmaSlow         = 21;                 // Slow EMA
input double   InpMinConfidence   = 0.75;               // Min Confidence to Trade
input int      InpCheckInterval   = 30;                 // Signal Check Interval (seconds)

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
// Indicator handles
int g_handleEmaFast;
int g_handleEmaSlow;
int g_handleAtr;

// Buffers
double g_emaFastBuffer[];
double g_emaSlowBuffer[];
double g_atrBuffer[];

// Position tracking (for hidden SL/TP management)
struct GridPosition
{
   ulong  ticket;
   double entryPrice;
   double hiddenSL;
   double hiddenTP;
   int    level;
   bool   breakevenHit;
   bool   trailingActive;
};

GridPosition g_positions[];
int g_positionCount = 0;

// Account tracking
double g_startBalance;
double g_highWaterMark;
double g_dailyStartBalance;
datetime g_lastDayReset;
datetime g_lastCheck;

// State
bool g_blocked = false;
string g_blockReason = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("BLUE GUARDIAN - ATLAS STYLE EA");
   Print("========================================");
   Print("Account: ", InpAccountName);
   Print("Magic: ", InpMagicNumber);
   Print("Symbol: ", InpSymbol, " | Timeframe: M1");
   Print("Lot Size: ", InpBaseLot, " - ", InpMaxLot);
   Print("TP: ", InpTPPoints, " points");
   Print("Max Positions: ", InpMaxPositions);
   Print("Grid Spacing: ", InpGridSpacing, " points");
   Print("Hidden SL/TP: ", InpHiddenSLTP);
   Print("========================================");

   // Create indicators on M1 timeframe
   g_handleEmaFast = iMA(InpSymbol, PERIOD_M1, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   g_handleEmaSlow = iMA(InpSymbol, PERIOD_M1, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   g_handleAtr = iATR(InpSymbol, PERIOD_M1, 14);

   if(g_handleEmaFast == INVALID_HANDLE || g_handleEmaSlow == INVALID_HANDLE || g_handleAtr == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   ArraySetAsSeries(g_emaFastBuffer, true);
   ArraySetAsSeries(g_emaSlowBuffer, true);
   ArraySetAsSeries(g_atrBuffer, true);

   // Initialize balance tracking
   g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_highWaterMark = g_startBalance;
   g_dailyStartBalance = g_startBalance;
   g_lastDayReset = TimeCurrent();
   g_lastCheck = 0;

   // Sync existing positions
   SyncPositions();

   Print("Account Balance: $", DoubleToString(g_startBalance, 2));
   Print("Existing positions synced: ", g_positionCount);
   Print("Initialization complete. Ready to trade.");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_handleEmaFast != INVALID_HANDLE) IndicatorRelease(g_handleEmaFast);
   if(g_handleEmaSlow != INVALID_HANDLE) IndicatorRelease(g_handleEmaSlow);
   if(g_handleAtr != INVALID_HANDLE) IndicatorRelease(g_handleAtr);

   Print("BG AtlasStyle EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // ALWAYS manage positions on every tick (critical for hidden SL/TP)
   ManagePositions();

   // Check interval for new signals
   if(TimeCurrent() - g_lastCheck < InpCheckInterval) return;
   g_lastCheck = TimeCurrent();

   // Daily reset
   CheckDailyReset();

   // Update high water mark
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(currentBalance > g_highWaterMark) g_highWaterMark = currentBalance;

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

   // Check for new entry signals
   CheckEntrySignal();
}

//+------------------------------------------------------------------+
//| Daily reset check                                                  |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
   MqlDateTime now, last;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(g_lastDayReset, last);

   if(now.day != last.day || now.mon != last.mon || now.year != last.year)
   {
      g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_lastDayReset = TimeCurrent();
      g_blocked = false;
      g_blockReason = "";
      Print("Daily reset. New baseline: $", DoubleToString(g_dailyStartBalance, 2));
   }
}

//+------------------------------------------------------------------+
//| Check drawdown limits                                              |
//+------------------------------------------------------------------+
bool CheckDrawdownLimits()
{
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(currentBalance, equity);

   if(g_dailyStartBalance <= 0 || g_highWaterMark <= 0) return true;

   // Daily DD
   double dailyDD = ((g_dailyStartBalance - current) / g_dailyStartBalance) * 100;
   if(dailyDD >= InpDailyDDLimit)
   {
      g_blockReason = StringFormat("Daily DD %.2f%% >= limit %.2f%%", dailyDD, InpDailyDDLimit);
      return false;
   }

   // Max DD
   double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
   if(maxDD >= InpMaxDDLimit)
   {
      g_blockReason = StringFormat("Max DD %.2f%% >= limit %.2f%%", maxDD, InpMaxDDLimit);
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Check for entry signal - ATLAS STYLE                              |
//+------------------------------------------------------------------+
void CheckEntrySignal()
{
   // Count current positions
   int posCount = CountOurPositions();
   if(posCount >= InpMaxPositions) return;

   // Get indicator values
   if(CopyBuffer(g_handleEmaFast, 0, 0, 5, g_emaFastBuffer) < 5) return;
   if(CopyBuffer(g_handleEmaSlow, 0, 0, 5, g_emaSlowBuffer) < 5) return;
   if(CopyBuffer(g_handleAtr, 0, 0, 5, g_atrBuffer) < 5) return;

   double emaFast = g_emaFastBuffer[1];
   double emaSlow = g_emaSlowBuffer[1];
   double atr = g_atrBuffer[1];

   // Get current price
   MqlTick tick;
   if(!SymbolInfoTick(InpSymbol, tick)) return;
   double price = tick.ask;

   // ATLAS STYLE: Primarily bullish bias
   // Signal: Fast EMA above Slow EMA for BUY
   bool buySignal = (emaFast > emaSlow);

   // Calculate confidence based on EMA separation
   double separation = MathAbs(emaFast - emaSlow) / atr;
   double confidence = MathMin(1.0, separation / 2.0);

   if(confidence < InpMinConfidence) return;

   // Check grid spacing from last entry
   if(posCount > 0)
   {
      double lastEntry = GetLastEntryPrice();
      double spacing = SymbolInfoDouble(InpSymbol, SYMBOL_POINT) * InpGridSpacing;

      // For BUY grid: only add if price is BELOW last entry (scaling in on dips)
      if(price > lastEntry - spacing) return;
   }

   // ATLAS STYLE: Only BUY in bullish mode
   if(InpOnlyBuy && buySignal)
   {
      OpenGridPosition(ORDER_TYPE_BUY, posCount + 1);
   }
   else if(!InpOnlyBuy)
   {
      // Full bi-directional mode (not Atlas style, but available)
      if(buySignal) OpenGridPosition(ORDER_TYPE_BUY, posCount + 1);
   }
}

//+------------------------------------------------------------------+
//| Open grid position - ATLAS STYLE (Hidden SL/TP)                   |
//+------------------------------------------------------------------+
void OpenGridPosition(ENUM_ORDER_TYPE type, int level)
{
   MqlTick tick;
   if(!SymbolInfoTick(InpSymbol, tick)) return;

   double price = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;
   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);

   // Calculate lot size (randomize slightly between base and max like Atlas)
   double lot = InpBaseLot + (MathRand() % 2) * 0.01;  // 0.06 or 0.07
   lot = MathMin(lot, InpMaxLot);
   lot = NormalizeLot(lot);

   // Calculate hidden TP (ATLAS STYLE: ~450 points above entry)
   double hiddenTP, hiddenSL;
   if(type == ORDER_TYPE_BUY)
   {
      hiddenTP = price + (InpTPPoints * point);
      hiddenSL = price - (InpTPPoints * 0.5 * point);  // Hidden SL at 50% of TP distance
   }
   else
   {
      hiddenTP = price - (InpTPPoints * point);
      hiddenSL = price + (InpTPPoints * 0.5 * point);
   }

   // Build order request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = InpSymbol;
   request.volume = lot;
   request.type = type;
   request.price = price;
   request.deviation = 30;
   request.magic = InpMagicNumber;
   request.comment = StringFormat("BUY L%d placed by expert", level);  // Matches Atlas comment style
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = GetFillingMode();

   // ATLAS STYLE: No visible SL/TP sent to broker
   if(InpHiddenSLTP)
   {
      request.sl = 0;
      request.tp = 0;
   }
   else
   {
      request.sl = hiddenSL;
      request.tp = hiddenTP;
   }

   if(!OrderSend(request, result))
   {
      Print("ERROR: Order failed - ", GetLastError());
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE)
   {
      // Track position for hidden SL/TP management
      int idx = ArraySize(g_positions);
      ArrayResize(g_positions, idx + 1);

      g_positions[idx].ticket = result.order;
      g_positions[idx].entryPrice = price;
      g_positions[idx].hiddenSL = hiddenSL;
      g_positions[idx].hiddenTP = hiddenTP;
      g_positions[idx].level = level;
      g_positions[idx].breakevenHit = false;
      g_positions[idx].trailingActive = false;
      g_positionCount = idx + 1;

      string typeStr = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
      Print(typeStr, " L", level, " placed by expert | Price: ", DoubleToString(price, 2),
            " | Lot: ", DoubleToString(lot, 2),
            " | Hidden TP: ", DoubleToString(hiddenTP, 2));
   }
   else
   {
      Print("Order rejected: ", result.comment, " (", result.retcode, ")");
   }
}

//+------------------------------------------------------------------+
//| Manage positions - Hidden SL/TP, Trailing, Breakeven              |
//+------------------------------------------------------------------+
void ManagePositions()
{
   // Sync first
   SyncPositions();

   MqlTick tick;
   if(!SymbolInfoTick(InpSymbol, tick)) return;

   for(int i = g_positionCount - 1; i >= 0; i--)
   {
      ulong ticket = g_positions[i].ticket;

      if(!PositionSelectByTicket(ticket))
      {
         // Position closed externally, remove from tracking
         RemovePosition(i);
         continue;
      }

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double entryPrice = g_positions[i].entryPrice;
      double hiddenTP = g_positions[i].hiddenTP;
      double hiddenSL = g_positions[i].hiddenSL;
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      // Calculate profit distance
      double profitDist = (posType == POSITION_TYPE_BUY) ?
                          (currentPrice - entryPrice) :
                          (entryPrice - currentPrice);

      double tpDist = MathAbs(hiddenTP - entryPrice);
      double progress = profitDist / tpDist;

      // === 1. HIDDEN TP HIT ===
      bool hitTP = false;
      if(posType == POSITION_TYPE_BUY && currentPrice >= hiddenTP) hitTP = true;
      if(posType == POSITION_TYPE_SELL && currentPrice <= hiddenTP) hitTP = true;

      if(hitTP)
      {
         Print("HIDDEN TP HIT - Closing position L", g_positions[i].level);
         ClosePosition(ticket);
         RemovePosition(i);
         continue;
      }

      // === 2. HIDDEN SL HIT ===
      bool hitSL = false;
      if(posType == POSITION_TYPE_BUY && currentPrice <= hiddenSL) hitSL = true;
      if(posType == POSITION_TYPE_SELL && currentPrice >= hiddenSL) hitSL = true;

      if(hitSL)
      {
         Print("HIDDEN SL HIT - Closing position L", g_positions[i].level);
         ClosePosition(ticket);
         RemovePosition(i);
         continue;
      }

      // === 3. BREAKEVEN ===
      if(InpBreakevenMove && !g_positions[i].breakevenHit && progress >= InpBreakevenTrigger)
      {
         double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
         if(posType == POSITION_TYPE_BUY)
         {
            g_positions[i].hiddenSL = entryPrice + (10 * point);  // Small buffer above entry
         }
         else
         {
            g_positions[i].hiddenSL = entryPrice - (10 * point);
         }
         g_positions[i].breakevenHit = true;
         Print("BREAKEVEN - L", g_positions[i].level, " SL moved to ", DoubleToString(g_positions[i].hiddenSL, 2));
      }

      // === 4. TRAILING STOP ===
      if(InpUseTrailing && g_positions[i].breakevenHit && progress >= InpTrailStartPct)
      {
         g_positions[i].trailingActive = true;
         double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);

         if(posType == POSITION_TYPE_BUY)
         {
            double newSL = currentPrice - (InpTrailDistance * point);
            if(newSL > g_positions[i].hiddenSL)
            {
               g_positions[i].hiddenSL = newSL;
               Print("TRAILING - L", g_positions[i].level, " SL trailed to ", DoubleToString(newSL, 2));
            }
         }
         else
         {
            double newSL = currentPrice + (InpTrailDistance * point);
            if(newSL < g_positions[i].hiddenSL)
            {
               g_positions[i].hiddenSL = newSL;
               Print("TRAILING - L", g_positions[i].level, " SL trailed to ", DoubleToString(newSL, 2));
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close position                                                     |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket)
{
   if(!PositionSelectByTicket(ticket)) return false;

   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);
   string symbol = PositionGetString(POSITION_SYMBOL);

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return false;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = symbol;
   request.volume = volume;
   request.deviation = 30;
   request.magic = InpMagicNumber;
   request.type_filling = GetFillingMode();

   if(posType == POSITION_TYPE_BUY)
   {
      request.type = ORDER_TYPE_SELL;
      request.price = tick.bid;
   }
   else
   {
      request.type = ORDER_TYPE_BUY;
      request.price = tick.ask;
   }

   if(!OrderSend(request, result))
   {
      Print("ERROR: Close failed - ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Sync positions with actual MT5 positions                          |
//+------------------------------------------------------------------+
void SyncPositions()
{
   // Remove positions that no longer exist
   for(int i = g_positionCount - 1; i >= 0; i--)
   {
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         ulong ticket = PositionGetTicket(p);
         if(ticket == g_positions[i].ticket)
         {
            found = true;
            break;
         }
      }
      if(!found) RemovePosition(i);
   }

   // Add positions we're not tracking
   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      ulong ticket = PositionGetTicket(p);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != InpSymbol) continue;

      // Check if already tracked
      bool tracked = false;
      for(int i = 0; i < g_positionCount; i++)
      {
         if(g_positions[i].ticket == ticket)
         {
            tracked = true;
            break;
         }
      }

      if(!tracked)
      {
         // Add to tracking with estimated values
         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         int idx = ArraySize(g_positions);
         ArrayResize(g_positions, idx + 1);

         g_positions[idx].ticket = ticket;
         g_positions[idx].entryPrice = entry;
         g_positions[idx].level = idx + 1;
         g_positions[idx].breakevenHit = false;
         g_positions[idx].trailingActive = false;

         if(posType == POSITION_TYPE_BUY)
         {
            g_positions[idx].hiddenTP = entry + (InpTPPoints * point);
            g_positions[idx].hiddenSL = entry - (InpTPPoints * 0.5 * point);
         }
         else
         {
            g_positions[idx].hiddenTP = entry - (InpTPPoints * point);
            g_positions[idx].hiddenSL = entry + (InpTPPoints * 0.5 * point);
         }

         g_positionCount = idx + 1;
         Print("Synced existing position: ", ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Remove position from tracking                                      |
//+------------------------------------------------------------------+
void RemovePosition(int index)
{
   if(index < 0 || index >= g_positionCount) return;

   for(int i = index; i < g_positionCount - 1; i++)
   {
      g_positions[i] = g_positions[i + 1];
   }

   g_positionCount--;
   ArrayResize(g_positions, g_positionCount);
}

//+------------------------------------------------------------------+
//| Count our positions                                                |
//+------------------------------------------------------------------+
int CountOurPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber &&
         PositionGetString(POSITION_SYMBOL) == InpSymbol)
      {
         count++;
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Get last entry price                                               |
//+------------------------------------------------------------------+
double GetLastEntryPrice()
{
   double lastPrice = 0;
   datetime lastTime = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != InpSymbol) continue;

      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      if(openTime > lastTime)
      {
         lastTime = openTime;
         lastPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      }
   }

   return lastPrice;
}

//+------------------------------------------------------------------+
//| Normalize lot size                                                 |
//+------------------------------------------------------------------+
double NormalizeLot(double lot)
{
   double minLot = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(InpSymbol, SYMBOL_VOLUME_STEP);

   lot = MathMax(lot, minLot);
   lot = MathMin(lot, maxLot);
   lot = MathFloor(lot / lotStep) * lotStep;

   return NormalizeDouble(lot, 2);
}

//+------------------------------------------------------------------+
//| Get filling mode                                                   |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
   uint filling = (uint)SymbolInfoInteger(InpSymbol, SYMBOL_FILLING_MODE);

   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;

   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;

   return ORDER_FILLING_RETURN;
}
//+------------------------------------------------------------------+
