//+------------------------------------------------------------------+
//|                                               BG_AtlasGrid.mq5   |
//|                            Blue Guardian - Atlas Grid Trading    |
//|                          BTCUSD M1 | BUY ONLY | Grid Scaling     |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property strict

//+------------------------------------------------------------------+
//| INPUT PARAMETERS - CONFIGURE PER ACCOUNT                          |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input string   AccountName       = "BG_100K";        // Account Name
input int      MagicNumber       = 365001;           // Magic Number

input group "=== TRADING SETTINGS ==="
input double   BaseLot           = 0.06;             // Base Lot Size
input double   MaxLot            = 0.07;             // Max Lot Size (varies 0.06-0.07)
input int      MaxPositions      = 5;                // Max Grid Positions
input int      GridSpacingPts    = 500;              // Grid Spacing (points)
input int      TakeProfitPts     = 450;              // Take Profit (points)
input bool     OnlyBuy           = true;             // Only BUY (bullish bias)
input int      CheckSeconds      = 30;               // Check Interval (seconds)

input group "=== SIGNAL SETTINGS ==="
input int      FastEMA           = 8;                // Fast EMA Period
input int      SlowEMA           = 21;               // Slow EMA Period
input double   MinConfidence     = 0.6;              // Min Signal Confidence (0-1)

input group "=== RISK MANAGEMENT ==="
input double   DailyDDLimit      = 4.5;              // Daily Drawdown Limit %
input double   MaxDDLimit        = 9.0;              // Max Drawdown Limit %
input bool     UseHiddenSLTP     = true;             // Hidden SL/TP (manage internally)
input double   HiddenSLMultiple  = 2.0;              // Hidden SL = TP * this

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                   |
//+------------------------------------------------------------------+
int g_handleEmaFast = INVALID_HANDLE;
int g_handleEmaSlow = INVALID_HANDLE;
int g_handleAtr = INVALID_HANDLE;

double g_emaFast[];
double g_emaSlow[];
double g_atr[];

double g_startBalance = 0;
double g_highWaterMark = 0;
double g_dailyStartBalance = 0;
datetime g_lastDayReset = 0;
datetime g_lastCheck = 0;

bool g_blocked = false;
string g_blockReason = "";

// Position tracking for hidden TP management
struct GridLevel
{
   ulong  ticket;
   double entry;
   double hiddenTP;
   double hiddenSL;
   int    level;
};
GridLevel g_grid[];
int g_gridCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("================================================");
   Print("  BLUE GUARDIAN ATLAS GRID EA v1.0");
   Print("  Account: ", AccountName);
   Print("  Magic: ", MagicNumber);
   Print("  Lot: ", BaseLot, " - ", MaxLot);
   Print("  Max Positions: ", MaxPositions);
   Print("  Grid Spacing: ", GridSpacingPts, " pts");
   Print("  TP: ", TakeProfitPts, " pts");
   Print("  Only BUY: ", OnlyBuy ? "YES" : "NO");
   Print("  Hidden SL/TP: ", UseHiddenSLTP ? "YES" : "NO");
   Print("================================================");

   // Create EMA indicators
   g_handleEmaFast = iMA(_Symbol, PERIOD_M1, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_handleEmaSlow = iMA(_Symbol, PERIOD_M1, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_handleAtr = iATR(_Symbol, PERIOD_M1, 14);

   if(g_handleEmaFast == INVALID_HANDLE || g_handleEmaSlow == INVALID_HANDLE || g_handleAtr == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicators");
      return INIT_FAILED;
   }

   ArraySetAsSeries(g_emaFast, true);
   ArraySetAsSeries(g_emaSlow, true);
   ArraySetAsSeries(g_atr, true);

   // Initialize balance tracking
   g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_highWaterMark = g_startBalance;
   g_dailyStartBalance = g_startBalance;
   g_lastDayReset = TimeCurrent();
   g_lastCheck = 0;

   // Sync existing positions
   SyncGrid();

   Print("Balance: $", DoubleToString(g_startBalance, 2));
   Print("Synced ", g_gridCount, " existing positions");
   Print("Initialization complete. READY TO TRADE.");
   Print("================================================");

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
   Print("BG Atlas Grid EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // ALWAYS manage positions on every tick (for hidden TP)
   ManageGrid();

   // Check interval for new entries
   if(TimeCurrent() - g_lastCheck < CheckSeconds) return;
   g_lastCheck = TimeCurrent();

   // Daily reset
   CheckDailyReset();

   // Update high water mark
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   if(bal > g_highWaterMark) g_highWaterMark = bal;

   // Risk check
   if(!CheckRisk())
   {
      if(!g_blocked)
      {
         g_blocked = true;
         Print("BLOCKED: ", g_blockReason);
      }
      return;
   }

   // Check for entry signals
   CheckEntry();
}

//+------------------------------------------------------------------+
//| Daily reset                                                        |
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
      Print("Daily reset. Baseline: $", DoubleToString(g_dailyStartBalance, 2));
   }
}

//+------------------------------------------------------------------+
//| Check risk limits                                                  |
//+------------------------------------------------------------------+
bool CheckRisk()
{
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(bal, eq);

   if(g_dailyStartBalance <= 0 || g_highWaterMark <= 0) return true;

   // Daily DD
   double dailyDD = ((g_dailyStartBalance - current) / g_dailyStartBalance) * 100;
   if(dailyDD >= DailyDDLimit)
   {
      g_blockReason = StringFormat("Daily DD %.2f%% >= %.2f%%", dailyDD, DailyDDLimit);
      return false;
   }

   // Max DD
   double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
   if(maxDD >= MaxDDLimit)
   {
      g_blockReason = StringFormat("Max DD %.2f%% >= %.2f%%", maxDD, MaxDDLimit);
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Check for entry signal                                             |
//+------------------------------------------------------------------+
void CheckEntry()
{
   // Count positions
   int posCount = CountPositions();
   if(posCount >= MaxPositions)
   {
      return;
   }

   // Get indicator values
   if(CopyBuffer(g_handleEmaFast, 0, 0, 3, g_emaFast) < 3) return;
   if(CopyBuffer(g_handleEmaSlow, 0, 0, 3, g_emaSlow) < 3) return;
   if(CopyBuffer(g_handleAtr, 0, 0, 3, g_atr) < 3) return;

   double emaF = g_emaFast[1];
   double emaS = g_emaSlow[1];
   double atr = g_atr[1];

   // Current tick
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   double price = tick.ask;
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // BUY signal: Fast EMA > Slow EMA
   bool buySignal = (emaF > emaS);

   // Calculate confidence based on EMA separation
   double separation = MathAbs(emaF - emaS);
   double confidence = MathMin(1.0, (separation / atr) * 0.5);

   if(confidence < MinConfidence)
   {
      return;
   }

   // Grid spacing check
   if(posCount > 0)
   {
      double lastEntry = GetLastEntryPrice();
      double spacing = GridSpacingPts * point;

      // For BUY grid: add positions when price dips below last entry
      if(price > lastEntry - spacing)
      {
         return;
      }
   }

   // Execute BUY
   if(OnlyBuy && buySignal)
   {
      OpenPosition(ORDER_TYPE_BUY, posCount + 1);
   }
   else if(!OnlyBuy && !buySignal)
   {
      OpenPosition(ORDER_TYPE_SELL, posCount + 1);
   }
}

//+------------------------------------------------------------------+
//| Open grid position                                                 |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE type, int level)
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double price = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // Lot size: randomly between BaseLot and MaxLot
   double lot = BaseLot + (MathRand() % 2) * 0.01;
   lot = MathMin(lot, MaxLot);
   lot = NormalizeLot(lot);

   // Calculate hidden TP/SL
   double hiddenTP, hiddenSL;
   if(type == ORDER_TYPE_BUY)
   {
      hiddenTP = price + (TakeProfitPts * point);
      hiddenSL = price - (TakeProfitPts * HiddenSLMultiple * point);
   }
   else
   {
      hiddenTP = price - (TakeProfitPts * point);
      hiddenSL = price + (TakeProfitPts * HiddenSLMultiple * point);
   }

   // Build request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = type;
   request.price = price;
   request.deviation = 30;
   request.magic = MagicNumber;
   request.comment = StringFormat("BG_L%d", level);
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = GetFilling();

   // Hidden SL/TP: don't send to broker
   if(UseHiddenSLTP)
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
      // Track position
      int idx = ArraySize(g_grid);
      ArrayResize(g_grid, idx + 1);

      g_grid[idx].ticket = result.order;
      g_grid[idx].entry = price;
      g_grid[idx].hiddenTP = hiddenTP;
      g_grid[idx].hiddenSL = hiddenSL;
      g_grid[idx].level = level;
      g_gridCount = idx + 1;

      string typeStr = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
      Print(typeStr, " L", level, " placed by expert | Price: ", DoubleToString(price, 2),
            " | Lot: ", DoubleToString(lot, 2),
            " | TP: ", DoubleToString(hiddenTP, 2));
   }
   else
   {
      Print("Order rejected: ", result.comment, " (", result.retcode, ")");
   }
}

//+------------------------------------------------------------------+
//| Manage grid - hidden TP/SL                                         |
//+------------------------------------------------------------------+
void ManageGrid()
{
   SyncGrid();

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   for(int i = g_gridCount - 1; i >= 0; i--)
   {
      ulong ticket = g_grid[i].ticket;

      if(!PositionSelectByTicket(ticket))
      {
         RemoveFromGrid(i);
         continue;
      }

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      // Check hidden TP hit
      bool hitTP = false;
      if(posType == POSITION_TYPE_BUY && currentPrice >= g_grid[i].hiddenTP) hitTP = true;
      if(posType == POSITION_TYPE_SELL && currentPrice <= g_grid[i].hiddenTP) hitTP = true;

      if(hitTP)
      {
         Print("HIDDEN TP HIT - Closing L", g_grid[i].level, " @ ", DoubleToString(currentPrice, 2));
         ClosePosition(ticket);
         RemoveFromGrid(i);
         continue;
      }

      // Check hidden SL hit
      bool hitSL = false;
      if(posType == POSITION_TYPE_BUY && currentPrice <= g_grid[i].hiddenSL) hitSL = true;
      if(posType == POSITION_TYPE_SELL && currentPrice >= g_grid[i].hiddenSL) hitSL = true;

      if(hitSL)
      {
         Print("HIDDEN SL HIT - Closing L", g_grid[i].level, " @ ", DoubleToString(currentPrice, 2));
         ClosePosition(ticket);
         RemoveFromGrid(i);
         continue;
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
   request.magic = MagicNumber;
   request.type_filling = GetFilling();

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
//| Sync grid with actual positions                                    |
//+------------------------------------------------------------------+
void SyncGrid()
{
   // Remove closed positions
   for(int i = g_gridCount - 1; i >= 0; i--)
   {
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         ulong t = PositionGetTicket(p);
         if(t == g_grid[i].ticket)
         {
            found = true;
            break;
         }
      }
      if(!found) RemoveFromGrid(i);
   }

   // Add untracked positions
   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      ulong ticket = PositionGetTicket(p);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      bool tracked = false;
      for(int i = 0; i < g_gridCount; i++)
      {
         if(g_grid[i].ticket == ticket)
         {
            tracked = true;
            break;
         }
      }

      if(!tracked)
      {
         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         int idx = ArraySize(g_grid);
         ArrayResize(g_grid, idx + 1);

         g_grid[idx].ticket = ticket;
         g_grid[idx].entry = entry;
         g_grid[idx].level = idx + 1;

         if(posType == POSITION_TYPE_BUY)
         {
            g_grid[idx].hiddenTP = entry + (TakeProfitPts * point);
            g_grid[idx].hiddenSL = entry - (TakeProfitPts * HiddenSLMultiple * point);
         }
         else
         {
            g_grid[idx].hiddenTP = entry - (TakeProfitPts * point);
            g_grid[idx].hiddenSL = entry + (TakeProfitPts * HiddenSLMultiple * point);
         }

         g_gridCount = idx + 1;
         Print("Synced position: ", ticket, " entry: ", DoubleToString(entry, 2));
      }
   }
}

//+------------------------------------------------------------------+
//| Remove position from grid tracking                                 |
//+------------------------------------------------------------------+
void RemoveFromGrid(int index)
{
   if(index < 0 || index >= g_gridCount) return;

   for(int i = index; i < g_gridCount - 1; i++)
   {
      g_grid[i] = g_grid[i + 1];
   }

   g_gridCount--;
   ArrayResize(g_grid, g_gridCount);
}

//+------------------------------------------------------------------+
//| Count our positions                                                |
//+------------------------------------------------------------------+
int CountPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
         PositionGetString(POSITION_SYMBOL) == _Symbol)
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
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

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
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lot = MathMax(lot, minLot);
   lot = MathMin(lot, maxLot);
   lot = MathFloor(lot / lotStep) * lotStep;

   return NormalizeDouble(lot, 2);
}

//+------------------------------------------------------------------+
//| Get filling mode                                                   |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFilling()
{
   uint filling = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);

   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;

   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;

   return ORDER_FILLING_RETURN;
}
//+------------------------------------------------------------------+
