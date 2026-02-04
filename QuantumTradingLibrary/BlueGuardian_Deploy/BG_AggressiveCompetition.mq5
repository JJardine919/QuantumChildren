//+------------------------------------------------------------------+
//|                                    BG_AggressiveCompetition.mq5  |
//|                    COMPETITION ONLY - NOT REAL MONEY             |
//|                    AGGRESSIVE GRID TRADING - NO RISK LIMITS      |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property strict
#property description "AGGRESSIVE Competition Grid Trader"
#property description "NO drawdown limits - MAXIMUM position size"
#property description "For competition accounts ONLY"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input string   AccountName       = "BG_COMPETITION";   // Account Name
input int      MagicNumber       = 366592;             // Magic Number (account ID)

input group "=== AGGRESSIVE TRADING SETTINGS ==="
input double   LotSize           = 0.50;               // Lot Size (AGGRESSIVE)
input int      MaxPositions      = 20;                 // Max Grid Positions (HIGH)
input int      GridSpacingPts    = 200;                // Grid Spacing (points) - TIGHT
input int      TakeProfitPts     = 300;                // Take Profit (points) - QUICK
input bool     BothDirections    = true;               // Trade BOTH directions
input int      CheckSeconds      = 5;                  // Check Interval (FAST)

input group "=== SIGNAL SETTINGS ==="
input int      FastEMA           = 5;                  // Fast EMA (responsive)
input int      SlowEMA           = 13;                 // Slow EMA (responsive)
input double   MinConfidence     = 0.3;                // Min Confidence (LOW - more trades)

input group "=== COMPETITION MODE - NO LIMITS ==="
input bool     IgnoreDrawdown    = true;               // Ignore Drawdown Limits
input bool     ScaleOnWins       = true;               // Increase lots on wins
input double   WinScaleFactor    = 1.2;                // Scale factor on wins

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                   |
//+------------------------------------------------------------------+
int g_handleEmaFast = INVALID_HANDLE;
int g_handleEmaSlow = INVALID_HANDLE;
int g_handleAtr = INVALID_HANDLE;

double g_emaFast[];
double g_emaSlow[];
double g_atr[];

datetime g_lastCheck = 0;
double g_currentLot = 0;
int g_winStreak = 0;

// Position tracking for hidden TP management
struct GridLevel
{
   ulong  ticket;
   double entry;
   double hiddenTP;
   int    direction; // 1=BUY, -1=SELL
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
   Print("  AGGRESSIVE COMPETITION GRID EA v1.0");
   Print("  *** FOR COMPETITION USE ONLY ***");
   Print("================================================");
   Print("  Account: ", AccountName);
   Print("  Magic: ", MagicNumber);
   Print("  Lot Size: ", LotSize, " (AGGRESSIVE)");
   Print("  Max Positions: ", MaxPositions);
   Print("  Grid Spacing: ", GridSpacingPts, " pts (TIGHT)");
   Print("  TP: ", TakeProfitPts, " pts (QUICK)");
   Print("  Both Directions: ", BothDirections ? "YES" : "NO");
   Print("  Ignore DD Limits: ", IgnoreDrawdown ? "YES" : "NO");
   Print("================================================");

   // Create EMA indicators on M1
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

   g_currentLot = LotSize;
   g_lastCheck = 0;

   // Sync existing positions
   SyncGrid();

   Print("Balance: $", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   Print("Synced ", g_gridCount, " existing positions");
   Print(">>> AGGRESSIVE MODE ACTIVE <<<");
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
   Print("Aggressive Competition EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // ALWAYS manage positions on every tick (for hidden TP)
   ManageGrid();

   // FAST check interval for competition
   if(TimeCurrent() - g_lastCheck < CheckSeconds) return;
   g_lastCheck = TimeCurrent();

   // NO risk checks in competition mode
   // Just trade aggressively

   // Check for entry signals
   CheckEntry();
}

//+------------------------------------------------------------------+
//| Check for entry signal                                             |
//+------------------------------------------------------------------+
void CheckEntry()
{
   // Count positions by direction
   int buyCount = 0, sellCount = 0;
   CountPositionsByDirection(buyCount, sellCount);

   int totalPos = buyCount + sellCount;
   if(totalPos >= MaxPositions) return;

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
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // Signal detection
   bool buySignal = (emaF > emaS);
   bool sellSignal = (emaF < emaS);

   // Calculate confidence based on EMA separation
   double separation = MathAbs(emaF - emaS);
   double confidence = MathMin(1.0, (separation / atr) * 0.5);

   // Lower threshold for more trades in competition
   if(confidence < MinConfidence) return;

   // Grid spacing check for BUY positions
   if(buySignal && buyCount < MaxPositions / 2)
   {
      double lastBuyEntry = GetLastEntryPrice(1);
      double spacing = GridSpacingPts * point;

      bool shouldOpenBuy = (buyCount == 0) || (tick.ask < lastBuyEntry - spacing);

      if(shouldOpenBuy)
      {
         OpenPosition(ORDER_TYPE_BUY, buyCount + 1);
      }
   }

   // Grid spacing check for SELL positions (if bi-directional)
   if(BothDirections && sellSignal && sellCount < MaxPositions / 2)
   {
      double lastSellEntry = GetLastEntryPrice(-1);
      double spacing = GridSpacingPts * point;

      bool shouldOpenSell = (sellCount == 0) || (tick.bid > lastSellEntry + spacing);

      if(shouldOpenSell)
      {
         OpenPosition(ORDER_TYPE_SELL, sellCount + 1);
      }
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

   // Calculate lot (scale on wins if enabled)
   double lot = g_currentLot;
   lot = NormalizeLot(lot);

   // Calculate hidden TP
   double hiddenTP;
   if(type == ORDER_TYPE_BUY)
   {
      hiddenTP = price + (TakeProfitPts * point);
   }
   else
   {
      hiddenTP = price - (TakeProfitPts * point);
   }

   // Build request - NO SL, only TP (hidden)
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = type;
   request.price = price;
   request.deviation = 50;  // Wide slippage for competition
   request.magic = MagicNumber;
   request.comment = StringFormat("COMP_L%d", level);
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = GetFilling();
   request.sl = 0;  // NO stop loss in competition
   request.tp = 0;  // Hidden TP

   if(!OrderSend(request, result))
   {
      Print("ERROR: Order failed - ", GetLastError(), " - ", result.comment);
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
      g_grid[idx].direction = (type == ORDER_TYPE_BUY) ? 1 : -1;
      g_grid[idx].level = level;
      g_gridCount = idx + 1;

      string typeStr = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
      Print(">>> ", typeStr, " L", level, " @ ", DoubleToString(price, 2),
            " | Lot: ", DoubleToString(lot, 2),
            " | TP: ", DoubleToString(hiddenTP, 2));
   }
   else
   {
      Print("Order rejected: ", result.comment, " (", result.retcode, ")");
   }
}

//+------------------------------------------------------------------+
//| Manage grid - hidden TP                                            |
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
         // Position closed - check if it was profit
         g_winStreak++;
         if(ScaleOnWins && g_winStreak >= 2)
         {
            g_currentLot = LotSize * MathPow(WinScaleFactor, MathMin(g_winStreak - 1, 5));
            Print("Win streak! Lot scaled to: ", DoubleToString(g_currentLot, 2));
         }
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
         Print(">>> TP HIT - Closing L", g_grid[i].level, " @ ", DoubleToString(currentPrice, 2));
         ClosePosition(ticket);

         // Scale up on wins
         g_winStreak++;
         if(ScaleOnWins && g_winStreak >= 2)
         {
            g_currentLot = LotSize * MathPow(WinScaleFactor, MathMin(g_winStreak - 1, 5));
            Print("Win streak! Lot scaled to: ", DoubleToString(g_currentLot, 2));
         }

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
   request.deviation = 50;
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
//| Count positions by direction                                       |
//+------------------------------------------------------------------+
void CountPositionsByDirection(int &buyCount, int &sellCount)
{
   buyCount = 0;
   sellCount = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY) buyCount++;
      else sellCount++;
   }
}

//+------------------------------------------------------------------+
//| Get last entry price by direction                                  |
//+------------------------------------------------------------------+
double GetLastEntryPrice(int direction)
{
   double lastPrice = 0;
   datetime lastTime = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      int posDir = (posType == POSITION_TYPE_BUY) ? 1 : -1;

      if(posDir != direction) continue;

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
         g_grid[idx].direction = (posType == POSITION_TYPE_BUY) ? 1 : -1;
         g_grid[idx].level = idx + 1;

         if(posType == POSITION_TYPE_BUY)
         {
            g_grid[idx].hiddenTP = entry + (TakeProfitPts * point);
         }
         else
         {
            g_grid[idx].hiddenTP = entry - (TakeProfitPts * point);
         }

         g_gridCount = idx + 1;
         Print("Synced position: ", ticket);
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
