//+------------------------------------------------------------------+
//|                                    BG_AggressiveCompetition.mq5  |
//|                    COMPETITION ONLY - NOT REAL MONEY             |
//|                    AGGRESSIVE GRID TRADING                       |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "2.00"
#property strict
#property description "AGGRESSIVE Competition Grid Trader"
#property description "Dynamic ATR SL/TP + Rolling SL + 50% Dynamic TP"
#property description "For competition accounts ONLY"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input string   AccountName       = "BG_COMPETITION";   // Account Name
input int      MagicNumber       = 366592;             // Magic Number (account ID)

input group "=== TRADING SETTINGS ==="
input double   LotSize           = 0.03;               // Lot Size
input int      MaxPositions      = 20;                 // Max Grid Positions
input int      GridSpacingPts    = 200;                // Grid Spacing (points)
input bool     BothDirections    = true;               // Trade BOTH directions
input int      CheckSeconds      = 5;                  // Check Interval

input group "=== RISK MANAGEMENT (ATR-BASED) ==="
input double   InpTPMultiplier   = 3.0;                // TP Multiplier (TP = SL x this)
input double   InpDynTPPercent   = 50.0;               // Dynamic TP % (partial close)
input double   InpRollingSLMult  = 1.5;                // Rolling SL Multiplier

input group "=== SIGNAL SETTINGS ==="
input int      FastEMA           = 5;                  // Fast EMA
input int      SlowEMA           = 13;                 // Slow EMA
input double   MinConfidence     = 0.3;                // Min Confidence

input group "=== COMPETITION MODE ==="
input bool     IgnoreDrawdown    = true;               // Ignore Drawdown Limits
input bool     ScaleOnWins       = true;               // Increase lots on wins

input group "=== STEALTH SETTINGS ==="
input bool     StealthMode       = false;              // Stealth Mode (hide EA identifiers)
input double   WinScaleFactor    = 1.2;                // Scale factor on wins

//+------------------------------------------------------------------+
//| CONSTANTS (HARDCODED - DO NOT MAKE INPUT)                         |
//+------------------------------------------------------------------+
#define SL_ATR_MULTIPLIER  1.0    // SL = ATR x 1.0 (HARDCODED)
#define ATR_PERIOD         14     // ATR period

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                   |
//+------------------------------------------------------------------+
int g_handleEmaFast = INVALID_HANDLE;
int g_handleEmaSlow = INVALID_HANDLE;
int g_handleAtr     = INVALID_HANDLE;

double g_emaFast[];
double g_emaSlow[];
double g_atr[];

datetime g_lastCheck = 0;
double g_currentLot = 0;
int g_winStreak = 0;

// Position tracking
struct GridLevel
{
   ulong  ticket;
   double entry;
   double sl;
   double tp;
   double dynTPPrice;     // 50% partial TP level
   bool   dynTPTaken;     // Has partial TP fired?
   double rollingSL;      // Current rolling SL level
   int    direction;      // 1=BUY, -1=SELL
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
   Print("  AGGRESSIVE COMPETITION GRID EA v2.0");
   Print("  *** FOR COMPETITION USE ONLY ***");
   Print("================================================");
   Print("  Account: ", AccountName);
   Print("  Magic: ", MagicNumber);
   Print("  Lot: ", LotSize);
   Print("  SL: ATR x ", DoubleToString(SL_ATR_MULTIPLIER, 1), " (hardcoded)");
   Print("  TP: SL x ", DoubleToString(InpTPMultiplier, 1), " (param)");
   Print("  Dynamic TP: ", DoubleToString(InpDynTPPercent, 0), "% partial close");
   Print("  Rolling SL: ", DoubleToString(InpRollingSLMult, 1), "x");
   Print("================================================");

   g_handleEmaFast = iMA(_Symbol, PERIOD_M1, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_handleEmaSlow = iMA(_Symbol, PERIOD_M1, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_handleAtr     = iATR(_Symbol, PERIOD_M1, ATR_PERIOD);

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

   SyncGrid();

   Print("Balance: $", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   Print("Synced ", g_gridCount, " existing positions");
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
   ManageGrid();

   if(TimeCurrent() - g_lastCheck < CheckSeconds) return;
   g_lastCheck = TimeCurrent();

   CheckEntry();
}

//+------------------------------------------------------------------+
//| Check for entry signal                                             |
//+------------------------------------------------------------------+
void CheckEntry()
{
   int buyCount = 0, sellCount = 0;
   CountPositionsByDirection(buyCount, sellCount);

   int totalPos = buyCount + sellCount;
   if(totalPos >= MaxPositions) return;

   if(CopyBuffer(g_handleEmaFast, 0, 0, 3, g_emaFast) < 3) return;
   if(CopyBuffer(g_handleEmaSlow, 0, 0, 3, g_emaSlow) < 3) return;
   if(CopyBuffer(g_handleAtr, 0, 0, 3, g_atr) < 3) return;

   double emaF = g_emaFast[1];
   double emaS = g_emaSlow[1];
   double atr  = g_atr[1];

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   bool buySignal  = (emaF > emaS);
   bool sellSignal = (emaF < emaS);

   double separation = MathAbs(emaF - emaS);
   double confidence = MathMin(1.0, (separation / atr) * 0.5);
   if(confidence < MinConfidence) return;

   if(buySignal && buyCount < MaxPositions / 2)
   {
      double lastBuyEntry = GetLastEntryPrice(1);
      double spacing = GridSpacingPts * point;
      if((buyCount == 0) || (tick.ask < lastBuyEntry - spacing))
         OpenPosition(ORDER_TYPE_BUY, buyCount + 1);
   }

   if(BothDirections && sellSignal && sellCount < MaxPositions / 2)
   {
      double lastSellEntry = GetLastEntryPrice(-1);
      double spacing = GridSpacingPts * point;
      if((sellCount == 0) || (tick.bid > lastSellEntry + spacing))
         OpenPosition(ORDER_TYPE_SELL, sellCount + 1);
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

   double lot = NormalizeLot(g_currentLot);

   // Get ATR
   if(CopyBuffer(g_handleAtr, 0, 0, 3, g_atr) < 3) return;
   double atr = g_atr[1];

   // SL = ATR x 1.0 (hardcoded multiplier)
   double slDistance = atr * SL_ATR_MULTIPLIER;

   // TP = SL_distance x TP_Multiplier (input parameter)
   double tpDistance = slDistance * InpTPMultiplier;

   // Dynamic TP = 50% of full TP distance
   double dynTPDistance = tpDistance * (InpDynTPPercent / 100.0);

   double sl, tp, dynTP;
   if(type == ORDER_TYPE_BUY)
   {
      sl    = price - slDistance;
      tp    = price + tpDistance;
      dynTP = price + dynTPDistance;
   }
   else
   {
      sl    = price + slDistance;
      tp    = price - tpDistance;
      dynTP = price - dynTPDistance;
   }

   // Normalize to tick size
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   sl    = MathRound(sl / tickSize) * tickSize;
   tp    = MathRound(tp / tickSize) * tickSize;
   dynTP = MathRound(dynTP / tickSize) * tickSize;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_DEAL;
   request.symbol       = _Symbol;
   request.volume       = lot;
   request.type         = type;
   request.price        = price;
   request.deviation    = 50;
   request.magic        = StealthMode ? 0 : MagicNumber;
   request.comment      = StealthMode ? "" : StringFormat("COMP_L%d", level);
   request.type_time    = ORDER_TIME_GTC;
   request.type_filling = GetFilling();
   request.sl           = sl;
   request.tp           = tp;

   if(!OrderSend(request, result))
   {
      Print("ERROR: Order failed - ", GetLastError(), " - ", result.comment);
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE)
   {
      int idx = ArraySize(g_grid);
      ArrayResize(g_grid, idx + 1);

      g_grid[idx].ticket     = result.order;
      g_grid[idx].entry      = price;
      g_grid[idx].sl         = sl;
      g_grid[idx].tp         = tp;
      g_grid[idx].dynTPPrice = dynTP;
      g_grid[idx].dynTPTaken = false;
      g_grid[idx].rollingSL  = sl;
      g_grid[idx].direction  = (type == ORDER_TYPE_BUY) ? 1 : -1;
      g_grid[idx].level      = level;
      g_gridCount = idx + 1;

      string typeStr = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
      Print(">>> ", typeStr, " L", level,
            " @ ", DoubleToString(price, 2),
            " | Lot: ", DoubleToString(lot, 2),
            " | SL: ", DoubleToString(sl, 2),
            " | TP: ", DoubleToString(tp, 2),
            " | DynTP(", DoubleToString(InpDynTPPercent, 0), "%): ", DoubleToString(dynTP, 2));
   }
   else
   {
      Print("Order rejected: ", result.comment, " (", result.retcode, ")");
   }
}

//+------------------------------------------------------------------+
//| Manage grid - Dynamic TP + Rolling SL                             |
//+------------------------------------------------------------------+
void ManageGrid()
{
   SyncGrid();

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   for(int i = g_gridCount - 1; i >= 0; i--)
   {
      ulong ticket = g_grid[i].ticket;

      if(!PositionSelectByTicket(ticket))
      {
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
      double openPrice    = PositionGetDouble(POSITION_PRICE_OPEN);
      double volume       = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      // --- DYNAMIC TP: Close 50% at partial target ---
      if(!g_grid[i].dynTPTaken)
      {
         bool hitDynTP = false;
         if(posType == POSITION_TYPE_BUY  && currentPrice >= g_grid[i].dynTPPrice) hitDynTP = true;
         if(posType == POSITION_TYPE_SELL && currentPrice <= g_grid[i].dynTPPrice) hitDynTP = true;

         if(hitDynTP)
         {
            double closeVol = NormalizeLot(volume * 0.5);
            if(closeVol > 0)
            {
               Print(">>> DYN TP ", DoubleToString(InpDynTPPercent, 0),
                     "% - Closing half L", g_grid[i].level,
                     " @ ", DoubleToString(currentPrice, 2));
               PartialClose(ticket, closeVol);
               g_grid[i].dynTPTaken = true;

               // Move SL to breakeven after partial TP
               double beSL = MathRound(openPrice / tickSize) * tickSize;
               ModifySL(ticket, beSL);
               g_grid[i].rollingSL = beSL;
               Print("   SL moved to breakeven: ", DoubleToString(beSL, 2));
            }
         }
      }

      // --- ROLLING SL: Trail SL by 1.5x multiplier ---
      if(g_grid[i].dynTPTaken)
      {
         double slDistance  = MathAbs(openPrice - g_grid[i].sl);
         double rollTarget  = slDistance * InpRollingSLMult;
         double newSL;

         if(posType == POSITION_TYPE_BUY)
         {
            double profit = currentPrice - openPrice;
            if(profit > rollTarget)
            {
               newSL = currentPrice - slDistance;
               newSL = MathRound(newSL / tickSize) * tickSize;
               if(newSL > g_grid[i].rollingSL)
               {
                  ModifySL(ticket, newSL);
                  Print("   Rolling SL up L", g_grid[i].level,
                        " -> ", DoubleToString(newSL, 2));
                  g_grid[i].rollingSL = newSL;
               }
            }
         }
         else
         {
            double profit = openPrice - currentPrice;
            if(profit > rollTarget)
            {
               newSL = currentPrice + slDistance;
               newSL = MathRound(newSL / tickSize) * tickSize;
               if(newSL < g_grid[i].rollingSL || g_grid[i].rollingSL == 0)
               {
                  ModifySL(ticket, newSL);
                  Print("   Rolling SL down L", g_grid[i].level,
                        " -> ", DoubleToString(newSL, 2));
                  g_grid[i].rollingSL = newSL;
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Partial close position                                             |
//+------------------------------------------------------------------+
bool PartialClose(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket)) return false;

   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   string symbol = PositionGetString(POSITION_SYMBOL);

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return false;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_DEAL;
   request.position     = ticket;
   request.symbol       = symbol;
   request.volume       = volume;
   request.deviation    = 50;
   request.magic        = MagicNumber;
   request.type_filling = GetFilling();

   if(posType == POSITION_TYPE_BUY)
   {
      request.type  = ORDER_TYPE_SELL;
      request.price = tick.bid;
   }
   else
   {
      request.type  = ORDER_TYPE_BUY;
      request.price = tick.ask;
   }

   if(!OrderSend(request, result))
   {
      Print("ERROR: Partial close failed - ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Modify stop loss on position                                       |
//+------------------------------------------------------------------+
bool ModifySL(ulong ticket, double newSL)
{
   if(!PositionSelectByTicket(ticket)) return false;

   double currentTP = PositionGetDouble(POSITION_TP);

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action   = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol   = PositionGetString(POSITION_SYMBOL);
   request.sl       = newSL;
   request.tp       = currentTP;

   if(!OrderSend(request, result))
   {
      Print("ERROR: ModifySL failed - ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Close full position                                                |
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

   request.action       = TRADE_ACTION_DEAL;
   request.position     = ticket;
   request.symbol       = symbol;
   request.volume       = volume;
   request.deviation    = 50;
   request.magic        = MagicNumber;
   request.type_filling = GetFilling();

   if(posType == POSITION_TYPE_BUY)
   {
      request.type  = ORDER_TYPE_SELL;
      request.price = tick.bid;
   }
   else
   {
      request.type  = ORDER_TYPE_BUY;
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
   for(int i = g_gridCount - 1; i >= 0; i--)
   {
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         ulong t = PositionGetTicket(p);
         if(t == g_grid[i].ticket) { found = true; break; }
      }
      if(!found) RemoveFromGrid(i);
   }

   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      ulong ticket = PositionGetTicket(p);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      bool tracked = false;
      for(int i = 0; i < g_gridCount; i++)
      {
         if(g_grid[i].ticket == ticket) { tracked = true; break; }
      }

      if(!tracked)
      {
         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double posSL = PositionGetDouble(POSITION_SL);
         double posTP = PositionGetDouble(POSITION_TP);
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         int idx = ArraySize(g_grid);
         ArrayResize(g_grid, idx + 1);

         g_grid[idx].ticket     = ticket;
         g_grid[idx].entry      = entry;
         g_grid[idx].sl         = posSL;
         g_grid[idx].tp         = posTP;
         g_grid[idx].direction  = (posType == POSITION_TYPE_BUY) ? 1 : -1;
         g_grid[idx].level      = idx + 1;
         g_grid[idx].rollingSL  = posSL;
         g_grid[idx].dynTPTaken = false;

         // Calculate dynTP from existing position
         double slDist = MathAbs(entry - posSL);
         double tpDist = slDist * InpTPMultiplier;
         double dynDist = tpDist * (InpDynTPPercent / 100.0);

         if(posType == POSITION_TYPE_BUY)
            g_grid[idx].dynTPPrice = entry + dynDist;
         else
            g_grid[idx].dynTPPrice = entry - dynDist;

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
      g_grid[i] = g_grid[i + 1];

   g_gridCount--;
   ArrayResize(g_grid, g_gridCount);
}

//+------------------------------------------------------------------+
//| Normalize lot size                                                 |
//+------------------------------------------------------------------+
double NormalizeLot(double lot)
{
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
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
