//+------------------------------------------------------------------+
//|                                        BlueGuardian_Quantum.mq5  |
//|                        Quantum Children - BRAIN Logic MQL5 Port  |
//|           Fixed Dollar Risk | Rolling SL | Dynamic TP            |
//+------------------------------------------------------------------+
//| Matches BRAIN_BG_CHALLENGE.py + MASTER_CONFIG.json logic:        |
//|   - Auto lot sizing: lot = MaxLoss / (sl_ticks * tick_value)     |
//|   - ATR-based SL distance with dollar risk cap                   |
//|   - Rolling SL trails at initialDist / 1.5x                     |
//|   - Dynamic TP closes at 50% of TP target                       |
//|   - Multi-factor confidence scoring                              |
//|   - Daily + Max drawdown limits for challenge accounts           |
//+------------------------------------------------------------------+
#property copyright "QuantumChildren"
#property version   "3.00"
#property description "BRAIN-matched EA: Fixed $1 risk, Rolling SL, Dynamic TP"

//+------------------------------------------------------------------+
//| Core Risk Settings (from MASTER_CONFIG.json)                     |
//+------------------------------------------------------------------+
input group "=== Core Risk Settings (MASTER_CONFIG) ==="
input double InpMaxLossDollars    = 1.00;      // Max Loss per Trade ($)
input double InpTpMultiplier      = 3.0;       // TP = Nx SL Distance
input double InpRollingSLMult     = 1.5;       // Rolling SL Divider
input int    InpDynamicTpPct      = 50;        // Dynamic TP Trigger (% of TP)
input bool   InpUseDynamicTp      = true;      // Enable Dynamic TP
input bool   InpUseRollingSL      = true;      // Enable Rolling SL
input double InpAtrMultiplier     = 0.0438;    // ATR Multiplier (SL distance)
input double InpConfidenceThresh  = 0.22;      // Min Confidence to Trade

input group "=== Account Settings ==="
input int    InpMagic             = 365001;    // Magic Number

input group "=== Drawdown Limits ==="
input double InpDailyDDLimit      = 4.5;       // Daily DD Limit %
input double InpMaxDDLimit        = 9.0;       // Max DD Limit %

input group "=== Signal Settings ==="
input int    InpEmaFast           = 8;         // Fast EMA
input int    InpEmaSlow           = 21;        // Slow EMA
input int    InpEma200            = 200;       // Trend EMA
input int    InpRsiPeriod         = 14;        // RSI Period
input int    InpAtrPeriod         = 14;        // ATR Period


//+------------------------------------------------------------------+
//| Position tracking - stores initial SL distance per trade         |
//+------------------------------------------------------------------+
struct PositionTrack
{
   ulong  ticket;
   double initialSLDist;   // Original SL distance at entry (for rolling SL + dynamic TP)
   bool   active;
};

PositionTrack g_tracks[];
int           g_trackCount = 0;

//+------------------------------------------------------------------+
//| Indicator handles and buffers                                    |
//+------------------------------------------------------------------+
int    hEmaFast, hEmaSlow, hEma200, hRsi, hAtr;
double bufEmaFast[], bufEmaSlow[], bufEma200[], bufRsi[], bufAtr[];

//+------------------------------------------------------------------+
//| Drawdown tracking                                                |
//+------------------------------------------------------------------+
double   g_dailyStartBal;
double   g_highWaterMark;
datetime g_lastDayReset;
bool     g_blocked;
string   g_blockReason;

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   hEmaFast = iMA(_Symbol, PERIOD_M5, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlow = iMA(_Symbol, PERIOD_M5, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   hEma200  = iMA(_Symbol, PERIOD_M5, InpEma200, 0, MODE_EMA, PRICE_CLOSE);
   hRsi     = iRSI(_Symbol, PERIOD_M5, InpRsiPeriod, PRICE_CLOSE);
   hAtr     = iATR(_Symbol, PERIOD_M5, InpAtrPeriod);

   if(hEmaFast == INVALID_HANDLE || hEmaSlow == INVALID_HANDLE ||
      hEma200 == INVALID_HANDLE  || hRsi == INVALID_HANDLE ||
      hAtr == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   ArraySetAsSeries(bufEmaFast, true);
   ArraySetAsSeries(bufEmaSlow, true);
   ArraySetAsSeries(bufEma200, true);
   ArraySetAsSeries(bufRsi, true);
   ArraySetAsSeries(bufAtr, true);

   g_dailyStartBal = AccountInfoDouble(ACCOUNT_BALANCE);
   g_highWaterMark = g_dailyStartBal;
   g_lastDayReset  = TimeCurrent();
   g_blocked       = false;
   g_blockReason   = "";

   // Sync any existing positions from a restart
   SyncExistingPositions();

   Print("================================================");
   Print("  QUANTUM CHILDREN - BRAIN Logic EA v3.00");
   Print("================================================");
   Print("  Symbol:     ", _Symbol);
   Print("  Magic:      ", InpMagic);
   Print("  Max Risk:   $", DoubleToString(InpMaxLossDollars, 2), " per trade");
   Print("  ATR Mult:   ", InpAtrMultiplier, " (SL distance)");
   Print("  TP:         ", InpTpMultiplier, "x SL ($", DoubleToString(InpMaxLossDollars * InpTpMultiplier, 2), " target)");
   Print("  Rolling SL: ", InpUseRollingSL ? "ON" : "OFF", " (", InpRollingSLMult, "x divider)");
   Print("  Dynamic TP: ", InpUseDynamicTp ? "ON" : "OFF", " (", InpDynamicTpPct, "% trigger)");
   Print("  Confidence: >= ", InpConfidenceThresh);
   Print("  DD Limits:  ", InpDailyDDLimit, "% daily / ", InpMaxDDLimit, "% max");
   Print("  Balance:    $", DoubleToString(g_dailyStartBal, 2));
   Print("================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(hEmaFast != INVALID_HANDLE) IndicatorRelease(hEmaFast);
   if(hEmaSlow != INVALID_HANDLE) IndicatorRelease(hEmaSlow);
   if(hEma200  != INVALID_HANDLE) IndicatorRelease(hEma200);
   if(hRsi     != INVALID_HANDLE) IndicatorRelease(hRsi);
   if(hAtr     != INVALID_HANDLE) IndicatorRelease(hAtr);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Manage existing positions on EVERY tick (rolling SL, dynamic TP)
   ManagePositions();

   // New M5 bar gate
   static datetime lastBar = 0;
   datetime curBar = iTime(_Symbol, PERIOD_M5, 0);
   if(curBar == 0 || curBar == lastBar) return;
   lastBar = curBar;

   // Daily reset
   CheckDailyReset();

   // Update high water mark
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   if(bal > g_highWaterMark) g_highWaterMark = bal;

   // Drawdown check
   if(!CheckDrawdownLimits())
   {
      if(!g_blocked) Print("BLOCKED: ", g_blockReason);
      g_blocked = true;
      return;
   }
   g_blocked = false;

   // Already have a position
   if(HasOpenPosition()) return;

   // Confidence check
   double confidence = CalculateConfidence();
   if(confidence < InpConfidenceThresh) return;

   // Signal
   int signal = GetSignal();
   if(signal == 0) return;

   // Execute with auto lot sizing
   ExecuteTrade(signal, confidence);
}

//+------------------------------------------------------------------+
//| EXECUTE TRADE - Fixed dollar risk with auto lot sizing           |
//| This is the core: lot = MaxLoss / (sl_ticks * tick_value)        |
//+------------------------------------------------------------------+
void ExecuteTrade(int direction, double confidence)
{
   if(CopyBuffer(hAtr, 0, 0, 3, bufAtr) < 3) return;
   double atr = bufAtr[1];
   if(atr <= 0) return;

   // SL distance from ATR (matching BRAIN: sl_distance = atr * ATR_MULTIPLIER)
   double slDist = atr * InpAtrMultiplier;

   // Enforce broker minimum stops level
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int stopsLevel = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist = (stopsLevel + 10) * point;
   if(slDist < minDist) slDist = minDist;

   // AUTO LOT SIZING: lot = MaxLossDollars / (sl_ticks * tick_value)
   // This ensures every trade risks EXACTLY $InpMaxLossDollars
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double slTicks   = slDist / tickSize;
   double lot;

   if(tickValue > 0 && slTicks > 0)
      lot = InpMaxLossDollars / (slTicks * tickValue);
   else
      lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   // Clamp lot to broker limits and round to step
   double volMin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volMax  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lot = MathMax(volMin, lot);
   lot = MathMin(volMax, lot);
   lot = MathFloor(lot / volStep) * volStep;
   lot = MathMax(volMin, lot);

   // TP distance = SL distance * multiplier
   double tpDist = slDist * InpTpMultiplier;

   // Price, SL, TP
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double price, sl, tp;
   ENUM_ORDER_TYPE orderType;

   if(direction > 0) // BUY
   {
      orderType = ORDER_TYPE_BUY;
      price = NormalizeDouble(tick.ask, digits);
      sl    = NormalizeDouble(price - slDist, digits);
      tp    = NormalizeDouble(price + tpDist, digits);
   }
   else // SELL
   {
      orderType = ORDER_TYPE_SELL;
      price = NormalizeDouble(tick.bid, digits);
      sl    = NormalizeDouble(price + slDist, digits);
      tp    = NormalizeDouble(price - tpDist, digits);
   }

   // Send order with VISIBLE SL/TP (broker enforced)
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_DEAL;
   request.symbol       = _Symbol;
   request.volume       = lot;
   request.type         = orderType;
   request.price        = price;
   request.sl           = sl;
   request.tp           = tp;
   request.magic        = InpMagic;
   request.comment      = "QC_Quantum";
   request.deviation    = 50;
   request.type_time    = ORDER_TIME_GTC;
   request.type_filling = GetFillingMode();

   if(!OrderSend(request, result))
   {
      Print("Order failed: ", GetLastError(),
            " | Price=", DoubleToString(price, digits),
            " | SL=", DoubleToString(sl, digits),
            " | TP=", DoubleToString(tp, digits));
      return;
   }

   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("Order rejected: ", result.retcode, " - ", result.comment);
      return;
   }

   // Track position with initial SL distance (for rolling SL + dynamic TP)
   AddTrack(result.order, slDist);

   string typeStr = (direction > 0) ? "BUY" : "SELL";
   Print(typeStr, " @ ", DoubleToString(price, digits),
         " | Lot: ", DoubleToString(lot, 2),
         " | SL: ", DoubleToString(sl, digits), " ($", DoubleToString(InpMaxLossDollars, 2), " risk)",
         " | TP: ", DoubleToString(tp, digits), " ($", DoubleToString(InpMaxLossDollars * InpTpMultiplier, 2), " target)",
         " | Conf: ", DoubleToString(confidence, 2));
}

//+------------------------------------------------------------------+
//| MANAGE POSITIONS - Rolling SL + Dynamic TP (every tick)          |
//| Mirrors BRAIN_BG_CHALLENGE.py manage_positions()                 |
//+------------------------------------------------------------------+
void ManagePositions()
{
   double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int stopsLvl  = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist = (stopsLvl + 10) * point;
   int digits     = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      double entry   = PositionGetDouble(POSITION_PRICE_OPEN);
      double current = PositionGetDouble(POSITION_PRICE_CURRENT);
      double curSL   = PositionGetDouble(POSITION_SL);
      double curTP   = PositionGetDouble(POSITION_TP);
      double volume  = PositionGetDouble(POSITION_VOLUME);
      bool   isBuy   = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);

      // Get initial SL distance from tracking (or estimate from ATR)
      double initialSLDist = GetInitialSLDist(ticket);
      if(initialSLDist <= 0) continue;

      // Current profit distance
      double profitDist = isBuy ? (current - entry) : (entry - current);

      // --- DYNAMIC TP: Close at X% of TP target ---
      if(InpUseDynamicTp)
      {
         double tpTarget   = initialSLDist * InpTpMultiplier;
         double dynTpThres = tpTarget * (InpDynamicTpPct / 100.0);

         if(profitDist >= dynTpThres && dynTpThres > 0)
         {
            double closePrice = isBuy ? SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                                        SymbolInfoDouble(_Symbol, SYMBOL_ASK);

            Print("DYNAMIC TP: Closing #", ticket, " at ", DoubleToString(profitDist / initialSLDist, 1),
                  "R profit ($", DoubleToString(profitDist * volume * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE), 2), ")");

            ClosePosition(ticket, volume, isBuy, closePrice);
            RemoveTrack(ticket);
            continue;
         }
      }

      // --- ROLLING SL: Trail at initialDist / 1.5x, maintain TP ratio ---
      if(InpUseRollingSL && profitDist > 0)
      {
         double rolledDist = initialSLDist / InpRollingSLMult;
         double newSL, newTP;

         if(isBuy)
         {
            newSL = NormalizeDouble(current - rolledDist, digits);
            newSL = MathMax(newSL, entry);  // Never below entry

            // Respect minimum stops distance
            if(newSL > current - minDist)
               newSL = NormalizeDouble(current - minDist, digits);

            // Only move SL forward
            if(curSL > 0 && newSL <= curSL) continue;

            // Move TP forward to maintain ratio
            double risk = current - newSL;
            newTP = NormalizeDouble(current + (risk * InpTpMultiplier), digits);
            if(curTP > 0) newTP = MathMax(newTP, curTP);  // Never decrease TP
         }
         else // SELL
         {
            newSL = NormalizeDouble(current + rolledDist, digits);
            newSL = MathMin(newSL, entry);  // Never above entry

            // Respect minimum stops distance
            if(newSL < current + minDist)
               newSL = NormalizeDouble(current + minDist, digits);

            // Only move SL forward (down for sells)
            if(curSL > 0 && newSL >= curSL) continue;

            // Move TP forward to maintain ratio
            double risk = newSL - current;
            newTP = NormalizeDouble(current - (risk * InpTpMultiplier), digits);
            if(curTP > 0) newTP = MathMin(newTP, curTP);  // Never increase TP (for sells)
         }

         // Only modify if SL actually changed
         if(MathAbs(newSL - curSL) > point || MathAbs(newTP - curTP) > point)
         {
            MqlTradeRequest req;
            MqlTradeResult  res;
            ZeroMemory(req);
            ZeroMemory(res);

            req.action   = TRADE_ACTION_SLTP;
            req.symbol   = _Symbol;
            req.position = ticket;
            req.sl       = newSL;
            req.tp       = newTP;

            if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE)
            {
               Print("TRAIL: #", ticket, " SL -> ", DoubleToString(newSL, digits),
                     " | TP -> ", DoubleToString(newTP, digits));
            }
         }
      }
   }

   // Clean up tracks for positions that no longer exist
   CleanupTracks();
}

//+------------------------------------------------------------------+
//| SIGNAL: EMA crossover + EMA200 trend filter                      |
//+------------------------------------------------------------------+
int GetSignal()
{
   if(CopyBuffer(hEmaFast, 0, 0, 3, bufEmaFast) < 3) return 0;
   if(CopyBuffer(hEmaSlow, 0, 0, 3, bufEmaSlow) < 3) return 0;
   if(CopyBuffer(hEma200, 0, 0, 3, bufEma200) < 3)   return 0;

   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // EMA crossover
   bool crossUp   = (bufEmaFast[2] <= bufEmaSlow[2]) && (bufEmaFast[1] > bufEmaSlow[1]);
   bool crossDown = (bufEmaFast[2] >= bufEmaSlow[2]) && (bufEmaFast[1] < bufEmaSlow[1]);

   // BUY: cross up + price above EMA200 (uptrend)
   if(crossUp && price > bufEma200[1])
      return 1;

   // SELL: cross down + price below EMA200 (downtrend)
   if(crossDown && price < bufEma200[1])
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| CONFIDENCE: Multi-factor score (EMA alignment, RSI, ATR)         |
//+------------------------------------------------------------------+
double CalculateConfidence()
{
   if(CopyBuffer(hEmaFast, 0, 0, 10, bufEmaFast) < 10) return 0;
   if(CopyBuffer(hEmaSlow, 0, 0, 10, bufEmaSlow) < 10) return 0;
   if(CopyBuffer(hEma200, 0, 0, 10, bufEma200) < 10)   return 0;
   if(CopyBuffer(hRsi, 0, 0, 10, bufRsi) < 10)         return 0;
   if(CopyBuffer(hAtr, 0, 0, 10, bufAtr) < 10)         return 0;

   double confidence = 0.0;
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // 1. EMA Alignment (30%)
   bool bullAlign = (price > bufEma200[1] && bufEmaFast[1] > bufEmaSlow[1] && bufEmaSlow[1] > bufEma200[1]);
   bool bearAlign = (price < bufEma200[1] && bufEmaFast[1] < bufEmaSlow[1] && bufEmaSlow[1] < bufEma200[1]);

   if(bullAlign || bearAlign)
      confidence += 0.30;
   else if(bufEmaFast[1] > bufEmaSlow[1] || bufEmaFast[1] < bufEmaSlow[1])
      confidence += 0.15;

   // 2. EMA Separation Consistency (20%)
   double sep1 = MathAbs(bufEmaFast[1] - bufEmaSlow[1]);
   double sep5 = MathAbs(bufEmaFast[5] - bufEmaSlow[5]);
   double sepChange = MathAbs(sep1 - sep5) / (sep5 + 1e-10);

   if(sepChange < 0.10)
      confidence += 0.20;
   else if(sepChange < 0.20)
      confidence += 0.10;

   // 3. RSI Range (25%)
   double rsi = bufRsi[1];
   if(rsi >= 40 && rsi <= 60)
      confidence += 0.25;
   else if(rsi >= 30 && rsi <= 70)
      confidence += 0.15;
   else
      confidence += 0.05;

   // 4. ATR Stability (25%)
   double atrChange = MathAbs(bufAtr[1] - bufAtr[5]) / (bufAtr[5] + 1e-10);
   if(atrChange < 0.15)
      confidence += 0.25;
   else if(atrChange < 0.30)
      confidence += 0.15;
   else
      confidence += 0.05;

   return MathMin(confidence, 1.0);
}

//+------------------------------------------------------------------+
//| DRAWDOWN LIMITS                                                  |
//+------------------------------------------------------------------+
bool CheckDrawdownLimits()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(balance, equity);

   if(g_dailyStartBal <= 0 || g_highWaterMark <= 0) return true;

   double dailyDD = ((g_dailyStartBal - current) / g_dailyStartBal) * 100;
   if(dailyDD >= InpDailyDDLimit)
   {
      g_blockReason = StringFormat("Daily DD %.2f%% >= %.2f%%", dailyDD, InpDailyDDLimit);
      return false;
   }

   double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
   if(maxDD >= InpMaxDDLimit)
   {
      g_blockReason = StringFormat("Max DD %.2f%% >= %.2f%%", maxDD, InpMaxDDLimit);
      return false;
   }

   return true;
}

void CheckDailyReset()
{
   MqlDateTime now, last;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(g_lastDayReset, last);

   if(now.day != last.day || now.mon != last.mon || now.year != last.year)
   {
      g_dailyStartBal = AccountInfoDouble(ACCOUNT_BALANCE);
      g_lastDayReset  = TimeCurrent();
      g_blocked       = false;
      g_blockReason   = "";
      Print("Daily reset. Balance: $", DoubleToString(g_dailyStartBal, 2));
   }
}

//+------------------------------------------------------------------+
//| HAS OPEN POSITION                                                |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| CLOSE POSITION                                                   |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket, double volume, bool isBuy, double closePrice)
{
   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_DEAL;
   request.position     = ticket;
   request.symbol       = _Symbol;
   request.volume       = volume;
   request.type         = isBuy ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   request.price        = NormalizeDouble(closePrice, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
   request.deviation    = 50;
   request.magic        = InpMagic;
   request.comment      = "QC_DynTP";
   request.type_filling = GetFillingMode();

   if(!OrderSend(request, result))
   {
      Print("Close failed: ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| FILLING MODE DETECTION                                           |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
   uint filling = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK) return ORDER_FILLING_FOK;
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC) return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| POSITION TRACKING - Store initial SL distance per trade          |
//+------------------------------------------------------------------+
void AddTrack(ulong ticket, double slDist)
{
   int idx = g_trackCount;
   ArrayResize(g_tracks, idx + 1);

   g_tracks[idx].ticket       = ticket;
   g_tracks[idx].initialSLDist = slDist;
   g_tracks[idx].active       = true;
   g_trackCount = idx + 1;
}

double GetInitialSLDist(ulong ticket)
{
   for(int i = 0; i < g_trackCount; i++)
   {
      if(g_tracks[i].ticket == ticket && g_tracks[i].active)
         return g_tracks[i].initialSLDist;
   }

   // Not tracked (EA restarted) - estimate from current ATR
   if(CopyBuffer(hAtr, 0, 0, 3, bufAtr) >= 3)
   {
      double est = bufAtr[1] * InpAtrMultiplier;
      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      int stopsLvl = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
      double minDist = (stopsLvl + 10) * point;
      if(est < minDist) est = minDist;
      return est;
   }

   return 0;
}

void RemoveTrack(ulong ticket)
{
   for(int i = 0; i < g_trackCount; i++)
   {
      if(g_tracks[i].ticket == ticket)
      {
         g_tracks[i].active = false;
         return;
      }
   }
}

void CleanupTracks()
{
   // Remove tracks for positions that no longer exist
   for(int i = g_trackCount - 1; i >= 0; i--)
   {
      if(!g_tracks[i].active) continue;

      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         if(PositionGetTicket(p) == g_tracks[i].ticket)
         {
            found = true;
            break;
         }
      }

      if(!found)
         g_tracks[i].active = false;
   }

   // Compact array
   int write = 0;
   for(int read = 0; read < g_trackCount; read++)
   {
      if(g_tracks[read].active)
      {
         if(write != read)
            g_tracks[write] = g_tracks[read];
         write++;
      }
   }
   g_trackCount = write;
   ArrayResize(g_tracks, g_trackCount);
}

void SyncExistingPositions()
{
   // On restart, create tracking for any existing positions
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      // Estimate initial SL distance from current SL or ATR
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = PositionGetDouble(POSITION_SL);
      double slDist = 0;

      if(sl > 0)
         slDist = MathAbs(entry - sl);

      if(slDist <= 0)
      {
         // No SL set or at entry, estimate from ATR
         if(CopyBuffer(hAtr, 0, 0, 3, bufAtr) >= 3)
            slDist = bufAtr[1] * InpAtrMultiplier;
      }

      if(slDist > 0)
      {
         AddTrack(ticket, slDist);
         Print("Synced position #", ticket, " | Est SL dist: ", DoubleToString(slDist, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)));
      }
   }
}
//+------------------------------------------------------------------+
