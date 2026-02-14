//+------------------------------------------------------------------+
//|                                          BlueGuardian_Elite.mq5 |
//|                              Elite Expert with +12 Compression   |
//|                    Based on expert_C7_E36_WR72 (89% simulated)   |
//+------------------------------------------------------------------+
#property copyright "QuantumChildren"
#property version   "1.00"


//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Account Settings ==="
input int      InpMagicNumber     = 365001;        // Magic Number
input double   InpVolume          = 0.01;          // Lot Size
input double   InpRiskPercent     = 0.5;           // Risk % per trade

input group "=== ATR Settings ==="
input double   SL_ATR_MULT        = 1.5;           // Stop Loss ATR Multiplier
input double   TP_ATR_MULT        = 3.0;           // Take Profit ATR Multiplier
input double   COMPRESSION_BOOST  = 12.0;          // Confidence Boost
input double   PARTIAL_TP_RATIO   = 0.5;           // Partial Take Profit Ratio (50%)
input double   BE_TRIGGER_PCT     = 0.30;          // Break-even Trigger (% of TP)
input double   TRAIL_TRIGGER_PCT  = 0.50;          // Trail Start (% of TP)

input group "=== Entry Filter Settings ==="
input int      InpATRPeriod       = 14;            // ATR Period
input int      InpEMAFast         = 8;             // Fast EMA
input int      InpEMASlow         = 21;            // Slow EMA
input int      InpEMA200          = 200;           // Trend EMA
input int      InpRSIPeriod       = 14;            // RSI Period
input double   InpConfidenceThresh = 0.22;         // Base Confidence Threshold

input group "=== Risk Management ==="
input double   InpDailyDDLimit    = 4.5;           // Daily DD Limit %
input double   InpMaxDDLimit      = 9.0;           // Max DD Limit %

input group "=== SL Dollar Cap ==="
input double   MaxSLDollars       = 50.0;          // Maximum SL loss in dollars per position (0=unlimited)

input group "=== Spread Filter ==="
input int      MaxSpreadPoints    = 100;           // Max spread to allow trade (points)

input group "=== Weekend Protection ==="
input bool     WeekendProtection  = true;          // Block new trades near weekend
input int      FridayCloseHour    = 21;            // Hour (UTC) to stop new entries on Friday

input group "=== Stealth Settings ==="
input bool     StealthMode        = false;         // Stealth Mode (hide EA identifiers)

input group "=== 12% ENTROPY REMOVAL (Logarithmic) ==="
input bool     InpUseEntropyRemoval = true;        // Enable 12% Entropy Removal
input int      InpEntropyLookback   = 50;          // Price lookback for entropy calc
input double   InpEntropyRemovalPct = 0.12;        // Entropy removal factor (12%)
input double   InpEntropyBlockLevel = 2.20;        // Block trades above this entropy

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
int g_atrHandle;
int g_emaFastHandle;
int g_emaSlowHandle;
int g_ema200Handle;
int g_rsiHandle;

double g_atrBuffer[];
double g_emaFastBuffer[];
double g_emaSlowBuffer[];
double g_ema200Buffer[];
double g_rsiBuffer[];

double g_startBalance;
double g_highWaterMark;
double g_dailyStartBalance;
datetime g_lastDayReset;

bool g_blocked = false;
string g_blockReason = "";

// Position tracking for hidden management
struct PositionData
{
   ulong  ticket;
   double entryPrice;
   double virtualSL;
   double virtualTP;
   double partialTP;
   bool   partialClosed;
   bool   breakEvenSet;
   bool   trailingActive;
};
PositionData g_positions[];

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   // Create indicator handles
   g_atrHandle = iATR(_Symbol, PERIOD_M5, InpATRPeriod);
   g_emaFastHandle = iMA(_Symbol, PERIOD_M5, InpEMAFast, 0, MODE_EMA, PRICE_CLOSE);
   g_emaSlowHandle = iMA(_Symbol, PERIOD_M5, InpEMASlow, 0, MODE_EMA, PRICE_CLOSE);
   g_ema200Handle = iMA(_Symbol, PERIOD_M5, InpEMA200, 0, MODE_EMA, PRICE_CLOSE);
   g_rsiHandle = iRSI(_Symbol, PERIOD_M5, InpRSIPeriod, PRICE_CLOSE);

   if(g_atrHandle == INVALID_HANDLE || g_emaFastHandle == INVALID_HANDLE ||
      g_emaSlowHandle == INVALID_HANDLE || g_ema200Handle == INVALID_HANDLE ||
      g_rsiHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles");
      return INIT_FAILED;
   }

   ArraySetAsSeries(g_atrBuffer, true);
   ArraySetAsSeries(g_emaFastBuffer, true);
   ArraySetAsSeries(g_emaSlowBuffer, true);
   ArraySetAsSeries(g_ema200Buffer, true);
   ArraySetAsSeries(g_rsiBuffer, true);

   // Initialize balance tracking
   g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_highWaterMark = g_startBalance;
   g_dailyStartBalance = g_startBalance;
   g_lastDayReset = TimeCurrent();

   Print("========================================");
   Print("BlueGuardian Elite Expert Initialized");
   Print("Based on: expert_C7_E36_WR72 (+12 Compression)");
   Print("Simulated Win Rate: 89.17%");
   Print("SL: ", SL_ATR_MULT, "x ATR | TP: ", TP_ATR_MULT, "x ATR");
   Print("Compression Boost: +", COMPRESSION_BOOST);
   Print("----------------------------------------");
   Print("12% ENTROPY REMOVAL (Logarithmic): ", InpUseEntropyRemoval ? "ENABLED" : "DISABLED");
   if(InpUseEntropyRemoval)
   {
      Print("  Lookback: ", InpEntropyLookback, " bars");
      Print("  Removal Factor: ", InpEntropyRemovalPct * 100, "%");
      Print("  Block Level: ", InpEntropyBlockLevel);
      Print("  Method: Shannon H = -SUM(p * log2(p))");
   }
   Print("========================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
   if(g_emaFastHandle != INVALID_HANDLE) IndicatorRelease(g_emaFastHandle);
   if(g_emaSlowHandle != INVALID_HANDLE) IndicatorRelease(g_emaSlowHandle);
   if(g_ema200Handle != INVALID_HANDLE) IndicatorRelease(g_ema200Handle);
   if(g_rsiHandle != INVALID_HANDLE) IndicatorRelease(g_rsiHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Always manage existing positions (hidden SL/TP)
   ManagePositions();

   // Check for new bar on M5
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, PERIOD_M5, 0);
   if(currentBarTime == lastBarTime) return;
   lastBarTime = currentBarTime;

   // Daily reset check
   CheckDailyReset();

   // Update high water mark
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(balance > g_highWaterMark) g_highWaterMark = balance;

   // Check drawdown limits
   if(!CheckDrawdownLimits())
   {
      if(!g_blocked) Print("BLOCKED: ", g_blockReason);
      g_blocked = true;
      return;
   }
   g_blocked = false;

   // Check if we already have a position
   if(HasOpenPosition()) return;

   // === 12% ENTROPY REMOVAL: Block if market is too chaotic ===
   if(IsEntropyBlocked()) return;

   // Calculate confidence (with +12 boost)
   double confidence = CalculateConfidence();

   // === 12% ENTROPY REMOVAL: Scale confidence by entropy factor ===
   double entropyFactor = GetEntropyRemovalFactor();
   double adjustedConfidence = confidence * entropyFactor;

   double adjustedThreshold = InpConfidenceThresh - (COMPRESSION_BOOST / 100.0);
   adjustedThreshold = MathMax(adjustedThreshold, 0.50);

   if(adjustedConfidence < adjustedThreshold)
   {
      // Low confidence (after entropy removal) - skip
      return;
   }

   // Get signal
   int signal = GetSignal();
   if(signal == 0) return;

   // Execute trade
   ExecuteTrade(signal);
}

//+------------------------------------------------------------------+
//| Calculate Shannon Entropy from Price Returns (Logarithmic)        |
//| Uses MathLog(p)/MathLog(2) -- identical to XAUUSD_GridCore        |
//| Returns normalized entropy [0..~2.32] for 5 bins                  |
//+------------------------------------------------------------------+
double CalculateShannonEntropy()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(_Symbol, PERIOD_M5, 0, InpEntropyLookback + 1, rates);
   if(copied < InpEntropyLookback + 1)
      return 99.0; // Cannot calculate -- return high entropy (safe: blocks trade)

   // Calculate percentage returns
   double changes[];
   ArrayResize(changes, copied - 1);

   for(int i = 0; i < copied - 1; i++)
   {
      if(rates[i+1].close == 0) continue;
      changes[i] = (rates[i].close - rates[i+1].close) / rates[i+1].close;
   }

   // Bin the changes into 5 categories (matching XAUUSD_GridCore pattern)
   int bins[5] = {0, 0, 0, 0, 0}; // Strong down, down, flat, up, strong up

   for(int i = 0; i < ArraySize(changes); i++)
   {
      if(changes[i] < -0.002)       bins[0]++;
      else if(changes[i] < -0.0005) bins[1]++;
      else if(changes[i] < 0.0005)  bins[2]++;
      else if(changes[i] < 0.002)   bins[3]++;
      else                           bins[4]++;
   }

   // Shannon entropy: H = -SUM(p * log2(p))
   double entropy = 0;
   int total = ArraySize(changes);
   if(total == 0) return 99.0;

   for(int i = 0; i < 5; i++)
   {
      if(bins[i] > 0)
      {
         double p = (double)bins[i] / total;
         entropy -= p * MathLog(p) / MathLog(2.0);
      }
   }

   return entropy; // Max ~2.32 for 5 bins (uniform distribution)
}

//+------------------------------------------------------------------+
//| Apply 12% Entropy Removal Factor                                  |
//| Returns scaling factor [0.0 .. 1.0] for confidence                |
//| At zero entropy: returns 1.0 (no reduction)                       |
//| At max entropy (2.32): returns 1.0 - (1.0 * 0.12) = 0.88         |
//+------------------------------------------------------------------+
double GetEntropyRemovalFactor()
{
   if(!InpUseEntropyRemoval)
      return 1.0; // Feature disabled -- no modification

   double entropy = CalculateShannonEntropy();
   double maxEntropy = MathLog(5.0) / MathLog(2.0); // ~2.322 for 5 bins

   // Normalize to [0..1]
   double normalizedEntropy = MathMin(1.0, entropy / maxEntropy);

   // Apply 12% removal: factor = 1.0 - (normalized * removal_pct)
   double factor = 1.0 - (normalizedEntropy * InpEntropyRemovalPct);
   factor = MathMax(0.0, MathMin(1.0, factor));

   // Log for transparency
   static datetime lastEntropyLog = 0;
   if(TimeCurrent() - lastEntropyLog > 300) // Every 5 min
   {
      Print("[ENTROPY_REMOVAL] Shannon H=", DoubleToString(entropy, 4),
            " | Normalized=", DoubleToString(normalizedEntropy, 4),
            " | Factor=", DoubleToString(factor, 4),
            " | Block=", (entropy >= InpEntropyBlockLevel ? "YES" : "NO"));
      lastEntropyLog = TimeCurrent();
   }

   return factor;
}

//+------------------------------------------------------------------+
//| Check if entropy is too high to trade                             |
//+------------------------------------------------------------------+
bool IsEntropyBlocked()
{
   if(!InpUseEntropyRemoval)
      return false;

   double entropy = CalculateShannonEntropy();
   if(entropy >= InpEntropyBlockLevel)
   {
      Print("[ENTROPY_REMOVAL] BLOCKED: Shannon entropy ", DoubleToString(entropy, 4),
            " >= threshold ", DoubleToString(InpEntropyBlockLevel, 4));
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Calculate Confidence Score (with +12 Compression Boost)           |
//+------------------------------------------------------------------+
double CalculateConfidence()
{
   if(CopyBuffer(g_emaFastHandle, 0, 0, 10, g_emaFastBuffer) < 10) return 0;
   if(CopyBuffer(g_emaSlowHandle, 0, 0, 10, g_emaSlowBuffer) < 10) return 0;
   if(CopyBuffer(g_ema200Handle, 0, 0, 10, g_ema200Buffer) < 10) return 0;
   if(CopyBuffer(g_rsiHandle, 0, 0, 10, g_rsiBuffer) < 10) return 0;
   if(CopyBuffer(g_atrHandle, 0, 0, 10, g_atrBuffer) < 10) return 0;

   double confidence = 0.0;
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // 1. EMA Alignment (30% weight)
   bool allAbove = (price > g_ema200Buffer[1] && g_emaFastBuffer[1] > g_emaSlowBuffer[1] && g_emaSlowBuffer[1] > g_ema200Buffer[1]);
   bool allBelow = (price < g_ema200Buffer[1] && g_emaFastBuffer[1] < g_emaSlowBuffer[1] && g_emaSlowBuffer[1] < g_ema200Buffer[1]);

   if(allAbove || allBelow)
      confidence += 0.30;
   else if((g_emaFastBuffer[1] > g_emaSlowBuffer[1]) || (g_emaFastBuffer[1] < g_emaSlowBuffer[1]))
      confidence += 0.15;

   // 2. EMA Separation Consistency (20% weight)
   double sep1 = MathAbs(g_emaFastBuffer[1] - g_emaSlowBuffer[1]);
   double sep2 = MathAbs(g_emaFastBuffer[5] - g_emaSlowBuffer[5]);
   double sepChange = MathAbs(sep1 - sep2) / (sep2 + 1e-10);

   if(sepChange < 0.10)
      confidence += 0.20;
   else if(sepChange < 0.20)
      confidence += 0.10;

   // 3. RSI Range (25% weight)
   double rsi = g_rsiBuffer[1];
   if(rsi >= 40 && rsi <= 60)
      confidence += 0.25;
   else if(rsi >= 30 && rsi <= 70)
      confidence += 0.15;
   else
      confidence += 0.05;

   // 4. ATR Stability (25% weight)
   double atrChange = MathAbs(g_atrBuffer[1] - g_atrBuffer[5]) / (g_atrBuffer[5] + 1e-10);
   if(atrChange < 0.15)
      confidence += 0.25;
   else if(atrChange < 0.30)
      confidence += 0.15;
   else
      confidence += 0.05;

   // Apply +12 compression boost
   confidence += (COMPRESSION_BOOST / 100.0);

   return MathMin(confidence, 1.0);
}

//+------------------------------------------------------------------+
//| Get Trading Signal                                                |
//+------------------------------------------------------------------+
int GetSignal()
{
   if(CopyBuffer(g_emaFastHandle, 0, 0, 3, g_emaFastBuffer) < 3) return 0;
   if(CopyBuffer(g_emaSlowHandle, 0, 0, 3, g_emaSlowBuffer) < 3) return 0;
   if(CopyBuffer(g_ema200Handle, 0, 0, 3, g_ema200Buffer) < 3) return 0;

   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   // EMA Crossover with trend filter
   bool fastAboveSlow = g_emaFastBuffer[1] > g_emaSlowBuffer[1];
   bool fastWasBelowSlow = g_emaFastBuffer[2] <= g_emaSlowBuffer[2];
   bool fastBelowSlow = g_emaFastBuffer[1] < g_emaSlowBuffer[1];
   bool fastWasAboveSlow = g_emaFastBuffer[2] >= g_emaSlowBuffer[2];

   // BUY: Fast crosses above Slow, price above EMA200
   if(fastAboveSlow && fastWasBelowSlow && price > g_ema200Buffer[1])
      return 1;

   // SELL: Fast crosses below Slow, price below EMA200
   if(fastBelowSlow && fastWasAboveSlow && price < g_ema200Buffer[1])
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Execute Trade with Hidden SL/TP                                   |
//+------------------------------------------------------------------+
void ExecuteTrade(int direction)
{
   // Weekend protection - stop opening new positions near market close
   if(WeekendProtection)
   {
      MqlDateTime dt;
      TimeCurrent(dt);
      if((dt.day_of_week == 5 && dt.hour >= FridayCloseHour) || dt.day_of_week == 6 || dt.day_of_week == 0)
      {
         Print("Weekend protection: Blocking new entry (Day=", dt.day_of_week, " Hour=", dt.hour, ")");
         return;
      }
   }

   if(CopyBuffer(g_atrHandle, 0, 0, 3, g_atrBuffer) < 3) return;

   double atr = g_atrBuffer[1];
   if(atr <= 0) return;

   // Spread check - skip trade if spread is too wide
   long currentSpread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(currentSpread > MaxSpreadPoints)
   {
      Print("SKIP: Spread ", currentSpread, " exceeds max ", MaxSpreadPoints, " - no entry");
      return;
   }

   double slDistance = atr * SL_ATR_MULT;

   // Cap SL to maximum dollar loss per position
   if(MaxSLDollars > 0)
   {
      double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tickSize > 0 && tickValue > 0)
      {
         double slDollars = (slDistance / tickSize) * tickValue * InpVolume;
         if(slDollars > MaxSLDollars)
         {
            double maxSlDistance = (MaxSLDollars / (tickValue * InpVolume)) * tickSize;
            Print("SL CAPPED: ", DoubleToString(slDistance/_Point, 0), " pts ($",
                  DoubleToString(slDollars, 2), ") -> ",
                  DoubleToString(maxSlDistance/_Point, 0), " pts ($",
                  DoubleToString(MaxSLDollars, 2), ")");
            slDistance = maxSlDistance;
         }
      }
   }

   double tpDistance = atr * TP_ATR_MULT;
   double partialTpDist = tpDistance * PARTIAL_TP_RATIO;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double price;
   ENUM_ORDER_TYPE orderType;
   double virtualSL, virtualTP, partialTP;

   if(direction > 0) // BUY
   {
      orderType = ORDER_TYPE_BUY;
      price = tick.ask;
      virtualSL = price - slDistance;
      virtualTP = price + tpDistance;
      partialTP = price + partialTpDist;
   }
   else // SELL
   {
      orderType = ORDER_TYPE_SELL;
      price = tick.bid;
      virtualSL = price + slDistance;
      virtualTP = price - tpDistance;
      partialTP = price - partialTpDist;
   }

   // Build order - HIDDEN SL/TP (0, 0)
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = InpVolume;
   request.type = orderType;
   request.price = price;
   request.sl = 0;  // Hidden
   request.tp = 0;  // Hidden
   request.deviation = 50;
   request.magic = StealthMode ? 0 : InpMagicNumber;
   request.comment = StealthMode ? "" : "BG_Elite_+12";
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = GetFillingMode();

   if(!OrderSend(request, result))
   {
      Print("Order failed: ", GetLastError());
      return;
   }

   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("Order rejected: ", result.comment);
      return;
   }

   // Track position
   int idx = ArraySize(g_positions);
   ArrayResize(g_positions, idx + 1);

   g_positions[idx].ticket = result.order;
   g_positions[idx].entryPrice = price;
   g_positions[idx].virtualSL = virtualSL;
   g_positions[idx].virtualTP = virtualTP;
   g_positions[idx].partialTP = partialTP;
   g_positions[idx].partialClosed = false;
   g_positions[idx].breakEvenSet = false;
   g_positions[idx].trailingActive = false;

   string typeStr = (direction > 0) ? "BUY" : "SELL";
   Print(typeStr, " @ ", DoubleToString(price, 2),
         " | Hidden SL: ", DoubleToString(virtualSL, 2),
         " | Hidden TP: ", DoubleToString(virtualTP, 2));

   // === EMERGENCY BROKER-SIDE SL (catastrophic backstop - 5x virtual SL) ===
   // If MT5 crashes or EA is removed, this wide SL protects the position.
   // Set far enough away (5x) that normal virtual SL management is not affected.
   {
      double emergency_sl_dist = slDistance * 5.0;  // 5x normal SL distance
      double emergencySL = (direction > 0) ?
          price - emergency_sl_dist :
          price + emergency_sl_dist;
      emergencySL = NormalizeDouble(emergencySL, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));

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
                " (5x SL dist = ", DoubleToString(emergency_sl_dist, 2), ")");
      else
          Print("WARNING: Emergency SL rejected: ", slRes.retcode, " - ", slRes.comment);
   }
}

//+------------------------------------------------------------------+
//| Manage Positions - Hidden SL/TP, Partial TP, BE, Trail           |
//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = ArraySize(g_positions) - 1; i >= 0; i--)
   {
      ulong ticket = g_positions[i].ticket;

      if(!PositionSelectByTicket(ticket))
      {
         // Position closed externally
         RemovePosition(i);
         continue;
      }

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double entry = g_positions[i].entryPrice;

      // Check Virtual SL/TP
      bool hitSL = false;
      bool hitTP = false;

      if(posType == POSITION_TYPE_BUY)
      {
         hitSL = (currentPrice <= g_positions[i].virtualSL);
         hitTP = (currentPrice >= g_positions[i].virtualTP);
      }
      else
      {
         hitSL = (currentPrice >= g_positions[i].virtualSL);
         hitTP = (currentPrice <= g_positions[i].virtualTP);
      }

      if(hitSL)
      {
         Print("HIDDEN SL HIT - Closing");
         ClosePosition(ticket, volume);
         RemovePosition(i);
         continue;
      }

      if(hitTP)
      {
         Print("HIDDEN TP HIT - Closing");
         ClosePosition(ticket, volume);
         RemovePosition(i);
         continue;
      }

      // Check Partial TP (50% at first target)
      if(!g_positions[i].partialClosed)
      {
         bool hitPartial = false;
         if(posType == POSITION_TYPE_BUY)
            hitPartial = (currentPrice >= g_positions[i].partialTP);
         else
            hitPartial = (currentPrice <= g_positions[i].partialTP);

         if(hitPartial)
         {
            double closeVol = NormalizeDouble(volume * PARTIAL_TP_RATIO, 2);
            if(closeVol >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN))
            {
               if(ClosePosition(ticket, closeVol))
               {
                  g_positions[i].partialClosed = true;
                  Print("PARTIAL TP: Closed 50%");
               }
            }
         }
      }

      // Check Break-Even
      double tpDist = MathAbs(g_positions[i].virtualTP - entry);
      double profitDist = (posType == POSITION_TYPE_BUY) ?
                          (currentPrice - entry) : (entry - currentPrice);
      double progress = profitDist / tpDist;

      if(!g_positions[i].breakEvenSet && progress >= BE_TRIGGER_PCT && profitDist > 0)
      {
         double buffer = 10 * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         if(posType == POSITION_TYPE_BUY)
            g_positions[i].virtualSL = entry + buffer;
         else
            g_positions[i].virtualSL = entry - buffer;

         g_positions[i].breakEvenSet = true;
         Print("BREAK-EVEN: SL moved to entry");
      }

      // Check Trailing
      if(g_positions[i].breakEvenSet && progress >= TRAIL_TRIGGER_PCT)
      {
         if(CopyBuffer(g_atrHandle, 0, 0, 3, g_atrBuffer) >= 3)
         {
            double trailDist = g_atrBuffer[1];

            if(posType == POSITION_TYPE_BUY)
            {
               double newSL = currentPrice - trailDist;
               if(newSL > g_positions[i].virtualSL)
               {
                  g_positions[i].virtualSL = newSL;
                  if(!g_positions[i].trailingActive)
                  {
                     g_positions[i].trailingActive = true;
                     Print("TRAILING: Started");
                  }
               }
            }
            else
            {
               double newSL = currentPrice + trailDist;
               if(newSL < g_positions[i].virtualSL)
               {
                  g_positions[i].virtualSL = newSL;
                  if(!g_positions[i].trailingActive)
                  {
                     g_positions[i].trailingActive = true;
                     Print("TRAILING: Started");
                  }
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close Position                                                    |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket)) return false;

   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return false;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = _Symbol;
   request.volume = volume;
   request.deviation = 50;
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
      Print("Close failed: ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Remove Position from Tracking                                     |
//+------------------------------------------------------------------+
void RemovePosition(int index)
{
   for(int i = index; i < ArraySize(g_positions) - 1; i++)
      g_positions[i] = g_positions[i + 1];
   ArrayResize(g_positions, ArraySize(g_positions) - 1);
}

//+------------------------------------------------------------------+
//| Has Open Position                                                 |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Check Drawdown Limits                                             |
//+------------------------------------------------------------------+
bool CheckDrawdownLimits()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(balance, equity);

   if(g_dailyStartBalance <= 0 || g_highWaterMark <= 0) return true;

   // Daily DD
   double dailyDD = ((g_dailyStartBalance - current) / g_dailyStartBalance) * 100;
   if(dailyDD >= InpDailyDDLimit)
   {
      g_blockReason = StringFormat("Daily DD %.2f%% >= %.2f%%", dailyDD, InpDailyDDLimit);
      return false;
   }

   // Max DD
   double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
   if(maxDD >= InpMaxDDLimit)
   {
      g_blockReason = StringFormat("Max DD %.2f%% >= %.2f%%", maxDD, InpMaxDDLimit);
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Check Daily Reset                                                 |
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
//| Get Filling Mode                                                  |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
   uint filling = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK) return ORDER_FILLING_FOK;
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC) return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}
//+------------------------------------------------------------------+
