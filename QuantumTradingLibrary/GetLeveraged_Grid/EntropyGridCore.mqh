//+------------------------------------------------------------------+
//|                                              EntropyGridCore.mqh |
//|                      Shared Grid Logic with Entropy Filtering    |
//|                      GetLeveraged Multi-Symbol Grid System       |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property version   "1.00"

//+------------------------------------------------------------------+
//| Entropy Filter State                                              |
//+------------------------------------------------------------------+
enum ENUM_ENTROPY_STATE
{
   ENTROPY_LOW,        // Market is predictable - TRADE
   ENTROPY_MEDIUM,     // Market is mixed - REDUCE SIZE
   ENTROPY_HIGH        // Market is chaotic - NO TRADE
};

//+------------------------------------------------------------------+
//| Grid Position Structure                                           |
//+------------------------------------------------------------------+
struct GridPosition
{
   ulong    ticket;
   double   entryPrice;
   double   virtualSL;       // Hidden SL - managed internally
   double   virtualTP;       // Hidden TP - managed internally
   double   partialTP;       // First target (50% of TP)
   double   volume;
   double   partialVolume;   // Volume to close at partial (50%)
   bool     partialClosed;   // Has partial TP been taken?
   bool     breakEvenSet;    // Has breakeven been triggered?
   bool     trailingActive;  // Is trailing active?
   datetime openTime;
   int      gridLevel;
};

//+------------------------------------------------------------------+
//| Entropy Grid Manager Class                                        |
//+------------------------------------------------------------------+
class CEntropyGridManager
{
private:
   // Configuration
   string         m_symbol;
   int            m_magic;
   double         m_accountId;

   // Entropy configuration (hard-coded per spec)
   double         m_confidenceThreshold;    // 0.80 base
   int            m_compressionBoost;       // +12
   double         m_slAtrMultiplier;        // 1.5x
   double         m_tpAtrMultiplier;        // 3.0x
   double         m_partialTpRatio;         // 0.5 (50%)
   double         m_breakEvenTrigger;       // 0.30 (30% of TP)
   double         m_trailStartTrigger;      // 0.50 (50% of TP)

   // Grid configuration
   int            m_maxPositions;
   double         m_gridSpacingAtr;
   double         m_baseLotSize;
   double         m_maxLotSize;
   double         m_riskPerTradePct;

   // Drawdown limits
   double         m_dailyDDLimit;
   double         m_maxDDLimit;

   // Indicator handles
   int            m_atrHandle;
   int            m_emaFastHandle;
   int            m_emaSlowHandle;
   int            m_ema200Handle;
   int            m_rsiHandle;

   // Buffers
   double         m_atrBuffer[];
   double         m_emaFastBuffer[];
   double         m_emaSlowBuffer[];
   double         m_ema200Buffer[];
   double         m_rsiBuffer[];

   // Position tracking
   GridPosition   m_positions[];
   int            m_positionCount;

   // Account tracking
   double         m_startBalance;
   double         m_highWaterMark;
   double         m_dailyStartBalance;
   datetime       m_lastDayReset;

   // State
   bool           m_blocked;
   string         m_blockReason;
   ENUM_ENTROPY_STATE m_currentEntropy;
   double         m_lastGridPrice;

public:
   // Constructor/Destructor
   CEntropyGridManager(void);
   ~CEntropyGridManager(void);

   // Initialization
   bool           Initialize(string symbol, int magic, double accountId);
   void           SetLotSizes(double baseLot, double maxLot);
   void           SetMaxPositions(int maxPos);
   void           SetDrawdownLimits(double dailyDD, double maxDD);
   void           SetRiskPercent(double riskPct);
   void           SetATRMultipliers(double slMult, double tpMult);
   void           SetPartialTPRatio(double ratio);
   void           SetBreakEvenTrigger(double trigger);
   void           SetTrailStartTrigger(double trigger);
   void           SetCompressionBoost(int boost);
   void           SetConfidenceThreshold(double threshold);
   void           Deinitialize(void);

   // Core Operations
   void           OnTick(void);
   bool           ProcessTick(void);

   // Entropy Calculation
   ENUM_ENTROPY_STATE CalculateEntropy(void);
   double         GetEntropyConfidence(void);
   double         GetATR(void);

   // Grid Operations
   bool           ShouldOpenGrid(int direction);
   bool           OpenGridPosition(int direction, int level);
   void           ManagePositions(void);

   // Hidden SL/TP Management
   void           CheckVirtualSLTP(void);
   void           CheckPartialTP(void);
   void           CheckBreakEven(void);
   void           CheckTrailingStop(void);

   // Risk Management
   bool           CheckDrawdownLimits(void);
   void           CheckDailyReset(void);
   double         CalculateLotSize(double slDistance);

   // Position Utilities
   int            CountPositions(void);
   void           SyncPositions(void);
   bool           ClosePosition(ulong ticket, double volume);
   void           RemovePosition(int index);
   int            FindPosition(ulong ticket);
   double         GetLastEntryPrice(void);

   // Utilities
   double         NormalizeLot(double lot);
   ENUM_ORDER_TYPE_FILLING GetFillingMode(void);
   string         GetStatusString(void);

   // Getters
   string         GetSymbol(void) { return m_symbol; }
   int            GetMagic(void) { return m_magic; }
   bool           IsBlocked(void) { return m_blocked; }
   string         GetBlockReason(void) { return m_blockReason; }
   ENUM_ENTROPY_STATE GetCurrentEntropy(void) { return m_currentEntropy; }
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CEntropyGridManager::CEntropyGridManager(void)
{
   m_symbol = "";
   m_magic = 0;
   m_accountId = 0;
   m_positionCount = 0;
   m_blocked = false;
   m_blockReason = "";
   m_currentEntropy = ENTROPY_HIGH;
   m_lastGridPrice = 0;

   // Hard-coded per specification
   m_confidenceThreshold = 0.22;
   m_compressionBoost = 12;
   m_slAtrMultiplier = 1.5;
   m_tpAtrMultiplier = 3.0;
   m_partialTpRatio = 0.50;
   m_breakEvenTrigger = 0.30;
   m_trailStartTrigger = 0.50;

   // Defaults
   m_maxPositions = 5;
   m_gridSpacingAtr = 0.5;
   m_baseLotSize = 0.01;
   m_maxLotSize = 0.10;
   m_riskPerTradePct = 0.5;
   m_dailyDDLimit = 4.5;
   m_maxDDLimit = 9.0;

   // Handles
   m_atrHandle = INVALID_HANDLE;
   m_emaFastHandle = INVALID_HANDLE;
   m_emaSlowHandle = INVALID_HANDLE;
   m_ema200Handle = INVALID_HANDLE;
   m_rsiHandle = INVALID_HANDLE;

   ArraySetAsSeries(m_atrBuffer, true);
   ArraySetAsSeries(m_emaFastBuffer, true);
   ArraySetAsSeries(m_emaSlowBuffer, true);
   ArraySetAsSeries(m_ema200Buffer, true);
   ArraySetAsSeries(m_rsiBuffer, true);
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CEntropyGridManager::~CEntropyGridManager(void)
{
   Deinitialize();
}

//+------------------------------------------------------------------+
//| Initialize                                                        |
//+------------------------------------------------------------------+
bool CEntropyGridManager::Initialize(string symbol, int magic, double accountId)
{
   m_symbol = symbol;
   m_magic = magic;
   m_accountId = accountId;

   // Create indicator handles on M5 timeframe
   m_atrHandle = iATR(m_symbol, PERIOD_M5, 14);
   m_emaFastHandle = iMA(m_symbol, PERIOD_M5, 8, 0, MODE_EMA, PRICE_CLOSE);
   m_emaSlowHandle = iMA(m_symbol, PERIOD_M5, 21, 0, MODE_EMA, PRICE_CLOSE);
   m_ema200Handle = iMA(m_symbol, PERIOD_M5, 200, 0, MODE_EMA, PRICE_CLOSE);
   m_rsiHandle = iRSI(m_symbol, PERIOD_M5, 14, PRICE_CLOSE);

   if(m_atrHandle == INVALID_HANDLE || m_emaFastHandle == INVALID_HANDLE ||
      m_emaSlowHandle == INVALID_HANDLE || m_ema200Handle == INVALID_HANDLE ||
      m_rsiHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles for ", m_symbol);
      return false;
   }

   // Initialize balance tracking
   m_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   m_highWaterMark = m_startBalance;
   m_dailyStartBalance = m_startBalance;
   m_lastDayReset = TimeCurrent();

   // Sync existing positions
   SyncPositions();

   Print("EntropyGridManager initialized for ", m_symbol, " | Magic: ", m_magic,
         " | Account: ", DoubleToString(m_accountId, 0));
   Print("  Entropy Filter: ENABLED | Confidence: ", m_confidenceThreshold,
         " | Boost: +", m_compressionBoost);
   Print("  ATR Multipliers: SL=", m_slAtrMultiplier, "x | TP=", m_tpAtrMultiplier, "x");
   Print("  Partial TP: ", m_partialTpRatio * 100, "% | BE at ", m_breakEvenTrigger * 100,
         "% | Trail at ", m_trailStartTrigger * 100, "%");

   return true;
}

//+------------------------------------------------------------------+
//| Set Lot Sizes                                                     |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetLotSizes(double baseLot, double maxLot)
{
   m_baseLotSize = baseLot;
   m_maxLotSize = maxLot;
}

//+------------------------------------------------------------------+
//| Set Max Positions                                                 |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetMaxPositions(int maxPos)
{
   m_maxPositions = maxPos;
}

//+------------------------------------------------------------------+
//| Set Drawdown Limits                                               |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetDrawdownLimits(double dailyDD, double maxDD)
{
   m_dailyDDLimit = dailyDD;
   m_maxDDLimit = maxDD;
}

//+------------------------------------------------------------------+
//| Set Risk Percent                                                  |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetRiskPercent(double riskPct)
{
   m_riskPerTradePct = riskPct;
}

//+------------------------------------------------------------------+
//| Set ATR Multipliers (allows parent EA to override defaults)        |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetATRMultipliers(double slMult, double tpMult)
{
   m_slAtrMultiplier = slMult;
   m_tpAtrMultiplier = tpMult;
}

//+------------------------------------------------------------------+
//| Set Partial TP Ratio                                               |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetPartialTPRatio(double ratio)
{
   m_partialTpRatio = ratio;
}

//+------------------------------------------------------------------+
//| Set Break Even Trigger                                             |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetBreakEvenTrigger(double trigger)
{
   m_breakEvenTrigger = trigger;
}

//+------------------------------------------------------------------+
//| Set Trail Start Trigger                                            |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetTrailStartTrigger(double trigger)
{
   m_trailStartTrigger = trigger;
}

//+------------------------------------------------------------------+
//| Set Compression Boost                                              |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetCompressionBoost(int boost)
{
   m_compressionBoost = boost;
}

//+------------------------------------------------------------------+
//| Set Confidence Threshold                                           |
//+------------------------------------------------------------------+
void CEntropyGridManager::SetConfidenceThreshold(double threshold)
{
   m_confidenceThreshold = threshold;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CEntropyGridManager::Deinitialize(void)
{
   if(m_atrHandle != INVALID_HANDLE) IndicatorRelease(m_atrHandle);
   if(m_emaFastHandle != INVALID_HANDLE) IndicatorRelease(m_emaFastHandle);
   if(m_emaSlowHandle != INVALID_HANDLE) IndicatorRelease(m_emaSlowHandle);
   if(m_ema200Handle != INVALID_HANDLE) IndicatorRelease(m_ema200Handle);
   if(m_rsiHandle != INVALID_HANDLE) IndicatorRelease(m_rsiHandle);

   m_atrHandle = INVALID_HANDLE;
   m_emaFastHandle = INVALID_HANDLE;
   m_emaSlowHandle = INVALID_HANDLE;
   m_ema200Handle = INVALID_HANDLE;
   m_rsiHandle = INVALID_HANDLE;
}

//+------------------------------------------------------------------+
//| OnTick - Called every tick                                        |
//+------------------------------------------------------------------+
void CEntropyGridManager::OnTick(void)
{
   // ALWAYS manage positions on every tick for hidden SL/TP
   ManagePositions();
}

//+------------------------------------------------------------------+
//| ProcessTick - Called on interval for new entries                  |
//+------------------------------------------------------------------+
bool CEntropyGridManager::ProcessTick(void)
{
   // Daily reset
   CheckDailyReset();

   // Update high water mark
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(currentBalance > m_highWaterMark) m_highWaterMark = currentBalance;

   // Check drawdown limits
   if(!CheckDrawdownLimits())
   {
      if(!m_blocked)
      {
         m_blocked = true;
         Print(m_symbol, " BLOCKED: ", m_blockReason);
      }
      return false;
   }

   // Calculate entropy
   m_currentEntropy = CalculateEntropy();

   // Only trade in low entropy (predictable) markets
   if(m_currentEntropy == ENTROPY_HIGH)
   {
      return true; // Not an error, just not trading
   }

   // Check position count
   int posCount = CountPositions();
   if(posCount >= m_maxPositions)
   {
      return true;
   }

   // Determine direction based on EMA alignment
   if(CopyBuffer(m_emaFastHandle, 0, 0, 3, m_emaFastBuffer) < 3) return true;
   if(CopyBuffer(m_emaSlowHandle, 0, 0, 3, m_emaSlowBuffer) < 3) return true;
   if(CopyBuffer(m_ema200Handle, 0, 0, 3, m_ema200Buffer) < 3) return true;

   double emaFast = m_emaFastBuffer[1];
   double emaSlow = m_emaSlowBuffer[1];
   double ema200 = m_ema200Buffer[1];
   double price = SymbolInfoDouble(m_symbol, SYMBOL_BID);

   // Bullish bias: Price > EMA200, Fast > Slow
   bool bullish = (price > ema200 && emaFast > emaSlow);
   // Bearish bias: Price < EMA200, Fast < Slow
   bool bearish = (price < ema200 && emaFast < emaSlow);

   int direction = 0;
   if(bullish) direction = 1;      // BUY
   else if(bearish) direction = -1; // SELL
   else return true;                // No clear direction

   // Check grid spacing
   if(posCount > 0)
   {
      double lastEntry = GetLastEntryPrice();
      double atr = GetATR();
      if(atr <= 0) return true;

      double spacing = atr * m_gridSpacingAtr;
      double priceMove = 0;

      if(direction > 0) // BUY - add on dips
      {
         priceMove = lastEntry - price;
      }
      else // SELL - add on rallies
      {
         priceMove = price - lastEntry;
      }

      if(priceMove < spacing) return true;
   }

   // Open new grid position
   if(ShouldOpenGrid(direction))
   {
      OpenGridPosition(direction, posCount + 1);
   }

   return true;
}

//+------------------------------------------------------------------+
//| Calculate Entropy State                                           |
//+------------------------------------------------------------------+
ENUM_ENTROPY_STATE CEntropyGridManager::CalculateEntropy(void)
{
   double confidence = GetEntropyConfidence();

   // Apply compression boost (+12)
   double adjustedThreshold = m_confidenceThreshold - (m_compressionBoost / 100.0);
   adjustedThreshold = MathMax(adjustedThreshold, 0.50); // Don't go below 50%

   if(confidence >= m_confidenceThreshold)
      return ENTROPY_LOW;    // Highly predictable - full trading
   else if(confidence >= adjustedThreshold)
      return ENTROPY_MEDIUM; // Somewhat predictable - reduced size
   else
      return ENTROPY_HIGH;   // Chaotic - no trading
}

//+------------------------------------------------------------------+
//| Get Entropy Confidence Score                                      |
//+------------------------------------------------------------------+
double CEntropyGridManager::GetEntropyConfidence(void)
{
   if(CopyBuffer(m_emaFastHandle, 0, 0, 10, m_emaFastBuffer) < 10) return 0;
   if(CopyBuffer(m_emaSlowHandle, 0, 0, 10, m_emaSlowBuffer) < 10) return 0;
   if(CopyBuffer(m_ema200Handle, 0, 0, 10, m_ema200Buffer) < 10) return 0;
   if(CopyBuffer(m_rsiHandle, 0, 0, 10, m_rsiBuffer) < 10) return 0;
   if(CopyBuffer(m_atrHandle, 0, 0, 10, m_atrBuffer) < 10) return 0;

   double confidence = 0.0;

   // 1. EMA Alignment (30% weight)
   // All EMAs aligned = low entropy
   double emaFast = m_emaFastBuffer[1];
   double emaSlow = m_emaSlowBuffer[1];
   double ema200 = m_ema200Buffer[1];
   double price = SymbolInfoDouble(m_symbol, SYMBOL_BID);

   bool allAbove = (price > ema200 && emaFast > emaSlow && emaSlow > ema200);
   bool allBelow = (price < ema200 && emaFast < emaSlow && emaSlow < ema200);

   if(allAbove || allBelow)
      confidence += 0.30;
   else if((emaFast > emaSlow) || (emaFast < emaSlow))
      confidence += 0.15;

   // 2. EMA Separation Consistency (20% weight)
   // Consistent separation = trending market = lower entropy
   double sep1 = MathAbs(m_emaFastBuffer[1] - m_emaSlowBuffer[1]);
   double sep2 = MathAbs(m_emaFastBuffer[5] - m_emaSlowBuffer[5]);
   double sepChange = MathAbs(sep1 - sep2) / (sep2 + 1e-10);

   if(sepChange < 0.10)
      confidence += 0.20; // Very consistent
   else if(sepChange < 0.20)
      confidence += 0.10; // Somewhat consistent

   // 3. RSI Range (25% weight)
   // RSI in 30-70 range = ranging/predictable
   // RSI extreme = trending/less predictable for mean reversion
   double rsi = m_rsiBuffer[1];
   if(rsi >= 40 && rsi <= 60)
      confidence += 0.25; // Balanced
   else if(rsi >= 30 && rsi <= 70)
      confidence += 0.15; // Normal range
   else
      confidence += 0.05; // Extreme

   // 4. ATR Stability (25% weight)
   // Stable ATR = predictable volatility
   double atr1 = m_atrBuffer[1];
   double atr5 = m_atrBuffer[5];
   double atrChange = MathAbs(atr1 - atr5) / (atr5 + 1e-10);

   if(atrChange < 0.15)
      confidence += 0.25;
   else if(atrChange < 0.30)
      confidence += 0.15;
   else
      confidence += 0.05;

   // Apply compression boost
   confidence += (m_compressionBoost / 100.0);

   return MathMin(confidence, 1.0);
}

//+------------------------------------------------------------------+
//| Get ATR Value                                                     |
//+------------------------------------------------------------------+
double CEntropyGridManager::GetATR(void)
{
   if(CopyBuffer(m_atrHandle, 0, 0, 3, m_atrBuffer) < 3) return 0;
   return m_atrBuffer[1];
}

//+------------------------------------------------------------------+
//| Should Open Grid                                                  |
//+------------------------------------------------------------------+
bool CEntropyGridManager::ShouldOpenGrid(int direction)
{
   // Check entropy
   if(m_currentEntropy == ENTROPY_HIGH) return false;

   // Check position limit
   if(CountPositions() >= m_maxPositions) return false;

   // Check drawdown
   if(m_blocked) return false;

   return true;
}

//+------------------------------------------------------------------+
//| Open Grid Position                                                |
//+------------------------------------------------------------------+
bool CEntropyGridManager::OpenGridPosition(int direction, int level)
{
   if(!ShouldOpenGrid(direction)) return false;

   double atr = GetATR();
   if(atr <= 0) return false;

   // Calculate SL/TP distances based on ATR
   double slDistance = atr * m_slAtrMultiplier;
   double tpDistance = atr * m_tpAtrMultiplier;
   double partialTpDistance = tpDistance * m_partialTpRatio;

   // Calculate lot size
   double lot = CalculateLotSize(slDistance);

   // Reduce lot size for medium entropy
   if(m_currentEntropy == ENTROPY_MEDIUM)
   {
      lot = lot * 0.5;
   }

   lot = NormalizeLot(lot);
   if(lot <= 0) return false;

   // Get current price
   MqlTick tick;
   if(!SymbolInfoTick(m_symbol, tick)) return false;

   double price;
   ENUM_ORDER_TYPE orderType;
   double virtualSL, virtualTP, partialTP;

   if(direction > 0) // BUY
   {
      orderType = ORDER_TYPE_BUY;
      price = tick.ask;
      virtualSL = price - slDistance;
      virtualTP = price + tpDistance;
      partialTP = price + partialTpDistance;
   }
   else // SELL
   {
      orderType = ORDER_TYPE_SELL;
      price = tick.bid;
      virtualSL = price + slDistance;
      virtualTP = price - tpDistance;
      partialTP = price - partialTpDistance;
   }

   // Build order request - HIDDEN SL/TP (no SL/TP sent to broker)
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = m_symbol;
   request.volume = lot;
   request.type = orderType;
   request.price = price;
   request.sl = 0;  // Hidden
   request.tp = 0;  // Hidden
   request.deviation = 50;
   request.magic = m_magic;
   request.comment = StringFormat("ENTROPY_L%d", level);
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = GetFillingMode();

   if(!OrderSend(request, result))
   {
      Print(m_symbol, " Order failed: ", GetLastError());
      return false;
   }

   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print(m_symbol, " Order rejected: ", result.comment, " (", result.retcode, ")");
      return false;
   }

   // Track position with virtual levels
   int idx = m_positionCount;
   ArrayResize(m_positions, m_positionCount + 1);

   m_positions[idx].ticket = result.order;
   m_positions[idx].entryPrice = price;
   m_positions[idx].virtualSL = virtualSL;
   m_positions[idx].virtualTP = virtualTP;
   m_positions[idx].partialTP = partialTP;
   m_positions[idx].volume = lot;
   m_positions[idx].partialVolume = lot * m_partialTpRatio;
   m_positions[idx].partialClosed = false;
   m_positions[idx].breakEvenSet = false;
   m_positions[idx].trailingActive = false;
   m_positions[idx].openTime = TimeCurrent();
   m_positions[idx].gridLevel = level;

   m_positionCount++;
   m_lastGridPrice = price;

   string typeStr = (direction > 0) ? "BUY" : "SELL";
   string entropyStr = (m_currentEntropy == ENTROPY_LOW) ? "LOW" : "MEDIUM";
   Print(m_symbol, " ", typeStr, " L", level, " @ ", DoubleToString(price, 2),
         " | Lot: ", DoubleToString(lot, 4),
         " | Hidden SL: ", DoubleToString(virtualSL, 2),
         " | Hidden TP: ", DoubleToString(virtualTP, 2),
         " | Entropy: ", entropyStr);

   return true;
}

//+------------------------------------------------------------------+
//| Manage Positions                                                  |
//+------------------------------------------------------------------+
void CEntropyGridManager::ManagePositions(void)
{
   SyncPositions();

   CheckVirtualSLTP();
   CheckPartialTP();
   CheckBreakEven();
   CheckTrailingStop();
}

//+------------------------------------------------------------------+
//| Check Virtual SL/TP                                               |
//+------------------------------------------------------------------+
void CEntropyGridManager::CheckVirtualSLTP(void)
{
   for(int i = m_positionCount - 1; i >= 0; i--)
   {
      ulong ticket = m_positions[i].ticket;

      if(!PositionSelectByTicket(ticket))
      {
         RemovePosition(i);
         continue;
      }

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double volume = PositionGetDouble(POSITION_VOLUME);

      bool hitSL = false;
      bool hitTP = false;

      if(posType == POSITION_TYPE_BUY)
      {
         hitSL = (currentPrice <= m_positions[i].virtualSL);
         hitTP = (currentPrice >= m_positions[i].virtualTP);
      }
      else
      {
         hitSL = (currentPrice >= m_positions[i].virtualSL);
         hitTP = (currentPrice <= m_positions[i].virtualTP);
      }

      if(hitSL)
      {
         Print(m_symbol, " HIDDEN SL HIT - Closing L", m_positions[i].gridLevel);
         ClosePosition(ticket, volume);
         RemovePosition(i);
      }
      else if(hitTP)
      {
         Print(m_symbol, " HIDDEN TP HIT - Closing L", m_positions[i].gridLevel);
         ClosePosition(ticket, volume);
         RemovePosition(i);
      }
   }
}

//+------------------------------------------------------------------+
//| Check Partial Take Profit (50% at first target)                   |
//+------------------------------------------------------------------+
void CEntropyGridManager::CheckPartialTP(void)
{
   for(int i = 0; i < m_positionCount; i++)
   {
      if(m_positions[i].partialClosed) continue;

      ulong ticket = m_positions[i].ticket;
      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      bool hitPartial = false;

      if(posType == POSITION_TYPE_BUY)
      {
         hitPartial = (currentPrice >= m_positions[i].partialTP);
      }
      else
      {
         hitPartial = (currentPrice <= m_positions[i].partialTP);
      }

      if(hitPartial)
      {
         double closeVolume = m_positions[i].partialVolume;
         closeVolume = NormalizeLot(closeVolume);

         double remainingVolume = PositionGetDouble(POSITION_VOLUME);
         if(closeVolume >= remainingVolume)
         {
            closeVolume = remainingVolume * 0.5;
            closeVolume = NormalizeLot(closeVolume);
         }

         if(closeVolume > 0 && ClosePosition(ticket, closeVolume))
         {
            m_positions[i].partialClosed = true;
            m_positions[i].volume -= closeVolume;
            Print(m_symbol, " PARTIAL TP: Closed ", DoubleToString(closeVolume, 4),
                  " lots from L", m_positions[i].gridLevel);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Break Even (at 30% of TP)                                   |
//+------------------------------------------------------------------+
void CEntropyGridManager::CheckBreakEven(void)
{
   for(int i = 0; i < m_positionCount; i++)
   {
      if(m_positions[i].breakEvenSet) continue;

      ulong ticket = m_positions[i].ticket;
      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double entryPrice = m_positions[i].entryPrice;
      double tpDistance = MathAbs(m_positions[i].virtualTP - entryPrice);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      double profitDist = (posType == POSITION_TYPE_BUY) ?
                          (currentPrice - entryPrice) :
                          (entryPrice - currentPrice);

      double progress = profitDist / tpDistance;

      if(progress >= m_breakEvenTrigger && profitDist > 0)
      {
         double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
         double buffer = 10 * point;

         if(posType == POSITION_TYPE_BUY)
         {
            m_positions[i].virtualSL = entryPrice + buffer;
         }
         else
         {
            m_positions[i].virtualSL = entryPrice - buffer;
         }

         m_positions[i].breakEvenSet = true;
         Print(m_symbol, " BREAKEVEN: L", m_positions[i].gridLevel,
               " SL moved to ", DoubleToString(m_positions[i].virtualSL, 2));
      }
   }
}

//+------------------------------------------------------------------+
//| Check Trailing Stop (at 50% of TP)                                |
//+------------------------------------------------------------------+
void CEntropyGridManager::CheckTrailingStop(void)
{
   double atr = GetATR();
   if(atr <= 0) return;

   double trailDistance = atr * 1.0; // Trail at 1 ATR

   for(int i = 0; i < m_positionCount; i++)
   {
      if(!m_positions[i].breakEvenSet) continue;

      ulong ticket = m_positions[i].ticket;
      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double entryPrice = m_positions[i].entryPrice;
      double tpDistance = MathAbs(m_positions[i].virtualTP - entryPrice);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      double profitDist = (posType == POSITION_TYPE_BUY) ?
                          (currentPrice - entryPrice) :
                          (entryPrice - currentPrice);

      double progress = profitDist / tpDistance;

      if(progress >= m_trailStartTrigger)
      {
         m_positions[i].trailingActive = true;

         if(posType == POSITION_TYPE_BUY)
         {
            double newSL = currentPrice - trailDistance;
            if(newSL > m_positions[i].virtualSL)
            {
               m_positions[i].virtualSL = newSL;
               Print(m_symbol, " TRAILING: L", m_positions[i].gridLevel,
                     " SL trailed to ", DoubleToString(newSL, 2));
            }
         }
         else
         {
            double newSL = currentPrice + trailDistance;
            if(newSL < m_positions[i].virtualSL)
            {
               m_positions[i].virtualSL = newSL;
               Print(m_symbol, " TRAILING: L", m_positions[i].gridLevel,
                     " SL trailed to ", DoubleToString(newSL, 2));
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Drawdown Limits                                             |
//+------------------------------------------------------------------+
bool CEntropyGridManager::CheckDrawdownLimits(void)
{
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(currentBalance, equity);

   if(m_dailyStartBalance <= 0 || m_highWaterMark <= 0) return true;

   // Daily DD
   double dailyDD = ((m_dailyStartBalance - current) / m_dailyStartBalance) * 100;
   if(dailyDD >= m_dailyDDLimit)
   {
      m_blockReason = StringFormat("Daily DD %.2f%% >= %.2f%%", dailyDD, m_dailyDDLimit);
      return false;
   }

   // Max DD
   double maxDD = ((m_highWaterMark - current) / m_highWaterMark) * 100;
   if(maxDD >= m_maxDDLimit)
   {
      m_blockReason = StringFormat("Max DD %.2f%% >= %.2f%%", maxDD, m_maxDDLimit);
      return false;
   }

   m_blocked = false;
   m_blockReason = "";
   return true;
}

//+------------------------------------------------------------------+
//| Check Daily Reset                                                 |
//+------------------------------------------------------------------+
void CEntropyGridManager::CheckDailyReset(void)
{
   MqlDateTime now, last;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(m_lastDayReset, last);

   if(now.day != last.day || now.mon != last.mon || now.year != last.year)
   {
      m_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      m_lastDayReset = TimeCurrent();
      m_blocked = false;
      m_blockReason = "";
      Print(m_symbol, " Daily reset. New baseline: $", DoubleToString(m_dailyStartBalance, 2));
   }
}

//+------------------------------------------------------------------+
//| Calculate Lot Size                                                |
//+------------------------------------------------------------------+
double CEntropyGridManager::CalculateLotSize(double slDistance)
{
   if(slDistance <= 0) return m_baseLotSize;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (m_riskPerTradePct / 100.0);

   double tickValue = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tickValue <= 0 || tickSize <= 0) return m_baseLotSize;

   double lots = riskAmount / (slDistance / tickSize * tickValue);

   lots = MathMax(lots, m_baseLotSize);
   lots = MathMin(lots, m_maxLotSize);

   return lots;
}

//+------------------------------------------------------------------+
//| Count Positions                                                   |
//+------------------------------------------------------------------+
int CEntropyGridManager::CountPositions(void)
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
      count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Sync Positions                                                    |
//+------------------------------------------------------------------+
void CEntropyGridManager::SyncPositions(void)
{
   // Remove closed positions
   for(int i = m_positionCount - 1; i >= 0; i--)
   {
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         ulong ticket = PositionGetTicket(p);
         if(ticket == m_positions[i].ticket)
         {
            found = true;
            break;
         }
      }
      if(!found) RemovePosition(i);
   }

   // Add untracked positions
   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      ulong ticket = PositionGetTicket(p);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;

      if(FindPosition(ticket) < 0)
      {
         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double atr = GetATR();
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         int idx = m_positionCount;
         ArrayResize(m_positions, m_positionCount + 1);

         m_positions[idx].ticket = ticket;
         m_positions[idx].entryPrice = entry;
         m_positions[idx].volume = PositionGetDouble(POSITION_VOLUME);
         m_positions[idx].partialVolume = m_positions[idx].volume * m_partialTpRatio;
         m_positions[idx].partialClosed = false;
         m_positions[idx].breakEvenSet = false;
         m_positions[idx].trailingActive = false;
         m_positions[idx].openTime = (datetime)PositionGetInteger(POSITION_TIME);
         m_positions[idx].gridLevel = idx + 1;

         if(posType == POSITION_TYPE_BUY)
         {
            m_positions[idx].virtualSL = entry - (atr * m_slAtrMultiplier);
            m_positions[idx].virtualTP = entry + (atr * m_tpAtrMultiplier);
            m_positions[idx].partialTP = entry + (atr * m_tpAtrMultiplier * m_partialTpRatio);
         }
         else
         {
            m_positions[idx].virtualSL = entry + (atr * m_slAtrMultiplier);
            m_positions[idx].virtualTP = entry - (atr * m_tpAtrMultiplier);
            m_positions[idx].partialTP = entry - (atr * m_tpAtrMultiplier * m_partialTpRatio);
         }

         m_positionCount++;
         Print(m_symbol, " Synced position: ", ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Close Position                                                    |
//+------------------------------------------------------------------+
bool CEntropyGridManager::ClosePosition(ulong ticket, double volume)
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

   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = symbol;
   request.volume = volume;
   request.deviation = 50;
   request.magic = m_magic;
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
      Print(m_symbol, " Close failed: ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Remove Position from Tracking                                     |
//+------------------------------------------------------------------+
void CEntropyGridManager::RemovePosition(int index)
{
   if(index < 0 || index >= m_positionCount) return;

   for(int i = index; i < m_positionCount - 1; i++)
   {
      m_positions[i] = m_positions[i + 1];
   }

   m_positionCount--;
   ArrayResize(m_positions, m_positionCount);
}

//+------------------------------------------------------------------+
//| Find Position Index                                               |
//+------------------------------------------------------------------+
int CEntropyGridManager::FindPosition(ulong ticket)
{
   for(int i = 0; i < m_positionCount; i++)
   {
      if(m_positions[i].ticket == ticket) return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Get Last Entry Price                                              |
//+------------------------------------------------------------------+
double CEntropyGridManager::GetLastEntryPrice(void)
{
   double lastPrice = 0;
   datetime lastTime = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;

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
//| Normalize Lot Size                                                |
//+------------------------------------------------------------------+
double CEntropyGridManager::NormalizeLot(double lot)
{
   double minLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);

   lot = MathMax(lot, minLot);
   lot = MathMin(lot, maxLot);
   lot = MathFloor(lot / lotStep) * lotStep;

   return NormalizeDouble(lot, 8);
}

//+------------------------------------------------------------------+
//| Get Filling Mode                                                  |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING CEntropyGridManager::GetFillingMode(void)
{
   uint filling = (uint)SymbolInfoInteger(m_symbol, SYMBOL_FILLING_MODE);

   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;

   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;

   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Get Status String                                                 |
//+------------------------------------------------------------------+
string CEntropyGridManager::GetStatusString(void)
{
   string entropyStr = "";
   switch(m_currentEntropy)
   {
      case ENTROPY_LOW: entropyStr = "LOW"; break;
      case ENTROPY_MEDIUM: entropyStr = "MEDIUM"; break;
      case ENTROPY_HIGH: entropyStr = "HIGH"; break;
   }

   return StringFormat("%s | Pos: %d/%d | Entropy: %s | Blocked: %s",
                       m_symbol, CountPositions(), m_maxPositions,
                       entropyStr, m_blocked ? "YES" : "NO");
}
//+------------------------------------------------------------------+
