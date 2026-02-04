//+------------------------------------------------------------------+
//|                                              XAUUSD_GridCore.mqh |
//|                        XAUUSD Grid Trading System - Core Logic   |
//|                    GetLeveraged Multi-Account Edition            |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library - DooDoo Edition"
#property version   "1.00"

//+------------------------------------------------------------------+
//| HARD-CODED ATR MULTIPLIERS (AS REQUIRED)                         |
//+------------------------------------------------------------------+
#define ATR_SL_MULTIPLIER       1.5     // Stop Loss = 1.5x ATR
#define ATR_TP_MULTIPLIER       3.0     // Take Profit = 3.0x ATR
#define ATR_PERIOD              14      // Standard ATR period
#define PARTIAL_TP_PERCENT      50      // 50% of position at first TP
#define TRAILING_ACTIVATION     0.50    // Activate trailing at 50% of TP
#define BREAKEVEN_ACTIVATION    0.30    // Move to BE at 30% of TP
#define COMPRESSION_BOOST       12      // +12 compression confidence boost

//+------------------------------------------------------------------+
//| Regime Enumeration                                               |
//+------------------------------------------------------------------+
enum ENUM_XAUUSD_REGIME
{
   XAUUSD_REGIME_BEARISH = -1,
   XAUUSD_REGIME_NEUTRAL = 0,
   XAUUSD_REGIME_BULLISH = 1
};

//+------------------------------------------------------------------+
//| LLM Adjustment Structure (from companion script)                 |
//+------------------------------------------------------------------+
struct LLMAdjustment
{
   double   slMultiplierAdj;    // Adjustment to SL multiplier (-0.5 to +0.5)
   double   tpMultiplierAdj;    // Adjustment to TP multiplier (-1.0 to +1.0)
   double   confidenceBoost;    // Additional confidence boost
   bool     tightenSLTP;        // True = tighter, False = wider
   string   volatilityRegime;   // "LOW", "NORMAL", "HIGH", "EXTREME"
   double   llmConfidence;      // LLM's confidence in its adjustment
   datetime lastUpdate;         // When LLM last updated
};

//+------------------------------------------------------------------+
//| Grid Order Structure (Hidden SL/TP)                              |
//+------------------------------------------------------------------+
struct XAUGridOrder
{
   ulong    ticket;
   double   entryPrice;
   double   virtualSL;          // Hidden SL - managed internally
   double   virtualTP;          // Hidden TP - managed internally
   double   partialTP;          // First target for partial close
   double   volume;
   double   partialVolume;      // Volume to close at partial TP
   bool     partialClosed;      // Has partial been taken?
   bool     breakEvenSet;       // Has BE been triggered?
   bool     trailingActive;     // Is trailing stop active?
   datetime openTime;
   int      gridLevel;
   double   atrAtEntry;         // ATR at time of entry
   double   actualSLMult;       // Actual SL multiplier used (after LLM adj)
   double   actualTPMult;       // Actual TP multiplier used (after LLM adj)
};

//+------------------------------------------------------------------+
//| Grid Configuration Structure                                     |
//+------------------------------------------------------------------+
struct XAUGridConfig
{
   // Risk Management
   double   riskPerGridPct;         // Risk per grid level
   bool     dynamicHiddenSLTP;      // Hidden orders, managed internally
   bool     trailingStopEnabled;
   bool     breakEvenEnabled;

   // Grid Settings
   int      maxOrdersPerExpert;     // Max orders per expert
   int      maxTotalOrders;         // Max total orders
   double   gridSpacingAtr;         // Grid spacing in ATR multiples

   // Compression/Entropy Filter
   double   confidenceThreshold;    // Base confidence threshold
   bool     useEntropyFilter;       // Only trade when market is predictable
   double   entropyThreshold;       // Max entropy allowed

   // Account
   double   accountBalance;
   double   dailyDDLimit;
   double   maxDDLimit;

   // LLM Integration
   bool     useLLMAdjustment;       // Use LLM for dynamic SL/TP
   string   llmSignalFile;          // Path to LLM signal file
};

//+------------------------------------------------------------------+
//| XAUUSD Grid Manager Class                                        |
//+------------------------------------------------------------------+
class CXAUGridManager
{
private:
   XAUGridConfig    m_config;
   XAUGridOrder     m_orders[];
   int              m_orderCount;
   int              m_magicNumber;
   string           m_symbol;
   ENUM_XAUUSD_REGIME m_regime;
   LLMAdjustment    m_llmAdj;

   // Tracking
   double           m_startBalance;
   double           m_highWaterMark;
   double           m_dailyStartBalance;
   datetime         m_lastDayReset;
   datetime         m_lastLLMRead;

   // Indicator Handles
   int              m_atrHandle;
   double           m_atrBuffer[];

   // EMA Handles for regime detection
   int              m_emaFastHandle;
   int              m_emaSlowHandle;
   int              m_ema200Handle;
   double           m_emaFastBuffer[];
   double           m_emaSlowBuffer[];
   double           m_ema200Buffer[];

   // Entropy calculation
   double           m_entropyBuffer[];

public:
   //--- Constructor/Destructor
   CXAUGridManager(void);
   ~CXAUGridManager(void);

   //--- Initialization
   bool     Initialize(int magic, ENUM_XAUUSD_REGIME regime);
   void     SetConfig(XAUGridConfig &config);
   void     Deinitialize(void);

   //--- Core Operations
   bool     ProcessTick(void);
   double   CalculateRegimeConfidence(void);
   double   CalculateEntropy(void);
   bool     ShouldOpenGrid(int direction);
   bool     OpenGridOrder(int direction, int gridLevel);
   void     ManageOpenPositions(void);

   //--- Hidden SL/TP Management
   void     CheckVirtualSLTP(void);
   void     CheckPartialTakeProfit(void);
   void     CheckBreakEven(void);
   void     CheckTrailingStop(void);

   //--- LLM Integration
   bool     ReadLLMAdjustment(void);
   double   GetAdjustedSLMultiplier(void);
   double   GetAdjustedTPMultiplier(void);

   //--- Helpers
   double   GetATR(void);
   double   CalculateLotSize(double slDistance);
   bool     IsWithinRiskLimits(void);
   int      CountOpenOrders(void);
   bool     CheckDailyDrawdown(void);
   bool     CheckMaxDrawdown(void);
   void     DailyReset(void);

   //--- Order Management
   bool     ClosePosition(ulong ticket, double volume);
   bool     ModifyPosition(ulong ticket, double sl, double tp);
   int      FindGridOrder(ulong ticket);
   void     RemoveGridOrder(int index);
   void     SyncGridOrders(void);

   //--- Getters
   int      GetMagicNumber(void) { return m_magicNumber; }
   string   GetSymbol(void) { return m_symbol; }
   ENUM_XAUUSD_REGIME GetRegime(void) { return m_regime; }
   LLMAdjustment GetLLMAdjustment(void) { return m_llmAdj; }
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CXAUGridManager::CXAUGridManager(void)
{
   m_orderCount = 0;
   m_magicNumber = 0;
   m_symbol = "XAUUSD";
   m_regime = XAUUSD_REGIME_NEUTRAL;
   m_atrHandle = INVALID_HANDLE;
   m_emaFastHandle = INVALID_HANDLE;
   m_emaSlowHandle = INVALID_HANDLE;
   m_ema200Handle = INVALID_HANDLE;
   m_lastLLMRead = 0;

   // Initialize LLM adjustment to neutral
   m_llmAdj.slMultiplierAdj = 0;
   m_llmAdj.tpMultiplierAdj = 0;
   m_llmAdj.confidenceBoost = 0;
   m_llmAdj.tightenSLTP = false;
   m_llmAdj.volatilityRegime = "NORMAL";
   m_llmAdj.llmConfidence = 0;
   m_llmAdj.lastUpdate = 0;

   ArraySetAsSeries(m_atrBuffer, true);
   ArraySetAsSeries(m_emaFastBuffer, true);
   ArraySetAsSeries(m_emaSlowBuffer, true);
   ArraySetAsSeries(m_ema200Buffer, true);
   ArraySetAsSeries(m_entropyBuffer, true);
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CXAUGridManager::~CXAUGridManager(void)
{
   Deinitialize();
}

//+------------------------------------------------------------------+
//| Initialize the XAUUSD Grid Manager                               |
//+------------------------------------------------------------------+
bool CXAUGridManager::Initialize(int magic, ENUM_XAUUSD_REGIME regime)
{
   m_magicNumber = magic;
   m_regime = regime;

   // Ensure XAUUSD is selected
   if(!SymbolSelect(m_symbol, true))
   {
      Print("XAUGridManager: Failed to select ", m_symbol);
      return false;
   }

   // Create indicator handles for M5 timeframe
   m_atrHandle = iATR(m_symbol, PERIOD_M5, ATR_PERIOD);
   m_emaFastHandle = iMA(m_symbol, PERIOD_M5, 8, 0, MODE_EMA, PRICE_CLOSE);
   m_emaSlowHandle = iMA(m_symbol, PERIOD_M5, 21, 0, MODE_EMA, PRICE_CLOSE);
   m_ema200Handle = iMA(m_symbol, PERIOD_M5, 200, 0, MODE_EMA, PRICE_CLOSE);

   if(m_atrHandle == INVALID_HANDLE || m_emaFastHandle == INVALID_HANDLE ||
      m_emaSlowHandle == INVALID_HANDLE || m_ema200Handle == INVALID_HANDLE)
   {
      Print("XAUGridManager: Failed to create indicator handles");
      return false;
   }

   // Initialize tracking
   m_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   m_highWaterMark = m_startBalance;
   m_dailyStartBalance = m_startBalance;
   m_lastDayReset = TimeCurrent();

   // Sync existing orders
   SyncGridOrders();

   Print("========================================");
   Print("XAUUSD GRID MANAGER INITIALIZED");
   Print("Magic: ", m_magicNumber);
   Print("Regime: ", EnumToString(m_regime));
   Print("ATR SL Multiplier: ", ATR_SL_MULTIPLIER, "x (HARD-CODED)");
   Print("ATR TP Multiplier: ", ATR_TP_MULTIPLIER, "x (HARD-CODED)");
   Print("ATR Period: ", ATR_PERIOD);
   Print("Partial TP: ", PARTIAL_TP_PERCENT, "% at first target");
   Print("Trailing Activation: ", TRAILING_ACTIVATION * 100, "% of TP");
   Print("Break-Even Activation: ", BREAKEVEN_ACTIVATION * 100, "% of TP");
   Print("Compression Boost: +", COMPRESSION_BOOST);
   Print("========================================");

   return true;
}

//+------------------------------------------------------------------+
//| Set Configuration                                                 |
//+------------------------------------------------------------------+
void CXAUGridManager::SetConfig(XAUGridConfig &config)
{
   m_config = config;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CXAUGridManager::Deinitialize(void)
{
   if(m_atrHandle != INVALID_HANDLE) IndicatorRelease(m_atrHandle);
   if(m_emaFastHandle != INVALID_HANDLE) IndicatorRelease(m_emaFastHandle);
   if(m_emaSlowHandle != INVALID_HANDLE) IndicatorRelease(m_emaSlowHandle);
   if(m_ema200Handle != INVALID_HANDLE) IndicatorRelease(m_ema200Handle);

   m_atrHandle = INVALID_HANDLE;
   m_emaFastHandle = INVALID_HANDLE;
   m_emaSlowHandle = INVALID_HANDLE;
   m_ema200Handle = INVALID_HANDLE;
}

//+------------------------------------------------------------------+
//| Process Tick - Main entry point                                   |
//+------------------------------------------------------------------+
bool CXAUGridManager::ProcessTick(void)
{
   // Daily reset check
   MqlDateTime now, last;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(m_lastDayReset, last);

   if(now.day != last.day || now.mon != last.mon || now.year != last.year)
   {
      DailyReset();
   }

   // Update high water mark
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(currentBalance > m_highWaterMark)
   {
      m_highWaterMark = currentBalance;
   }

   // Check risk limits
   if(!IsWithinRiskLimits())
   {
      return false;
   }

   // Read LLM adjustments if enabled
   if(m_config.useLLMAdjustment)
   {
      // Read every 30 seconds max
      if(TimeCurrent() - m_lastLLMRead >= 30)
      {
         ReadLLMAdjustment();
         m_lastLLMRead = TimeCurrent();
      }
   }

   // Manage open positions (hidden SL/TP, trailing, BE)
   ManageOpenPositions();

   // Check entropy filter - only trade when market is predictable
   if(m_config.useEntropyFilter)
   {
      double entropy = CalculateEntropy();
      if(entropy > m_config.entropyThreshold)
      {
         // Market is too chaotic - skip trading
         return true;
      }
   }

   // Check regime confidence with compression boost
   double confidence = CalculateRegimeConfidence();
   double adjustedThreshold = m_config.confidenceThreshold;

   // Apply compression boost
   confidence += (COMPRESSION_BOOST / 100.0);

   // Apply LLM confidence boost if available
   if(m_config.useLLMAdjustment && m_llmAdj.llmConfidence > 0.5)
   {
      confidence += m_llmAdj.confidenceBoost;
   }

   if(confidence < adjustedThreshold)
   {
      return true; // Not confident enough, but not an error
   }

   // Check if we can open new grid orders
   int orderCount = CountOpenOrders();
   if(orderCount >= m_config.maxOrdersPerExpert)
   {
      return true; // At max capacity
   }

   return true;
}

//+------------------------------------------------------------------+
//| Calculate Entropy (Market Predictability)                        |
//+------------------------------------------------------------------+
double CXAUGridManager::CalculateEntropy(void)
{
   // Get recent price data for entropy calculation
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(m_symbol, PERIOD_M5, 0, 50, rates);
   if(copied < 50) return 999.0; // Return high entropy if insufficient data

   // Calculate Shannon entropy of price changes
   double changes[];
   ArrayResize(changes, copied - 1);

   for(int i = 0; i < copied - 1; i++)
   {
      changes[i] = (rates[i].close - rates[i+1].close) / rates[i+1].close;
   }

   // Bin the changes into categories
   int bins[5] = {0, 0, 0, 0, 0}; // Strong down, down, flat, up, strong up

   for(int i = 0; i < ArraySize(changes); i++)
   {
      if(changes[i] < -0.002) bins[0]++;
      else if(changes[i] < -0.0005) bins[1]++;
      else if(changes[i] < 0.0005) bins[2]++;
      else if(changes[i] < 0.002) bins[3]++;
      else bins[4]++;
   }

   // Calculate entropy
   double entropy = 0;
   int total = ArraySize(changes);

   for(int i = 0; i < 5; i++)
   {
      if(bins[i] > 0)
      {
         double p = (double)bins[i] / total;
         entropy -= p * MathLog(p) / MathLog(2.0);
      }
   }

   return entropy; // Max entropy ~2.32 for 5 bins
}

//+------------------------------------------------------------------+
//| Calculate Regime Confidence                                       |
//+------------------------------------------------------------------+
double CXAUGridManager::CalculateRegimeConfidence(void)
{
   if(CopyBuffer(m_emaFastHandle, 0, 0, 5, m_emaFastBuffer) < 5) return 0;
   if(CopyBuffer(m_emaSlowHandle, 0, 0, 5, m_emaSlowBuffer) < 5) return 0;
   if(CopyBuffer(m_ema200Handle, 0, 0, 5, m_ema200Buffer) < 5) return 0;

   double emaFast = m_emaFastBuffer[1];
   double emaSlow = m_emaSlowBuffer[1];
   double ema200 = m_ema200Buffer[1];
   double price = SymbolInfoDouble(m_symbol, SYMBOL_BID);

   double confidence = 0.0;

   switch(m_regime)
   {
      case XAUUSD_REGIME_BULLISH:
         // Bullish: Price > EMA200, Fast > Slow, both above 200
         if(price > ema200) confidence += 0.35;
         if(emaFast > emaSlow) confidence += 0.35;
         if(emaSlow > ema200) confidence += 0.30;
         break;

      case XAUUSD_REGIME_BEARISH:
         // Bearish: Price < EMA200, Fast < Slow, both below 200
         if(price < ema200) confidence += 0.35;
         if(emaFast < emaSlow) confidence += 0.35;
         if(emaSlow < ema200) confidence += 0.30;
         break;

      case XAUUSD_REGIME_NEUTRAL:
         // Neutral: Price near EMA200, EMAs converging
         double distFromEma200 = MathAbs(price - ema200) / ema200;
         double emaDiff = MathAbs(emaFast - emaSlow) / emaSlow;

         // Less distance = more neutral
         if(distFromEma200 < 0.005) confidence += 0.40;
         else if(distFromEma200 < 0.01) confidence += 0.25;
         else confidence += 0.10;

         // Converging EMAs = more neutral
         if(emaDiff < 0.003) confidence += 0.40;
         else if(emaDiff < 0.006) confidence += 0.25;
         else confidence += 0.10;
         break;
   }

   return MathMin(confidence, 1.0);
}

//+------------------------------------------------------------------+
//| Read LLM Adjustment from File                                     |
//+------------------------------------------------------------------+
bool CXAUGridManager::ReadLLMAdjustment(void)
{
   if(m_config.llmSignalFile == "" || m_config.llmSignalFile == NULL)
      return false;

   int fileHandle = FileOpen(m_config.llmSignalFile, FILE_READ|FILE_TXT|FILE_COMMON);
   if(fileHandle == INVALID_HANDLE)
   {
      return false;
   }

   // Read the JSON-like content
   string content = "";
   while(!FileIsEnding(fileHandle))
   {
      content += FileReadString(fileHandle) + "\n";
   }
   FileClose(fileHandle);

   // Parse simple key-value pairs
   // Format expected:
   // sl_adj=0.2
   // tp_adj=0.5
   // confidence_boost=0.05
   // volatility_regime=HIGH
   // llm_confidence=0.85

   if(StringFind(content, "sl_adj=") >= 0)
   {
      int pos = StringFind(content, "sl_adj=") + 7;
      int endPos = StringFind(content, "\n", pos);
      string val = StringSubstr(content, pos, endPos - pos);
      m_llmAdj.slMultiplierAdj = StringToDouble(val);
   }

   if(StringFind(content, "tp_adj=") >= 0)
   {
      int pos = StringFind(content, "tp_adj=") + 7;
      int endPos = StringFind(content, "\n", pos);
      string val = StringSubstr(content, pos, endPos - pos);
      m_llmAdj.tpMultiplierAdj = StringToDouble(val);
   }

   if(StringFind(content, "confidence_boost=") >= 0)
   {
      int pos = StringFind(content, "confidence_boost=") + 17;
      int endPos = StringFind(content, "\n", pos);
      string val = StringSubstr(content, pos, endPos - pos);
      m_llmAdj.confidenceBoost = StringToDouble(val);
   }

   if(StringFind(content, "volatility_regime=") >= 0)
   {
      int pos = StringFind(content, "volatility_regime=") + 18;
      int endPos = StringFind(content, "\n", pos);
      m_llmAdj.volatilityRegime = StringSubstr(content, pos, endPos - pos);
   }

   if(StringFind(content, "llm_confidence=") >= 0)
   {
      int pos = StringFind(content, "llm_confidence=") + 15;
      int endPos = StringFind(content, "\n", pos);
      string val = StringSubstr(content, pos, endPos - pos);
      m_llmAdj.llmConfidence = StringToDouble(val);
   }

   if(StringFind(content, "tighten=") >= 0)
   {
      int pos = StringFind(content, "tighten=") + 8;
      int endPos = StringFind(content, "\n", pos);
      string val = StringSubstr(content, pos, endPos - pos);
      m_llmAdj.tightenSLTP = (val == "true" || val == "1");
   }

   m_llmAdj.lastUpdate = TimeCurrent();

   Print("LLM Adjustment loaded: SL_adj=", m_llmAdj.slMultiplierAdj,
         " TP_adj=", m_llmAdj.tpMultiplierAdj,
         " Regime=", m_llmAdj.volatilityRegime,
         " Conf=", m_llmAdj.llmConfidence);

   return true;
}

//+------------------------------------------------------------------+
//| Get Adjusted SL Multiplier                                        |
//+------------------------------------------------------------------+
double CXAUGridManager::GetAdjustedSLMultiplier(void)
{
   double baseMult = ATR_SL_MULTIPLIER;

   if(!m_config.useLLMAdjustment || m_llmAdj.llmConfidence < 0.5)
      return baseMult;

   // Apply LLM adjustment (clamped to reasonable range)
   double adj = MathMax(-0.5, MathMin(0.5, m_llmAdj.slMultiplierAdj));
   double result = baseMult + adj;

   // Ensure minimum SL multiplier
   return MathMax(1.0, result);
}

//+------------------------------------------------------------------+
//| Get Adjusted TP Multiplier                                        |
//+------------------------------------------------------------------+
double CXAUGridManager::GetAdjustedTPMultiplier(void)
{
   double baseMult = ATR_TP_MULTIPLIER;

   if(!m_config.useLLMAdjustment || m_llmAdj.llmConfidence < 0.5)
      return baseMult;

   // Apply LLM adjustment (clamped to reasonable range)
   double adj = MathMax(-1.0, MathMin(1.0, m_llmAdj.tpMultiplierAdj));
   double result = baseMult + adj;

   // Ensure minimum TP multiplier (at least 2:1 R:R)
   return MathMax(2.0, result);
}

//+------------------------------------------------------------------+
//| Check if should open new grid order                               |
//+------------------------------------------------------------------+
bool CXAUGridManager::ShouldOpenGrid(int direction)
{
   // Direction: 1 = BUY, -1 = SELL, 0 = BOTH

   // Check regime alignment
   if(m_regime == XAUUSD_REGIME_BULLISH && direction < 0) return false;
   if(m_regime == XAUUSD_REGIME_BEARISH && direction > 0) return false;

   // Check entropy filter
   if(m_config.useEntropyFilter)
   {
      double entropy = CalculateEntropy();
      if(entropy > m_config.entropyThreshold)
      {
         return false;
      }
   }

   // Check confidence with compression boost
   double confidence = CalculateRegimeConfidence() + (COMPRESSION_BOOST / 100.0);
   if(confidence < m_config.confidenceThreshold)
   {
      return false;
   }

   // Check order limits
   if(CountOpenOrders() >= m_config.maxOrdersPerExpert)
   {
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Open a grid order with Hidden SL/TP                               |
//+------------------------------------------------------------------+
bool CXAUGridManager::OpenGridOrder(int direction, int gridLevel)
{
   if(!ShouldOpenGrid(direction)) return false;

   double atr = GetATR();
   if(atr <= 0) return false;

   // Get adjusted multipliers (base + LLM adjustment)
   double slMult = GetAdjustedSLMultiplier();
   double tpMult = GetAdjustedTPMultiplier();

   // Calculate virtual SL/TP (HIDDEN - not sent to broker)
   double slDistance = atr * slMult;
   double tpDistance = atr * tpMult;
   double partialTpDistance = tpDistance * (PARTIAL_TP_PERCENT / 100.0);

   // Calculate lot size based on risk
   double lotSize = CalculateLotSize(slDistance);
   if(lotSize <= 0) return false;

   double price;
   ENUM_ORDER_TYPE orderType;
   double virtualSL, virtualTP, partialTP;

   if(direction > 0) // BUY
   {
      orderType = ORDER_TYPE_BUY;
      price = SymbolInfoDouble(m_symbol, SYMBOL_ASK);
      virtualSL = price - slDistance;
      virtualTP = price + tpDistance;
      partialTP = price + partialTpDistance;
   }
   else // SELL
   {
      orderType = ORDER_TYPE_SELL;
      price = SymbolInfoDouble(m_symbol, SYMBOL_BID);
      virtualSL = price + slDistance;
      virtualTP = price - tpDistance;
      partialTP = price - partialTpDistance;
   }

   // Build order request - NO SL/TP sent to broker (HIDDEN)
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = m_symbol;
   request.volume = lotSize;
   request.type = orderType;
   request.price = price;
   request.sl = 0;  // HIDDEN - managed internally
   request.tp = 0;  // HIDDEN - managed internally
   request.deviation = 50;
   request.magic = m_magicNumber;
   request.comment = StringFormat("XAUG_%d_L%d", m_magicNumber, gridLevel);
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result))
   {
      Print("XAUUSD Grid order failed: ", GetLastError());
      return false;
   }

   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("XAUUSD Grid order rejected: ", result.comment);
      return false;
   }

   // Store grid order with virtual levels
   int idx = m_orderCount;
   ArrayResize(m_orders, m_orderCount + 1);

   m_orders[idx].ticket = result.deal;
   m_orders[idx].entryPrice = price;
   m_orders[idx].virtualSL = virtualSL;
   m_orders[idx].virtualTP = virtualTP;
   m_orders[idx].partialTP = partialTP;
   m_orders[idx].volume = lotSize;
   m_orders[idx].partialVolume = lotSize * (PARTIAL_TP_PERCENT / 100.0);
   m_orders[idx].partialClosed = false;
   m_orders[idx].breakEvenSet = false;
   m_orders[idx].trailingActive = false;
   m_orders[idx].openTime = TimeCurrent();
   m_orders[idx].gridLevel = gridLevel;
   m_orders[idx].atrAtEntry = atr;
   m_orders[idx].actualSLMult = slMult;
   m_orders[idx].actualTPMult = tpMult;

   m_orderCount++;

   Print("========================================");
   Print("XAUUSD GRID ORDER OPENED");
   Print("Direction: ", (direction > 0 ? "BUY" : "SELL"));
   Print("Entry: ", DoubleToString(price, 2));
   Print("Hidden SL: ", DoubleToString(virtualSL, 2), " (", slMult, "x ATR)");
   Print("Hidden TP: ", DoubleToString(virtualTP, 2), " (", tpMult, "x ATR)");
   Print("Partial TP: ", DoubleToString(partialTP, 2), " (", PARTIAL_TP_PERCENT, "%)");
   Print("ATR: ", DoubleToString(atr, 2));
   Print("Grid Level: ", gridLevel);
   Print("LLM Regime: ", m_llmAdj.volatilityRegime);
   Print("========================================");

   return true;
}

//+------------------------------------------------------------------+
//| Manage Open Positions                                             |
//+------------------------------------------------------------------+
void CXAUGridManager::ManageOpenPositions(void)
{
   // Sync grid orders with actual positions
   SyncGridOrders();

   // Check each order
   CheckVirtualSLTP();
   CheckPartialTakeProfit();

   if(m_config.breakEvenEnabled)
   {
      CheckBreakEven();
   }

   if(m_config.trailingStopEnabled)
   {
      CheckTrailingStop();
   }
}

//+------------------------------------------------------------------+
//| Check Virtual SL/TP (Hidden Order Management)                     |
//+------------------------------------------------------------------+
void CXAUGridManager::CheckVirtualSLTP(void)
{
   for(int i = m_orderCount - 1; i >= 0; i--)
   {
      ulong ticket = m_orders[i].ticket;

      // Find the actual position
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         if(PositionSelectByTicket(PositionGetTicket(p)))
         {
            if(PositionGetTicket(p) == ticket)
            {
               found = true;
               break;
            }
         }
      }

      if(!found)
      {
         // Try direct selection
         if(PositionSelectByTicket(ticket))
         {
            found = true;
         }
      }

      if(!found)
      {
         RemoveGridOrder(i);
         continue;
      }

      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double volume = PositionGetDouble(POSITION_VOLUME);

      // Check hidden SL
      bool hitSL = false;
      bool hitTP = false;

      if(posType == POSITION_TYPE_BUY)
      {
         hitSL = (currentPrice <= m_orders[i].virtualSL);
         hitTP = (currentPrice >= m_orders[i].virtualTP);
      }
      else // SELL
      {
         hitSL = (currentPrice >= m_orders[i].virtualSL);
         hitTP = (currentPrice <= m_orders[i].virtualTP);
      }

      if(hitSL)
      {
         Print("XAUUSD HIDDEN SL HIT: Closing position ", ticket, " at ", currentPrice);
         ClosePosition(ticket, volume);
         RemoveGridOrder(i);
      }
      else if(hitTP)
      {
         Print("XAUUSD HIDDEN TP HIT: Closing position ", ticket, " at ", currentPrice);
         ClosePosition(ticket, volume);
         RemoveGridOrder(i);
      }
   }
}

//+------------------------------------------------------------------+
//| Check Partial Take Profit (50% at first target)                   |
//+------------------------------------------------------------------+
void CXAUGridManager::CheckPartialTakeProfit(void)
{
   for(int i = 0; i < m_orderCount; i++)
   {
      if(m_orders[i].partialClosed) continue;

      ulong ticket = m_orders[i].ticket;
      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      bool hitPartial = false;

      if(posType == POSITION_TYPE_BUY)
      {
         hitPartial = (currentPrice >= m_orders[i].partialTP);
      }
      else
      {
         hitPartial = (currentPrice <= m_orders[i].partialTP);
      }

      if(hitPartial)
      {
         double closeVolume = m_orders[i].partialVolume;

         // Normalize volume
         double minLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
         double lotStep = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
         closeVolume = MathMax(closeVolume, minLot);
         closeVolume = MathFloor(closeVolume / lotStep) * lotStep;
         closeVolume = NormalizeDouble(closeVolume, 2);

         if(ClosePosition(ticket, closeVolume))
         {
            m_orders[i].partialClosed = true;
            m_orders[i].volume -= closeVolume;
            Print("XAUUSD PARTIAL TP: Closed ", PARTIAL_TP_PERCENT, "% (", closeVolume, " lots) from ticket ", ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Break-Even (at 30% of TP progress)                          |
//+------------------------------------------------------------------+
void CXAUGridManager::CheckBreakEven(void)
{
   for(int i = 0; i < m_orderCount; i++)
   {
      if(m_orders[i].breakEvenSet) continue;

      ulong ticket = m_orders[i].ticket;
      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double entryPrice = m_orders[i].entryPrice;
      double tpDistance = MathAbs(m_orders[i].virtualTP - entryPrice);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      // Calculate progress to TP
      double progress = MathAbs(currentPrice - entryPrice) / tpDistance;

      // Move to BE at BREAKEVEN_ACTIVATION (30%)
      if(progress >= BREAKEVEN_ACTIVATION)
      {
         bool inProfit = (posType == POSITION_TYPE_BUY && currentPrice > entryPrice) ||
                         (posType == POSITION_TYPE_SELL && currentPrice < entryPrice);

         if(inProfit)
         {
            // Move virtual SL to break-even + small buffer
            double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
            double buffer = 50 * point; // 50 points buffer for XAUUSD

            if(posType == POSITION_TYPE_BUY)
            {
               m_orders[i].virtualSL = entryPrice + buffer;
            }
            else
            {
               m_orders[i].virtualSL = entryPrice - buffer;
            }

            m_orders[i].breakEvenSet = true;
            Print("XAUUSD BREAK-EVEN: Moved virtual SL to entry for ticket ", ticket, " (", progress * 100, "% progress)");
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Trailing Stop (activate at 50% of TP)                       |
//+------------------------------------------------------------------+
void CXAUGridManager::CheckTrailingStop(void)
{
   double atr = GetATR();
   if(atr <= 0) return;

   double trailDistance = atr * 1.0; // Trail at 1 ATR

   for(int i = 0; i < m_orderCount; i++)
   {
      ulong ticket = m_orders[i].ticket;
      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double entryPrice = m_orders[i].entryPrice;
      double tpDistance = MathAbs(m_orders[i].virtualTP - entryPrice);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      // Calculate progress
      double progress = MathAbs(currentPrice - entryPrice) / tpDistance;

      // Activate trailing at TRAILING_ACTIVATION (50%)
      if(progress < TRAILING_ACTIVATION) continue;

      // Check if in profit direction
      bool inProfit = (posType == POSITION_TYPE_BUY && currentPrice > entryPrice) ||
                      (posType == POSITION_TYPE_SELL && currentPrice < entryPrice);

      if(!inProfit) continue;

      m_orders[i].trailingActive = true;

      double newSL = m_orders[i].virtualSL;

      if(posType == POSITION_TYPE_BUY)
      {
         double proposedSL = currentPrice - trailDistance;
         if(proposedSL > m_orders[i].virtualSL && proposedSL > entryPrice)
         {
            newSL = proposedSL;
         }
      }
      else // SELL
      {
         double proposedSL = currentPrice + trailDistance;
         if(proposedSL < m_orders[i].virtualSL && proposedSL < entryPrice)
         {
            newSL = proposedSL;
         }
      }

      if(newSL != m_orders[i].virtualSL)
      {
         m_orders[i].virtualSL = newSL;
         Print("XAUUSD TRAILING: Updated virtual SL to ", DoubleToString(newSL, 2), " for ticket ", ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Get ATR Value                                                     |
//+------------------------------------------------------------------+
double CXAUGridManager::GetATR(void)
{
   if(m_atrHandle == INVALID_HANDLE) return 0;

   if(CopyBuffer(m_atrHandle, 0, 0, 3, m_atrBuffer) < 3)
   {
      // Fallback calculation
      MqlRates rates[];
      ArraySetAsSeries(rates, true);

      int copied = CopyRates(m_symbol, PERIOD_M5, 0, ATR_PERIOD + 1, rates);
      if(copied < ATR_PERIOD + 1) return 0;

      double sum = 0;
      for(int i = 0; i < ATR_PERIOD; i++)
      {
         double tr = MathMax(rates[i].high - rates[i].low,
                    MathMax(MathAbs(rates[i].high - rates[i+1].close),
                           MathAbs(rates[i].low - rates[i+1].close)));
         sum += tr;
      }
      return sum / ATR_PERIOD;
   }

   return m_atrBuffer[1];
}

//+------------------------------------------------------------------+
//| Calculate Lot Size Based on Risk                                  |
//+------------------------------------------------------------------+
double CXAUGridManager::CalculateLotSize(double slDistance)
{
   if(slDistance <= 0) return 0;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (m_config.riskPerGridPct / 100.0);

   // Get tick value for XAUUSD
   double tickValue = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(m_symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tickValue <= 0 || tickSize <= 0) return 0;

   // Calculate lots
   double lots = riskAmount / (slDistance / tickSize * tickValue);

   // Normalize
   double minLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);

   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);
   lots = MathFloor(lots / lotStep) * lotStep;

   return NormalizeDouble(lots, 2);
}

//+------------------------------------------------------------------+
//| Check if Within Risk Limits                                       |
//+------------------------------------------------------------------+
bool CXAUGridManager::IsWithinRiskLimits(void)
{
   if(!CheckDailyDrawdown())
   {
      Print("XAUGridManager: Daily drawdown limit reached");
      return false;
   }

   if(!CheckMaxDrawdown())
   {
      Print("XAUGridManager: Max drawdown limit reached");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Count Open Orders for This Expert                                 |
//+------------------------------------------------------------------+
int CXAUGridManager::CountOpenOrders(void)
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         if(PositionGetInteger(POSITION_MAGIC) == m_magicNumber &&
            PositionGetString(POSITION_SYMBOL) == m_symbol)
         {
            count++;
         }
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Check Daily Drawdown                                              |
//+------------------------------------------------------------------+
bool CXAUGridManager::CheckDailyDrawdown(void)
{
   double current = MathMin(AccountInfoDouble(ACCOUNT_BALANCE),
                           AccountInfoDouble(ACCOUNT_EQUITY));

   if(m_dailyStartBalance <= 0) return true;

   double dd = ((m_dailyStartBalance - current) / m_dailyStartBalance) * 100;

   return (dd < m_config.dailyDDLimit);
}

//+------------------------------------------------------------------+
//| Check Max Drawdown                                                |
//+------------------------------------------------------------------+
bool CXAUGridManager::CheckMaxDrawdown(void)
{
   double current = MathMin(AccountInfoDouble(ACCOUNT_BALANCE),
                           AccountInfoDouble(ACCOUNT_EQUITY));

   if(m_highWaterMark <= 0) return true;

   double dd = ((m_highWaterMark - current) / m_highWaterMark) * 100;

   return (dd < m_config.maxDDLimit);
}

//+------------------------------------------------------------------+
//| Daily Reset                                                       |
//+------------------------------------------------------------------+
void CXAUGridManager::DailyReset(void)
{
   m_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   m_lastDayReset = TimeCurrent();
   Print("XAUGridManager: Daily reset. New baseline: $", DoubleToString(m_dailyStartBalance, 2));
}

//+------------------------------------------------------------------+
//| Close Position                                                    |
//+------------------------------------------------------------------+
bool CXAUGridManager::ClosePosition(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket)) return false;

   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   string symbol = PositionGetString(POSITION_SYMBOL);

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = symbol;
   request.volume = volume;
   request.deviation = 50;
   request.magic = m_magicNumber;
   request.type_filling = ORDER_FILLING_IOC;

   if(posType == POSITION_TYPE_BUY)
   {
      request.type = ORDER_TYPE_SELL;
      request.price = SymbolInfoDouble(symbol, SYMBOL_BID);
   }
   else
   {
      request.type = ORDER_TYPE_BUY;
      request.price = SymbolInfoDouble(symbol, SYMBOL_ASK);
   }

   if(!OrderSend(request, result))
   {
      Print("XAUUSD Close position failed: ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Modify Position                                                   |
//+------------------------------------------------------------------+
bool CXAUGridManager::ModifyPosition(ulong ticket, double sl, double tp)
{
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.sl = sl;
   request.tp = tp;

   if(!OrderSend(request, result))
   {
      Print("XAUUSD Modify position failed: ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Find Grid Order by Ticket                                         |
//+------------------------------------------------------------------+
int CXAUGridManager::FindGridOrder(ulong ticket)
{
   for(int i = 0; i < m_orderCount; i++)
   {
      if(m_orders[i].ticket == ticket)
         return i;
   }
   return -1;
}

//+------------------------------------------------------------------+
//| Remove Grid Order from Tracking                                   |
//+------------------------------------------------------------------+
void CXAUGridManager::RemoveGridOrder(int index)
{
   if(index < 0 || index >= m_orderCount) return;

   for(int i = index; i < m_orderCount - 1; i++)
   {
      m_orders[i] = m_orders[i + 1];
   }

   m_orderCount--;
   ArrayResize(m_orders, m_orderCount);
}

//+------------------------------------------------------------------+
//| Sync Grid Orders with Actual Positions                            |
//+------------------------------------------------------------------+
void CXAUGridManager::SyncGridOrders(void)
{
   // Remove closed orders from tracking
   for(int i = m_orderCount - 1; i >= 0; i--)
   {
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         if(PositionSelectByTicket(PositionGetTicket(p)))
         {
            if(PositionGetTicket(p) == m_orders[i].ticket)
            {
               found = true;
               break;
            }
         }
      }

      if(!found)
      {
         RemoveGridOrder(i);
      }
   }

   // Add any positions we're not tracking (from restart)
   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      ulong ticket = PositionGetTicket(p);
      if(!PositionSelectByTicket(ticket)) continue;

      if(PositionGetInteger(POSITION_MAGIC) != m_magicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;

      if(FindGridOrder(ticket) < 0)
      {
         // Add to tracking with estimated values
         int idx = m_orderCount;
         ArrayResize(m_orders, m_orderCount + 1);

         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double atr = GetATR();
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         double slMult = GetAdjustedSLMultiplier();
         double tpMult = GetAdjustedTPMultiplier();

         m_orders[idx].ticket = ticket;
         m_orders[idx].entryPrice = entry;
         m_orders[idx].volume = PositionGetDouble(POSITION_VOLUME);
         m_orders[idx].partialVolume = m_orders[idx].volume * (PARTIAL_TP_PERCENT / 100.0);
         m_orders[idx].partialClosed = false;
         m_orders[idx].breakEvenSet = false;
         m_orders[idx].trailingActive = false;
         m_orders[idx].openTime = (datetime)PositionGetInteger(POSITION_TIME);
         m_orders[idx].gridLevel = 0;
         m_orders[idx].atrAtEntry = atr;
         m_orders[idx].actualSLMult = slMult;
         m_orders[idx].actualTPMult = tpMult;

         if(posType == POSITION_TYPE_BUY)
         {
            m_orders[idx].virtualSL = entry - (atr * slMult);
            m_orders[idx].virtualTP = entry + (atr * tpMult);
            m_orders[idx].partialTP = entry + (atr * tpMult * (PARTIAL_TP_PERCENT / 100.0));
         }
         else
         {
            m_orders[idx].virtualSL = entry + (atr * slMult);
            m_orders[idx].virtualTP = entry - (atr * tpMult);
            m_orders[idx].partialTP = entry - (atr * tpMult * (PARTIAL_TP_PERCENT / 100.0));
         }

         m_orderCount++;
         Print("XAUUSD: Synced existing position: ", ticket);
      }
   }
}
//+------------------------------------------------------------------+
