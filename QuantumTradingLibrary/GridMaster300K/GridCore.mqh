//+------------------------------------------------------------------+
//|                                                   GridCore.mqh   |
//|                           GridMaster 300K - Core Grid Logic      |
//|                           Prop Firm Challenge - Conservative     |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property version   "2.00"

//+------------------------------------------------------------------+
//| Regime Enumeration                                               |
//+------------------------------------------------------------------+
enum ENUM_REGIME_TYPE
{
   REGIME_BEARISH = -1,
   REGIME_NEUTRAL = 0,
   REGIME_BULLISH = 1
};

//+------------------------------------------------------------------+
//| Grid Order Structure                                             |
//+------------------------------------------------------------------+
struct GridOrder
{
   ulong    ticket;
   double   entryPrice;
   double   virtualSL;      // Hidden SL - managed internally
   double   virtualTP;      // Hidden TP - managed internally
   double   partialTP;      // First target for partial close
   double   volume;
   double   partialVolume;  // Volume to close at partial TP
   bool     partialClosed;  // Has partial been taken?
   bool     breakEvenSet;   // Has BE been triggered?
   datetime openTime;
   int      gridLevel;
};

//+------------------------------------------------------------------+
//| Grid Configuration Structure                                     |
//+------------------------------------------------------------------+
struct GridConfig
{
   // Risk Management
   double   rewardRatio;           // 3:1 ratio
   double   slAtrMultiplier;       // 1.5x ATR for SL
   double   tpAtrMultiplier;       // 3.0x ATR for TP (3:1 reward)
   double   partialTpRatio;        // 0.5 = 50% at first target
   bool     dynamicHiddenSLTP;     // Hidden orders, managed internally
   bool     trailingStopEnabled;
   bool     breakEvenEnabled;

   // Grid Settings
   int      maxOrdersPerExpert;    // 10 max per expert
   int      maxTotalOrders;        // 30 total max
   double   gridSpacingAtr;        // Grid spacing in ATR multiples
   double   riskPerGridPct;        // Risk per grid level

   // Compression Filter
   double   confidenceThreshold;   // 0.80 = 80% confidence required
   int      compressionBoost;      // +12 compression boost

   // Account
   double   accountBalance;
   double   dailyDDLimit;
   double   maxDDLimit;
};

//+------------------------------------------------------------------+
//| Grid Manager Class                                               |
//+------------------------------------------------------------------+
class CGridManager
{
private:
   GridConfig    m_config;
   GridOrder     m_orders[];
   int           m_orderCount;
   int           m_magicNumber;
   string        m_symbol;
   ENUM_REGIME_TYPE m_regime;

   // Tracking
   double        m_startBalance;
   double        m_highWaterMark;
   double        m_dailyStartBalance;
   datetime      m_lastDayReset;

   // Stealth
   bool          m_stealthMode;

   // ATR Handle
   int           m_atrHandle;
   double        m_atrBuffer[];

   // EMA Handles for regime detection
   int           m_emaFastHandle;
   int           m_emaSlowHandle;
   int           m_ema200Handle;
   double        m_emaFastBuffer[];
   double        m_emaSlowBuffer[];
   double        m_ema200Buffer[];

public:
   //--- Constructor/Destructor
   CGridManager(void);
   ~CGridManager(void);

   //--- Initialization
   bool     Initialize(int magic, string symbol, ENUM_REGIME_TYPE regime);
   void     SetConfig(GridConfig &config);
   void     SetStealthMode(bool stealth) { m_stealthMode = stealth; }
   void     Deinitialize(void);

   //--- Core Operations
   bool     ProcessTick(void);
   double   CalculateRegimeConfidence(void);
   bool     ShouldOpenGrid(int direction);
   bool     OpenGridOrder(int direction, int gridLevel);
   void     ManageOpenPositions(void);

   //--- Hidden SL/TP Management
   void     CheckVirtualSLTP(void);
   void     CheckPartialTakeProfit(void);
   void     CheckBreakEven(void);
   void     CheckTrailingStop(void);

   //--- Helpers
   double   GetATR(int period);
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
   ENUM_REGIME_TYPE GetRegime(void) { return m_regime; }
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CGridManager::CGridManager(void)
{
   m_orderCount = 0;
   m_magicNumber = 0;
   m_symbol = "";
   m_regime = REGIME_NEUTRAL;
   m_stealthMode = false;
   m_atrHandle = INVALID_HANDLE;
   m_emaFastHandle = INVALID_HANDLE;
   m_emaSlowHandle = INVALID_HANDLE;
   m_ema200Handle = INVALID_HANDLE;

   ArraySetAsSeries(m_atrBuffer, true);
   ArraySetAsSeries(m_emaFastBuffer, true);
   ArraySetAsSeries(m_emaSlowBuffer, true);
   ArraySetAsSeries(m_ema200Buffer, true);
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CGridManager::~CGridManager(void)
{
   Deinitialize();
}

//+------------------------------------------------------------------+
//| Initialize the Grid Manager                                       |
//+------------------------------------------------------------------+
bool CGridManager::Initialize(int magic, string symbol, ENUM_REGIME_TYPE regime)
{
   m_magicNumber = magic;
   m_symbol = symbol;
   m_regime = regime;

   // Create indicator handles
   m_atrHandle = iATR(m_symbol, PERIOD_M5, 14);
   m_emaFastHandle = iMA(m_symbol, PERIOD_M5, 8, 0, MODE_EMA, PRICE_CLOSE);
   m_emaSlowHandle = iMA(m_symbol, PERIOD_M5, 21, 0, MODE_EMA, PRICE_CLOSE);
   m_ema200Handle = iMA(m_symbol, PERIOD_M5, 200, 0, MODE_EMA, PRICE_CLOSE);

   if(m_atrHandle == INVALID_HANDLE || m_emaFastHandle == INVALID_HANDLE ||
      m_emaSlowHandle == INVALID_HANDLE || m_ema200Handle == INVALID_HANDLE)
   {
      Print("GridManager: Failed to create indicator handles");
      return false;
   }

   // Initialize tracking
   m_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   m_highWaterMark = m_startBalance;
   m_dailyStartBalance = m_startBalance;
   m_lastDayReset = TimeCurrent();

   // Sync existing orders
   SyncGridOrders();

   Print("GridManager initialized: Magic=", m_magicNumber, " Symbol=", m_symbol,
         " Regime=", EnumToString(m_regime));

   return true;
}

//+------------------------------------------------------------------+
//| Set Configuration                                                 |
//+------------------------------------------------------------------+
void CGridManager::SetConfig(GridConfig &config)
{
   m_config = config;
}

//+------------------------------------------------------------------+
//| Deinitialize                                                      |
//+------------------------------------------------------------------+
void CGridManager::Deinitialize(void)
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
bool CGridManager::ProcessTick(void)
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

   // Manage open positions (hidden SL/TP, trailing, BE)
   ManageOpenPositions();

   // Check regime confidence
   double confidence = CalculateRegimeConfidence();
   if(confidence < m_config.confidenceThreshold)
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
//| Calculate Regime Confidence                                       |
//+------------------------------------------------------------------+
double CGridManager::CalculateRegimeConfidence(void)
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
      case REGIME_BULLISH:
         // Bullish: Price > EMA200, Fast > Slow, both above 200
         if(price > ema200) confidence += 0.35;
         if(emaFast > emaSlow) confidence += 0.35;
         if(emaSlow > ema200) confidence += 0.30;

         // Apply compression boost
         confidence += (m_config.compressionBoost / 100.0);
         break;

      case REGIME_BEARISH:
         // Bearish: Price < EMA200, Fast < Slow, both below 200
         if(price < ema200) confidence += 0.35;
         if(emaFast < emaSlow) confidence += 0.35;
         if(emaSlow < ema200) confidence += 0.30;

         // Apply compression boost
         confidence += (m_config.compressionBoost / 100.0);
         break;

      case REGIME_NEUTRAL:
         // Neutral: Price near EMA200, EMAs converging
         double distFromEma200 = MathAbs(price - ema200) / ema200;
         double emaDiff = MathAbs(emaFast - emaSlow) / emaSlow;

         // Less distance = more neutral
         if(distFromEma200 < 0.01) confidence += 0.40;
         else if(distFromEma200 < 0.02) confidence += 0.25;
         else confidence += 0.10;

         // Converging EMAs = more neutral
         if(emaDiff < 0.005) confidence += 0.40;
         else if(emaDiff < 0.01) confidence += 0.25;
         else confidence += 0.10;

         // Apply compression boost
         confidence += (m_config.compressionBoost / 100.0);
         break;
   }

   return MathMin(confidence, 1.0);
}

//+------------------------------------------------------------------+
//| Check if should open new grid order                               |
//+------------------------------------------------------------------+
bool CGridManager::ShouldOpenGrid(int direction)
{
   // Direction: 1 = BUY, -1 = SELL, 0 = BOTH

   // Check regime alignment
   if(m_regime == REGIME_BULLISH && direction < 0) return false;
   if(m_regime == REGIME_BEARISH && direction > 0) return false;

   // Check confidence
   double confidence = CalculateRegimeConfidence();
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
//| Open a grid order                                                 |
//+------------------------------------------------------------------+
bool CGridManager::OpenGridOrder(int direction, int gridLevel)
{
   if(!ShouldOpenGrid(direction)) return false;

   double atr = GetATR(14);
   if(atr <= 0) return false;

   // Calculate virtual SL/TP (hidden - not sent to broker)
   double slDistance = atr * m_config.slAtrMultiplier;
   double tpDistance = atr * m_config.tpAtrMultiplier;
   double partialTpDistance = tpDistance * m_config.partialTpRatio;

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

   // Build order request - NO SL/TP sent to broker (hidden)
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = m_symbol;
   request.volume = lotSize;
   request.type = orderType;
   request.price = price;
   request.sl = 0;  // Hidden - managed internally
   request.tp = 0;  // Hidden - managed internally
   request.deviation = 50;
   request.magic = m_stealthMode ? 0 : m_magicNumber;
   request.comment = m_stealthMode ? "" : StringFormat("GRID_%d_L%d", m_magicNumber, gridLevel);
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = ORDER_FILLING_IOC;

   if(!OrderSend(request, result))
   {
      Print("Grid order failed: ", GetLastError());
      return false;
   }

   if(result.retcode != TRADE_RETCODE_DONE)
   {
      Print("Grid order rejected: ", result.comment);
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
   m_orders[idx].partialVolume = lotSize * m_config.partialTpRatio;
   m_orders[idx].partialClosed = false;
   m_orders[idx].breakEvenSet = false;
   m_orders[idx].openTime = TimeCurrent();
   m_orders[idx].gridLevel = gridLevel;

   m_orderCount++;

   Print("GRID ORDER: ", (direction > 0 ? "BUY" : "SELL"), " @ ", price,
         " | Hidden SL: ", virtualSL, " | Hidden TP: ", virtualTP,
         " | Level: ", gridLevel);

   return true;
}

//+------------------------------------------------------------------+
//| Manage Open Positions                                             |
//+------------------------------------------------------------------+
void CGridManager::ManageOpenPositions(void)
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
void CGridManager::CheckVirtualSLTP(void)
{
   for(int i = m_orderCount - 1; i >= 0; i--)
   {
      ulong ticket = m_orders[i].ticket;

      // Find the actual position
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         if(PositionGetTicket(p) == ticket)
         {
            found = true;
            break;
         }
         if(PositionSelectByTicket(ticket))
         {
            found = true;
            break;
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
         Print("HIDDEN SL HIT: Closing position ", ticket);
         ClosePosition(ticket, volume);
         RemoveGridOrder(i);
      }
      else if(hitTP)
      {
         Print("HIDDEN TP HIT: Closing position ", ticket);
         ClosePosition(ticket, volume);
         RemoveGridOrder(i);
      }
   }
}

//+------------------------------------------------------------------+
//| Check Partial Take Profit                                         |
//+------------------------------------------------------------------+
void CGridManager::CheckPartialTakeProfit(void)
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
            Print("PARTIAL TP: Closed ", closeVolume, " lots from ticket ", ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Break-Even                                                  |
//+------------------------------------------------------------------+
void CGridManager::CheckBreakEven(void)
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

      // Move to BE when price moves 50% to TP
      double progress = MathAbs(currentPrice - entryPrice) / tpDistance;

      if(progress >= 0.50)
      {
         bool inProfit = (posType == POSITION_TYPE_BUY && currentPrice > entryPrice) ||
                         (posType == POSITION_TYPE_SELL && currentPrice < entryPrice);

         if(inProfit)
         {
            // Move virtual SL to break-even + small buffer
            double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
            double buffer = 20 * point;

            if(posType == POSITION_TYPE_BUY)
            {
               m_orders[i].virtualSL = entryPrice + buffer;
            }
            else
            {
               m_orders[i].virtualSL = entryPrice - buffer;
            }

            m_orders[i].breakEvenSet = true;
            Print("BREAK-EVEN: Moved virtual SL to entry for ticket ", ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check Trailing Stop                                               |
//+------------------------------------------------------------------+
void CGridManager::CheckTrailingStop(void)
{
   double atr = GetATR(14);
   if(atr <= 0) return;

   double trailDistance = atr * 1.0; // Trail at 1 ATR

   for(int i = 0; i < m_orderCount; i++)
   {
      if(!m_orders[i].breakEvenSet) continue; // Only trail after BE is set

      ulong ticket = m_orders[i].ticket;
      if(!PositionSelectByTicket(ticket)) continue;

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      double entryPrice = m_orders[i].entryPrice;
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

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
         Print("TRAILING: Updated virtual SL to ", newSL, " for ticket ", ticket);
      }
   }
}

//+------------------------------------------------------------------+
//| Get ATR Value                                                     |
//+------------------------------------------------------------------+
double CGridManager::GetATR(int period)
{
   if(m_atrHandle == INVALID_HANDLE) return 0;

   if(CopyBuffer(m_atrHandle, 0, 0, 3, m_atrBuffer) < 3)
   {
      // Fallback calculation
      MqlRates rates[];
      ArraySetAsSeries(rates, true);

      int copied = CopyRates(m_symbol, PERIOD_M5, 0, period + 1, rates);
      if(copied < period + 1) return 0;

      double sum = 0;
      for(int i = 0; i < period; i++)
      {
         double tr = MathMax(rates[i].high - rates[i].low,
                    MathMax(MathAbs(rates[i].high - rates[i+1].close),
                           MathAbs(rates[i].low - rates[i+1].close)));
         sum += tr;
      }
      return sum / period;
   }

   return m_atrBuffer[1];
}

//+------------------------------------------------------------------+
//| Calculate Lot Size Based on Risk                                  |
//+------------------------------------------------------------------+
double CGridManager::CalculateLotSize(double slDistance)
{
   if(slDistance <= 0) return 0;

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (m_config.riskPerGridPct / 100.0);

   // Get tick value
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
bool CGridManager::IsWithinRiskLimits(void)
{
   if(!CheckDailyDrawdown())
   {
      Print("GridManager: Daily drawdown limit reached");
      return false;
   }

   if(!CheckMaxDrawdown())
   {
      Print("GridManager: Max drawdown limit reached");
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Count Open Orders for This Expert                                 |
//+------------------------------------------------------------------+
int CGridManager::CountOpenOrders(void)
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
bool CGridManager::CheckDailyDrawdown(void)
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
bool CGridManager::CheckMaxDrawdown(void)
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
void CGridManager::DailyReset(void)
{
   m_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   m_lastDayReset = TimeCurrent();
   Print("GridManager: Daily reset. New baseline: $", DoubleToString(m_dailyStartBalance, 2));
}

//+------------------------------------------------------------------+
//| Close Position                                                    |
//+------------------------------------------------------------------+
bool CGridManager::ClosePosition(ulong ticket, double volume)
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
   request.magic = m_stealthMode ? 0 : m_magicNumber;
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
      Print("Close position failed: ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Modify Position                                                   |
//+------------------------------------------------------------------+
bool CGridManager::ModifyPosition(ulong ticket, double sl, double tp)
{
   // Virtual SL/TP management only - NOT sent to broker
   // All SL/TP is managed internally via virtual tracking in grid position struct
   // This function exists for interface compatibility only
   return PositionSelectByTicket(ticket);
}

//+------------------------------------------------------------------+
//| Find Grid Order by Ticket                                         |
//+------------------------------------------------------------------+
int CGridManager::FindGridOrder(ulong ticket)
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
void CGridManager::RemoveGridOrder(int index)
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
void CGridManager::SyncGridOrders(void)
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
         double atr = GetATR(14);
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         m_orders[idx].ticket = ticket;
         m_orders[idx].entryPrice = entry;
         m_orders[idx].volume = PositionGetDouble(POSITION_VOLUME);
         m_orders[idx].partialVolume = m_orders[idx].volume * m_config.partialTpRatio;
         m_orders[idx].partialClosed = false;
         m_orders[idx].breakEvenSet = false;
         m_orders[idx].openTime = (datetime)PositionGetInteger(POSITION_TIME);
         m_orders[idx].gridLevel = 0;

         if(posType == POSITION_TYPE_BUY)
         {
            m_orders[idx].virtualSL = entry - (atr * m_config.slAtrMultiplier);
            m_orders[idx].virtualTP = entry + (atr * m_config.tpAtrMultiplier);
            m_orders[idx].partialTP = entry + (atr * m_config.tpAtrMultiplier * m_config.partialTpRatio);
         }
         else
         {
            m_orders[idx].virtualSL = entry + (atr * m_config.slAtrMultiplier);
            m_orders[idx].virtualTP = entry - (atr * m_config.tpAtrMultiplier);
            m_orders[idx].partialTP = entry - (atr * m_config.tpAtrMultiplier * m_config.partialTpRatio);
         }

         m_orderCount++;
         Print("Synced existing position: ", ticket);
      }
   }
}
//+------------------------------------------------------------------+
