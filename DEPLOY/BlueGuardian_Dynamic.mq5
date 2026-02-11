//+------------------------------------------------------------------+
//|                                        BlueGuardian_Dynamic.mq5 |
//|                         Dynamic SL/TP with LLM Integration       |
//|                              Ready for MQL5 Market Signal        |
//+------------------------------------------------------------------+
#property copyright "QuantumChildren"
#property version   "2.00"
#property description "Dynamic TP/SL with LLM monitoring"
// NOTE: #property strict removed - MQL4 only, not valid in MQL5

//+------------------------------------------------------------------+
//| HARD-CODED CORE VALUES - DO NOT CHANGE                           |
//+------------------------------------------------------------------+
#define ATR_MULT           0.0438    // Hard-coded ATR multiplier for initial TP
#define TP_RATIO           3         // Take Profit ratio
#define SL_BASE            1         // Stop Loss base
#define SL_MULT            1.5       // Stop Loss multiplier
#define DYN_TP_PCT         50        // Dynamic TP adjustment %
#define USE_DYN_TP         true      // Use Dynamic TP
#define ROLLING_SL         true      // Rolling/Trailing SL enabled
#define INITIAL_SL_POINTS  50        // Initial SL in points (LLM monitors)

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Account Settings ==="
input int      InpMagicNumber     = 365100;        // Magic Number
input double   InpVolume          = 0.01;          // Lot Size
input string   InpSymbol          = "BTC";         // Symbol (blank = current)

input group "=== LLM Integration ==="
input bool     InpUseLLM          = true;          // Enable LLM Monitoring
input string   InpLLMConfigFile   = "llm_config.json";  // LLM Config File
input int      InpLLMCheckSecs    = 60;            // LLM Check Interval (seconds)
input bool     InpLLMEmergencyOff = true;          // Allow LLM Emergency Shutoff

input group "=== Risk Management ==="
input double   InpDailyDDLimit    = 4.5;           // Daily DD Limit %
input double   InpMaxDDLimit      = 9.0;           // Max DD Limit %

input group "=== Stealth Settings ==="
input bool     StealthMode        = false;         // Stealth Mode (hide EA identifiers)

input group "=== Signal Settings ==="
input int      InpEMAFast         = 8;             // Fast EMA
input int      InpEMASlow         = 21;            // Slow EMA
input int      InpATRPeriod       = 14;            // ATR Period

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
string g_symbol;
int g_atrHandle;
int g_emaFastHandle;
int g_emaSlowHandle;

double g_atrBuffer[];
double g_emaFastBuffer[];
double g_emaSlowBuffer[];

// Cached symbol properties
double g_point;
int    g_digits;
double g_tickSize;
int    g_stopsLevel;

// LLM state
datetime g_lastLLMCheck = 0;
bool g_llmEmergencyStop = false;
double g_llmSLAdjust = 0;
double g_llmTPAdjust = 0;

// Balance tracking
double g_startBalance;
double g_highWaterMark;
double g_dailyStartBalance;
datetime g_lastDayReset;
bool g_blocked = false;

// Position tracking
struct DynamicPosition
{
   ulong  ticket;
   double entryPrice;
   double initialSL;
   double initialTP;
   double currentSL;      // Rolling SL
   double currentTP;      // Dynamic TP
   double highWaterPrice;  // For rolling SL
   double lowWaterPrice;   // For rolling SL (shorts)
   bool   inProfit;
   int    tpExtensions;    // FIX: Track TP extension count to prevent runaway
   datetime openTime;
};
DynamicPosition g_positions[];

//+------------------------------------------------------------------+
//| Normalize price to symbol tick size                                |
//+------------------------------------------------------------------+
double NormalizePrice(double price)
{
   if(g_tickSize <= 0) return NormalizeDouble(price, g_digits);
   return NormalizeDouble(MathRound(price / g_tickSize) * g_tickSize, g_digits);
}

//+------------------------------------------------------------------+
//| Enforce minimum stops level distance                              |
//+------------------------------------------------------------------+
void EnforceStopsLevel(double price, int direction, double &sl, double &tp)
{
   double minDist = g_stopsLevel * g_point;
   if(minDist <= 0) return;

   if(direction > 0) // BUY
   {
      if(sl > 0 && (price - sl) < minDist)
         sl = NormalizePrice(price - minDist);
      if(tp > 0 && (tp - price) < minDist)
         tp = NormalizePrice(price + minDist);
   }
   else // SELL
   {
      if(sl > 0 && (sl - price) < minDist)
         sl = NormalizePrice(price + minDist);
      if(tp > 0 && (price - tp) < minDist)
         tp = NormalizePrice(price - minDist);
   }
}

//+------------------------------------------------------------------+
//| Validate and resolve symbol name                                  |
//+------------------------------------------------------------------+
bool ValidateSymbol(string &symbol)
{
   // Try exact match first
   if(SymbolSelect(symbol, true))
      return true;

   // Try common suffixes
   string suffixes[] = {"USD", "USDT", ".i", "m", ".raw", ".ecn", ".std", "usd"};
   for(int i = 0; i < ArraySize(suffixes); i++)
   {
      string test = symbol + suffixes[i];
      if(SymbolSelect(test, true))
      {
         Print("Symbol resolved: ", symbol, " -> ", test);
         symbol = test;
         return true;
      }
   }

   return false;
}

//+------------------------------------------------------------------+
//| Find actual position ticket after order fill                      |
//+------------------------------------------------------------------+
ulong FindPositionTicket(ulong orderTicket)
{
   // First try: look through open positions by magic/symbol
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong posTicket = PositionGetTicket(i);
      if(posTicket == 0) continue;
      if(!PositionSelectByTicket(posTicket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;

      // Check if this position was opened by our order
      if(PositionGetInteger(POSITION_IDENTIFIER) == (long)orderTicket)
         return posTicket;
   }

   // Second try: use deal history to find position
   if(HistorySelectByPosition((long)orderTicket))
   {
      for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
      {
         ulong dealTicket = HistoryDealGetTicket(i);
         if(dealTicket == 0) continue;
         ulong posId = (ulong)HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID);
         if(posId > 0)
         {
            // Try selecting this as a position
            if(PositionSelectByTicket(posId))
               return posId;
         }
      }
   }

   // Third try: find most recent position matching magic/symbol
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong posTicket = PositionGetTicket(i);
      if(posTicket == 0) continue;
      if(!PositionSelectByTicket(posTicket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;
      return posTicket;
   }

   return 0;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol = (InpSymbol == "") ? _Symbol : InpSymbol;

   // Validate symbol
   if(!ValidateSymbol(g_symbol))
   {
      Print("ERROR: Symbol '", g_symbol, "' not found. Trying current chart symbol.");
      g_symbol = _Symbol;
   }

   // Cache symbol properties
   g_point      = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
   g_digits     = (int)SymbolInfoInteger(g_symbol, SYMBOL_DIGITS);
   g_tickSize   = SymbolInfoDouble(g_symbol, SYMBOL_TRADE_TICK_SIZE);
   g_stopsLevel = (int)SymbolInfoInteger(g_symbol, SYMBOL_TRADE_STOPS_LEVEL);

   if(g_point <= 0)
   {
      Print("ERROR: Invalid symbol properties for ", g_symbol);
      return INIT_FAILED;
   }

   // Create indicators
   g_atrHandle = iATR(g_symbol, PERIOD_M5, InpATRPeriod);
   g_emaFastHandle = iMA(g_symbol, PERIOD_M5, InpEMAFast, 0, MODE_EMA, PRICE_CLOSE);
   g_emaSlowHandle = iMA(g_symbol, PERIOD_M5, InpEMASlow, 0, MODE_EMA, PRICE_CLOSE);

   if(g_atrHandle == INVALID_HANDLE || g_emaFastHandle == INVALID_HANDLE || g_emaSlowHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicators for ", g_symbol);
      return INIT_FAILED;
   }

   ArraySetAsSeries(g_atrBuffer, true);
   ArraySetAsSeries(g_emaFastBuffer, true);
   ArraySetAsSeries(g_emaSlowBuffer, true);

   // Initialize balance
   g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_highWaterMark = g_startBalance;
   g_dailyStartBalance = g_startBalance;
   g_lastDayReset = TimeCurrent();

   Print("================================================");
   Print("BlueGuardian Dynamic v2.00 - Initialized");
   Print("Symbol: ", g_symbol, " (digits=", g_digits, " point=", g_point, " tickSize=", g_tickSize, ")");
   Print("Stops Level: ", g_stopsLevel, " points");
   Print("------------------------------------------------");
   Print("HARD-CODED VALUES:");
   Print("  ATR_MULT:     ", ATR_MULT);
   Print("  TP_RATIO:     ", TP_RATIO);
   Print("  SL_BASE:      ", SL_BASE);
   Print("  SL_MULT:      ", SL_MULT);
   Print("  DYN_TP_PCT:   ", DYN_TP_PCT, "%");
   Print("  USE_DYN_TP:   ", USE_DYN_TP);
   Print("  ROLLING_SL:   ", ROLLING_SL);
   Print("  INITIAL_SL:   ", INITIAL_SL_POINTS, " points");
   Print("------------------------------------------------");
   Print("LLM Integration: ", InpUseLLM ? "ENABLED" : "DISABLED");
   Print("================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_atrHandle != INVALID_HANDLE) { IndicatorRelease(g_atrHandle); g_atrHandle = INVALID_HANDLE; }
   if(g_emaFastHandle != INVALID_HANDLE) { IndicatorRelease(g_emaFastHandle); g_emaFastHandle = INVALID_HANDLE; }
   if(g_emaSlowHandle != INVALID_HANDLE) { IndicatorRelease(g_emaSlowHandle); g_emaSlowHandle = INVALID_HANDLE; }
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update high water mark on every tick
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   g_highWaterMark = MathMax(g_highWaterMark, MathMax(balance, equity));

   // Check LLM for emergency stop
   if(InpUseLLM && InpLLMEmergencyOff)
   {
      CheckLLMStatus();
      if(g_llmEmergencyStop)
      {
         CloseAllPositions("LLM Emergency Stop");
         return;
      }
   }

   // Manage existing positions (rolling SL, dynamic TP)
   ManagePositions();

   // Check for new bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(g_symbol, PERIOD_M5, 0);
   if(currentBar == 0 || currentBar == lastBar) return;  // FIX: guard against iTime returning 0
   lastBar = currentBar;

   // Daily reset
   CheckDailyReset();

   // DD check
   if(!CheckDrawdownLimits()) return;

   // Skip if position exists
   if(HasOpenPosition()) return;

   // Get signal
   int signal = GetSignal();
   if(signal == 0) return;

   // Execute
   ExecuteTrade(signal);
}

//+------------------------------------------------------------------+
//| Calculate Initial TP using ATR_MULT (0.0438)                      |
//+------------------------------------------------------------------+
double CalculateInitialTP(double entryPrice, int direction)
{
   if(CopyBuffer(g_atrHandle, 0, 0, 3, g_atrBuffer) < 3) return 0;

   double atr = g_atrBuffer[1];
   double tpDistance = atr * ATR_MULT * TP_RATIO;

   if(direction > 0) // BUY
      return NormalizePrice(entryPrice + tpDistance);
   else // SELL
      return NormalizePrice(entryPrice - tpDistance);
}

//+------------------------------------------------------------------+
//| Calculate Initial SL (50 points, then monitored by LLM)           |
//+------------------------------------------------------------------+
double CalculateInitialSL(double entryPrice, int direction)
{
   double slDistance = INITIAL_SL_POINTS * g_point * SL_MULT;

   if(direction > 0) // BUY
      return NormalizePrice(entryPrice - slDistance);
   else // SELL
      return NormalizePrice(entryPrice + slDistance);
}

//+------------------------------------------------------------------+
//| Get Trading Signal                                                |
//+------------------------------------------------------------------+
int GetSignal()
{
   if(CopyBuffer(g_emaFastHandle, 0, 0, 3, g_emaFastBuffer) < 3) return 0;
   if(CopyBuffer(g_emaSlowHandle, 0, 0, 3, g_emaSlowBuffer) < 3) return 0;

   // EMA Crossover
   bool crossUp = (g_emaFastBuffer[2] <= g_emaSlowBuffer[2]) && (g_emaFastBuffer[1] > g_emaSlowBuffer[1]);
   bool crossDown = (g_emaFastBuffer[2] >= g_emaSlowBuffer[2]) && (g_emaFastBuffer[1] < g_emaSlowBuffer[1]);

   if(crossUp) return 1;
   if(crossDown) return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Execute Trade                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade(int direction)
{
   MqlTick tick;
   if(!SymbolInfoTick(g_symbol, tick)) return;

   double price = NormalizePrice((direction > 0) ? tick.ask : tick.bid);
   double initialTP = CalculateInitialTP(price, direction);
   double initialSL = CalculateInitialSL(price, direction);

   if(initialTP == 0) return;

   // Enforce minimum stops distance
   EnforceStopsLevel(price, direction, initialSL, initialTP);

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = g_symbol;
   request.volume = InpVolume;
   request.type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price = price;
   request.sl = initialSL;
   request.tp = initialTP;
   request.deviation = 50;
   request.magic = StealthMode ? 0 : InpMagicNumber;
   request.comment = StealthMode ? "" : "BG_Dynamic";
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = GetFillingMode();

   if(!OrderSend(request, result))
   {
      Print("Order failed: ", GetLastError(),
            " | Price=", DoubleToString(price, g_digits),
            " | SL=", DoubleToString(initialSL, g_digits),
            " | TP=", DoubleToString(initialTP, g_digits));
      return;
   }

   if(result.retcode != TRADE_RETCODE_DONE && result.retcode != TRADE_RETCODE_PLACED)
   {
      Print("Order rejected: ", result.retcode, " - ", result.comment);
      return;
   }

   // FIX: Find actual position ticket (result.order is an ORDER ticket, not position ticket)
   Sleep(100); // Brief wait for position to register
   ulong posTicket = FindPositionTicket(result.order);
   if(posTicket == 0)
   {
      Print("WARNING: Could not find position ticket for order ", result.order, ". Using order ticket as fallback.");
      posTicket = result.order;
   }

   // Get actual fill price from position
   double fillPrice = price;
   if(PositionSelectByTicket(posTicket))
      fillPrice = PositionGetDouble(POSITION_PRICE_OPEN);

   // Track position
   int idx = ArraySize(g_positions);
   ArrayResize(g_positions, idx + 1);

   g_positions[idx].ticket = posTicket;
   g_positions[idx].entryPrice = fillPrice;
   g_positions[idx].initialSL = initialSL;
   g_positions[idx].initialTP = initialTP;
   g_positions[idx].currentSL = initialSL;
   g_positions[idx].currentTP = initialTP;
   g_positions[idx].highWaterPrice = fillPrice;
   g_positions[idx].lowWaterPrice = fillPrice;
   g_positions[idx].inProfit = false;
   g_positions[idx].tpExtensions = 0;  // FIX: Initialize extension counter
   g_positions[idx].openTime = TimeCurrent();

   string typeStr = (direction > 0) ? "BUY" : "SELL";
   Print(typeStr, " @ ", DoubleToString(fillPrice, g_digits),
         " | SL: ", DoubleToString(initialSL, g_digits),
         " | TP: ", DoubleToString(initialTP, g_digits),
         " | ATR_MULT: ", ATR_MULT,
         " | Ticket: ", posTicket);
}

//+------------------------------------------------------------------+
//| Manage Positions - Rolling SL, Dynamic TP                         |
//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = ArraySize(g_positions) - 1; i >= 0; i--)
   {
      ulong ticket = g_positions[i].ticket;

      if(!PositionSelectByTicket(ticket))
      {
         RemovePosition(i);
         continue;
      }

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double entry = g_positions[i].entryPrice;

      // Update high/low water marks
      if(posType == POSITION_TYPE_BUY)
      {
         if(currentPrice > g_positions[i].highWaterPrice)
            g_positions[i].highWaterPrice = currentPrice;
      }
      else
      {
         if(currentPrice < g_positions[i].lowWaterPrice)
            g_positions[i].lowWaterPrice = currentPrice;
      }

      // Check if in profit
      if(posType == POSITION_TYPE_BUY)
         g_positions[i].inProfit = (currentPrice > entry);
      else
         g_positions[i].inProfit = (currentPrice < entry);

      // === ROLLING SL (if enabled) ===
      if(ROLLING_SL && g_positions[i].inProfit)
      {
         double newSL = g_positions[i].currentSL;
         double trailDistance = INITIAL_SL_POINTS * g_point;

         if(posType == POSITION_TYPE_BUY)
         {
            double proposedSL = NormalizePrice(g_positions[i].highWaterPrice - trailDistance);
            if(proposedSL > g_positions[i].currentSL && proposedSL > entry)
            {
               newSL = proposedSL;
            }
         }
         else
         {
            double proposedSL = NormalizePrice(g_positions[i].lowWaterPrice + trailDistance);
            if(proposedSL < g_positions[i].currentSL && proposedSL < entry)
            {
               newSL = proposedSL;
            }
         }

         // Update SL if changed
         if(MathAbs(newSL - g_positions[i].currentSL) > g_point)
         {
            if(ModifyPositionSL(ticket, newSL))
            {
               g_positions[i].currentSL = newSL;
               Print("ROLLING SL: Moved to ", DoubleToString(newSL, g_digits));
            }
         }
      }

      // === DYNAMIC TP (if enabled) ===
      if(USE_DYN_TP && g_positions[i].inProfit)
      {
         double tpDistance = MathAbs(g_positions[i].initialTP - entry);
         if(tpDistance <= 0) continue;  // FIX: guard against division by zero

         double currentDistance = MathAbs(currentPrice - entry);
         double progress = currentDistance / tpDistance;

         // FIX: Use extension counter to determine next trigger threshold
         // Extension 0 triggers at 50%, extension 1 at 100%, extension 2 at 150%, etc.
         double nextThreshold = (g_positions[i].tpExtensions + 1) * (DYN_TP_PCT / 100.0);

         if(progress >= nextThreshold)
         {
            double extension = tpDistance * (DYN_TP_PCT / 100.0);
            double newTP;

            if(posType == POSITION_TYPE_BUY)
               newTP = NormalizePrice(g_positions[i].currentTP + extension);
            else
               newTP = NormalizePrice(g_positions[i].currentTP - extension);

            if(MathAbs(newTP - g_positions[i].currentTP) > g_point * 10)
            {
               if(ModifyPositionTP(ticket, newTP))
               {
                  g_positions[i].currentTP = newTP;
                  g_positions[i].tpExtensions++;  // FIX: Increment to prevent repeat firing
                  Print("DYNAMIC TP: Extended to ", DoubleToString(newTP, g_digits),
                        " (extension #", g_positions[i].tpExtensions, ")");
               }
            }
         }
      }

      // === LLM ADJUSTMENTS (if available) ===
      if(InpUseLLM && (g_llmSLAdjust != 0 || g_llmTPAdjust != 0))
      {
         ApplyLLMAdjustments(i);
      }
   }
}

//+------------------------------------------------------------------+
//| Modify Position SL                                                |
//+------------------------------------------------------------------+
bool ModifyPositionSL(ulong ticket, double newSL)
{
   if(!PositionSelectByTicket(ticket)) return false;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_SLTP;
   request.symbol = g_symbol;       // FIX: required field
   request.position = ticket;
   request.sl = NormalizePrice(newSL);
   request.tp = PositionGetDouble(POSITION_TP);

   if(!OrderSend(request, result) || (result.retcode != TRADE_RETCODE_DONE && result.retcode != TRADE_RETCODE_PLACED))
   {
      Print("ModifySL failed: ticket=", ticket, " retcode=", result.retcode, " ", result.comment);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Modify Position TP                                                |
//+------------------------------------------------------------------+
bool ModifyPositionTP(ulong ticket, double newTP)
{
   if(!PositionSelectByTicket(ticket)) return false;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_SLTP;
   request.symbol = g_symbol;       // FIX: required field
   request.position = ticket;
   request.sl = PositionGetDouble(POSITION_SL);
   request.tp = NormalizePrice(newTP);

   if(!OrderSend(request, result) || (result.retcode != TRADE_RETCODE_DONE && result.retcode != TRADE_RETCODE_PLACED))
   {
      Print("ModifyTP failed: ticket=", ticket, " retcode=", result.retcode, " ", result.comment);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Check LLM Status from Config File                                 |
//+------------------------------------------------------------------+
void CheckLLMStatus()
{
   if(TimeCurrent() - g_lastLLMCheck < InpLLMCheckSecs) return;
   g_lastLLMCheck = TimeCurrent();

   string filename = InpLLMConfigFile;
   int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_COMMON);

   if(handle == INVALID_HANDLE)
      return;

   string content = "";
   while(!FileIsEnding(handle))
   {
      content += FileReadString(handle) + " ";  // FIX: add separator between lines
   }
   FileClose(handle);

   // Parse emergency_stop
   g_llmEmergencyStop = (StringFind(content, "\"emergency_stop\": true") >= 0 ||
                         StringFind(content, "\"emergency_stop\":true") >= 0);

   // Parse SL adjustment
   int slPos = StringFind(content, "\"sl_adjust\":");
   if(slPos >= 0)
   {
      string slStr = StringSubstr(content, slPos + 13, 20);
      // Trim to just the numeric value
      int commaPos = StringFind(slStr, ",");
      int bracePos = StringFind(slStr, "}");
      int endPos = StringLen(slStr);
      if(commaPos >= 0 && commaPos < endPos) endPos = commaPos;
      if(bracePos >= 0 && bracePos < endPos) endPos = bracePos;
      slStr = StringSubstr(slStr, 0, endPos);
      StringTrimLeft(slStr);
      StringTrimRight(slStr);
      g_llmSLAdjust = StringToDouble(slStr);
   }

   // Parse TP adjustment
   int tpPos = StringFind(content, "\"tp_adjust\":");
   if(tpPos >= 0)
   {
      string tpStr = StringSubstr(content, tpPos + 13, 20);
      int commaPos = StringFind(tpStr, ",");
      int bracePos = StringFind(tpStr, "}");
      int endPos = StringLen(tpStr);
      if(commaPos >= 0 && commaPos < endPos) endPos = commaPos;
      if(bracePos >= 0 && bracePos < endPos) endPos = bracePos;
      tpStr = StringSubstr(tpStr, 0, endPos);
      StringTrimLeft(tpStr);
      StringTrimRight(tpStr);
      g_llmTPAdjust = StringToDouble(tpStr);
   }

   if(g_llmEmergencyStop)
   {
      Print("LLM EMERGENCY STOP TRIGGERED!");
   }
}

//+------------------------------------------------------------------+
//| Apply LLM Adjustments to Position                                 |
//+------------------------------------------------------------------+
void ApplyLLMAdjustments(int posIdx)
{
   if(posIdx < 0 || posIdx >= ArraySize(g_positions)) return;

   ulong ticket = g_positions[posIdx].ticket;
   if(!PositionSelectByTicket(ticket)) return;

   double currentSL = PositionGetDouble(POSITION_SL);
   double currentTP = PositionGetDouble(POSITION_TP);

   bool modified = false;
   double newSL = currentSL;
   double newTP = currentTP;

   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

   // Apply SL adjustment (in points)
   if(g_llmSLAdjust != 0)
   {
      if(posType == POSITION_TYPE_BUY)
         newSL = currentSL + (g_llmSLAdjust * g_point);
      else
         newSL = currentSL - (g_llmSLAdjust * g_point);
      newSL = NormalizePrice(newSL);
      modified = true;
      Print("LLM SL Adjust: ", g_llmSLAdjust, " points");
   }

   // Apply TP adjustment (in points)
   if(g_llmTPAdjust != 0)
   {
      if(posType == POSITION_TYPE_BUY)
         newTP = currentTP + (g_llmTPAdjust * g_point);
      else
         newTP = currentTP - (g_llmTPAdjust * g_point);
      newTP = NormalizePrice(newTP);
      modified = true;
      Print("LLM TP Adjust: ", g_llmTPAdjust, " points");
   }

   if(modified)
   {
      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action = TRADE_ACTION_SLTP;
      request.symbol = g_symbol;       // FIX: required field
      request.position = ticket;
      request.sl = newSL;
      request.tp = newTP;

      if(OrderSend(request, result) && (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED))
      {
         g_positions[posIdx].currentSL = newSL;
         g_positions[posIdx].currentTP = newTP;
      }
      else
      {
         Print("LLM adjustment failed: retcode=", result.retcode, " ", result.comment);
      }
   }

   // Reset adjustments after applying
   g_llmSLAdjust = 0;
   g_llmTPAdjust = 0;
}

//+------------------------------------------------------------------+
//| Close All Positions                                               |
//+------------------------------------------------------------------+
void CloseAllPositions(string reason)
{
   Print("CLOSING ALL POSITIONS: ", reason);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;  // FIX: guard against 0 ticket
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;

      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      MqlTick tick;
      if(!SymbolInfoTick(g_symbol, tick))  // FIX: check return value
      {
         Print("WARNING: Cannot get tick for ", g_symbol, " - skipping ticket ", ticket);
         continue;
      }

      MqlTradeRequest request;
      MqlTradeResult result;
      ZeroMemory(request);
      ZeroMemory(result);

      request.action = TRADE_ACTION_DEAL;
      request.position = ticket;
      request.symbol = g_symbol;
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

      if(!OrderSend(request, result) || result.retcode != TRADE_RETCODE_DONE)
         Print("Close failed: ticket=", ticket, " retcode=", result.retcode, " ", result.comment);
   }

   ArrayResize(g_positions, 0);
}

//+------------------------------------------------------------------+
//| Remove Position from Tracking                                     |
//+------------------------------------------------------------------+
void RemovePosition(int index)
{
   int size = ArraySize(g_positions);
   if(index < 0 || index >= size) return;  // FIX: bounds check

   for(int i = index; i < size - 1; i++)
      g_positions[i] = g_positions[i + 1];
   ArrayResize(g_positions, size - 1);
}

//+------------------------------------------------------------------+
//| Has Open Position                                                 |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;  // FIX: guard against 0 ticket
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;
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

   double dailyDD = ((g_dailyStartBalance - current) / g_dailyStartBalance) * 100;
   if(dailyDD >= InpDailyDDLimit)
   {
      if(!g_blocked) Print("BLOCKED: Daily DD ", DoubleToString(dailyDD, 2), "%");
      g_blocked = true;
      return false;
   }

   double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
   if(maxDD >= InpMaxDDLimit)
   {
      if(!g_blocked) Print("BLOCKED: Max DD ", DoubleToString(maxDD, 2), "%");
      g_blocked = true;
      return false;
   }

   g_blocked = false;
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

   if(now.day != last.day)
   {
      g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_lastDayReset = TimeCurrent();
      g_blocked = false;
      Print("Daily reset. Balance: $", DoubleToString(g_dailyStartBalance, 2));
   }
}

//+------------------------------------------------------------------+
//| Get Filling Mode                                                  |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFillingMode()
{
   uint filling = (uint)SymbolInfoInteger(g_symbol, SYMBOL_FILLING_MODE);
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK) return ORDER_FILLING_FOK;
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC) return ORDER_FILLING_IOC;
   return ORDER_FILLING_RETURN;
}
//+------------------------------------------------------------------+
