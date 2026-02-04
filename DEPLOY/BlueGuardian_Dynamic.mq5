//+------------------------------------------------------------------+
//|                                        BlueGuardian_Dynamic.mq5 |
//|                         Dynamic SL/TP with LLM Integration       |
//|                              Ready for MQL5 Market Signal        |
//+------------------------------------------------------------------+
#property copyright "QuantumChildren"
#property version   "1.00"
#property description "Dynamic TP/SL with LLM monitoring"
#property strict

//+------------------------------------------------------------------+
//| HARD-CODED CORE VALUES - DO NOT CHANGE                           |
//+------------------------------------------------------------------+
#define ATR_MULT           0.0438    // Hard-coded ATR multiplier for initial TP
#define TP_RATIO           3         // Take Profit ratio
#define SL_BASE            1         // Stop Loss base
#define SL_MULT            1.5       // Stop Loss multiplier
#define DYN_TP_PCT         30        // Dynamic TP adjustment %
#define USE_DYN_TP         true      // Use Dynamic TP
#define ROLLING_SL         true      // Rolling/Trailing SL enabled
#define INITIAL_SL_POINTS  50        // Initial SL in points (LLM monitors)

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== Account Settings ==="
input int      InpMagicNumber     = 365100;        // Magic Number
input double   InpVolume          = 0.01;          // Lot Size
input string   InpSymbol          = "";            // Symbol (blank = current)

input group "=== LLM Integration ==="
input bool     InpUseLLM          = true;          // Enable LLM Monitoring
input string   InpLLMConfigFile   = "llm_config.json";  // LLM Config File
input int      InpLLMCheckSecs    = 60;            // LLM Check Interval (seconds)
input bool     InpLLMEmergencyOff = true;          // Allow LLM Emergency Shutoff

input group "=== Risk Management ==="
input double   InpDailyDDLimit    = 4.5;           // Daily DD Limit %
input double   InpMaxDDLimit      = 9.0;           // Max DD Limit %

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
   double highWaterPrice; // For rolling SL
   double lowWaterPrice;  // For rolling SL (shorts)
   bool   inProfit;
   datetime openTime;
};
DynamicPosition g_positions[];

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol = (InpSymbol == "") ? _Symbol : InpSymbol;

   // Create indicators
   g_atrHandle = iATR(g_symbol, PERIOD_M5, InpATRPeriod);
   g_emaFastHandle = iMA(g_symbol, PERIOD_M5, InpEMAFast, 0, MODE_EMA, PRICE_CLOSE);
   g_emaSlowHandle = iMA(g_symbol, PERIOD_M5, InpEMASlow, 0, MODE_EMA, PRICE_CLOSE);

   if(g_atrHandle == INVALID_HANDLE || g_emaFastHandle == INVALID_HANDLE || g_emaSlowHandle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicators");
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
   Print("BlueGuardian Dynamic - Initialized");
   Print("Symbol: ", g_symbol);
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
   if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
   if(g_emaFastHandle != INVALID_HANDLE) IndicatorRelease(g_emaFastHandle);
   if(g_emaSlowHandle != INVALID_HANDLE) IndicatorRelease(g_emaSlowHandle);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check LLM for emergency stop
   if(InpUseLLM && InpLLMEmergencyOff)
   {
      CheckLLMStatus();
      if(g_llmEmergencyStop)
      {
         // Close all positions and halt
         CloseAllPositions("LLM Emergency Stop");
         return;
      }
   }

   // Manage existing positions (rolling SL, dynamic TP)
   ManagePositions();

   // Check for new bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(g_symbol, PERIOD_M5, 0);
   if(currentBar == lastBar) return;
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
      return entryPrice + tpDistance;
   else // SELL
      return entryPrice - tpDistance;
}

//+------------------------------------------------------------------+
//| Calculate Initial SL (50 points, then monitored by LLM)           |
//+------------------------------------------------------------------+
double CalculateInitialSL(double entryPrice, int direction)
{
   double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
   double slDistance = INITIAL_SL_POINTS * point * SL_MULT;

   if(direction > 0) // BUY
      return entryPrice - slDistance;
   else // SELL
      return entryPrice + slDistance;
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

   double price = (direction > 0) ? tick.ask : tick.bid;
   double initialTP = CalculateInitialTP(price, direction);
   double initialSL = CalculateInitialSL(price, direction);

   if(initialTP == 0) return;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = g_symbol;
   request.volume = InpVolume;
   request.type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   request.price = price;
   request.sl = initialSL;  // Set initial SL (LLM will monitor)
   request.tp = initialTP;  // Set initial TP (dynamic adjustment enabled)
   request.deviation = 50;
   request.magic = InpMagicNumber;
   request.comment = "BG_Dynamic";
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
   g_positions[idx].initialSL = initialSL;
   g_positions[idx].initialTP = initialTP;
   g_positions[idx].currentSL = initialSL;
   g_positions[idx].currentTP = initialTP;
   g_positions[idx].highWaterPrice = price;
   g_positions[idx].lowWaterPrice = price;
   g_positions[idx].inProfit = false;
   g_positions[idx].openTime = TimeCurrent();

   string typeStr = (direction > 0) ? "BUY" : "SELL";
   Print(typeStr, " @ ", DoubleToString(price, 5),
         " | SL: ", DoubleToString(initialSL, 5),
         " | TP: ", DoubleToString(initialTP, 5),
         " | ATR_MULT: ", ATR_MULT);
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
      bool wasInProfit = g_positions[i].inProfit;
      if(posType == POSITION_TYPE_BUY)
         g_positions[i].inProfit = (currentPrice > entry);
      else
         g_positions[i].inProfit = (currentPrice < entry);

      // === ROLLING SL (if enabled) ===
      if(ROLLING_SL && g_positions[i].inProfit)
      {
         double newSL = g_positions[i].currentSL;
         double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
         double trailDistance = INITIAL_SL_POINTS * point;

         if(posType == POSITION_TYPE_BUY)
         {
            double proposedSL = g_positions[i].highWaterPrice - trailDistance;
            if(proposedSL > g_positions[i].currentSL && proposedSL > entry)
            {
               newSL = proposedSL;
            }
         }
         else
         {
            double proposedSL = g_positions[i].lowWaterPrice + trailDistance;
            if(proposedSL < g_positions[i].currentSL && proposedSL < entry)
            {
               newSL = proposedSL;
            }
         }

         // Update SL if changed
         if(MathAbs(newSL - g_positions[i].currentSL) > point)
         {
            ModifyPositionSL(ticket, newSL);
            g_positions[i].currentSL = newSL;
            Print("ROLLING SL: Moved to ", DoubleToString(newSL, 5));
         }
      }

      // === DYNAMIC TP (if enabled) ===
      if(USE_DYN_TP && g_positions[i].inProfit)
      {
         double tpDistance = MathAbs(g_positions[i].initialTP - entry);
         double currentDistance = MathAbs(currentPrice - entry);
         double progress = currentDistance / tpDistance;

         // If we've hit DYN_TP_PCT of our target, extend TP by DYN_TP_PCT
         if(progress >= (DYN_TP_PCT / 100.0))
         {
            double extension = tpDistance * (DYN_TP_PCT / 100.0);
            double newTP;

            if(posType == POSITION_TYPE_BUY)
               newTP = g_positions[i].currentTP + extension;
            else
               newTP = g_positions[i].currentTP - extension;

            double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);
            if(MathAbs(newTP - g_positions[i].currentTP) > point * 10)
            {
               ModifyPositionTP(ticket, newTP);
               g_positions[i].currentTP = newTP;
               Print("DYNAMIC TP: Extended to ", DoubleToString(newTP, 5));
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
   request.position = ticket;
   request.sl = newSL;
   request.tp = PositionGetDouble(POSITION_TP);

   return OrderSend(request, result) && result.retcode == TRADE_RETCODE_DONE;
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
   request.position = ticket;
   request.sl = PositionGetDouble(POSITION_SL);
   request.tp = newTP;

   return OrderSend(request, result) && result.retcode == TRADE_RETCODE_DONE;
}

//+------------------------------------------------------------------+
//| Check LLM Status from Config File                                 |
//+------------------------------------------------------------------+
void CheckLLMStatus()
{
   if(TimeCurrent() - g_lastLLMCheck < InpLLMCheckSecs) return;
   g_lastLLMCheck = TimeCurrent();

   // Read LLM config file
   string filename = InpLLMConfigFile;
   int handle = FileOpen(filename, FILE_READ | FILE_TXT | FILE_COMMON);

   if(handle == INVALID_HANDLE)
   {
      // No config file - LLM not running, use defaults
      return;
   }

   string content = "";
   while(!FileIsEnding(handle))
   {
      content += FileReadString(handle);
   }
   FileClose(handle);

   // Parse JSON-like content (simple parsing)
   // Expected format: {"emergency_stop": false, "sl_adjust": 0, "tp_adjust": 0}

   g_llmEmergencyStop = (StringFind(content, "\"emergency_stop\": true") >= 0 ||
                         StringFind(content, "\"emergency_stop\":true") >= 0);

   // Parse SL adjustment
   int slPos = StringFind(content, "\"sl_adjust\":");
   if(slPos >= 0)
   {
      string slStr = StringSubstr(content, slPos + 13, 10);
      g_llmSLAdjust = StringToDouble(slStr);
   }

   // Parse TP adjustment
   int tpPos = StringFind(content, "\"tp_adjust\":");
   if(tpPos >= 0)
   {
      string tpStr = StringSubstr(content, tpPos + 13, 10);
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
   double point = SymbolInfoDouble(g_symbol, SYMBOL_POINT);

   bool modified = false;
   double newSL = currentSL;
   double newTP = currentTP;

   // Apply SL adjustment (in points)
   if(g_llmSLAdjust != 0)
   {
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY)
         newSL = currentSL + (g_llmSLAdjust * point);
      else
         newSL = currentSL - (g_llmSLAdjust * point);
      modified = true;
      Print("LLM SL Adjust: ", g_llmSLAdjust, " points");
   }

   // Apply TP adjustment (in points)
   if(g_llmTPAdjust != 0)
   {
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      if(posType == POSITION_TYPE_BUY)
         newTP = currentTP + (g_llmTPAdjust * point);
      else
         newTP = currentTP - (g_llmTPAdjust * point);
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
      request.position = ticket;
      request.sl = newSL;
      request.tp = newTP;

      if(OrderSend(request, result) && result.retcode == TRADE_RETCODE_DONE)
      {
         g_positions[posIdx].currentSL = newSL;
         g_positions[posIdx].currentTP = newTP;
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
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != g_symbol) continue;

      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      MqlTick tick;
      SymbolInfoTick(g_symbol, tick);

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

      OrderSend(request, result);
   }

   ArrayResize(g_positions, 0);
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
