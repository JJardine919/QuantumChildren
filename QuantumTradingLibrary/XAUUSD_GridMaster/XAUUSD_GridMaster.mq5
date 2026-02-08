//+------------------------------------------------------------------+
//|                                             XAUUSD_GridMaster.mq5|
//|                   XAUUSD Grid Trading Expert Advisor              |
//|                   GetLeveraged Multi-Account Edition              |
//|                   With LLM Dynamic SL/TP Integration              |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library - DooDoo Edition"
#property version   "1.00"
#property description "XAUUSD Grid Trading System with Neural Network"
#property description "Features:"
#property description "- ATR-based Dynamic SL/TP (1.5x SL, 3.0x TP - HARD-CODED)"
#property description "- Hidden SL/TP (managed internally)"
#property description "- Entropy filtering (only trade predictable markets)"
#property description "- +12 Compression boost to confidence threshold"
#property description "- Partial TP (50% at first target)"
#property description "- Trailing stop at 50% of TP"
#property description "- Break-even at 30% of TP"
#property description "- LLM integration for dynamic adjustments"
#property description ""
#property description "GetLeveraged Accounts: 113328, 113326, 107245"
#property strict

//--- Include Core Grid Logic
#include "XAUUSD_GridCore.mqh"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT CONFIGURATION ==="
input string   InpAccountName       = "GL_113328";   // Account Name (for logging)
input int      InpMagicBullish      = 113001;        // Magic Number (BULLISH)
input int      InpMagicBearish      = 113002;        // Magic Number (BEARISH)
input int      InpMagicNeutral      = 113003;        // Magic Number (NEUTRAL)
input double   InpAccountBalance    = 10000;         // Account Balance (for DD calc)
input double   InpDailyDDLimit      = 5.0;           // Daily Drawdown Limit %
input double   InpMaxDDLimit        = 10.0;          // Max Drawdown Limit %

input group "=== RISK MANAGEMENT ==="
input double   InpRiskPerGrid       = 0.25;          // Risk Per Grid Level %
input bool     InpDynamicHiddenSLTP = true;          // Use Hidden SL/TP
input bool     InpTrailingStop      = true;          // Trailing Stop Enabled
input bool     InpBreakEven         = true;          // Break-Even Enabled

input group "=== GRID SETTINGS ==="
input int      InpMaxOrdersPerExpert = 5;            // Max Orders Per Expert
input int      InpMaxTotalOrders    = 15;            // Max Total Orders
input double   InpGridSpacingAtr    = 0.75;          // Grid Spacing (ATR multiples)

input group "=== ENTROPY FILTER (PREDICTABILITY) ==="
input double   InpConfidenceThreshold = 0.22;        // Base Confidence Threshold
input bool     InpUseEntropyFilter  = true;          // Use Entropy Filter
input double   InpEntropyThreshold  = 2.0;           // Max Entropy (lower = more selective)

input group "=== LLM INTEGRATION ==="
input bool     InpUseLLM            = true;          // Use LLM for Dynamic SL/TP
input string   InpLLMSignalFile     = "xauusd_llm_signal.txt"; // LLM Signal File (Common folder)

input group "=== TRADING ==="
input bool     InpTradeEnabled      = true;          // Trading Enabled
input int      InpCheckInterval     = 30;            // Check Interval (seconds)

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
CXAUGridManager   g_bullishManager;
CXAUGridManager   g_bearishManager;
CXAUGridManager   g_neutralManager;
XAUGridConfig     g_config;
datetime          g_lastCheck = 0;

// Grid level tracking
int               g_bullishLevel = 0;
int               g_bearishLevel = 0;
int               g_neutralBuyLevel = 0;
int               g_neutralSellLevel = 0;

// Last grid prices
double            g_lastBullishPrice = 0;
double            g_lastBearishPrice = 0;
double            g_lastNeutralBuyPrice = 0;
double            g_lastNeutralSellPrice = 0;

// Range for neutral trading
double            g_rangeHigh = 0;
double            g_rangeLow = 0;

// Stats
int               g_totalTrades = 0;
double            g_totalProfit = 0;
datetime          g_startTime = 0;

//+------------------------------------------------------------------+
//| Expert Initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("");
   Print("========================================================");
   Print("     XAUUSD GRIDMASTER - GETLEVERAGED EDITION");
   Print("========================================================");
   Print("Account Name: ", InpAccountName);
   Print("Symbol: XAUUSD");
   Print("");
   Print("--- ATR-BASED SL/TP (HARD-CODED) ---");
   Print("Stop Loss: 1.5x ATR");
   Print("Take Profit: 3.0x ATR");
   Print("ATR Period: 14");
   Print("Partial TP: 50% at first target");
   Print("Trailing Activation: 50% of TP");
   Print("Break-Even Activation: 30% of TP");
   Print("");
   Print("--- COMPRESSION BOOST ---");
   Print("Confidence Boost: +12");
   Print("Base Threshold: ", InpConfidenceThreshold * 100, "%");
   Print("Effective Threshold: ", (InpConfidenceThreshold - 0.12) * 100, "% (after boost)");
   Print("");
   Print("--- ENTROPY FILTER ---");
   Print("Entropy Filter: ", InpUseEntropyFilter ? "ENABLED" : "DISABLED");
   Print("Max Entropy: ", InpEntropyThreshold);
   Print("(Only trade when market is predictable)");
   Print("");
   Print("--- LLM INTEGRATION ---");
   Print("LLM Dynamic SL/TP: ", InpUseLLM ? "ENABLED" : "DISABLED");
   Print("Signal File: ", InpLLMSignalFile);
   Print("");
   Print("--- GRID LIMITS ---");
   Print("Max Per Expert: ", InpMaxOrdersPerExpert);
   Print("Max Total: ", InpMaxTotalOrders);
   Print("Grid Spacing: ", InpGridSpacingAtr, "x ATR");
   Print("");
   Print("--- MAGIC NUMBERS ---");
   Print("BULLISH: ", InpMagicBullish);
   Print("BEARISH: ", InpMagicBearish);
   Print("NEUTRAL: ", InpMagicNeutral);
   Print("========================================================");
   Print("");

   //--- Setup Configuration
   g_config.riskPerGridPct = InpRiskPerGrid;
   g_config.dynamicHiddenSLTP = InpDynamicHiddenSLTP;
   g_config.trailingStopEnabled = InpTrailingStop;
   g_config.breakEvenEnabled = InpBreakEven;
   g_config.maxOrdersPerExpert = InpMaxOrdersPerExpert;
   g_config.maxTotalOrders = InpMaxTotalOrders;
   g_config.gridSpacingAtr = InpGridSpacingAtr;
   g_config.confidenceThreshold = InpConfidenceThreshold;
   g_config.useEntropyFilter = InpUseEntropyFilter;
   g_config.entropyThreshold = InpEntropyThreshold;
   g_config.accountBalance = InpAccountBalance;
   g_config.dailyDDLimit = InpDailyDDLimit;
   g_config.maxDDLimit = InpMaxDDLimit;
   g_config.useLLMAdjustment = InpUseLLM;
   g_config.llmSignalFile = InpLLMSignalFile;

   //--- Initialize All Three Grid Managers
   if(!g_bullishManager.Initialize(InpMagicBullish, XAUUSD_REGIME_BULLISH))
   {
      Print("ERROR: Failed to initialize BULLISH Grid Manager");
      return INIT_FAILED;
   }
   g_bullishManager.SetConfig(g_config);

   if(!g_bearishManager.Initialize(InpMagicBearish, XAUUSD_REGIME_BEARISH))
   {
      Print("ERROR: Failed to initialize BEARISH Grid Manager");
      return INIT_FAILED;
   }
   g_bearishManager.SetConfig(g_config);

   if(!g_neutralManager.Initialize(InpMagicNeutral, XAUUSD_REGIME_NEUTRAL))
   {
      Print("ERROR: Failed to initialize NEUTRAL Grid Manager");
      return INIT_FAILED;
   }
   g_neutralManager.SetConfig(g_config);

   //--- Initialize range detection
   DetectRange();

   g_startTime = TimeCurrent();

   //--- Print trading permissions
   Print("");
   Print("--- PERMISSION CHECK ---");
   Print("Trade Allowed: ", AccountInfoInteger(ACCOUNT_TRADE_ALLOWED));
   Print("Expert Allowed: ", AccountInfoInteger(ACCOUNT_TRADE_EXPERT));
   Print("Terminal Trade: ", TerminalInfoInteger(TERMINAL_TRADE_ALLOWED));
   Print("MQL Trade: ", MQLInfoInteger(MQL_TRADE_ALLOWED));
   Print("");

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      Print("!!! WARNING: MQL Trading not allowed - check EA properties !!!");
   }

   Print("XAUUSD GridMaster initialized successfully");
   Print("========================================================");
   Print("");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_bullishManager.Deinitialize();
   g_bearishManager.Deinitialize();
   g_neutralManager.Deinitialize();

   Print("");
   Print("========================================================");
   Print("XAUUSD GRIDMASTER SESSION SUMMARY");
   Print("Account: ", InpAccountName);
   Print("Total Trades: ", g_totalTrades);
   Print("Session Duration: ", (TimeCurrent() - g_startTime) / 3600, " hours");
   Print("Reason: ", reason);
   Print("========================================================");
   Print("");
}

//+------------------------------------------------------------------+
//| Expert Tick Function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Always manage positions every tick
   g_bullishManager.ManageOpenPositions();
   g_bearishManager.ManageOpenPositions();
   g_neutralManager.ManageOpenPositions();

   //--- Check interval for new orders
   if(TimeCurrent() - g_lastCheck < InpCheckInterval)
   {
      return;
   }
   g_lastCheck = TimeCurrent();

   //--- Check global risk limits
   if(!CheckGlobalRiskLimits())
   {
      return;
   }

   //--- Process each manager
   g_bullishManager.ProcessTick();
   g_bearishManager.ProcessTick();
   g_neutralManager.ProcessTick();

   if(!InpTradeEnabled)
   {
      return;
   }

   //--- Trading permission check
   if(!CanTrade())
   {
      static datetime lastWarn = 0;
      if(TimeCurrent() - lastWarn > 300)
      {
         Print("XAUUSD GridMaster: Cannot trade - check permissions");
         lastWarn = TimeCurrent();
      }
      return;
   }

   //--- Update range detection periodically
   static datetime lastRangeUpdate = 0;
   if(TimeCurrent() - lastRangeUpdate > 3600) // Every hour
   {
      DetectRange();
      lastRangeUpdate = TimeCurrent();
   }

   //--- Check global order limit
   int totalOrders = CountAllOpenOrders();
   if(totalOrders >= InpMaxTotalOrders)
   {
      return;
   }

   //--- Determine dominant regime and activate appropriate expert
   ENUM_XAUUSD_REGIME dominantRegime = DetectDominantRegime();

   //--- Log status periodically
   static datetime lastLog = 0;
   if(TimeCurrent() - lastLog > 300) // Every 5 min
   {
      double entropy = g_bullishManager.CalculateEntropy();
      LLMAdjustment llmAdj = g_bullishManager.GetLLMAdjustment();

      Print("");
      Print("=== XAUUSD STATUS UPDATE ===");
      Print("Regime: ", EnumToString(dominantRegime));
      Print("Entropy: ", DoubleToString(entropy, 3), " (max: ", InpEntropyThreshold, ")");
      Print("Total Positions: ", totalOrders, "/", InpMaxTotalOrders);
      Print("Balance: $", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
      Print("Equity: $", DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2));
      if(InpUseLLM && llmAdj.llmConfidence > 0)
      {
         Print("LLM Regime: ", llmAdj.volatilityRegime);
         Print("LLM Confidence: ", llmAdj.llmConfidence * 100, "%");
      }
      Print("===========================");
      Print("");

      lastLog = TimeCurrent();
   }

   //--- Execute grid strategy based on regime
   switch(dominantRegime)
   {
      case XAUUSD_REGIME_BULLISH:
         CheckBullishGrid();
         break;

      case XAUUSD_REGIME_BEARISH:
         CheckBearishGrid();
         break;

      case XAUUSD_REGIME_NEUTRAL:
         CheckNeutralGrid();
         break;
   }
}

//+------------------------------------------------------------------+
//| Check if we can trade                                             |
//+------------------------------------------------------------------+
bool CanTrade()
{
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) return false;
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) return false;
   if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) return false;
   if(!TerminalInfoInteger(TERMINAL_CONNECTED)) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Detect Dominant Regime                                            |
//+------------------------------------------------------------------+
ENUM_XAUUSD_REGIME DetectDominantRegime()
{
   double bullishConf = g_bullishManager.CalculateRegimeConfidence();
   double bearishConf = g_bearishManager.CalculateRegimeConfidence();
   double neutralConf = g_neutralManager.CalculateRegimeConfidence();

   // Apply compression boost
   double boost = COMPRESSION_BOOST / 100.0;
   bullishConf += boost;
   bearishConf += boost;
   neutralConf += boost;

   // Only consider if above threshold
   bool bullishValid = (bullishConf >= InpConfidenceThreshold);
   bool bearishValid = (bearishConf >= InpConfidenceThreshold);
   bool neutralValid = (neutralConf >= InpConfidenceThreshold);

   // Find highest confidence
   if(bullishValid && bullishConf >= bearishConf && bullishConf >= neutralConf)
      return XAUUSD_REGIME_BULLISH;

   if(bearishValid && bearishConf >= bullishConf && bearishConf >= neutralConf)
      return XAUUSD_REGIME_BEARISH;

   if(neutralValid)
      return XAUUSD_REGIME_NEUTRAL;

   // Default to neutral
   return XAUUSD_REGIME_NEUTRAL;
}

//+------------------------------------------------------------------+
//| Check Global Risk Limits                                          |
//+------------------------------------------------------------------+
bool CheckGlobalRiskLimits()
{
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(currentBalance, currentEquity);

   static double highWaterMark = 0;
   static double dailyStartBalance = 0;
   static datetime lastDayReset = 0;

   if(highWaterMark == 0) highWaterMark = InpAccountBalance;
   if(dailyStartBalance == 0) dailyStartBalance = InpAccountBalance;
   if(lastDayReset == 0) lastDayReset = TimeCurrent();

   // Update high water mark
   if(currentBalance > highWaterMark)
      highWaterMark = currentBalance;

   // Daily reset
   MqlDateTime now, last;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(lastDayReset, last);

   if(now.day != last.day)
   {
      dailyStartBalance = currentBalance;
      lastDayReset = TimeCurrent();
      Print("XAUUSD GridMaster: Daily reset - Baseline: $", DoubleToString(dailyStartBalance, 2));
   }

   // Daily DD check
   if(dailyStartBalance > 0)
   {
      double dailyDD = ((dailyStartBalance - current) / dailyStartBalance) * 100;
      if(dailyDD >= InpDailyDDLimit)
      {
         Print("XAUUSD GridMaster: BLOCKED - Daily DD limit: ", DoubleToString(dailyDD, 2), "%");
         return false;
      }
   }

   // Max DD check
   if(highWaterMark > 0)
   {
      double maxDD = ((highWaterMark - current) / highWaterMark) * 100;
      if(maxDD >= InpMaxDDLimit)
      {
         Print("XAUUSD GridMaster: BLOCKED - Max DD limit: ", DoubleToString(maxDD, 2), "%");
         return false;
      }
   }

   return true;
}

//+------------------------------------------------------------------+
//| Count All Open Orders                                             |
//+------------------------------------------------------------------+
int CountAllOpenOrders()
{
   int count = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         int magic = (int)PositionGetInteger(POSITION_MAGIC);

         if(magic == InpMagicBullish || magic == InpMagicBearish || magic == InpMagicNeutral)
         {
            if(PositionGetString(POSITION_SYMBOL) == "XAUUSD")
            {
               count++;
            }
         }
      }
   }

   return count;
}

//+------------------------------------------------------------------+
//| Detect Trading Range                                              |
//+------------------------------------------------------------------+
void DetectRange()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates("XAUUSD", PERIOD_H1, 0, 24, rates);
   if(copied < 24) return;

   g_rangeHigh = rates[0].high;
   g_rangeLow = rates[0].low;

   for(int i = 1; i < copied; i++)
   {
      if(rates[i].high > g_rangeHigh) g_rangeHigh = rates[i].high;
      if(rates[i].low < g_rangeLow) g_rangeLow = rates[i].low;
   }

   Print("XAUUSD Range: High=", DoubleToString(g_rangeHigh, 2), " | Low=", DoubleToString(g_rangeLow, 2));
}

//+------------------------------------------------------------------+
//| Check BULLISH Grid Opportunity                                    |
//+------------------------------------------------------------------+
void CheckBullishGrid()
{
   if(!g_bullishManager.ShouldOpenGrid(1))
      return;

   double currentPrice = SymbolInfoDouble("XAUUSD", SYMBOL_ASK);
   double atr = g_bullishManager.GetATR();
   if(atr <= 0) return;

   double gridSpacing = atr * InpGridSpacingAtr;

   if(g_bullishLevel == 0)
   {
      if(g_bullishManager.OpenGridOrder(1, 1))
      {
         g_bullishLevel = 1;
         g_lastBullishPrice = currentPrice;
         g_totalTrades++;
         Print("XAUUSD: BULLISH BUY opened at level 1");
      }
   }
   else
   {
      // For BULLISH, add more buys as price drops (better entry)
      double priceMove = g_lastBullishPrice - currentPrice;
      if(priceMove >= gridSpacing && g_bullishLevel < InpMaxOrdersPerExpert)
      {
         int nextLevel = g_bullishLevel + 1;
         if(g_bullishManager.OpenGridOrder(1, nextLevel))
         {
            g_bullishLevel = nextLevel;
            g_lastBullishPrice = currentPrice;
            g_totalTrades++;
            Print("XAUUSD: BULLISH BUY opened at level ", nextLevel);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check BEARISH Grid Opportunity                                    |
//+------------------------------------------------------------------+
void CheckBearishGrid()
{
   if(!g_bearishManager.ShouldOpenGrid(-1))
      return;

   double currentPrice = SymbolInfoDouble("XAUUSD", SYMBOL_BID);
   double atr = g_bearishManager.GetATR();
   if(atr <= 0) return;

   double gridSpacing = atr * InpGridSpacingAtr;

   if(g_bearishLevel == 0)
   {
      if(g_bearishManager.OpenGridOrder(-1, 1))
      {
         g_bearishLevel = 1;
         g_lastBearishPrice = currentPrice;
         g_totalTrades++;
         Print("XAUUSD: BEARISH SELL opened at level 1");
      }
   }
   else
   {
      // For BEARISH, add more sells as price rises (better entry)
      double priceMove = currentPrice - g_lastBearishPrice;
      if(priceMove >= gridSpacing && g_bearishLevel < InpMaxOrdersPerExpert)
      {
         int nextLevel = g_bearishLevel + 1;
         if(g_bearishManager.OpenGridOrder(-1, nextLevel))
         {
            g_bearishLevel = nextLevel;
            g_lastBearishPrice = currentPrice;
            g_totalTrades++;
            Print("XAUUSD: BEARISH SELL opened at level ", nextLevel);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check NEUTRAL Grid Opportunity                                    |
//+------------------------------------------------------------------+
void CheckNeutralGrid()
{
   if(!g_neutralManager.ShouldOpenGrid(0))
      return;

   double currentPrice = SymbolInfoDouble("XAUUSD", SYMBOL_BID);
   double atr = g_neutralManager.GetATR();
   if(atr <= 0) return;

   double gridSpacing = atr * InpGridSpacingAtr;
   double rangeMid = (g_rangeHigh + g_rangeLow) / 2;
   double rangeQuarter = (g_rangeHigh - g_rangeLow) / 4;

   int maxPerSide = InpMaxOrdersPerExpert / 2;

   // BUY zone (lower part of range)
   if(currentPrice < rangeMid - rangeQuarter && g_neutralBuyLevel < maxPerSide)
   {
      bool shouldOpen = false;

      if(g_neutralBuyLevel == 0)
      {
         shouldOpen = true;
      }
      else
      {
         double priceMove = g_lastNeutralBuyPrice - currentPrice;
         shouldOpen = (priceMove >= gridSpacing);
      }

      if(shouldOpen)
      {
         int level = g_neutralBuyLevel + 1;
         if(g_neutralManager.OpenGridOrder(1, level))
         {
            g_neutralBuyLevel = level;
            g_lastNeutralBuyPrice = currentPrice;
            g_totalTrades++;
            Print("XAUUSD: NEUTRAL BUY opened at level ", level);
         }
      }
   }

   // SELL zone (upper part of range)
   if(currentPrice > rangeMid + rangeQuarter && g_neutralSellLevel < maxPerSide)
   {
      bool shouldOpen = false;

      if(g_neutralSellLevel == 0)
      {
         shouldOpen = true;
      }
      else
      {
         double priceMove = currentPrice - g_lastNeutralSellPrice;
         shouldOpen = (priceMove >= gridSpacing);
      }

      if(shouldOpen)
      {
         int level = g_neutralSellLevel + 1;
         if(g_neutralManager.OpenGridOrder(-1, level))
         {
            g_neutralSellLevel = level;
            g_lastNeutralSellPrice = currentPrice;
            g_totalTrades++;
            Print("XAUUSD: NEUTRAL SELL opened at level ", level);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Chart Event Handler                                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   // Future: Dashboard controls
}

//+------------------------------------------------------------------+
//| Timer Function                                                    |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Future: Periodic reporting
}
//+------------------------------------------------------------------+
