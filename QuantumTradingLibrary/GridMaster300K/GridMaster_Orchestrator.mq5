//+------------------------------------------------------------------+
//|                                     GridMaster_Orchestrator.mq5  |
//|                        GridMaster 300K - Master Orchestrator     |
//|                       Prop Firm $300K Challenge - Conservative   |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property version   "2.00"
#property description "Master Orchestrator for 3 Grid Experts"
#property description "Coordinates BEARISH + BULLISH + NEUTRAL experts"
#property description "Manages total order limits and risk across all experts"

//--- Include Core Grid Logic
#include "GridCore.mqh"

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input double   InpAccountBalance    = 300000;    // Account Balance
input double   InpDailyDDLimit      = 5.0;       // Daily Drawdown Limit %
input double   InpMaxDDLimit        = 10.0;      // Max Drawdown Limit %
input double   InpPayoutOnPass      = 2500;      // Payout on Pass

input group "=== RISK MANAGEMENT (ALL EXPERTS) ==="
input double   InpRewardRatio       = 3.0;       // Reward Ratio (3:1)
input double   InpSLAtrMultiplier   = 1.5;       // SL ATR Multiplier
input double   InpTPAtrMultiplier   = 3.0;       // TP ATR Multiplier (3x for 3:1)
input double   InpPartialTPRatio    = 0.5;       // Partial TP Ratio (50% at first target)
input double   InpRiskPerGrid       = 0.15;      // Risk Per Grid Level %
input bool     InpDynamicHiddenSLTP = true;      // Dynamic Hidden SL/TP
input bool     InpTrailingStop      = true;      // Trailing Stop Enabled
input bool     InpBreakEven         = true;      // Break-Even Enabled

input group "=== GRID SETTINGS ==="
input int      InpMaxOrdersPerExpert = 10;       // Max Orders Per Expert
input int      InpMaxTotalOrders    = 30;        // Max Total Orders
input double   InpGridSpacingAtr    = 0.5;       // Grid Spacing (ATR multiples)

input group "=== COMPRESSION FILTER ==="
input double   InpConfidenceThreshold = 0.22;    // Confidence Threshold (22%)
input int      InpCompressionBoost  = 12;        // Compression Boost Per Expert

input group "=== EXPERT MAGIC NUMBERS ==="
input int      InpMagicBearish      = 300001;    // Magic Number (BEARISH)
input int      InpMagicBullish      = 300002;    // Magic Number (BULLISH)
input int      InpMagicNeutral      = 300003;    // Magic Number (NEUTRAL)

input group "=== TRADING ==="
input bool     InpTradeEnabled      = true;      // Trading Enabled
input int      InpCheckInterval     = 30;        // Check Interval (seconds)

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CGridManager   g_bearishManager;
CGridManager   g_bullishManager;
CGridManager   g_neutralManager;
GridConfig     g_config;
datetime       g_lastCheck = 0;

// Grid level tracking per expert
int            g_bearishLevel = 0;
int            g_bullishLevel = 0;
int            g_neutralBuyLevel = 0;
int            g_neutralSellLevel = 0;

// Last grid prices
double         g_lastBearishPrice = 0;
double         g_lastBullishPrice = 0;
double         g_lastNeutralBuyPrice = 0;
double         g_lastNeutralSellPrice = 0;

// Range for neutral trading
double         g_rangeHigh = 0;
double         g_rangeLow = 0;

// Stats
int            g_totalTrades = 0;
double         g_totalProfit = 0;
datetime       g_startTime = 0;

//+------------------------------------------------------------------+
//| Expert Initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("GRIDMASTER 300K - ORCHESTRATOR");
   Print("========================================");
   Print("Account: $", DoubleToString(InpAccountBalance, 2));
   Print("Target Payout: $", DoubleToString(InpPayoutOnPass, 2));
   Print("========================================");
   Print("COMPRESSION FILTER:");
   Print("  - Confidence Threshold: ", InpConfidenceThreshold * 100, "%");
   Print("  - Boost Per Expert: +", InpCompressionBoost);
   Print("  - Total Active Boost: +", InpCompressionBoost * 3);
   Print("========================================");
   Print("RISK MANAGEMENT:");
   Print("  - Reward Ratio: ", InpRewardRatio, ":1");
   Print("  - SL: ", InpSLAtrMultiplier, "x ATR");
   Print("  - TP: ", InpTPAtrMultiplier, "x ATR");
   Print("  - Partial TP: ", InpPartialTPRatio * 100, "% at first target");
   Print("  - Hidden SL/TP: ", InpDynamicHiddenSLTP ? "ENABLED" : "DISABLED");
   Print("  - Trailing Stop: ", InpTrailingStop ? "ENABLED" : "DISABLED");
   Print("  - Break-Even: ", InpBreakEven ? "ENABLED" : "DISABLED");
   Print("========================================");
   Print("GRID LIMITS:");
   Print("  - Max Per Expert: ", InpMaxOrdersPerExpert);
   Print("  - Max Total: ", InpMaxTotalOrders);
   Print("========================================");
   Print("EXPERTS:");
   Print("  - BEARISH (SELL) Magic: ", InpMagicBearish);
   Print("  - BULLISH (BUY) Magic: ", InpMagicBullish);
   Print("  - NEUTRAL (BOTH) Magic: ", InpMagicNeutral);
   Print("========================================");

   //--- Setup Configuration
   g_config.rewardRatio = InpRewardRatio;
   g_config.slAtrMultiplier = InpSLAtrMultiplier;
   g_config.tpAtrMultiplier = InpTPAtrMultiplier;
   g_config.partialTpRatio = InpPartialTPRatio;
   g_config.dynamicHiddenSLTP = InpDynamicHiddenSLTP;
   g_config.trailingStopEnabled = InpTrailingStop;
   g_config.breakEvenEnabled = InpBreakEven;
   g_config.maxOrdersPerExpert = InpMaxOrdersPerExpert;
   g_config.maxTotalOrders = InpMaxTotalOrders;
   g_config.gridSpacingAtr = InpGridSpacingAtr;
   g_config.riskPerGridPct = InpRiskPerGrid;
   g_config.confidenceThreshold = InpConfidenceThreshold;
   g_config.compressionBoost = InpCompressionBoost;
   g_config.accountBalance = InpAccountBalance;
   g_config.dailyDDLimit = InpDailyDDLimit;
   g_config.maxDDLimit = InpMaxDDLimit;

   //--- Initialize All Three Grid Managers
   if(!g_bearishManager.Initialize(InpMagicBearish, _Symbol, REGIME_BEARISH))
   {
      Print("ERROR: Failed to initialize BEARISH Grid Manager");
      return INIT_FAILED;
   }
   g_bearishManager.SetConfig(g_config);

   if(!g_bullishManager.Initialize(InpMagicBullish, _Symbol, REGIME_BULLISH))
   {
      Print("ERROR: Failed to initialize BULLISH Grid Manager");
      return INIT_FAILED;
   }
   g_bullishManager.SetConfig(g_config);

   if(!g_neutralManager.Initialize(InpMagicNeutral, _Symbol, REGIME_NEUTRAL))
   {
      Print("ERROR: Failed to initialize NEUTRAL Grid Manager");
      return INIT_FAILED;
   }
   g_neutralManager.SetConfig(g_config);

   //--- Initialize range detection for neutral
   DetectRange();

   g_startTime = TimeCurrent();

   Print("ORCHESTRATOR: All three experts initialized successfully");
   Print("========================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_bearishManager.Deinitialize();
   g_bullishManager.Deinitialize();
   g_neutralManager.Deinitialize();

   Print("========================================");
   Print("ORCHESTRATOR SESSION SUMMARY");
   Print("Total Trades: ", g_totalTrades);
   Print("Total Profit: $", DoubleToString(g_totalProfit, 2));
   Print("========================================");
}

//+------------------------------------------------------------------+
//| Expert Tick Function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Manage positions every tick for all experts
   g_bearishManager.ManageOpenPositions();
   g_bullishManager.ManageOpenPositions();
   g_neutralManager.ManageOpenPositions();

   //--- Check interval for new orders
   if(TimeCurrent() - g_lastCheck < InpCheckInterval)
   {
      return;
   }
   g_lastCheck = TimeCurrent();

   //--- Check global risk limits first
   if(!CheckGlobalRiskLimits())
   {
      return;
   }

   //--- Process each expert
   g_bearishManager.ProcessTick();
   g_bullishManager.ProcessTick();
   g_neutralManager.ProcessTick();

   if(!InpTradeEnabled)
   {
      return;
   }

   //--- Update range detection periodically
   static datetime lastRangeUpdate = 0;
   if(TimeCurrent() - lastRangeUpdate > 3600) // Every hour
   {
      DetectRange();
      lastRangeUpdate = TimeCurrent();
   }

   //--- Check global order limit before opening new orders
   int totalOrders = CountAllOpenOrders();
   if(totalOrders >= InpMaxTotalOrders)
   {
      return; // At max capacity
   }

   //--- Determine dominant regime and activate appropriate expert
   ENUM_REGIME_TYPE dominantRegime = DetectDominantRegime();

   switch(dominantRegime)
   {
      case REGIME_BEARISH:
         CheckBearishGrid();
         break;

      case REGIME_BULLISH:
         CheckBullishGrid();
         break;

      case REGIME_NEUTRAL:
         CheckNeutralGrid();
         break;
   }
}

//+------------------------------------------------------------------+
//| Detect Dominant Regime                                            |
//+------------------------------------------------------------------+
ENUM_REGIME_TYPE DetectDominantRegime()
{
   double bearishConf = g_bearishManager.CalculateRegimeConfidence();
   double bullishConf = g_bullishManager.CalculateRegimeConfidence();
   double neutralConf = g_neutralManager.CalculateRegimeConfidence();

   // Only consider if above threshold
   bool bearishValid = (bearishConf >= InpConfidenceThreshold);
   bool bullishValid = (bullishConf >= InpConfidenceThreshold);
   bool neutralValid = (neutralConf >= InpConfidenceThreshold);

   // Find highest confidence
   if(bearishValid && bearishConf >= bullishConf && bearishConf >= neutralConf)
      return REGIME_BEARISH;

   if(bullishValid && bullishConf >= bearishConf && bullishConf >= neutralConf)
      return REGIME_BULLISH;

   if(neutralValid)
      return REGIME_NEUTRAL;

   // Default to neutral if none meet threshold
   return REGIME_NEUTRAL;
}

//+------------------------------------------------------------------+
//| Check Global Risk Limits                                          |
//+------------------------------------------------------------------+
bool CheckGlobalRiskLimits()
{
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(currentBalance, currentEquity);

   // Use high water mark
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
      Print("ORCHESTRATOR: Daily reset - New baseline: $", DoubleToString(dailyStartBalance, 2));
   }

   // Daily DD check
   if(dailyStartBalance > 0)
   {
      double dailyDD = ((dailyStartBalance - current) / dailyStartBalance) * 100;
      if(dailyDD >= InpDailyDDLimit)
      {
         Print("ORCHESTRATOR: BLOCKED - Daily DD limit reached: ", DoubleToString(dailyDD, 2), "%");
         return false;
      }
   }

   // Max DD check
   if(highWaterMark > 0)
   {
      double maxDD = ((highWaterMark - current) / highWaterMark) * 100;
      if(maxDD >= InpMaxDDLimit)
      {
         Print("ORCHESTRATOR: BLOCKED - Max DD limit reached: ", DoubleToString(maxDD, 2), "%");
         return false;
      }
   }

   return true;
}

//+------------------------------------------------------------------+
//| Count All Open Orders Across All Experts                          |
//+------------------------------------------------------------------+
int CountAllOpenOrders()
{
   int count = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionSelectByTicket(PositionGetTicket(i)))
      {
         int magic = (int)PositionGetInteger(POSITION_MAGIC);

         if(magic == InpMagicBearish || magic == InpMagicBullish || magic == InpMagicNeutral)
         {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol)
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

   int copied = CopyRates(_Symbol, PERIOD_H1, 0, 24, rates);
   if(copied < 24) return;

   g_rangeHigh = rates[0].high;
   g_rangeLow = rates[0].low;

   for(int i = 1; i < copied; i++)
   {
      if(rates[i].high > g_rangeHigh) g_rangeHigh = rates[i].high;
      if(rates[i].low < g_rangeLow) g_rangeLow = rates[i].low;
   }

   Print("ORCHESTRATOR: Range detected - High: ", g_rangeHigh, " | Low: ", g_rangeLow);
}

//+------------------------------------------------------------------+
//| Check BEARISH Grid Opportunity                                    |
//+------------------------------------------------------------------+
void CheckBearishGrid()
{
   if(!g_bearishManager.ShouldOpenGrid(-1))
      return;

   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr = g_bearishManager.GetATR(14);
   if(atr <= 0) return;

   double gridSpacing = atr * InpGridSpacingAtr;

   if(g_bearishLevel == 0)
   {
      if(g_bearishManager.OpenGridOrder(-1, 1))
      {
         g_bearishLevel = 1;
         g_lastBearishPrice = currentPrice;
         g_totalTrades++;
         Print("ORCHESTRATOR: BEARISH SELL opened at level 1");
      }
   }
   else
   {
      double priceMove = currentPrice - g_lastBearishPrice;
      if(priceMove >= gridSpacing && g_bearishLevel < InpMaxOrdersPerExpert)
      {
         int nextLevel = g_bearishLevel + 1;
         if(g_bearishManager.OpenGridOrder(-1, nextLevel))
         {
            g_bearishLevel = nextLevel;
            g_lastBearishPrice = currentPrice;
            g_totalTrades++;
            Print("ORCHESTRATOR: BEARISH SELL opened at level ", nextLevel);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check BULLISH Grid Opportunity                                    |
//+------------------------------------------------------------------+
void CheckBullishGrid()
{
   if(!g_bullishManager.ShouldOpenGrid(1))
      return;

   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double atr = g_bullishManager.GetATR(14);
   if(atr <= 0) return;

   double gridSpacing = atr * InpGridSpacingAtr;

   if(g_bullishLevel == 0)
   {
      if(g_bullishManager.OpenGridOrder(1, 1))
      {
         g_bullishLevel = 1;
         g_lastBullishPrice = currentPrice;
         g_totalTrades++;
         Print("ORCHESTRATOR: BULLISH BUY opened at level 1");
      }
   }
   else
   {
      double priceMove = g_lastBullishPrice - currentPrice;
      if(priceMove >= gridSpacing && g_bullishLevel < InpMaxOrdersPerExpert)
      {
         int nextLevel = g_bullishLevel + 1;
         if(g_bullishManager.OpenGridOrder(1, nextLevel))
         {
            g_bullishLevel = nextLevel;
            g_lastBullishPrice = currentPrice;
            g_totalTrades++;
            Print("ORCHESTRATOR: BULLISH BUY opened at level ", nextLevel);
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

   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr = g_neutralManager.GetATR(14);
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
            Print("ORCHESTRATOR: NEUTRAL BUY opened at level ", level);
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
            Print("ORCHESTRATOR: NEUTRAL SELL opened at level ", level);
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
   // Future: Add dashboard and manual controls
}

//+------------------------------------------------------------------+
//| Timer Function (if needed)                                        |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Future: Periodic reporting
}
//+------------------------------------------------------------------+
