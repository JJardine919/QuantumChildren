//+------------------------------------------------------------------+
//|                                        GridExpert_Neutral.mq5    |
//|                              GridMaster 300K - NEUTRAL Expert    |
//|                       Prop Firm $300K Challenge - Conservative   |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property version   "2.00"
#property description "NEUTRAL Regime Grid Expert (Range Trading)"
#property description "+12 Compression Boost | 80% Confidence Threshold"
#property description "Hidden Dynamic ATR-based SL/TP"
#property description "Trailing Stop + Break-Even Enabled"

//--- Include Core Grid Logic
#include "GridCore.mqh"

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input int      InpMagicNumber       = 300003;    // Magic Number (NEUTRAL)
input double   InpAccountBalance    = 300000;    // Account Balance
input double   InpDailyDDLimit      = 5.0;       // Daily Drawdown Limit %
input double   InpMaxDDLimit        = 10.0;      // Max Drawdown Limit %

input group "=== RISK MANAGEMENT ==="
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
input int      InpMaxTotalOrders    = 30;        // Max Total Orders (all experts)
input double   InpGridSpacingAtr    = 0.5;       // Grid Spacing (ATR multiples)

input group "=== COMPRESSION FILTER ==="
input double   InpConfidenceThreshold = 0.80;    // Confidence Threshold (80%)
input int      InpCompressionBoost  = 12;        // Compression Boost

input group "=== TRADING ==="
input bool     InpTradeEnabled      = true;      // Trading Enabled
input int      InpCheckInterval     = 30;        // Check Interval (seconds)

//+------------------------------------------------------------------+
//| Global Variables                                                  |
//+------------------------------------------------------------------+
CGridManager   g_gridManager;
GridConfig     g_config;
datetime       g_lastCheck = 0;
int            g_currentBuyLevel = 0;
int            g_currentSellLevel = 0;
double         g_lastBuyPrice = 0;
double         g_lastSellPrice = 0;
double         g_rangeHigh = 0;
double         g_rangeLow = 0;

//+------------------------------------------------------------------+
//| Expert Initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("GRID EXPERT: NEUTRAL REGIME (RANGE)");
   Print("Magic Number: ", InpMagicNumber);
   Print("Compression Boost: +", InpCompressionBoost);
   Print("Confidence Threshold: ", InpConfidenceThreshold * 100, "%");
   Print("Risk Management: ", InpRewardRatio, ":1 R:R");
   Print("SL: ", InpSLAtrMultiplier, "x ATR | TP: ", InpTPAtrMultiplier, "x ATR");
   Print("Hidden SL/TP: ", InpDynamicHiddenSLTP ? "ENABLED" : "DISABLED");
   Print("Trailing Stop: ", InpTrailingStop ? "ENABLED" : "DISABLED");
   Print("Break-Even: ", InpBreakEven ? "ENABLED" : "DISABLED");
   Print("Max Orders: ", InpMaxOrdersPerExpert, " per expert");
   Print("Direction: BOTH (Range Trading)");
   Print("Account: $", DoubleToString(InpAccountBalance, 2));
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

   //--- Initialize Grid Manager
   if(!g_gridManager.Initialize(InpMagicNumber, _Symbol, REGIME_NEUTRAL))
   {
      Print("ERROR: Failed to initialize Grid Manager");
      return INIT_FAILED;
   }

   g_gridManager.SetConfig(g_config);

   //--- Initialize range detection
   DetectRange();

   Print("NEUTRAL Grid Expert initialized successfully");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_gridManager.Deinitialize();
   Print("NEUTRAL Grid Expert deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert Tick Function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check interval
   if(TimeCurrent() - g_lastCheck < InpCheckInterval)
   {
      //--- Still manage positions every tick
      g_gridManager.ManageOpenPositions();
      return;
   }
   g_lastCheck = TimeCurrent();

   //--- Process tick (risk checks, position management)
   if(!g_gridManager.ProcessTick())
   {
      return; // Risk limits exceeded
   }

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

   //--- Check if we should open new grid orders
   CheckGridOpportunity();
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

   Print("NEUTRAL: Range detected - High: ", g_rangeHigh, " | Low: ", g_rangeLow);
}

//+------------------------------------------------------------------+
//| Check for Grid Trading Opportunity                                |
//+------------------------------------------------------------------+
void CheckGridOpportunity()
{
   //--- NEUTRAL expert trades BOTH directions (range trading)
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr = g_gridManager.GetATR(14);

   if(atr <= 0)
   {
      return;
   }

   double gridSpacing = atr * InpGridSpacingAtr;
   double rangeMid = (g_rangeHigh + g_rangeLow) / 2;
   double rangeQuarter = (g_rangeHigh - g_rangeLow) / 4;

   //--- Total orders check (split between buy and sell)
   int totalOrders = g_gridManager.CountOpenOrders();
   int maxPerSide = InpMaxOrdersPerExpert / 2; // 5 buys + 5 sells = 10 total

   //--- Check for BUY opportunities (lower part of range)
   if(currentPrice < rangeMid - rangeQuarter)
   {
      if(g_currentBuyLevel < maxPerSide && totalOrders < InpMaxOrdersPerExpert)
      {
         if(g_gridManager.ShouldOpenGrid(0)) // 0 = NEUTRAL allows both
         {
            bool shouldOpen = false;

            if(g_currentBuyLevel == 0)
            {
               shouldOpen = true;
            }
            else
            {
               double priceMove = g_lastBuyPrice - currentPrice;
               shouldOpen = (priceMove >= gridSpacing);
            }

            if(shouldOpen)
            {
               int level = g_currentBuyLevel + 1;
               if(g_gridManager.OpenGridOrder(1, level)) // 1 = BUY
               {
                  g_currentBuyLevel = level;
                  g_lastBuyPrice = currentPrice;
                  Print("NEUTRAL GRID: Opened BUY at level ", level, " (range low zone)");
               }
            }
         }
      }
   }

   //--- Check for SELL opportunities (upper part of range)
   if(currentPrice > rangeMid + rangeQuarter)
   {
      if(g_currentSellLevel < maxPerSide && totalOrders < InpMaxOrdersPerExpert)
      {
         if(g_gridManager.ShouldOpenGrid(0)) // 0 = NEUTRAL allows both
         {
            bool shouldOpen = false;

            if(g_currentSellLevel == 0)
            {
               shouldOpen = true;
            }
            else
            {
               double priceMove = currentPrice - g_lastSellPrice;
               shouldOpen = (priceMove >= gridSpacing);
            }

            if(shouldOpen)
            {
               int level = g_currentSellLevel + 1;
               if(g_gridManager.OpenGridOrder(-1, level)) // -1 = SELL
               {
                  g_currentSellLevel = level;
                  g_lastSellPrice = currentPrice;
                  Print("NEUTRAL GRID: Opened SELL at level ", level, " (range high zone)");
               }
            }
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
   // Future: Add chart buttons for manual control
}
//+------------------------------------------------------------------+
