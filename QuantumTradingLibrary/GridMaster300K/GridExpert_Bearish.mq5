//+------------------------------------------------------------------+
//|                                        GridExpert_Bearish.mq5    |
//|                              GridMaster 300K - BEARISH Expert    |
//|                       Prop Firm $300K Challenge - Conservative   |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property version   "2.00"
#property description "BEARISH Regime Grid Expert"
#property description "+12 Compression Boost | 80% Confidence Threshold"
#property description "Hidden Dynamic ATR-based SL/TP"
#property description "Trailing Stop + Break-Even Enabled"

//--- Include Core Grid Logic
#include "GridCore.mqh"

//+------------------------------------------------------------------+
//| Input Parameters                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input int      InpMagicNumber       = 300001;    // Magic Number (BEARISH)
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
input double   InpConfidenceThreshold = 0.22;    // Confidence Threshold (22%)
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
int            g_currentGridLevel = 0;
double         g_lastGridPrice = 0;

//+------------------------------------------------------------------+
//| Expert Initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================");
   Print("GRID EXPERT: BEARISH REGIME");
   Print("Magic Number: ", InpMagicNumber);
   Print("Compression Boost: +", InpCompressionBoost);
   Print("Confidence Threshold: ", InpConfidenceThreshold * 100, "%");
   Print("Risk Management: ", InpRewardRatio, ":1 R:R");
   Print("SL: ", InpSLAtrMultiplier, "x ATR | TP: ", InpTPAtrMultiplier, "x ATR");
   Print("Hidden SL/TP: ", InpDynamicHiddenSLTP ? "ENABLED" : "DISABLED");
   Print("Trailing Stop: ", InpTrailingStop ? "ENABLED" : "DISABLED");
   Print("Break-Even: ", InpBreakEven ? "ENABLED" : "DISABLED");
   Print("Max Orders: ", InpMaxOrdersPerExpert, " per expert");
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
   if(!g_gridManager.Initialize(InpMagicNumber, _Symbol, REGIME_BEARISH))
   {
      Print("ERROR: Failed to initialize Grid Manager");
      return INIT_FAILED;
   }

   g_gridManager.SetConfig(g_config);

   Print("BEARISH Grid Expert initialized successfully");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_gridManager.Deinitialize();
   Print("BEARISH Grid Expert deinitialized. Reason: ", reason);
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

   //--- Check if we should open new grid orders
   CheckGridOpportunity();
}

//+------------------------------------------------------------------+
//| Check for Grid Trading Opportunity                                |
//+------------------------------------------------------------------+
void CheckGridOpportunity()
{
   //--- BEARISH expert only opens SELL orders
   int direction = -1; // SELL only

   //--- Check if conditions met
   if(!g_gridManager.ShouldOpenGrid(direction))
   {
      return;
   }

   //--- Get current price and ATR for grid spacing
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double atr = g_gridManager.GetATR(14);

   if(atr <= 0)
   {
      return;
   }

   double gridSpacing = atr * InpGridSpacingAtr;

   //--- Check if price has moved enough for new grid level
   if(g_currentGridLevel == 0)
   {
      //--- First order
      if(g_gridManager.OpenGridOrder(direction, 1))
      {
         g_currentGridLevel = 1;
         g_lastGridPrice = currentPrice;
         Print("BEARISH GRID: Opened first SELL at level 1");
      }
   }
   else
   {
      //--- Subsequent orders - check grid spacing
      double priceMove = currentPrice - g_lastGridPrice;

      //--- For BEARISH (SELL), we add more sells as price rises (better entry)
      if(priceMove >= gridSpacing)
      {
         int nextLevel = g_currentGridLevel + 1;

         if(nextLevel <= InpMaxOrdersPerExpert)
         {
            if(g_gridManager.OpenGridOrder(direction, nextLevel))
            {
               g_currentGridLevel = nextLevel;
               g_lastGridPrice = currentPrice;
               Print("BEARISH GRID: Opened SELL at level ", nextLevel);
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
