//+------------------------------------------------------------------+
//|                                          MultiSymbol_Launcher.mq5 |
//|                      Multi-Symbol Grid Trading Orchestrator       |
//|                      Deploys XAU, BTC, ETH to All 3 GL Accounts   |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property description "Multi-Symbol Grid Launcher for GetLeveraged"
#property description "Manages XAUUSD, BTCUSD, ETHUSD across 3 accounts"
#property description "Entropy filtering | Hidden SL/TP | ATR-based levels"
#property description "Accounts: 113328, 113326, 107245"

//--- Include shared entropy grid logic
#include "EntropyGridCore.mqh"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SELECTION ==="
input int      InpAccountSelector  = 1;           // Account (1=113328, 2=113326, 3=107245)

input group "=== SYMBOL SELECTION ==="
input bool     InpTradeXAUUSD      = true;        // Trade XAUUSD (Gold)
input bool     InpTradeBTCUSD      = true;        // Trade BTCUSD (Bitcoin)
input bool     InpTradeETHUSD      = true;        // Trade ETHUSD (Ethereum)

input group "=== LOT SIZES BY SYMBOL ==="
input double   InpXauBaseLot       = 0.01;        // XAUUSD Base Lot
input double   InpXauMaxLot        = 0.04;        // XAUUSD Max Lot
input double   InpBtcBaseLot       = 0.01;        // BTCUSD Base Lot
input double   InpBtcMaxLot        = 0.04;        // BTCUSD Max Lot
input double   InpEthBaseLot       = 0.01;        // ETHUSD Base Lot
input double   InpEthMaxLot        = 0.04;        // ETHUSD Max Lot

input group "=== GRID CONFIGURATION ==="
input int      InpMaxPositionsPerSymbol = 5;      // Max Positions Per Symbol
input double   InpRiskPercent      = 0.5;         // Risk Per Trade %

input group "=== RISK MANAGEMENT ==="
input double   InpDailyDDLimit     = 4.5;         // Daily Drawdown Limit %
input double   InpMaxDDLimit       = 9.0;         // Max Drawdown Limit %

input group "=== TIMING ==="
input int      InpCheckInterval    = 30;          // Signal Check Interval (seconds)
input bool     InpTradeEnabled     = true;        // Trading Enabled

//+------------------------------------------------------------------+
//| MAGIC NUMBER SCHEME                                               |
//+------------------------------------------------------------------+
// XAUUSD: 112xxx (Account selector as last digit)
// BTCUSD: 113xxx
// ETHUSD: 114xxx
// This allows unique identification per symbol per account

//+------------------------------------------------------------------+
//| ACCOUNT MAPPING                                                   |
//+------------------------------------------------------------------+
// Account 1: 113328 / GetLeveraged-Trade
// Account 2: 113326 / GetLeveraged-Trade
// Account 3: 107245 / GetLeveraged-Trade

//+------------------------------------------------------------------+
//| FIXED SPECIFICATIONS (All symbols share these settings)           |
//+------------------------------------------------------------------+
// Entropy Filtering: ENABLED
// Compression Boost: +12
// ATR Multipliers: SL=1.5x, TP=3.0x (3:1 reward ratio)
// Partial TP: 50% at first target
// Break-Even: At 30% of TP
// Trailing: At 50% of TP
// Hidden SL/TP: Yes (managed internally)

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
CEntropyGridManager  g_xauManager;
CEntropyGridManager  g_btcManager;
CEntropyGridManager  g_ethManager;

bool                 g_xauActive = false;
bool                 g_btcActive = false;
bool                 g_ethActive = false;

datetime             g_lastCheck = 0;
int                  g_accountId = 0;

//+------------------------------------------------------------------+
//| Get Account ID from Selector                                      |
//+------------------------------------------------------------------+
int GetAccountId(int selector)
{
   switch(selector)
   {
      case 1: return 113328;
      case 2: return 113326;
      case 3: return 107245;
      default: return 113328;
   }
}

//+------------------------------------------------------------------+
//| Check if symbol is available for trading                          |
//+------------------------------------------------------------------+
bool IsSymbolAvailable(string symbol)
{
   if(!SymbolSelect(symbol, true))
   {
      Print("WARNING: Symbol ", symbol, " not available in Market Watch");
      return false;
   }

   // Check if symbol has valid tick data
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
   {
      Print("WARNING: Cannot get tick data for ", symbol);
      return false;
   }

   if(tick.bid <= 0 || tick.ask <= 0)
   {
      Print("WARNING: Invalid prices for ", symbol);
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Expert Initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("================================================================");
   Print("MULTI-SYMBOL GRID LAUNCHER - GETLEVERAGED");
   Print("================================================================");

   // Determine account
   g_accountId = GetAccountId(InpAccountSelector);

   Print("Account: ", g_accountId, " / GetLeveraged-Trade");
   Print("Account Selector: ", InpAccountSelector);
   Print("----------------------------------------------------------------");
   Print("SYMBOLS TO TRADE:");
   Print("  - XAUUSD (Gold): ", InpTradeXAUUSD ? "ENABLED" : "DISABLED");
   Print("  - BTCUSD (Bitcoin): ", InpTradeBTCUSD ? "ENABLED" : "DISABLED");
   Print("  - ETHUSD (Ethereum): ", InpTradeETHUSD ? "ENABLED" : "DISABLED");
   Print("----------------------------------------------------------------");
   Print("ENTROPY FILTER: ENABLED (All Symbols)");
   Print("  - Confidence Threshold: 80%");
   Print("  - Compression Boost: +12");
   Print("  - Trades only in LOW entropy (predictable) markets");
   Print("----------------------------------------------------------------");
   Print("ATR-BASED LEVELS (All Symbols):");
   Print("  - Stop Loss: 1.5x ATR");
   Print("  - Take Profit: 3.0x ATR (3:1 reward ratio)");
   Print("  - Partial TP: 50% at first target");
   Print("  - Break-Even: At 30% of TP distance");
   Print("  - Trailing Stop: At 50% of TP distance");
   Print("----------------------------------------------------------------");
   Print("HIDDEN SL/TP: ENABLED (broker sees no stops)");
   Print("================================================================");

   // Initialize XAUUSD
   if(InpTradeXAUUSD && IsSymbolAvailable("XAUUSD"))
   {
      int xauMagic = 112000 + InpAccountSelector;
      if(g_xauManager.Initialize("XAUUSD", xauMagic, g_accountId))
      {
         g_xauManager.SetLotSizes(InpXauBaseLot, InpXauMaxLot);
         g_xauManager.SetMaxPositions(InpMaxPositionsPerSymbol);
         g_xauManager.SetDrawdownLimits(InpDailyDDLimit, InpMaxDDLimit);
         g_xauManager.SetRiskPercent(InpRiskPercent);
         g_xauActive = true;
         Print("XAUUSD Manager initialized | Magic: ", xauMagic);
      }
      else
      {
         Print("ERROR: Failed to initialize XAUUSD manager");
      }
   }

   // Initialize BTCUSD
   if(InpTradeBTCUSD && IsSymbolAvailable("BTCUSD"))
   {
      int btcMagic = 113000 + InpAccountSelector;
      if(g_btcManager.Initialize("BTCUSD", btcMagic, g_accountId))
      {
         g_btcManager.SetLotSizes(InpBtcBaseLot, InpBtcMaxLot);
         g_btcManager.SetMaxPositions(InpMaxPositionsPerSymbol);
         g_btcManager.SetDrawdownLimits(InpDailyDDLimit, InpMaxDDLimit);
         g_btcManager.SetRiskPercent(InpRiskPercent);
         g_btcActive = true;
         Print("BTCUSD Manager initialized | Magic: ", btcMagic);
      }
      else
      {
         Print("ERROR: Failed to initialize BTCUSD manager");
      }
   }

   // Initialize ETHUSD
   if(InpTradeETHUSD && IsSymbolAvailable("ETHUSD"))
   {
      int ethMagic = 114000 + InpAccountSelector;
      if(g_ethManager.Initialize("ETHUSD", ethMagic, g_accountId))
      {
         g_ethManager.SetLotSizes(InpEthBaseLot, InpEthMaxLot);
         g_ethManager.SetMaxPositions(InpMaxPositionsPerSymbol);
         g_ethManager.SetDrawdownLimits(InpDailyDDLimit, InpMaxDDLimit);
         g_ethManager.SetRiskPercent(InpRiskPercent);
         g_ethActive = true;
         Print("ETHUSD Manager initialized | Magic: ", ethMagic);
      }
      else
      {
         Print("ERROR: Failed to initialize ETHUSD manager");
      }
   }

   // Summary
   int activeCount = (g_xauActive ? 1 : 0) + (g_btcActive ? 1 : 0) + (g_ethActive ? 1 : 0);

   if(activeCount == 0)
   {
      Print("ERROR: No symbols could be initialized!");
      return INIT_FAILED;
   }

   Print("================================================================");
   Print("INITIALIZATION COMPLETE");
   Print("Active Symbols: ", activeCount, "/3");
   Print("Account Balance: $", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   Print("================================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_xauActive) g_xauManager.Deinitialize();
   if(g_btcActive) g_btcManager.Deinitialize();
   if(g_ethActive) g_ethManager.Deinitialize();

   Print("Multi-Symbol Launcher stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert Tick Function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // ALWAYS manage positions on every tick (critical for hidden SL/TP)
   if(g_xauActive) g_xauManager.OnTick();
   if(g_btcActive) g_btcManager.OnTick();
   if(g_ethActive) g_ethManager.OnTick();

   // Check interval for new entries
   if(TimeCurrent() - g_lastCheck < InpCheckInterval) return;
   g_lastCheck = TimeCurrent();

   // Trading check
   if(!InpTradeEnabled) return;

   // Check permissions
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      static datetime lastWarn = 0;
      if(TimeCurrent() - lastWarn > 300)
      {
         Print("WARNING: MQL trading not allowed - check EA properties");
         lastWarn = TimeCurrent();
      }
      return;
   }

   // Process each symbol for new entries
   if(g_xauActive) g_xauManager.ProcessTick();
   if(g_btcActive) g_btcManager.ProcessTick();
   if(g_ethActive) g_ethManager.ProcessTick();

   // Periodic status log
   static datetime lastLog = 0;
   if(TimeCurrent() - lastLog > 300) // Every 5 minutes
   {
      PrintStatus();
      lastLog = TimeCurrent();
   }
}

//+------------------------------------------------------------------+
//| Print Comprehensive Status                                        |
//+------------------------------------------------------------------+
void PrintStatus()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin = AccountInfoDouble(ACCOUNT_MARGIN);
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);

   Print("================================================================");
   Print("MULTI-SYMBOL STATUS REPORT");
   Print("================================================================");
   Print("Account: ", g_accountId, " | Balance: $", DoubleToString(balance, 2),
         " | Equity: $", DoubleToString(equity, 2));
   Print("Margin: $", DoubleToString(margin, 2),
         " | Free Margin: $", DoubleToString(freeMargin, 2));
   Print("----------------------------------------------------------------");

   if(g_xauActive)
   {
      Print("XAUUSD: ", g_xauManager.GetStatusString());
   }

   if(g_btcActive)
   {
      Print("BTCUSD: ", g_btcManager.GetStatusString());
   }

   if(g_ethActive)
   {
      Print("ETHUSD: ", g_ethManager.GetStatusString());
   }

   Print("================================================================");
}

//+------------------------------------------------------------------+
//| Chart Event Handler                                               |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
   // Future: Add dashboard with symbol toggles
}

//+------------------------------------------------------------------+
//| Timer Function                                                    |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Future: Periodic comprehensive reporting
}
//+------------------------------------------------------------------+
