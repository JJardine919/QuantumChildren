//+------------------------------------------------------------------+
//|                                             ETHUSD_GridTrader.mq5 |
//|                      Ethereum Grid Trading EA with Entropy Filter |
//|                      GetLeveraged Accounts - 3 Account Deployment |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property description "ETHUSD Grid Trading EA with Entropy Filtering"
#property description "Trades only in predictable (low entropy) market conditions"
#property description "Hidden SL/TP | ATR-based levels | Partial TP at 50%"
#property description "For GetLeveraged account: 107245 (GL_3 ACTIVE)"

//--- Include shared entropy grid logic
#include "EntropyGridCore.mqh"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT CONFIGURATION ==="
input int      InpAccountSelector  = 3;           // Account (1=113328 DISABLED, 2=113326 DISABLED, 3=107245 ACTIVE)
input int      InpMagicBase        = 114000;      // Magic Number Base (ETH uses 114xxx)

input group "=== TRADING CONFIGURATION ==="
input double   InpBaseLot          = 0.01;        // Base Lot Size
input double   InpMaxLot           = 0.04;        // Max Lot Size
input int      InpMaxPositions     = 5;           // Max Grid Positions
input double   InpRiskPercent      = 0.5;         // Risk Per Trade %

input group "=== RISK MANAGEMENT ==="
input double   InpDailyDDLimit     = 4.5;         // Daily Drawdown Limit %
input double   InpMaxDDLimit       = 9.0;         // Max Drawdown Limit %
input double   InpMaxSLDollars     = 50.0;        // Max SL Dollar Loss Per Position

input group "=== SPREAD FILTER ==="
input int      InpMaxSpreadPoints  = 300;         // Max spread to allow trade (points) - ETH typical

input group "=== TIMING ==="
input int      InpCheckInterval    = 30;          // Signal Check Interval (seconds)
input bool     InpTradeEnabled     = true;        // Trading Enabled

//+------------------------------------------------------------------+
//| FIXED SPECIFICATIONS (Hard-coded per requirements)                |
//+------------------------------------------------------------------+
// Symbol: ETHUSD (Ethereum)
// Entropy Filtering: ENABLED
// Compression Boost: +12
// ATR Multipliers: SL=1.5x, TP=3.0x (3:1 reward ratio)
// Partial TP: 50% at first target
// Break-Even: At 30% of TP
// Trailing: At 50% of TP
// Hidden SL/TP: Yes (managed internally)

//+------------------------------------------------------------------+
//| ACCOUNT MAPPING                                                   |
//+------------------------------------------------------------------+
// Account 1: 113328 / GetLeveraged-Trade  ** DISABLED **
// Account 2: 113326 / GetLeveraged-Trade  ** DISABLED **
// Account 3: 107245 / GetLeveraged-Trade  ** ACTIVE **

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
CEntropyGridManager  g_gridManager;
datetime             g_lastCheck = 0;
int                  g_accountId = 0;
int                  g_magic = 0;

//+------------------------------------------------------------------+
//| Get Account ID from Selector                                      |
//| NOTE: Accounts 113328 (GL_2) and 113326 (GL_1) are DISABLED.     |
//|       Only 107245 (GL_3) is active. Selecting a disabled account  |
//|       will print a warning and return -1.                          |
//+------------------------------------------------------------------+
int GetAccountId(int selector)
{
   switch(selector)
   {
      case 1: // 113328 - GL_2 -- DISABLED
         Print("CRITICAL WARNING: Account 113328 (GL_2) is DISABLED! Do NOT trade on this account.");
         Print("CRITICAL WARNING: Change InpAccountSelector to 3 (107245) immediately.");
         return -1;
      case 2: // 113326 - GL_1 -- DISABLED
         Print("CRITICAL WARNING: Account 113326 (GL_1) is DISABLED! Do NOT trade on this account.");
         Print("CRITICAL WARNING: Change InpAccountSelector to 3 (107245) immediately.");
         return -1;
      case 3: return 107245;  // GL_3 -- ACTIVE
      default:
         Print("CRITICAL WARNING: Unknown account selector ", selector, ". Defaulting to 107245 (GL_3).");
         return 107245;
   }
}

//+------------------------------------------------------------------+
//| Expert Initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("========================================================");
   Print("ETHUSD GRID TRADER - ENTROPY FILTERED");
   Print("========================================================");

   // Determine account and magic number
   g_accountId = GetAccountId(InpAccountSelector);
   g_magic = InpMagicBase + InpAccountSelector;

   Print("Symbol: ETHUSD (Ethereum)");
   Print("Account: ", g_accountId, " / GetLeveraged-Trade");
   Print("Magic: ", g_magic);
   Print("--------------------------------------------------------");
   Print("ENTROPY FILTER: ENABLED");
   Print("  - Confidence Threshold: 22%");
   Print("  - Compression Boost: +12");
   Print("  - Trades only in LOW entropy (predictable) markets");
   Print("--------------------------------------------------------");
   Print("ATR-BASED LEVELS:");
   Print("  - Stop Loss: 1.5x ATR");
   Print("  - Take Profit: 3.0x ATR (3:1 reward ratio)");
   Print("  - Partial TP: 50% at first target");
   Print("  - Break-Even: At 30% of TP distance");
   Print("  - Trailing Stop: At 50% of TP distance");
   Print("--------------------------------------------------------");
   Print("HIDDEN SL/TP: ENABLED (broker sees no stops)");
   Print("========================================================");

   // Initialize grid manager
   if(!g_gridManager.Initialize("ETHUSD", g_magic, g_accountId))
   {
      Print("ERROR: Failed to initialize grid manager");
      return INIT_FAILED;
   }

   // Configure
   g_gridManager.SetLotSizes(InpBaseLot, InpMaxLot);
   g_gridManager.SetMaxPositions(InpMaxPositions);
   g_gridManager.SetDrawdownLimits(InpDailyDDLimit, InpMaxDDLimit);
   g_gridManager.SetRiskPercent(InpRiskPercent);
   g_gridManager.SetMaxSpreadPoints(InpMaxSpreadPoints);
   g_gridManager.SetMaxSLDollars(InpMaxSLDollars);

   Print("ETHUSD Grid Trader initialized successfully");
   Print("Account Balance: $", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   Print("========================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert Deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_gridManager.Deinitialize();
   Print("ETHUSD Grid Trader stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert Tick Function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // ALWAYS manage positions on every tick (critical for hidden SL/TP)
   g_gridManager.OnTick();

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

   // Process for new entries
   g_gridManager.ProcessTick();

   // Periodic status log
   static datetime lastLog = 0;
   if(TimeCurrent() - lastLog > 300) // Every 5 minutes
   {
      string status = g_gridManager.GetStatusString();
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);

      Print("STATUS: ", status);
      Print("  Balance: $", DoubleToString(balance, 2),
            " | Equity: $", DoubleToString(equity, 2));
      lastLog = TimeCurrent();
   }
}
//+------------------------------------------------------------------+
