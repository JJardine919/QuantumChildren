//+------------------------------------------------------------------+
//|                                               BG_Diagnostic.mq5  |
//|                  Blue Guardian Connection & Trading Diagnostic   |
//|                           Run this to identify WHY trades fail   |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property strict
#property description "DIAGNOSTIC TOOL - Checks why trades are not being placed"
#property description "Attach to chart and check Experts tab for output"

input int TestMagic = 999999;  // Test Magic Number

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("================================================");
   Print("  BLUE GUARDIAN DIAGNOSTIC TOOL");
   Print("================================================");

   // 1. Account Information
   Print("");
   Print("=== ACCOUNT INFO ===");
   Print("Account Number: ", AccountInfoInteger(ACCOUNT_LOGIN));
   Print("Account Name: ", AccountInfoString(ACCOUNT_NAME));
   Print("Account Server: ", AccountInfoString(ACCOUNT_SERVER));
   Print("Account Company: ", AccountInfoString(ACCOUNT_COMPANY));
   Print("Account Currency: ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Account Balance: $", DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
   Print("Account Equity: $", DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2));
   Print("Account Margin: $", DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN), 2));
   Print("Account Free Margin: $", DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2));
   Print("Account Leverage: 1:", AccountInfoInteger(ACCOUNT_LEVERAGE));

   // 2. Trading Permissions
   Print("");
   Print("=== TRADING PERMISSIONS ===");
   bool tradeAllowed = AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   bool tradeExpert = AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   bool terminalTrade = TerminalInfoInteger(TERMINAL_TRADE_ALLOWED);
   bool connected = TerminalInfoInteger(TERMINAL_CONNECTED);
   bool dlls = TerminalInfoInteger(TERMINAL_DLLS_ALLOWED);
   bool mqlTrade = MQLInfoInteger(MQL_TRADE_ALLOWED);

   Print("Account Trade Allowed: ", tradeAllowed ? "YES" : "NO <<<< PROBLEM!");
   Print("Account Trade Expert: ", tradeExpert ? "YES" : "NO <<<< PROBLEM!");
   Print("Terminal Trade Allowed: ", terminalTrade ? "YES" : "NO <<<< PROBLEM!");
   Print("Terminal Connected: ", connected ? "YES" : "NO <<<< PROBLEM!");
   Print("DLLs Allowed: ", dlls ? "YES" : "NO");
   Print("MQL Trade Allowed: ", mqlTrade ? "YES" : "NO <<<< PROBLEM!");

   if(!tradeAllowed)
      Print(">>> FIX: Enable 'Allow Algo Trading' in account settings or server has disabled trading");
   if(!tradeExpert)
      Print(">>> FIX: Enable 'Allow Automated Trading' button (Ctrl+E) in terminal");
   if(!terminalTrade)
      Print(">>> FIX: Click the 'Algo Trading' button in toolbar");
   if(!mqlTrade)
      Print(">>> FIX: In EA properties, check 'Allow Algo Trading' checkbox");

   // 3. Symbol Information
   Print("");
   Print("=== SYMBOL INFO ===");
   string sym = _Symbol;
   Print("Current Symbol: ", sym);
   Print("Symbol Description: ", SymbolInfoString(sym, SYMBOL_DESCRIPTION));

   bool symbolTrade = SymbolInfoInteger(sym, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED;
   Print("Symbol Trade Mode: ", symbolTrade ? "ENABLED" : "DISABLED <<<< PROBLEM!");

   long tradeMode = SymbolInfoInteger(sym, SYMBOL_TRADE_MODE);
   switch((int)tradeMode)
   {
      case SYMBOL_TRADE_MODE_DISABLED: Print("  Trade Mode: DISABLED - Cannot trade this symbol!"); break;
      case SYMBOL_TRADE_MODE_LONGONLY: Print("  Trade Mode: LONG ONLY"); break;
      case SYMBOL_TRADE_MODE_SHORTONLY: Print("  Trade Mode: SHORT ONLY"); break;
      case SYMBOL_TRADE_MODE_CLOSEONLY: Print("  Trade Mode: CLOSE ONLY - Cannot open new positions!"); break;
      case SYMBOL_TRADE_MODE_FULL: Print("  Trade Mode: FULL"); break;
   }

   double minLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   Print("Min Lot: ", minLot);
   Print("Max Lot: ", maxLot);
   Print("Lot Step: ", lotStep);

   double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   Print("Tick Value: ", tickValue);
   Print("Tick Size: ", tickSize);

   long filling = SymbolInfoInteger(sym, SYMBOL_FILLING_MODE);
   Print("Filling Mode Flags: ", filling);
   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK) Print("  - FOK Supported");
   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC) Print("  - IOC Supported");

   // 4. Current Price
   Print("");
   Print("=== CURRENT PRICES ===");
   MqlTick tick;
   if(SymbolInfoTick(sym, tick))
   {
      Print("Bid: ", tick.bid);
      Print("Ask: ", tick.ask);
      Print("Spread: ", (tick.ask - tick.bid) / SymbolInfoDouble(sym, SYMBOL_POINT), " points");
   }
   else
   {
      Print("ERROR: Cannot get tick data! <<<< PROBLEM!");
   }

   // 5. Try a Test Order (dry run)
   Print("");
   Print("=== TEST ORDER (DRY RUN) ===");
   TestOrder();

   Print("");
   Print("================================================");
   Print("  DIAGNOSTIC COMPLETE - Check issues above");
   Print("================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Test order - check if order would be accepted                      |
//+------------------------------------------------------------------+
void TestOrder()
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
   {
      Print("Cannot get tick - no test possible");
      return;
   }

   MqlTradeRequest request;
   MqlTradeCheckResult checkResult;
   ZeroMemory(request);
   ZeroMemory(checkResult);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   request.type = ORDER_TYPE_BUY;
   request.price = tick.ask;
   request.sl = 0;
   request.tp = 0;
   request.deviation = 50;
   request.magic = TestMagic;
   request.comment = "TEST_ORDER";
   request.type_filling = GetFilling();
   request.type_time = ORDER_TIME_GTC;

   Print("Testing BUY order:");
   Print("  Symbol: ", request.symbol);
   Print("  Volume: ", request.volume);
   Print("  Price: ", request.price);
   Print("  Filling: ", EnumToString(request.type_filling));

   bool checkOK = OrderCheck(request, checkResult);

   Print("OrderCheck Result: ", checkOK ? "PASSED" : "FAILED");
   Print("  Return Code: ", checkResult.retcode);
   Print("  Comment: ", checkResult.comment);
   Print("  Balance Impact: ", checkResult.balance);
   Print("  Margin Required: ", checkResult.margin);
   Print("  Free Margin After: ", checkResult.margin_free);

   if(!checkOK)
   {
      Print("");
      Print(">>> ORDER WOULD BE REJECTED <<<");

      switch((int)checkResult.retcode)
      {
         case TRADE_RETCODE_MARKET_CLOSED:
            Print("REASON: Market is CLOSED");
            break;
         case TRADE_RETCODE_NO_MONEY:
            Print("REASON: Insufficient funds. Need: ", checkResult.margin, " Have: ", AccountInfoDouble(ACCOUNT_MARGIN_FREE));
            break;
         case TRADE_RETCODE_INVALID_VOLUME:
            Print("REASON: Invalid volume. Min: ", SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));
            break;
         case TRADE_RETCODE_INVALID_PRICE:
            Print("REASON: Invalid price");
            break;
         case TRADE_RETCODE_INVALID_STOPS:
            Print("REASON: Invalid stops");
            break;
         case TRADE_RETCODE_TRADE_DISABLED:
            Print("REASON: Trading is DISABLED for this account!");
            break;
         case TRADE_RETCODE_INVALID_FILL:
            Print("REASON: Invalid fill mode - try different filling type");
            break;
         default:
            Print("REASON: Unknown - code ", checkResult.retcode);
      }
   }
   else
   {
      Print(">>> ORDER WOULD BE ACCEPTED - Trading should work!");
   }
}

//+------------------------------------------------------------------+
//| Get best filling mode                                              |
//+------------------------------------------------------------------+
ENUM_ORDER_TYPE_FILLING GetFilling()
{
   uint filling = (uint)SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);

   if((filling & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)
      return ORDER_FILLING_FOK;

   if((filling & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC)
      return ORDER_FILLING_IOC;

   return ORDER_FILLING_RETURN;
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Do nothing - diagnostic only runs once at init
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Diagnostic tool removed");
}
//+------------------------------------------------------------------+
