//+------------------------------------------------------------------+
//|                                               BG_ForceTrade.mq5  |
//|                         FORCE TRADE TEST - Places ONE trade NOW  |
//|                         Use to verify trading is actually working |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property version   "1.00"
#property strict
#property description "FORCE TRADE - Places ONE BUY immediately"
#property description "Use to test if terminal can actually trade"
#property description "REMOVE AFTER TEST!"

input int MagicNumber = 123456;  // Test Magic Number
input double LotSize = 0.01;     // Test Lot Size (minimum)

bool g_tradePlaced = false;

//+------------------------------------------------------------------+
//| Expert initialization                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("===========================================");
   Print("  FORCE TRADE TEST");
   Print("===========================================");
   Print("This EA will place ONE BUY trade immediately");
   Print("to verify that trading is working.");
   Print("");
   Print("Magic: ", MagicNumber);
   Print("Lot: ", LotSize);
   Print("Symbol: ", _Symbol);
   Print("");

   // Permission check
   Print("--- PERMISSIONS ---");
   bool ok = true;

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      Print("FAIL: MQL Trade not allowed - check EA properties!");
      ok = false;
   }
   else Print("PASS: MQL Trade allowed");

   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      Print("FAIL: Terminal Trade not allowed - press Ctrl+E!");
      ok = false;
   }
   else Print("PASS: Terminal Trade allowed");

   if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))
   {
      Print("FAIL: Account Trade not allowed - server disabled!");
      ok = false;
   }
   else Print("PASS: Account Trade allowed");

   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      Print("FAIL: Not connected to server!");
      ok = false;
   }
   else Print("PASS: Connected to server");

   Print("");

   if(ok)
   {
      Print("All permissions OK - will attempt trade on first tick...");
   }
   else
   {
      Print("PERMISSION ISSUES - trade will likely fail!");
   }

   Print("===========================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   if(g_tradePlaced) return;  // Only trade once

   Print("");
   Print(">>> ATTEMPTING FORCE TRADE NOW <<<");
   Print("");

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
   {
      Print("ERROR: Cannot get tick data!");
      return;
   }

   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lot = MathMax(LotSize, minLot);
   lot = NormalizeDouble(lot, 2);

   Print("Price: ", tick.ask);
   Print("Lot: ", lot, " (min: ", minLot, ")");

   // Build request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = ORDER_TYPE_BUY;
   request.price = tick.ask;
   request.sl = 0;
   request.tp = 0;
   request.deviation = 50;
   request.magic = MagicNumber;
   request.comment = "FORCE_TEST";
   request.type_filling = GetFilling();
   request.type_time = ORDER_TIME_GTC;

   Print("Filling mode: ", EnumToString(request.type_filling));

   // Check first
   MqlTradeCheckResult checkResult;
   bool checkOK = OrderCheck(request, checkResult);

   Print("");
   Print("OrderCheck: ", checkOK ? "PASSED" : "FAILED");
   Print("  Retcode: ", checkResult.retcode);
   Print("  Comment: ", checkResult.comment);
   Print("  Margin: ", checkResult.margin);
   Print("  Free: ", checkResult.margin_free);

   if(!checkOK)
   {
      Print("");
      Print("!!! ORDER CHECK FAILED - Trade will not be sent !!!");
      DecodeError(checkResult.retcode);
      g_tradePlaced = true;  // Don't keep trying
      return;
   }

   // Send order
   Print("");
   Print("Sending order...");

   bool sendOK = OrderSend(request, result);

   Print("");
   Print("OrderSend: ", sendOK ? "SENT" : "FAILED");
   Print("  Retcode: ", result.retcode);
   Print("  Comment: ", result.comment);
   Print("  Order: ", result.order);
   Print("  Price: ", result.price);
   Print("  Volume: ", result.volume);

   if(result.retcode == TRADE_RETCODE_DONE)
   {
      Print("");
      Print("=====================================");
      Print("  SUCCESS! TRADE PLACED!");
      Print("  Ticket: ", result.order);
      Print("  Price: ", result.price);
      Print("=====================================");
      Print("");
      Print("TRADING IS WORKING!");
      Print("You can now use BG_SimpleGrid or BG_AtlasGrid.");
      Print("REMOVE THIS TEST EA NOW!");
   }
   else
   {
      Print("");
      Print("!!! ORDER REJECTED !!!");
      DecodeError(result.retcode);
   }

   g_tradePlaced = true;  // Only try once
}

//+------------------------------------------------------------------+
//| Decode error                                                       |
//+------------------------------------------------------------------+
void DecodeError(int code)
{
   Print("");
   switch(code)
   {
      case 10004: Print("REASON: Requote - price changed"); break;
      case 10006: Print("REASON: Request rejected by server"); break;
      case 10013: Print("REASON: Invalid request"); break;
      case 10014: Print("REASON: Invalid volume - check min lot"); break;
      case 10015: Print("REASON: Invalid price"); break;
      case 10016: Print("REASON: Invalid stops"); break;
      case 10017: Print("REASON: TRADE IS DISABLED by server!"); break;
      case 10018: Print("REASON: Market is CLOSED"); break;
      case 10019: Print("REASON: NOT ENOUGH MONEY - need more margin"); break;
      case 10020: Print("REASON: Prices changed - try again"); break;
      case 10021: Print("REASON: No quotes available"); break;
      case 10026: Print("REASON: Autotrading disabled by SERVER"); break;
      case 10027: Print("REASON: Autotrading disabled by TERMINAL - press Ctrl+E"); break;
      case 10031: Print("REASON: Invalid fill type - broker doesn't support this"); break;
      case 10034: Print("REASON: Order limit reached"); break;
      case 10035: Print("REASON: Volume limit reached"); break;
      default: Print("REASON: Error code ", code); break;
   }
}

//+------------------------------------------------------------------+
//| Get filling mode                                                   |
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
//| Expert deinitialization                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("Force Trade Test removed");
}
//+------------------------------------------------------------------+
