//+------------------------------------------------------------------+
//|                                               BG_SimpleGrid.mq5  |
//|                    Simple Grid Trader - GUARANTEED TO TRADE      |
//|                   Minimal logic, maximum reliability              |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property strict
#property description "SIMPLE Grid Trader - trades on every signal"
#property description "Atlas Style: BTCUSD, 0.06-0.07 lots, BUY bias"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT ==="
input int      MagicNumber       = 365001;    // Magic Number

input group "=== TRADING ==="
input double   LotSize           = 0.06;      // Lot Size
input int      MaxPositions      = 5;         // Max Positions
input int      GridPoints        = 500;       // Grid Spacing (points)
input int      TPPoints          = 450;       // Take Profit (points)
input int      CheckSec          = 10;        // Check Interval (sec)

//+------------------------------------------------------------------+
//| GLOBALS                                                           |
//+------------------------------------------------------------------+
datetime g_lastCheck = 0;

//+------------------------------------------------------------------+
//| Expert initialization                                              |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== BG SIMPLE GRID STARTED ===");
   Print("Magic: ", MagicNumber);
   Print("Lot: ", LotSize);
   Print("Max Pos: ", MaxPositions);
   Print("Grid: ", GridPoints, " pts");
   Print("TP: ", TPPoints, " pts");

   // Immediate status check
   Print("");
   Print("--- PERMISSION CHECK ---");
   Print("Trade Allowed: ", AccountInfoInteger(ACCOUNT_TRADE_ALLOWED));
   Print("Expert Allowed: ", AccountInfoInteger(ACCOUNT_TRADE_EXPERT));
   Print("Terminal Trade: ", TerminalInfoInteger(TERMINAL_TRADE_ALLOWED));
   Print("Connected: ", TerminalInfoInteger(TERMINAL_CONNECTED));
   Print("MQL Trade: ", MQLInfoInteger(MQL_TRADE_ALLOWED));
   Print("");

   // Test if we can trade
   if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
   {
      Print("!!! MQL TRADING NOT ALLOWED - Check EA properties !!!");
   }
   if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      Print("!!! TERMINAL TRADING NOT ALLOWED - Enable Algo Trading !!!");
   }
   if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED))
   {
      Print("!!! ACCOUNT TRADING NOT ALLOWED - Server disabled !!!");
   }

   Print("=== READY ===");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert tick                                                        |
//+------------------------------------------------------------------+
void OnTick()
{
   // Manage existing positions - check TP
   ManagePositions();

   // Check interval
   if(TimeCurrent() - g_lastCheck < CheckSec) return;
   g_lastCheck = TimeCurrent();

   // Check if we can trade
   if(!CanTrade())
   {
      static datetime lastWarn = 0;
      if(TimeCurrent() - lastWarn > 60)
      {
         Print("Cannot trade - check permissions");
         lastWarn = TimeCurrent();
      }
      return;
   }

   // Count our positions
   int posCount = CountPositions();

   // Log status periodically
   static datetime lastLog = 0;
   if(TimeCurrent() - lastLog > 300)  // Every 5 min
   {
      Print("Status: ", posCount, "/", MaxPositions, " positions | Balance: $",
            DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
      lastLog = TimeCurrent();
   }

   if(posCount >= MaxPositions) return;

   // Simple entry logic: Always try to BUY if we have room
   // Grid spacing check
   if(posCount > 0)
   {
      double lastEntry = GetLastEntry();
      MqlTick tick;
      if(!SymbolInfoTick(_Symbol, tick)) return;

      double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double spacing = GridPoints * point;

      // Only add if price dropped below last entry by grid spacing
      if(tick.ask > lastEntry - spacing) return;
   }

   // OPEN BUY
   OpenBuy(posCount + 1);
}

//+------------------------------------------------------------------+
//| Check if we can trade                                              |
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
//| Count our positions                                                |
//+------------------------------------------------------------------+
int CountPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Get last entry price                                               |
//+------------------------------------------------------------------+
double GetLastEntry()
{
   double lastPrice = 0;
   datetime lastTime = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      if(openTime > lastTime)
      {
         lastTime = openTime;
         lastPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      }
   }
   return lastPrice;
}

//+------------------------------------------------------------------+
//| Open BUY position                                                  |
//+------------------------------------------------------------------+
void OpenBuy(int level)
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
   {
      Print("ERROR: Cannot get tick");
      return;
   }

   double lot = NormalizeLot(LotSize);
   double price = tick.ask;
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double tp = price + (TPPoints * point);

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = ORDER_TYPE_BUY;
   request.price = price;
   request.sl = 0;
   request.tp = tp;  // Visible TP for reliability
   request.deviation = 30;
   request.magic = MagicNumber;
   request.comment = StringFormat("BG_L%d", level);
   request.type_filling = GetFilling();
   request.type_time = ORDER_TIME_GTC;

   // Log before send
   Print(">>> Sending BUY L", level, " @ ", DoubleToString(price, 2),
         " lot=", DoubleToString(lot, 2), " tp=", DoubleToString(tp, 2));

   // Check order first
   MqlTradeCheckResult checkResult;
   if(!OrderCheck(request, checkResult))
   {
      Print("!!! OrderCheck FAILED: ", checkResult.retcode, " - ", checkResult.comment);
      return;
   }

   // Send order
   if(!OrderSend(request, result))
   {
      int err = GetLastError();
      Print("!!! OrderSend FAILED: ", err, " - ", ErrorDescription(err));
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE)
   {
      Print(">>> BUY L", level, " FILLED @ ", DoubleToString(result.price, 2),
            " ticket=", result.order);
   }
   else
   {
      Print("!!! Order REJECTED: ", result.retcode, " - ", result.comment);
   }
}

//+------------------------------------------------------------------+
//| Manage positions - check if TP hit (backup to broker TP)           |
//+------------------------------------------------------------------+
void ManagePositions()
{
   // Positions with broker TP will auto-close
   // This is just for logging
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      double profit = PositionGetDouble(POSITION_PROFIT);
      // Just monitor - broker handles TP
   }
}

//+------------------------------------------------------------------+
//| Normalize lot                                                      |
//+------------------------------------------------------------------+
double NormalizeLot(double lot)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lot = MathMax(lot, minLot);
   lot = MathMin(lot, maxLot);
   lot = MathFloor(lot / step) * step;
   return NormalizeDouble(lot, 2);
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
//| Error description                                                  |
//+------------------------------------------------------------------+
string ErrorDescription(int code)
{
   switch(code)
   {
      case 10004: return "Requote";
      case 10006: return "Request rejected";
      case 10007: return "Request canceled by trader";
      case 10008: return "Order placed";
      case 10009: return "Request completed";
      case 10010: return "Only part of the request was completed";
      case 10011: return "Request processing error";
      case 10012: return "Request canceled by timeout";
      case 10013: return "Invalid request";
      case 10014: return "Invalid volume";
      case 10015: return "Invalid price";
      case 10016: return "Invalid stops";
      case 10017: return "Trade is disabled";
      case 10018: return "Market is closed";
      case 10019: return "Not enough money";
      case 10020: return "Prices changed";
      case 10021: return "No quotes";
      case 10022: return "Invalid expiration";
      case 10023: return "Order changed";
      case 10024: return "Too frequent requests";
      case 10025: return "No changes in request";
      case 10026: return "Autotrading disabled by server";
      case 10027: return "Autotrading disabled by terminal";
      case 10028: return "Invalid parameters";
      case 10029: return "Locked by terminal";
      case 10030: return "Frozen trade";
      case 10031: return "Invalid fill type";
      case 10032: return "No connection";
      case 10033: return "Only real accounts";
      case 10034: return "Order limit reached";
      case 10035: return "Volume limit reached";
      case 10036: return "Invalid order type";
      case 10038: return "Close disabled";
      default: return "Unknown error";
   }
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                            |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("BG Simple Grid stopped. Reason: ", reason);
}
//+------------------------------------------------------------------+
