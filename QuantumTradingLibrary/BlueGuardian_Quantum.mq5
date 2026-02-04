//+------------------------------------------------------------------+
//|                                        BlueGuardian_Quantum.mq5 |
//|                                    Blue Guardian $100K Challenge |
//|                                      Quantum Fusion Trading Bot  |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading"
#property version   "1.00"
#property strict

//--- Input parameters
input double   InpVolume = 0.01;           // Lot Size
input int      InpEmaFast = 8;             // Fast EMA Period
input int      InpEmaSlow = 21;            // Slow EMA Period
input double   InpAtrMultiplier = 1.5;     // ATR Multiplier for SL
input double   InpTpRatio = 3.0;           // TP Ratio (relative to SL)
input double   InpBePercent = 0.5;         // Break-Even Trigger (% of TP)
input int      InpMagic = 365001;          // Magic Number

//--- Global variables
int handleEmaFast;
int handleEmaSlow;
int handleAtr;
double emaFastBuffer[];
double emaSlowBuffer[];
double atrBuffer[];

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Create indicators
   handleEmaFast = iMA(_Symbol, PERIOD_M5, InpEmaFast, 0, MODE_EMA, PRICE_CLOSE);
   handleEmaSlow = iMA(_Symbol, PERIOD_M5, InpEmaSlow, 0, MODE_EMA, PRICE_CLOSE);
   handleAtr = iATR(_Symbol, PERIOD_M5, 14);

   if(handleEmaFast == INVALID_HANDLE || handleEmaSlow == INVALID_HANDLE || handleAtr == INVALID_HANDLE)
   {
      Print("Error creating indicators");
      return INIT_FAILED;
   }

   ArraySetAsSeries(emaFastBuffer, true);
   ArraySetAsSeries(emaSlowBuffer, true);
   ArraySetAsSeries(atrBuffer, true);

   Print("BlueGuardian Quantum EA initialized");
   Print("Account: ", AccountInfoInteger(ACCOUNT_LOGIN));
   Print("Balance: $", AccountInfoDouble(ACCOUNT_BALANCE));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(handleEmaFast);
   IndicatorRelease(handleEmaSlow);
   IndicatorRelease(handleAtr);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Manage existing positions (Break-Even)
   ManagePositions();

   //--- Check for new signals only if no open position
   if(!HasOpenPosition())
   {
      int signal = CheckSignal();
      if(signal != 0)
      {
         ExecuteTrade(signal);
      }
   }
}

//+------------------------------------------------------------------+
//| Check for trading signal                                           |
//+------------------------------------------------------------------+
int CheckSignal()
{
   if(CopyBuffer(handleEmaFast, 0, 0, 3, emaFastBuffer) < 3) return 0;
   if(CopyBuffer(handleEmaSlow, 0, 0, 3, emaSlowBuffer) < 3) return 0;

   //--- EMA Crossover
   // Current bar
   double fastCurr = emaFastBuffer[1];
   double slowCurr = emaSlowBuffer[1];
   // Previous bar
   double fastPrev = emaFastBuffer[2];
   double slowPrev = emaSlowBuffer[2];

   //--- Buy signal: fast crosses above slow
   if(fastPrev <= slowPrev && fastCurr > slowCurr)
      return 1;

   //--- Sell signal: fast crosses below slow
   if(fastPrev >= slowPrev && fastCurr < slowCurr)
      return -1;

   return 0;
}

//+------------------------------------------------------------------+
//| Execute trade                                                      |
//+------------------------------------------------------------------+
void ExecuteTrade(int signal)
{
   if(CopyBuffer(handleAtr, 0, 0, 2, atrBuffer) < 2) return;
   double atr = atrBuffer[1];

   double price, sl, tp;
   ENUM_ORDER_TYPE orderType;

   if(signal > 0) // BUY
   {
      orderType = ORDER_TYPE_BUY;
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = price - (atr * InpAtrMultiplier);
      tp = price + (atr * InpAtrMultiplier * InpTpRatio);
   }
   else // SELL
   {
      orderType = ORDER_TYPE_SELL;
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = price + (atr * InpAtrMultiplier);
      tp = price - (atr * InpAtrMultiplier * InpTpRatio);
   }

   MqlTradeRequest request = {};
   MqlTradeResult result = {};

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = InpVolume;
   request.type = orderType;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.magic = InpMagic;
   request.comment = "BG_Quantum";
   request.type_filling = ORDER_FILLING_IOC;

   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("TRADE: ", (signal > 0 ? "BUY" : "SELL"), " @ ", price, " | SL: ", sl, " | TP: ", tp);
      }
      else
      {
         Print("Trade failed: ", result.comment);
      }
   }
}

//+------------------------------------------------------------------+
//| Manage positions - Break-Even                                      |
//+------------------------------------------------------------------+
void ManagePositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double current = PositionGetDouble(POSITION_PRICE_CURRENT);
      double tp = PositionGetDouble(POSITION_TP);
      double sl = PositionGetDouble(POSITION_SL);

      if(tp == 0) continue;

      double totalDist = MathAbs(tp - entry);
      double currentDist = MathAbs(current - entry);
      double progress = currentDist / totalDist;

      if(progress >= InpBePercent)
      {
         bool isBuy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
         bool isAtBe = (isBuy && sl >= entry) || (!isBuy && sl <= entry);

         if(!isAtBe)
         {
            double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
            double newSl = entry + (isBuy ? 50 * point : -50 * point);

            MqlTradeRequest request = {};
            MqlTradeResult result = {};

            request.action = TRADE_ACTION_SLTP;
            request.position = ticket;
            request.sl = newSl;
            request.tp = tp;

            if(OrderSend(request, result))
            {
               if(result.retcode == TRADE_RETCODE_DONE)
               {
                  Print("BE: Moved SL to entry for ticket ", ticket);
               }
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check if we have an open position                                  |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetInteger(POSITION_MAGIC) == InpMagic &&
         PositionGetString(POSITION_SYMBOL) == _Symbol)
      {
         return true;
      }
   }
   return false;
}
//+------------------------------------------------------------------+
