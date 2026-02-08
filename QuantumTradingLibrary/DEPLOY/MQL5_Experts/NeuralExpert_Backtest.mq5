//+------------------------------------------------------------------+
//|                                      NeuralExpert_Backtest.mq5   |
//|                              Quantum Children Trading Systems    |
//|    Backtestable EA: ONNX neural signals + fixed dollar risk      |
//+------------------------------------------------------------------+
#property copyright "Quantum Children"
#property version   "1.00"
#property description "Neural LSTM inference via ONNX with fixed-dollar risk."
#property description "Auto-detects symbol, loads trained expert, trades signals."
#property description "SL=$1.00, TP=3x, Confidence>=0.22 (from MASTER_CONFIG)."

#include <NeuralExpert.mqh>
#include <Trade\Trade.mqh>

//+------------------------------------------------------------------+
//| INPUTS                                                           |
//+------------------------------------------------------------------+
input group "=== MODEL ==="
input string InpModelFile     = "";       // ONNX file (blank = auto from symbol)

input group "=== RISK (from MASTER_CONFIG) ==="
input double InpMaxLossDollars = 1.00;    // Max loss per trade ($)
input double InpTPMultiplier   = 3.0;     // TP = SL * this
input double InpConfThreshold  = 0.22;    // Min confidence to trade
input double InpLotSize        = 0.01;    // Lot size

input group "=== TRADE MANAGEMENT ==="
input int    InpMagicNumber    = 777001;  // Magic number
input int    InpMaxPositions   = 1;       // Max open positions
input int    InpBarsBetween    = 1;       // Min bars between new trades
input bool   InpAllowBuy       = true;    // Allow BUY signals
input bool   InpAllowSell      = true;    // Allow SELL signals

input group "=== DEBUG ==="
input bool   InpShowChart      = true;    // Show info on chart
input bool   InpDebugLog       = true;    // Verbose logging

//+------------------------------------------------------------------+
//| GLOBALS                                                          |
//+------------------------------------------------------------------+
NeuralExpert g_expert;
CTrade       g_trade;
datetime     g_last_bar    = 0;
datetime     g_last_trade  = 0;
int          g_bar_count   = 0;

// Stats
int    g_total_signals = 0;
int    g_trades_opened = 0;
int    g_buy_signals   = 0;
int    g_sell_signals  = 0;
int    g_hold_signals  = 0;
int    g_filtered      = 0;   // below confidence threshold

//+------------------------------------------------------------------+
//| Auto-detect ONNX model from chart symbol                        |
//+------------------------------------------------------------------+
string AutoDetectModel()
{
   string sym = _Symbol;
   int dot = StringFind(sym, ".");
   if(dot > 0) sym = StringSubstr(sym, 0, dot);

   if(sym == "BTCUSD")  return "expert_BTCUSD_special.onnx";
   if(sym == "AUDNZD")  return "expert_rank01_AUDNZD.onnx";
   if(sym == "XAUUSD")  return "expert_rank11_XAUUSD.onnx";
   if(sym == "ETHUSD")  return "expert_rank21_ETHUSD.onnx";
   if(sym == "EURCAD")  return "expert_rank26_EURCAD.onnx";
   if(sym == "GBPUSD")  return "expert_rank27_GBPUSD.onnx";
   if(sym == "EURNZD")  return "expert_rank40_EURNZD.onnx";
   if(sym == "NZDCHF")  return "expert_rank50_NZDCHF.onnx";
   return "";
}

//+------------------------------------------------------------------+
//| Count our open positions                                         |
//+------------------------------------------------------------------+
int CountPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(PositionGetSymbol(i) == _Symbol)
         if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
            count++;
   }
   return count;
}

//+------------------------------------------------------------------+
//| Calculate SL distance in points for fixed dollar risk            |
//| Formula: sl_points = MaxLoss / (tick_value * lots)               |
//+------------------------------------------------------------------+
double CalcSLPoints()
{
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_val <= 0) return 0;

   double sl_points = InpMaxLossDollars / (tick_val * InpLotSize);
   return NormalizeDouble(sl_points, 0);
}

//+------------------------------------------------------------------+
//| Init                                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   string model = InpModelFile;
   if(model == "") model = AutoDetectModel();

   if(model == "")
   {
      Print("[BT] No ONNX model for ", _Symbol);
      return INIT_FAILED;
   }

   Print("[BT] ========================================");
   Print("[BT] NeuralExpert Backtest EA");
   Print("[BT] Symbol:     ", _Symbol);
   Print("[BT] Model:      ", model);
   Print("[BT] Max Loss:   $", DoubleToString(InpMaxLossDollars, 2));
   Print("[BT] TP Multi:   ", InpTPMultiplier, "x");
   Print("[BT] Confidence: >= ", InpConfThreshold);
   Print("[BT] Lot:        ", InpLotSize);
   Print("[BT] Magic:      ", InpMagicNumber);
   Print("[BT] ========================================");

   // Load model
   g_expert.SetDebugMode(InpDebugLog);
   if(!g_expert.Load(model))
   {
      Print("[BT] FAILED to load: ", model);
      return INIT_FAILED;
   }

   // Setup trade object
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Validate SL calculation
   double sl_pts = CalcSLPoints();
   double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   Print("[BT] Tick value: ", tick_val,
         " | SL distance: ", sl_pts, " pts",
         " | TP distance: ", sl_pts * InpTPMultiplier, " pts");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Deinit                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Comment("");
   Print("[BT] ========================================");
   Print("[BT] Session Results");
   Print("[BT] Total signals:  ", g_total_signals);
   Print("[BT] BUY signals:    ", g_buy_signals);
   Print("[BT] SELL signals:   ", g_sell_signals);
   Print("[BT] HOLD signals:   ", g_hold_signals);
   Print("[BT] Filtered (low conf): ", g_filtered);
   Print("[BT] Trades opened:  ", g_trades_opened);
   Print("[BT] ========================================");
}

//+------------------------------------------------------------------+
//| Tick                                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- New M1 bar only
   datetime bar_time = iTime(_Symbol, PERIOD_M1, 0);
   if(bar_time == g_last_bar) return;
   g_last_bar = bar_time;

   //--- Run inference
   int direction = g_expert.Predict();
   NeuralPrediction pred = g_expert.GetLastPrediction();
   if(!pred.valid) return;

   g_total_signals++;
   if(direction == 1)       g_buy_signals++;
   else if(direction == -1) g_sell_signals++;
   else                     g_hold_signals++;

   //--- Chart display
   if(InpShowChart)
   {
      string dir_str = (direction == 1) ? "BUY" : (direction == -1) ? "SELL" : "HOLD";
      Comment(StringFormat(
         "[NeuralExpert BT]  %s\n"
         "Signal: %s  Conf: %.4f\n"
         "P(B):%.3f P(S):%.3f P(H):%.3f\n"
         "Trades: %d  Positions: %d",
         _Symbol, dir_str, pred.confidence,
         pred.prob_buy, pred.prob_sell, pred.prob_hold,
         g_trades_opened, CountPositions()
      ));
   }

   //--- Filter: HOLD = no action
   if(direction == 0) return;

   //--- Filter: confidence threshold
   if(pred.confidence < InpConfThreshold)
   {
      g_filtered++;
      return;
   }

   //--- Filter: direction allowed?
   if(direction == 1 && !InpAllowBuy) return;
   if(direction == -1 && !InpAllowSell) return;

   //--- Filter: max positions
   if(CountPositions() >= InpMaxPositions) return;

   //--- Filter: bar spacing between trades
   if(g_last_trade != 0)
   {
      int bars_since = (int)((bar_time - g_last_trade) / PeriodSeconds(PERIOD_M1));
      if(bars_since < InpBarsBetween) return;
   }

   //--- Calculate SL/TP
   double sl_pts = CalcSLPoints();
   if(sl_pts <= 0)
   {
      Print("[BT] Cannot calculate SL -- tick_value issue");
      return;
   }

   double tp_pts = sl_pts * InpTPMultiplier;
   double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   //--- Open trade
   double price, sl_price, tp_price;
   bool result = false;

   if(direction == 1)  // BUY
   {
      price    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl_price = NormalizeDouble(price - sl_pts * point, _Digits);
      tp_price = NormalizeDouble(price + tp_pts * point, _Digits);

      result = g_trade.Buy(InpLotSize, _Symbol, price, sl_price, tp_price,
                           StringFormat("NE BUY c=%.3f", pred.confidence));
   }
   else  // SELL
   {
      price    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl_price = NormalizeDouble(price + sl_pts * point, _Digits);
      tp_price = NormalizeDouble(price - tp_pts * point, _Digits);

      result = g_trade.Sell(InpLotSize, _Symbol, price, sl_price, tp_price,
                            StringFormat("NE SELL c=%.3f", pred.confidence));
   }

   if(result)
   {
      g_trades_opened++;
      g_last_trade = bar_time;

      if(InpDebugLog)
      {
         string dir_str = (direction == 1) ? "BUY" : "SELL";
         Print("[BT] TRADE #", g_trades_opened,
               " | ", dir_str,
               " @ ", DoubleToString(price, _Digits),
               " | SL=", DoubleToString(sl_price, _Digits),
               " | TP=", DoubleToString(tp_price, _Digits),
               " | conf=", DoubleToString(pred.confidence, 4));
      }
   }
   else
   {
      Print("[BT] Order failed: ", g_trade.ResultRetcodeDescription());
   }
}
//+------------------------------------------------------------------+
