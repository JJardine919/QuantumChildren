//+------------------------------------------------------------------+
//|                                          NeuralExpert_Test.mq5   |
//|                              Quantum Children Trading Systems    |
//|        Live inference test -- NO TRADING, log only               |
//+------------------------------------------------------------------+
#property copyright "Quantum Children"
#property version   "1.00"
#property description "Test harness for NeuralExpert ONNX inference."
#property description "Attach to any chart with a matching .onnx model."
#property description "Logs predictions to Experts tab. Does NOT trade."

#include <NeuralExpert.mqh>

//--- Inputs
input string InpModelFile  = "";    // ONNX file (blank = auto-detect from symbol)
input int    InpBarDelay   = 1;     // Bars between predictions (1 = every new bar)
input bool   InpShowChart  = true;  // Show prediction on chart via Comment()

//--- Globals
NeuralExpert g_expert;
datetime     g_last_bar_time = 0;
int          g_predict_count = 0;
int          g_buy_count     = 0;
int          g_sell_count    = 0;
int          g_hold_count    = 0;
int          g_bar_counter   = 0;

//--- Model file mapping: symbol -> ONNX filename
string AutoDetectModel()
{
   string sym = _Symbol;

   // Strip trailing suffixes (e.g., BTCUSD.r, XAUUSD.a)
   int dot = StringFind(sym, ".");
   if(dot > 0)
      sym = StringSubstr(sym, 0, dot);

   // Map to our exported models
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
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   //--- Determine model file
   string model = InpModelFile;
   if(model == "")
      model = AutoDetectModel();

   if(model == "")
   {
      Print("[TEST] No ONNX model found for ", _Symbol,
            ". Supported: BTCUSD, AUDNZD, XAUUSD, ETHUSD, EURCAD, GBPUSD, EURNZD, NZDCHF");
      return INIT_FAILED;
   }

   Print("[TEST] ========================================");
   Print("[TEST] NeuralExpert Live Inference Test");
   Print("[TEST] Symbol: ", _Symbol);
   Print("[TEST] Model:  ", model);
   Print("[TEST] Period:  M1 (hardcoded in NeuralExpert)");
   Print("[TEST] Mode:   LOG ONLY - no trades");
   Print("[TEST] ========================================");

   //--- Load model
   if(!g_expert.Load(model))
   {
      Print("[TEST] FAILED to load model: ", model);
      return INIT_FAILED;
   }

   Print("[TEST] Model loaded OK. Waiting for first new bar...");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Comment("");

   Print("[TEST] ========================================");
   Print("[TEST] Session Summary");
   Print("[TEST] Total predictions: ", g_predict_count);
   Print("[TEST] BUY:  ", g_buy_count,
         " (", (g_predict_count > 0 ? DoubleToString(100.0 * g_buy_count / g_predict_count, 1) : "0"), "%)");
   Print("[TEST] SELL: ", g_sell_count,
         " (", (g_predict_count > 0 ? DoubleToString(100.0 * g_sell_count / g_predict_count, 1) : "0"), "%)");
   Print("[TEST] HOLD: ", g_hold_count,
         " (", (g_predict_count > 0 ? DoubleToString(100.0 * g_hold_count / g_predict_count, 1) : "0"), "%)");
   Print("[TEST] ========================================");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Only run on new M1 bar
   datetime current_bar = iTime(_Symbol, PERIOD_M1, 0);
   if(current_bar == g_last_bar_time)
      return;
   g_last_bar_time = current_bar;

   //--- Respect bar delay
   g_bar_counter++;
   if(g_bar_counter < InpBarDelay)
      return;
   g_bar_counter = 0;

   //--- Run inference
   uint tick_start = GetTickCount();
   int direction = g_expert.Predict();
   uint elapsed = GetTickCount() - tick_start;

   NeuralPrediction pred = g_expert.GetLastPrediction();

   if(!pred.valid)
   {
      if(InpShowChart)
         Comment("[NeuralExpert TEST]\nWaiting for indicators to warm up...");
      return;
   }

   //--- Count
   g_predict_count++;
   if(direction == 1)       g_buy_count++;
   else if(direction == -1) g_sell_count++;
   else                     g_hold_count++;

   //--- Direction string
   string dir_str = "HOLD";
   if(direction == 1)       dir_str = "BUY";
   else if(direction == -1) dir_str = "SELL";

   //--- Log
   Print("[TEST] #", g_predict_count,
         " | ", dir_str,
         " | conf=", DoubleToString(pred.confidence, 4),
         " | P(BUY)=", DoubleToString(pred.prob_buy, 4),
         " P(SELL)=", DoubleToString(pred.prob_sell, 4),
         " P(HOLD)=", DoubleToString(pred.prob_hold, 4),
         " | ", elapsed, "ms");

   //--- Chart display
   if(InpShowChart)
   {
      string chart_text = StringFormat(
         "[NeuralExpert TEST]  %s\n"
         "-----------------------------------\n"
         "Signal:     %s\n"
         "Confidence: %.4f\n"
         "-----------------------------------\n"
         "P(BUY):     %.4f\n"
         "P(SELL):    %.4f\n"
         "P(HOLD):    %.4f\n"
         "-----------------------------------\n"
         "Inference:  %d ms\n"
         "Predictions: %d (B:%d S:%d H:%d)\n"
         "Bar: %s",
         _Symbol,
         dir_str,
         pred.confidence,
         pred.prob_buy,
         pred.prob_sell,
         pred.prob_hold,
         elapsed,
         g_predict_count, g_buy_count, g_sell_count, g_hold_count,
         TimeToString(current_bar, TIME_DATE | TIME_MINUTES)
      );
      Comment(chart_text);
   }
}
//+------------------------------------------------------------------+
