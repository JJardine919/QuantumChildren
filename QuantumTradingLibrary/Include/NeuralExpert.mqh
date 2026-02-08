//+------------------------------------------------------------------+
//|                                            NeuralExpert.mqh      |
//|                              Quantum Children Trading Systems    |
//|         Native ONNX inference -- trained LSTM weights in MQL5    |
//+------------------------------------------------------------------+
#property copyright "Quantum Children"
#property version   "1.00"

/*
  +==================================================================+
  |              NEURAL EXPERT -- ONNX BRIDGE                         |
  +==================================================================+
  | Loads a PyTorch LSTM expert exported to ONNX and runs inference   |
  | natively inside MQL5. No Python needed at runtime.               |
  |                                                                   |
  | Model spec:                                                       |
  |   Input:  [1, 30, 8]  (batch, seq_len, features)                 |
  |   Output: [1, 3]      (probabilities: SELL, HOLD, BUY)           |
  |   Arch:   LSTM(8,128,2) -> Linear(128,3) -> Softmax              |
  |                                                                   |
  | Features (8) -- must match training pipeline exactly:             |
  |   [0] rsi, [1] macd, [2] macd_signal, [3] bb_upper,              |
  |   [4] bb_lower, [5] momentum, [6] roc, [7] atr                   |
  |                                                                   |
  | Class ordering (from training target encoding):                   |
  |   [0] HOLD, [1] BUY, [2] SELL                                    |
  +==================================================================+

  USAGE:
  ----------------------------------------------------------------
  #include <NeuralExpert.mqh>

  NeuralExpert expert;

  void OnInit()
  {
      expert.Load("expert_BTCUSD_special.onnx");
  }

  void OnTick()
  {
      int signal = expert.Predict();  // 1=BUY, -1=SELL, 0=HOLD
      double conf = expert.GetConfidence();
  }
*/

//+------------------------------------------------------------------+
//| Prediction result structure                                      |
//+------------------------------------------------------------------+
struct NeuralPrediction
{
   double   prob_sell;     // P(SELL)
   double   prob_hold;     // P(HOLD)
   double   prob_buy;      // P(BUY)
   int      direction;     // 1=BUY, -1=SELL, 0=HOLD
   double   confidence;    // max probability
   bool     valid;
};

//+------------------------------------------------------------------+
//| Main class                                                       |
//+------------------------------------------------------------------+
class NeuralExpert
{
private:
   long              m_handle;        // ONNX model handle
   string            m_model_file;    // filename
   bool              m_loaded;
   bool              m_debug;

   //--- Model dimensions
   int               m_seq_length;    // 30
   int               m_input_size;    // 8
   int               m_output_size;   // 3

   //--- Feature indicator handles
   int               m_handle_rsi;       // RSI(14)
   int               m_handle_macd;      // MACD(12,26,9)
   int               m_handle_bb;        // Bollinger Bands(20,2)
   int               m_handle_momentum;  // Momentum(10)
   int               m_handle_roc;       // ROC(10)
   int               m_handle_atr;       // ATR(14)

   //--- Last prediction
   NeuralPrediction  m_last;

   //--- Build input tensor from market data
   //--- Features: [rsi, macd, macd_signal, bb_upper, bb_lower, momentum, roc, atr]
   //--- Normalization: per-feature z-score over the sequence window, clipped to [-4, 4]
   bool BuildInput(float &input_data[])
   {
      int total = m_seq_length * m_input_size;
      ArrayResize(input_data, total);
      ArrayInitialize(input_data, 0.0f);

      // Copy indicator buffers (series mode: index 0 = latest bar)
      double rsi_buf[], macd_buf[], signal_buf[], bb_upper[], bb_lower[];
      double momentum_buf[], close_buf[];
      ArraySetAsSeries(rsi_buf, true);
      ArraySetAsSeries(macd_buf, true);
      ArraySetAsSeries(signal_buf, true);
      ArraySetAsSeries(bb_upper, true);
      ArraySetAsSeries(bb_lower, true);
      ArraySetAsSeries(momentum_buf, true);
      ArraySetAsSeries(close_buf, true);

      if(CopyBuffer(m_handle_rsi, 0, 0, m_seq_length, rsi_buf) < m_seq_length)
         return false;
      if(CopyBuffer(m_handle_macd, 0, 0, m_seq_length, macd_buf) < m_seq_length)
         return false;
      if(CopyBuffer(m_handle_macd, 1, 0, m_seq_length, signal_buf) < m_seq_length)
         return false;
      if(CopyBuffer(m_handle_bb, 1, 0, m_seq_length, bb_upper) < m_seq_length)
         return false;
      if(CopyBuffer(m_handle_bb, 2, 0, m_seq_length, bb_lower) < m_seq_length)
         return false;
      if(CopyBuffer(m_handle_momentum, 0, 0, m_seq_length, momentum_buf) < m_seq_length)
         return false;

      // ATR
      double atr_buf[];
      ArraySetAsSeries(atr_buf, true);
      if(CopyBuffer(m_handle_atr, 0, 0, m_seq_length, atr_buf) < m_seq_length)
         return false;

      // ROC: close / close[10] (manual, matching Python training)
      // Need extra bars for ROC lookback
      int roc_period = 10;
      int bars_needed = m_seq_length + roc_period;
      MqlRates rates[];
      ArraySetAsSeries(rates, true);
      if(CopyRates(_Symbol, PERIOD_M1, 0, bars_needed, rates) < bars_needed)
         return false;

      double roc_buf[];
      ArrayResize(roc_buf, m_seq_length);
      for(int i = 0; i < m_seq_length; i++)
      {
         double prev_close = rates[i + roc_period].close;
         roc_buf[i] = (prev_close > 0) ? (rates[i].close - prev_close) / prev_close * 100.0 : 0.0;
      }

      // Build raw feature matrix [seq_length x 8] in series order
      double raw[][8];
      ArrayResize(raw, m_seq_length);
      for(int i = 0; i < m_seq_length; i++)
      {
         raw[i][0] = rsi_buf[i];
         raw[i][1] = macd_buf[i];
         raw[i][2] = signal_buf[i];
         raw[i][3] = bb_upper[i];
         raw[i][4] = bb_lower[i];
         raw[i][5] = momentum_buf[i];
         raw[i][6] = roc_buf[i];
         raw[i][7] = atr_buf[i];
      }

      // Per-feature z-score normalization over the window
      for(int f = 0; f < 8; f++)
      {
         double sum = 0;
         for(int i = 0; i < m_seq_length; i++)
            sum += raw[i][f];
         double mean = sum / m_seq_length;

         double var = 0;
         for(int i = 0; i < m_seq_length; i++)
            var += MathPow(raw[i][f] - mean, 2);
         double std = MathSqrt(var / m_seq_length) + 1e-8;

         for(int i = 0; i < m_seq_length; i++)
         {
            double z = (raw[i][f] - mean) / std;
            // Clip to [-4, 4] matching training
            raw[i][f] = MathMax(-4.0, MathMin(4.0, z));
         }
      }

      // Fill output tensor: chronological order (oldest first)
      for(int i = 0; i < m_seq_length; i++)
      {
         int bar = m_seq_length - 1 - i;  // reverse: oldest first
         int base = i * m_input_size;

         for(int f = 0; f < 8; f++)
            input_data[base + f] = (float)raw[bar][f];
      }

      return true;
   }

public:
   //+------------------------------------------------------------------+
   //| Constructor                                                      |
   //+------------------------------------------------------------------+
   NeuralExpert()
   {
      m_handle       = INVALID_HANDLE;
      m_model_file   = "";
      m_loaded       = false;
      m_debug        = true;
      m_seq_length   = 30;
      m_input_size   = 8;
      m_output_size  = 3;
      m_handle_rsi      = INVALID_HANDLE;
      m_handle_macd     = INVALID_HANDLE;
      m_handle_bb       = INVALID_HANDLE;
      m_handle_momentum = INVALID_HANDLE;
      m_handle_roc      = INVALID_HANDLE;
      m_handle_atr      = INVALID_HANDLE;

      ZeroMemory(m_last);
   }

   //+------------------------------------------------------------------+
   //| Destructor                                                       |
   //+------------------------------------------------------------------+
   ~NeuralExpert()
   {
      Unload();
      if(m_handle_rsi != INVALID_HANDLE)      IndicatorRelease(m_handle_rsi);
      if(m_handle_macd != INVALID_HANDLE)     IndicatorRelease(m_handle_macd);
      if(m_handle_bb != INVALID_HANDLE)       IndicatorRelease(m_handle_bb);
      if(m_handle_momentum != INVALID_HANDLE) IndicatorRelease(m_handle_momentum);
      if(m_handle_roc != INVALID_HANDLE)      IndicatorRelease(m_handle_roc);
      if(m_handle_atr != INVALID_HANDLE)      IndicatorRelease(m_handle_atr);
   }

   //+------------------------------------------------------------------+
   //| Configuration                                                    |
   //+------------------------------------------------------------------+
   void SetDebugMode(bool enabled) { m_debug = enabled; }
   void SetSeqLength(int len)      { m_seq_length = len; }

   //+------------------------------------------------------------------+
   //| Load ONNX model from MQL5/Files/                                 |
   //+------------------------------------------------------------------+
   bool Load(string filename)
   {
      Unload();
      m_model_file = filename;

      m_handle = OnnxCreate(filename, ONNX_DEFAULT);
      if(m_handle == INVALID_HANDLE)
      {
         Print("[NE] ONNX load failed: ", filename, " err=", GetLastError());
         return false;
      }

      // Define input shape: [1, seq_length, input_size]
      const long input_shape[] = {1, m_seq_length, m_input_size};
      if(!OnnxSetInputShape(m_handle, 0, input_shape))
      {
         Print("[NE] Failed to set input shape, err=", GetLastError());
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }

      // Define output shape: [1, output_size]
      const long output_shape[] = {1, m_output_size};
      if(!OnnxSetOutputShape(m_handle, 0, output_shape))
      {
         Print("[NE] Failed to set output shape, err=", GetLastError());
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }

      // Create indicator handles matching training features:
      // [rsi(14), macd(12,26,9), bb(20,2), momentum(10), roc(manual), atr(14)]
      m_handle_rsi      = iRSI(_Symbol, PERIOD_M1, 14, PRICE_CLOSE);
      m_handle_macd     = iMACD(_Symbol, PERIOD_M1, 12, 26, 9, PRICE_CLOSE);
      m_handle_bb       = iBands(_Symbol, PERIOD_M1, 20, 0, 2.0, PRICE_CLOSE);
      m_handle_momentum = iMomentum(_Symbol, PERIOD_M1, 10, PRICE_CLOSE);
      m_handle_atr      = iATR(_Symbol, PERIOD_M1, 14);
      // ROC is computed manually in BuildInput() from close prices

      if(m_handle_rsi == INVALID_HANDLE ||
         m_handle_macd == INVALID_HANDLE ||
         m_handle_bb == INVALID_HANDLE ||
         m_handle_momentum == INVALID_HANDLE ||
         m_handle_atr == INVALID_HANDLE)
      {
         Print("[NE] Failed to create indicators");
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         return false;
      }

      m_loaded = true;
      Print("[NE] Loaded: ", filename);
      return true;
   }

   //+------------------------------------------------------------------+
   //| Unload model                                                     |
   //+------------------------------------------------------------------+
   void Unload()
   {
      if(m_handle != INVALID_HANDLE)
      {
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
      }
      m_loaded = false;
   }

   bool IsLoaded() { return m_loaded; }

   //+------------------------------------------------------------------+
   //| Run inference: returns direction (1=BUY, -1=SELL, 0=HOLD)       |
   //+------------------------------------------------------------------+
   int Predict()
   {
      m_last.valid = false;

      if(!m_loaded)
      {
         if(m_debug) Print("[NE] Model not loaded");
         return 0;
      }

      // Build input tensor
      float input_data[];
      if(!BuildInput(input_data))
         return 0;

      // Run inference
      float output_data[];
      ArrayResize(output_data, m_output_size);

      if(!OnnxRun(m_handle, ONNX_NO_CONVERSION, input_data, output_data))
      {
         Print("[NE] OnnxRun failed, err=", GetLastError());
         return 0;
      }

      // Parse output: [P(HOLD), P(BUY), P(SELL)] -- matches training target encoding
      m_last.prob_hold = output_data[0];
      m_last.prob_buy  = output_data[1];
      m_last.prob_sell = output_data[2];

      // Find argmax
      if(m_last.prob_buy > m_last.prob_sell && m_last.prob_buy > m_last.prob_hold)
      {
         m_last.direction  = 1;
         m_last.confidence = m_last.prob_buy;
      }
      else if(m_last.prob_sell > m_last.prob_buy && m_last.prob_sell > m_last.prob_hold)
      {
         m_last.direction  = -1;
         m_last.confidence = m_last.prob_sell;
      }
      else
      {
         m_last.direction  = 0;
         m_last.confidence = m_last.prob_hold;
      }

      m_last.valid = true;

      if(m_debug)
         Print("[NE] Predict: dir=", m_last.direction,
               " conf=", DoubleToString(m_last.confidence, 4),
               " [SELL=", DoubleToString(m_last.prob_sell, 4),
               " HOLD=", DoubleToString(m_last.prob_hold, 4),
               " BUY=", DoubleToString(m_last.prob_buy, 4), "]");

      return m_last.direction;
   }

   //+------------------------------------------------------------------+
   //| Accessors                                                        |
   //+------------------------------------------------------------------+
   double            GetConfidence()      { return m_last.confidence; }
   NeuralPrediction  GetLastPrediction()  { return m_last; }
   double            GetProbBuy()         { return m_last.prob_buy; }
   double            GetProbSell()        { return m_last.prob_sell; }
   double            GetProbHold()        { return m_last.prob_hold; }
};

//+------------------------------------------------------------------+
