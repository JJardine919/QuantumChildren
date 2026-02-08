//+------------------------------------------------------------------+
//|                                     BG_AtlasGrid_JardinesGate.mq5|
//|                            Blue Guardian - Atlas Grid Trading    |
//|                  BTCUSD M1 | SHORT BIAS | Grid + Jardine's Gate  |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "2.00"
#property strict

//+------------------------------------------------------------------+
//| INPUT PARAMETERS - CONFIGURE PER ACCOUNT                          |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input string   AccountName       = "BG_100K";        // Account Name
input int      MagicNumber       = 365001;           // Magic Number

input group "=== TRADING SETTINGS ==="
input double   BaseLot           = 0.01;             // Base Lot Size
input double   MaxLot            = 0.04;             // Max Lot Size
input int      MaxPositions      = 5;                // Max Grid Positions
input int      GridSpacingPts    = 500;              // Grid Spacing (points)
input int      TakeProfitPts     = 450;              // Take Profit (points)
input bool     OnlyBuy           = false;            // false=SELL allowed (SHORT bias)
input int      CheckSeconds      = 30;               // Check Interval (seconds)

input group "=== SIGNAL SETTINGS ==="
input int      FastEMA           = 8;                // Fast EMA Period
input int      SlowEMA           = 21;               // Slow EMA Period
input double   MinConfidence     = 0.6;              // Min Signal Confidence (0-1)

input group "=== RISK MANAGEMENT ==="
input double   DailyDDLimit      = 4.5;              // Daily Drawdown Limit %
input double   MaxDDLimit        = 9.0;              // Max Drawdown Limit %
input bool     UseHiddenSLTP     = true;             // Hidden SL/TP (manage internally)
input double   HiddenSLMultiple  = 2.0;              // Hidden SL = TP * this

input group "=== JARDINE'S GATE ALGORITHM ==="
input bool     UseJardinesGate   = true;             // Enable Jardine's Gate Filter
input double   JG_EntropyClean   = 0.90;             // G1: Entropy threshold (blocks if >)
input double   JG_InterferenceMin= 0.50;             // G2: Min interference (expert agreement)
input double   JG_ConfidenceMin  = 0.20;             // G3: Min signal confidence
input double   JG_ProbabilityMin = 0.35;             // G4: Min trade probability
input int      JG_DirectionBias  = -1;               // G5: -1=SHORT (80-90% WR on BTCUSD)
input double   JG_KillSwitchWR   = 0.50;             // G6: Min win rate (kill switch)
input int      JG_KillSwitchLB   = 10;               // G6: Lookback trades for kill switch
input bool     JG_DebugMode      = true;             // Print gate debug info
input int      JG_EntropyBars    = 50;               // Bars for entropy calculation

//+------------------------------------------------------------------+
//| JARDINE'S GATE ENUMS                                              |
//+------------------------------------------------------------------+
enum ENUM_JG_GATE
{
   GATE_PASSED = 0,
   GATE_1_ENTROPY = 1,
   GATE_2_INTERFERENCE = 2,
   GATE_3_CONFIDENCE = 3,
   GATE_4_PROBABILITY = 4,
   GATE_5_DIRECTION = 5,
   GATE_6_KILLSWITCH = 6
};

//+------------------------------------------------------------------+
//| JARDINE'S GATE TRADE RESULT STRUCTURE                             |
//+------------------------------------------------------------------+
struct JG_TradeResult
{
   datetime time;
   int      direction;
   double   pnl;
   bool     win;
};

//+------------------------------------------------------------------+
//| JARDINE'S GATE CLASS                                              |
//+------------------------------------------------------------------+
class CJardinesGate
{
private:
   double m_entropy_clean;
   double m_confidence_min;
   double m_interference_min;
   double m_probability_min;
   int    m_direction_bias;
   double m_kill_switch_wr;
   int    m_kill_switch_lookback;
   bool   m_debug_mode;

   JG_TradeResult m_trade_history[];
   int            m_history_size;
   ENUM_JG_GATE   m_last_block_gate;
   string         m_last_block_reason;

   // Stats
   int m_total_checks;
   int m_passed;
   int m_blocked_g1, m_blocked_g2, m_blocked_g3;
   int m_blocked_g4, m_blocked_g5, m_blocked_g6;

public:
   CJardinesGate()
   {
      m_entropy_clean = 0.90;
      m_confidence_min = 0.20;
      m_interference_min = 0.50;
      m_probability_min = 0.35;
      m_direction_bias = -1;  // SHORT bias (80-90% WR)
      m_kill_switch_wr = 0.50;
      m_kill_switch_lookback = 10;
      m_debug_mode = true;
      m_history_size = 0;
      m_last_block_gate = GATE_PASSED;
      m_last_block_reason = "";

      m_total_checks = 0;
      m_passed = 0;
      m_blocked_g1 = m_blocked_g2 = m_blocked_g3 = 0;
      m_blocked_g4 = m_blocked_g5 = m_blocked_g6 = 0;

      ArrayResize(m_trade_history, 0);
   }

   void SetThresholds(double entropy_clean, double confidence_min,
                      double interference_min, double probability_min)
   {
      m_entropy_clean = entropy_clean;
      m_confidence_min = confidence_min;
      m_interference_min = interference_min;
      m_probability_min = probability_min;
   }

   void SetDirectionBias(int bias) { m_direction_bias = bias; }
   void SetKillSwitch(double min_wr, int lookback)
   {
      m_kill_switch_wr = min_wr;
      m_kill_switch_lookback = lookback;
   }
   void SetDebugMode(bool enabled) { m_debug_mode = enabled; }

   //+------------------------------------------------------------------+
   //| Entropy Factor: E(H)                                              |
   //| 1.0 if H < 0.90, linear decay 0.90-0.99, 0.1 if H >= 0.99        |
   //+------------------------------------------------------------------+
   double CalculateEntropyFactor(double entropy)
   {
      if(entropy < m_entropy_clean)
         return 1.0;
      else if(entropy >= 0.99)
         return 0.1;
      else
         return 1.0 - 9.0 * (entropy - m_entropy_clean);
   }

   //+------------------------------------------------------------------+
   //| Trade Probability: P(trade) = |psi|^2 * E(entropy) * I * C       |
   //+------------------------------------------------------------------+
   double CalculateProbability(double amplitude_squared, double entropy,
                               double interference, double confidence)
   {
      double entropy_factor = CalculateEntropyFactor(entropy);
      double probability = amplitude_squared * entropy_factor * interference * confidence;
      return MathMax(0.0, MathMin(1.0, probability));
   }

   //+------------------------------------------------------------------+
   //| Calculate entropy from price array (compression-based)           |
   //+------------------------------------------------------------------+
   double CalculateEntropyFromPrices(const double &prices[])
   {
      int size = ArraySize(prices);
      if(size < 10) return 1.0;

      // Method 1: Coefficient of variation of returns
      double returns[];
      ArrayResize(returns, size - 1);

      double sum = 0, sum_sq = 0;
      for(int i = 1; i < size; i++)
      {
         returns[i-1] = (prices[i] - prices[i-1]) / (prices[i-1] + 0.0000001);
         sum += MathAbs(returns[i-1]);
      }

      double mean_abs = sum / (size - 1);
      double mean_ret = sum / (size - 1);

      for(int i = 0; i < size - 1; i++)
         sum_sq += MathPow(returns[i] - mean_ret, 2);

      double std_dev = MathSqrt(sum_sq / (size - 1));
      double cv_entropy = MathMin(1.0, std_dev / (mean_abs + 0.0000001));

      // Method 2: Autocorrelation
      double autocorr = 0;
      if(size > 11)
      {
         double sum_xy = 0, sum_x = 0, sum_y = 0, sum_x2 = 0, sum_y2 = 0;
         int n = size - 2;
         for(int i = 0; i < n; i++)
         {
            sum_xy += returns[i] * returns[i+1];
            sum_x += returns[i];
            sum_y += returns[i+1];
            sum_x2 += returns[i] * returns[i];
            sum_y2 += returns[i+1] * returns[i+1];
         }
         double denom = MathSqrt((n*sum_x2 - sum_x*sum_x) * (n*sum_y2 - sum_y*sum_y));
         if(denom > 0)
            autocorr = (n*sum_xy - sum_x*sum_y) / denom;
      }
      double autocorr_entropy = 1.0 - MathAbs(autocorr);

      // Method 3: Directional consistency
      int up = 0, down = 0;
      for(int i = 0; i < size - 1; i++)
      {
         if(returns[i] > 0) up++;
         else if(returns[i] < 0) down++;
      }
      double direction_entropy = 1.0 - MathAbs((double)(up - down) / (size - 1));

      // Weighted combination
      double entropy = 0.4 * cv_entropy + 0.3 * autocorr_entropy + 0.3 * direction_entropy;

      return MathMax(0.0, MathMin(1.0, entropy));
   }

   //+------------------------------------------------------------------+
   //| MAIN FILTER: Six-gate check                                       |
   //| Signal -> [G1] -> [G2] -> [G3] -> [G4] -> [G5] -> [G6] -> EXEC   |
   //+------------------------------------------------------------------+
   bool ShouldTrade(double entropy, double interference, double confidence,
                    int direction, double amplitude_squared = 0.5)
   {
      m_total_checks++;
      m_last_block_gate = GATE_PASSED;
      m_last_block_reason = "";

      // G1: Entropy Gate - blocks noisy markets
      if(entropy > m_entropy_clean)
      {
         m_blocked_g1++;
         m_last_block_gate = GATE_1_ENTROPY;
         m_last_block_reason = StringFormat("entropy=%.3f > %.2f", entropy, m_entropy_clean);
         if(m_debug_mode) Print("[JG] G1 BLOCK: ", m_last_block_reason);
         return false;
      }

      // G2: Interference Gate - requires expert agreement
      if(interference < m_interference_min)
      {
         m_blocked_g2++;
         m_last_block_gate = GATE_2_INTERFERENCE;
         m_last_block_reason = StringFormat("interference=%.3f < %.2f", interference, m_interference_min);
         if(m_debug_mode) Print("[JG] G2 BLOCK: ", m_last_block_reason);
         return false;
      }

      // G3: Confidence Gate - signal clarity
      if(confidence < m_confidence_min)
      {
         m_blocked_g3++;
         m_last_block_gate = GATE_3_CONFIDENCE;
         m_last_block_reason = StringFormat("confidence=%.3f < %.2f", confidence, m_confidence_min);
         if(m_debug_mode) Print("[JG] G3 BLOCK: ", m_last_block_reason);
         return false;
      }

      // Calculate probability for G4
      double probability = CalculateProbability(amplitude_squared, entropy, interference, confidence);

      // G4: Probability Gate - final composite score
      if(probability < m_probability_min)
      {
         m_blocked_g4++;
         m_last_block_gate = GATE_4_PROBABILITY;
         m_last_block_reason = StringFormat("probability=%.3f < %.2f", probability, m_probability_min);
         if(m_debug_mode) Print("[JG] G4 BLOCK: ", m_last_block_reason);
         return false;
      }

      // G5: Direction Bias Gate
      if(m_direction_bias != 0)  // 0 = QEF_BOTH
      {
         if(direction != m_direction_bias)
         {
            m_blocked_g5++;
            m_last_block_gate = GATE_5_DIRECTION;
            string dir_str = (direction == 1) ? "LONG" : "SHORT";
            string bias_str = (m_direction_bias == 1) ? "LONG" : "SHORT";
            m_last_block_reason = StringFormat("direction=%s != %s", dir_str, bias_str);
            if(m_debug_mode) Print("[JG] G5 BLOCK: ", m_last_block_reason);
            return false;
         }
      }

      // G6: Kill Switch Gate
      if(IsKillSwitchTriggered())
      {
         m_blocked_g6++;
         m_last_block_gate = GATE_6_KILLSWITCH;
         double recent_wr = GetRecentWinRate();
         m_last_block_reason = StringFormat("recent_wr=%.2f < %.2f", recent_wr, m_kill_switch_wr);
         if(m_debug_mode) Print("[JG] G6 BLOCK: ", m_last_block_reason);
         return false;
      }

      // ALL GATES PASSED
      m_passed++;
      if(m_debug_mode)
         Print("[JG] ALL GATES PASSED -> EXECUTE (entropy=",
               DoubleToString(entropy,3), " prob=", DoubleToString(probability,3), ")");

      return true;
   }

   //+------------------------------------------------------------------+
   //| Record trade result for kill switch tracking                      |
   //+------------------------------------------------------------------+
   void RecordTrade(int direction, double pnl)
   {
      int size = ArraySize(m_trade_history);
      ArrayResize(m_trade_history, size + 1);

      m_trade_history[size].time = TimeCurrent();
      m_trade_history[size].direction = direction;
      m_trade_history[size].pnl = pnl;
      m_trade_history[size].win = (pnl > 0);

      m_history_size = size + 1;

      // Keep only recent history
      if(m_history_size > m_kill_switch_lookback * 2)
      {
         for(int i = 0; i < m_history_size - m_kill_switch_lookback; i++)
            m_trade_history[i] = m_trade_history[i + m_kill_switch_lookback];
         ArrayResize(m_trade_history, m_kill_switch_lookback);
         m_history_size = m_kill_switch_lookback;
      }
   }

   bool IsKillSwitchTriggered()
   {
      if(m_history_size < m_kill_switch_lookback)
         return false;
      return GetRecentWinRate() < m_kill_switch_wr;
   }

   double GetRecentWinRate()
   {
      if(m_history_size == 0) return 1.0;

      int start = MathMax(0, m_history_size - m_kill_switch_lookback);
      int wins = 0;
      int count = 0;

      for(int i = start; i < m_history_size; i++)
      {
         if(m_trade_history[i].win) wins++;
         count++;
      }

      return (count > 0) ? (double)wins / count : 1.0;
   }

   string GetLastBlockReason() { return m_last_block_reason; }
   ENUM_JG_GATE GetLastBlockGate() { return m_last_block_gate; }

   string GetStats()
   {
      double pass_rate = (m_total_checks > 0) ? (double)m_passed / m_total_checks * 100 : 0;

      return StringFormat(
         "JARDINE'S GATE STATS | Checks: %d | Passed: %d (%.1f%%) | "
         "G1:%d G2:%d G3:%d G4:%d G5:%d G6:%d",
         m_total_checks, m_passed, pass_rate,
         m_blocked_g1, m_blocked_g2, m_blocked_g3,
         m_blocked_g4, m_blocked_g5, m_blocked_g6
      );
   }

   void ResetStats()
   {
      m_total_checks = 0;
      m_passed = 0;
      m_blocked_g1 = m_blocked_g2 = m_blocked_g3 = 0;
      m_blocked_g4 = m_blocked_g5 = m_blocked_g6 = 0;
   }
};

//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                   |
//+------------------------------------------------------------------+
int g_handleEmaFast = INVALID_HANDLE;
int g_handleEmaSlow = INVALID_HANDLE;
int g_handleAtr = INVALID_HANDLE;

double g_emaFast[];
double g_emaSlow[];
double g_atr[];

double g_startBalance = 0;
double g_highWaterMark = 0;
double g_dailyStartBalance = 0;
datetime g_lastDayReset = 0;
datetime g_lastCheck = 0;

bool g_blocked = false;
string g_blockReason = "";

// Jardine's Gate filter
CJardinesGate g_jardinesGate;

// Position tracking for hidden TP management
struct GridLevel
{
   ulong  ticket;
   double entry;
   double hiddenTP;
   double hiddenSL;
   int    level;
};
GridLevel g_grid[];
int g_gridCount = 0;

// Track last closed position for kill switch updates
ulong g_lastClosedTicket = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("================================================");
   Print("  BLUE GUARDIAN ATLAS GRID EA v2.0");
   Print("  + JARDINE'S GATE ALGORITHM (SHORT BIAS)");
   Print("================================================");
   Print("  Account: ", AccountName);
   Print("  Magic: ", MagicNumber);
   Print("  Lot: ", BaseLot, " - ", MaxLot);
   Print("  Max Positions: ", MaxPositions);
   Print("  Grid Spacing: ", GridSpacingPts, " pts");
   Print("  TP: ", TakeProfitPts, " pts");
   Print("  Only BUY: ", OnlyBuy ? "YES" : "NO (SELL allowed)");
   Print("  Hidden SL/TP: ", UseHiddenSLTP ? "YES" : "NO");
   Print("------------------------------------------------");
   Print("  JARDINE'S GATE: ", UseJardinesGate ? "ENABLED" : "DISABLED");
   if(UseJardinesGate)
   {
      Print("    G1 Entropy Threshold: ", JG_EntropyClean);
      Print("    G2 Min Interference: ", JG_InterferenceMin);
      Print("    G3 Min Confidence: ", JG_ConfidenceMin);
      Print("    G4 Min Probability: ", JG_ProbabilityMin);
      Print("    G5 Direction Bias: ", JG_DirectionBias == 1 ? "LONG" : (JG_DirectionBias == -1 ? "SHORT" : "BOTH"));
      Print("    G6 Kill Switch WR: ", JG_KillSwitchWR, " (", JG_KillSwitchLB, " trades)");
   }
   Print("================================================");

   // Create EMA indicators
   g_handleEmaFast = iMA(_Symbol, PERIOD_M1, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_handleEmaSlow = iMA(_Symbol, PERIOD_M1, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   g_handleAtr = iATR(_Symbol, PERIOD_M1, 14);

   if(g_handleEmaFast == INVALID_HANDLE || g_handleEmaSlow == INVALID_HANDLE || g_handleAtr == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicators");
      return INIT_FAILED;
   }

   ArraySetAsSeries(g_emaFast, true);
   ArraySetAsSeries(g_emaSlow, true);
   ArraySetAsSeries(g_atr, true);

   // Initialize Jardine's Gate
   g_jardinesGate.SetThresholds(JG_EntropyClean, JG_ConfidenceMin, JG_InterferenceMin, JG_ProbabilityMin);
   g_jardinesGate.SetDirectionBias(JG_DirectionBias);
   g_jardinesGate.SetKillSwitch(JG_KillSwitchWR, JG_KillSwitchLB);
   g_jardinesGate.SetDebugMode(JG_DebugMode);

   // Initialize balance tracking
   g_startBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_highWaterMark = g_startBalance;
   g_dailyStartBalance = g_startBalance;
   g_lastDayReset = TimeCurrent();
   g_lastCheck = 0;

   // Sync existing positions
   SyncGrid();

   Print("Balance: $", DoubleToString(g_startBalance, 2));
   Print("Synced ", g_gridCount, " existing positions");
   Print("Initialization complete. READY TO TRADE.");
   Print("================================================");

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_handleEmaFast != INVALID_HANDLE) IndicatorRelease(g_handleEmaFast);
   if(g_handleEmaSlow != INVALID_HANDLE) IndicatorRelease(g_handleEmaSlow);
   if(g_handleAtr != INVALID_HANDLE) IndicatorRelease(g_handleAtr);

   // Print final Jardine's Gate stats
   if(UseJardinesGate)
      Print(g_jardinesGate.GetStats());

   Print("BG Atlas Grid EA stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
   // ALWAYS manage positions on every tick (for hidden TP)
   ManageGrid();

   // Check interval for new entries
   if(TimeCurrent() - g_lastCheck < CheckSeconds) return;
   g_lastCheck = TimeCurrent();

   // Daily reset
   CheckDailyReset();

   // Update high water mark
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   if(bal > g_highWaterMark) g_highWaterMark = bal;

   // Risk check
   if(!CheckRisk())
   {
      if(!g_blocked)
      {
         g_blocked = true;
         Print("BLOCKED: ", g_blockReason);
      }
      return;
   }

   // Check for entry signals
   CheckEntry();
}

//+------------------------------------------------------------------+
//| Daily reset                                                        |
//+------------------------------------------------------------------+
void CheckDailyReset()
{
   MqlDateTime now, last;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(g_lastDayReset, last);

   if(now.day != last.day || now.mon != last.mon || now.year != last.year)
   {
      g_dailyStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_lastDayReset = TimeCurrent();
      g_blocked = false;
      g_blockReason = "";
      Print("Daily reset. Baseline: $", DoubleToString(g_dailyStartBalance, 2));

      // Print Jardine's Gate daily stats
      if(UseJardinesGate)
      {
         Print(g_jardinesGate.GetStats());
         g_jardinesGate.ResetStats();
      }
   }
}

//+------------------------------------------------------------------+
//| Check risk limits                                                  |
//+------------------------------------------------------------------+
bool CheckRisk()
{
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double current = MathMin(bal, eq);

   if(g_dailyStartBalance <= 0 || g_highWaterMark <= 0) return true;

   // Daily DD
   double dailyDD = ((g_dailyStartBalance - current) / g_dailyStartBalance) * 100;
   if(dailyDD >= DailyDDLimit)
   {
      g_blockReason = StringFormat("Daily DD %.2f%% >= %.2f%%", dailyDD, DailyDDLimit);
      return false;
   }

   // Max DD
   double maxDD = ((g_highWaterMark - current) / g_highWaterMark) * 100;
   if(maxDD >= MaxDDLimit)
   {
      g_blockReason = StringFormat("Max DD %.2f%% >= %.2f%%", maxDD, MaxDDLimit);
      return false;
   }

   return true;
}

//+------------------------------------------------------------------+
//| Calculate market entropy from recent prices                        |
//+------------------------------------------------------------------+
double CalculateMarketEntropy()
{
   double prices[];
   ArrayResize(prices, JG_EntropyBars);
   ArraySetAsSeries(prices, true);

   if(CopyClose(_Symbol, PERIOD_M1, 0, JG_EntropyBars, prices) < JG_EntropyBars)
      return 1.0;  // High entropy if can't get data

   return g_jardinesGate.CalculateEntropyFromPrices(prices);
}

//+------------------------------------------------------------------+
//| Calculate interference (expert agreement) from multiple EMAs       |
//+------------------------------------------------------------------+
double CalculateInterference()
{
   // Use multiple timeframes/periods for "expert" agreement
   int handles[4];
   handles[0] = iMA(_Symbol, PERIOD_M1, 5, 0, MODE_EMA, PRICE_CLOSE);
   handles[1] = iMA(_Symbol, PERIOD_M1, 13, 0, MODE_EMA, PRICE_CLOSE);
   handles[2] = iMA(_Symbol, PERIOD_M1, 34, 0, MODE_EMA, PRICE_CLOSE);
   handles[3] = iMA(_Symbol, PERIOD_M1, 55, 0, MODE_EMA, PRICE_CLOSE);

   double values[4];
   double buffers[];
   ArrayResize(buffers, 3);

   int bullish = 0;
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   for(int i = 0; i < 4; i++)
   {
      if(handles[i] == INVALID_HANDLE) continue;

      if(CopyBuffer(handles[i], 0, 0, 3, buffers) >= 3)
      {
         values[i] = buffers[1];
         if(price > values[i]) bullish++;
      }

      IndicatorRelease(handles[i]);
   }

   // Interference = agreement ratio (0-1)
   // 4/4 or 0/4 = 1.0 (full agreement), 2/4 = 0.0 (no agreement)
   double agreement = MathAbs((double)bullish - 2.0) / 2.0;

   return agreement;
}

//+------------------------------------------------------------------+
//| Check for entry signal                                             |
//+------------------------------------------------------------------+
void CheckEntry()
{
   // Count positions
   int posCount = CountPositions();
   if(posCount >= MaxPositions)
   {
      return;
   }

   // Get indicator values
   if(CopyBuffer(g_handleEmaFast, 0, 0, 3, g_emaFast) < 3) return;
   if(CopyBuffer(g_handleEmaSlow, 0, 0, 3, g_emaSlow) < 3) return;
   if(CopyBuffer(g_handleAtr, 0, 0, 3, g_atr) < 3) return;

   double emaF = g_emaFast[1];
   double emaS = g_emaSlow[1];
   double atr = g_atr[1];

   // Current tick
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;
   double price = tick.bid;  // Use bid for SELL
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // Signal detection
   bool buySignal = (emaF > emaS);
   bool sellSignal = (emaF < emaS);

   // Determine direction (1=LONG, -1=SHORT)
   int direction = buySignal ? 1 : -1;

   // Calculate confidence based on EMA separation
   double separation = MathAbs(emaF - emaS);
   double confidence = MathMin(1.0, (separation / atr) * 0.5);

   if(confidence < MinConfidence)
   {
      return;
   }

   // Grid spacing check
   if(posCount > 0)
   {
      double lastEntry = GetLastEntryPrice();
      double spacing = GridSpacingPts * point;

      // For SELL grid: add positions when price rises above last entry
      if(!OnlyBuy && sellSignal)
      {
         if(price < lastEntry + spacing)
            return;
      }
      // For BUY grid: add positions when price dips below last entry
      else if(OnlyBuy && buySignal)
      {
         if(price > lastEntry - spacing)
            return;
      }
   }

   //=== JARDINE'S GATE FILTER ===
   if(UseJardinesGate)
   {
      // Calculate quantum metrics
      double entropy = CalculateMarketEntropy();
      double interference = CalculateInterference();

      // Check all 6 gates
      if(!g_jardinesGate.ShouldTrade(entropy, interference, confidence, direction))
      {
         // Signal blocked by Jardine's Gate
         return;
      }
   }

   // Execute trade based on signal and OnlyBuy setting
   if(OnlyBuy && buySignal)
   {
      OpenPosition(ORDER_TYPE_BUY, posCount + 1);
   }
   else if(!OnlyBuy && sellSignal)
   {
      OpenPosition(ORDER_TYPE_SELL, posCount + 1);
   }
}

//+------------------------------------------------------------------+
//| Open grid position                                                 |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE type, int level)
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   double price = (type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // Lot size: randomly between BaseLot and MaxLot
   double lot = BaseLot + (MathRand() % 2) * 0.01;
   lot = MathMin(lot, MaxLot);
   lot = NormalizeLot(lot);

   // Calculate hidden TP/SL
   double hiddenTP, hiddenSL;
   if(type == ORDER_TYPE_BUY)
   {
      hiddenTP = price + (TakeProfitPts * point);
      hiddenSL = price - (TakeProfitPts * HiddenSLMultiple * point);
   }
   else
   {
      hiddenTP = price - (TakeProfitPts * point);
      hiddenSL = price + (TakeProfitPts * HiddenSLMultiple * point);
   }

   // Build request
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = lot;
   request.type = type;
   request.price = price;
   request.deviation = 30;
   request.magic = MagicNumber;
   request.comment = StringFormat("BG_JG_L%d", level);
   request.type_time = ORDER_TIME_GTC;
   request.type_filling = GetFilling();

   // Hidden SL/TP: don't send to broker
   if(UseHiddenSLTP)
   {
      request.sl = 0;
      request.tp = 0;
   }
   else
   {
      request.sl = hiddenSL;
      request.tp = hiddenTP;
   }

   if(!OrderSend(request, result))
   {
      Print("ERROR: Order failed - ", GetLastError());
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE)
   {
      // Track position
      int idx = ArraySize(g_grid);
      ArrayResize(g_grid, idx + 1);

      g_grid[idx].ticket = result.order;
      g_grid[idx].entry = price;
      g_grid[idx].hiddenTP = hiddenTP;
      g_grid[idx].hiddenSL = hiddenSL;
      g_grid[idx].level = level;
      g_gridCount = idx + 1;

      string typeStr = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";
      Print(typeStr, " L", level, " placed [JARDINE'S GATE PASSED] | Price: ", DoubleToString(price, 2),
            " | Lot: ", DoubleToString(lot, 2),
            " | TP: ", DoubleToString(hiddenTP, 2));
   }
   else
   {
      Print("Order rejected: ", result.comment, " (", result.retcode, ")");
   }
}

//+------------------------------------------------------------------+
//| Manage grid - hidden TP/SL                                         |
//+------------------------------------------------------------------+
void ManageGrid()
{
   SyncGrid();

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   for(int i = g_gridCount - 1; i >= 0; i--)
   {
      ulong ticket = g_grid[i].ticket;

      if(!PositionSelectByTicket(ticket))
      {
         // Position closed - record for Jardine's Gate kill switch
         RecordClosedTrade(ticket);
         RemoveFromGrid(i);
         continue;
      }

      double currentPrice = PositionGetDouble(POSITION_PRICE_CURRENT);
      ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      // Check hidden TP hit
      bool hitTP = false;
      if(posType == POSITION_TYPE_BUY && currentPrice >= g_grid[i].hiddenTP) hitTP = true;
      if(posType == POSITION_TYPE_SELL && currentPrice <= g_grid[i].hiddenTP) hitTP = true;

      if(hitTP)
      {
         Print("HIDDEN TP HIT - Closing L", g_grid[i].level, " @ ", DoubleToString(currentPrice, 2));
         double pnl = PositionGetDouble(POSITION_PROFIT);
         int dir = (posType == POSITION_TYPE_BUY) ? 1 : -1;

         ClosePosition(ticket);

         // Record for Jardine's Gate
         if(UseJardinesGate)
            g_jardinesGate.RecordTrade(dir, pnl);

         RemoveFromGrid(i);
         continue;
      }

      // Check hidden SL hit
      bool hitSL = false;
      if(posType == POSITION_TYPE_BUY && currentPrice <= g_grid[i].hiddenSL) hitSL = true;
      if(posType == POSITION_TYPE_SELL && currentPrice >= g_grid[i].hiddenSL) hitSL = true;

      if(hitSL)
      {
         Print("HIDDEN SL HIT - Closing L", g_grid[i].level, " @ ", DoubleToString(currentPrice, 2));
         double pnl = PositionGetDouble(POSITION_PROFIT);
         int dir = (posType == POSITION_TYPE_BUY) ? 1 : -1;

         ClosePosition(ticket);

         // Record for Jardine's Gate
         if(UseJardinesGate)
            g_jardinesGate.RecordTrade(dir, pnl);

         RemoveFromGrid(i);
         continue;
      }
   }
}

//+------------------------------------------------------------------+
//| Record externally closed trades for kill switch                    |
//+------------------------------------------------------------------+
void RecordClosedTrade(ulong ticket)
{
   if(!UseJardinesGate) return;
   if(ticket == g_lastClosedTicket) return;  // Already recorded

   // Try to get PnL from history
   if(HistorySelectByPosition(ticket))
   {
      int total = HistoryDealsTotal();
      for(int i = total - 1; i >= 0; i--)
      {
         ulong dealTicket = HistoryDealGetTicket(i);
         if(HistoryDealGetInteger(dealTicket, DEAL_POSITION_ID) == (long)ticket)
         {
            double pnl = HistoryDealGetDouble(dealTicket, DEAL_PROFIT);
            ENUM_DEAL_TYPE dealType = (ENUM_DEAL_TYPE)HistoryDealGetInteger(dealTicket, DEAL_TYPE);

            // Determine original direction (opposite of closing deal)
            int dir = (dealType == DEAL_TYPE_SELL) ? 1 : -1;  // Sell closes Buy, Buy closes Sell

            g_jardinesGate.RecordTrade(dir, pnl);
            g_lastClosedTicket = ticket;
            return;
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Close position                                                     |
//+------------------------------------------------------------------+
bool ClosePosition(ulong ticket)
{
   if(!PositionSelectByTicket(ticket)) return false;

   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);
   string symbol = PositionGetString(POSITION_SYMBOL);

   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return false;

   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = symbol;
   request.volume = volume;
   request.deviation = 30;
   request.magic = MagicNumber;
   request.type_filling = GetFilling();

   if(posType == POSITION_TYPE_BUY)
   {
      request.type = ORDER_TYPE_SELL;
      request.price = tick.bid;
   }
   else
   {
      request.type = ORDER_TYPE_BUY;
      request.price = tick.ask;
   }

   if(!OrderSend(request, result))
   {
      Print("ERROR: Close failed - ", GetLastError());
      return false;
   }

   return (result.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
//| Sync grid with actual positions                                    |
//+------------------------------------------------------------------+
void SyncGrid()
{
   // Remove closed positions
   for(int i = g_gridCount - 1; i >= 0; i--)
   {
      bool found = false;
      for(int p = PositionsTotal() - 1; p >= 0; p--)
      {
         ulong t = PositionGetTicket(p);
         if(t == g_grid[i].ticket)
         {
            found = true;
            break;
         }
      }
      if(!found) RemoveFromGrid(i);
   }

   // Add untracked positions
   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      ulong ticket = PositionGetTicket(p);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      bool tracked = false;
      for(int i = 0; i < g_gridCount; i++)
      {
         if(g_grid[i].ticket == ticket)
         {
            tracked = true;
            break;
         }
      }

      if(!tracked)
      {
         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         int idx = ArraySize(g_grid);
         ArrayResize(g_grid, idx + 1);

         g_grid[idx].ticket = ticket;
         g_grid[idx].entry = entry;
         g_grid[idx].level = idx + 1;

         if(posType == POSITION_TYPE_BUY)
         {
            g_grid[idx].hiddenTP = entry + (TakeProfitPts * point);
            g_grid[idx].hiddenSL = entry - (TakeProfitPts * HiddenSLMultiple * point);
         }
         else
         {
            g_grid[idx].hiddenTP = entry - (TakeProfitPts * point);
            g_grid[idx].hiddenSL = entry + (TakeProfitPts * HiddenSLMultiple * point);
         }

         g_gridCount = idx + 1;
         Print("Synced position: ", ticket, " entry: ", DoubleToString(entry, 2));
      }
   }
}

//+------------------------------------------------------------------+
//| Remove position from grid tracking                                 |
//+------------------------------------------------------------------+
void RemoveFromGrid(int index)
{
   if(index < 0 || index >= g_gridCount) return;

   for(int i = index; i < g_gridCount - 1; i++)
   {
      g_grid[i] = g_grid[i + 1];
   }

   g_gridCount--;
   ArrayResize(g_grid, g_gridCount);
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
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber &&
         PositionGetString(POSITION_SYMBOL) == _Symbol)
      {
         count++;
      }
   }
   return count;
}

//+------------------------------------------------------------------+
//| Get last entry price                                               |
//+------------------------------------------------------------------+
double GetLastEntryPrice()
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
//| Normalize lot size                                                 |
//+------------------------------------------------------------------+
double NormalizeLot(double lot)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   lot = MathMax(lot, minLot);
   lot = MathMin(lot, maxLot);
   lot = MathFloor(lot / lotStep) * lotStep;

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
