//+------------------------------------------------------------------+
//|                                     JardinesGate_v3.mqh          |
//|                              Quantum Children Trading Systems    |
//|                        Jardine's Gate Algorithm v3.0              |
//|                    Neural-TE Integration (10 Gates)               |
//+------------------------------------------------------------------+
#property copyright "Quantum Children"
#property version   "3.00"

/*
  +==================================================================+
  |          JARDINE'S GATE ALGORITHM v3.0 -- NEURAL-TE               |
  +==================================================================+
  | Ten-gate quantum filter. Extends v2.0 (6 gates) with:            |
  |   G7: Neural Mosaic Consensus                                     |
  |   G8: Genomic Shock Adaptive                                      |
  |   G9: Speciation / Cross-Instrument Check                         |
  |   G10: TE Domestication Validation                                |
  |                                                                    |
  | "Only the worthy signals pass through all ten gates."             |
  |                                                                    |
  | Based on:                                                          |
  |   - L1 retrotransposon somatic mosaicism (Muotri 2005)            |
  |   - TE-driven speciation (Serrato-Capuchina & Matute 2018)        |
  |   - Genomic Shock Hypothesis (McClintock 1984)                    |
  +==================================================================+

  USAGE:
  ----------------------------------------------------------------
  #include <JardinesGate_v3.mqh>

  CJardinesGateV3 gate;

  void OnInit()
  {
      gate.SetThresholds(0.90, 0.20, 0.50, 0.35);
      gate.SetNeuralParams(0.70, 7);
      gate.SetGenomicShockParams(0.8, 3.0);
  }

  void OnTick()
  {
      double entropy = CalculateEntropy();
      double interference = GetExpertAgreement();
      double confidence = GetSignalConfidence();
      int direction = GetSignalDirection();

      // Neural mosaic votes from Python bridge
      double neuron_votes[];
      int n_neurons = ReadNeuronVotes(neuron_votes);

      // Genomic shock from Python bridge
      double shock_score = ReadShockScore();

      // Cross-instrument correlation
      double cross_corr = ReadCrossCorrelation();

      // Domestication boost
      double dom_boost = ReadDomesticationBoost();

      if(!gate.ShouldTradeV3(entropy, interference, confidence,
                              direction, neuron_votes, n_neurons,
                              shock_score, cross_corr, dom_boost))
          return;  // Blocked

      // Execute trade...
  }
*/

//+------------------------------------------------------------------+
//| Enums                                                            |
//+------------------------------------------------------------------+
enum ENUM_JGV3_GATE
{
   GATE_V3_PASSED = 0,
   GATE_V3_1_ENTROPY = 1,
   GATE_V3_2_INTERFERENCE = 2,
   GATE_V3_3_CONFIDENCE = 3,
   GATE_V3_4_PROBABILITY = 4,
   GATE_V3_5_DIRECTION = 5,
   GATE_V3_6_KILLSWITCH = 6,
   GATE_V3_7_NEURAL_CONSENSUS = 7,
   GATE_V3_8_GENOMIC_SHOCK = 8,
   GATE_V3_9_SPECIATION = 9,
   GATE_V3_10_DOMESTICATION = 10
};

enum ENUM_JGV3_SHOCK
{
   SHOCK_CALM = 0,
   SHOCK_NORMAL = 1,
   SHOCK_ELEVATED = 2,
   SHOCK_ACTIVE = 3,
   SHOCK_EXTREME = 4
};

enum ENUM_JGV3_SPECIATION
{
   SPEC_NO_DONOR = 0,
   SPEC_SAME_SPECIES = 1,
   SPEC_HYBRID_ZONE = 2,
   SPEC_REPRODUCTIVE_ISOLATION = 3
};

//+------------------------------------------------------------------+
//| Trade Result Structure                                           |
//+------------------------------------------------------------------+
struct JGV3_TradeResult
{
   datetime time;
   int      direction;
   double   pnl;
   bool     win;
   string   active_tes;  // CSV of active TE names
};

//+------------------------------------------------------------------+
//| Main Filter Class v3.0                                           |
//+------------------------------------------------------------------+
class CJardinesGateV3
{
private:
   //--- Original thresholds (G1-G6)
   double m_entropy_clean;
   double m_confidence_min;
   double m_interference_min;
   double m_probability_min;
   int    m_direction_bias;
   double m_kill_switch_wr;
   int    m_kill_switch_lookback;

   //--- NEW: Neural mosaic params (G7)
   double m_neural_consensus_min;
   int    m_expected_neurons;

   //--- NEW: Genomic shock params (G8)
   double m_shock_calm_threshold;
   double m_shock_extreme_threshold;
   ENUM_JGV3_SHOCK m_current_shock;

   //--- NEW: Speciation params (G9)
   double m_speciation_same_species;
   double m_speciation_hybrid_zone;

   //--- NEW: Domestication params (G10)
   double m_domestication_min_boost;

   //--- State
   JGV3_TradeResult m_trade_history[];
   int              m_history_size;
   bool             m_debug_mode;
   ENUM_JGV3_GATE   m_last_block_gate;
   string           m_last_block_reason;

   //--- Stats
   int m_total_checks;
   int m_passed;
   int m_blocked[11];  // Index 1-10 for each gate

public:
   //+------------------------------------------------------------------+
   //| Constructor                                                      |
   //+------------------------------------------------------------------+
   CJardinesGateV3()
   {
      // Original thresholds
      m_entropy_clean        = 0.90;
      m_confidence_min       = 0.20;
      m_interference_min     = 0.50;
      m_probability_min      = 0.35;
      m_direction_bias       = -1;   // SHORT bias
      m_kill_switch_wr       = 0.50;
      m_kill_switch_lookback = 10;

      // Neural mosaic (G7)
      m_neural_consensus_min = 0.70;
      m_expected_neurons     = 7;

      // Genomic shock (G8)
      m_shock_calm_threshold    = 0.8;
      m_shock_extreme_threshold = 3.0;
      m_current_shock           = SHOCK_NORMAL;

      // Speciation (G9)
      m_speciation_same_species = 0.6;
      m_speciation_hybrid_zone  = 0.3;

      // Domestication (G10)
      m_domestication_min_boost = 1.0;

      // State
      m_history_size = 0;
      m_debug_mode = true;
      m_last_block_gate = GATE_V3_PASSED;
      m_last_block_reason = "";

      m_total_checks = 0;
      m_passed = 0;
      ArrayInitialize(m_blocked, 0);
      ArrayResize(m_trade_history, 0);
   }

   //+------------------------------------------------------------------+
   //| Configuration                                                    |
   //+------------------------------------------------------------------+
   void SetThresholds(double entropy_clean, double confidence_min,
                      double interference_min, double probability_min)
   {
      m_entropy_clean    = entropy_clean;
      m_confidence_min   = confidence_min;
      m_interference_min = interference_min;
      m_probability_min  = probability_min;
   }

   void SetDirectionBias(int bias) { m_direction_bias = bias; }
   void SetKillSwitch(double min_wr, int lookback)
   {
      m_kill_switch_wr = min_wr;
      m_kill_switch_lookback = lookback;
   }
   void SetDebugMode(bool enabled) { m_debug_mode = enabled; }

   // NEW v3.0 configuration
   void SetNeuralParams(double consensus_min, int expected_neurons)
   {
      m_neural_consensus_min = consensus_min;
      m_expected_neurons = expected_neurons;
   }

   void SetGenomicShockParams(double calm_thresh, double extreme_thresh)
   {
      m_shock_calm_threshold = calm_thresh;
      m_shock_extreme_threshold = extreme_thresh;
   }

   void SetSpeciationParams(double same_species, double hybrid_zone)
   {
      m_speciation_same_species = same_species;
      m_speciation_hybrid_zone = hybrid_zone;
   }

   void SetDomesticationParams(double min_boost)
   {
      m_domestication_min_boost = min_boost;
   }

   //+------------------------------------------------------------------+
   //| Entropy Factor                                                   |
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
   //| Probability Formula                                              |
   //| P(trade) = |psi|^2 x E(entropy) x interference x confidence     |
   //+------------------------------------------------------------------+
   double CalculateProbability(double amplitude_squared,
                               double entropy,
                               double interference,
                               double confidence)
   {
      double entropy_factor = CalculateEntropyFactor(entropy);
      double probability = amplitude_squared * entropy_factor * interference * confidence;
      return MathMax(0.0, MathMin(1.0, probability));
   }

   //+------------------------------------------------------------------+
   //| MAIN FILTER: 10-Gate Check                                       |
   //|                                                                   |
   //| Signal --[G1]--[G2]--[G3]--[G4]--[G5]--[G6]--                   |
   //|         --[G7]--[G8]--[G9]--[G10]--> EXECUTE                    |
   //+------------------------------------------------------------------+
   bool ShouldTradeV3(
      double entropy,
      double interference,
      double confidence,
      int    direction,
      double &neuron_votes[],
      int    n_neurons,
      double shock_score,
      double cross_corr,
      double domestication_boost,
      double amplitude_squared = 0.5
   )
   {
      m_total_checks++;
      m_last_block_gate = GATE_V3_PASSED;
      m_last_block_reason = "";

      //--- G1: Entropy Gate (unchanged from v2.0) ---
      if(entropy > m_entropy_clean)
      {
         m_blocked[1]++;
         m_last_block_gate = GATE_V3_1_ENTROPY;
         m_last_block_reason = StringFormat("entropy=%.3f > %.2f", entropy, m_entropy_clean);
         if(m_debug_mode) Print("[JGv3] G1 BLOCK: ", m_last_block_reason);
         return false;
      }

      //--- G2: Interference Gate (unchanged) ---
      if(interference < m_interference_min)
      {
         m_blocked[2]++;
         m_last_block_gate = GATE_V3_2_INTERFERENCE;
         m_last_block_reason = StringFormat("interference=%.3f < %.2f", interference, m_interference_min);
         if(m_debug_mode) Print("[JGv3] G2 BLOCK: ", m_last_block_reason);
         return false;
      }

      //--- G3: Confidence Gate (unchanged) ---
      if(confidence < m_confidence_min)
      {
         m_blocked[3]++;
         m_last_block_gate = GATE_V3_3_CONFIDENCE;
         m_last_block_reason = StringFormat("confidence=%.3f < %.2f", confidence, m_confidence_min);
         if(m_debug_mode) Print("[JGv3] G3 BLOCK: ", m_last_block_reason);
         return false;
      }

      //--- Calculate probability for G4 ---
      double probability = CalculateProbability(amplitude_squared, entropy, interference, confidence);

      //--- G4: Probability Gate (unchanged) ---
      if(probability < m_probability_min)
      {
         m_blocked[4]++;
         m_last_block_gate = GATE_V3_4_PROBABILITY;
         m_last_block_reason = StringFormat("probability=%.3f < %.2f", probability, m_probability_min);
         if(m_debug_mode) Print("[JGv3] G4 BLOCK: ", m_last_block_reason);
         return false;
      }

      //--- G5: Direction Bias Gate (unchanged) ---
      if(m_direction_bias != 0)
      {
         if(direction != m_direction_bias)
         {
            m_blocked[5]++;
            m_last_block_gate = GATE_V3_5_DIRECTION;
            string dir_str = (direction > 0) ? "LONG" : "SHORT";
            string bias_str = (m_direction_bias > 0) ? "LONG" : "SHORT";
            m_last_block_reason = StringFormat("direction=%s != %s", dir_str, bias_str);
            if(m_debug_mode) Print("[JGv3] G5 BLOCK: ", m_last_block_reason);
            return false;
         }
      }

      //--- G6: Kill Switch Gate (unchanged) ---
      if(IsKillSwitchTriggered())
      {
         m_blocked[6]++;
         m_last_block_gate = GATE_V3_6_KILLSWITCH;
         double recent_wr = GetRecentWinRate();
         m_last_block_reason = StringFormat("recent_wr=%.2f < %.2f", recent_wr, m_kill_switch_wr);
         if(m_debug_mode) Print("[JGv3] G6 BLOCK: ", m_last_block_reason);
         return false;
      }

      //=== NEW GATES (v3.0) ===

      //--- G7: Neural Mosaic Consensus ---
      double neural_consensus = 0.0;  // hoisted for debug print at end
      if(n_neurons > 0)
      {
         int long_votes = 0, short_votes = 0;
         for(int i = 0; i < n_neurons; i++)
         {
            if(neuron_votes[i] > 0) long_votes++;
            else if(neuron_votes[i] < 0) short_votes++;
         }

         double consensus = (double)MathMax(long_votes, short_votes) / n_neurons;
         neural_consensus = consensus;

         if(consensus < m_neural_consensus_min)
         {
            m_blocked[7]++;
            m_last_block_gate = GATE_V3_7_NEURAL_CONSENSUS;
            m_last_block_reason = StringFormat("neural_consensus=%.2f < %.2f (L=%d S=%d N=%d)",
                                               consensus, m_neural_consensus_min,
                                               long_votes, short_votes,
                                               n_neurons - long_votes - short_votes);
            if(m_debug_mode) Print("[JGv3] G7 BLOCK: ", m_last_block_reason);
            return false;
         }

         // Also check direction agreement with TE concordance
         int neural_dir = (long_votes > short_votes) ? 1 : -1;
         if(neural_dir != direction && direction != 0)
         {
            m_blocked[7]++;
            m_last_block_gate = GATE_V3_7_NEURAL_CONSENSUS;
            m_last_block_reason = StringFormat("neural_dir=%d != signal_dir=%d",
                                               neural_dir, direction);
            if(m_debug_mode) Print("[JGv3] G7 BLOCK: ", m_last_block_reason);
            return false;
         }
      }

      //--- G8: Genomic Shock Gate ---
      // Extreme shock = TRIM28 suppression = block all trades
      if(shock_score >= m_shock_extreme_threshold)
      {
         m_blocked[8]++;
         m_last_block_gate = GATE_V3_8_GENOMIC_SHOCK;
         m_last_block_reason = StringFormat("shock=%.2f >= %.2f (EXTREME/TRIM28)",
                                            shock_score, m_shock_extreme_threshold);
         if(m_debug_mode) Print("[JGv3] G8 BLOCK: ", m_last_block_reason);
         return false;
      }

      //--- G9: Speciation / Cross-Instrument Check ---
      // Block if in hybrid zone (conflicting cross-instrument signals)
      double abs_corr = MathAbs(cross_corr);
      if(abs_corr >= m_speciation_hybrid_zone && abs_corr < m_speciation_same_species)
      {
         // Hybrid zone: signals are ambiguous
         m_blocked[9]++;
         m_last_block_gate = GATE_V3_9_SPECIATION;
         m_last_block_reason = StringFormat("cross_corr=%.3f in hybrid zone [%.2f, %.2f)",
                                            cross_corr, m_speciation_hybrid_zone,
                                            m_speciation_same_species);
         if(m_debug_mode) Print("[JGv3] G9 BLOCK: ", m_last_block_reason);
         return false;
      }

      //--- G10: TE Domestication Validation ---
      if(domestication_boost < m_domestication_min_boost)
      {
         m_blocked[10]++;
         m_last_block_gate = GATE_V3_10_DOMESTICATION;
         m_last_block_reason = StringFormat("dom_boost=%.2f < %.2f",
                                            domestication_boost, m_domestication_min_boost);
         if(m_debug_mode) Print("[JGv3] G10 BLOCK: ", m_last_block_reason);
         return false;
      }

      //--- ALL 10 GATES PASSED ---
      m_passed++;
      if(m_debug_mode)
         Print("[JGv3] ALL 10 GATES PASSED -> EXECUTE (entropy=",
               DoubleToString(entropy,3), " prob=", DoubleToString(probability,3),
               " consensus=", DoubleToString(neural_consensus, 3),
               " shock=", DoubleToString(shock_score, 2), ")");

      return true;
   }

   //+------------------------------------------------------------------+
   //| Simplified check (backward compatible with v2.0)                 |
   //+------------------------------------------------------------------+
   bool ShouldTrade(double entropy, double interference, double confidence,
                    int direction, double amplitude_squared = 0.5)
   {
      // Call v3 with neutral neural/shock/speciation/domestication values
      double empty[];
      ArrayResize(empty, 0);
      return ShouldTradeV3(entropy, interference, confidence, direction,
                           empty, 0,     // no neurons
                           1.0,          // normal shock
                           0.0,          // no cross-corr
                           1.0,          // neutral domestication
                           amplitude_squared);
   }

   //+------------------------------------------------------------------+
   //| Kill Switch                                                       |
   //+------------------------------------------------------------------+
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
      int wins = 0, count = 0;
      for(int i = start; i < m_history_size; i++)
      {
         if(m_trade_history[i].win) wins++;
         count++;
      }
      return (count > 0) ? (double)wins / count : 1.0;
   }

   //+------------------------------------------------------------------+
   //| Record trade result                                               |
   //+------------------------------------------------------------------+
   void RecordTrade(int direction, double pnl, string active_tes = "")
   {
      int size = ArraySize(m_trade_history);
      ArrayResize(m_trade_history, size + 1);
      m_trade_history[size].time = TimeCurrent();
      m_trade_history[size].direction = direction;
      m_trade_history[size].pnl = pnl;
      m_trade_history[size].win = (pnl > 0);
      m_trade_history[size].active_tes = active_tes;
      m_history_size = size + 1;

      if(m_history_size > m_kill_switch_lookback * 2)
      {
         for(int i = 0; i < m_history_size - m_kill_switch_lookback; i++)
            m_trade_history[i] = m_trade_history[i + m_kill_switch_lookback];
         ArrayResize(m_trade_history, m_kill_switch_lookback);
         m_history_size = m_kill_switch_lookback;
      }
   }

   //+------------------------------------------------------------------+
   //| Getters                                                           |
   //+------------------------------------------------------------------+
   string GetLastBlockReason()        { return m_last_block_reason; }
   ENUM_JGV3_GATE GetLastBlockGate()  { return m_last_block_gate; }

   //+------------------------------------------------------------------+
   //| Statistics                                                        |
   //+------------------------------------------------------------------+
   string GetStats()
   {
      double pass_rate = (m_total_checks > 0) ? (double)m_passed / m_total_checks * 100 : 0;

      return StringFormat(
         "\n+================================================================+\n"
         "|        JARDINE'S GATE v3.0 NEURAL-TE STATS                      |\n"
         "+================================================================+\n"
         "| Total Checks:     %6d                                       |\n"
         "| Passed:           %6d  (%.1f%%)                              |\n"
         "+----------------------------------------------------------------+\n"
         "| G1  Entropy:          %6d                                    |\n"
         "| G2  Interference:     %6d                                    |\n"
         "| G3  Confidence:       %6d                                    |\n"
         "| G4  Probability:      %6d                                    |\n"
         "| G5  Direction:        %6d                                    |\n"
         "| G6  Kill Switch:      %6d                                    |\n"
         "| G7  Neural Consensus: %6d  (NEW)                             |\n"
         "| G8  Genomic Shock:    %6d  (NEW)                             |\n"
         "| G9  Speciation:       %6d  (NEW)                             |\n"
         "| G10 Domestication:    %6d  (NEW)                             |\n"
         "+================================================================+",
         m_total_checks, m_passed, pass_rate,
         m_blocked[1], m_blocked[2], m_blocked[3],
         m_blocked[4], m_blocked[5], m_blocked[6],
         m_blocked[7], m_blocked[8], m_blocked[9], m_blocked[10]
      );
   }

   //+------------------------------------------------------------------+
   //| Reset                                                             |
   //+------------------------------------------------------------------+
   void ResetStats()
   {
      m_total_checks = 0;
      m_passed = 0;
      ArrayInitialize(m_blocked, 0);
   }
};

//+------------------------------------------------------------------+
//| Global helper (backward compatible)                              |
//+------------------------------------------------------------------+
bool JGV3_ShouldTrade(double entropy, double interference,
                       double confidence, int direction)
{
   static CJardinesGateV3 gate;
   return gate.ShouldTrade(entropy, interference, confidence, direction);
}

//+------------------------------------------------------------------+
