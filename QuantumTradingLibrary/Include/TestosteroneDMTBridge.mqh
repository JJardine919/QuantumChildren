//+------------------------------------------------------------------+
//|                                     TestosteroneDMTBridge.mqh    |
//|                              Quantum Children Trading Systems    |
//|                     Testosterone-DMT TE Bridge -- MQL5 Side      |
//+------------------------------------------------------------------+
#property copyright "Quantum Children"
#property version   "1.00"

/*
  +==================================================================+
  |       TESTOSTERONE-DMT TE BRIDGE -- MQL5 INTERFACE                |
  +==================================================================+
  | Reads testosterone_dmt_signal.json written by the Python bridge  |
  | (testosterone_dmt_bridge.py) and exposes the bridge decision to  |
  | MQL5 Expert Advisors.                                             |
  |                                                                   |
  | Architecture:                                                     |
  |   8-qubit quantum circuit / 4096 shots                            |
  |   4 heavy processing layers (Testosterone's 4 rings)              |
  |   5-channel DMT pattern recognition                               |
  |   4 strict decision gates (ALL must pass)                         |
  |   Aromatase controller: aggressive -> defensive transition        |
  |                                                                   |
  | Strategy Profile: AGGRESSIVE TREND-FOLLOWING                      |
  |   - Wider stop losses (hold through noise)                        |
  |   - Bigger take profit targets (let winners run)                  |
  |   - Position size multipliers (1.5x to 2.5x in full T mode)      |
  |   - Aromatase flip (defensive when drawdown or vol spike)         |
  |                                                                   |
  | Signal flow:                                                      |
  |   Python TEQA -> testosterone_dmt_bridge.py -> JSON signal file   |
  |   -> TestosteroneDMTBridge (this file) -> EA                      |
  |                                                                   |
  | Regime States (Aromatase Controller):                             |
  |   FULL_TESTOSTERONE (aggressive) -- trend following, wide stops   |
  |   AROMATIZING (transitional) -- balanced, normal sizing           |
  |   FULL_ESTROGEN (defensive) -- tight stops, reduced sizing        |
  +==================================================================+

  USAGE:
  ----------------------------------------------------------------
  #include <TestosteroneDMTBridge.mqh>

  TestosteroneDMTBridge testoBridge;

  void OnInit()
  {
      testoBridge.SetSignalFile("testosterone_dmt_signal.json");
      testoBridge.SetStaleTimeout(120);
  }

  void OnTick()
  {
      if(!testoBridge.ReadSignal())
          return;

      TESTO_Signal sig = testoBridge.GetSignal();

      // sig.action               = "boost", "suppress", "neutral"
      // sig.strength             = signal strength (0-1)
      // sig.position_multiplier  = lot size multiplier
      // sig.stop_multiplier      = SL distance multiplier
      // sig.target_multiplier    = TP distance multiplier
      // sig.regime               = "aggressive" / "transitional" / "defensive"
      // sig.all_gates_passed     = true if ALL 4 gates passed
      // sig.trend_direction      = 1 (up) or -1 (down)

      if(sig.action == "boost" && sig.all_gates_passed)
      {
          double lots = BaseLotSize() * sig.position_multiplier;
          double sl   = BaseStopLoss() * sig.stop_multiplier;
          double tp   = sl * sig.target_multiplier;
          // Execute trade with adjusted parameters
      }
  }
*/

#include <TransposableEdge.mqh>

//+------------------------------------------------------------------+
//| Parsed bridge signal structure                                    |
//+------------------------------------------------------------------+
struct TESTO_Signal
{
   // Core decision
   string   action;               // "boost", "suppress", "neutral"
   double   strength;             // Signal strength (0-1)

   // Testosterone layers
   double   trend_strength;       // Layer 1: Trend detection output
   int      trend_direction;      // Layer 1: 1=up, -1=down, 0=range
   double   momentum;             // Layer 2: Momentum amplification
   double   acceleration;         // Layer 2: Momentum acceleration
   bool     dht_converted;        // Layer 2: DHT conversion active

   // Position sizing (Anabolic:Androgenic ratio)
   double   position_multiplier;  // 0.5x to 2.5x
   double   anabolic_component;   // Growth/profit component
   double   androgenic_component; // Aggression/entry component

   // Exit timing (Aromatase)
   string   exit_strategy;        // "trend_following", "balanced", "defensive"
   double   stop_multiplier;      // SL distance multiplier (0.8 to 1.5)
   double   target_multiplier;    // TP distance multiplier (1.5 to 3.0)
   string   regime;               // "aggressive", "transitional", "defensive"
   double   aromatization_level;  // 0.0 (full T) to 1.0 (full E)

   // DMT pattern recognition
   double   dmt_consensus;        // Pattern consensus rate
   double   dmt_confidence;       // Average DMT confidence
   int      dmt_polarity;         // Dominant direction (-1, 0, +1)
   bool     all_channels_agree;   // All 5 DMT channels aligned

   // Decision gates (4 gates, ALL must pass)
   bool     all_gates_passed;     // True = trade authorized
   int      gates_passed_count;   // How many of 4 gates passed

   // Quantum circuit
   double   vote_long;            // Quantum vote for LONG
   double   vote_short;           // Quantum vote for SHORT
   double   vote_bias;            // net bias (long - short)
   double   shannon_entropy;      // Measurement entropy
   double   novelty;              // Entropy / max entropy

   // Meta
   double   processing_time_ms;
   string   version;
   datetime signal_time;
   bool     valid;
};

//+------------------------------------------------------------------+
//| Testosterone-DMT Bridge Reader                                   |
//+------------------------------------------------------------------+
class TestosteroneDMTBridge
{
private:
   string       m_filename;
   int          m_stale_timeout;
   TESTO_Signal m_signal;
   datetime     m_last_read;

   bool ParseJSON(string json);
   string ExtractString(string json, string key);
   double ExtractDouble(string json, string key);
   int    ExtractInt(string json, string key);
   bool   ExtractBool(string json, string key);

public:
   TestosteroneDMTBridge()
   {
      m_filename = "testosterone_dmt_signal.json";
      m_stale_timeout = 120;
      m_last_read = 0;
      ZeroMemory(m_signal);
   }

   void SetSignalFile(string filename) { m_filename = filename; }
   void SetStaleTimeout(int seconds)   { m_stale_timeout = seconds; }

   //--- Read and parse the signal file
   bool ReadSignal();

   //--- Check if signal is fresh
   bool IsFresh();

   //--- Get the parsed signal
   TESTO_Signal GetSignal() { return m_signal; }

   //--- Quick checks
   bool ShouldTrade();           // All gates passed + action is boost
   bool IsAggressive();          // Full testosterone regime
   bool IsDefensive();           // Full estrogen regime (aromatized)

   //--- Get sizing parameters
   double GetPositionMultiplier();
   double GetStopMultiplier();
   double GetTargetMultiplier();

   //--- Get quantum vote
   double GetQuantumBias();
};

//+------------------------------------------------------------------+
//| Read and parse the JSON signal file                              |
//+------------------------------------------------------------------+
bool TestosteroneDMTBridge::ReadSignal()
{
   ZeroMemory(m_signal);
   m_signal.valid = false;

   int handle = FileOpen(m_filename, FILE_READ|FILE_TXT|FILE_COMMON);
   if(handle == INVALID_HANDLE)
   {
      handle = FileOpen(m_filename, FILE_READ|FILE_TXT);
      if(handle == INVALID_HANDLE)
         return false;
   }

   string json = "";
   while(!FileIsEnding(handle))
   {
      json += FileReadString(handle);
   }
   FileClose(handle);

   if(StringLen(json) < 10)
      return false;

   return ParseJSON(json);
}

//+------------------------------------------------------------------+
//| Parse JSON into TESTO_Signal                                     |
//+------------------------------------------------------------------+
bool TestosteroneDMTBridge::ParseJSON(string json)
{
   // Core decision
   m_signal.action             = ExtractString(json, "action");
   m_signal.strength           = ExtractDouble(json, "strength");

   // Testosterone layers
   m_signal.trend_strength     = ExtractDouble(json, "trend_strength");
   m_signal.trend_direction    = ExtractInt(json, "trend_direction");
   m_signal.momentum           = ExtractDouble(json, "momentum");
   m_signal.acceleration       = ExtractDouble(json, "acceleration");
   m_signal.dht_converted      = ExtractBool(json, "dht_converted");

   // Position sizing
   m_signal.position_multiplier  = ExtractDouble(json, "position_size_multiplier");
   m_signal.anabolic_component   = ExtractDouble(json, "anabolic_component");
   m_signal.androgenic_component = ExtractDouble(json, "androgenic_component");

   // Exit timing
   m_signal.exit_strategy      = ExtractString(json, "exit_strategy");
   m_signal.stop_multiplier    = ExtractDouble(json, "stop_multiplier");
   m_signal.target_multiplier  = ExtractDouble(json, "target_multiplier");
   m_signal.regime             = ExtractString(json, "regime");
   m_signal.aromatization_level = ExtractDouble(json, "aromatization_level");

   // DMT
   m_signal.dmt_consensus      = ExtractDouble(json, "consensus_rate");
   m_signal.dmt_confidence     = ExtractDouble(json, "avg_confidence");
   m_signal.dmt_polarity       = ExtractInt(json, "dominant_polarity");
   m_signal.all_channels_agree = ExtractBool(json, "all_channels_agree");

   // Gates
   m_signal.all_gates_passed   = ExtractBool(json, "all_gates_passed");
   m_signal.gates_passed_count = ExtractInt(json, "gates_passed_count");

   // Quantum
   m_signal.vote_long          = ExtractDouble(json, "vote_long");
   m_signal.vote_short         = ExtractDouble(json, "vote_short");
   m_signal.vote_bias          = ExtractDouble(json, "vote_bias");
   m_signal.shannon_entropy    = ExtractDouble(json, "shannon_entropy");
   m_signal.novelty            = ExtractDouble(json, "novelty");

   // Meta
   m_signal.processing_time_ms = ExtractDouble(json, "processing_time_ms");
   m_signal.version            = ExtractString(json, "version");
   m_signal.signal_time        = TimeCurrent();
   m_signal.valid              = true;
   m_last_read                 = TimeCurrent();

   return true;
}

//+------------------------------------------------------------------+
//| Check if signal is fresh                                         |
//+------------------------------------------------------------------+
bool TestosteroneDMTBridge::IsFresh()
{
   if(!m_signal.valid) return false;
   return (TimeCurrent() - m_last_read) < m_stale_timeout;
}

//+------------------------------------------------------------------+
//| Should we trade? All 4 gates must pass + action is boost         |
//+------------------------------------------------------------------+
bool TestosteroneDMTBridge::ShouldTrade()
{
   if(!m_signal.valid || !IsFresh()) return false;
   return m_signal.all_gates_passed && m_signal.action == "boost";
}

//+------------------------------------------------------------------+
//| Is the system in full aggressive (testosterone) mode?            |
//+------------------------------------------------------------------+
bool TestosteroneDMTBridge::IsAggressive()
{
   return m_signal.regime == "aggressive";
}

//+------------------------------------------------------------------+
//| Is the system in defensive (estrogen/aromatized) mode?           |
//+------------------------------------------------------------------+
bool TestosteroneDMTBridge::IsDefensive()
{
   return m_signal.regime == "defensive";
}

//+------------------------------------------------------------------+
//| Get position size multiplier                                     |
//+------------------------------------------------------------------+
double TestosteroneDMTBridge::GetPositionMultiplier()
{
   if(!m_signal.valid) return 1.0;
   return m_signal.position_multiplier;
}

//+------------------------------------------------------------------+
//| Get stop loss distance multiplier                                |
//+------------------------------------------------------------------+
double TestosteroneDMTBridge::GetStopMultiplier()
{
   if(!m_signal.valid) return 1.0;
   return m_signal.stop_multiplier;
}

//+------------------------------------------------------------------+
//| Get take profit distance multiplier                              |
//+------------------------------------------------------------------+
double TestosteroneDMTBridge::GetTargetMultiplier()
{
   if(!m_signal.valid) return 2.0;
   return m_signal.target_multiplier;
}

//+------------------------------------------------------------------+
//| Get quantum circuit vote bias                                    |
//+------------------------------------------------------------------+
double TestosteroneDMTBridge::GetQuantumBias()
{
   if(!m_signal.valid) return 0.0;
   return m_signal.vote_bias;
}

//+------------------------------------------------------------------+
//| JSON helpers                                                     |
//+------------------------------------------------------------------+
string TestosteroneDMTBridge::ExtractString(string json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";

   int colon = StringFind(json, ":", pos + StringLen(search));
   if(colon < 0) return "";

   int q1 = StringFind(json, "\"", colon + 1);
   if(q1 < 0) return "";

   int q2 = StringFind(json, "\"", q1 + 1);
   if(q2 < 0) return "";

   return StringSubstr(json, q1 + 1, q2 - q1 - 1);
}

double TestosteroneDMTBridge::ExtractDouble(string json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return 0.0;

   int colon = StringFind(json, ":", pos + StringLen(search));
   if(colon < 0) return 0.0;

   string numStr = "";
   int i = colon + 1;
   int len = StringLen(json);
   while(i < len && (StringGetCharacter(json, i) == ' ' || StringGetCharacter(json, i) == '\t'))
      i++;

   while(i < len)
   {
      ushort ch = StringGetCharacter(json, i);
      if((ch >= '0' && ch <= '9') || ch == '.' || ch == '-' || ch == '+' || ch == 'e' || ch == 'E')
         numStr += ShortToString(ch);
      else
         break;
      i++;
   }

   return StringToDouble(numStr);
}

int TestosteroneDMTBridge::ExtractInt(string json, string key)
{
   return (int)ExtractDouble(json, key);
}

bool TestosteroneDMTBridge::ExtractBool(string json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return false;

   int colon = StringFind(json, ":", pos + StringLen(search));
   if(colon < 0) return false;

   // Look for "true" or "false" after colon
   int truePos = StringFind(json, "true", colon);
   int falsePos = StringFind(json, "false", colon);

   if(truePos > 0 && (falsePos < 0 || truePos < falsePos))
      return true;
   return false;
}
//+------------------------------------------------------------------+
