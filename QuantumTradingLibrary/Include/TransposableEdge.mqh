//+------------------------------------------------------------------+
//|                                         TransposableEdge.mqh     |
//|                              Quantum Children Trading Systems    |
//|                     MQL5 Bridge: Python TEQA -> Jardine's Gate   |
//+------------------------------------------------------------------+
#property copyright "Quantum Children"
#property version   "1.00"

/*
  +==================================================================+
  |              TRANSPOSABLE EDGE -- MQL5 BRIDGE                     |
  +==================================================================+
  | Reads te_quantum_signal.json written by the Python TEQA engine   |
  | and feeds values through the QuantumEdgeFilter 6-gate system.    |
  |                                                                   |
  | Signal flow:                                                      |
  |   Python TEQA -> te_quantum_signal.json -> TransposableEdge      |
  |              -> QuantumEdgeFilter.ShouldTrade() -> EA             |
  |                                                                   |
  | TE-specific pre-filters (applied before gates):                   |
  |   - pirna_silenced:    piRNA silencing -> block trade             |
  |   - shock_active:      genomic shock   -> block trade             |
  |   - ectopic_inversion: ectopic TE      -> flip direction          |
  +==================================================================+

  USAGE:
  ----------------------------------------------------------------
  #include <TransposableEdge.mqh>

  TransposableEdge te;

  void OnInit()
  {
      te.SetStaleTimeout(120);  // signal older than 120s = stale
  }

  void OnTick()
  {
      if(!te.ShouldTrade())
          return;

      TE_Signal sig = te.GetLastSignal();

      // sig.direction   = 1 (LONG) or -1 (SHORT)
      // sig.confidence  = 0.0-1.0
      // sig.lot_scale   = position size multiplier

      // Execute trade with sig values...
  }
*/

#include <QuantumEdgeFilter.mqh>

//+------------------------------------------------------------------+
//| Parsed signal structure                                          |
//+------------------------------------------------------------------+
struct TE_Signal
{
   int      direction;        // 1=LONG, -1=SHORT
   double   confidence;       // 0.0-1.0
   double   entropy_adj;      // entropy adjustment
   double   interference;     // expert agreement
   double   amplitude_sq;     // |psi|^2
   double   lot_scale;        // position size multiplier
   bool     pirna_silenced;   // piRNA silencing active
   bool     shock_active;     // genomic shock active
   bool     ectopic_inversion;// ectopic TE inversion
   datetime parse_time;       // when signal was parsed
   bool     valid;            // parse succeeded
};

//+------------------------------------------------------------------+
//| Main bridge class                                                |
//+------------------------------------------------------------------+
class TransposableEdge
{
private:
   QuantumEdgeFilter  m_filter;
   TE_Signal          m_signal;
   string             m_filename;
   int                m_stale_seconds;
   bool               m_debug;
   string             m_last_block;

   //--- JSON parsing helpers ---

   bool ReadJsonFile(string &content)
   {
      int handle = FileOpen(m_filename, FILE_READ | FILE_TXT | FILE_ANSI);
      if(handle == INVALID_HANDLE)
      {
         if(m_debug)
            Print("[TE] FileOpen failed: ", m_filename, " error=", GetLastError());
         return false;
      }

      content = "";
      while(!FileIsEnding(handle))
         content += FileReadString(handle) + "\n";

      FileClose(handle);
      return (StringLen(content) > 2);
   }

   double ParseDouble(const string &json, const string &key)
   {
      string search = "\"" + key + "\"";
      int pos = StringFind(json, search);
      if(pos < 0) return 0.0;

      // Find colon after key
      int colon = StringFind(json, ":", pos + StringLen(search));
      if(colon < 0) return 0.0;

      // Extract value: skip whitespace, read until , or } or \n
      int start = colon + 1;
      int len = StringLen(json);

      // Skip whitespace
      while(start < len)
      {
         ushort ch = StringGetCharacter(json, start);
         if(ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
            break;
         start++;
      }

      // Find end of value
      int end = start;
      while(end < len)
      {
         ushort ch = StringGetCharacter(json, end);
         if(ch == ',' || ch == '}' || ch == '\n' || ch == '\r')
            break;
         end++;
      }

      string val = StringSubstr(json, start, end - start);
      StringTrimRight(val);
      StringTrimLeft(val);

      return StringToDouble(val);
   }

   int ParseInt(const string &json, const string &key)
   {
      return (int)ParseDouble(json, key);
   }

   bool ParseBool(const string &json, const string &key)
   {
      string search = "\"" + key + "\"";
      int pos = StringFind(json, search);
      if(pos < 0) return false;

      int colon = StringFind(json, ":", pos + StringLen(search));
      if(colon < 0) return false;

      // Look for "true" after the colon
      int check_end = MathMin(colon + 20, StringLen(json));
      string fragment = StringSubstr(json, colon + 1, check_end - colon - 1);
      StringToLower(fragment);

      return (StringFind(fragment, "true") >= 0);
   }

   string ParseString(const string &json, const string &key)
   {
      string search = "\"" + key + "\"";
      int pos = StringFind(json, search);
      if(pos < 0) return "";

      int colon = StringFind(json, ":", pos + StringLen(search));
      if(colon < 0) return "";

      int quote1 = StringFind(json, "\"", colon + 1);
      if(quote1 < 0) return "";

      int quote2 = StringFind(json, "\"", quote1 + 1);
      if(quote2 < 0) return "";

      return StringSubstr(json, quote1 + 1, quote2 - quote1 - 1);
   }

   bool IsSignalStale(const string &json)
   {
      if(m_stale_seconds <= 0) return false;

      string ts = ParseString(json, "timestamp");
      if(StringLen(ts) < 19) return false;

      // Parse "2026-02-07T15:44:03" format
      string date_part = StringSubstr(ts, 0, 10);   // 2026-02-07
      string time_part = StringSubstr(ts, 11, 8);    // 15:44:03

      // Build MQL5 datetime string "2026.02.07 15:44:03"
      StringReplace(date_part, "-", ".");
      string dt_str = date_part + " " + time_part;

      datetime signal_time = StringToTime(dt_str);
      datetime now = TimeCurrent();

      if(signal_time <= 0) return false;

      long age = (long)(now - signal_time);
      if(age > m_stale_seconds)
      {
         if(m_debug)
            Print("[TE] Signal stale: age=", age, "s > ", m_stale_seconds, "s");
         return true;
      }
      return false;
   }

public:
   //+------------------------------------------------------------------+
   //| Constructor                                                      |
   //+------------------------------------------------------------------+
   TransposableEdge()
   {
      m_filename       = "te_quantum_signal.json";
      m_stale_seconds  = 120;
      m_debug          = true;
      m_last_block     = "";

      ZeroMemory(m_signal);
      m_signal.valid = false;
   }

   //+------------------------------------------------------------------+
   //| Configuration                                                    |
   //+------------------------------------------------------------------+
   void SetFilename(string filename)       { m_filename = filename; }
   void SetStaleTimeout(int seconds)       { m_stale_seconds = seconds; }
   void SetDebugMode(bool enabled)         { m_debug = enabled; m_filter.SetDebugMode(enabled); }

   // Pass-through to QuantumEdgeFilter configuration
   void SetThresholds(double entropy_clean, double confidence_min,
                      double interference_min, double probability_min)
   {
      m_filter.SetThresholds(entropy_clean, confidence_min,
                             interference_min, probability_min);
   }

   void SetDirectionBias(int bias)         { m_filter.SetDirectionBias(bias); }
   void SetKillSwitch(double wr, int n)    { m_filter.SetKillSwitch(wr, n); }

   //+------------------------------------------------------------------+
   //| GetSignal: Read JSON, parse, return key values                   |
   //| Returns: true if valid signal parsed                             |
   //+------------------------------------------------------------------+
   bool GetSignal(int &direction, double &confidence, double &lot_scale)
   {
      string json = "";
      if(!ReadJsonFile(json))
      {
         m_signal.valid = false;
         m_last_block = "file_read_failed";
         return false;
      }

      if(IsSignalStale(json))
      {
         m_signal.valid = false;
         m_last_block = "signal_stale";
         return false;
      }

      // Parse jardines_gate section
      m_signal.direction    = ParseInt(json, "direction");
      m_signal.confidence   = ParseDouble(json, "confidence");
      m_signal.entropy_adj  = ParseDouble(json, "entropy_adj");
      m_signal.interference = ParseDouble(json, "interference");
      m_signal.amplitude_sq = ParseDouble(json, "amplitude_sq");

      // Parse position section
      m_signal.lot_scale    = ParseDouble(json, "lot_scale");

      // Parse filters section
      m_signal.pirna_silenced    = ParseBool(json, "pirna_silenced");
      m_signal.shock_active      = ParseBool(json, "shock_active");
      m_signal.ectopic_inversion = ParseBool(json, "ectopic_inversion");

      m_signal.parse_time = TimeCurrent();
      m_signal.valid      = true;

      // Apply ectopic inversion (TE inserted in wrong orientation -> flip)
      if(m_signal.ectopic_inversion)
      {
         m_signal.direction *= -1;
         if(m_debug)
            Print("[TE] Ectopic inversion applied: direction flipped to ", m_signal.direction);
      }

      // Sanity: direction must be 1 or -1
      if(m_signal.direction != 1 && m_signal.direction != -1)
      {
         m_signal.valid = false;
         m_last_block = StringFormat("invalid_direction=%d", m_signal.direction);
         if(m_debug)
            Print("[TE] Invalid direction: ", m_signal.direction);
         return false;
      }

      // Output
      direction  = m_signal.direction;
      confidence = m_signal.confidence;
      lot_scale  = m_signal.lot_scale;

      if(m_debug)
         Print("[TE] Signal parsed: dir=", direction,
               " conf=", DoubleToString(confidence, 4),
               " lot=", DoubleToString(lot_scale, 2),
               " entropy=", DoubleToString(m_signal.entropy_adj, 4),
               " interf=", DoubleToString(m_signal.interference, 4),
               " amp2=", DoubleToString(m_signal.amplitude_sq, 6),
               " pirna=", m_signal.pirna_silenced,
               " shock=", m_signal.shock_active,
               " ectopic=", m_signal.ectopic_inversion);

      return true;
   }

   //+------------------------------------------------------------------+
   //| ShouldTrade: Read signal + run through QuantumEdgeFilter gates   |
   //| Returns: true only if signal is valid AND passes all 6 gates     |
   //+------------------------------------------------------------------+
   bool ShouldTrade()
   {
      int    direction  = 0;
      double confidence = 0;
      double lot_scale  = 0;

      if(!GetSignal(direction, confidence, lot_scale))
         return false;

      //--- TE pre-filter: piRNA silencing blocks everything
      if(m_signal.pirna_silenced)
      {
         m_last_block = "pirna_silenced";
         if(m_debug)
            Print("[TE] BLOCKED: piRNA silencing active - transposon suppressed");
         return false;
      }

      //--- TE pre-filter: genomic shock blocks everything
      if(m_signal.shock_active)
      {
         m_last_block = "shock_active";
         if(m_debug)
            Print("[TE] BLOCKED: Genomic shock active - TRIM28 suppression");
         return false;
      }

      //--- Pass through the 6-gate filter
      bool pass = m_filter.ShouldTrade(
         m_signal.entropy_adj,
         m_signal.interference,
         m_signal.confidence,
         m_signal.direction,
         m_signal.amplitude_sq
      );

      if(!pass)
      {
         m_last_block = "gate:" + m_filter.GetLastBlockReason();
         return false;
      }

      m_last_block = "";
      return true;
   }

   //+------------------------------------------------------------------+
   //| Record a completed trade (for kill switch tracking)              |
   //+------------------------------------------------------------------+
   void RecordTrade(int direction, double pnl)
   {
      m_filter.RecordTrade(direction, pnl);
   }

   //+------------------------------------------------------------------+
   //| Accessors                                                        |
   //+------------------------------------------------------------------+
   TE_Signal  GetLastSignal()      { return m_signal; }
   string     GetLastBlockReason() { return m_last_block; }
   string     GetFilterStats()     { return m_filter.GetStats(); }

   void PrintThresholds()          { m_filter.PrintThresholds(); }
};

//+------------------------------------------------------------------+
//| Global helper for simple usage                                   |
//+------------------------------------------------------------------+
bool TE_ShouldTrade()
{
   static TransposableEdge te;
   return te.ShouldTrade();
}

//+------------------------------------------------------------------+
