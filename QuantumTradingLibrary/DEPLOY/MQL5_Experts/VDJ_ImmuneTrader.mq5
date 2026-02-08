//+------------------------------------------------------------------+
//|                                            VDJ_ImmuneTrader.mq5  |
//|                     Quantum Trading Library -- Immune System EA   |
//|                                                                  |
//|  Reads VDJ antibody consensus signals from Python VDJ engine     |
//|  and integrates with Jardine's Gate for trade execution.          |
//|                                                                  |
//|  Biological basis: RAG1/RAG2 V(D)J recombination                 |
//|  The EA is the "effector cell" that acts on antibody signals.     |
//+------------------------------------------------------------------+
#property copyright "Quantum Trading Library"
#property link      "https://quantumtradinglib.com"
#property version   "1.00"
#property strict
#property description "VDJ Immune System Trader - reads antibody consensus from Python engine"

//+------------------------------------------------------------------+
//| INPUT PARAMETERS                                                  |
//+------------------------------------------------------------------+
input group "=== ACCOUNT SETTINGS ==="
input string   AccountName       = "VDJ_IMMUNE";      // Account identifier
input int      MagicNumber       = 700100;             // Magic number
input string   SignalFilePath    = "C:\\Users\\jimjj\\Music\\QuantumChildren\\QuantumTradingLibrary\\vdj_antibody_signal.json";

input group "=== TRADING SETTINGS ==="
input double   BaseLot           = 0.01;               // Base lot size
input double   MaxLossDollars    = 1.00;               // Max loss per trade ($)
input double   TPMultiplier      = 3.0;                // TP = SL * this
input int      MaxPositions      = 3;                  // Max simultaneous positions
input int      CheckIntervalSec  = 30;                 // Signal check interval (seconds)

input group "=== VDJ IMMUNE SYSTEM SETTINGS ==="
input double   MinConsensusConf  = 0.60;               // Min antibody consensus confidence
input int      MinActiveAntibody = 2;                  // Min active antibodies for signal
input double   MinAntibodyWR     = 0.60;               // Min individual antibody win rate
input double   MinAntibodyPF     = 1.2;                // Min individual antibody profit factor
input bool     RequireGatePass   = true;               // Require VDJ gate pass from Python
input int      SignalStaleSeconds= 120;                // Signal considered stale after N sec

input group "=== RISK MANAGEMENT ==="
input double   DailyDDLimitPct   = 4.5;                // Daily DD limit %
input double   MaxDDLimitPct     = 9.0;                // Max DD limit %
input bool     UseTrailingStop   = true;               // Enable trailing stop
input double   TrailActivationATR= 1.0;                // Activate trail after N * ATR profit
input double   TrailDistanceATR  = 1.5;                // Trail distance = N * ATR

input group "=== DEBUG ==="
input bool     DebugMode         = true;               // Print debug info
input bool     DryRun            = false;              // Simulate only (no real trades)


//+------------------------------------------------------------------+
//| STRUCTURES                                                        |
//+------------------------------------------------------------------+

// Individual antibody signal from JSON
struct AntibodySignal
{
   string id;
   int    direction;       // 1=LONG, -1=SHORT
   double confidence;
   double win_rate;
   double profit_factor;
   string v_type;          // Entry signal type
   string d_type;          // Regime filter type
   string j_type;          // Exit strategy type
};

// Parsed VDJ signal from JSON file
struct VDJSignal
{
   string   version;
   datetime timestamp;
   int      direction;
   double   confidence;
   bool     gate_pass;
   int      n_active;
   int      n_long;
   int      n_short;
   double   weighted_long;
   double   weighted_short;
   int      memory_cells_total;
   int      generation;
   // Individual antibodies
   AntibodySignal antibodies[];
   // Validity
   bool     is_valid;
   bool     is_stale;
};

// Trade tracking for feedback
struct ImmuneTradeRecord
{
   ulong    ticket;
   string   antibody_id;       // Which antibody triggered this
   int      direction;
   double   entry_price;
   datetime entry_time;
   double   sl;
   double   tp;
   string   v_type;
   string   d_type;
   string   j_type;
};


//+------------------------------------------------------------------+
//| GLOBAL VARIABLES                                                  |
//+------------------------------------------------------------------+
datetime   g_last_check = 0;
datetime   g_last_signal_time = 0;
VDJSignal  g_current_signal;
ImmuneTradeRecord g_trades[];
int        g_total_trades = 0;
int        g_wins = 0;
int        g_losses = 0;

// Daily tracking
double     g_daily_start_balance = 0;
datetime   g_daily_start_date = 0;


//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== VDJ IMMUNE SYSTEM TRADER v1.00 ===");
   Print("Account:       ", AccountName);
   Print("Magic:         ", MagicNumber);
   Print("Signal file:   ", SignalFilePath);
   Print("Min consensus: ", MinConsensusConf);
   Print("Min antibodies:", MinActiveAntibody);
   Print("Max loss:      $", DoubleToString(MaxLossDollars, 2));
   Print("TP multiplier: ", TPMultiplier);
   Print("Dry run:       ", DryRun ? "YES" : "NO");
   Print("========================================");

   // Initialize daily tracking
   g_daily_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_daily_start_date = TimeCurrent();

   // Set timer for periodic checks
   EventSetTimer(CheckIntervalSec);

   return(INIT_SUCCEEDED);
}


//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("VDJ Immune Trader stopped. Trades: ", g_total_trades,
         " W:", g_wins, " L:", g_losses);
}


//+------------------------------------------------------------------+
//| Timer function -- main execution loop                              |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Reset daily stats at new day
   MqlDateTime dt;
   TimeCurrent(dt);
   MqlDateTime dt_start;
   TimeToStruct(g_daily_start_date, dt_start);
   if(dt.day != dt_start.day)
   {
      g_daily_start_balance = AccountInfoDouble(ACCOUNT_BALANCE);
      g_daily_start_date = TimeCurrent();
   }

   // Check drawdown limits
   if(!CheckDrawdownLimits())
   {
      if(DebugMode) Print("[VDJ] Drawdown limit reached -- holding");
      return;
   }

   // Read VDJ signal file
   g_current_signal = ReadVDJSignal();

   if(!g_current_signal.is_valid)
   {
      if(DebugMode) Print("[VDJ] No valid signal");
      return;
   }

   if(g_current_signal.is_stale)
   {
      if(DebugMode) Print("[VDJ] Signal is stale (>", SignalStaleSeconds, "s old)");
      return;
   }

   // Manage existing positions (trailing stop, etc.)
   ManageExistingPositions();

   // Check if we should open new positions
   if(CountMyPositions() >= MaxPositions)
   {
      if(DebugMode) Print("[VDJ] Max positions reached (", MaxPositions, ")");
      return;
   }

   // Apply immune system gates
   if(!PassImmuneGates(g_current_signal))
   {
      if(DebugMode) Print("[VDJ] Immune gates BLOCKED trade");
      return;
   }

   // Execute trade based on antibody consensus
   ExecuteImmuneTrade(g_current_signal);
}


//+------------------------------------------------------------------+
//| Read and parse VDJ signal JSON file                                |
//+------------------------------------------------------------------+
VDJSignal ReadVDJSignal()
{
   VDJSignal sig;
   sig.is_valid = false;
   sig.is_stale = false;

   // Check if file exists
   if(!FileIsExist(SignalFilePath, FILE_COMMON))
   {
      // Try direct path
      int handle = FileOpen(SignalFilePath, FILE_READ|FILE_TXT|FILE_ANSI|FILE_SHARE_READ);
      if(handle == INVALID_HANDLE)
      {
         // Try common files directory
         string common_path = "vdj_antibody_signal.json";
         handle = FileOpen(common_path, FILE_READ|FILE_TXT|FILE_ANSI|FILE_SHARE_READ|FILE_COMMON);
      }

      if(handle == INVALID_HANDLE)
      {
         if(DebugMode) Print("[VDJ] Cannot open signal file");
         return sig;
      }

      string content = "";
      while(!FileIsEnding(handle))
         content += FileReadString(handle) + "\n";
      FileClose(handle);

      sig = ParseVDJJson(content);
   }

   return sig;
}


//+------------------------------------------------------------------+
//| Parse VDJ JSON content (simplified JSON parser)                    |
//+------------------------------------------------------------------+
VDJSignal ParseVDJJson(string content)
{
   VDJSignal sig;
   sig.is_valid = false;
   sig.is_stale = false;

   if(StringLen(content) < 10)
      return sig;

   // Extract key fields using string search
   sig.direction = (int)ExtractJsonDouble(content, "\"direction\"");
   sig.confidence = ExtractJsonDouble(content, "\"confidence\"");
   sig.gate_pass = ExtractJsonBool(content, "\"gate_pass\"");
   sig.n_active = (int)ExtractJsonDouble(content, "\"n_active\"");
   sig.n_long = (int)ExtractJsonDouble(content, "\"n_long\"");
   sig.n_short = (int)ExtractJsonDouble(content, "\"n_short\"");
   sig.weighted_long = ExtractJsonDouble(content, "\"weighted_long\"");
   sig.weighted_short = ExtractJsonDouble(content, "\"weighted_short\"");
   sig.memory_cells_total = (int)ExtractJsonDouble(content, "\"memory_cells_total\"");
   sig.generation = (int)ExtractJsonDouble(content, "\"generation\"");

   // Extract timestamp string
   string ts = ExtractJsonString(content, "\"timestamp\"");
   if(StringLen(ts) > 0)
   {
      // Parse ISO timestamp: 2026-02-08T14:30:00.000000
      sig.timestamp = ParseISOTimestamp(ts);
   }

   // Check staleness
   datetime now = TimeCurrent();
   if(sig.timestamp > 0 && (now - sig.timestamp) > SignalStaleSeconds)
      sig.is_stale = true;

   // Parse individual antibodies (simplified)
   // Look for antibodies array
   int ab_start = StringFind(content, "\"antibodies\"");
   if(ab_start >= 0)
   {
      int arr_start = StringFind(content, "[", ab_start);
      int arr_end = StringFind(content, "]", arr_start);
      if(arr_start >= 0 && arr_end > arr_start)
      {
         string arr_content = StringSubstr(content, arr_start, arr_end - arr_start + 1);
         ParseAntibodyArray(arr_content, sig.antibodies);
      }
   }

   sig.is_valid = (sig.n_active >= 0 && sig.confidence >= 0);
   return sig;
}


//+------------------------------------------------------------------+
//| Parse antibody array from JSON                                     |
//+------------------------------------------------------------------+
void ParseAntibodyArray(string arr, AntibodySignal &antibodies[])
{
   // Count objects by counting opening braces
   int count = 0;
   for(int i = 0; i < StringLen(arr); i++)
      if(StringGetCharacter(arr, i) == '{')
         count++;

   if(count == 0) return;
   ArrayResize(antibodies, count);

   int pos = 0;
   for(int idx = 0; idx < count; idx++)
   {
      int obj_start = StringFind(arr, "{", pos);
      int obj_end = StringFind(arr, "}", obj_start);
      if(obj_start < 0 || obj_end < 0) break;

      string obj = StringSubstr(arr, obj_start, obj_end - obj_start + 1);

      antibodies[idx].id = ExtractJsonString(obj, "\"id\"");
      antibodies[idx].direction = (int)ExtractJsonDouble(obj, "\"dir\"");
      antibodies[idx].confidence = ExtractJsonDouble(obj, "\"conf\"");
      antibodies[idx].win_rate = ExtractJsonDouble(obj, "\"wr\"");
      antibodies[idx].profit_factor = ExtractJsonDouble(obj, "\"pf\"");
      antibodies[idx].v_type = ExtractJsonString(obj, "\"v\"");
      antibodies[idx].d_type = ExtractJsonString(obj, "\"d\"");
      antibodies[idx].j_type = ExtractJsonString(obj, "\"j\"");

      pos = obj_end + 1;
   }
}


//+------------------------------------------------------------------+
//| JSON helper: extract double value                                  |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
   int pos = StringFind(json, key);
   if(pos < 0) return 0.0;

   int colon = StringFind(json, ":", pos + StringLen(key));
   if(colon < 0) return 0.0;

   // Skip whitespace after colon
   int start = colon + 1;
   while(start < StringLen(json) && StringGetCharacter(json, start) == ' ')
      start++;

   // Read until comma, brace, or bracket
   string num = "";
   for(int i = start; i < StringLen(json); i++)
   {
      ushort ch = StringGetCharacter(json, i);
      if(ch == ',' || ch == '}' || ch == ']' || ch == '\n')
         break;
      num += CharToString((uchar)ch);
   }
   StringTrimLeft(num);
   StringTrimRight(num);

   return StringToDouble(num);
}


//+------------------------------------------------------------------+
//| JSON helper: extract string value                                  |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
   int pos = StringFind(json, key);
   if(pos < 0) return "";

   int colon = StringFind(json, ":", pos + StringLen(key));
   if(colon < 0) return "";

   int quote1 = StringFind(json, "\"", colon + 1);
   if(quote1 < 0) return "";

   int quote2 = StringFind(json, "\"", quote1 + 1);
   if(quote2 < 0) return "";

   return StringSubstr(json, quote1 + 1, quote2 - quote1 - 1);
}


//+------------------------------------------------------------------+
//| JSON helper: extract boolean value                                 |
//+------------------------------------------------------------------+
bool ExtractJsonBool(string json, string key)
{
   int pos = StringFind(json, key);
   if(pos < 0) return false;

   int colon = StringFind(json, ":", pos + StringLen(key));
   if(colon < 0) return false;

   string rest = StringSubstr(json, colon + 1, 10);
   StringTrimLeft(rest);

   return (StringFind(rest, "true") == 0);
}


//+------------------------------------------------------------------+
//| Parse ISO timestamp to datetime                                    |
//+------------------------------------------------------------------+
datetime ParseISOTimestamp(string iso)
{
   // Format: 2026-02-08T14:30:00.000000
   if(StringLen(iso) < 19) return 0;

   string date_part = StringSubstr(iso, 0, 10);
   string time_part = StringSubstr(iso, 11, 8);

   // Replace dashes and colons for MQL parsing
   StringReplace(date_part, "-", ".");
   string combined = date_part + " " + time_part;

   return StringToTime(combined);
}


//+------------------------------------------------------------------+
//| IMMUNE GATES -- VDJ-specific trade filters                         |
//+------------------------------------------------------------------+
bool PassImmuneGates(VDJSignal &sig)
{
   // Gate I1: Python-side gate pass
   if(RequireGatePass && !sig.gate_pass)
   {
      if(DebugMode) Print("[VDJ GATE I1] Python gate_pass=false -> BLOCKED");
      return false;
   }

   // Gate I2: Minimum active antibodies (polyclonal response)
   if(sig.n_active < MinActiveAntibody)
   {
      if(DebugMode) Print("[VDJ GATE I2] n_active=", sig.n_active,
                          " < min=", MinActiveAntibody, " -> BLOCKED");
      return false;
   }

   // Gate I3: Minimum consensus confidence
   if(sig.confidence < MinConsensusConf)
   {
      if(DebugMode) Print("[VDJ GATE I3] confidence=",
                          DoubleToString(sig.confidence, 4),
                          " < min=", DoubleToString(MinConsensusConf, 4),
                          " -> BLOCKED");
      return false;
   }

   // Gate I4: Direction must be non-zero
   if(sig.direction == 0)
   {
      if(DebugMode) Print("[VDJ GATE I4] direction=0 -> BLOCKED");
      return false;
   }

   // Gate I5: Check individual antibody quality
   int quality_count = 0;
   for(int i = 0; i < ArraySize(sig.antibodies); i++)
   {
      if(sig.antibodies[i].win_rate >= MinAntibodyWR
         && sig.antibodies[i].profit_factor >= MinAntibodyPF)
         quality_count++;
   }
   if(quality_count < 1)
   {
      if(DebugMode) Print("[VDJ GATE I5] No quality antibodies "
                          "(WR>=", MinAntibodyWR, " PF>=", MinAntibodyPF,
                          ") -> BLOCKED");
      return false;
   }

   if(DebugMode)
      Print("[VDJ GATES] ALL PASSED | dir=", sig.direction,
            " conf=", DoubleToString(sig.confidence, 4),
            " active=", sig.n_active,
            " quality=", quality_count);

   return true;
}


//+------------------------------------------------------------------+
//| Execute trade based on immune system consensus                     |
//+------------------------------------------------------------------+
void ExecuteImmuneTrade(VDJSignal &sig)
{
   string symbol = _Symbol;
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double lot_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   if(tick_value <= 0 || tick_size <= 0)
   {
      Print("[VDJ] ERROR: Invalid tick_value or tick_size for ", symbol);
      return;
   }

   // Calculate lot size for fixed dollar risk
   // ATR-based SL distance
   double atr = 0;
   int atr_handle = iATR(symbol, PERIOD_CURRENT, 14);
   if(atr_handle != INVALID_HANDLE)
   {
      double atr_buf[];
      if(CopyBuffer(atr_handle, 0, 0, 1, atr_buf) > 0)
         atr = atr_buf[0];
      IndicatorRelease(atr_handle);
   }

   if(atr <= 0)
   {
      Print("[VDJ] ERROR: ATR = 0, cannot calculate SL");
      return;
   }

   double sl_distance = atr;  // 1 ATR stop loss
   double tp_distance = sl_distance * TPMultiplier;

   // Lot calculation: MaxLossDollars / (sl_in_ticks * tick_value)
   double sl_ticks = sl_distance / tick_size;
   double lot = MaxLossDollars / (sl_ticks * tick_value);
   lot = MathMax(min_lot, MathFloor(lot / lot_step) * lot_step);
   lot = MathMin(lot, BaseLot * 4);  // Cap at 4x base

   // Price and SL/TP
   double price, sl, tp;
   ENUM_ORDER_TYPE order_type;

   if(sig.direction > 0)
   {
      price = SymbolInfoDouble(symbol, SYMBOL_ASK);
      sl = price - sl_distance;
      tp = price + tp_distance;
      order_type = ORDER_TYPE_BUY;
   }
   else
   {
      price = SymbolInfoDouble(symbol, SYMBOL_BID);
      sl = price + sl_distance;
      tp = price - tp_distance;
      order_type = ORDER_TYPE_SELL;
   }

   if(DryRun)
   {
      Print("[VDJ DRY RUN] Would ", (sig.direction > 0 ? "BUY" : "SELL"),
            " ", DoubleToString(lot, 2), " ", symbol,
            " @ ", DoubleToString(price, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
            " SL=", DoubleToString(sl, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
            " TP=", DoubleToString(tp, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
            " | Antibodies: ", sig.n_active,
            " | Conf: ", DoubleToString(sig.confidence, 4));
      return;
   }

   // Build comment with antibody info
   string best_ab = "";
   if(ArraySize(sig.antibodies) > 0)
      best_ab = sig.antibodies[0].id;
   string comment = StringFormat("VDJ|ab=%s|gen=%d|n=%d",
                                  best_ab, sig.generation, sig.n_active);

   // Execute trade
   MqlTradeRequest request = {};
   MqlTradeResult  result = {};

   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = symbol;
   request.volume    = lot;
   request.type      = order_type;
   request.price     = price;
   request.sl        = sl;
   request.tp        = tp;
   request.magic     = MagicNumber;
   request.comment   = comment;
   request.deviation = 30;

   if(!OrderSend(request, result))
   {
      Print("[VDJ] OrderSend FAILED: ", result.retcode, " - ", result.comment);
      return;
   }

   if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
   {
      g_total_trades++;

      Print("[VDJ] TRADE EXECUTED: ",
            (sig.direction > 0 ? "BUY" : "SELL"), " ",
            DoubleToString(lot, 2), " ", symbol,
            " @ ", DoubleToString(result.price, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
            " | SL=", DoubleToString(sl, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
            " | TP=", DoubleToString(tp, (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
            " | Antibodies=", sig.n_active,
            " | Confidence=", DoubleToString(sig.confidence, 4));

      // Record trade for feedback
      ImmuneTradeRecord rec;
      rec.ticket = result.order;
      rec.antibody_id = best_ab;
      rec.direction = sig.direction;
      rec.entry_price = result.price;
      rec.entry_time = TimeCurrent();
      rec.sl = sl;
      rec.tp = tp;
      if(ArraySize(sig.antibodies) > 0)
      {
         rec.v_type = sig.antibodies[0].v_type;
         rec.d_type = sig.antibodies[0].d_type;
         rec.j_type = sig.antibodies[0].j_type;
      }

      int sz = ArraySize(g_trades);
      ArrayResize(g_trades, sz + 1);
      g_trades[sz] = rec;
   }
}


//+------------------------------------------------------------------+
//| Manage existing positions (trailing stop, etc.)                    |
//+------------------------------------------------------------------+
void ManageExistingPositions()
{
   if(!UseTrailingStop) return;

   double atr = 0;
   int atr_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
   if(atr_handle != INVALID_HANDLE)
   {
      double atr_buf[];
      if(CopyBuffer(atr_handle, 0, 0, 1, atr_buf) > 0)
         atr = atr_buf[0];
      IndicatorRelease(atr_handle);
   }
   if(atr <= 0) return;

   double activation = atr * TrailActivationATR;
   double trail_dist = atr * TrailDistanceATR;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
      double pos_sl = PositionGetDouble(POSITION_SL);
      double pos_tp = PositionGetDouble(POSITION_TP);
      long pos_type = PositionGetInteger(POSITION_TYPE);
      double current_price;

      if(pos_type == POSITION_TYPE_BUY)
      {
         current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double profit_dist = current_price - pos_open;

         if(profit_dist >= activation)
         {
            double new_sl = current_price - trail_dist;
            if(new_sl > pos_sl)
            {
               ModifySL(ticket, new_sl, pos_tp);
            }
         }
      }
      else if(pos_type == POSITION_TYPE_SELL)
      {
         current_price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double profit_dist = pos_open - current_price;

         if(profit_dist >= activation)
         {
            double new_sl = current_price + trail_dist;
            if(new_sl < pos_sl || pos_sl == 0)
            {
               ModifySL(ticket, new_sl, pos_tp);
            }
         }
      }
   }
}


//+------------------------------------------------------------------+
//| Modify SL of a position                                            |
//+------------------------------------------------------------------+
bool ModifySL(ulong ticket, double new_sl, double tp)
{
   MqlTradeRequest request = {};
   MqlTradeResult  result = {};

   request.action   = TRADE_ACTION_SLTP;
   request.position = ticket;
   request.symbol   = _Symbol;
   request.sl       = NormalizeDouble(new_sl, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
   request.tp       = tp;

   if(!OrderSend(request, result))
   {
      if(DebugMode) Print("[VDJ] Trail SL modify failed: ", result.retcode);
      return false;
   }
   return true;
}


//+------------------------------------------------------------------+
//| Count positions with our magic number                              |
//+------------------------------------------------------------------+
int CountMyPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) == MagicNumber
         && PositionGetString(POSITION_SYMBOL) == _Symbol)
         count++;
   }
   return count;
}


//+------------------------------------------------------------------+
//| Check drawdown limits                                              |
//+------------------------------------------------------------------+
bool CheckDrawdownLimits()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   // Daily drawdown
   if(g_daily_start_balance > 0)
   {
      double daily_dd = (g_daily_start_balance - equity) / g_daily_start_balance * 100;
      if(daily_dd >= DailyDDLimitPct)
      {
         Print("[VDJ] DAILY DD LIMIT: ", DoubleToString(daily_dd, 2),
               "% >= ", DoubleToString(DailyDDLimitPct, 2), "%");
         return false;
      }
   }

   // Max drawdown
   if(balance > 0)
   {
      double max_dd = (balance - equity) / balance * 100;
      if(max_dd >= MaxDDLimitPct)
      {
         Print("[VDJ] MAX DD LIMIT: ", DoubleToString(max_dd, 2),
               "% >= ", DoubleToString(MaxDDLimitPct, 2), "%");
         return false;
      }
   }

   return true;
}


//+------------------------------------------------------------------+
//| OnTick -- for additional responsiveness                            |
//+------------------------------------------------------------------+
void OnTick()
{
   // Trailing stop management on every tick for precision
   if(UseTrailingStop)
      ManageExistingPositions();
}


//+------------------------------------------------------------------+
//| OnTradeTransaction -- track trade outcomes for immune feedback     |
//+------------------------------------------------------------------+
void OnTradeTransaction(
   const MqlTradeTransaction &trans,
   const MqlTradeRequest &request,
   const MqlTradeResult &result)
{
   // Track closed positions for win/loss stats
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      if(trans.deal_type == DEAL_TYPE_BUY || trans.deal_type == DEAL_TYPE_SELL)
      {
         // Check if this is a closing deal
         ulong deal_ticket = trans.deal;
         if(deal_ticket > 0)
         {
            if(HistoryDealSelect(deal_ticket))
            {
               long entry = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);
               long magic = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);

               if(entry == DEAL_ENTRY_OUT && magic == MagicNumber)
               {
                  double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
                  double swap = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
                  double commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
                  double net_pnl = profit + swap + commission;

                  if(net_pnl > 0)
                     g_wins++;
                  else
                     g_losses++;

                  double wr = (g_wins + g_losses > 0) ?
                              (double)g_wins / (g_wins + g_losses) * 100 : 0;

                  Print("[VDJ OUTCOME] ",
                        (net_pnl > 0 ? "WIN" : "LOSS"),
                        " $", DoubleToString(net_pnl, 2),
                        " | Record: W=", g_wins, " L=", g_losses,
                        " WR=", DoubleToString(wr, 1), "%");
               }
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
