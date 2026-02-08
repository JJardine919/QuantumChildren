"""
ALGORITHM: TE_Domestication
============================
Closed-loop learning for transposable element trading patterns.

Biological basis:
    - RAG1/RAG2 recombinase: Transib DNA transposon domesticated by vertebrate
      immune system to create adaptive immunity (V(D)J recombination)
    - Arc protein: Ty3/gypsy retrotransposon domesticated for synaptic RNA
      transfer between neurons (memory formation)
    - CRISPR: Casposon-derived system domesticated by prokaryotes for defense
    - SETMAR: Hsmar1 mariner transposon fused with SET domain for DNA repair

Trading analogy:
    - TE activation patterns that consistently precede profitable trades get
      "domesticated" -- they become permanent signal enhancers
    - Patterns that precede losses remain wild (neutral) or get suppressed
    - This is STDP (spike-timing-dependent plasticity) for transposons:
      TE fires before profit = strengthen, TE fires before loss = weaken

Implementation: Python (SQLite persistence, MT5 deal history polling)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: 1.0

============================================================
PSEUDOCODE
============================================================
"""

# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────

DOMESTICATION_MIN_TRADES   = 20     # Minimum observations before domestication
DOMESTICATION_MIN_WR       = 0.70   # Promote to domesticated at 70%+ WR
DOMESTICATION_DE_MIN_WR    = 0.60   # De-domesticate only below 60% WR (hysteresis)
DOMESTICATION_EXPIRY_DAYS  = 30     # Re-evaluate after 30 days of inactivity
SIGNAL_MATCH_WINDOW_SEC    = 120    # 2 min window to match signal → trade
POLL_LOOKBACK_SEC          = 300    # 5 min lookback for closed deals
# Boost formula: sigmoid centered at 65% WR, max boost ~1.30
# boost = 1.0 + 0.30 * sigmoid(-15 * (WR - 0.65))

"""
============================================================
ALGORITHM TE_Domestication
============================================================

DEFINE TE_Pattern AS:
    active_tes[]      : list of TE family names active at signal time
    combo             : sorted(active_tes) joined by "+"
    pattern_hash      : MD5(combo)[:16]

DEFINE DomesticationRecord AS:
    pattern_hash      : TEXT PRIMARY KEY
    te_combo          : TEXT
    win_count         : INTEGER
    loss_count        : INTEGER
    win_rate          : REAL
    domesticated      : BOOLEAN  (0 or 1)
    boost_factor      : REAL     (1.0 = neutral, >1.0 = boosted, <1.0 = suppressed)
    first_seen        : TIMESTAMP
    last_seen         : TIMESTAMP

STORAGE:
    domestication_db  : SQLite "teqa_domestication.db"
    signal_history_db : SQLite "teqa_signal_history.db"
    processed_tickets : in-memory SET (dedup closed deals)

────────────────────────────────────────────────────────────
PHASE 1: SIGNAL EMISSION (runs every TEQA cycle, ~60s)
────────────────────────────────────────────────────────────

ON teqa_cycle(bars, symbol):

    # Step 1-6: TE activation → quantum circuit → neural consensus
    activations = compute_te_activations(bars)
    shock       = detect_genomic_shock(bars)
    activations = adjust_for_shock(activations, shock)
    consensus   = run_neural_mosaic_quantum(activations)

    # Step 7: DOMESTICATION READ (the feedback point)
    active_tes = [a.te FOR a IN activations WHERE a.strength > 0.5]
    boost = domestication_db.GET_BOOST(active_tes)
    #   GET_BOOST:
    #     hash = MD5(sorted(active_tes).join("+"))[:16]
    #     row = SELECT boost_factor, domesticated
    #           FROM domesticated_patterns WHERE pattern_hash = hash
    #     IF row AND row.domesticated: RETURN row.boost_factor
    #     ELSE: RETURN 1.0

    # Step 8: Apply boost to confidence
    confidence = concordance * 0.4
                + consensus  * 0.4
                + min(0.2, boost - 1.0)

    # Step 9: Emit signal JSON with active_tes list
    signal = {
        direction, confidence, active_tes, gates, shock, consensus, boost, ...
    }
    WRITE signal → te_quantum_signal.json
    INSERT signal → signal_history_db   # For later matching

────────────────────────────────────────────────────────────
PHASE 2: TRADE EXECUTION (runs in BRAIN loop, same cycle)
────────────────────────────────────────────────────────────

ON brain_cycle(symbol):

    signal = READ te_quantum_signal.json via TEQABridge
    IF signal.is_blocked: HOLD
    IF signal.confidence < CONFIDENCE_THRESHOLD: HOLD

    # Combine with LSTM prediction
    action, conf = apply_to_lstm(lstm_action, lstm_conf, signal)

    IF action IN (BUY, SELL) AND regime == CLEAN:
        mt5.order_send(
            symbol   = symbol,
            type     = action,
            magic    = MAGIC_NUMBER,        # ← critical for matching
            sl       = price -/+ sl_dist,   # Fixed $1.00 max loss
            tp       = price +/- tp_dist,
        )
        # Trade is now OPEN in MT5 with our magic number

────────────────────────────────────────────────────────────
PHASE 3: OUTCOME HARVESTING (runs at end of each BRAIN cycle)
────────────────────────────────────────────────────────────

ON feedback_poll():

    # Pull recently closed deals from MT5
    deals = mt5.history_deals_get(
        from = now - POLL_LOOKBACK_SEC,
        to   = now
    )

    FOR deal IN deals:

        # Filter: only OUR closing deals
        IF deal.entry != DEAL_ENTRY_OUT: SKIP
        IF deal.magic NOT IN magic_numbers: SKIP
        IF deal.ticket IN processed_tickets: SKIP

        # Determine original trade direction
        # (closing BUY = was SHORT, closing SELL = was LONG)
        original_direction = -1 IF deal.close_type == BUY ELSE 1

        # Match to TEQA signal that triggered this trade
        matched_signal = signal_history_db.QUERY(
            SELECT active_tes
            FROM signals
            WHERE symbol    = deal.symbol
              AND direction = original_direction
              AND gates_pass = 1
              AND timestamp BETWEEN (deal.time - MATCH_WINDOW)
                                AND deal.time
            ORDER BY timestamp DESC
            LIMIT 1
        )

        IF matched_signal IS NULL: SKIP  # Manual trade or expired signal

        # Extract the TE combo that was active when we entered
        active_tes = JSON_PARSE(matched_signal.active_tes)

        # Determine outcome
        won = (deal.profit > 0)

        # Feed to domestication tracker
        RECORD_PATTERN(active_tes, won)

        processed_tickets.ADD(deal.ticket)

────────────────────────────────────────────────────────────
PHASE 4: DOMESTICATION UPDATE (called by RECORD_PATTERN)
────────────────────────────────────────────────────────────

FUNCTION RECORD_PATTERN(active_tes[], won):

    combo = sorted(active_tes).join("+")
    hash  = MD5(combo)[:16]

    row = domestication_db.SELECT WHERE pattern_hash = hash

    IF row EXISTS:
        row.win_count  += 1 IF won ELSE 0
        row.loss_count += 0 IF won ELSE 1
        total = row.win_count + row.loss_count
        row.win_rate = row.win_count / total

        # DOMESTICATION CHECK with HYSTERESIS
        # Two thresholds prevent oscillation at the boundary:
        #   ON  threshold: WR >= 0.70 (promote to domesticated)
        #   OFF threshold: WR <  0.60 (revoke domestication)
        #   Between 0.60 and 0.70: maintain current state
        was_domesticated = row.domesticated

        IF was_domesticated:
            # Already domesticated: only revoke below de-domestication threshold
            IF row.win_rate < DOMESTICATION_DE_MIN_WR:
                row.domesticated = FALSE
            # Between 0.60 and 0.70: stays domesticated (hysteresis band)
        ELSE:
            # Not yet domesticated: must meet full criteria to promote
            IF total >= DOMESTICATION_MIN_TRADES AND row.win_rate >= DOMESTICATION_MIN_WR:
                row.domesticated = TRUE

        # Sigmoid boost: near-zero below 55% WR, steep ramp 60-80%, saturates ~1.30
        IF row.domesticated:
            row.boost_factor = 1.0 + 0.30 * sigmoid(-15 * (row.win_rate - 0.65))
            #
            # Examples at different win rates:
            #   55% WR → boost ≈ 1.02  (near-zero boost)
            #   60% WR → boost ≈ 1.07  (starting to ramp)
            #   65% WR → boost ≈ 1.15  (midpoint of sigmoid)
            #   70% WR → boost ≈ 1.23  (steep ramp zone)
            #   80% WR → boost ≈ 1.29  (approaching saturation)
            #   90% WR → boost ≈ 1.30  (saturated)
            #
        ELSE:
            row.boost_factor = 1.0

        row.last_seen = NOW
        domestication_db.UPDATE row

    ELSE:  # First time seeing this TE combination
        domestication_db.INSERT(
            pattern_hash = hash,
            te_combo     = combo,
            win_count    = 1 IF won ELSE 0,
            loss_count   = 0 IF won ELSE 1,
            win_rate     = 1.0 IF won ELSE 0.0,
            domesticated = FALSE,
            boost_factor = 1.0,
            first_seen   = NOW,
            last_seen    = NOW,
        )

────────────────────────────────────────────────────────────
THE LOOP (steady-state behavior)
────────────────────────────────────────────────────────────

LOOP every ~60 seconds:

    ┌─────────────────────────────────────────────────────┐
    │  TEQA v3.0 analyze()                                │
    │    TE activations → quantum circuit → neural vote   │
    │    → get_boost(active_tes)  ←─────────────────┐     │
    │    → emit signal JSON + log to history DB      │     │
    └─────────────┬──────────────────────────────────┘     │
                  ↓                                        │
    ┌─────────────────────────────────────────────────┐   │
    │  BRAIN run_cycle()                              │   │
    │    read signal → combine with LSTM              │   │
    │    → execute trade (mt5.order_send, magic #)    │   │
    │    → feedback_poll() ──┐                        │   │
    └────────────────────────┼────────────────────────┘   │
                             ↓                             │
    ┌─────────────────────────────────────────────────┐   │
    │  TradeOutcomePoller                             │   │
    │    mt5.history_deals_get() → filter by magic    │   │
    │    → match deal to signal (symbol+dir+time)     │   │
    │    → extract active_tes from matched signal     │   │
    │    → record_pattern(active_tes, won=profit>0)   │   │
    └─────────────┬──────────────────────────────────┘   │
                  ↓                                        │
    ┌─────────────────────────────────────────────────┐   │
    │  TEDomesticationTracker (with hysteresis)        │   │
    │    update win/loss counts in SQLite             │   │
    │    ON at WR>=0.70, OFF at WR<0.60, hold between │   │
    │    boost = 1.0 + 0.30 * sigmoid(-15*(WR-0.65)) │   │
    │    → stored in teqa_domestication.db ───────────┘   │
    └─────────────────────────────────────────────────────┘

INVARIANT:
    "TE patterns that precede profit are amplified.
     TE patterns that precede loss remain wild.
     The system learns which transposon combinations make money."

BIOLOGICAL PARALLEL:
    active_tes[]           = transposon activation fingerprint
    pattern_hash           = genomic locus identifier
    win_count / loss_count = selection pressure (fitness)
    domesticated           = exaptation (TE co-opted for host benefit)
    boost_factor           = expression level after domestication
    RECORD_PATTERN()       = STDP (spike-timing-dependent plasticity)
    GET_BOOST()            = domesticated gene expression check

CONVERGENCE:
    After ~20 trades per TE combo, the system identifies which
    combinations of L1_Neuronal + Alu_Exonization + HERV_Synapse + ...
    actually predict profitable market moves. These get a permanent
    confidence boost. Losing combos stay at 1.0x. The domestication
    DB becomes a learned filter that improves with every trade.

DATABASES:
    teqa_domestication.db    ← pattern win/loss/boost (persistent learning)
    teqa_signal_history.db   ← every signal emitted (for deal matching)
    MT5 deal history         ← trade outcomes (profit/loss per ticket)

FILES:
    teqa_v3_neural_te.py     → TEDomesticationTracker class
    teqa_feedback.py         → TradeOutcomePoller class
    teqa_signal_history.py   → SignalHistoryDB class
    teqa_live.py             → emits signals + logs to history
    teqa_bridge.py           → reads signals for BRAIN scripts
    BRAIN_ATLAS.py           → trades + polls feedback each cycle
"""
