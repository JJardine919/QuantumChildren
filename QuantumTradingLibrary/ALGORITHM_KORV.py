"""
ALGORITHM: KoRV -- "Real-Time Retroviral Domestication"
========================================================
New signal onboarding system that manages untested indicators through
staged integration, population-specific adaptation, and methylation control.

Biological basis:
    - KoRV (Koala Retrovirus) is a gammaretrovirus that invaded the koala
      germline approximately 50,000 years ago -- evolutionarily YESTERDAY
    - Unlike ancient endogenous retroviruses (ERVs) that were domesticated
      millions of years ago, KoRV domestication is happening RIGHT NOW
    - Northern Australian koalas carry KoRV in every cell (germline integration).
      Southern koalas are still being infected exogenously (horizontal transmission)
    - We can observe every stage of domestication simultaneously across
      different koala populations:

      Stage A - INFECTION: KoRV enters a new koala via horizontal transmission.
        The virus is fully active, replicating freely. The host immune system
        has not yet recognized it. High viral load, unpredictable outcomes.

      Stage B - IMMUNE RESPONSE: The host piRNA pathway and APOBEC3 enzymes
        begin targeting KoRV sequences. CpG methylation accumulates at KoRV
        promoter regions, progressively silencing viral gene expression.
        Some copies are heavily methylated (silenced), others resist.

      Stage C - TOLERANCE: Defective KoRV copies (solo LTRs, truncated env)
        accumulate. The host stops wasting immune resources on non-functional
        copies. These neutral insertions persist as genomic passengers.

      Stage D - DOMESTICATION: Rare KoRV copies that landed near host genes
        and provide a selective advantage (e.g., immune regulation via the
        env protein's immunomodulatory domain) get FIXED in the population.
        The formerly parasitic sequence becomes beneficial host machinery.

    - KEY INSIGHT: Different koala populations are at different stages for
      the SAME viral sequence. Northern populations (50,000 years of coevolution)
      have more domesticated copies than southern populations (recent infection).
      Selective pressure RATE determines domestication speed.

    - KoRV subtypes (A through J) show differential domestication:
        KoRV-A: Oldest, most integrated, lowest pathogenicity
        KoRV-B: Associated with chlamydia susceptibility, actively suppressed
        KoRV-J: Newest variant, most virulent, least domesticated
      This proves that EACH new variant goes through the full cycle independently.

Trading analogy -- "New Signal Onboarding":
    When a NEW signal type, indicator, or data source is added to the trading
    system, it is like a new retrovirus entering the koala genome:
    - The system does not know if the signal is profitable, neutral, or toxic
    - Blindly trusting it is dangerous (like a new virus replicating freely)
    - Blindly rejecting it loses potential alpha (like an immune overreaction)
    - KoRV algorithm manages staged onboarding with population-specific tracking

    Stage 1: INFECTION (Probationary Entry)
        New signal enters the system unfiltered for a probationary window of
        N trades. ALL outcomes are tracked per population (instrument/timeframe).
        No weight adjustments yet -- pure observation. Like KoRV freely
        replicating before the immune system notices it.

    Stage 2: IMMUNE RESPONSE (Methylation / Silencing)
        After the probationary window, if the signal's posterior loss rate
        exceeds the toxicity threshold, methylation kicks in. The signal's
        influence weight is progressively reduced (methylation = weight decay).
        Heavy methylation = near-zero weight. Like CpG methylation silencing
        KoRV promoters. Speed of silencing scales with loss severity.

    Stage 3: TOLERANCE (Neutral Coexistence)
        If the signal is neither profitable nor toxic (win rate near 50%),
        it enters tolerance. The system stops wasting compute on actively
        processing it but does not delete it. Periodic re-evaluation occurs.
        Like defective KoRV copies persisting as harmless passengers.

    Stage 4: DOMESTICATION (Permanent Integration)
        If the signal proves consistently profitable across sufficient trades
        with strong posterior win rate and profit factor, it graduates to
        permanent toolkit status. Its weight gets boosted. Like KoRV env
        protein being co-opted for immune regulation. Domesticated signals
        are NEVER automatically removed -- only manual intervention or
        catastrophic performance collapse triggers de-domestication.

    Population variation:
        The SAME new signal may be domesticated for XAUUSD M5 (population A)
        while still under immune response for BTCUSD H1 (population B).
        Each {signal_type, instrument, timeframe} tuple is tracked independently.
        This mirrors how northern vs southern koalas have different KoRV
        domestication statuses for the exact same viral sequence.

    Recombination risk:
        New signals can interact with existing domesticated signals in
        unpredictable ways. Adding a "new virus" to a genome with existing
        ERVs can cause recombination events. The system tracks pairwise
        interaction effects between new signals and existing toolkit members.

Implementation: Python (SQLite persistence, config_loader for trading values)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: KORV-1.0

============================================================
PSEUDOCODE
============================================================
"""

# ----------------------------------------------------------
# CONSTANTS (from config_loader where applicable)
# ----------------------------------------------------------

# Onboarding stages
STAGE_INFECTION       = "infection"       # Probationary: observe only
STAGE_IMMUNE_RESPONSE = "immune_response" # Silencing: weight decay active
STAGE_TOLERANCE       = "tolerance"       # Neutral: parked, periodic re-eval
STAGE_DOMESTICATED    = "domesticated"    # Permanent: integrated with boost

# Probationary window
INFECTION_MIN_TRADES            = 15    # Minimum trades before stage transition
INFECTION_MIN_DAYS              = 7     # Minimum calendar days in probation
INFECTION_MAX_TRADES            = 100   # Force evaluation even if ambiguous

# Immune response (methylation/silencing)
METHYLATION_TOXICITY_THRESHOLD  = 0.60  # Posterior loss rate >= 60% triggers silencing
METHYLATION_INITIAL_WEIGHT      = 0.70  # First methylation: 70% weight
METHYLATION_HEAVY_WEIGHT        = 0.30  # Heavy methylation: 30% weight
METHYLATION_SILENCED_WEIGHT     = 0.05  # Near-total silencing: 5% weight
METHYLATION_ESCALATION_LR       = 0.70  # Escalate to heavy at LR >= 70%
METHYLATION_SILENCE_LR          = 0.80  # Full silence at LR >= 80%
IMMUNE_RESPONSE_SPEED_FAST      = 10    # Fast immune: 10 trades to escalate
IMMUNE_RESPONSE_SPEED_SLOW      = 30    # Slow immune: 30 trades to escalate

# Tolerance (neutral zone)
TOLERANCE_WR_LOW                = 0.42  # Below 42% posterior WR = not neutral
TOLERANCE_WR_HIGH               = 0.58  # Above 58% posterior WR = not neutral
TOLERANCE_REEVAL_INTERVAL_DAYS  = 14    # Re-evaluate every 14 days
TOLERANCE_MAX_IDLE_DAYS         = 60    # Remove from tolerance after 60 days idle

# Domestication (permanent integration)
DOMESTICATION_MIN_TRADES        = 30    # Need 30+ trades to domesticate
DOMESTICATION_MIN_POSTERIOR_WR  = 0.62  # Posterior WR >= 62%
DOMESTICATION_MIN_PROFIT_FACTOR = 1.40  # PF >= 1.40
DOMESTICATION_BOOST_BASE        = 1.10  # Base boost for newly domesticated
DOMESTICATION_BOOST_MAX         = 1.35  # Maximum boost cap
DOMESTICATION_DE_TRIGGER_WR     = 0.50  # De-domesticate if WR drops below 50%
DOMESTICATION_DE_TRIGGER_TRADES = 20    # Need 20 post-domestication trades to trigger

# Bayesian prior (Beta distribution)
PRIOR_ALPHA                     = 8     # Prior wins (conservative)
PRIOR_BETA                      = 8     # Prior losses (conservative)

# Population tracking
POPULATION_KEY_FORMAT           = "{signal_type}|{instrument}|{timeframe}"

# Recombination risk (interaction tracking)
INTERACTION_MIN_CO_OCCURRENCES  = 10    # Track interaction after 10 co-fires
INTERACTION_SYNERGY_THRESHOLD   = 0.10  # WR improvement > 10% = synergy
INTERACTION_TOXICITY_THRESHOLD  = -0.10 # WR degradation > 10% = interference

# Endogenization report
ENDOGENIZATION_REPORT_INTERVAL  = 3600  # Seconds between full status reports

"""
============================================================
ALGORITHM KoRV_Domestication
============================================================

DEFINE SignalOnboardRecord AS:
    record_id               : TEXT PRIMARY KEY (MD5 of population_key))
    signal_type             : TEXT (name of the new signal/indicator)
    instrument              : TEXT (e.g. "XAUUSD", "BTCUSD")
    timeframe               : TEXT (e.g. "M5", "H1")
    population_key          : TEXT (signal_type|instrument|timeframe)
    stage                   : TEXT ("infection", "immune_response", "tolerance", "domesticated")
    win_count               : INTEGER
    loss_count              : INTEGER
    total_trades            : INTEGER
    posterior_wr            : REAL (Bayesian posterior win rate)
    posterior_lr            : REAL (Bayesian posterior loss rate = 1 - posterior_wr)
    profit_factor           : REAL (avg_win / avg_loss)
    total_pnl               : REAL (cumulative P/L)
    avg_win                 : REAL
    avg_loss                : REAL
    current_weight          : REAL (1.0 = full, 0.05 = silenced, >1.0 = boosted)
    methylation_level       : REAL (0.0 = unmethylated, 1.0 = fully methylated)
    domestication_boost     : REAL (1.0 = neutral, up to 1.35 = max boost)
    stage_entered_at        : TIMESTAMP
    first_seen              : TIMESTAMP
    last_seen               : TIMESTAMP
    last_reeval             : TIMESTAMP
    immune_response_speed   : REAL (trades per methylation escalation step)
    de_domestication_count  : INTEGER (how many times it was de-domesticated)
    notes                   : TEXT

DEFINE InteractionRecord AS:
    interaction_id          : TEXT PRIMARY KEY
    signal_a                : TEXT (signal type A)
    signal_b                : TEXT (signal type B -- may be existing toolkit signal)
    instrument              : TEXT
    timeframe               : TEXT
    co_occurrence_count     : INTEGER
    co_win_count            : INTEGER
    co_loss_count           : INTEGER
    individual_wr_a         : REAL (signal A's standalone WR)
    individual_wr_b         : REAL (signal B's standalone WR)
    combined_wr             : REAL (WR when both fire together)
    synergy_score           : REAL (combined_wr - max(individual_wr_a, individual_wr_b))
    classification          : TEXT ("synergistic", "neutral", "interfering")
    first_seen              : TIMESTAMP
    last_seen               : TIMESTAMP

STORAGE:
    korv_db                 : SQLite "korv_domestication.db"
    Table: signal_onboard   : SignalOnboardRecord rows
    Table: interactions     : InteractionRecord rows
    Table: stage_transitions: audit log of every stage change
    Table: population_summary: aggregated stats per instrument

------------------------------------------------------------
PHASE 1: INFECTION -- New Signal Registration
------------------------------------------------------------

ON register_new_signal(signal_type, instrument, timeframe):

    population_key = f"{signal_type}|{instrument}|{timeframe}"
    record_id = MD5(population_key)[:16]

    # Check if already registered
    existing = korv_db.SELECT FROM signal_onboard WHERE record_id = record_id
    IF existing IS NOT NULL:
        RETURN existing  # Already onboarding

    # Register as new infection
    korv_db.INSERT INTO signal_onboard(
        record_id           = record_id,
        signal_type         = signal_type,
        instrument          = instrument,
        timeframe           = timeframe,
        population_key      = population_key,
        stage               = STAGE_INFECTION,
        win_count           = 0,
        loss_count          = 0,
        total_trades        = 0,
        posterior_wr        = PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_BETA),  # 0.50
        posterior_lr        = PRIOR_BETA / (PRIOR_ALPHA + PRIOR_BETA),   # 0.50
        profit_factor       = 0.0,
        total_pnl           = 0.0,
        current_weight      = 1.0,   # Full weight during probation
        methylation_level   = 0.0,   # Unmethylated (no silencing)
        domestication_boost = 1.0,   # No boost
        stage_entered_at    = NOW,
        first_seen          = NOW,
        last_seen           = NOW,
        immune_response_speed = IMMUNE_RESPONSE_SPEED_SLOW,
        de_domestication_count = 0,
    )

    LOG_STAGE_TRANSITION(record_id, "none", STAGE_INFECTION, "New signal registered")

------------------------------------------------------------
PHASE 2: OUTCOME RECORDING -- Track Every Trade
------------------------------------------------------------

ON record_outcome(signal_type, instrument, timeframe, won, pnl):

    population_key = f"{signal_type}|{instrument}|{timeframe}"
    record_id = MD5(population_key)[:16]

    row = korv_db.SELECT FROM signal_onboard WHERE record_id = record_id
    IF row IS NULL:
        register_new_signal(signal_type, instrument, timeframe)
        row = korv_db.SELECT FROM signal_onboard WHERE record_id = record_id

    # Update counts
    row.win_count  += 1 IF won ELSE 0
    row.loss_count += 0 IF won ELSE 1
    row.total_trades = row.win_count + row.loss_count
    row.total_pnl += pnl
    row.last_seen = NOW

    # Update running averages
    IF won AND pnl > 0:
        row.avg_win = running_average(row.avg_win, pnl, row.win_count)
    ELIF NOT won AND pnl < 0:
        row.avg_loss = running_average(row.avg_loss, abs(pnl), row.loss_count)

    # Bayesian posterior
    row.posterior_wr = (PRIOR_ALPHA + row.win_count) /
                       (PRIOR_ALPHA + PRIOR_BETA + row.total_trades)
    row.posterior_lr = 1.0 - row.posterior_wr

    # Profit factor
    IF row.avg_loss > 0:
        row.profit_factor = row.avg_win / row.avg_loss
    ELSE:
        row.profit_factor = 99.0 IF row.avg_win > 0 ELSE 0.0

    korv_db.UPDATE signal_onboard SET ... WHERE record_id = record_id

    # Trigger stage evaluation
    evaluate_stage_transition(row)

------------------------------------------------------------
PHASE 3: STAGE EVALUATION ENGINE
------------------------------------------------------------

FUNCTION evaluate_stage_transition(row):

    current_stage = row.stage

    # ========================================
    # FROM INFECTION (probationary)
    # ========================================
    IF current_stage == STAGE_INFECTION:

        # Must meet minimum observation window
        IF row.total_trades < INFECTION_MIN_TRADES:
            RETURN  # Still collecting data

        days_since_entry = (NOW - row.stage_entered_at).days
        IF days_since_entry < INFECTION_MIN_DAYS AND row.total_trades < INFECTION_MAX_TRADES:
            RETURN  # Still in calendar probation

        # Evaluate: Is this signal toxic, neutral, or beneficial?

        # TOXIC: High posterior loss rate -> Immune Response
        IF row.posterior_lr >= METHYLATION_TOXICITY_THRESHOLD:
            transition_to_immune_response(row)
            RETURN

        # BENEFICIAL: High posterior WR + sufficient PF -> Domesticate
        IF (row.posterior_wr >= DOMESTICATION_MIN_POSTERIOR_WR
            AND row.profit_factor >= DOMESTICATION_MIN_PROFIT_FACTOR
            AND row.total_trades >= DOMESTICATION_MIN_TRADES):
            transition_to_domesticated(row)
            RETURN

        # NEUTRAL: Neither toxic nor beneficial -> Tolerance
        IF TOLERANCE_WR_LOW <= row.posterior_wr <= TOLERANCE_WR_HIGH:
            transition_to_tolerance(row)
            RETURN

        # AMBIGUOUS: Between neutral and beneficial, or between neutral and toxic
        # Stay in infection, continue collecting data
        IF row.total_trades >= INFECTION_MAX_TRADES:
            # Forced decision after max trades
            IF row.posterior_wr > 0.50:
                transition_to_tolerance(row)  # Lean neutral-positive
            ELSE:
                transition_to_immune_response(row)  # Lean toxic

    # ========================================
    # FROM IMMUNE RESPONSE (silencing)
    # ========================================
    ELIF current_stage == STAGE_IMMUNE_RESPONSE:

        trades_in_stage = count_trades_since_stage_entered(row)

        # Methylation escalation based on continued poor performance
        IF row.posterior_lr >= METHYLATION_SILENCE_LR:
            # Extreme toxicity: fast-track to full silencing
            row.methylation_level = min(1.0, row.methylation_level + 0.20)
            row.current_weight = max(METHYLATION_SILENCED_WEIGHT,
                                     1.0 - row.methylation_level)

        ELIF row.posterior_lr >= METHYLATION_ESCALATION_LR:
            # Escalating methylation
            escalation_rate = 0.10 if trades_in_stage < IMMUNE_RESPONSE_SPEED_FAST else 0.05
            row.methylation_level = min(1.0, row.methylation_level + escalation_rate)
            row.current_weight = max(METHYLATION_HEAVY_WEIGHT,
                                     1.0 - row.methylation_level)

        ELIF row.posterior_lr >= METHYLATION_TOXICITY_THRESHOLD:
            # Maintaining methylation at current level
            PASS  # No change -- holding position

        ELSE:
            # Performance improving! De-methylation possible
            IF row.posterior_wr >= DOMESTICATION_MIN_POSTERIOR_WR AND trades_in_stage >= 20:
                # Remarkable recovery: promote to domesticated
                transition_to_domesticated(row)
                RETURN
            ELIF row.posterior_lr < METHYLATION_TOXICITY_THRESHOLD:
                # Signal stopped being toxic: move to tolerance
                row.methylation_level = max(0.0, row.methylation_level - 0.10)
                row.current_weight = 1.0 - row.methylation_level
                IF row.methylation_level <= 0.0:
                    transition_to_tolerance(row)
                    RETURN

    # ========================================
    # FROM TOLERANCE (neutral)
    # ========================================
    ELIF current_stage == STAGE_TOLERANCE:

        # Periodic re-evaluation
        days_since_reeval = (NOW - row.last_reeval).days
        IF days_since_reeval < TOLERANCE_REEVAL_INTERVAL_DAYS:
            RETURN  # Not time yet

        row.last_reeval = NOW

        # Check for idleness
        days_idle = (NOW - row.last_seen).days
        IF days_idle > TOLERANCE_MAX_IDLE_DAYS:
            # Signal went dormant -- remove from active consideration
            LOG_STAGE_TRANSITION(row.record_id, STAGE_TOLERANCE, "dormant",
                                 "Idle for {days_idle} days")
            RETURN

        # Re-evaluate performance with recent data
        IF row.posterior_lr >= METHYLATION_TOXICITY_THRESHOLD:
            # Turned toxic: move to immune response
            transition_to_immune_response(row)
            RETURN

        IF (row.posterior_wr >= DOMESTICATION_MIN_POSTERIOR_WR
            AND row.profit_factor >= DOMESTICATION_MIN_PROFIT_FACTOR
            AND row.total_trades >= DOMESTICATION_MIN_TRADES):
            # Became beneficial: domesticate!
            transition_to_domesticated(row)
            RETURN

        # Still neutral: stay in tolerance

    # ========================================
    # FROM DOMESTICATED (permanent)
    # ========================================
    ELIF current_stage == STAGE_DOMESTICATED:

        # De-domestication is rare and requires strong evidence
        trades_since_domesticated = count_trades_since_stage_entered(row)

        IF trades_since_domesticated >= DOMESTICATION_DE_TRIGGER_TRADES:
            # Recalculate recent performance (post-domestication window only)
            recent_wr = compute_recent_wr(row, window=DOMESTICATION_DE_TRIGGER_TRADES)

            IF recent_wr < DOMESTICATION_DE_TRIGGER_WR:
                # Performance collapsed: de-domesticate
                row.de_domestication_count += 1
                transition_to_immune_response(row)
                LOG_STAGE_TRANSITION(row.record_id, STAGE_DOMESTICATED,
                    STAGE_IMMUNE_RESPONSE,
                    f"De-domesticated: recent WR={recent_wr:.1%}, count={row.de_domestication_count}")
                RETURN

        # Update boost based on current performance
        # Sigmoid boost centered at 60% WR, capped at DOMESTICATION_BOOST_MAX
        wr_delta = row.posterior_wr - 0.55
        boost = DOMESTICATION_BOOST_BASE + (DOMESTICATION_BOOST_MAX - DOMESTICATION_BOOST_BASE) * sigmoid(10 * wr_delta)
        row.domestication_boost = min(DOMESTICATION_BOOST_MAX, boost)
        row.current_weight = row.domestication_boost

------------------------------------------------------------
STAGE TRANSITION FUNCTIONS
------------------------------------------------------------

FUNCTION transition_to_immune_response(row):
    previous = row.stage
    row.stage = STAGE_IMMUNE_RESPONSE
    row.stage_entered_at = NOW
    row.methylation_level = 0.30  # Start at 30% methylation
    row.current_weight = METHYLATION_INITIAL_WEIGHT  # 70% weight
    row.domestication_boost = 1.0

    # Faster immune response if signal has been toxic before
    IF row.de_domestication_count > 0:
        row.immune_response_speed = IMMUNE_RESPONSE_SPEED_FAST
        row.methylation_level = 0.50  # Start heavier
        row.current_weight = METHYLATION_HEAVY_WEIGHT

    LOG_STAGE_TRANSITION(row.record_id, previous, STAGE_IMMUNE_RESPONSE,
                         f"Posterior LR={row.posterior_lr:.1%}")
    korv_db.UPDATE signal_onboard SET ... WHERE record_id = row.record_id

FUNCTION transition_to_tolerance(row):
    previous = row.stage
    row.stage = STAGE_TOLERANCE
    row.stage_entered_at = NOW
    row.last_reeval = NOW
    row.methylation_level = 0.0
    row.current_weight = 1.0  # Full weight but no boost
    row.domestication_boost = 1.0

    LOG_STAGE_TRANSITION(row.record_id, previous, STAGE_TOLERANCE,
                         f"Posterior WR={row.posterior_wr:.1%}")
    korv_db.UPDATE signal_onboard SET ... WHERE record_id = row.record_id

FUNCTION transition_to_domesticated(row):
    previous = row.stage
    row.stage = STAGE_DOMESTICATED
    row.stage_entered_at = NOW
    row.methylation_level = 0.0  # Fully unmethylated
    row.domestication_boost = DOMESTICATION_BOOST_BASE
    row.current_weight = DOMESTICATION_BOOST_BASE

    LOG_STAGE_TRANSITION(row.record_id, previous, STAGE_DOMESTICATED,
                         f"Posterior WR={row.posterior_wr:.1%}, PF={row.profit_factor:.2f}")
    korv_db.UPDATE signal_onboard SET ... WHERE record_id = row.record_id

------------------------------------------------------------
PHASE 4: WEIGHT QUERY (called during signal generation)
------------------------------------------------------------

FUNCTION get_signal_weight(signal_type, instrument, timeframe) -> float:

    population_key = f"{signal_type}|{instrument}|{timeframe}"
    record_id = MD5(population_key)[:16]

    row = korv_db.SELECT FROM signal_onboard WHERE record_id = record_id

    IF row IS NULL:
        RETURN 1.0  # Unknown signal -- no modification

    RETURN row.current_weight
    # Returns:
    #   INFECTION:       1.00 (unmodified, under observation)
    #   IMMUNE_RESPONSE: 0.05 to 0.70 (methylation-dependent)
    #   TOLERANCE:       1.00 (neutral, no boost or suppression)
    #   DOMESTICATED:    1.10 to 1.35 (boosted)

------------------------------------------------------------
PHASE 5: RECOMBINATION RISK -- Interaction Tracking
------------------------------------------------------------

ON record_interaction(signal_a, signal_b, instrument, timeframe, won, pnl):

    # Track when two signals fire simultaneously
    key = sorted([signal_a, signal_b]).join("|") + f"|{instrument}|{timeframe}"
    interaction_id = MD5(key)[:16]

    row = korv_db.SELECT FROM interactions WHERE interaction_id = interaction_id

    IF row IS NULL:
        korv_db.INSERT INTO interactions(
            interaction_id = interaction_id,
            signal_a = signal_a,
            signal_b = signal_b,
            instrument = instrument,
            timeframe = timeframe,
            co_occurrence_count = 1,
            co_win_count = 1 IF won ELSE 0,
            co_loss_count = 0 IF won ELSE 1,
            first_seen = NOW,
            last_seen = NOW,
        )
        RETURN

    row.co_occurrence_count += 1
    row.co_win_count += 1 IF won ELSE 0
    row.co_loss_count += 0 IF won ELSE 1
    row.last_seen = NOW

    # Compute interaction effect after sufficient data
    IF row.co_occurrence_count >= INTERACTION_MIN_CO_OCCURRENCES:
        row.combined_wr = row.co_win_count / row.co_occurrence_count

        # Look up individual win rates
        row.individual_wr_a = get_population_wr(signal_a, instrument, timeframe)
        row.individual_wr_b = get_population_wr(signal_b, instrument, timeframe)

        expected_wr = max(row.individual_wr_a, row.individual_wr_b)
        row.synergy_score = row.combined_wr - expected_wr

        IF row.synergy_score > INTERACTION_SYNERGY_THRESHOLD:
            row.classification = "synergistic"
        ELIF row.synergy_score < INTERACTION_TOXICITY_THRESHOLD:
            row.classification = "interfering"
        ELSE:
            row.classification = "neutral"

    korv_db.UPDATE interactions SET ... WHERE interaction_id = interaction_id

------------------------------------------------------------
PHASE 6: ENDOGENIZATION REPORT
------------------------------------------------------------

FUNCTION generate_endogenization_report() -> Dict:

    # Population-level summary: how many signals at each stage?
    counts = korv_db.SELECT
        stage, COUNT(*) as count,
        AVG(posterior_wr) as avg_wr,
        AVG(current_weight) as avg_weight
    FROM signal_onboard
    GROUP BY stage

    # Per-instrument breakdown
    instrument_stats = korv_db.SELECT
        instrument, stage, COUNT(*),
        AVG(posterior_wr), AVG(profit_factor)
    FROM signal_onboard
    GROUP BY instrument, stage

    # Interaction hotspots
    synergies = korv_db.SELECT * FROM interactions
        WHERE classification = "synergistic"
        ORDER BY synergy_score DESC LIMIT 10

    interferences = korv_db.SELECT * FROM interactions
        WHERE classification = "interfering"
        ORDER BY synergy_score ASC LIMIT 10

    # Recent stage transitions
    transitions = korv_db.SELECT * FROM stage_transitions
        ORDER BY timestamp DESC LIMIT 20

    RETURN {
        "timestamp": NOW,
        "stage_counts": counts,
        "instrument_breakdown": instrument_stats,
        "top_synergies": synergies,
        "top_interferences": interferences,
        "recent_transitions": transitions,
        "total_signals_tracked": SUM(counts.count),
        "domestication_rate": counts["domesticated"].count / total IF total > 0 ELSE 0,
    }

------------------------------------------------------------
THE LOOP (steady-state behavior)
------------------------------------------------------------

LOOP on every trade signal evaluation:

    +-----------------------------------------------------+
    |  New signal source registered?                       |
    |    -> register_new_signal() -> enters INFECTION      |
    +-----------------------------------------------------+
                           |
                           v
    +-----------------------------------------------------+
    |  Trade executed using this signal                    |
    |    -> record_outcome(signal, instrument, tf, won, pnl)|
    |    -> updates counts, posterior WR/LR, PF             |
    |    -> evaluate_stage_transition()                     |
    +-------------------+---------------------------------+
                        |
           +------------+------------+
           |            |            |
           v            v            v
    +-----------+ +-----------+ +-----------+
    | INFECTION | | TOLERANCE | | DOMEST.   |
    | (observe) | | (neutral) | | (boosted) |
    +-----------+ +-----------+ +-----------+
           |            |            |
           v            |            v
    +----------------+  |    +------------------+
    | IMMUNE RESP.   |  |    | De-domestication |
    | (methylation)  |  |    | (if WR collapses)|
    | weight -> 0.05 |  |    +--------+---------+
    +-------+--------+  |             |
            |           |             v
            v           |    +------------------+
    +---------------+   +--->| IMMUNE RESPONSE  |
    | Recovery?     |        +------------------+
    | WR improves   |
    +-------+-------+
            |
            v
    +---------------+
    | -> TOLERANCE   |
    | or DOMESTICATE |
    +---------------+

    CONCURRENTLY:

    +-----------------------------------------------------+
    |  When multiple signals fire together:                |
    |    -> record_interaction(sig_a, sig_b, ...)          |
    |    -> track synergy or interference                  |
    +-----------------------------------------------------+

    PERIODICALLY (every ENDOGENIZATION_REPORT_INTERVAL seconds):

    +-----------------------------------------------------+
    |  generate_endogenization_report()                    |
    |    -> stage counts, instrument breakdown             |
    |    -> interaction hotspots                            |
    |    -> recent transitions                             |
    +-----------------------------------------------------+

INVARIANT:
    "Every new signal must earn its place in the genome.
     The system trusts NOTHING by default and domesticates
     ONLY what proves itself -- per population, per stage,
     with full methylation control at every step."

BIOLOGICAL PARALLEL:
    signal_type             = retroviral sequence (KoRV-A, KoRV-B, etc.)
    population_key          = koala population (northern, southern, specific colony)
    stage                   = integration status (exogenous -> endogenous)
    methylation_level       = CpG methylation at KoRV LTR promoter
    current_weight          = viral gene expression level
    domestication_boost     = co-opted function (env immunomodulation)
    posterior_wr            = fitness effect on host
    interaction_record      = recombination between KoRV and existing ERVs
    de_domestication_count  = reversion to pathogenic state
    immune_response_speed   = piRNA + APOBEC3 defense kinetics
    tolerance               = defective KoRV copies (solo LTRs, truncated)
    endogenization_report   = genomic census of ERV integration states

CONVERGENCE:
    After onboarding N new signal types across M instruments:
    - Toxic signals are silenced within 15-30 trades (methylation)
    - Neutral signals park in tolerance, re-evaluated periodically
    - Profitable signals graduate to permanent toolkit within 30-100 trades
    - Population-specific behavior emerges: same signal, different weights
      per instrument -- exactly like KoRV in northern vs southern koalas
    - Interaction tracking identifies synergistic signal pairs (recombination
      that benefits the host) and toxic combinations (pathogenic recombination)

DATABASES:
    korv_domestication.db   <- signal onboarding status, interactions, transitions
    Uses CONFIDENCE_THRESHOLD from config_loader for gate integration

FILES:
    korv.py                 -> KoRVDomesticationEngine class
    ALGORITHM_KORV.py       -> this pseudocode specification
    BRAIN_*.py              -> calls get_signal_weight() per cycle
    teqa_v3_neural_te.py    -> existing TE system (integration point)
"""
