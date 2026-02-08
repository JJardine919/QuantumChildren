"""
ALGORITHM: Toxoplasma -- "Regime Behavior Modification"
========================================================
Detects when the market is hijacking your strategies through regime shifts,
scores the degree of behavioral infection, and activates countermeasures.

Biological basis:
    - Toxoplasma gondii is an obligate intracellular parasite that infects
      approximately 30% of all humans and most warm-blooded animals
    - Its lifecycle REQUIRES passage through a feline definitive host:
      oocysts shed in cat feces -> intermediate host ingests -> tissue cysts
      form in brain/muscle -> cat eats intermediate host -> sexual reproduction
      in cat gut -> back to oocysts
    - The problem: intermediate hosts (rodents, humans) must be EATEN by cats
      for the lifecycle to complete. Natural selection therefore favored
      Toxoplasma strains that could MODIFY HOST BEHAVIOR to increase predation

    Mechanisms of behavioral modification:
    1. DOPAMINE MANIPULATION: Toxoplasma encodes tyrosine hydroxylase (TH),
       the rate-limiting enzyme in dopamine synthesis. Infected tissue cysts
       produce excess dopamine directly in the brain, particularly in the
       amygdala and nucleus accumbens. This increases:
         - Risk-taking behavior (rodents lose fear of cat odor)
         - Impulsivity (faster reaction to stimuli but poorer accuracy)
         - Novelty-seeking (exploration of dangerous areas)

    2. EPIGENETIC HIJACKING: Toxoplasma secretes effector proteins (ROP16,
       GRA15, GRA16) into the host cell nucleus that:
         - Modify histone acetylation at immune gene promoters
         - Alter DNA methylation patterns
         - Activate host transposable elements (TEs) that are normally silenced
         - Specifically: L1 and SINE elements show increased expression in
           infected neurons, disrupting normal gene regulation

    3. TESTOSTERONE MODULATION: Infected males show elevated testosterone,
       further increasing risk-taking and aggression -- the parasite
       hijacks the host's hormonal system to serve its own ends

    4. CHRONIC vs ACUTE: Acute infection (tachyzoites) causes active
       inflammation and immune response. Chronic infection (bradyzoites in
       tissue cysts) is the "stealth mode" -- the parasite evades immunity
       and persistently modifies behavior for the lifetime of the host.
       The two phases require DIFFERENT host responses.

    KEY INSIGHT: An external agent (parasite/market) can SYSTEMATICALLY
    HIJACK the host's (trader's) decision-making system by manipulating
    the neurotransmitter environment (volatility/volume/regime signals)
    that drives behavioral choices.

Trading analogy -- "Market Regime Hijacking":
    The MARKET is the parasite. It modifies YOUR SYSTEM'S behavior through
    regime changes, creating an environment where your strategies act against
    their own nature -- just as Toxoplasma makes rodents run toward cats.

    Infection vectors:
    - TRENDING REGIME infects mean-reversion strategies:
        Mean-reversion buys dips in a downtrend = "running toward the cat"
        The market's "dopamine" (volatility spikes) makes the system chase
        falling knives with increased conviction
    - RANGING REGIME infects momentum strategies:
        Momentum buys breakouts in a range = "running toward the cat"
        False breakouts are the market's trap -- whipsaws that drain capital
    - VOLATILE REGIME infects ALL strategies:
        Like acute Toxoplasma infection -- the pathogen is overwhelming.
        Excessive volatility makes the system overtrade (dopamine proxy)

    Defense mechanisms:
    - BEHAVIORAL BASELINE: Like a healthy immune system knowing "self" from
      "non-self", each strategy has a baseline behavioral profile
    - INFECTION SCORING: Deviation from baseline = degree of parasitic control
    - ANTI-PARASITIC RESPONSE: When infection exceeds threshold:
        * Reduce position sizing (limit pathogen replication)
        * Invert contrarian signals (if market wants you to buy, consider sell)
        * Increase confidence threshold (require stronger evidence)
        * Shorten holding periods (reduce exposure to manipulation)
    - CHRONIC vs ACUTE: Short-term regime anomalies (acute) need temporary
      defense. Long-term regime changes (chronic) need strategic adaptation.

    TE Epigenetic Integration:
    - During "infected" periods, track which TE patterns get activated
    - These are the market's manipulation tools -- the same way Toxoplasma
      activates host L1/SINE elements to disrupt gene regulation
    - Build a map: which TE activation patterns are associated with regime
      hijacking? This feeds back into the domestication system.

Implementation: Python (SQLite persistence, config_loader)
Integration: TEQA v3.0 pipeline, VDJ Recombination, TE Domestication

Authors: DooDoo + Claude
Date:    2026-02-08
Version: TOXOPLASMA-1.0

============================================================
PSEUDOCODE
============================================================
"""

# ----------------------------------------------------------------
# CONSTANTS (defaults -- never hardcode trading values)
# ----------------------------------------------------------------

# Behavioral baseline calibration
BASELINE_MIN_TRADES        = 30     # Min trades to establish baseline profile
BASELINE_WINDOW_TRADES     = 100    # Rolling window of recent trades for baseline
BASELINE_RECALIBRATE_HOURS = 24     # Re-evaluate baseline every 24 hours

# Infection detection
INFECTION_LOOKBACK_TRADES  = 20     # Recent trades to measure current behavior
INFECTION_MILD_THRESHOLD   = 0.35   # Below this = healthy
INFECTION_MODERATE_THRESHOLD = 0.55 # Moderate infection
INFECTION_SEVERE_THRESHOLD = 0.75   # Severe infection -- full countermeasures
INFECTION_CRITICAL         = 0.90   # Critical -- shut down strategy entirely

# Dopamine proxy (volatility/volume excitement index)
DOPAMINE_VOL_LOOKBACK      = 20     # Bars for volatility baseline
DOPAMINE_SPIKE_MULT        = 1.8    # ATR spike > 1.8x baseline = dopamine surge
DOPAMINE_VOLUME_SPIKE_MULT = 2.0    # Volume > 2x average = dopamine surge
DOPAMINE_DECAY_RATE        = 0.85   # Dopamine level decays by 15% per cycle

# Anti-parasitic countermeasures
COUNTERMEASURE_SIZE_REDUCTION    = 0.50  # Reduce position to 50% under infection
COUNTERMEASURE_CONFIDENCE_BOOST  = 0.15  # Add 0.15 to confidence threshold
COUNTERMEASURE_HOLD_REDUCTION    = 0.60  # Shorten max hold time by 40%
COUNTERMEASURE_INVERSION_THRESH  = 0.80  # Invert signals above this infection score

# Chronic vs Acute classification
ACUTE_WINDOW_HOURS   = 4     # Infection lasting < 4 hours = acute
CHRONIC_WINDOW_HOURS = 48    # Infection lasting > 48 hours = chronic
CHRONIC_ADAPTATION_RATE = 0.02  # Slow baseline shift rate for chronic infection

# TE Epigenetic tracking
TE_INFECTION_WINDOW_SEC = 300   # 5 min window to associate TE patterns with infection
TE_MANIPULATION_MIN_COUNT = 5   # Min observations to flag a TE pattern as manipulation

"""
============================================================
ALGORITHM Toxoplasma
============================================================

DEFINE StrategyType AS ENUM:
    MOMENTUM         # Trend-following, breakout strategies
    MEAN_REVERSION   # Counter-trend, revert-to-mean strategies
    VOLATILITY       # Volatility-based strategies (straddle, gamma)
    HYBRID           # Mixed strategy types

DEFINE InfectionPhase AS ENUM:
    HEALTHY          # No regime hijacking detected
    ACUTE            # Short-term regime anomaly (< 4 hours)
    CHRONIC          # Long-term regime shift (> 48 hours)
    CRITICAL         # Emergency -- strategy completely compromised

DEFINE RegimeType AS ENUM:
    TRENDING_UP      # Strong upward trend
    TRENDING_DOWN    # Strong downward trend
    RANGING          # Sideways, mean-reverting market
    VOLATILE         # High volatility, no clear direction
    COMPRESSED       # Low volatility, pre-breakout squeeze
    TRANSITIONING    # Regime change in progress (most dangerous)

DEFINE BehavioralBaseline AS:
    strategy_id       : TEXT            # Unique strategy identifier
    strategy_type     : StrategyType    # MOMENTUM, MEAN_REVERSION, etc.
    symbol            : TEXT            # Trading instrument
    # Historical behavior profile
    baseline_win_rate : REAL            # Normal win rate (e.g., 0.58)
    baseline_avg_hold : REAL            # Normal average holding time (bars)
    baseline_trade_freq : REAL          # Normal trades per day
    baseline_avg_pnl  : REAL            # Normal average P/L per trade
    baseline_sharpe   : REAL            # Normal Sharpe ratio
    # Regime-specific baselines
    wr_trending_up    : REAL            # Win rate in uptrend
    wr_trending_down  : REAL            # Win rate in downtrend
    wr_ranging        : REAL            # Win rate in range
    wr_volatile       : REAL            # Win rate in volatile
    # Calibration metadata
    total_trades      : INTEGER         # Total trades used for baseline
    last_calibrated   : TIMESTAMP       # When baseline was last updated
    calibration_valid : BOOLEAN         # True if enough trades exist

DEFINE InfectionScore AS:
    strategy_id       : TEXT
    symbol            : TEXT
    timestamp         : TIMESTAMP
    # Component scores (each 0.0 to 1.0)
    win_rate_deviation : REAL     # How far current WR is from baseline
    hold_time_deviation : REAL    # Abnormal holding periods
    frequency_deviation : REAL    # Abnormal trade frequency
    pnl_deviation      : REAL    # Abnormal P/L distribution
    regime_mismatch    : REAL     # Strategy type vs current regime conflict
    dopamine_level     : REAL     # Current volatility/volume excitement
    # Composite score
    infection_score    : REAL     # Weighted composite (0 = healthy, 1 = fully infected)
    infection_phase    : InfectionPhase
    # Infection duration tracking
    infection_start    : TIMESTAMP OR NULL
    infection_duration_hours : REAL

DEFINE DopamineState AS:
    current_level      : REAL     # 0.0 to 1.0 (0 = calm, 1 = manic)
    atr_ratio          : REAL     # Current ATR / baseline ATR
    volume_ratio       : REAL     # Current volume / baseline volume
    spike_count_1h     : INTEGER  # Number of vol/volume spikes in last hour
    decay_factor       : REAL     # How fast dopamine returns to normal

DEFINE CountermeasureSet AS:
    position_size_mult  : REAL    # 0.0 to 1.0 (1.0 = full size, 0.5 = half)
    confidence_offset   : REAL    # Added to confidence threshold
    hold_time_mult      : REAL    # 0.0 to 1.0 (1.0 = normal, 0.6 = shortened)
    signal_inversion    : BOOLEAN # True = flip signals (buy -> sell, sell -> buy)
    strategy_pause      : BOOLEAN # True = halt strategy entirely
    # Which countermeasures are active
    active_measures     : LIST[TEXT]

DEFINE TEManipulationRecord AS:
    te_combo_hash      : TEXT PRIMARY KEY
    te_combo           : TEXT     # Sorted TE names joined by "+"
    # Infection context
    seen_during_infection : INTEGER  # Times this TE combo appeared during infection
    seen_during_healthy   : INTEGER  # Times during healthy periods
    infection_correlation : REAL     # Ratio: infection / (infection + healthy)
    # Outcome during infection
    infection_wins     : INTEGER
    infection_losses   : INTEGER
    infection_wr       : REAL
    # Classification
    is_manipulation_tool : BOOLEAN   # True if strongly correlated with infection
    first_seen         : TIMESTAMP
    last_seen          : TIMESTAMP

STORAGE:
    toxoplasma_db     : SQLite "toxoplasma_infection.db"
        TABLE strategy_baselines     -- behavioral baseline per strategy/symbol
        TABLE infection_log          -- time-series of infection scores
        TABLE dopamine_log           -- dopamine state over time
        TABLE countermeasure_log     -- what countermeasures were activated and when
        TABLE te_manipulation        -- TE patterns associated with infection
    config_loader     : imports CONFIDENCE_THRESHOLD, MAX_LOSS_DOLLARS (never hardcoded)

------------------------------------------------------------
PHASE 1: BEHAVIORAL BASELINE CALIBRATION
------------------------------------------------------------

ON calibrate_baseline(strategy_id, strategy_type, symbol, trade_history[]):
    #
    # Establish the "healthy self" -- what does this strategy look like
    # when it is NOT being manipulated by the market?
    #
    # This is analogous to the immune system learning "self" antigens
    # during thymic selection. Without a baseline, you cannot detect infection.
    #

    IF len(trade_history) < BASELINE_MIN_TRADES:
        RETURN BehavioralBaseline(calibration_valid=FALSE)

    # Use the most recent BASELINE_WINDOW_TRADES for calibration
    recent = trade_history[-BASELINE_WINDOW_TRADES:]

    baseline = BehavioralBaseline()
    baseline.strategy_id = strategy_id
    baseline.strategy_type = strategy_type
    baseline.symbol = symbol
    baseline.total_trades = len(recent)
    baseline.last_calibrated = NOW

    # Overall statistics
    wins = [t FOR t IN recent WHERE t.profit > 0]
    baseline.baseline_win_rate = len(wins) / len(recent)
    baseline.baseline_avg_hold = MEAN(t.hold_time FOR t IN recent)
    baseline.baseline_trade_freq = len(recent) / time_span_days(recent)
    baseline.baseline_avg_pnl = MEAN(t.profit FOR t IN recent)

    # Regime-specific win rates
    FOR regime IN [TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE]:
        regime_trades = [t FOR t IN recent WHERE t.regime == regime]
        IF len(regime_trades) >= 5:
            baseline["wr_" + regime] = len([t FOR t IN regime_trades WHERE t.profit > 0]) / len(regime_trades)
        ELSE:
            baseline["wr_" + regime] = baseline.baseline_win_rate  # Use overall as fallback

    baseline.calibration_valid = TRUE
    STORE baseline -> toxoplasma_db.strategy_baselines

    RETURN baseline

------------------------------------------------------------
PHASE 2: DOPAMINE STATE COMPUTATION
------------------------------------------------------------

ON compute_dopamine(bars, symbol):
    #
    # The "dopamine proxy" -- volatility and volume spikes act as the
    # neurotransmitter that makes the system want to trade more aggressively.
    #
    # Toxoplasma increases dopamine in the amygdala -> increased risk-taking.
    # High volatility + high volume = the market's dopamine injection.
    #
    # When dopamine is high, strategies deviate from baseline behavior:
    #   - Trade more frequently (impulsivity)
    #   - Take larger positions (risk-seeking)
    #   - Hold longer than they should (novelty-seeking)
    #

    # ATR ratio: current vs baseline
    atr_current = ATR(bars[-5:], period=5)
    atr_baseline = ATR(bars[-DOPAMINE_VOL_LOOKBACK:], period=DOPAMINE_VOL_LOOKBACK)
    atr_ratio = atr_current / (atr_baseline + 1e-10)

    # Volume ratio: current vs baseline
    vol_current = MEAN(bars[-5:].volume)
    vol_baseline = MEAN(bars[-DOPAMINE_VOL_LOOKBACK:].volume)
    volume_ratio = vol_current / (vol_baseline + 1e-10)

    # Spike counting: how many bars in last hour had extreme moves
    spike_count = COUNT(bar IN bars[-60:]
                        WHERE bar.atr > atr_baseline * DOPAMINE_SPIKE_MULT
                        OR bar.volume > vol_baseline * DOPAMINE_VOLUME_SPIKE_MULT)

    # Composite dopamine level (0 to 1)
    # Sigmoid mapping: low ratios -> near 0, high ratios -> near 1
    atr_component = sigmoid(3.0 * (atr_ratio - 1.5))       # Fires above 1.5x ATR
    vol_component = sigmoid(2.0 * (volume_ratio - 2.0))     # Fires above 2x volume
    spike_component = MIN(1.0, spike_count / 10.0)          # 10+ spikes = max

    dopamine_level = (
        atr_component * 0.40
        + vol_component * 0.35
        + spike_component * 0.25
    )

    # Apply decay from previous cycle (dopamine does not spike and stay forever)
    previous_dopamine = GET_PREVIOUS_DOPAMINE(symbol)
    IF previous_dopamine IS NOT NULL:
        # Current level cannot be lower than decayed previous level
        # (prevents false "calm" readings during sustained volatility)
        decayed = previous_dopamine.current_level * DOPAMINE_DECAY_RATE
        dopamine_level = MAX(dopamine_level, decayed)

    state = DopamineState(
        current_level = CLAMP(dopamine_level, 0.0, 1.0),
        atr_ratio = atr_ratio,
        volume_ratio = volume_ratio,
        spike_count_1h = spike_count,
        decay_factor = DOPAMINE_DECAY_RATE,
    )

    STORE state -> toxoplasma_db.dopamine_log
    RETURN state

------------------------------------------------------------
PHASE 3: INFECTION SCORING
------------------------------------------------------------

ON compute_infection_score(strategy_id, symbol, baseline, recent_trades[], dopamine, current_regime):
    #
    # The core diagnostic: how badly is this strategy "infected"?
    #
    # Each component measures deviation from the behavioral baseline.
    # Like a blood test for Toxoplasma antibodies -- multiple markers
    # combined into a composite infection index.
    #
    # Components:
    #   1. Win rate deviation:   Is the strategy winning less than normal?
    #   2. Hold time deviation:  Is the strategy holding too long/short?
    #   3. Frequency deviation:  Is the strategy trading too much/little?
    #   4. P/L deviation:        Is the P/L distribution abnormal?
    #   5. Regime mismatch:      Is the strategy type fighting the regime?
    #   6. Dopamine influence:   Is the market flooding the system with noise?
    #

    IF NOT baseline.calibration_valid OR len(recent_trades) < 5:
        RETURN InfectionScore(infection_score=0.0, infection_phase=HEALTHY)

    recent = recent_trades[-INFECTION_LOOKBACK_TRADES:]

    # 1. Win rate deviation (0 to 1)
    # How much worse is current WR vs baseline for this regime?
    expected_wr = baseline["wr_" + current_regime]  # Regime-specific expected WR
    IF expected_wr IS NULL OR expected_wr == 0:
        expected_wr = baseline.baseline_win_rate
    actual_wr = len([t FOR t IN recent WHERE t.profit > 0]) / len(recent)
    # Bayesian smoothing: combine with prior
    posterior_wr = (8 + actual_wr * len(recent)) / (16 + len(recent))
    wr_deviation = CLAMP((expected_wr - posterior_wr) / (expected_wr + 1e-10), 0.0, 1.0)
    # Only counts if performance is WORSE than expected (positive deviation)

    # 2. Hold time deviation (0 to 1)
    actual_hold = MEAN(t.hold_time FOR t IN recent)
    hold_ratio = actual_hold / (baseline.baseline_avg_hold + 1e-10)
    # Both too-long and too-short holding are infection signs
    hold_deviation = CLAMP(ABS(hold_ratio - 1.0), 0.0, 1.0)

    # 3. Frequency deviation (0 to 1)
    actual_freq = len(recent) / time_span_days(recent)
    freq_ratio = actual_freq / (baseline.baseline_trade_freq + 1e-10)
    # Overtrading is the primary infection vector (dopamine-driven)
    freq_deviation = CLAMP(MAX(0, freq_ratio - 1.0), 0.0, 1.0)

    # 4. P/L deviation (0 to 1)
    actual_avg_pnl = MEAN(t.profit FOR t IN recent)
    IF baseline.baseline_avg_pnl > 0:
        pnl_deviation = CLAMP((baseline.baseline_avg_pnl - actual_avg_pnl) / baseline.baseline_avg_pnl, 0.0, 1.0)
    ELSE:
        pnl_deviation = CLAMP(-actual_avg_pnl, 0.0, 1.0)

    # 5. Regime mismatch (0 to 1)
    # This is the critical "running toward the cat" detector
    mismatch = compute_regime_mismatch(baseline.strategy_type, current_regime)
    # Returns 0.0 if strategy is well-suited for this regime
    # Returns 1.0 if strategy is maximally mismatched
    #
    # Mismatch matrix:
    #   MOMENTUM    + RANGING       = 0.90  (false breakout hell)
    #   MOMENTUM    + VOLATILE      = 0.60  (some trends in vol, but risky)
    #   MOMENTUM    + TRENDING_UP   = 0.05  (this is its home turf)
    #   MOMENTUM    + TRENDING_DOWN = 0.10  (can short the trend)
    #   MEAN_REVERT + TRENDING_UP   = 0.85  (catching falling knives)
    #   MEAN_REVERT + TRENDING_DOWN = 0.85  (catching falling knives)
    #   MEAN_REVERT + RANGING       = 0.05  (this is its home turf)
    #   MEAN_REVERT + VOLATILE      = 0.70  (overshoots prevent reversion)
    #   VOLATILITY  + COMPRESSED    = 0.80  (no vol to trade)
    #   HYBRID      + any           = 0.30  (partial mismatch always)

    # 6. Dopamine influence (0 to 1)
    dopamine_component = dopamine.current_level

    # COMPOSITE INFECTION SCORE
    # Weighted combination reflecting biological importance:
    #   - Regime mismatch is the primary infection vector (highest weight)
    #   - Win rate deviation is the strongest symptom
    #   - Dopamine amplifies all other components (multiplicative)
    infection_score = (
        wr_deviation * 0.25
        + hold_deviation * 0.10
        + freq_deviation * 0.15
        + pnl_deviation * 0.15
        + mismatch * 0.25
        + dopamine_component * 0.10
    )

    # Dopamine AMPLIFICATION effect
    # When dopamine is high, all infection signals are amplified
    # (Toxoplasma's dopamine production makes infection worse)
    amplification = 1.0 + 0.5 * dopamine_component
    infection_score = CLAMP(infection_score * amplification, 0.0, 1.0)

    # Phase classification
    infection_start = GET_INFECTION_START(strategy_id, symbol)
    IF infection_score < INFECTION_MILD_THRESHOLD:
        phase = HEALTHY
        infection_start = NULL
    ELSE:
        IF infection_start IS NULL:
            infection_start = NOW
        duration_hours = (NOW - infection_start).total_hours()
        IF duration_hours < ACUTE_WINDOW_HOURS:
            phase = ACUTE
        ELIF duration_hours > CHRONIC_WINDOW_HOURS:
            phase = CHRONIC
        ELSE:
            # In between: classification depends on severity
            IF infection_score >= INFECTION_CRITICAL:
                phase = CRITICAL
            ELIF infection_score >= INFECTION_SEVERE_THRESHOLD:
                # Treat as acute with elevated response
                phase = ACUTE
            ELSE:
                phase = ACUTE

    score = InfectionScore(
        strategy_id = strategy_id,
        symbol = symbol,
        timestamp = NOW,
        win_rate_deviation = wr_deviation,
        hold_time_deviation = hold_deviation,
        frequency_deviation = freq_deviation,
        pnl_deviation = pnl_deviation,
        regime_mismatch = mismatch,
        dopamine_level = dopamine_component,
        infection_score = infection_score,
        infection_phase = phase,
        infection_start = infection_start,
        infection_duration_hours = duration_hours IF phase != HEALTHY ELSE 0,
    )

    STORE score -> toxoplasma_db.infection_log
    RETURN score

------------------------------------------------------------
PHASE 4: ANTI-PARASITIC COUNTERMEASURES
------------------------------------------------------------

ON activate_countermeasures(infection_score, baseline, current_regime):
    #
    # When infection is detected, activate the immune response.
    #
    # Like the host immune system fighting Toxoplasma:
    #   - Mild infection: IFN-gamma response (moderate suppression)
    #   - Severe infection: full inflammatory cascade
    #   - Chronic: shift to tolerance + adaptation
    #
    # The goal is NOT to eliminate trading (that would be like total immune
    # suppression = death). The goal is to modify behavior to resist the
    # parasite's manipulation while maintaining the ability to trade
    # when genuine opportunities exist.
    #

    measures = CountermeasureSet(
        position_size_mult = 1.0,
        confidence_offset = 0.0,
        hold_time_mult = 1.0,
        signal_inversion = FALSE,
        strategy_pause = FALSE,
        active_measures = [],
    )

    score = infection_score.infection_score
    phase = infection_score.infection_phase

    IF phase == HEALTHY:
        RETURN measures  # No countermeasures needed

    # ---- MILD INFECTION (0.35 - 0.55) ----
    # IFN-gamma equivalent: moderate position reduction + confidence tightening
    IF score >= INFECTION_MILD_THRESHOLD:
        # Linear interpolation: 35% -> no reduction, 55% -> 25% reduction
        reduction = LERP(0.0, 0.25, (score - 0.35) / 0.20)
        measures.position_size_mult = 1.0 - reduction
        measures.confidence_offset = reduction * COUNTERMEASURE_CONFIDENCE_BOOST
        measures.active_measures.APPEND("mild_position_reduction")
        measures.active_measures.APPEND("mild_confidence_tightening")

    # ---- MODERATE INFECTION (0.55 - 0.75) ----
    # Full inflammatory response: significant reduction + shortened holds
    IF score >= INFECTION_MODERATE_THRESHOLD:
        measures.position_size_mult = COUNTERMEASURE_SIZE_REDUCTION  # 50%
        measures.confidence_offset = COUNTERMEASURE_CONFIDENCE_BOOST  # +0.15
        measures.hold_time_mult = COUNTERMEASURE_HOLD_REDUCTION  # 60%
        measures.active_measures.APPEND("moderate_position_reduction")
        measures.active_measures.APPEND("moderate_confidence_tightening")
        measures.active_measures.APPEND("hold_time_shortened")

    # ---- SEVERE INFECTION (0.75 - 0.90) ----
    # Cytokine storm equivalent: consider signal inversion
    IF score >= INFECTION_SEVERE_THRESHOLD:
        measures.position_size_mult = 0.25  # 25% size
        measures.confidence_offset = 0.25   # Much tighter threshold
        measures.hold_time_mult = 0.40      # Very short holds

        # SIGNAL INVERSION: if the market is hijacking your system,
        # doing the OPPOSITE of what it wants you to do is the correct play
        IF score >= COUNTERMEASURE_INVERSION_THRESH:
            measures.signal_inversion = TRUE
            measures.active_measures.APPEND("signal_inversion_active")
        measures.active_measures.APPEND("severe_position_reduction")
        measures.active_measures.APPEND("severe_confidence_lockdown")

    # ---- CRITICAL INFECTION (>0.90) ----
    # Organ failure equivalent: halt the strategy entirely
    IF score >= INFECTION_CRITICAL:
        measures.strategy_pause = TRUE
        measures.position_size_mult = 0.0
        measures.active_measures.APPEND("STRATEGY_PAUSED_CRITICAL_INFECTION")

    # ---- CHRONIC ADAPTATION ----
    # For chronic infections (> 48 hours), shift the baseline slowly
    # This is like the immune system learning to coexist with the parasite
    IF phase == CHRONIC:
        # Reduce countermeasure intensity over time (tolerance development)
        duration = infection_score.infection_duration_hours
        tolerance = MIN(0.5, duration * CHRONIC_ADAPTATION_RATE)
        # Partially relax measures (but never fully)
        measures.position_size_mult = MIN(0.75, measures.position_size_mult + tolerance * 0.5)
        measures.confidence_offset = MAX(0.05, measures.confidence_offset - tolerance * 0.1)
        measures.active_measures.APPEND("chronic_tolerance_adaptation")

    STORE measures -> toxoplasma_db.countermeasure_log
    RETURN measures

------------------------------------------------------------
PHASE 5: TE EPIGENETIC TRACKING
------------------------------------------------------------

ON track_te_manipulation(active_tes[], infection_score, trade_outcome):
    #
    # Toxoplasma activates host transposable elements (L1, SINE) as part
    # of its epigenetic hijacking strategy. Similarly, certain TE activation
    # patterns in our system may be ASSOCIATED with regime manipulation.
    #
    # If specific TE combos consistently appear during infection periods
    # AND those combos lead to losses, they are "manipulation tools" --
    # the market's equivalent of Toxoplasma's effector proteins.
    #
    # This information feeds BACK into the TE Domestication system:
    # TE patterns flagged as manipulation tools should get SUPPRESSED
    # (increased methylation) rather than domesticated.
    #

    combo = sorted(active_tes).join("+")
    hash = MD5(combo)[:16]

    row = toxoplasma_db.SELECT WHERE te_combo_hash = hash

    is_infected = (infection_score.infection_score >= INFECTION_MILD_THRESHOLD)
    won = (trade_outcome.profit > 0)

    IF row EXISTS:
        IF is_infected:
            row.seen_during_infection += 1
            IF won: row.infection_wins += 1
            ELSE:   row.infection_losses += 1
        ELSE:
            row.seen_during_healthy += 1

        total_infection = row.seen_during_infection
        total_healthy = row.seen_during_healthy
        row.infection_correlation = total_infection / (total_infection + total_healthy + 1e-10)

        total_inf_trades = row.infection_wins + row.infection_losses
        row.infection_wr = row.infection_wins / (total_inf_trades + 1e-10) IF total_inf_trades > 0 ELSE 0.5

        # Flag as manipulation tool if:
        # 1. Strongly correlated with infection (>60% of appearances during infection)
        # 2. Low win rate during infection (<45%)
        # 3. Seen enough times to be statistically meaningful
        row.is_manipulation_tool = (
            row.infection_correlation > 0.60
            AND row.infection_wr < 0.45
            AND total_inf_trades >= TE_MANIPULATION_MIN_COUNT
        )

        row.last_seen = NOW
        toxoplasma_db.UPDATE row

    ELSE:
        toxoplasma_db.INSERT(
            te_combo_hash = hash,
            te_combo = combo,
            seen_during_infection = 1 IF is_infected ELSE 0,
            seen_during_healthy = 0 IF is_infected ELSE 1,
            infection_correlation = 1.0 IF is_infected ELSE 0.0,
            infection_wins = 1 IF (is_infected AND won) ELSE 0,
            infection_losses = 1 IF (is_infected AND NOT won) ELSE 0,
            infection_wr = 1.0 IF won ELSE 0.0,
            is_manipulation_tool = FALSE,
            first_seen = NOW,
            last_seen = NOW,
        )

------------------------------------------------------------
PHASE 6: REGIME MISMATCH MATRIX
------------------------------------------------------------

FUNCTION compute_regime_mismatch(strategy_type, current_regime) -> REAL:
    #
    # The mismatch matrix encodes how badly a strategy type conflicts
    # with the current market regime. This is the "infection vector" --
    # the route by which the market hijacks the strategy.
    #
    # High mismatch = the market is trying to make you trade against
    # your strategy's nature. This is the "running toward the cat" moment.
    #

    MISMATCH_MATRIX = {
        # (strategy_type, regime): mismatch score 0.0 to 1.0
        (MOMENTUM, TRENDING_UP):     0.05,  # Home turf
        (MOMENTUM, TRENDING_DOWN):   0.10,  # Can short
        (MOMENTUM, RANGING):         0.90,  # False breakout hell
        (MOMENTUM, VOLATILE):        0.60,  # Some trend in volatility but risky
        (MOMENTUM, COMPRESSED):      0.40,  # Pre-breakout, could go either way
        (MOMENTUM, TRANSITIONING):   0.70,  # Most dangerous

        (MEAN_REVERSION, TRENDING_UP):     0.85,  # Catching falling knives (inverse)
        (MEAN_REVERSION, TRENDING_DOWN):   0.85,  # Catching falling knives
        (MEAN_REVERSION, RANGING):         0.05,  # Home turf
        (MEAN_REVERSION, VOLATILE):        0.70,  # Overshoots prevent reversion
        (MEAN_REVERSION, COMPRESSED):      0.20,  # Some mean-reversion in squeezes
        (MEAN_REVERSION, TRANSITIONING):   0.80,  # Dangerous

        (VOLATILITY, TRENDING_UP):     0.30,
        (VOLATILITY, TRENDING_DOWN):   0.30,
        (VOLATILITY, RANGING):         0.40,
        (VOLATILITY, VOLATILE):        0.05,  # Home turf
        (VOLATILITY, COMPRESSED):      0.80,  # No vol to trade
        (VOLATILITY, TRANSITIONING):   0.15,  # Vol strategies like transitions

        (HYBRID, TRENDING_UP):     0.25,
        (HYBRID, TRENDING_DOWN):   0.25,
        (HYBRID, RANGING):         0.30,
        (HYBRID, VOLATILE):        0.35,
        (HYBRID, COMPRESSED):      0.40,
        (HYBRID, TRANSITIONING):   0.45,
    }

    RETURN MISMATCH_MATRIX.GET((strategy_type, current_regime), 0.30)

------------------------------------------------------------
THE LOOP (steady-state behavior)
------------------------------------------------------------

LOOP every ~60 seconds (aligned with TEQA cycle):

    FOR EACH active_strategy IN registered_strategies:

        # Step 1: Classify current regime
        regime = classify_regime(bars)

        # Step 2: Compute dopamine state
        dopamine = compute_dopamine(bars, symbol)

        # Step 3: Get or calibrate behavioral baseline
        baseline = GET_OR_CALIBRATE_BASELINE(strategy, symbol, trade_history)
        IF baseline.last_calibrated + BASELINE_RECALIBRATE_HOURS < NOW:
            baseline = calibrate_baseline(strategy, trade_history)

        # Step 4: Score infection level
        infection = compute_infection_score(
            strategy.id, symbol, baseline,
            recent_trades, dopamine, regime
        )

        # Step 5: Activate countermeasures based on infection
        measures = activate_countermeasures(infection, baseline, regime)

        # Step 6: APPLY countermeasures to trading decisions
        IF measures.strategy_pause:
            BLOCK all signals for this strategy
        ELSE:
            MODIFY signal.confidence += measures.confidence_offset
            MODIFY position.size *= measures.position_size_mult
            MODIFY max_hold_time *= measures.hold_time_mult
            IF measures.signal_inversion:
                FLIP signal.direction  # BUY -> SELL, SELL -> BUY

        # Step 7: Track TE patterns during infection
        IF infection.infection_score >= INFECTION_MILD_THRESHOLD:
            active_tes = GET_CURRENT_TE_ACTIVATIONS()
            track_te_manipulation(active_tes, infection, last_trade_outcome)

        # Step 8: Emit toxoplasma status to signal file
        WRITE toxoplasma_status -> toxoplasma_signal.json

------------------------------------------------------------
INTEGRATION POINTS
------------------------------------------------------------

INTEGRATION with TEQA v3.0:
    - Toxoplasma reads TE activations from TEActivationEngine
    - Feeds manipulation flags back to TE Domestication tracker:
      IF te_pattern.is_manipulation_tool:
          domestication_db.SUPPRESS(te_pattern, penalty=0.15)
    - Toxoplasma countermeasures modify the TEQA confidence output:
      final_confidence = teqa_confidence + toxoplasma.confidence_offset

INTEGRATION with VDJ Recombination:
    - Antibody fitness evaluation incorporates infection context:
      IF antibody was selected during high infection period:
          DISCOUNT fitness by infection_score * 0.3
    - Memory B cells that only win during infection are flagged
      (they may be "manipulation-dependent" -- false positives)

INTEGRATION with BRAIN scripts:
    - BRAIN reads toxoplasma_signal.json before each trade
    - Applies position_size_mult to lot calculation
    - Uses confidence_offset to adjust CONFIDENCE_THRESHOLD (from config_loader)
    - Respects strategy_pause to skip trading cycles
    - Logs infection status in trade metadata for post-hoc analysis

INTEGRATION with TE Domestication:
    - TE patterns flagged as manipulation tools receive a domestication PENALTY
    - This prevents the system from "domesticating" patterns that only work
      because the market is tricking the system into bad trades
    - Like the immune system learning to recognize Toxoplasma effector proteins
      as "non-self" rather than incorporating them as "self"

------------------------------------------------------------
INVARIANT
------------------------------------------------------------

INVARIANT:
    "The market is a parasite that hijacks your behavior through regime shifts.
     Toxoplasma detects the hijacking and flips the script.
     When the market makes you want to buy, it is time to sell.
     The system that survives is the one that recognizes when it is infected."

BIOLOGICAL PARALLEL:
    behavioral_baseline      = healthy immune system (knows "self")
    infection_score          = parasitic load measurement
    dopamine_state           = Toxoplasma TH enzyme output (dopamine in amygdala)
    regime_mismatch          = parasite lifecycle requirement (rodent must go to cat)
    countermeasures          = IFN-gamma / IL-12 immune response
    signal_inversion         = "don't run toward the cat" behavioral override
    strategy_pause           = quarantine (isolate infected host)
    chronic_adaptation       = bradyzoite tolerance (live with the parasite)
    te_manipulation_tracking = identifying effector proteins (ROP16, GRA15, GRA16)
    is_manipulation_tool     = non-self antigen recognition

CONVERGENCE:
    After ~30 trades per strategy per regime, the system learns:
    1. What each strategy looks like when healthy (baseline)
    2. What infection looks like (deviation + dopamine + mismatch)
    3. Which TE patterns are manipulation tools vs genuine signals
    4. How to dose countermeasures proportionally (not over-react)
    5. When chronic infection requires adaptation vs resistance

    The result: a trading system that does not "run toward the cat"
    when the market's dopamine (volatility/volume spikes) tells it to.
    Instead, it recognizes the manipulation and either resists or inverts.

DATABASES:
    toxoplasma_infection.db  <- infection scores, baselines, dopamine, countermeasures
    teqa_domestication.db    <- TE patterns (receives manipulation flags)
    MT5 deal history         <- trade outcomes for baseline calibration

FILES:
    toxoplasma.py            -> ToxoplasmaEngine class (main implementation)
    toxoplasma_signal.json   -> current infection status for BRAIN scripts
    ALGORITHM_TOXOPLASMA.py  -> this specification
    BRAIN_ATLAS.py           -> reads toxoplasma status each cycle
    teqa_v3_neural_te.py     -> provides TE activations
    config_loader.py         -> CONFIDENCE_THRESHOLD, MAX_LOSS_DOLLARS
"""
