"""
ALGORITHM #8: Syncytin -- "Strategy Fusion into Hybrid Organisms"
==================================================================
Fuse complementary strategies into multi-regime hybrid organisms
with shared risk envelopes and compartmentalized signal logic.

Biological basis:
    Syncytin is a protein derived from a retroviral envelope (env) gene
    that was domesticated by mammals. Its history:

    1. ORIGIN: An ancient retrovirus used its envelope protein to FUSE
       its membrane with host cell membranes, injecting viral DNA into
       the host cell. This was an invasion mechanism.

    2. ENDOGENIZATION: The retrovirus integrated into the host germline
       and was passed vertically to offspring. Over millions of years,
       most viral genes were silenced or deleted by the host's defense
       mechanisms (piRNA, methylation, APOBEC3).

    3. DOMESTICATION: The env gene was SPARED from silencing because it
       provided a selective advantage: cell-cell fusion. The host co-opted
       this viral fusion protein for its own purposes.

    4. PLACENTAL FORMATION: Syncytin drives the formation of the
       syncytiotrophoblast -- the multinucleated outer layer of the
       placenta. Trophoblast cells fuse their membranes together,
       creating a continuous barrier between mother and fetus.

    5. DUAL FUNCTION: The syncytiotrophoblast serves two critical roles:
       a) IMMUNE BARRIER: Protects the fetus (which carries paternal
          antigens) from the mother's immune system. Without this barrier,
          the mother's T cells would attack the fetus as foreign tissue.
       b) NUTRIENT EXCHANGE: Despite being a barrier, the fused layer
          allows selective transport of nutrients, oxygen, and waste
          between maternal and fetal blood supplies.

    6. CONVERGENT DOMESTICATION: This happened INDEPENDENTLY in multiple
       mammalian lineages. Different retroviruses donated different
       syncytin genes:
       - Syncytin-1 (HERV-W): Primates, ~25 million years ago
       - Syncytin-2 (HERV-FRD): Primates, ~40 million years ago
       - Syncytin-A/B: Rodents, different retroviral source
       - Syncytin-Rum1: Ruminants, yet another retroviral source
       - Syncytin-Car1: Carnivores, yet another retroviral source
       Each lineage independently discovered that viral fusion proteins
       make excellent placental glue.

    7. FUSOGENIC MECHANISM: Syncytin has a fusion peptide domain that
       inserts into the target cell membrane, pulling the two lipid
       bilayers together until they merge. The merged cell inherits
       cytoplasm, organelles, and nuclei from both parent cells.

    8. RECEPTOR DEPENDENCY: Syncytin-1 binds ASCT2 (alanine/serine/
       cysteine transporter 2) on the target cell. Without the correct
       receptor, fusion cannot occur. This is the COMPATIBILITY CHECK --
       not all cells can fuse with all other cells.

    KEY BIOLOGICAL INSIGHT:
    A mechanism designed for INVASION (viral cell fusion) was repurposed
    for NURTURING (placental formation). The fusion creates something
    neither parent cell could be alone: a multinucleated syncytium with
    shared resources but protected compartments.

Trading analogy -- Strategy Fusion:
    Take two independent strategies that are each profitable but in
    DIFFERENT market conditions. FUSE them into a single hybrid organism
    that inherits the best properties of both.

    Like syncytin fusing trophoblast cells:
    - Each strategy is a "cell" with its own nucleus (signal logic)
    - The hybrid has a shared membrane (risk management envelope)
    - The two nuclei maintain their independent signal generation
    - The shared cytoplasm allows nutrient exchange (equity sharing)
    - The immune barrier prevents one strategy's losses from killing
      the other (compartmentalized drawdown limits)

    The fusion creates a MULTI-REGIME ORGANISM that neither parent
    strategy could be alone:
    - Strategy A wins in trends but bleeds in ranges
    - Strategy B wins in ranges but bleeds in trends
    - The hybrid routes to A during trends and B during ranges
    - The shared risk envelope limits total exposure
    - Profits from one sub-strategy subsidize the other's drawdowns

    FUSION FITNESS: The hybrid must outperform either parent alone.
    If it does not, the fusion was unproductive and should be dissolved
    (defusion). Like a placenta that fails to form properly -- the
    pregnancy is not viable.

    INDEPENDENT DOMESTICATION: Just as different mammals domesticated
    different syncytins from different retroviruses, the system can
    explore multiple fusion candidates simultaneously. Each fusion
    is an independent experiment.

Integration:
    - Reads VDJ memory cells (Algorithm #1) as fusion candidates
    - Reads TE domestication DB (Algorithm #2) for fitness data
    - Uses regime detection from config_loader for regime routing
    - Stores fusion records in SQLite for persistent learning
    - Emits hybrid signals to JSON for BRAIN script consumption
    - Monitors hybrid fitness and triggers defusion if degraded

Authors: DooDoo + Claude
Date:    2026-02-08
Version: SYNCYTIN-1.0

============================================================
PSEUDOCODE
============================================================
"""

# ──────────────────────────────────────────────────────────
# CONSTANTS (no hardcoded trading values -- config_loader)
# ──────────────────────────────────────────────────────────

# Fusion candidate screening
MIN_PARENT_TRADES          = 20     # Each parent must have >= 20 trades
MIN_PARENT_WIN_RATE        = 0.55   # Each parent must have >= 55% WR
MAX_RETURN_CORRELATION     = 0.30   # Parents must be negatively or weakly correlated
MIN_REGIME_COMPLEMENTARITY = 0.40   # At least 40% non-overlapping regime coverage

# Compatibility scoring (receptor check)
COMPATIBILITY_THRESHOLD    = 0.60   # Minimum compatibility score for fusion
RECEPTOR_ASCT2_FIELDS      = [      # Fields checked for receptor compatibility:
    "regime_coverage",              #   Which regimes each strategy covers
    "return_correlation",           #   Correlation of return streams
    "drawdown_overlap",             #   Do they draw down at the same time?
    "signal_frequency_ratio",       #   Are trade frequencies roughly balanced?
]

# Envelope protein (shared risk management)
ENVELOPE_MAX_COMBINED_SL   = "config_loader.MAX_LOSS_DOLLARS"  # From config_loader
ENVELOPE_MAX_DRAWDOWN_PCT  = 0.05   # 5% max drawdown for the hybrid
ENVELOPE_POSITION_LIMIT    = 2      # Max concurrent positions across both sub-strategies

# Syncytiotrophoblast (the fused hybrid)
REGIME_DETECTION_LOOKBACK  = 50     # Bars to look back for regime classification
REGIME_SWITCH_COOLDOWN     = 5      # Minimum bars between regime switches
EQUITY_SHARING_RATE        = 0.20   # 20% of one sub-strategy's profit can subsidize the other

# Immune barrier (compartmentalized risk)
COMPARTMENT_MAX_LOSS_PCT   = 0.60   # Each sub-strategy can use at most 60% of total risk budget
COMPARTMENT_KILL_THRESHOLD = 0.03   # If one sub-strategy hits 3% DD, temporarily suspend it

# Fitness monitoring
FUSION_MIN_TRADES          = 30     # Minimum trades before judging hybrid fitness
FUSION_MIN_IMPROVEMENT     = 0.05   # Hybrid must be 5% better than best parent
DEFUSION_DEGRADATION_PCT   = -0.10  # Defuse if hybrid underperforms best parent by 10%
FUSION_REEVAL_INTERVAL     = 50     # Re-evaluate fusion fitness every 50 trades

# Limits
MAX_ACTIVE_FUSIONS         = 10     # Maximum concurrent hybrid organisms
MAX_FUSION_CANDIDATES      = 50     # Maximum candidate pairs to screen per cycle

"""
============================================================
ALGORITHM Syncytin -- Strategy Fusion
============================================================

DEFINE Strategy AS:
    strategy_id       : TEXT         -- hash of strategy parameters
    strategy_type     : TEXT         -- "vdj_antibody" | "te_pattern" | "custom"
    source_id         : TEXT         -- antibody_id or pattern_hash
    signal_logic      : CALLABLE    -- function(bars) -> {direction, confidence}
    regime_affinity   : DICT        -- {TRENDING: 0.8, RANGING: 0.2, ...}
    fitness_metrics   : DICT        -- {win_rate, profit_factor, sharpe, ...}
    return_stream     : FLOAT[]     -- historical trade returns
    trade_count       : INTEGER
    regime_trades     : DICT        -- {TRENDING: {wins, losses}, RANGING: {wins, losses}}

DEFINE FusionCandidate AS:
    strategy_a        : Strategy
    strategy_b        : Strategy
    compatibility     : REAL        -- compatibility score [0.0, 1.0]
    return_corr       : REAL        -- Pearson correlation of return streams
    regime_complement : REAL        -- fraction of non-overlapping regime coverage
    drawdown_overlap  : REAL        -- correlation of drawdown periods
    frequency_ratio   : REAL        -- min(freq_a, freq_b) / max(freq_a, freq_b)
    receptor_match    : BOOLEAN     -- TRUE if compatible (ASCT2 receptor check)

DEFINE HybridOrganism AS:
    hybrid_id         : TEXT PRIMARY KEY
    strategy_a_id     : TEXT         -- nucleus A
    strategy_b_id     : TEXT         -- nucleus B
    fusion_type       : TEXT         -- "regime_switch" | "weighted_blend" | "cascade"

    # Shared membrane (envelope protein)
    envelope          : EnvelopeProtein

    # Immune barrier (compartmentalized risk)
    compartment_a     : RiskCompartment
    compartment_b     : RiskCompartment

    # Routing logic
    current_regime    : TEXT         -- TRENDING | RANGING | VOLATILE | etc.
    active_strategy   : TEXT         -- "A" | "B" | "BOTH"
    regime_switch_bar : INTEGER      -- last bar where regime switched

    # Fitness tracking
    hybrid_trades     : INTEGER
    hybrid_returns    : FLOAT[]
    parent_a_solo     : FLOAT[]     -- what A would have done alone
    parent_b_solo     : FLOAT[]     -- what B would have done alone
    fusion_alpha      : REAL        -- hybrid return - max(solo_a, solo_b) return

    # Status
    status            : TEXT         -- "active" | "probation" | "defused"
    created_at        : TIMESTAMP
    last_evaluated    : TIMESTAMP

DEFINE EnvelopeProtein AS:
    max_sl_dollars    : REAL         -- from config_loader.MAX_LOSS_DOLLARS
    max_drawdown_pct  : REAL         -- maximum drawdown for the hybrid
    position_limit    : INTEGER      -- max concurrent positions
    current_positions : INTEGER      -- current open positions
    current_drawdown  : REAL         -- current drawdown of the hybrid

DEFINE RiskCompartment AS:
    strategy_id       : TEXT
    budget_pct        : REAL         -- fraction of total risk budget (0.0 to 1.0)
    current_dd        : REAL         -- current drawdown in this compartment
    suspended         : BOOLEAN      -- TRUE if compartment hit kill threshold
    trades_since_susp : INTEGER      -- trades since last suspension
    equity_balance    : REAL         -- running equity for this compartment

STORAGE:
    syncytin_db       : SQLite "syncytin_fusions.db"
    Tables:
        hybrid_organisms  -- all fusion records
        fusion_trades     -- individual trade log per hybrid
        fusion_candidates -- screened candidate pairs
        fusion_fitness    -- periodic fitness evaluations

────────────────────────────────────────────────────────────
PHASE 1: CANDIDATE SCREENING (identify fusible strategy pairs)
────────────────────────────────────────────────────────────

ON screen_candidates(strategy_pool[]):

    # Step 1: Filter strategies that meet minimum quality
    qualified = []
    FOR strategy IN strategy_pool:
        IF strategy.trade_count >= MIN_PARENT_TRADES
           AND strategy.fitness_metrics.win_rate >= MIN_PARENT_WIN_RATE:
            qualified.APPEND(strategy)

    # Step 2: Generate all unique pairs
    pairs = COMBINATIONS(qualified, 2)

    # Step 3: Score each pair for fusion compatibility
    candidates = []
    FOR (A, B) IN pairs:

        # 3a. Return correlation (want NEGATIVE or LOW)
        corr = PEARSON_CORRELATION(A.return_stream, B.return_stream)
        IF corr > MAX_RETURN_CORRELATION:
            SKIP  # Too correlated = redundant, not complementary

        # 3b. Regime complementarity (want HIGH)
        # Measure how much their regime affinities differ
        regime_overlap = 0.0
        FOR regime IN [TRENDING, RANGING, VOLATILE, COMPRESSED, BREAKOUT]:
            overlap = MIN(A.regime_affinity[regime], B.regime_affinity[regime])
            regime_overlap += overlap
        regime_complement = 1.0 - (regime_overlap / num_regimes)
        IF regime_complement < MIN_REGIME_COMPLEMENTARITY:
            SKIP  # Too similar regime coverage

        # 3c. Drawdown overlap (want LOW)
        dd_corr = PEARSON_CORRELATION(
            rolling_drawdown(A.return_stream),
            rolling_drawdown(B.return_stream)
        )

        # 3d. Signal frequency balance (want BALANCED)
        freq_a = A.trade_count / A.observation_period
        freq_b = B.trade_count / B.observation_period
        freq_ratio = MIN(freq_a, freq_b) / MAX(freq_a, freq_b)

        # 3e. Composite compatibility score
        # Receptor check: all conditions must be met
        compatibility = (
            (1.0 - corr) * 0.35            # Low correlation is good
            + regime_complement * 0.30       # Complementary regimes
            + (1.0 - dd_corr) * 0.20        # Non-overlapping drawdowns
            + freq_ratio * 0.15              # Balanced frequency
        )

        receptor_match = (compatibility >= COMPATIBILITY_THRESHOLD)

        candidates.APPEND(FusionCandidate(
            strategy_a=A, strategy_b=B,
            compatibility=compatibility,
            return_corr=corr,
            regime_complement=regime_complement,
            drawdown_overlap=dd_corr,
            frequency_ratio=freq_ratio,
            receptor_match=receptor_match,
        ))

    # Step 4: Sort by compatibility, keep top N
    candidates.SORT(key=compatibility, descending=True)
    candidates = candidates[:MAX_FUSION_CANDIDATES]

    RETURN [c FOR c IN candidates WHERE c.receptor_match]

────────────────────────────────────────────────────────────
PHASE 2: FUSION (create the hybrid organism)
────────────────────────────────────────────────────────────

ON fuse(candidate: FusionCandidate) -> HybridOrganism:

    A = candidate.strategy_a
    B = candidate.strategy_b

    # Step 1: Determine fusion type based on regime complementarity
    IF candidate.regime_complement > 0.70:
        fusion_type = "regime_switch"
        # Strong complementarity: route entirely to A or B based on regime
    ELIF candidate.regime_complement > 0.40:
        fusion_type = "weighted_blend"
        # Moderate complementarity: weighted average of both signals
    ELSE:
        fusion_type = "cascade"
        # Low complementarity: A generates candidates, B confirms/rejects

    # Step 2: Build envelope protein (shared risk management)
    # Uses config_loader values -- NO hardcoded trading values
    envelope = EnvelopeProtein(
        max_sl_dollars    = config_loader.MAX_LOSS_DOLLARS,
        max_drawdown_pct  = ENVELOPE_MAX_DRAWDOWN_PCT,
        position_limit    = ENVELOPE_POSITION_LIMIT,
        current_positions = 0,
        current_drawdown  = 0.0,
    )

    # Step 3: Build risk compartments (immune barrier)
    # Budget allocation weighted by each strategy's fitness
    total_fitness = A.fitness_metrics.fitness + B.fitness_metrics.fitness
    budget_a = A.fitness_metrics.fitness / total_fitness
    budget_b = B.fitness_metrics.fitness / total_fitness
    # Clamp to compartment limits
    budget_a = CLAMP(budget_a, 1.0 - COMPARTMENT_MAX_LOSS_PCT, COMPARTMENT_MAX_LOSS_PCT)
    budget_b = 1.0 - budget_a

    compartment_a = RiskCompartment(
        strategy_id=A.strategy_id,
        budget_pct=budget_a,
        current_dd=0.0,
        suspended=FALSE,
        trades_since_susp=0,
        equity_balance=0.0,
    )
    compartment_b = RiskCompartment(
        strategy_id=B.strategy_id,
        budget_pct=budget_b,
        current_dd=0.0,
        suspended=FALSE,
        trades_since_susp=0,
        equity_balance=0.0,
    )

    # Step 4: Create the hybrid
    hybrid_id = MD5(A.strategy_id + "|" + B.strategy_id + "|" + fusion_type)[:16]

    hybrid = HybridOrganism(
        hybrid_id=hybrid_id,
        strategy_a_id=A.strategy_id,
        strategy_b_id=B.strategy_id,
        fusion_type=fusion_type,
        envelope=envelope,
        compartment_a=compartment_a,
        compartment_b=compartment_b,
        current_regime="UNKNOWN",
        active_strategy="BOTH",
        regime_switch_bar=0,
        hybrid_trades=0,
        hybrid_returns=[],
        parent_a_solo=[],
        parent_b_solo=[],
        fusion_alpha=0.0,
        status="active",
        created_at=NOW(),
        last_evaluated=NOW(),
    )

    syncytin_db.INSERT(hybrid)
    RETURN hybrid

────────────────────────────────────────────────────────────
PHASE 3: SIGNAL ROUTING (the syncytiotrophoblast in action)
────────────────────────────────────────────────────────────

ON hybrid_signal(hybrid: HybridOrganism, bars: np.ndarray) -> Signal:

    # Step 1: Detect current market regime
    regime = detect_regime(bars, lookback=REGIME_DETECTION_LOOKBACK)
    #   detect_regime returns one of:
    #     TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, COMPRESSED, BREAKOUT

    # Step 2: Check regime switch cooldown
    bars_since_switch = current_bar - hybrid.regime_switch_bar
    IF regime != hybrid.current_regime AND bars_since_switch >= REGIME_SWITCH_COOLDOWN:
        hybrid.current_regime = regime
        hybrid.regime_switch_bar = current_bar

    # Step 3: Route to appropriate sub-strategy based on fusion type
    IF hybrid.fusion_type == "regime_switch":
        # Pure routing: one strategy gets full control
        affinity_a = strategy_a.regime_affinity.get(regime, 0.0)
        affinity_b = strategy_b.regime_affinity.get(regime, 0.0)

        IF affinity_a > affinity_b:
            hybrid.active_strategy = "A"
            signal = strategy_a.signal_logic(bars)
        ELSE:
            hybrid.active_strategy = "B"
            signal = strategy_b.signal_logic(bars)

    ELIF hybrid.fusion_type == "weighted_blend":
        # Weighted combination of both signals
        signal_a = strategy_a.signal_logic(bars)
        signal_b = strategy_b.signal_logic(bars)

        affinity_a = strategy_a.regime_affinity.get(regime, 0.5)
        affinity_b = strategy_b.regime_affinity.get(regime, 0.5)
        total_aff = affinity_a + affinity_b

        weight_a = affinity_a / total_aff
        weight_b = affinity_b / total_aff

        blended_direction = SIGN(
            signal_a.direction * signal_a.confidence * weight_a
            + signal_b.direction * signal_b.confidence * weight_b
        )
        blended_confidence = (
            signal_a.confidence * weight_a
            + signal_b.confidence * weight_b
        )
        signal = Signal(direction=blended_direction, confidence=blended_confidence)
        hybrid.active_strategy = "BOTH"

    ELIF hybrid.fusion_type == "cascade":
        # Strategy A proposes, Strategy B confirms
        signal_a = strategy_a.signal_logic(bars)
        IF signal_a.confidence > CONFIDENCE_THRESHOLD:
            signal_b = strategy_b.signal_logic(bars)
            IF SIGN(signal_a.direction) == SIGN(signal_b.direction):
                # Both agree: high confidence
                signal = Signal(
                    direction=signal_a.direction,
                    confidence=MAX(signal_a.confidence, signal_b.confidence),
                )
            ELSE:
                # Disagreement: no trade
                signal = Signal(direction=0, confidence=0.0)
        ELSE:
            signal = Signal(direction=0, confidence=0.0)
        hybrid.active_strategy = "A+B"

    # Step 4: Envelope check (shared risk management)
    IF hybrid.envelope.current_positions >= hybrid.envelope.position_limit:
        signal.direction = 0  # Block: position limit reached

    IF hybrid.envelope.current_drawdown > hybrid.envelope.max_drawdown_pct:
        signal.direction = 0  # Block: drawdown limit reached

    # Step 5: Compartment check (immune barrier)
    active_compartment = hybrid.compartment_a IF hybrid.active_strategy == "A" \
                         ELSE hybrid.compartment_b

    IF active_compartment.suspended:
        signal.direction = 0  # Block: this compartment is suspended

    RETURN signal

────────────────────────────────────────────────────────────
PHASE 4: TRADE OUTCOME & COMPARTMENT ACCOUNTING
────────────────────────────────────────────────────────────

ON record_hybrid_trade(hybrid: HybridOrganism, pnl: REAL, active: TEXT):

    # Step 1: Record in hybrid's return stream
    hybrid.hybrid_returns.APPEND(pnl)
    hybrid.hybrid_trades += 1

    # Step 2: Update the appropriate compartment
    IF active == "A":
        compartment = hybrid.compartment_a
    ELSE:
        compartment = hybrid.compartment_b

    compartment.equity_balance += pnl

    # Update compartment drawdown
    IF compartment.equity_balance < 0:
        compartment.current_dd = ABS(compartment.equity_balance)
    ELSE:
        compartment.current_dd = 0.0

    # Step 3: Immune barrier check -- suspend compartment if DD too high
    IF compartment.current_dd > COMPARTMENT_KILL_THRESHOLD * account_equity:
        compartment.suspended = TRUE
        compartment.trades_since_susp = 0
        LOG("IMMUNE BARRIER: Compartment %s suspended (DD=%.4f)", active, compartment.current_dd)

    # Step 4: Nutrient exchange (equity sharing)
    # If one compartment is profitable, it can subsidize the other
    other = hybrid.compartment_b IF active == "A" ELSE hybrid.compartment_a
    IF compartment.equity_balance > 0 AND other.equity_balance < 0:
        subsidy = compartment.equity_balance * EQUITY_SHARING_RATE
        other.equity_balance += subsidy
        compartment.equity_balance -= subsidy
        LOG("NUTRIENT EXCHANGE: %.2f from %s to %s", subsidy, active, other_label)

    # Step 5: Check if suspended compartment can resume
    IF compartment.suspended:
        compartment.trades_since_susp += 1
        IF compartment.current_dd < COMPARTMENT_KILL_THRESHOLD * 0.5:
            compartment.suspended = FALSE
            LOG("IMMUNE BARRIER: Compartment %s resumed", active)

    # Step 6: Update envelope
    hybrid.envelope.current_drawdown = MAX(
        hybrid.compartment_a.current_dd,
        hybrid.compartment_b.current_dd,
    )

    # Step 7: Record what each parent would have done solo (for alpha calc)
    # (This requires running both strategy signal_logic on each bar
    #  regardless of which was active -- shadow execution)
    # shadow_a = strategy_a.evaluate_at_bar(bars, current_bar)
    # shadow_b = strategy_b.evaluate_at_bar(bars, current_bar)
    # hybrid.parent_a_solo.APPEND(shadow_a.pnl)
    # hybrid.parent_b_solo.APPEND(shadow_b.pnl)

    syncytin_db.UPDATE(hybrid)

────────────────────────────────────────────────────────────
PHASE 5: FUSION FITNESS MONITORING (is the hybrid viable?)
────────────────────────────────────────────────────────────

ON evaluate_fusion_fitness(hybrid: HybridOrganism):

    IF hybrid.hybrid_trades < FUSION_MIN_TRADES:
        RETURN  # Not enough data yet

    # Compute hybrid metrics
    hybrid_wr = wins / hybrid.hybrid_trades
    hybrid_pf = avg_win / avg_loss
    hybrid_sharpe = mean(returns) / std(returns) * sqrt(252)
    hybrid_total = SUM(hybrid.hybrid_returns)

    # Compute solo parent metrics (from shadow execution)
    solo_a_total = SUM(hybrid.parent_a_solo)
    solo_b_total = SUM(hybrid.parent_b_solo)
    best_parent_total = MAX(solo_a_total, solo_b_total)

    # Fusion alpha = how much better is the hybrid than the best parent alone?
    IF best_parent_total != 0:
        fusion_alpha = (hybrid_total - best_parent_total) / ABS(best_parent_total)
    ELSE:
        fusion_alpha = 0.0 IF hybrid_total == 0 ELSE 1.0

    hybrid.fusion_alpha = fusion_alpha

    # Decision logic
    IF fusion_alpha >= FUSION_MIN_IMPROVEMENT:
        hybrid.status = "active"
        LOG("FUSION FIT: Hybrid %s is viable (alpha=%.2f%%)", hybrid.hybrid_id, fusion_alpha*100)

    ELIF fusion_alpha >= 0:
        hybrid.status = "probation"
        LOG("FUSION MARGINAL: Hybrid %s on probation (alpha=%.2f%%)", hybrid.hybrid_id, fusion_alpha*100)

    ELIF fusion_alpha < DEFUSION_DEGRADATION_PCT:
        hybrid.status = "defused"
        LOG("DEFUSION: Hybrid %s dissolved (alpha=%.2f%%)", hybrid.hybrid_id, fusion_alpha*100)

    syncytin_db.UPDATE(hybrid)

────────────────────────────────────────────────────────────
PHASE 6: REGIME DETECTION ENGINE
────────────────────────────────────────────────────────────

ON detect_regime(bars, lookback) -> RegimeType:

    close = bars[-lookback:, 3]
    high  = bars[-lookback:, 1]
    low   = bars[-lookback:, 2]

    # ATR ratio for volatility assessment
    atr_recent = ATR(high[-5:], low[-5:], close[-5:])
    atr_full   = ATR(high, low, close)
    atr_ratio  = atr_recent / (atr_full + 1e-10)

    # EMA spread for trend assessment
    ema_fast = EMA(close, 8)
    ema_slow = EMA(close, 21)
    ema_diff = (ema_fast - ema_slow) / (ema_slow + 1e-10)

    # Bollinger width for compression assessment
    sma     = MEAN(close[-20:])
    std_dev = STD(close[-20:])
    bb_width = (2 * std_dev) / (sma + 1e-10)

    # Classification
    IF atr_ratio < 0.6:
        RETURN COMPRESSED
    IF atr_ratio > 1.5:
        RETURN VOLATILE
    IF ABS(ema_diff) > 0.005:
        IF ema_diff > 0:
            RETURN TRENDING_UP
        ELSE:
            RETURN TRENDING_DOWN
    IF bb_width < 0.02:
        RETURN RANGING
    # Check for breakout (compression -> expansion)
    IF atr_ratio > 1.3 AND was_compressed_recently:
        RETURN BREAKOUT

    RETURN RANGING  # Default

────────────────────────────────────────────────────────────
THE LOOP (steady-state behavior)
────────────────────────────────────────────────────────────

LOOP every ~60 seconds (integrated with BRAIN cycle):

    ┌──────────────────────────────────────────────────────────┐
    │  PHASE 1: CANDIDATE SCREENING                            │
    │    Runs periodically (every N cycles or on new data)     │
    │    Reads VDJ memory cells + TE domestication records     │
    │    Scores compatibility, filters by receptor match       │
    │    Stores top candidates in syncytin_db                  │
    └───────────────┬──────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────────────────────────┐
    │  PHASE 2: FUSION                                         │
    │    For each compatible candidate not yet fused:           │
    │    Determine fusion type (regime_switch/blend/cascade)    │
    │    Build envelope protein (shared risk from config)       │
    │    Build compartments (immune barrier)                    │
    │    Create HybridOrganism and store                        │
    └───────────────┬──────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────────────────────────┐
    │  PHASE 3: SIGNAL ROUTING (every cycle)                   │
    │    Detect current market regime                           │
    │    Route to appropriate sub-strategy                      │
    │    Apply envelope checks (position limit, drawdown)       │
    │    Apply compartment checks (suspension)                  │
    │    Emit signal for BRAIN consumption                      │
    └───────────────┬──────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────────────────────────┐
    │  PHASE 4: OUTCOME RECORDING                              │
    │    Record PnL to appropriate compartment                  │
    │    Check immune barrier (suspend if DD too high)          │
    │    Execute nutrient exchange (equity sharing)             │
    │    Shadow-execute both parents for alpha tracking         │
    └───────────────┬──────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────────────────────────┐
    │  PHASE 5: FITNESS MONITORING (every N trades)            │
    │    Compare hybrid returns to solo parent returns          │
    │    Compute fusion alpha                                   │
    │    Status: active / probation / defused                   │
    │    Defuse unviable hybrids                                │
    └──────────────────────────────────────────────────────────┘

INVARIANT:
    "Strategies that complement each other are fused into hybrid organisms.
     The hybrid inherits both nuclei (signal logic) but shares one membrane
     (risk management). Fusion must produce alpha over either parent alone.
     If it does not, the organism is dissolved."

BIOLOGICAL PARALLEL:
    Strategy A / B          = trophoblast cells with distinct nuclei
    compatibility_score     = ASCT2 receptor expression (can they fuse?)
    return_correlation      = antigenic compatibility (too similar = no benefit)
    EnvelopeProtein         = syncytin fusogenic domain (shared risk membrane)
    RiskCompartment         = nuclear envelope (protects each nucleus)
    regime routing          = syncytiotrophoblast nutrient transport selectivity
    equity_sharing          = placental nutrient exchange
    compartment suspension  = immune barrier (maternal T-cell exclusion)
    fusion_alpha            = fetal viability (is the pregnancy successful?)
    defusion                = spontaneous abortion (non-viable fusion)
    multiple active fusions = convergent domestication (different syncytins)

CROSS-ALGORITHM INTEGRATION:
    Algorithm #1 (VDJ Recombination):
        VDJ memory B cells serve as FUSION CANDIDATES. Each memory cell
        is a proven micro-strategy that can be fused with a complementary
        one. The antibody's regime affinity data feeds the compatibility
        scoring.

    Algorithm #2 (TE Domestication):
        Domesticated TE patterns provide FITNESS DATA for screening.
        A strategy built on heavily domesticated patterns (high boost)
        is a stronger fusion candidate than one on weakly domesticated
        patterns.

    Algorithm #3 (CRISPR-Cas):
        CRISPR spacers can block trades even within a hybrid. If the
        current conditions match a known losing fingerprint, the CRISPR
        system overrides the hybrid's signal -- the immune memory is
        respected even inside the fused organism.

    Algorithm #4 (Electric Organs):
        Convergent patterns across instruments strengthen fusion candidates.
        If Strategy A's winning pattern was convergently domesticated across
        3+ instruments, it is a higher-quality fusion candidate.

    Algorithm #5 (Protective Deletion):
        Protective deletion suppresses toxic sub-strategies. If one
        nucleus of the hybrid relies on a deleted (suppressed) pattern,
        that compartment is weakened accordingly.

    Algorithm #6 (KoRV):
        New signal types onboarded via KoRV can become fusion candidates
        once they pass through staged integration and reach "domesticated"
        status. The KoRV pipeline feeds new strategies into the syncytin
        fusion pool.

DATABASES:
    syncytin_fusions.db      -- hybrid organisms, candidates, trades, fitness
    vdj_antibodies.db        -- memory B cells (fusion candidate source)
    teqa_domestication.db    -- TE pattern fitness (screening data)

FILES:
    syncytin.py              -- SyncytinFusionEngine class (this algorithm)
    vdj_recombination.py     -- VDJ engine (provides memory cells)
    ALGORITHM_SYNCYTIN.py    -- This pseudocode specification
    config_loader.py         -- Source of truth for all trading values
"""
