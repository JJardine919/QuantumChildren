"""
ALGORITHM: Protective_Deletion -- "Loss-of-Function Defense"
=============================================================
Closed-loop suppression of toxic transposable element trading patterns.

Biological basis:
    - CCR5-delta32: A 32-nucleotide deletion in the CCR5 chemokine receptor
      gene provides resistance to HIV-1 infection. Heterozygous carriers
      (one deleted copy) get partial resistance with delayed disease
      progression. Homozygous carriers (two deleted copies) get near-complete
      resistance -- the virus cannot bind because the receptor is gone.

    - CASP12 pseudogene: Loss-of-function mutation in Caspase-12 became
      fixed in most human populations because it reduces sepsis susceptibility.
      The "broken" gene is actually protective.

    - Myostatin knockout: Loss of the myostatin gene removes the brake on
      muscle growth. The deletion of a suppressor IS the gain.

    - PCSK9 loss-of-function: Carriers of PCSK9 LoF variants have
      dramatically lower LDL cholesterol and 88% reduced coronary heart
      disease risk. Removing the receptor removes the attack vector.

    Principle: Sometimes removing a receptor from the cell surface is the
    strongest possible defense. The cell becomes immune to attack because
    the attack vector no longer exists.

Trading analogy:
    - Some TE signal combinations are TOXIC -- they consistently precede
      losing trades. They are attack vectors against the account.
    - Instead of leaving them at neutral (1.0x boost), actively SUPPRESS
      them with a factor < 1.0x.
    - "Delete the receptor" = remove the signal's ability to influence
      trading decisions.
    - Heterozygous deletion (partial evidence, 15-25 trades):
          suppression_factor = 0.5 (partial resistance)
    - Homozygous deletion (strong evidence, 25+ trades):
          suppression_factor = 0.1 (near-complete resistance)
    - This is the INVERSE of TE domestication. Domestication boosts
      winning combos above 1.0x. Protective deletion suppresses losing
      combos below 1.0x. Together they form a complete learning system.

    - Drawdown pressure: Under account stress (drawdown), the detection
      thresholds drop. This mirrors how selective pressure intensifies in
      hostile environments -- organisms under stress evolve faster because
      the cost of NOT adapting is death.

    - Hysteresis: Once flagged as toxic, a pattern only recovers when its
      loss rate drops below 50%. This prevents oscillation at the boundary.

    - Heterozygote advantage tracking: In some market regimes, partial
      suppression (0.5x) outperforms full deletion (0.1x). This is
      analogous to sickle-cell trait: heterozygous carriers have an
      advantage over both homozygous states. The system tracks this.

Implementation: Python (SQLite persistence, extends domestication DB)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: PROTECTIVE-DELETION-1.0

============================================================
PSEUDOCODE
============================================================
"""

# ----------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------

# Toxic pattern detection thresholds
DELETION_HETEROZYGOUS_MIN_TRADES   = 15     # Stage 1: partial evidence required
DELETION_HETEROZYGOUS_MIN_LR       = 0.65   # Stage 1: loss rate >= 65%
DELETION_HETEROZYGOUS_FACTOR       = 0.50   # Stage 1: suppression multiplier

DELETION_HOMOZYGOUS_MIN_TRADES     = 25     # Stage 2: strong evidence required
DELETION_HOMOZYGOUS_MIN_LR         = 0.70   # Stage 2: loss rate >= 70%
DELETION_HOMOZYGOUS_FACTOR         = 0.10   # Stage 2: near-complete suppression

# Hysteresis: recovery threshold
DELETION_RECOVERY_LR               = 0.50   # Unflag only when loss rate < 50%

# Drawdown pressure (adaptive thresholds under stress)
DRAWDOWN_PRESSURE_MILD             = 0.03   # 3% drawdown: mild pressure
DRAWDOWN_PRESSURE_MODERATE         = 0.05   # 5% drawdown: moderate pressure
DRAWDOWN_PRESSURE_SEVERE           = 0.08   # 8% drawdown: severe pressure
DRAWDOWN_LR_REDUCTION_MILD         = 0.03   # Lower LR threshold by 3%
DRAWDOWN_LR_REDUCTION_MODERATE     = 0.05   # Lower LR threshold by 5%
DRAWDOWN_LR_REDUCTION_SEVERE       = 0.08   # Lower LR threshold by 8%
DRAWDOWN_TRADES_REDUCTION_MILD     = 2      # Reduce min trades by 2
DRAWDOWN_TRADES_REDUCTION_MODERATE = 4      # Reduce min trades by 4
DRAWDOWN_TRADES_REDUCTION_SEVERE   = 6      # Reduce min trades by 6

# Allele frequency monitoring
ALLELE_FREQ_WARNING_THRESHOLD      = 0.30   # Warn if >30% of patterns are suppressed
ALLELE_FREQ_CRITICAL_THRESHOLD     = 0.50   # Critical: over half the patterns are toxic

# Quantum integration
QUANTUM_SUPPRESSION_ANGLE_SCALE    = 0.3    # Scale factor for reduced qubit rotation

# Bayesian prior for loss rate estimation (mirrors domestication)
DELETION_PRIOR_ALPHA               = 10     # Beta prior alpha (losses)
DELETION_PRIOR_BETA                = 10     # Beta prior beta (wins)

# Heterozygote advantage tracking window
HET_ADVANTAGE_EVAL_TRADES          = 50     # Evaluate het vs hom after 50 trades
HET_ADVANTAGE_TRACKING_WINDOW      = 30     # Days to track advantage

# Expiry: re-evaluate suppressed patterns after inactivity
DELETION_EXPIRY_DAYS               = 45     # Longer than domestication (deletion is cautious)

"""
============================================================
ALGORITHM Protective_Deletion
============================================================

DEFINE ToxicPattern AS:
    pattern_hash        : TEXT PRIMARY KEY (same as domestication)
    te_combo            : TEXT (sorted TE names joined by "+")
    win_count           : INTEGER
    loss_count          : INTEGER
    loss_rate           : REAL (loss_count / total)
    posterior_lr        : REAL (Bayesian posterior loss rate)
    deletion_stage      : TEXT ("none", "heterozygous", "homozygous")
    suppression_factor  : REAL (1.0 = neutral, 0.5 = het, 0.1 = hom)
    flagged_at          : TIMESTAMP (when first flagged as toxic)
    recovered_at        : TIMESTAMP (when unflagged, NULL if still toxic)
    first_seen          : TIMESTAMP
    last_seen           : TIMESTAMP
    total_pnl           : REAL (cumulative P/L when this pattern fired)
    avg_loss_magnitude  : REAL (average size of losses)
    drawdown_at_flag    : REAL (account drawdown when flagged)
    het_advantage_score : REAL (tracks if partial > full suppression)

STORAGE:
    deletion_db         : SQLite "teqa_protective_deletion.db"
    Uses SAME signal_history_db as domestication for deal matching
    processed_tickets   : shared with domestication (dedup)

------------------------------------------------------------
PHASE 1: TOXIC PATTERN DETECTION (runs on every trade outcome)
------------------------------------------------------------

ON record_outcome(active_tes[], won, profit, account_drawdown_pct):

    combo = sorted(active_tes).join("+")
    hash  = MD5(combo)[:16]

    row = deletion_db.SELECT WHERE pattern_hash = hash

    IF row EXISTS:
        row.win_count  += 1 IF won ELSE 0
        row.loss_count += 0 IF won ELSE 1
        total = row.win_count + row.loss_count
        row.loss_rate = row.loss_count / total

        # Bayesian posterior loss rate (mirrors domestication posterior WR)
        # Beta(alpha + losses, beta + wins) -> posterior mean of loss rate
        row.posterior_lr = (DELETION_PRIOR_ALPHA + row.loss_count) /
                           (DELETION_PRIOR_ALPHA + DELETION_PRIOR_BETA + total)

        # Accumulate P/L tracking
        row.total_pnl += profit
        IF NOT won:
            row.avg_loss_magnitude = running_average(row.avg_loss_magnitude, abs(profit))

    ELSE:  # First time seeing this TE combination
        deletion_db.INSERT(
            pattern_hash = hash,
            te_combo     = combo,
            win_count    = 1 IF won ELSE 0,
            loss_count   = 0 IF won ELSE 1,
            loss_rate    = 0.0 IF won ELSE 1.0,
            posterior_lr = (DELETION_PRIOR_ALPHA + (0 IF won ELSE 1)) /
                           (DELETION_PRIOR_ALPHA + DELETION_PRIOR_BETA + 1),
            deletion_stage      = "none",
            suppression_factor  = 1.0,
            first_seen          = NOW,
            last_seen           = NOW,
            total_pnl           = profit,
        )
        RETURN  # Not enough data to evaluate yet

------------------------------------------------------------
PHASE 2: DELETION CLASSIFICATION (two-stage with hysteresis)
------------------------------------------------------------

    # Compute adaptive thresholds based on drawdown pressure
    drawdown_lr_offset = compute_drawdown_pressure(account_drawdown_pct)
    effective_het_lr  = DELETION_HETEROZYGOUS_MIN_LR - drawdown_lr_offset.lr_reduction
    effective_hom_lr  = DELETION_HOMOZYGOUS_MIN_LR - drawdown_lr_offset.lr_reduction
    effective_het_trades = max(8, DELETION_HETEROZYGOUS_MIN_TRADES - drawdown_lr_offset.trades_reduction)
    effective_hom_trades = max(15, DELETION_HOMOZYGOUS_MIN_TRADES - drawdown_lr_offset.trades_reduction)

    previous_stage = row.deletion_stage

    # HYSTERESIS: Different thresholds for flagging vs unflagging
    IF previous_stage == "homozygous":
        # Already fully suppressed: only recover if loss rate drops below 50%
        IF row.posterior_lr < DELETION_RECOVERY_LR:
            row.deletion_stage = "none"
            row.suppression_factor = 1.0
            row.recovered_at = NOW
        # Stays homozygous otherwise (hysteresis prevents oscillation)

    ELIF previous_stage == "heterozygous":
        # Partially suppressed: can escalate to homozygous or recover
        IF row.posterior_lr < DELETION_RECOVERY_LR:
            row.deletion_stage = "none"
            row.suppression_factor = 1.0
            row.recovered_at = NOW
        ELIF total >= effective_hom_trades AND row.posterior_lr >= effective_hom_lr:
            row.deletion_stage = "homozygous"
            row.suppression_factor = DELETION_HOMOZYGOUS_FACTOR
            # Check heterozygote advantage before full deletion
            IF het_advantage_detected(row):
                row.suppression_factor = DELETION_HETEROZYGOUS_FACTOR  # Stay partial
                row.het_advantage_score += 1

    ELSE:  # deletion_stage == "none"
        # Not yet flagged: evaluate for initial flagging
        IF total >= effective_hom_trades AND row.posterior_lr >= effective_hom_lr:
            # Strong evidence: skip straight to homozygous
            row.deletion_stage = "homozygous"
            row.suppression_factor = DELETION_HOMOZYGOUS_FACTOR
            row.flagged_at = NOW
            row.drawdown_at_flag = account_drawdown_pct
        ELIF total >= effective_het_trades AND row.posterior_lr >= effective_het_lr:
            # Partial evidence: heterozygous deletion
            row.deletion_stage = "heterozygous"
            row.suppression_factor = DELETION_HETEROZYGOUS_FACTOR
            row.flagged_at = NOW
            row.drawdown_at_flag = account_drawdown_pct

    row.last_seen = NOW
    deletion_db.UPDATE row

------------------------------------------------------------
PHASE 3: SUPPRESSION APPLICATION (called during signal generation)
------------------------------------------------------------

FUNCTION get_suppression(active_tes[]) -> float:

    combo = sorted(active_tes).join("+")
    hash  = MD5(combo)[:16]

    row = deletion_db.SELECT WHERE pattern_hash = hash
    IF row IS NULL: RETURN 1.0  # Unknown pattern, no suppression

    IF row.deletion_stage == "none": RETURN 1.0

    # Check expiry: don't suppress patterns that haven't been seen recently
    IF row.last_seen older than DELETION_EXPIRY_DAYS:
        RETURN 1.0  # Pattern may have changed, give it another chance

    RETURN row.suppression_factor  # 0.5 or 0.1

------------------------------------------------------------
PHASE 4: COMBINED SIGNAL MODIFICATION
      (works alongside domestication boost)
------------------------------------------------------------

ON teqa_cycle(bars, symbol):

    # ... existing TE activation + quantum circuit + neural consensus ...

    active_tes = [a.te FOR a IN activations WHERE a.strength > 0.3]

    # DOMESTICATION (existing): patterns that WIN get boosted
    boost = domestication_db.GET_BOOST(active_tes)  # >= 1.0

    # PROTECTIVE DELETION (new): patterns that LOSE get suppressed
    suppression = deletion_db.GET_SUPPRESSION(active_tes)  # <= 1.0

    # Combined signal modifier: boost * suppression
    # If domesticated:  1.25 * 1.0 = 1.25 (boosted)
    # If neutral:       1.0  * 1.0 = 1.00 (unchanged)
    # If het-deleted:   1.0  * 0.5 = 0.50 (partially suppressed)
    # If hom-deleted:   1.0  * 0.1 = 0.10 (nearly muted)
    # If BOTH boosted and suppressed (conflict): boost * suppression
    #   e.g., 1.25 * 0.5 = 0.625 (suppression wins -- safety first)
    combined_modifier = boost * suppression

    # Apply to confidence
    confidence = concordance * 0.3
                + consensus  * 0.3
                + min(0.4, (combined_modifier - 1.0))

    # QUANTUM INTEGRATION: Suppressed patterns get reduced qubit angles
    # Domesticated patterns get AMPLIFIED rotation (existing)
    # Deleted patterns get DAMPENED rotation (new)
    # qubit_angle *= suppression_factor (e.g., 0.1 = 90% reduction)
    IF suppression < 1.0:
        FOR each TE in active_tes:
            qubit_rotation_angle[TE.qubit_index] *= suppression * QUANTUM_SUPPRESSION_ANGLE_SCALE

------------------------------------------------------------
PHASE 5: DRAWDOWN PRESSURE ENGINE
         (adaptive thresholds under account stress)
------------------------------------------------------------

FUNCTION compute_drawdown_pressure(drawdown_pct) -> PressureResult:

    # Drawdown = (peak_equity - current_equity) / peak_equity
    # Under stress, the system becomes more aggressive at deleting
    # toxic patterns. This mirrors increased selective pressure in
    # hostile environments.

    IF drawdown_pct >= DRAWDOWN_PRESSURE_SEVERE:
        RETURN PressureResult(
            lr_reduction = DRAWDOWN_LR_REDUCTION_SEVERE,
            trades_reduction = DRAWDOWN_TRADES_REDUCTION_SEVERE,
            pressure_level = "SEVERE"
        )
    ELIF drawdown_pct >= DRAWDOWN_PRESSURE_MODERATE:
        RETURN PressureResult(
            lr_reduction = DRAWDOWN_LR_REDUCTION_MODERATE,
            trades_reduction = DRAWDOWN_TRADES_REDUCTION_MODERATE,
            pressure_level = "MODERATE"
        )
    ELIF drawdown_pct >= DRAWDOWN_PRESSURE_MILD:
        RETURN PressureResult(
            lr_reduction = DRAWDOWN_LR_REDUCTION_MILD,
            trades_reduction = DRAWDOWN_TRADES_REDUCTION_MILD,
            pressure_level = "MILD"
        )
    ELSE:
        RETURN PressureResult(
            lr_reduction = 0.0,
            trades_reduction = 0,
            pressure_level = "NONE"
        )

    # Example at 6% drawdown (MODERATE pressure):
    #   effective_het_lr  = 0.65 - 0.05 = 0.60 (lower threshold)
    #   effective_hom_lr  = 0.70 - 0.05 = 0.65
    #   effective_het_trades = max(8, 15 - 4) = 11
    #   effective_hom_trades = max(15, 25 - 4) = 21
    #   Patterns get flagged FASTER under drawdown pressure

------------------------------------------------------------
PHASE 6: ALLELE FREQUENCY MONITORING
         (population-level health metric)
------------------------------------------------------------

FUNCTION compute_allele_frequency() -> AlleleFrequencyReport:

    total_patterns = COUNT(*) FROM deletion_db
    het_patterns   = COUNT(*) WHERE deletion_stage = "heterozygous"
    hom_patterns   = COUNT(*) WHERE deletion_stage = "homozygous"

    suppressed = het_patterns + hom_patterns
    allele_freq = suppressed / total_patterns IF total_patterns > 0 ELSE 0

    # Health assessment
    IF allele_freq > ALLELE_FREQ_CRITICAL_THRESHOLD:
        health = "CRITICAL"
        # Over half the genome is deleted -- the system may be
        # suppressing too aggressively, or the market regime has
        # fundamentally shifted. Consider resetting.
    ELIF allele_freq > ALLELE_FREQ_WARNING_THRESHOLD:
        health = "WARNING"
        # Many patterns are toxic. The system is under stress.
    ELSE:
        health = "HEALTHY"

    RETURN AlleleFrequencyReport(
        total_patterns, het_patterns, hom_patterns,
        allele_freq, health,
        avg_suppression = mean(suppression_factor) across all suppressed,
        worst_pattern = pattern with highest posterior_lr
    )

------------------------------------------------------------
PHASE 7: HETEROZYGOTE ADVANTAGE DETECTION
         (partial suppression can outperform full deletion)
------------------------------------------------------------

FUNCTION het_advantage_detected(row) -> bool:

    # In some market regimes, partial suppression (het = 0.5x) produces
    # better outcomes than full deletion (hom = 0.1x). This is analogous
    # to sickle-cell trait: heterozygous carriers of HbS have an
    # advantage over both normal (malaria susceptible) and homozygous
    # (sickle-cell disease) individuals.
    #
    # Mechanism: A fully deleted signal provides ZERO information.
    # A partially suppressed signal still provides SOME information,
    # which can be useful in transitional market regimes.
    #
    # Detection: Track P/L during het vs hom periods.
    # If het-period P/L > hom-period P/L over sufficient trades,
    # maintain partial suppression instead of escalating to full.

    IF row.het_period_pnl IS NULL OR row.het_period_trades < HET_ADVANTAGE_EVAL_TRADES:
        RETURN FALSE  # Not enough data

    het_avg_pnl = row.het_period_pnl / row.het_period_trades
    hom_avg_pnl = row.hom_period_pnl / row.hom_period_trades IF row.hom_period_trades > 0 ELSE -inf

    RETURN het_avg_pnl > hom_avg_pnl

------------------------------------------------------------
THE LOOP (steady-state behavior)
------------------------------------------------------------

LOOP every ~60 seconds:

    +-----------------------------------------------------+
    |  TEQA v3.0 analyze()                                |
    |    TE activations -> quantum circuit -> neural vote  |
    |    -> get_boost(active_tes)  <-----------+          |
    |    -> get_suppression(active_tes) <------+---+      |
    |    -> combined = boost * suppression      |   |      |
    |    -> emit signal JSON + log to history   |   |      |
    +--------------+----------------------------+   |      |
                   |                                 |      |
    +--------------v----------------------------+   |      |
    |  BRAIN run_cycle()                        |   |      |
    |    read signal -> combine with LSTM       |   |      |
    |    -> execute trade (mt5.order_send)      |   |      |
    |    -> feedback_poll() --+                 |   |      |
    +-------------------------+-----+           |   |      |
                              |     |           |   |      |
    +-------------------------v-----+--------+  |   |      |
    |  TradeOutcomePoller                    |  |   |      |
    |    mt5.history_deals_get() -> filter   |  |   |      |
    |    -> match deal to signal             |  |   |      |
    |    -> extract active_tes               |  |   |      |
    |    -> record_pattern(tes, won, profit) |  |   |      |
    +--------+-------+----------------------+  |   |      |
             |       |                          |   |      |
    +--------v-------v--+    +---------v--------v--+      |
    | TEDomesticationTrk |    | ProtectiveDeletion  |      |
    | win_rate >= 0.70   |    | loss_rate >= 0.65   |      |
    | boost >= 1.0       |    | suppression <= 1.0  |      |
    | "amplify winners"  |    | "delete losers"     |      |
    | stored in dom.db --+    | stored in del.db ---+      |
    +--------------------+    +---------------------+      |
                                                           |
    +------------------------------------------------------+
    |  DrawdownPressure                                    |
    |    Monitors account equity curve                     |
    |    Under stress: lower thresholds (faster deletion)  |
    |    Reports pressure level to deletion engine         |
    +------------------------------------------------------+
    |
    +------------------------------------------------------+
    |  AlleleFrequencyMonitor                              |
    |    Tracks % of patterns that are suppressed          |
    |    HEALTHY < 30% | WARNING 30-50% | CRITICAL > 50%  |
    |    Alerts if too many patterns are toxic             |
    +------------------------------------------------------+

INVARIANT:
    "TE patterns that precede loss are suppressed.
     The signal's ability to influence decisions is DELETED.
     The receptor is removed. The attack vector is gone."

BIOLOGICAL PARALLEL:
    active_tes[]           = cell surface receptors
    pattern_hash           = receptor gene locus
    loss_rate              = disease susceptibility
    deletion_stage         = allele state (wild-type / het / hom)
    suppression_factor     = receptor expression level (0.1 = nearly gone)
    "heterozygous"         = one copy deleted (partial resistance)
    "homozygous"           = both copies deleted (near-complete resistance)
    drawdown_pressure      = environmental selective pressure
    allele_frequency       = population-level deletion frequency
    het_advantage          = overdominance (sickle-cell analogy)
    recovery               = reversion mutation (rare, requires evidence)
    RECORD_OUTCOME()       = pathogen exposure (selection event)
    GET_SUPPRESSION()      = receptor surface check

CONVERGENCE:
    After ~15 trades per TE combo, the system identifies which
    combinations of L1_Neuronal + Alu_Exonization + HERV_Synapse + ...
    consistently predict LOSING market moves. These get their signal
    strength slashed: first to 50% (heterozygous), then to 10%
    (homozygous). The deletion DB becomes a learned immune defense
    that blocks toxic signal pathways before they cause losses.

    Combined with domestication (which amplifies winners), the full
    system now has both POSITIVE selection (boost winners) and
    NEGATIVE selection (suppress losers). This dual selection
    pressure is how real genomes evolve: gain-of-function AND
    loss-of-function mutations both contribute to fitness.

DATABASES:
    teqa_protective_deletion.db  <- toxic pattern tracking (persistent)
    teqa_domestication.db        <- winning pattern tracking (existing)
    teqa_signal_history.db       <- signal emission log (shared)
    MT5 deal history             <- trade outcomes (profit/loss)

FILES:
    protective_deletion.py       -> ProtectiveDeletionTracker class
                                    ToxicPatternDetector
                                    SuppressionEngine
                                    DrawdownPressure
                                    AlleleFrequencyMonitor
    teqa_v3_neural_te.py         -> TEDomesticationTracker (existing)
    teqa_feedback.py             -> TradeOutcomePoller (existing, extended)
    BRAIN_ATLAS.py               -> trades + polls feedback each cycle
"""
