"""
ALGORITHM #4: Electric Organs -- Convergent Signal Evolution
=============================================================
Cross-instrument convergent pattern detection for universal edge discovery.

Biological basis:
    Electric organs evolved INDEPENDENTLY at least 6 times in unrelated
    fish lineages:
      1. Electric eels (Gymnotiformes) -- South America
      2. Torpedo rays (Torpediniformes) -- Worldwide
      3. Electric catfish (Malapteruridae) -- Africa
      4. Elephantfish (Mormyridae) -- Africa
      5. Stargazers (Uranoscopidae) -- Worldwide
      6. Skates (Rajiformes) -- Worldwide

    Every single time, the SAME solution emerged:
      - Muscle cells transform into ELECTROCYTES
      - Sodium channel genes (SCN) get UPREGULATED
      - Contractile genes (actin/myosin) get DOWNREGULATED
      - Transposable elements played a role in the regulatory rewiring

    This is CONVERGENT EVOLUTION at its purest: when the same optimal
    solution independently arises in unrelated organisms, it is strong
    evidence that this solution is a genuine adaptation -- not noise, not
    drift, not phylogenetic inheritance. It is physics finding the same
    answer to the same problem.

Trading analogy -- Convergent Signal Discovery:
    - Each instrument (XAUUSD, BTCUSD, ETHUSD, NAS100) is an independent
      "species" with its own evolutionary history (domestication DB)
    - If 3 or 4 INDEPENDENT instruments all domesticate the SAME TE
      pattern combo, that pattern is probably not overfitting -- it is
      a UNIVERSAL market edge, like a law of physics
    - Single-instrument domestication might be curve-fitting
    - Cross-instrument convergent domestication is REAL SIGNAL

    Convergent patterns get a SUPER-BOOST on top of individual
    domestication, because they carry cross-validated evidence.

Integration:
    - Reads domestication DBs from multiple instruments
    - Identifies TE combos independently domesticated in 3+ instruments
    - Computes convergence score and electrocyte status
    - Applies sodium channel amplification (super-boost) or
      contractile suppression (super-suppress) to TEQA pipeline
    - Persists convergence findings in SQLite for incremental learning
    - Verifies independence via rolling correlation checks

Authors: DooDoo + Claude
Date:    2026-02-08
Version: ELECTRIC-ORGANS-1.0

============================================================
PSEUDOCODE
============================================================
"""

# ──────────────────────────────────────────────────────────
# CONSTANTS (no hardcoded trading values -- config_loader)
# ──────────────────────────────────────────────────────────

# Convergence thresholds
CONVERGENCE_MIN_INSTRUMENTS    = 3     # Minimum instruments sharing pattern
CONVERGENCE_ELECTROCYTE_THRESH = 0.60  # Ratio to achieve electrocyte status
CONVERGENCE_SCAN_INTERVAL_SEC  = 300   # 5 min between convergence scans

# Super-boost / super-suppress multipliers
SODIUM_CHANNEL_BOOST           = 1.50  # 1.5x on top of domestication boost
CONTRACTILE_SUPPRESS           = 0.30  # 0.3x for convergent losers

# Independence verification
MIN_INDEPENDENCE_THRESHOLD     = 0.40  # Max rolling correlation for "independent"
INDEPENDENCE_WINDOW_BARS       = 100   # Window for rolling correlation
INDEPENDENCE_CHECK_INTERVAL    = 3600  # Re-verify independence every hour

# Pattern staleness
CONVERGENCE_EXPIRY_DAYS        = 60    # Re-evaluate convergence after 60 days

# Instruments considered as independent lineages
DEFAULT_LINEAGES = ["XAUUSD", "BTCUSD", "ETHUSD", "NAS100"]

"""
============================================================
ALGORITHM Electric_Organs -- Convergent Signal Evolution
============================================================

DEFINE Lineage AS:
    symbol            : TEXT   (e.g., "BTCUSD")
    domestication_db  : PATH   (e.g., "teqa_domestication_BTCUSD.db")
    domesticated_set  : SET of pattern_hashes currently domesticated
    is_active         : BOOL   (has recent data)

DEFINE ConvergentPattern AS:
    pattern_hash      : TEXT PRIMARY KEY
    te_combo          : TEXT   (e.g., "Alu+CACTA+L1_Neuronal")
    n_instruments     : INTEGER (how many lineages domesticated this)
    n_observed        : INTEGER (how many lineages have seen this at all)
    convergence_score : REAL   (n_instruments / n_observed)
    is_electrocyte    : BOOL   (convergence_score >= 0.60)
    super_boost       : REAL   (1.50 if electrocyte winner, 0.30 if loser)
    avg_win_rate      : REAL   (mean posterior WR across instruments)
    avg_profit_factor : REAL   (mean PF across instruments)
    first_detected    : TIMESTAMP
    last_verified     : TIMESTAMP
    independence_ok   : BOOL   (verified non-correlated)
    lineages_present  : TEXT   (JSON list of symbols that domesticated this)

STORAGE:
    convergence_db    : SQLite "electric_organs_convergence.db"
    per-instrument    : reads existing "teqa_domestication_{symbol}.db" files
    signal_output     : "electric_organs_signal.json"

────────────────────────────────────────────────────────────
PHASE 1: LINEAGE DISCOVERY (startup + periodic)
────────────────────────────────────────────────────────────

ON startup():

    lineages = []
    FOR symbol IN configured_symbols:
        db_path = find_domestication_db(symbol)
        #   Search order:
        #     1. teqa_domestication_{symbol}.db
        #     2. teqa_domestication.db (shared DB with symbol column)
        IF db_path EXISTS:
            lineages.APPEND(Lineage(symbol, db_path))
            LOG("Found lineage: {symbol} -> {db_path}")
        ELSE:
            LOG("No domestication DB for {symbol}, skipping")

    IF len(lineages) < 2:
        WARN("Need >= 2 lineages for convergence detection")
        RETURN

    LOG("Electric Organs initialized with {len(lineages)} lineages")

────────────────────────────────────────────────────────────
PHASE 2: CONVERGENCE SCAN (runs every 5 minutes)
────────────────────────────────────────────────────────────

ON convergence_scan():

    # Step 1: Harvest domesticated patterns from each lineage
    all_domesticated = {}   # pattern_hash -> {symbols: [], win_rates: [], ...}
    all_observed = {}       # pattern_hash -> set of symbols that have seen it

    FOR lineage IN lineages:
        domesticated, observed = harvest_lineage(lineage)
        #
        # harvest_lineage():
        #   conn = sqlite3.connect(lineage.domestication_db)
        #   domesticated_rows = SELECT pattern_hash, te_combo, posterior_wr,
        #                              profit_factor, win_count, loss_count
        #                       FROM domesticated_patterns
        #                       WHERE domesticated = 1
        #   observed_rows = SELECT DISTINCT pattern_hash
        #                   FROM domesticated_patterns
        #                   WHERE (win_count + loss_count) >= 5
        #
        FOR row IN domesticated_rows:
            all_domesticated[row.hash].symbols.ADD(lineage.symbol)
            all_domesticated[row.hash].win_rates.APPEND(row.posterior_wr)
            all_domesticated[row.hash].profit_factors.APPEND(row.profit_factor)
            all_domesticated[row.hash].te_combo = row.te_combo

        FOR row IN observed_rows:
            all_observed[row.hash].ADD(lineage.symbol)

    # Step 2: Identify convergent patterns
    convergent_patterns = []

    FOR hash, data IN all_domesticated.items():
        n_instruments = len(data.symbols)
        n_observed = len(all_observed.get(hash, data.symbols))
        convergence_score = n_instruments / max(1, n_observed)

        IF n_instruments >= CONVERGENCE_MIN_INSTRUMENTS:
            # This TE combo was independently domesticated in 3+ instruments
            avg_wr = mean(data.win_rates)
            avg_pf = mean(data.profit_factors)

            is_electrocyte = (convergence_score >= CONVERGENCE_ELECTROCYTE_THRESH)

            # Determine super-boost direction
            IF is_electrocyte AND avg_wr >= 0.65 AND avg_pf >= 1.5:
                # SODIUM CHANNEL AMPLIFICATION
                # Winner pattern: super-boost
                super_boost = SODIUM_CHANNEL_BOOST  # 1.5x
            ELIF is_electrocyte AND avg_wr < 0.45:
                # CONTRACTILE SUPPRESSION
                # Convergent loser: super-suppress
                super_boost = CONTRACTILE_SUPPRESS  # 0.3x
            ELSE:
                # Convergent but ambiguous: mild boost
                super_boost = 1.0 + 0.25 * (convergence_score - 0.5)

            pattern = ConvergentPattern(
                pattern_hash = hash,
                te_combo = data.te_combo,
                n_instruments = n_instruments,
                n_observed = n_observed,
                convergence_score = convergence_score,
                is_electrocyte = is_electrocyte,
                super_boost = super_boost,
                avg_win_rate = avg_wr,
                avg_profit_factor = avg_pf,
                lineages_present = data.symbols,
            )

            convergent_patterns.APPEND(pattern)

    # Step 3: Persist to convergence DB
    FOR pattern IN convergent_patterns:
        convergence_db.UPSERT(pattern)

    LOG("Convergence scan: found {len(convergent_patterns)} convergent "
        "patterns from {len(lineages)} lineages")

    RETURN convergent_patterns

────────────────────────────────────────────────────────────
PHASE 3: INDEPENDENCE VERIFICATION
────────────────────────────────────────────────────────────

ON verify_independence(convergent_patterns, bars_dict):
    # bars_dict: {symbol: np.ndarray} -- recent bars per instrument

    # The biological principle: convergent evolution is only meaningful
    # if the lineages are INDEPENDENT. If electric eels and torpedo rays
    # were actually the same species, their shared solution would mean nothing.
    #
    # Similarly, if BTCUSD and ETHUSD are 95% correlated, a shared
    # domesticated pattern might just be the same signal duplicated.

    FOR pattern IN convergent_patterns:
        symbols = pattern.lineages_present
        pairs = all_pairs(symbols)
        max_corr = 0.0

        FOR (s1, s2) IN pairs:
            IF s1 IN bars_dict AND s2 IN bars_dict:
                bars1 = bars_dict[s1]
                bars2 = bars_dict[s2]

                # Compute rolling correlation of returns
                returns1 = diff(log(bars1.close))
                returns2 = diff(log(bars2.close))

                # Align by time if needed, then correlate
                corr = rolling_correlation(
                    returns1[-INDEPENDENCE_WINDOW_BARS:],
                    returns2[-INDEPENDENCE_WINDOW_BARS:]
                )

                max_corr = max(max_corr, abs(corr))

        # Independence check
        pattern.independence_ok = (max_corr < MIN_INDEPENDENCE_THRESHOLD)

        IF NOT pattern.independence_ok:
            # Reduce confidence in this convergence
            # Not completely zero -- correlated instruments CAN still
            # provide convergent evidence, just weaker evidence
            pattern.super_boost = 1.0 + (pattern.super_boost - 1.0) * 0.3
            LOG("Convergent pattern {pattern.te_combo} failed independence "
                "check (max_corr={max_corr:.2f}), reducing boost")

────────────────────────────────────────────────────────────
PHASE 4: SIGNAL APPLICATION (runs every TEQA cycle)
────────────────────────────────────────────────────────────

ON apply_convergence_boost(active_tes, symbol, domestication_boost):
    # Called from TEQA pipeline after domestication boost is computed.
    # This is the "electrocyte transformation" step.

    combo = sorted(active_tes).join("+")
    hash = MD5(combo)[:16]

    # Look up in convergence DB
    row = convergence_db.SELECT WHERE pattern_hash = hash

    IF row IS NULL:
        # Not a convergent pattern
        RETURN domestication_boost

    IF row.is_electrocyte AND row.independence_ok:
        # ELECTROCYTE TRANSFORMATION COMPLETE
        # This TE combo has been validated across independent instruments.
        # Apply super-boost on top of domestication boost.
        #
        # Biological parallel: the muscle cell has been fully transformed
        # into an electrocyte. Sodium channels are maximally expressed.
        # Contractile proteins are completely suppressed.

        final_boost = domestication_boost * row.super_boost

        LOG("[ELECTROCYTE] {combo} | convergence={row.convergence_score:.2f} | "
            "{row.n_instruments} instruments | boost={final_boost:.3f}")

        RETURN final_boost

    ELIF row.convergence_score >= 0.40:
        # Partial convergence -- some evidence but not full electrocyte
        # This is like an intermediate cell between muscle and electrocyte
        partial_factor = 1.0 + (row.super_boost - 1.0) * 0.5
        final_boost = domestication_boost * partial_factor

        LOG("[PARTIAL] {combo} | convergence={row.convergence_score:.2f} | "
            "partial_boost={final_boost:.3f}")

        RETURN final_boost

    ELSE:
        RETURN domestication_boost

────────────────────────────────────────────────────────────
PHASE 5: CONVERGENT LOSER SUPPRESSION
────────────────────────────────────────────────────────────

ON check_convergent_loser(active_tes, symbol):
    # Separate check for convergent LOSERS.
    # If a TE combo is NOT domesticated in any instrument -- and has been
    # OBSERVED in 3+ instruments and failed in all of them -- that is
    # convergent evidence of a BAD pattern.

    combo = sorted(active_tes).join("+")
    hash = MD5(combo)[:16]

    n_failed = 0
    n_observed = 0

    FOR lineage IN lineages:
        row = lineage.db.SELECT WHERE pattern_hash = hash
        IF row AND (row.win_count + row.loss_count) >= DOMESTICATION_MIN_TRADES:
            n_observed += 1
            IF row.domesticated == 0 AND row.posterior_wr < 0.45:
                n_failed += 1

    IF n_observed >= 3 AND n_failed >= 3:
        # CONTRACTILE SUPPRESSION
        # This pattern fails across multiple independent instruments.
        # It is the opposite of convergent evolution -- convergent extinction.
        # Suppress it hard.
        RETURN CONTRACTILE_SUPPRESS  # 0.3x

    RETURN 1.0  # No convergent suppression

────────────────────────────────────────────────────────────
THE LOOP (steady-state behavior)
────────────────────────────────────────────────────────────

LOOP:

    ┌─────────────────────────────────────────────────────────┐
    │  Every 5 min: CONVERGENCE SCAN                          │
    │    Read domestication DBs for each instrument           │
    │    Identify patterns domesticated in 3+ instruments     │
    │    Compute convergence score per pattern                 │
    │    Verify independence via correlation check             │
    │    Persist to electric_organs_convergence.db             │
    └──────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────────────┐
    │  Every TEQA cycle (~60s): APPLY BOOST                   │
    │    Get active_tes from current signal                    │
    │    Look up in convergence DB                             │
    │    IF electrocyte: apply SODIUM_CHANNEL_BOOST (1.5x)    │
    │    IF convergent loser: CONTRACTILE_SUPPRESS (0.3x)     │
    │    Combined boost = domestication * convergence          │
    │    Feed into confidence calculation                      │
    └──────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────────────┐
    │  Every hour: INDEPENDENCE RE-VERIFICATION               │
    │    Recompute rolling correlations                        │
    │    Downgrade patterns that lost independence             │
    │    Upgrade patterns that gained independence             │
    └─────────────────────────────────────────────────────────┘

INVARIANT:
    "TE patterns that independently evolve as winners in MULTIPLE unrelated
     instruments are universal edges. Convergent evolution cannot be faked.
     Cross-instrument validation is the strongest evidence against overfitting."

BIOLOGICAL PARALLEL:
    instrument         = species (independent evolutionary lineage)
    domesticated combo = adapted trait (locally beneficial mutation)
    convergent pattern = convergent evolution (same solution, different lineage)
    electrocyte        = universal optimum (physics-driven solution)
    sodium channel     = amplified expression of convergent winner
    contractile gene   = suppressed expression of divergent loser
    independence check = phylogenetic independence (not shared ancestry)
    convergence score  = n_lineages_adapted / n_lineages_observed

CONVERGENCE WITH OTHER ALGORITHMS:
    Algorithm #1 (VDJ Recombination):
        Memory B cells from VDJ engine contribute to domestication DBs.
        Convergent antibodies across instruments become super-antibodies.

    Algorithm #2 (TE Domestication):
        Electric Organs reads the OUTPUT of domestication (per-instrument).
        It does NOT replace domestication -- it AMPLIFIES it when confirmed
        by independent lineages.

    Algorithm #3 (CRISPR-Cas):
        Patterns flagged as convergent losers by Electric Organs can be
        added to the CRISPR spacer array for permanent suppression.

DATABASES:
    electric_organs_convergence.db  <- convergent patterns + scores
    teqa_domestication_{symbol}.db  <- per-instrument domestication (READ ONLY)
    electric_organs_signal.json     <- current convergence signal output

FILES:
    electric_organs.py              -> ConvergentSignalEngine class
    teqa_v3_neural_te.py            -> TEDomesticationTracker (read by this algo)
    BRAIN_*.py                      -> apply_convergence_boost() call point
"""
