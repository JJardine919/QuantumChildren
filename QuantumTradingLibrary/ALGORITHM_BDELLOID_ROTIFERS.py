"""
ALGORITHM: Bdelloid_Rotifers -- "Horizontal Gene Transfer"
===========================================================
Strategy DNA theft system that harvests proven components from high-performing
strategies and transplants them into struggling strategies during drawdown-
triggered desiccation events.

Biological basis:
    Bdelloid rotifers are microscopic animals (phylum Rotifera, class Bdelloidea)
    that have reproduced asexually for over 80 MILLION years. By all conventional
    evolutionary theory, they should be extinct -- asexual species accumulate
    deleterious mutations (Muller's ratchet) and lack the genetic diversity to
    adapt to changing environments. Yet bdelloid rotifers are one of the most
    successful animal groups on Earth, with over 460 described species.

    Their secret: MASSIVE horizontal gene transfer (HGT).

    The mechanism:
    1. DESICCATION: Bdelloid rotifers live in ephemeral aquatic habitats (moss,
       lichen, temporary pools). When their habitat dries out, they enter a state
       called anhydrobiosis -- complete metabolic shutdown. Their cells lose
       >95% of water content. During this process, their DNA SHATTERS into
       fragments. The double-strand breaks occur throughout the genome.

    2. FOREIGN DNA UPTAKE: While desiccated, the rotifer's cell membranes become
       permeable. Environmental DNA from bacteria, fungi, protists, and even
       other rotifers can enter the cell alongside debris. This DNA soup
       contains functional gene fragments from diverse organisms.

    3. REASSEMBLY WITH FOREIGN DNA: Upon rehydration, the rotifer's DNA repair
       machinery (homologous recombination, non-homologous end joining, and
       transposable element-mediated repair) reassembles the shattered genome.
       During this process, foreign DNA fragments are INCORPORATED into the
       rotifer's own genome at break points. TEs play a critical role as they
       provide sequence homology for integration at DSB sites.

    4. FOREIGN GENE EXPRESSION: Remarkably, ~8-10% of bdelloid rotifer genes
       are of non-metazoan origin. These horizontally acquired genes are:
       - Transcribed and translated (they actually WORK)
       - Under purifying selection (they are maintained because they are useful)
       - Often involve stress response, toxin degradation, and metabolic functions
       - Some provide capabilities that no animal genome normally encodes

    5. DESICCATION RESISTANCE ACCUMULATION: Rotifers that survive multiple
       desiccation-rehydration cycles accumulate MORE foreign genes. Each cycle
       is an opportunity for additional HGT. Survivors become genetically richer
       over time -- the exact opposite of Muller's ratchet.

    Key biological facts:
    - Adineta ricciae genome: 8% non-metazoan genes
    - Adineta vaga genome: ~8,000 foreign genes from >500 donor species
    - Foreign genes cluster at telomeric/subtelomeric regions (hotspots)
    - TEs (especially non-LTR retrotransposons) mark integration sites
    - The DNA repair system EXPECTS foreign fragments and uses them as patches
    - Desiccation-tolerant species have MORE HGT than desiccation-intolerant ones

Trading analogy -- "Strategy DNA Theft":
    When a strategy enters a drawdown (desiccation), its parameters become
    plastic -- this is the moment to STEAL proven components from other
    strategies that are currently performing well.

    Instead of only evolving strategies internally (sexual recombination as in
    VDJ, or mutation as in affinity maturation), this algorithm enables
    CROSS-STRATEGY COMPONENT THEFT. A momentum strategy in drawdown might
    absorb the exit logic from a successful mean-reversion strategy, or steal
    the regime filter from a breakout strategy that is thriving.

    The biological insight: diversity does not require sex. It requires
    ENVIRONMENTAL STRESS + PERMEABLE BOUNDARIES + FOREIGN DNA AVAILABILITY.

    In trading terms:
    - Drawdown = desiccation = DNA shattering trigger
    - Strategy parameters = genomic DNA
    - Other strategies' components = foreign DNA from environment
    - Parameter replacement = HGT incorporation at DSB sites
    - TE-mediated integration = TE activation weights guide which components
      to incorporate and where
    - Survival through multiple drawdowns = accumulated foreign gene resilience

Implementation: Python (SQLite persistence, config_loader for trading values)

Authors: DooDoo + Claude
Date:    2026-02-08
Version: BDELLOID-HGT-1.0

============================================================
PSEUDOCODE
============================================================
"""

# ----------------------------------------------------------
# CONSTANTS (from config_loader where applicable)
# ----------------------------------------------------------

# Desiccation trigger thresholds
DESICCATION_DRAWDOWN_PCT        = 0.10    # 10% drawdown from peak triggers desiccation
DESICCATION_LOSING_STREAK       = 5       # 5 consecutive losses triggers desiccation
DESICCATION_WIN_RATE_FLOOR      = 0.40    # WR below 40% over recent window triggers it
DESICCATION_EVAL_WINDOW         = 20      # Look at last 20 trades for WR floor check

# Foreign DNA scanning
DONOR_MIN_WIN_RATE              = 0.58    # Donor must have WR >= 58% to donate
DONOR_MIN_TRADES                = 15      # Donor needs >= 15 trades to be credible
DONOR_MIN_PROFIT_FACTOR         = 1.30    # Donor needs PF >= 1.30
DONOR_RECENCY_DAYS              = 30      # Only consider donors active in last 30 days

# HGT incorporation
HGT_MAX_COMPONENTS_PER_EVENT    = 3       # Max foreign components absorbed per desiccation
HGT_COMPONENT_ACCEPT_PROB       = 0.60    # 60% chance of accepting a scanned foreign component
HGT_PARAM_BLEND_RATIO           = 0.70    # 70% donor param, 30% original (not full overwrite)
HGT_TE_INTEGRATION_BOOST        = 0.15    # TE-active integration sites get 15% higher acceptance

# Reassembly
REASSEMBLY_QUARANTINE_TRADES    = 10      # After reassembly, quarantine for 10 trades before re-evaluation
REASSEMBLY_VALIDATION_MIN_WR    = 0.50    # Must achieve 50% WR during quarantine to keep foreign DNA
REASSEMBLY_REVERT_ON_FAILURE    = True    # Revert to pre-desiccation state if quarantine fails

# Foreign gene tracking
FOREIGN_GENE_MAX_AGE_DAYS       = 90      # Re-evaluate foreign genes after 90 days
FOREIGN_GENE_MIN_CONTRIBUTION   = 0.05    # Foreign gene must contribute >= 5% improvement to persist

# Desiccation resistance
RESISTANCE_BONUS_PER_SURVIVAL   = 0.02    # Each survived desiccation adds 2% resistance
RESISTANCE_MAX_BONUS            = 0.20    # Cap at 20% total resistance bonus
RESISTANCE_REDUCES_TRIGGER      = True    # Resistant strategies need deeper drawdown to trigger

# TE integration role
TE_INTEGRATION_FAMILIES = [
    "L1_Neuronal",          # Non-LTR retrotransposon: marks integration sites (biological basis)
    "Alu_Exonization",      # SINE: provides sequence homology for recombination
    "HERV_Synapse",         # ERV: facilitates cross-strategy gene flow
    "Mariner_Tc1",          # DNA transposon: cut-and-paste at integration sites
    "hobo",                 # DNA transposon: mediator of hybrid dysgenesis
]
TE_INTEGRATION_WEIGHT_THRESHOLD = 0.50    # TE must have activation > 0.50 to facilitate HGT

# Bayesian prior for strategy fitness tracking
PRIOR_ALPHA                     = 8
PRIOR_BETA                      = 8

# Component types available for HGT
COMPONENT_TYPES = [
    "entry_logic",          # V segment / entry signal parameters
    "exit_logic",           # J segment / exit strategy parameters
    "regime_filter",        # D segment / market regime classifier
    "te_weights",           # TE activation weight profile
    "risk_params",          # Position sizing, SL/TP ratios (NOT dollar values)
    "timing_params",        # Session filters, time-of-day weights
]

"""
============================================================
ALGORITHM Bdelloid_Rotifers_HGT
============================================================

DEFINE StrategyOrganism AS:
    strategy_id             : TEXT PRIMARY KEY (MD5 of strategy name + instrument)
    strategy_name           : TEXT (e.g. "momentum_breakout", "mean_revert_rsi")
    instrument              : TEXT (e.g. "XAUUSD", "BTCUSD")
    timeframe               : TEXT (e.g. "M5", "H1")

    # Core DNA (strategy parameters)
    components              : JSON {
        "entry_logic":   { type, params },
        "exit_logic":    { type, params },
        "regime_filter": { type, params },
        "te_weights":    { family: weight, ... },
        "risk_params":   { ... },
        "timing_params": { ... },
    }

    # Performance tracking
    win_count               : INTEGER
    loss_count              : INTEGER
    total_trades            : INTEGER
    posterior_wr            : REAL (Bayesian posterior win rate)
    profit_factor           : REAL
    total_pnl               : REAL
    peak_pnl                : REAL (high-water mark)
    current_drawdown_pct    : REAL (current drawdown from peak)
    consecutive_losses      : INTEGER
    recent_wr               : REAL (WR over last DESICCATION_EVAL_WINDOW trades)

    # Desiccation state
    is_desiccated           : BOOLEAN (currently in desiccation state)
    desiccation_count       : INTEGER (total desiccation events survived)
    desiccation_resistance  : REAL (0.0 to RESISTANCE_MAX_BONUS)
    last_desiccation        : TIMESTAMP
    pre_desiccation_snapshot: JSON (full component state before shattering)

    # Foreign gene inventory
    foreign_genes           : JSON [
        {
            "gene_id":          TEXT,
            "component_type":   TEXT,
            "donor_strategy_id":TEXT,
            "donor_name":       TEXT,
            "donated_params":   JSON,
            "incorporated_at":  TIMESTAMP,
            "desiccation_event":INTEGER,
            "te_integration_site": TEXT (which TE family facilitated),
            "contribution_score":REAL (measured improvement),
            "status":           TEXT ("active", "neutral", "rejected"),
        },
        ...
    ]

    # Reassembly state
    in_quarantine           : BOOLEAN
    quarantine_start        : TIMESTAMP
    quarantine_trades       : INTEGER
    quarantine_wins         : INTEGER

    # Metadata
    created_at              : TIMESTAMP
    last_updated            : TIMESTAMP
    generation              : INTEGER (increments each desiccation cycle)

DEFINE DesiccationEvent AS:
    event_id                : TEXT PRIMARY KEY
    strategy_id             : TEXT
    triggered_at            : TIMESTAMP
    trigger_reason          : TEXT ("drawdown", "losing_streak", "wr_floor")
    drawdown_at_trigger     : REAL
    components_shattered    : JSON (list of component types that shattered)
    donor_strategies_scanned: INTEGER
    foreign_genes_incorporated: INTEGER
    reassembly_completed_at : TIMESTAMP
    quarantine_passed       : BOOLEAN
    net_improvement         : REAL (PnL change post vs pre desiccation)

DEFINE DonorScan AS:
    scan_id                 : TEXT PRIMARY KEY
    event_id                : TEXT (FK to DesiccationEvent)
    donor_strategy_id       : TEXT
    donor_name              : TEXT
    component_type          : TEXT
    donor_fitness           : REAL (donor's posterior_wr at scan time)
    donor_pf                : REAL
    te_compatibility        : REAL (TE activation overlap score)
    accepted                : BOOLEAN
    acceptance_reason       : TEXT

STORAGE:
    bdelloid_db             : SQLite "bdelloid_rotifers.db"
    Table: strategies       : StrategyOrganism rows
    Table: desiccation_events: DesiccationEvent rows
    Table: donor_scans      : DonorScan rows
    Table: foreign_gene_log : audit log of all foreign gene incorporations
    Table: hgt_statistics   : aggregated HGT success/failure rates

------------------------------------------------------------
PHASE 1: DESICCATION DETECTION (continuous monitoring)
------------------------------------------------------------

ON check_desiccation(strategy_id):

    strategy = bdelloid_db.SELECT FROM strategies WHERE strategy_id = strategy_id

    IF strategy.is_desiccated OR strategy.in_quarantine:
        RETURN  # Already in desiccation or quarantine

    # Calculate effective trigger threshold (resistance lowers sensitivity)
    effective_drawdown_trigger = DESICCATION_DRAWDOWN_PCT
    IF RESISTANCE_REDUCES_TRIGGER:
        effective_drawdown_trigger += strategy.desiccation_resistance
        # Resistant strategies need deeper drawdown to trigger
        # e.g., 10% base + 8% resistance = needs 18% drawdown

    # TRIGGER 1: Drawdown from peak exceeds threshold
    IF strategy.current_drawdown_pct >= effective_drawdown_trigger:
        trigger_desiccation(strategy, "drawdown")
        RETURN

    # TRIGGER 2: Consecutive losing streak
    IF strategy.consecutive_losses >= DESICCATION_LOSING_STREAK:
        trigger_desiccation(strategy, "losing_streak")
        RETURN

    # TRIGGER 3: Recent win rate below floor
    IF (strategy.total_trades >= DESICCATION_EVAL_WINDOW
        AND strategy.recent_wr < DESICCATION_WIN_RATE_FLOOR):
        trigger_desiccation(strategy, "wr_floor")
        RETURN

------------------------------------------------------------
PHASE 2: DNA SHATTERING (parameter decomposition)
------------------------------------------------------------

FUNCTION trigger_desiccation(strategy, reason):

    # Snapshot current state (for potential rollback)
    strategy.pre_desiccation_snapshot = deepcopy(strategy.components)
    strategy.is_desiccated = TRUE
    strategy.desiccation_count += 1
    strategy.last_desiccation = NOW
    strategy.generation += 1

    # Determine WHICH components shatter (not all of them)
    # Components with worse recent performance are more likely to shatter
    shattered_components = []

    FOR component_type IN COMPONENT_TYPES:
        # Component-level contribution analysis:
        # Components that contributed to recent losses shatter first
        shatter_probability = compute_component_shatter_prob(strategy, component_type)
        IF random() < shatter_probability:
            shattered_components.append(component_type)

    # At least one component must shatter
    IF len(shattered_components) == 0:
        # Force-shatter the weakest component
        weakest = find_weakest_component(strategy)
        shattered_components.append(weakest)

    # Create desiccation event record
    event_id = MD5(f"{strategy.strategy_id}|{NOW}|{strategy.desiccation_count}")[:16]
    event = DesiccationEvent(
        event_id = event_id,
        strategy_id = strategy.strategy_id,
        triggered_at = NOW,
        trigger_reason = reason,
        drawdown_at_trigger = strategy.current_drawdown_pct,
        components_shattered = shattered_components,
    )
    bdelloid_db.INSERT INTO desiccation_events ...

    LOG("[HGT] Desiccation triggered for {strategy.strategy_name}: "
        "reason={reason}, drawdown={strategy.current_drawdown_pct:.1%}, "
        "shattered={shattered_components}, cycle={strategy.desiccation_count}")

    # Proceed to foreign DNA scanning
    scan_and_incorporate(strategy, event, shattered_components)

FUNCTION compute_component_shatter_prob(strategy, component_type) -> float:
    # Base probability: 40% for any component
    base_prob = 0.40

    # Higher probability if this component type has foreign genes that
    # have stopped contributing (old transplants that went stale)
    stale_foreign_count = count_stale_foreign_genes(strategy, component_type)
    staleness_bonus = min(0.30, stale_foreign_count * 0.10)

    # Higher probability for entry/exit logic (these are most market-sensitive)
    if component_type in ("entry_logic", "exit_logic"):
        base_prob += 0.10

    RETURN min(0.90, base_prob + staleness_bonus)

------------------------------------------------------------
PHASE 3: FOREIGN DNA SCANNING (donor strategy survey)
------------------------------------------------------------

FUNCTION scan_and_incorporate(strategy, event, shattered_components):

    # Query all other active strategies for potential donors
    all_strategies = bdelloid_db.SELECT FROM strategies
        WHERE strategy_id != strategy.strategy_id
          AND total_trades >= DONOR_MIN_TRADES
          AND posterior_wr >= DONOR_MIN_WIN_RATE
          AND profit_factor >= DONOR_MIN_PROFIT_FACTOR
          AND last_updated >= (NOW - DONOR_RECENCY_DAYS)
        ORDER BY posterior_wr DESC

    IF len(all_strategies) == 0:
        LOG("[HGT] No viable donors found. Reassembling with original DNA.")
        reassemble_strategy(strategy, event, incorporated=[])
        RETURN

    # Scan donors for compatible components
    incorporated_genes = []
    donor_scan_count = 0

    FOR component_type IN shattered_components:
        IF len(incorporated_genes) >= HGT_MAX_COMPONENTS_PER_EVENT:
            BREAK  # Cap on foreign DNA per desiccation event

        FOR donor IN all_strategies:
            donor_scan_count += 1

            # Check TE compatibility at integration site
            te_compatibility = compute_te_compatibility(strategy, donor, component_type)

            # Acceptance decision
            accept_prob = HGT_COMPONENT_ACCEPT_PROB

            # TE-mediated boost: higher acceptance if TEs at integration site are active
            IF te_compatibility > TE_INTEGRATION_WEIGHT_THRESHOLD:
                accept_prob += HGT_TE_INTEGRATION_BOOST

            # Cross-instrument penalty: same instrument donors preferred
            IF donor.instrument != strategy.instrument:
                accept_prob *= 0.70  # 30% penalty for cross-instrument transfer

            # Donor fitness bonus: better donors more likely accepted
            fitness_bonus = (donor.posterior_wr - DONOR_MIN_WIN_RATE) * 0.50
            accept_prob += fitness_bonus

            # Record the scan
            scan = DonorScan(
                scan_id = MD5(f"{event.event_id}|{donor.strategy_id}|{component_type}")[:16],
                event_id = event.event_id,
                donor_strategy_id = donor.strategy_id,
                donor_name = donor.strategy_name,
                component_type = component_type,
                donor_fitness = donor.posterior_wr,
                donor_pf = donor.profit_factor,
                te_compatibility = te_compatibility,
            )

            IF random() < accept_prob:
                # ACCEPT: Incorporate foreign DNA
                foreign_gene = incorporate_foreign_gene(
                    strategy, donor, component_type, event
                )
                incorporated_genes.append(foreign_gene)
                scan.accepted = TRUE
                scan.acceptance_reason = f"prob={accept_prob:.2f}, te_compat={te_compatibility:.2f}"
                bdelloid_db.INSERT INTO donor_scans ...
                BREAK  # Move to next shattered component

            ELSE:
                scan.accepted = FALSE
                scan.acceptance_reason = f"prob={accept_prob:.2f}, random rejected"
                bdelloid_db.INSERT INTO donor_scans ...

    event.donor_strategies_scanned = donor_scan_count
    event.foreign_genes_incorporated = len(incorporated_genes)
    bdelloid_db.UPDATE desiccation_events ...

    LOG("[HGT] Scanned {donor_scan_count} donors, incorporated {len(incorporated_genes)} "
        "foreign genes for {strategy.strategy_name}")

    # Proceed to reassembly
    reassemble_strategy(strategy, event, incorporated_genes)

------------------------------------------------------------
PHASE 4: HGT INCORPORATION (foreign gene integration)
------------------------------------------------------------

FUNCTION incorporate_foreign_gene(strategy, donor, component_type, event) -> Dict:

    # Extract donor component parameters
    donor_params = donor.components[component_type]

    # Blend donor params with remaining original params (not full overwrite)
    # This models the biological reality: HGT doesn't replace the entire gene,
    # it inserts a foreign fragment that gets partially integrated
    blended_params = {}
    original_params = strategy.components[component_type]

    FOR key IN union(donor_params.keys(), original_params.keys()):
        IF key IN donor_params AND key IN original_params:
            # Both have this param: blend
            IF isinstance(donor_params[key], (int, float)):
                blended_params[key] = (
                    HGT_PARAM_BLEND_RATIO * donor_params[key]
                    + (1.0 - HGT_PARAM_BLEND_RATIO) * original_params[key]
                )
            ELSE:
                # Non-numeric: take donor's version
                blended_params[key] = donor_params[key]
        ELIF key IN donor_params:
            # Only donor has it: adopt foreign gene
            blended_params[key] = donor_params[key]
        ELSE:
            # Only original has it: keep (undamaged DNA fragment)
            blended_params[key] = original_params[key]

    # Identify which TE family facilitated integration
    integration_te = select_integration_te(strategy)

    # Create foreign gene record
    gene_id = MD5(f"{donor.strategy_id}|{component_type}|{NOW}")[:16]
    foreign_gene = {
        "gene_id":           gene_id,
        "component_type":    component_type,
        "donor_strategy_id": donor.strategy_id,
        "donor_name":        donor.strategy_name,
        "donated_params":    donor_params,
        "incorporated_at":   NOW,
        "desiccation_event": strategy.desiccation_count,
        "te_integration_site": integration_te,
        "contribution_score": 0.0,  # Measured after quarantine
        "status":            "active",
    }

    # Apply blended params to strategy
    strategy.components[component_type] = blended_params

    # Add to foreign gene inventory
    strategy.foreign_genes.append(foreign_gene)

    LOG("[HGT] Foreign gene incorporated: {component_type} from {donor.strategy_name} "
        "via TE:{integration_te}, gene_id={gene_id[:8]}")

    RETURN foreign_gene

FUNCTION compute_te_compatibility(strategy, donor, component_type) -> float:
    # TE compatibility measures how well the TE activation profiles match
    # at the integration site for this component type
    #
    # Biological analogy: TEs at DSB sites provide sequence homology for
    # foreign DNA integration. More active TEs = more integration sites.

    strategy_te_weights = strategy.components.get("te_weights", {})
    donor_te_weights = donor.components.get("te_weights", {})

    compatibility = 0.0
    n_checked = 0

    FOR te_family IN TE_INTEGRATION_FAMILIES:
        s_weight = strategy_te_weights.get(te_family, 0.0)
        d_weight = donor_te_weights.get(te_family, 0.0)

        # Both having high activation = good integration site
        # Like both genomes having active TEs at homologous positions
        overlap = min(s_weight, d_weight)
        compatibility += overlap
        n_checked += 1

    IF n_checked > 0:
        compatibility /= n_checked

    RETURN compatibility

FUNCTION select_integration_te(strategy) -> str:
    # Select which TE family facilitates the integration
    # Prefer TEs with highest activation (most active integration sites)
    te_weights = strategy.components.get("te_weights", {})

    best_te = None
    best_weight = -1.0

    FOR te_family IN TE_INTEGRATION_FAMILIES:
        w = te_weights.get(te_family, 0.0)
        IF w > best_weight:
            best_weight = w
            best_te = te_family

    RETURN best_te IF best_te IS NOT NULL ELSE "L1_Neuronal"

------------------------------------------------------------
PHASE 5: REASSEMBLY (genome reconstruction)
------------------------------------------------------------

FUNCTION reassemble_strategy(strategy, event, incorporated_genes):

    # The strategy is now reconstructed with a hybrid genome:
    # - Original components that did NOT shatter (preserved DNA)
    # - Blended components where foreign genes were incorporated
    # - Potentially some shattered components with no donor (repaired from original)

    # Enter quarantine: observe N trades before judging the reassembled strategy
    strategy.is_desiccated = FALSE
    strategy.in_quarantine = TRUE
    strategy.quarantine_start = NOW
    strategy.quarantine_trades = 0
    strategy.quarantine_wins = 0

    # Update desiccation resistance (each survival makes the organism tougher)
    strategy.desiccation_resistance = min(
        RESISTANCE_MAX_BONUS,
        strategy.desiccation_resistance + RESISTANCE_BONUS_PER_SURVIVAL
    )

    bdelloid_db.UPDATE strategies SET ... WHERE strategy_id = strategy.strategy_id

    LOG("[HGT] Strategy {strategy.strategy_name} reassembled: "
        "{len(incorporated_genes)} foreign genes, "
        "generation={strategy.generation}, "
        "resistance={strategy.desiccation_resistance:.1%}")

    event.reassembly_completed_at = NOW
    bdelloid_db.UPDATE desiccation_events ...

------------------------------------------------------------
PHASE 6: QUARANTINE EVALUATION
------------------------------------------------------------

ON record_quarantine_outcome(strategy_id, won, pnl):

    strategy = bdelloid_db.SELECT FROM strategies WHERE strategy_id = strategy_id

    IF NOT strategy.in_quarantine:
        RETURN  # Not in quarantine

    strategy.quarantine_trades += 1
    IF won:
        strategy.quarantine_wins += 1

    # Check if quarantine period is complete
    IF strategy.quarantine_trades >= REASSEMBLY_QUARANTINE_TRADES:
        quarantine_wr = strategy.quarantine_wins / strategy.quarantine_trades

        IF quarantine_wr >= REASSEMBLY_VALIDATION_MIN_WR:
            # QUARANTINE PASSED: Keep foreign DNA
            strategy.in_quarantine = FALSE
            strategy.consecutive_losses = 0  # Reset streak

            # Score each foreign gene's contribution
            score_foreign_gene_contributions(strategy)

            LOG("[HGT] Quarantine PASSED for {strategy.strategy_name}: "
                "WR={quarantine_wr:.1%} over {strategy.quarantine_trades} trades")

            # Update the desiccation event record
            event = get_latest_desiccation_event(strategy)
            IF event:
                event.quarantine_passed = TRUE
                bdelloid_db.UPDATE desiccation_events ...

        ELSE:
            # QUARANTINE FAILED: Revert to pre-desiccation state
            IF REASSEMBLY_REVERT_ON_FAILURE AND strategy.pre_desiccation_snapshot:
                strategy.components = strategy.pre_desiccation_snapshot
                # Mark all foreign genes from this cycle as rejected
                FOR gene IN strategy.foreign_genes:
                    IF gene["desiccation_event"] == strategy.desiccation_count:
                        gene["status"] = "rejected"

                LOG("[HGT] Quarantine FAILED for {strategy.strategy_name}: "
                    "WR={quarantine_wr:.1%}, REVERTING to pre-desiccation state")
            ELSE:
                LOG("[HGT] Quarantine FAILED for {strategy.strategy_name}: "
                    "WR={quarantine_wr:.1%}, NO revert (snapshot unavailable)")

            strategy.in_quarantine = FALSE

            # Update the desiccation event record
            event = get_latest_desiccation_event(strategy)
            IF event:
                event.quarantine_passed = FALSE
                bdelloid_db.UPDATE desiccation_events ...

    bdelloid_db.UPDATE strategies SET ... WHERE strategy_id = strategy.strategy_id

------------------------------------------------------------
PHASE 7: FOREIGN GENE MAINTENANCE (ongoing evaluation)
------------------------------------------------------------

ON evaluate_foreign_genes(strategy_id):

    strategy = bdelloid_db.SELECT FROM strategies WHERE strategy_id = strategy_id

    FOR gene IN strategy.foreign_genes:
        IF gene["status"] != "active":
            CONTINUE

        # Check age
        days_since_incorporation = (NOW - gene["incorporated_at"]).days
        IF days_since_incorporation > FOREIGN_GENE_MAX_AGE_DAYS:
            # Re-evaluate contribution
            IF gene["contribution_score"] < FOREIGN_GENE_MIN_CONTRIBUTION:
                gene["status"] = "neutral"
                LOG("[HGT] Foreign gene {gene['gene_id'][:8]} from "
                    "{gene['donor_name']} marked NEUTRAL: "
                    "contribution={gene['contribution_score']:.3f}")

    bdelloid_db.UPDATE strategies SET ... WHERE strategy_id = strategy.strategy_id

FUNCTION score_foreign_gene_contributions(strategy):
    # Compare post-desiccation performance to pre-desiccation performance
    # Attribute improvement to each incorporated foreign gene

    pre_wr = compute_pre_desiccation_wr(strategy)
    post_wr = strategy.quarantine_wins / max(1, strategy.quarantine_trades)

    improvement = post_wr - pre_wr

    # Distribute improvement across foreign genes from this cycle
    active_genes = [g for g in strategy.foreign_genes
                    IF g["desiccation_event"] == strategy.desiccation_count
                    AND g["status"] == "active"]

    IF len(active_genes) > 0:
        per_gene_contribution = improvement / len(active_genes)
        FOR gene IN active_genes:
            gene["contribution_score"] = per_gene_contribution

------------------------------------------------------------
PHASE 8: PERFORMANCE TRACKING (update strategy metrics)
------------------------------------------------------------

ON record_trade_outcome(strategy_id, won, pnl):

    strategy = bdelloid_db.SELECT FROM strategies WHERE strategy_id = strategy_id

    # Update core metrics
    strategy.win_count += 1 IF won ELSE 0
    strategy.loss_count += 0 IF won ELSE 1
    strategy.total_trades = strategy.win_count + strategy.loss_count
    strategy.total_pnl += pnl
    strategy.last_updated = NOW

    # Update consecutive losses
    IF won:
        strategy.consecutive_losses = 0
    ELSE:
        strategy.consecutive_losses += 1

    # Bayesian posterior WR
    strategy.posterior_wr = (PRIOR_ALPHA + strategy.win_count) /
                            (PRIOR_ALPHA + PRIOR_BETA + strategy.total_trades)

    # Update peak PnL and drawdown
    IF strategy.total_pnl > strategy.peak_pnl:
        strategy.peak_pnl = strategy.total_pnl
    IF strategy.peak_pnl > 0:
        strategy.current_drawdown_pct = (strategy.peak_pnl - strategy.total_pnl) / strategy.peak_pnl
    ELSE:
        strategy.current_drawdown_pct = 0.0

    # Recent WR (sliding window)
    strategy.recent_wr = compute_recent_wr(strategy, DESICCATION_EVAL_WINDOW)

    # Profit factor
    IF strategy.avg_loss > 0:
        strategy.profit_factor = strategy.avg_win / strategy.avg_loss
    ELSE:
        strategy.profit_factor = 99.0 IF strategy.avg_win > 0 ELSE 0.0

    bdelloid_db.UPDATE strategies SET ... WHERE strategy_id = strategy.strategy_id

    # If in quarantine, feed to quarantine tracker
    IF strategy.in_quarantine:
        record_quarantine_outcome(strategy_id, won, pnl)
        RETURN  # Don't check desiccation during quarantine

    # Check for desiccation trigger
    check_desiccation(strategy_id)

------------------------------------------------------------
PHASE 9: HGT STATISTICS & REPORTING
------------------------------------------------------------

FUNCTION generate_hgt_report() -> Dict:

    # Strategy-level summary
    strategies = bdelloid_db.SELECT * FROM strategies

    total_strategies = len(strategies)
    total_desiccations = SUM(s.desiccation_count FOR s IN strategies)
    total_foreign_genes = SUM(len(s.foreign_genes) FOR s IN strategies)
    active_foreign_genes = SUM(
        len([g for g in s.foreign_genes if g["status"] == "active"])
        FOR s IN strategies
    )

    # Foreign gene origin map: which donors contribute most?
    donor_frequency = {}
    FOR s IN strategies:
        FOR g IN s.foreign_genes:
            IF g["status"] == "active":
                donor_frequency[g["donor_name"]] = donor_frequency.get(g["donor_name"], 0) + 1

    # Component type distribution
    component_distribution = {}
    FOR s IN strategies:
        FOR g IN s.foreign_genes:
            IF g["status"] == "active":
                ct = g["component_type"]
                component_distribution[ct] = component_distribution.get(ct, 0) + 1

    # Desiccation survival rate
    events = bdelloid_db.SELECT * FROM desiccation_events
    quarantine_passed = len([e for e in events if e.quarantine_passed])
    survival_rate = quarantine_passed / max(1, len(events))

    # Average foreign gene percentage per strategy
    foreign_pcts = []
    FOR s IN strategies:
        total_components = len(COMPONENT_TYPES)
        foreign_count = len([g for g in s.foreign_genes if g["status"] == "active"])
        foreign_pcts.append(foreign_count / total_components)

    RETURN {
        "timestamp":            NOW,
        "total_strategies":     total_strategies,
        "total_desiccations":   total_desiccations,
        "total_foreign_genes":  total_foreign_genes,
        "active_foreign_genes": active_foreign_genes,
        "quarantine_survival_rate": survival_rate,
        "avg_foreign_gene_pct": mean(foreign_pcts) IF foreign_pcts ELSE 0,
        "top_donors":           sorted(donor_frequency.items(), key=lambda x: -x[1])[:10],
        "component_distribution": component_distribution,
        "avg_desiccation_resistance": mean(s.desiccation_resistance FOR s IN strategies),
    }

------------------------------------------------------------
THE LOOP (steady-state behavior)
------------------------------------------------------------

LOOP on every trade outcome:

    +-----------------------------------------------------+
    |  Trade executed by any strategy                     |
    |    -> record_trade_outcome(strategy_id, won, pnl)   |
    |    -> updates WR, PF, drawdown, consecutive losses  |
    +-------------------+---------------------------------+
                        |
                        v
    +-----------------------------------------------------+
    |  In quarantine?                                     |
    |  YES -> record_quarantine_outcome()                 |
    |         -> if quarantine complete:                   |
    |            PASS -> keep foreign DNA, score genes     |
    |            FAIL -> revert to pre-desiccation state   |
    |  NO  -> check_desiccation()                         |
    +-------------------+---------------------------------+
                        |
                        v (if desiccation triggered)
    +-----------------------------------------------------+
    |  trigger_desiccation()                              |
    |    -> snapshot current state (rollback safety)       |
    |    -> determine which components shatter            |
    |    -> scan donors for foreign DNA                   |
    |    -> incorporate foreign genes (blended params)    |
    |    -> reassemble with hybrid genome                 |
    |    -> enter quarantine                              |
    +-----------------------------------------------------+

    PERIODICALLY (every FOREIGN_GENE_MAX_AGE_DAYS):

    +-----------------------------------------------------+
    |  evaluate_foreign_genes() for each strategy         |
    |    -> mark stale/non-contributing genes as neutral   |
    |    -> stale genes increase shatter prob next cycle   |
    +-----------------------------------------------------+

    PERIODICALLY (on demand or scheduled):

    +-----------------------------------------------------+
    |  generate_hgt_report()                              |
    |    -> desiccation/survival stats                     |
    |    -> foreign gene census                            |
    |    -> donor contribution map                         |
    |    -> component type distribution                    |
    +-----------------------------------------------------+

INVARIANT:
    "Strategies that suffer drawdowns do not die -- they STEAL from the
     strong. Every desiccation event is an opportunity for horizontal
     gene transfer. Diversity comes not from breeding but from theft
     under duress. The survivors accumulate foreign genes that make
     them more robust with every crisis they endure."

BIOLOGICAL PARALLEL:
    strategy_organism          = bdelloid rotifer individual
    strategy components        = rotifer genome (native DNA)
    drawdown/losing streak     = desiccation event (habitat dries out)
    component shattering       = double-strand DNA breaks during desiccation
    donor scanning             = environmental DNA soup available during anhydrobiosis
    foreign gene incorporation = horizontal gene transfer at DSB repair sites
    TE-mediated integration    = non-LTR retrotransposons marking integration sites
    param blending (70/30)     = partial gene integration (not full chromosome replacement)
    quarantine                 = rehydration recovery period
    quarantine pass/fail       = organism survival/death after rehydration
    pre-desiccation snapshot   = genome checkpoint before stress
    desiccation resistance     = accumulated stress tolerance (trehalose, LEA proteins)
    foreign gene inventory     = non-metazoan genes in rotifer genome (~8-10%)
    contribution scoring       = purifying selection on horizontally acquired genes
    gene status "neutral"      = defective foreign gene copies (like pseudogenes)
    generation counter         = desiccation-rehydration cycle count

CONVERGENCE:
    After N desiccation cycles across M strategies:
    - Strategies accumulate foreign genes that WORK (proven donor components)
    - Stale foreign genes are marked neutral and eventually replaced
    - Each strategy becomes a unique chimera of original + borrowed DNA
    - The best donor strategies become "gene reservoirs" (like bacteria in
      the rotifer's environment that provide useful genes)
    - Desiccation resistance grows: robust strategies need deeper drawdowns
      to trigger reconstruction, reducing unnecessary churn
    - Foreign gene percentage per strategy converges toward 8-10% (matching
      the biological observation in Adineta vaga)
    - The population of strategies becomes MORE diverse over time, not less --
      countering Muller's ratchet through systematic component theft

DATABASES:
    bdelloid_rotifers.db       <- strategy state, foreign genes, desiccation events
    Uses CONFIDENCE_THRESHOLD from config_loader for gate integration

FILES:
    bdelloid_rotifers.py       -> BdelloidHGTEngine class
    ALGORITHM_BDELLOID_ROTIFERS.py -> this pseudocode specification
    vdj_recombination.py       -> source of antibody components (donor pool)
    teqa_v3_neural_te.py       -> TE activation engine (integration sites)
"""
