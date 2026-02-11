"""
ALGORITHM: Cancer_Cell -- "Oncogenic Strategy Acceleration"
============================================================
Full cancer simulation: rapidly mutate strategies, spread winners across all
markets, temporarily bypass safety filters to find hidden edges, and run
the entire population through the quantum computer at GPU speed.

Biological basis:
    Cancer is uncontrolled cell proliferation arising from accumulated mutations
    in genes that regulate the cell cycle. Normal cells divide ~50 times
    (the Hayflick limit, controlled by telomere shortening) before entering
    senescence. Cancer cells bypass ALL of these controls:

    1. ONCOGENE ACTIVATION (gain-of-function mutations):
       - RAS family (KRAS, NRAS, HRAS): ~30% of all cancers
         RAS proteins are molecular switches. When mutated, they are STUCK ON,
         sending continuous "grow and divide" signals even without external
         growth factor stimulation.
       - MYC: Transcription factor that drives cell cycle progression.
         Amplified MYC = cells that divide without restraint.
       - HER2: Receptor tyrosine kinase that amplifies growth signaling.
         Overexpression = 100x more receptors = hypersensitivity to growth signals.

    2. TUMOR SUPPRESSOR INACTIVATION (loss-of-function mutations):
       - TP53 ("Guardian of the Genome"): Present in ~50% of all cancers.
         Normal p53 detects DNA damage and either repairs it or triggers
         apoptosis (programmed cell death). When p53 is mutated/deleted,
         damaged cells survive and accumulate MORE mutations. Two-hit
         hypothesis: both alleles must be lost (Knudson's model).
       - RB1 (Retinoblastoma protein): Controls the G1/S checkpoint.
         Normal RB binds E2F transcription factors, preventing S phase.
         Loss of RB = cells skip the checkpoint entirely.
       - BRCA1/BRCA2: Homologous recombination repair. Loss = genomic
         instability, accumulation of structural variants. The genome
         becomes a mutation factory.
       - APC (Adenomatous Polyposis Coli): Wnt pathway regulator.
         Loss = constitutive Wnt signaling = colorectal cancer.

    3. TELOMERASE ACTIVATION (immortality):
       - Normal cells have a division counter (telomeres shorten each cycle).
         After ~50 divisions, critically short telomeres trigger senescence.
       - Cancer cells reactivate TERT (telomerase reverse transcriptase),
         which adds telomeric repeats (TTAGGG) back to chromosome ends.
       - Result: unlimited replicative potential. The cell becomes immortal.
       - ~85% of cancers use TERT. The rest use ALT (Alternative Lengthening
         of Telomeres) via recombination.

    4. WARBURG EFFECT (metabolic reprogramming):
       - Normal cells use oxidative phosphorylation (36 ATP/glucose) when
         oxygen is available. They only switch to glycolysis (2 ATP/glucose)
         under anaerobic conditions.
       - Cancer cells use glycolysis EVEN IN THE PRESENCE OF OXYGEN
         (aerobic glycolysis). This is wildly inefficient per glucose molecule
         but provides two advantages:
         a) SPEED: Glycolysis is 10-100x faster than OXPHOS. Cancer cells
            prioritize THROUGHPUT over efficiency.
         b) BIOSYNTHETIC PRECURSORS: Glycolytic intermediates feed into
            nucleotide, amino acid, and lipid synthesis -- the raw materials
            needed for rapid cell division.
       - Analogy: Using GPU parallel processing instead of sequential CPU.
         Lower per-unit efficiency, but massively higher throughput.

    5. ANGIOGENESIS (building blood supply):
       - Tumors larger than 1-2mm cannot survive on diffusion alone.
         They need their own blood supply.
       - Cancer cells secrete VEGF (Vascular Endothelial Growth Factor),
         which stimulates nearby blood vessels to sprout new branches
         toward the tumor (angiogenic switch).
       - This diverts nutrients and oxygen from healthy tissue to the tumor.
       - Analogy: Allocating more capital/compute to promising strategy
         clusters at the expense of underperformers.

    6. METASTASIS (distant colonization):
       - The hallmark of malignancy. Cancer cells detach from the primary
         tumor, enter the bloodstream (intravasation), survive circulation,
         exit at a distant site (extravasation), and colonize new tissue.
       - Steps:
         a) Epithelial-Mesenchymal Transition (EMT): cells lose adhesion,
            gain motility. They transform from stationary to migratory.
         b) Intravasation: penetrate blood vessel walls
         c) Circulation: survive immune attack in the bloodstream
         d) Extravasation: exit at distant site
         e) Colonization: adapt to the new tissue microenvironment and
            establish secondary tumors
       - Not all cells can metastasize. It requires a specific combination
         of mutations AND compatibility with the target tissue ("seed and
         soil" hypothesis by Stephen Paget, 1889).
       - Analogy: A winning strategy on BTCUSD "metastasizes" to XAUUSD,
         ETHUSD, etc. But it must pass a compatibility check first.

    7. IMMUNE EVASION:
       - Normal immune system (NK cells, T cells) kills abnormal cells.
       - Cancer cells evade via:
         a) PD-L1 expression: binds PD-1 on T cells, sending "don't kill me"
         b) MHC-I downregulation: hide antigens from T cell recognition
         c) Treg recruitment: attract regulatory T cells that suppress immunity
         d) IDO expression: depletes tryptophan in the microenvironment,
            starving T cells
       - Analogy: Strategies that bypass the CRISPR-Cas blocker and
         the Toxoplasma infection detector by generating novel signals
         that don't match any known loss patterns.

    MULTI-HIT MODEL (Vogelstein's "Hallmarks of Cancer"):
    Cancer requires 3-7 driver mutations accumulated over years/decades.
    Each mutation alone is insufficient. The combination creates a cell
    that grows without restraint, ignores stop signals, is immortal,
    builds its own blood supply, and can colonize distant organs.

    Our trading analogy: a SINGLE mutation to a strategy is rarely
    transformative. The algorithm combines MULTIPLE simultaneous mutations
    (parameter changes + TE pattern changes + safety bypass + cross-symbol
    spread) to find the rare combinations that produce breakthrough
    performance. Most mutants die (natural selection). The survivors are
    the "cancer" -- strategies that grow without the normal constraints.

Trading analogy -- "Oncogenic Strategy Acceleration":

    The STRATEGY is the cell. Normal strategies operate within constraints
    (confidence thresholds, regime detection, CRISPR blocks, position limits).
    The Cancer Cell algorithm TEMPORARILY removes these constraints to
    explore the parameter space aggressively, then re-applies them to
    validate which mutations actually produce alpha.

    Phase 1 - ONCOGENE ACTIVATION (Mitosis):
        Rapidly generate N mutant variants of each strategy by mutating
        parameters: TE weights, confidence thresholds, regime sensitivity,
        holding periods, SL/TP ratios. Each mutant is a "daughter cell."
        Use GPU to run all mutants through quantum encoder simultaneously.

    Phase 2 - TUMOR SUPPRESSOR BYPASS (Checkpoint Removal):
        Temporarily disable CRISPR-Cas blocks, Toxoplasma countermeasures,
        and confidence thresholds IN SIMULATION ONLY. Let mutant strategies
        trade freely in backtested data to find edges that the normal
        immune system would have blocked.
        CRITICAL: This runs in SIMULATION, not live. SL stays sacred.

    Phase 3 - WARBURG ACCELERATION (GPU Batch Processing):
        Run the entire mutant population through the Qiskit quantum encoder
        in parallel batches on the GPU. Prioritize throughput over per-unit
        efficiency. The Warburg effect: 10x more mutations evaluated per
        second, even if each evaluation is less precise.

    Phase 4 - ANGIOGENESIS (Resource Allocation):
        Mutant clusters that show positive fitness get more "blood supply"
        (more mutations, more backtesting time, more symbols to evaluate).
        Underperforming clusters get starved (no further mutation budget).

    Phase 5 - METASTASIS (Cross-Symbol Spread):
        Winning mutants attempt to "metastasize" to other symbols.
        Compatibility check (the "seed and soil" test): backtest the
        mutant on the target symbol. If it passes, it colonizes.
        If it fails, the metastasis is rejected.

    Phase 6 - TELOMERASE (Immortality for Winners):
        Mutants that pass all tests are given "telomerase" -- they are
        promoted to the permanent strategy pool with no expiry. Their
        parameters are written to the TE Domestication database as
        new domesticated patterns.

    Phase 7 - IMMUNE EVASION CHECK (Re-enable Defenses):
        After the cancer simulation completes, RE-ENABLE all safety
        systems (CRISPR, Toxoplasma, confidence thresholds). Run the
        surviving mutants through the full immune gauntlet one final time.
        Only mutants that survive WITH defenses active are promoted.
        This is the "immune checkpoint" -- the last line of defense
        against false positives.

    CRITICAL SAFETY:
        - Cancer Cell runs in SIMULATION/BACKTEST mode only
        - SL remains sacred ($1.00 max loss per trade -- from config_loader)
        - Tumor suppressor bypass is TIME-LIMITED (max 1 hour per run)
        - All mutants must pass immune checkpoint before live deployment
        - No live trades are placed by the cancer simulation
        - Results are written to cancer_cell.db for BRAIN scripts to read

Implementation: Python (SQLite, Qiskit, torch+DirectML, config_loader)
Integration: TEQA v3.0, TE Domestication, CRISPR-Cas, Toxoplasma, BRAIN

Authors: DooDoo + Claude
Date:    2026-02-09
Version: CANCER-CELL-1.0

============================================================
PSEUDOCODE
============================================================
"""

# ---------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------

# Mitosis (strategy mutation)
MITOSIS_POPULATION_SIZE     = 200     # Number of mutant daughters per parent
MITOSIS_MUTATION_RATE       = 0.30    # 30% of parameters mutated per daughter
MITOSIS_MUTATION_STRENGTH   = 0.20    # Max 20% parameter change per mutation
MITOSIS_PARENT_STRATEGIES   = 10      # Top N strategies used as parent cells
MITOSIS_GENERATIONS         = 5       # Mutation rounds before selection

# Tumor suppressor bypass
BYPASS_MAX_DURATION_SEC     = 3600    # Max 1 hour of unconstrained simulation
BYPASS_CONFIDENCE_FLOOR     = 0.05    # Even in bypass mode, minimum confidence
BYPASS_CRISPR_DISABLED      = True    # Disable CRISPR-Cas blocks in sim
BYPASS_TOXOPLASMA_DISABLED  = True    # Disable Toxoplasma countermeasures in sim
BYPASS_REGIME_DISABLED      = True    # Disable regime detection in sim

# Warburg effect (GPU acceleration)
WARBURG_BATCH_SIZE          = 64      # Quantum circuits processed per GPU batch
WARBURG_SHOTS_FAST          = 2048    # Reduced shots for speed (vs 8192 normal)
WARBURG_USE_GPU             = True    # Use DirectML GPU acceleration

# Angiogenesis (resource allocation)
ANGIO_TOP_PERCENT           = 0.20    # Top 20% get more resources
ANGIO_STARVE_PERCENT        = 0.50    # Bottom 50% get starved
ANGIO_RESOURCE_MULTIPLIER   = 3.0     # Top mutants get 3x more mutation budget

# Metastasis (cross-symbol spread)
METASTASIS_MIN_FITNESS      = 0.65    # Minimum fitness to attempt metastasis
METASTASIS_SOIL_TEST_BARS   = 500     # Bars of target symbol data for soil test
METASTASIS_COLONIZE_WR      = 0.55    # Min win rate on target to colonize
METASTASIS_MAX_SYMBOLS      = 5       # Max symbols a mutant can spread to

# Telomerase (immortality)
TELOMERASE_PROMOTION_WR     = 0.65    # Win rate threshold for immortality
TELOMERASE_PROMOTION_PF     = 1.5     # Profit factor threshold for immortality
TELOMERASE_MIN_TRADES       = 30      # Minimum simulated trades before promotion

# Immune checkpoint (re-validation)
IMMUNE_CHECKPOINT_WR        = 0.58    # Must still achieve 58% WITH defenses on
IMMUNE_CHECKPOINT_PF        = 1.2     # Profit factor with defenses on
IMMUNE_CHECKPOINT_BARS      = 300     # Bars for final validation run

# Multi-hit model
MULTI_HIT_MIN_DRIVERS       = 3       # Minimum simultaneous mutations for breakthrough
MULTI_HIT_MAX_DRIVERS       = 7       # Maximum mutations to test

"""
============================================================
ALGORITHM Cancer_Cell
============================================================

DEFINE MutationType AS ENUM:
    ONCOGENE_ACTIVATION     # Amplify a signal weight (gain-of-function)
    SUPPRESSOR_DELETION     # Remove a gate/filter (loss-of-function)
    METABOLIC_SHIFT         # Change processing speed/precision tradeoff
    RECEPTOR_MUTATION       # Alter symbol/timeframe sensitivity
    TELOMERE_EXTENSION      # Remove strategy expiry/decay

DEFINE CancerPhase AS ENUM:
    MITOSIS                 # Rapid cell division (mutation generation)
    G1_CHECKPOINT_BYPASS    # Tumor suppressor inactivation
    S_PHASE                 # DNA replication (strategy evaluation)
    G2_CHECKPOINT_BYPASS    # Genome integrity check bypass
    M_PHASE                 # Cell division (strategy selection)
    METASTASIS              # Distant colonization
    ANGIOGENESIS            # Resource allocation
    IMMUNE_EVASION          # Final validation with defenses on
    APOPTOSIS               # Programmed death of failed mutants

DEFINE StrategyMutant AS:
    mutant_id              : TEXT PRIMARY KEY     # UUID
    parent_id              : TEXT                 # Parent strategy ID
    generation             : INT                 # Mutation generation (0 = parent)
    # Mutated parameters (delta from parent)
    te_weight_deltas       : DICT[str, float]    # Per-TE family weight changes
    confidence_delta       : FLOAT               # Change to confidence threshold
    regime_sensitivity     : FLOAT               # 0 = ignore regime, 1 = strict
    hold_time_mult         : FLOAT               # Multiplier on max hold time
    sl_ratio               : FLOAT               # SL as fraction of MAX_LOSS_DOLLARS
    tp_ratio               : FLOAT               # TP multiplier override
    # Mutation metadata
    mutation_types         : LIST[MutationType]   # Which mutations were applied
    driver_count           : INT                 # Number of simultaneous driver mutations
    # Fitness metrics (filled after evaluation)
    fitness_score          : FLOAT               # Composite fitness
    win_rate               : FLOAT               # Simulated win rate
    profit_factor          : FLOAT               # Simulated profit factor
    total_trades           : INT                 # Simulated trade count
    sharpe_ratio           : FLOAT               # Simulated Sharpe
    max_drawdown           : FLOAT               # Simulated max DD
    # Lifecycle
    is_alive               : BOOL                # Survived selection?
    has_telomerase         : BOOL                # Promoted to immortal?
    metastasized_to        : LIST[TEXT]          # Symbols successfully colonized
    created_at             : TIMESTAMP
    promoted_at            : TIMESTAMP OR NULL

DEFINE TumorCluster AS:
    cluster_id             : TEXT PRIMARY KEY
    parent_symbol          : TEXT                # Original symbol
    mutants                : LIST[StrategyMutant]
    avg_fitness            : FLOAT
    resource_budget        : INT                # Remaining mutation iterations
    generation             : INT
    # Angiogenesis state
    vegf_signal            : FLOAT               # Resource demand signal
    blood_supply           : FLOAT               # Allocated resources (0-1)

STORAGE:
    cancer_db              : SQLite "cancer_cell.db"
        TABLE mutants           -- all strategy mutants (alive + dead)
        TABLE clusters          -- tumor clusters with resource budgets
        TABLE metastasis_log    -- cross-symbol colonization attempts
        TABLE promotion_log     -- mutants promoted to live via telomerase
        TABLE immune_checkpoint -- final validation results
        TABLE run_history       -- metadata for each cancer run
    config_loader          : imports MAX_LOSS_DOLLARS, CONFIDENCE_THRESHOLD
                             (NEVER hardcoded)

--------------------------------------------------------------------
PHASE 1: MITOSIS (Oncogene Activation -- Rapid Strategy Mutation)
--------------------------------------------------------------------

ON mitosis_cycle(parent_strategies[], symbol, bars):

    # Step 1: Select parent cells (top performing strategies)
    parents = SELECT TOP MITOSIS_PARENT_STRATEGIES FROM parent_strategies
              ORDER BY fitness DESC

    all_mutants = []

    FOR parent IN parents:

        # Step 2: Generate MITOSIS_POPULATION_SIZE daughter cells
        FOR i IN range(MITOSIS_POPULATION_SIZE):

            # Step 3: Multi-hit model -- pick 3-7 simultaneous mutations
            n_drivers = RANDOM(MULTI_HIT_MIN_DRIVERS, MULTI_HIT_MAX_DRIVERS)
            mutation_types = RANDOM_SAMPLE(MutationType, n_drivers)

            mutant = DEEP_COPY(parent)
            mutant.mutant_id = UUID()
            mutant.parent_id = parent.id
            mutant.generation = parent.generation + 1
            mutant.driver_count = n_drivers
            mutant.mutation_types = mutation_types

            FOR mutation IN mutation_types:

                IF mutation == ONCOGENE_ACTIVATION:
                    # Amplify a random TE family weight (gain-of-function)
                    # Like RAS stuck in the ON position
                    te_family = RANDOM_CHOICE(ALL_TE_FAMILIES)
                    delta = RANDOM_UNIFORM(0.1, MITOSIS_MUTATION_STRENGTH)
                    mutant.te_weight_deltas[te_family.name] += delta

                IF mutation == SUPPRESSOR_DELETION:
                    # Reduce or eliminate a gate/filter strength
                    # Like losing p53 -- removing a checkpoint
                    gate_to_weaken = RANDOM_CHOICE([
                        "confidence_threshold",
                        "regime_sensitivity",
                        "crispr_sensitivity",
                        "toxoplasma_sensitivity",
                    ])
                    mutant[gate_to_weaken] *= RANDOM_UNIFORM(0.3, 0.8)

                IF mutation == METABOLIC_SHIFT:
                    # Warburg effect: trade precision for speed
                    mutant.hold_time_mult *= RANDOM_UNIFORM(0.5, 1.5)
                    mutant.tp_ratio *= RANDOM_UNIFORM(0.8, 1.3)

                IF mutation == RECEPTOR_MUTATION:
                    # Alter sensitivity to specific market signals
                    te_family = RANDOM_CHOICE(ALL_TE_FAMILIES)
                    mutant.te_weight_deltas[te_family.name] *= RANDOM_UNIFORM(-0.5, 2.0)

                IF mutation == TELOMERE_EXTENSION:
                    # Remove decay/expiry mechanisms
                    # This is evaluated later -- marked as a candidate
                    pass

            all_mutants.APPEND(mutant)

    LOG f"CANCER: Mitosis complete -- {len(all_mutants)} mutants from "
        f"{len(parents)} parent cells, {MULTI_HIT_MIN_DRIVERS}-"
        f"{MULTI_HIT_MAX_DRIVERS} driver mutations each"

    RETURN all_mutants

--------------------------------------------------------------------
PHASE 2: TUMOR SUPPRESSOR BYPASS (Checkpoint Removal -- Simulation)
--------------------------------------------------------------------

ON bypass_checkpoints(mutants[], symbol, bars):
    #
    # CRITICAL: This runs in SIMULATION only.
    # We disable the immune system (CRISPR, Toxoplasma, regime detection)
    # to see what the mutants would do without constraints.
    #
    # Like p53 deletion: the cell divides even with damaged DNA.
    # We want to see which "damaged" strategies actually produce alpha.
    #
    # Time limit: BYPASS_MAX_DURATION_SEC (default 1 hour)
    # SL remains sacred -- even cancer cells respect MAX_LOSS_DOLLARS
    #

    start_time = NOW
    simulated_results = []

    FOR mutant IN mutants:

        IF (NOW - start_time).seconds > BYPASS_MAX_DURATION_SEC:
            LOG "CANCER: Bypass time limit reached, stopping simulation"
            BREAK

        # Create a simulation context with disabled checkpoints
        sim_context = SimulationContext(
            crispr_enabled = NOT BYPASS_CRISPR_DISABLED,
            toxoplasma_enabled = NOT BYPASS_TOXOPLASMA_DISABLED,
            regime_detection = NOT BYPASS_REGIME_DISABLED,
            confidence_threshold = BYPASS_CONFIDENCE_FLOOR,
            # SL REMAINS ENFORCED (from config_loader)
            max_loss_dollars = MAX_LOSS_DOLLARS,
        )

        # Run the mutant through historical data
        result = simulate_strategy(
            mutant = mutant,
            bars = bars,
            context = sim_context,
        )

        mutant.win_rate = result.win_rate
        mutant.profit_factor = result.profit_factor
        mutant.total_trades = result.total_trades
        mutant.sharpe_ratio = result.sharpe_ratio
        mutant.max_drawdown = result.max_drawdown

        # Composite fitness (same formula as ETARE genetic algorithm)
        mutant.fitness_score = (
            result.profit_factor * 0.30
            + result.sharpe_ratio * 0.25
            + result.win_rate * 0.25
            + (1.0 - result.max_drawdown) * 0.20
        )

        simulated_results.APPEND(mutant)

    LOG f"CANCER: Bypass complete -- simulated {len(simulated_results)} mutants"
    RETURN simulated_results

--------------------------------------------------------------------
PHASE 3: WARBURG ACCELERATION (GPU Batch Quantum Processing)
--------------------------------------------------------------------

ON warburg_quantum_batch(mutants[], bars):
    #
    # The Warburg effect: cancer cells use glycolysis even with oxygen.
    # Inefficient per unit, but 10-100x faster throughput.
    #
    # We run quantum circuits in GPU-accelerated batches using DirectML.
    # Reduced shots (2048 vs 8192) for speed. More mutations evaluated
    # per second, even if each evaluation is noisier.
    #

    import torch
    import torch_directml
    device = torch_directml.device()  # AMD RX 6800 XT

    # Pre-compute features for all mutants in batch
    batched_features = []
    FOR mutant IN mutants:
        # Apply TE weight deltas to the standard feature vector
        base_activations = te_engine.compute_all_activations(bars)
        modified = []
        FOR act IN base_activations:
            weight = 1.0 + mutant.te_weight_deltas.GET(act.te, 0.0)
            modified.APPEND(act.strength * weight)
        batched_features.APPEND(modified)

    # Convert to GPU tensor
    feature_tensor = torch.tensor(batched_features, dtype=torch.float32, device=device)

    # Process in batches of WARBURG_BATCH_SIZE
    all_quantum_features = []
    FOR batch_start IN range(0, len(mutants), WARBURG_BATCH_SIZE):
        batch = feature_tensor[batch_start : batch_start + WARBURG_BATCH_SIZE]

        # Run quantum circuits with reduced shots for speed
        quantum_results = []
        FOR features IN batch.cpu().numpy():
            # Build quantum circuit (33 qubits, one per TE family)
            qc = QuantumCircuit(N_QUBITS, N_QUBITS)
            angles = normalize_to_pi(features)

            FOR i IN range(N_QUBITS):
                qc.ry(angles[i], i)

            # Entanglement ring
            FOR i IN range(N_QUBITS - 1):
                qc.cz(i, i + 1)
            qc.cz(N_QUBITS - 1, 0)

            qc.measure_all()

            job = AerSimulator().run(qc, shots=WARBURG_SHOTS_FAST)
            counts = job.result().get_counts()

            # Extract quantum features
            probs = counts_to_probabilities(counts, N_QUBITS)
            entropy = shannon_entropy(probs)
            dominant = max(probs)
            significant = sum(1 for p in probs if p > 0.03)
            variance = np.var(probs)

            quantum_results.APPEND([entropy, dominant, significant, variance])

        all_quantum_features.EXTEND(quantum_results)

    # Attach quantum features to mutants
    FOR mutant, qf IN zip(mutants, all_quantum_features):
        mutant.quantum_entropy = qf[0]
        mutant.quantum_dominant = qf[1]
        mutant.quantum_significant = qf[2]
        mutant.quantum_variance = qf[3]

    LOG f"CANCER: Warburg batch complete -- {len(mutants)} mutants "
        f"processed in {len(mutants)//WARBURG_BATCH_SIZE + 1} GPU batches"

    RETURN mutants

--------------------------------------------------------------------
PHASE 4: ANGIOGENESIS (Resource Allocation to Winning Clusters)
--------------------------------------------------------------------

ON angiogenesis(clusters[]):
    #
    # Tumors that grow past 1-2mm need their own blood supply.
    # Cancer cells secrete VEGF to attract blood vessels.
    #
    # Similarly: mutant clusters that show positive fitness get more
    # resources (more mutation iterations, more backtesting time).
    # Underperforming clusters are starved.
    #

    # Sort clusters by average fitness
    clusters.SORT(key=lambda c: c.avg_fitness, reverse=True)

    n_clusters = len(clusters)
    top_n = int(n_clusters * ANGIO_TOP_PERCENT)
    starve_n = int(n_clusters * ANGIO_STARVE_PERCENT)

    FOR i, cluster IN enumerate(clusters):

        IF i < top_n:
            # TOP 20%: Full angiogenesis -- pump resources in
            cluster.vegf_signal = 1.0
            cluster.blood_supply = 1.0
            cluster.resource_budget *= ANGIO_RESOURCE_MULTIPLIER
            LOG f"CANCER: Angiogenesis -- cluster {cluster.cluster_id[:8]} "
                f"(fitness={cluster.avg_fitness:.3f}) gets {ANGIO_RESOURCE_MULTIPLIER}x resources"

        ELIF i >= n_clusters - starve_n:
            # BOTTOM 50%: Anti-angiogenic -- starve the tumor
            cluster.vegf_signal = 0.0
            cluster.blood_supply = 0.0
            cluster.resource_budget = 0
            # Mark all mutants in cluster as dead (apoptosis)
            FOR mutant IN cluster.mutants:
                mutant.is_alive = FALSE
            LOG f"CANCER: Apoptosis -- cluster {cluster.cluster_id[:8]} "
                f"(fitness={cluster.avg_fitness:.3f}) starved"

        ELSE:
            # MIDDLE 30%: Maintenance -- survive but don't grow
            cluster.vegf_signal = 0.3
            cluster.blood_supply = 0.3
            cluster.resource_budget = max(1, cluster.resource_budget // 2)

    RETURN clusters

--------------------------------------------------------------------
PHASE 5: METASTASIS (Cross-Symbol Strategy Spread)
--------------------------------------------------------------------

ON metastasis(survivors[], all_symbols[], bars_by_symbol):
    #
    # The hallmark of malignancy: spread to distant organs.
    #
    # A winning mutant on BTCUSD "metastasizes" to XAUUSD, ETHUSD, etc.
    # But not all organs are compatible -- the "seed and soil" test
    # checks if the mutant can thrive in the target tissue.
    #

    metastasis_log = []

    FOR mutant IN survivors:

        IF mutant.fitness_score < METASTASIS_MIN_FITNESS:
            CONTINUE  # Too weak to metastasize

        # Attempt EMT (Epithelial-Mesenchymal Transition)
        # The mutant must be adaptable enough to survive migration
        adaptability = mutant.sharpe_ratio / (mutant.max_drawdown + 0.01)
        IF adaptability < 1.0:
            CONTINUE  # Cannot survive the bloodstream (too fragile)

        colonized_count = 0

        FOR target_symbol IN all_symbols:

            IF target_symbol == mutant.parent_symbol:
                CONTINUE  # Already there

            IF colonized_count >= METASTASIS_MAX_SYMBOLS:
                BREAK  # Reached colonization limit

            # SEED AND SOIL TEST: can this mutant thrive on the target?
            target_bars = bars_by_symbol[target_symbol][-METASTASIS_SOIL_TEST_BARS:]

            soil_result = simulate_strategy(
                mutant = mutant,
                bars = target_bars,
                context = SimulationContext(
                    # Full defenses ON for soil test
                    crispr_enabled = TRUE,
                    toxoplasma_enabled = TRUE,
                    regime_detection = TRUE,
                    confidence_threshold = CONFIDENCE_THRESHOLD,
                    max_loss_dollars = MAX_LOSS_DOLLARS,
                ),
            )

            IF soil_result.win_rate >= METASTASIS_COLONIZE_WR AND soil_result.total_trades >= 10:
                # SUCCESSFUL COLONIZATION
                mutant.metastasized_to.APPEND(target_symbol)
                colonized_count += 1

                metastasis_log.APPEND({
                    "mutant_id": mutant.mutant_id,
                    "source_symbol": mutant.parent_symbol,
                    "target_symbol": target_symbol,
                    "soil_wr": soil_result.win_rate,
                    "soil_pf": soil_result.profit_factor,
                    "soil_trades": soil_result.total_trades,
                    "timestamp": NOW,
                })

                LOG f"CANCER: Metastasis! {mutant.mutant_id[:8]} colonized "
                    f"{target_symbol} (WR={soil_result.win_rate:.2%}, "
                    f"PF={soil_result.profit_factor:.2f})"
            ELSE:
                LOG f"CANCER: Metastasis rejected -- {mutant.mutant_id[:8]} "
                    f"failed soil test on {target_symbol} "
                    f"(WR={soil_result.win_rate:.2%})"

    STORE metastasis_log -> cancer_db.metastasis_log
    RETURN survivors

--------------------------------------------------------------------
PHASE 6: TELOMERASE (Immortality for Winners)
--------------------------------------------------------------------

ON telomerase_activation(survivors[]):
    #
    # Normal cells die after ~50 divisions (Hayflick limit).
    # Cancer cells reactivate TERT, adding telomeric repeats back.
    # Result: unlimited replicative potential.
    #
    # We grant "immortality" to mutants that pass the threshold:
    # they are promoted to the permanent strategy pool with no expiry.
    #

    promoted = []

    FOR mutant IN survivors:

        IF NOT mutant.is_alive:
            CONTINUE

        IF mutant.total_trades < TELOMERASE_MIN_TRADES:
            CONTINUE  # Not enough evidence

        IF (mutant.win_rate >= TELOMERASE_PROMOTION_WR
                AND mutant.profit_factor >= TELOMERASE_PROMOTION_PF):

            mutant.has_telomerase = TRUE
            mutant.promoted_at = NOW
            promoted.APPEND(mutant)

            LOG f"CANCER: Telomerase activated! {mutant.mutant_id[:8]} "
                f"is IMMORTAL (WR={mutant.win_rate:.2%}, "
                f"PF={mutant.profit_factor:.2f}, "
                f"trades={mutant.total_trades}, "
                f"colonized={len(mutant.metastasized_to)} symbols)"

    LOG f"CANCER: Telomerase -- {len(promoted)}/{len(survivors)} promoted to immortal"
    RETURN promoted

--------------------------------------------------------------------
PHASE 7: IMMUNE CHECKPOINT (Re-enable Defenses and Validate)
--------------------------------------------------------------------

ON immune_checkpoint(promoted_mutants[], bars_by_symbol):
    #
    # The final test. Re-enable ALL defenses:
    # - CRISPR-Cas adaptive memory (block known losers)
    # - Toxoplasma regime detection (detect hijacking)
    # - Full confidence threshold (from config_loader)
    # - Full regime detection
    #
    # Only mutants that STILL perform with defenses on are deployed.
    # This is immunotherapy -- PD-1/PD-L1 checkpoint inhibitors that
    # allow the immune system to recognize and kill cancer cells.
    # But in our case, we WANT the immune system to work.
    # Mutants that can survive immune scrutiny are truly exceptional.
    #

    validated = []

    FOR mutant IN promoted_mutants:

        # Test on primary symbol
        primary_bars = bars_by_symbol[mutant.parent_symbol][-IMMUNE_CHECKPOINT_BARS:]

        checkpoint_result = simulate_strategy(
            mutant = mutant,
            bars = primary_bars,
            context = SimulationContext(
                crispr_enabled = TRUE,
                toxoplasma_enabled = TRUE,
                regime_detection = TRUE,
                confidence_threshold = CONFIDENCE_THRESHOLD,  # from config_loader
                max_loss_dollars = MAX_LOSS_DOLLARS,          # from config_loader
            ),
        )

        IF (checkpoint_result.win_rate >= IMMUNE_CHECKPOINT_WR
                AND checkpoint_result.profit_factor >= IMMUNE_CHECKPOINT_PF
                AND checkpoint_result.total_trades >= 10):

            validated.APPEND(mutant)

            # Write to TE Domestication DB as new domesticated pattern
            active_tes = GET_ACTIVE_TES_FROM_MUTANT(mutant)
            domestication_db.PROMOTE_PATTERN(
                active_tes = active_tes,
                win_rate = checkpoint_result.win_rate,
                boost_factor = 1.0 + 0.30 * sigmoid(-15 * (checkpoint_result.win_rate - 0.65)),
                source = "CANCER_CELL",
            )

            cancer_db.INSERT INTO promotion_log (
                mutant_id, parent_id, symbol, win_rate, profit_factor,
                sharpe_ratio, total_trades, metastasized_to,
                driver_count, mutation_types, promoted_at
            )

            LOG f"CANCER: IMMUNE CHECKPOINT PASSED! {mutant.mutant_id[:8]} "
                f"validated with defenses ON "
                f"(WR={checkpoint_result.win_rate:.2%}, "
                f"PF={checkpoint_result.profit_factor:.2f})"
        ELSE:
            LOG f"CANCER: Immune checkpoint KILLED {mutant.mutant_id[:8]} "
                f"(WR={checkpoint_result.win_rate:.2%} -- defenses caught it)"

    LOG f"CANCER: Immune checkpoint -- {len(validated)}/{len(promoted_mutants)} "
        f"survived with full defenses"

    RETURN validated

--------------------------------------------------------------------
THE LOOP (Full Cancer Simulation Cycle)
--------------------------------------------------------------------

ON run_cancer_simulation(symbols, bars_by_symbol, parent_strategies):
    #
    # The full multi-phase cancer simulation.
    # Called periodically (e.g., daily, or on demand).
    # Runs entirely in simulation -- NO live trades.
    #

    LOG "================================================================"
    LOG "CANCER CELL SIMULATION: INITIATING"
    LOG f"  Symbols: {symbols}"
    LOG f"  Parent strategies: {len(parent_strategies)}"
    LOG f"  Population target: {MITOSIS_POPULATION_SIZE * MITOSIS_PARENT_STRATEGIES}"
    LOG "================================================================"

    run_start = NOW
    all_validated = []

    FOR symbol IN symbols:

        bars = bars_by_symbol[symbol]
        LOG f"\n--- CANCER: Processing {symbol} ---"

        # Phase 1: MITOSIS
        mutants = mitosis_cycle(parent_strategies, symbol, bars)

        # Phase 3: WARBURG (run quantum BEFORE simulation for TE-aware eval)
        mutants = warburg_quantum_batch(mutants, bars)

        # Cluster mutants by parent (for angiogenesis)
        clusters = group_mutants_by_parent(mutants)

        # Phase 1.5: Iterative mutation with angiogenesis
        FOR gen IN range(MITOSIS_GENERATIONS):

            # Phase 2: TUMOR SUPPRESSOR BYPASS (simulate without defenses)
            FOR cluster IN clusters:
                IF cluster.resource_budget > 0:
                    cluster.mutants = bypass_checkpoints(
                        cluster.mutants, symbol, bars
                    )
                    cluster.avg_fitness = MEAN(m.fitness_score
                                               FOR m IN cluster.mutants)

            # Phase 4: ANGIOGENESIS (allocate resources)
            clusters = angiogenesis(clusters)

            # Spawn new mutations from surviving clusters
            FOR cluster IN clusters:
                IF cluster.resource_budget > 0:
                    new_mutants = mitosis_cycle(
                        [m FOR m IN cluster.mutants WHERE m.is_alive],
                        symbol, bars
                    )
                    cluster.mutants.EXTEND(new_mutants)
                    cluster.resource_budget -= 1

        # Collect all surviving mutants
        survivors = [m FOR cluster IN clusters
                       FOR m IN cluster.mutants
                       WHERE m.is_alive]

        LOG f"CANCER: {symbol} -- {len(survivors)} survivors from "
            f"{MITOSIS_POPULATION_SIZE * MITOSIS_PARENT_STRATEGIES} initial mutants"

        # Phase 5: METASTASIS (spread to other symbols)
        survivors = metastasis(survivors, symbols, bars_by_symbol)

        # Phase 6: TELOMERASE (grant immortality)
        promoted = telomerase_activation(survivors)

        # Phase 7: IMMUNE CHECKPOINT (final validation with defenses ON)
        validated = immune_checkpoint(promoted, bars_by_symbol)

        all_validated.EXTEND(validated)

    # Store run summary
    cancer_db.INSERT INTO run_history (
        run_start = run_start,
        run_end = NOW,
        symbols = JSON(symbols),
        total_mutants_generated = MITOSIS_POPULATION_SIZE * MITOSIS_PARENT_STRATEGIES * len(symbols),
        total_survivors = len(all_validated),
        promoted_to_live = len(all_validated),
        avg_fitness = MEAN(m.fitness_score FOR m IN all_validated) IF all_validated ELSE 0,
        avg_win_rate = MEAN(m.win_rate FOR m IN all_validated) IF all_validated ELSE 0,
        metastasis_count = SUM(len(m.metastasized_to) FOR m IN all_validated),
    )

    LOG "================================================================"
    LOG f"CANCER CELL SIMULATION: COMPLETE"
    LOG f"  Duration: {NOW - run_start}"
    LOG f"  Mutants generated: {MITOSIS_POPULATION_SIZE * MITOSIS_PARENT_STRATEGIES * len(symbols)}"
    LOG f"  Survivors: {len(all_validated)}"
    LOG f"  Immune checkpoint pass rate: {len(all_validated) / max(1, MITOSIS_POPULATION_SIZE * len(symbols)) * 100:.1f}%"
    IF all_validated:
        LOG f"  Avg WR: {MEAN(m.win_rate FOR m IN all_validated):.2%}"
        LOG f"  Avg PF: {MEAN(m.profit_factor FOR m IN all_validated):.2f}"
        LOG f"  Metastasized symbols: {SUM(len(m.metastasized_to) FOR m IN all_validated)}"
    LOG "================================================================"

    RETURN all_validated

--------------------------------------------------------------------
INTEGRATION POINTS
--------------------------------------------------------------------

INTEGRATION with TEQA v3.0:
    - Cancer Cell reads TE family definitions from ALL_TE_FAMILIES (33 families)
    - Mutates TE weights as "oncogene activation" (gain-of-function)
    - Runs modified TE activations through the quantum encoder
    - Writes validated mutant TE patterns to domestication DB

INTEGRATION with TE Domestication:
    - Validated mutants are promoted to domesticated status
    - Their TE weight profiles become new domesticated patterns
    - Source tag "CANCER_CELL" distinguishes them from organic domestication

INTEGRATION with CRISPR-Cas:
    - Phase 2 (bypass): CRISPR is DISABLED in simulation to explore
    - Phase 7 (immune checkpoint): CRISPR is RE-ENABLED for validation
    - Mutants that survive CRISPR with full defenses are truly robust

INTEGRATION with Toxoplasma:
    - Phase 2 (bypass): Toxoplasma is DISABLED in simulation
    - Phase 7 (immune checkpoint): Toxoplasma is RE-ENABLED
    - Mutants must survive regime hijacking detection

INTEGRATION with VDJ Recombination:
    - Cancer Cell can use VDJ-generated antibodies as parent strategies
    - Validated cancer mutants can be fed back as VDJ parents
    - Cross-pollination between the two evolutionary systems

INTEGRATION with BRAIN scripts:
    - BRAIN scripts read cancer_db.promotion_log for new strategies
    - Promoted mutants are loaded as additional signal sources
    - Metastasis information tells BRAIN which symbols to apply mutants to
    - No live trades are placed by cancer simulation itself

INTEGRATION with GPU:
    - Phase 3 (Warburg) uses torch_directml for AMD RX 6800 XT
    - Quantum circuits run in batches of WARBURG_BATCH_SIZE
    - Reduced shots (2048) for throughput over precision
    - Feature tensors pre-computed on GPU before quantum encoding

--------------------------------------------------------------------
GATE INTEGRATION
--------------------------------------------------------------------

Cancer Cell operates OUTSIDE the normal gate pipeline.
It does not add a new gate to Jardine's Gate system.
Instead, it FEEDS RESULTS into existing gates:
    - Gate G10 (Domestication): receives promoted TE patterns
    - Gate G12 (CRISPR-Cas): informed of patterns that survive immune checkpoint

Cancer Cell is a DISCOVERY mechanism, not a trade filter.
It runs offline/periodically and produces new strategies for the
existing pipeline to use.

--------------------------------------------------------------------
SAFETY INVARIANTS
--------------------------------------------------------------------

INVARIANT 1: "Cancer simulation runs in BACKTEST/SIMULATION mode ONLY.
              No live trades are placed by any cancer phase."

INVARIANT 2: "MAX_LOSS_DOLLARS ($1.00) is enforced even in tumor suppressor
              bypass mode. SL is sacred -- cancer cells do not override it."

INVARIANT 3: "All promoted mutants must pass immune checkpoint with FULL
              defenses enabled before deployment to live trading."

INVARIANT 4: "Tumor suppressor bypass has a hard time limit of
              BYPASS_MAX_DURATION_SEC (1 hour). Cannot run indefinitely."

INVARIANT 5: "All trading values come from config_loader. No hardcoding."

--------------------------------------------------------------------
BIOLOGICAL PARALLEL
--------------------------------------------------------------------

BIOLOGICAL PARALLEL:
    parent_strategies        = healthy cells (normal tissue)
    mitosis_cycle()          = cell division with accumulated mutations
    mutation_types           = oncogene/suppressor/metabolic driver mutations
    driver_count             = multi-hit model (Vogelstein's 3-7 driver mutations)
    bypass_checkpoints()     = p53 deletion / RB loss (checkpoint removal)
    warburg_quantum_batch()  = aerobic glycolysis (speed over efficiency)
    angiogenesis()           = VEGF secretion / blood vessel recruitment
    metastasis()             = EMT + intravasation + colonization
    seed_and_soil_test()     = Paget's hypothesis (tissue compatibility)
    telomerase_activation()  = TERT reactivation (immortality)
    immune_checkpoint()      = PD-1/PD-L1 checkpoint (immune re-engagement)
    apoptosis                = programmed cell death of failed mutants
    cancer_db                = tumor registry (tracking all mutations)
    fitness_score            = clonal fitness (growth advantage)
    is_alive                 = survived selection pressure
    has_telomerase           = unlimited replicative potential
    vegf_signal              = resource demand signal
    blood_supply             = allocated compute/capital

CONVERGENCE:
    From ~2000 initial mutants per symbol, expect:
    - ~400 survive angiogenesis (top 20% get resources)
    - ~100 reach telomerase threshold (5% of initial)
    - ~20-40 pass immune checkpoint (1-2% of initial)
    - ~5-15 successfully metastasize to other symbols

    The surviving 1-2% represent strategies that:
    1. Found genuine edges (not noise)
    2. Work across multiple market regimes (bypassed filters AND survived checkpoint)
    3. Can spread to multiple symbols (robust, not symbol-specific)
    4. Survive the full immune system (CRISPR + Toxoplasma + regime detection)

    These are the "cancer" -- strategies that grow without the normal
    constraints because they have found REAL alpha, not just escaped
    the safety systems. The immune checkpoint ensures only true positives
    survive.

DATABASES:
    cancer_cell.db           <- mutants, clusters, metastasis, promotions
    teqa_domestication.db    <- receives promoted TE patterns
    crispr_cas.db            <- informed of immune-checkpoint survivors
    toxoplasma_infection.db  <- infection context for bypass evaluation

FILES:
    cancer_cell.py           -> CancerCellEngine class (main implementation)
    ALGORITHM_CANCER_CELL.py -> this specification
    teqa_v3_neural_te.py     -> TE family definitions, activation engine
    config_loader.py         -> MAX_LOSS_DOLLARS, CONFIDENCE_THRESHOLD
    BRAIN_ATLAS.py           -> reads promotion_log for new strategies
"""
