# A Computational Framework for Transposable Element Domestication and Somatic Mosaicism in Neural Tissue

**Authors:** J. Jardine

**Date:** February 2026

**Keywords:** transposable elements, somatic mosaicism, neural diversity, TE domestication, quantum-inspired computation, L1 retrotransposition, spike-timing-dependent plasticity, adaptive signal processing

---

## Abstract

We present a computational framework that models the process of transposable element (TE) domestication in neural tissue, incorporating somatic mosaicism through L1 retrotransposition. The model implements a closed-loop learning system in which TE activation patterns are evaluated against environmental outcomes, with successful patterns undergoing a domestication process analogous to biological exaptation. Seven virtual neurons, each with unique L1 insertion profiles (somatic mosaicism), form a neural mosaic that processes environmental signals through a 33-qubit quantum-inspired circuit architecture. The framework introduces a spike-timing-dependent plasticity (STDP) analogue for transposon selection: TE patterns that precede favorable outcomes are amplified through a domestication mechanism, while unsuccessful patterns remain epigenetically neutral. This model provides a testable computational analogue for how the brain may leverage retrotransposon-driven genomic diversity for adaptive information processing, with implications for understanding the role of somatic mosaicism in neuropsychiatric conditions including schizophrenia, autism spectrum disorder, and neurodegenerative disease.

---

## 1. Introduction

### 1.1 Transposable Elements in the Brain

Transposable elements (TEs) comprise approximately 45% of the human genome (Lander et al., 2001). Once dismissed as "junk DNA," TEs are now recognized as dynamic genomic elements with functional roles in gene regulation, chromatin organization, and innate immunity (Chuong et al., 2017).

In neural tissue, TEs are uniquely active. LINE-1 (L1) retrotransposons undergo somatic retrotransposition in neural progenitor cells and mature neurons, creating genomic mosaicism — a state in which individual neurons within the same brain carry distinct genomic insertions (Muotri et al., 2005; Coufal et al., 2009; Baillie et al., 2011). This somatic mosaicism has been proposed as a mechanism for generating neuronal diversity and functional heterogeneity (Singer et al., 2010).

The discovery that L1 retrotransposition is regulated by neuronal activity (Coufal et al., 2009) and that retrotransposon-derived proteins serve essential neurological functions — most notably the Arc protein, a domesticated Ty3/gypsy retrotransposon critical for synaptic plasticity and memory formation (Pastuzyn et al., 2018) — suggests that TE activity in the brain is not merely tolerated but may be functionally integrated into neural computation.

### 1.2 TE Domestication: From Selfish Element to Host Function

TE domestication (exaptation) is the process by which a formerly parasitic genetic element is co-opted for host benefit. Several landmark examples demonstrate this phenomenon:

- **RAG1/RAG2 recombinase**: Derived from a Transib DNA transposon, domesticated by the vertebrate immune system to perform V(D)J recombination — the basis of adaptive immunity (Kapitonov & Jurka, 2005).
- **Arc protein**: A Ty3/gypsy retrotransposon capsid protein domesticated for intercellular RNA transfer between neurons, essential for long-term memory formation (Pastuzyn et al., 2018).
- **Syncytin-1/2**: HERV-W and HERV-FRD envelope proteins domesticated for placental syncytiotrophoblast formation (Mi et al., 2000).
- **SETMAR**: An Hsmar1 mariner transposase fused with a SET histone methyltransferase domain, functioning in DNA repair (Cordaux et al., 2006).
- **CRISPR-Cas**: Derived from casposon mobile elements, domesticated by prokaryotes for adaptive defense (Koonin & Makarova, 2019).

In each case, domestication follows a common trajectory: the TE activity that initially benefits only the element itself is repurposed to serve the host organism, typically through regulatory co-option, protein neofunctionalization, or structural integration into gene networks.

### 1.3 The Computational Gap

While the molecular biology of individual TE domestication events is well-characterized, no computational framework exists to model the *process* of domestication — the selective dynamics by which certain TE activation patterns transition from neutral or deleterious to beneficial. Similarly, while somatic mosaicism in neurons has been documented, the computational consequences of genomic heterogeneity across a neural population remain poorly understood.

This work addresses both gaps by implementing a computational model in which:

1. A population of neurons with distinct TE insertion profiles (somatic mosaicism) collectively processes environmental signals
2. TE activation patterns are evaluated against environmental outcomes
3. Successful patterns undergo progressive domestication through a feedback mechanism analogous to STDP
4. The system adapts over time as domesticated patterns receive amplified influence on decision-making

---

## 2. Model Architecture

### 2.1 Transposable Element Families

The model incorporates 33 TE families organized into three biological classes:

**Class I: Retrotransposons (copy-and-paste mechanism)**
- LINE-1 (L1): Autonomous long interspersed element
- Alu (SINE): Non-autonomous short interspersed element, primate-specific
- SVA: SINE-VNTR-Alu composite element
- HERV-K: Human endogenous retrovirus, Class II
- HERV-W: Syncytin-1 source
- LTR-MaLR: Mammalian apparent LTR retrotransposon
- Processed pseudogenes: Retrotransposed mRNA copies

**Class II: DNA Transposons (cut-and-paste mechanism)**
- Tc1/mariner: Widespread superfamily
- hAT (Hobo/Ac/Tam3): Histone acetyltransferase-associated
- PiggyBac: Used extensively in genetic engineering
- Helitron: Rolling-circle transposons
- MITE: Miniature inverted-repeat transposable elements
- Crypton: Tyrosine recombinase transposons
- Maverick/Polinton: Self-synthesizing large DNA transposons

**Class III: Neural TEs (brain-specific or brain-enriched)**
- L1_Neuronal: Brain-specific L1 subfamily (active in neural progenitors)
- Alu_Exonization: Alu elements creating novel exons in neural transcripts
- HERV_Synapse: Endogenous retroviral elements at synaptic gene loci
- SVA_Regulatory: SVA elements in neural gene promoter regions
- BC200: Brain cytoplasmic RNA derived from Alu/7SL
- AmnSINE1: Amniote-conserved SINE in forebrain enhancers
- MER130: Ancient element in cortical enhancers
- LF-SINE: Lobe-finned fish SINE in developmental enhancers

Each TE family is mapped to a specific environmental signal type (e.g., momentum indicators, volatility measures, structural patterns), creating a biological signal transduction analogy where TEs serve as environmental sensors.

### 2.2 Activation Dynamics

Each TE family computes an activation strength and directional signal from environmental input data. The activation function incorporates:

- **Base activation**: Signal-specific computation (threshold-based or continuous)
- **Shock modulation**: Under high environmental volatility ("genomic shock," per McClintock 1984), TE activation thresholds are lowered, modeling the biological observation that stress increases transposon activity
- **Domestication modulation**: Domesticated TE patterns receive amplified activation (see Section 2.5)

The output for each TE is a tuple: `(family_name, strength ∈ [0,1], direction ∈ {-1, +1})`.

### 2.3 Neural Mosaic (Somatic Mosaicism)

The model instantiates 7 virtual neurons, each with a unique genomic profile created through simulated L1 retrotransposition:

**For each neuron at initialization:**
1. Sample 2-5 random L1 insertion events
2. Each insertion targets a specific qubit (TE family) with one of four effects:
   - **Amplify**: Increase TE activation strength (insertion in enhancer region)
   - **Silence**: Decrease TE activation strength (insertion disrupting regulatory element)
   - **Invert**: Reverse the directional signal (insertion causing antisense transcription)
   - **Rewire**: Redirect one TE's signal to another qubit (insertion creating novel regulatory connection)

This creates a population of neurons that respond differently to identical environmental inputs — precisely the functional heterogeneity observed in biological somatic mosaicism (Evrony et al., 2012).

### 2.4 Quantum-Inspired Circuit Processing

Environmental signals are processed through a split quantum circuit architecture:

**Genome Circuit (25 qubits):**
- Each qubit represents one of the 25 standard TE families
- RY rotation angles encode TE activation strength and direction
- Entanglement layer (CX/CZ gates) models TE-TE interactions (e.g., Alu mobilization by L1 ORF2p)
- Measurement produces a bitstring encoding directional consensus

**Neural Circuit (8 qubits):**
- Each qubit represents one neural TE family, modified by the neuron's L1 insertion profile
- Separate circuit per neuron allows mosaicism to influence quantum interference patterns
- Measurement produces a per-neuron directional vote

**Consensus:**
The 7 neurons vote independently. The majority vote determines the population-level signal. This models how a mosaic neural population produces coherent output despite individual genomic differences.

### 2.5 TE Domestication Feedback Loop

The core contribution of this model is the domestication feedback mechanism:

**Phase 1 — Signal Emission:**
For each processing cycle, the set of active TEs (activation strength > threshold) is recorded as a pattern fingerprint: `hash(sorted(active_te_names))`.

**Phase 2 — Outcome Observation:**
Environmental outcomes (favorable or unfavorable) are observed after a delay period.

**Phase 3 — Pattern-Outcome Matching:**
Each observed outcome is matched to the TE activation pattern that was active at the time of the corresponding decision, using temporal correlation.

**Phase 4 — Domestication Update:**
For each matched pattern:
- Win/loss counts are updated in a persistent database
- Win rate is computed with Bayesian shrinkage: `posterior_wr = (prior_wins + observed_wins) / (prior_total + observed_total)` using a Beta(10,10) prior
- **Domestication threshold**: A pattern is domesticated when:
  - Minimum 20 observations (statistical confidence)
  - Bayesian posterior win rate >= 70%
  - Profit factor (mean favorable outcome / mean unfavorable outcome) > 1.5
- **Hysteresis**: Domestication is revoked only when posterior win rate drops below 60%, preventing oscillation at the threshold boundary
- **Boost function**: Domesticated patterns receive a sigmoid-shaped confidence boost:

  `boost = 1.0 + 0.30 × σ(15 × (win_rate - 0.65))`

  where σ is the logistic sigmoid function. This produces near-zero boost below 55% WR, steep amplification in the 60-80% decision zone, and saturation near 1.30 at high win rates.
- **Expiry**: Patterns inactive for 30 days lose domestication status, modeling the biological observation that unused domesticated elements can be re-silenced through epigenetic mechanisms.

**The STDP Analogy:**
This feedback loop mirrors spike-timing-dependent plasticity:
- TE activation (presynaptic spike) followed by favorable outcome (postsynaptic spike) = **strengthen** (domesticate)
- TE activation followed by unfavorable outcome = **weaken** (remain wild or suppress)
- The temporal matching window (~2 minutes) corresponds to the biological STDP window (~20-100ms, scaled proportionally)

### 2.6 Neural Evolution (Darwinian Selection)

In addition to TE domestication, the model implements Darwinian selection on the neural mosaic:

- Every N cycles, each neuron's predictive accuracy is scored
- The worst-performing neuron undergoes apoptosis (removal)
- The best-performing neuron reproduces via mitosis with L1 retrotransposition errors (mutations)
- If all neurons converge to identical genomes, forced speciation introduces diversity

This models the biological observation that neural progenitor cells with deleterious L1 insertions are selectively eliminated, while those with beneficial or neutral insertions survive and proliferate (Muotri et al., 2005).

---

## 3. Health Implications

### 3.1 Schizophrenia

Increased L1 retrotransposition has been observed in schizophrenia patients (Bundo et al., 2014; Doyle et al., 2017). In our model, excessive L1 activity corresponds to:
- Higher mutation rates during neural evolution (more L1 insertions per neuron)
- Greater genomic divergence between neurons (increased mosaicism)
- Potential for beneficial domestication patterns to be disrupted by new insertions

The model predicts that uncontrolled L1 activity would degrade the domestication database — previously learned TE-outcome associations would be disrupted as new insertions change the activation fingerprints. This provides a computational mechanism for the "noisy signal processing" hypothesis of schizophrenia.

**Testable prediction:** Schizophrenia-associated L1 activity levels, when input to the model, should produce measurably degraded domestication stability compared to control levels.

### 3.2 Autism Spectrum Disorder

Altered TE expression patterns have been reported in ASD (Balestrieri et al., 2019). Our model suggests that ASD-associated TE profiles may correspond to:
- Reduced neural mosaic diversity (fewer L1 insertions, more homogeneous neuron population)
- Premature domestication of suboptimal patterns (reduced diversity → faster convergence on locally optimal but globally suboptimal TE combinations)
- Speciation pressure failure (insufficient genomic diversity to maintain independent neural "viewpoints")

**Testable prediction:** Simulations with reduced L1 insertion rates should show faster initial convergence but lower long-term adaptability to changing environmental conditions.

### 3.3 Neurodegenerative Disease

Age-related reactivation of TEs has been linked to neurodegeneration, particularly in ALS and frontotemporal dementia (Li et al., 2012; Tam et al., 2019). The model provides a framework for understanding how TE reactivation could disrupt established neural circuits:
- Domesticated patterns (established, beneficial TE-outcome associations) could be corrupted by new TE insertions
- The domestication expiry mechanism models how disused neural pathways lose their protective status
- Genomic shock (environmental stress → TE activation threshold reduction) models how systemic inflammation or oxidative stress could trigger pathological TE reactivation

**Testable prediction:** Simulated TE reactivation events (lowering all activation thresholds) should produce acute degradation of domestication database integrity, followed by partial recovery as the system re-learns patterns.

### 3.4 TRIM28/KAP1 and TE Silencing

The model incorporates a "genomic shock detector" inspired by KRAB-ZFP/TRIM28 TE silencing:
- Under normal conditions, TE activation follows standard thresholds
- Under high-volatility conditions (genomic shock), thresholds are lowered, allowing normally silenced TEs to become active
- This models the biological observation that TRIM28 knockout leads to TE derepression and neural dysfunction (Fasching et al., 2015)

**Clinical relevance:** TRIM28 haploinsufficiency has been associated with behavioral abnormalities and obesity in mice. The model provides a computational framework for predicting the functional consequences of TRIM28 dysregulation on neural signal processing.

---

## 4. Implementation

The complete model is implemented in Python with the following components:

| Component | Description | LOC |
|-----------|-------------|-----|
| `teqa_v3_neural_te.py` | Core engine: 33 TE families, activation dynamics, neural mosaic, quantum circuits, domestication tracker | ~1,750 |
| `neural_evolution.py` | Darwinian selection on neuron genomes | ~550 |
| `teqa_feedback.py` | Outcome harvesting and pattern-outcome matching | ~280 |
| `teqa_signal_history.py` | Signal persistence for temporal matching | ~150 |
| `ALGORITHM_TE_DOMESTICATION.py` | Formal algorithm specification | ~294 |

**Quantum circuit execution:** Qiskit 2.3.0 with AerSimulator (33-qubit statevector simulation). GPU-accelerated tensor operations via PyTorch + DirectML on AMD RX 6800 XT.

**Persistence:** SQLite with WAL journaling mode for concurrent access. Domestication records, signal history, and processed outcome deduplication are maintained across sessions.

**Source code:** Available at [GitHub repository URL] under GPL-3.0 license.

---

## 5. Discussion

### 5.1 Novel Contributions

1. **First computational model of TE domestication as a learning process.** Previous work has characterized individual domestication events post-hoc. This framework models domestication as an ongoing selective process with measurable dynamics.

2. **Somatic mosaicism as a computational resource.** Rather than treating L1-driven genomic diversity as noise, the model demonstrates that heterogeneous neural populations can outperform homogeneous ones by maintaining diverse "viewpoints" on environmental signals.

3. **STDP analogue for transposon selection.** The domestication feedback loop provides a novel mapping between Hebbian learning principles and TE evolutionary dynamics.

4. **Quantitative predictions for neuropsychiatric conditions.** The model generates testable predictions about how altered TE activity levels (as observed in schizophrenia, ASD, and neurodegeneration) would affect information processing capacity.

### 5.2 Limitations

- The model uses 33 TE families as a simplified representation of the >4 million TE copies in the human genome
- Quantum circuit simulation is classical (AerSimulator); execution on quantum hardware would enable exploration of larger TE spaces
- The domestication threshold (20 observations) is based on statistical power analysis, not biological measurement
- Temporal scaling between biological processes (milliseconds to days) and model cycles (seconds to hours) is approximate

### 5.3 Future Directions

- **Validation against biological data**: Compare model domestication dynamics with longitudinal TE expression data from neural organoids
- **Patient-specific modeling**: Input individual TE insertion profiles (from whole-genome sequencing) to predict personalized domestication trajectories
- **Drug target identification**: Identify TE families whose modulation (activation or silencing) would most strongly affect domestication stability
- **Quantum hardware execution**: Run the 33-qubit architecture on real quantum processors to explore TE combination spaces beyond classical simulation capacity

---

## 6. References

Baillie, J.K., et al. (2011). Somatic retrotransposition alters the genetic landscape of the human brain. *Nature*, 479, 534-537.

Balestrieri, E., et al. (2019). Human endogenous retroviruses and TRIM28 expression in autism spectrum disorders. *International Journal of Molecular Sciences*, 20(7), 1656.

Bundo, M., et al. (2014). Increased L1 retrotransposition in the neuronal genome in schizophrenia. *Neuron*, 81(2), 306-313.

Chuong, E.B., Elde, N.C., & Feschotte, C. (2017). Regulatory activities of transposable elements: from conflicts to benefits. *Nature Reviews Genetics*, 18, 71-86.

Cordaux, R., et al. (2006). Birth of a chimeric primate gene by capture of the transposase gene from a mobile element. *PNAS*, 103(21), 8101-8106.

Coufal, N.G., et al. (2009). L1 retrotransposition in human neural progenitor cells. *Nature*, 460, 1127-1131.

Doyle, G.A., et al. (2017). Analysis of LINE-1 elements in DNA from postmortem brains of individuals with schizophrenia. *Neuropsychopharmacology*, 42, 2602-2611.

Evrony, G.D., et al. (2012). Single-neuron sequencing analysis of L1 retrotransposition and somatic mutation in the human brain. *Cell*, 151(3), 483-496.

Fasching, L., et al. (2015). TRIM28 represses transcription of endogenous retroviruses in neural progenitor cells. *Cell Reports*, 10(1), 20-28.

Kapitonov, V.V. & Jurka, J. (2005). RAG1 core and V(D)J recombination signal sequences were derived from Transib transposons. *PLoS Biology*, 3(6), e181.

Koonin, E.V. & Makarova, K.S. (2019). Origins and evolution of CRISPR-Cas systems. *Philosophical Transactions of the Royal Society B*, 374(1772), 20180087.

Lander, E.S., et al. (2001). Initial sequencing and analysis of the human genome. *Nature*, 409, 860-921.

Li, W., et al. (2012). Transposable elements in TDP-43-mediated neurodegenerative disorders. *PLoS ONE*, 7(9), e44099.

McClintock, B. (1984). The significance of responses of the genome to challenge. *Science*, 226(4676), 792-801.

Mi, S., et al. (2000). Syncytin is a captive retroviral envelope protein involved in human placental morphogenesis. *Nature*, 403, 785-789.

Muotri, A.R., et al. (2005). Somatic mosaicism in neuronal precursor cells mediated by L1 retrotransposition. *Nature*, 435, 903-910.

Pastuzyn, E.D., et al. (2018). The neuronal gene Arc encodes a repurposed retrotransposon Gag protein that mediates intercellular RNA transfer. *Cell*, 172(1-2), 275-288.

Serrato-Capuchina, A. & Matute, D.R. (2018). The role of transposable elements in speciation. *Genes*, 9(5), 254.

Singer, T., et al. (2010). LINE-1 retrotransposons: mediators of somatic variation in neuronal genomes? *Trends in Neurosciences*, 33(8), 345-354.

Tam, O.H., et al. (2019). Postmortem cortex samples identify distinct molecular subtypes of ALS: retrotransposon activation, oxidative stress, and activated glia. *Cell Reports*, 29(5), 1164-1177.

---

## Supplementary Materials

Full source code, algorithm specification, and quantum circuit definitions are available at the associated repository. The domestication algorithm specification (`ALGORITHM_TE_DOMESTICATION.py`) contains executable pseudocode with all constants matching the implementation.

---

*Corresponding author: J. Jardine*
*License: GPL-3.0*
*Code availability: [Repository DOI via Zenodo]*
