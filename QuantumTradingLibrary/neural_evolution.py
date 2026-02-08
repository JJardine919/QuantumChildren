"""
NEURAL MOSAIC EVOLUTION ENGINE
===============================
Darwinian selection on quantum circuit topology.

The 7 neurons in TEQA's NeuralMosaicEngine get random L1 insertions
at creation and never change. This module adds:

  1. SCORING:     After each TEQA cycle, compare each neuron's vote
                  to the actual market move (rolling window accuracy)
  2. SELECTION:   Best neuron reproduces, worst neuron dies
  3. REPRODUCTION: Copy genome + mutate (add/remove/modify L1 insertions)
  4. SPECIATION:   If all neurons converge to same genome, force diversity
  5. PERSISTENCE: Save evolved genomes to JSON so they survive restarts

Biology mapping:
  - Scoring       = natural selection (fitness = prediction accuracy)
  - Reproduction  = mitosis with L1 retrotransposition errors
  - Death         = apoptosis of poorly performing neurons
  - Speciation    = reproductive isolation maintains population diversity
  - Persistence   = epigenetic memory across cell generations

The mosaic gets smarter over time. Every cycle is a generation.

Authors: DooDoo + Claude
Date:    2026-02-08
Version: NEURAL-EVOLUTION-1.0
"""

import json
import math
import os
import random
import logging
import hashlib
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from teqa_v3_neural_te import (
    NeuralMosaicEngine,
    NeuralGenome,
    L1Insertion,
    ALL_TE_FAMILIES,
    N_QUBITS,
    N_QUBITS_GENOME,
    N_QUBITS_NEURAL,
    TEClass,
)

log = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

# Minimum cycles before evolution kicks in (need data to score)
MIN_CYCLES_BEFORE_EVOLUTION = 10

# Rolling window for neuron accuracy scoring
SCORING_WINDOW_SIZE = 50

# Mutation rates
MUTATION_ADD_INSERTION_PROB = 0.30      # Probability of adding a new L1 insertion
MUTATION_REMOVE_INSERTION_PROB = 0.15   # Probability of removing an existing insertion
MUTATION_MODIFY_INSERTION_PROB = 0.40   # Probability of modifying an existing insertion
MUTATION_MAGNITUDE_JITTER = 0.15       # Max change to insertion magnitude per mutation

# Speciation pressure thresholds
GENOME_SIMILARITY_THRESHOLD = 0.85     # If avg pairwise similarity > this, force diversity
DIVERSITY_INJECTION_FRACTION = 0.30    # What fraction of population to replace for diversity
MIN_UNIQUE_GENOMES = 3                 # Force at least this many distinct genome shapes

# Evolution frequency
EVOLVE_EVERY_N_CYCLES = 5             # Run selection/reproduction every N cycles
                                       # (not every cycle -- let neurons prove themselves)

# Persistence
DEFAULT_GENOME_FILE = "evolved_genomes.json"

# Elite protection
ELITE_COUNT = 1                        # Top N neurons are immune from replacement


# ============================================================
# NEURON SCORECARD
# ============================================================

@dataclass
class NeuronScore:
    """Tracks a neuron's prediction accuracy over a rolling window."""
    neuron_id: int
    # Rolling window of (vote, actual_direction) tuples
    predictions: deque = field(default_factory=lambda: deque(maxlen=SCORING_WINDOW_SIZE))
    # Cumulative stats
    total_correct: int = 0
    total_predictions: int = 0
    # Streak tracking
    current_streak: int = 0       # Positive = consecutive correct, negative = consecutive wrong
    best_streak: int = 0
    worst_streak: int = 0
    # Generation tracking
    generation: int = 0           # How many times this neuron has been born/reproduced
    parent_id: Optional[int] = None
    born_at_cycle: int = 0

    @property
    def accuracy(self) -> float:
        """Rolling window accuracy (0.0 to 1.0)."""
        if len(self.predictions) == 0:
            return 0.5  # Prior: assume average until proven otherwise
        correct = sum(1 for vote, actual in self.predictions if vote == actual)
        return correct / len(self.predictions)

    @property
    def n_predictions(self) -> int:
        return len(self.predictions)

    @property
    def has_enough_data(self) -> bool:
        return len(self.predictions) >= MIN_CYCLES_BEFORE_EVOLUTION

    def record(self, vote: int, actual_direction: int):
        """Record a prediction and its outcome."""
        self.predictions.append((vote, actual_direction))
        self.total_predictions += 1
        if vote == actual_direction:
            self.total_correct += 1
            self.current_streak = max(1, self.current_streak + 1)
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = min(-1, self.current_streak - 1)
            self.worst_streak = min(self.worst_streak, self.current_streak)


# ============================================================
# GENOME FINGERPRINTING
# ============================================================

def genome_fingerprint(neuron: NeuralGenome) -> str:
    """
    Create a hashable fingerprint of a neuron's genome.
    Two neurons with identical L1 insertions should have identical fingerprints.
    """
    # Sort insertions by target qubit for deterministic ordering
    insertion_tuples = sorted(
        (ins.target_qubit, ins.effect, ins.rewire_target, round(ins.magnitude, 3))
        for ins in neuron.insertions
    )
    raw = str(insertion_tuples)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def population_fingerprint(neurons: List[NeuralGenome]) -> str:
    """
    Create a hashable fingerprint of the ENTIRE neural mosaic population.
    Changes when ANY neuron's genome is modified (evolution event).
    Used by TEDomesticationTracker to detect topology changes.
    """
    individual_fps = sorted(genome_fingerprint(n) for n in neurons)
    combined = "|".join(individual_fps)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def genome_similarity(a: NeuralGenome, b: NeuralGenome) -> float:
    """
    Compute similarity between two neuron genomes (0.0 = completely different, 1.0 = identical).
    Uses Jaccard similarity on target qubits + effect type overlap.
    """
    set_a = set(
        (ins.target_qubit, ins.effect)
        for ins in a.insertions
    )
    set_b = set(
        (ins.target_qubit, ins.effect)
        for ins in b.insertions
    )
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return intersection / union


# ============================================================
# MUTATION ENGINE
# ============================================================

class MutationEngine:
    """
    Mutates neuron genomes during reproduction.
    Biologically: L1 retrotransposition errors during cell division.
    """

    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random()

    def mutate(self, genome: NeuralGenome) -> NeuralGenome:
        """
        Create a mutated copy of a genome.
        Preserves biological accuracy: L1 insertions preferentially target
        neural_target TEs but can land anywhere with low probability.
        """
        child = deepcopy(genome)

        # 1. Possibly ADD a new L1 insertion
        if self.rng.random() < MUTATION_ADD_INSERTION_PROB:
            self._add_insertion(child)

        # 2. Possibly REMOVE an existing insertion
        if child.insertions and self.rng.random() < MUTATION_REMOVE_INSERTION_PROB:
            self._remove_insertion(child)

        # 3. Possibly MODIFY existing insertions
        for i in range(len(child.insertions)):
            if self.rng.random() < MUTATION_MODIFY_INSERTION_PROB:
                self._modify_insertion(child, i)

        # Rebuild activation_modifiers from insertions
        child.activation_modifiers = self._rebuild_modifiers(child.insertions)

        # Reset vote state (new neuron has no prior)
        child.vote = 0
        child.confidence = 0.0

        return child

    def _add_insertion(self, genome: NeuralGenome):
        """Add a new L1 insertion to the genome."""
        # L1 preferentially targets neural_target TEs (70%) but can hit others (30%)
        if self.rng.random() < 0.70:
            eligible = [
                te.qubit_index for te in ALL_TE_FAMILIES
                if te.neural_target
            ]
        else:
            eligible = list(range(N_QUBITS))

        if not eligible:
            eligible = list(range(N_QUBITS))

        # Don't insert into a qubit that already has an insertion
        existing_targets = {ins.target_qubit for ins in genome.insertions}
        available = [q for q in eligible if q not in existing_targets]
        if not available:
            return  # All eligible qubits already have insertions

        target = self.rng.choice(available)
        effect = self.rng.choice(["enhance", "disrupt", "invert", "rewire"])
        magnitude = self.rng.uniform(0.3, 0.8)

        rewire_target = target
        if effect == "rewire":
            candidates = [q for q in range(N_QUBITS) if q != target]
            rewire_target = self.rng.choice(candidates)

        genome.insertions.append(L1Insertion(
            target_qubit=target,
            effect=effect,
            rewire_target=rewire_target,
            magnitude=magnitude,
        ))

    def _remove_insertion(self, genome: NeuralGenome):
        """Remove a random L1 insertion (L1 excision -- rare in biology but happens)."""
        idx = self.rng.randint(0, len(genome.insertions) - 1)
        genome.insertions.pop(idx)

    def _modify_insertion(self, genome: NeuralGenome, idx: int):
        """Modify an existing insertion's properties."""
        ins = genome.insertions[idx]

        # Choose what to modify
        modification = self.rng.choice(["magnitude", "effect", "rewire"])

        if modification == "magnitude":
            # Jitter the magnitude
            delta = self.rng.uniform(-MUTATION_MAGNITUDE_JITTER, MUTATION_MAGNITUDE_JITTER)
            ins.magnitude = max(0.1, min(0.95, ins.magnitude + delta))

        elif modification == "effect":
            # Change the effect type
            effects = ["enhance", "disrupt", "invert", "rewire"]
            effects.remove(ins.effect)
            ins.effect = self.rng.choice(effects)
            if ins.effect == "rewire":
                candidates = [q for q in range(N_QUBITS) if q != ins.target_qubit]
                ins.rewire_target = self.rng.choice(candidates)

        elif modification == "rewire" and ins.effect == "rewire":
            # Change the rewire target
            candidates = [q for q in range(N_QUBITS) if q != ins.target_qubit]
            ins.rewire_target = self.rng.choice(candidates)

    def _rebuild_modifiers(self, insertions: List[L1Insertion]) -> Dict[int, float]:
        """Rebuild activation_modifiers dict from insertions list."""
        modifiers = {}
        for ins in insertions:
            if ins.effect == "enhance":
                modifiers[ins.target_qubit] = 1.0 + ins.magnitude
            elif ins.effect == "disrupt":
                modifiers[ins.target_qubit] = 1.0 - ins.magnitude * 0.7
            elif ins.effect == "invert":
                modifiers[ins.target_qubit] = -1.0
            elif ins.effect == "rewire":
                modifiers[ins.target_qubit] = 1.0
        return modifiers

    def create_random_genome(self, neuron_id: int) -> NeuralGenome:
        """Create a completely new random genome (for diversity injection)."""
        n_jumps = self.rng.randint(2, 6)

        eligible_targets = [
            te.qubit_index for te in ALL_TE_FAMILIES
            if te.neural_target or self.rng.random() < 0.15
        ]
        targets = self.rng.sample(
            eligible_targets, min(n_jumps, len(eligible_targets))
        )

        insertions = []
        for target in targets:
            effect = self.rng.choice(["enhance", "disrupt", "invert", "rewire"])
            magnitude = self.rng.uniform(0.3, 0.8)
            rewire_target = target
            if effect == "rewire":
                candidates = [q for q in range(N_QUBITS) if q != target]
                rewire_target = self.rng.choice(candidates)
            insertions.append(L1Insertion(
                target_qubit=target,
                effect=effect,
                rewire_target=rewire_target,
                magnitude=magnitude,
            ))

        modifiers = self._rebuild_modifiers(insertions)
        return NeuralGenome(
            neuron_id=neuron_id,
            insertions=insertions,
            activation_modifiers=modifiers,
        )


# ============================================================
# NEURAL EVOLUTION ENGINE
# ============================================================

class NeuralEvolutionEngine:
    """
    Wraps NeuralMosaicEngine with Darwinian evolution.

    After each TEQA cycle:
      1. Record each neuron's vote and the eventual market direction
      2. Every N cycles, run selection/reproduction:
         a. Rank neurons by rolling accuracy
         b. Best neuron reproduces (copy + mutate)
         c. Worst neuron gets replaced by offspring
         d. Check speciation pressure -- force diversity if converged
      3. Persist evolved genomes to disk

    Usage:
        # In TEQAv3Engine.__init__:
        self.evolution = NeuralEvolutionEngine(self.mosaic)

        # After each analyze() call:
        self.evolution.record_votes(actual_direction)

        # The engine handles evolution timing internally.
    """

    def __init__(
        self,
        mosaic: NeuralMosaicEngine,
        genome_file: str = None,
        evolve_every: int = EVOLVE_EVERY_N_CYCLES,
        seed: int = None,
    ):
        self.mosaic = mosaic
        self.evolve_every = evolve_every
        self.rng = random.Random(seed)
        self.mutator = MutationEngine(rng=self.rng)
        self.cycle_count = 0
        self.generation = 0

        # Scoring for each neuron
        self.scores: Dict[int, NeuronScore] = {}
        for neuron in mosaic.neurons:
            self.scores[neuron.neuron_id] = NeuronScore(
                neuron_id=neuron.neuron_id,
                generation=0,
                born_at_cycle=0,
            )

        # Genome persistence
        if genome_file is None:
            self.genome_file = str(
                Path(__file__).parent / DEFAULT_GENOME_FILE
            )
        else:
            self.genome_file = genome_file

        # Evolution history for analytics
        self.evolution_log: List[Dict] = []

        # Try to load persisted genomes
        self._load_genomes()

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def record_votes(self, actual_direction: int) -> Optional[Dict]:
        """
        Record the outcome of the latest TEQA cycle.

        Called AFTER analyze() when we know what the market actually did.
        actual_direction: 1 (went up), -1 (went down), 0 (flat)

        Returns evolution event dict if evolution occurred, else None.
        """
        self.cycle_count += 1

        # Record each neuron's vote vs actual
        for neuron in self.mosaic.neurons:
            score = self.scores.get(neuron.neuron_id)
            if score is None:
                score = NeuronScore(
                    neuron_id=neuron.neuron_id,
                    generation=0,
                    born_at_cycle=self.cycle_count,
                )
                self.scores[neuron.neuron_id] = score
            score.record(neuron.vote, actual_direction)

        # Check if it's time to evolve
        if self.cycle_count % self.evolve_every == 0:
            # Need minimum data before evolving
            enough_data = sum(
                1 for s in self.scores.values() if s.has_enough_data
            )
            if enough_data >= len(self.mosaic.neurons) // 2:
                return self._evolve()

        return None

    def record_trade_outcome(
        self,
        neuron_votes: Dict[int, int],
        actual_direction: int,
    ) -> Optional[Dict]:
        """
        Alternative API: record specific neuron votes from a trade signal
        along with the actual market direction after that signal.

        neuron_votes: {neuron_id: vote (-1, 0, 1), ...}
        actual_direction: what the market actually did

        This is for the feedback loop from teqa_feedback.py.
        """
        self.cycle_count += 1

        for neuron_id, vote in neuron_votes.items():
            score = self.scores.get(neuron_id)
            if score is None:
                score = NeuronScore(
                    neuron_id=neuron_id,
                    generation=0,
                    born_at_cycle=self.cycle_count,
                )
                self.scores[neuron_id] = score
            score.record(vote, actual_direction)

        if self.cycle_count % self.evolve_every == 0:
            enough_data = sum(
                1 for s in self.scores.values() if s.has_enough_data
            )
            if enough_data >= len(self.mosaic.neurons) // 2:
                return self._evolve()

        return None

    def get_leaderboard(self) -> List[Dict]:
        """Get neuron rankings for display/logging."""
        board = []
        for neuron in self.mosaic.neurons:
            score = self.scores.get(neuron.neuron_id)
            fp = genome_fingerprint(neuron)
            board.append({
                "neuron_id": neuron.neuron_id,
                "accuracy": score.accuracy if score else 0.5,
                "n_predictions": score.n_predictions if score else 0,
                "total_correct": score.total_correct if score else 0,
                "streak": score.current_streak if score else 0,
                "best_streak": score.best_streak if score else 0,
                "generation": score.generation if score else 0,
                "n_insertions": len(neuron.insertions),
                "fingerprint": fp,
            })
        board.sort(key=lambda x: x["accuracy"], reverse=True)
        return board

    def get_population_stats(self) -> Dict:
        """Population-level statistics for analytics."""
        accuracies = [
            self.scores[n.neuron_id].accuracy
            for n in self.mosaic.neurons
            if n.neuron_id in self.scores
        ]
        fingerprints = [genome_fingerprint(n) for n in self.mosaic.neurons]
        unique_genomes = len(set(fingerprints))

        # Pairwise similarity
        sims = []
        neurons = self.mosaic.neurons
        for i in range(len(neurons)):
            for j in range(i + 1, len(neurons)):
                sims.append(genome_similarity(neurons[i], neurons[j]))
        avg_similarity = float(np.mean(sims)) if sims else 0.0

        return {
            "cycle": self.cycle_count,
            "generation": self.generation,
            "n_neurons": len(neurons),
            "unique_genomes": unique_genomes,
            "avg_accuracy": float(np.mean(accuracies)) if accuracies else 0.5,
            "best_accuracy": float(max(accuracies)) if accuracies else 0.5,
            "worst_accuracy": float(min(accuracies)) if accuracies else 0.5,
            "accuracy_spread": float(max(accuracies) - min(accuracies)) if accuracies else 0.0,
            "avg_pairwise_similarity": avg_similarity,
            "speciation_pressure": avg_similarity > GENOME_SIMILARITY_THRESHOLD,
            "avg_insertions": float(np.mean([len(n.insertions) for n in neurons])),
        }

    # ----------------------------------------------------------
    # EVOLUTION CORE
    # ----------------------------------------------------------

    def _evolve(self) -> Dict:
        """
        Run one generation of Darwinian selection.

        Steps:
          1. Rank all neurons by accuracy
          2. Protect elite(s)
          3. Best non-elite reproduces -> offspring replaces worst
          4. Check speciation pressure -> diversity injection if needed
          5. Persist genomes
          6. Log evolution event
        """
        self.generation += 1
        event = {
            "generation": self.generation,
            "cycle": self.cycle_count,
            "timestamp": datetime.now().isoformat(),
            "actions": [],
        }

        # Step 1: Rank by accuracy
        ranked = self._rank_neurons()

        if len(ranked) < 3:
            log.warning("Not enough neurons to evolve (need >= 3, have %d)", len(ranked))
            return event

        best_neuron_id = ranked[0][0]
        worst_neuron_id = ranked[-1][0]
        best_accuracy = ranked[0][1]
        worst_accuracy = ranked[-1][1]

        log.info(
            "[EVOLUTION] Gen %d | Best: neuron_%d (%.1f%%) | Worst: neuron_%d (%.1f%%)",
            self.generation,
            best_neuron_id, best_accuracy * 100,
            worst_neuron_id, worst_accuracy * 100,
        )

        # Step 2: Selection + Reproduction
        # Find the best neuron and worst neuron objects
        best_neuron = next(
            (n for n in self.mosaic.neurons if n.neuron_id == best_neuron_id), None
        )
        worst_idx = next(
            (i for i, n in enumerate(self.mosaic.neurons) if n.neuron_id == worst_neuron_id),
            None,
        )

        if best_neuron is not None and worst_idx is not None:
            # Only replace if the accuracy gap is meaningful
            if best_accuracy - worst_accuracy > 0.05:
                # Reproduce: best -> offspring
                offspring = self.mutator.mutate(best_neuron)
                offspring.neuron_id = worst_neuron_id  # Take the dead neuron's slot

                # Replace worst with offspring
                old_fingerprint = genome_fingerprint(self.mosaic.neurons[worst_idx])
                self.mosaic.neurons[worst_idx] = offspring
                new_fingerprint = genome_fingerprint(offspring)

                # Reset scoring for the new neuron
                self.scores[worst_neuron_id] = NeuronScore(
                    neuron_id=worst_neuron_id,
                    generation=self.generation,
                    parent_id=best_neuron_id,
                    born_at_cycle=self.cycle_count,
                )

                event["actions"].append({
                    "type": "SELECTION",
                    "parent": best_neuron_id,
                    "parent_accuracy": best_accuracy,
                    "replaced": worst_neuron_id,
                    "replaced_accuracy": worst_accuracy,
                    "old_fingerprint": old_fingerprint,
                    "new_fingerprint": new_fingerprint,
                    "n_insertions": len(offspring.insertions),
                })

                log.info(
                    "[EVOLUTION] SELECTION: neuron_%d (%.1f%%) reproduced -> "
                    "replaced neuron_%d (%.1f%%) | fingerprint %s -> %s",
                    best_neuron_id, best_accuracy * 100,
                    worst_neuron_id, worst_accuracy * 100,
                    old_fingerprint, new_fingerprint,
                )
            else:
                event["actions"].append({
                    "type": "NO_SELECTION",
                    "reason": "accuracy_gap_too_small",
                    "gap": best_accuracy - worst_accuracy,
                })
                log.info(
                    "[EVOLUTION] No selection: accuracy gap too small (%.1f%%)",
                    (best_accuracy - worst_accuracy) * 100,
                )

        # Step 3: Speciation pressure check
        speciation_event = self._check_speciation_pressure()
        if speciation_event:
            event["actions"].append(speciation_event)

        # Step 4: Persist genomes
        self._save_genomes()

        # Step 5: Log
        event["population_stats"] = self.get_population_stats()
        event["leaderboard"] = self.get_leaderboard()
        self.evolution_log.append(event)

        # Trim log to last 100 events
        if len(self.evolution_log) > 100:
            self.evolution_log = self.evolution_log[-100:]

        return event

    def _rank_neurons(self) -> List[Tuple[int, float]]:
        """
        Rank neurons by accuracy (best first).
        Returns list of (neuron_id, accuracy) tuples.
        """
        rankings = []
        for neuron in self.mosaic.neurons:
            score = self.scores.get(neuron.neuron_id)
            if score:
                rankings.append((neuron.neuron_id, score.accuracy))
            else:
                rankings.append((neuron.neuron_id, 0.5))  # Default prior

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _check_speciation_pressure(self) -> Optional[Dict]:
        """
        Check if the population has converged too much.
        If so, inject diversity by replacing some neurons with random genomes.

        This prevents the mosaic from collapsing into a monoculture where
        all neurons have the same genome (which defeats the purpose of
        having a diverse population).
        """
        neurons = self.mosaic.neurons
        if len(neurons) < 3:
            return None

        # Check genome uniqueness
        fingerprints = [genome_fingerprint(n) for n in neurons]
        unique_count = len(set(fingerprints))

        # Check average pairwise similarity
        sims = []
        for i in range(len(neurons)):
            for j in range(i + 1, len(neurons)):
                sims.append(genome_similarity(neurons[i], neurons[j]))
        avg_similarity = float(np.mean(sims)) if sims else 0.0

        needs_diversity = (
            unique_count < MIN_UNIQUE_GENOMES
            or avg_similarity > GENOME_SIMILARITY_THRESHOLD
        )

        if not needs_diversity:
            return None

        # Diversity injection: replace the bottom N neurons with random genomes
        ranked = self._rank_neurons()
        n_to_replace = max(1, int(len(neurons) * DIVERSITY_INJECTION_FRACTION))

        # Never replace the top-ranked neuron(s)
        replaceable = ranked[ELITE_COUNT:]
        # Take from the bottom of the replaceable list
        to_replace = replaceable[-n_to_replace:]

        replaced_ids = []
        for neuron_id, accuracy in to_replace:
            idx = next(
                (i for i, n in enumerate(neurons) if n.neuron_id == neuron_id),
                None,
            )
            if idx is not None:
                old_fp = genome_fingerprint(neurons[idx])
                new_genome = self.mutator.create_random_genome(neuron_id)
                neurons[idx] = new_genome

                # Reset scoring
                self.scores[neuron_id] = NeuronScore(
                    neuron_id=neuron_id,
                    generation=self.generation,
                    born_at_cycle=self.cycle_count,
                )

                replaced_ids.append({
                    "neuron_id": neuron_id,
                    "old_accuracy": accuracy,
                    "old_fingerprint": old_fp,
                    "new_fingerprint": genome_fingerprint(new_genome),
                })

        log.info(
            "[EVOLUTION] SPECIATION PRESSURE: diversity injection | "
            "unique=%d/%d avg_sim=%.2f | replaced %d neurons: %s",
            unique_count, len(neurons), avg_similarity,
            len(replaced_ids), [r["neuron_id"] for r in replaced_ids],
        )

        return {
            "type": "SPECIATION_PRESSURE",
            "trigger": {
                "unique_genomes": unique_count,
                "total_neurons": len(neurons),
                "avg_similarity": avg_similarity,
                "threshold": GENOME_SIMILARITY_THRESHOLD,
            },
            "replaced": replaced_ids,
        }

    # ----------------------------------------------------------
    # PERSISTENCE
    # ----------------------------------------------------------

    def _save_genomes(self):
        """Save evolved genomes to JSON for restart recovery."""
        try:
            data = {
                "version": "NEURAL-EVOLUTION-1.0",
                "timestamp": datetime.now().isoformat(),
                "generation": self.generation,
                "cycle_count": self.cycle_count,
                "neurons": [],
                "scores": {},
            }

            for neuron in self.mosaic.neurons:
                neuron_data = {
                    "neuron_id": neuron.neuron_id,
                    "insertions": [
                        {
                            "target_qubit": ins.target_qubit,
                            "effect": ins.effect,
                            "rewire_target": ins.rewire_target,
                            "magnitude": ins.magnitude,
                        }
                        for ins in neuron.insertions
                    ],
                    "activation_modifiers": {
                        str(k): v for k, v in neuron.activation_modifiers.items()
                    },
                    "fingerprint": genome_fingerprint(neuron),
                }
                data["neurons"].append(neuron_data)

            # Save scores (rolling predictions are NOT saved -- they rebuild)
            for nid, score in self.scores.items():
                data["scores"][str(nid)] = {
                    "neuron_id": score.neuron_id,
                    "total_correct": score.total_correct,
                    "total_predictions": score.total_predictions,
                    "generation": score.generation,
                    "parent_id": score.parent_id,
                    "born_at_cycle": score.born_at_cycle,
                    "best_streak": score.best_streak,
                    "worst_streak": score.worst_streak,
                }

            # Atomic write: write to temp file then rename
            tmp_path = self.genome_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.genome_file)

            log.info(
                "[EVOLUTION] Genomes saved to %s (gen=%d, %d neurons)",
                self.genome_file, self.generation, len(data["neurons"]),
            )

        except Exception as e:
            log.error("[EVOLUTION] Failed to save genomes: %s", e)

    def _load_genomes(self):
        """Load persisted genomes if available."""
        if not os.path.exists(self.genome_file):
            log.info("[EVOLUTION] No persisted genomes found at %s -- starting fresh", self.genome_file)
            return

        try:
            with open(self.genome_file, "r") as f:
                data = json.load(f)

            if data.get("version") != "NEURAL-EVOLUTION-1.0":
                log.warning("[EVOLUTION] Genome file version mismatch, ignoring")
                return

            saved_neurons = data.get("neurons", [])
            if len(saved_neurons) != len(self.mosaic.neurons):
                log.warning(
                    "[EVOLUTION] Saved neuron count (%d) != mosaic size (%d), ignoring",
                    len(saved_neurons), len(self.mosaic.neurons),
                )
                return

            # Restore genomes
            for i, neuron_data in enumerate(saved_neurons):
                neuron = self.mosaic.neurons[i]
                neuron.neuron_id = neuron_data["neuron_id"]

                # Rebuild insertions
                neuron.insertions = [
                    L1Insertion(
                        target_qubit=ins["target_qubit"],
                        effect=ins["effect"],
                        rewire_target=ins["rewire_target"],
                        magnitude=ins["magnitude"],
                    )
                    for ins in neuron_data["insertions"]
                ]

                # Rebuild modifiers
                neuron.activation_modifiers = {
                    int(k): v
                    for k, v in neuron_data.get("activation_modifiers", {}).items()
                }

                neuron.vote = 0
                neuron.confidence = 0.0

            # Restore metadata
            self.generation = data.get("generation", 0)
            self.cycle_count = data.get("cycle_count", 0)

            # Restore score metadata (but NOT rolling predictions -- they need fresh data)
            saved_scores = data.get("scores", {})
            for nid_str, score_data in saved_scores.items():
                nid = int(nid_str)
                if nid in self.scores:
                    self.scores[nid].total_correct = score_data.get("total_correct", 0)
                    self.scores[nid].total_predictions = score_data.get("total_predictions", 0)
                    self.scores[nid].generation = score_data.get("generation", 0)
                    self.scores[nid].parent_id = score_data.get("parent_id")
                    self.scores[nid].born_at_cycle = score_data.get("born_at_cycle", 0)
                    self.scores[nid].best_streak = score_data.get("best_streak", 0)
                    self.scores[nid].worst_streak = score_data.get("worst_streak", 0)

            log.info(
                "[EVOLUTION] Loaded genomes from %s | gen=%d, cycle=%d, %d neurons",
                self.genome_file, self.generation, self.cycle_count, len(saved_neurons),
            )

        except Exception as e:
            log.warning("[EVOLUTION] Failed to load genomes: %s -- starting fresh", e)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    print("=" * 76)
    print("  NEURAL MOSAIC EVOLUTION ENGINE -- Standalone Test")
    print("=" * 76)

    # Create a mosaic
    mosaic = NeuralMosaicEngine(n_neurons=7, seed=42)

    # Create evolution engine (use temp file for test)
    test_genome_file = str(
        Path(__file__).parent / "test_evolved_genomes.json"
    )
    evo = NeuralEvolutionEngine(
        mosaic,
        genome_file=test_genome_file,
        evolve_every=5,
        seed=42,
    )

    print(f"\n  Initial population ({len(mosaic.neurons)} neurons):")
    for entry in evo.get_leaderboard():
        print(f"    Neuron {entry['neuron_id']}: "
              f"acc={entry['accuracy']:.1%} "
              f"ins={entry['n_insertions']} "
              f"fp={entry['fingerprint']}")

    # Simulate 100 cycles with known market directions
    print(f"\n  Simulating 100 TEQA cycles...")
    rng = random.Random(123)
    evolution_events = 0

    for cycle in range(100):
        # Simulate: each neuron votes randomly but some are biased
        actual_direction = rng.choice([-1, 1])

        # Bias neuron 0 to be good (70% correct), neuron 6 to be bad (30% correct)
        for neuron in mosaic.neurons:
            if neuron.neuron_id == 0:
                neuron.vote = actual_direction if rng.random() < 0.70 else -actual_direction
            elif neuron.neuron_id == 6:
                neuron.vote = actual_direction if rng.random() < 0.30 else -actual_direction
            else:
                neuron.vote = actual_direction if rng.random() < 0.50 else -actual_direction

        result = evo.record_votes(actual_direction)
        if result and result.get("actions"):
            evolution_events += 1

    print(f"\n  After 100 cycles ({evolution_events} evolution events):")
    for entry in evo.get_leaderboard():
        print(f"    Neuron {entry['neuron_id']}: "
              f"acc={entry['accuracy']:.1%} "
              f"pred={entry['n_predictions']} "
              f"gen={entry['generation']} "
              f"ins={entry['n_insertions']} "
              f"fp={entry['fingerprint']}")

    stats = evo.get_population_stats()
    print(f"\n  Population stats:")
    print(f"    Generation:      {stats['generation']}")
    print(f"    Avg accuracy:    {stats['avg_accuracy']:.1%}")
    print(f"    Best accuracy:   {stats['best_accuracy']:.1%}")
    print(f"    Worst accuracy:  {stats['worst_accuracy']:.1%}")
    print(f"    Unique genomes:  {stats['unique_genomes']}/{stats['n_neurons']}")
    print(f"    Avg similarity:  {stats['avg_pairwise_similarity']:.2f}")
    print(f"    Speciation:      {'YES' if stats['speciation_pressure'] else 'no'}")

    # Test persistence
    print(f"\n  Testing persistence...")
    evo._save_genomes()
    fingerprints_before = [genome_fingerprint(n) for n in mosaic.neurons]

    # Create new mosaic + load
    mosaic2 = NeuralMosaicEngine(n_neurons=7, seed=999)  # Different seed
    evo2 = NeuralEvolutionEngine(
        mosaic2,
        genome_file=test_genome_file,
        evolve_every=5,
    )
    fingerprints_after = [genome_fingerprint(n) for n in mosaic2.neurons]

    if fingerprints_before == fingerprints_after:
        print("    Persistence: PASS (genomes survived restart)")
    else:
        print("    Persistence: FAIL")
        print(f"      Before: {fingerprints_before}")
        print(f"      After:  {fingerprints_after}")

    # Cleanup test file
    try:
        os.remove(test_genome_file)
    except OSError:
        pass

    print("\n" + "=" * 76)
    print("  Test complete.")
    print("=" * 76)
