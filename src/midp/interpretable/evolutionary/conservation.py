"""
Conservation analysis for metalloprotein sequences.

This module implements advanced conservation scoring methods that account for
amino acid properties and metal-binding preferences.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from scipy.stats import entropy

from src.midp.core.constants import (
    AMINO_ACID_PROPERTIES,
    CONSERVATION_THRESHOLDS,
    METAL_BINDING_PREFERENCES,
)

logger = logging.getLogger(__name__)


class ConservationAnalyzer:
    """
    Advanced conservation analysis for metalloproteins.

    Implements multiple conservation scoring methods:
    - Shannon entropy-based conservation
    - Property-based conservation (e.g., hydrophobicity, charge)
    - Substitution matrix-based scoring (BLOSUM)
    - Metal-binding propensity conservation
    """

    def __init__(
        self,
        use_blosum: bool = True,
        blosum_matrix: str = "BLOSUM62",
        property_weight: float = 0.3,
    ):
        """
        Initialize conservation analyzer.

        Args:
            use_blosum: Whether to use BLOSUM substitution scoring
            blosum_matrix: Which BLOSUM matrix to use
            property_weight: Weight for property-based conservation (vs entropy)
        """
        self.use_blosum = use_blosum
        self.property_weight = property_weight

        # Load substitution matrix
        if use_blosum:
            self.subst_matrix = substitution_matrices.load(blosum_matrix)

    def calculate_conservation_scores(
        self,
        msa: MultipleSeqAlignment,
        seq_weights: Optional[np.ndarray] = None,
        metal_positions: Optional[list[int]] = None,
    ) -> np.ndarray:
        """
        Calculate comprehensive conservation scores for each position.

        Args:
            msa: Multiple sequence alignment
            seq_weights: Optional sequence weights to correct for bias
            metal_positions: Known metal-binding positions for enhanced scoring

        Returns:
            Array of conservation scores (0-1) for each position
        """
        n_positions = msa.get_alignment_length()

        # Calculate different conservation components
        entropy_scores = self._calculate_entropy_conservation(msa, seq_weights)
        property_scores = self._calculate_property_conservation(msa, seq_weights)

        # Combine scores
        combined_scores = (
            1 - self.property_weight
        ) * entropy_scores + self.property_weight * property_scores

        # Apply BLOSUM-based smoothing if enabled
        if self.use_blosum:
            combined_scores = self._apply_blosum_smoothing(
                msa, combined_scores, seq_weights
            )

        # Boost scores for metal-binding positions
        if metal_positions:
            combined_scores = self._boost_metal_positions(
                combined_scores, metal_positions, msa
            )

        return combined_scores

    def _calculate_entropy_conservation(
        self, msa: MultipleSeqAlignment, seq_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate Shannon entropy-based conservation scores."""
        n_positions = msa.get_alignment_length()
        n_sequences = len(msa)
        entropy_scores = np.zeros(n_positions)

        if seq_weights is None:
            seq_weights = np.ones(n_sequences)

        for pos in range(n_positions):
            # Calculate weighted amino acid frequencies
            aa_counts = {}
            total_weight = 0.0

            for i, record in enumerate(msa):
                aa = record.seq[pos]
                if aa != "-" and aa != "X":
                    weight = seq_weights[i]
                    aa_counts[aa] = aa_counts.get(aa, 0) + weight
                    total_weight += weight

            if total_weight == 0:
                entropy_scores[pos] = 0.0
                continue

            # Calculate Shannon entropy
            aa_freqs = np.array(list(aa_counts.values())) / total_weight
            shannon_entropy = entropy(aa_freqs, base=2)

            # Normalize (max entropy = log2(20) for 20 amino acids)
            max_entropy = np.log2(20)
            entropy_scores[pos] = 1.0 - (shannon_entropy / max_entropy)

        return entropy_scores

    def _calculate_property_conservation(
        self, msa: MultipleSeqAlignment, seq_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate conservation based on amino acid properties."""
        n_positions = msa.get_alignment_length()
        n_sequences = len(msa)
        property_scores = np.zeros(n_positions)

        if seq_weights is None:
            seq_weights = np.ones(n_sequences)

        # Define property groups
        property_groups = {
            "hydrophobic": set("AILMFVW"),
            "aromatic": set("FWY"),
            "polar": set("STNQ"),
            "positive": set("RKH"),
            "negative": set("DE"),
            "small": set("AGST"),
            "tiny": set("AGS"),
            "proline": set("P"),
            "cysteine": set("C"),
            "metal_binding": set("CHMDE"),
        }

        for pos in range(n_positions):
            # Calculate property frequencies
            property_freqs = {prop: 0.0 for prop in property_groups}
            total_weight = 0.0

            for i, record in enumerate(msa):
                aa = record.seq[pos]
                if aa != "-" and aa != "X":
                    weight = seq_weights[i]
                    total_weight += weight

                    # Add to property groups
                    for prop, aa_set in property_groups.items():
                        if aa in aa_set:
                            property_freqs[prop] += weight

            if total_weight == 0:
                property_scores[pos] = 0.0
                continue

            # Normalize frequencies
            for prop in property_freqs:
                property_freqs[prop] /= total_weight

            # Calculate property conservation
            # High score if one property dominates
            max_prop_freq = max(property_freqs.values())

            # Additional scoring for specific important properties
            metal_binding_freq = property_freqs["metal_binding"]
            cysteine_freq = property_freqs["cysteine"]

            # Weighted combination
            property_scores[pos] = (
                0.6 * max_prop_freq + 0.2 * metal_binding_freq + 0.2 * cysteine_freq
            )

        return property_scores

    def _apply_blosum_smoothing(
        self,
        msa: MultipleSeqAlignment,
        base_scores: np.ndarray,
        seq_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply BLOSUM-based smoothing to conservation scores."""
        n_positions = msa.get_alignment_length()
        n_sequences = len(msa)
        smoothed_scores = np.copy(base_scores)

        if seq_weights is None:
            seq_weights = np.ones(n_sequences)

        for pos in range(n_positions):
            # Get amino acid distribution at this position
            aa_counts = {}
            total_weight = 0.0

            for i, record in enumerate(msa):
                aa = record.seq[pos]
                if aa != "-" and aa != "X":
                    weight = seq_weights[i]
                    aa_counts[aa] = aa_counts.get(aa, 0) + weight
                    total_weight += weight

            if total_weight == 0 or len(aa_counts) <= 1:
                continue

            # Calculate average BLOSUM score between observed amino acids
            blosum_conservation = 0.0
            comparisons = 0

            aa_list = list(aa_counts.keys())
            for i in range(len(aa_list)):
                for j in range(i, len(aa_list)):
                    aa1, aa2 = aa_list[i], aa_list[j]

                    try:
                        # Get BLOSUM score from the new matrix format
                        score = self.subst_matrix[aa1, aa2]
                        weight_product = (
                            aa_counts[aa1] * aa_counts[aa2] / (total_weight**2)
                        )
                        blosum_conservation += score * weight_product
                        comparisons += weight_product
                    except KeyError:
                        # Skip if amino acids not in matrix
                        continue

            if comparisons > 0:
                # Normalize BLOSUM score (typical range -4 to 11 for BLOSUM62)
                min_score = self.subst_matrix.min()
                max_score = self.subst_matrix.max()
                score_range = max_score - min_score
                if score_range > 0:
                    normalized_blosum = (blosum_conservation - min_score) / score_range
                    normalized_blosum = max(0, min(1, normalized_blosum))

                    # Blend with base score
                    smoothed_scores[pos] = (
                        0.7 * base_scores[pos] + 0.3 * normalized_blosum
                    )

        return smoothed_scores

    def _boost_metal_positions(
        self, scores: np.ndarray, metal_positions: list[int], msa: MultipleSeqAlignment
    ) -> np.ndarray:
        """Boost conservation scores for known metal-binding positions."""
        boosted_scores = np.copy(scores)

        for pos in metal_positions:
            if pos <= len(scores):
                idx = pos - 1  # Convert to 0-indexed

                # Check if metal-binding amino acids are conserved
                metal_aa_count = 0
                total_count = 0

                for record in msa:
                    aa = record.seq[idx]
                    if aa != "-" and aa != "X":
                        total_count += 1
                        if aa in "CHMDE":  # Metal-binding residues
                            metal_aa_count += 1

                if total_count > 0:
                    metal_freq = metal_aa_count / total_count
                    if metal_freq > 0.5:  # Majority are metal-binding
                        # Boost score by up to 20%
                        boost_factor = 1.0 + 0.2 * metal_freq
                        boosted_scores[idx] = min(1.0, scores[idx] * boost_factor)

        return boosted_scores

    def identify_conserved_metal_motifs(
        self, msa: MultipleSeqAlignment, conservation_scores: np.ndarray
    ) -> list[dict[str, any]]:
        """
        Identify conserved motifs that match known metal-binding patterns.

        Returns:
            List of dictionaries containing motif information
        """
        motifs = []

        # Common metal-binding motifs (simplified patterns)
        metal_patterns = {
            "CxxC": (r"C..C", "Zinc finger-like"),
            "HxxH": (r"H..H", "Zinc/Iron binding"),
            "CxxCH": (r"C..CH", "Iron-sulfur cluster"),
            "DxDxD": (r"D.D.D", "Calcium binding"),
            "CxxHxxC": (r"C..H..C", "Zinc binding"),
        }

        # Get consensus sequence
        consensus = self._get_consensus_sequence(msa)

        import re

        for motif_name, (pattern, description) in metal_patterns.items():
            # Find pattern matches in consensus
            for match in re.finditer(pattern, consensus):
                start, end = match.span()

                # Check if region is conserved
                region_conservation = np.mean(conservation_scores[start:end])

                if (
                    region_conservation
                    > CONSERVATION_THRESHOLDS["moderately_conserved"]
                ):
                    motif_info = {
                        "name": motif_name,
                        "description": description,
                        "start": start + 1,  # 1-indexed
                        "end": end,
                        "sequence": consensus[start:end],
                        "conservation": region_conservation,
                        "positions": list(range(start + 1, end + 1)),
                    }
                    motifs.append(motif_info)

        return motifs

    def _get_consensus_sequence(self, msa: MultipleSeqAlignment) -> str:
        """Generate consensus sequence from MSA."""
        n_positions = msa.get_alignment_length()
        consensus = []

        for pos in range(n_positions):
            # Count amino acids at this position
            aa_counts = {}

            for record in msa:
                aa = record.seq[pos]
                if aa != "-":
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1

            if aa_counts:
                # Get most common amino acid
                consensus_aa = max(aa_counts, key=aa_counts.get)
                consensus.append(consensus_aa)
            else:
                consensus.append("X")

        return "".join(consensus)

    def calculate_relative_entropy(
        self,
        msa: MultipleSeqAlignment,
        background_freqs: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Calculate relative entropy (Kullback-Leibler divergence) for each position.

        This measures how much the amino acid distribution at each position
        differs from background frequencies.
        """
        n_positions = msa.get_alignment_length()
        relative_entropies = np.zeros(n_positions)

        # Default background frequencies (from UniProt)
        if background_freqs is None:
            background_freqs = {
                "A": 0.0825,
                "R": 0.0553,
                "N": 0.0406,
                "D": 0.0545,
                "C": 0.0137,
                "Q": 0.0393,
                "E": 0.0675,
                "G": 0.0707,
                "H": 0.0227,
                "I": 0.0596,
                "L": 0.0966,
                "K": 0.0584,
                "M": 0.0242,
                "F": 0.0386,
                "P": 0.0470,
                "S": 0.0656,
                "T": 0.0534,
                "W": 0.0108,
                "Y": 0.0292,
                "V": 0.0687,
            }

        for pos in range(n_positions):
            # Calculate position-specific frequencies
            pos_freqs = {}
            total = 0

            for record in msa:
                aa = record.seq[pos]
                if aa != "-" and aa != "X" and aa in background_freqs:
                    pos_freqs[aa] = pos_freqs.get(aa, 0) + 1
                    total += 1

            if total == 0:
                continue

            # Normalize frequencies
            for aa in pos_freqs:
                pos_freqs[aa] /= total

            # Calculate KL divergence
            kl_div = 0.0
            for aa, freq in pos_freqs.items():
                if freq > 0 and aa in background_freqs:
                    kl_div += freq * np.log2(freq / background_freqs[aa])

            relative_entropies[pos] = kl_div

        # Normalize to 0-1 range
        max_re = np.max(relative_entropies) if np.max(relative_entropies) > 0 else 1.0
        return relative_entropies / max_re
