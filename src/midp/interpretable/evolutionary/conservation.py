"""
Conservation analysis for metalloprotein sequences.

This module implements advanced conservation scoring methods that account for
amino acid properties and metal-binding preferences.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
import re

import numpy as np
from Bio.Align import MultipleSeqAlignment, substitution_matrices
from Bio.Substitution import MatrixInfo
from scipy.stats import entropy

from ...core.constants import (
    AMINO_ACID_PROPERTIES,
    CONSERVATION_THRESHOLDS,
    METAL_BINDING_PREFERENCES,
)
from ...core.data_structures import MetalType
from ...core.exceptions import (
    ValidationError,
    ScientificCalculationError,
    DataAccessError,
)
from .config import EvolutionaryConfig

logger = logging.getLogger(__name__)

# Get configuration instance
config = EvolutionaryConfig.get_instance()


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
        blosum_matrix: Optional[str] = None,
        property_weight: Optional[float] = None,
    ):
        """
        Initialize conservation analyzer.

        Args:
            use_blosum: Whether to use BLOSUM substitution scoring
            blosum_matrix: Which BLOSUM matrix to use (default from config)
            property_weight: Weight for property-based conservation (vs entropy) (default from config)
        """
        self.use_blosum = use_blosum

        # Load parameters from config
        cons_config = config.conservation
        self.blosum_matrix = blosum_matrix or cons_config["blosum_matrix"]
        self.property_weight = property_weight or cons_config["property_weight"]

        # Load property groups from config
        self.property_groups = cons_config["property_groups"]

        # Load thresholds
        self.highly_conserved = cons_config["highly_conserved"]
        self.moderately_conserved = cons_config["moderately_conserved"]
        self.variable = cons_config["variable"]

        # Load BLOSUM matrix
        if self.use_blosum:
            try:
                self.blosum = getattr(MatrixInfo, self.blosum_matrix)
            except AttributeError:
                raise ValueError(f"Invalid BLOSUM matrix: {self.blosum_matrix}")

    def calculate_conservation_scores(
        self,
        msa: MultipleSeqAlignment,
        seq_weights: Optional[np.ndarray] = None,
        metal_positions: Optional[Dict[int, MetalType]] = None,
    ) -> np.ndarray:
        """
        Calculate comprehensive conservation scores for each position.

        Args:
            msa: Multiple sequence alignment
            seq_weights: Optional sequence weights to correct for bias
            metal_positions: Dict mapping positions to metal types

        Returns:
            Array of conservation scores (0-1) for each position

        Raises:
            ValidationError: If input validation fails
        """
        try:
            # Validate MSA
            if len(msa) < 10:
                raise ValidationError(
                    f"MSA must have at least 10 sequences, got {len(msa)}"
                )

            n_positions = msa.get_alignment_length()

            # Validate sequence weights
            if seq_weights is not None:
                if len(seq_weights) != len(msa):
                    raise ValidationError(
                        f"Length mismatch: {len(seq_weights)} weights for {len(msa)} sequences"
                    )
                weight_sum = np.sum(seq_weights)
                if not np.isclose(weight_sum, len(msa), rtol=1e-5):
                    raise ValidationError(
                        f"Sequence weights should sum to {len(msa)}, got {weight_sum}"
                    )

            # Validate metal positions
            if metal_positions:
                invalid_pos = [pos for pos in metal_positions if pos > n_positions]
                if invalid_pos:
                    raise ValidationError(
                        f"Metal positions {invalid_pos} exceed sequence length {n_positions}"
                    )

            # Calculate different conservation components
            entropy_scores = self._calculate_entropy_conservation(msa, seq_weights)
            property_scores = self._calculate_property_conservation(msa, seq_weights)

            # Combine scores
            scores = (
                1 - self.property_weight
            ) * entropy_scores + self.property_weight * property_scores

            # Apply BLOSUM-based smoothing if enabled
            if self.use_blosum:
                scores = self._apply_blosum_smoothing(msa, scores, seq_weights)

            # Boost scores for metal-binding positions
            if metal_positions:
                scores = self._boost_metal_positions(scores, metal_positions, msa)

            return scores

        except Exception as e:
            if not isinstance(e, (ValidationError, ScientificCalculationError)):
                raise ScientificCalculationError(
                    f"Conservation calculation failed: {str(e)}"
                )
            raise

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

        # Calculate property frequencies
        property_freqs = {prop: 0.0 for prop in self.property_groups}
        total_weight = 0.0

        for i, record in enumerate(msa):
            aa = record.seq[pos]
            if aa != "-" and aa != "X":
                weight = seq_weights[i]
                total_weight += weight

                # Add to property groups
                for prop, aa_set in self.property_groups.items():
                    if aa in aa_set:
                        property_freqs[prop] += weight

        if total_weight == 0:
            property_scores = np.zeros(n_positions)
            return property_scores

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
        property_scores = (
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
        if not self.blosum:
            logger.warning("BLOSUM matrix not available, skipping smoothing")
            return base_scores

        try:
            n_positions = msa.get_alignment_length()
            n_sequences = len(msa)
            smoothed_scores = np.copy(base_scores)

            if seq_weights is None:
                seq_weights = np.ones(n_sequences)

            # Validate matrix bounds
            min_score = float(self.blosum.min())
            max_score = float(self.blosum.max())
            score_range = max_score - min_score

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
                            score = self.blosum[aa1, aa2]
                            weight_product = (
                                aa_counts[aa1] * aa_counts[aa2] / (total_weight**2)
                            )
                            blosum_conservation += score * weight_product
                            comparisons += weight_product
                        except KeyError:
                            logger.debug(
                                f"Amino acid pair {aa1}-{aa2} not found in BLOSUM matrix"
                            )
                            continue

                if comparisons > 0:
                    # Normalize BLOSUM score
                    normalized_blosum = (blosum_conservation - min_score) / score_range
                    normalized_blosum = max(0, min(1, normalized_blosum))

                    # Blend with base score
                    smoothed_scores[pos] = (
                        0.7 * base_scores[pos] + 0.3 * normalized_blosum
                    )

            return smoothed_scores

        except Exception as e:
            if not isinstance(e, (KeyError, AttributeError)):
                raise ScientificCalculationError(f"BLOSUM smoothing failed: {str(e)}")
            raise

    def _boost_metal_positions(
        self,
        scores: np.ndarray,
        metal_positions: Dict[int, MetalType],
        msa: MultipleSeqAlignment,
    ) -> np.ndarray:
        """
        Boost conservation scores for known metal-binding positions.

        Args:
            scores: Base conservation scores
            metal_positions: Dict mapping positions to metal types
            msa: Multiple sequence alignment

        Returns:
            Boosted conservation scores
        """
        try:
            boosted_scores = np.copy(scores)

            for pos, metal_type in metal_positions.items():
                if pos > len(scores):
                    raise ValidationError(
                        f"Metal position {pos} exceeds sequence length {len(scores)}"
                    )

                idx = pos - 1  # Convert to 0-indexed

                # Get metal-specific preferences
                if metal_type not in METAL_BINDING_PREFERENCES:
                    raise ValidationError(
                        f"No binding preferences defined for {metal_type}"
                    )

                preferences = METAL_BINDING_PREFERENCES[metal_type]
                primary_aas = set(preferences.get("primary", []))
                secondary_aas = set(preferences.get("secondary", []))

                if not primary_aas and not secondary_aas:
                    raise ValidationError(
                        f"No binding residues defined for {metal_type}"
                    )

                # Check conservation of preferred residues
                primary_count = 0
                secondary_count = 0
                total_count = 0

                for record in msa:
                    aa = record.seq[idx]
                    if aa != "-" and aa != "X":
                        total_count += 1
                        if aa in primary_aas:
                            primary_count += 1
                        elif aa in secondary_aas:
                            secondary_count += 1

                if total_count > 0:
                    # Calculate weighted frequency
                    primary_freq = primary_count / total_count
                    secondary_freq = secondary_count / total_count

                    # Apply metal-specific boost factors
                    base_boost = preferences.get("conservation_boost", 0.2)
                    if base_boost <= 0:
                        raise ValidationError(
                            f"Invalid conservation boost factor for {metal_type}: {base_boost}"
                        )

                    # Higher boost for primary binding residues
                    primary_boost = base_boost * 1.5 * primary_freq
                    secondary_boost = base_boost * secondary_freq

                    total_boost = primary_boost + secondary_boost

                    # Apply boost with upper limit
                    boosted_scores[idx] = min(1.0, scores[idx] * (1.0 + total_boost))

                    if total_boost > 0:
                        logger.debug(
                            f"Boosted pos {pos} ({metal_type}) by factor {1.0 + total_boost:.2f}"
                        )

            return boosted_scores

        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ScientificCalculationError(
                    f"Failed to boost metal positions: {str(e)}"
                )
            raise

    def identify_conserved_metal_motifs(
        self, msa: MultipleSeqAlignment, conservation_scores: np.ndarray
    ) -> list[dict[str, any]]:
        """
        Identify conserved motifs that match known metal-binding patterns.

        Returns:
            List of dictionaries containing motif information
        """
        motifs = []
        consensus = self._get_consensus_sequence(msa)

        # Get motif patterns from config
        motif_patterns = config.metal_binding["motif_patterns"]
        motif_threshold = config.metal_binding["conservation_thresholds"][
            "motif_detection"
        ]

        for motif_name, pattern in motif_patterns.items():
            for match in re.finditer(pattern, consensus):
                start, end = match.span()

                # Check if motif positions are conserved
                motif_scores = conservation_scores[start:end]
                if np.mean(motif_scores) > motif_threshold:
                    motifs.append({
                        "name": motif_name,
                        "start": start + 1,  # 1-indexed
                        "end": end,
                        "sequence": match.group(),
                        "score": float(np.mean(motif_scores)),
                        "positions": list(range(start + 1, end + 1)),
                    })

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
