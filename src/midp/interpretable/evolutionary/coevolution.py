"""
Coevolution analysis for detecting functionally coupled residues.

This module implements methods to identify coevolving residue pairs that may
indicate functional coupling, allosteric networks, or coordinated metal binding.
"""

import hashlib
import logging
import multiprocessing as mp
import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from Bio.Align import MultipleSeqAlignment
from memory_profiler import profile
from numba import jit
from scipy import linalg
from scipy.spatial.distance import squareform
from scipy.stats import chi2, entropy
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.midp.core.constants import METAL_BINDING_PREFERENCES
from src.midp.core.data_structures import MetalSite, MetalType
from src.midp.core.exceptions import (
    DataAccessError,
    ScientificCalculationError,
    ValidationError,
)

from .config import EvolutionaryConfig

logger = logging.getLogger(__name__)

# Get configuration instance
config = EvolutionaryConfig.get_instance()


# Numba-optimized MI calculation for a single position pair
@jit(nopython=True)
def _calculate_position_pair_mi(
    joint_probs: np.ndarray,
    marginal_i_probs: np.ndarray,
    marginal_j_probs: np.ndarray,
) -> float:
    """Calculate mutual information between two positions using pre-computed probabilities."""
    mi = 0.0
    for idx_i in range(joint_probs.shape[0]):
        for idx_j in range(joint_probs.shape[1]):
            if (
                joint_probs[idx_i, idx_j] > 0
                and marginal_i_probs[idx_i] > 0
                and marginal_j_probs[idx_j] > 0
            ):
                mi += joint_probs[idx_i, idx_j] * np.log2(
                    joint_probs[idx_i, idx_j]
                    / (marginal_i_probs[idx_i] * marginal_j_probs[idx_j])
                )
    return mi


def _process_sequence_chunk(args):
    """Process a chunk of sequences for parallel MI calculation."""
    start_idx, end_idx, msa_chunk, pos_i, pos_j, aa_to_idx, seq_weights_chunk = args

    joint_counts = np.zeros((21, 21))
    marginal_i = np.zeros(21)
    marginal_j = np.zeros(21)
    total_weight = 0.0

    for k, record in enumerate(msa_chunk):
        aa_i = record.seq[pos_i]
        aa_j = record.seq[pos_j]

        if aa_i in aa_to_idx and aa_j in aa_to_idx:
            idx_i = aa_to_idx[aa_i]
            idx_j = aa_to_idx[aa_j]
            weight = seq_weights_chunk[k]

            joint_counts[idx_i, idx_j] += weight
            marginal_i[idx_i] += weight
            marginal_j[idx_j] += weight
            total_weight += weight

    return joint_counts, marginal_i, marginal_j, total_weight


class CoevolutionAnalyzer:
    """
    Detects coevolving residue pairs using mutual information analysis.

    Implements:
    - Mutual Information (MI) with APC correction
    - Network analysis of coevolution patterns
    - Metal-specific coevolution detection

    The analysis uses a position-specific mutual information approach with
    the Average Product Correction (APC) to reduce phylogenetic and sampling bias.
    This helps identify functionally coupled residues, including those involved
    in metal binding and allosteric networks.
    """

    def __init__(
        self,
        min_separation: Optional[int] = None,
        significance_threshold: Optional[float] = None,
        n_workers: Optional[int] = None,
        chunk_size: Optional[int] = None,
        pseudocount: float = 0.5,
    ):
        """
        Initialize coevolution analyzer.

        Args:
            min_separation: Minimum sequence separation for coupling
            significance_threshold: P-value threshold for significant coupling
            n_workers: Number of parallel workers (default: CPU count)
            chunk_size: Size of sequence chunks for batch processing
            pseudocount: Pseudocount for frequency calculations
        """
        # Load parameters from config
        coevo_config = config.coevolution
        self.min_separation = min_separation or coevo_config["min_separation"]
        self.pseudocount = pseudocount
        self.significance_threshold = (
            significance_threshold or coevo_config["significance_threshold"]
        )
        self.n_workers = n_workers or coevo_config["n_workers"] or mp.cpu_count()
        self.chunk_size = chunk_size or coevo_config["chunk_size"]
        self.use_apc = coevo_config["apc_correction"]

        # Alphabet for calculations
        self.alphabet = list("ACDEFGHIKLMNPQRSTVWY-")
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.alphabet)}
        self.n_states = len(self.alphabet)

        # Cache for frequency calculations
        self._freq_cache = {}
        self._cache_dir = Path(tempfile.gettempdir()) / "midp_coevo_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @profile
    def _calculate_mutual_information(
        self, msa: MultipleSeqAlignment, seq_weights: np.ndarray
    ) -> np.ndarray:
        """Calculate weighted mutual information between all position pairs."""
        try:
            n_positions = msa.get_alignment_length()
            n_sequences = len(msa)
            mi_matrix = np.zeros((n_positions, n_positions))

            # Validate inputs
            if len(seq_weights) != n_sequences:
                raise ValidationError(
                    f"Sequence weights length ({len(seq_weights)}) "
                    f"does not match MSA size ({n_sequences})"
                )

            # Calculate or load cached single-site frequencies
            cache_key = self._get_cache_key(msa)
            single_freqs = self._load_cached_frequencies(cache_key)

            if single_freqs is None:
                single_freqs = self._calculate_single_site_frequencies(msa, seq_weights)
                self._save_cached_frequencies(cache_key, single_freqs)

            # Process position pairs in parallel
            position_pairs = [
                (i, j) for i in range(n_positions) for j in range(i + 1, n_positions)
            ]

            # Split sequences into chunks for large MSAs
            if n_sequences > self.chunk_size:
                sequence_chunks = [
                    (i, min(i + self.chunk_size, n_sequences))
                    for i in range(0, n_sequences, self.chunk_size)
                ]
            else:
                sequence_chunks = [(0, n_sequences)]

            # Create process pool
            with mp.Pool(self.n_workers) as pool:
                # Process chunks with progress bar
                with tqdm(total=len(position_pairs), desc="Calculating MI") as pbar:
                    for i, j in position_pairs:
                        # Prepare chunk arguments
                        chunk_args = [
                            (
                                start,
                                end,
                                msa[start:end],
                                i,
                                j,
                                self.aa_to_idx,
                                seq_weights[start:end],
                            )
                            for start, end in sequence_chunks
                        ]

                        # Process chunks in parallel
                        chunk_results = pool.map(_process_sequence_chunk, chunk_args)

                        # Combine chunk results
                        joint_counts = sum(r[0] for r in chunk_results)
                        marginal_i = sum(r[1] for r in chunk_results)
                        marginal_j = sum(r[2] for r in chunk_results)
                        total_weight = sum(r[3] for r in chunk_results)

                        # Add pseudocounts
                        pseudocount = self.pseudocount
                        joint_counts += pseudocount
                        marginal_i += pseudocount * self.n_states
                        marginal_j += pseudocount * self.n_states
                        total_weight += pseudocount * self.n_states * self.n_states

                        # Calculate probabilities
                        joint_probs = joint_counts / total_weight
                        marginal_i_probs = marginal_i / total_weight
                        marginal_j_probs = marginal_j / total_weight

                        # Calculate MI using numba-optimized function
                        mi = _calculate_position_pair_mi(
                            joint_probs, marginal_i_probs, marginal_j_probs
                        )

                        mi_matrix[i, j] = mi
                        mi_matrix[j, i] = mi

                        pbar.update(1)

            return mi_matrix

        except Exception as e:
            if not isinstance(e, (ValidationError, ScientificCalculationError)):
                raise ScientificCalculationError(
                    f"Mutual information calculation failed: {str(e)}"
                )
            raise

    def _get_cache_key(self, msa: MultipleSeqAlignment) -> str:
        """Generate cache key for MSA frequencies."""
        # Hash MSA content for cache key
        msa_str = "\n".join(str(record.seq) for record in msa)
        return hashlib.md5(msa_str.encode()).hexdigest()

    def _load_cached_frequencies(self, cache_key: str) -> Optional[np.ndarray]:
        """Load cached frequencies if available."""
        cache_file = self._cache_dir / f"freq_{cache_key}.npy"
        if cache_file.exists():
            try:
                return np.load(str(cache_file))
            except Exception as e:
                logger.warning(f"Failed to load frequency cache: {e}")
        return None

    def _save_cached_frequencies(self, cache_key: str, frequencies: np.ndarray):
        """Save frequencies to cache."""
        cache_file = self._cache_dir / f"freq_{cache_key}.npy"
        try:
            np.save(str(cache_file), frequencies)
        except Exception as e:
            logger.warning(f"Failed to save frequency cache: {e}")

    def calculate_coevolution_matrix(
        self,
        msa: MultipleSeqAlignment,
        seq_weights: Optional[np.ndarray] = None,
        metal_sites: Optional[list[MetalSite]] = None,
    ) -> np.ndarray:
        """
        Calculate the full coevolution matrix using mutual information.

        Args:
            msa: Multiple sequence alignment
            seq_weights: Sequence weights to reduce bias
            metal_sites: Known metal sites for enhanced analysis

        Returns:
            Symmetric matrix of coevolution scores with APC correction
        """
        # Calculate MI matrix with APC correction
        coevo_matrix = self._calculate_mi_apc(msa, seq_weights)

        # Apply sequence separation filter
        coevo_matrix = self._apply_separation_filter(coevo_matrix)

        # Enhance scores for metal-coordinating positions
        if metal_sites:
            coevo_matrix = self._enhance_metal_coevolution(
                coevo_matrix, metal_sites, msa
            )

        return coevo_matrix

    def _calculate_mi_apc(
        self, msa: MultipleSeqAlignment, seq_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate Mutual Information with Average Product Correction."""
        n_positions = msa.get_alignment_length()
        n_sequences = len(msa)

        if seq_weights is None:
            seq_weights = np.ones(n_sequences) / n_sequences

        # Calculate frequencies and MI
        mi_matrix = np.zeros((n_positions, n_positions))

        # Pre-calculate single-site frequencies
        single_freqs = self._calculate_single_site_frequencies(msa, seq_weights)

        # Calculate pairwise MI
        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                # Calculate joint frequencies
                joint_freq = self._calculate_joint_frequencies(msa, i, j, seq_weights)

                # Calculate MI
                mi = 0.0
                for a in range(self.n_states):
                    for b in range(self.n_states):
                        if (
                            single_freqs[i, a] > 0
                            and single_freqs[j, b] > 0
                            and joint_freq[a, b] > 0
                        ):
                            mi += joint_freq[a, b] * np.log(
                                joint_freq[a, b]
                                / (single_freqs[i, a] * single_freqs[j, b])
                            )

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        # Apply APC correction
        mi_matrix = self._apply_apc(mi_matrix)

        # Calculate significance
        mi_matrix = self._calculate_mi_significance(mi_matrix, n_sequences)

        return mi_matrix

    def _calculate_single_site_frequencies(
        self, msa: MultipleSeqAlignment, seq_weights: np.ndarray
    ) -> np.ndarray:
        """Calculate amino acid frequencies for each position."""
        n_positions = msa.get_alignment_length()
        freqs = np.zeros((n_positions, self.n_states))
        pseudocount = self.pseudocount

        for pos in range(n_positions):
            # Add pseudocounts
            freqs[pos, :] = pseudocount / self.n_states
            total_weight = pseudocount

            for seq_idx, record in enumerate(msa):
                aa = record.seq[pos]
                if aa in self.aa_to_idx:
                    aa_idx = self.aa_to_idx[aa]
                    freqs[pos, aa_idx] += seq_weights[seq_idx]
                    total_weight += seq_weights[seq_idx]

            # Normalize
            freqs[pos, :] /= total_weight

        return freqs

    def _calculate_joint_frequencies(
        self, msa: MultipleSeqAlignment, pos1: int, pos2: int, seq_weights: np.ndarray
    ) -> np.ndarray:
        """Calculate joint amino acid frequencies for two positions."""
        joint_freq = np.zeros((self.n_states, self.n_states))
        pseudocount = self.pseudocount

        # Add pseudocounts
        joint_freq[:, :] = pseudocount / (self.n_states**2)
        total_weight = pseudocount

        for seq_idx, record in enumerate(msa):
            aa1 = record.seq[pos1]
            aa2 = record.seq[pos2]

            if aa1 in self.aa_to_idx and aa2 in self.aa_to_idx:
                idx1 = self.aa_to_idx[aa1]
                idx2 = self.aa_to_idx[aa2]
                joint_freq[idx1, idx2] += seq_weights[seq_idx]
                total_weight += seq_weights[seq_idx]

        # Normalize
        joint_freq /= total_weight

        return joint_freq

    def _apply_apc(self, mi_matrix: np.ndarray) -> np.ndarray:
        """Apply Average Product Correction to MI matrix."""
        n_positions = mi_matrix.shape[0]

        # Calculate row/column means (excluding diagonal)
        row_means = np.zeros(n_positions)
        for i in range(n_positions):
            mask = np.ones(n_positions, dtype=bool)
            mask[i] = False
            # Also exclude positions within min_separation
            for j in range(
                max(0, i - self.min_separation + 1),
                min(n_positions, i + self.min_separation),
            ):
                mask[j] = False

            if np.any(mask):
                row_means[i] = np.mean(mi_matrix[i, mask])

        # Overall mean
        overall_mean = np.mean(row_means)

        # Apply correction
        mi_corrected = np.copy(mi_matrix)
        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                if overall_mean > 0:
                    apc = (row_means[i] * row_means[j]) / overall_mean
                    mi_corrected[i, j] -= apc
                    mi_corrected[j, i] -= apc

                # Ensure non-negative
                if mi_corrected[i, j] < 0:
                    mi_corrected[i, j] = 0
                    mi_corrected[j, i] = 0

        return mi_corrected

    def _calculate_mi_significance(
        self, mi_matrix: np.ndarray, n_sequences: int
    ) -> np.ndarray:
        """
        Calculate statistical significance of MI values.

        Uses chi-squared approximation for MI significance.
        """
        # Degrees of freedom for chi-squared test
        df = (self.n_states - 1) ** 2

        # Convert MI to chi-squared statistic
        # chi2 = 2 * n_sequences * MI
        chi2_matrix = 2 * n_sequences * mi_matrix

        # Calculate p-values
        p_values = np.zeros_like(mi_matrix)
        for i in range(mi_matrix.shape[0]):
            for j in range(i + 1, mi_matrix.shape[1]):
                if chi2_matrix[i, j] > 0:
                    p_values[i, j] = 1 - chi2.cdf(chi2_matrix[i, j], df)
                    p_values[j, i] = p_values[i, j]

        # Convert to Z-scores for significance
        # Higher Z-score = more significant
        z_scores = np.zeros_like(mi_matrix)

        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        for i in range(mi_matrix.shape[0]):
            for j in range(i + 1, mi_matrix.shape[1]):
                if p_values[i, j] < 1.0:
                    z_scores[i, j] = -np.log10(p_values[i, j] + epsilon)
                    z_scores[j, i] = z_scores[i, j]

        # Normalize to 0-1 range
        max_z = np.max(z_scores)
        if max_z > 0:
            z_scores /= max_z

        return z_scores

    def _apply_separation_filter(self, coevo_matrix: np.ndarray) -> np.ndarray:
        """Apply minimum sequence separation filter."""
        n_positions = coevo_matrix.shape[0]
        filtered = np.copy(coevo_matrix)

        for i in range(n_positions):
            for j in range(
                max(0, i - self.min_separation + 1),
                min(n_positions, i + self.min_separation),
            ):
                filtered[i, j] = 0
                filtered[j, i] = 0

        return filtered

    def _enhance_metal_coevolution(
        self,
        coevo_matrix: np.ndarray,
        metal_sites: list[MetalSite],
        msa: MultipleSeqAlignment,
    ) -> np.ndarray:
        """Enhance coevolution scores for metal-coordinating positions."""
        enhanced = np.copy(coevo_matrix)

        # Get all metal-binding positions
        metal_positions = set()
        for site in metal_sites:
            for residue in site.get_coordinating_residues():
                if residue.position <= coevo_matrix.shape[0]:
                    metal_positions.add(residue.position - 1)  # 0-indexed

        # Boost coevolution between metal-binding positions
        for pos1 in metal_positions:
            for pos2 in metal_positions:
                if pos1 != pos2 and abs(pos1 - pos2) >= self.min_separation:
                    # Check if positions show coordinated conservation
                    if self._check_coordinated_conservation(msa, pos1, pos2):
                        # Get boost factor based on metal type
                        metal_type = next(
                            site.metal_type.value
                            for site in metal_sites
                            if any(
                                r.position - 1 == pos1
                                for r in site.get_coordinating_residues()
                            )
                        )
                        boost_factor = config.get_metal_boost_factor(metal_type)
                        enhanced[pos1, pos2] *= boost_factor
                        enhanced[pos2, pos1] *= boost_factor

        # Normalize
        max_score = np.max(enhanced)
        if max_score > 1.0:
            enhanced /= max_score

        return enhanced

    def _check_coordinated_conservation(
        self, msa: MultipleSeqAlignment, pos1: int, pos2: int
    ) -> bool:
        """Check if two positions show coordinated conservation patterns."""
        # Get metal binding residues from config
        metal_binding_aas = set(config.get_property_group("metal_binding"))

        coordinated_count = 0
        total_count = 0

        for record in msa:
            aa1 = record.seq[pos1]
            aa2 = record.seq[pos2]

            if aa1 != "-" and aa2 != "-":
                total_count += 1
                # Both are metal-binding or both are not
                if (aa1 in metal_binding_aas) == (aa2 in metal_binding_aas):
                    coordinated_count += 1

        if total_count == 0:
            return False

        # Consider coordinated if >70% sequences show same pattern
        return (coordinated_count / total_count) > config.metal_binding[
            "conservation_thresholds"
        ]["binding_site"]

    def identify_coevolution_networks(
        self, coevo_matrix: np.ndarray, threshold: Optional[float] = None
    ) -> list[set[int]]:
        """
        Identify networks of coevolving positions.

        Args:
            coevo_matrix: Coevolution score matrix
            threshold: Score threshold for including edges

        Returns:
            List of position sets forming coevolution networks
        """
        if threshold is None:
            # Use top 5% of scores as threshold
            upper_tri = coevo_matrix[np.triu_indices_from(coevo_matrix, k=1)]
            threshold = np.percentile(upper_tri[upper_tri > 0], 95)

        # Build adjacency matrix
        adj_matrix = coevo_matrix > threshold

        # Find connected components
        n_positions = adj_matrix.shape[0]
        visited = np.zeros(n_positions, dtype=bool)
        networks = []

        for start_pos in range(n_positions):
            if visited[start_pos]:
                continue

            # BFS to find connected component
            network = set()
            queue = [start_pos]

            while queue:
                pos = queue.pop(0)
                if visited[pos]:
                    continue

                visited[pos] = True
                network.add(pos)

                # Add connected positions
                connected = np.where(adj_matrix[pos])[0]
                for next_pos in connected:
                    if not visited[next_pos]:
                        queue.append(next_pos)

            if len(network) > 1:  # Only keep networks with multiple positions
                networks.append(network)

        # Sort by network size
        networks.sort(key=len, reverse=True)

        return networks

    def calculate_sector_decomposition(
        self, coevo_matrix: np.ndarray, n_sectors: int = 3
    ) -> dict[int, list[int]]:
        """
        Decompose coevolution matrix into functional sectors.

        Uses spectral clustering on the coevolution matrix.

        Args:
            coevo_matrix: Coevolution score matrix
            n_sectors: Number of sectors to identify

        Returns:
            Dictionary mapping sector ID to list of positions
        """
        from sklearn.cluster import SpectralClustering

        # Convert to similarity matrix (already is one)
        similarity = coevo_matrix

        # Perform spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_sectors, affinity="precomputed", random_state=42
        )

        labels = clustering.fit_predict(similarity)

        # Group positions by sector
        sectors = {}
        for pos, label in enumerate(labels):
            if label not in sectors:
                sectors[label] = []
            sectors[label].append(pos + 1)  # 1-indexed

        return sectors

    def _validate_msa_for_coevolution(self, msa: MultipleSeqAlignment) -> None:
        """Validate MSA is suitable for coevolution analysis."""
        try:
            n_sequences = len(msa)
            n_positions = msa.get_alignment_length()

            if n_sequences < config.coevolution["min_sequences"]:
                raise ValidationError(
                    f"MSA has too few sequences for coevolution analysis: {n_sequences} "
                    f"(minimum: {config.coevolution['min_sequences']})"
                )

            if n_positions < config.coevolution["min_positions"]:
                raise ValidationError(
                    f"MSA has too few positions for coevolution analysis: {n_positions} "
                    f"(minimum: {config.coevolution['min_positions']})"
                )

            # Check sequence diversity
            unique_seqs = set(str(record.seq) for record in msa)
            if len(unique_seqs) < config.coevolution["min_unique_sequences"]:
                raise ValidationError(
                    f"MSA has too few unique sequences: {len(unique_seqs)} "
                    f"(minimum: {config.coevolution['min_unique_sequences']})"
                )

            # Check gap content
            gap_fractions = []
            for pos in range(n_positions):
                gaps = sum(1 for record in msa if record.seq[pos] == "-")
                gap_fractions.append(gaps / n_sequences)

            max_gap_fraction = max(gap_fractions)
            if max_gap_fraction > config.coevolution["max_gap_fraction"]:
                raise ValidationError(
                    f"MSA has positions with too many gaps: {max_gap_fraction:.2%} "
                    f"(maximum: {config.coevolution['max_gap_fraction']:.2%})"
                )

        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(f"MSA validation failed: {str(e)}")
            raise
