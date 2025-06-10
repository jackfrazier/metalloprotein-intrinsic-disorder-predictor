"""
Coevolution analysis for detecting functionally coupled residues.

This module implements methods to identify coevolving residue pairs that may
indicate functional coupling, allosteric networks, or coordinated metal binding.
"""

import logging
from typing import Optional, Dict, List, Set

import numpy as np
from Bio.Align import MultipleSeqAlignment
from scipy.spatial.distance import squareform
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

from src.midp.core.constants import COEVOLUTION_PARAMETERS
from src.midp.core.data_structures import MetalSite

logger = logging.getLogger(__name__)


class CoevolutionAnalyzer:
    """
    Detects coevolving residue pairs using multiple methods.

    Implements:
    - Mutual Information (MI) with APC correction
    - Direct Coupling Analysis (DCA) approximation
    - Network analysis of coevolution patterns
    - Metal-specific coevolution detection
    """

    def __init__(
        self,
        method: str = "mi_apc",
        min_separation: int = 5,
        significance_threshold: float = 0.05,
    ):
        """
        Initialize coevolution analyzer.

        Args:
            method: Coevolution detection method ("mi_apc", "dca", "both")
            min_separation: Minimum sequence separation for coupling
            significance_threshold: P-value threshold for significant coupling
        """
        self.method = method
        self.min_separation = min_separation
        self.significance_threshold = significance_threshold

        # Alphabet for calculations
        self.alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.alphabet)}
        self.n_states = len(self.alphabet)

    def calculate_coevolution_matrix(
        self,
        msa: MultipleSeqAlignment,
        seq_weights: Optional[np.ndarray] = None,
        metal_sites: Optional[list[MetalSite]] = None,
    ) -> np.ndarray:
        """
        Calculate the full coevolution matrix.

        Args:
            msa: Multiple sequence alignment
            seq_weights: Sequence weights to reduce bias
            metal_sites: Known metal sites for enhanced analysis

        Returns:
            Symmetric matrix of coevolution scores
        """
        if self.method == "mi_apc":
            coevo_matrix = self._calculate_mi_apc(msa, seq_weights)
        elif self.method == "dca":
            coevo_matrix = self._calculate_dca_scores(msa, seq_weights)
        else:  # "both"
            mi_matrix = self._calculate_mi_apc(msa, seq_weights)
            dca_matrix = self._calculate_dca_scores(msa, seq_weights)
            # Combine methods with equal weight
            coevo_matrix = 0.5 * mi_matrix + 0.5 * dca_matrix

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
        pseudocount = COEVOLUTION_PARAMETERS["pseudocount"]

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
        pseudocount = COEVOLUTION_PARAMETERS["pseudocount"]

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

    def _calculate_dca_scores(
        self, msa: MultipleSeqAlignment, seq_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate Direct Coupling Analysis scores.

        This is a simplified DCA implementation using inverse covariance.
        """
        n_positions = msa.get_alignment_length()

        # Convert MSA to numerical matrix
        msa_matrix = self._msa_to_numerical(msa)

        if seq_weights is None:
            seq_weights = np.ones(len(msa)) / len(msa)

        # Weight sequences
        weighted_msa = msa_matrix * seq_weights[:, np.newaxis]

        # Calculate covariance matrix
        cov_matrix = np.cov(weighted_msa.T)

        # Add regularization to ensure invertibility
        reg_param = 0.01
        cov_matrix += reg_param * np.eye(n_positions)

        # Calculate inverse covariance (precision matrix)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix inversion failed, using pseudoinverse")
            inv_cov = np.linalg.pinv(cov_matrix)

        # Convert to coupling strengths
        coupling_matrix = np.abs(inv_cov)

        # Zero diagonal
        np.fill_diagonal(coupling_matrix, 0)

        # Normalize
        max_coupling = np.max(coupling_matrix)
        if max_coupling > 0:
            coupling_matrix /= max_coupling

        return coupling_matrix

    def _msa_to_numerical(self, msa: MultipleSeqAlignment) -> np.ndarray:
        """Convert MSA to numerical matrix for DCA."""
        n_sequences = len(msa)
        n_positions = msa.get_alignment_length()

        # Simple numerical encoding (more sophisticated encoding possible)
        numerical_msa = np.zeros((n_sequences, n_positions))

        for seq_idx, record in enumerate(msa):
            for pos_idx, aa in enumerate(record.seq):
                if aa in self.aa_to_idx:
                    # Use hydrophobicity as numerical feature
                    # (could use more sophisticated encoding)
                    hydrophobicity_scale = {
                        "A": 1.8,
                        "R": -4.5,
                        "N": -3.5,
                        "D": -3.5,
                        "C": 2.5,
                        "Q": -3.5,
                        "E": -3.5,
                        "G": -0.4,
                        "H": -3.2,
                        "I": 4.5,
                        "L": 3.8,
                        "K": -3.9,
                        "M": 1.9,
                        "F": 2.8,
                        "P": -1.6,
                        "S": -0.8,
                        "T": -0.7,
                        "W": -0.9,
                        "Y": -1.3,
                        "V": 4.2,
                    }
                    numerical_msa[seq_idx, pos_idx] = hydrophobicity_scale.get(aa, 0)

        # Standardize features
        scaler = StandardScaler()
        numerical_msa = scaler.fit_transform(numerical_msa)

        return numerical_msa

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
                        # Boost by 20%
                        enhanced[pos1, pos2] *= 1.2
                        enhanced[pos2, pos1] *= 1.2

        # Normalize
        max_score = np.max(enhanced)
        if max_score > 1.0:
            enhanced /= max_score

        return enhanced

    def _check_coordinated_conservation(
        self, msa: MultipleSeqAlignment, pos1: int, pos2: int
    ) -> bool:
        """Check if two positions show coordinated conservation patterns."""
        # Simple check: positions co-vary in metal-binding residues
        metal_binding_aas = set("CHMDE")

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
        return (coordinated_count / total_count) > 0.7

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
