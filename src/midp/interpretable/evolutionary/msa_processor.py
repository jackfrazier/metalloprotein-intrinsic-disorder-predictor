"""
MSA processing utilities for evolutionary analysis.

This module provides tools for MSA manipulation, filtering, and quality assessment
specifically tailored for metalloprotein analysis.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.midp.core.constants import MSA_QUALITY_THRESHOLDS
from src.midp.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class MSAProcessor:
    """
    Utilities for processing and filtering multiple sequence alignments.

    Provides methods for:
    - MSA filtering and quality control
    - Gap handling and column removal
    - Sequence weighting and clustering
    - Metal-binding site preservation during filtering
    """

    def __init__(
        self,
        max_gap_fraction: float = 0.5,
        min_coverage: float = 0.7,
        identity_threshold: float = 0.9,
    ):
        """
        Initialize MSA processor.

        Args:
            max_gap_fraction: Maximum allowed gap fraction per column
            min_coverage: Minimum sequence coverage required
            identity_threshold: Threshold for removing redundant sequences
        """
        self.max_gap_fraction = max_gap_fraction
        self.min_coverage = min_coverage
        self.identity_threshold = identity_threshold

    def filter_msa(
        self, msa: MultipleSeqAlignment, preserve_positions: Optional[list[int]] = None
    ) -> MultipleSeqAlignment:
        """
        Filter MSA to remove low-quality sequences and columns.

        Args:
            msa: Input MSA
            preserve_positions: Positions to preserve (e.g., metal-binding sites)

        Returns:
            Filtered MSA
        """
        # First, filter sequences by coverage
        filtered_seqs = self._filter_sequences_by_coverage(msa)

        # Remove highly similar sequences
        filtered_seqs = self._remove_redundant_sequences(filtered_seqs)

        # Filter columns by gap content
        filtered_msa = self._filter_columns_by_gaps(filtered_seqs, preserve_positions)

        # Validate filtered MSA
        if len(filtered_msa) < MSA_QUALITY_THRESHOLDS["min_sequences"]:
            logger.warning(f"Filtered MSA has only {len(filtered_msa)} sequences")

        return filtered_msa

    def _filter_sequences_by_coverage(
        self, msa: MultipleSeqAlignment
    ) -> MultipleSeqAlignment:
        """Filter sequences with too many gaps."""
        filtered_records = []

        for record in msa:
            seq_str = str(record.seq)
            gap_count = seq_str.count("-")
            coverage = 1.0 - (gap_count / len(seq_str))

            if coverage >= self.min_coverage:
                filtered_records.append(record)
            else:
                logger.debug(
                    f"Removing sequence {record.id} with " f"coverage {coverage:.2%}"
                )

        return MultipleSeqAlignment(filtered_records)

    def _remove_redundant_sequences(
        self, msa: MultipleSeqAlignment
    ) -> MultipleSeqAlignment:
        """Remove sequences that are too similar to others."""
        n_seqs = len(msa)
        if n_seqs <= MSA_QUALITY_THRESHOLDS["min_sequences"]:
            return msa  # Don't filter if we have few sequences

        # Calculate pairwise identities
        keep_indices = []
        keep_indices.append(0)  # Always keep query sequence

        for i in range(1, n_seqs):
            seq_i = str(msa[i].seq)
            is_redundant = False

            for j in keep_indices:
                seq_j = str(msa[j].seq)
                identity = self._calculate_sequence_identity(seq_i, seq_j)

                if identity > self.identity_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                keep_indices.append(i)

        filtered_records = [msa[i] for i in keep_indices]

        logger.info(f"Removed {n_seqs - len(filtered_records)} redundant sequences")

        return MultipleSeqAlignment(filtered_records)

    def _filter_columns_by_gaps(
        self, msa: MultipleSeqAlignment, preserve_positions: Optional[list[int]] = None
    ) -> MultipleSeqAlignment:
        """Remove columns with too many gaps."""
        n_seqs = len(msa)
        n_cols = msa.get_alignment_length()

        # Identify columns to keep
        keep_columns = []
        preserve_set = set(preserve_positions) if preserve_positions else set()

        for col_idx in range(n_cols):
            # Always preserve specified positions
            if col_idx in preserve_set:
                keep_columns.append(col_idx)
                continue

            # Count gaps in column
            gap_count = sum(1 for record in msa if record.seq[col_idx] == "-")
            gap_fraction = gap_count / n_seqs

            if gap_fraction <= self.max_gap_fraction:
                keep_columns.append(col_idx)

        # Create new alignment with filtered columns
        filtered_records = []
        for record in msa:
            filtered_seq = "".join(record.seq[i] for i in keep_columns)
            filtered_record = SeqRecord(
                Seq(filtered_seq), id=record.id, description=record.description
            )
            filtered_records.append(filtered_record)

        logger.info(
            f"Filtered {n_cols - len(keep_columns)} columns with high gap content"
        )

        return MultipleSeqAlignment(filtered_records)

    def _calculate_sequence_identity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence identity between two aligned sequences."""
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != "-")

        # Count positions where both sequences have residues
        valid_positions = sum(1 for a, b in zip(seq1, seq2) if a != "-" and b != "-")

        if valid_positions == 0:
            return 0.0

        return matches / valid_positions

    def extract_metal_binding_columns(
        self, msa: MultipleSeqAlignment, metal_positions: list[int]
    ) -> dict[int, list[str]]:
        """
        Extract columns corresponding to metal-binding positions.

        Args:
            msa: Multiple sequence alignment
            metal_positions: list of metal-binding positions (1-indexed)

        Returns:
            dictionary mapping positions to lists of amino acids
        """
        metal_columns = {}

        for pos in metal_positions:
            if pos <= msa.get_alignment_length():
                col_idx = pos - 1  # Convert to 0-indexed
                amino_acids = []

                for record in msa:
                    aa = record.seq[col_idx]
                    if aa != "-" and aa != "X":
                        amino_acids.append(aa)

                metal_columns[pos] = amino_acids

        return metal_columns

    def calculate_positional_weights(self, msa: MultipleSeqAlignment) -> np.ndarray:
        """
        Calculate position-specific sequence weights.

        Uses the Henikoff & Henikoff position-based method to reduce
        bias from overrepresented sequences.
        """
        n_seqs = len(msa)
        n_cols = msa.get_alignment_length()

        # Initialize weights
        weights = np.zeros((n_seqs, n_cols))

        for col_idx in range(n_cols):
            # Count amino acid types at this position
            aa_counts = {}
            seq_indices = {}

            for seq_idx, record in enumerate(msa):
                aa = record.seq[col_idx]
                if aa != "-":
                    if aa not in aa_counts:
                        aa_counts[aa] = 0
                        seq_indices[aa] = []
                    aa_counts[aa] += 1
                    seq_indices[aa].append(seq_idx)

            # Calculate weights
            n_types = len(aa_counts)
            if n_types > 0:
                for aa, indices in seq_indices.items():
                    weight = 1.0 / (n_types * aa_counts[aa])
                    for idx in indices:
                        weights[idx, col_idx] = weight

        # Average weights across positions for each sequence
        seq_weights = np.mean(weights, axis=1)

        # Normalize to sum to number of sequences
        seq_weights = seq_weights / np.sum(seq_weights) * n_seqs

        return seq_weights

    def identify_conserved_motifs(
        self,
        msa: MultipleSeqAlignment,
        window_size: int = 5,
        conservation_threshold: float = 0.8,
    ) -> list[tuple[int, int, float]]:
        """
        Identify conserved sequence motifs in the MSA.

        Args:
            msa: Multiple sequence alignment
            window_size: Size of sliding window
            conservation_threshold: Minimum conservation score

        Returns:
            list of (start, end, score) tuples for conserved motifs
        """
        n_cols = msa.get_alignment_length()
        conservation_scores = self._calculate_conservation_scores(msa)

        motifs = []
        i = 0

        while i < n_cols - window_size + 1:
            # Calculate average conservation in window
            window_score = np.mean(conservation_scores[i : i + window_size])

            if window_score >= conservation_threshold:
                # Extend motif as long as conservation remains high
                start = i
                end = i + window_size

                while (
                    end < n_cols
                    and conservation_scores[end] >= conservation_threshold * 0.9
                ):
                    end += 1

                motifs.append((start + 1, end, window_score))  # 1-indexed
                i = end
            else:
                i += 1

        return motifs

    def _calculate_conservation_scores(self, msa: MultipleSeqAlignment) -> np.ndarray:
        """Calculate simple conservation scores for each position."""
        n_cols = msa.get_alignment_length()
        scores = np.zeros(n_cols)

        for col_idx in range(n_cols):
            # Count amino acid frequencies
            aa_counts = {}
            total = 0

            for record in msa:
                aa = record.seq[col_idx]
                if aa != "-" and aa != "X":
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1
                    total += 1

            if total > 0:
                # Calculate entropy
                entropy = 0.0
                for count in aa_counts.values():
                    freq = count / total
                    if freq > 0:
                        entropy -= freq * np.log2(freq)

                # Convert to conservation (max entropy for 20 AA is log2(20))
                max_entropy = np.log2(20)
                scores[col_idx] = 1.0 - (entropy / max_entropy)

        return scores

    def save_filtered_msa(
        self, msa: MultipleSeqAlignment, output_path: str, format: str = "stockholm"
    ):
        """Save filtered MSA to file."""
        AlignIO.write(msa, output_path, format)
        logger.info(f"Saved filtered MSA to {output_path}")


def create_metal_aware_msa_filter(metal_positions: list[int]) -> MSAProcessor:
    """
    Create an MSA processor that preserves metal-binding positions.

    Args:
        metal_positions: list of metal-binding positions to preserve

    Returns:
        Configured MSAProcessor instance
    """
    processor = MSAProcessor(
        max_gap_fraction=MSA_QUALITY_THRESHOLDS["max_gap_fraction"],
        min_coverage=MSA_QUALITY_THRESHOLDS["min_coverage"],
    )

    # Store metal positions for use in filtering
    processor.metal_positions = metal_positions

    return processor
