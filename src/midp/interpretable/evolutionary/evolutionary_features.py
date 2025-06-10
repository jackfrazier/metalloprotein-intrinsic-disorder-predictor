"""
Evolutionary feature extraction for metalloproteins.

This module implements evolutionary analysis to identify functionally important
residues through conservation, coevolution, and metal-binding motif detection.
Based on principles from DyNoPy and related coevolutionary analysis methods.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.spatial.distance import squareform
from scipy.stats import entropy

from src.midp.core.constants import (
    AMINO_ACID_PROPERTIES,
    COEVOLUTION_PARAMETERS,
    CONSERVATION_THRESHOLDS,
    METAL_BINDING_PREFERENCES,
    MSA_QUALITY_THRESHOLDS,
)
from src.midp.core.data_structures import (
    EvolutionaryFeatures,
    MetalType,
    ProteinData,
    ResidueType,
)
from src.midp.core.exceptions import (
    DataAccessError,
    EvolutionaryAnalysisError,
    ValidationError,
)
from src.midp.core.interfaces import EvolutionaryAnalyzer

logger = logging.getLogger(__name__)


class MetalloproteinEvolutionaryAnalyzer(EvolutionaryAnalyzer):
    """
    Analyzes evolutionary patterns in metalloproteins to identify functional sites.

    This implementation focuses on:
    1. Conservation analysis with phylogenetic correction
    2. Coevolution detection using mutual information and DCA
    3. Metal-binding motif identification
    4. Functional site prediction from evolutionary couplings
    """

    def __init__(
        self,
        use_hhblits: bool = True,
        database_path: Optional[Path] = None,
        n_iterations: int = 3,
        e_value_threshold: float = 1e-3,
    ):
        """
        Initialize the evolutionary analyzer.

        Args:
            use_hhblits: Whether to use HHblits for MSA generation
            database_path: Path to sequence database (e.g., UniRef30)
            n_iterations: Number of HHblits iterations
            e_value_threshold: E-value threshold for sequence inclusion
        """
        self.use_hhblits = use_hhblits
        self.database_path = database_path
        self.n_iterations = n_iterations
        self.e_value_threshold = e_value_threshold

        # Metal-binding patterns from literature
        self.metal_binding_patterns = {
            "zinc_finger_CCHH": "C..C.{12,}H..H",
            "zinc_finger_CCCC": "C..C.{12,}C..C",
            "iron_sulfur_4Fe4S": "C..C..C.{5,}C",
            "copper_type1": "H.{5,}C.{3,}H.{3,}M",
            "calcium_ef_hand": "D.{3}[DNS].{3}[DNS]",
            "heme_binding": "C..CH",
        }

    def analyze(
        self, protein_data: ProteinData, msa_file: Optional[Path] = None
    ) -> EvolutionaryFeatures:
        """
        Perform comprehensive evolutionary analysis.

        Args:
            protein_data: Protein information
            msa_file: Optional pre-computed MSA file

        Returns:
            EvolutionaryFeatures with conservation, coevolution, and functional sites
        """
        try:
            # Get or generate MSA
            if msa_file and msa_file.exists():
                logger.info(f"Loading MSA from {msa_file}")
                msa = AlignIO.read(str(msa_file), "stockholm")
            else:
                logger.info("Generating MSA")
                msa = self._generate_msa(protein_data)

            # Validate MSA quality
            self._validate_msa(msa)

            # Calculate conservation with phylogenetic correction
            conservation_scores = self.compute_conservation(msa)

            # Calculate coevolution matrix
            coevolution_matrix = self.compute_coevolution(msa)

            # Identify functional sites
            functional_sites = self._identify_functional_sites(
                conservation_scores, coevolution_matrix, protein_data
            )

            # Calculate phylogenetic diversity
            phylogenetic_diversity = self._calculate_phylogenetic_diversity(msa)

            # Create and return features
            features = EvolutionaryFeatures(
                conservation_scores=conservation_scores,
                coevolution_matrix=coevolution_matrix,
                functional_sites=functional_sites,
                phylogenetic_diversity=phylogenetic_diversity,
            )

            # Add metal-specific analysis
            self._analyze_metal_binding_evolution(features, protein_data, msa)

            logger.info(
                f"Evolutionary analysis complete: "
                f"{len(features.functional_sites)} functional sites identified"
            )

            return features

        except Exception as e:
            raise EvolutionaryAnalysisError(
                "comprehensive_analysis", f"Failed to analyze evolution: {str(e)}"
            )

    def compute_conservation(self, msa: MultipleSeqAlignment) -> np.ndarray:
        """
        Calculate position-specific conservation scores with phylogenetic correction.

        Uses a combination of:
        - Shannon entropy
        - Property-based conservation (hydrophobicity, charge, etc.)
        - Phylogenetic weighting to reduce bias
        """
        n_positions = msa.get_alignment_length()
        conservation_scores = np.zeros(n_positions)

        # Calculate sequence weights to correct for phylogenetic bias
        seq_weights = self._calculate_sequence_weights(msa)

        # Amino acid groups for property conservation
        aa_groups = {
            "hydrophobic": set("AILMFVW"),
            "aromatic": set("FWY"),
            "polar": set("STNQ"),
            "charged": set("DEKR"),
            "small": set("AGST"),
            "proline": set("P"),
            "cysteine": set("C"),
            "histidine": set("H"),
        }

        for pos in range(n_positions):
            # Get weighted amino acid frequencies
            aa_counts = {}
            total_weight = 0.0

            for i, record in enumerate(msa):
                aa = record.seq[pos]
                if aa != "-" and aa != "X":
                    aa_counts[aa] = aa_counts.get(aa, 0) + seq_weights[i]
                    total_weight += seq_weights[i]

            if total_weight == 0:
                conservation_scores[pos] = 0.0
                continue

            # Normalize frequencies
            aa_freqs = {aa: count / total_weight for aa, count in aa_counts.items()}

            # Calculate Shannon entropy
            entropy_score = 0.0
            for freq in aa_freqs.values():
                if freq > 0:
                    entropy_score -= freq * np.log2(freq)

            # Normalize entropy (max entropy = log2(20) for 20 amino acids)
            entropy_score = 1.0 - (entropy_score / np.log2(20))

            # Calculate property conservation
            property_scores = []
            for group_name, group_aas in aa_groups.items():
                group_freq = sum(aa_freqs.get(aa, 0) for aa in group_aas)
                if group_freq > 0.8:  # Highly conserved property
                    property_scores.append(1.0)
                elif group_freq > 0.6:  # Moderately conserved
                    property_scores.append(0.5)
                else:
                    property_scores.append(0.0)

            # Combine entropy and property conservation
            property_conservation = np.mean(property_scores) if property_scores else 0.0
            conservation_scores[pos] = 0.7 * entropy_score + 0.3 * property_conservation

            # Boost conservation for metal-binding residues
            if self._is_metal_binding_position(aa_freqs):
                conservation_scores[pos] = min(1.0, conservation_scores[pos] * 1.2)

        return conservation_scores

    def compute_coevolution(self, msa: MultipleSeqAlignment) -> np.ndarray:
        """
        Calculate pairwise coevolution scores using mutual information with APC.

        This implementation follows the approach from:
        - Dunn et al. (2008) for mutual information
        - DyNoPy methodology for identifying functionally coupled residues
        """
        n_positions = msa.get_alignment_length()

        # Calculate sequence weights
        seq_weights = self._calculate_sequence_weights(msa)

        # Calculate weighted mutual information matrix
        mi_matrix = self._calculate_mutual_information(msa, seq_weights)

        # Apply average product correction (APC) to reduce phylogenetic bias
        if COEVOLUTION_PARAMETERS["apc_correction"]:
            mi_matrix = self._apply_apc_correction(mi_matrix)

        # Apply minimum separation filter
        min_sep = COEVOLUTION_PARAMETERS["min_separation"]
        for i in range(n_positions):
            for j in range(max(0, i - min_sep + 1), min(n_positions, i + min_sep)):
                mi_matrix[i, j] = 0.0
                mi_matrix[j, i] = 0.0

        # Identify significant coevolution (top 5% of scores)
        threshold = np.percentile(
            mi_matrix[np.triu_indices(n_positions, k=min_sep)], 95
        )
        mi_matrix[mi_matrix < threshold] = 0.0

        return mi_matrix

    def _generate_msa(self, protein_data: ProteinData) -> MultipleSeqAlignment:
        """Generate MSA using HHblits or fallback methods."""
        if self.use_hhblits and self.database_path:
            return self._run_hhblits(protein_data)
        else:
            # Fallback: Use BLAST against UniProt
            return self._run_blast_fallback(protein_data)

    def _run_hhblits(self, protein_data: ProteinData) -> MultipleSeqAlignment:
        """Run HHblits to generate MSA."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write query sequence
            query_file = tmp_path / "query.fasta"
            with open(query_file, "w") as f:
                f.write(f">{protein_data.protein_id}\n{protein_data.sequence}\n")

            # Output files
            msa_file = tmp_path / "msa.a3m"

            # Run HHblits
            cmd = [
                "hhblits",
                "-i",
                str(query_file),
                "-d",
                str(self.database_path),
                "-oa3m",
                str(msa_file),
                "-n",
                str(self.n_iterations),
                "-e",
                str(self.e_value_threshold),
                "-cpu",
                "4",
                "-v",
                "0",
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"HHblits failed: {e.stderr}")
                raise EvolutionaryAnalysisError("MSA generation", "HHblits failed")

            # Parse A3M format
            return self._parse_a3m(msa_file)

    def _run_blast_fallback(self, protein_data: ProteinData) -> MultipleSeqAlignment:
        """Fallback MSA generation using BLAST."""
        # This is a simplified implementation
        # In production, you would query UniProt or nr database
        logger.warning("Using fallback MSA generation - results may be limited")

        # Create a minimal MSA with just the query sequence
        record = SeqRecord(
            Seq(protein_data.sequence), id=protein_data.protein_id, description=""
        )

        return MultipleSeqAlignment([record])

    def _parse_a3m(self, a3m_file: Path) -> MultipleSeqAlignment:
        """Parse A3M format MSA file."""
        records = []
        with open(a3m_file) as f:
            current_id = None
            current_seq = []

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        # Remove lowercase letters (insertions in A3M format)
                        seq = "".join(
                            c for c in "".join(current_seq) if not c.islower()
                        )
                        records.append(
                            SeqRecord(Seq(seq), id=current_id, description="")
                        )
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            # Don't forget the last sequence
            if current_id and current_seq:
                seq = "".join(c for c in "".join(current_seq) if not c.islower())
                records.append(SeqRecord(Seq(seq), id=current_id, description=""))

        return MultipleSeqAlignment(records)

    def _validate_msa(self, msa: MultipleSeqAlignment):
        """Validate MSA quality against thresholds."""
        n_sequences = len(msa)
        n_positions = msa.get_alignment_length()

        # Check minimum sequences
        if n_sequences < MSA_QUALITY_THRESHOLDS["min_sequences"]:
            logger.warning(f"Low sequence count: {n_sequences}")

        # Calculate effective sequences (sequence clustering at 62% identity)
        n_effective = self._calculate_effective_sequences(msa)
        if n_effective < MSA_QUALITY_THRESHOLDS["min_effective_sequences"]:
            logger.warning(f"Low effective sequence count: {n_effective}")

        # Check gap statistics
        gap_fractions = []
        for pos in range(n_positions):
            gaps = sum(1 for record in msa if record.seq[pos] == "-")
            gap_fractions.append(gaps / n_sequences)

        mean_gap_fraction = np.mean(gap_fractions)
        if mean_gap_fraction > MSA_QUALITY_THRESHOLDS["max_gap_fraction"]:
            logger.warning(f"High gap fraction: {mean_gap_fraction:.2%}")

    def _calculate_sequence_weights(self, msa: MultipleSeqAlignment) -> np.ndarray:
        """Calculate sequence weights to correct for phylogenetic bias."""
        n_sequences = len(msa)
        weights = np.ones(n_sequences)

        # Simple position-based sequence weighting (Henikoff & Henikoff)
        for pos in range(msa.get_alignment_length()):
            # Count different amino acids at this position
            aa_counts = {}
            for i, record in enumerate(msa):
                aa = record.seq[pos]
                if aa != "-":
                    if aa not in aa_counts:
                        aa_counts[aa] = []
                    aa_counts[aa].append(i)

            # Weight sequences inversely by amino acid frequency
            for aa, seq_indices in aa_counts.items():
                weight = 1.0 / (len(aa_counts) * len(seq_indices))
                for idx in seq_indices:
                    weights[idx] += weight

        # Normalize weights
        weights = weights / np.sum(weights) * n_sequences

        return weights

    def _calculate_effective_sequences(self, msa: MultipleSeqAlignment) -> int:
        """Calculate number of effective sequences at 62% identity threshold."""
        n_sequences = len(msa)
        identity_threshold = 0.62

        # Build identity matrix
        clustered = np.zeros(n_sequences, dtype=bool)
        n_clusters = 0

        for i in range(n_sequences):
            if clustered[i]:
                continue

            # Start new cluster
            n_clusters += 1
            clustered[i] = True

            # Find similar sequences
            seq_i = str(msa[i].seq)
            for j in range(i + 1, n_sequences):
                if clustered[j]:
                    continue

                seq_j = str(msa[j].seq)
                identity = sum(
                    1 for a, b in zip(seq_i, seq_j) if a == b and a != "-"
                ) / len(seq_i)

                if identity >= identity_threshold:
                    clustered[j] = True

        return n_clusters

    def _is_metal_binding_position(self, aa_freqs: dict[str, float]) -> bool:
        """Check if position shows metal-binding amino acid preferences."""
        metal_binding_aas = set("CHMDE")  # Common metal-binding residues
        metal_freq = sum(aa_freqs.get(aa, 0) for aa in metal_binding_aas)
        return metal_freq > 0.6

    def _calculate_mutual_information(
        self, msa: MultipleSeqAlignment, seq_weights: np.ndarray
    ) -> np.ndarray:
        """Calculate weighted mutual information between all position pairs."""
        n_positions = msa.get_alignment_length()
        mi_matrix = np.zeros((n_positions, n_positions))

        # Pseudocount for probability estimation
        pseudocount = COEVOLUTION_PARAMETERS["pseudocount"]

        # Alphabet (20 amino acids + gap)
        alphabet = list("ACDEFGHIKLMNPQRSTVWY-")
        aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}

        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                # Calculate joint and marginal probabilities
                joint_counts = np.zeros((21, 21))
                marginal_i = np.zeros(21)
                marginal_j = np.zeros(21)

                total_weight = 0.0
                for k, record in enumerate(msa):
                    aa_i = record.seq[i]
                    aa_j = record.seq[j]

                    if aa_i in aa_to_idx and aa_j in aa_to_idx:
                        idx_i = aa_to_idx[aa_i]
                        idx_j = aa_to_idx[aa_j]
                        weight = seq_weights[k]

                        joint_counts[idx_i, idx_j] += weight
                        marginal_i[idx_i] += weight
                        marginal_j[idx_j] += weight
                        total_weight += weight

                if total_weight == 0:
                    continue

                # Add pseudocounts and normalize
                joint_counts += pseudocount
                marginal_i += pseudocount * 21
                marginal_j += pseudocount * 21
                total_weight += pseudocount * 21 * 21

                joint_probs = joint_counts / total_weight
                marginal_i_probs = marginal_i / total_weight
                marginal_j_probs = marginal_j / total_weight

                # Calculate mutual information
                mi = 0.0
                for idx_i in range(21):
                    for idx_j in range(21):
                        if joint_probs[idx_i, idx_j] > 0:
                            mi += joint_probs[idx_i, idx_j] * np.log2(
                                joint_probs[idx_i, idx_j]
                                / (marginal_i_probs[idx_i] * marginal_j_probs[idx_j])
                            )

                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi

        return mi_matrix

    def _apply_apc_correction(self, mi_matrix: np.ndarray) -> np.ndarray:
        """Apply Average Product Correction to reduce phylogenetic noise."""
        n_positions = mi_matrix.shape[0]

        # Calculate average MI for each position
        avg_mi = np.zeros(n_positions)
        for i in range(n_positions):
            # Exclude diagonal and positions within min_separation
            mask = np.ones(n_positions, dtype=bool)
            mask[i] = False
            min_sep = COEVOLUTION_PARAMETERS["min_separation"]
            mask[max(0, i - min_sep + 1) : min(n_positions, i + min_sep)] = False

            if np.any(mask):
                avg_mi[i] = np.mean(mi_matrix[i, mask])

        # Calculate overall average
        overall_avg = np.mean(avg_mi)

        # Apply APC correction
        mi_corrected = np.copy(mi_matrix)
        for i in range(n_positions):
            for j in range(i + 1, n_positions):
                apc = (avg_mi[i] * avg_mi[j]) / overall_avg
                mi_corrected[i, j] -= apc
                mi_corrected[j, i] -= apc

                # Ensure non-negative
                if mi_corrected[i, j] < 0:
                    mi_corrected[i, j] = 0
                    mi_corrected[j, i] = 0

        return mi_corrected

    def _identify_functional_sites(
        self,
        conservation_scores: np.ndarray,
        coevolution_matrix: np.ndarray,
        protein_data: ProteinData,
    ) -> list[tuple[int, float]]:
        """Identify functional sites from conservation and coevolution patterns."""
        functional_sites = []

        # High conservation sites
        high_cons_threshold = CONSERVATION_THRESHOLDS["highly_conserved"]
        for i, score in enumerate(conservation_scores):
            if score > high_cons_threshold:
                functional_sites.append((i + 1, score))  # 1-indexed

        # Coevolution hubs (positions with many strong couplings)
        n_positions = len(conservation_scores)
        coevolution_degree = np.sum(coevolution_matrix > 0, axis=1)

        # Top 10% most connected positions
        hub_threshold = np.percentile(coevolution_degree, 90)
        for i in range(n_positions):
            if coevolution_degree[i] > hub_threshold:
                # Boost score if also conserved
                base_score = conservation_scores[i]
                hub_score = min(1.0, base_score * 1.5)

                # Check if already in list
                existing = [s for s in functional_sites if s[0] == i + 1]
                if not existing:
                    functional_sites.append((i + 1, hub_score))
                elif existing[0][1] < hub_score:
                    # Update with higher score
                    functional_sites = [
                        (p, s) for p, s in functional_sites if p != i + 1
                    ]
                    functional_sites.append((i + 1, hub_score))

        # Metal-binding positions from structure
        if protein_data.metal_sites:
            for site in protein_data.metal_sites:
                for residue in site.get_coordinating_residues():
                    pos = residue.position
                    # Check conservation at metal-binding position
                    if pos <= len(conservation_scores):
                        score = conservation_scores[pos - 1]
                        if score > 0.5:  # Moderately conserved
                            existing = [s for s in functional_sites if s[0] == pos]
                            if not existing:
                                functional_sites.append((pos, min(1.0, score * 1.3)))

        # Sort by score
        functional_sites.sort(key=lambda x: x[1], reverse=True)

        return functional_sites

    def _calculate_phylogenetic_diversity(self, msa: MultipleSeqAlignment) -> float:
        """Calculate phylogenetic diversity of the MSA."""
        # Simple metric: average pairwise sequence identity
        n_sequences = min(len(msa), 100)  # Limit for computational efficiency

        if n_sequences < 2:
            return 0.0

        identities = []
        for i in range(n_sequences):
            for j in range(i + 1, n_sequences):
                seq_i = str(msa[i].seq)
                seq_j = str(msa[j].seq)

                matches = sum(1 for a, b in zip(seq_i, seq_j) if a == b and a != "-")
                length = sum(1 for a, b in zip(seq_i, seq_j) if a != "-" or b != "-")

                if length > 0:
                    identities.append(matches / length)

        # Diversity is inverse of average identity
        avg_identity = np.mean(identities) if identities else 1.0
        diversity = 1.0 - avg_identity

        return diversity

    def _analyze_metal_binding_evolution(
        self,
        features: EvolutionaryFeatures,
        protein_data: ProteinData,
        msa: MultipleSeqAlignment,
    ):
        """Add metal-specific evolutionary analysis."""
        # Look for conserved metal-binding motifs
        sequence = protein_data.sequence

        import re

        for motif_name, pattern in self.metal_binding_patterns.items():
            matches = list(re.finditer(pattern, sequence))
            for match in matches:
                start, end = match.span()
                # Check conservation in motif region
                motif_conservation = np.mean(features.conservation_scores[start:end])

                if motif_conservation > 0.7:
                    logger.info(f"Found conserved {motif_name} at {start+1}-{end}")

                    # Add to functional sites if not already present
                    for pos in range(start + 1, end + 1):
                        if not any(s[0] == pos for s in features.functional_sites):
                            features.functional_sites.append((pos, motif_conservation))

        # Analyze coevolution networks around known metal sites
        if protein_data.metal_sites:
            for metal_site in protein_data.metal_sites:
                metal_positions = [
                    r.position - 1
                    for r in metal_site.get_coordinating_residues()
                    if r.position <= len(features.conservation_scores)
                ]

                # Find positions that coevolve with metal-binding residues
                metal_network = set()
                for pos in metal_positions:
                    # Get strongly coupled positions
                    coupled = np.where(features.coevolution_matrix[pos] > 0)[0]
                    metal_network.update(coupled)

                logger.info(
                    f"Metal site at {metal_site.center} has coevolution network "
                    f"of {len(metal_network)} residues"
                )
