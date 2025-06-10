"""
Evolutionary feature extraction for metalloproteins.

This module implements evolutionary analysis to identify functionally important
residues through conservation, coevolution, and metal-binding motif detection.
Based on principles from DyNoPy and related coevolutionary analysis methods.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment, PairwiseAligner, substitution_matrices
from Bio.Blast import NCBIXML, NCBIWWW
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from scipy.spatial.distance import squareform
from scipy.stats import entropy

from midp.core.constants import (
    AMINO_ACID_PROPERTIES,
    COEVOLUTION_PARAMETERS,
    CONSERVATION_THRESHOLDS,
    METAL_BINDING_PREFERENCES,
    MSA_QUALITY_THRESHOLDS,
)
from midp.core.data_structures import (
    EvolutionaryFeatures,
    MetalType,
    ProteinData,
    ResidueType,
    MetalSite,
)
from midp.core.exceptions import (
    DataAccessError,
    EvolutionaryAnalysisError,
    ValidationError,
    ScientificCalculationError,
)
from midp.core.interfaces import EvolutionaryAnalyzer
from .config import EvolutionaryConfig
from .conservation import ConservationAnalyzer
from .coevolution import CoevolutionAnalyzer

logger = logging.getLogger(__name__)

# Get configuration instance
config = EvolutionaryConfig.get_instance()


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
        *,  # Force keyword arguments
        use_hhblits: bool = True,
        database_path: Optional[Path] = None,
        n_iterations: int = 3,
        e_value_threshold: float = 1e-3,
    ) -> None:
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

        # Load MSA parameters from config
        msa_config = config.msa
        self.min_sequences = msa_config["min_sequences"]
        self.min_effective_sequences = msa_config["min_effective_sequences"]
        self.max_gap_fraction = msa_config["max_gap_fraction"]
        self.min_coverage = msa_config["min_coverage"]
        self.identity_threshold = msa_config["identity_threshold"]
        self.chunk_size = msa_config["chunk_size"]

        # Metal-binding patterns from literature
        self.metal_binding_patterns = {
            "zinc_finger_CCHH": "C..C.{12,}H..H",
            "zinc_finger_CCCC": "C..C.{12,}C..C",
            "iron_sulfur_4Fe4S": "C..C..C.{5,}C",
            "copper_type1": "H.{5,}C.{3,}H.{3,}M",
            "calcium_ef_hand": "D.{3}[DNS].{3}[DNS]",
            "heme_binding": "C..CH",
            "iron_2his_cluster": "H.{3,7}H",
            "calcium_dx_motif": "[DN].[DN].{2}[DENQ]",
            "copper_met_rich": "M.{0,3}[HM].{0,3}M",
        }

        # Initialize analyzers with config
        self.conservation_analyzer = ConservationAnalyzer()
        self.coevolution_analyzer = CoevolutionAnalyzer()

    def analyze(
        self,
        protein_data: ProteinData,
        msa_file: Optional[Path] = None,
    ) -> EvolutionaryFeatures:
        """Perform comprehensive evolutionary analysis.

        Args:
        ----
            protein_data: Protein information
            msa_file: Optional pre-computed MSA file

        Returns:
        -------
            EvolutionaryFeatures with conservation, coevolution, and functional sites

        Raises:
        ------
            EvolutionaryAnalysisError: If analysis fails
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
            conservation_scores = (
                self.conservation_analyzer.calculate_conservation_scores(
                    msa, self._calculate_sequence_weights(msa), None
                )
            )

            # Calculate coevolution matrix
            coevolution_matrix = self.coevolution_analyzer.calculate_coevolution_matrix(
                msa, self._calculate_sequence_weights(msa)
            )

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
            ) from e

    def _validate_msa(self, msa: MultipleSeqAlignment):
        """Validate MSA meets requirements."""
        n_sequences = len(msa)
        n_positions = msa.get_alignment_length()

        if n_sequences < self.min_sequences:
            raise ValidationError(
                f"MSA has too few sequences: {n_sequences} "
                f"(minimum: {self.min_sequences})"
            )

        # Calculate effective sequences
        n_effective = self._calculate_effective_sequences(msa)
        if n_effective < self.min_effective_sequences:
            raise ValidationError(
                f"MSA has too few effective sequences: {n_effective} "
                f"(minimum: {self.min_effective_sequences})"
            )

        # Check gap content
        gap_fractions = []
        for pos in range(n_positions):
            gaps = sum(1 for record in msa if record.seq[pos] == "-")
            gap_fractions.append(gaps / n_sequences)

        max_gap_fraction = max(gap_fractions)
        if max_gap_fraction > self.max_gap_fraction:
            raise ValidationError(
                f"MSA has positions with too many gaps: {max_gap_fraction:.2%} "
                f"(maximum: {self.max_gap_fraction:.2%})"
            )

        # Check coverage
        coverage = 1 - (sum(gap_fractions) / n_positions)
        if coverage < self.min_coverage:
            raise ValidationError(
                f"MSA has insufficient coverage: {coverage:.2%} "
                f"(minimum: {self.min_coverage:.2%})"
            )

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
        """Check if position shows metal-binding amino acid enrichment."""
        metal_binding_aas = set(config.get_property_group("metal_binding"))
        metal_binding_freq = sum(
            freq for aa, freq in aa_freqs.items() if aa in metal_binding_aas
        )
        return (
            metal_binding_freq
            > config.metal_binding["conservation_thresholds"]["binding_site"]
        )

    def _calculate_coordination_score(
        self,
        metal_site: MetalSite,
        conservation_scores: np.ndarray,
        coevolution_matrix: np.ndarray,
    ) -> float:
        """Calculate coordination geometry consistency score."""
        coordinating_positions = [
            r.position - 1 for r in metal_site.get_coordinating_residues()
        ]

        # Get boost factor for metal type
        boost_factor = config.get_metal_boost_factor(metal_site.metal_type.value)

        # Calculate average conservation of coordinating residues
        cons_score = np.mean(conservation_scores[coordinating_positions])

        # Calculate average coevolution between coordinating residues
        coevo_scores = []
        for i, pos1 in enumerate(coordinating_positions):
            for pos2 in coordinating_positions[i + 1 :]:
                coevo_scores.append(coevolution_matrix[pos1, pos2])
        coevo_score = np.mean(coevo_scores) if coevo_scores else 0.0

        # Combine scores with metal-specific boost
        return boost_factor * (0.7 * cons_score + 0.3 * coevo_score)

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

            try:
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
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                except subprocess.CalledProcessError as e:
                    raise EvolutionaryAnalysisError(
                        "msa_generation",
                        f"HHblits failed for {protein_data.protein_id}: {e.stderr}",
                    )
                except FileNotFoundError:
                    raise EvolutionaryAnalysisError(
                        "msa_generation",
                        "HHblits executable not found. Please ensure HHblits is installed and in PATH.",
                    )

                # Parse A3M format
                try:
                    return self._parse_a3m(msa_file)
                except (ValueError, IOError) as e:
                    raise EvolutionaryAnalysisError(
                        "msa_parsing",
                        f"Failed to parse HHblits output for {protein_data.protein_id}: {str(e)}",
                    )

            except (IOError, OSError) as e:
                raise DataAccessError(
                    f"Failed to access temporary files for {protein_data.protein_id}: {str(e)}"
                )

    def _run_blast_fallback(self, protein_data: ProteinData) -> MultipleSeqAlignment:
        """Generate MSA using BLAST search against nr database."""
        # Cache directory for BLAST results
        cache_dir = Path(tempfile.gettempdir()) / "midp_blast_cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise DataAccessError(f"Failed to create BLAST cache directory: {str(e)}")

        @lru_cache(maxsize=1000)
        def run_blast_search(sequence: str) -> list[SeqRecord]:
            """Run BLAST search with caching."""
            sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
            cache_file = cache_dir / f"blast_{sequence_hash}.pkl"

            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
                except (IOError, pickle.UnpicklingError) as e:
                    logger.warning(f"Failed to load BLAST cache: {e}")
                    # Continue to run new BLAST search

            try:
                # Run BLAST search
                logger.info("Running BLAST search against nr database")
                result_handle = NCBIWWW.qblast(
                    program="blastp",
                    database="nr",
                    sequence=sequence,
                    expect=1e-3,
                    hitlist_size=500,
                    gapcosts="11 1",
                    matrix_name="BLOSUM62",
                )

                # Parse results
                try:
                    blast_records = NCBIXML.parse(result_handle)
                    record = next(blast_records)
                except (ValueError, StopIteration) as e:
                    raise EvolutionaryAnalysisError(
                        "blast_parsing",
                        f"Failed to parse BLAST results for {protein_data.protein_id}: {str(e)}",
                    )

                # Filter hits and extract sequences
                sequences = []
                query_length = len(sequence)

                for alignment in record.alignments:
                    for hsp in alignment.hsps:
                        coverage = (hsp.align_length / query_length) * 100
                        if hsp.expect < 1e-3 and coverage > 70:
                            hit_seq = SeqRecord(
                                Seq(hsp.sbjct),
                                id=alignment.accession,
                                description=alignment.title[:50],
                            )
                            sequences.append(hit_seq)

                # Cache results
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump(sequences, f)
                except (IOError, pickle.PicklingError) as e:
                    logger.warning(f"Failed to cache BLAST results: {e}")

                return sequences

            except Exception as e:
                if "Resource temporarily unavailable" in str(e):
                    raise EvolutionaryAnalysisError(
                        "blast_search",
                        f"BLAST server temporarily unavailable for {protein_data.protein_id}. Please retry later.",
                    )
                else:
                    raise EvolutionaryAnalysisError(
                        "blast_search",
                        f"BLAST search failed for {protein_data.protein_id}: {str(e)}",
                    )

        # Try BLAST search with retries
        max_retries = 3
        retry_delay = 60  # seconds
        sequences = []

        for attempt in range(max_retries):
            try:
                sequences = run_blast_search(protein_data.sequence)
                break
            except EvolutionaryAnalysisError as e:
                if "temporarily unavailable" in str(e) and attempt < max_retries - 1:
                    logger.warning(
                        f"BLAST attempt {attempt + 1} failed, retrying in {retry_delay}s"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

        if not sequences:
            logger.warning(
                f"No BLAST hits found for {protein_data.protein_id}, using single sequence"
            )
            return MultipleSeqAlignment([
                SeqRecord(
                    Seq(protein_data.sequence),
                    id=protein_data.protein_id,
                    description="",
                )
            ])

        # Add query sequence at the start
        sequences.insert(
            0,
            SeqRecord(
                Seq(protein_data.sequence),
                id=protein_data.protein_id,
                description="query",
            ),
        )

        # Align sequences
        try:
            aligner = PairwiseAligner()
            aligner.mode = "global"
            aligner.open_gap_score = -10
            aligner.extend_gap_score = -0.5
            aligner.target_end_gap_score = 0.0
            aligner.query_end_gap_score = 0.0
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

            aligned_records = []
            query_seq = str(sequences[0].seq)

            for record in sequences:
                if record.id == protein_data.protein_id:
                    aligned_records.append(record)
                    continue

                try:
                    alignments = aligner.align(query_seq, str(record.seq))
                    if alignments:
                        best = alignments[0]
                        aligned_seq = ""
                        target_idx = 0

                        for q, t in zip(best.path[0], best.path[1]):
                            if q == -1:
                                aligned_seq += "-"
                            else:
                                aligned_seq += str(record.seq)[target_idx]
                                target_idx += 1

                        aligned_records.append(
                            SeqRecord(
                                Seq(aligned_seq),
                                id=record.id,
                                description=record.description,
                            ),
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to align sequence {record.id} for {protein_data.protein_id}: {e}"
                    )
                    continue

            if not aligned_records:
                raise EvolutionaryAnalysisError(
                    "sequence_alignment",
                    f"Failed to align any sequences for {protein_data.protein_id}",
                )

            msa = MultipleSeqAlignment(aligned_records)
            logger.info(f"Generated MSA with {len(msa)} sequences using BLAST")
            return msa

        except Exception as e:
            raise EvolutionaryAnalysisError(
                "sequence_alignment",
                f"Sequence alignment failed for {protein_data.protein_id}: {str(e)}",
            )

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

        # Track metal-specific scores
        metal_scores = {metal_type: 0.0 for metal_type in MetalType}
        total_motifs = 0
        coordination_scores = []

        for motif_name, pattern in self.metal_binding_patterns.items():
            matches = list(re.finditer(pattern, sequence))
            for match in matches:
                start, end = match.span()
                # Check conservation in motif region
                motif_conservation = np.mean(features.conservation_scores[start:end])

                if motif_conservation > 0.7:
                    logger.info(f"Found conserved {motif_name} at {start + 1}-{end}")

                    # Add to functional sites if not already present
                    for pos in range(start + 1, end + 1):
                        if not any(s[0] == pos for s in features.functional_sites):
                            features.functional_sites.append((pos, motif_conservation))

                    # Update metal-specific scores based on motif type
                    if "zinc" in motif_name:
                        metal_scores[MetalType.ZN2] += motif_conservation
                    elif "iron" in motif_name:
                        metal_scores[MetalType.FE2] += motif_conservation
                        metal_scores[MetalType.FE3] += motif_conservation * 0.8
                    elif "copper" in motif_name:
                        metal_scores[MetalType.CU2] += motif_conservation
                        metal_scores[MetalType.CU1] += motif_conservation * 0.8
                    elif "calcium" in motif_name:
                        metal_scores[MetalType.CA2] += motif_conservation
                    total_motifs += 1

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

                # Calculate coordination-aware score
                coord_score = self._calculate_coordination_score(
                    metal_site,
                    features.conservation_scores,
                    features.coevolution_matrix,
                )
                coordination_scores.append(coord_score)

                # Update metal-specific score
                metal_scores[metal_site.metal_type] += coord_score

                logger.info(
                    f"Metal site at {metal_site.center} has coevolution network "
                    f"of {len(metal_network)} residues (coord score: {coord_score:.3f})"
                )

        # Normalize metal-specific scores
        if total_motifs > 0:
            for metal_type in metal_scores:
                metal_scores[metal_type] /= total_motifs

        # Calculate overall coordination consistency
        coordination_consistency = (
            np.mean(coordination_scores) if coordination_scores else None
        )

        # Update features
        features.metal_specific_scores = metal_scores
        features.coordination_consistency = coordination_consistency

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
