"""Unit tests for evolutionary analysis module.

Tests edge cases and error handling for conservation and coevolution analysis.
"""

import logging
from typing import List, Optional

import numpy as np
import pytest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from src.midp.core.data_structures import (
    MetalSite,
    MetalType,
    ProteinData,
    ResidueType,
    Residue,
    MetalLigand,
    CoordinationGeometry,
)
from src.midp.interpretable.evolutionary.evolutionary_features import (
    CoevolutionAnalyzer,
    ConservationAnalyzer,
    MetalloproteinEvolutionaryAnalyzer,
)

# Set up logging
logger = logging.getLogger(__name__)


@pytest.fixture()
def mock_protein_data():
    """Create mock protein data for testing."""
    sequence = "MKCPFCGHKLAMQRLMDAHQGK"

    # Create residues for metal binding
    cys2 = Residue(
        residue_type=ResidueType.CYS,
        position=2,
        chain_id="A",
        coordinates=(1.0, 1.0, 1.0),
    )
    cys5 = Residue(
        residue_type=ResidueType.CYS,
        position=5,
        chain_id="A",
        coordinates=(2.0, 2.0, 2.0),
    )

    # Create metal ligands
    ligand1 = MetalLigand(
        residue=cys2,
        atom_name="SG",
        coordinates=(1.0, 1.0, 1.0),
        bond_length=2.3,
    )
    ligand2 = MetalLigand(
        residue=cys5,
        atom_name="SG",
        coordinates=(2.0, 2.0, 2.0),
        bond_length=2.3,
    )

    metal_sites = [
        MetalSite(
            metal_type=MetalType.ZN2,
            center=(0.0, 0.0, 0.0),
            ligands=[ligand1, ligand2],
            geometry=CoordinationGeometry.TETRAHEDRAL,
            geometry_rmsd=0.1,
        )
    ]

    return ProteinData(
        protein_id="TEST1",
        sequence=sequence,
        metal_sites=metal_sites,
        chain_ids=["A"],
    )


@pytest.fixture
def create_msa():
    """Factory fixture to create MSAs with different properties."""

    def _create_msa(
        sequences: List[str], ids: Optional[List[str]] = None
    ) -> MultipleSeqAlignment:
        if ids is None:
            ids = [f"seq_{i}" for i in range(len(sequences))]
        records = [
            SeqRecord(Seq(seq), id=id_, description="")
            for seq, id_ in zip(sequences, ids)
        ]
        return MultipleSeqAlignment(records)

    return _create_msa


class TestEvolutionaryAnalyzer:
    """Test MetalloproteinEvolutionaryAnalyzer edge cases."""

    def test_empty_msa(self, mock_protein_data, create_msa, caplog):
        """Test handling of empty MSA."""
        analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)
        empty_msa = create_msa([])

        with pytest.raises(ValueError) as exc_info:
            analyzer._validate_msa(empty_msa)
        assert "MSA must have at least" in str(exc_info.value)
        assert "too few sequences" in caplog.text

    def test_single_sequence_msa(self, mock_protein_data, create_msa, caplog):
        """Test handling of MSA with single sequence."""
        analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)
        single_msa = create_msa([mock_protein_data.sequence])

        with pytest.raises(ValueError) as exc_info:
            analyzer._validate_msa(single_msa)
        assert "MSA must have at least" in str(exc_info.value)
        assert "too few sequences" in caplog.text

    def test_no_homologs(self, mock_protein_data, caplog):
        """Test handling of protein with no homologs."""
        analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)

        # Mock BLAST search to return no hits
        def mock_blast(*args, **kwargs):
            return create_msa([mock_protein_data.sequence])

        analyzer._run_blast_fallback = mock_blast

        with caplog.at_level(logging.WARNING):
            features = analyzer.analyze(mock_protein_data)

        assert "No BLAST hits found" in caplog.text
        assert features.conservation_scores is not None
        assert features.coevolution_matrix is None
        assert len(features.functional_sites) == 0

    def test_terminal_metal_sites(self, create_msa):
        """Test handling of metal sites at sequence termini."""
        sequence = "CXXXC"  # 5 residues with metal sites at ends

        # Create residues for metal binding
        cys1 = Residue(
            residue_type=ResidueType.CYS,
            position=1,
            chain_id="A",
            coordinates=(1.0, 1.0, 1.0),
        )
        cys5 = Residue(
            residue_type=ResidueType.CYS,
            position=5,
            chain_id="A",
            coordinates=(2.0, 2.0, 2.0),
        )

        # Create metal ligands
        ligand1 = MetalLigand(
            residue=cys1,
            atom_name="SG",
            coordinates=(1.0, 1.0, 1.0),
            bond_length=2.3,
        )
        ligand2 = MetalLigand(
            residue=cys5,
            atom_name="SG",
            coordinates=(2.0, 2.0, 2.0),
            bond_length=2.3,
        )

        metal_sites = [
            MetalSite(
                metal_type=MetalType.ZN2,
                center=(0.0, 0.0, 0.0),
                ligands=[ligand1, ligand2],
                geometry=CoordinationGeometry.TETRAHEDRAL,
                geometry_rmsd=0.1,
            )
        ]

        protein_data = ProteinData(
            protein_id="TEST2",
            sequence=sequence,
            metal_sites=metal_sites,
            chain_ids=["A"],
        )

        # Create MSA with terminal metal sites
        msa = create_msa([
            "CXXXC",
            "CXXXC",
            "CXXXC",
        ])

        analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)
        features = analyzer.analyze(protein_data, msa_file=None)

        assert features.conservation_scores is not None
        assert len(features.conservation_scores) == 5
        assert features.conservation_scores[0] > 0.5  # Terminal C should be conserved
        assert features.conservation_scores[-1] > 0.5  # Terminal C should be conserved

    def test_high_gap_msa(self, mock_protein_data, create_msa):
        """Test handling of MSA with >90% gaps."""
        sequences = [
            "M----G-K--M--L--A--K",  # >90% gaps
            "-K---G-K--M--L--A--K",
            "M----G----M--L--A---",
        ]
        high_gap_msa = create_msa(sequences)

        analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)

        with pytest.raises(ValueError) as exc_info:
            analyzer._validate_msa(high_gap_msa)
        assert "too many gaps" in str(exc_info.value)

    def test_short_sequence(self, create_msa):
        """Test handling of extremely short sequences (<20 AA)."""
        sequence = "MCGK"  # 4 residues
        protein_data = ProteinData(
            protein_id="TEST3",
            sequence=sequence,
            metal_sites=[],
            chain_ids=["A"],
        )

        msa = create_msa([
            "MCGK",
            "MCGK",
            "MCGK",
        ])

        analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)
        features = analyzer.analyze(protein_data, msa_file=None)

        assert features.conservation_scores is not None
        assert len(features.conservation_scores) == 4
        assert features.coevolution_matrix is None  # Too short for coevolution

    def test_invalid_metal_positions(self, create_msa):
        """Test handling of invalid metal positions (out of bounds)."""
        sequence = "MCGK"

        # Create residues for metal binding
        cys1 = Residue(
            residue_type=ResidueType.CYS,
            position=1,
            chain_id="A",
            coordinates=(1.0, 1.0, 1.0),
        )
        cys10 = Residue(  # Invalid position
            residue_type=ResidueType.CYS,
            position=10,
            chain_id="A",
            coordinates=(2.0, 2.0, 2.0),
        )

        # Create metal ligands
        ligand1 = MetalLigand(
            residue=cys1,
            atom_name="SG",
            coordinates=(1.0, 1.0, 1.0),
            bond_length=2.3,
        )
        ligand2 = MetalLigand(
            residue=cys10,
            atom_name="SG",
            coordinates=(2.0, 2.0, 2.0),
            bond_length=2.3,
        )

        metal_sites = [
            MetalSite(
                metal_type=MetalType.ZN2,
                center=(0.0, 0.0, 0.0),
                ligands=[ligand1, ligand2],
                geometry=CoordinationGeometry.TETRAHEDRAL,
                geometry_rmsd=0.1,
            )
        ]

        protein_data = ProteinData(
            protein_id="TEST4",
            sequence=sequence,
            metal_sites=metal_sites,
            chain_ids=["A"],
        )

        msa = create_msa([sequence] * 3)
        analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)

        with pytest.warns(UserWarning):
            features = analyzer.analyze(protein_data, msa_file=None)

        assert features.conservation_scores is not None
        assert len(features.metal_specific_scores) == 0  # Should skip invalid sites


class TestConservationAnalyzer:
    """Test ConservationAnalyzer edge cases."""

    def test_identical_sequences(self, create_msa):
        """Test conservation analysis with all identical sequences."""
        sequence = "MKCPFCGHK"
        # Create more sequences to meet minimum requirement
        msa = create_msa([sequence] * 10)  # 10 identical sequences

        analyzer = ConservationAnalyzer()
        scores = analyzer.calculate_conservation_scores(msa)

        assert np.allclose(scores, 1.0)  # All positions should be perfectly conserved

    def test_all_gaps_position(self, create_msa):
        """Test handling of positions with all gaps."""
        sequences = [
            "M-CPF",
            "M-CPF",
            "M-CPF",
        ] * 4  # Multiply to meet minimum requirement
        msa = create_msa(sequences)

        analyzer = ConservationAnalyzer()
        scores = analyzer.calculate_conservation_scores(msa)

        assert scores[1] == 0.0  # Gap position should have zero conservation


class TestCoevolutionAnalyzer:
    """Test CoevolutionAnalyzer edge cases."""

    def test_no_significant_pairs(self, create_msa):
        """Test coevolution analysis with no significant pairs."""
        # Create MSA where positions vary independently
        sequences = [
            "ABCDE",
            "ABCDE",
            "VWXYZ",
            "VWXYZ",
        ] * 3  # Multiply to meet minimum requirement
        msa = create_msa(sequences)

        analyzer = CoevolutionAnalyzer()
        matrix = analyzer.calculate_coevolution_matrix(msa)

        # No position pairs should show strong coevolution
        assert np.all(matrix <= 0.5)

        # Network analysis should find no significant networks
        networks = analyzer.identify_coevolution_networks(matrix, threshold=0.5)
        assert len(networks) == 0

    def test_insufficient_sequences(self, create_msa):
        """Test handling of MSA with too few sequences for coevolution."""
        sequences = ["ABCDE", "VWXYZ"]  # Only 2 sequences
        msa = create_msa(sequences)

        analyzer = CoevolutionAnalyzer()
        with pytest.warns(UserWarning):
            matrix = analyzer.calculate_coevolution_matrix(msa)

        assert matrix is None or np.all(matrix == 0)

    def test_perfect_coevolution(self, create_msa):
        """Test detection of perfectly coevolving positions."""
        sequences = [
            "ABCDE",
            "ABCDE",
            "VWXYZ",
            "VWXYZ",
            "ABCDE",
        ] * 2  # Multiply to meet minimum requirement
        msa = create_msa(sequences)

        analyzer = CoevolutionAnalyzer()
        matrix = analyzer.calculate_coevolution_matrix(msa)

        # First two positions should perfectly covary
        assert matrix[0, 1] > 0.8
        assert matrix[1, 0] > 0.8


def test_end_to_end(mock_protein_data, create_msa):
    """End-to-end test of evolutionary analysis pipeline."""
    # Create realistic MSA with enough sequences
    sequences = [
        "MKCPFCGHKLAMQRLMDAHQGK",  # Original
        "MKCPYCGHKLAMQRLMDAHQGK",  # Conservative mutation
        "MKCPFCGHRLAMQKLMDAHQGK",  # Nearby mutation
        "MKCPFCDHKLAMQRLMDAHQGK",  # Metal-binding mutation
        "MKCPFCGHKLAMQRLMDAHQGK",  # Identical
    ] * 2  # Multiply to meet minimum requirement
    msa = create_msa(sequences)

    analyzer = MetalloproteinEvolutionaryAnalyzer(use_hhblits=False)
    features = analyzer.analyze(mock_protein_data, msa_file=None)

    # Verify all components are present
    assert features.conservation_scores is not None
    assert features.coevolution_matrix is not None
    assert len(features.functional_sites) > 0
    assert features.phylogenetic_diversity is not None
    assert len(features.metal_specific_scores) > 0

    # Verify metal binding positions are identified
    metal_positions = {2, 5}  # From mock_protein_data
    functional_positions = {pos for pos, _ in features.functional_sites}
    assert metal_positions.intersection(functional_positions)

    # Verify coevolution between metal binding residues
    assert features.coevolution_matrix[1, 4] > 0.5  # Positions 2 and 5 (0-based)
