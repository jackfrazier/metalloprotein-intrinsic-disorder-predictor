#!/usr/bin/env python3
"""
Test script for evolutionary analysis module.

This script demonstrates how to use the evolutionary analysis components
to identify functionally important residues in metalloproteins.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import logging

import numpy as np

from src.midp.data.loaders import load_protein
from src.midp.interpretable.evolutionary import MetalloproteinEvolutionaryAnalyzer

# Set up logging with debug level
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_single_protein(pdb_id: str, description: str, analyzer) -> None:
    """Analyze a single protein and display its evolutionary features."""
    print(f"\n\nAnalyzing {pdb_id} - {description}")
    print("-" * 40)

    try:
        logger.info(f"Loading protein {pdb_id}")
        protein_data = load_protein(pdb_id)

        if protein_data is None:
            logger.error(f"Failed to load protein {pdb_id}")
            return

        logger.info(
            f"Successfully loaded protein with {len(protein_data.sequence)} residues"
        )
        print(f"✓ Loaded protein with {len(protein_data.sequence)} residues")
        print(f"  Metal sites: {len(protein_data.metal_sites)}")

        evolutionary_features = analyzer.analyze(protein_data)
        display_evolutionary_results(protein_data, evolutionary_features)
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        logger.error(f"Error analyzing {pdb_id}", exc_info=True)


def display_evolutionary_results(protein_data, evolutionary_features) -> None:
    """Display the evolutionary analysis results."""
    print("\nEvolutionary Analysis Results:")
    print(
        f"  - Functional sites identified: {len(evolutionary_features.functional_sites)}"
    )

    display_functional_sites(protein_data, evolutionary_features)
    display_conservation_stats(evolutionary_features)
    display_metal_site_conservation(protein_data, evolutionary_features)
    display_coevolution_results(protein_data, evolutionary_features)

    if evolutionary_features.phylogenetic_diversity:
        print(
            f"\n  Phylogenetic diversity: {evolutionary_features.phylogenetic_diversity:.3f}"
        )


def run_mock_msa_analysis() -> None:
    """Run analysis on mock MSA data for demonstration."""
    print("\n\nAdvanced Analysis Demo:")
    print("-" * 40)

    mock_msa = create_mock_msa()
    analyze_mock_msa(mock_msa)


def main():
    """Test evolutionary analysis on a metalloprotein."""
    print("=" * 60)
    print("MIDP Evolutionary Analysis Test")
    print("=" * 60)

    # Let's just test with one protein first to diagnose the issue
    test_proteins = [
        ("1ZAA", "Zinc Finger Protein"),
    ]

    analyzer = MetalloproteinEvolutionaryAnalyzer(
        use_hhblits=False,
        n_iterations=3,
        e_value_threshold=1e-3,
    )

    for pdb_id, description in test_proteins:
        analyze_single_protein(pdb_id, description, analyzer)

    print("\n" + "=" * 60)
    print("Evolutionary analysis test completed!")
    print("=" * 60)

    run_mock_msa_analysis()


# Helper functions for display_evolutionary_results
def display_functional_sites(protein_data, evolutionary_features) -> None:
    if evolutionary_features.functional_sites:
        print("\n  Top functional sites:")
        for pos, score in evolutionary_features.functional_sites[:10]:
            residue = (
                protein_data.sequence[pos - 1]
                if pos <= len(protein_data.sequence)
                else "?"
            )
            print(f"    Position {pos} ({residue}): score = {score:.3f}")


def display_conservation_stats(evolutionary_features) -> None:
    conservation_scores = evolutionary_features.conservation_scores
    print(f"\n  Conservation statistics:")
    print(f"    - Mean conservation: {np.mean(conservation_scores):.3f}")
    print(f"    - Max conservation: {np.max(conservation_scores):.3f}")
    print(
        f"    - Highly conserved positions (>0.8): {np.sum(conservation_scores > 0.8)}"
    )


def display_metal_site_conservation(protein_data, evolutionary_features) -> None:
    if not protein_data.metal_sites:
        return

    conservation_scores = evolutionary_features.conservation_scores
    print("\n  Metal-binding site conservation:")
    for i, site in enumerate(protein_data.metal_sites):
        metal_positions = [r.position for r in site.get_coordinating_residues()]
        metal_conservation = [
            conservation_scores[p - 1]
            for p in metal_positions
            if p <= len(conservation_scores)
        ]

        if metal_conservation:
            print(f"    Metal site {i+1} ({site.metal_type.value}):")
            print(f"      - Coordinating residues: {metal_positions}")
            print(f"      - Mean conservation: {np.mean(metal_conservation):.3f}")


def display_coevolution_results(protein_data, evolutionary_features) -> None:
    if evolutionary_features.coevolution_matrix is None:
        return

    coevo_matrix = evolutionary_features.coevolution_matrix
    strong_pairs = find_strong_coevolving_pairs(protein_data, coevo_matrix)

    if strong_pairs:
        print(f"\n  Strongly coevolving pairs: {len(strong_pairs)}")
        strong_pairs.sort(key=lambda x: x[2], reverse=True)
        for pos1, pos2, score in strong_pairs[:5]:
            res1 = (
                protein_data.sequence[pos1 - 1]
                if pos1 <= len(protein_data.sequence)
                else "?"
            )
            res2 = (
                protein_data.sequence[pos2 - 1]
                if pos2 <= len(protein_data.sequence)
                else "?"
            )
            print(f"    {pos1}({res1}) - {pos2}({res2}): {score:.3f}")


def find_strong_coevolving_pairs(protein_data, coevo_matrix):
    n_positions = coevo_matrix.shape[0]
    strong_pairs = []
    for i in range(n_positions):
        for j in range(i + 1, n_positions):
            if coevo_matrix[i, j] > 0.8:
                strong_pairs.append((i + 1, j + 1, coevo_matrix[i, j]))
    return strong_pairs


def create_mock_msa():
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Align import MultipleSeqAlignment

    sequences = [
        "MKCPFCGHKLAMQRLMDAHQGK",  # Query with C-x-x-C motif
        "MKCPYCGHKLAMQRLMDAHQGK",  # Conserved motif
        "MKCPFCGHRLAMQKLMDAHQGK",  # Slight variation
        "MKCPFCDHKLAMQRLMDAHQGK",  # C->D mutation
        "MKCPFCGHKLAMQRLMDAHQGK",  # Identical to query
    ]

    records = [
        SeqRecord(Seq(seq), id=f"seq_{i}", description="")
        for i, seq in enumerate(sequences)
    ]
    return MultipleSeqAlignment(records)


def analyze_mock_msa(mock_msa):
    print("Analyzing mock MSA with metal-binding motif...")

    from src.midp.interpretable.evolutionary.conservation import ConservationAnalyzer
    from src.midp.interpretable.evolutionary.coevolution import CoevolutionAnalyzer

    cons_analyzer = ConservationAnalyzer()
    conservation = cons_analyzer.calculate_conservation_scores(mock_msa)
    motifs = cons_analyzer.identify_conserved_metal_motifs(mock_msa, conservation)

    print(f"\nConserved metal-binding motifs found: {len(motifs)}")
    for motif in motifs:
        print(
            f"  - {motif['name']} at positions {motif['start']}-{motif['end']}: "
            f"{motif['sequence']} (conservation: {motif['conservation']:.3f})"
        )

    coevo_analyzer = CoevolutionAnalyzer()
    coevo_matrix = coevo_analyzer.calculate_coevolution_matrix(mock_msa)
    networks = coevo_analyzer.identify_coevolution_networks(coevo_matrix)

    print(f"\nCoevolution networks identified: {len(networks)}")
    for i, network in enumerate(networks[:3]):
        positions = sorted(network)
        print(f"  Network {i+1}: positions {positions}")


if __name__ == "__main__":
    main()
