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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Test evolutionary analysis on a metalloprotein."""
    
    print("=" * 60)
    print("MIDP Evolutionary Analysis Test")
    print("=" * 60)
    
    # Test proteins
    test_proteins = [
        ("1ZAA", "Zinc Finger Protein"),  # Small zinc-binding protein
        ("1CA2", "Carbonic Anhydrase II"),  # Larger zinc enzyme
    ]
    
    # Initialize evolutionary analyzer
    analyzer = MetalloproteinEvolutionaryAnalyzer(
        use_hhblits=False,  # Use fallback for demo
        n_iterations=3,
        e_value_threshold=1e-3
    )
    
    for pdb_id, description in test_proteins:
        print(f"\n\nAnalyzing {pdb_id} - {description}")
        print("-" * 40)
        
        try:
            # Load protein
            logger.info(f"Loading protein {pdb_id}")
            protein_data = load_protein(pdb_id)
            
            print(f"✓ Loaded protein with {len(protein_data.sequence)} residues")
            print(f"  Metal sites: {len(protein_data.metal_sites)}")
            
            # Perform evolutionary analysis
            logger.info("Performing evolutionary analysis")
            
            # Note: This will use a minimal MSA since we're not using HHblits
            evolutionary_features = analyzer.analyze(protein_data)
            
            # Display results
            print("\nEvolutionary Analysis Results:")
            print(f"  - Functional sites identified: {len(evolutionary_features.functional_sites)}")
            
            # Show top 10 functional sites
            if evolutionary_features.functional_sites:
                print("\n  Top functional sites:")
                for pos, score in evolutionary_features.functional_sites[:10]:
                    residue = protein_data.sequence[pos-1] if pos <= len(protein_data.sequence) else '?'
                    print(f"    Position {pos} ({residue}): score = {score:.3f}")
            
            # Show conservation statistics
            conservation_scores = evolutionary_features.conservation_scores
            print(f"\n  Conservation statistics:")
            print(f"    - Mean conservation: {np.mean(conservation_scores):.3f}")
            print(f"    - Max conservation: {np.max(conservation_scores):.3f}")
            print(f"    - Highly conserved positions (>0.8): "
                  f"{np.sum(conservation_scores > 0.8)}")
            
            # Check metal-binding site conservation
            if protein_data.metal_sites:
                print("\n  Metal-binding site conservation:")
                for i, site in enumerate(protein_data.metal_sites):
                    metal_positions = [r.position for r in site.get_coordinating_residues()]
                    metal_conservation = [conservation_scores[p-1] for p in metal_positions 
                                        if p <= len(conservation_scores)]
                    
                    if metal_conservation:
                        print(f"    Metal site {i+1} ({site.metal_type.value}):")
                        print(f"      - Coordinating residues: {metal_positions}")
                        print(f"      - Mean conservation: {np.mean(metal_conservation):.3f}")
            
            # Analyze coevolution if available
            if evolutionary_features.coevolution_matrix is not None:
                coevo_matrix = evolutionary_features.coevolution_matrix
                
                # Find strongly coevolving pairs
                n_positions = coevo_matrix.shape[0]
                strong_pairs = []
                
                for i in range(n_positions):
                    for j in range(i+1, n_positions):
                        if coevo_matrix[i, j] > 0.8:  # Strong coevolution
                            strong_pairs.append((i+1, j+1, coevo_matrix[i, j]))
                
                if strong_pairs:
                    print(f"\n  Strongly coevolving pairs: {len(strong_pairs)}")
                    # Show top 5
                    strong_pairs.sort(key=lambda x: x[2], reverse=True)
                    for pos1, pos2, score in strong_pairs[:5]:
                        res1 = protein_data.sequence[pos1-1] if pos1 <= len(protein_data.sequence) else '?'
                        res2 = protein_data.sequence[pos2-1] if pos2 <= len(protein_data.sequence) else '?'
                        print(f"    {pos1}({res1}) - {pos2}({res2}): {score:.3f}")
            
            # Phylogenetic diversity
            if evolutionary_features.phylogenetic_diversity:
                print(f"\n  Phylogenetic diversity: {evolutionary_features.phylogenetic_diversity:.3f}")
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            logger.error(f"Error analyzing {pdb_id}", exc_info=True)
    
    print("\n" + "=" * 60)
    print("Evolutionary analysis test completed!")
    print("=" * 60)
    
    # Demonstrate advanced features
    print("\n\nAdvanced Analysis Demo:")
    print("-" * 40)
    
    # Create a mock MSA for demonstration
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Align import MultipleSeqAlignment
    
    # Create a simple MSA with metal-binding motif conservation
    sequences = [
        "MKCPFCGHKLAMQRLMDAHQGK",  # Query with C-x-x-C motif
        "MKCPYCGHKLAMQRLMDAHQGK",  # Conserved motif
        "MKCPFCGHRLAMQKLMDAHQGK",  # Slight variation
        "MKCPFCDHKLAMQRLMDAHQGK",  # C->D mutation
        "MKCPFCGHKLAMQRLMDAHQGK",  # Identical to query
    ]
    
    records = [SeqRecord(Seq(seq), id=f"seq_{i}", description="") 
               for i, seq in enumerate(sequences)]
    mock_msa = MultipleSeqAlignment(records)
    
    print("Analyzing mock MSA with metal-binding motif...")
    
    # Test conservation analyzer
    from src.midp.interpretable.evolutionary.conservation import ConservationAnalyzer
    cons_analyzer = ConservationAnalyzer()
    
    conservation = cons_analyzer.calculate_conservation_scores(mock_msa)
    motifs = cons_analyzer.identify_conserved_metal_motifs(mock_msa, conservation)
    
    print(f"\nConserved metal-binding motifs found: {len(motifs)}")
    for motif in motifs:
        print(f"  - {motif['name']} at positions {motif['start']}-{motif['end']}: "
              f"{motif['sequence']} (conservation: {motif['conservation']:.3f})")
    
    # Test coevolution analyzer
    from src.midp.interpretable.evolutionary.coevolution import CoevolutionAnalyzer
    coevo_analyzer = CoevolutionAnalyzer()
    
    coevo_matrix = coevo_analyzer.calculate_coevolution_matrix(mock_msa)
    networks = coevo_analyzer.identify_coevolution_networks(coevo_matrix)
    
    print(f"\nCoevolution networks identified: {len(networks)}")
    for i, network in enumerate(networks[:3]):  # Show first 3
        positions = sorted(network)
        print(f"  Network {i+1}: positions {positions}")


if __name__ == "__main__":
    main() 