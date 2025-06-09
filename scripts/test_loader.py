#!/usr/bin/env python3
"""
Test script for the protein loader.

This script demonstrates how to load proteins from both local files
and the RCSB PDB database.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.midp.data.loaders import load_protein


def main():
    """Test the protein loader functionality."""
    
    print("=" * 60)
    print("MIDP Protein Loader Test")
    print("=" * 60)
    
    # Test 1: Load a small metalloprotein from RCSB PDB
    print("\nTest 1: Loading Crambin (1CRN) from RCSB PDB...")
    print("-" * 40)
    
    try:
        # Crambin is a small plant protein with no metals, good for testing
        protein1 = load_protein("1CRN")
        
        print(f"✓ Successfully loaded {protein1.protein_id}")
        print(f"  - Sequence length: {len(protein1.sequence)}")
        print(f"  - Number of chains: {len(protein1.chain_ids)}")
        print(f"  - Number of residues: {len(protein1.residues)}")
        print(f"  - Metal sites detected: {len(protein1.metal_sites)}")
        print(f"  - Disorder regions: {len(protein1.disorder_regions)}")
        print(f"  - Global disorder content: {protein1.global_disorder_content:.2%}")
        print(f"  - Resolution: {protein1.resolution} Å")
        print(f"  - Method: {protein1.experimental_method}")
        
        # Show first 20 amino acids
        print(f"  - First 20 AA: {protein1.sequence[:20]}...")
        
    except Exception as e:
        print(f"✗ Failed to load 1CRN: {e}")
    
    # Test 2: Load a zinc finger protein
    print("\n\nTest 2: Loading a Zinc Finger Protein (1ZAA) from RCSB PDB...")
    print("-" * 40)
    
    try:
        # 1ZAA is a zinc finger protein
        protein2 = load_protein("1ZAA")
        
        print(f"✓ Successfully loaded {protein2.protein_id}")
        print(f"  - Sequence length: {len(protein2.sequence)}")
        print(f"  - Number of chains: {len(protein2.chain_ids)}")
        print(f"  - Metal sites detected: {len(protein2.metal_sites)}")
        
        # Show metal site details
        if protein2.metal_sites:
            print("\n  Metal Site Details:")
            for i, site in enumerate(protein2.metal_sites):
                print(f"    Site {i+1}:")
                print(f"      - Metal type: {site.metal_type.value}")
                print(f"      - Coordination number: {site.coordination_number}")
                print(f"      - Geometry: {site.geometry.value}")
                print(f"      - Ligands: {site.ligand_composition}")
                print(f"      - Position: ({site.center.x:.1f}, {site.center.y:.1f}, {site.center.z:.1f})")
        
    except Exception as e:
        print(f"✗ Failed to load 1ZAA: {e}")
    
    # Test 3: Load a larger metalloprotein
    print("\n\nTest 3: Loading Carbonic Anhydrase (1CA2) from RCSB PDB...")
    print("-" * 40)
    
    try:
        # 1CA2 is human carbonic anhydrase II with zinc
        protein3 = load_protein("1CA2")
        
        print(f"✓ Successfully loaded {protein3.protein_id}")
        print(f"  - Sequence length: {len(protein3.sequence)}")
        print(f"  - Metal sites detected: {len(protein3.metal_sites)}")
        
        # Show disorder analysis
        if protein3.disorder_regions:
            print(f"\n  Disorder Analysis:")
            print(f"    - Number of disordered regions: {len(protein3.disorder_regions)}")
            for i, region in enumerate(protein3.disorder_regions[:3]):  # Show first 3
                print(f"    - Region {i+1}: residues {region.start}-{region.end} "
                      f"(length: {region.length}, probability: {region.disorder_probability:.2f})")
        
    except Exception as e:
        print(f"✗ Failed to load 1CA2: {e}")
    
    # Test 4: Test validation with invalid input
    print("\n\nTest 4: Testing validation with invalid PDB ID...")
    print("-" * 40)
    
    try:
        protein4 = load_protein("XXXX")
        print("✗ Should have failed but didn't!")
    except Exception as e:
        print(f"✓ Correctly failed with: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main() 