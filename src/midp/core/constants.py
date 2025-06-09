"""
Scientific constants and configuration defaults for the MIDP system.

This module centralizes all constants to ensure consistency across
the interpretable and black box tracks.
"""

from typing import Dict, List, Tuple

# ============================================================================
# PHYSICAL AND CHEMICAL CONSTANTS
# ============================================================================

# Van der Waals radii in Angstroms (from Bondi, 1964)
VAN_DER_WAALS_RADII: Dict[str, float] = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'F': 1.47, 'P': 1.80, 'S': 1.80, 'CL': 1.75,
    'FE': 1.40, 'ZN': 1.39, 'CU': 1.40, 'CA': 1.97,
    'MG': 1.73, 'MN': 1.39, 'CO': 1.39, 'NI': 1.39
}

# Ionic radii for common oxidation states (Shannon, 1976)
IONIC_RADII: Dict[str, float] = {
    'FE2+': 0.78, 'FE3+': 0.65, 'ZN2+': 0.74,
    'CU+': 0.77, 'CU2+': 0.73, 'CA2+': 1.00,
    'MG2+': 0.72, 'MN2+': 0.83, 'CO2+': 0.74,
    'NI2+': 0.69
}

# Standard amino acid properties
AMINO_ACID_PROPERTIES: Dict[str, Dict[str, float]] = {
    # Format: {hydrophobicity, pKa_side_chain, molecular_weight}
    'A': {'hydrophobicity': 1.8, 'pKa': None, 'mw': 89.1},
    'R': {'hydrophobicity': -4.5, 'pKa': 12.5, 'mw': 174.2},
    'N': {'hydrophobicity': -3.5, 'pKa': None, 'mw': 132.1},
    'D': {'hydrophobicity': -3.5, 'pKa': 3.9, 'mw': 133.1},
    'C': {'hydrophobicity': 2.5, 'pKa': 8.3, 'mw': 121.2},
    'Q': {'hydrophobicity': -3.5, 'pKa': None, 'mw': 146.2},
    'E': {'hydrophobicity': -3.5, 'pKa': 4.3, 'mw': 147.1},
    'G': {'hydrophobicity': -0.4, 'pKa': None, 'mw': 75.1},
    'H': {'hydrophobicity': -3.2, 'pKa': 6.0, 'mw': 155.2},
    'I': {'hydrophobicity': 4.5, 'pKa': None, 'mw': 131.2},
    'L': {'hydrophobicity': 3.8, 'pKa': None, 'mw': 131.2},
    'K': {'hydrophobicity': -3.9, 'pKa': 10.5, 'mw': 146.2},
    'M': {'hydrophobicity': 1.9, 'pKa': None, 'mw': 149.2},
    'F': {'hydrophobicity': 2.8, 'pKa': None, 'mw': 165.2},
    'P': {'hydrophobicity': -1.6, 'pKa': None, 'mw': 115.1},
    'S': {'hydrophobicity': -0.8, 'pKa': None, 'mw': 105.1},
    'T': {'hydrophobicity': -0.7, 'pKa': None, 'mw': 119.1},
    'W': {'hydrophobicity': -0.9, 'pKa': None, 'mw': 204.2},
    'Y': {'hydrophobicity': -1.3, 'pKa': 10.1, 'mw': 181.2},
    'V': {'hydrophobicity': 4.2, 'pKa': None, 'mw': 117.1}
}

# ============================================================================
# METAL COORDINATION PARAMETERS
# ============================================================================

# Ideal coordination geometries (bond angles in degrees)
COORDINATION_GEOMETRIES: Dict[str, Dict[str, any]] = {
    'linear': {
        'coordination_number': 2,
        'angles': [(180.0,)],
        'tolerance': 10.0
    },
    'trigonal_planar': {
        'coordination_number': 3,
        'angles': [(120.0, 120.0, 120.0)],
        'tolerance': 15.0
    },
    'tetrahedral': {
        'coordination_number': 4,
        'angles': [(109.5, 109.5, 109.5, 109.5, 109.5, 109.5)],
        'tolerance': 15.0
    },
    'square_planar': {
        'coordination_number': 4,
        'angles': [(90.0, 90.0, 90.0, 90.0, 180.0, 180.0)],
        'tolerance': 15.0
    },
    'trigonal_bipyramidal': {
        'coordination_number': 5,
        'angles': [(90.0, 90.0, 120.0, 120.0, 120.0, 180.0)],
        'tolerance': 15.0
    },
    'square_pyramidal': {
        'coordination_number': 5,
        'angles': [(90.0, 90.0, 90.0, 90.0, 180.0)],
        'tolerance': 15.0
    },
    'octahedral': {
        'coordination_number': 6,
        'angles': [(90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 180.0)],
        'tolerance': 15.0
    }
}

# Typical metal-ligand bond lengths (Angstroms)
METAL_LIGAND_DISTANCES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Format: {metal: {ligand_atom: (min_dist, max_dist)}}
    'FE': {
        'N': (1.9, 2.3), 'O': (1.8, 2.2), 'S': (2.2, 2.6)
    },
    'ZN': {
        'N': (1.9, 2.2), 'O': (1.9, 2.2), 'S': (2.2, 2.5)
    },
    'CU': {
        'N': (1.9, 2.3), 'O': (1.9, 2.3), 'S': (2.2, 2.6)
    },
    'CA': {
        'O': (2.2, 2.6), 'N': (2.3, 2.7)
    },
    'MG': {
        'O': (2.0, 2.3), 'N': (2.1, 2.4)
    }
}

# Metal binding preferences by residue type
METAL_BINDING_PREFERENCES: Dict[str, List[str]] = {
    'FE': ['C', 'H', 'M', 'E', 'D'],
    'ZN': ['C', 'H', 'E', 'D'],
    'CU': ['H', 'C', 'M'],
    'CA': ['D', 'E', 'S', 'T', 'N'],
    'MG': ['D', 'E', 'S', 'T']
}

# ============================================================================
# DISORDER PREDICTION PARAMETERS
# ============================================================================

# Disorder propensity by amino acid (normalized scale)
DISORDER_PROPENSITY: Dict[str, float] = {
    'A': 0.06, 'R': 0.18, 'N': 0.01, 'D': 0.19,
    'C': -0.20, 'Q': 0.32, 'E': 0.74, 'G': 0.17,
    'H': -0.04, 'I': -0.49, 'L': -0.34, 'K': 0.59,
    'M': -0.23, 'F': -0.44, 'P': 0.99, 'S': 0.34,
    'T': 0.12, 'W': -0.51, 'Y': -0.34, 'V': -0.48
}

# Disorder prediction thresholds
DISORDER_THRESHOLDS = {
    'confident_disorder': 0.5,
    'confident_order': 0.3,
    'twilight_zone': (0.3, 0.5)
}

# ============================================================================
# FRUSTRATION ANALYSIS PARAMETERS
# ============================================================================

# Contact distance cutoffs (Angstroms)
CONTACT_CUTOFFS = {
    'ca_ca': 8.0,  # CA-CA distance for contact
    'heavy_atom': 4.5,  # Heavy atom contact
    'vdw_clash': 0.8  # Fraction of VdW sum for clash
}

# Frustration thresholds
FRUSTRATION_THRESHOLDS = {
    'highly_frustrated': 0.78,  # > 78th percentile
    'neutral': (0.22, 0.78),  # 22nd-78th percentile
    'minimally_frustrated': 0.22  # < 22nd percentile
}

# Energy function parameters
ENERGY_PARAMETERS = {
    'temperature': 300.0,  # Kelvin
    'dielectric_constant': 80.0,  # Water
    'ionic_strength': 0.15  # Physiological
}

# ============================================================================
# EVOLUTIONARY ANALYSIS PARAMETERS
# ============================================================================

# Conservation score thresholds
CONSERVATION_THRESHOLDS = {
    'highly_conserved': 0.8,
    'moderately_conserved': 0.6,
    'variable': 0.4
}

# Coevolution parameters
COEVOLUTION_PARAMETERS = {
    'min_separation': 5,  # Minimum sequence separation
    'pseudocount': 0.5,  # For mutual information
    'apc_correction': True  # Average product correction
}

# MSA quality thresholds
MSA_QUALITY_THRESHOLDS = {
    'min_sequences': 100,
    'min_effective_sequences': 50,
    'max_gap_fraction': 0.5,
    'min_coverage': 0.7
}

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

# Graph neural network parameters
GNN_PARAMETERS = {
    'num_layers': 6,
    'hidden_dim': 256,
    'num_heads': 8,
    'dropout': 0.1,
    'layer_norm': True,
    'residual_connections': True
}

# Protein language model parameters
PLM_PARAMETERS = {
    'model_name': 'esm2_t33_650M_UR50D',
    'fine_tune_layers': 4,
    'learning_rate': 1e-5,
    'warmup_steps': 1000
}

# Training parameters
TRAINING_PARAMETERS = {
    'batch_size': 32,
    'max_sequence_length': 1024,
    'gradient_accumulation_steps': 4,
    'max_epochs': 100,
    'early_stopping_patience': 10,
    'mixed_precision': True
}

# ============================================================================
# API AND DEPLOYMENT PARAMETERS
# ============================================================================

# API rate limits
API_LIMITS = {
    'max_sequence_length': 2000,
    'max_requests_per_minute': 60,
    'max_batch_size': 10,
    'timeout_seconds': 300
}

# AWS resource configurations
AWS_RESOURCES = {
    's3_bucket': 'midp-models',
    'sagemaker_instance_type': 'ml.g4dn.xlarge',
    'lambda_memory': 3008,  # MB
    'lambda_timeout': 900  # seconds
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file'],
    'file_path': 'logs/midp.log'
}

# ============================================================================
# VALIDATION PARAMETERS
# ============================================================================

# Input validation limits
VALIDATION_LIMITS = {
    'min_sequence_length': 20,
    'max_sequence_length': 5000,
    'valid_amino_acids': set('ACDEFGHIKLMNPQRSTVWY'),
    'min_structure_resolution': 0.0,  # Angstroms
    'max_structure_resolution': 10.0,  # Angstroms
    'max_missing_residues_fraction': 0.1
}

# Output validation ranges
OUTPUT_RANGES = {
    'confidence_score': (0.0, 1.0),
    'functional_score': (0.0, 1.0),
    'disorder_score': (0.0, 1.0),
    'frustration_index': (-5.0, 5.0),
    'conservation_score': (0.0, 1.0)
}

# ============================================================================
# SCIENTIFIC REFERENCES
# ============================================================================

REFERENCES = {
    'frustration': 'Ferreiro et al. (2007) PNAS 104(50):19819-24',
    'disorder_propensity': 'Uversky et al. (2000) Proteins 41(3):415-27',
    'metal_coordination': 'Harding (2001) Acta Cryst. D57:401-411',
    'conservation': 'Capra & Singh (2007) Bioinformatics 23(15):1875-82',
    'coevolution': 'Dunn et al. (2008) Bioinformatics 24(3):333-40'
}
