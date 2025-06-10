"""
Core data structures for the Metalloprotein Intrinsically Disordered Predictor.

This module defines the fundamental data structures used throughout the system,
ensuring consistency between the interpretable and black box tracks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
from pathlib import Path
import numpy as np
import torch
from Bio.PDB import Structure


class MetalType(Enum):
    """Supported metal ion types with their common oxidation states."""

    FE2 = "Fe2+"  # Iron(II)
    FE3 = "Fe3+"  # Iron(III)
    ZN2 = "Zn2+"  # Zinc(II)
    CU1 = "Cu+"  # Copper(I)
    CU2 = "Cu2+"  # Copper(II)
    CA2 = "Ca2+"  # Calcium(II)
    MG2 = "Mg2+"  # Magnesium(II)
    MN2 = "Mn2+"  # Manganese(II)
    CO2 = "Co2+"  # Cobalt(II)
    NI2 = "Ni2+"  # Nickel(II)
    OTHER = "Other"


class CoordinationGeometry(Enum):
    """Common metal coordination geometries."""

    TETRAHEDRAL = "tetrahedral"
    SQUARE_PLANAR = "square_planar"
    OCTAHEDRAL = "octahedral"
    TRIGONAL_BIPYRAMIDAL = "trigonal_bipyramidal"
    SQUARE_PYRAMIDAL = "square_pyramidal"
    TRIGONAL_PLANAR = "trigonal_planar"
    LINEAR = "linear"
    UNKNOWN = "unknown"


class ResidueType(Enum):
    """Amino acid types with metal-binding propensity annotation."""

    # Metal-binding residues
    CYS = ("C", "Cysteine", True)
    HIS = ("H", "Histidine", True)
    MET = ("M", "Methionine", True)
    GLU = ("E", "Glutamate", True)
    ASP = ("D", "Aspartate", True)

    # Sometimes metal-binding
    TYR = ("Y", "Tyrosine", False)
    SER = ("S", "Serine", False)
    THR = ("T", "Threonine", False)
    ASN = ("N", "Asparagine", False)
    GLN = ("Q", "Glutamine", False)

    # Non-metal-binding
    ALA = ("A", "Alanine", False)
    ARG = ("R", "Arginine", False)
    GLY = ("G", "Glycine", False)
    ILE = ("I", "Isoleucine", False)
    LEU = ("L", "Leucine", False)
    LYS = ("K", "Lysine", False)
    PHE = ("F", "Phenylalanine", False)
    PRO = ("P", "Proline", False)
    TRP = ("W", "Tryptophan", False)
    VAL = ("V", "Valine", False)

    def __init__(self, one_letter: str, full_name: str, metal_binding: bool):
        self.one_letter = one_letter
        self.full_name = full_name
        self.metal_binding = metal_binding


@dataclass
class Coordinates3D:
    """3D coordinates with optional confidence."""

    x: float
    y: float
    z: float
    confidence: Optional[float] = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: "Coordinates3D") -> float:
        """Calculate Euclidean distance to another point."""
        return np.linalg.norm(self.to_array() - other.to_array())


@dataclass
class Residue:
    """Represents a single amino acid residue."""

    residue_type: ResidueType
    position: int  # 1-indexed position in sequence
    chain_id: str
    coordinates: Optional[Coordinates3D] = None  # CA atom position
    secondary_structure: Optional[str] = None  # H, E, C, etc.
    disorder_score: Optional[float] = None  # 0-1, higher = more disordered
    conservation_score: Optional[float] = None  # From MSA analysis
    frustration_index: Optional[float] = None  # Local frustration

    @property
    def is_metal_binding(self) -> bool:
        """Check if residue type commonly binds metals."""
        return self.residue_type.metal_binding

    @property
    def one_letter_code(self) -> str:
        """Get one-letter amino acid code."""
        return self.residue_type.one_letter


@dataclass
class MetalLigand:
    """Represents a ligand coordinating a metal ion."""

    residue: Residue
    atom_name: str  # e.g., "SG" for cysteine sulfur
    coordinates: Coordinates3D
    bond_length: float  # Distance to metal in Angstroms
    bond_order: Optional[float] = None  # Estimated bond order

    @property
    def ligand_type(self) -> str:
        """Get ligand atom type (e.g., 'S', 'N', 'O')."""
        # Map common atom names to element types
        atom_type_map = {
            "SG": "S",  # Cysteine sulfur
            "SD": "S",  # Methionine sulfur
            "ND1": "N",
            "ND2": "N",
            "NE2": "N",  # Histidine nitrogens
            "OE1": "O",
            "OE2": "O",  # Glutamate oxygens
            "OD1": "O",
            "OD2": "O",  # Aspartate oxygens
            "O": "O",  # Backbone oxygen
            "N": "N",  # Backbone nitrogen
        }
        return atom_type_map.get(self.atom_name, "X")


@dataclass
class MetalSite:
    """Represents a metal coordination site."""

    metal_type: MetalType
    center: Coordinates3D
    ligands: List[MetalLigand]
    geometry: CoordinationGeometry
    geometry_rmsd: float  # Deviation from ideal geometry
    occupancy: float = 1.0  # Fractional occupancy
    b_factor: Optional[float] = None  # Temperature factor

    @property
    def coordination_number(self) -> int:
        """Get the number of coordinating ligands."""
        return len(self.ligands)

    @property
    def ligand_composition(self) -> Dict[str, int]:
        """Get composition of ligand types."""
        composition = {}
        for ligand in self.ligands:
            ligand_type = ligand.ligand_type
            composition[ligand_type] = composition.get(ligand_type, 0) + 1
        return composition

    def get_coordinating_residues(self) -> List[Residue]:
        """Get list of residues coordinating the metal."""
        return [ligand.residue for ligand in self.ligands]


@dataclass
class DisorderRegion:
    """Represents an intrinsically disordered region."""

    start: int  # 1-indexed
    end: int  # 1-indexed, inclusive
    disorder_probability: float  # Average disorder score
    functional_score: Optional[float] = None  # Likelihood of functional importance
    metal_binding_potential: Optional[float] = None  # Potential for metal coordination

    @property
    def length(self) -> int:
        """Get the length of the disordered region."""
        return self.end - self.start + 1

    def contains_position(self, position: int) -> bool:
        """Check if a position falls within this region."""
        return self.start <= position <= self.end


@dataclass
class EvolutionaryFeatures:
    """Evolutionary analysis results."""

    conservation_scores: np.ndarray  # Per-residue conservation
    coevolution_matrix: Optional[np.ndarray] = None  # Pairwise coevolution
    functional_sites: List[Tuple[int, float]] = field(
        default_factory=list,
    )  # (position, score)
    phylogenetic_diversity: Optional[float] = None
    metal_specific_scores: Dict[MetalType, float] = field(
        default_factory=dict
    )  # Metal-type specific binding scores
    coordination_consistency: Optional[float] = (
        None  # Score for coordination geometry consistency
    )

    def get_highly_conserved_positions(self, threshold: float = 0.8) -> List[int]:
        """Get positions with high conservation scores."""
        return [
            i + 1
            for i, score in enumerate(self.conservation_scores)
            if score > threshold
        ]


@dataclass
class FrustrationFeatures:
    """Local frustration analysis results."""

    frustration_indices: np.ndarray  # Per-residue frustration
    contact_map: np.ndarray  # Binary contact matrix
    energy_matrix: np.ndarray  # Pairwise interaction energies
    frustrated_contacts: List[Tuple[int, int, float]] = field(
        default_factory=list
    )  # (i, j, frustration)

    def get_highly_frustrated_residues(self, threshold: float = 2.0) -> List[int]:
        """Get residues with high frustration."""
        return [
            i + 1 for i, fi in enumerate(self.frustration_indices) if fi > threshold
        ]


@dataclass
class ProteinData:
    """Core data structure containing all protein information."""

    # Basic information
    protein_id: str
    sequence: str
    chain_ids: List[str]

    # Structural data
    structure: Optional[Structure.Structure] = None  # BioPython structure
    residues: List[Residue] = field(default_factory=list)

    # Metal coordination
    metal_sites: List[MetalSite] = field(default_factory=list)

    # Disorder information
    disorder_regions: List[DisorderRegion] = field(default_factory=list)
    global_disorder_content: Optional[float] = None  # Fraction of disordered residues

    # Interpretable features
    evolutionary_features: Optional[EvolutionaryFeatures] = None
    frustration_features: Optional[FrustrationFeatures] = None

    # Additional metadata
    source_file: Optional[Path] = None
    experimental_method: Optional[str] = None  # X-ray, NMR, etc.
    resolution: Optional[float] = None  # For X-ray structures

    @property
    def length(self) -> int:
        """Get sequence length."""
        return len(self.sequence)

    @property
    def has_structure(self) -> bool:
        """Check if 3D structure is available."""
        return self.structure is not None

    @property
    def num_metal_sites(self) -> int:
        """Get number of metal binding sites."""
        return len(self.metal_sites)

    @property
    def is_valid(self) -> bool:
        """Validate protein data integrity."""
        return len(self.sequence) > 0 and all(
            aa in "ACDEFGHIKLMNPQRSTVWY" for aa in self.sequence
        )

    def get_metal_binding_residues(self) -> List[Residue]:
        """Get all residues involved in metal coordination."""
        binding_residues = []
        for site in self.metal_sites:
            binding_residues.extend(site.get_coordinating_residues())
        return list(set(binding_residues))  # Remove duplicates

    def get_disorder_score_at_position(self, position: int) -> Optional[float]:
        """Get disorder score for a specific position."""
        for residue in self.residues:
            if residue.position == position:
                return residue.disorder_score
        return None


@dataclass
class InterpretableResult:
    """Results from the interpretable track."""

    functional_score: float  # 0-1, likelihood of functional importance
    metal_binding_score: float  # 0-1, likelihood of metal coordination

    # Component scores
    evolutionary_score: float
    frustration_score: float
    geometry_score: float

    # Key features identified
    key_residues: List[int]  # Important positions
    predicted_metal_sites: List[
        Tuple[int, MetalType, float]
    ]  # (position, metal, confidence)

    # Explanation components
    evolutionary_explanation: str
    frustration_explanation: str
    metal_explanation: str

    def get_summary_explanation(self) -> str:
        """Generate a concise explanation of the prediction."""
        return f"{self.evolutionary_explanation} {self.frustration_explanation} {self.metal_explanation}"


@dataclass
class BlackBoxResult:
    """Results from the black box track."""

    functional_score: float  # 0-1, likelihood of functional importance
    confidence: float  # Model confidence in prediction

    # Neural network outputs
    attention_weights: Optional[torch.Tensor] = None  # Attention over sequence
    hidden_representations: Optional[torch.Tensor] = None  # Learned features

    # Conformational predictions
    conformational_ensemble: Optional[List[torch.Tensor]] = None  # Predicted structures
    dynamics_score: float = 0.0  # Conformational flexibility score

    # Feature importance (extracted via gradient analysis)
    residue_importance: Optional[np.ndarray] = None  # Per-residue importance scores


@dataclass
class PredictionResult:
    """Unified prediction result combining both tracks."""

    # Ensemble prediction
    ensemble_score: float  # Final functional importance score
    confidence: float  # Overall confidence in prediction

    # Track-specific results
    interpretable_result: InterpretableResult
    blackbox_result: BlackBoxResult

    # Agreement metrics
    track_agreement: float  # 0-1, how well tracks agree
    agreement_regions: List[Tuple[int, int]]  # Regions where tracks agree

    # Final predictions
    functional_sites: List[Tuple[int, int, float]]  # (start, end, score)
    metal_binding_sites: List[MetalSite]  # Predicted metal sites

    # Explanation
    explanation: str  # Human-readable explanation
    confidence_breakdown: Dict[str, float]  # Confidence by component

    # Metadata
    prediction_time: float  # Time taken for prediction
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary for API responses."""
        return {
            "ensemble_score": self.ensemble_score,
            "confidence": self.confidence,
            "track_agreement": self.track_agreement,
            "functional_sites": [
                {"start": start, "end": end, "score": score}
                for start, end, score in self.functional_sites
            ],
            "metal_binding_sites": [
                {
                    "metal_type": site.metal_type.value,
                    "position": [site.center.x, site.center.y, site.center.z],
                    "coordination_number": site.coordination_number,
                    "geometry": site.geometry.value,
                }
                for site in self.metal_binding_sites
            ],
            "explanation": self.explanation,
            "confidence_breakdown": self.confidence_breakdown,
            "model_version": self.model_version,
        }


# Type aliases for common patterns
SequencePosition = int  # 1-indexed position in sequence
ConfidenceScore = float  # 0-1 confidence value
ResidueRange = Tuple[int, int]  # Start and end positions (inclusive)
ContactPair = Tuple[int, int]  # Pair of residue positions
