"""
Protein data loaders for PDB files.

This module provides functionality to load protein structures from PDB files,
either from local filesystem or by downloading from RCSB PDB database.
"""

import logging
import os
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set
from urllib.error import HTTPError, URLError

import numpy as np
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB.Structure import Structure
from Bio.SeqUtils import seq1

from ..core.constants import (
    METAL_LIGAND_DISTANCES,
    VALIDATION_LIMITS,
    VAN_DER_WAALS_RADII,
)
from ..core.data_structures import (
    ProteinData,
    MetalSite,
    MetalType,
    Residue,
    ResidueType,
    Coordinates3D,
    MetalLigand,
    DisorderRegion,
    CoordinationGeometry,
)
from ..core.exceptions import (
    StructureValidationError,
    FileNotFoundError,
    DataAccessError,
    ValidationError,
)

logger = logging.getLogger(__name__)

# Metal atom names commonly found in PDB files
METAL_ATOMS = {
    "FE",
    "ZN",
    "CU",
    "CA",
    "MG",
    "MN",
    "CO",
    "NI",
    "MO",
    "W",
    "FE2",
    "FE3",
    "ZN2",
    "CU1",
    "CU2",
    "CA2",
    "MG2",
    "MN2",
    "CO2",
    "NI2",
}

# Common ligand atoms that coordinate metals
LIGAND_ATOMS = {"N", "O", "S", "CL", "BR", "I", "F"}


class ProteinLoader:
    """
    Loads protein structures from PDB files with comprehensive parsing.

    This class handles:
    - Local PDB file loading
    - Remote PDB downloading from RCSB
    - Metal site detection
    - Disorder region prediction
    - Data validation
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        validate_structure: bool = True,
        detect_metals: bool = True,
        predict_disorder: bool = True,
    ):
        """
        Initialize the protein loader.

        Args:
            cache_dir: Directory to cache downloaded PDB files
            validate_structure: Whether to validate loaded structures
            detect_metals: Whether to detect metal binding sites
            predict_disorder: Whether to predict disorder regions
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "midp_pdb_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.validate_structure = validate_structure
        self.detect_metals = detect_metals
        self.predict_disorder = predict_disorder

        # Initialize parsers
        self.pdb_parser = PDBParser(QUIET=True, PERMISSIVE=True)
        self.mmcif_parser = MMCIFParser(QUIET=True)

        logger.info(f"Initialized ProteinLoader with cache at {self.cache_dir}")

    def load_protein(
        self,
        source: str,
        protein_id: Optional[str] = None,
        chain_ids: Optional[List[str]] = None,
    ) -> ProteinData:
        """
        Load a protein from either a file path or PDB ID.

        Args:
            source: Either a file path or a 4-character PDB ID
            protein_id: Optional custom ID for the protein
            chain_ids: Optional list of chain IDs to include

        Returns:
            ProteinData object with all extracted information

        Raises:
            FileNotFoundError: If local file doesn't exist
            DataAccessError: If PDB download fails
            StructureValidationError: If structure is invalid
        """
        # Determine if source is a file or PDB ID
        if len(source) == 4 and source.isalnum():
            # Likely a PDB ID
            structure = self._download_and_parse_pdb(source)
            source_file = self.cache_dir / f"{source}.pdb"
        else:
            # File path
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(str(source_path), "PDB")
            structure = self._parse_pdb_file(source_path)
            source_file = source_path

        # Use provided ID or extract from structure
        if protein_id is None:
            protein_id = structure.id

        # Extract basic information
        sequence, residues, chains = self._extract_sequence_and_residues(
            structure, chain_ids
        )

        # Create base protein data
        protein_data = ProteinData(
            protein_id=protein_id,
            sequence=sequence,
            chain_ids=chains,
            structure=structure,
            residues=residues,
            source_file=source_file,
            experimental_method=self._get_experimental_method(structure),
            resolution=self._get_resolution(structure),
        )

        # Detect metal sites if requested
        if self.detect_metals:
            protein_data.metal_sites = self._detect_metal_sites(structure, residues)
            logger.info(f"Detected {len(protein_data.metal_sites)} metal sites")

        # Predict disorder regions if requested
        if self.predict_disorder:
            disorder_regions, disorder_scores = self._predict_disorder_regions(
                sequence, residues
            )
            protein_data.disorder_regions = disorder_regions
            protein_data.global_disorder_content = np.mean(disorder_scores)

            # Update residue disorder scores
            for i, residue in enumerate(residues):
                if i < len(disorder_scores):
                    residue.disorder_score = disorder_scores[i]

        # Validate if requested
        if self.validate_structure:
            self._validate_protein_data(protein_data)

        logger.info(
            f"Successfully loaded protein {protein_id} with "
            f"{len(sequence)} residues and {len(protein_data.metal_sites)} metal sites"
        )

        return protein_data

    def _download_and_parse_pdb(self, pdb_id: str) -> Structure:
        """Download PDB file from RCSB and parse it."""
        pdb_id = pdb_id.upper()
        cached_file = self.cache_dir / f"{pdb_id}.pdb"

        # Check cache first
        if cached_file.exists():
            logger.info(f"Loading {pdb_id} from cache")
            return self._parse_pdb_file(cached_file)

        # Download from RCSB
        logger.info(f"Downloading {pdb_id} from RCSB PDB")
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        try:
            urllib.request.urlretrieve(url, cached_file)
        except (HTTPError, URLError) as e:
            # Try mmCIF format
            logger.warning(f"PDB format failed, trying mmCIF for {pdb_id}")
            url = f"https://files.rcsb.org/download/{pdb_id}.cif"
            cif_file = self.cache_dir / f"{pdb_id}.cif"

            try:
                urllib.request.urlretrieve(url, cif_file)
                return self._parse_mmcif_file(cif_file)
            except (HTTPError, URLError) as e2:
                raise DataAccessError(
                    "RCSB PDB", pdb_id, "download", f"Failed to download: {str(e2)}"
                )

        return self._parse_pdb_file(cached_file)

    def _parse_pdb_file(self, pdb_file: Path) -> Structure:
        """Parse a PDB file using BioPython."""
        try:
            structure = self.pdb_parser.get_structure(pdb_file.stem, str(pdb_file))
            return structure
        except Exception as e:
            raise StructureValidationError(str(pdb_file), f"Failed to parse: {str(e)}")

    def _parse_mmcif_file(self, cif_file: Path) -> Structure:
        """Parse an mmCIF file using BioPython."""
        try:
            structure = self.mmcif_parser.get_structure(cif_file.stem, str(cif_file))
            return structure
        except Exception as e:
            raise StructureValidationError(
                str(cif_file), f"Failed to parse mmCIF: {str(e)}"
            )

    def _extract_sequence_and_residues(
        self, structure: Structure, chain_ids: Optional[List[str]] = None
    ) -> Tuple[str, List[Residue], List[str]]:
        """Extract sequence and residue information from structure."""
        sequence_parts = []
        residues = []
        chains_found = []

        logger.info(f"Starting sequence extraction from structure {structure.id}")

        for model in structure:
            logger.info(f"Processing model {model.id}")
            for chain in model:
                logger.info(f"Processing chain {chain.id}")
                # Skip if not in requested chains
                if chain_ids and chain.id not in chain_ids:
                    logger.info(
                        f"Skipping chain {chain.id} - not in requested chains {chain_ids}"
                    )
                    continue

                chains_found.append(chain.id)
                residue_count = 0
                skipped_count = 0

                for bio_residue in chain:
                    # Log residue info
                    logger.info(
                        f"Examining residue: {bio_residue.id} {bio_residue.get_resname()}"
                    )

                    # Skip hetero atoms and water
                    if bio_residue.id[0] != " ":  # Skip hetero and water
                        logger.info(
                            f"Skipping residue {bio_residue.id} - hetero or water"
                        )
                        skipped_count += 1
                        continue

                    # Get residue information
                    resname = bio_residue.get_resname().strip()

                    try:
                        # Try using seq1 first (handles modified residues better)
                        one_letter = seq1(resname)
                        if not one_letter:
                            # Fallback to standard mapping
                            one_letter = protein_letters_3to1.get(resname)
                            logger.info(
                                f"Used fallback mapping for {resname} -> {one_letter}"
                            )
                        else:
                            logger.info(
                                f"Successfully mapped {resname} -> {one_letter}"
                            )
                    except Exception as e:
                        logger.info(f"Exception in residue conversion: {str(e)}")
                        one_letter = protein_letters_3to1.get(resname)

                    if not one_letter:
                        logger.warning(
                            f"Unknown residue {resname} at position {bio_residue.id[1]} "
                            f"in chain {chain.id}, skipping"
                        )
                        skipped_count += 1
                        continue

                    residue_count += 1
                    sequence_parts.append(one_letter)

                    # Get CA coordinates if available
                    ca_coords = None
                    if "CA" in bio_residue:
                        ca = bio_residue["CA"]
                        ca_coords = Coordinates3D(
                            x=ca.coord[0],
                            y=ca.coord[1],
                            z=ca.coord[2],
                            confidence=ca.get_bfactor(),
                        )

                    # Map to our ResidueType enum
                    try:
                        residue_type = ResidueType[resname]
                    except KeyError:
                        # Handle non-standard amino acids
                        logger.warning(
                            f"Non-standard amino acid {resname} at position {bio_residue.id[1]} "
                            f"in chain {chain.id}, using closest standard type"
                        )
                        # Map to closest standard residue based on one letter code
                        standard_resname = None
                        for std_name, std_one in protein_letters_3to1.items():
                            if std_one == one_letter:
                                standard_resname = std_name
                                break
                        residue_type = ResidueType[
                            standard_resname or "ALA"
                        ]  # Default to ALA if no match

                    residue = Residue(
                        residue_type=residue_type,
                        position=len(residues) + 1,
                        chain_id=chain.id,
                        coordinates=ca_coords,
                        secondary_structure=None,  # Will be computed later
                        disorder_score=None,  # Will be computed later
                        conservation_score=None,  # Will be computed later
                        frustration_index=None,  # Will be computed later
                    )

                    residues.append(residue)

                logger.info(
                    f"Chain {chain.id} stats: {residue_count} residues processed, {skipped_count} skipped"
                )

            # Only process first model
            break

        sequence = "".join(sequence_parts)

        if not sequence:
            logger.error("No amino acid residues found in structure")
            raise StructureValidationError(structure.id, "No amino acid residues found")

        logger.info(f"Final sequence length: {len(sequence)}")
        return sequence, residues, chains_found

    def _detect_metal_sites(
        self, structure: Structure, residues: List[Residue]
    ) -> List[MetalSite]:
        """Detect metal binding sites in the structure."""
        metal_sites = []

        for model in structure:
            # Find all metal atoms
            metal_atoms = []
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        element = (
                            atom.element.upper() if atom.element else atom.name[:2]
                        )
                        if element in METAL_ATOMS:
                            metal_atoms.append((atom, residue, chain))

            # For each metal, find coordinating ligands
            for metal_atom, metal_res, metal_chain in metal_atoms:
                ligands = self._find_metal_ligands(metal_atom, model, metal_res.id[1])

                if not ligands:
                    logger.warning(f"No ligands found for metal {metal_atom.name}")
                    continue

                # Determine metal type
                element = (
                    metal_atom.element.upper()
                    if metal_atom.element
                    else metal_atom.name[:2]
                )
                metal_type = self._get_metal_type(element)

                # Determine coordination geometry
                geometry, rmsd = self._determine_coordination_geometry(
                    metal_atom, [lig[0] for lig in ligands]
                )

                # Create MetalLigand objects
                metal_ligands = []
                for lig_atom, lig_res, lig_chain, distance in ligands:
                    # Find corresponding residue in our list
                    residue_match = None
                    for res in residues:
                        if (
                            res.chain_id == lig_chain.id
                            and res.position == lig_res.id[1]
                        ):
                            residue_match = res
                            break

                    if residue_match:
                        metal_ligand = MetalLigand(
                            residue=residue_match,
                            atom_name=lig_atom.name,
                            coordinates=Coordinates3D(
                                x=lig_atom.coord[0],
                                y=lig_atom.coord[1],
                                z=lig_atom.coord[2],
                            ),
                            bond_length=distance,
                        )
                        metal_ligands.append(metal_ligand)

                # Create MetalSite
                metal_site = MetalSite(
                    metal_type=metal_type,
                    center=Coordinates3D(
                        x=metal_atom.coord[0],
                        y=metal_atom.coord[1],
                        z=metal_atom.coord[2],
                    ),
                    ligands=metal_ligands,
                    geometry=geometry,
                    geometry_rmsd=rmsd,
                    occupancy=metal_atom.get_occupancy(),
                    b_factor=metal_atom.get_bfactor(),
                )

                metal_sites.append(metal_site)

            # Only process first model
            break

        return metal_sites

    def _find_metal_ligands(
        self, metal_atom: PDB.Atom.Atom, model: PDB.Model.Model, metal_resnum: int
    ) -> List[Tuple]:
        """Find atoms coordinating a metal ion."""
        ligands = []
        metal_coord = metal_atom.coord
        metal_element = metal_atom.element.upper() if metal_atom.element else "FE"

        # Get distance thresholds for this metal
        if metal_element in METAL_LIGAND_DISTANCES:
            distance_ranges = METAL_LIGAND_DISTANCES[metal_element]
        else:
            # Default distances
            distance_ranges = {"N": (1.8, 2.5), "O": (1.8, 2.5), "S": (2.0, 2.8)}

        # Search all atoms in the model
        for chain in model:
            for residue in chain:
                # Skip the metal's own residue
                if residue.id[1] == metal_resnum:
                    continue

                for atom in residue:
                    # Check if it's a potential ligand atom
                    element = atom.element.upper() if atom.element else atom.name[0]
                    if element not in LIGAND_ATOMS:
                        continue

                    # Calculate distance
                    distance = np.linalg.norm(atom.coord - metal_coord)

                    # Check if within bonding distance
                    if element in distance_ranges:
                        min_dist, max_dist = distance_ranges[element]
                        if min_dist <= distance <= max_dist:
                            ligands.append((atom, residue, chain, distance))
                    elif distance < 3.0:  # Default max distance
                        ligands.append((atom, residue, chain, distance))

        # Sort by distance
        ligands.sort(key=lambda x: x[3])

        # Return closest ligands (up to expected coordination number)
        return ligands[:8]  # Max coordination number is typically 8

    def _get_metal_type(self, element: str) -> MetalType:
        """Map element name to MetalType enum."""
        mapping = {
            "FE": MetalType.FE2,  # Default to Fe2+
            "FE2": MetalType.FE2,
            "FE3": MetalType.FE3,
            "ZN": MetalType.ZN2,
            "CU": MetalType.CU2,  # Default to Cu2+
            "CU1": MetalType.CU1,
            "CU2": MetalType.CU2,
            "CA": MetalType.CA2,
            "MG": MetalType.MG2,
            "MN": MetalType.MN2,
            "CO": MetalType.CO2,
            "NI": MetalType.NI2,
        }

        return mapping.get(element.upper(), MetalType.OTHER)

    def _determine_coordination_geometry(
        self, metal_atom: PDB.Atom.Atom, ligand_atoms: List[PDB.Atom.Atom]
    ) -> Tuple[CoordinationGeometry, float]:
        """Determine the coordination geometry of a metal site."""
        n_ligands = len(ligand_atoms)

        if n_ligands == 0:
            return CoordinationGeometry.UNKNOWN, float("inf")
        elif n_ligands == 1:
            return CoordinationGeometry.UNKNOWN, 0.0
        elif n_ligands == 2:
            return CoordinationGeometry.LINEAR, self._calculate_geometry_rmsd(
                metal_atom, ligand_atoms, CoordinationGeometry.LINEAR
            )
        elif n_ligands == 3:
            return CoordinationGeometry.TRIGONAL_PLANAR, self._calculate_geometry_rmsd(
                metal_atom, ligand_atoms, CoordinationGeometry.TRIGONAL_PLANAR
            )
        elif n_ligands == 4:
            # Could be tetrahedral or square planar
            tet_rmsd = self._calculate_geometry_rmsd(
                metal_atom, ligand_atoms, CoordinationGeometry.TETRAHEDRAL
            )
            sq_rmsd = self._calculate_geometry_rmsd(
                metal_atom, ligand_atoms, CoordinationGeometry.SQUARE_PLANAR
            )

            if tet_rmsd < sq_rmsd:
                return CoordinationGeometry.TETRAHEDRAL, tet_rmsd
            else:
                return CoordinationGeometry.SQUARE_PLANAR, sq_rmsd
        elif n_ligands == 5:
            # Could be trigonal bipyramidal or square pyramidal
            tbp_rmsd = self._calculate_geometry_rmsd(
                metal_atom, ligand_atoms, CoordinationGeometry.TRIGONAL_BIPYRAMIDAL
            )
            spy_rmsd = self._calculate_geometry_rmsd(
                metal_atom, ligand_atoms, CoordinationGeometry.SQUARE_PYRAMIDAL
            )

            if tbp_rmsd < spy_rmsd:
                return CoordinationGeometry.TRIGONAL_BIPYRAMIDAL, tbp_rmsd
            else:
                return CoordinationGeometry.SQUARE_PYRAMIDAL, spy_rmsd
        elif n_ligands == 6:
            return CoordinationGeometry.OCTAHEDRAL, self._calculate_geometry_rmsd(
                metal_atom, ligand_atoms, CoordinationGeometry.OCTAHEDRAL
            )
        else:
            return CoordinationGeometry.UNKNOWN, float("inf")

    def _calculate_geometry_rmsd(
        self,
        metal_atom: PDB.Atom.Atom,
        ligand_atoms: List[PDB.Atom.Atom],
        geometry: CoordinationGeometry,
    ) -> float:
        """Calculate RMSD from ideal geometry."""
        # This is a simplified calculation
        # In practice, you would align to ideal geometry templates
        # For now, return a placeholder value
        return 0.5

    def _predict_disorder_regions(
        self, sequence: str, residues: List[Residue]
    ) -> Tuple[List[DisorderRegion], np.ndarray]:
        """
        Predict intrinsically disordered regions.

        This is a simplified predictor based on amino acid composition.
        In production, you would use tools like IUPred or DISOPRED.
        """
        from ..core.constants import DISORDER_PROPENSITY, DISORDER_THRESHOLDS

        # Calculate per-residue disorder scores
        disorder_scores = np.zeros(len(sequence))

        for i, aa in enumerate(sequence):
            if aa in DISORDER_PROPENSITY:
                # Simple sliding window approach
                window_size = 9
                start = max(0, i - window_size // 2)
                end = min(len(sequence), i + window_size // 2 + 1)

                window_score = 0.0
                for j in range(start, end):
                    if sequence[j] in DISORDER_PROPENSITY:
                        window_score += DISORDER_PROPENSITY[sequence[j]]

                disorder_scores[i] = (window_score / (end - start) + 1.0) / 2.0

        # Smooth scores
        from scipy.ndimage import gaussian_filter1d

        disorder_scores = gaussian_filter1d(disorder_scores, sigma=2.0)

        # Identify disorder regions
        disorder_regions = []
        in_disorder = False
        start_pos = 0

        threshold = DISORDER_THRESHOLDS["confident_disorder"]

        for i, score in enumerate(disorder_scores):
            if score > threshold and not in_disorder:
                in_disorder = True
                start_pos = i + 1  # 1-indexed
            elif score <= threshold and in_disorder:
                in_disorder = False
                if i - start_pos >= 5:  # Minimum length
                    region = DisorderRegion(
                        start=start_pos,
                        end=i,  # 1-indexed, inclusive
                        disorder_probability=np.mean(
                            disorder_scores[start_pos - 1 : i]
                        ),
                    )
                    disorder_regions.append(region)

        # Handle case where disorder extends to the end
        if in_disorder and len(sequence) - start_pos >= 5:
            region = DisorderRegion(
                start=start_pos,
                end=len(sequence),
                disorder_probability=np.mean(disorder_scores[start_pos - 1 :]),
            )
            disorder_regions.append(region)

        return disorder_regions, disorder_scores

    def _get_experimental_method(self, structure: Structure) -> Optional[str]:
        """Extract experimental method from structure."""
        if hasattr(structure, "header") and "structure_method" in structure.header:
            return structure.header["structure_method"]
        return None

    def _get_resolution(self, structure: Structure) -> Optional[float]:
        """Extract resolution from structure."""
        if hasattr(structure, "header") and "resolution" in structure.header:
            try:
                return float(structure.header["resolution"])
            except (ValueError, TypeError):
                return None
        return None

    def _validate_protein_data(self, protein_data: ProteinData):
        """Validate the loaded protein data."""
        # Check sequence length
        if len(protein_data.sequence) < VALIDATION_LIMITS["min_sequence_length"]:
            raise ValidationError(
                f"Sequence too short: {len(protein_data.sequence)} < "
                f"{VALIDATION_LIMITS['min_sequence_length']}"
            )

        if len(protein_data.sequence) > VALIDATION_LIMITS["max_sequence_length"]:
            raise ValidationError(
                f"Sequence too long: {len(protein_data.sequence)} > "
                f"{VALIDATION_LIMITS['max_sequence_length']}"
            )

        # Check for valid amino acids
        valid_aas = VALIDATION_LIMITS["valid_amino_acids"]
        invalid_positions = [
            i + 1 for i, aa in enumerate(protein_data.sequence) if aa not in valid_aas
        ]

        if invalid_positions:
            from ..core.exceptions import SequenceValidationError

            raise SequenceValidationError(protein_data.sequence, invalid_positions)

        # Check resolution if available
        if protein_data.resolution is not None:
            if (
                protein_data.resolution < VALIDATION_LIMITS["min_structure_resolution"]
                or protein_data.resolution
                > VALIDATION_LIMITS["max_structure_resolution"]
            ):
                logger.warning(f"Unusual resolution: {protein_data.resolution} Å")

        # Check for missing residues
        if protein_data.has_structure:
            n_missing = sum(
                1 for res in protein_data.residues if res.coordinates is None
            )
            missing_fraction = n_missing / len(protein_data.residues)

            if missing_fraction > VALIDATION_LIMITS["max_missing_residues_fraction"]:
                logger.warning(
                    f"High fraction of missing residues: {missing_fraction:.2%}"
                )


# Convenience function
def load_protein(source: str, **kwargs) -> ProteinData:
    """
    Convenience function to load a protein.

    Args:
        source: Either a file path or PDB ID
        **kwargs: Additional arguments passed to ProteinLoader

    Returns:
        Loaded ProteinData object
    """
    loader = ProteinLoader()
    return loader.load_protein(source, **kwargs)
