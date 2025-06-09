"""
Abstract interfaces defining contracts for predictors and analyzers.

This module ensures consistency between interpretable and black box tracks
by defining common interfaces that both must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path

from .data_structures import (
    ProteinData, PredictionResult, InterpretableResult, BlackBoxResult,
    MetalSite, DisorderRegion, EvolutionaryFeatures, FrustrationFeatures
)


class BasePredictor(ABC):
    """Common interface for both interpretable and black box tracks."""
    
    @abstractmethod
    def predict(self, protein_data: ProteinData) -> PredictionResult:
        """
        Generate predictions for a protein.
        
        Args:
            protein_data: Complete protein information
            
        Returns:
            PredictionResult with track-specific results
        """
        pass
    
    @abstractmethod
    def get_confidence(self, protein_data: ProteinData) -> float:
        """
        Estimate prediction confidence for the given protein.
        
        Args:
            protein_data: Complete protein information
            
        Returns:
            Confidence score between 0 and 1
        """
        pass
    
    @abstractmethod
    def validate_input(self, protein_data: ProteinData) -> Tuple[bool, Optional[str]]:
        """
        Validate that the input protein data is suitable for this predictor.
        
        Args:
            protein_data: Protein data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self, protein_data: ProteinData) -> Dict[str, np.ndarray]:
        """
        Get feature importance scores for the prediction.
        
        Args:
            protein_data: Complete protein information
            
        Returns:
            Dictionary mapping feature names to importance arrays
        """
        pass


class InterpretablePredictor(BasePredictor):
    """Interface for the interpretable track predictor."""
    
    @abstractmethod
    def predict(self, protein_data: ProteinData) -> InterpretableResult:
        """Generate interpretable predictions with explanations."""
        pass
    
    @abstractmethod
    def explain_prediction(self, result: InterpretableResult) -> str:
        """Generate human-readable explanation of the prediction."""
        pass


class BlackBoxPredictor(BasePredictor):
    """Interface for the black box track predictor."""
    
    @abstractmethod
    def predict(self, protein_data: ProteinData) -> BlackBoxResult:
        """Generate black box predictions with neural network outputs."""
        pass
    
    @abstractmethod
    def extract_attention_weights(self, protein_data: ProteinData) -> np.ndarray:
        """Extract attention weights for interpretability."""
        pass


class MetalAnalyzer(ABC):
    """Interface for metal coordination analysis."""
    
    @abstractmethod
    def detect_metal_sites(self, protein_data: ProteinData) -> List[MetalSite]:
        """
        Detect potential metal binding sites in the protein.
        
        Args:
            protein_data: Protein structure and sequence information
            
        Returns:
            List of identified metal sites
        """
        pass
    
    @abstractmethod
    def score_metal_site(self, site: MetalSite) -> float:
        """
        Score the quality/likelihood of a metal binding site.
        
        Args:
            site: Metal site to score
            
        Returns:
            Score between 0 and 1
        """
        pass
    
    @abstractmethod
    def predict_metal_type(self, protein_data: ProteinData, position: int) -> Dict[str, float]:
        """
        Predict likely metal types for a binding site.
        
        Args:
            protein_data: Protein information
            position: Residue position of interest
            
        Returns:
            Dictionary mapping metal types to probabilities
        """
        pass


class DisorderPredictor(ABC):
    """Interface for disorder prediction."""
    
    @abstractmethod
    def predict_disorder(self, sequence: str) -> np.ndarray:
        """
        Predict per-residue disorder scores.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Array of disorder scores (0-1) for each residue
        """
        pass
    
    @abstractmethod
    def identify_disorder_regions(self, disorder_scores: np.ndarray) -> List[DisorderRegion]:
        """
        Identify continuous disordered regions from scores.
        
        Args:
            disorder_scores: Per-residue disorder scores
            
        Returns:
            List of disorder regions
        """
        pass


class EvolutionaryAnalyzer(ABC):
    """Interface for evolutionary analysis."""
    
    @abstractmethod
    def analyze(self, protein_data: ProteinData, msa_file: Optional[Path] = None) -> EvolutionaryFeatures:
        """
        Perform evolutionary analysis on the protein.
        
        Args:
            protein_data: Protein information
            msa_file: Optional pre-computed MSA file
            
        Returns:
            Evolutionary features including conservation and coevolution
        """
        pass
    
    @abstractmethod
    def compute_conservation(self, msa: Any) -> np.ndarray:
        """
        Compute per-position conservation scores.
        
        Args:
            msa: Multiple sequence alignment
            
        Returns:
            Conservation scores for each position
        """
        pass
    
    @abstractmethod
    def compute_coevolution(self, msa: Any) -> np.ndarray:
        """
        Compute pairwise coevolution scores.
        
        Args:
            msa: Multiple sequence alignment
            
        Returns:
            Symmetric matrix of coevolution scores
        """
        pass


class FrustrationAnalyzer(ABC):
    """Interface for frustration analysis."""
    
    @abstractmethod
    def calculate(self, protein_data: ProteinData) -> FrustrationFeatures:
        """
        Calculate local frustration for the protein.
        
        Args:
            protein_data: Protein structure information
            
        Returns:
            Frustration features including indices and energy matrices
        """
        pass
    
    @abstractmethod
    def identify_frustrated_contacts(self, 
                                   frustration_features: FrustrationFeatures,
                                   threshold: float = 2.0) -> List[Tuple[int, int, float]]:
        """
        Identify highly frustrated contacts.
        
        Args:
            frustration_features: Calculated frustration features
            threshold: Frustration threshold
            
        Returns:
            List of (residue1, residue2, frustration_score) tuples
        """
        pass


class FeatureExtractor(ABC):
    """Interface for feature extraction from protein data."""
    
    @abstractmethod
    def extract_sequence_features(self, sequence: str) -> np.ndarray:
        """Extract features from amino acid sequence."""
        pass
    
    @abstractmethod
    def extract_structure_features(self, protein_data: ProteinData) -> np.ndarray:
        """Extract features from 3D structure."""
        pass
    
    @abstractmethod
    def extract_metal_features(self, metal_sites: List[MetalSite]) -> np.ndarray:
        """Extract features from metal coordination sites."""
        pass


class ModelCheckpointer(ABC):
    """Interface for model checkpointing and loading."""
    
    @abstractmethod
    def save_checkpoint(self, model: Any, path: Path, metadata: Dict[str, Any]):
        """Save model checkpoint with metadata."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Path) -> Tuple[Any, Dict[str, Any]]:
        """Load model checkpoint and metadata."""
        pass
    
    @abstractmethod
    def save_to_s3(self, local_path: Path, s3_key: str):
        """Upload checkpoint to S3."""
        pass
    
    @abstractmethod
    def load_from_s3(self, s3_key: str, local_path: Path):
        """Download checkpoint from S3."""
        pass


class DataValidator(ABC):
    """Interface for data validation."""
    
    @abstractmethod
    def validate_pdb(self, pdb_file: Path) -> Tuple[bool, Optional[str]]:
        """Validate PDB file format and content."""
        pass
    
    @abstractmethod
    def validate_sequence(self, sequence: str) -> Tuple[bool, Optional[str]]:
        """Validate amino acid sequence."""
        pass
    
    @abstractmethod
    def validate_metal_sites(self, metal_sites: List[MetalSite]) -> Tuple[bool, Optional[str]]:
        """Validate metal coordination sites."""
        pass


class IntegrationStrategy(ABC):
    """Interface for combining predictions from multiple tracks."""
    
    @abstractmethod
    def combine(self, 
                interpretable_result: InterpretableResult,
                blackbox_result: BlackBoxResult,
                confidence_weights: Dict[str, float]) -> PredictionResult:
        """
        Combine results from both tracks into unified prediction.
        
        Args:
            interpretable_result: Results from interpretable track
            blackbox_result: Results from black box track
            confidence_weights: Confidence weights for each track
            
        Returns:
            Unified prediction result
        """
        pass
    
    @abstractmethod
    def calculate_agreement(self,
                          interpretable_result: InterpretableResult,
                          blackbox_result: BlackBoxResult) -> float:
        """Calculate agreement score between tracks."""
        pass


class ConfidenceCalibrator(ABC):
    """Interface for confidence score calibration."""
    
    @abstractmethod
    def calibrate(self, raw_confidence: float, features: np.ndarray) -> float:
        """
        Calibrate raw confidence score.
        
        Args:
            raw_confidence: Uncalibrated confidence from model
            features: Additional features for calibration
            
        Returns:
            Calibrated confidence score
        """
        pass
    
    @abstractmethod
    def fit(self, raw_confidences: np.ndarray, true_outcomes: np.ndarray):
        """Fit calibration model on historical data."""
        pass


# Factory interfaces for creating components
class PredictorFactory(ABC):
    """Factory for creating predictor instances."""
    
    @abstractmethod
    def create_interpretable_predictor(self, config: Dict[str, Any]) -> InterpretablePredictor:
        """Create interpretable track predictor."""
        pass
    
    @abstractmethod
    def create_blackbox_predictor(self, config: Dict[str, Any]) -> BlackBoxPredictor:
        """Create black box track predictor."""
        pass
    
    @abstractmethod
    def create_integration_predictor(self,
                                   interpretable: InterpretablePredictor,
                                   blackbox: BlackBoxPredictor,
                                   config: Dict[str, Any]) -> BasePredictor:
        """Create integration layer predictor."""
        pass
