# Core Functions for Minimal Working System

## 1. Data Structure Foundation

Location: idp_predictor/core/data_structures.py

```python
@dataclass
class ProteinData:
    """Core data structure for protein information."""
    sequence: str
    structure: Optional[Structure]  # BioPython structure
    metal_sites: List[MetalSite]
    disorder_regions: List[Tuple[int, int]]


@dataclass
class PredictionResult:
    """Unified prediction result from both tracks."""
    ensemble_score: float
    interpretable_components: Dict[str, float]
    blackbox_confidence: float
    agreement_level: float
    explanation: Optional[str]
```

## 2. Common Interface Definition

Location: idp_predictor/core/interfaces.py

```python
class BasePredictor(ABC):
    """Common interface for both tracks."""

    @abstractmethod
    def predict(self, protein_data: ProteinData) -> PredictionResult:
        """Generate predictions for a protein."""
        pass

    @abstractmethod
    def get_confidence(self, protein_data: ProteinData) -> float:
        """Estimate prediction confidence."""
        pass
```

## 3. Data Loading Pipeline

Location: idp_predictor/data/loaders.py

```python
def load_protein(pdb_path: Path) -> ProteinData:
    """Load and validate protein data from PDB file."""
   # Parse structure
   # Extract sequence
   # Identify metal sites
   # Predict disorder regions
   # Return unified ProteinData object
```

## 4. Interpretable Track Core

Location: idp_predictor/interpretable/interpretable_predictor.py

```python
class InterpretablePredictor(BasePredictor):
    """Main interpretable track implementation."""

    def __init__(self, config: InterpretableConfig):
        self.evolutionary_analyzer = EvolutionaryAnalyzer(config.evolutionary)
        self.frustration_analyzer = FrustrationAnalyzer(config.frustration)
        self.metal_analyzer = MetalCoordinationAnalyzer(config.metal)
        self.integrator = DecisionTreeIntegrator()

    def predict(self, protein_data: ProteinData) -> PredictionResult:
        # Run each analyzer
        # Combine results through decision tree
        # Generate explanation
        # Return structured result
```

## 5. Black Box Track Core

Location: idp_predictor/blackbox/blackbox_predictor.py

```python
class BlackBoxPredictor(BasePredictor):
    """Main black box track implementation."""

    def __init__(self, config: BlackBoxConfig):
        self.encoder = HierarchicalGraphEncoder(config.encoder)
        self.dynamics_model = ConformationalDynamicsModel(config.dynamics)
        self.load_pretrained_weights(config.checkpoint_path)

    def predict(self, protein_data: ProteinData) -> PredictionResult:
        # Encode protein features
        # Process through hierarchical GNN
        # Generate conformational predictions
        # Return structured result
```

## 6. Integration Layer Core

Location: idp_predictor/integration/integration_predictor.py

```python
class IntegrationPredictor:
    """Combines predictions from both tracks."""

    def __init__(self, interpretable: InterpretablePredictor,
                 blackbox: BlackBoxPredictor,
                 config: IntegrationConfig):
        self.interpretable = interpretable
        self.blackbox = blackbox
        self.calibrator = ConfidenceCalibrator()
        self.agreement_analyzer = AgreementAnalyzer()

    def predict(self, protein_data: ProteinData) -> PredictionResult:
        # Get predictions from both tracks
        # Analyze agreement
        # Apply ensemble strategy
        # Return unified result
```

## 7. API Entry Point

Location: idp_predictor/api/app.py

```python
app = FastAPI(title="IDP Metalloprotein Predictor")

@app.post("/predict")
async def predict_protein(pdb_file: UploadFile) -> PredictionResponse:
    """Main prediction endpoint."""
    # Load protein data
    # Run through integration predictor
    # Format response
    # Return JSON result
```

## Execution Order for Basic Prediction

### Initialization Phase

- Load configuration files
- Initialize both track predictors
- Create integration predictor
- Start API server

### Request Processing

- Receive PDB file through API
- load_protein() validates and extracts features
- Create ProteinData object

### Interpretable Track Execution

- EvolutionaryAnalyzer.analyze() processes sequences
- FrustrationAnalyzer.calculate() computes energy landscapes
- MetalCoordinationAnalyzer.assess() evaluates binding sites
- DecisionTreeIntegrator.combine() merges results

### Black Box Track Execution (parallel)

- HierarchicalGraphEncoder.encode() creates graph representation
- ConformationalDynamicsModel.forward() processes through GNN
- Generate prediction scores and confidence

### Integration Phase

- AgreementAnalyzer.compare() assesses track alignment
- ConfidenceCalibrator.calibrate() adjusts confidence scores
- EnsembleStrategy.combine() merges predictions

### Response Generation

- Format results as PredictionResponse
- Include explanations from interpretable track
- Add confidence metrics and agreement scores
- Return JSON response to client

### Minimal Implementation Checklist

To achieve a working prototype, implement these components in order:

1. Core data structures and interfaces (2 files, ~200 lines)
2. Basic data loader for PDB files (1 file, ~150 lines)
3. Simplified interpretable predictor with one analyzer (3 files, ~400 lines)
4. Minimal black box predictor with pre-trained ESM (2 files, ~300 lines)
5. Basic integration logic with simple averaging (1 file, ~150 lines)
6. FastAPI endpoint for predictions (2 files, ~200 lines)
7. Basic tests for each component (5 files, ~500 lines)

Total for minimal system: ~15 files, ~1,900 lines of code
This provides a functional system that can be iteratively enhanced with additional analyzers, more sophisticated models, and advanced integration strategies.
