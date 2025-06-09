# File Architecture

metalloprotein-intrinsically-disordered-predictor/
├── .cursor/
│   ├── rules/
│   │    └── midc-development.mdc
│   └── docs/                          # Project documentation
│       ├── project-architecture.md
│       ├── important_functions.md
│       └── file-structure.md
├── src/midp/
│   ├── __init__.py
│   ├── core/                          # Shared core functionality
│   │   ├── __init__.py
│   │   ├── data_structures.py        # Common data classes
│   │   ├── interfaces.py             # Abstract interfaces for both tracks
│   │   ├── constants.py              # Scientific constants and configs
│   │   └── exceptions.py             # Custom exception classes
│   ├── data/                          # Data handling and preprocessing
│   │   ├── __init__.py
│   │   ├── loaders.py                # PDB and sequence loaders
│   │   ├── preprocessors.py          # Feature extraction pipelines
│   │   ├── datasets.py               # PyTorch dataset classes
│   │   ├── augmentation.py           # Data augmentation strategies
│   │   └── validation.py             # Data quality checks
│   ├── interpretable/                 # Interpretable track implementation
│   │   ├── __init__.py
│   │   ├── evolutionary/
│   │   │   ├── __init__.py
│   │   │   ├── msa_processor.py     # Multiple sequence alignment handling
│   │   │   ├── conservation.py       # Conservation score calculation
│   │   │   ├── coevolution.py       # Coevolution network analysis
│   │   │   └── evolutionary_features.py
│   │   ├── frustration/
│   │   │   ├── __init__.py
│   │   │   ├── energy_calculations.py # Statistical potential calculations
│   │   │   ├── frustratometer.py     # Core frustration algorithm
│   │   │   ├── frustration_mapper.py  # Spatial frustration mapping
│   │   │   └── metal_frustration.py  # Metal-specific extensions
│   │   ├── metal_coordination/
│   │   │   ├── __init__.py
│   │   │   ├── geometry_templates.py  # Coordination geometry definitions
│   │   │   ├── metal_detector.py     # Metal site identification
│   │   │   ├── coordination_scorer.py # Geometry scoring functions
│   │   │   └── electronic_analysis.py # Ligand field calculations
│   │   ├── integration/
│   │   │   ├── __init__.py
│   │   │   ├── decision_tree.py      # Interpretable integration logic
│   │   │   └── feature_combiner.py   # Feature aggregation
│   │   └── interpretable_predictor.py # Main interpretable track class
│   ├── blackbox/                      # Black box track implementation
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── encoders.py           # Feature encoding modules
│   │   │   ├── graph_layers.py       # Custom GNN layers
│   │   │   ├── attention_layers.py   # Multi-head attention implementations
│   │   │   ├── hierarchical_gnn.py   # Main hierarchical GNN model
│   │   │   └── dynamics_predictor.py # Conformational dynamics module
│   │   ├── protein_language_models/
│   │   │   ├── __init__.py
│   │   │   ├── esm_wrapper.py        # ESM model integration
│   │   │   ├── fine_tuning.py        # Domain-specific fine-tuning
│   │   │   └── embedding_processor.py # PLM embedding handling
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── losses.py             # Custom loss functions
│   │   │   ├── optimizers.py         # Optimization strategies
│   │   │   ├── schedulers.py         # Learning rate scheduling
│   │   │   └── mixed_precision.py    # FP16 training utilities
│   │   └── blackbox_predictor.py     # Main black box track class
│   ├── integration/                   # Integration layer
│   │   ├── __init__.py
│   │   ├── confidence_calibration.py  # Confidence estimation
│   │   ├── agreement_analyzer.py      # Track agreement analysis
│   │   ├── ensemble_strategies.py     # Prediction combination logic
│   │   ├── knowledge_distillation.py  # Pattern extraction from black box
│   │   └── integration_predictor.py   # Main integration class
│   ├── cloud/                         # AWS deployment utilities
│   │   ├── __init__.py
│   │   ├── s3_handler.py             # S3 storage management
│   │   ├── sagemaker_wrapper.py      # SageMaker integration
│   │   ├── batch_processor.py         # AWS Batch job handling
│   │   └── monitoring.py              # CloudWatch integration
│   ├── api/                           # API implementation
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPI application
│   │   ├── models.py                 # Pydantic models
│   │   ├── endpoints.py              # API endpoint definitions
│   │   └── middleware.py             # Request processing middleware
│   ├── cli/                          # Command-line interfaces
│   │   ├── __init__.py
│   │   ├── train.py                  # Training script
│   │   ├── predict.py                # Prediction script
│   │   ├── validate.py               # Validation script
│   │   └── deploy.py                 # Deployment script
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── logging_config.py         # Logging configuration
│       ├── metrics.py                # Performance metrics
│       ├── visualization.py          # Result visualization
│       └── profiling.py              # Memory and performance profiling
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── unit/                         # Unit tests
│   │   ├── interpretable/
│   │   ├── blackbox/
│   │   └── integration/
│   ├── integration/                  # Integration tests
│   │   ├── test_dual_track.py
│   │   └── test_api.py
│   └── fixtures/                     # Test data and fixtures
│       ├── proteins/
│       └── configs/
├── configs/                          # Configuration files
│   ├── default.yaml                  # Default configuration
│   ├── interpretable.yaml            # Interpretable track config
│   ├── blackbox.yaml                 # Black box track config
│   └── aws.yaml                      # AWS deployment config
├── scripts/                          # Utility scripts
│   ├── download_data.py              # Data download script
│   ├── prepare_dataset.py            # Dataset preparation
│   ├── benchmark.py                  # Performance benchmarking
│   └── validate_installation.py      # Installation verification
├── notebooks/                        # Jupyter notebooks
│   ├── exploratory/                  # Data exploration
│   ├── experiments/                  # Experimental notebooks
│   └── tutorials/                    # Usage tutorials
├── docker/                           # Docker configurations
│   ├── Dockerfile.cpu                # CPU-only container
│   ├── Dockerfile.gpu                # GPU-enabled container
│   └── docker-compose.yml            # Multi-service composition
├── .github/                          # GitHub configurations
│   └── workflows/
│       ├── tests.yml                 # CI testing workflow
│       └── deploy.yml                # CD deployment workflow
├── pyproject.toml                    # Poetry configuration
├── README.md                         # Project documentation
├── LICENSE                           # License file
└── .gitignore                        # Git ignore rules
