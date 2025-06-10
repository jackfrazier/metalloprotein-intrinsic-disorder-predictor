"""
Custom exceptions for the MIDP system.

This module defines specific exception types to handle various error
scenarios in a structured way across both tracks.
"""

from typing import Optional, List, Dict, Any


class MIDPException(Exception):
    """Base exception class for all MIDP-specific errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


# ============================================================================
# DATA VALIDATION EXCEPTIONS
# ============================================================================


class ValidationError(MIDPException):
    """Raised when input data fails validation."""

    pass


class SequenceValidationError(ValidationError):
    """Raised when protein sequence contains invalid characters or format."""

    def __init__(self, sequence: str, invalid_positions: List[int]):
        message = f"Invalid amino acid sequence at positions: {invalid_positions}"
        details = {
            "sequence_length": len(sequence),
            "invalid_positions": invalid_positions,
            "invalid_chars": [
                sequence[i - 1] for i in invalid_positions if i <= len(sequence)
            ],
        }
        super().__init__(message, details)


class StructureValidationError(ValidationError):
    """Raised when protein structure data is invalid or corrupted."""

    def __init__(self, pdb_file: str, reason: str):
        message = f"Invalid structure file '{pdb_file}': {reason}"
        details = {"file": pdb_file, "reason": reason}
        super().__init__(message, details)


class MetalSiteValidationError(ValidationError):
    """Raised when metal coordination site data is invalid."""

    def __init__(self, metal_type: str, reason: str):
        message = f"Invalid metal site ({metal_type}): {reason}"
        details = {"metal_type": metal_type, "reason": reason}
        super().__init__(message, details)


# ============================================================================
# SCIENTIFIC CALCULATION EXCEPTIONS
# ============================================================================


class ScientificCalculationError(MIDPException):
    """Base class for errors in scientific calculations."""

    pass


class FrustrationCalculationError(ScientificCalculationError):
    """Raised when frustration analysis fails."""

    def __init__(self, protein_id: str, step: str, reason: str):
        message = f"Frustration calculation failed for {protein_id} at {step}"
        details = {"protein_id": protein_id, "step": step, "reason": reason}
        super().__init__(message, details)


class CoordinationGeometryError(ScientificCalculationError):
    """Raised when metal coordination geometry analysis fails."""

    def __init__(self, geometry_type: str, coordination_number: int):
        message = f"Invalid coordination geometry: {geometry_type} with CN={coordination_number}"
        details = {
            "geometry_type": geometry_type,
            "coordination_number": coordination_number,
        }
        super().__init__(message, details)


class EvolutionaryAnalysisError(ScientificCalculationError):
    """Raised when evolutionary analysis fails."""

    def __init__(self, analysis_type: str, reason: str):
        message = f"Evolutionary analysis '{analysis_type}' failed: {reason}"
        details = {"analysis_type": analysis_type, "reason": reason}
        super().__init__(message, details)


# ============================================================================
# MODEL-RELATED EXCEPTIONS
# ============================================================================


class ModelError(MIDPException):
    """Base class for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a required model checkpoint is not found."""

    def __init__(self, model_name: str, path: str):
        message = f"Model '{model_name}' not found at {path}"
        details = {"model_name": model_name, "path": path}
        super().__init__(message, details)


class ModelLoadingError(ModelError):
    """Raised when model loading fails."""

    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load model '{model_name}': {reason}"
        details = {"model_name": model_name, "reason": reason}
        super().__init__(message, details)


class PredictionError(ModelError):
    """Raised when prediction fails."""

    def __init__(self, track: str, reason: str):
        message = f"Prediction failed in {track} track: {reason}"
        details = {"track": track, "reason": reason}
        super().__init__(message, details)


# ============================================================================
# INTEGRATION EXCEPTIONS
# ============================================================================


class IntegrationError(MIDPException):
    """Base class for integration layer errors."""

    pass


class TrackDisagreementError(IntegrationError):
    """Raised when tracks disagree beyond acceptable threshold."""

    def __init__(
        self, interpretable_score: float, blackbox_score: float, threshold: float
    ):
        disagreement = abs(interpretable_score - blackbox_score)
        message = (
            f"Track disagreement ({disagreement:.3f}) exceeds threshold ({threshold})"
        )
        details = {
            "interpretable_score": interpretable_score,
            "blackbox_score": blackbox_score,
            "disagreement": disagreement,
            "threshold": threshold,
        }
        super().__init__(message, details)


class ConfidenceCalibrationError(IntegrationError):
    """Raised when confidence calibration fails."""

    def __init__(self, raw_confidence: float, reason: str):
        message = f"Failed to calibrate confidence ({raw_confidence}): {reason}"
        details = {"raw_confidence": raw_confidence, "reason": reason}
        super().__init__(message, details)


# ============================================================================
# I/O AND DATA ACCESS EXCEPTIONS
# ============================================================================


class DataAccessError(MIDPException):
    """Base class for data access errors."""

    pass


class FileNotFoundError(DataAccessError):
    """Raised when a required file is not found."""

    def __init__(self, file_path: str, file_type: str):
        message = f"{file_type} file not found: {file_path}"
        details = {"file_path": file_path, "file_type": file_type}
        super().__init__(message, details)


class S3AccessError(DataAccessError):
    """Raised when S3 access fails."""

    def __init__(self, bucket: str, key: str, operation: str, reason: str):
        message = f"S3 {operation} failed for s3://{bucket}/{key}: {reason}"
        details = {
            "bucket": bucket,
            "key": key,
            "operation": operation,
            "reason": reason,
        }
        super().__init__(message, details)


class DatabaseConnectionError(DataAccessError):
    """Raised when database connection fails."""

    def __init__(self, db_type: str, reason: str):
        message = f"Failed to connect to {db_type} database: {reason}"
        details = {"db_type": db_type, "reason": reason}
        super().__init__(message, details)


# ============================================================================
# API AND SERVICE EXCEPTIONS
# ============================================================================


class APIError(MIDPException):
    """Base class for API-related errors."""

    def __init__(
        self, message: str, status_code: int, details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, details)
        self.status_code = status_code


class RateLimitExceededError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, limit: int, window: str):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        details = {"limit": limit, "window": window}
        super().__init__(message, 429, details)


class InvalidRequestError(APIError):
    """Raised when API request is invalid."""

    def __init__(self, field: str, reason: str):
        message = f"Invalid request: {field} - {reason}"
        details = {"field": field, "reason": reason}
        super().__init__(message, 400, details)


class ServiceUnavailableError(APIError):
    """Raised when a required service is unavailable."""

    def __init__(self, service: str, reason: str):
        message = f"Service '{service}' is unavailable: {reason}"
        details = {"service": service, "reason": reason}
        super().__init__(message, 503, details)


# ============================================================================
# RESOURCE MANAGEMENT EXCEPTIONS
# ============================================================================


class ResourceError(MIDPException):
    """Base class for resource-related errors."""

    pass


class InsufficientMemoryError(ResourceError):
    """Raised when system runs out of memory."""

    def __init__(self, required_gb: float, available_gb: float, operation: str):
        message = f"Insufficient memory for {operation}: need {required_gb}GB, have {available_gb}GB"
        details = {
            "required_gb": required_gb,
            "available_gb": available_gb,
            "operation": operation,
        }
        super().__init__(message, details)


class GPUNotAvailableError(ResourceError):
    """Raised when GPU is required but not available."""

    def __init__(self, reason: str):
        message = f"GPU not available: {reason}"
        details = {"reason": reason}
        super().__init__(message, details)


class TimeoutError(ResourceError):
    """Raised when operation times out."""

    def __init__(self, operation: str, timeout_seconds: int):
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        details = {"operation": operation, "timeout_seconds": timeout_seconds}
        super().__init__(message, details)


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================


class ConfigurationError(MIDPException):
    """Base class for configuration errors."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, config_file: Optional[str] = None):
        message = f"Missing required configuration: {config_key}"
        if config_file:
            message += f" in {config_file}"
        details = {"config_key": config_key, "config_file": config_file}
        super().__init__(message, details)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration value is invalid."""

    def __init__(self, config_key: str, value: Any, reason: str):
        message = f"Invalid configuration for {config_key}: {reason}"
        details = {"config_key": config_key, "value": value, "reason": reason}
        super().__init__(message, details)
