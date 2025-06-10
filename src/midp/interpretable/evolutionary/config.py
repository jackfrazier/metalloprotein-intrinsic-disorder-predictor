"""
Configuration management for the evolutionary analysis module.

This module handles loading and validating configuration parameters
from the evolutionary.yaml file.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any

import yaml
from omegaconf import OmegaConf


class EvolutionaryConfig:
    """Manages configuration for evolutionary analysis components."""

    _instance = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize configuration from yaml file."""
        if not hasattr(self, "_config"):
            self._load_config()

    def _load_config(self):
        """Load configuration from yaml file."""
        config_path = Path(__file__).parent / "evolutionary.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Convert to OmegaConf for better access patterns
        self._config = OmegaConf.create(raw_config["evolutionary"])

    @property
    def msa(self) -> Dict[str, Any]:
        """Get MSA processing parameters."""
        return OmegaConf.to_container(self._config.msa)

    @property
    def conservation(self) -> Dict[str, Any]:
        """Get conservation analysis parameters."""
        return OmegaConf.to_container(self._config.conservation)

    @property
    def coevolution(self) -> Dict[str, Any]:
        """Get coevolution analysis parameters."""
        return OmegaConf.to_container(self._config.coevolution)

    @property
    def metal_binding(self) -> Dict[str, Any]:
        """Get metal binding analysis parameters."""
        return OmegaConf.to_container(self._config.metal_binding)

    def get_metal_boost_factor(self, metal_type: str) -> float:
        """Get boost factor for specific metal type."""
        return self._config.metal_binding.boost_factors.get(metal_type, 1.0)

    def get_metal_motif_pattern(self, motif_name: str) -> Optional[str]:
        """Get regex pattern for specific metal binding motif."""
        return self._config.metal_binding.motif_patterns.get(motif_name)

    def get_property_group(self, group_name: str) -> str:
        """Get amino acids in specific property group."""
        return self._config.conservation.property_groups.get(group_name, "")

    @classmethod
    def get_instance(cls) -> "EvolutionaryConfig":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
