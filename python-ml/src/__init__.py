"""Core package initialization for AR Mirror Python ML Service"""

__version__ = "1.0.0"
__author__ = "AR Mirror Team"

# Make key components easily importable
from .sizing_pipeline import (
    SizingPipeline,
    BodyMeasurements,
    FitResult,
    FitDecision,
)

__all__ = [
    "SizingPipeline",
    "BodyMeasurements",
    "FitResult",
    "FitDecision",
]
