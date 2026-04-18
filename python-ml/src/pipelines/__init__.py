"""Neural network pipelines"""

try:
    from .neural_pipeline import *  # type: ignore[import]
    __all__ = ["NeuralPipeline"]
except ImportError:
    __all__ = []
