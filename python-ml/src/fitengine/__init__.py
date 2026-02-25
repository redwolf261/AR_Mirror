"""
FitEngine — Men's Formal Wear Size Engine
Phase 1: heuristic MVP  |  Phase 2: DualViewRegressor  |  Phase 3: ONNX/WASM SaaS
"""

__version__ = "0.1.0"
__all__ = [
    "FitEnginePipeline",
    "STARBodyModel",
    "PoseDetector",
    "BodySegmentor",
    "SizeChart",
    "AlignmentGuide",
]

try:
    from fitengine.pipeline   import FitEnginePipeline
except Exception:  # pragma: no cover
    FitEnginePipeline = None  # type: ignore[assignment,misc]

try:
    from fitengine.body_model import STARBodyModel
except Exception:  # pragma: no cover
    STARBodyModel = None  # type: ignore[assignment,misc]

try:
    from fitengine.detector   import PoseDetector
except Exception:  # pragma: no cover
    PoseDetector = None  # type: ignore[assignment,misc]

try:
    from fitengine.segmentor  import BodySegmentor
except Exception:  # pragma: no cover
    BodySegmentor = None  # type: ignore[assignment,misc]

try:
    from fitengine.size_chart import SizeChart
except Exception:  # pragma: no cover
    SizeChart = None  # type: ignore[assignment,misc]

try:
    from fitengine.alignment  import AlignmentGuide
except Exception:  # pragma: no cover
    AlignmentGuide = None  # type: ignore[assignment,misc]
