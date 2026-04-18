"""Core vision components for AR Try-On system"""

from .depth_estimator import DepthEstimator
from .frame_synchronizer import FrameSynchronizer
from .gpu_config import GPUConfig

__all__ = [
    'DepthEstimator',
    'FrameSynchronizer',
    'GPUConfig',
]

try:
    from .session_logger import SessionLogger, get_session_logger, BodyMeasurements, FitDecision, FrameMetadata
    __all__ += ['SessionLogger', 'get_session_logger', 'BodyMeasurements', 'FitDecision', 'FrameMetadata']
except ImportError:
    pass

try:
    from .sku_bias_corrector import SKUBiasCorrector, get_sku_corrector
    __all__ += ['SKUBiasCorrector', 'get_sku_corrector']
except ImportError:
    pass

try:
    from .smplx_body_reconstruction import SMPLXBodyReconstructor, SMPLXMigrationStub, smpl_to_smplx_params
    __all__ += ['SMPLXBodyReconstructor', 'SMPLXMigrationStub', 'smpl_to_smplx_params']
except ImportError:
    pass

# Optional imports — only available when their dependencies are installed
try:
    from .semantic_parser import SemanticParser, create_occlusion_aware_composite
    __all__ += ['SemanticParser', 'create_occlusion_aware_composite']
except ImportError:
    pass

try:
    from .body_aware_fitter import BodyAwareGarmentFitter
    __all__ += ['BodyAwareGarmentFitter']
except ImportError:
    pass

try:
    from .live_pose_converter import LivePoseConverter, LiveBodySegmenter
    __all__ += ['LivePoseConverter', 'LiveBodySegmenter']
except ImportError:
    pass

try:
    from .parsing_backends import MediaPipeParsingBackend
    __all__ += ['MediaPipeParsingBackend']
except ImportError:
    pass
