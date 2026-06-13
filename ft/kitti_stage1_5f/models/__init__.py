"""Model components for kitti_stage1_5f experiments."""

from .lifting import OccAnyRecon5FrameBackbone, Stage1LiftingModule
from .bevdet3d_local import BEVDetOcc3DHead, CustomResNet3D, LSSFPN3D
from .stage1_ssc import Stage1SSCModel
from .stage1_ssc_mono import Stage1SSCMonoModel
from .lidar_fusion import (
    LidarImageFusionModule,
    Sorted3DTokenFusionLayer,
    VoxelFeatureEncoder,
    WindowedCrossAttnLayer,
    WindowedSelfAttnLayer,
)
from .stage1_ssc_mono_lidar import Stage1SSCMonoLidarModel
from .stage1_ssc_bevdetocc_lidar import Stage1SSCBEVDetOccLidarModel
from .stage1_ssc_bevdetocc_lidar_dense_depth import (
    SingleScaleDPTDepthHead,
    Stage1SSCBEVDetOccLidarDenseDepthModel,
    dense_metric_depth_loss,
)
from .stage1_ssc_bevdetocc_lidar_pointmap import (
    PostFusionPointmapHead,
    Stage1SSCBEVDetOccLidarPointmapModel,
    pointmap_reconstruction_loss,
)
from .stage1_pointmap_ablation import (
    Stage1PointmapOriginalModel,
    Stage1PointmapPostFusionOnlyModel,
    Stage1SSCBEVDetOccLidarPointmapDenseDepthModel,
)

__all__ = [
    "OccAnyRecon5FrameBackbone",
    "Stage1LiftingModule",
    "BEVDetOcc3DHead",
    "CustomResNet3D",
    "LSSFPN3D",
    "Stage1SSCModel",
    "Stage1SSCMonoModel",
    "Stage1SSCMonoLidarModel",
    "Stage1SSCBEVDetOccLidarModel",
    "Stage1SSCBEVDetOccLidarDenseDepthModel",
    "Stage1SSCBEVDetOccLidarPointmapModel",
    "Stage1PointmapOriginalModel",
    "Stage1PointmapPostFusionOnlyModel",
    "Stage1SSCBEVDetOccLidarPointmapDenseDepthModel",
    "SingleScaleDPTDepthHead",
    "PostFusionPointmapHead",
    "dense_metric_depth_loss",
    "pointmap_reconstruction_loss",
    "LidarImageFusionModule",
    "Sorted3DTokenFusionLayer",
    "VoxelFeatureEncoder",
    "WindowedCrossAttnLayer",
    "WindowedSelfAttnLayer",
]
