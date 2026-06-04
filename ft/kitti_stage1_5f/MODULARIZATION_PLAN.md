# kitti_stage1_5f 模块化重组建议

这个文件记录当前 `ft/kitti_stage1_5f` 微调仓库的重组方向。建议等当前实验跑完后再做结构调整，避免中途改动影响 checkpoint、import 路径和复现实验。

## 当前主要问题

1. `tools/train.py` 责任过重  
   目前同时处理 CLI 参数、数据集构造、模型构造、forward dispatch、loss dispatch、DDP、checkpoint、日志和训练循环。`exp` 字符串分支散落在多个位置，后续加实验会继续膨胀。

2. LiDAR/fusion 相关模型文件过大  
   `models/lidar_fusion.py` 接近 1900 行，混合了 VFE、window attention、sorted 3D attention、memory voxel、fusion module 等多种概念。  
   `models/stage1_ssc_bevdetocc_lidar.py` 也超过 1000 行，包含 geometry adapter、LSS depth lift、dense depth loss、memory、warper、完整模型等。

3. 多个 Stage-1 模型变体重复代码  
   `stage1_ssc.py`、`stage1_ssc_mono.py`、`stage1_ssc_mono_lidar.py` 都重复创建 OccAny backbone、lifting、head，并重复处理 backbone eval/freeze 语义。

4. Dataset 变体存在重复逻辑  
   LiDAR dataset 的 raw velodyne 加载、`T_cam_from_velo`、`K_per_frame`、`image_hw`、collate 逻辑可以抽成更稳定的 mixin/helper。

5. 工具脚本重复构造逻辑  
   `train.py`、`eval.py`、`visualize_val_prediction.py` 都有各自的 model/dataset/device/path 处理，容易 drift。

## 推荐目标结构

```text
ft/kitti_stage1_5f/
  experiments/
    __init__.py
    registry.py          # exp -> dataset/model/loss/checkpoint 策略
    specs.py             # ExperimentSpec dataclass

  engine/
    __init__.py
    train_loop.py        # train_one_epoch / eval_one_epoch
    checkpoint.py        # save/resume/backbone_hash
    losses.py            # loss dispatch + depth aux loss
    batch.py             # move_to_device / model_forward dispatch
    optim.py             # lr schedule / optimizer

  datasets/
    __init__.py
    constants.py
    calib.py
    base.py              # Kitti5FrameStage1Dataset
    mono.py              # MonoScene CP 扩展
    lidar.py             # LiDAR mixin / sweep loading
    collate.py

  models/
    __init__.py
    lifting.py
    bevdet3d_local.py
    stage1/
      __init__.py
      base.py            # OccAny backbone + lifting 公共骨架
      light.py
      monoscene.py
      monoscene_lidar.py
      bevdetocc_lidar.py
    fusion/
      __init__.py
      voxel_encoder.py
      window_attention.py
      sorted3d_attention.py
      lidar_image.py
      memory_voxel.py

  heads/
    __init__.py
    light_occ_3d_unet.py
    monoscene_adapter.py
    monoscene_occ_head.py
    monoscene/

  losses/
    __init__.py
    monoscene.py
    depth.py

  visualization/
    __init__.py
    voxels.py
    plots.py
    projection.py

  tools/
    __init__.py
    train.py             # 只 parse args + 调 engine
    eval.py
    preprocess_voxels.py
    sanity_lidar_fusion.py
    visualize_patch_projection.py
    visualize_val_prediction.py
```

## 建议迁移顺序

### 1. 先引入 experiments registry

优先新增 `experiments/specs.py` 和 `experiments/registry.py`，统一管理实验变体：

- `light`
- `monoscene`
- `monoscene_lidar`
- `bevdetocc_lidar`

每个实验注册一个 `ExperimentSpec`，包含：

- `model_cls`
- `dataset_cls`
- `collate_fn`
- `criterion_factory`
- `uses_lidar`
- `uses_monoscene_loss`
- `checkpoint_policy`
- `ddp_find_unused_parameters`
- `syncbn_default`

这样可以优先消除 `train.py`、`eval.py`、`visualize_val_prediction.py` 中重复的 `if args.exp == ...`。

### 2. 再拆 engine

把 `tools/train.py` 中非 CLI 的逻辑迁到：

- `engine/batch.py`: `_move_views_to_device`、`_move_points_to_device`、model forward dispatch
- `engine/losses.py`: MonoScene loss、BEVDet depth loss、dense depth loss dispatch
- `engine/train_loop.py`: `train_one_epoch`、`eval_one_epoch`
- `engine/checkpoint.py`: resume、save、backbone hash、checkpoint payload
- `engine/optim.py`: lr schedule、optimizer 构造

目标是让 `tools/train.py` 变成薄入口，只负责：

1. parse args
2. init distributed/device
3. 调 registry 构造 dataset/model/loss
4. 调 engine 训练

### 3. 抽 Stage-1 公共模型骨架

新增 `models/stage1/base.py`，统一处理：

- `OccAnyRecon5FrameBackbone`
- checkpoint load
- `Stage1LiftingModule`
- backbone freeze
- `train()` 中保持 backbone eval
- backbone/lifting 的公共 forward 部分

各变体只负责自己的额外模块：

- `light.py`: LightOcc3DUNet head
- `monoscene.py`: MonoSceneOccHead
- `monoscene_lidar.py`: LiDAR fusion + MonoSceneOccHead
- `bevdetocc_lidar.py`: BEVDet-OCC/LSS/memory branch

建议所有模型输出统一成 dict，例如：

```python
{
    "ssc_logit": ssc_logit,
    "aux": {...},
}
```

这样训练循环不需要区分 tensor 输出和 dict 输出。

### 4. 拆 dataset helper

建议拆出：

- `datasets/constants.py`: `KITTI_SSC_CLASS_NAMES`、`KITTI_SPLITS`
- `datasets/calib.py`: `_parse_calib`、`_T_cami_from_cam0`、`_static_T_velo_from_cam2`、`_T_cami_from_velo`
- `datasets/base.py`: 原始 `Kitti5FrameStage1Dataset`
- `datasets/mono.py`: `compute_CP_mega_matrix`、`Kitti5FrameStage1MonoDataset`
- `datasets/lidar.py`: raw velodyne 加载和 LiDAR mixin
- `datasets/collate.py`: 所有 collate 函数

LiDAR 相关 dataset 可以共享一个 helper/mixin，避免 `Kitti5FrameStage1LidarDataset` 和 `Kitti5FrameStage1MonoLidarDataset` 复制加载逻辑。

### 5. 最后拆大模型文件

`models/lidar_fusion.py` 和 `models/stage1_ssc_bevdetocc_lidar.py` 风险最大，建议最后拆。

迁移时保留旧文件做兼容 re-export，例如：

```python
# models/lidar_fusion.py
from .fusion.voxel_encoder import VoxelFeatureEncoder
from .fusion.window_attention import WindowedCrossAttnLayer, WindowedSelfAttnLayer
from .fusion.sorted3d_attention import Sorted3DTokenFusionLayer
from .fusion.lidar_image import LidarImageFusionModule
from .fusion.memory_voxel import MemoryVoxel3DFusion
```

这样已有脚本和 checkpoint 逻辑不需要同时大面积改 import。

## 兼容性策略

重组时建议分阶段保留旧 import 路径：

- `losses_monoscene.py` 保留，内部 re-export `losses.monoscene`
- `models/lidar_fusion.py` 保留，内部 re-export `models.fusion.*`
- `models/stage1_ssc*.py` 可先保留，内部 re-export `models.stage1.*`

这样可以降低一次性迁移风险，也方便逐步验证。

## 最小验证命令

每完成一个迁移步骤，先跑：

```bash
python -m compileall ft/kitti_stage1_5f
python -m ft.kitti_stage1_5f.tools.train --help
python -m ft.kitti_stage1_5f.tools.eval --help
```

如果改了可视化相关模块，再跑：

```bash
python -m ft.kitti_stage1_5f.tools.visualize_val_prediction --help
python -m ft.kitti_stage1_5f.tools.visualize_patch_projection --help
```

## 推荐第一步

实验结束后，优先做：

1. 新增 `experiments/specs.py`
2. 新增 `experiments/registry.py`
3. 把 `train.py` 中的 `_build_dataset`、`_collate_fn`、model class 选择、criterion 选择迁到 registry

这一步收益最大，风险最低，也不会马上触碰大模型实现。
