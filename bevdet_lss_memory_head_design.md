# BEVDet LSS Memory Head Design Notes

Date: 2026-06-01

## Goal

Replace the current MonoScene occupancy head path with a BEVDet-style LSS and
3D BEV encoder head, while keeping the OccAny reconstruction backbone frozen.

The old `checkpoint-19.pth` path is not reused. Only OccAny reconstruction
weights are loaded; all newly added fusion, LSS, memory, temporal aggregation,
and BEVDet occupancy head modules are randomly initialized.

## Confirmed Pipeline

1. Run frozen OccAny on the 5-frame KITTI input.
   - Actual image resolution: `160x512`.
   - Patch size: `16`.
   - Token grid: `10x32`.
   - OccAny output feature: `t_rec` with shape `(B, N, 10, 32, 768)`.

2. Perform 2D-space cross-attention between image tokens and projected voxel
   features.
   - Query side: per-frame 2D `t_rec` image tokens.
   - Key/value side: LiDAR voxel features projected into the 2D patch grid.
   - Output remains per-frame 2D tokens: `(B, N, 10, 32, 768)`.

3. Project fused 2D tokens into BEVDet LSS input channels.
   - Use `1x1 Conv2d` from `768 -> 256`.
   - Per-frame 2D feature shape becomes `(B, N, 256, 10, 32)`.

4. Run LSS / depth distribution / view transformer.
   - Use KITTI current target grid, not the nuScenes grid from the BEVDet config.
   - Target occupancy grid remains `(256, 256, 32)` in target-frame KITTI
     velodyne coordinates.
   - Stereo cost volume is disabled or left empty.
   - Sparse LiDAR depth supervision will be added later after generating KITTI
     LiDAR-projected `gt_depth`.

5. Build per-frame memory voxel features.
   - Each frame's memory voxel volume is built in that frame's own velodyne
     coordinate system.
   - This memory branch is randomly initialized.
   - Do not reuse the previous `checkpoint-19.pth` NATTEN / memory weights.

6. Fuse per-frame LSS feature with same-frame memory voxel feature using NATTEN
   cross-attention.
   - Query: per-frame LSS 3D feature.
   - Key/value: memory voxel feature in the same frame coordinate system.
   - Output is an enhanced LSS feature.
   - Concatenate enhanced LSS feature and memory feature:
     `concat[LSS_feature, memory_feature]`.

7. Warp all per-frame 3D features to the target frame.
   - Use the existing multi-frame KITTI camera/world calibration path.
   - After warping, perform temporal aggregation in the target frame.
   - Confirmed aggregation strategy: concatenate the 5 warped frame features
     along channels, then use a `Conv3d` projection to the BEV encoder input
     width.

8. Run BEVDet-style occupancy head.
   - Use BEVDet `LSSFPN3D + final_conv + predicter`.
   - Output 20 KITTI semantic occupancy classes.
   - Externally expose logits in the current KITTI training/eval format:
     `ssc_logit: (B, 20, 256, 256, 32)`.

## Resolved Questions

### 1. Should `checkpoint-19.pth` be reused?

No. It is completely unused. Only the OccAny reconstruction backbone is loaded
and frozen. All other modules are randomly initialized.

### 2. What input resolution should be used?

Use the current actual configuration:

- Image resolution: `160x512`.
- Patch/token grid: `10x32`.

Do not switch to `512x512 / 32x32`.

### 3. Which occupancy grid should LSS use?

Use the current KITTI target grid:

- Grid size: `(256, 256, 32)`.
- Coordinate system: target-frame KITTI velodyne frame.

Do not use the nuScenes BEVDet grid from
`bevdet-occ-r50-4d-stereo-24e.py`.

### 4. Should BEVDet stereo cost volume be used?

No. Disable it or pass an empty stereo path. The model should use monocular
depth distribution plus LSS for now.

### 5. How should 768-channel OccAny tokens connect to LSS?

Use a `1x1 Conv2d` projection from `768` channels to `256` channels before
the LSS view transformer.

### 6. Should old NATTEN weights be reused?

No. The NATTEN/memory modules are randomly initialized. Only OccAny is frozen.

### 7. How should per-frame temporal features be used?

For each frame:

1. Run LSS in that frame's coordinate system.
2. Build memory voxel features in that frame's coordinate system.
3. Fuse LSS and memory with NATTEN cross-attention.
4. Concatenate LSS and memory features.
5. Warp the result to the target frame.

After all frames are in the target frame, concatenate across time and use a
`Conv3d` projection for temporal aggregation.

### 8. What final head should be used?

Use BEVDet's `LSSFPN3D + final_conv + predicter`, modified for 20 KITTI
classes and wrapped so the output matches the existing KITTI loss/eval
interface.

### 9. Should depth supervision be included now?

Not yet. The implementation should leave room for sparse KITTI LiDAR-projected
`gt_depth`, which will be generated later.

## Implementation Constraints

- Do not modify the frozen OccAny reconstruction backbone.
- Do not load or depend on `checkpoint-19.pth`.
- Keep all non-OccAny modules trainable and randomly initialized.
- Keep the external training/evaluation output key compatible with the current
  KITTI pipeline: `ssc_logit`.
- Preserve the current KITTI target occupancy layout expected by the loss and
  evaluation code.

