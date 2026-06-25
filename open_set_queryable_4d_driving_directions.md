# 自动驾驶中的开集合与图像几何基础模型研究方向整理

## 1. 核心判断

结合 CVPR 2026 自动驾驶、3D/4D 视觉、open-vocabulary、world model、occupancy 和 VLM/VLA 等热点，可以得到一个总体判断：

> 下一阶段自动驾驶感知的机会，不是再做一个更强的 BEV 检测/占据网络，而是构建一个“几何可信、语义开放、时序可查询、能服务规划”的 3D/4D scene representation。

传统路线大致是：

```text
多相机图像 → BEV / voxel → 固定类别检测 / 占据
```

更值得尝试的新路线是：

```text
多相机视频 → 3D/4D 几何基础表征 → open-vocabulary / occupancy / prediction / planning query
```

关键变化包括：

1. 从固定类别感知转向 open-vocabulary / open-set 感知；
2. 从静态 BEV/voxel 转向动态 4D scene representation；
3. 从单任务 head 转向 queryable interface；
4. 从纯视觉语义转向 metric 3D/4D geometry grounding；
5. 从 perception-only 转向能服务 planning / world model / VLA 的场景底座。

---

## 2. 方向一：Open-Vocabulary 3D / 4D Occupancy

### 2.1 问题动机

传统 occupancy 通常预测固定类别，例如：

- car
- truck
- pedestrian
- road
- sidewalk
- barrier
- traffic cone

但真实自动驾驶场景中存在大量长尾和未知物体：

- 施工牌
- 掉落物
- 三轮车
- 动物
- 临时路障
- 奇怪的改装车
- 不规则障碍物

闭集 occupancy 无法很好覆盖这些情况。

因此可以尝试：

> Open-Vocabulary 4D Occupancy from Multi-Camera Videos

### 2.2 基本形式

```text
multi-camera video
   ↓
geometry foundation encoder, e.g. VGGT / D4RT-like model
   ↓
4D scene tokens / voxel tokens / point tokens / Gaussian tokens
   ↓
open-vocabulary occupancy query
```

查询形式可以是：

```text
query("construction cone")
query("fallen object")
query("animal")
query("temporary barrier")
query("unknown obstacle")
```

### 2.3 与 Ov3R / OccAny 的关系

Ov3R 的启发是：

- 不只是重建后贴 CLIP label；
- 而是让 object-level CLIP semantics 进入 3D reconstruction 表示本身；
- 最后通过 2D-3D OVS 做开放词表 3D segmentation。

OccAny 的启发是：

- generalized / unconstrained urban 3D occupancy；
- 支持 monocular、sequential、surround-view；
- 尝试摆脱强 in-domain annotation 和固定 sensor-rig priors。

可以进一步推进到：

> generalized occupancy + open-vocabulary semantics + temporal 4D consistency

### 2.4 可能论文题目

- **OpenOcc4D: Open-Vocabulary 4D Occupancy from Multi-Camera Videos**
- **OV-DriveOcc: Open-Vocabulary 3D Occupancy for Autonomous Driving via Geometry Foundation Models**

### 2.5 关键创新点

1. **几何先对齐**  
   使用图像几何模型获得 depth / pointmap / correspondence / camera-aware features。

2. **语义再提升**  
   使用 Grounded-SAM、SAM、CLIP、DINO 等 2D foundation model 产生 mask-level / region-level semantics。

3. **多视角一致性过滤**  
   只有跨视角、跨时间一致的 open-vocabulary 语义才写入 3D / 4D 表示。

4. **segment-level 分类，而不是 voxel-level 分类**  
   先形成 3D region / object / stuff segment，再和文本 embedding 匹配。

### 2.6 优缺点

优点：

- 和已有 occupancy 经验高度匹配；
- 实验相对可控；
- 比单纯闭集 occupancy 更有新意；
- 可以自然连接 VLM / planning / OOD。

缺点：

- open-vocabulary 自动驾驶评测较难；
- 可能需要自己设计长尾类别、OOD 物体、文本查询或 pseudo-label benchmark。

---

## 3. 方向二：Geometry-Grounded Driving VLM / VLA

### 3.1 问题动机

现有驾驶 VLM 往往能描述图像，但不真正理解 metric 3D geometry。

例如模型可能知道：

```text
前方有车
```

但不知道：

```text
车距离 ego 多远？
是否在本车道？
速度如何？
是否遮挡行人？
是否会与 ego trajectory 冲突？
```

因此，Driving VLM / VLA 的核心瓶颈不是语言能力，而是 3D/4D geometry grounding。

### 3.2 基本思路

```text
multi-camera video
   ↓
4D geometry encoder
   ↓
scene memory / 4D tokens
   ↓
VLM / VLA query interface
   ↓
risk reasoning / motion prediction / planning explanation
```

不是让 VLM 直接看 6 张图，而是让它访问一个几何化的 scene memory。

### 3.3 可以尝试的问题

```text
Q: Which object is most likely to conflict with ego vehicle in 3 seconds?
Q: Is there a hidden pedestrian risk behind the parked vehicle?
Q: Which region is drivable but unsafe?
Q: What object should the planner yield to?
Q: Explain the risk using 3D positions and future motion.
```

这类任务比普通 caption / QA 更有价值，因为它要求模型真正利用 3D / 4D 几何。

### 3.4 可能论文题目

- **GeoDriveGPT: Geometry-Grounded Driving VLM with 4D Scene Memory**
- **4D Scene Memory for Vision-Language-Action Autonomous Driving**

### 3.5 关键创新点

1. 不改变大 VLM 主体，只插入一个 geometry adapter；
2. 4D scene tokens 作为外部 memory，被 VLM cross-attend；
3. query 不只是语言，也可以包括 ego trajectory、future timestamp、3D point、object proposal；
4. 评测不只看 QA accuracy，还看 risk perception、motion prediction、planning score。

### 3.6 风险

这个方向潜力很大，但系统复杂度也高：

- 需要驾驶 VLM 数据；
- 需要构造 QA / reasoning / planning 数据；
- 需要规划相关评测；
- 容易变成系统工程。

更稳的做法是：

> 不从零训练 VLA，而是做一个可插拔的 geometry-grounding module。

---

## 4. 方向三：Queryable 4D Driving Scene Representation

### 4.1 为什么重要

D4RT / VGGT / DUSt3R / MASt3R / Ov3R 这类图像几何模型说明，未来 3D 视觉可能不再是传统 pipeline：

```text
估深度 → 估位姿 → 重建 → 下游任务
```

而是走向：

```text
multi-view / video encoder → unified 3D/4D representation → queryable output
```

自动驾驶中可以进一步设计：

> 一个多相机视频 encoder，输出可查询的 4D driving scene representation。

### 4.2 Queryable 接口

它不一定直接输出 dense BEV，也不一定直接输出完整 voxel，而是支持各种 query：

```text
query(x, y, z, t) → occupancy / semantic / velocity / uncertainty
query(object_id, t) → future position / trajectory / visibility
query(region, t) → drivable / risky / unknown
query(text, t) → open-vocabulary object / region
query(ego_plan) → collision risk / comfort / feasibility
```

### 4.3 相比单纯 occupancy 的优势

Occupancy 只是 4D scene representation 的一种 readout。

统一 queryable representation 可以服务：

| Query 类型 | 输出 |
|---|---|
| 3D point query | occupied / free / unknown |
| semantic query | road / vehicle / pedestrian / barrier / open-vocab text |
| motion query | velocity / flow / future position |
| planning query | collision probability / drivable cost |
| language query | text-conditioned object / risk / scene explanation |

### 4.4 可能论文题目

- **Queryable 4D Driving Scene Representation from Multi-Camera Videos**

### 4.5 最小可行版本

不需要一开始覆盖所有任务，可以分阶段：

第一阶段：

```text
multi-camera video → 4D scene tokens → occupancy / depth query
```

第二阶段：

```text
+ open-vocabulary semantic query
```

第三阶段：

```text
+ future occupancy / planning cost query
```

### 4.6 评价

这是最推荐认真考虑的主线。

原因：

- 可以承接已有 occupancy / multi-camera 背景；
- 又能对接 D4RT / VGGT / open-vocabulary / world model / VLA 热点；
- 论文叙事空间比单纯 occupancy 更大；
- 可从小任务开始，逐渐扩展到博士阶段长期方向。

---

## 5. 方向四：2D Foundation Model Supervised 3D Driving Representation

### 5.1 问题动机

3D occupancy label 昂贵，跨数据集泛化困难。可以尝试不完全依赖 dense 3D label，而是用 2D foundation model + 几何一致性监督 3D / 4D 表示。

可以利用：

- SAM / SAM2：object masks；
- Grounded-SAM：文本条件 object masks；
- CLIP / SigLIP：open-vocabulary semantic embedding；
- DINO：dense visual features / objectness；
- Depth Anything / VGGT：dense depth / pointmap / geometry features；
- 多相机标定：2D → 3D lifting；
- 时序 ego pose：跨帧一致性。

### 5.2 基本形式

```text
2D masks / 2D open-vocab labels
   + multi-view geometry
   + temporal consistency
   ↓
pseudo 3D / 4D semantic regions
   ↓
train 3D scene representation
```

### 5.3 与 OccAny 的区别

OccAny 更强调：

- generalized urban occupancy；
- unconstrained scenes；
- metric occupancy；
- segmentation forcing；
- novel-view rendering。

可以进一步强调：

1. open-vocabulary 语义；
2. 多帧 4D 一致性；
3. segment-level 3D object / stuff representation；
4. 不需要 dense occupancy label；
5. 可被 VLM / planning 使用。

### 5.4 可能论文题目

- **Learning Open-Vocabulary 4D Driving Scenes from 2D Foundation Models**
- **From 2D Masks to 4D Driving Occupancy: Weakly-Supervised Open-Set Scene Representation**

### 5.5 优点

这个方向适合算力有限的学校场景：

- 可以依赖 frozen foundation models；
- 可以构造 pseudo labels；
- 不一定需要从零训练巨大模型；
- 容易和已有 occupancy 框架结合。

---

## 6. 方向五：Open-Set Risk Perception

### 6.1 问题动机

很多 open-vocabulary / open-set 工作关注：

```text
这个东西叫什么？
```

但对自动驾驶来说，更重要的是：

```text
它是否影响驾驶？
```

模型不一定需要知道一个未知物体到底是纸箱、石头、轮胎还是塑料袋，但必须知道：

```text
它是否占据可行驶空间？
是否有碰撞风险？
是否会移动？
是否需要减速或绕行？
```

因此可以做：

> Open-Set Risk Occupancy / Open-Set Drivable Hazard Detection

### 6.2 输出定义

输出可以不是固定类别，而是：

```text
free / occupied / unknown
static risk / dynamic risk
drivable but unsafe
unknown obstacle
text-aligned semantic hint
```

模型可以表达：

```text
我不确定它是什么，但它是一个占据 ego lane 的未知障碍物。
```

这比强行分类成某个具体类别更符合自动驾驶安全需求。

### 6.3 可能论文题目

- **Open-Set Risk-Aware 3D Occupancy for Autonomous Driving**
- **Unknown but Unsafe: Open-Set 4D Risk Perception from Multi-Camera Videos**

### 6.4 关键实验

| 场景 | 目标 |
|---|---|
| OOD object pasted into driving scenes | 测未知障碍物发现 |
| rare object categories | 测长尾泛化 |
| object on drivable area vs sidewalk | 测风险理解 |
| occlusion / partial visibility | 测不确定性 |
| future ego trajectory crossing object | 测 planning relevance |

### 6.5 评价

这个方向很有自动驾驶特色，也容易做出差异化。缺点是 benchmark 可能需要自己设计，但这也可能成为论文贡献。

---

## 7. 方向六：Gaussian / Point / Token-Based Driving Scene Representation

### 7.1 问题动机

Dense BEV / voxel 表示计算量大，而且不够灵活。现在很多 3D 视觉和 world model 工作开始从 dense grid 转向：

- dynamic 3D Gaussian tokens；
- point tokens；
- object-centric tokens；
- region tokens；
- sparse 4D scene tokens。

### 7.2 Token 内容

每个 token 可以存储：

```text
position
scale / extent
semantic embedding
motion vector
visibility
uncertainty
open-vocabulary descriptor
```

### 7.3 为什么适合 open-vocabulary

CLIP / DINO / SAM 这类模型天然更偏 region / object，而不是单个 voxel。因此，open-vocabulary 语义更适合落在：

```text
object token / region token / Gaussian token
```

而不是逐 voxel 分类。

### 7.4 可能论文题目

- **Sparse Open-Vocabulary 4D Scene Tokens for Autonomous Driving**
- **Semantic 4D Gaussians for Open-Set Driving Scene Understanding**

### 7.5 优势

1. 比 dense voxel 更省；
2. 更适合 instance / object / region；
3. 更容易接语言 query；
4. 可以和 world model / rendering 接上；
5. 可用于 future rollout。

### 7.6 风险

1. metric accuracy 不一定比 voxel 稳；
2. 如果不落到 occupancy / detection / planning，容易被认为只是 fancy representation。

因此如果做这个方向，最好绑定明确下游任务，例如：

- open-set occupancy；
- future occupancy；
- planning cost；
- risk perception。

---

## 8. 方向七：Foundation Geometry Model Adaptation for Driving

### 8.1 问题动机

VGGT / D4RT / DUSt3R / MASt3R / Ov3R 等通用图像几何模型很强，但直接迁移到自动驾驶会遇到问题：

- 多数在普通视频 / 室内 / object-centric 数据上训练；
- 自动驾驶是大尺度户外场景；
- 多相机 rig 固定但视角跨度大；
- 动态物体更多；
- 远距离目标小；
- 需要 metric scale；
- 需要与 ego pose / calibration 对齐；
- 不允许随意漂移。

因此可以研究：

> Driving Adapter for Visual Geometry Foundation Models

不是重新训练大模型，而是加轻量 adapter，让通用几何模型适应 driving。

### 8.2 可能论文题目

- **Adapting Visual Geometry Foundation Models for Multi-Camera Driving Perception**
- **DriveGeo: Metric and Temporal Adaptation of Visual Geometry Foundation Models for Autonomous Driving**

### 8.3 可以解决的问题

1. scale alignment；
2. camera rig calibration conditioning；
3. dynamic object separation；
4. temporal consistency；
5. metric depth / pointmap refinement；
6. downstream occupancy / detection readout。

### 8.4 评价

这个方向可落地性较强，适合从已有开源模型出发做增量创新。

---

## 9. 推荐优先级

### 第一推荐：Queryable 4D Driving Scene Representation

这是最值得认真考虑的主线。

它综合了：

- D4RT / VGGT 式图像几何模型；
- occupancy；
- open-vocabulary；
- world model；
- planning；
- VLM / VLA。

推荐题目：

> Queryable 4D Driving Scene Representation from Multi-Camera Videos

最小实验：

```text
depth / occupancy / semantic query
```

后续扩展：

```text
future occupancy / planning query / language query
```

---

### 第二推荐：Open-Vocabulary 4D Occupancy

这是最贴合已有 occupancy 经验的方向。

推荐题目：

> Open-Vocabulary 4D Occupancy from Multi-Camera Videos

核心创新：

```text
2D foundation semantics
+ geometry foundation model
+ temporal consistency
+ 3D/4D segment-level text alignment
```

这个方向比单纯 occupancy 更有新意，也比完整 VLA 更可控。

---

### 第三推荐：Open-Set Risk-Aware Occupancy

这是最有自动驾驶安全特色的方向。

推荐题目：

> Unknown but Unsafe: Open-Set Risk-Aware 3D Occupancy for Autonomous Driving

核心思想：

```text
未知物体也要知道是否危险
```

该方向能够连接 planning、安全、OOD 和 open-set perception，容易讲出自动驾驶必要性。

---

## 10. 一个较强的最终方案

### 10.1 题目方向

> Open-Set Queryable 4D Occupancy for Autonomous Driving

### 10.2 核心问题

现有 occupancy 方法通常依赖：

- 固定类别；
- 固定数据集；
- 固定 sensor rig；
- dense 3D occupancy label；
- 闭集语义定义。

因此难以处理：

- 长尾未知物体；
- 开放语义查询；
- 动态风险判断；
- 跨数据集泛化；
- VLM / planning / world model 调用。

### 10.3 方法框架

```text
Multi-camera video
   ↓
Visual geometry foundation encoder
   - depth / pointmap / correspondence / temporal geometry
   ↓
2D foundation semantic teacher
   - SAM / Grounded-SAM / CLIP / DINO
   ↓
4D scene token construction
   - point / voxel / region / Gaussian token
   ↓
Open-set query decoder
   - occupancy query
   - text query
   - unknown-risk query
   - future occupancy query
```

### 10.4 输出

```text
occupied / free / unknown
closed-set semantic occupancy
open-vocabulary region semantics
unknown obstacle score
future occupancy / risk map
```

### 10.5 监督信号

可以混合使用：

1. nuScenes / Occ3D / OpenOccupancy 的 occupancy label；
2. 多相机 depth / LiDAR projection；
3. 2D segmentation masks；
4. CLIP text-image alignment；
5. temporal consistency；
6. ego trajectory / planning collision signal。

### 10.6 论文卖点

这篇论文可以讲成：

> 我们不是只做更高 mIoU 的 occupancy，而是让自动驾驶场景表示同时具备 metric geometry、open-vocabulary semantics、temporal 4D consistency、unknown risk awareness。

相比传统方法，它的差异在于：

| 传统 occupancy | Open-Set Queryable 4D Occupancy |
|---|---|
| 固定类别 | 开放词表 / 未知风险 |
| 静态 3D | 动态 4D |
| dense voxel output | queryable scene representation |
| perception-only | 可服务 VLM / planning / world model |
| mIoU 导向 | geometry + semantics + risk + future |

---

## 11. 实施建议

### 11.1 不建议一开始做完整 VLA / world model

完整 driving VLA / world model 很热，但复杂度高：

- 数据构造困难；
- 训练成本大；
- 评测复杂；
- 容易变成系统工程；
- 难以在短期内形成稳定结果。

### 11.2 更稳的策略

先做一个强的：

> 4D scene representation / open-set occupancy

并证明它可以作为：

- VLM 的 geometric memory；
- planning 的 risk map；
- world model 的 scene token；
- occupancy / detection / prediction 的统一底座。

### 11.3 推荐定位

论文不要一开始声称“解决自动驾驶大模型”，而是定位为：

```text
我们提供一个 open-set, queryable, geometry-grounded 4D driving representation。
它可以作为 downstream driving VLM / planning / world model 的基础接口。
```

这个定位更稳，也更适合作为博士阶段长期研究主线。

---

## 12. 简短结论

最值得尝试的方向可以概括为：

> 构建一个 open-set、queryable、geometry-grounded 的 4D driving scene representation。

它应当同时具备：

1. **metric geometry**：知道物体和空间在哪里；
2. **open-vocabulary semantics**：不局限于固定类别；
3. **temporal consistency**：不是单帧，而是动态 4D；
4. **unknown risk awareness**：不知道是什么，也要知道是否危险；
5. **queryable interface**：可被 occupancy、VLM、planning、world model 调用。

最终建议优先探索：

```text
Open-Set Queryable 4D Occupancy for Autonomous Driving
```

这是一个兼具论文新意、自动驾驶价值、已有基础可复用性和长期扩展潜力的方向。
