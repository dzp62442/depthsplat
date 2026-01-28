# PandaSet 数据集适配方案（DepthSplat / comp_svfgs）

> 本文档仅描述**计划实施方案**与对照依据，不进行代码修改。后续实现将严格遵循本文方案。

## 1. 目标与约束
- **目标**：在 DepthSplat（comp_svfgs 分支）中接入 PandaSet 数据集，进行**零样本泛化**实验；训练/验证/测试配置与 Omni-Scene 保持一致。
- **数据位置**：PandaSet 已用 SVF-GS 的 `scripts/preprocess_pandaset.py` 处理完成，通过软链接放在本项目 `datasets/PandaSet` 路径下。
- **动态物体掩码**：不需要生成/加载动态掩码，全部视为静态；如需字段占位，统一使用全 1 mask。
- **PCC 指标**：如果需要计算，直接使用 **Metric3D-v2 的尺度深度**（无需 DepthAnything 相对深度）。
- **坐标系**：本项目**不做** `flip_yz`；仅 SVF-GS 需要。

## 2. 现有 Omni-Scene（nuScenes 风格）实现回顾（对照基准）
参考 `docs/OmniScene数据集实验文档.md` 与代码实现：
- **配置入口**：`config/dataset/omniscene.yaml` + `config/experiment/omniscene_112x200.yaml` / `omniscene_224x400.yaml`。
- **数据集实现**：`src/dataset/dataset_omniscene.py`。
  - 读取 `bins_*_3.2m.json` 与 `bin_infos_3.2m/*.pkl`。
  - 输入为 6 路环视 key-frame，输出为同 bin 中其他帧 + 输入帧拼接。
- **图像/内参/掩码加载**：`src/dataset/utils_omniscene.py::load_conditions`。
  - resize 同步缩放内参，并**归一化内参（按宽高归一化）**。
  - 输出掩码来自 `*_mask_small`；输入掩码为全 1。
- **PCC 触发逻辑**：`src/model/model_wrapper.py` 中，当 `target` 里存在 `rel_depth` 时会渲染深度并计算 `pcc`。
- **Patch shim**：`src/dataset/shims/patch_shim.py` 会同步裁剪 `image/masks/rel_depth`。

PandaSet 的接入将**复用上述框架**，仅调整**数据路径与组织方式**。

## 3. PandaSet 处理后数据组织（来自 SVF-GS）
SVF-GS 文档与脚本：
- 文档：`~/Projects/SVF-GS/docs/PandaSet 数据集适配方案.md`
- 预处理脚本：`~/Projects/SVF-GS/scripts/preprocess_pandaset.py`
- 数据加载：`~/Projects/SVF-GS/data/pandaset_dataset.py`

处理后的数据格式（已完成）：
```
datasets/PandaSet/
  raw/                       # 原始 PandaSet（不改动）
  processed/
    bins_train.json
    bins_test.json
    bin_infos/
      pandaset_<seq>_<frame>.pkl
    params/
      <seq>/<camera>/<frame>.json     # camera_intrinsic
    dptm/
      <seq>/<camera>/<frame>_dpt.npy  # Metric3D-v2 深度
      <seq>/<camera>/<frame>_conf.npy
```
说明：
- `bin_infos/*.pkl` 内含每个 bin 的 6 路相机数据与 `sensor2lidar_*` 信息（与 Omni-Scene 风格一致）。
- `params/` 与 `dptm/` 的组织方式与 SVF-GS 保持一致，便于复用数据。

## 4. 计划实现方案（不改代码，先定方案）

### 4.1 新增配置（与 Omni-Scene 保持一致）
1) **新增数据集配置**：`config/dataset/pandaset.yaml`
- 结构以 `config/dataset/omniscene.yaml` 为模板。
- `name: pandaset`；`roots: [datasets/PandaSet]`。
- `image_shape` 等参数与 Omni-Scene 保持一致（224x400 默认，支持 112x200）。
- `defaults.view_sampler` 依旧用 `all`。

2) **新增实验配置**：
- `config/experiment/pandaset_112x200.yaml`
- `config/experiment/pandaset_224x400.yaml`
- 内容基本拷贝 Omni-Scene 实验配置：
  - `override /dataset: pandaset`
  - `override /model/encoder: depthsplat`
  - `override /loss: [mse, lpips]`
  - `trainer.max_steps`、`batch_size`、`lpips` 权重等保持一致。
- `dataset.roots` 指向 `datasets/PandaSet`。

> 说明：虽然 Omni-Scene 训练命令常用 `train.use_dynamic_mask=true`，但 PandaSet 不需要动态掩码；我们会在数据集中输出全 1 mask，从而即便开启也不影响结果。

### 4.2 新增数据集实现
1) **新增数据集类**：`src/dataset/dataset_pandaset.py`
- 新增 `DatasetPandaSetCfg`（字段结构对齐 `DatasetOmniSceneCfg`）。
- `DatasetPandaSet` 逻辑对齐 Omni-Scene，**但输出仅包含 6 张输入视图（输入=输出）**。

2) **数据读取逻辑**（对照 `src/dataset/dataset_omniscene.py`）：
- **仅支持 train/val/test 三划分**，并与 SVF-GS 的数据加载逻辑保持一致：
  - `train` → `processed/bins_train.json` 全量。
  - `val` → `processed/bins_test.json` 按 SVF-GS 的 `val` 抽样策略均匀选取（默认 10 个）。
  - `test` → `processed/bins_test.json` 全量。
- `__getitem__`：
  - 读取 `processed/bin_infos/<bin>.pkl`。
  - 按相机顺序（与 Omni-Scene 一致）：
    `CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT`。
  - 对每路相机仅取 index=0 的 key-frame（与 SVF-GS 一致）。
  - 输入/输出均使用这 6 张图像。

3) **相机位姿与内参**
- 新增 `src/dataset/utils_pandaset.py`（或在 `dataset_pandaset.py` 内部实现）提供：
  - `load_info(info)`：
    - `c2w = sensor2lidar_transform` **直接使用，不做 flip_yz**。
    - `w2c` 计算与 `utils_omniscene.py` 保持一致。
  - `load_conditions(img_paths, reso, processed_root, load_rel_depth)`：
    - 从 `processed/params/<seq>/<cam>/<frame>.json` 读取内参。
    - resize 时同步缩放内参；并按 DepthSplat 约定进行**归一化**：
      - `ck[0, :] /= W`, `ck[1, :] /= H`。
    - 加载 Metric3D 深度 `*_dpt.npy`（可忽略 `_conf.npy`）。
    - mask 统一为全 1（静态场景）。

4) **返回数据结构**（与 Omni-Scene 完全一致）：
- `context` 与 `target` 均包含：
  - `extrinsics`, `intrinsics`, `image`, `near`, `far`, `index`。
  - `masks`：全 1。
  - `rel_depth`：仅在 `stage=test` 且需要 PCC 时加载（值为 Metric3D 深度）。

### 4.3 注册数据集
- 在 `src/dataset/__init__.py` 中加入：
  - `"pandaset": DatasetPandaSet`
- 将 `DatasetCfg` union 扩展到 `DatasetPandaSetCfg`。

### 4.4 PCC 指标处理（按需求简化）
- DepthSplat 现有逻辑：`target` 中存在 `rel_depth` 则渲染深度并计算 `pcc`。
- PandaSet 方案：
  - **直接用 Metric3D-v2 尺度深度**作为 `rel_depth` 填入。
  - 不再引入 DepthAnything 相对深度。
  - 因为 PCC 不需要报告，可默认关闭或仅在需要时开启 `compute_scores`。

### 4.5 动态掩码
- PandaSet 无动态 mask 文件；`load_conditions` 直接构造全 1 mask。
- 即使训练命令中误开 `train.use_dynamic_mask=true`，也不会影响训练。

## 5. 配置对齐说明（训练/测试节奏）
与 Omni-Scene 统一：
- `trainer.max_steps`、`data_loader.batch_size`、`loss` 配置按 Omni-Scene 实验文件保持一致。
- 推理/评估与 Omni-Scene 保持一致（同样由 `test.compute_scores`、`trainer.val_check_interval` 等控制）。

## 6. 验证与自检清单（实施后）
- **路径与索引**：随机读取 `processed/bin_infos/*.pkl`，6 路相机路径存在。
- **内参对齐**：resize 后内参正确缩放并完成归一化。
- **输出结构**：`context/target` 形状与 Omni-Scene 一致：
  - `image`: `[V, 3, H, W]`；`intrinsics`: `[V, 3, 3]`。
- **PCC**：若启用，确认 `target.rel_depth` 维度匹配 `output.depth`。
- **无 flip_yz**：检查 `load_info` 中未引入轴翻转。

---

## 7. 待你审阅的关键点
- PandaSet 数据组织（`datasets/PandaSet` 结构）是否与当前软链接一致？
- PCC 是否完全不需要（若完全不需要，可默认不加载 `rel_depth`）？

审阅通过后，我将按以上方案实现代码接入。
