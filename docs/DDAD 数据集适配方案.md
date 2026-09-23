# DDAD 数据集适配方案（DepthSplat / comp_svfgs）

更新日期：2026-09-23。状态：**DDAD 并列 Dataset、配置和十八视角评估已实现；已完成全量 CPU 加载及小样本 GPU 验收，未启动本项目全量 GPU 实验。**

SVF-GS 的零样本泛化实验与协议文档已更新完成。本项目读取同一份已发布 DDAD 十八视角数据，当前 `processed/manifest_test.json` 为 `complete=true`、`num_bins=324`。前馈方法正式口径为 `final/all_18` 的 PSNR/SSIM/LPIPS。

本方案与 [PandaSet 数据集适配方案](<PandaSet 数据集适配方案.md>) 并列，沿用其原有适配方式，不通过抽取共享 Dataset 基类改造 PandaSet。

## 1. 目标与架构边界

使用 OmniScene 自训权重或官方完整 DepthSplat GS 权重，从 DDAD 中央帧六路 RGB 和相机参数生成高斯，评估前后时刻十二路新视角及中央六路重建。全程不在 DDAD 训练或微调。

按 PandaSet 的现有文件分工新增：

| 文件 | 职责 |
| --- | --- |
| `src/dataset/dataset_ddad.py` | `DatasetDDADCfg`、`DatasetDDAD`，清单、子集和 6→18 组装 |
| `src/dataset/utils_ddad.py` | 路径、图像、相机内参/位姿及参考深度加载 |
| `config/dataset/ddad.yaml` | DDAD 根目录、处理目录与测试子集配置 |
| `config/experiment/ddad_112x200.yaml` | 与 PandaSet 并列的 112×200 配置 |
| `config/experiment/ddad_224x400.yaml` | 与 PandaSet 并列的 224×400 配置 |

在 `src/dataset/__init__.py` 增加 `"ddad": DatasetDDAD` 并扩展 `DatasetCfg` union，继续使用原 DataModule 和 `context/target/scene` 接口。分组指标由原 `ModelWrapper.test_step/on_test_end` 统一处理。

DDAD 入口从升级后的 PandaSet 结构平行实现，但不继承 `DatasetPandaSet`，不新增 `TemporalDataset` 等中间层，也不要求 PandaSet 为复用代码而迁移。保持少量并列加载代码符合本轮范围，不引入跨仓库 Python 依赖、DGP 或 Metric3D 模型依赖。

## 2. 数据来源与正式索引

截至本次检查，`datasets/DDAD` 已链接到 `/home/B_UserData/dongzhipeng/Datasets/DDAD`。默认读取 `datasets/DDAD/processed`；数据生成与维护仍由 SVF-GS 负责，本项目不再预处理 raw。

项目 test 来自 **DDAD 官方 validation**：原始 50 个场景、3950 个环视帧，发布清单保留 324 bins、49 个有效场景。它不是官方隐藏 test。生产者按中央步长 10、两侧累计 XY 距离最接近 1.6m、最小单侧距离 0.1m 的规则选帧；本项目不重采样原始轨迹或另加筛选。

```text
datasets/DDAD/processed/
  selection_test.json
  manifest_test.json
  bins_test.json
  bin_infos/ddad_<scene>_<center_camera01_stem>.pkl
  images_small/<scene>/<camera>/<source_stem>.jpg
  params_small/<scene>/<camera>/<source_stem>.json
  dptm_small/<scene>/<camera>/<source_stem>_dpt.npy
  dptm_small/<scene>/<camera>/<source_stem>_conf.npy
  dptm_small/<scene>/<camera>/<source_stem>_meta.json
```

schema 为 `svfgs_temporal18_v1`。每个 bin 的六个 `sensor_info[CAM_*]` 均为 `[center,before,after]`，并包含显式 scene_id、帧索引和 selection 身份。每图资产的 `data_path/intrinsic_path/depth_path/confidence_path/depth_meta_path` 都由索引提供。

DDAD 与 PandaSet 的关键区别：

- 深度目录是 `dptm_small`，而非 PandaSet 的 `dptm`；优先读取索引路径，不让目录差异散落到评估代码。
- 文件名采用各相机自己的 source stem/datum 身份，不从中央前相机时间戳推导其他图像路径。
- 原始时间字段可能存在不同编码，本项目不据此换算时间或重建路径。
- 正式数据不要求 `bins_train.json`。训练若无独立清单必须报错，不回退到 test；旧 `processed_only_input` 不参与十八视角实验。

## 3. 相机顺序、几何与十八视角组装

### 3.1 固定映射

| 项目相机语义 | DDAD 物理相机 | 过去/未来目标索引 | 中央目标索引 |
| --- | --- | --- | --- |
| CAM_FRONT | CAMERA_01 | 0 / 1 | 12 |
| CAM_FRONT_RIGHT | CAMERA_06 | 2 / 3 | 13 |
| CAM_FRONT_LEFT | CAMERA_05 | 4 / 5 | 14 |
| CAM_BACK | CAMERA_09 | 6 / 7 | 15 |
| CAM_BACK_LEFT | CAMERA_07 | 8 / 9 | 16 |
| CAM_BACK_RIGHT | CAMERA_08 | 10 / 11 | 17 |

正确物理顺序是 **01,06,05,09,07,08**，不能采用旧的编号升序映射。加载器按统一 CAM_* 顺序读取已发布索引，并验证其 camera 字段与 manifest 映射一致，RGB、pose、K 和深度同时遵守这套顺序。

### 3.2 加载流程

`DatasetDDAD.__getitem__` 对每个 bin 执行：

1. 读取、检查索引，按上表取六个 `[0]`，形成 context。
2. 依次取各相机 `[1]`、`[2]`，形成十二路 novel 目标。
3. 在十二路目标后拼接已加载的中央六路，各字段同步拼接，避免重复读取。
4. 返回原有 `context/target/scene` 结构，`scene=bin_token`；额外返回索引中的 `scene_id`。

无 batch 时，context 包含六路 RGB、归一化 K、OpenCV c2w、near/far 和本地 0～5 索引；target 对应十八路，索引 0～17。RGB 分别为 `(6,3,H,W)` 和 `(18,3,H,W)`；测试参考深度为 `target.rel_depth=(18,H,W)`。末尾六路 RGB/K/c2w/深度须与中央输入所加载的资产一致。

`utils_ddad.py` 沿用 PandaSet 的 `load_info/load_conditions` 风格，只处理 DDAD 资产。相对路径必须以配置的 processed 根目录解析；不存在时明确报错，不跨根目录找同名文件。

### 3.3 坐标和图像处理

生产者已计算每张图像相对于中央 LiDAR 的变换：

```text
T_center_lidar_from_camera(t,k)
    = inverse(T_world_from_lidar(center)) @ T_world_from_camera(t,k)
```

DepthSplat 直接以该 `sensor2lidar_transform` 作为 OpenCV c2w，不加 `flip_yz`，不以静态标定外参重算，不把前后时刻分别归一化到自身 LiDAR。上述变换适用于三时刻全部图像。

逐图读取处理后的 224×400 RGB 和对应 K；需要 112×200 时按现有 PandaSet 方式同步缩放，再按宽高归一化 K。只在本项目做必要的 resize/既有 patch shim，不重做主点居中裁剪、去畸变或深度尺度校正。

Metric3D 深度用于 PCC，不作为编码器额外输入；无需向模型注入置信度。RGB 指标保持全图，mask 可为全 1 占位，不声称实际场景没有动态物体。

默认 patch shim 的有效块大小为 16，`112×200` 实际裁成 `112×192`，`224×400` 不裁剪。记录实际尺寸，并保持 RGB/K/参考深度对齐；不能把配置名当作与 SVF-GS 指标视域完全一致的证据。本轮不因接入 DDAD 而改变已有模型预处理。

## 4. 配置和输入检查

新配置以现有 PandaSet YAML 为模板，只改变数据集名、根目录和实验标识；保持编码器/损失结构及 `near=0.5, far=100`，不因预处理深度上限为 300m 就擅自改变模型深度范围。数据覆盖超出模型范围的影响作为后续分析项。

DDAD 与 PandaSet 使用相同的配置字段：

- `dataset.processed_root=null`：默认 `roots[0]/processed`，允许覆盖到临时验收数据。
- `dataset.test_split=total`：total 读完整清单，mini 均匀抽最多 100 个，demo 最多 10 个；val 也最多取 10 个。保留原 `train/val/test` Stage 类型。

保留 `view_sampler: all`，由 Dataset 中的发布索引确定 6→18 内容，不切换为 RE10K 的 evaluation sampler。

在原类/工具内检查 manifest 完成标记、schema/dataset、selection 身份、bins 顺序与唯一性；逐 bin 检查哈希、camera 映射、每路三时刻和资产引用。RGB、K、位姿、深度形状与有限性应明确验证。不合格样本不能静默跳过，也不能复制中央视角伪造 novel 目标。

生产者 manifest/索引是跨方法共享的数据依据，不将 SVF-GS 的 config 构造函数或完整 Python 环境作为 DepthSplat 加载的依赖。

## 5. 分组评估与结果保存

复用 PandaSet 升级时对原 `ModelWrapper` 的小范围统计扩展，通过 `src/evaluation/zero_shot.py` 中的辅助函数汇总，不另写 DDAD 专用评估循环。前馈方法正式报告 all_18 的 RGB 指标；novel_12/input_6 保留作诊断，与优化式方法比较时才并列报告 novel_12：

| 组别 | target 切片 | total 预期 bin 数 | 视角评估次数 |
| --- | --- | ---: | ---: |
| all_18 | 0:18 | 324 | 5832 |
| novel_12 | 0:12 | 324 | 3888 |
| input_6 | 12:18 | 324 | 1944 |

PSNR/SSIM/LPIPS 逐视角计算后组内平均，再按 bin 等权汇总。PCC 对每个 bin 的组内深度展平计算，再按 bin 平均，记录 `pcc_reference=metric3d_v2`。Metric3D 为预测参考，不能称为深度真值或与 OmniScene 的 DA-v2 参考混用。

沿用现有整体 JSON（新协议对应 all_18），补充同一格式的 `per_bin_metrics.csv`、`evaluation_summary.json`、`data_provenance.json`，字段和聚合规则见 PandaSet 方案第 4 节。保留明确 scene_id，记录 checkpoint 身份、selection 哈希、split、相机顺序、深度参考及配置/实际分辨率。

新输出目录区分目标数据集、权重来源、split 和 temporal18。旧六视角结果不覆盖；若样本与预处理不同，也不能与新 input_6 直接作同协议比较。

## 6. 权重与运行示例

DDAD 沿用现有 `checkpointing.pretrained_model` 和严格加载；可使用自训 OmniScene 或官方 RE10K small/base/large 完整 GS 权重。官方两视角训练与本次六输入推理的区别需要在实验设置中注明，不另加训练或微调步骤。

当前优先评估 OmniScene 自训 base 和 small，官方 RE10K base 仅作后续可选对照。以下为自训 base 的全量命令，供用户正式运行；本次开发没有执行全量 GPU 实验：

```bash
conda activate depthsplat
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python -m src.main +experiment=ddad_112x200 \
mode=test \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=checkpoints/omniscene-112x200-depthsplat-base/checkpoints/epoch_0-step_100000.ckpt \
output_dir=outputs/depthsplat-ddad-112x200-base-omniscene/total/temporal18
```

自训 small 使用 `num_scales=1, upsample_factor=4, lowest_feature_resolution=4, monodepth_vit_type=vits`，checkpoint 为 `checkpoints/omniscene-112x200-depthsplat-small/checkpoints/epoch_0-step_100000.ckpt`，输出目录同步改为 `small-omniscene`。README 提供两种规模的完整命令。

后续如需官方 RE10K base，保留 base 结构参数，checkpoint 改为 `pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth`，输出目录中的来源改为 `base-re10k`。无需改代码；官方两视角训练和当前六输入推理需明确区分。

默认 `dataset.test_split=total`，Hydra 日志自动写入 `${output_dir}/logs/hydra`。首轮使用单 GPU、test batch size 1；临时验收把数据根目录、output_dir 和重定向的进程日志指向 /tmp，不写入正式数据和历史实验目录。

## 7. 实施与验收

1. 在 PandaSet 原架构中完成十八视角加载后，按同样的文件分工新增 DDAD Dataset/工具及 YAML，并注册 DatasetCfg。无需修改 PandaSet 的继承结构或配置命名。
2. CPU 合成数据验证：六相机语义、过去/未来交错、末尾中央复用、按配置根目录解析、K/pose/深度对应、小数据集采样无重复及不完整数据报错。
3. CPU 正式清单检查：全部 324 个 bin 的引用与身份闭合；抽取 000150 常规样本和 000173/000189 长间隔样本检查几何及实际图像内容。验证正前/正后与左右朝向，不能只验证矩阵可逆。
4. 指标及权重回归：三组样本数/平均方式正确，与 PandaSet 使用同一统计实现；OmniScene/RE10K 现有路径不受影响，自训/官方对应权重严格加载通过。
5. 后续 GPU 小样本：验证 6→18 网络前向、RGB/深度渲染、分块拼接和三组有限指标；目标 RGB/深度变化不应改变生成的高斯及目标渲染。
6. 正式实验单独执行：三组均覆盖 324 个唯一 token，无未解释的缺失、重复或非有限值，并记录运行和数据身份。本项目完成状态以实际 DepthSplat 验收/结果为准，不能用 SVF-GS 的完成记录代替。


### 本轮验证记录

已遍历正式 324 个 bin 完成 CPU 加载检查。GPU 在临时 processed 的 2 个 bin 上分别加载 OmniScene 自训 base 和 small，均严格加载并完成十八视角评估；small 另做完整与 chunk=6 对照，PNG 逐文件一致，三组 RGB/PCC 指标差异小于 1e-6。三组各覆盖 2 个唯一 token，不代表全量结果。

临时产物位于 `/tmp/depthsplat-zero-shot-lWPn3Q/ddad-omniscene-base/` 和同级 `ddad-omniscene-small{,-full}/`。详细共同验证范围见 PandaSet 文档“本轮验证记录”；正式全量实验由用户另行启动。

## 8. 对照来源

- 本项目 [PandaSet 方案](<PandaSet 数据集适配方案.md>) 及现有 `dataset_pandaset.py/utils_pandaset.py`，作为架构模板。
- SVF-GS：`../SVF-GS/docs/零样本泛化实验/DDAD 数据集适配方案.md` 和同目录的十八视角规划（路径相对本项目根目录）。
- SVF-GS 实现：`configs/build_config.py::build_zero_shot_config`、`data/temporal_dataset.py`、`data/transforms/temporal_loading.py`、`tools/temporal_data.py`、`tools/ablation_metrics.py`。对齐发布数据与视角/指标契约，保留 DepthSplat 自身的 OpenCV 相机约定。
