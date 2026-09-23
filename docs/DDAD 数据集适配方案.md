# DDAD 数据集适配方案（DepthSplat / comp_svfgs）

更新日期：2026-09-23。状态：**DDAD 全图及可选自车遮挡掩码评估已实现。** 掩码指标已与 SVF-GS 做 CPU 数值对照，并完成自训 small/base 各两样本的 GPU 开关对照；本轮未启动全量 GPU 实验。

SVF-GS 的零样本泛化实验与协议文档已更新完成。本项目读取同一份已发布 DDAD 十八视角数据，当前 `processed/manifest_test.json` 为 `complete=true`、`num_bins=324`。前馈方法正式口径为 `final/all_18` 的 PSNR/SSIM/LPIPS。

本方案与 [PandaSet 数据集适配方案](<PandaSet 数据集适配方案.md>) 并列，沿用其原有适配方式，不通过抽取共享 Dataset 基类改造 PandaSet。

本次对照 SVF-GS 提交 `af39b31d984ba128020282764265fa38d31ce767` 的文档、加载器、指标和汇总代码。目标是可选排除 novel_12 中模板标记的自车区域，**不改变模型推理或输入视角评价，也不保证分数必然提高**。

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

Metric3D 深度用于 PCC，不作为编码器额外输入；无需向模型注入置信度。默认 RGB 指标保持全图，原 `target.masks` 仍为全 1 占位，不声称实际场景没有动态物体。新增自车掩码使用独立 `target.eval_mask`，不替换该字段，具体规则见第 5.1 节。

默认 patch shim 的有效块大小为 16，`112×200` 实际裁成 `112×192`，`224×400` 不裁剪。记录实际尺寸，并保持 RGB/K/参考深度对齐；不能把配置名当作与 SVF-GS 指标视域完全一致的证据。本轮不因接入 DDAD 而改变已有模型预处理。

### 3.4 已发布自车掩码资产

直接读取共享 `datasets/DDAD/processed/ego_masks/vidar_v1/`，不在本项目下载、生成或修改掩码，不重新处理 RGB/Metric3D，不改 `selection_test.json`、`manifest_test.json` 或 bin 索引。

```text
ego_masks/vidar_v1/
  manifest.json
  source_1216x1936/CAMERA_*.png    # 6 张原始模板
  cleaned_1216x1936/CAMERA_*.png   # 6 张原始分辨率去噪模板
  224x400/<pixel_sha256>.png      # 19 张去重的处理后掩码
```

- 来源为 TRI-ML/vidar，固定上游提交 `0d84851ce4d86a9f132f8027898ff981e751db79`；这是共享相机模板，不是逐帧/逐场景自车分割。灰度 PNG 的 **0 表示排除、255 表示有效**，前相机 CAMERA_01 全白。
- 当前资产修订为 `native_area_opening_128_v1`：在原始 1216×1936 上，以黑色无效区域为前景，删除面积 ≤128 像素的 8 邻域连通域；再按 RGB 同一浮点像素变换 A，以 `cv2.remap / INTER_NEAREST` 生成 224×400 图像。无效边界填 0，不膨胀保留区域。DepthSplat 不重复去噪，尤其不能在缩小后再按 128 像素筛选。
- `scene_camera_to_variant[scene_id][camera] → variants[variant_id].mask_id → masks[mask_id]` 确定文件。22 种标定/裁剪变换引用 19 张 PNG，覆盖 49 场景、294 个场景/相机组合；同一场景同一相机的过去/未来帧复用，但不同主点/裁剪不得混用。
- 掩码 manifest 的 schema 是 `svfgs_ddad_ego_mask_v1`。当前 SHA256 为 `473adeaf83e9d1f086e107a234bd89eed248e8bfcbdc3784762f49635384f793`。这是本次核对的资产快照，运行时必须计算实际哈希，不能仅凭目录名判断版本。
- 模板仍可能漏盖少量后方车身边缘（例如 000156、000191）；几何对齐不等于精确分割，不根据预测误差调整模板，不声称完全去除所有自车像素。

已只读核对上述 6 张原始、6 张去噪、19 张处理 PNG 的文件哈希，以及掩码清单与正式 selection 的两种哈希关联；本项目加载和指标验收范围见第 7 节。

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

### 5.1 可选自车掩码评估

#### 开关与最小接入

唯一用户开关为 `test.eval_use_ego_mask=false`，位置沿用 DepthSplat 的 `test.*` 配置习惯，对应 SVF-GS 的顶层 `eval_use_ego_mask`，不是直接复制其命令。协议参数和输出目录解析集中在 `src/evaluation/ego_mask.py`。

- 在 `TestCfg` 声明布尔值；DDAD Dataset 的内部同名字段由 `${test.eval_use_ego_mask}` 插值传入，用户不重复指定。`src/main.py` 开始测试前校验：开启只允许 `mode=test`、`dataset.name=ddad`；Dataset 也只允许 test Stage。PandaSet、OmniScene、训练及训练期验证误用时明确报错。
- 默认关闭时不读取掩码 manifest/PNG，不要求掩码目录存在，不构建 spatial LPIPS，保留原全图数值和输出路径。它与 `train.use_dynamic_mask`、深度置信度或有效性掩码无关。
- 在原 `dataset_ddad.py/utils_ddad.py` 内接入加载，不改 DataModule 接口，不抽取共享 Dataset 基类，不把 PandaSet 的工具改造成通用框架。可在 DDAD 工具内封装轻量加载器，按 worker、mask_id 和加载分辨率缓存二值张量；不创建磁盘缓存。
- 只新增 `target.eval_mask`：未 batch 为布尔 `(18,H,W)`，batch 后为 `(B,18,H,W)`。前 12 张按照第 3.1 节的目标相机次序取共享掩码，后 6 张**始终为全 1**；前相机的两个 novel 目标也为全 1。顶层可携带 `eval_mask_manifest_sha256`，与 Dataset 的 evaluation metadata 交叉核对。
- 掩码仅进入 `ModelWrapper.test_step` 的指标计算。编码器输入、目标相机、生成高斯、RGB/深度渲染、训练损失和保存的渲染图全部不变；不把 target GT 填充值送入模型或写回渲染张量。

#### 加载校验与 patch 对齐

以 SVF-GS `DDADEgoMasks` 的检查为准，在本地实现同等约束，不运行时导入另一个仓库：

1. 按配置的 processed 根目录解析 manifest 和资产路径，禁止越界或缺文件时跨目录搜索。核对 dataset/schema、上游提交、`selection_sha256`、`selection_file_sha256`、相机顺序、源图/派生图文件哈希、灰度 L 模式和 0/255 二值语义；记录去噪修订和规则。
2. 启动时核对完整 selection 的场景/相机引用覆盖和原始 K；逐图核对 asset_id、scene_id、camera、source_hw、image_hw、处理后的 K 及 A。按 `principal_center_float_v1` 复算浮点 A，检查 `K_processed=A @ K_raw`，使用 SVF-GS 的 `rtol=0, atol=1e-6`。不能仅因分辨率相同就复用别的标定，不能取整主点裁剪框。
3. 224×400 掩码用 **PIL NEAREST** 缩放到 Dataset 的配置尺寸，再以 `==255` 转 bool；RGB 仍沿用现有加载方式。遇到未知场景、损坏文件、身份/标定不匹配或空有效区域时直接报错，不以全 1 兜底或跳过 bin。
4. `src/dataset/shims/patch_shim.py` 已在 `image/masks/rel_depth` 之外显式同步裁剪 `eval_mask`。112×200 时先缩放、再按与 RGB 完全相同的窗口左右各裁 4 列成为 112×192；**不能直接将 224×400 缩放为 112×192**。其他尺寸使用 shim 算出的实际窗口。
5. 指标入口检查 mask/RGB/depth 的尺寸、device、视角数和 bool 类型；在最终裁剪后的掩码上计算有效像素和 SSIM 支持区。

关闭开关或全 1 掩码时必须回归原结果。即使掩码规则与 SVF-GS 一致，DepthSplat 的 112×192 实际视域和原 RGB resize 仍不同；本次不改变既有模型预处理，也不声称配置名相同就构成严格同像素比较。跨方法对照须同时记录配置尺寸、实际尺寸、裁剪和插值。

#### 与 SVF-GS 一致的指标定义

在 `src/evaluation/metrics.py` 增加统一的 masked RGB/PCC 入口，规则对应 SVF-GS `compute_image_metrics()`、`compute_eval_pcc()`。协议参数固定记录，不作为逐实验调分数的自由开关。设 `M=1` 为有效区域：

| 指标 | 部分有效视角的计算规则 |
| --- | --- |
| PSNR | GT/pred 先按原函数裁到 [0,1]；`MSE = sum(M * (pred-gt)^2) / (3 * sum(M))`，再取 `-10*log10(MSE)`。只以有效 RGB 元素作分母。 |
| SSIM | `structural_similarity(..., win_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=True, channel_axis=0, data_range=1.0, full=True)`；对最终 M 用全 1 的 11×11 方形结构腐蚀，图外视为无效，仅平均完整窗口落在有效区域内的中心及三个通道。这同时排除图像四周 5 像素边界。不对预测/GT 涂黑后计算。 |
| LPIPS | 使用同一 VGG 权重、`normalize=True`；独立缓存 `LPIPS(net="vgg", spatial=True).eval()`，不修改已有全图 LPIPS 实例。指标内部构造 `pred_eval=where(M,pred,gt)`，获取 `(N,H,W)` 距离图后取 `sum(M*distance)/sum(M)`。规则名 `gt_fill_invalid_spatial_valid_mean`。 |
| PCC | 对每个 bin/组先用 M 筛选预测与 Metric3D 参考深度，再展平计算 Pearson，规则名 `valid_group_flatten`。不把无效位置置零当作样本，不平均逐图 PCC，也不额外加入置信度或正深度筛选。 |

RGB 实现先调用原全图函数得到十八路分数，仅替换 `M` 非全 1 的视角；因此 input_6 和 CAMERA_01 保持原分数。SSIM/LPIPS 沿用原输入，不额外作 PSNR 使用的 [0,1] clamp。Masked LPIPS 保留 VGG 的卷积上下文，GT 填充用于阻断无效区预测误差向有效区传播，不等同于标准全图 LPIPS 或逐像素独立误差。

每视角无有效像素、无完整 SSIM 窗口必须报错；PCC 有效点少于 2、存在非有限值或任一深度方差为零也报错，全 1 时通过检查后走原 PCC。注意本项目原 `compute_pcc` 有三维 jaxtyping 注解，筛选出的向量不能未经适配直接传入。PSNR 的零 MSE 不额外加 epsilon：按原公式得到非有限值后由现有完整性检查拒绝，不伪造有限高分。

#### 分组和输出一致性

`test_step` 只计算一套逐视角 RGB 值，同时交给原日志、整体 JSON 和 `view_group_records`，避免 novel_12 使用 masked 值而 all_18 又重新计算全图。对每个 bin，三组定义为：

- `novel_12`：前 12 路使用各自 M 的分数等权平均。
- `input_6`：后 6 路原全图分数等权平均。
- `all_18`：上述十八路等权平均，即 RGB 分数满足 `(12 * novel_12 + 6 * input_6) / 18`，再按 bin 等权汇总。**不按各视角剩余有效像素数重新加权**。

PCC 则分别在每个 bin 的三组有效像素集合内展平后计算，再平均各 bin；不满足上述 RGB 的加权关系。前馈方法主结果仍是 `final/all_18` 的 PSNR/SSIM/LPIPS，novel_12/input_6 为诊断，PCC 为独立深度诊断。

### 5.2 掩码实验身份与目录

- 关闭为 `pixel_protocol=full_image`、`mask_manifest_sha256=""`；开启为 `pixel_protocol=ddad_ego_novel12_v1` 和实际 manifest 文件哈希。CSV 每行、summary 顶层与各视角组均记录身份；input_6 虽然全图计算，也保留该次 masked run 的统一身份，与 SVF-GS 一致。
- provenance 和 summary 的 `eval_mask` 记录 source_commit、资产修订/清理规则、模板精度限制、manifest 路径/哈希、加载插值、SSIM/LPIPS/PCC 参数；延续 checkpoint/selection/实际尺寸记录。协议参数来自同一份本地配置，加载器、指标和汇总不得分别维护冲突默认值。
- 汇总前检查所有记录及 provenance 的协议/哈希一致；拒绝缺掩码哈希、全图协议却携带掩码哈希或混合版本。旧 CSV 缺字段只解释为 `full_image`，不能自动升级为 masked；跨方法比较也必须匹配掩码修订和实际像素范围。
- 把用户的 `output_dir` 视为原全图目录，开启时自动追加 `_ego_novel12_v1`，关闭时保持原路径，例如 `.../temporal18` 与 `.../temporal18_ego_novel12_v1`。Hydra 日志、metrics、图像使用同一个解析后的目录，不增加命令行路径参数，也不重复追加后缀。
- 路径推导应在 Hydra 创建日志前可用，并由主程序复用；不能仅在 `test_step` 改输出路径，否则 Hydra 日志仍会混入全图目录。保留用户原始目录和实际目录以便追溯。若已有目录的协议或 manifest 哈希不同，应报错要求换输出根目录，禁止覆盖、混合去噪前后资产的结果。
- 掩码是测试身份，不改变模型结构、checkpoint 选择、严格权重加载或 OmniScene 源训练配置；自训 base/small 仍是首要验证对象，官方 RE10K base 为后续可选对照。

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

**启用掩码：** 上述 DDAD base/small 命令仅增加 `test.eval_use_ego_mask=true`。完整十八路仍然渲染，total 仍覆盖 324 个 bin；关闭时省略该参数。输出目录后缀及 Hydra 日志按第 5.2 节自动推导，不要求用户再写 `hydra.run.dir`。PandaSet 命令不增加此参数。

## 7. 实施与验收

1. 在 PandaSet 原架构中完成十八视角加载后，按同样的文件分工新增 DDAD Dataset/工具及 YAML，并注册 DatasetCfg。无需修改 PandaSet 的继承结构或配置命名。
2. CPU 合成数据验证：六相机语义、过去/未来交错、末尾中央复用、按配置根目录解析、K/pose/深度对应、小数据集采样无重复及不完整数据报错。
3. CPU 正式清单检查：全部 324 个 bin 的引用与身份闭合；抽取 000150 常规样本和 000173/000189 长间隔样本检查几何及实际图像内容。验证正前/正后与左右朝向，不能只验证矩阵可逆。
4. 指标及权重回归：三组样本数/平均方式正确，与 PandaSet 使用同一统计实现；OmniScene/RE10K 现有路径不受影响，自训/官方对应权重严格加载通过。
5. 后续 GPU 小样本：验证 6→18 网络前向、RGB/深度渲染、分块拼接和三组有限指标；目标 RGB/深度变化不应改变生成的高斯及目标渲染。
6. 正式实验单独执行：三组均覆盖 324 个唯一 token，无未解释的缺失、重复或非有限值，并记录运行和数据身份。本项目完成状态以实际 DepthSplat 验收/结果为准，不能用 SVF-GS 的完成记录代替。


### 既有全图路径验证记录

已遍历正式 324 个 bin 完成 CPU 加载检查。GPU 在临时 processed 的 2 个 bin 上分别加载 OmniScene 自训 base 和 small，均严格加载并完成十八视角评估；small 另做完整与 chunk=6 对照，PNG 逐文件一致，三组 RGB/PCC 指标差异小于 1e-6。三组各覆盖 2 个唯一 token，不代表全量结果。

临时产物位于 `/tmp/depthsplat-zero-shot-lWPn3Q/ddad-omniscene-base/` 和同级 `ddad-omniscene-small{,-full}/`。详细共同验证范围见 PandaSet 文档“既有全图路径验证记录”；正式全量实验由用户另行启动。

### 自车掩码升级实施清单与验收要求

| 文件范围 | 实现职责 |
| --- | --- |
| `config/main.yaml`、`TestCfg`、`config/dataset/ddad.yaml`、`DatasetDDADCfg` | 唯一测试开关及向 DDAD 的内部插值传递；非法 mode/dataset 拒绝 |
| `src/dataset/dataset_ddad.py`、`utils_ddad.py` | 共享资产身份/几何检查、worker 内缓存、独立 eval_mask 和元数据 |
| `src/dataset/types.py`、`shims/patch_shim.py` | eval_mask 类型及同步裁剪；无字段时原行为不变 |
| `src/evaluation/metrics.py`、`zero_shot.py`、`src/model/model_wrapper.py` | masked 指标、一次逐视角计算、三组 PCC/汇总及协议身份检查 |
| `src/main.py`、DDAD experiment YAML | 共用的最终输出路径推导、Hydra 日志隔离和已有结果身份检查 |
| `src/evaluation/ego_mask.py`、`src/config.py` | 单一协议配置和 Hydra resolver；类型化配置解析内部布尔插值 |
| `tests/test_zero_shot.py`、`tests/test_ego_mask.py` | 加载、指标对照、关闭回归和目录自动推导测试 |

1. **配置隔离**：默认关闭且掩码目录不存在仍可加载；PandaSet/OmniScene/训练误用开关报错；small/base 和 112×200/224×400 配置均能组合；保持严格权重加载，未涉及模型参数。
2. **资产与几何**：覆盖正式 324 个 bin 的场景/相机引用，特别是四组完整标定；缺文件、错误 scene/camera/K/A、篡改哈希、空 mask 均失败。检查前 12 路过去/未来映射、后 6 路全 1，以及 nearest 后再 patch crop 的逐像素结果。
3. **跨仓库指标一致性**：以固定、相同的 GT/pred/depth/mask 张量，在最终 112×192 和不裁剪的 224×400 上分别对照 SVF-GS 两个指标入口。使用真实 VGG LPIPS（不能只用 mock），同设备/依赖下以 `rtol=1e-5, atol=1e-6` 为验收阈值；失败时查明数值或定义差异，不能直接放宽。参考结果可离线导出，生产运行不引入 SVF-GS 依赖。
4. **指标性质**：关闭与全 1 回归原函数；只改变无效区预测不改变 masked PSNR/SSIM/LPIPS/PCC，改变有效区可改变分数；独立核对有效 MSE 分母、SSIM 窗口腐蚀、LPIPS GT 填充和 PCC 有效集合。验证不同掩码面积下仍是视角/bin 等权平均及完整/分块渲染的顺序一致。
5. **结果身份**：日志、CSV、summary、旧整体 JSON 的 all_18 一致；混合全图/masked 或不同哈希拒绝；Hydra、图像和指标均在同一自动隔离目录，已有全图目录不变。资产修订不改变协议名时也能通过哈希发现冲突。
6. **后续 GPU 小样本**：在独立临时目录用自训 small/base 比较开关前后，要求高斯、十八路 RGB/深度和 input_6 指标保持一致。使用正式 processed 只读配合 demo/显式少量 bin，或准备身份闭合的临时夹具；不能把正式掩码 manifest 原样配给 selection 哈希已改变的旧两-bin目录。这里不启动全量实验，也不把 SVF-GS 的测试通过视为本项目已通过。

### 本次掩码升级验收记录

- 开启本地数据及 SVF-GS 指标参考检查后，标准库 unittest 共 23 项全部通过；全图回归仍包含 PandaSet 264 和 DDAD 324 个 bin 的 CPU 加载检查。
- CPU 使用合成夹具覆盖 mask 顺序/缓存、nearest 后 patch crop、损坏及空掩码拒绝、无掩码回归、真实 VGG LPIPS 的无效区域预测不变性、SSIM 窗口和 PSNR/PCC 定义，以及混合协议/哈希拒绝。
- 对同一组固定张量，在 112×192、224×400 上与 SVF-GS 当前 `compute_image_metrics/compute_eval_pcc` 比较，真实 VGG LPIPS/PSNR/SSIM/PCC 均通过 `rtol=1e-5, atol=1e-6` 对照。参考仓库仅由可选测试读取，生产代码不依赖它。
- 正式 DDAD 的全部 324 个 bin 已完成 CPU 掩码引用/标定检查，实际缓存 19 张共享 mask；没有改动共享数据。
- GPU 只选同一正式清单中两个 bin，自训 small/base 各进行关闭和开启对照。两种规模的高斯、十八路原始 RGB/深度张量哈希、保存 PNG 和 input_6 四项指标完全一致；small 对照同时覆盖完整渲染与 chunk=6。各次三组记录完整，旧整体 JSON 与 final/all_18 一致。
- 全部调试产物和日志位于 `/tmp/depthsplat-ego-smoke-nLRnP0/`，provenance 标注 `development_bin_limit=2`；这些是功能验收，不代表正式质量结果或效率测量。目录后缀及 Hydra 日志由开关自动推导，没有在命令行重复指定。

## 8. 对照来源

- 本项目 [PandaSet 方案](<PandaSet 数据集适配方案.md>) 及现有 `dataset_pandaset.py/utils_pandaset.py`，作为架构模板。
- SVF-GS：`../SVF-GS/docs/零样本泛化实验/DDAD 数据集适配方案.md` 和同目录的十八视角规划（路径相对本项目根目录）。
- SVF-GS 实现：`configs/build_config.py::build_zero_shot_config/build_eval_mask_config`、`data/temporal_dataset.py`、`data/transforms/ego_mask.py::DDADEgoMasks`、`tools/temporal_data.py::centered_crop`、`tools/metrics.py::compute_image_metrics/compute_eval_pcc`、`tools/ablation_metrics.py`、`tools/ablation_results.py`、`tools/ablation_analysis.py` 及 `tests/test_ego_mask_evaluation.py`。对齐发布资产与指标契约，保留 DepthSplat 自身的 OpenCV 相机约定。
