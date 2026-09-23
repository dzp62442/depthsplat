# PandaSet 数据集适配方案（DepthSplat / comp_svfgs）

更新日期：2026-09-23。状态：**六输入、十八目标视角加载与分组评估已实现；已完成全量 CPU 数据加载检查和小样本 GPU 验收，未启动本项目全量 GPU 实验。**

DDAD 可选自车掩码已接入公共评估入口，**仅允许 DDAD 测试使用，不适用于 PandaSet**。本项目保留 PandaSet 原有全图加载、模型和评价口径。

SVF-GS 的零样本泛化实验与协议文档已更新完成。本项目直接消费同一份已发布数据，不再生成数据。当前 PandaSet `processed/manifest_test.json` 为 `complete=true`，包含 264 个 bin。前馈方法正式口径为 `final/all_18` 的 PSNR/SSIM/LPIPS，其他组保留作诊断。

并列数据集方案见 [DDAD 数据集适配方案](<DDAD 数据集适配方案.md>)。

## 1. 目标与改动边界

目标是在同一冻结 checkpoint 下，用 PandaSet 中央帧六路 RGB 和相机参数生成高斯，渲染十二个跨时刻新视角及六个输入视角，不在目标数据集训练或微调。

**保留现有 PandaSet 适配架构，只修改加载内容和必要的评估统计。**

- 继续由 `src/dataset/dataset_pandaset.py` 定义 `DatasetPandaSetCfg/ DatasetPandaSet`，由 `src/dataset/utils_pandaset.py` 提供 `load_info/load_conditions` 等工具。
- 保留现有配置名称、Dataset 注册、DataModule、`context/target/scene` 返回接口，以及模型、渲染器和 checkpoint 加载流程。
- 不新增共享 `TemporalDataset` 基类，不把 PandaSet 改造成另一套数据框架，也不运行时导入 SVF-GS 的 Python 模块。
- 不在本项目重做选帧、裁剪、Metric3D 预处理或索引生成，不清理共享数据和历史结果。
- 主要改动是把 `target=input` 改为“前后时刻十二路 + 中央六路”，再补充显式子集选择、三组指标和必要的数据身份记录。

本方案仅扩展零样本测试。保留原有 train 入口的职责，但没有独立 `bins_train.json` 时应明确报错，不能用测试清单代替训练清单。

## 2. 升级前后与已有数据

| 项目 | 原六视角实现 | 当前十八视角实现 |
| --- | --- | --- |
| 输入 | 中央六路 RGB/相机 | 不变 |
| 目标 | 六路输入重建，`target=input` | 十二路新视角 + 六路输入重建 |
| 测试采样 | 代码硬编码均匀抽 100 个 | 配置选择 total / mini / demo |
| 深度 | 测试时加载 Metric3D，填入 `target.rel_depth` | 加载十八路对应深度，继续用于 PCC |
| 指标 | 全部目标合并的 PSNR/SSIM/LPIPS/PCC | 增加 all_18 / novel_12 / input_6 |
| 数据位置 | `datasets/PandaSet/processed` | 不变，可显式覆盖 processed 根目录 |

截至本次检查，`datasets/PandaSet` 已链接到 `/home/B_UserData/dongzhipeng/Datasets/PandaSet`。发布数据为 264 bins、38 个有效序列，来自当前本地 39 个序列，不代表整个公开 PandaSet。

生产者的固定选帧规则是：每序列原始帧号 0、10、20……作为候选中心，在完整原始轨迹两侧选累计 XY 距离最接近 1.6m 的帧；缺少时间侧或任一侧累计位移小于 0.1m 时排除。本项目只读取最终清单，不重复执行该规则，也不附加距离或时间筛选。

```text
datasets/PandaSet/processed/
  selection_test.json
  manifest_test.json
  bins_test.json
  bin_infos/pandaset_<seq>_<center_frame>.pkl
  images_small/<seq>/<camera>/<frame>.jpg
  params_small/<seq>/<camera>/<frame>.json
  dptm/<seq>/<camera>/<frame>_dpt.npy
  dptm/<seq>/<camera>/<frame>_conf.npy
  dptm/<seq>/<camera>/<frame>_meta.json
```

当前 schema 是 `svfgs_temporal18_v1`。每个 bin 包含 `bin_token`、`scene_id`、`frame_indices`、`selection_sha256`，以及六路 `sensor_info[CAM_*] = [center_info, before_info, after_info]`。每条传感器记录直接提供 `data_path`、`intrinsic_path`、`depth_path`、`confidence_path`、`depth_meta_path` 和 `sensor2lidar_transform` 等字段。

正式清单不要求存在 `bins_train.json`；旧 `processed_only_input` 不作为新协议的兼容兜底。

## 3. 十八视角加载：在原文件内修改

### 3.1 Dataset 初始化与子集

在现有 `DatasetPandaSetCfg` 和 `config/dataset/pandaset.yaml` 中已增加两个小范围配置字段：

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `processed_root` | `null` | 默认取 `roots[0]/processed`；验收时可指定临时目录 |
| `test_split` | `total` | `total / mini / demo`，仅控制测试清单子集 |

保留 `Stage = train/val/test`，不另造 Stage 类型。测试阶段按发布清单原顺序加载；total 取全部，mini 用 `linspace(0,N-1,min(100,N))` 均匀抽样，demo/val 最多取 10 个。小清单不得重复采样，空清单报错。

初始化时核对 manifest 完成标记、schema、dataset、selection 哈希和 bins 顺序/唯一性；逐 bin 核对索引身份、六相机映射及每相机恰好三时刻。缺失或旧索引明确报错，不能复制中央视角补足十八路或静默跳样本。检查直接放在原 Dataset/工具模块内，不建立额外的缓存或版本管理框架。

### 3.2 `__getitem__` 最小改造

沿用当前中央输入加载逻辑，并在其后补充两侧视角：

1. 按现有六相机顺序读取各 `sensor_info[cam][0]`，调用原有工具加载中央 RGB、K、位姿与测试参考深度。
2. 按同一相机顺序依次收集 `sensor_info[cam][1]`、`[2]`，得到十二路过去/未来视角。
3. 通过同一工具加载十二路数据，再把中央六路已加载张量拼接到末尾，不重复读图。
4. 图像、K、c2w、near/far、mask、深度和视角身份同步按此顺序排列。
5. 保留 `scene=bin_token`，额外返回索引中明确的 `scene_id`，用于逐 bin 指标；不从 `_bin` 字符串猜测 PandaSet 场景。

| 相机语义 | PandaSet 相机 | 过去/未来目标索引 | 中央目标索引 |
| --- | --- | --- | --- |
| CAM_FRONT | front_camera | 0 / 1 | 12 |
| CAM_FRONT_RIGHT | front_right_camera | 2 / 3 | 13 |
| CAM_FRONT_LEFT | front_left_camera | 4 / 5 | 14 |
| CAM_BACK | back_camera | 6 / 7 | 15 |
| CAM_BACK_LEFT | left_camera | 8 / 9 | 16 |
| CAM_BACK_RIGHT | right_camera | 10 / 11 | 17 |

无 batch 时，`context.image=(6,3,H,W)`，`target.image=(18,3,H,W)`；对应相机分别为 `(6,4,4)/(18,4,4)` 与 `(6,3,3)/(18,3,3)`，`target.rel_depth=(18,H,W)`。沿用现有视角索引字段，输出 0～17；输入仍为本地 0～5，中央对应关系由上表确定。

### 3.3 工具函数、路径与几何

继续使用 `utils_pandaset.py` 的函数分工。`load_conditions` 增加 `infos`、`manifest` 参数以读取明确路径并核对资产；十八视角调用必须提供它们，原有四项返回元组保持不变。清单和索引检查也位于同一工具文件，未替换 Dataset 接口。

- 新索引的相对路径以当前配置的 `processed_root` 为基准解析，不先尝试从 cwd 或其他旧根目录读取同名资产。正式测试只读 processed RGB，不回退到 raw 图像。
- `c2w = sensor2lidar_transform`，直接保留 OpenCV 相机约定，不做 `flip_yz`。三个时刻已在生产者端统一到中央 LiDAR 坐标系，加载器不再分别归一化。
- 保留逐图 K 读取与 resize 同步缩放，随后 `K[0,:]/=W`、`K[1,:]/=H`。所有时刻使用各自图像的位姿与内参。
- Metric3D 深度按图像尺寸同步 resize，保持米制数值；只放入 target 供评估，不作为 DepthSplat 额外输入。网络不需要置信度张量，按需核对资产元信息，不把置信度引入模型。
- 缺图、缺深度、尺寸不匹配或非有限数值应报出 bin/相机/路径；不以零深度或另一时刻图像代替。
- mask 沿用全 1 的接口占位，RGB 指标按全图计算；不能据此描述为“所有物体静态”。当前 `train.use_dynamic_mask` 不是本次测试协议的一部分。

### 3.4 Patch shim 与实际评估尺寸

保留现有模型预处理。本项目默认 `shim_patch_size=4`、`downscale_factor=4`，有效块大小为 16；配置 `112×200` 经 patch shim 后实际为 `112×192`，`224×400` 则不发生该裁剪。

记录配置尺寸、最终输入/目标尺寸，并检查 RGB、mask、深度和归一化 K 同步对齐。配置名不代表实际指标尺寸。若要与 SVF-GS 严格按相同的 112×200 视域比较，需另行验证预处理对齐，不能在这次数据加载升级中悄悄修改已有 checkpoint 的输入处理。

## 4. 指标分组：沿用现有测试流程

在 `src/model/model_wrapper.py::test_step/on_test_end` 中扩展已有统计，复用 `src/evaluation/metrics.py` 的指标函数。`src/evaluation/zero_shot.py` 仅提供分组、汇总与写文件函数，没有独立评估循环或框架。正式报告 all_18 的 RGB 指标；novel_12/input_6 作诊断，只有与优化式方法比较时再并列报告 novel_12。

| 组别 | target 切片 | 含义 |
| --- | --- | --- |
| `all_18` | `0:18` | 全部目标视角 |
| `novel_12` | `0:12` | 跨时刻新视角泛化 |
| `input_6` | `12:18` | 中央输入视角重建 |

PSNR/SSIM/LPIPS 一次计算十八个逐视角分数，再在每个 bin 内按组平均，最后等权平均各 bin。PCC 与 SVF-GS 当前实现对齐：每个 bin 的组内深度展平计算一次 Pearson 相关系数，再平均各 bin；不是逐图 PCC 的平均。

测试保持 `test.compute_scores=true`，依据 `target.rel_depth` 渲染深度并计算 PCC，分块渲染也必须拼接完整十八路深度。记录 `pcc_reference=metric3d_v2`；它是预测深度参考，不是传感器真值，也不同于 OmniScene 的 DA-v2 参考。不得通过关闭 `compute_scores` 来只关闭 PCC，因为这也会关闭 RGB 指标。

保留 `scores_*_all.json` 和 `scores_all_avg.json` 的整体结果语义，在新协议运行中对应 all_18；增补：

- `per_bin_metrics.csv`：bin_token、scene_id、view_group、PSNR/SSIM/LPIPS/PCC。
- `evaluation_summary.json`：三组结果、请求/实际样本数、缺失/重复及非有限值情况。
- `data_provenance.json`：schema、selection 哈希、相机顺序、split、深度参考、配置/实际分辨率和实际加载的 checkpoint 路径/哈希。

这些文件作为现有输出的补充。分组扩展限定在本次十八视角协议，原 OmniScene/RE10K 的调用和已有统计不应被无关改变。首轮用单 GPU、test batch size 1 验证，不把多卡评估改造混入本轮。

### 4.1 DDAD 自车掩码升级的兼容边界

详细规则见 [DDAD 方案第 5 节](<DDAD 数据集适配方案.md>)。与 SVF-GS 一致，novel_12 可选排除自车、input_6 保持全图；本项目只在 DDAD 接入独立 `target.eval_mask`，不向 PandaSet 提供模板、不改其 Dataset/工具架构，也不把 `target.masks` 或 `train.use_dynamic_mask` 改作自车评价开关。

- 唯一用户参数 `test.eval_use_ego_mask=false`；PandaSet 必须保持关闭，误设 true 明确报错。关闭时不读取 DDAD 掩码资产，不增加 PandaSet 的数据依赖。
- 公共 `metrics.py` 增加可选 masked 入口，但 `mask=None` 时逐视角 PSNR/SSIM/LPIPS/PCC 沿用原函数。逐视角全 1 时也走原 RGB 函数，不能把所有数据集统一切到 spatial LPIPS 或改变 SSIM 参数。
- `patch_shim` 仅在 `eval_mask` 存在时同步裁剪；PandaSet 既有 RGB/K/masks/rel_depth 处理不变。保留实际 112×192 的尺寸记录，不借本轮升级调整模型预处理。
- 公共 CSV/summary/provenance 已补充 `pixel_protocol`、`mask_manifest_sha256`。PandaSet 固定为 `full_image` 和空哈希；缺字段的历史记录只按全图解释。禁止把 DDAD masked 的 novel_12/all_18 与 full_image 或不同掩码版本作为同协议结果混合汇总。
- 原目录和自动 Hydra 日志推导保持不变。新增目录后缀仅由 DDAD 掩码开关触发，不更改 PandaSet 启动命令。

回归要求：相同 PandaSet bin/checkpoint 开关默认关闭时，输入/渲染及三组数值与本轮升级前一致；缺少掩码目录也能完成原全图加载。新增身份字段可以改变 JSON/CSV 结构，但不能改变原指标数值或样本覆盖。既有全图测试记录不作为新掩码路径已实现、已验收的证据。

## 5. 权重与运行接口

自训 OmniScene checkpoint 和官方完整 GS 权重继续通过 `checkpointing.pretrained_model` 选择，保留严格加载。small/base/large 使用各自匹配的结构参数；不新增自动查找或复制权重机制。

当前优先评估 OmniScene 自训 base 和 small，官方 RE10K base 仅作后续可选对照。以下为自训 base 的全量命令；仅提供给用户正式运行，本次开发没有执行全量 GPU 实验：

```bash
conda activate depthsplat
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
python -m src.main +experiment=pandaset_112x200 \
mode=test \
model.encoder.num_scales=2 \
model.encoder.upsample_factor=2 \
model.encoder.lowest_feature_resolution=4 \
model.encoder.monodepth_vit_type=vitb \
checkpointing.pretrained_model=checkpoints/omniscene-112x200-depthsplat-base/checkpoints/epoch_0-step_100000.ckpt \
output_dir=outputs/depthsplat-pandaset-112x200-base-omniscene/total/temporal18
```

自训 small 使用 `num_scales=1, upsample_factor=4, lowest_feature_resolution=4, monodepth_vit_type=vits`，checkpoint 为 `checkpoints/omniscene-112x200-depthsplat-small/checkpoints/epoch_0-step_100000.ckpt`，输出目录同步改为 `small-omniscene`。README 提供两种规模的完整命令。

后续切换官方 RE10K base 时，保持 base 结构参数，将 checkpoint 改为 `pretrained/depthsplat-gs-base-re10k-256x256-view2-ca7b6795.pth`，输出目录中的来源改为 `base-re10k`。无需改代码，两种来源分开报告，不覆盖旧六视角结果。

默认 `dataset.test_split=total`，Hydra 日志自动写入 `${output_dir}/logs/hydra`，指标仍写入 `${output_dir}/metrics`，无需重复指定路径。`PYTHONNOUSERSITE=1` 用于避免用户级 PyTorch 覆盖 Conda 环境。临时验收时把 processed_root、output_dir 及重定向的进程日志指向临时目录即可。

## 6. 实施清单与验收

| 文件 | 修改范围 |
| --- | --- |
| `src/dataset/dataset_pandaset.py` | 原类内增加清单检查/测试配置，目标从 6 路扩为 18 路，返回 scene_id |
| `src/dataset/utils_pandaset.py` | 原工具函数内支持索引显式路径、配置根目录及必要校验 |
| `config/dataset/pandaset.yaml` | 增加 processed_root/test_split；原两份 experiment 保持结构 |
| `src/dataset/types.py` | 补充 scene_id 等必要元数据类型 |
| `src/model/model_wrapper.py`、`src/evaluation/zero_shot.py` | 原测试入口调用分组/汇总辅助函数，保存可追溯输出 |
| `src/main.py` | 记录实际 checkpoint 身份，零样本测试绕开会清理 outputs/local 的 LocalLogger |
| 测试与 README | 增加针对性验证及正式命令，不重构数据框架 |

本轮保留了 PandaSet 原类和工具结构，DDAD 采用并列文件实现。以下为持续验收要求：

1. CPU 合成样本：验证视角内容顺序、末尾六路复用、三时刻 pose/K、子集无重复，以及错误索引明确拒绝。
2. CPU 真实数据：检查正式 264 个 bin 的引用闭合和数据身份；普通及长间隔样本检查投影、RGB/深度/K 对齐。独立检查 patch shim 后的有效尺寸。
3. 指标回归：用可手算分数验证三组统计和 PCC 聚合；保留 OmniScene/RE10K 原行为，确认 checkpoint 严格加载。
4. 后续 GPU 验收：临时目录内选 2～3 个 bin，检查十八路 RGB/深度、完整与分块渲染一致，改变 target GT 不影响高斯生成/渲染。此步骤不代表全量实验。
5. 正式运行后：total 三组各覆盖 264 个唯一 bin，分别对应 4752/3168/1584 个视角评估次数。以实际发布清单为准，缺失或无效结果不能标成完整。

历史六视角数据与新数据的抽样、预处理可能不同，不能把新 input_6 与旧结果的差异归因于“只增加了目标视角”。


### 既有全图路径验证记录

- 此前全图开发验收使用标准库 unittest，无需安装 pytest；当时 12 项测试全部通过，包含遍历 PandaSet 264、DDAD 324 个 bin，核对十八路 RGB/K/c2w/深度及末尾六路复用，CUDA 未初始化。新增掩码功能的验收单列于 DDAD 文档，不以旧记录代替。
- GPU 主验收使用两数据集各 2 个临时 bin：OmniScene 自训 base/small × PandaSet/DDAD 四种组合均严格加载并完成十八路推理。另已验证 PandaSet + 官方 RE10K base 兼容性，作为后续可选入口，不替代自训主实验。
- DDAD 完整渲染与 chunk=6 对照：保存 PNG 逐文件一致，三组 RGB/PCC 指标差异小于 1e-6；每次输出 36 张渲染和 36 张 GT。
- 原 scores_all_avg 与 final/all_18 对齐；清单、checkpoint 哈希及实际 112×192 尺寸已读回核对。预热后无计时样本时保存 null，不用两样本测试报告效率结论。
- 临时产物：`/tmp/depthsplat-zero-shot-lWPn3Q/`，不作为正式质量结果。全量推理由用户另行启动。

## 7. 对照来源

- 本项目：`src/dataset/dataset_pandaset.py`、`src/dataset/utils_pandaset.py`、`src/model/model_wrapper.py`。
- SVF-GS 文档：`../SVF-GS/docs/零样本泛化实验/` 下的十八视角规划、PandaSet 方案（路径相对本项目根目录）。
- SVF-GS 当前实现：`data/temporal_dataset.py`、`data/transforms/temporal_loading.py`、`tools/temporal_data.py`、`tools/ablation_metrics.py`。参考数据契约及指标定义，不复制其 Dataset 继承结构或 OpenGL 轴翻转。
