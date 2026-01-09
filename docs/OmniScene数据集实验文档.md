# OmniScene 数据集实验说明

## 配置概览
- 入口配置位于 `config/dataset/omniscene.yaml`，整体结构沿用 `re10k` 配置，只需将 `name` 设置为 `omniscene`、`roots` 指向 `datasets/omniscene`。图像尺寸使用 `224x400` 并保持黑色背景，`defaults.view_sampler` 选为 `all`，方便在任意阶段提供全部输入视角。
- OmniScene 相关实验使用 `+experiment=omniscene_112x200` 或 `+experiment=omniscene_224x400`，命令示例见 `README.md` 开头处的训练/测试脚本。运行方式仍是 `python -m src.main`，Hydra 会先加载 `config/main.yaml` 基础配置，再由 experiment 覆盖 batch size、验证间隔、最大步数等参数。
- 训练/验证/测试的节奏控制依旧由 `trainer.val_check_interval`、`train.eval_model_every_n_val`、`trainer.max_steps` 等字段决定。虽然 `config/main.yaml` 提供了默认值，但 README 中的运行指令会统一覆盖为 `trainer.val_check_interval=0.01`（约每 1k step 触发验证）和 `train.eval_model_every_n_val=10`（约每 10k step 运行一次完整测试），以便在 OmniScene 上更密集地监控训练过程；其它配置项（batch size、checkpoint 目录等）则按实验文件适配即可。

## 数据加载流程
1. **注册入口**：`src/dataset/__init__.py` 中把 `omniscene` 名称映射到 `DatasetOmniScene`，因此只要配置里写 `dataset.name=omniscene`，`DataModule` 就会实例化对应的数据集。
2. **数据类实现**：`src/dataset/dataset_omniscene.py` 定义 `DatasetOmniScene`。该类读取 OmniScene（nuScenes 派生）提供的 `bins_*_3.2m.json` 列表：
   - train 阶段使用 `bins_train_3.2m.json` 全量 bin。
   - val 阶段用 `bins_val_3.2m.json` 的前 30000 个，每隔 3000 取 1 个，再取前 10 个用于快速可视化。
   - test 阶段默认仍走 “mini-test” 模式，相当于 `self.bin_tokens = self.bin_tokens[0::14][:2048]`（每 14 个取 1 个）。若需要完整测试集，手动注释该行。
   - demo 阶段使用 `bins_dynamic_demo` 列表。
3. **样本构造**：`__getitem__` 读取 `bin_infos_3.2m/{token}.pkl`，对每个摄像头（6 个环视视角）取 key-frame 的传感器参数，并调用 `src/dataset/utils_omniscene.py` 提供的 `load_info` 得到图像路径与 `c2w`/`w2c`。输入视图固定为 key-frame；输出视图额外从同一 bin 中选取 index `[1, 2]` 的帧（共 12 张），然后把输入帧再拼回输出，方便渲染 supervision。
4. **图像与掩码加载**：`load_conditions` 负责：
   - 读取 JPEG（`samples_small`/`sweeps_small`），缩放到配置分辨率，并按比例调整/归一化内参。
   - 输出视图加载 `samples_mask_small`/`sweeps_mask_small` 中的动态物体掩码；输入视图直接生成全 1 掩码。
   - 返回 `(images, masks, intrinsics)`，不再读取 depth/disp 数据。
5. **返回结构**：数据集最终返回包含 `context` 和 `target` 的字典，字段包括：
   - `extrinsics`（`c2w`）、`intrinsics`、`image`、`near`、`far`、`index`。
   - `target` 额外带 `masks`，供动静态区域过滤使用。
   这些字段与其它数据集完全一致，可直接被 `DataModule`/`ModelWrapper` 处理。
   - `context` 包含 6 个输入视图，`target` 包含 18 个输出视图（含输入帧）。

## 主程序调用方式
- 主入口 `src/main.py` 通过 Hydra 解析配置后实例化 `DataModule`，从而自动创建 `DatasetOmniScene`。训练或测试命令只需传入 `+experiment=omniscene_*`（或在配置中覆盖 `dataset=omniscene`）即可加载该数据集。
- 在 `src/model/model_wrapper.py` 的 `training_step` 中，当 `train.use_dynamic_mask=true` 时会读取 `batch["target"]["masks"]` 构造掩码，把动态区域从损失中剔除；OmniScene loader 已提供有效掩码，因此无需额外修改模型代码。
- 模型内部（EncoderDepthSplat + DecoderSplattingCUDA）会自行预测深度、计算射线并渲染图像，数据集中只需提供 RGB 和相机参数。
- 由于注意力模块要求输入尺寸是 8 的倍数，`EncoderDepthSplat` 在其 `get_data_shim` 中调用 `src/dataset/shims/patch_shim.py` 的 `apply_patch_shim_to_views` 对 `context/target` 做中心裁剪（独立于 `load_conditions` 的 resize）。原实现只处理图像和内参；针对 OmniScene 的动态掩码，我们补充了 mask 支持——若视图包含 `masks` 字段，裁剪同时更新掩码，从而在 `train.use_dynamic_mask` 下保持像素对齐。
- 测试阶段新增了 `test.save_video_omniscene` 开关（与 `save_video` 并列）。开启后，`ModelWrapper.test_step` 会基于输出序列最后 6 个环视姿态生成自定义轨迹：向前/向后平移并串联 360° 环视路径，通过 `interpolate_extrinsics` 与固定内参、near/far 组合成连续相机序列，再调用解码器渲染 OmniScene 风格的环视视频并保存至 `videos_omniscene/<scene>.mp4`。

## PCC 指标补充方案
1. **相对深度加载（仅 test）**：
   - 参考 SVF-GS 的 `data/transforms/loading.py`：读取 DepthAnything-v2 预测的 disparity（`samples_dpt_small`/`sweeps_dpt_small` 下 `.npy`），若发生 resize 需同步缩放；再用 inverse disparity 转为相对深度，并限制最大/最小比值为 50（与原实现一致），最终做 min-max 归一化到 `[0, 1]`。
   - DepthSplat 侧建议在 `src/dataset/utils_omniscene.py::load_conditions` 中受 `load_rel_depth` 控制返回 `rel_depth`（`[v, h, w]`），`DatasetOmniScene` 仅在 `stage="test"` 时传入 `load_rel_depth=True` 并在 `target` 中加入 `rel_depth`；其它阶段保持 `rel_depth=None`。
   - 若需要 patch shim（`apply_patch_shim_to_views`）保证分辨率可整除，新增对 `rel_depth` 的中心裁剪逻辑，保持与 `image/masks/intrinsics` 对齐。
2. **测试时渲染深度**：
   - 渲染端已支持深度输出：`DecoderSplattingCUDA.forward` 在 `depth_mode` 非空时会走 `render_depth_cuda` 并返回 `DecoderOutput.depth`（`[b, v, h, w]`）。
   - 当前 `ModelWrapper.test_step` 与 `run_full_test_sets_eval` 均传入 `depth_mode=None`，因此测试阶段不会产出深度。方案是：当 `target["rel_depth"]` 存在且需要 PCC 时，将 `decoder.forward(..., depth_mode="depth")`（或根据需求选 `"disparity"/"relative_disparity"`）并保留 `output.depth`，注意 chunk 渲染分支也要同步开启。
   - 为避免额外开销，可新增测试配置开关（例如 `test.render_depth_for_metrics`）或复用 `compute_scores` 条件，仅在计算 PCC 时开启深度渲染。
3. **PCC 计算位置**：
   - 参考 SVF-GS 的 `tools/metrics.py`，使用 `torchmetrics.PearsonCorrCoef` 实现 `get_pcc/compute_pcc`，放在 `src/evaluation/metrics.py` 与 PSNR/SSIM/LPIPS 同文件。
   - 在 `ModelWrapper.test_step` 与 `run_full_test_sets_eval` 中，与 PSNR/SSIM/LPIPS 同步计算 `compute_pcc(target_rel_depth, pred_depth)`；仅当 `target["rel_depth"]` 与 `output.depth` 同时存在时启用。
4. **PCC 统计与汇总**：
   - 复用 `test_step_outputs` 与 `on_test_end` 的汇总逻辑，新增 `pcc` key，输出 `scores_pcc_all.json` 并写入 `scores_all_avg.json`。
   - 在 `run_full_test_sets_eval` 的 `scores_dict` 中加入 `pcc` 字段，确保与 `psnr/ssim/lpips` 同层级记录与打印。

## 与 SVF-GS 的差异与注意事项
1. **返回字段**：SVF-GS 的 `data/dataloader.py` 还会返回 DepthAnything/Metric3D 的深度、置信度、射线、`w2i` 等信息；DepthSplat 作为通用前馈高斯重建框架，只需要输入/输出图像、相机内外参、near/far、索引及动态掩码。深度与射线由模型在运行时计算，动态掩码则借助已有 mask 接口融入损失。
2. **掩码接入**：原 RE10k/DL3DV 数据集没有动态掩码；OmniScene loader 在 `load_conditions` 中加载 `*_mask_small` 掩码并通过 `train.use_dynamic_mask` 接入训练流程。这是新引入但与项目现有 mask 逻辑兼容的扩展。
3. **坐标系细节**：SVF-GS 的 `load_info` 会对 `sensor2lidar` 做 `flip_yz` 以匹配其渲染坐标系；DepthSplat 中的 `utils_omniscene.load_info` 已注释 `flip_yz`，直接使用 LiDAR->Camera 的原始变换即可适配当前解算流程。
4. **测试模式**：SVF-GS 同时提供 `test` 和 `mini-test` 两种 split。本项目为了保持与其它数据集一致的接口，只保留 `stage="test"`，但默认沿用 mini-test 的抽样策略（每 14 个 bin 取 1 个）。如需完整测试集，可注释 `DatasetOmniScene.__init__` 中的抽样语句。
5. **配置继承**：OmniScene 的实验配置以 `re10k` 为模板，根据实际 GPU 负载调整 batch size（一般为 1）、验证频率和测试间隔，并设置不同的输出目录/预训练权重。这样可以最大化复用 DepthSplat 的既有训练流程。

## 小结
OmniScene 数据集已经与 DepthSplat 的 Hydra 配置、数据模块和模型框架完成对接。只需切换实验配置即可复用现有训练/评测流程；若未来需要与 SVF-GS 完全对齐，可在 `DatasetOmniScene` 中按需恢复深度、射线等扩展字段或调整测试抽样策略。
