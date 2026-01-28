import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json
import pickle as pkl
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from einops import repeat
from torch.utils.data import Dataset
import cv2

from .types import Stage
from .dataset import DatasetCfgCommon
from .view_sampler import ViewSampler
from .utils_pandaset import load_info, load_conditions

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@dataclass
class DatasetPandaSetCfg(DatasetCfgCommon):
    name: Literal["pandaset"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False


class DatasetPandaSet(Dataset):
    cfg: DatasetPandaSetCfg
    stage: Stage
    view_sampler: ViewSampler

    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        cfg: DatasetPandaSetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        load_rel_depth: bool | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        self.reso = cfg.image_shape
        self.data_root = Path(cfg.roots[0])
        self.processed_root = self.data_root / "processed"
        self.load_rel_depth = stage == "test" if load_rel_depth is None else load_rel_depth
        if stage != "test":
            self.load_rel_depth = False

        if stage == "train":
            bins_path = self.processed_root / "bins_train.json"
            self.bin_tokens = json.load(open(bins_path))[
                "bins"
            ]
        elif stage == "val":
            bins_path = self.processed_root / "bins_test.json"
            self.bin_tokens = json.load(open(bins_path))[
                "bins"
            ]
            if len(self.bin_tokens) > 0:
                num_samples = min(10, len(self.bin_tokens))
                indices = np.linspace(0, len(self.bin_tokens) - 1, num_samples, dtype=int)
                self.bin_tokens = [self.bin_tokens[i] for i in indices]
        elif stage == "test":
            bins_path = self.processed_root / "bins_test.json"
            self.bin_tokens = json.load(open(bins_path))[
                "bins"
            ]
            indices = np.linspace(0, len(self.bin_tokens) - 1, 100, dtype=int)  # mini-test
            self.bin_tokens = [self.bin_tokens[i] for i in indices]
        else:
            raise ValueError(f"不支持的阶段: {stage}")

    def __len__(self):
        return len(self.bin_tokens)

    def __getitem__(self, index):
        bin_token = self.bin_tokens[index]
        with open(self.processed_root / "bin_infos" / f"{bin_token}.pkl", "rb") as f:
            bin_info = pkl.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0] for sensor in self.camera_types
        }

        input_img_paths, input_c2ws = [], []
        for cam in self.camera_types:
            info = copy.deepcopy(sensor_info_center[cam])
            img_path, c2w, _ = load_info(info, self.data_root)
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)

        input_imgs, input_masks, input_cks, input_rel_depths = load_conditions(
            input_img_paths,
            self.reso,
            self.processed_root,
            load_rel_depth=self.load_rel_depth,
        )
        input_cks = torch.as_tensor(input_cks, dtype=torch.float32)

        # PandaSet 仅使用输入视图进行监督（输入=输出）
        output_imgs = input_imgs
        output_masks = input_masks
        output_c2ws = input_c2ws
        output_cks = input_cks
        output_rel_depths = input_rel_depths

        context = {
            "extrinsics": input_c2ws,
            "intrinsics": input_cks,
            "image": input_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(input_c2ws)),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(input_c2ws)),
            "index": torch.arange(len(input_c2ws)),
        }

        target = {
            "extrinsics": output_c2ws,
            "intrinsics": output_cks,
            "image": output_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(output_c2ws)),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(output_c2ws)),
            "index": torch.arange(len(output_c2ws)),
            "masks": output_masks,
        }
        if output_rel_depths is not None:
            target["rel_depth"] = output_rel_depths

        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
