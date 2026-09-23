import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
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
from .utils_ddad import load_info, load_conditions, load_manifest, load_bin_info

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@dataclass
class DatasetDDADCfg(DatasetCfgCommon):
    name: Literal["ddad"]
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
    processed_root: Path | None = None
    test_split: Literal["total", "mini", "demo"] = "total"


class DatasetDDAD(Dataset):
    cfg: DatasetDDADCfg
    stage: Stage
    view_sampler: ViewSampler

    camera_types = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    camera_map = dict(zip(camera_types, ["CAMERA_01", "CAMERA_06", "CAMERA_05",
                                         "CAMERA_09", "CAMERA_07", "CAMERA_08"]))

    def __init__(self, cfg: DatasetDDADCfg, stage: Stage, view_sampler: ViewSampler,
                 load_rel_depth: bool | None = None):
        super().__init__()
        self.cfg, self.stage, self.view_sampler = cfg, stage, view_sampler
        if stage not in ("train", "val", "test") or cfg.test_split not in ("total", "mini", "demo"):
            raise ValueError(f"Unsupported stage/split: {stage}/{cfg.test_split}")
        if not 0 < cfg.near < cfg.far:
            raise ValueError("DDAD requires 0 < near < far")
        self.near, self.far = cfg.near, cfg.far
        self.reso = cfg.image_shape
        self.data_root = Path(cfg.roots[0])
        self.processed_root = Path(cfg.processed_root) if cfg.processed_root is not None else self.data_root / "processed"
        self.load_rel_depth = stage == "test" and (load_rel_depth is None or load_rel_depth)
        storage_split = "train" if stage == "train" else "test"
        self.manifest, self.bin_tokens = load_manifest(
            self.processed_root, "ddad", self.camera_map, storage_split)
        if stage == "val" or (stage == "test" and cfg.test_split != "total"):
            size = 100 if stage == "test" and cfg.test_split == "mini" else 10
            indices = np.linspace(0, len(self.bin_tokens) - 1, min(size, len(self.bin_tokens)), dtype=int)
            self.bin_tokens = [self.bin_tokens[i] for i in indices]

    def __len__(self):
        return len(self.bin_tokens)

    def evaluation_metadata(self):
        return dict(dataset="ddad", schema=self.manifest["schema"],
                    split=self.cfg.test_split if self.stage == "test" else self.stage,
                    processed_root=str(self.processed_root.resolve()), num_bins=len(self),
                    bin_tokens=list(self.bin_tokens), selection_sha256=self.manifest["selection_sha256"],
                    protocol=self.manifest["protocol"], depth_model=self.manifest["depth_model"],
                    pcc_reference="metric3d_v2", configured_image_shape=list(self.reso),
                    input_views=6, output_views=18, rgb_resize="PIL bicubic", depth_resize="PIL bilinear")

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        info = load_bin_info(self.processed_root, token, self.manifest, self.camera_map)
        center = [info["sensor_info"][cam][0] for cam in self.camera_types]
        novel = [sensor for cam in self.camera_types for sensor in info["sensor_info"][cam][1:]]
        input_paths, input_c2ws = [], []
        output_paths, output_c2ws = [], []
        for sensor in center:
            path, c2w, _ = load_info(sensor, self.data_root, self.processed_root)
            input_paths.append(path)
            input_c2ws.append(c2w)
        for sensor in novel:
            path, c2w, _ = load_info(sensor, self.data_root, self.processed_root)
            output_paths.append(path)
            output_c2ws.append(c2w)
        input_imgs, input_masks, input_cks, input_depths = load_conditions(
            input_paths, self.reso, self.processed_root, self.load_rel_depth, center, self.manifest)
        output_imgs, output_masks, output_cks, output_depths = load_conditions(
            output_paths, self.reso, self.processed_root, self.load_rel_depth, novel, self.manifest)
        input_c2ws = torch.from_numpy(np.stack(input_c2ws))
        output_c2ws = torch.cat((torch.from_numpy(np.stack(output_c2ws)), input_c2ws))
        output_imgs = torch.cat((output_imgs, input_imgs))
        output_masks = torch.cat((output_masks, input_masks))
        output_cks = torch.cat((output_cks, input_cks))
        context = {
            "extrinsics": input_c2ws, "intrinsics": input_cks, "image": input_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=6),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=6),
            "index": torch.arange(6),
        }
        target = {
            "extrinsics": output_c2ws, "intrinsics": output_cks, "image": output_imgs,
            "near": repeat(torch.tensor(self.near, dtype=torch.float32), "-> v", v=18),
            "far": repeat(torch.tensor(self.far, dtype=torch.float32), "-> v", v=18),
            "index": torch.arange(18), "masks": output_masks,
        }
        if output_depths is not None:
            target["rel_depth"] = torch.cat((output_depths, input_depths))
        return {"context": context, "target": target, "scene": token,
                "scene_id": str(info["scene_id"]), "evaluation_protocol": self.manifest["schema"]}
