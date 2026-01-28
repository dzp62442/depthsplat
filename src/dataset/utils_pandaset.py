import json
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
import torch


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    h, w, c = x.shape
    assert c == 1 or c == 3 or c == 4
    if c == 3:
        return x
    if c == 1:
        return np.concatenate([x, x, x], axis=2)
    if c == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resolve_image_path(img_path: str | Path, data_root: Path) -> Path:
    """解析图像路径，兼容 raw/images_small 的相对/绝对路径。"""
    path = Path(img_path)
    if path.exists():
        return path

    parts = path.parts
    if "images_small" in parts:
        images_idx = parts.index("images_small")
        suffix = Path(*parts[images_idx + 1 :])
        candidate = Path(data_root) / "processed" / "images_small" / suffix
        if candidate.exists():
            return candidate

    if "raw" in parts:
        raw_idx = parts.index("raw")
        suffix = Path(*parts[raw_idx + 1 :])
        candidate = Path(data_root) / "raw" / suffix
        if candidate.exists():
            return candidate

    if "PandaSet" in parts:
        pandaset_idx = parts.index("PandaSet")
        suffix = Path(*parts[pandaset_idx + 1 :])
        candidate = Path(data_root) / suffix
        if candidate.exists():
            return candidate

    return Path(data_root) / path


def parse_seq_cam_frame(img_path: Path) -> tuple[str, str, str]:
    parts = img_path.parts
    if "images_small" in parts:
        idx = parts.index("images_small")
        return parts[idx + 1], parts[idx + 2], img_path.stem
    if "raw" in parts:
        idx = parts.index("raw")
        seq_id = parts[idx + 1]
        cam_name = parts[idx + 3] if parts[idx + 2] == "camera" else parts[idx + 2]
        return seq_id, cam_name, img_path.stem
    return img_path.parents[1].name, img_path.parent.name, img_path.stem


def load_info(info: dict, data_root: Path):
    """读取路径与位姿信息（不做 flip_yz）。"""
    img_path = resolve_image_path(info["data_path"], data_root)
    # 使用 lidar 坐标系作为世界坐标系
    c2w = info["sensor2lidar_transform"]

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t

    return str(img_path), c2w, w2c


def load_conditions(
    img_paths: list[str],
    reso: list[int],
    processed_root: Path,
    load_rel_depth: bool = False,
):
    """加载图像、内参、Metric3D 深度与静态 mask。"""

    def maybe_resize(img, tgt_reso, ck):
        if not isinstance(img, PIL.Image.Image):
            img = Image.fromarray(img)
        resize_flag = False
        if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
            fx_scaled, fy_scaled = fx * scale_w, fy * scale_h
            cx_scaled, cy_scaled = cx * scale_w, cy * scale_h
            ck = np.array(
                [[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]]
            )
            img = img.resize((tgt_reso[1], tgt_reso[0]))
            resize_flag = True
        return np.array(img), ck, resize_flag

    imgs, cks = [], []
    rel_depths = [] if load_rel_depth else None
    masks = []

    processed_root = Path(processed_root)

    for img_path in img_paths:
        img_path = Path(img_path)
        # 支持 processed/images_small 与 raw 路径
        seq_id, cam_name, frame_id = parse_seq_cam_frame(img_path)

        # 读取内参
        param_path = processed_root / "params_small" / seq_id / cam_name / f"{frame_id}.json"
        with open(param_path, "r", encoding="utf-8") as f:
            param = json.load(f)
        ck = np.array(param["camera_intrinsic"], dtype=np.float32)

        # 读取图像并同步缩放内参
        img = Image.open(img_path)
        img, ck, resize_flag = maybe_resize(img, reso, ck)
        ck[0, :] = ck[0, :] / reso[1]
        ck[1, :] = ck[1, :] / reso[0]
        img = HWC3(img)
        imgs.append(img)
        cks.append(ck)

        # Metric3D-v2 尺度深度（用于 PCC）
        if load_rel_depth:
            depth_path = processed_root / "dptm" / seq_id / cam_name / f"{frame_id}_dpt.npy"
            depth = np.load(depth_path).astype(np.float32)
            if resize_flag:
                depth = Image.fromarray(depth)
                depth = depth.resize((reso[1], reso[0]), Image.BILINEAR)
                depth = np.array(depth)
            rel_depths.append(depth)

        # PandaSet 视为全静态
        mask = np.ones(tuple(reso), dtype=np.float32)
        masks.append(mask)

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    masks = torch.from_numpy(np.stack(masks, axis=0)).bool()
    cks = torch.as_tensor(cks, dtype=torch.float32)
    rel_depths_tensor = (
        None
        if rel_depths is None
        else torch.from_numpy(np.stack(rel_depths, axis=0)).float()
    )

    return imgs, masks, cks, rel_depths_tensor
