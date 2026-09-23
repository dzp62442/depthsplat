"""DDAD processed assets. Poses remain OpenCV c2w for DepthSplat."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import torch

SCHEMA = "svfgs_temporal18_v1"


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def asset_path(processed_root, relative):
    root = Path(processed_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset escapes processed_root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_manifest(processed_root, dataset, camera_map, split):
    root = Path(processed_root)
    manifest = read_json(root / f"manifest_{split}.json")
    selection = read_json(root / f"selection_{split}.json")
    listing = read_json(root / f"bins_{split}.json")
    payload = {k: v for k, v in selection.items() if k != "selection_sha256"}
    checksum = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode()).hexdigest()
    protocol = manifest["protocol"]
    if (manifest["schema"] != SCHEMA or selection["schema"] != SCHEMA or
            manifest["dataset"] != dataset or selection["dataset"] != dataset or
            selection["split"] != split or not manifest["complete"]):
        raise ValueError(f"Expected completed {dataset} temporal18 data: {root}")
    if not checksum == selection["selection_sha256"] == manifest["selection_sha256"] == listing["selection_sha256"]:
        raise ValueError(f"Selection identity mismatch: {root}")
    if (protocol != selection["protocol"] or protocol["schema"] != SCHEMA or
            protocol["dataset"] != dataset or protocol["camera_order"] != list(camera_map) or
            protocol["camera_map"] != camera_map or protocol["image_hw"] != [224, 400] or
            protocol["depth_reference"] != "metric3d_v2" or
            protocol["crop_method"] != "principal_center_float_v1"):
        raise ValueError(f"Unsupported temporal18 protocol: {root}")
    tokens = listing["bins"]
    if (not tokens or len(set(tokens)) != len(tokens) or
            tokens != [row["bin_token"] for row in selection["bins"]] or
            manifest["num_bins"] != len(tokens) or set(manifest["bin_info_sha256"]) != set(tokens)):
        raise ValueError(f"Published bin coverage mismatch: {root}")
    return manifest, tokens


def load_bin_info(processed_root, token, manifest, camera_map):
    path = asset_path(processed_root, f"bin_infos/{token}.pkl")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["bin_info_sha256"][token]:
        raise ValueError(f"Bin identity mismatch: {token}")
    info = pickle.loads(raw)
    if (info["schema"] != SCHEMA or info["bin_token"] != token or not info["scene_id"] or
            info["selection_sha256"] != manifest["selection_sha256"]):
        raise ValueError(f"Bin schema/selection mismatch: {token}")
    indices = info["frame_indices"]
    if not indices["before"] < indices["center"] < indices["after"]:
        raise ValueError(f"Invalid temporal order: {token}")
    for camera, physical_name in camera_map.items():
        sensors = info["sensor_info"][camera]
        if len(sensors) != 3 or any(s["camera"] != physical_name for s in sensors):
            raise ValueError(f"Expected [center, before, after] for {token}/{camera}")
        if len({s["asset_id"] for s in sensors}) != 3:
            raise ValueError(f"Repeated temporal asset: {token}/{camera}")
    return info


def load_info(info: dict, data_root: Path, processed_root: Path | None = None):
    """Read an indexed image and its OpenCV camera-to-center-LiDAR pose."""
    root = Path(processed_root) if processed_root is not None else Path(data_root) / "processed"
    path = asset_path(root, info["data_path"])
    c2w = np.asarray(info["sensor2lidar_transform"], dtype=np.float32)
    if (c2w.shape != (4, 4) or not np.isfinite(c2w).all() or
            not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-4)):
        raise ValueError(f"Invalid OpenCV c2w: {path}")
    # Keep the existing row-vector w2c return convention; Dataset uses c2w.
    return str(path), c2w, np.linalg.inv(c2w).T.copy()


def load_conditions(img_paths: list[str], reso: list[int], processed_root: Path,
                    load_rel_depth: bool = False, infos: list[dict] | None = None,
                    manifest: dict | None = None):
    """Load indexed RGB/K and optional metric-depth references, without axis flips."""
    if infos is None or manifest is None or len(img_paths) != len(infos):
        raise ValueError("Temporal18 loading requires matching sensor records and a manifest")
    height, width = reso
    if min(height, width) <= 0:
        raise ValueError(f"Invalid image_shape: {reso}")
    source_h, source_w = manifest["protocol"]["image_hw"]
    images, intrinsics, depths = [], [], []
    for img_path, info in zip(img_paths, infos):
        path = asset_path(processed_root, info["data_path"])
        if path != Path(img_path).resolve():
            raise ValueError(f"Image/sensor mismatch: {img_path}")
        param = read_json(asset_path(processed_root, info["intrinsic_path"]))
        if (param["selection_sha256"] != manifest["selection_sha256"] or
                param["asset_id"] != info["asset_id"] or param["image_hw"] != [source_h, source_w] or
                file_digest(path) != param["image_sha256"]):
            raise ValueError(f"RGB/selection identity mismatch: {path}")
        k = np.asarray(param["camera_intrinsic"], dtype=np.float32)
        expected_k = np.asarray(param["pixel_transform"]) @ np.asarray(info["intrinsic"])
        if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or
                not np.allclose(k, expected_k, rtol=1e-5, atol=1e-5) or
                not np.allclose(k[:2, 2], [source_w / 2, source_h / 2], atol=1e-4)):
            raise ValueError(f"Invalid processed intrinsics: {path}")
        with Image.open(path) as source:
            if source.size != (source_w, source_h):
                raise ValueError(f"Invalid source image shape: {path}")
            image = source.convert("RGB")
            if (height, width) != (source_h, source_w):
                # Match the existing PandaSet loader's PIL RGB resize.
                image = image.resize((width, height), Image.Resampling.BICUBIC)
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
        k = k.copy()
        k[0] *= width / source_w
        k[1] *= height / source_h
        k[0] /= width
        k[1] /= height
        intrinsics.append(k)

        if load_rel_depth:
            metadata = read_json(asset_path(processed_root, info["depth_meta_path"]))
            if (metadata["synthetic"] or metadata["reference"] != "metric3d_v2" or
                    metadata["model"] != manifest["depth_model"] or
                    metadata["selection_sha256"] != manifest["selection_sha256"] or
                    metadata["image_sha256"] != param["image_sha256"]):
                raise ValueError(f"Depth/RGB identity mismatch: {path}")
            depth_path = asset_path(processed_root, info["depth_path"])
            if file_digest(depth_path) != metadata["depth_path_sha256"]:
                raise ValueError(f"Depth checksum mismatch: {depth_path}")
            depth = np.load(depth_path, allow_pickle=False)
            if (depth.shape != (source_h, source_w) or depth.dtype != np.float32 or
                    not np.isfinite(depth).all() or depth.min() < 0 or depth.max() <= 0 or
                    depth.max() > manifest["protocol"]["depth_max_m"]):
                raise ValueError(f"Invalid metric depth: {depth_path}")
            if (height, width) != (source_h, source_w):
                depth = np.asarray(Image.fromarray(depth).resize((width, height), Image.Resampling.BILINEAR))
            depths.append(depth.copy())

    rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    masks = torch.ones((len(images), height, width), dtype=torch.bool)
    ks = torch.from_numpy(np.stack(intrinsics))
    rel_depth = torch.from_numpy(np.stack(depths)) if load_rel_depth else None
    return rgb, masks, ks, rel_depth
