"""DDAD processed assets. Poses remain OpenCV c2w for DepthSplat."""

import hashlib
import json
import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..evaluation.ego_mask import build_eval_mask_config

SCHEMA = "svfgs_temporal18_v1"


def centered_crop(intrinsic, source_hw, target_hw):
    """SVF-GS principal_center_float_v1; validate only, never resample RGB here."""
    k = np.asarray(intrinsic, dtype=np.float64)
    height, width = source_hw
    target_h, target_w = target_hw
    cx, cy = k[0, 2], k[1, 2]
    half_h = min(cy, height - 1 - cy, min(cx, width - 1 - cx) * target_h / target_w)
    if not np.isfinite(k).all() or half_h <= 0:
        raise ValueError("Principal point must be strictly inside the source image")
    half_w = half_h * target_w / target_h
    sx, sy = target_w / (2 * half_w), target_h / (2 * half_h)
    affine = np.array([[sx, 0, target_w / 2 - sx * cx],
                       [0, sy, target_h / 2 - sy * cy], [0, 0, 1]], dtype=np.float64)
    return affine, affine @ k


class DDADEgoMasks:
    """Read-only shared templates, cached per Dataset/worker at its load resolution."""

    def __init__(self, processed_root, resolution):
        self.cfg = cfg = build_eval_mask_config()
        self.processed_root = Path(processed_root)
        self.resolution = tuple(resolution)
        path = asset_path(self.processed_root, cfg["manifest_path"])
        self.root = path.parent
        self.manifest = manifest = read_json(path)
        self.manifest_sha256 = file_digest(path)
        self.cache = {}
        selection_path = asset_path(self.processed_root, "selection_test.json")
        selection = read_json(selection_path)
        if (selection["dataset"] != "ddad" or manifest["dataset"] != "ddad" or
                manifest["schema"] != cfg["schema"] or manifest["source_commit"] != cfg["source_commit"] or
                manifest["selection_sha256"] != selection["selection_sha256"] or
                manifest["selection_file_sha256"] != file_digest(selection_path)):
            raise ValueError("DDAD ego mask dataset/selection/source identity mismatch")
        if (manifest["transform"]["interpolation"] != "INTER_NEAREST" or
                manifest["transform"]["name"] != selection["protocol"]["crop_method"] or
                manifest["transform"]["image_hw"] != cfg["image_hw"] or
                cfg["image_hw"] != selection["protocol"]["image_hw"] or
                manifest["mask_values"] != {"0": "exclude ego-vehicle region", "255": "valid region"}):
            raise ValueError("DDAD ego mask geometry/encoding protocol mismatch")
        cameras = [selection["protocol"]["camera_map"][c] for c in selection["protocol"]["camera_order"]]
        if manifest["camera_order"] != cameras:
            raise ValueError("DDAD ego mask camera order mismatch")
        for entries in (manifest["source_masks"], manifest.get("cleaned_source_masks", {}), manifest["masks"]):
            for entry in entries.values():
                if file_digest(asset_path(self.root, entry["path"])) != entry["sha256"]:
                    raise ValueError("DDAD ego mask file checksum mismatch")
        for asset in selection["assets"].values():
            scene, camera, _ = asset["asset_id"].split("/")
            variant = self._variant(scene, camera)
            if not np.allclose(variant["intrinsic_raw"], asset["intrinsic"], rtol=0, atol=cfg["geometry_atol"]):
                raise ValueError(f"DDAD ego mask raw intrinsics mismatch: {scene}/{camera}")

    def _variant(self, scene, camera):
        manifest = self.manifest
        try:
            variant = manifest["variants"][manifest["scene_camera_to_variant"][scene][camera]]
            source = manifest["source_masks"][camera]
            manifest["masks"][variant["mask_id"]]
        except KeyError as exc:
            raise ValueError(f"Missing DDAD ego mask mapping: {scene}/{camera}") from exc
        if variant["camera"] != camera or variant["source_sha256"] != source["sha256"]:
            raise ValueError(f"DDAD ego mask camera/source mismatch: {scene}/{camera}")
        return variant

    def load(self, scene, infos):
        masks = []
        for info in infos:
            camera = info["camera"]
            if info["asset_id"].split("/")[:2] != [scene, camera]:
                raise ValueError("DDAD ego mask target scene/camera mismatch")
            variant = self._variant(scene, camera)
            param = read_json(asset_path(self.processed_root, info["intrinsic_path"]))
            if (param["asset_id"] != info["asset_id"] or
                    param["selection_sha256"] != self.manifest["selection_sha256"] or
                    param["source_hw"] != variant["source_hw"] or param["image_hw"] != self.cfg["image_hw"]):
                raise ValueError("DDAD ego mask target asset/shape mismatch")
            affine, intrinsic = centered_crop(info["intrinsic"], param["source_hw"], param["image_hw"])
            pairs = [(variant["intrinsic_raw"], info["intrinsic"]), (variant["pixel_transform"], affine),
                     (variant["camera_intrinsic"], intrinsic), (param["pixel_transform"], affine),
                     (param["camera_intrinsic"], intrinsic)]
            if any(not np.allclose(a, b, rtol=0, atol=self.cfg["geometry_atol"]) for a, b in pairs):
                raise ValueError("DDAD ego mask target crop/intrinsics mismatch")
            key = variant["mask_id"]
            if key not in self.cache:
                entry = self.manifest["masks"][key]
                path = asset_path(self.root, entry["path"])
                if file_digest(path) != entry["sha256"]:
                    raise ValueError("DDAD ego mask changed after initialization")
                with Image.open(path) as source:
                    pixels = np.asarray(source)
                    if (source.mode != "L" or list(pixels.shape) != self.cfg["image_hw"] or
                            not np.isin(pixels, [0, 255]).all()):
                        raise ValueError("DDAD ego mask must be a binary image of the configured size")
                    resized = source.resize(self.resolution[::-1], Image.Resampling.NEAREST)
                    valid = torch.from_numpy(np.asarray(resized) == 255)
                if not valid.any():
                    raise ValueError("DDAD ego mask has no valid pixels")
                self.cache[key] = valid
            masks.append(self.cache[key])
        return torch.stack(masks)

    def metadata(self):
        return dict(pixel_protocol=self.cfg["pixel_protocol"], mask_manifest_sha256=self.manifest_sha256,
                    manifest_path=str((self.root / "manifest.json").resolve()),
                    source_commit=self.manifest["source_commit"], settings=deepcopy(self.cfg),
                    asset_revision=self.manifest.get("asset_revision"), cleanup=self.manifest.get("cleanup"),
                    template_quality=self.manifest["quality_note"])


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
