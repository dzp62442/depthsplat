"""CPU contracts; opt in to full local-data checks with DEPTHSPLAT_REAL_DATA=1."""
import hashlib
import importlib
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from hydra import compose, initialize
from torch.utils.data import DataLoader

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.shims.patch_shim import apply_patch_shim
from src.evaluation.zero_shot import view_group_records, summarize_records, write_zero_shot_results
from src.global_cfg import set_cfg
from src.model.model_wrapper import ModelWrapper
from omegaconf import OmegaConf


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_cfg(name, root=None, split="total", resolution="112x200"):
    overrides = [f"+experiment={name}_{resolution}", "mode=test", f"dataset.test_split={split}"]
    if root is not None:
        overrides.append(f"dataset.processed_root={root}")
    with initialize(version_base=None, config_path="../config"):
        return load_typed_root_config(compose(config_name="main", overrides=overrides))


def fixture(root, name):
    cls = getattr(importlib.import_module(f"src.dataset.dataset_{name}"),
                  "DatasetPandaSet" if name == "pandaset" else "DatasetDDAD")
    schema = "svfgs_temporal18_v1"
    tokens = [f"{name}_fixture_{i}" for i in (10, 20)]
    protocol = dict(schema=schema, dataset=name, camera_map=cls.camera_map,
                    camera_order=cls.camera_types, image_hw=[224, 400], depth_reference="metric3d_v2",
                    crop_method="principal_center_float_v1", depth_max_m=300.0)
    selection = dict(schema=schema, dataset=name, split="test", protocol=protocol,
                     bins=[dict(bin_token=t) for t in tokens])
    sha = digest_bytes(json.dumps(selection, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
    selection["selection_sha256"] = sha
    model = {"reference": "metric3d_v2", "checkpoint_sha256": "unit-test-only"}
    hashes = {}
    for bin_index, token in enumerate(tokens):
        sensors = {}
        for cam_index, (cam, physical) in enumerate(cls.camera_map.items()):
            sensors[cam] = []
            for role in range(3):
                stem = f"{bin_index}/{physical}/{role}"
                rgb_path = root / f"images_small/{stem}.png"
                rgb_path.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.full((224, 400, 3), 10 + cam_index * 20 + role * 4, dtype=np.uint8)
                Image.fromarray(pixels).save(rgb_path)
                image_sha = digest_bytes(rgb_path.read_bytes())
                depth_path = root / f"depth/{stem}.npy"
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(depth_path, np.full((224, 400), 5 + cam_index + role, dtype=np.float32))
                k = [[300 + cam_index, 0, 200], [0, 310 + cam_index, 112], [0, 0, 1]]
                info = dict(asset_id=stem, camera=physical, data_path=str(rgb_path.relative_to(root)),
                            intrinsic_path=f"params/{stem}.json", depth_path=str(depth_path.relative_to(root)),
                            depth_meta_path=f"depth/{stem}_meta.json", intrinsic=k)
                pose = np.eye(4, dtype=np.float32)
                pose[:3, 3] = [cam_index, role, bin_index]
                info["sensor2lidar_transform"] = pose
                write_json(root / info["intrinsic_path"], dict(selection_sha256=sha, asset_id=stem,
                           image_hw=[224, 400], image_sha256=image_sha, camera_intrinsic=k,
                           pixel_transform=np.eye(3).tolist()))
                write_json(root / info["depth_meta_path"], dict(synthetic=False, reference="metric3d_v2",
                           model=model, selection_sha256=sha, image_sha256=image_sha,
                           depth_path_sha256=digest_bytes(depth_path.read_bytes())))
                sensors[cam].append(info)
        info = dict(schema=schema, bin_token=token, scene_id="fixture", selection_sha256=sha,
                    frame_indices=dict(before=8, center=10, after=12), sensor_info=sensors)
        pkl_path = root / f"bin_infos/{token}.pkl"
        pkl_path.parent.mkdir(exist_ok=True)
        raw = pickle.dumps(info)
        pkl_path.write_bytes(raw)
        hashes[token] = digest_bytes(raw)
    write_json(root / "selection_test.json", selection)
    write_json(root / "bins_test.json", dict(bins=tokens, selection_sha256=sha))
    write_json(root / "manifest_test.json", dict(schema=schema, dataset=name, complete=True,
               num_bins=2, selection_sha256=sha, protocol=protocol, depth_model=model, bin_info_sha256=hashes))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="depthsplat-loader-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("pandaset", "ddad"):
            fixture(self.root / name, name)

    def dataset(self, name="pandaset", split="total", stage="test"):
        return get_dataset(make_cfg(name, self.root / name, split).dataset, stage, None)

    def test_order_geometry_and_patch_shim(self):
        for name in ("pandaset", "ddad"):
            ds = self.dataset(name)
            data = ds[0]
            self.assertEqual(data["scene_id"], "fixture")
            self.assertEqual(data["context"]["image"].shape, (6, 3, 112, 200))
            self.assertEqual(data["target"]["image"].shape, (18, 3, 112, 200))
            for key in ("image", "intrinsics", "extrinsics", "near", "far"):
                torch.testing.assert_close(data["target"][key][12:], data["context"][key], rtol=0, atol=0)
            # Each camera's past/future, followed by all centers.
            expected_roles = [1, 2] * 6 + [0] * 6
            self.assertEqual(data["target"]["extrinsics"][:, 1, 3].tolist(), expected_roles)
            torch.testing.assert_close(data["context"]["intrinsics"][:, 0, 0],
                                       torch.arange(300, 306).float() / 400)
            self.assertTrue(torch.all(data["target"]["intrinsics"][:, :2, 2] == 0.5))
            self.assertEqual(data["target"]["rel_depth"][0, 0, 0].item(), 6)
            batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
            cropped = apply_patch_shim(batch, 16)
            self.assertEqual(cropped["target"]["image"].shape, (1, 18, 3, 112, 192))
            self.assertEqual(cropped["target"]["rel_depth"].shape, (1, 18, 112, 192))
            self.assertEqual(cropped["evaluation_protocol"], ["svfgs_temporal18_v1"])
            # Center crop subtracts four pixels, with normalized focal length adjusted.
            old = batch["target"]["intrinsics"][0, 0]
            new = cropped["target"]["intrinsics"][0, 0]
            x = 0.2
            self.assertAlmostEqual((old[0, 0]*x + old[0, 2]).item()*200 - 4,
                                   (new[0, 0]*x + new[0, 2]).item()*192, places=4)

    def test_subsets_and_no_train_fallback(self):
        for name in ("pandaset", "ddad"):
            for split in ("total", "mini", "demo"):
                ds = self.dataset(name, split)
                self.assertEqual(len(ds), 2)
                self.assertEqual(len(set(ds.bin_tokens)), 2)
            ds = self.dataset(name, stage="val")
            self.assertEqual(len(ds), 2)
            self.assertNotIn("rel_depth", ds[0]["target"])
            with self.assertRaises(FileNotFoundError):
                self.dataset(name, stage="train")

    def test_target_gt_cannot_change_encoder_inputs(self):
        for name in ("pandaset", "ddad"):
            batch = next(iter(DataLoader(self.dataset(name), batch_size=1, num_workers=0)))
            context = {key: value.clone() for key, value in batch["context"].items()}
            batch["target"]["image"][:, :12] = 0
            batch["target"]["rel_depth"][:, :12] = 999
            for key, value in context.items():
                torch.testing.assert_close(batch["context"][key], value, rtol=0, atol=0)
            self.assertFalse(torch.cuda.is_initialized())

    def test_manifest_and_selection_rejection(self):
        for field, value in [("complete", False), ("schema", "old"), ("dataset", "wrong")]:
            path = self.root / "pandaset/manifest_test.json"
            original = json.loads(path.read_text())
            invalid = {**original, field: value}
            write_json(path, invalid)
            with self.assertRaises(ValueError):
                self.dataset()
            write_json(path, original)
        path = self.root / "pandaset/selection_test.json"
        value = json.loads(path.read_text())
        value["bins"].reverse()
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.dataset()

    def test_explicit_processed_root_and_changed_bin(self):
        ds = self.dataset("ddad")
        self.assertEqual(ds.evaluation_metadata()["processed_root"], str(self.root / "ddad"))
        path = self.root / "ddad/bin_infos" / f"{ds.bin_tokens[0]}.pkl"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "identity"):
            ds[0]

    def test_missing_depth_and_changed_intrinsics(self):
        for name in ("pandaset", "ddad"):
            ds = self.dataset(name)
            info = pickle.loads((self.root / name / "bin_infos" / f"{ds.bin_tokens[0]}.pkl").read_bytes())
            sensor = info["sensor_info"]["CAM_FRONT"][1]
            depth = self.root / name / sensor["depth_path"]
            depth.unlink()
            with self.assertRaises(FileNotFoundError):
                ds[0]
            k_path = self.root / name / info["sensor_info"]["CAM_FRONT"][0]["intrinsic_path"]
            value = json.loads(k_path.read_text())
            value["camera_intrinsic"][0][0] += 10
            write_json(k_path, value)
            with self.assertRaisesRegex(ValueError, "intrinsics"):
                ds[0]

    def test_wrong_camera_and_temporal_copy_rejected(self):
        for name in ("pandaset", "ddad"):
            ds = self.dataset(name)
            root = self.root / name
            path = root / "bin_infos" / f"{ds.bin_tokens[0]}.pkl"
            info = pickle.loads(path.read_bytes())
            info["sensor_info"]["CAM_FRONT"][1]["camera"] = "wrong"
            raw = pickle.dumps(info)
            path.write_bytes(raw)
            # Re-sign the bin to exercise semantic checks independently of hashes.
            ds.manifest["bin_info_sha256"][ds.bin_tokens[0]] = digest_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Expected"):
                ds[0]

    def test_configuration_registration_and_existing_defaults(self):
        for name in ("pandaset", "ddad"):
            cfg = make_cfg(name, self.root / name, resolution="224x400")
            ds = get_dataset(cfg.dataset, "test", None)
            self.assertEqual(ds[0]["target"]["image"].shape, (18, 3, 224, 400))
        for name in ("omniscene_112x200", "re10k"):
            with initialize(version_base=None, config_path="../config"):
                cfg = load_typed_root_config(compose(config_name="main", overrides=[f"+experiment={name}"]))
            self.assertFalse(hasattr(cfg.dataset, "processed_root"))
        self.assertFalse(torch.cuda.is_initialized())

    def test_test_start_invalidates_previous_summary(self):
        out = self.root / "smoke"
        set_cfg(OmegaConf.create({"output_dir": str(out)}))
        write_json(out / "metrics/evaluation_summary.json", {"complete": True})
        wrapper = SimpleNamespace(
            trainer=SimpleNamespace(world_size=1, test_dataloaders=SimpleNamespace(dataset=self.dataset())),
            train_cfg=SimpleNamespace(forward_depth_only=False),
            test_cfg=SimpleNamespace(stablize_camera=False, compute_scores=True),
            checkpoint_provenance={"path": "fixture-only"})
        ModelWrapper.on_test_start(wrapper)
        summary = json.loads((out / "metrics/evaluation_summary.json").read_text())
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["expected_bins"], 2)
        self.assertEqual(wrapper.zero_shot_metadata["dataset"], "pandaset")


class ConfigTests(unittest.TestCase):
    def test_zero_shot_log_directory_follows_output_directory(self):
        for name in ("pandaset", "ddad"):
            for resolution in ("112x200", "224x400"):
                with self.subTest(dataset=name, resolution=resolution):
                    with initialize(version_base=None, config_path="../config"):
                        cfg = compose(config_name="main", return_hydra_config=True, overrides=[
                            f"+experiment={name}_{resolution}", "mode=test",
                            "output_dir=/tmp/depthsplat-config-test"])
                    self.assertEqual(cfg.hydra.run.dir, "/tmp/depthsplat-config-test/logs/hydra")
                    self.assertEqual(cfg.dataset.test_split, "total")
                    cfg.output_dir = "/tmp/depthsplat-config-test-another"
                    self.assertEqual(cfg.hydra.run.dir, "/tmp/depthsplat-config-test-another/logs/hydra")
        for name in ("omniscene_112x200", "re10k"):
            with initialize(version_base=None, config_path="../config"):
                cfg = compose(config_name="main", return_hydra_config=True,
                              overrides=[f"+experiment={name}"])
            self.assertNotIn("${output_dir}", OmegaConf.to_container(cfg.hydra, resolve=False)["run"]["dir"])


class MetricTests(unittest.TestCase):
    def records(self, token="one"):
        values = torch.arange(18, dtype=torch.float32)
        reference = torch.arange(18 * 4 * 5, dtype=torch.float32).reshape(18, 4, 5)
        predicted = reference * 2 + 7
        predicted[:12] = torch.flip(predicted[:12], dims=[0])
        metrics = dict(psnr=values, ssim=values / 20, lpips=values / 30)
        rows = view_group_records(token, "scene", metrics, reference, predicted)
        for row in rows:
            selection = {"all_18": slice(18), "novel_12": slice(12), "input_6": slice(12, 18)}[row["view_group"]]
            corr = torch.corrcoef(torch.stack([reference[selection].flatten(), predicted[selection].flatten()]))[0, 1]
            self.assertAlmostEqual(row["pcc"], corr.item(), places=5)
        return rows

    def test_group_means_and_pcc(self):
        rows = self.records()
        self.assertEqual([r["psnr"] for r in rows], [8.5, 5.5, 14.5])
        rows += self.records("two")
        summary = summarize_records(rows, ["one", "two"])
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["primary_result"], "final/all_18")
        self.assertEqual(summary["final/all_18"]["num_bins"], 2)
        self.assertEqual(summary["final/all_18"]["psnr"], 8.5)

    def test_missing_duplicate_nonfinite(self):
        rows = self.records()
        self.assertFalse(summarize_records(rows, ["one", "two"])["complete"])
        self.assertFalse(summarize_records(rows + rows[:1], ["one"])["complete"])
        rows[0]["pcc"] = float("nan")
        with tempfile.TemporaryDirectory(prefix="depthsplat-metrics-test-") as directory:
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                write_zero_shot_results(directory, rows, {"bin_tokens": ["one"]})
            summary = json.loads((Path(directory) / "evaluation_summary.json").read_text())
            self.assertFalse(summary["complete"])
            self.assertIsNone(summary["final/all_18"]["pcc"])


@unittest.skipUnless(os.environ.get("DEPTHSPLAT_REAL_DATA") == "1", "Local datasets opt-in")
class RealDataTests(unittest.TestCase):
    def test_all_published_bins(self):
        for name, count in (("pandaset", 264), ("ddad", 324)):
            ds = get_dataset(make_cfg(name).dataset, "test", None)
            self.assertEqual(len(ds), count)
            for idx in range(len(ds)):
                batch = ds[idx]
                self.assertEqual(batch["target"]["image"].shape, (18, 3, 112, 200))
                self.assertTrue(torch.isfinite(batch["target"]["rel_depth"]).all())
                for key in ("image", "intrinsics", "extrinsics"):
                    self.assertTrue(torch.equal(batch["context"][key], batch["target"][key][12:]))
            print(f"{name}: {count} bins, 18-view RGB/K/pose/depth verified", flush=True)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
