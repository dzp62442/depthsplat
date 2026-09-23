"""Evaluation-only masks: synthetic CPU contracts and opt-in SVF-GS comparison."""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.shims.patch_shim import apply_patch_shim
from src.dataset.utils_ddad import DDADEgoMasks, file_digest, read_json, load_bin_info
from src.evaluation import metrics
from src.evaluation.ego_mask import build_eval_mask_config, resolve_eval_output_dir, validate_eval_mask_mode
from src.evaluation.zero_shot import check_existing_pixel_identity, evaluation_pixel_identity, summarize_records, view_group_records, write_zero_shot_results
from test_zero_shot import fixture, write_json, digest_bytes


def config(root=None, enabled=True, resolution="112x200"):
    overrides = [f"+experiment=ddad_{resolution}", "mode=test", f"test.eval_use_ego_mask={str(enabled).lower()}"]
    if root is not None:
        overrides.append(f"dataset.processed_root={root}")
    with initialize(version_base=None, config_path="../config"):
        return load_typed_root_config(compose(config_name="main", overrides=overrides))


def mask_fixture(root):
    fixture(root, "ddad")
    selection = read_json(root / "selection_test.json")
    infos = [pickle.loads(p.read_bytes()) for p in sorted((root / "bin_infos").glob("*.pkl"))]
    selection["assets"] = {s["asset_id"]: dict(asset_id=s["asset_id"], intrinsic=s["intrinsic"])
                           for info in infos for sensors in info["sensor_info"].values() for s in sensors}
    payload = {k: v for k, v in selection.items() if k != "selection_sha256"}
    sha = digest_bytes(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
    selection["selection_sha256"] = sha
    write_json(root / "selection_test.json", selection)
    manifest = read_json(root / "manifest_test.json")
    manifest["selection_sha256"] = sha
    for i, info in enumerate(infos):
        info.update(scene_id=str(i), selection_sha256=sha)
        for sensors in info["sensor_info"].values():
            for sensor in sensors:
                for key in ("intrinsic_path", "depth_meta_path"):
                    value = read_json(root / sensor[key])
                    value["selection_sha256"] = sha
                    if key == "intrinsic_path":
                        # The existing fixture's centered K has an identity crop at this raw size.
                        value["source_hw"] = [225, 401]
                    write_json(root / sensor[key], value)
        path = root / "bin_infos" / f"{info['bin_token']}.pkl"
        path.write_bytes(pickle.dumps(info))
        manifest["bin_info_sha256"][info["bin_token"]] = file_digest(path)
    write_json(root / "manifest_test.json", manifest)
    write_json(root / "bins_test.json", dict(bins=[i["bin_token"] for i in infos], selection_sha256=sha))
    cfg = build_eval_mask_config()
    directory = root / Path(cfg["manifest_path"]).parent
    directory.mkdir(parents=True)
    sources, masks, variants = {}, {}, {}
    for i, sensors in enumerate(infos[0]["sensor_info"].values()):
        sensor = sensors[0]
        camera = sensor["camera"]
        pixels = np.full((224, 400), 255, np.uint8)
        if i:
            pixels[-i * 12:, :] = 0
            pixels[:, :i * 4] = 0
        path = directory / f"{camera}.png"
        Image.fromarray(pixels).save(path)
        sources[camera] = masks[camera] = dict(path=path.name, sha256=file_digest(path))
        variants[camera] = dict(camera=camera, mask_id=camera, source_sha256=file_digest(path),
                                source_hw=[225, 401], intrinsic_raw=sensor["intrinsic"],
                                camera_intrinsic=sensor["intrinsic"], pixel_transform=np.eye(3).tolist())
    mask_manifest = dict(schema=cfg["schema"], dataset="ddad", source_commit=cfg["source_commit"],
                         selection_sha256=sha, selection_file_sha256=file_digest(root / "selection_test.json"),
                         camera_order=list(sources), source_masks=sources, masks=masks, variants=variants,
                         scene_camera_to_variant={str(i): {c: c for c in sources} for i in range(2)},
                         mask_values={"0": "exclude ego-vehicle region", "255": "valid region"},
                         transform=dict(name="principal_center_float_v1", interpolation="INTER_NEAREST", image_hw=[224, 400]),
                         quality_note="synthetic unit-test masks only")
    write_json(root / cfg["manifest_path"], mask_manifest)


class MaskLoaderTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="depthsplat-mask-test-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        mask_fixture(self.root)

    def test_order_resize_crop_and_unchanged_inputs(self):
        for resolution, hw in (("112x200", (112, 200)), ("224x400", (224, 400))):
            ds = get_dataset(config(self.root, resolution=resolution).dataset, "test", None)
            plain = get_dataset(config(self.root, False, resolution).dataset, "test", None)[0]
            data = ds[0]
            area = data["target"]["eval_mask"]
            self.assertEqual(area.shape, (18, *hw))
            self.assertTrue(area[12:].all() and area[:2].all())
            for i, camera in enumerate(ds.camera_map.values()):
                with Image.open(self.root / "ego_masks/vidar_v1" / f"{camera}.png") as image:
                    expected = torch.from_numpy(np.asarray(image.resize(hw[::-1], Image.Resampling.NEAREST)) == 255)
                self.assertTrue(torch.equal(area[i*2], expected) and torch.equal(area[i*2+1], expected))
            for group in ("context", "target"):
                for key, value in plain[group].items():
                    self.assertTrue(torch.equal(value, data[group][key]))
            batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
            crop = apply_patch_shim(batch, 16)
            expected = area[..., 4:-4] if hw[1] == 200 else area
            self.assertTrue(torch.equal(crop["target"]["eval_mask"][0], expected))
            ds[1]
            self.assertEqual(len(ds.ego_masks.cache), 6)
        for stage in ("train", "val"):
            with self.assertRaisesRegex(ValueError, "test stage"):
                get_dataset(config(self.root).dataset, stage, None)

    def test_corrupted_assets_rejected(self):
        path = self.root / build_eval_mask_config()["manifest_path"]
        original = read_json(path)
        for corruption in ("sha", "camera", "intrinsic", "crop", "scene", "selection"):
            with self.subTest(corruption=corruption):
                value = deepcopy(original)
                first = next(iter(value["variants"]))
                if corruption == "sha": value["masks"][first]["sha256"] = "wrong"
                elif corruption == "camera": value["variants"][first]["camera"] = "wrong"
                elif corruption == "intrinsic": value["variants"][first]["intrinsic_raw"][0][0] += 1
                elif corruption == "crop": value["variants"][first]["pixel_transform"][0][2] += 1
                elif corruption == "scene": value["scene_camera_to_variant"].clear()
                else: value["selection_file_sha256"] = "wrong"
                write_json(path, value)
                with self.assertRaises(ValueError):
                    get_dataset(config(self.root).dataset, "test", None)[0]
        write_json(path, original)
        (path.parent / original["masks"][first]["path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            get_dataset(config(self.root).dataset, "test", None)
        # Disabled loading must not depend on any mask files.
        get_dataset(config(self.root, False).dataset, "test", None)[0]

    def test_nonbinary_empty_and_wrong_shape_masks_rejected(self):
        path = self.root / build_eval_mask_config()["manifest_path"]
        original = read_json(path)
        camera = original["camera_order"][1]
        png = path.parent / original["masks"][camera]["path"]
        for shape, value in (((224, 400), 127), ((224, 400), 0), ((112, 200), 255)):
            with self.subTest(shape=shape, value=value):
                Image.fromarray(np.full(shape, value, np.uint8)).save(png)
                manifest = deepcopy(original)
                sha = file_digest(png)
                manifest["masks"][camera]["sha256"] = sha
                manifest["source_masks"][camera]["sha256"] = sha
                manifest["variants"][camera]["source_sha256"] = sha
                write_json(path, manifest)
                with self.assertRaises(ValueError):
                    get_dataset(config(self.root).dataset, "test", None)[0]


class MaskConfigTests(unittest.TestCase):
    def test_routing_and_mode_validation(self):
        for resolution in ("112x200", "224x400"):
            with initialize(version_base=None, config_path="../config"):
                raw = compose(config_name="main", return_hydra_config=True, overrides=[
                    f"+experiment=ddad_{resolution}", "mode=test", "test.eval_use_ego_mask=true",
                    "output_dir=/tmp/masked-eval"])
            self.assertEqual(raw.hydra.run.dir, "/tmp/masked-eval_ego_novel12_v1/logs/hydra")
            self.assertTrue(config(enabled=True, resolution=resolution).dataset.eval_use_ego_mask)
        out = resolve_eval_output_dir("/tmp/run/", "test", "ddad", True)
        self.assertEqual(resolve_eval_output_dir(out, "test", "ddad", True), out)
        for mode, dataset in (("train", "ddad"), ("test", "pandaset"), ("test", "omniscene")):
            with self.assertRaises(ValueError): validate_eval_mask_mode(mode, dataset, True)
        with self.assertRaises(ValueError): validate_eval_mask_mode("test", "ddad", "true")

    def test_hydra_launch_writes_only_resolved_directory(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="depthsplat-hydra-mask-") as temp:
            code = f'''import hydra
from src.evaluation.ego_mask import resolve_eval_output_dir
@hydra.main(version_base=None, config_path={str(project / "config")!r}, config_name="main")
def run(cfg):
    print(resolve_eval_output_dir(cfg.output_dir, cfg.mode, cfg.dataset.name, cfg.test.eval_use_ego_mask))
run()
'''
            command = [sys.executable, "-B", "-c", code, "+experiment=ddad_112x200", "mode=test",
                       "test.eval_use_ego_mask=true", f"output_dir={temp}/run"]
            subprocess.run(command, cwd=project, check=True, capture_output=True, text=True, timeout=30)
            self.assertTrue((Path(temp) / "run_ego_novel12_v1/logs/hydra/.hydra/config.yaml").is_file())
            self.assertFalse((Path(temp) / "run").exists())


class MaskMetricTests(unittest.TestCase):
    def test_real_lpips_invariance_and_full_view_regression(self):
        torch.manual_seed(20)
        gt = torch.rand(2, 3, 48, 64)
        pred = (gt + .07 * torch.randn_like(gt)).clip(0, 1)
        mask = torch.ones(2, 48, 64, dtype=torch.bool)
        mask[0, 31:] = False
        cfg = build_eval_mask_config()
        plain = metrics.compute_image_metrics(gt, pred)
        full = metrics.compute_image_metrics(gt, pred, torch.ones_like(mask), cfg)
        masked = metrics.compute_image_metrics(gt, pred, mask, cfg)
        changed = pred.clone()
        changed[0, :, 31:] = 1 - changed[0, :, 31:]
        ignored = metrics.compute_image_metrics(gt, changed, mask, cfg)
        changed[0, :, :12] = 1 - changed[0, :, :12]
        valid_change = metrics.compute_image_metrics(gt, changed, mask, cfg)
        for name in plain:
            self.assertTrue(torch.equal(full[name], plain[name]))
            self.assertTrue(torch.equal(masked[name][1], plain[name][1]))
            self.assertTrue(torch.equal(masked[name], ignored[name]))
            self.assertFalse(torch.isclose(masked[name][0], valid_change[name][0]))

    def test_psnr_ssim_support_and_pcc(self):
        cfg = build_eval_mask_config()
        gt, pred = torch.full((2, 3, 32, 40), .2), torch.full((2, 3, 32, 40), .3)
        mask = torch.ones(2, 32, 40, dtype=torch.bool)
        mask[0, 22:] = False
        pred[0, :, 22:] = .9
        result = metrics.compute_image_metrics(gt, pred, mask, cfg)
        torch.testing.assert_close(result["psnr"], torch.full((2,), 20.), rtol=1e-5, atol=1e-5)
        _, ssim_map = structural_similarity(gt[0].numpy(), pred[0].numpy(), win_size=11,
                                            gaussian_weights=True, channel_axis=0, data_range=1., full=True)
        self.assertAlmostEqual(result["ssim"][0].item(), float(ssim_map[:, 5:17, 5:-5].mean()), places=6)
        for height, message in ((0, "no valid pixels"), (5, "no valid SSIM windows")):
            empty = torch.zeros_like(mask)
            empty[:, :height] = True
            with self.assertRaisesRegex(ValueError, message): metrics.compute_image_metrics(gt, pred, empty, cfg)
        depth = torch.arange(20.).reshape(1, 4, 5)
        rendered = depth * 2 + 1
        valid = torch.ones_like(depth, dtype=torch.bool)
        valid[:, -1] = False
        rendered[:, -1] = -100
        self.assertAlmostEqual(metrics.compute_eval_pcc(depth, rendered, valid).item(), 1., places=6)
        with self.assertRaisesRegex(ValueError, "nonconstant"):
            metrics.compute_eval_pcc(depth, torch.ones_like(depth), valid)

    def test_group_identity_and_rerun_protection(self):
        cfg = build_eval_mask_config()
        depth = torch.arange(18 * 12 * 12.).reshape(18, 12, 12)
        mask = torch.ones_like(depth, dtype=torch.bool)
        mask[:12, -2:] = False
        scores = {k: torch.arange(18.) for k in ("psnr", "ssim", "lpips")}
        rows = view_group_records("one", "scene", scores, depth, depth * 2 + 1, mask, cfg, "mask-sha")
        self.assertEqual([r["psnr"] for r in rows], [8.5, 5.5, 14.5])
        summary = summarize_records(rows, ["one"])
        self.assertEqual(summary["final/input_6"]["pixel_protocol"], cfg["pixel_protocol"])
        mixed = deepcopy(rows)
        mixed[0]["mask_manifest_sha256"] = "another"
        with self.assertRaisesRegex(ValueError, "Mixed"): summarize_records(mixed, ["one"])
        for invalid in (None, ""):
            with self.assertRaisesRegex(ValueError, "matching mask"):
                evaluation_pixel_identity([dict(pixel_protocol=cfg["pixel_protocol"], mask_manifest_sha256=invalid)])
        metadata = dict(bin_tokens=["one"], pixel_protocol=cfg["pixel_protocol"], mask_manifest_sha256="mask-sha")
        with tempfile.TemporaryDirectory(prefix="depthsplat-mask-results-") as temp:
            write_zero_shot_results(temp, rows, metadata)
            original = (Path(temp) / "evaluation_summary.json").read_bytes()
            for invalid in ({}, {**metadata, "mask_manifest_sha256": "changed"}):
                with self.assertRaisesRegex(ValueError, "differs"): check_existing_pixel_identity(temp, invalid)
            self.assertEqual((Path(temp) / "evaluation_summary.json").read_bytes(), original)


@unittest.skipUnless(os.environ.get("DEPTHSPLAT_SVFGS_REFERENCE"), "SVF-GS metrics opt-in")
class ReferenceTests(unittest.TestCase):
    def test_identical_metrics_to_svfgs(self):
        path = Path(os.environ["DEPTHSPLAT_SVFGS_REFERENCE"]) / "tools/metrics.py"
        spec = importlib.util.spec_from_file_location("svfgs_reference_metrics", path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        cfg = build_eval_mask_config()
        for hw in ((112, 192), (224, 400)):
            torch.manual_seed(23)
            gt = torch.rand(2, 3, *hw)
            pred = (gt + .1 * torch.randn_like(gt)).clip(0, 1)
            mask = torch.ones(2, *hw, dtype=torch.bool)
            mask[0, -hw[0] // 3:] = False
            actual = metrics.compute_image_metrics(gt, pred, mask, cfg)
            expected = reference.compute_image_metrics(gt, pred, mask, cfg)
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(metrics.compute_eval_pcc(gt[:, 0], pred[:, 0], mask),
                                       reference.compute_eval_pcc(gt[:, 0], pred[:, 0], mask), rtol=1e-5, atol=1e-6)


@unittest.skipUnless(os.environ.get("DEPTHSPLAT_REAL_DATA") == "1", "Local datasets opt-in")
class RealMaskTests(unittest.TestCase):
    def test_all_ddad_mask_references(self):
        ds = get_dataset(config().dataset, "test", None)
        self.assertEqual(len(ds), 324)
        for token in ds.bin_tokens:
            info = load_bin_info(ds.processed_root, token, ds.manifest, ds.camera_map)
            novel = [sensor for cam in ds.camera_types for sensor in info["sensor_info"][cam][1:]]
            mask = ds.ego_masks.load(str(info["scene_id"]), novel)
            self.assertEqual(mask.shape, (12, 112, 200))
            self.assertTrue(mask.reshape(12, -1).any(1).all())
        self.assertEqual(len(ds.ego_masks.cache), 19)
        self.assertFalse(torch.cuda.is_initialized())
