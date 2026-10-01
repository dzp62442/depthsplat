"""DDAD-only reference-frame alignment; all temporary assets live under /tmp."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.dataset import get_dataset
from src.dataset import utils_ddad
from src.geometry.projection import get_world_rays, project
from test_ego_mask import mask_fixture
from test_zero_shot import fixture, make_cfg


# Independent expected mapping: forward -> +Y, left -> -X, up -> +Z.
ALIGN = torch.tensor([[0., -1., 0., 0.], [1., 0., 0., 0.],
                      [0., 0., 1., 0.], [0., 0., 0., 1.]])


class DDADCoordinateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="depthsplat-ddad-frame-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        mask_fixture(self.root)

    def dataset(self, stage="test", masked=False, root=None):
        cfg = make_cfg("ddad", self.root if root is None else root).dataset
        cfg.eval_use_ego_mask = masked
        return get_dataset(cfg, stage, None)

    def test_load_info_rotates_both_R_and_t_and_recomputes_inverse(self):
        ds = self.dataset()
        info = utils_ddad.load_bin_info(self.root, ds.bin_tokens[0], ds.manifest, ds.camera_map)
        sensor = info["sensor_info"]["CAM_FRONT"][0]
        c, s = np.cos(.37), np.sin(.37)
        yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        camera_to_flu = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        raw = np.eye(4, dtype=np.float32)
        raw[:3, :3] = yaw @ camera_to_flu
        raw[:3, 3] = [2.1, -3.2, 1.4]
        sensor["sensor2lidar_transform"] = raw.copy()
        _, aligned, w2c_row = utils_ddad.load_info(sensor, self.root, self.root)
        np.testing.assert_allclose(aligned, ALIGN.numpy() @ raw, rtol=0, atol=0)
        np.testing.assert_allclose(aligned[:3, 3], [3.2, 2.1, 1.4], atol=1e-6)
        np.testing.assert_allclose(w2c_row.T @ aligned, np.eye(4), atol=1e-6)
        np.testing.assert_allclose(aligned @ w2c_row.T, np.eye(4), atol=1e-6)
        np.testing.assert_array_equal(sensor["sensor2lidar_transform"], raw)
        self.assertFalse(np.shares_memory(aligned, sensor["sensor2lidar_transform"]))

    def test_all_views_stages_and_mask_modes_preserve_relative_geometry(self):
        train_root = self.root / "train_assets"
        fixture(train_root, "ddad", split="train")
        for stage, masked in (("train", False), ("val", False), ("test", False), ("test", True)):
            with self.subTest(stage=stage, masked=masked):
                ds = self.dataset(stage, masked, train_root if stage == "train" else self.root)
                # Emulate the old loader only inside this test to compare every field.
                with patch.object(utils_ddad, "DDAD_REFERENCE_TO_MODEL", np.eye(4, dtype=np.float32)):
                    original = ds[0]
                aligned = ds[0]
                for side in ("context", "target"):
                    torch.testing.assert_close(aligned[side]["extrinsics"],
                                               ALIGN @ original[side]["extrinsics"], rtol=0, atol=0)
                    for key in original[side]:
                        if key != "extrinsics":
                            torch.testing.assert_close(aligned[side][key], original[side][key], rtol=0, atol=0)
                for key in original.keys() - {"context", "target"}:
                    self.assertEqual(aligned[key], original[key])
                torch.testing.assert_close(aligned["target"]["extrinsics"][12:],
                                           aligned["context"]["extrinsics"], rtol=0, atol=0)
                old, new = original["target"]["extrinsics"], aligned["target"]["extrinsics"]
                old_relative = old[:, None].inverse() @ original["context"]["extrinsics"][None]
                new_relative = new[:, None].inverse() @ aligned["context"]["extrinsics"][None]
                torch.testing.assert_close(new_relative, old_relative, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(torch.cdist(new[:, :3, 3], new[:, :3, 3]),
                                           torch.cdist(old[:, :3, 3], old[:, :3, 3]), atol=1e-6, rtol=1e-6)
                # Downstream ray origins/directions derive from the new c2w; no stale cache.
                pixels = torch.tensor([[.2, .3], [.5, .5], [.8, .7]])
                ks = aligned["target"]["intrinsics"][:, None]
                origins, directions = get_world_rays(pixels, old[:, None], ks)
                new_origins, new_directions = get_world_rays(pixels, new[:, None], ks)
                torch.testing.assert_close(new_origins, origins @ ALIGN[:3, :3].T, rtol=0, atol=0)
                torch.testing.assert_close(new_directions, directions @ ALIGN[:3, :3].T, rtol=0, atol=0)
                # Backproject one camera's points and reproject them into all 18 cameras.
                points = origins[0] + directions[0] * torch.tensor([3., 5., 9.])[:, None]
                old_pixels, old_visible = project(points, old[:, None], ks)
                new_pixels, new_visible = project(points @ ALIGN[:3, :3].T, new[:, None], ks)
                torch.testing.assert_close(new_pixels, old_pixels, atol=2e-6, rtol=1e-5)
                self.assertTrue(torch.equal(new_visible, old_visible))
                # A second load must not apply the rotation twice to shared source records.
                torch.testing.assert_close(ds[0]["target"]["extrinsics"], new, rtol=0, atol=0)

    def test_metadata_describes_axes_without_changing_protocol(self):
        for masked in (False, True):
            ds = self.dataset(masked=masked)
            metadata = ds.evaluation_metadata()
            self.assertEqual(metadata["camera_frame"], "nuscenes_axes_x_right_y_forward_z_up")
            self.assertEqual(metadata["reference_to_model"], ALIGN.tolist())
            self.assertEqual(metadata["protocol"], ds.manifest["protocol"])
            self.assertEqual(metadata["selection_sha256"], ds.manifest["selection_sha256"])
            self.assertEqual(metadata["pixel_protocol"], "ddad_ego_novel12_v1" if masked else "full_image")
            self.assertEqual((metadata["input_views"], metadata["output_views"]), (6, 18))


@unittest.skipUnless(os.environ.get("DEPTHSPLAT_REAL_DATA") == "1", "Local datasets opt-in")
class RealDDADCoordinateTests(unittest.TestCase):
    def test_all_published_camera_poses(self):
        ds = get_dataset(make_cfg("ddad").dataset, "test", None)
        self.assertEqual(len(ds), 324)
        count = 0
        for token in ds.bin_tokens:
            info = utils_ddad.load_bin_info(ds.processed_root, token, ds.manifest, ds.camera_map)
            old_poses, new_poses = [], []
            for camera in ds.camera_types:
                for sensor in info["sensor_info"][camera]:
                    raw = np.asarray(sensor["sensor2lidar_transform"], dtype=np.float32)
                    _, aligned, w2c_row = utils_ddad.load_info(sensor, ds.data_root, ds.processed_root)
                    np.testing.assert_array_equal(aligned, ALIGN.numpy() @ raw)
                    np.testing.assert_allclose(w2c_row.T @ aligned, np.eye(4), atol=2e-5)
                    old_poses.append(raw)
                    new_poses.append(aligned)
                    count += 1
            old, new = np.stack(old_poses), np.stack(new_poses)
            np.testing.assert_allclose(np.linalg.inv(new[:, None]) @ new[None],
                                       np.linalg.inv(old[:, None]) @ old[None], atol=2e-5, rtol=1e-5)
            # Front camera's OpenCV +Z optical axis now predominantly points +Y.
            self.assertGreater(new[0, 1, 2], .9)
        self.assertEqual(count, 324 * 18)
        self.assertFalse(torch.cuda.is_initialized())
        print(f"DDAD: {count} camera poses aligned; relative poses and inverses verified", flush=True)
