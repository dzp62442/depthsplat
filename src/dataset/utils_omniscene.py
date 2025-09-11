import math
from collections import defaultdict
import json
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Basic types
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# Tensor dtype
# for jaxtyping usage, see https://github.com/google/jaxtyping/blob/main/API.md
from jaxtyping import Bool, Complex, Float, Inexact, Int, Integer, Num, Shaped, UInt

# Config type
from omegaconf import DictConfig

# PyTorch Tensor type
from torch import Tensor

# Runtime type checking decorator
from typeguard import typechecked as typechecker


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def load_info(info):  # TODO: 似乎不应该再 flip_yz
    img_path = info["data_path"]
    # use lidar coordinate of the key frame as the world coordinate
    c2w = info["sensor2lidar_transform"]
    # opencv cam -> opengl cam, maybe not necessary!
    # flip_yz = np.eye(4)
    # flip_yz[1, 1] = -1
    # flip_yz[2, 2] = -1
    # c2w = c2w@flip_yz

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    
    return img_path, c2w, w2c


def load_conditions(img_paths, reso, is_input=False, load_rel_depth=False):
    
    def maybe_resize(img, tgt_reso, ck):
        if not isinstance(img, PIL.Image.Image):
            img = Image.fromarray(img)
        resize_flag = False
        if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
            # img.resize((w, h))
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
            fx_scaled, fy_scaled, cx_scaled, cy_scaled = fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h
            ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
            img = img.resize((tgt_reso[1], tgt_reso[0]))
            resize_flag = True
        return np.array(img), ck, resize_flag
    
    imgs, cks = [], []
    depths = []
    depths_m = []
    confs_m = []
    masks = []
    for img_path in img_paths:      
        # param
        param_path = img_path.replace("samples", "samples_param_small") # 224x400 resolution
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        param = json.load(open(param_path))
        ck = np.array(param["camera_intrinsic"])

        # img
        img_path = img_path.replace("samples", "samples_small")
        img_path = img_path.replace("sweeps", "sweeps_small")
        img = Image.open(img_path)
        h, w = img.height, img.width
        img, ck, resize_flag = maybe_resize(img, reso, ck)
        ck[0, :] = ck[0, :] / reso[1]  # 第一行除以图像宽度
        ck[1, :] = ck[1, :] / reso[0]  # 第二行除以图像高度
        img = HWC3(img)
        imgs.append(img)
        cks.append(ck)

        # relative depth from DepthAnything-v2
        if load_rel_depth:
            depth_path = img_path.replace("sweeps_small", "sweeps_dpt_small")
            depth_path = depth_path.replace("samples_small", "samples_dpt_small")
            depth_path = depth_path.replace(".jpg", ".npy")
            disp = np.load(depth_path).astype(np.float32)
            if resize_flag:
                disp = Image.fromarray(disp)
                disp = disp.resize((reso[1], reso[0]), Image.BILINEAR)
                disp = np.array(disp)
            # inverse disparity to relative depth
            # clamping the farthest depth to 50x of the nearest
            range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
            max = disp.max()
            min = max / range
            depth = 1 / np.maximum(disp, min)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depths.append(depth)
        else:
            depths.append(False)
        
        # metric depth from Metric3D-v2
        depthm_path = img_path.replace("sweeps_small", "sweeps_dptm_small")
        depthm_path = depthm_path.replace("samples_small", "samples_dptm_small")
        depthm_path = depthm_path.replace(".jpg", "_dpt.npy")
        conf_path = depthm_path.replace("_dpt.npy", "_conf.npy")
        dptm = np.load(depthm_path).astype(np.float32)
        conf = np.load(conf_path).astype(np.float32)
        if resize_flag:
            dptm = Image.fromarray(dptm)
            dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
            dptm = np.array(dptm)
            conf = Image.fromarray(conf)
            conf = conf.resize((reso[1], reso[0]), Image.BILINEAR)
            conf = np.array(conf)
        depths_m.append(dptm)
        confs_m.append(conf)

        # 动态物体掩码
        if is_input:  # 输入图像使用全白掩码
            mask = np.ones(tuple(reso), dtype=np.float32)
        else:  # 输出图像读取真实掩码
            mask_path = img_path.replace("sweeps_small", "sweeps_mask_small")
            mask_path = mask_path.replace("samples_small", "samples_mask_small")
            mask_path = mask_path.replace(".jpg", ".png")
            mask = Image.open(mask_path).convert('L')  # convert to grayscale
            if resize_flag:
                mask = mask.resize((reso[1], reso[0]), Image.BILINEAR)
            mask = np.array(mask).astype(np.float32)
            mask = mask / 255.0
        masks.append(mask)

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0  # [v c h w]
    depths = torch.from_numpy(np.stack(depths, axis=0)).float()  # [v h w]
    depths_m = torch.from_numpy(np.stack(depths_m, axis=0)).float()  # [v h w]
    confs_m = torch.from_numpy(np.stack(confs_m, axis=0)).float()  # [v h w]
    masks = torch.from_numpy(np.stack(masks, axis=0)).float()  # [v h w]
    cks = torch.as_tensor(cks, dtype=torch.float32)

    return imgs, depths, depths_m, confs_m, masks, cks

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    return directions


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
    normalize=True,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    if normalize:
        rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d