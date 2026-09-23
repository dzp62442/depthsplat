from functools import cache

import numpy as np
import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor
from torchmetrics import PearsonCorrCoef
from scipy.ndimage import binary_erosion

from .ego_mask import build_eval_mask_config


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@cache
def get_pcc(device: torch.device) -> PearsonCorrCoef:
    return PearsonCorrCoef().to(device)


@torch.no_grad()
def compute_pcc(
    ground_truth: Float[Tensor, "batch height width"],
    predicted: Float[Tensor, "batch height width"],
) -> Float[Tensor, ""]:
    value = get_pcc(predicted.device).forward(
        ground_truth.reshape(-1), predicted.reshape(-1)
    )
    return value.mean()


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


@cache
def get_spatial_lpips(device: torch.device, net: str) -> LPIPS:
    return LPIPS(net=net, spatial=True).to(device).eval()


def validate_eval_mask(mask, shape, device):
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape) or mask.device != device:
        raise ValueError("Evaluation mask must be boolean and match target shape/device")
    if not mask.reshape(mask.shape[0], -1).any(dim=1).all():
        raise ValueError("Evaluation mask has no valid pixels in a target view")


@torch.no_grad()
def compute_image_metrics(ground_truth, predicted, mask=None, mask_cfg=None):
    """SVF-GS masked metrics; retain original functions for every full-image view."""
    if ground_truth.shape != predicted.shape or ground_truth.device != predicted.device:
        raise ValueError("RGB metric tensors must have matching shapes/devices")
    if (mask is None) != (mask_cfg is None):
        raise ValueError("Evaluation mask and its metric configuration must be provided together")
    if mask is not None:
        validate_eval_mask(mask, (ground_truth.shape[0], *ground_truth.shape[-2:]), ground_truth.device)
        expected = build_eval_mask_config()
        for key in ("ssim_win_size", "ssim_sigma", "ssim_use_sample_covariance", "lpips_net",
                    "lpips_normalize", "lpips_rule", "pcc_rule"):
            if mask_cfg.get(key) != expected[key]:
                raise ValueError(f"Unsupported masked metric configuration: {key}")
    metrics = {name: fn(ground_truth, predicted) for name, fn in
               (("psnr", compute_psnr), ("ssim", compute_ssim), ("lpips", compute_lpips))}
    if mask is None:
        return metrics
    partial = ~mask.flatten(1).all(dim=1)
    if not partial.any():
        return metrics
    valid = mask[partial]
    gt, pred = ground_truth[partial], predicted[partial]
    squared = (gt.clip(0, 1) - pred.clip(0, 1)).square()
    mse = torch.where(valid[:, None], squared, 0).sum(dim=(1, 2, 3)) / (gt.shape[1] * valid.sum(dim=(1, 2)))
    metrics["psnr"][partial] = -10 * mse.log10()
    window = mask_cfg["ssim_win_size"]
    scores = []
    for target, prediction, area in zip(gt, pred, valid):
        valid_centers = binary_erosion(area.cpu().numpy(), structure=np.ones((window, window), dtype=bool))
        if not valid_centers.any():
            raise ValueError("Evaluation mask has no valid SSIM windows")
        _, distance = structural_similarity(target.cpu().numpy(), prediction.cpu().numpy(),
                                           win_size=window, gaussian_weights=True, sigma=mask_cfg["ssim_sigma"],
                                           use_sample_covariance=mask_cfg["ssim_use_sample_covariance"],
                                           channel_axis=0, data_range=1.0, full=True)
        scores.append(distance[:, valid_centers].mean())
    metrics["ssim"][partial] = torch.as_tensor(scores, dtype=predicted.dtype, device=predicted.device)
    pred_eval = torch.where(valid[:, None], pred, gt)
    distance = get_spatial_lpips(predicted.device, mask_cfg["lpips_net"])(
        gt, pred_eval, normalize=mask_cfg["lpips_normalize"])[:, 0]
    metrics["lpips"][partial] = torch.where(valid, distance, 0).sum(dim=(1, 2)) / valid.sum(dim=(1, 2))
    return metrics


@torch.no_grad()
def compute_eval_pcc(ground_truth, predicted, mask=None):
    if ground_truth.shape != predicted.shape or ground_truth.device != predicted.device:
        raise ValueError("PCC tensors must have matching shapes/devices")
    if mask is None:
        return compute_pcc(ground_truth, predicted)
    validate_eval_mask(mask, ground_truth.shape, ground_truth.device)
    gt, pred = ground_truth[mask], predicted[mask]
    if (gt.numel() < 2 or not torch.isfinite(gt).all() or not torch.isfinite(pred).all() or
            gt.var(unbiased=False) == 0 or pred.var(unbiased=False) == 0):
        raise ValueError("Masked PCC requires finite nonconstant depths and at least two valid pixels")
    if mask.all():
        return compute_pcc(ground_truth, predicted)
    # Keep the existing jaxtyping 3-D contract while flattening only valid samples.
    return compute_pcc(gt.reshape(1, 1, -1), pred.reshape(1, 1, -1))
