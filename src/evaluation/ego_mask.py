"""DDAD evaluation-only protocol and output routing (no model dependencies)."""

from pathlib import Path

from omegaconf import OmegaConf


def build_eval_mask_config():
    # Match SVF-GS build_eval_mask_config; do not tune these per experiment.
    return dict(schema="svfgs_ddad_ego_mask_v1", manifest_path="ego_masks/vidar_v1/manifest.json",
                source_commit="0d84851ce4d86a9f132f8027898ff981e751db79",
                pixel_protocol="ddad_ego_novel12_v1", output_suffix="_ego_novel12_v1",
                image_hw=[224, 400], resize="PIL_NEAREST", novel_views=12, input_views=6,
                invalid_value=0, valid_value=255, geometry_atol=1e-6,
                ssim_win_size=11, ssim_sigma=1.5, ssim_use_sample_covariance=True,
                lpips_net="vgg", lpips_normalize=True,
                lpips_rule="gt_fill_invalid_spatial_valid_mean", pcc_rule="valid_group_flatten")


def validate_eval_mask_mode(mode, dataset, enabled):
    if not isinstance(enabled, bool):
        raise ValueError("test.eval_use_ego_mask must be boolean")
    if enabled and (mode != "test" or dataset != "ddad"):
        raise ValueError("test.eval_use_ego_mask requires mode=test and dataset=ddad")


def resolve_eval_output_dir(output_dir, mode, dataset, enabled):
    validate_eval_mask_mode(mode, dataset, enabled)
    if not enabled:
        return output_dir
    if not output_dir:
        raise ValueError("Masked evaluation requires an explicit output_dir")
    path = Path(output_dir)
    suffix = build_eval_mask_config()["output_suffix"]
    return str(path if path.name.endswith(suffix) else path.with_name(path.name + suffix))


OmegaConf.register_new_resolver("depthsplat_eval_output_dir", resolve_eval_output_dir, replace=True)
