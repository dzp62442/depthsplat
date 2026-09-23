"""Small reporting helpers for the existing PandaSet/DDAD test loop."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .metrics import compute_eval_pcc
from .ego_mask import build_eval_mask_config

VIEW_GROUPS = {"all_18": slice(0, 18), "novel_12": slice(0, 12), "input_6": slice(12, 18)}
METRICS = ("psnr", "ssim", "lpips", "pcc")


def checkpoint_metadata(path, strict):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "strict_load": strict}


def evaluation_pixel_identity(records):
    identities = {(r.get("pixel_protocol", "full_image"), r.get("mask_manifest_sha256", "")) for r in records}
    if len(identities) != 1:
        raise ValueError("Mixed evaluation pixel protocols or mask manifests")
    protocol, sha = identities.pop()
    if protocol not in ("full_image", build_eval_mask_config()["pixel_protocol"]):
        raise ValueError(f"Unknown evaluation pixel protocol: {protocol}")
    if not isinstance(sha, str) or (protocol == "full_image") != (sha == ""):
        raise ValueError("Evaluation pixel protocol requires matching mask manifest identity")
    return protocol, sha


def check_existing_pixel_identity(out_dir, metadata):
    """Refuse incompatible reruns before overwriting summaries or rendered images."""
    expected = evaluation_pixel_identity([metadata])
    for name in ("data_provenance.json", "evaluation_summary.json"):
        path = Path(out_dir) / name
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                previous = json.load(handle)
            if evaluation_pixel_identity([previous]) != expected:
                raise ValueError(f"Existing evaluation pixel protocol/mask differs: {path}; use another output_dir")


def view_group_records(token, scene_id, image_metrics, reference_depth, predicted_depth,
                       mask=None, mask_cfg=None, mask_manifest_sha256=None):
    if (reference_depth is None or predicted_depth is None or
            reference_depth.shape != predicted_depth.shape or reference_depth.shape[0] != 18):
        raise ValueError(f"Expected 18 matching target/rendered depths: {token}")
    if set(image_metrics) != {"psnr", "ssim", "lpips"} or any(v.shape != (18,) for v in image_metrics.values()):
        raise ValueError(f"Expected 18 per-view RGB metrics: {token}")
    if ((mask is None) != (mask_cfg is None) or
            (mask is not None and not mask_manifest_sha256) or (mask is None and mask_manifest_sha256)):
        raise ValueError("Evaluation mask records require matching configuration and manifest identity")
    records = []
    for name, indices in VIEW_GROUPS.items():
        row = {"bin_token": token, "scene_id": scene_id, "stage": "final", "view_group": name}
        row.update(pixel_protocol="full_image" if mask_cfg is None else mask_cfg["pixel_protocol"],
                   mask_manifest_sha256="" if mask is None else mask_manifest_sha256)
        row.update({key: values[indices].double().mean().item() for key, values in image_metrics.items()})
        row["pcc"] = compute_eval_pcc(reference_depth[indices].contiguous(),
                                      predicted_depth[indices].contiguous(),
                                      None if mask is None else mask[indices]).item()
        records.append(row)
    return records


def summarize_records(records, expected_tokens):
    expected = set(expected_tokens)
    if not expected or len(expected) != len(expected_tokens):
        raise ValueError("Expected a nonempty, unique evaluation list")
    protocol, sha = evaluation_pixel_identity(records)
    summary = {
        "pixel_protocol": protocol,
        "mask_manifest_sha256": sha,
        "primary_result": "final/all_18",
        "primary_metrics": ["psnr", "ssim", "lpips"],
        "diagnostic_groups": ["final/novel_12", "final/input_6"],
        "pcc_reference": "metric3d_v2",
        "aggregation": "per-view RGB mean within bin, then equal bin mean; PCC flattened within each bin/group",
        "expected_bins": len(expected),
        "complete": True,
    }
    if any(r["view_group"] not in VIEW_GROUPS or r["stage"] != "final" for r in records):
        raise ValueError("Unknown evaluation stage/view group")
    for group in VIEW_GROUPS:
        rows = [r for r in records if r["view_group"] == group]
        counts = Counter(r["bin_token"] for r in rows)
        invalid = [{"bin_token": r["bin_token"], "metric": name}
                   for r in rows for name in METRICS if not math.isfinite(r[name])]
        missing, unexpected = sorted(expected - counts.keys()), sorted(counts.keys() - expected)
        duplicates = sorted(token for token, count in counts.items() if count != 1)
        complete = not (invalid or missing or unexpected or duplicates)
        result = {"num_bins": len(counts), "num_records": len(rows), "complete": complete,
                  "pixel_protocol": protocol, "mask_manifest_sha256": sha,
                  "missing_bins": missing, "unexpected_bins": unexpected,
                  "duplicate_bins": duplicates, "nonfinite_metrics": invalid}
        for name in METRICS:
            values = [r[name] for r in rows]
            result[name] = math.fsum(values) / len(values) if values and all(map(math.isfinite, values)) else None
        summary[f"final/{group}"] = result
        summary["complete"] &= complete
    return summary


def write_zero_shot_results(out_dir, records, metadata):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_records(records, metadata["bin_tokens"])
    if evaluation_pixel_identity(records) != evaluation_pixel_identity([metadata]):
        raise ValueError("Result/provenance evaluation pixel identity mismatch")
    check_existing_pixel_identity(out_dir, metadata)
    if "eval_mask" in metadata:
        if evaluation_pixel_identity([metadata["eval_mask"]]) != evaluation_pixel_identity([metadata]):
            raise ValueError("Inconsistent evaluation mask metadata")
        summary["eval_mask"] = metadata["eval_mask"]
        for group in VIEW_GROUPS:
            summary[f"final/{group}"]["eval_mask"] = metadata["eval_mask"]
    fields = ["bin_token", "scene_id", "stage", "view_group", "pixel_protocol", "mask_manifest_sha256", *METRICS]
    with (out_dir / "per_bin_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    for filename, payload in (("evaluation_summary.json", summary), ("data_provenance.json", metadata)):
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    if not summary["complete"]:
        raise RuntimeError(f"Incomplete/nonfinite temporal18 evaluation; see {out_dir / 'evaluation_summary.json'}")
    return summary
