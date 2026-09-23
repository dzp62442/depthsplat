"""Small reporting helpers for the existing PandaSet/DDAD test loop."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .metrics import compute_pcc

VIEW_GROUPS = {"all_18": slice(0, 18), "novel_12": slice(0, 12), "input_6": slice(12, 18)}
METRICS = ("psnr", "ssim", "lpips", "pcc")


def checkpoint_metadata(path, strict):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "strict_load": strict}


def view_group_records(token, scene_id, image_metrics, reference_depth, predicted_depth):
    if (reference_depth is None or predicted_depth is None or
            reference_depth.shape != predicted_depth.shape or reference_depth.shape[0] != 18):
        raise ValueError(f"Expected 18 matching target/rendered depths: {token}")
    if set(image_metrics) != {"psnr", "ssim", "lpips"} or any(v.shape != (18,) for v in image_metrics.values()):
        raise ValueError(f"Expected 18 per-view RGB metrics: {token}")
    records = []
    for name, indices in VIEW_GROUPS.items():
        row = {"bin_token": token, "scene_id": scene_id, "stage": "final", "view_group": name}
        row.update({key: values[indices].double().mean().item() for key, values in image_metrics.items()})
        row["pcc"] = compute_pcc(reference_depth[indices].contiguous(),
                                 predicted_depth[indices].contiguous()).item()
        records.append(row)
    return records


def summarize_records(records, expected_tokens):
    expected = set(expected_tokens)
    if not expected or len(expected) != len(expected_tokens):
        raise ValueError("Expected a nonempty, unique evaluation list")
    summary = {
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
    fields = ["bin_token", "scene_id", "stage", "view_group", *METRICS]
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
