"""Reproduce all quantitative results and figures.

    python scripts/run_pipeline.py

Writes results/metrics.csv, results/summary.json and results/figures/*.png.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lpguide import viz  # noqa: E402
from lpguide.config import IOU_ROI, MM_PER_PX_AXIAL, RESULTS_DIR  # noqa: E402
from lpguide.data import load_global_registration, load_mri_points, load_sweep  # noqa: E402
from lpguide.pipeline import cloud_to_mri_distance, run_sweep, us_point_cloud  # noqa: E402
from figures import fig_3d, fig_gallery, fig_metrics, fig_pipeline  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
METHOD_LABELS = {
    "global": "No correction (tracker pose only)",
    "auto_translation": "Corrected to automatically detected bone",
    "translation": "Corrected to local refinement (shift only)",
    "rigid": "Corrected to local refinement (shift + rotation)",
}


def detector_error(profile_auto, profile_label):
    cols = np.arange(len(profile_label))[IOU_ROI[1]]
    a, l = profile_auto[cols], profile_label[cols]
    both = ~np.isnan(a) & ~np.isnan(l)
    if not both.any():
        return np.nan, 0.0
    err = np.abs(a[both] - l[both]) * MM_PER_PX_AXIAL
    return float(np.median(err)), float(both.sum() / max((~np.isnan(l)).sum(), 1))


def summarise(res, mri):
    rows, summary = [], {"n_frames": len(res.frame_ids), "n_annotated": int(res.labelled.sum())}
    tr = res.per_frame["translation"]
    for i, fid in enumerate(res.frame_ids):
        row = {"frame": fid, "annotated": bool(res.labelled[i])}
        if res.labelled[i]:
            row["iou_global"] = tr[i].iou_before
            row["err_global_mm"] = tr[i].err_before_mm
            for m in ("auto_translation", "translation", "rigid"):
                r = res.per_frame[m][i]
                row[f"iou_{m}"] = r.iou_after if r else np.nan
                row[f"err_{m}_mm"] = r.err_after_mm if r else np.nan
            row["detector_median_err_mm"], row["detector_coverage"] = detector_error(
                res.profiles_auto[i], res.profiles_label[i]
            )
        rows.append(row)

    ann = [r for r in rows if r["annotated"]]
    methods = {}
    for m in ("global", "auto_translation", "translation", "rigid"):
        iou = np.array([r[f"iou_{m}"] for r in ann], float)
        err = np.array([r[f"err_{m}_mm"] for r in ann], float)
        methods[m] = {
            "label": METHOD_LABELS[m],
            "iou_mean": float(np.nanmean(iou)),
            "iou_std": float(np.nanstd(iou)),
            "surface_err_mean_mm": float(np.nanmean(err)),
            "surface_err_median_mm": float(np.nanmedian(err)),
            "n": int(np.sum(~np.isnan(iou))),
        }
    summary["methods"] = methods
    det = np.array([r["detector_median_err_mm"] for r in ann], float)
    summary["bone_detector"] = {
        "median_abs_err_mm": float(np.nanmedian(det)),
        "mean_coverage": float(np.mean([r["detector_coverage"] for r in ann])),
    }

    # 3D consistency: lift annotated bone surfaces with the global vs refined poses.
    idx = np.flatnonzero(res.labelled)
    profiles = [res.profiles_label[i] for i in idx]
    T_glob = [res.T_global[i] for i in idx]
    T_ref = [tr[i].T_mri_probe for i in idx]
    d_glob = cloud_to_mri_distance(us_point_cloud(profiles, T_glob), mri)
    d_ref = cloud_to_mri_distance(us_point_cloud(profiles, T_ref), mri)
    summary["us_cloud_to_mri_mm"] = {
        "global_median": float(np.median(d_glob)),
        "refined_median": float(np.median(d_ref)),
    }
    return rows, summary




def main():
    viz.style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sweep, mri, T_mri_world = load_sweep(), load_mri_points(), load_global_registration()
    print(f"Loaded {len(sweep.frames)} tracked frames and {len(mri)} MRI points")

    res = run_sweep(sweep, mri, T_mri_world)
    rows, summary = summarise(res, mri)

    with open(RESULTS_DIR / "metrics.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "frame", k != "annotated", k))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nAnnotated frames: {summary['n_annotated']} / {summary['n_frames']}")
    print(f"{'method':<52}{'IoU [%]':>10}{'depth err [mm]':>17}")
    for m in summary["methods"].values():
        print(f"{m['label']:<52}{m['iou_mean']:>10.1f}{m['surface_err_mean_mm']:>17.2f}")
    d = summary["us_cloud_to_mri_mm"]
    print(f"\n3D US-to-MRI distance (median): {d['global_median']:.2f} -> {d['refined_median']:.2f} mm")
    print(f"Bone detector median error: {summary['bone_detector']['median_abs_err_mm']:.2f} mm")

    rep = int(np.flatnonzero(np.array(sweep.frame_ids) == 22)[0])
    fig_pipeline(sweep, res, rep, FIG_DIR / "pipeline.png")
    fig_metrics(rows, summary, FIG_DIR / "metrics.png")
    fig_3d(res, mri, FIG_DIR / "reconstruction_3d.png")
    fig_gallery(sweep, res, FIG_DIR / "safe_region_gallery.png")
    print(f"\nFigures written to {FIG_DIR.relative_to(RESULTS_DIR.parent)}/")


if __name__ == "__main__":
    main()
