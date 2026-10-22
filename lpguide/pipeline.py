"""End-to-end processing of a tracked sweep."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .bone_detection import detect_bone_surface
from .config import IOU_ROI
from .data import Sweep
from .geometry import image_points_to_mri, mri_profile, profile_from_label
from .registration import FrameResult, iou, refine_frame, surface_error_mm
from .safe_region import safe_region_mask


def _has_bone(profile, min_cols=10):
    return np.sum(~np.isnan(profile[IOU_ROI[1]])) >= min_cols


@dataclass
class SweepResult:
    frame_ids: list
    labelled: np.ndarray  # bool, frame shows an annotated spinous process
    profiles_label: list
    profiles_auto: list
    per_frame: dict = field(default_factory=dict)  # method -> list[FrameResult | None]
    T_global: np.ndarray = None  # (N,4,4) MRI<-probe from the global registration
    T_track: np.ndarray = None  # (N,4,4) MRI<-probe used for display
    safe_masks: list = None
    mri_display_profiles: list = None


def run_sweep(sweep: Sweep, mri_points, T_mri_world, with_safe_regions=True):
    n = len(sweep.frames)
    labels = [profile_from_label(l) for l in sweep.labels]
    autos = [detect_bone_surface(f) for f in sweep.frames]
    labelled = np.array([_has_bone(p) for p in labels])
    T_global = np.array([T_mri_world @ T for T in sweep.T_world_probe])

    res = SweepResult(sweep.frame_ids, labelled, labels, autos, T_global=T_global)

    # Methods evaluated on annotated frames. Each starts from the same global
    # (landmark + ICP) registration so frames are scored independently.
    methods = {
        "translation": dict(source=labels, allow_rotation=False),  # as in the paper
        "rigid": dict(source=labels, allow_rotation=True),
        "auto_translation": dict(source=autos, allow_rotation=False),
    }
    for name, m in methods.items():
        out = []
        for i in range(n):
            if not labelled[i] or not _has_bone(m["source"][i]):
                out.append(None)
                continue
            r = refine_frame(mri_points, T_global[i], m["source"][i], m["allow_rotation"])
            # Always score against the manual annotation.
            out.append(
                FrameResult(
                    r.T_mri_probe,
                    labels[i],
                    r.profile_before,
                    r.profile_after,
                    iou(r.profile_before, labels[i]),
                    iou(r.profile_after, labels[i]),
                    surface_error_mm(r.profile_before, labels[i]),
                    surface_error_mm(r.profile_after, labels[i]),
                )
            )
        res.per_frame[name] = out

    # Dynamic tracking for display: refine when bone is visible, otherwise
    # carry the most recent world->MRI correction forward (Algorithm 2).
    T_track = np.empty_like(T_global)
    T_mri_world_now = T_mri_world
    for i in range(n):
        r = res.per_frame["translation"][i]
        if r is not None:
            T_track[i] = r.T_mri_probe
            T_mri_world_now = r.T_mri_probe @ np.linalg.inv(sweep.T_world_probe[i])
        else:
            T_track[i] = T_mri_world_now @ sweep.T_world_probe[i]
    res.T_track = T_track
    res.mri_display_profiles = [mri_profile(mri_points, T) for T in T_track]
    if with_safe_regions:
        res.safe_masks = [safe_region_mask(T) for T in T_track]
    return res


def us_point_cloud(profiles, T_mri_probe_list):
    """Lift per-frame bone profiles into one 3D point cloud (MRI frame)."""
    clouds = []
    for p, T in zip(profiles, T_mri_probe_list):
        cols = np.flatnonzero(~np.isnan(p))
        if len(cols):
            clouds.append(image_points_to_mri(cols, p[cols], T))
    return np.vstack(clouds) if clouds else np.zeros((0, 3))


def cloud_to_mri_distance(cloud, mri_points):
    d, _ = cKDTree(mri_points).query(cloud)
    return d
