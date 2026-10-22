"""MRI <-> ultrasound registration and overlap metrics."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import IOU_ROI, MM_PER_PX_AXIAL
from .geometry import in_plane_to_probe, mri_profile, profile_to_mask, profile_to_points


def icp_2d(source, target, allow_rotation=False, max_iter=50, tol=1e-4):
    """Rigid 2D ICP. Returns the 3x3 transform A minimising
    mean ||A(source_i) - nearest_target||^2.
    """
    tree = cKDTree(target)
    A = np.eye(3)
    src = source.copy()
    prev_err = np.inf
    for _ in range(max_iter):
        dist, idx = tree.query(src)
        matched = target[idx]
        if allow_rotation:
            mu_s, mu_t = src.mean(0), matched.mean(0)
            U, _, Vt = np.linalg.svd((src - mu_s).T @ (matched - mu_t))
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            t = mu_t - R @ mu_s
        else:
            R, t = np.eye(2), (matched - src).mean(0)
        step = np.eye(3)
        step[:2, :2], step[:2, 2] = R, t
        A = step @ A
        src = src @ R.T + t
        err = np.mean(dist**2)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return A


def iou(profile_a, profile_b):
    """IoU of the acoustic-window masks (tissue above bone) inside IOU_ROI, in %."""
    a, b = profile_to_mask(profile_a)[IOU_ROI], profile_to_mask(profile_b)[IOU_ROI]
    return 100.0 * np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


def surface_error_mm(profile_a, profile_b):
    """Mean absolute depth difference of two bone surfaces inside IOU_ROI, in mm."""
    cols = np.arange(len(profile_a))[IOU_ROI[1]]
    a, b = profile_a[cols], profile_b[cols]
    both = ~np.isnan(a) & ~np.isnan(b)
    if not both.any():
        return np.nan
    return float(np.mean(np.abs(a[both] - b[both])) * MM_PER_PX_AXIAL)


@dataclass
class FrameResult:
    T_mri_probe: np.ndarray  # pose after refinement
    profile_us: np.ndarray
    profile_before: np.ndarray
    profile_after: np.ndarray
    iou_before: float
    iou_after: float
    err_before_mm: float
    err_after_mm: float


def refine_frame(mri_points, T_mri_probe, profile_us, allow_rotation=False, min_points=10):
    """Local in-plane refinement of one tracked frame (Algorithm 2 in the paper).

    The tracked pose predicts an MRI bone profile; the bone surface seen in
    ultrasound is ICP-registered onto it and the residual in-plane motion is
    folded back into the probe pose.
    """
    profile_before = mri_profile(mri_points, T_mri_probe)
    us_pts, mri_pts = profile_to_points(profile_us), profile_to_points(profile_before)

    T_new = T_mri_probe
    if len(us_pts) >= min_points and len(mri_pts) >= min_points:
        A = icp_2d(us_pts, mri_pts, allow_rotation=allow_rotation)
        T_new = T_mri_probe @ in_plane_to_probe(A)
    profile_after = mri_profile(mri_points, T_new)

    return FrameResult(
        T_mri_probe=T_new,
        profile_us=profile_us,
        profile_before=profile_before,
        profile_after=profile_after,
        iou_before=iou(profile_before, profile_us),
        iou_after=iou(profile_after, profile_us),
        err_before_mm=surface_error_mm(profile_before, profile_us),
        err_after_mm=surface_error_mm(profile_after, profile_us),
    )
