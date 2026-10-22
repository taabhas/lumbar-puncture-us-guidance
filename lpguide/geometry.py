"""Coordinate frames, virtual MRI slicing and bone-surface profiles.

Frames
------
world  : Vicon motion-capture frame
probe  : probe tip, tracked by Vicon
image  : B-mode image plane, origin at the probe tip, x lateral, y depth (mm)
mri    : MRI phantom frame

A *bone-surface profile* is a length-W array holding, for every image column,
the row at which bone is first encountered (NaN where no bone is visible).
Both the ultrasound labels and the virtual MRI slices are reduced to this
representation so they can be compared and registered.
"""

import numpy as np

from .config import (
    IMAGE_SHAPE,
    MM_PER_PX_AXIAL,
    MM_PER_PX_LATERAL,
    PROBE_TIP_PX,
    SLAB_HALF_THICKNESS_MM,
    T_IMG_PROBE,
)


def to_h(points: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 4) homogeneous."""
    return np.hstack([points, np.ones((len(points), 1))])


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (to_h(points) @ T.T)[:, :3]


# --- pixels <-> image-plane millimetres ---------------------------------------


def px_to_mm(cols, rows):
    x = (np.asarray(cols, float) - PROBE_TIP_PX[0]) * MM_PER_PX_LATERAL
    y = (np.asarray(rows, float) - PROBE_TIP_PX[1]) * MM_PER_PX_AXIAL
    return x, y


def image_points_to_mri(cols, rows, T_mri_probe):
    """Lift pixels of the image plane into 3D MRI coordinates."""
    x, y = px_to_mm(cols, rows)
    pts_img = np.column_stack([x, y, np.zeros_like(x)])
    return transform(T_mri_probe @ T_IMG_PROBE, pts_img)


# --- virtual MRI slicing --------------------------------------------------------


def mri_profile(mri_points: np.ndarray, T_mri_probe: np.ndarray, shape=IMAGE_SHAPE):
    """Render the bone-surface profile the probe *should* see at this pose.

    MRI points inside the elevational slab of the image plane are projected
    onto it, and for every column the shallowest one is kept.
    """
    T_img_mri = T_IMG_PROBE @ np.linalg.inv(T_mri_probe)
    p = transform(T_img_mri, mri_points)

    rows_px, cols_px = shape
    x_min = -PROBE_TIP_PX[0] * MM_PER_PX_LATERAL
    x_max = (cols_px - PROBE_TIP_PX[0]) * MM_PER_PX_LATERAL
    y_lim = rows_px * MM_PER_PX_AXIAL
    keep = (
        (np.abs(p[:, 2]) < SLAB_HALF_THICKNESS_MM)
        & (p[:, 0] >= x_min)
        & (p[:, 0] < x_max)
        & (np.abs(p[:, 1]) < y_lim)
    )
    p = p[keep]

    cols = np.trunc(p[:, 0] / MM_PER_PX_LATERAL).astype(int) + PROBE_TIP_PX[0]
    rows = np.trunc(p[:, 1] / MM_PER_PX_AXIAL).astype(int) + PROBE_TIP_PX[1]

    profile = np.full(cols_px, np.inf)
    np.minimum.at(profile, cols, rows)
    profile[np.isinf(profile)] = np.nan
    return profile


# --- profiles, masks, point sets ----------------------------------------------------


def profile_from_label(label_bgr: np.ndarray) -> np.ndarray:
    """First red pixel in each column of a hand-traced label image."""
    b, g, r = (label_bgr[..., i].astype(int) for i in range(3))
    red = (r > 100) & (r > g + 10) & (r > b + 10)
    has_red = red.any(axis=0)
    profile = np.full(label_bgr.shape[1], np.nan)
    profile[has_red] = red.argmax(axis=0)[has_red]
    return profile


def profile_to_mask(profile: np.ndarray, n_rows=IMAGE_SHAPE[0]) -> np.ndarray:
    """Boolean mask of the acoustic window: tissue *above* the bone surface."""
    rows = np.arange(n_rows)[:, None]
    surface = np.where(np.isnan(profile), np.inf, np.maximum(profile, 0))
    return rows < surface[None, :]


def profile_to_points(profile: np.ndarray) -> np.ndarray:
    """Profile -> (N, 2) image-plane points in mm."""
    cols = np.flatnonzero(~np.isnan(profile))
    x, y = px_to_mm(cols, profile[cols])
    return np.column_stack([x, y])


def in_plane_to_probe(A: np.ndarray) -> np.ndarray:
    """Embed a 3x3 rigid transform of the image plane as a 4x4 probe-frame motion."""
    T = np.eye(4)
    T[:2, :2] = A[:2, :2]
    T[:2, 3] = A[:2, 2]
    return T_IMG_PROBE @ T @ T_IMG_PROBE
