"""Safe needle-insertion regions defined in MRI and rendered into ultrasound."""

import cv2
import numpy as np

from .config import IMAGE_SHAPE, SAFE_CONES, SLAB_HALF_THICKNESS_MM, T_IMG_PROBE
from .geometry import px_to_mm, transform


def inside_cones(points_mri: np.ndarray, cones=SAFE_CONES) -> np.ndarray:
    """True for points inside any of the (solid) insertion cones."""
    inside = np.zeros(len(points_mri), bool)
    for cone in cones:
        d = points_mri - np.asarray(cone["apex"])
        h = -d[:, 1]  # height above the apex, towards the skin
        r_allowed = cone["radius"] * h / cone["height"]
        r = np.hypot(d[:, 0], d[:, 2])
        inside |= (h >= 0) & (h <= cone["height"]) & (r <= r_allowed)
    return inside


def safe_region_mask(T_mri_probe, shape=IMAGE_SHAPE, n_elevation=9, cones=SAFE_CONES):
    """Pixels of the image whose beam slab intersects an insertion cone.

    The B-mode beam has a finite elevational thickness, so (like the MRI
    slice) the cones are projected from a slab around the image plane.
    """
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    x, y = px_to_mm(cols.ravel(), rows.ravel())
    T_mri_img = T_mri_probe @ T_IMG_PROBE
    mask = np.zeros(x.size, bool)
    for z in np.linspace(-SLAB_HALF_THICKNESS_MM, SLAB_HALF_THICKNESS_MM, n_elevation):
        pts = np.column_stack([x, y, np.full_like(x, z)])
        mask |= inside_cones(transform(T_mri_img, pts), cones)
    return mask.reshape(shape)


def overlay(image_bgr, mask, color=(80, 220, 80), alpha=0.4):
    out = image_bgr.copy()
    tint = np.zeros_like(out)
    tint[:] = color
    blended = cv2.addWeighted(out, 1 - alpha, tint, alpha, 0)
    out[mask] = blended[mask]
    return out
