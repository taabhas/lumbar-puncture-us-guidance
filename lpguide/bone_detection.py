"""Hybrid bone-surface detector for paramedian lumbar ultrasound.

Bone in B-mode appears as a bright echo followed by an acoustic shadow. For
every image column we find

* the *brightest* pixel (strong specular reflection at the bone apex), and
* the *transition* row that best separates a brighter region above from a
  dark (shadowed) region below,

and pick whichever is consistent. The resulting profile is smoothed.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def transition_row(column, clip=90, shadow_max=40, margin=5):
    """Row that maximises mean(intensity above) - mean(intensity below).

    Returns -1 if the column is (almost) uniformly bright or dark, or if no
    candidate has a dark enough region below it.
    """
    col = np.minimum(column.astype(float), clip)
    n = len(col)
    mean_all = col.mean()
    if mean_all > 60 or mean_all < 2:
        return -1

    csum = np.cumsum(col)
    idx = np.arange(n)
    valid = (idx >= margin) & (idx <= n - margin)
    above = csum / (idx + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        below = (csum[-1] - csum) / (n - idx - 1)
    contrast = np.where(valid & (below < shadow_max), above - below, -np.inf)
    best = int(np.argmax(contrast))
    return best if np.isfinite(contrast[best]) and contrast[best] > 0 else -1


def detect_bone_surface(
    image_bgr,
    bright_threshold=50,
    max_gap=50,
    row_range=(20, 160),
    border=50,
    sigma=5,
    return_debug=False,
):
    """Estimate the bone-surface profile (row per column, NaN = no bone)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    n_cols = gray.shape[1]

    bright = np.full(n_cols, np.nan)
    trans = np.full(n_cols, np.nan)
    chosen = np.full(n_cols, np.nan)
    for c in range(border, n_cols - border):
        column = gray[:, c]
        if column.max() > bright_threshold:
            bright[c] = int(np.argmax(column))
        t = transition_row(column)
        if t >= 0:
            trans[c] = t
        if not (row_range[0] < t < row_range[1]):
            continue
        # Prefer the bright apex when it lies just above the shadow boundary.
        if not np.isnan(bright[c]) and t - bright[c] < max_gap:
            chosen[c] = bright[c]
        else:
            chosen[c] = t

    profile = chosen.copy()
    ok = ~np.isnan(chosen)
    if ok.sum() > 1:
        profile[ok] = np.round(gaussian_filter1d(chosen[ok], sigma))

    if return_debug:
        return profile, {"brightest": bright, "transition": trans}
    return profile
