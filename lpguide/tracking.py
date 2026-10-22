"""Probe pose from the three Vicon markers on the ultrasound cuff."""

import numpy as np

from .config import CUFF_MARKERS_CAD


def frame_from_markers(markers: np.ndarray) -> np.ndarray:
    """Build a right-handed frame from three markers.

    Origin at the marker centroid, x towards marker 1, z normal to the marker
    plane. Returns the 4x4 pose of that frame in the markers' coordinates.
    """
    m1, m2, _ = markers
    centroid = markers.mean(axis=0)
    x = (m1 - centroid) / np.linalg.norm(m1 - centroid)
    r = (m2 - centroid) / np.linalg.norm(m2 - centroid)
    z = np.cross(x, r)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)

    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = centroid
    return T


# Fixed offset from the marker-cuff frame to the probe tip (from CAD).
T_CUFF_TIP = np.linalg.inv(frame_from_markers(CUFF_MARKERS_CAD))


def probe_poses_from_markers(markers: np.ndarray) -> np.ndarray:
    """(N, 3, 3) marker positions in world -> (N, 4, 4) probe-tip poses in world.

    Samples where the markers were occluded (all zeros) get a NaN pose so that
    frame indices stay aligned with the images.
    """
    poses = np.full((len(markers), 4, 4), np.nan)
    for i, m in enumerate(markers):
        if np.linalg.norm(m[0]) == 0 and np.linalg.norm(m[1]) == 0:
            continue
        poses[i] = frame_from_markers(m) @ T_CUFF_TIP
    return poses
