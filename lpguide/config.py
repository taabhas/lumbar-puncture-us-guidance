"""Geometry and calibration constants for the GE 9L probe / Logiq E9 setup.

All lengths are in millimetres, all image coordinates are (column, row) pixels
of the *cropped* B-mode image (see ``CROP``).
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# --- Ultrasound image -------------------------------------------------------
# The frame grabber records the full 640x480 Logiq screen; the B-mode sector
# occupies rows 135:315 and columns 255:565.
CROP = (slice(135, 315), slice(255, 565))
IMAGE_SHAPE = (180, 310)  # rows, cols after cropping

# Pixel spacing measured from the on-screen depth scale.
MM_PER_PX_LATERAL = 43 / 180  # along the transducer face (image x)
MM_PER_PX_AXIAL = 48 / 210  # into the tissue (image y)

# Pixel location of the probe tip (origin of the image frame).
PROBE_TIP_PX = (165, 0)

# The B-mode plane is not infinitely thin; MRI points within this distance of
# the image plane (elevational direction) are projected into the slice.
SLAB_HALF_THICKNESS_MM = 16.125

# Central region of the image used to score MRI/US overlap.
IOU_ROI = (slice(0, 80), slice(65, 245))

# Image frame <-> probe-tip frame. Image x is the probe's z axis (along the
# spine), image y is the probe's y axis (depth). The matrix is its own inverse.
T_IMG_PROBE = np.array(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=float,
)

# --- Vicon marker cuff --------------------------------------------------------
# Marker positions in the probe-tip frame, taken from the cuff CAD model.
CUFF_MARKERS_CAD = np.array(
    [
        [21.80, -49.34, -0.54],
        [-26.41, -14.19, 33.61],
        [35.29, -14.15, -32.61],
    ]
)

# --- Safe needle-insertion cones (MRI frame) ----------------------------------
# One cone per target interspace. The apex sits in the interspinous gap and the
# cone opens towards the skin (-y in the MRI frame).
SAFE_CONES = [
    {"name": "L1/L2", "apex": (-6.5, 122.5, 28.0), "height": 90.0, "radius": 12.0},
    {"name": "L2/L3", "apex": (-6.5, 122.5, -7.0), "height": 90.0, "radius": 12.0},
    {"name": "L3/L4", "apex": (-6.5, 122.5, -38.0), "height": 90.0, "radius": 7.0},
]
