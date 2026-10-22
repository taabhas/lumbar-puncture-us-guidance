"""Loaders for the phantom dataset shipped in ``data/``."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CROP, DATA_DIR
from .tracking import probe_poses_from_markers


def _frame_index(path: Path) -> int:
    return int(re.search(r"(\d+)", path.stem).group(1))


def load_image(path, crop=True) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image[CROP] if crop else image


def load_mri_points(path=DATA_DIR / "mri" / "phantom_mri_pointcloud.npy") -> np.ndarray:
    """(N, 3) bone-surface points of the phantom segmented from MRI, in mm."""
    return np.load(path)


def load_global_registration(path=DATA_DIR / "registration" / "global_registration.json"):
    """4x4 transform taking Vicon world coordinates to MRI coordinates."""
    return np.array(json.loads(Path(path).read_text())["T_mri_world"])


def load_vicon_markers(path) -> np.ndarray:
    """Read a Vicon CSV (timestamp, m1xyz, m2xyz, m3xyz) into an (N, 3, 3) array."""
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    return raw[:, 1:10].reshape(-1, 3, 3)


@dataclass
class Sweep:
    """A tracked ultrasound sweep: synchronised frames, labels and probe poses."""

    frame_ids: list
    frames: list  # cropped BGR images
    labels: list  # cropped BGR images with the bone surface traced in red
    T_world_probe: np.ndarray  # (N, 4, 4)


def load_sweep(sweep_dir=DATA_DIR / "sweep") -> Sweep:
    sweep_dir = Path(sweep_dir)
    frame_paths = sorted((sweep_dir / "frames").glob("*.png"), key=_frame_index)
    frames = [load_image(p) for p in frame_paths]
    labels = [load_image(sweep_dir / "labels" / p.name) for p in frame_paths]
    markers = load_vicon_markers(sweep_dir / "vicon_markers.csv")
    if len(markers) != len(frames):
        raise ValueError(f"{len(frames)} frames but {len(markers)} Vicon samples")
    return Sweep(
        frame_ids=[_frame_index(p) for p in frame_paths],
        frames=frames,
        labels=labels,
        T_world_probe=probe_poses_from_markers(markers),
    )
