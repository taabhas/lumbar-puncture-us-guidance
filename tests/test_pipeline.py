"""Regression tests: the pipeline must reproduce the reported numbers."""

import numpy as np
import pytest

from lpguide.config import DATA_DIR
from lpguide.data import load_global_registration, load_mri_points, load_sweep
from lpguide.geometry import in_plane_to_probe, mri_profile
from lpguide.pipeline import run_sweep
from lpguide.registration import icp_2d


@pytest.fixture(scope="module")
def sweep():
    return load_sweep()


@pytest.fixture(scope="module")
def result(sweep):
    return run_sweep(sweep, load_mri_points(), load_global_registration(), with_safe_regions=False)


def test_probe_poses_match_recorded(sweep):
    """Raw Vicon markers -> probe poses reproduces the poses saved during the experiment."""
    ref = np.load(DATA_DIR / "sweep" / "probe_poses_reference.npy")
    np.testing.assert_allclose(sweep.T_world_probe, ref, atol=1e-9)


def test_icp_2d_recovers_shift():
    rng = np.random.default_rng(0)
    target = np.column_stack([np.linspace(-10, 10, 200), np.sin(np.linspace(-3, 3, 200)) * 5])
    source = target + np.array([1.5, -0.8]) + rng.normal(0, 0.01, target.shape)
    A = icp_2d(source, target)
    np.testing.assert_allclose(A[:2, 2], [-1.5, 0.8], atol=0.05)


def test_in_plane_motion_shifts_mri_profile(sweep):
    """Moving the probe 1 mm deeper must move the predicted bone surface up by ~1 mm."""
    mri = load_mri_points()
    T = load_global_registration() @ sweep.T_world_probe[0]
    A = np.eye(3)
    A[1, 2] = 1.0  # image y (depth) + 1 mm
    p0, p1 = mri_profile(mri, T), mri_profile(mri, T @ in_plane_to_probe(A))
    both = ~np.isnan(p0) & ~np.isnan(p1)
    assert np.median(p1[both] - p0[both]) == pytest.approx(-1.0 / (48 / 210), abs=1.0)


def test_reported_metrics(result):
    assert result.labelled.sum() == 30
    tr = [r for r in result.per_frame["translation"] if r is not None]
    iou_before = np.mean([r.iou_before for r in tr])
    iou_after = np.mean([r.iou_after for r in tr])
    err_after = np.mean([r.err_after_mm for r in tr])
    assert iou_before == pytest.approx(67.4, abs=0.1)
    assert iou_after == pytest.approx(87.2, abs=0.1)
    assert err_after == pytest.approx(1.97, abs=0.05)
    assert all(r.iou_after > r.iou_before for r in tr)
