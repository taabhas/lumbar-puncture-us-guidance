"""Plotting helpers shared by the result figures and the demo animation."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .config import IMAGE_SHAPE, MM_PER_PX_LATERAL, SAFE_CONES
from .geometry import image_points_to_mri

# Colours (hex for matplotlib, BGR tuples for OpenCV overlays).
C_US = "#eb6834"  # bone surface seen in ultrasound
C_MRI = "#5aa3f0"  # bone surface predicted from MRI (after refinement)
C_MRI_BEFORE = "#c3c2b7"  # ... before refinement
C_SAFE = "#1baf7a"  # safe insertion region
C_PLANE = "#3b3b38"  # ultrasound image plane in 3D views
C_BEFORE, C_AFTER = "#a3a29c", "#2a78d6"  # metric plots
INK, INK_MUTED = "#0b0b0b", "#52514e"


def bgr(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def sector_columns(frame_bgr, thresh=4):
    """Columns that contain ultrasound signal (the rest of the crop is black)."""
    return frame_bgr.max(axis=(0, 2)) > thresh


def sector_bounds(frame_bgr):
    """First and last+1 column of the imaged sector."""
    cols = np.flatnonzero(sector_columns(frame_bgr))
    return int(cols[0]), int(cols[-1]) + 1


def crop_sector(img, frame_bgr, scale=2):
    c0, c1 = sector_bounds(frame_bgr)
    return img[:, c0 * scale : c1 * scale]


def _polyline(img, profile, color, scale, thickness, dashed=False):
    cols = np.flatnonzero(~np.isnan(profile))
    if len(cols) < 2:
        return
    pts = np.column_stack([cols, profile[cols]]) * scale
    # Split where columns are not contiguous.
    breaks = np.flatnonzero(np.diff(cols) > 3) + 1
    for seg in np.split(pts, breaks):
        seg = seg.astype(np.int32)
        if dashed:
            for k in range(0, len(seg) - 1, 12):
                cv2.polylines(img, [seg[k : k + 7]], False, color, thickness, cv2.LINE_AA)
        else:
            cv2.polylines(img, [seg], False, color, thickness, cv2.LINE_AA)


def frame_overlay(
    frame_bgr,
    profile_us=None,
    profile_mri=None,
    profile_mri_before=None,
    safe_mask=None,
    scale=2,
    safe_alpha=0.35,
):
    """Ultrasound frame with bone surfaces and safe region drawn on top."""
    img = frame_bgr.copy()
    sector = sector_columns(frame_bgr)

    def clip(profile):
        return None if profile is None else np.where(sector, profile, np.nan)

    profile_us, profile_mri, profile_mri_before = map(clip, (profile_us, profile_mri, profile_mri_before))
    if safe_mask is not None:
        m = safe_mask & sector[None, :]
        tint = np.zeros_like(img)
        tint[:] = bgr(C_SAFE)
        img[m] = cv2.addWeighted(img, 1 - safe_alpha, tint, safe_alpha, 0)[m]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if safe_mask is not None:
        m = (safe_mask & sector[None, :]).astype(np.uint8)
        m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, bgr(C_SAFE), 1, cv2.LINE_AA)
    if profile_mri_before is not None:
        _polyline(img, profile_mri_before, bgr(C_MRI_BEFORE), scale, 2, dashed=True)
    if profile_mri is not None:
        _polyline(img, profile_mri, bgr(C_MRI), scale, 2)
    if profile_us is not None:
        _polyline(img, profile_us, bgr(C_US), scale, 2)
    return img


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def add_scale_bar(ax, scale=2, mm=10, color="white"):
    """10 mm scale bar in the lower-left corner of an image axis."""
    px = mm / MM_PER_PX_LATERAL * scale
    h = IMAGE_SHAPE[0] * scale
    x0, y0 = 12, h - 14
    ax.plot([x0, x0 + px], [y0, y0], color=color, lw=2.5, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - 6, f"{mm} mm", color=color, ha="center", va="bottom", fontsize=8)


# --- 3D scene ---------------------------------------------------------------------


def cone_wireframe(cone, n=24):
    ax_, ay, az = cone["apex"]
    h = np.linspace(0, cone["height"] * 0.55, 8)
    t = np.linspace(0, 2 * np.pi, n)
    H, Tt = np.meshgrid(h, t)
    R = cone["radius"] * H / cone["height"]
    return ax_ + R * np.cos(Tt), ay - H, az + R * np.sin(Tt)


def image_plane_corners(T_mri_probe):
    rows, cols = IMAGE_SHAPE
    c = np.array([0, cols - 1, cols - 1, 0, 0])
    r = np.array([0, 0, rows - 1, rows - 1, 0])
    return image_points_to_mri(c, r, T_mri_probe)


def draw_scene(
    ax, mri_points, T_mri_probe=None, us_cloud=None, cones=SAFE_CONES, sub=6, zoom=1.0, mri_color="#6f6e69",
    xlim=(-75, 70), label_plane=True,
):
    p = mri_points[::sub]
    # Plot with MRI y (depth) as the vertical axis, skin side up.
    ax.scatter(p[:, 2], p[:, 0], -p[:, 1], s=0.5, c=mri_color, alpha=0.35, linewidths=0)
    for cone in cones:
        X, Y, Z = cone_wireframe(cone)
        ax.plot_surface(Z, X, -Y, color=C_SAFE, alpha=0.25, linewidth=0)
    if us_cloud is not None and len(us_cloud):
        ax.scatter(us_cloud[:, 2], us_cloud[:, 0], -us_cloud[:, 1], s=1.2, c=C_US, linewidths=0)
    if T_mri_probe is not None:
        q = image_plane_corners(T_mri_probe)
        ax.plot(q[:, 2], q[:, 0], -q[:, 1], color=C_PLANE, lw=1.4)
        verts = [list(zip(q[:4, 2], q[:4, 0], -q[:4, 1]))]
        ax.add_collection3d(Poly3DCollection(verts, color=C_PLANE, alpha=0.08))
        tip = T_mri_probe[:3, 3]
        ax.scatter([tip[2]], [tip[0]], [-tip[1]], s=18, c=C_PLANE)
        if label_plane:
            ax.text(tip[2], tip[0], -tip[1] + 6, "ultrasound image", fontsize=8, color=C_PLANE, ha="center")
    lims = (xlim, (-50, 35), (-135, -55))
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])
    ax.set_box_aspect([hi - lo for lo, hi in lims], zoom=zoom)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-62)
