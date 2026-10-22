"""Result figures written by run_pipeline.py."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lpguide import viz
from lpguide.bone_detection import detect_bone_surface
from lpguide.config import DATA_DIR, IMAGE_SHAPE, T_IMG_PROBE
from lpguide.data import load_vicon_markers
from lpguide.geometry import profile_to_points, px_to_mm, transform
from lpguide.pipeline import us_point_cloud
from lpguide.registration import icp_2d
from lpguide.tracking import frame_from_markers

LEGEND_HANDLES = [
    plt.Line2D([], [], color=viz.C_US, lw=2.5, label="bone surface seen in ultrasound"),
    plt.Line2D([], [], color=viz.C_MRI, lw=2.5, label="bone surface predicted from MRI"),
    plt.Rectangle((0, 0), 1, 1, fc=viz.C_SAFE, alpha=0.55, ec=viz.C_SAFE, label="safe needle corridor (from MRI)"),
]


def _image_panel(ax, img, frame):
    ax.imshow(viz.to_rgb(viz.crop_sector(img, frame)))
    ax.set_axis_off()


# --- 1. illustrated pipeline ------------------------------------------------------------


def _panel_legend(ax, handles, ncol=1):
    """Legend placed just below a panel so it never covers the data."""
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.02), frameon=False,
        fontsize=8, ncol=ncol, handlelength=1.6, borderaxespad=0, labelspacing=0.3, columnspacing=1.0,
    )


def _line(color, label, **kw):
    return plt.Line2D([], [], color=color, lw=2.2, label=label, **kw)


def _dots(color, label):
    return plt.Line2D([], [], ls="", marker="o", ms=5, color=color, label=label)


def _patch(fc, label, ec=None, alpha=0.55):
    return plt.Rectangle((0, 0), 1, 1, fc=fc, ec=ec or fc, alpha=alpha, label=label)


AXIS_COLORS = ["#e34948", "#008300", "#2a78d6"]


def _panel_tracking(ax, markers, T_world_probe):
    """Cuff markers -> centroid frame -> probe tip -> image plane, in world coordinates."""
    T_c = frame_from_markers(markers)
    origin = T_world_probe[:3, 3]

    def P(p):  # plot with world z up, centred on the probe tip
        p = np.atleast_2d(p) - origin
        return p[:, 0], p[:, 1], p[:, 2]

    tri = np.vstack([markers, markers[:1]])
    ax.plot(*P(tri), color="#a3a29c", lw=1.2)
    ax.scatter(*P(markers), s=60, color="#3b3b38", depthshade=False)
    for k, m in enumerate(markers):
        x, y, z = P(m)
        ax.text(x[0], y[0], z[0] + 6, f"M{k + 1}", fontsize=8, color=viz.INK_MUTED, ha="center")

    for T in (T_c, T_world_probe):
        for axis, color in zip(range(3), AXIS_COLORS):
            ax.quiver(*P(T[:3, 3]), *(T[:3, axis] * 18), color=color, lw=1.6, arrow_length_ratio=0.25)
    ax.plot(*P(np.vstack([T_c[:3, 3], origin])), color=viz.INK_MUTED, lw=1, ls="--")

    rows, cols = IMAGE_SHAPE
    x, y = px_to_mm(np.array([0, cols - 1, cols - 1, 0, 0]), np.array([0, 0, rows - 1, rows - 1, 0]))
    corners = transform(T_world_probe @ T_IMG_PROBE, np.column_stack([x, y, np.zeros_like(x)]))
    ax.plot(*P(corners), color=viz.C_PLANE, lw=1.3)
    c = corners[:4] - origin
    ax.add_collection3d(Poly3DCollection([c.tolist()], color=viz.C_PLANE, alpha=0.08))
    pts = np.vstack([markers - origin, c])
    mid, span = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2
    for setter, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setter(m - span, m + span)
    ax.set_box_aspect((1, 1, 1), zoom=1.25)
    ax.view_init(elev=14, azim=-35)
    ax.set_axis_off()


def _panel_icp(ax, r):
    us = profile_to_points(r.profile_us)
    before = profile_to_points(r.profile_before)
    after = profile_to_points(r.profile_after)
    A = icp_2d(us, before)
    d = -A[:2, 2]  # motion applied to the MRI prediction

    ax.plot(before[:, 0], before[:, 1], ".", ms=2.5, color=viz.C_MRI_BEFORE)
    ax.plot(after[:, 0], after[:, 1], ".", ms=2.5, color=viz.C_MRI)
    ax.plot(us[:, 0], us[:, 1], "-", lw=2.5, color=viz.C_US)
    near = np.abs(before[:, 0] - us[:, 0].mean()) < 6
    start = np.array([us[:, 0].mean(), before[near, 1].min()])
    ax.add_patch(FancyArrowPatch(start, start + d, arrowstyle="-|>", mutation_scale=14, color=viz.INK, lw=1.5))
    ax.text(start[0] + 3, start[1] + d[1] / 2 + 2, f"shift\n{np.hypot(*d):.1f} mm", fontsize=8, color=viz.INK)
    ax.set_xlim(-38, 30)
    ax.set_ylim(42, -2)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral [mm]", fontsize=8.5, labelpad=1)
    ax.set_ylabel("depth [mm]", fontsize=8.5)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_pipeline(sweep, res, i, path):
    frame = sweep.frames[i]
    r = res.per_frame["translation"][i]
    _, dbg = detect_bone_surface(frame, return_debug=True)
    markers = load_vicon_markers(DATA_DIR / "sweep" / "vicon_markers.csv")[i]

    fig = plt.figure(figsize=(18, 4.8))
    W, gap, left, bottom, h = 0.168, 0.034, 0.012, 0.34, 0.5
    boxes = [[left + k * (W + gap), bottom, W, h] for k in range(5)]
    steps = [
        ("Vicon markers", "1  Track the probe", "rigid-body fit of 3 cuff markers,\nCAD offset to the probe tip"),
        ("B-mode frame", "2  Detect bone", "brightest echo + shadow transition\nper image column"),
        ("MRI + global registration", "3  Slice MRI at tracked pose", "predicted bone surface is ~1 cm off\n(calibration and tracking error)"),
        (None, "4  Refine pose with 2D ICP", "align ultrasound surface to MRI\nsurface, fold correction into pose"),
        (None, "5  Render safe corridor", "insertion cones defined in MRI,\nintersected with the image"),
    ]

    axes = []
    for k, box in enumerate(boxes):
        ax = fig.add_axes(box, projection="3d" if k == 0 else None)
        axes.append(ax)
        tag, title, caption = steps[k]
        fig.text(box[0], 0.95, title, fontsize=12, weight="bold", color=viz.INK)
        if tag:
            fig.text(
                box[0], 0.88, f"input: {tag}", fontsize=8.5, color=viz.INK_MUTED,
                bbox=dict(boxstyle="round,pad=0.3", fc="#f3f2ee", ec="none"),
            )
        fig.text(box[0], 0.1, caption, fontsize=8.5, color=viz.INK_MUTED, va="center", linespacing=1.4)

    # 1 tracking
    _panel_tracking(axes[0], markers, sweep.T_world_probe[i])
    _panel_legend(
        axes[0],
        [
            _dots("#3b3b38", "Vicon markers (M1-M3)"),
            _line(AXIS_COLORS[0], "x / y / z axes of cuff and probe-tip frames"),
            _patch("#e6e5e0", "ultrasound image plane", ec=viz.C_PLANE, alpha=1),
        ],
    )

    # 2 detection
    _image_panel(axes[1], viz.frame_overlay(frame), frame)
    c0, _ = viz.sector_bounds(frame)
    cols = np.arange(frame.shape[1]) - c0
    axes[1].plot(cols * 2, dbg["brightest"] * 2, ".", ms=2, color="#2a78d6")
    axes[1].plot(cols * 2, dbg["transition"] * 2, ".", ms=2, color="#eda100")
    axes[1].plot(cols * 2, res.profiles_auto[i] * 2, "-", lw=2.2, color=viz.C_US)
    viz.add_scale_bar(axes[1])
    _panel_legend(
        axes[1],
        [
            _dots("#2a78d6", "brightest echo per column"),
            _dots("#eda100", "shadow transition per column"),
            _line(viz.C_US, "detected bone surface"),
        ],
    )

    # 3 slice at tracked pose
    _image_panel(axes[2], viz.frame_overlay(frame, r.profile_us, r.profile_before), frame)
    _panel_legend(
        axes[2],
        [_line(viz.C_US, "bone surface in ultrasound"), _line(viz.C_MRI, "bone surface from MRI, tracked pose")],
    )

    # 4 refinement
    pos = axes[3].get_position()
    axes[3].set_position([pos.x0 + 0.03, pos.y0 + 0.04, pos.width - 0.03, pos.height - 0.04])
    _panel_icp(axes[3], r)
    axes[3].legend(
        handles=[
            _line(viz.C_US, "ultrasound surface"),
            _dots(viz.C_MRI_BEFORE, "MRI surface, tracked pose"),
            _dots(viz.C_MRI, "MRI surface, refined pose"),
        ],
        loc="upper left", bbox_to_anchor=(-0.18, -0.2), frameon=False, fontsize=8,
        handlelength=1.6, borderaxespad=0, labelspacing=0.3,
    )

    # 5 safe corridor
    _image_panel(axes[4], viz.frame_overlay(frame, r.profile_us, r.profile_after, safe_mask=res.safe_masks[i]), frame)
    _panel_legend(
        axes[4],
        [
            _line(viz.C_US, "bone surface in ultrasound"),
            _line(viz.C_MRI, "bone surface from MRI, refined pose"),
            _patch(viz.C_SAFE, "safe needle corridor"),
        ],
    )

    for k in range(4):
        x = boxes[k][0] + W
        fig.add_artist(
            FancyArrowPatch(
                (x + 0.004, bottom + h / 2), (x + gap - 0.004, bottom + h / 2),
                transform=fig.transFigure, arrowstyle="-|>", mutation_scale=18, color="#8d8c86", lw=1.8,
            )
        )
    fig.savefig(path)
    plt.close(fig)


# --- 2. metrics ------------------------------------------------------------------------

CONFIGS = [
    ("global", "No correction\n(tracker pose only)"),
    ("auto_translation", "Corrected to automatically\ndetected bone"),
    ("translation", "Corrected to local\nrefinement (shift only)"),
    ("rigid", "Corrected to local\nrefinement (shift + rotation)"),
]


def _tile(fig, rect, value, label):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06", fc="#f3f2ee", ec="none", transform=ax.transAxes))
    ax.text(0.06, 0.6, value, fontsize=21, weight="bold", color=viz.INK, transform=ax.transAxes, va="center")
    ax.text(0.06, 0.24, label, fontsize=9.5, color=viz.INK_MUTED, transform=ax.transAxes, va="center")


def _distribution(ax, ann, key, unit, better):
    rng = np.random.default_rng(0)
    for row, (m, _) in enumerate(CONFIGS):
        v = np.array([r[f"{key}_{m}"] for r in ann], float)
        v = v[~np.isnan(v)]
        y = len(CONFIGS) - 1 - row
        color = viz.C_BEFORE if m == "global" else viz.C_AFTER
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(plt.Rectangle((q1, y - 0.22), q3 - q1, 0.44, fc=color, alpha=0.18, ec="none"))
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=22, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([med, med], [y - 0.3, y + 0.3], color=viz.INK, lw=2.2, zorder=4)
        ax.text(med, y + 0.33, f"median {med:.1f} {unit}", fontsize=8.5, color=viz.INK, ha="center", va="bottom")
    ax.set_yticks(range(len(CONFIGS)), [c[1] for c in CONFIGS][::-1])
    ax.set_ylim(-0.6, len(CONFIGS) - 0.1)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"{unit}   ({better})", fontsize=9, color=viz.INK_MUTED)


def fig_metrics(rows, summary, path):
    ann = [r for r in rows if r["annotated"]]
    ann_err = [{f"err_{m}": r[f"err_{m}_mm"] for m, _ in CONFIGS} for r in ann]
    ann_iou = [{f"iou_{m}": r[f"iou_{m}"] for m, _ in CONFIGS} for r in ann]
    g, t = summary["methods"]["global"], summary["methods"]["translation"]
    improved = sum(r["iou_translation"] > r["iou_global"] for r in ann)

    fig = plt.figure(figsize=(13, 6.6))
    tiles = [
        (f"{g['surface_err_mean_mm']:.1f} → {t['surface_err_mean_mm']:.1f} mm", "mean bone-surface depth error"),
        (f"{g['iou_mean']:.0f}% → {t['iou_mean']:.0f}%", "overlap (IoU) of MRI prediction and ultrasound"),
        (f"{improved} / {len(ann)}", "annotated frames improved by refinement"),
    ]
    for k, (v, l) in enumerate(tiles):
        _tile(fig, [0.03 + k * 0.325, 0.8, 0.30, 0.16], v, l)

    ax1 = fig.add_axes([0.17, 0.1, 0.36, 0.56])
    ax2 = fig.add_axes([0.62, 0.1, 0.36, 0.56])
    ax1.set_xlim(0, 17)
    ax2.set_xlim(45, 100)
    _distribution(ax1, ann_err, "err", "mm", "lower is better")
    _distribution(ax2, ann_iou, "iou", "%", "higher is better")
    ax1.set_title("Bone-surface depth error per frame", loc="left", pad=12)
    ax2.set_title("Overlap (IoU) per frame", loc="left", pad=12)
    ax2.set_yticklabels([])
    fig.text(
        0.03, 0.015,
        f"Each dot is one of the {len(ann)} annotated frames; shaded band = interquartile range, black tick = median.",
        fontsize=8.5, color=viz.INK_MUTED,
    )
    fig.savefig(path)
    plt.close(fig)



# --- 3. 3D reconstruction ---------------------------------------------------------------


def fig_3d(res, mri, path):
    idx = np.flatnonzero(res.labelled)
    profiles = [res.profiles_label[i] for i in idx]
    fig = plt.figure(figsize=(13, 4.2))
    for k, (title, Ts) in enumerate(
        [
            ("Tracked poses (global registration)", [res.T_global[i] for i in idx]),
            ("After local MRI-ultrasound refinement", [res.per_frame["translation"][i].T_mri_probe for i in idx]),
        ]
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        viz.draw_scene(ax, mri, us_cloud=us_point_cloud(profiles, Ts), zoom=1.3)
        ax.set_title(title, y=0.95)
    handles = [
        plt.Line2D([], [], ls="", marker="o", color="#6f6e69", label="MRI bone surface (ground truth)"),
        plt.Line2D([], [], ls="", marker="o", color=viz.C_US, label="bone surface reconstructed from tracked ultrasound"),
        plt.Line2D([], [], ls="", marker="^", color=viz.C_SAFE, label="safe insertion cones (L1/L2, L2/L3, L3/L4)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(wspace=-0.05, left=0, right=1, top=1, bottom=0.06)
    fig.savefig(path)
    plt.close(fig)


# --- 4. gallery ---------------------------------------------------------------------


def fig_gallery(sweep, res, path, n=8):
    idx = [i for i in range(len(sweep.frames)) if res.safe_masks[i].any()]
    idx = [idx[k] for k in np.linspace(0, len(idx) - 1, n).astype(int)]
    fig, axes = plt.subplots(2, n // 2, figsize=(10, 5.6))
    for ax, i in zip(axes.ravel(), idx):
        img = viz.frame_overlay(
            sweep.frames[i],
            res.profiles_label[i] if res.labelled[i] else None,
            res.mri_display_profiles[i],
            safe_mask=res.safe_masks[i],
        )
        _image_panel(ax, img, sweep.frames[i])
        state = "bone visible, refined" if res.labelled[i] else "interspace, tracked"
        ax.set_title(f"frame {sweep.frame_ids[i]} · {state}", fontsize=8.5, loc="left", fontweight="normal", color=viz.INK_MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=LEGEND_HANDLES, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.savefig(path)
    plt.close(fig)
