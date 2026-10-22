"""Render the demo animation (GIF + MP4) of the guidance pipeline.

    python scripts/make_demo.py

Left: live ultrasound with the bone surface, the MRI-predicted bone surface and
the safe needle region. Right: the tracked image plane moving over the MRI
spine while the ultrasound bone reconstruction builds up.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lpguide import viz  # noqa: E402
from lpguide.config import RESULTS_DIR  # noqa: E402
from lpguide.data import load_global_registration, load_mri_points, load_sweep  # noqa: E402
from lpguide.pipeline import run_sweep, us_point_cloud  # noqa: E402

BG, INK, INK_MUTED = "#ffffff", "#0b0b0b", "#52514e"
FPS = 5


def render_frames(out_dir):
    sweep, mri = load_sweep(), load_mri_points()
    res = run_sweep(sweep, mri, load_global_registration())
    n = len(sweep.frames)

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    for i in range(n):
        fig = plt.figure(figsize=(10, 4.2), facecolor=BG, dpi=160)
        fig.text(0.02, 0.93, "Ultrasound-guided lumbar puncture", fontsize=15, weight="bold")
        fig.text(
            0.02,
            0.87,
            "Tracked ultrasound registered to an MRI model of the spine; safe needle "
            "corridors rendered live in the image",
            fontsize=9.5,
            color=INK_MUTED,
        )

        ax = fig.add_axes([0.02, 0.1, 0.4, 0.72])
        img = viz.frame_overlay(
            sweep.frames[i],
            res.profiles_label[i] if res.labelled[i] else None,
            res.mri_display_profiles[i],
            safe_mask=res.safe_masks[i],
        )
        ax.imshow(viz.to_rgb(viz.crop_sector(img, sweep.frames[i])))
        ax.set_axis_off()
        status = "bone visible: pose refined against MRI" if res.labelled[i] else "interspace: pose tracked"
        ax.text(
            0.0, -0.04, f"frame {i + 1}/{n}  ·  {status}", transform=ax.transAxes,
            fontsize=8.5, color=INK_MUTED, va="top",
        )

        # 3D axes render into a square viewport; make it large and transparent
        # so the wide sweep is not cropped at the sides.
        ax3 = fig.add_axes([0.41, -0.32, 0.6, 1.5], projection="3d")
        ax3.set_facecolor("none")
        seen = [k for k in range(i + 1) if res.labelled[k]]
        cloud = us_point_cloud([res.profiles_label[k] for k in seen], [res.T_track[k] for k in seen])
        viz.draw_scene(
            ax3, mri, T_mri_probe=res.T_track[i], us_cloud=cloud, zoom=1.2, mri_color="#3b3b38",
            xlim=(-125, 125),
        )
        ax3.view_init(elev=18, azim=-62 + 10 * np.sin(2 * np.pi * i / n))

        handles = [
            plt.Line2D([], [], color=viz.C_US, lw=2.5, label="bone (ultrasound)"),
            plt.Line2D([], [], color=viz.C_MRI, lw=2.5, label="bone (MRI model)"),
            plt.Rectangle((0, 0), 1, 1, color=viz.C_SAFE, alpha=0.6, label="safe needle corridor"),
            plt.Rectangle((0, 0), 1, 1, fc="#e6e5e0", ec=viz.C_PLANE, label="ultrasound image plane"),
        ]
        leg = fig.legend(
            handles=handles, loc="lower right", ncol=2, frameon=False, fontsize=8.5,
            bbox_to_anchor=(0.99, 0.0),
        )
        for t in leg.get_texts():
            t.set_color(INK)
        fig.savefig(out_dir / f"{i:03d}.png", facecolor=BG)
        plt.close(fig)
        print(f"\rrendered {i + 1}/{n}", end="", flush=True)
    print()


def encode(frames_dir, out_dir):
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found: install it to encode the GIF/MP4 (frames kept in %s)" % frames_dir)
    pattern = str(frames_dir / "%03d.png")
    mp4, gif = out_dir / "demo.mp4", out_dir / "demo.gif"
    # Hold the last frame for a beat before looping.
    common = ["-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern]
    subprocess.run(
        ["ffmpeg", *common, "-vf", "tpad=stop_mode=clone:stop_duration=1.5,scale=1600:-2:flags=lanczos",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart", str(mp4)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", *common, "-vf",
         "tpad=stop_mode=clone:stop_duration=1.5,scale=1200:-1:flags=lanczos,split[a][b];"
         "[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
         "-loop", "0", str(gif)],
        check=True,
    )
    return mp4, gif


def main():
    out_dir = RESULTS_DIR / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        render_frames(tmp)
        shutil.copy(tmp / "012.png", out_dir / "demo_still.png")
        for p in encode(tmp, out_dir):
            print(f"wrote {p.relative_to(RESULTS_DIR.parent)} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
