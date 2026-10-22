# Dataset

All data were acquired on a commercial lumbar-puncture training phantom
(Ultrasound Lumbar Puncture Simulator, GT Simulators). **No human subject data.**

| Path | Content |
|---|---|
| `mri/phantom_mri_pointcloud.npy` | (183 768, 3) bone-surface points of the phantom segmented from a 3T MRI (GE Signa Premier), mm, MRI frame |
| `sweep/frames/frame_XX.png` | 44 B-mode frames (640x480 screen capture of a GE Logiq E9 with a 9L linear probe, paramedian view) |
| `sweep/labels/frame_XX.png` | Same frames with the spinous-process bone surface hand-traced in red (30 frames; the 14 interspace frames show no spinous process and carry no trace) |
| `sweep/vicon_markers.csv` | Vicon sample per frame: `timestamp, m1x, m1y, m1z, m2x, ..., m3z` (mm) for the three cuff markers |
| `sweep/probe_poses_reference.npy` | Probe-tip poses saved during the experiment; used only by the tests to check that `lpguide.tracking` reproduces them |
| `registration/global_registration.json` | World -> MRI transform from the offline landmark + ICP registration |

Frame `XX` and row `XX - 10` of the Vicon CSV were recorded in the same
acquisition loop iteration.
