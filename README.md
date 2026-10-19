# MarmoVR Viewer

> [!IMPORTANT]
> **Run this module on the Tracking PC under Linux.**
>
> Before launching, set **`resultsDir`**, **`expID`**, **`ext2D`**, and **`ext3D`** in `Settings/Settings.txt` to match the recorded experiment.
>
> The supplied configuration contains a Windows results path. Replace it with the appropriate Linux path.

Offline replay and visualization for the MarmoVR virtual reality setup. The application reads recorded camera videos and tracking results, displays detection and DeepLabCut keypoint overlays, and shows reconstructed positions for inspection and quality control.

## Features

- Four-camera video replay with tracking and keypoint overlays.
- Full-frame or cropped-image display with per-camera rotation in crop mode.
- Detection-based or DLC-derived “cyclop” position display.
- Keypoint marker sizes weighted by confidence.
- Position trails with an adjustable length.
- XY and 3D position plots with velocity and gaze-direction vectors.
- A PyQt5 interface for subject and trial selection.
- Playback, pause, frame stepping, timeline navigation, and playback speeds from 1/8× to 8×.
- MP4 export of camera panels and position plots.
- Optional UDP position exchange with the rendering computer.

## Project files

| Path | Purpose |
| --- | --- |
| `Viewer.py` | Application entry point, result loading, video replay, position plots, networking, export, and GUI. |
| `Settings/Settings.txt` | Experiment paths, result suffixes, display options, playback settings, and camera configuration. |

Recorded videos and tracking results are not included in this archive. The viewer uses previously computed results and does not require a trained model or run DLC inference.

## Requirements

Use a Python environment with a graphical desktop session on the Linux Tracking PC. The application depends on:

- Python and NumPy.
- OpenCV with graphical window support.
- Matplotlib with an interactive plotting backend.
- PyQt5.

The archive does not provide an environment specification or pinned dependency list. Use an existing compatible environment when available. For a separate environment, the following commands install the imported packages, but do not define a validated version combination:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python matplotlib PyQt5
```

Use `opencv-python` rather than a headless build, since video replay opens graphical windows.

## Configuration and launch

Edit `Settings/Settings.txt` before launching. This file uses **tab-separated setting names and Python values**, not assignment statements. Preserve tabs, quotes, and Python boolean syntax when editing.

Set the following values, adapting the path, experiment ID, and suffixes to the local data:

```text
resultsDir	'/disks/ssd/Results'
expID	'Intruder_Task'
ext2D	'_pos2D+DLC (new)'
ext3D	'_pos3D+DLC (new)'
xMonitWin	340
```

The suffixes above match the supplied viewer configuration. They must match the actual filenames: for example, recordings using `_pos2D+DLC2D.npy` and `_pos3D+DLC3D.npy` require `ext2D` and `ext3D` to be changed accordingly.

Activate the selected environment and start the application from the project folder:

```bash
source .venv/bin/activate
python Viewer.py
```

The application resolves relative paths from the directory containing `Viewer.py`.

## Setup configuration

Before the first replay:

1. Check `resultsDir` and `expID`. The viewer loads the experiment index from `resultsDir/<expID>/<expID>_files.tsv`. Use an explicit experiment ID; see troubleshooting below.
2. Check that trial files exist under `resultsDir/<expID>/<subjectID>/`, using the filename stem recorded in the experiment index.
3. Match `ext2D` and `ext3D` to the saved NumPy result suffixes, excluding `.npy`.
4. Keep `camList = [0, 1, 2, 3]` for the supplied setup. Check `cropSize` and `rotateCamList` against the recorded videos.
5. Choose `viewMode = 2` for camera videos or `3` for position plots. Set `showTrack`, `showDLC`, `useCyclop`, and `trailFrames` as needed.
6. Set `startFrame` and `endFrame` to restrict playback. Frame indices start at zero; `endFrame = 0` uses the available recording length.
7. Leave `sendPos3D = False` for local inspection, or check the rendering address and UDP port in `Viewer.py` before enabling streaming.

Use `imgMode = 'full'` to display full frames at half resolution, or `'crop'` to display the configured region of interest. With `useCyclop = True`, the viewer displays the DLC-derived cyclop position; with `False`, it displays the detected position. This follows the implementation; the supplied settings comment describes the toggle in reverse.

## Replay a trial

1. Launch `Viewer.py` in the selected environment.
2. Select the **Subject** and **Trial** in the interface.
3. Choose **2D videos** or **3D plots**.
4. Adjust **Show DLC**, **Show Track**, **Use cyclop**, and the trail length.
5. Enable **Save video** before starting if an export is needed; review the export limitation below.
6. Click **Start player**, then **Play**.
7. Use **Play/Pause**, **−1 / +1**, **−10 / +10**, the timeline slider, and the speed selector to inspect the results.
8. Click **Stop player** before selecting another trial or changing the view mode.

In video mode, `showTriang = True` adds reconstructed position coordinates to the camera panel when the 3D result file is available.

## Network integration

| Connection | Default configuration |
| --- | --- |
| Tracking computer address in `IP` | `192.168.0.2` |
| Rendering computer | `192.168.0.1` |
| UDP positions sent to rendering | Port `50771` |

Set `sendPos3D = True` to stream reconstructed positions during replay. Update `IP['Rendering']` and `UDPserverRendering` near the top of `Viewer.py` to match the installation.

Messages contain three comma-separated coordinates formatted as `x,y,z`, with three decimal places. The transmitted position always comes from `pos(Cyclop)`, independently of the **Use cyclop** display toggle.

## Input data and exported outputs

The experiment index is a tab-separated file with a header row. The viewer reads the experiment ID from column 1, subject ID from column 2, and trial filename stem from column 5.

Trial inputs are read from `resultsDir/<expID>/<subjectID>/`:

| File | Purpose |
| --- | --- |
| `<trial>_cam0.mp4` … `<trial>_cam3.mp4` | Recorded camera videos for 2D replay. |
| `<trial><ext2D>.npy` | Structured 2D tracking and keypoint results for video replay. |
| `<trial><ext3D>.npy` | Structured 3D results for position plots, triangulation coordinates, and UDP streaming. |

Use structured NumPy arrays matching the MarmoVR tracking schema. The viewer expects fields such as `pos(UL)_cam0`, `pos(0)_cam0`, camera-specific keypoint positions and probabilities, and 3D fields including `imgIndex`, `time`, `pos(Cyclop)`, `vel(Cyclop)`, and `gazeDir`. Required fields depend on the selected visualization. Keypoint names are defined in `keyColors` in `Viewer.py`.

The 3D plot player reads the 3D results without opening camera videos. Standard DLC CSV files cannot be loaded directly.

When **Save video** is enabled, exports are written beside the input files:

| View mode | Output filename |
| --- | --- |
| 2D videos | `<trial>_panel.mp4` |
| 3D plots | `<trial>_plot3D.mp4` |

The suffix ` (with DLC)` is inserted before `.mp4` when DLC display is enabled. Exports use OpenCV's `mp4v` codec. Stop the player to close the output file, and rename previous exports before repeating an export if they need to be preserved.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Experiment index cannot be opened | Replace the supplied Windows path with the Linux results path; check `expID` and `<expID>_files.tsv`. |
| Startup fails with an empty `expID` | Use an explicit experiment ID. Although the settings mention manual log selection, this version does not initialize `resLogFile` when `expID` is empty. |
| Video or NumPy file cannot be opened | Check the experiment/subject folders, trial filename stem, camera filenames, and result suffixes. |
| Missing field or incorrect overlays | Check the result array schema, keypoint names, frame correspondence, crop mode, and camera rotations. |
| Camera subset or reordered cameras fail | Retain `[0, 1, 2, 3]`; the video player mixes camera IDs and list indices. |
| GUI or video window does not appear | Check the graphical session, PyQt5 installation, Matplotlib backend, and desktop OpenCV package. |
| Export raises an error about `saveNoShow` | The setting is loaded into `self.s`, but not copied to `self.saveNoShow`. Initialize the runtime attribute from the setting before using export. |
| 3D export fails at `tostring_rgb()` | This canvas method is unavailable in some newer Matplotlib versions; use a compatible environment or adapt the image conversion. |
| Rendering receives no position | Check `sendPos3D`, the 3D result file, rendering IP address, UDP port, and network access. |

The export path has the implementation limitations described above. Playback and export have not been validated against an included sample dataset, since the archive contains no recordings.
