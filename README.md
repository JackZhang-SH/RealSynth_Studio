# RS Studio (RealSynth Dataset Studio) — Blender Add-on

> 🎥 **Watch the demo video (feature overview & usage tutorial)**
[![Watch the video](https://img.youtube.com/vi/JmlZSmdW96Q/maxresdefault.jpg)](https://youtu.be/JmlZSmdW96Q)

RS Studio is a **Blender add-on for dataset generation and camera virtualization**.  
It supports importing or generating multi-view camera rigs, rendering multi-frame datasets, and exporting data in formats commonly used by **COLMAP / NeRF / Instant-NGP / 3D Gaussian Splatting (3DGS)** pipelines, including **COLMAP-format 3DGS datasets with mesh-sampled ground-truth point clouds**.

If you're interested in the research context behind RS Studio—especially **TACV** and its 3DGS variant—the following repos provide the method details and training pipelines:

## 🔗 Related Repositories (TACV Ecosystem)

TACV (Time-Archival Camera Virtualization) is designed for dynamic sports scenes (e.g., stadium events) where camera parameters can be pre-calibrated and remain consistent. 

Instead of treating the entire dynamic sequence as a single reconstruction problem, TACV organizes the scene into a time-indexed set of “archival” multi-view snapshots, enabling **per-time-step 3D reconstruction** and consistent evaluation across time.

RS Studio aligns naturally with TACV: Blender cameras are explicitly defined with known intrinsics/extrinsics, matching TACV’s pre-calibrated stadium-rig assumption. RS Studio focuses on generating and exporting the resulting per-frame datasets, which can then be used for downstream training and evaluation.

- **TACV Method (origin of TACV export in RS Studio)**
  - **Time-Archival Camera Virtualization for Visual Performance and Sports (TACV)**
  - Repo: [Time-Archival-Camera-Virtualization-for-Visual-Performance-and-Sports](https://github.com/JackZhang-SH/Time-Archival-Camera-Virtualization-for-Visual-Performance-and-Sports)
  - What it is: the core TACV method and the reference definition/usage of the TACV dataset format.

- **TACV 3DGS Variant (recommended to pair with RS Studio exports)**
  - **Time Archival 3DGS**
  - Repo: [TimeArchival3DGS](https://github.com/JackZhang-SH/TimeArchival3DGS)
  - Best match with RS Studio: use **COLMAP + 3DGS (Surface Points)** export in RS Studio to generate
    COLMAP-style frames + `fused_points.ply`, a convenient setup for 3DGS initialization and training.

Below is a quick UI pointer and screenshot.

> UI entry: **3D Viewport → Sidebar (N-panel) → RS Studio**

![RS Studio UI](./assets/3.png)

---


## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [UI Overview](#ui-overview)
  - [Main Panel: RS Studio](#main-panel-rs-studio)
  - [Panel: Camera Track → 3DGS](#panel-camera-track--3dgs)
- [Supported Formats](#supported-formats)
- [Exported Dataset Layouts](#exported-dataset-layouts)
- [Camera Naming & Splits](#camera-naming--splits)
- [Development Notes](#development-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Key Features
- **Camera import** from multiple dataset formats (COLMAP, Instant-NGP, NeRF-Synthetic, TACV, CMU Panoptic).
- **Camera generation tools** for building repeatable multi-view rigs (hemisphere / sphere / elliptical ring).
- **Stadium-layout camera generator** (ring + tile + sideline + goal-line rigs) with stable naming/indexing.
- **Dataset export** with per-frame folders, automated rendering, and optional point cloud generation for 3DGS-ready COLMAP datasets.
- **3DGS point cloud routes**
  - **Surface-sampling route**: sample points on mesh surfaces and fuse via voxel grid.
  - **Depth-based route**: render depth and back-project to points (depending on export mode).
- **Utility tools** such as lighting/weather presets and camera track export.

---

## Installation

### Requirements
- **Blender 4.0+**
- Optional (feature-dependent):
  - `pysolar` for sun-position lighting (Outdoor mode).
  - `colmap` CLI if you want automatic COLMAP TXT → BIN conversion (nice-to-have).

### Option A — Install from ZIP (recommended)
1. Download this repository as a ZIP (or a release ZIP).
2. In Blender: **Edit → Preferences → Add-ons → Install…**
3. Select the ZIP and install.
4. Enable the add-on: search for **“RealSynth Dataset Studio”**.

### Option B — Install from source
1. Clone the repo.
2. Copy the add-on folder into your Blender add-ons directory:
   - Windows: `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\`
   - macOS: `~/Library/Application Support/Blender/<version>/scripts/addons/`
   - Linux: `~/.config/blender/<version>/scripts/addons/`
3. Enable the add-on in Blender.


## Optional: Install `pysolar` (for Sun-Position Outdoor Lighting)

RS Studio uses the `pysolar` Python package to compute physically-based sun direction from date/time/location.
This is optional — you only need it if you want the **Sun-Position Outdoor Lighting** feature.

### 1) Find Blender’s bundled Python

Blender ships with its own Python interpreter. You must install `pysolar` into **that** Python (not your system Python).

Typical locations:

- **Windows**
  - `C:\Program Files\Blender Foundation\Blender <version>\<version>\python\bin\python.exe`
- **macOS**
  - `/Applications/Blender.app/Contents/Resources/<version>/python/bin/python3`
- **Linux**
  - `<blender_install_dir>/<version>/python/bin/python3`

> Tip: If you cannot find it, search your Blender installation folder for `python.exe` (Windows) or `python3` (macOS/Linux).

---

### 2) Install via pip

#### Windows (PowerShell)

```powershell
$blenderPy = "C:\Program Files\Blender Foundation\Blender 4.0\4.0\python\bin\python.exe"
& $blenderPy -m pip install --upgrade pip
& $blenderPy -m pip install pysolar
```

#### macOS / Linux (Terminal)

```bash
BLENDER_PY="/path/to/blender/<version>/python/bin/python3"
"$BLENDER_PY" -m pip install --upgrade pip
"$BLENDER_PY" -m pip install pysolar
```

---

### 3) Verify in Blender

Restart Blender, then open **Scripting → Python Console** and run:

```python
import pysolar
print("pysolar OK")
```

If you see `pysolar OK`, you’re done.

---

### Troubleshooting

- **`pip` not found**
  - Try: `"$BLENDER_PY" -m ensurepip --upgrade`
  - Then rerun the install commands.

- **Permission error (macOS/Linux)**
  - Use `--user`:
    - `"$BLENDER_PY" -m pip install --user pysolar`

- **Wrong Python**
  - If `import pysolar` works in your system Python but not inside Blender, it means you installed it into the wrong interpreter.
    Make sure you are using Blender’s bundled Python path above.

---

## Quick Start

### 1) Prepare your scene
- Open your Blender scene (static or animated).
- Set materials/textures if you want realistic RGB renders.
- Set render resolution and engine as desired.

### 2) Create or import cameras
**Generate cameras**
- Open **RS Studio** panel.
- Choose a sampling strategy (HEMI / SPHERE / ELLIPSE).
- Set `Images Per Frame`, `Radius/Height`, and `Target Point`.
- Click **Generate Cameras**.

**Import cameras**
- Choose an import format (COLMAP / NGP / TACV / NeRF-Synth / Panoptic).
- Set `Import Directory` and `Import Scale`, then click **Import Cameras**.

### 3) Export a dataset
- Set:
  - **Output Directory**
  - **Start Frame / End Frame**
  - **Export Format**
- Click **Generate Dataset**.

---

## UI Overview

### Main Panel: RS Studio
Location: **3D Viewport → Sidebar (N-panel) → RS Studio → “RealSynth Dataset Studio”**

This main panel is organized into multiple boxes (partitions). Each partition is explained below.

---

#### 1) Import Cameras
Use this to bring an existing multi-view camera rig into Blender.

- **Import Format**
  - `COLMAP` (expects `cameras.txt` + `images.txt`)
  - `Instant-NGP` (expects `transforms*.json`)
  - `TACV` (expects `transforms.json`, `transforms_valid.json`, `transforms_test.json`)
  - `NeRF-Synthetic` (expects `transforms_train.json`, `transforms_val.json`, `transforms_test.json`)
  - `CMU Panoptic` (expects a calibration `.json`)
- **Import Directory**
  - Folder containing the files for the selected format.
- **Import Scale**
  - Applies a uniform scale when converting dataset coordinates into Blender units.
  - Useful if the source dataset is in meters but your Blender scene is scaled differently.
- **Import Cameras**
  - Creates Blender camera objects and assigns intrinsics/extrinsics.

---

#### 2) Lighting
Quick lighting presets for consistent dataset rendering.

- **Location**
  - `INDOOR` / `OUTDOOR`
- **Weather**
  - `SUNNY`, `RAIN`, `FOG`
- **Fog Visibility / Rain Intensity**
  - Only shown when the corresponding weather is selected.
- **Date / Hour / Latitude / Longitude** (Outdoor only)
  - Used to estimate sun direction (requires `pysolar`).
- **Apply Lighting**
  - Builds the lights + world setup created by RS Studio.
- **Clear Lighting**
  - Removes the lighting setup created by RS Studio.

Tip: If you want perfectly repeatable renders, keep lighting settings fixed across runs.

---

#### 3) Camera Generator
Generates a set of RS Studio cameras around a target point.

- **Images Per Frame**
  - Number of cameras to generate (one camera per sampled position).
- **Sampling Strategy**
  - `HEMI`: Fibonacci hemisphere sampling
  - `SPHERE`: Fibonacci sphere sampling
  - `ELLIPSE`: Elliptical ring sampling
- **Radius / Height**
  - `Radius`: used by all strategies
  - `Height`: used by hemisphere/sphere samplers (controls the vertical offset)
  - Ellipse mode uses `Radius` as major axis `a`.
- **Ellipse Minor / Ellipse Center** (Ellipse only)
  - `Ellipse Minor`: minor axis `b`
  - `Ellipse Center`: XYZ offset of the ellipse center
- **Focal Length (mm)**
  - Sets the camera lens value for generated cameras (perspective cameras).
- **Target Point**
  - Generated cameras will look at this point.
- **Generate Cameras**
  - Creates cameras named like `RS_Studio_<id>_train`.
  - Also creates marker objects for visibility and organizes cameras into split collections.
- **Clear Cameras**
  - Removes all RS Studio cameras and their markers.
- **Set Split (Train / Valid / Test)**
  - Select cameras in the viewport and assign them to a split.
  - This renames cameras with suffixes (`_train`, `_valid`, `_test`) and moves them into split collections.

---

#### 4) Stadium Layout
Generates stadium-style rigs with stable naming and indexing.

- **Pitch Length / Pitch Width (meters)**
  - Physical size of the field used as a reference for placing cameras.
- **Ring Cameras**
  - `Ring K`: number of cameras around the ring
  - `Ring Radius`, `Ring Height`
  - `Ring Lens (mm)`
- **Tile Cameras**
  - Grid layout: `Tiles NX`, `Tiles NY`
  - `Tile Cams Per Zone`: multiple side views per tile zone
  - `Tile Side Mode`: `LEFT`, `RIGHT`, or `BOTH`
  - `Tile Lens (mm)`
  - `Tile Cam Height`, `Tile Target Height`
  - `Row Stagger X Ratio`: offsets alternating rows to avoid perfect alignment
- **Sideline Cameras**
  - Enable/disable sideline cameras
  - `Cams Per Side`, `Out Offset`, `Height`, `Lens (mm)`, `Target Height`
- **Goal-line Cameras**
  - Enable/disable goal cameras
  - `Cams Per Side`, `Behind Offset`, `Height`, `Lens (mm)`, `Target Height`
- **Generate Stadium Cameras**
  - Creates cameras tagged with `rs_group="stadium"` so they can be cleared independently.
- **Clear Stadium Cameras**
  - Removes stadium cameras and helper empties (e.g., `TileCenter_*`).

---

#### 5) Dataset Generation
Renders and exports datasets across a frame range.

- **Output Directory**
  - Root folder. RS Studio writes `frame_<F>/...` subfolders into it.
- **Frame Range**
  - `Start Frame`, `End Frame`
- **Export Format**
  - `Instant-NGP`
  - `NeRF-Synthetic`
  - `TACV`
  - `COLMAP (poses-only)`
  - `COLMAP + 3DGS (Surface Points)`
  - `COLMAP + 3DGS (Depth Points)`
  - `Depth (.pt/.npy)`
- **NGP Settings**
  - `AABB Scale`
- **COLMAP + 3DGS (Surface Points)** (mesh surface sampling route)
  - **Mesh**: choose a mesh object to sample points from
  - **Voxel Size**: voxel-grid fusion resolution
  - **Max Points**: cap after fusion
  - **Color Mode**: choose how point colors are assigned
  - **AABB Bounds (optional)**: restrict sampling to a 3D box (OpenCV/COLMAP world coordinates)
  - **Thickness**: epsilon and two-sided options for thin geometry
- **COLMAP + 3DGS (Depth Points)** (depth render + back-projection route)
  - `Depth Stride`: subsampling factor (1 = dense, larger = faster)
  - `Voxel Size`, `Max Points`: fusion settings for back-projected points
- **Generate Dataset**
  - Starts incremental rendering across frames and cameras.
- **Cancel Generation**
  - Stops an in-progress generation job.

---

### Panel: Camera Track → 3DGS
This panel exports a **per-frame camera track** (intrinsics + OpenCV extrinsics) as a JSON file. Useful for 3DGS rendering tools, replay cameras, or downstream evaluation.

#### Track Controls
- **Track Camera**
  - The Blender camera object you want to animate/export.
- **Track Mode**
  - `KEYFRAMED`: you animate the camera manually; RS Studio only exports.
  - `ORBIT`: auto-build an orbit path around a target.
  - `LINEAR`: auto-build a linear interpolation path.
  - `FOLLOW`: chase-camera behavior that keeps an offset behind the target.
- **Frame Range**
  - Start/end frames for exporting the track.
- **Target Point**
  - The point to orbit/look at.

#### Mode-specific parameters
- **ORBIT**
  - `Orbit Frames`, `Orbit Radius`, `Orbit Height`, `Clockwise`
- **LINEAR**
  - `Linear Start`, `Linear End`
- **FOLLOW**
  - `Follow Offset`, `Follow Smoothing`

#### Actions
- **Build Track**
  - Generates camera keyframes for ORBIT/LINEAR/FOLLOW modes.
- **Output JSON Path**
  - Destination for the exported JSON file.
- **Export Track**
  - Writes a JSON list (one entry per frame), including `(fx, fy, cx, cy)` and OpenCV `(R, T)` plus `znear/zfar`.

---

## Supported Formats

### Import formats (cameras into Blender)
- **COLMAP** (`cameras.txt`, `images.txt`)
- **Instant-NGP** (`transforms*.json`)
- **TACV** (`transforms.json`, `transforms_valid.json`, `transforms_test.json`)
- **NeRF-Synthetic** (`transforms_train.json`, `transforms_val.json`, `transforms_test.json`)
- **CMU Panoptic** (calibration `.json`)

### Export formats (datasets from Blender)
Depending on the chosen export mode, RS Studio can produce:
- **NeRF-style transforms** per frame
- **COLMAP pose-only** datasets (cameras/images + placeholder `points3D.txt`)
- **COLMAP + 3DGS surface-sampled points** (`points3D.*` + `fused_points.ply`)
- **COLMAP + 3DGS depth points** (depth render + back-projection + `fused_points.ply`)
- **Depth per camera** (depth `.npy`, optional `.pt` if Torch is available)

---

## Exported Dataset Layouts
RS Studio exports **one folder per animation frame**:

### Instant-NGP (per frame)
```
<output_dir>/
  frame_<F>/
    train/
      render_0000.png
      render_0001.png
      ...
    transforms.json
```

### TACV (per frame)
```
<output_dir>/
  frame_<F>/
    train/
      0.png
      1.png
      ...
    transforms.json
    transforms_test.json   # if any cameras are marked as _test
```

### COLMAP pose-only (per frame)
```
<output_dir>/
  frame_<F>/
    images/
      0000.png
      0001.png
      ...
    sparse/
      0/
        cameras.txt
        images.txt
        points3D.txt        # placeholder (empty)
```

### COLMAP + 3DGS (Surface Points) (per frame)
```
<output_dir>/
  frame_<F>/
    images/
    sparse/0/               # cameras/images/points3D (TXT + optional BIN)
    fused_points.ply        # fused point cloud for 3DGS initialization
```

### COLMAP + 3DGS (Depth Points) (per frame)
Similar to the surface route, but points are reconstructed from depth rendering and fused into `fused_points.ply`.

---

## Camera Naming & Splits

### RS Studio camera naming
- Generated cameras follow:
  - `RS_Studio_<id>_train`
  - `RS_Studio_<id>_valid`
  - `RS_Studio_<id>_test`

### Splits
- Use **Set Split (Train / Valid / Test)** to:
  - rename selected cameras with the split suffix
  - move them into split collections
  - colorize markers for readability

---

## Development Notes
For local development/testing outside Blender, a minimal Python environment is provided in `environment.yml` (uses `fake-bpy-module-4.0`).

Typical workflow:
1. Develop/edit scripts.
2. Reload the add-on in Blender (disable/enable in Preferences). The add-on entry-point supports hot reload.

---

## Troubleshooting
- **Nothing happens after clicking Generate Dataset**
  - Check Blender system console for Python exceptions.
  - Verify `Output Directory` is writable.
- **Sun-position lighting fails**
  - Install `pysolar` into Blender’s Python environment.
- **COLMAP BIN files look stale**
  - RS Studio attempts to regenerate BIN from TXT if `colmap` is available.
  - If `colmap` is missing, RS Studio may delete stale BIN files so pipelines read TXT.

---

## Contributing
Issues and PRs are welcome. If you add new export formats or camera rigs, please also update:
- `operators.py` (properties + operators)
- `ui.py` (controls)
- This README (UI + output layout sections)

---

## License
Choose a license before publishing (MIT / Apache-2.0 / GPL-3.0 are common for Blender add-ons).
Once chosen, add a `LICENSE` file and update this section.

---

## Citation
If you use RS Studio in academic work, please cite it:

```bibtex
@misc{rsstudio,
  title        = {RealSynth Dataset Studio (RS Studio)},
  author       = {Yunxiao Zhang},
  year         = {2025},
  howpublished = {GitHub repository},
}
```
