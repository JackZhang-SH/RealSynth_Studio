# core.py
"""
Core implementation: multi‑camera dataset generator for Blender.

Key classes
-----------
SamplingStrategy                – abstract camera‑position sampler
FibonacciSphereSampling         – default uniform sphere sampling
CameraRig                       – creates/maintains one camera per sample
FrameDatasetRenderer            – renders one frame using the rig
DatasetGenerator                – drives multiple frames incrementally

2025‑05‑30  (multi‑camera refactor)
----------------------------------
* Each sample position is represented by **its own** camera object (``camera
  {id}``) that persists across frames.  This emulates a real‑world rig where
  many cameras capture simultaneously.
* The original logic that teleported a single camera has been retired.
* A future feature may let users pick a subset of cameras (e.g. for a test
  split).  The design already stores the full ``self.cameras`` list in
  ``DatasetGenerator`` so filtering can be implemented later without major
  refactoring.

All identifiers + comments are English‑only per project guidelines.
"""
from __future__ import annotations
import shutil
from ast import Dict, Set
import json
import math
from abc import ABC, abstractmethod
import os
from pathlib import Path
import subprocess
from typing import Iterator, List, Sequence, Tuple
import mathutils as mu
import bpy
import bpy.path as bpath
from mathutils import Vector
from enum import Enum, auto
import json, math, random
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector, Matrix
def fit_plane_svd(pts_xyz: np.ndarray):
    """
    Fit a plane n^T (x - c) = 0 by SVD.
    Returns: centroid c (3,), unit normal n (3,), rms distance (float)
    """
    assert pts_xyz.ndim == 2 and pts_xyz.shape[1] == 3 and pts_xyz.shape[0] >= 3
    c = pts_xyz.mean(axis=0)
    Q = pts_xyz - c
    _, _, vh = np.linalg.svd(Q, full_matrices=False)
    n = vh[-1, :]
    n = n / (np.linalg.norm(n) + 1e-12)
    dists = Q @ n
    rms = float(np.sqrt(np.mean(dists**2)))
    return c, n, rms

def build_K(fx, fy, cx, cy):
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K

def extrinsics_cv_from_matrix_world(M_bl):
    """
    Compute OpenCV extrinsics from Blender camera world matrix.
    Returns (R_wc: mathutils.Matrix 3x3, C_w: mathutils.Vector 3)
    """
    from mathutils import Matrix
    # Axis converters: OpenCV-world -> Blender-world (A), Blender-local -> OpenCV-local (T)
    A = Matrix((
        (1, 0,  0, 0),
        (0, 0,  1, 0),
        (0,-1,  0, 0),
        (0, 0,  0, 1),
    ))
    T = Matrix((
        (1, 0,  0, 0),
        (0,-1,  0, 0),
        (0, 0,-1, 0),
        (0, 0,  0, 1),
    ))
    # world->camera(OpenCV)
    M_w2cv = T @ M_bl.inverted() @ A
    R_cw = M_w2cv.to_3x3()
    R_wc = R_cw.transposed()
    t_cw = M_w2cv.to_translation()
    C_w  = -(R_wc @ t_cw)
    return R_wc, C_w

def Rt_from_colmap_quat(qw, qx, qy, qz, tx, ty, tz):
    """
    Returns R_cw (3x3), t_cw (3,), R_wc (3x3), C_w (3,)
    """
    import mathutils as mu
    R_cw = mu.Quaternion((qw,qx,qy,qz)).to_matrix()
    R_wc = R_cw.transposed()
    t_cw = mu.Vector((tx,ty,tz))
    C_w = -(R_wc @ t_cw)
    R_cw = np.array(R_cw, dtype=np.float64)
    R_wc = np.array(R_wc, dtype=np.float64)
    C_w  = np.array([C_w.x, C_w.y, C_w.z], dtype=np.float64)
    t_cw = np.array([tx,ty,tz], dtype=np.float64)
    return R_cw, t_cw, R_wc, C_w

def project_cv(K, R_wc, C_w, Xw):
    """
    Xw: (N,3) world points. Returns pixel coords (u,v) and depths Zc.
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    R_cw = R_wc.T
    Xc = (R_cw @ (Xw.T - C_w.reshape(3,1))).T  # (N,3)
    Zc = Xc[:,2].copy()
    uv1 = (K @ (Xc.T / np.clip(Zc, 1e-12, None))).T
    return uv1[:,0], uv1[:,1], Zc

def _intrinsics_from_cam(camdat, scene):
    """
    Robust intrinsics from Blender's view_frame(), guaranteed consistent with
    world_to_camera_view(): handles lens shift, sensor fit, pixel aspect, etc.
    Returns: w,h, fx,fy, cx,cy, dist
    """
    # Output render size in pixels
    scale = scene.render.resolution_percentage / 100.0
    w = int(round(scene.render.resolution_x * scale))
    h = int(round(scene.render.resolution_y * scale))

    # Camera frustum corners in *camera local* (Blender) coords.
    # Each corner v has (x,y,z) with z < 0 (looking along -Z).
    # We convert to normalized projection plane by dividing by -z:
    #   x_n = x/(-z), y_n = y/(-z)
    vf = camdat.view_frame(scene=scene)  # sequence of 4 corners
    xs = [v.x / (-v.z) for v in vf]
    ys = [v.y / (-v.z) for v in vf]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Solve linear maps so that:
    #   u = fx * x_n + cx,     u(x_min)=0, u(x_max)=w
    #   v_top = -fy * y_n + cy, v_top(y_max)=0 (top), v_top(y_min)=h (bottom)
    fx = w / (x_max - x_min)
    fy = h / (y_max - y_min)
    cx = -fx * x_min
    cy =  fy * y_max   # top-left image origin

    dist = {k: float(getattr(camdat, k, 0.0)) for k in ("k1", "k2", "p1", "p2")}
    return w, h, fx, fy, cx, cy, dist
def _colmap_supports_input_type_option(colmap_exe: str) -> bool:
    """Return True iff `colmap model_converter --help` mentions --input_type."""
    try:
        out = subprocess.run(
            [colmap_exe, "model_converter", "--help"],
            capture_output=True, text=True, check=False
        )
        blob = (out.stdout or "") + (out.stderr or "")
        return "--input_type" in blob
    except Exception:
        return False
def _refresh_or_drop_colmap_bins(sparse0: Path) -> None:
    """
    Ensure 3DGS will not read stale *.bin files.
    - If COLMAP exists: regenerate BIN from current TXT (force TXT when possible).
    - If COLMAP is missing or conversion fails: delete stale BIN to force TXT read.
    """
    import os, shutil, subprocess

    imgs_bin = sparse0 / "images.bin"
    cams_bin = sparse0 / "cameras.bin"
    pts_bin  = sparse0 / "points3D.bin"

    colmap_exe = shutil.which("colmap")
    if colmap_exe:
        # If --input_type is unavailable (old COLMAP), remove old BIN first so TXT is used.
        supports_input_type = _colmap_supports_input_type_option(colmap_exe)
        if not supports_input_type:
            for p in (imgs_bin, cams_bin, pts_bin):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

        cmd = [
            colmap_exe, "model_converter",
            "--input_path",  os.fspath(sparse0),
            "--output_path", os.fspath(sparse0),
        ]
        if supports_input_type:
            cmd += ["--input_type", "TXT"]
        cmd += ["--output_type", "BIN"]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except Exception as e:
            print("[RS-Studio] ⚠ model_converter failed; removing stale *.bin to force TXT:", e)

    # No COLMAP or conversion failed → remove BIN so loaders must read TXT
    for p in (imgs_bin, cams_bin, pts_bin):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

# ──────────────────────────────────────────────────────────── #
#  New: export-format enumeration
# ──────────────────────────────────────────────────────────── #
class ExportFormat(Enum):
    NGP          = auto()
    NERF_SYNTH   = auto()
    TACV         = auto()          # Time-Archival Camera Virtualisation
    COLMAP_POSES = auto()      # ----- COLMAP (poses-only) ----
    COLMAP_3DGS_MESH = auto()   # ← NEW: surface-sampling route
    COLMAP_3DGS_DEPTH = auto()  # Depth back-projection route
# ──────────────────────────────────────────────────────────── #
#  New: abstract writer & concrete NGP / NeRF-Synthetic writer
# ──────────────────────────────────────────────────────────── #
class DatasetWriter(ABC):
    """Strategy object that decides *where* each rendered PNG lands and
    eventually writes one (or many) transforms-*.json."""

    def __init__(self, root_out: Path, cam0: bpy.types.Camera) -> None:
        self.root = root_out
        self.cam0 = cam0               # all cameras share intrinsics
        self._scene = bpy.context.scene

        scale  = self._scene.render.resolution_percentage / 100.0
        self.w = int(round(self._scene.render.resolution_x * scale))
        self.h = int(round(self._scene.render.resolution_y * scale))
        # --- focal lengths computed from field-of-view (sensor-fit agnostic) ----
        self.fx = 0.5 * self.w / math.tan(cam0.angle_x * 0.5)
        self.fy = 0.5 * self.h / math.tan(cam0.angle_y * 0.5)

        # global translation-scale (user-controlled)
        self._scale = getattr(bpy.context.scene.rs_settings, "scale", 1.0)

    # --------- public API every subclass must offer --------- #
    @abstractmethod
    def filepath_for(self, cam_obj: bpy.types.Object, global_idx: int) -> Path:
        ...

    @abstractmethod
    def register_frame(
        self, cam_obj: bpy.types.Object, rel_path: str
    ) -> None:
        ...

    @abstractmethod
    def finish(self) -> None:         # called once after all frames done
        ...
    # 额外步数（供进度条预估）；渲染张数外的附加步骤写在这里
    @property
    def extra_steps(self) -> int:
        return 0

    # 若子类需要耗时后处理（例如调用 COLMAP），实现为生成器并在其中 yield
    def postprocess_iter(self) -> Iterator[None]:
        if False:
            yield None
    # apply global scale to translation part of a 4×4 matrix
    def _matrix(self, mat):
        m = [list(r) for r in mat]
        if self._scale != 1.0:
            m[0][3] *= self._scale
            m[1][3] *= self._scale
            m[2][3] *= self._scale
        return m
    
# ---------- Instant-NGP (unchanged behaviour) --------------- #
class NGPDatasetWriter(DatasetWriter):
    def __init__(self, root_out: Path, cam0: bpy.types.Camera) -> None:
        super().__init__(root_out, cam0)
        self._per_frame_meta: dict[int, list] = {}

    def filepath_for(self, cam_obj, global_idx):
        frame = self._scene.frame_current
        frame_dir = self.root / f"frame_{frame}" / "train"
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir / f"render_{global_idx:04d}.png"

    def register_frame(self, cam_obj, rel_path):
        frame = self._scene.frame_current
        # 把前缀 "frame_{idx}/" 剥掉，保证相对路径正确
        prefix = f"frame_{frame}/"
        local_fp = rel_path[len(prefix):] if rel_path.startswith(prefix) else rel_path
        if not local_fp.startswith("./"):
            local_fp = "./" + local_fp            # → "./train/render_0000.png"

        self._per_frame_meta.setdefault(frame, []).append({
            "file_path":        local_fp,
            "transform_matrix": self._matrix(cam_obj.matrix_world),
        })


    def finish(self):
        scene = bpy.context.scene
        w, h, fx, fy, cx, cy, dist = _intrinsics_from_cam(self.cam0, scene)
        aabb = scene.rs_settings.aabb_scale

        for frame, frames_meta in self._per_frame_meta.items():
            out = {
                "camera_angle_x": self.cam0.angle_x,
                "camera_angle_y": self.cam0.angle_y,
                "fl_x": fx, "fl_y": fy,
                **dist,
                "cx": cx, "cy": cy,
                "w":  w,  "h":  h,
                "aabb_scale": aabb,
                "scale": self._scale,
                "frames": frames_meta,
            }
            (self.root / f"frame_{frame}"
             / "transforms.json").write_text(json.dumps(out, indent=4))


    
# ---------- NeRF-Synthetic writer --------------------------- #
_SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


class NeRFSyntheticWriter(DatasetWriter):
    def __init__(self, root_out: Path, cam0):
        super().__init__(root_out, cam0)
        # frame  -> split -> running idx
        self._seq:    dict[int, dict[str, int]]   = {}
        # frame  -> split -> list[dict]
        self._frames: dict[int, dict[str, list]]  = {}

    # ---------- helpers -------------------------------------------------- #
    def _split_of(self, cam_obj: bpy.types.Object) -> str:
        for suf in _SPLIT_MAP:
            if cam_obj.name.endswith(f"_{suf}"):
                return suf
        return "train"

    # ---------- public API ------------------------------------------------ #
    def filepath_for(self, cam_obj, _):
        frame  = self._scene.frame_current
        split  = self._split_of(cam_obj)               # 'train' | 'valid' | 'test'

        # incrementing sequence number per (frame, split)
        self._seq.setdefault(frame, {}).setdefault(split, 0)
        idx = self._seq[frame][split]
        self._seq[frame][split] += 1

        frame_dir = self.root / f"frame_{frame}" / _SPLIT_MAP[split]
        frame_dir.mkdir(parents=True, exist_ok=True)

        # RETURN PATH WITHOUT ".png" SUFFIX:
        # Blender will add ".png" when saving; Nerfstudio JSON expects no ".png" here.
        return frame_dir / f"r_{idx}"

    def register_frame(self, cam_obj, rel_path):
        frame  = self._scene.frame_current
        split  = self._split_of(cam_obj)
        prefix = f"frame_{frame}/"
        local_fp = rel_path[len(prefix):] if rel_path.startswith(prefix) else rel_path

        # If the rel_path comes in without ".png", ensure any consumer adds ".png" when loading.
        if not local_fp.startswith("./"):
            local_fp = "./" + local_fp

        self._frames.setdefault(frame, {}).setdefault(split, []).append({
            "file_path":        local_fp,
            "rotation":         math.pi / 100.0,
            "transform_matrix": self._matrix(cam_obj.matrix_world),
        })

    def finish(self):
        w,h,fx,fy,cx,cy,_ = _intrinsics_from_cam(self.cam0, bpy.context.scene)
        common = {
            "camera_angle_x": self.cam0.angle_x,
            "camera_angle_y": self.cam0.angle_y,
            "fl_x": fx, "fl_y": fy,
            "cx": cx,  "cy": cy,
            "w":  w,   "h":  h,
        }

        for frame, split_dict in self._frames.items():
            frame_dir = self.root / f"frame_{frame}"
            for split, frames in split_dict.items():
                if not frames:
                    continue
                data = {**common, "frames": frames}
                fname = f"transforms_{_SPLIT_MAP[split]}.json"
                (frame_dir / fname).write_text(json.dumps(data, indent=4))

    
# ---------- TACV writer ------------------------------------ #
class TACVDatasetWriter(DatasetWriter):
    """
    Time-Archival Camera Virtualisation (TACV) format

    Directory layout
    ├── frame_1/
    │   ├── train/          ← all rendered PNGs (train + test)
    │   ├── transforms.json
    │   └── transforms_test.json
    ├── frame_2/
    │   └── …               (same structure)
    """
    _DIST_KEYS = ("k1", "k2", "p1", "p2")          # 畸变系数可选

    def __init__(self, root_out: Path, cam0) -> None:
        super().__init__(root_out, cam0)
        # per-frame → list[dict]
        self._train: dict[int, list] = {}
        self._test:  dict[int, list] = {}
        self._seq:   dict[int, int]  = {}           # frame → running idx

   # --------------------------- helpers --------------------------- #
    @staticmethod
    def _split_of(cam: bpy.types.Object) -> str:
        return "test" if cam.name.endswith("_test") else "train"


    # ------------------------------------------------ public
    def filepath_for(self, cam_obj: bpy.types.Object, _global_idx: int) -> Path:
        frame = self._scene.frame_current
        frame_dir = self.root / f"frame_{frame}" / "train"
        frame_dir.mkdir(parents=True, exist_ok=True)

        seq = self._seq.setdefault(frame, 0)
        filename = f"{seq}.png"        # 0.png, 1.png, …
        self._seq[frame] += 1
        return frame_dir / filename

    def register_frame(self, cam_obj, rel_path):
        frame = self._scene.frame_current
        split = "test" if cam_obj.name.endswith("_test") else "train"

        w,h,fx,fy,cx,cy,dist = _intrinsics_from_cam(cam_obj.data, self._scene)
        prefix = f"frame_{frame}/"
        local_fp = rel_path[len(prefix):] if rel_path.startswith(prefix) else rel_path
        if not local_fp.startswith("./"): local_fp = "./"+local_fp

        item = {
            "camera_angle_x": cam_obj.data.angle_x,
            "camera_angle_y": cam_obj.data.angle_y,
            "fl_x": fx, "fl_y": fy,
            **dist,
            "cx": cx, "cy": cy,
            "w": w, "h": h,
            "aabb_scale": 2,
            "scale": self._scale,
            "transform_matrix": self._matrix(cam_obj.matrix_world),
            "file_path": local_fp,
        }
        (self._test if split=="test" else self._train)\
            .setdefault(frame, []).append(item)

    # 写文件：一个 frame 一对 transforms*.json ------------------------------- #
    def finish(self):
        for frame in set(self._train) | set(self._test):
            frame_dir = self.root / f"frame_{frame}"
            train_list = self._train.get(frame, [])
            test_list  = self._test.get(frame, [])

            if train_list:
                (frame_dir / "transforms.json").write_text(
                    json.dumps(train_list, indent=4)
                )
            if test_list:
                (frame_dir / "transforms_test.json").write_text(
                    json.dumps(test_list, indent=4)
                )

class ColmapPoseWriter(DatasetWriter):
    """
    Directory layout per frame (unchanged)::

        frame_1/
            images/0000.png 0001.png …
            sparse/0/
                cameras.txt / .bin   (one line per Blender camera)
                images.txt  / .bin   (one line per rendered image)
                points3D.txt         (empty placeholder)
    """

    def __init__(self, root_out: Path, cam0):
        super().__init__(root_out, cam0)
        self._seq_per_frame: dict[int, int] = {}            # frame → running idx
        self._frames: dict[int, list[tuple[bpy.types.Object, Path]]] = {}

        # 解析场景分辨率一次即可
        scene = bpy.context.scene
        scale = scene.render.resolution_percentage / 100.0
        self._width  = int(scene.render.resolution_x * scale)
        self._height = int(scene.render.resolution_y * scale)

    # ------------------------------------------------------------ helpers ----
    def _blender_to_colmap(self, cam_obj: bpy.types.Object):
        """
        Convert *Blender* camera-to-world (matrix_world) into COLMAP
        world-to-camera (OpenCV 左手系) 7-tuple:
            (qw, qx, qy, qz, tx, ty, tz)
        The math is the exact逆变换 of the importer’s
            M_bl  =  A · (M_w2cv)⁻¹ · T
        so that round-tripping poses is lossless.
        """
        import mathutils as mu

        # Blender matrix_world  (camera → Blender-world)
        M_bl = cam_obj.matrix_world

        # ── constant axis-conversion matrices (same as importer) ────────────
        A = mu.Matrix((
            (1, 0,  0, 0),   # OpenCV-world → Blender-world
            (0, 0,  1, 0),
            (0,-1,  0, 0),
            (0, 0,  0, 1),
        ))
        T = mu.Matrix((
            (1, 0,  0, 0),   # Blender-local → OpenCV-local
            (0,-1,  0, 0),
            (0, 0,-1, 0),
            (0, 0,  0, 1),
        ))

        # M_w2cv  =  T · M_bl⁻¹ · A
        M_w2cv = T @ M_bl.inverted() @ A

        R = M_w2cv.to_3x3()
        t = M_w2cv.to_translation()
        # Apply global scene scale so COLMAP poses match fused_points
        s = float(getattr(bpy.context.scene.rs_settings, "scale", 1.0) or 1.0)
        if s != 1.0:
            t *= s
        q = R.to_quaternion()                         # (w, x, y, z)

        return (q.w, q.x, q.y, q.z, t.x, t.y, t.z)

    # ------------------------------------------------ dataset-writer API -----
    def filepath_for(self, cam_obj, _global_idx):
        """
        Called by FrameDatasetRenderer for every image.
        Returns absolute path where the render should be saved.
        """
        frame = bpy.context.scene.frame_current
        seq   = self._seq_per_frame.setdefault(frame, 0)

        img_dir = self.root / f"frame_{frame}" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        path = img_dir / f"{seq:04d}.png"
        self._seq_per_frame[frame] += 1
        return path

    def register_frame(self, cam_obj, img_path: Path):
        """
        Record (camera object, absolute image path) for later writing txt/bin.
        """
        frame = bpy.context.scene.frame_current
        self._frames.setdefault(frame, []).append((cam_obj, img_path))

    # ------------------------------------------------ write txt & call colmap
    def finish(self):
        import shutil, subprocess, sys, os

        for frame, items in self._frames.items():
            sparse0 = self.root / f"frame_{frame}" / "sparse" / "0"
            sparse0.mkdir(parents=True, exist_ok=True)

            # -------- cameras.txt : one line per camera --------------------
            cam_lines: list[str] = []
            for cam_id, (cam_obj, _) in enumerate(items, start=1):
                camdat = cam_obj.data
                # keep intrinsics consistent with import/export path
                _w,_h,fx,fy,cx,cy,_ = _intrinsics_from_cam(camdat, bpy.context.scene)
                Wi, Hi = int(round(_w)), int(round(_h))
                cam_lines.append(
                    f"{cam_id} PINHOLE {Wi} {Hi} "
                    f"{fx} {fy} {cx} {cy}\n"
                )
            (sparse0 / "cameras.txt").write_text("".join(cam_lines))

            # -------- images.txt : one line per rendered PNG ---------------
            img_lines: list[str] = []
            for img_id, (cam_obj, img_abs) in enumerate(items, start=1):
                qw,qx,qy,qz, tx,ty,tz = self._blender_to_colmap(cam_obj)
                # COLMAP 期望相对 images/ 路径，且用正斜杠
                rel_name = Path(img_abs).name
                camera_id = img_id         # 1-to-1，对应上面的 cam_id
                img_lines.append(
                    f"{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} "
                    f"{camera_id} {rel_name}\n\n"
                )
            (sparse0 / "images.txt").write_text("".join(img_lines))

            # -------- 空 points3D.txt 占位，让 model_converter 不报错 ------
            (sparse0 / "points3D.txt").touch(exist_ok=True)
            # --- Sanity check: compare Blender vs exported K/R/t (1~2 cams) ---
            try:
                if items:
                    _reproj_sanity_check(bpy.context.scene, items[0][0], samples=400)
                    if len(items) > 3:
                        _reproj_sanity_check(bpy.context.scene, items[len(items)//2][0], samples=400)
            except Exception as _e:
                print("[RS-Studio] reprojection sanity-check skipped:", _e)
            # Refresh BIN from TXT (or drop stale BIN to force TXT read)
            _refresh_or_drop_colmap_bins(sparse0)
                
from bpy_extras.object_utils import world_to_camera_view as _w2c_view
import random as _rand
def _reproj_sanity_check(scene, cam_obj, *, samples=300):
    """
    Compare Blender's own projection vs our exported K,R,t.
    Return (rmse_px, avg_abs_u, avg_abs_v). Print a few worst cases.
    """
    w,h,fx,fy,cx,cy,_ = _intrinsics_from_cam(cam_obj.data, scene)
    R_wc, C_w = extrinsics_cv_from_matrix_world(cam_obj.matrix_world)  # OpenCV
    # gather some world points (mesh verts) – evaluated depsgraph
    try:
        dg = bpy.context.evaluated_depsgraph_get()
    except AttributeError:
        # Older API fallback
        dg = bpy.context.view_layer.depsgraph
    verts = []
    for inst in dg.object_instances:
        o = inst.object.evaluated_get(dg)
        if o.type != 'MESH' or o.hide_render: continue
        me = o.to_mesh(); 
        if not me: continue
        mw = inst.matrix_world
        try:
            for v in me.vertices:
                P = (mw @ v.co).to_3d()
                verts.append(P)
        finally:
            o.to_mesh_clear()
        if len(verts) > 5000:
            break
    if not verts:
        print("[RS-Studio] sanity-check: no mesh verts found")
        return 0.0,0.0,0.0
    _rand.shuffle(verts)
    verts = verts[:samples]

    import math
    se = 0.0; su = 0.0; sv = 0.0; worst = []
    for P_bl in verts:
        # Blender's projection
        uvz = _w2c_view(scene, cam_obj, P_bl)  # (x_norm in [0,1], y from bottom)
        uB = float(uvz.x * w)
        vB = float((1.0 - uvz.y) * h)

        # Our OpenCV projection
        # Blender-world -> OpenCV-world
        A_cv_from_bl = Matrix(((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1))).inverted()
        P_cv = (A_cv_from_bl.to_3x3() @ P_bl)
        Xc = R_wc.transposed() @ (P_cv - C_w)
        if Xc.z <= 1e-8:  # behind
            continue
        u = fx * (Xc.x / Xc.z) + cx
        v = fy * (Xc.y / Xc.z) + cy

        du = abs(u - uB); dv = abs(v - vB)
        su += du; sv += dv
        se += du*du + dv*dv
        worst.append((du+dv, uB, vB, u, v))

    n = max(1, len(worst))
    rmse = math.sqrt(se / n)
    worst.sort(reverse=True)
    print(f"[RS-Studio] reprojection check on '{cam_obj.name}': "
          f"RMSE={rmse:.3f}px  mean|Δu|={su/n:.3f}px  mean|Δv|={sv/n:.3f}px")
    for ssum,uB,vB,uC,vC in worst[:5]:
        print(f"   worst Δ≈{ssum:.2f}px  Blender=({uB:.1f},{vB:.1f})  Ours=({uC:.1f},{vC:.1f})")
    return rmse, su/n, sv/n


class Colmap3DGSDepthWriter(ColmapPoseWriter):
    """
    Export a COLMAP dataset ready for 3DGS by:
      1) rendering per-view color (PNG) and depth (EXR, Z pass);
      2) back-projecting pixels with (fx, fy, cx, cy) (top-left origin);
      3) fusing by voxel grid; writing fused_points.ply;
      4) writing sparse model (cameras.txt, images.txt, points3D.txt) and
         converting to BIN if COLMAP is available.
    All coordinates are in OpenCV world to match images.txt (and 3DGS).
    """
    def __init__(self, root_out: Path, cam0):
        super().__init__(root_out, cam0)
        self._depth_paths: dict[Path, Path] = {}  # color-abs-path -> depth-abs-path

        scn = bpy.context.scene
        # Ensure Z pass and compositor
        for vl in scn.view_layers:
            vl.use_pass_z = True
        scn.render.use_compositing = True
        scn.use_nodes = True
        self._fo_node, self._fo_slot = self._ensure_depth_output_node(scn)

        # Render color as PNG (3DGS expects standard images)
        imgset = scn.render.image_settings
        imgset.file_format = 'PNG'
        imgset.color_mode = 'RGB'
        imgset.color_depth = '8'

        # Tunables (read from scene.rs_settings if present)
        rs = getattr(scn, "rs_settings", None)
        self._stride     = int(getattr(rs, "depth_stride", 4))
        self._voxel      = float(getattr(rs, "pointcloud_voxel", 0.01))
        self._max_points = int(getattr(rs, "pointcloud_max_points", 2_000_000))

        # Force Cycles for reliable Z pass
        self._prev_engine = scn.render.engine
        try:
            engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        except Exception:
            engines = {"CYCLES", "BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"}
        if "CYCLES" in engines:
            scn.render.engine = "CYCLES"

    # ---- compositor setup ---------------------------------------------------
    @staticmethod
    def _ensure_depth_output_node(scene: bpy.types.Scene):
        """Ensure a compositor pipeline that writes depth as EXR (32-bit, BW)."""
        nt = scene.node_tree
        rl = next((n for n in nt.nodes if n.type == 'R_LAYERS'), None)
        if rl is None:
            rl = nt.nodes.new("CompositorNodeRLayers")
        rl.scene = scene
        try:
            rl.layer = scene.view_layers.active.name
        except Exception:
            pass
        comp = next((n for n in nt.nodes if n.type == 'COMPOSITE'), None)
        if comp is None:
            comp = nt.nodes.new("CompositorNodeComposite")
        try:
            nt.links.new(rl.outputs.get("Image"), comp.inputs.get("Image"))
        except Exception:
            pass
        fo = next((n for n in nt.nodes if n.name == "RS_Depth_Output" and n.type == 'OUTPUT_FILE'), None)
        if fo is None:
            fo = nt.nodes.new("CompositorNodeOutputFile")
            fo.name = "RS_Depth_Output"
        if not fo.file_slots:
            fo.file_slots.new("DepthZ")
        slot = fo.file_slots[0]
        slot.use_node_format = True
        fo.base_path = "//"
        fo.format.file_format = 'OPEN_EXR'
        fo.format.color_mode  = 'BW'
        fo.format.color_depth = '32'
        depth_out = rl.outputs.get("Depth") or rl.outputs.get("Z")
        if depth_out is not None:
            for l in list(nt.links):
                if l.to_node == fo and l.to_socket == fo.inputs[0]:
                    nt.links.remove(l)
            nt.links.new(depth_out, fo.inputs[0])
        return fo, slot
    def _write_colmap_points3D_bin(self, bin_path: Path,
                                   pts: list[tuple[float,float,float,float,float,float,int,int,int]]) -> None:
        """Minimal COLMAP-compatible points3D.bin with empty tracks."""
        import struct
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<Q", len(pts)))
            pid = 1
            for x, y, z, nx, ny, nz, r, g, b in pts:
                f.write(struct.pack("<QdddBBBdQ",
                                    pid, float(x), float(y), float(z),
                                    int(r), int(g), int(b),
                                    0.0, 0))  # ERROR=0.0, track_len=0
                pid += 1

    def _prepare_depth_path_for_seq(self, frame: int, seq: int, out_dir: Path) -> Path:
        depth_dir = out_dir / f"frame_{frame}" / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        self._fo_node.base_path = os.fspath(depth_dir)
        name = f"f{frame:04d}_c{seq:04d}"
        self._fo_slot.path = name
        return depth_dir / f"{name}.exr"

    def _resolve_depth_file(self, expected: Path) -> Path | None:
        if expected.exists():
            return expected
        cand = sorted(expected.parent.glob(expected.stem + "*.exr"))
        return cand[-1] if cand else None

    # ---- hook: return color path and schedule depth path --------------------
    def filepath_for(self, cam_obj, _global_idx):
        frame = bpy.context.scene.frame_current
        seq   = self._seq_per_frame.setdefault(frame, 0)
        img_dir = self.root / f"frame_{frame}" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        color_path = img_dir / f"{seq:04d}.png"
        depth_path = self._prepare_depth_path_for_seq(frame, seq, self.root)
        self._depth_paths[color_path.resolve()] = depth_path
        self._seq_per_frame[frame] = seq + 1
        return color_path

    def register_frame(self, cam_obj, img_path: Path):
        super().register_frame(cam_obj, img_path)

    # ---- main finish: run poses, then fuse depth into points ----------------
    def finish(self):
        # First, write cameras/images (poses-only)
        super().finish()
        # Then, for each frame: fuse depth → write PLY and points3D.txt, convert BIN
        for frame, items in self._frames.items():
            frame_dir  = self.root / f"frame_{frame}"
            sparse0    = frame_dir / "sparse" / "0"
            points_txt = sparse0 / "points3D.txt"
            # pts: list of (x, y, z, nx, ny, nz, r, g, b) in OpenCV world
            pts_cv = self._fuse_points_for_frame(frame, items)
            fused_ply = frame_dir / "fused_points.ply"
            self._write_ply(fused_ply, pts_cv)
            # convenience copy next to COLMAP sparse
            try:
                import shutil as _sh
                _sh.copyfile(fused_ply, sparse0 / "points3D.ply")
            except Exception:
                pass
            # Overwrite points3D.txt with actual points (no tracks in TXT)
            with points_txt.open("w", encoding="utf8") as f:
                pid = 1
                for (x, y, z, nx, ny, nz, r, g, b) in pts_cv:
                    f.write(f"{pid} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0\n")
                    pid += 1
            # Try convert to BIN (old COLMAP may not support --input_type)
            colmap_exe = shutil.which("colmap")
            bin_points = sparse0 / "points3D.bin"
            need_fallback = False
            if colmap_exe:
                try:
                    args = [
                        colmap_exe, "model_converter",
                        "--input_path",  os.fspath(sparse0),
                        "--output_path", os.fspath(sparse0),
                    ]
                    if _colmap_supports_input_type_option(colmap_exe):
                        args += ["--input_type", "TXT"]
                    args += ["--output_type", "BIN"]
                    res = subprocess.run(args, check=True, capture_output=True, text=True)
                    # verify output looks sane
                    if (not bin_points.exists()) or (bin_points.stat().st_size <= 64):
                        need_fallback = True
                except Exception as e:
                    print("[RS-Studio] ⚠ model_converter failed; falling back to internal writer:", e)
                    need_fallback = True
            else:
                need_fallback = True

            if need_fallback:
                self._write_colmap_points3D_bin(bin_points, pts_cv)
        # Restore engine
        try:
            bpy.context.scene.render.engine = self._prev_engine
        except Exception:
            pass

    # ---- depth fusion core --------------------------------------------------
    def _fuse_points_for_frame(self, frame: int, items: list[tuple[bpy.types.Object, Path]]):
        """
        Return a list of (x, y, z, r, g, b) in OpenCV world.
       Voxel grid merges duplicates and controls count.
        """
        scn = bpy.context.scene
        vox = max(self._voxel, 1e-6)
        stride = max(self._stride, 1)
        max_pts = max(self._max_points, 1)
        s_global = float(getattr(bpy.context.scene.rs_settings, "scale", 1.0) or 1.0)

        accum: dict[tuple[int, int, int], list[float]] = {}
        counts: dict[tuple[int, int, int], int] = {}
        for cam_obj, img_abs in items:
            color_abs = (self.root / img_abs).resolve()
            depth_expected = self._depth_paths.get(color_abs) or self._depth_paths.get((self.root / img_abs))
            if depth_expected is None:
                continue
            depth_abs = self._resolve_depth_file(depth_expected)
            if depth_abs is None:
                print(f"[RS-Studio] Depth EXR not found for {color_abs.name} – looked for {depth_expected}")
                continue

            col_img = bpy.data.images.load(os.fspath(color_abs), check_existing=True)
            dep_img = bpy.data.images.load(os.fspath(depth_abs), check_existing=True)
            try:
                dep_img.use_view_as_render = False
                dep_img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            w, h = int(col_img.size[0]), int(col_img.size[1])
            w0,h0,fx,fy,cx,cy,_ = _intrinsics_from_cam(cam_obj.data, scn)
            if int(w0) != int(w) or int(h0) != int(h):
                sx, sy = (w / w0), (h / h0)
                fx *= sx; fy *= sy; cx *= sx; cy *= sy

            z_near = float(cam_obj.data.clip_start)
            z_far  = float(cam_obj.data.clip_end)

            col_px = col_img.pixels[:]
            dep_px = dep_img.pixels[:]
            col_ch = int(getattr(col_img, "channels", 4) or 4)
            dep_ch = int(getattr(dep_img, "channels", 1) or 1)
            R_wc_mu, C_w_mu = extrinsics_cv_from_matrix_world(cam_obj.matrix_world)
            delta = max(1, min(stride, 4))
            for v in range(0, h, stride):
                for u in range(0, w, stride):
                    # Read the same pixel for color and depth (top-left origin)
                    v_img = (h - 1 - v)                      # Blender stores bottom→top
                    base  = v_img * w + u
                    off_c = base * col_ch
                    off_d = base * dep_ch
                    Z = float(dep_px[off_d])                 # first channel
                    if not (z_near < Z < z_far) or not math.isfinite(Z) or Z <= 0.0:
                        continue
                    u_c = u + 0.5
                    v_td = v + 0.5
                    sxn = (u_c - cx) / fx
                    syn = (v_td - cy) / fy
                    Xc, Yc, Zc = sxn * Z, syn * Z, Z

                    # --- estimate normal in camera space via finite differences ---
                    # neighbors at (u+delta, v) and (u, v+delta); fall back to view-normal if invalid
                    def _read_depth(_u, _v) -> float | None:
                        if 0 <= _u < w and 0 <= _v < h:
                            _v_img = (h - 1 - _v)
                            _base  = _v_img * w + _u
                            _off_d = _base * dep_ch
                            _Z = float(dep_px[_off_d])
                            if (z_near < _Z < z_far) and math.isfinite(_Z) and _Z > 0.0:
                                return _Z
                        return None

                    Zx = _read_depth(min(w-1, u + delta), v)
                    Zy = _read_depth(u, min(h-1, v + delta))
                    # current point in camera
                    P_c = mu.Vector((Xc, Yc, Zc))
                    # neighbor points in camera if valid
                    nx_valid = Zx is not None
                    ny_valid = Zy is not None
                    if nx_valid:
                        sxn_x = ((u + delta) + 0.5 - cx) / fx
                        Pcx = mu.Vector((sxn_x * Zx, syn * Zx, Zx))
                    if ny_valid:
                        syn_y = ((v + delta) + 0.5 - cy) / fy
                        Pcy = mu.Vector((sxn * Zy, syn_y * Zy, Zy))
                    if nx_valid and ny_valid:
                        t_u = (Pcx - P_c)
                        t_v = (Pcy - P_c)
                        n_c = t_u.cross(t_v)
                        if n_c.length > 1e-12:
                            n_c.normalize()
                        else:
                            n_c = -P_c
                            if n_c.length > 1e-12: n_c.normalize()
                    else:
                        # fallback: view-direction normal (points toward camera)
                        n_c = -P_c
                        if n_c.length > 1e-12: n_c.normalize()

                    # camera->world (OpenCV world)
                    P_w = (R_wc_mu @ P_c) + C_w_mu
                    N_w = (R_wc_mu @ n_c).normalized()
                    
                    if s_global != 1.0:
                        P_w = P_w * s_global
                    xw, yw, zw = float(P_w.x), float(P_w.y), float(P_w.z)
                    nx, ny, nz = float(N_w.x), float(N_w.y), float(N_w.z)
                    r = int(max(0, min(255, round(col_px[off_c + 0] * 255.0))))
                    g = int(max(0, min(255, round(col_px[off_c + 1] * 255.0))))
                    b = int(max(0, min(255, round(col_px[off_c + 2] * 255.0))))

                    key = (int(round(xw / vox)), int(round(yw / vox)), int(round(zw / vox)))
                    if key not in accum:
                        accum[key]  = [xw, yw, zw, nx, ny, nz, float(r), float(g), float(b)]
                        counts[key] = 1
                    else:
                        acc = accum[key]
                        acc[0] += xw; acc[1] += yw; acc[2] += zw
                        acc[3] += nx; acc[4] += ny; acc[5] += nz
                        acc[6] += r;  acc[7] += g;  acc[8] += b
                        counts[key] += 1
                    if len(accum) >= max_pts:
                        break
                if len(accum) >= max_pts:
                    break

            bpy.data.images.remove(col_img)
            bpy.data.images.remove(dep_img)

        out: list[tuple[float, float, float, float, float, float, int, int, int]] = []
        for k, acc in accum.items():
            c = counts[k]
            x = acc[0] / c; y = acc[1] / c; z = acc[2] / c
            nx = acc[3] / c; ny = acc[4] / c; nz = acc[5] / c
            # re-normalize averaged normals for PLY
            nlen = math.sqrt(nx*nx + ny*ny + nz*nz) if 'math' in globals() else 0.0
            if nlen > 1e-12:
                nx /= nlen; ny /= nlen; nz /= nlen
            r = int(round(acc[6] / c)); g = int(round(acc[7] / c)); b = int(round(acc[8] / c))
            out.append((x, y, z, nx, ny, nz, r, g, b))
        return out

    @staticmethod
    def _write_ply(path: Path, pts: list[tuple[float, float, float, float, float, float, int, int, int]]):
        """Write binary little-endian PLY: XYZ + Normal + RGB."""
        import struct
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        with open(path, "wb") as f:
            f.write(header)
            pack = struct.Struct("<ffffffBBB").pack
            for x, y, z, nx, ny, nz, r, g, b in pts:
                f.write(pack(float(x), float(y), float(z),
                            float(nx), float(ny), float(nz),
                            int(r), int(g), int(b)))
# ------------------------------------------------------------------ 3DGS Writer (Surface-sampling route C)
class Colmap3DGSSurfaceWriter(ColmapPoseWriter):
    """
    Export a COLMAP dataset ready for 3DGS by:
    1) rendering per-view color PNGs (no depth rendering);
    2) sampling points directly on mesh surfaces (uniform-by-area);
    3) NO camera-visibility filtering (by design for Surface route):
       we do not test frustum/image-bounds/backface/occlusion. Points are
       sampled uniformly-by-area on the mesh and fused by voxel grid.
    4) optional voxel-grid merge (averaging color) to control count;
    5) write fused_points.ply and sparse points3D.txt (OpenCV world).
    Notes:
    - This avoids per-view depth quirks and removes inside/occluded surfaces.
    - For albedo: attempts UV → ImageTexture sampling; falls back to vertex
      color (if present) or material base-color (constant) -> white.
    - UDIM is treated tile-aware when Blender Image is TILED (best-effort).
    """

    def __init__(self, root_out: Path, cam0):
        super().__init__(root_out, cam0)
        scn = bpy.context.scene
        # we only need color PNGs for training
        imgset = scn.render.image_settings
        imgset.file_format = 'PNG'
        imgset.color_mode = 'RGB'
        imgset.color_depth = '8'

    def _write_colmap_points3D_bin(self, bin_path: Path,
                                   pts: list[tuple[float,float,float,float,float,float,int,int,int]]) -> None:
        """
        Write a minimal COLMAP-compatible points3D.bin:
        [num_points: uint64]
        For each point:
          point3D_id: uint64
          X,Y,Z:      float64 x3
          R,G,B:      uint8   x3
          ERROR:      float64
          track_len:  uint64  (0 here)
          (no (image_id:int32, point2D_idx:int32) pairs since empty track)
        Normals are not stored in COLMAP's points3D.bin, so they are ignored here.
        """
        import struct
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<Q", len(pts)))
            pid = 1
            for x, y, z, nx, ny, nz, r, g, b in pts:
                f.write(struct.pack("<QdddBBBdQ",
                                    pid, float(x), float(y), float(z),
                                    int(r), int(g), int(b),
                                    0.0, 0))  # ERROR=0.0, track_len=0
                pid += 1

    def finish(self):
        # 1) Write cameras.txt / images.txt / empty points3D.txt via parent
        super().finish()

        for frame, items in self._frames.items():
            frame_dir  = self.root / f"frame_{frame}"

            # (Surface route) We do not need depth at all. If a lingering "depth" folder
            # exists (e.g., from a previous Depth run), remove it to keep the layout clean.
            # Policy: delete *.exr files under depth/, then remove the directory if empty.
            try:
                depth_dir = frame_dir / "depth"
                if depth_dir.exists() and depth_dir.is_dir():
                    for exr in list(depth_dir.glob("*.exr")):
                        try:
                            exr.unlink()
                        except Exception:
                            pass
                    # remove the folder if it is now empty
                    try:
                        next(depth_dir.iterdir())
                    except StopIteration:
                        depth_dir.rmdir()
            except Exception:
                # Never fail the export just because cleanup did not succeed.
                pass
            sparse0    = frame_dir / "sparse" / "0"
            txt_points = sparse0 / "points3D.txt"
            bin_points = sparse0 / "points3D.bin"

            # 2) Build surface-sampled points (OpenCV world, with normals)
            pts = self._sample_surface_points_for_frame(frame, items)
            # pts: [(x,y,z, nx,ny,nz, r,g,b), ...]

            # 3) Write fused_points.ply (with normals), then sync to COLMAP path
            fused_ply = frame_dir / "fused_points.ply"
            self._write_ply(fused_ply, pts)
            # sync for 3DGS convenience
            try:
                import shutil
                shutil.copyfile(fused_ply, sparse0 / "points3D.ply")
            except Exception as _:
                pass  # never fail the pipeline on a convenience copy

            # 4) Strict points3D.txt (no header/comment, no track in TXT)
            with txt_points.open("w", encoding="utf8") as f:
                pid = 1
                for x, y, z, nx, ny, nz, r, g, b in pts:
                    # POINT3D_ID X Y Z R G B ERROR
                    f.write(f"{pid} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0\n")
                    pid += 1

            # 5) Try TXT->BIN via COLMAP; explicit input/output types; then validate size
            colmap_exe = shutil.which("colmap")
            need_fallback = False
            if colmap_exe:
                try:
                    import subprocess, os
                    cmd = [
                        colmap_exe, "model_converter",
                        "--input_path",  os.fspath(sparse0),
                        "--output_path", os.fspath(sparse0),
                    ]
                    if _colmap_supports_input_type_option(colmap_exe):
                        cmd += ["--input_type", "TXT"]
                    cmd += ["--output_type", "BIN"]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    if (not bin_points.exists()) or (bin_points.stat().st_size <= 64):
                        need_fallback = True
                except Exception:
                    need_fallback = True
            else:
                need_fallback = True

            # 6) Fallback: write our own minimal points3D.bin (empty tracks)
            if need_fallback:
                self._write_colmap_points3D_bin(bin_points, pts)
                
    def _write_ply(self, path: Path, pts: list[tuple[float,float,float,float,float,float,int,int,int]]) -> None:
            """Write binary little-endian PLY with XYZ + Normal + RGB in that exact order."""
            import struct
            path.parent.mkdir(parents=True, exist_ok=True)
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(pts)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
            with open(path, "wb") as f:
                f.write(header)
                pack = struct.Struct("<ffffffBBB").pack
                for x, y, z, nx, ny, nz, r, g, b in pts:
                    f.write(pack(float(x), float(y), float(z),
                                float(nx), float(ny), float(nz),
                                int(r), int(g), int(b)))
    # ---- core: surface sampling with visibility filtering ------------------
    def _sample_surface_points_for_frame(self, frame: int, items: list[tuple[bpy.types.Object, Path]]):
        """
        Return list of (x, y, z, r, g, b) in OpenCV world.
        Steps:
          • gather triangles from visible MESH objects (evaluated depsgraph)
          • sample ~target_count points (area-weighted)
          • for each point, test visibility against multiple cameras
          • optional voxel merge (color-averaging in world space)
        """
        # OpenCV-world -> Blender-world (A_bl_from_cv)
        A_bl_from_cv = mu.Matrix((
            (1, 0,  0, 0),
            (0, 0,  1, 0),
            (0,-1,  0, 0),
            (0, 0,  0, 1),
        ))
        # Blender-world -> OpenCV-world
        A_cv_from_bl = A_bl_from_cv.inverted()
        scn = bpy.context.scene
        rs  = getattr(scn, "rs_settings", None)

        # parameters (read from UI; provide robust defaults)
        target_count     = int(getattr(rs, "mesh_points_target", 800_000))
        voxel            = float(getattr(rs, "mesh_voxel", 0.01))
        color_mode       = str(getattr(rs, "mesh_color_mode", "TEXTURE"))  # TEXTURE / VCOL / MATERIAL / WHITE
        # visibility filtering is disabled for Surface route
        vis_filter   = False
        min_visible  = 1
        backface     = False
        max_check    = 0
        # Axis-aligned bounds in OpenCV world (optional)
        bounds_enable = bool(getattr(rs, "mesh_bounds_enable", False))
        bmin = getattr(rs, "mesh_bounds_min", (-1e9, -1e9, -1e9))
        bmax = getattr(rs, "mesh_bounds_max", ( 1e9,  1e9,  1e9))
        try:
            bmin_cv = mu.Vector((float(bmin[0]), float(bmin[1]), float(bmin[2])))
            bmax_cv = mu.Vector((float(bmax[0]), float(bmax[1]), float(bmax[2])))
        except Exception:
            bmin_cv = mu.Vector((-1e9, -1e9, -1e9))
            bmax_cv = mu.Vector(( 1e9,  1e9,  1e9))
        # depsgraph & evaluated scene for accurate geometry + ray cast
        dg = bpy.context.evaluated_depsgraph_get()
        # 1) collect triangles (world space) grouped by *object* (layered sampling)
        # bucket schema: { "name": str, "tris": list[dict], "area": float, "cdf": np.ndarray }
        buckets: list[dict] = []
        bucket_by_name: dict[str, int] = {}
        tot_area = 0.0
        _img_cache: dict = {}    # Image -> (np_pixels, w, h, ch, is_udim, tiles_dict)
        _mat_img_cache: dict = {}  # Material -> preferred Image or None

        def _find_image_from_material(mat: bpy.types.Material):
            """Return an Image used as albedo source if possible (first try BaseColor)."""
            if mat is None or not mat.use_nodes or mat.node_tree is None:
                return None
            nt = mat.node_tree
            # Prefer the ImageTexture feeding Principled BSDF Base Color
            bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if bsdf and "Base Color" in bsdf.inputs:
                for lk in bsdf.inputs["Base Color"].links:
                    if lk.from_node.type == 'TEX_IMAGE' and lk.from_node.image is not None:
                        return lk.from_node.image
            # Fallback: first TEX_IMAGE
            for n in nt.nodes:
                if n.type == 'TEX_IMAGE' and n.image is not None:
                    return n.image
            return None

        def _ensure_image_cached(img: bpy.types.Image):
            """Return (pix_np, w, h, ch, is_tiled_udim, tiles_dict). Robust to empty images."""
            if img in _img_cache:
                return _img_cache[img]
            ch = int(getattr(img, "channels", 4) or 4)
            is_tiled = (getattr(img, "source", "") == 'TILED')
            if is_tiled:
                # Build per-tile cache: {tile_number -> (np_pixels, w, h, ch)}
                tiles = {}
                for t in getattr(img, "tiles", []):
                    try:
                        img.tile_number = t.number
                        w = int(img.size[0]); h = int(img.size[1])
                        if w < 1 or h < 1:
                            continue
                        arr = np.array(img.pixels[:], dtype=np.float32)
                        if arr.size < w * h * ch:
                            continue
                        tiles[t.number] = (arr, w, h, ch)
                    except Exception:
                        continue
                _img_cache[img] = (None, 0, 0, ch, True, tiles)
            else:
                w = int(img.size[0]); h = int(img.size[1])
                if w < 1 or h < 1:
                    _img_cache[img] = (np.empty(0, dtype=np.float32), 0, 0, ch, False, None)
                else:
                    arr = np.array(img.pixels[:], dtype=np.float32)
                    _img_cache[img] = (arr, w, h, ch, False, None)
            return _img_cache[img]

        def _sample_image_rgba(img: bpy.types.Image, uv: Vector):
            """Return (r,g,b,a) in 0..255 from Image/UDIM; fall back to white on any issue."""
            if img is None:
                return (255, 255, 255, 255)
            pix, w, h, ch, is_udim, tiles = _ensure_image_cached(img)
            u, v = float(uv.x), float(uv.y)
            if is_udim:
                ut = int(math.floor(u)); vt = int(math.floor(v))
                lu = u - ut; lv = v - vt
                tile_no = 1001 + ut + 10 * vt
                if not tiles or tile_no not in tiles:
                    return (255, 255, 255, 255)
                arr, tw, th, tch = tiles[tile_no]
                if tw < 1 or th < 1 or arr.size < tw * th * tch:
                    return (255, 255, 255, 255)
                x = min(tw - 1, max(0, int(lu * (tw - 1))))
                y = min(th - 1, max(0, int(lv * (th - 1))))
                idx = (y * tw + x) * tch
                r = int(round(255.0 * arr[idx + 0])) if tch >= 1 else 255
                g = int(round(255.0 * arr[idx + 1])) if tch >= 2 else 255
                b = int(round(255.0 * arr[idx + 2])) if tch >= 3 else 255
                a = int(round(255.0 * arr[idx + 3])) if tch >= 4 else 255
                return (r, g, b, a)
            else:
                if w < 1 or h < 1 or pix.size < w * h * ch:
                    return (255, 255, 255, 255)
                x = min(w - 1, max(0, int(u * (w - 1))))
                y = min(h - 1, max(0, int(v * (h - 1))))
                idx = (y * w + x) * ch
                r = int(round(255.0 * pix[idx + 0])) if ch >= 1 else 255
                g = int(round(255.0 * pix[idx + 1])) if ch >= 2 else 255
                b = int(round(255.0 * pix[idx + 2])) if ch >= 3 else 255
                a = int(round(255.0 * pix[idx + 3])) if ch >= 4 else 255
                return (r, g, b, a)

        # Traverse *evaluated instances* so collection/geometry instances are included.
        for inst in dg.object_instances:
            obj_eval = inst.object.evaluated_get(dg)
            if obj_eval.type != 'MESH' or obj_eval.hide_render:
                continue

            me = obj_eval.to_mesh()
            if me is None:
                continue
            mw = inst.matrix_world

            try:
                # We only need loop triangles in 4.4; do NOT call calc_normals* (removed).
                if hasattr(me, "calc_loop_triangles"):
                    me.calc_loop_triangles()

                uv_layer = me.uv_layers.active
                has_uv = uv_layer is not None
                mats = obj_eval.material_slots

                # ensure we have a bucket for this object
                bname = obj_eval.name_full if hasattr(obj_eval, "name_full") else obj_eval.name
                if bname not in bucket_by_name:
                    bucket_by_name[bname] = len(buckets)
                    buckets.append({"name": bname, "tris": [], "area": 0.0, "cdf": None})

                b_idx = bucket_by_name[bname]
                BU = buckets[b_idx]

                for lt in me.loop_triangles:
                    vid = lt.vertices
                    li  = lt.loops

                    # World-space triangle vertices
                    p0 = (mw @ me.vertices[vid[0]].co).to_3d()
                    p1 = (mw @ me.vertices[vid[1]].co).to_3d()
                    p2 = (mw @ me.vertices[vid[2]].co).to_3d()
                    if bounds_enable:
                        p0_cv = A_cv_from_bl.to_3x3() @ p0
                        p1_cv = A_cv_from_bl.to_3x3() @ p1
                        p2_cv = A_cv_from_bl.to_3x3() @ p2
                        tri_min = mu.Vector((
                            min(p0_cv.x, p1_cv.x, p2_cv.x),
                            min(p0_cv.y, p1_cv.y, p2_cv.y),
                            min(p0_cv.z, p1_cv.z, p2_cv.z),
                        ))
                        tri_max = mu.Vector((
                            max(p0_cv.x, p1_cv.x, p2_cv.x),
                            max(p0_cv.y, p1_cv.y, p2_cv.y),
                            max(p0_cv.z, p1_cv.z, p2_cv.z),
                        ))
                        # Grow a little to catch border triangles (use voxel as tolerance)
                        eps = max(1e-6, voxel)
                        tri_min -= mu.Vector((eps, eps, eps))
                        tri_max += mu.Vector((eps, eps, eps))
                        # No intersection -> skip
                        if (tri_max.x < bmin_cv.x or tri_min.x > bmax_cv.x or
                            tri_max.y < bmin_cv.y or tri_min.y > bmax_cv.y or
                            tri_max.z < bmin_cv.z or tri_min.z > bmax_cv.z):
                            continue
                    # Flat normal and area
                    e1 = p1 - p0
                    e2 = p2 - p0
                    n_flat = e1.cross(e2)
                    area = float(n_flat.length * 0.5)
                    if area <= 0.0:
                        continue
                    n_flat.normalize()

                    # Per-corner normals = flat normal (sufficient for culling/shading)
                    n0 = n1 = n2 = n_flat

                    # UVs
                    if has_uv:
                        uv0 = uv_layer.data[li[0]].uv.copy()
                        uv1 = uv_layer.data[li[1]].uv.copy()
                        uv2 = uv_layer.data[li[2]].uv.copy()
                    else:
                        uv0 = uv1 = uv2 = Vector((0.5, 0.5))

                    # Optional texture image per material (cached)
                    img = None
                    if color_mode == "TEXTURE" and len(mats) and lt.polygon_index < len(me.polygons):
                        midx = me.polygons[lt.polygon_index].material_index
                        mat = mats[midx].material if midx < len(mats) else None
                        if mat not in _mat_img_cache:
                            _mat_img_cache[mat] = _find_image_from_material(mat)
                        img = _mat_img_cache[mat]

                    tri_rec = dict(
                        p0=p0, p1=p1, p2=p2,
                        n0=n0, n1=n1, n2=n2,
                        uv0=uv0, uv1=uv1, uv2=uv2,
                        img=img, base_rgba=(255, 255, 255, 255),
                        area=area
                    )
                    BU["tris"].append(tri_rec)
                    BU["area"] += area
                    tot_area   += area
            finally:
                # Always free the evaluated mesh to avoid leaks
                obj_eval.to_mesh_clear()

        if tot_area <= 0.0 or not buckets:
            print("[RS-Studio] No triangles found (hidden-from-render? or all instances culled?).")
            return []
        # Drop buckets that ended up with no triangles (after AABB/filters)
        buckets = [b for b in buckets if len(b["tris"]) > 0 and b["area"] > 0.0]
        if not buckets:
            print("[RS-Studio] No triangles survived AABB/filters; skip point cloud generation.")
            return []

        # --- per-object layered weights to mitigate huge-ground dominance ---
        # object weight = (area) ** gamma ; gamma in (0,1] flattens dominance (default 0.5)
        obj_gamma = float(getattr(rs, "mesh_object_gamma", 0.5) or 0.5)
        obj_cap   = float(getattr(rs, "mesh_object_cap_ratio", 0.33) or 0.33)  # max share per object
        # Build per-object CDF and per-triangle CDFs
        obj_weights = np.array([b["area"] ** obj_gamma for b in buckets], dtype=np.float64)
        obj_cdf = np.cumsum(obj_weights); obj_cdf /= obj_cdf[-1]
        for BU in buckets:
            areas_local = np.fromiter(
                (t["area"] for t in BU["tris"]),
                dtype=np.float64,
                count=len(BU["tris"])
            )
            # areas_local is guaranteed non-empty after the bucket filter above
            cdf_local = np.cumsum(areas_local)
            cdf_local /= cdf_local[-1]
            BU["cdf"] = cdf_local
        obj_accept_cnt: dict[int, int] = {}

        # 2) stochastic sampling with layered object pick + per-object area pick
        rng = np.random.default_rng(12345)
        # voxel merge accumulators
        vox = max(voxel, 1e-6)
        accum: dict[tuple[int,int,int], list[float]] = {}
        counts: dict[tuple[int,int,int], int] = {}

        # Oversampling budget:
        # - without AABB: modest trials;
        # - with AABB: much larger budget since many samples may fall outside.
        rng = np.random.default_rng(12345)
        oversample = 10 if not bounds_enable else 200
        max_trials = max(target_count * oversample, target_count * 3)

        accepted_voxels = 0
        trials = 0

        while accepted_voxels < target_count and trials < max_trials:
            trials += 1
            # (A) pick an object bucket by weighted CDF; respect per-object cap
            tries_obj = 0
            while True:
                b_idx = int(np.searchsorted(obj_cdf, rng.random(), side="right"))
                if b_idx >= len(buckets): b_idx = len(buckets) - 1
                cap_ok = (obj_accept_cnt.get(b_idx, 0) < int(obj_cap * target_count))
                if cap_ok or tries_obj > 8:    # avoid infinite loop if all capped
                    break
                tries_obj += 1
            BU = buckets[b_idx]

            # (B) pick a triangle inside the chosen object by its local CDF
            lcdf = BU["cdf"]
            t_idx = int(np.searchsorted(lcdf, rng.random(), side="right"))
            if t_idx >= len(BU["tris"]): t_idx = len(BU["tris"]) - 1
            t = BU["tris"][t_idx]

            # barycentric sampling (uniform by area)
            r1 = rng.random(); r2 = rng.random()
            su = math.sqrt(r1)
            b0 = 1.0 - su
            b1 = su * (1.0 - r2)
            b2 = su * r2

            Pw = (t["p0"] * b0 + t["p1"] * b1 + t["p2"] * b2)
            Nw = (t["n0"] * b0 + t["n1"] * b1 + t["n2"] * b2).normalized()
            uv = (t["uv0"] * b0 + t["uv1"] * b1 + t["uv2"] * b2)

            # Keep only points inside the user AABB (OpenCV world) if enabled
            if bounds_enable:
                P_cv_for_aabb = A_cv_from_bl.to_3x3() @ Pw
                if not (bmin_cv.x <= P_cv_for_aabb.x <= bmax_cv.x and
                        bmin_cv.y <= P_cv_for_aabb.y <= bmax_cv.y and
                        bmin_cv.z <= P_cv_for_aabb.z <= bmax_cv.z):
                    continue
            else:
                P_cv_for_aabb = A_cv_from_bl.to_3x3() @ Pw          


            # color
            if color_mode == "TEXTURE" and (t["img"] is not None):
                r,g,b,a = _sample_image_rgba(t["img"], uv)
            elif color_mode == "VCOL":
                r,g,b,a = (220,220,220,255)
            elif color_mode == "MATERIAL":
                r,g,b,a = (200,200,200,255)
            else:
                r,g,b,a = (255,255,255,255)

            # voxel key (Blender world)
            k = (int(round(Pw.x/vox)), int(round(Pw.y/vox)), int(round(Pw.z/vox)))
            if k not in accum:
                accum[k]  = [Pw.x, Pw.y, Pw.z, float(Nw.x), float(Nw.y), float(Nw.z), float(r), float(g), float(b)]
                counts[k] = 1
                accepted_voxels += 1
                # count toward this object's cap only when a *new* voxel is created
                obj_accept_cnt[b_idx] = obj_accept_cnt.get(b_idx, 0) + 1                
            else:
                acc = accum[k]; c = counts[k]
                acc[0] += Pw.x; acc[1] += Pw.y; acc[2] += Pw.z
                acc[3] += Nw.x; acc[4] += Nw.y; acc[5] += Nw.z
                acc[6] += r;    acc[7] += g;    acc[8] += b
                counts[k] = c + 1

        # If AABB is enabled and we did not reach the target, tell the user explicitly.
        if bounds_enable and accepted_voxels < target_count:
            print(f"[RS-Studio] AABB target {target_count} not reached: got {accepted_voxels} "
                f"after {trials} trials. Consider expanding bounds or increasing budget.")

        # average within voxels; convert to OpenCV world
        out: list[tuple[float,float,float,float,float,float,int,int,int]] = []
        R_cv_from_bl = A_cv_from_bl.to_3x3()
        s = float(getattr(bpy.context.scene.rs_settings, "scale", 1.0) or 1.0)
        for k, acc in accum.items():
            c = counts[k]
            # 平均位置（Blender 世界）
            P_bl = mu.Vector((acc[0]/c, acc[1]/c, acc[2]/c))
            # 平均法线（Blender 世界 → 归一化）
            N_bl = mu.Vector((acc[3]/c, acc[4]/c, acc[5]/c)).normalized()
            # 坐标与法线都转到 OpenCV 世界
            P_cv = R_cv_from_bl @ P_bl
            N_cv = (R_cv_from_bl @ N_bl).normalized()
            if s != 1.0:
                P_cv *= s
            # 颜色
            rr = int(round(acc[6]/c)); gg = int(round(acc[7]/c)); bb = int(round(acc[8]/c))
            out.append((float(P_cv.x), float(P_cv.y), float(P_cv.z),
                        float(N_cv.x), float(N_cv.y), float(N_cv.z),
                        rr, gg, bb))
        # --- micro thickness (optional): duplicate points along normals ±ε ---
        thickness_eps = float(getattr(rs, "mesh_thickness_eps", 0.0) or 0.0)
        thickness_two_sided = bool(getattr(rs, "mesh_thickness_two_sided", True))
        if thickness_eps > 0.0 and len(out) > 0:
            extra = []
            for x,y,z, nx,ny,nz, r,g,b in out:
                # +epsilon along normal
                extra.append((x + thickness_eps*nx,
                              y + thickness_eps*ny,
                              z + thickness_eps*nz,
                              nx, ny, nz, r, g, b))
                if thickness_two_sided:
                    # -epsilon along normal
                    extra.append((x - thickness_eps*nx,
                                  y - thickness_eps*ny,
                                  z - thickness_eps*nz,
                                  nx, ny, nz, r, g, b))
            out.extend(extra)
        print(
            f"[RS-Studio] frame {frame}: tot_area={tot_area:.3f} "
            f"target(in-AABB)={target_count} vox={voxel} trials={trials} "
            f"fused_voxels={len(accum)} out_points={len(out)}"
            f"[layered: gamma={obj_gamma} cap={obj_cap} thickness={thickness_eps}]"
        )
        return out


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
class SamplingStrategy(ABC):
    """Interface for producing *n* camera positions on a sphere of *radius*."""

    @abstractmethod
    def sample(self, n: int, radius: float) -> Sequence[Vector]:
        raise NotImplementedError


class FibonacciSphereSampling(SamplingStrategy):
    """Evenly distribute points on a sphere using the Fibonacci spiral."""

    _golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    def sample(self, n: int, radius: float) -> List[Vector]:
        pts: List[Vector] = []
        for i in range(n):
            z = 1.0 - (2 * i + 1) / n                  # in [-1, 1]
            theta = self._golden_angle * i
            r_xy = math.sqrt(max(0.0, 1.0 - z * z))
            x, y = r_xy * math.cos(theta), r_xy * math.sin(theta)
            pts.append(Vector((x, y, z)) * radius)
        return pts

class FibonacciHemisphereSampling(FibonacciSphereSampling):
    """Fibonacci spiral constrained to the *upper* hemisphere (z ≥ 0)."""

    def sample(self, n: int, radius: float) -> List[Vector]:          # type: ignore[override]
        pts: List[Vector] = []
        i = 0
        while len(pts) < n:
            z = 1.0 - (2 * i + 1) / (2 * n)          # step twice as fine
            if z >= 0:
                theta = self._golden_angle * i
                r_xy = math.sqrt(max(0.0, 1.0 - z * z))
                x, y = r_xy * math.cos(theta), r_xy * math.sin(theta)
                pts.append(Vector((x, y, z)) * radius)
            i += 1
        return pts

# --------------------------------------------------------------------------- #
# Camera rig helper
# --------------------------------------------------------------------------- #
class CameraRig:
    """Maintain one Blender camera per sample position.

    The first camera re‑uses the *template* camera supplied by the user.  All
    additional cameras are *duplicates* of that template so optical parameters
    (lens, sensor size, etc.) remain identical.
    """

    def __init__(
        self,
        template_camera: bpy.types.Object,
        positions: Sequence[Vector],
        target: Vector,
        base_name: str = "camera",
    ) -> None:
        if template_camera.type != "CAMERA":
            raise TypeError("template_camera must be a CAMERA object")

        self.cameras: List[bpy.types.Object] = []

        # Ensure we are operating in object mode for safe ops
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        # Use the template itself as camera 0 --------------------------------
        cam0 = template_camera
        cam0.name = f"{base_name} 0"
        self._place_camera(cam0, positions[0], target)
        self.cameras.append(cam0)

        # Duplicate for remaining positions ---------------------------------
        for idx, pos in enumerate(positions[1:], start=1):
            name = f"{base_name} {idx}"
            cam_obj = bpy.data.objects.get(name)
            if cam_obj is None:  # create new duplicate only if not present
                cam_obj = self._duplicate_camera(cam0, name)
            self._place_camera(cam_obj, pos, target)
            self.cameras.append(cam_obj)

    # ---------------------------------- helpers --------------------------- #
    @staticmethod
    def _duplicate_camera(source: bpy.types.Object, name: str) -> bpy.types.Object:
        """Deep‑copy *source* camera (including its data block) and link it."""
        dup_obj = source.copy()
        dup_obj.data = source.data.copy()
        dup_obj.name = name
        bpy.context.collection.objects.link(dup_obj)
        return dup_obj

    @staticmethod
    def _place_camera(cam: bpy.types.Object, position: Vector, target: Vector) -> None:
        """Set *cam* to *position* and orient it to look at *target*."""
        cam.location = position
        direction = target - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    # Allow ``for cam in rig: ...``
    def __iter__(self):
        return iter(self.cameras)



class DatasetGenerator:
    # 新增 export_format 参数
    def __init__(self, *, cameras, start_frame, end_frame,
                 export_fmt: ExportFormat = ExportFormat.NGP):
        if end_frame < start_frame:
            raise ValueError("end_frame must be ≥ start_frame")
        self._colmap_ok = bool(shutil.which("colmap"))
        self.cameras = list(cameras)
        self.start   = int(start_frame)
        self.end     = int(end_frame)
        self.export_fmt = export_fmt
        frame_cnt = self.end - self.start + 1
        if export_fmt is ExportFormat.COLMAP_3DGS_MESH and self._colmap_ok:
            per_frame_extra = 3
        else:
            per_frame_extra = 0
        self.total_images = len(self.cameras) * frame_cnt + per_frame_extra
                
    # core.py  — inside class DatasetGenerator
    def iter_generate(self, output_dir: str | Path):
        from bpy.types import RenderSettings
        scene = bpy.context.scene
        prev_engine = scene.render.engine
        engines = {e.identifier for e in RenderSettings.bl_rna.properties["engine"].enum_items}
        preferred = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "CYCLES"
        if scene.render.engine == "BLENDER_WORKBENCH":
            scene.render.engine = preferred

        writer = None  # ← ensure we can call finish() even if something fails early
        try:
            root_out = Path(bpath.abspath(str(output_dir))).resolve()
            root_out.mkdir(parents=True, exist_ok=True)

            cam0 = self.cameras[0].data
            # --- choose writer ---
            if self.export_fmt is ExportFormat.COLMAP_POSES:
                writer = ColmapPoseWriter(root_out, cam0)
            elif self.export_fmt is ExportFormat.NERF_SYNTH:
                writer = NeRFSyntheticWriter(root_out, cam0)
            elif self.export_fmt is ExportFormat.TACV:
                writer = TACVDatasetWriter(root_out, cam0)
            elif self.export_fmt is ExportFormat.COLMAP_3DGS_MESH:
                writer = Colmap3DGSSurfaceWriter(root_out, cam0)
            elif self.export_fmt is ExportFormat.COLMAP_3DGS_DEPTH:
                writer = Colmap3DGSDepthWriter(root_out, cam0)            
            else:
                writer = NGPDatasetWriter(root_out, cam0)

            self._writer_inst = writer

            # --- render loop ---
            done = 0
            for frame_idx in range(self.start, self.end + 1):
                scene.frame_set(frame_idx)
                for cam in self.cameras:
                    path = writer.filepath_for(cam, done)
                    scene.camera = cam
                    scene.render.filepath = str(path)
                    bpy.ops.render.render(write_still=True, use_viewport=False)

                    # register relative path
                    rel = path.relative_to(root_out).as_posix()
                    writer.register_frame(cam, rel)

                    done += 1
                    yield done, self.total_images

            # --- optional postprocess ---
            for _ in writer.postprocess_iter():
                done += 1
                yield done, self.total_images

            # normal finish
            writer.finish()

        except Exception as e:
            print("[RS-Studio] Generation aborted:", e)
            import traceback; traceback.print_exc()
            # still try to salvage poses/points/ply
            try:
                if writer is not None:
                    writer.finish()
            except Exception as e2:
                print("[RS-Studio] finish() also failed:", e2)

        finally:
            scene.render.engine = prev_engine