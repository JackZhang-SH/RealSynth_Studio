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
    scale = scene.render.resolution_percentage / 100.0
    w = scene.render.resolution_x * scale
    h = scene.render.resolution_y * scale

    # ALWAYS recompute from current FOV and size
    fx = 0.5 * w / math.tan(camdat.angle_x * 0.5)
    fy = 0.5 * h / math.tan(camdat.angle_y * 0.5)

    # ALWAYS derive principal point from normalized shifts
    cx = (0.5 + camdat.shift_x) * w
    cy = (0.5 - camdat.shift_y) * h

    dist = {k: float(camdat.get(k, 0.0)) for k in ("k1","k2","p1","p2")}
    return w, h, fx, fy, cx, cy, dist
# ──────────────────────────────────────────────────────────── #
#  New: export-format enumeration
# ──────────────────────────────────────────────────────────── #
class ExportFormat(Enum):
    NGP          = auto()
    NERF_SYNTH   = auto()
    TACV         = auto()          # Time-Archival Camera Virtualisation
    COLMAP_POSES = auto()      # ----- COLMAP (poses-only) ----
    COLMAP_3DGS_MESH = auto()   # ← NEW: surface-sampling route
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
        t = M_w2cv.to_translation()    # 可选全局缩放
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
                cam_lines.append(
                    f"{cam_id} PINHOLE {self._width} {self._height} "
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

            # -------- auto-convert .txt → .bin 若系统有 colmap ----------
            colmap_exe = shutil.which("colmap")
            if not colmap_exe:
                print("[RS-Studio] ⚠  未检测到 'colmap' 命令，已保留 .txt 文件",
                      file=sys.stdout)
                continue

            # 确保 Windows 路径含空格时也能执行：使用 list 传参
            cmd = [
                colmap_exe, "model_converter",
                "--input_path",  os.fspath(sparse0),
                "--output_path", os.fspath(sparse0),
                "--output_type", "BIN",
            ]
            print("[RS-Studio] ▶  running:", " ".join(cmd), file=sys.stdout)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print("[RS-Studio] ❌  colmap model_converter failed:", e,
                      file=sys.stderr)
# ------------------------------------------------------------------ 3DGS Writer (Surface-sampling route C)
class Colmap3DGSSurfaceWriter(ColmapPoseWriter):
    """
    Export a COLMAP dataset ready for 3DGS by:
    1) rendering per-view color PNGs (no depth rendering);
    2) sampling points directly on mesh surfaces (uniform-by-area);
    3) filtering by camera visibility ("what cameras would actually see"):
         - view frustum test (fx, fy, cx, cy; near/far; image bounds)
         - optional backface culling (normal faces camera)
         - occlusion ray-cast against the evaluated scene
       A point is kept if visible by >= min_visible_cameras;
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

    # ---- main finish: write poses (txt), then sample surfaces & write ply / points3D
    def finish(self):
        # 1) write cameras.txt / images.txt / empty points3D.txt (poses-only)
        super().finish()

        # 2) for each frame: build surface point cloud with visibility filter
        for frame, items in self._frames.items():
            frame_dir = self.root / f"frame_{frame}"
            sparse0   = frame_dir / "sparse" / "0"
            points_txt = sparse0 / "points3D.txt"

            pts = self._sample_surface_points_for_frame(frame, items)
            self._write_ply(frame_dir / "fused_points.ply", pts)

            # 3) write points3D.txt (OpenCV world)
            with points_txt.open("w", encoding="utf8") as f:
                f.write("# 3DGS initialization points generated from mesh surface sampling (OpenCV world)\n")
                pid = 1
                for (x, y, z, r, g, b) in pts:
                    f.write(f"{pid} {x} {y} {z} {r} {g} {b} 0.0 0\n")
                    pid += 1

            # optional BIN conversion if COLMAP exists
            colmap_exe = shutil.which("colmap")
            if colmap_exe:
                try:
                    subprocess.run(
                        [colmap_exe, "model_converter",
                         "--input_path",  os.fspath(sparse0),
                         "--output_path", os.fspath(sparse0),
                         "--output_type", "BIN"],
                        check=True, capture_output=True, text=True
                    )
                except subprocess.CalledProcessError as e:
                    print("[RS-Studio] ⚠ model_converter failed; keep .txt:", e.stderr)
    def _write_ply(self, path: Path, pts: list[tuple[float,float,float,int,int,int]]) -> None:
        """Write binary little-endian PLY with XYZ + RGB."""
        import struct
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        with open(path, "wb") as f:
            f.write(header)
            pack = struct.Struct("<fffBBB").pack
            for x, y, z, r, g, b in pts:
                f.write(pack(float(x), float(y), float(z), int(r), int(g), int(b)))
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
        min_vis_cams     = int(getattr(rs, "mesh_min_visible_cameras", 1))
        backface_cull    = bool(getattr(rs, "mesh_backface_cull", True))
        max_cams_check   = int(getattr(rs, "mesh_max_cameras_check", 12))
        vis_filter       = bool(getattr(rs, "mesh_visibility_filter", True))
        color_mode       = str(getattr(rs, "mesh_color_mode", "TEXTURE"))  # TEXTURE / VCOL / MATERIAL / WHITE
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
        # intrinsics cache per cam
        cam_infos = []



        # depsgraph & evaluated scene for accurate geometry + ray cast
        dg = bpy.context.evaluated_depsgraph_get()

        # 1) collect triangles (world space) + UV/material hooks  [Blender 4.4 safe]
        tris: list[dict] = []    # each: {p0,p1,p2,n0,n1,n2,area,uv0,uv1,uv2,img,base_rgba}
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

                    tris.append(dict(
                        p0=p0, p1=p1, p2=p2,
                        n0=n0, n1=n1, n2=n2,
                        uv0=uv0, uv1=uv1, uv2=uv2,
                        img=img, base_rgba=(255, 255, 255, 255),
                        area=area
                    ))
                    tot_area += area
            finally:
                # Always free the evaluated mesh to avoid leaks
                obj_eval.to_mesh_clear()

        if tot_area <= 0.0 or not tris:
            print("[RS-Studio] No triangles found (hidden-from-render? or all instances culled?).")
            return []

        # 2) stochastic sampling per area
        rng = np.random.default_rng(12345)
        # cumulative distribution
        areas = np.fromiter((t["area"] for t in tris), dtype=np.float64, count=len(tris))
        cdf = np.cumsum(areas); cdf /= cdf[-1]

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
            # area-weighted tri pick
            idx = int(np.searchsorted(cdf, rng.random(), side="right"))
            if idx >= len(tris):
                idx = len(tris) - 1
            t = tris[idx]

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
                accum[k]  = [Pw.x, Pw.y, Pw.z, float(r), float(g), float(b)]
                counts[k] = 1
                accepted_voxels += 1
            else:
                acc = accum[k]; c = counts[k]
                acc[0] += Pw.x; acc[1] += Pw.y; acc[2] += Pw.z
                acc[3] += r;    acc[4] += g;    acc[5] += b
                counts[k] = c + 1

        # If AABB is enabled and we did not reach the target, tell the user explicitly.
        if bounds_enable and accepted_voxels < target_count:
            print(f"[RS-Studio] AABB target {target_count} not reached: got {accepted_voxels} "
                f"after {trials} trials. Consider expanding bounds or increasing budget.")

        # average within voxels; convert to OpenCV world
        out: list[tuple[float,float,float,int,int,int]] = []
        for k, acc in accum.items():
            c = counts[k]
            P_bl = mu.Vector((acc[0]/c, acc[1]/c, acc[2]/c))
            P_cv = A_cv_from_bl.to_3x3() @ P_bl
            out.append((
                float(P_cv.x), float(P_cv.y), float(P_cv.z),
                int(round(acc[3]/c)), int(round(acc[4]/c)), int(round(acc[5]/c))
            ))

        print(
            f"[RS-Studio] frame {frame}: tris={len(tris)} tot_area={tot_area:.3f} "
            f"target(in-AABB)={target_count} vox={voxel} trials={trials} "
            f"fused_voxels={len(accum)} out_points={len(out)}"
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