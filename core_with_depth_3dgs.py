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
# ---------- small helpers: plane fit & (de)projection ----------

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

# ---------- main diagnostics: per-camera plane normals & deltas ----------

def debug_report_depth_planes(root_out: Path, frame: int, items, *,
                              max_cam=8,  # limit cameras for speed
                              step=8,     # sampling stride on the image grid
                              max_samples_per_cam=40000,
                              write_ply=True):
    """
    For each camera in `items` (the same list received by _fuse_points_for_frame),
    back-project a sparse grid using the current math, then:
      - Fit a plane (SVD) -> normal, RMS
      - Reproject back -> RMSE in pixels
      - Repeat under toggles to see which factor bends the plane:
          transform_mode: 'matrix_world' vs 'Rt'
          depth_mode:     'Z' vs 'RAY' (convert ray length to Zc)
          use_half_pixel: True vs False
    A combined JSON report is written to frame_{frame}/debug_backproj_report.json
    and optional per-camera PLY files are saved for visualization.
    """
    import mathutils as mu
    scn = bpy.context.scene
    frame_dir = Path(root_out) / f"frame_{frame}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # scene pixel aspect for sanity
    pasx = float(scn.render.pixel_aspect_x)
    pasy = float(scn.render.pixel_aspect_y)

    # toggles we will sweep
    transform_modes = ["matrix_world", "Rt"]
    depth_modes     = ["Z", "RAY"]
    half_pixel_opts = [True, False]

    # utility to load pixels (Non-Color) and return numpy arrays
    def load_np_pixels(img_path):
        img = bpy.data.images.load(os.fspath(img_path), check_existing=True)
        try:
            img.use_view_as_render = False
            img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        w, h = img.size[0], img.size[1]
        ch = int(getattr(img, "channels", 4) or 4)
        px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)  # Blender stores bottom→top
        bpy.data.images.remove(img)
        return px, w, h, ch

    # collect (color_path -> depth_path) used in this frame (same as exporter)
    # We reconstruct expected depth paths from filename convention
    color2depth = {}
    for cam_obj, img_rel in items:
        color_abs = (Path(root_out) / img_rel).resolve()
        # depth alongside: frame_X/depth/f####_c####.exr
        # Resolve any suffix Blender may add.
        depth_dir = color_abs.parents[1] / "depth"
        stem = f"f{frame:04d}_c{int(color_abs.stem):04d}"
        cand = sorted(depth_dir.glob(stem + "*.exr"))
        if cand:
            color2depth[color_abs] = cand[-1]

    report = {
        "frame": frame,
        "pixel_aspect": [pasx, pasy],
        "cameras": []
    }

    # iterate cameras
    for idx, (cam_obj, img_rel) in enumerate(items[:max_cam]):
        color_abs = (Path(root_out) / img_rel).resolve()
        depth_abs = color2depth.get(color_abs)
        if depth_abs is None or not depth_abs.exists():
            continue

        dep_np, w, h, dep_ch = load_np_pixels(depth_abs)
        # depth channel: take the first channel [.., .., 0]
        dep_np = dep_np[..., 0]

        # intrinsics
        w0,h0,fx,fy,cx,cy,_ = _intrinsics_from_cam(cam_obj.data, scn)
        if int(w0) != int(w) or int(h0) != int(h):
            fx *= (w / w0);  fy *= (h / h0)
            cx *= (w / w0);  cy *= (h / h0)

        # extrinsics
        # Replace the "static-like" call with explicit math that does not depend on self._scale
        M_bl = cam_obj.matrix_world
        A = mu.Matrix((
            (1, 0,  0, 0),  # OpenCV-world → Blender-world
            (0, 0,  1, 0),
            (0,-1,  0, 0),
            (0, 0,  0, 1),
        ))
        T = mu.Matrix((
            (1, 0,  0, 0),  # Blender-local → OpenCV-local
            (0,-1,  0, 0),
            (0, 0,-1, 0),
            (0, 0,  0, 1),
        ))
        M_w2cv = T @ M_bl.inverted() @ A
        R_cw = np.array(M_w2cv.to_3x3(), dtype=np.float64)
        R_wc = R_cw.T
        t_cw = np.array(M_w2cv.to_translation(), dtype=np.float64) * 1.0  # keep debug free of global scale
        C_w  = -R_wc @ t_cw
        Mw = np.array(cam_obj.matrix_world, dtype=np.float64)

        # sample pixels
        uu = np.arange(0, w, max(1, step))
        vv = np.arange(0, h, max(1, step))
        samp = [(int(u), int(v)) for v in vv for u in uu]
        random.shuffle(samp)
        samp = samp[:max_samples_per_cam]

        cam_entry = {
            "name": cam_obj.name,
            "image": os.fspath(color_abs),
            "depth": os.fspath(depth_abs),
            "w": int(w), "h": int(h),
            "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
            "metrics": []
        }

        for transform_mode in transform_modes:
            for depth_mode in depth_modes:
                for use_half_pixel in half_pixel_opts:
                    pts_world = []
                    uv_used   = []

                    # back-project
                    for (u, v) in samp:
                        # read depth from Blender's bottom→top buffer
                        vi = h - 1 - v
                        Z = float(dep_np[vi, u])
                        if not (cam_obj.data.clip_start < Z < cam_obj.data.clip_end):
                            continue

                        u_c = u + (0.5 if use_half_pixel else 0.0)
                        v_c = v + (0.5 if use_half_pixel else 0.0)
                        sx = (u_c - cx) / fx
                        sy = (v_c - cy) / fy

                        if depth_mode == "RAY":  # interpret depth as ray length L
                            rl = math.sqrt(1.0 + sx*sx + sy*sy)
                            Zc = Z / rl
                        else:                    # interpret depth as projection depth Zc
                            Zc = Z

                        Xc, Yc = sx*Zc, sy*Zc
                        if transform_mode == "matrix_world":
                            # OpenCV->Blender local: (Xb,Yb,Zb)=(Xc,-Yc,-Zc)
                            Xb, Yb, Zb = Xc, -Yc, -Zc
                            Pw = (Mw @ np.array([Xb, Yb, Zb, 1.0])).ravel()
                            Xw, Yw, Zw = Pw[0], Pw[1], Pw[2]
                        else:  # "Rt"
                            Xw, Yw, Zw = (R_wc @ np.array([Xc, Yc, Zc])) + C_w

                        pts_world.append([Xw, Yw, Zw])
                        uv_used.append([u_c, v_c])

                    if len(pts_world) < 100:
                        continue

                    P = np.asarray(pts_world, dtype=np.float64)
                    U = np.asarray(uv_used, dtype=np.float64)

                    # plane fit
                    c, n, rms = fit_plane_svd(P)

                    # reprojection error with the same intrinsics/extrinsics
                    K = build_K(fx, fy, cx, cy)
                    u_hat, v_hat, Zc_hat = project_cv(K, R_wc, C_w, P)
                    E = np.sqrt((u_hat - U[:,0])**2 + (v_hat - U[:,1])**2)
                    rmse = float(np.sqrt(np.mean(E**2)))
                    q95  = float(np.percentile(E, 95.0))

                    # pack metrics
                    cam_entry["metrics"].append({
                        "transform_mode": transform_mode,
                        "depth_mode": depth_mode,
                        "use_half_pixel": bool(use_half_pixel),
                        "num_points": int(P.shape[0]),
                        "plane_normal": [float(n[0]), float(n[1]), float(n[2])],
                        "plane_rms": rms,
                        "reproj_rmse_px": rmse,
                        "reproj_q95_px": q95
                    })

                    # optional: write per-camera PLY for quick visual check
                    if write_ply:
                        ply_path = frame_dir / f"debug_cam_{cam_obj.name}_{transform_mode}_{depth_mode}_{'half' if use_half_pixel else 'nohalf'}.ply"
                        with open(ply_path, "w", encoding="ascii") as f:
                            f.write("ply\nformat ascii 1.0\n")
                            f.write(f"element vertex {P.shape[0]}\n")
                            f.write("property float x\nproperty float y\nproperty float z\n")
                            f.write("end_header\n")
                            for x,y,z in P:
                                f.write(f"{x} {y} {z}\n")

        report["cameras"].append(cam_entry)

    # pairwise normal angles for best settings (choose the combo with min reproj RMSE per cam)
    def best_metric(entry):
        return min(entry["metrics"], key=lambda m: (m["reproj_rmse_px"], m["plane_rms"]))

    normals = []
    for cam_e in report["cameras"]:
        bm = best_metric(cam_e)
        normals.append((cam_e["name"], np.array(bm["plane_normal"], dtype=np.float64)))

    pairwise = []
    for i in range(len(normals)):
        for j in range(i+1, len(normals)):
            a_name, a_n = normals[i]
            b_name, b_n = normals[j]
            cosang = np.clip(float(a_n @ b_n) / (np.linalg.norm(a_n)*np.linalg.norm(b_n) + 1e-12), -1.0, 1.0)
            ang = float(np.degrees(np.arccos(abs(cosang))))  # abs to ignore normal flips
            pairwise.append({"a": a_name, "b": b_name, "angle_deg": ang})
    report["pairwise_best_normal_angles_deg"] = pairwise

    # write JSON
    out_json = frame_dir / "debug_backproj_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # console summary
    print("\n[RS-Studio][DEBUG] Depth back-projection diagnostics")
    print("  Pixel aspect (x,y):", pasx, pasy)
    for cam_e in report["cameras"]:
        bm = best_metric(cam_e)
        print(f"  Cam {cam_e['name']}: best transform={bm['transform_mode']}, depth={bm['depth_mode']}, half={bm['use_half_pixel']}, "
              f"n={bm['num_points']}  plane_rms={bm['plane_rms']:.6f}  reproj_rmse={bm['reproj_rmse_px']:.4f}px")
    for r in report["pairwise_best_normal_angles_deg"]:
        print(f"    Angle({r['a']}, {r['b']}) = {r['angle_deg']:.6f} deg")
    print(f"  -> JSON report: {out_json}\n")
# ------------------------------------------------------------------ NEW ---- #
# core.py ────────────────────────────────────────────────────────────
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
    COLMAP_3DGS  = auto()   
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
        t = M_w2cv.to_translation() * self._scale     # 可选全局缩放
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

# ------------------------------------------------------------------ 3DGS Writer (Depth-fusion route B)
class Colmap3DGSDepthWriter(ColmapPoseWriter):
    """
    Export a COLMAP dataset ready for 3DGS by:
    1) rendering per-view color (PNG) and depth (EXR, Z pass);
    2) back-projecting pixels to camera coordinates using (fx, fy, cx, cy);
    3) converting to Blender-local (X, Y, Z_bl = Xc, -Yc, -Zc) and then to world;
    4) voxel-grid fusion (averaging color), writing fused_points.ply;
    5) writing sparse model (cameras.txt, images.txt, points3D.txt) and
       converting to BIN via `colmap model_converter` when available.

    Notes:
    - v axis follows image row index (top->down). This matches cy we derive.
    - depth is in metres (Blender units). Background yields 0 or very large Z,
      we clip by [clip_start, clip_end].
    - Tracks in points3D are emitted with length 0 (allowed by COLMAP parser).
    """
    def __init__(self, root_out: Path, cam0):
        super().__init__(root_out, cam0)
        self._depth_paths: dict[Path, Path] = {}  # color-abs-path -> depth-abs-path

        scn = bpy.context.scene
        # ensure Z pass and compositor
        for view_layer in scn.view_layers:
            view_layer.use_pass_z = True
        scn.render.use_compositing = True
        scn.use_nodes = True

        self._fo_node, self._fo_slot = self._ensure_depth_output_node(scn)

        # render color as PNG (3DGS expects regular images)
        imgset = scn.render.image_settings
        imgset.file_format = 'PNG'
        imgset.color_mode = 'RGB'
        imgset.color_depth = '8'

        # tunables (read from scene.rs_settings if present)
        rs = getattr(scn, "rs_settings", None)
        self._stride = int(getattr(rs, "depth_stride", 4))             # sample every N pixels
        self._voxel = float(getattr(rs, "pointcloud_voxel", 0.01))     # metres
        self._max_points = int(getattr(rs, "pointcloud_max_points", 2_000_000))
        # --- force Cycles for reliable Z pass ---
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
        """
        Ensure a compositor pipeline that actually runs and writes Z as EXR.
        """
        nt = scene.node_tree

        # Render Layers
        rl = next((n for n in nt.nodes if n.type == 'R_LAYERS'), None)
        if rl is None:
            rl = nt.nodes.new("CompositorNodeRLayers")
        rl.scene = scene
        try:
            rl.layer = scene.view_layers.active.name
        except Exception:
            pass

        # Composite (ensure compositor runs)
        comp = next((n for n in nt.nodes if n.type == 'COMPOSITE'), None)
        if comp is None:
            comp = nt.nodes.new("CompositorNodeComposite")
        try:
            nt.links.new(rl.outputs.get("Image"), comp.inputs.get("Image"))
        except Exception:
            pass

        # File Output (EXR, 32-bit, single channel)
        fo = next((n for n in nt.nodes if n.name == "RS_Depth_Output" and n.type == 'OUTPUT_FILE'), None)
        if fo is None:
            fo = nt.nodes.new("CompositorNodeOutputFile")
            fo.name = "RS_Depth_Output"

        if not fo.file_slots:
            fo.file_slots.new("DepthZ")
        slot = fo.file_slots[0]
        slot.use_node_format = True  # <<< important: slot follows node format

        fo.base_path = "//"
        fo.format.file_format = 'OPEN_EXR'
        fo.format.color_mode  = 'BW'
        fo.format.color_depth = '32'

        # Link Depth (some versions call it "Z")
        depth_out = rl.outputs.get("Depth") or rl.outputs.get("Z")
        if depth_out is not None:
            # remove any prior link into fo.inputs[0]
            for l in list(nt.links):
                if l.to_node == fo and l.to_socket == fo.inputs[0]:
                    nt.links.remove(l)
            nt.links.new(depth_out, fo.inputs[0])

        return fo, slot
    def _resolve_depth_file(self, expected: Path) -> Path | None:
        """Return the actual EXR file path. Blender's File Output often appends frame digits."""
        if expected.exists():
            return expected
        cand = sorted(expected.parent.glob(expected.stem + "*.exr"))
        return cand[-1] if cand else None
    # ---- tell the compositor where to write the EXR for this render --------
    def _prepare_depth_path_for_seq(self, frame: int, seq: int, out_dir: Path) -> Path:
        depth_dir = out_dir / f"frame_{frame}" / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        # File Output node uses base_path + slot.path; no frame digits if we set exact name.
        self._fo_node.base_path = os.fspath(depth_dir)
        name = f"f{frame:04d}_c{seq:04d}"  # yields .../0000.exr
        self._fo_slot.path = name
        return depth_dir / f"{name}.exr"

    # ---- override file path hook to also schedule depth writing -------------
    def filepath_for(self, cam_obj, _global_idx):
        frame = bpy.context.scene.frame_current
        seq   = self._seq_per_frame.setdefault(frame, 0)

        # color path (PNG), identical to ColmapPoseWriter
        img_dir = self.root / f"frame_{frame}" / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        color_path = img_dir / f"{seq:04d}.png"

        # schedule EXR depth output through compositor
        depth_path = self._prepare_depth_path_for_seq(frame, seq, self.root)
        # remember mapping (absolute paths)
        self._depth_paths[color_path.resolve()] = depth_path

        self._seq_per_frame[frame] = seq + 1
        return color_path

    # keep registration logic (camera + image path)
    def register_frame(self, cam_obj, img_path: Path):
        super().register_frame(cam_obj, img_path)

    # ---- main finish: write poses (txt), fuse points, write ply & points3D --
    def finish(self):
        # write cameras.txt / images.txt / empty points3D.txt first (poses-only)
        super().finish()

        # then per-frame: fuse depth → PLY, and write points3D.txt with 0-length tracks
        for frame, items in self._frames.items():
            frame_dir = self.root / f"frame_{frame}"
            sparse0   = frame_dir / "sparse" / "0"
            points_txt = sparse0 / "points3D.txt"

            fused_pts_cv = self._fuse_points_for_frame(frame, items)   # OpenCV world
            # 1) write PLY in OpenCV world (for Open3D viewer)
            self._write_ply(frame_dir / "fused_points.ply", fused_pts_cv)
            debug_report_depth_planes(self.root, frame, items, step=max(4, self._stride), write_ply=True)
            # 2) write points3D.txt directly in OpenCV world
            with points_txt.open("w", encoding="utf8") as f:
                f.write("# 3DGS initialization points generated from depth fusion (OpenCV world)\n")
                pid = 1
                for (x, y, z, r, g, b) in fused_pts_cv:
                    f.write(f"{pid} {x} {y} {z} {r} {g} {b} 0.0 0\n")
                    pid += 1

            # optional: convert to BIN if COLMAP exists
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
                    # restore previous engine
        
        try:
            bpy.context.scene.render.engine = self._prev_engine
        except Exception:
            pass
    def _blender_to_colmap(self, cam_obj):
        import mathutils as mu
        M_bl = cam_obj.matrix_world
        A = mu.Matrix(((1,0,0,0),(0,0,1,0),(0,-1,0,0),(0,0,0,1)))
        T = mu.Matrix(((1,0,0,0),(0,-1,0,0),(0,0,-1,0),(0,0,0,1)))
        M_w2cv = T @ M_bl.inverted() @ A
        R = M_w2cv.to_3x3()
        t = M_w2cv.to_translation()      # NOTE: no global scaling here
        q = R.to_quaternion()
        return (q.w, q.x, q.y, q.z, t.x, t.y, t.z)
    # ---- depth fusion core ---------------------------------------------------
    def _fuse_points_for_frame(self, frame: int, items: list[tuple[bpy.types.Object, Path]]):
        """
        Return a list of (x, y, z, r, g, b) in world coordinates.
        Voxel-grid is used to merge duplicates and to tame point count.
        """
        scn = bpy.context.scene
        vox = max(self._voxel, 1e-6)
        stride = max(self._stride, 1)
        max_pts = max(self._max_points, 1)

        accum: dict[tuple[int, int, int], list[float]] = {}
        counts: dict[tuple[int, int, int], int] = {}
        import mathutils as mu
        for cam_obj, img_abs in items:
            color_abs = (self.root / img_abs).resolve()
            # --- map color → expected depth, then resolve actual filename ---
            depth_expected = self._depth_paths.get(color_abs) or self._depth_paths.get((self.root / img_abs))
            if depth_expected is None:
                continue
            depth_abs = self._resolve_depth_file(depth_expected)
            if depth_abs is None:
                print(f"[RS-Studio] Depth EXR not found for {color_abs.name} – looked for {depth_expected}")
                continue

            # load color & depth images
            col_img = bpy.data.images.load(os.fspath(color_abs), check_existing=True)
            dep_img = bpy.data.images.load(os.fspath(depth_abs), check_existing=True)
            # avoid any display transform while we read raw pixels
            try:
                dep_img.use_view_as_render = False
                dep_img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
            w, h = col_img.size[0], col_img.size[1]
            # intrinsics for this camera
            w0,h0,fx,fy,cx,cy,_ = _intrinsics_from_cam(cam_obj.data, scn)
            # guard (should match render size)
            if int(w0) != int(w) or int(h0) != int(h):
                fx *= (w / w0);  fy *= (h / h0)
                cx *= (w / w0);  cy *= (h / h0)

            # near/far to filter invalid depths
            z_near = float(cam_obj.data.clip_start)
            z_far  = float(cam_obj.data.clip_end)

            col_px = col_img.pixels[:]   # RGBA ... length = w*h*4
            dep_px = dep_img.pixels[:]
            col_ch = int(getattr(col_img, "channels", 4) or 4)  # usually 4 for PNG
            dep_ch = int(getattr(dep_img, "channels", 1) or 1)  # EXR depth is often 1
            R_wc_mu, C_w_mu = extrinsics_cv_from_matrix_world(cam_obj.matrix_world)
            taken = 0
            for v in range(0, h, stride):
                for u in range(0, w, stride):
                    # --- read & geometry must refer to the *same* pixel center ---
                    # Blender stores pixels bottom→top; our geometry uses top→down.
                    # So convert once, then use that single (u_c, v_c) everywhere.
                    v_img = (h - 1 - v)
                    base  = v_img * w + u
                    off_c = base * col_ch
                    off_d = base * dep_ch
                    Z = float(dep_px[off_d])  # first channel only
                    if not (z_near < Z < z_far) or not math.isfinite(Z) or Z <= 0.0:
                        continue
                    # pixel centers in their respective conventions
                    u_c = u + 0.5                              # x: left→right
                    v_td = v + 0.5                             # y:  top→down  (for cx,cy)
                    # now back-project using the *same* sample (u_c, v_td)
                    sx = (u_c - cx) / fx
                    sy = (v_td - cy) / fy
                    # Depth pass here is *projection depth* Zc (confirmed)

                    Xc = sx * Z
                    Yc = sy * Z
                    Zc = Z
                    # Blender local then world
                    P_w = (R_wc_mu @ mu.Vector((Xc, Yc, Zc))) + C_w_mu
                    xw, yw, zw = P_w.x, P_w.y, P_w.z
                    # color in 0..255
                    r = int(max(0, min(255, round(col_px[off_c + 0] * 255.0))))
                    g = int(max(0, min(255, round(col_px[off_c + 1] * 255.0))))
                    b = int(max(0, min(255, round(col_px[off_c + 2] * 255.0))))

                    key = (int(round(xw / vox)), int(round(yw / vox)), int(round(zw / vox)))
                    if key not in accum:
                        accum[key]  = [xw, yw, zw, float(r), float(g), float(b)]
                        counts[key] = 1
                    else:
                        acc = accum[key]
                        acc[0] += xw; acc[1] += yw; acc[2] += zw
                        acc[3] += r;  acc[4] += g;  acc[5] += b
                        counts[key] += 1

                    taken += 1
                    if len(accum) >= max_pts:
                        break
                if len(accum) >= max_pts:
                    break

            # clean up RAM
            bpy.data.images.remove(col_img)
            bpy.data.images.remove(dep_img)

        # average within voxels
        out: list[tuple[float, float, float, int, int, int]] = []
        for k, acc in accum.items():
            c = counts[k]
            x = acc[0] / c; y = acc[1] / c; z = acc[2] / c
            r = int(round(acc[3] / c)); g = int(round(acc[4] / c)); b = int(round(acc[5] / c))
            out.append((x, y, z, r, g, b))
        return out

    # ---- write ASCII PLY ----------------------------------------------------
    @staticmethod
    def _write_ply(path: Path, pts: list[tuple[float, float, float, int, int, int]]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for x,y,z,r,g,b in pts:
                f.write(f"{x} {y} {z} {r} {g} {b}\n")
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
        if export_fmt is ExportFormat.COLMAP_3DGS and self._colmap_ok:
            per_frame_extra = 3
        else:
            per_frame_extra = 0
        self.total_images = len(self.cameras) * frame_cnt + per_frame_extra
        
    def iter_generate(self, output_dir: str | Path):
        from bpy.types import RenderSettings

        scene = bpy.context.scene
        prev_engine = scene.render.engine

        # 枚举所有可用引擎，优先选 EEVEE Next，否则退回 Cycles
        engines = {e.identifier for e in RenderSettings.bl_rna.properties["engine"].enum_items}
        preferred = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "CYCLES"

        if scene.render.engine == "BLENDER_WORKBENCH":
            scene.render.engine = preferred        # 切换到能渲染贴图的引擎
        try:  
            root_out = Path(bpath.abspath(str(output_dir))).resolve()
            root_out.mkdir(parents=True, exist_ok=True)

            cam0 = self.cameras[0].data
            frame_cnt = self.end - self.start + 1

            # ---------- 创建 writer 实例 ----------
            if self.export_fmt is ExportFormat.COLMAP_3DGS:
                # New route B: depth-fusion writer
                writer = Colmap3DGSDepthWriter(root_out, cam0)
            elif self.export_fmt is ExportFormat.COLMAP_POSES:
                writer = ColmapPoseWriter(root_out, cam0)
            elif self.export_fmt is ExportFormat.NERF_SYNTH:
                writer = NeRFSyntheticWriter(root_out, cam0)
            elif self.export_fmt is ExportFormat.TACV:
                writer = TACVDatasetWriter(root_out, cam0)
            else:                                        # 默认 Instant-NGP
                writer = NGPDatasetWriter(root_out, cam0)

            self._writer_inst = writer                   # 供后处理阶段使用

            # ---------- 渲染循环 ----------
            done = 0
            for frame_idx in range(self.start, self.end + 1):
                bpy.context.scene.frame_set(frame_idx)
                for cam in self.cameras:
                    path = writer.filepath_for(cam, done)
                    bpy.context.scene.camera = cam
                    bpy.context.scene.render.filepath = str(path)
                    bpy.ops.render.render(write_still=True, use_viewport=False)

                    rel = path.relative_to(root_out).as_posix()
                    writer.register_frame(cam, rel)

                    done += 1
                    yield done, self.total_images

            # ---------- 后处理 ----------
            for _ in writer.postprocess_iter():
                done += 1
                yield done, self.total_images

            writer.finish()
        finally:
            scene.render.engine = prev_engine      # 恢复用户原设置