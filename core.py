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
from typing import Iterator, List, Sequence, Tuple

import bpy
import bpy.path as bpath
from mathutils import Vector
from enum import Enum, auto
# ------------------------------------------------------------------ NEW ---- #
# core.py ────────────────────────────────────────────────────────────
def _intrinsics_from_cam(camdat: bpy.types.Camera, scene: bpy.types.Scene):
    """Extract w,h,fx,fy,cx,cy,dist from *camdat* **without** losing precision."""
    scale = scene.render.resolution_percentage / 100.0
    w = scene.render.resolution_x * scale
    h = scene.render.resolution_y * scale

    fx = float(camdat.get("fl_x", 0.5 * w / math.tan(camdat.angle_x * 0.5)))
    fy = float(camdat.get("fl_y", 0.5 * h / math.tan(camdat.angle_y * 0.5)))


    if "cx" in camdat.keys() and "cy" in camdat.keys():
        cx = float(camdat["cx"])
        cy = float(camdat["cy"])
    else:                                       # fallback: derive from shifts
        cx = (0.5 + camdat.shift_x) * w
        cy = (0.5 - camdat.shift_y) * h
    # ───── REPLACE ↑ 这一段 ─────

    dist = {k: float(camdat.get(k, 0.0)) for k in ("k1", "k2", "p1", "p2")}
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
                # 计算 fx, fy, cx, cy
                fx = camdat.lens / camdat.sensor_width  * self._width
                fy = fx
                if camdat.sensor_fit == 'VERTICAL':
                    fy = camdat.lens / camdat.sensor_height * self._height
                cx, cy = self._width / 2.0, self._height / 2.0
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
# ------------------------------------------------------------------ 3DGS Writer
class Colmap3DGSWriter(ColmapPoseWriter):
    """
    对每个 frame 依次执行：
        feature_extractor  →  exhaustive_matcher  →  mapper  →  model_converter
    生成真实的 cameras / images / points3D .bin 文件。
    若系统找不到 COLMAP，则自动降级为 ColmapPoseWriter（只写位姿）。
    """
    def __init__(self, root_out: Path, cam0, frame_cnt: int, colmap_ok: bool):
        super().__init__(root_out, cam0)
        self._frame_cnt   = frame_cnt
        self._colmap_ok   = colmap_ok

    # 满足抽象接口——这里交给 postprocess_iter 真正收尾
    def finish(self):
        pass

    # 每帧：feature + match + mapper + converter  ⇒ 4 步
    @property
    def extra_steps(self) -> int:
        return 0 if not self._colmap_ok else self._frame_cnt * 4

    # ------------------------------ 后处理主逻辑 ------------------------------ #
    def postprocess_iter(self) -> Iterator[None]:
        """
        ① 如果用户未安装 COLMAP 或任何一步出错，则自动退回“poses-only”导出；
        ② 在 Blender Console 打印完整 stderr，方便定位；
        ③ 始终保证 cameras.txt / images.txt / points3D.(txt|bin) 被落盘。
        """
        import subprocess, os, shutil

        # --------- 如果之前检测到没有 colmap，直接写位姿文件并结束 --------- #
        if not self._colmap_ok:
            super().finish()
            return

        # --------- 对每个 frame_x 逐帧调用 COLMAP 管道 --------- #
        for frame, _items in self._frames.items():
            frame_dir   = self.root / f"frame_{frame}"
            img_dir     = frame_dir / "images"
            db_path     = frame_dir / "database.db"
            sparse_root = frame_dir / "sparse"
            sparse_root.mkdir(parents=True, exist_ok=True)

            # 把 subprocess.run 写成一个小工具，统一错误处理
            def _run(cmd: list[str], step: str):
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as e:
                    # 记录日志到控制台并告知用户
                    print(f"[RS-Studio] ❌ COLMAP {step} failed:\n{e.stderr}")
                    print(f"[RS-Studio] ⚠  COLMAP {step} failed on frame_{frame}; "
                          f"falling back to poses-only.")
                    # 标记失败：后续 frame 也不再跑 COLMAP
                    self._colmap_ok = False
                    return False
                return True

            # 1) feature_extractor
            if not _run([
                "colmap", "feature_extractor",
                "--ImageReader.camera_model", "PINHOLE",
                "--database_path",  os.fspath(db_path),
                "--image_path",     os.fspath(img_dir),
                # ❗ 已去掉 --ImageReader.single_camera 1
            ], "feature_extractor"):
                break
            yield None  # +1 进度

            # 2) exhaustive_matcher
            if not _run([
                "colmap", "exhaustive_matcher",
                "--database_path", os.fspath(db_path),
            ], "exhaustive_matcher"):
                break
            yield None  # +1

            # 3) mapper
            if not _run([
                "colmap", "mapper",
                "--database_path", os.fspath(db_path),
                "--image_path",    os.fspath(img_dir),
                "--output_path",   os.fspath(sparse_root),
                "--Mapper.num_threads", "8",
            ], "mapper"):
                break

            # 如果 mapper 没产生 0 号模型，就把第一个模型重命名为 0
            model_dirs = sorted(p for p in sparse_root.iterdir() if p.is_dir())
            model_0 = sparse_root / "0"
            if not model_0.exists() and model_dirs:
                model_dirs[0].rename(model_0)
            yield None  # +1

            # 4) txt / bin 互转，确保生成 .bin
            if not _run([
                "colmap", "model_converter",
                "--input_path",  os.fspath(model_0),
                "--output_path", os.fspath(model_0),
                "--output_type", "BIN",
            ], "model_converter"):
                break
            yield None  # +1

        # --------- 如果任何一步失败，则 fallback 为 poses-only --------- #
        if not self._colmap_ok:
            super().finish()  # 写 cameras / images / points3D.txt

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
                if self._colmap_ok:
                    writer = Colmap3DGSWriter(root_out, cam0, frame_cnt, True)
                else:
                    writer = ColmapPoseWriter(root_out, cam0)
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