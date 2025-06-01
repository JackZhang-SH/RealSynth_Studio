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

import json
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import bpy
import bpy.path as bpath
from mathutils import Vector

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


# --------------------------------------------------------------------------- #
# Single‑frame renderer (incremental)
# --------------------------------------------------------------------------- #
class FrameDatasetRenderer:
    """Render **one** Blender timeline frame using a fixed camera rig."""

    def __init__(self, frame_idx: int, cameras: Sequence[bpy.types.Object]):
        self.frame_idx = frame_idx
        self.cameras = list(cameras)
        self._frames_meta: List[dict] = []

    # --------------------------------------------------------------------- #
    # Public: incremental generator
    # --------------------------------------------------------------------- #
    def iter_render(self, root_out: Path) -> Iterator[None]:
        """Render one image at a time and yield control after each."""
        scene = bpy.context.scene
        scene.frame_set(self.frame_idx)

        frame_dir = root_out / f"frame_{self.frame_idx}"
        train_dir = frame_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        for view_idx, cam in enumerate(self.cameras):
            scene.camera = cam
            scene.render.filepath = str(train_dir / f"render_{view_idx:04d}.png")

            # -- render just one image ------------------------------------- #
            bpy.ops.render.render(write_still=True, use_viewport=True)

            # -- record metadata ------------------------------------------- #
            self._frames_meta.append(
                {
                    "file_path": f"train/render_{view_idx:04d}.png",
                    "transform_matrix": self._matrix_to_list(cam.matrix_world),
                }
            )

            yield None  # hand control back to modal operator

        # -- write transforms.json once all views are done ----------------- #
        self._write_transforms_json(frame_dir)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _matrix_to_list(mat) -> List[List[float]]:
        return [list(row) for row in mat]

    def _write_transforms_json(self, frame_dir: Path) -> None:
        """Write NeRF‑style transforms.json using collected frame metadata."""
        cam = self.cameras[0].data  # all cameras share identical intrinsics
        scene = bpy.context.scene

        scale = scene.render.resolution_percentage / 100.0
        width = scene.render.resolution_x * scale
        height = scene.render.resolution_y * scale
        focal_mm = cam.lens
        sensor_mm = cam.sensor_width
        focal_px = focal_mm / sensor_mm * width

        data = {
            "camera_angle_x": cam.angle_x,
            "camera_angle_y": cam.angle_y,
            "fl_x": focal_px,
            "fl_y": focal_px,
            "cx": width * 0.5,
            "cy": height * 0.5,
            "w": width,
            "h": height,
            "frames": self._frames_meta,
        }

        with open(frame_dir / "transforms.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


# --------------------------------------------------------------------------- #
# Multi‑frame driver (also incremental)
# --------------------------------------------------------------------------- #
class DatasetGenerator:
    """Drive timeline frames using *existing* RS Studio cameras."""

    def __init__(
        self,
        cameras: Sequence[bpy.types.Object],
        start_frame: int,
        end_frame: int,
    ) -> None:
        if not cameras:
            raise ValueError("No RS Studio cameras supplied")

        self.cameras = list(cameras)
        self.start = start_frame
        self.end = end_frame

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #
    @property
    def total_images(self) -> int:
        return (self.end - self.start + 1) * len(self.cameras)

    # --------------------------------------------------------------------- #
    # Public incremental generator
    # --------------------------------------------------------------------- #
    def iter_generate(self, output_dir: str | Path) -> Iterator[Tuple[int, int]]:
        """Incrementally generate the whole dataset."""
        root_out = Path(bpath.abspath(str(output_dir))).resolve()
        root_out.mkdir(parents=True, exist_ok=True)

        done, total = 0, self.total_images

        for frame_idx in range(self.start, self.end + 1):
            renderer = FrameDatasetRenderer(frame_idx=frame_idx, cameras=self.cameras)
            for _ in renderer.iter_render(root_out):
                done += 1
                yield done, total
