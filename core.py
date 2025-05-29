# core.py
"""
Core implementation: view-point sampling, per-frame rendering, and metadata
export.  Provides an *incremental* generator so the UI can refresh after every
single image render.

Key classes
-----------
SamplingStrategy                – abstract camera-position sampler
FibonacciSphereSampling         – default uniform sphere sampling
FrameDatasetRenderer            – renders one frame (many camera views)
DatasetGenerator                – drives multiple frames incrementally
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
    """Interface for producing n camera positions on a sphere of given radius."""

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


# --------------------------------------------------------------------------- #
# Single-frame renderer  (incremental)
# --------------------------------------------------------------------------- #
class FrameDatasetRenderer:
    """Render a single timeline frame from many camera positions."""

    def __init__(
        self,
        frame_idx: int,
        camera: bpy.types.Object,
        strategy: SamplingStrategy,
        images_per_frame: int,
        target: Vector,
        radius: float,
    ) -> None:
        self.frame_idx = frame_idx
        self.camera = camera
        self.strategy = strategy
        self.images_per_frame = images_per_frame
        self.target = target
        self.radius = radius

        self._frames_meta: list[dict] = []

    # --------------------------------------------------------------------- #
    # Public: incremental generator
    # --------------------------------------------------------------------- #
    def iter_render(self, root_out: Path) -> Iterator[None]:
        """
        Render **one image**, yield control, repeat.  After the last image,
        transforms.json is written.  Yields `None` after *each* image so
        external callers can update the UI.
        """
        scene = bpy.context.scene
        scene.frame_set(self.frame_idx)

        frame_dir = root_out / f"frame_{self.frame_idx}"
        train_dir = frame_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        positions = self.strategy.sample(self.images_per_frame, self.radius)

        for view_idx, pos in enumerate(positions):
            # -- position + orient camera ---------------------------------- #
            self._set_camera(pos)
            scene.render.filepath = str(train_dir / f"render_{view_idx:04d}.png")

            # -- render just one image ------------------------------------- #
            bpy.ops.render.render(write_still=True, use_viewport=True)

            # -- record metadata ------------------------------------------- #
            self._frames_meta.append(
                {
                    "file_path": f"train/render_{view_idx:04d}.png",
                    "transform_matrix": self._matrix_to_list(
                        self.camera.matrix_world
                    ),
                }
            )

            yield None  # hand control back to modal operator

        # -- write transforms.json once all views are done ----------------- #
        self._write_transforms_json(frame_dir)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _set_camera(self, position: Vector) -> None:
        """Move camera to *position* and turn it to look at *target*."""
        self.camera.location = position
        direction = self.target - self.camera.location
        self.camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    @staticmethod
    def _matrix_to_list(mat) -> List[List[float]]:
        return [list(row) for row in mat]

    def _write_transforms_json(self, frame_dir: Path) -> None:
        """Write NeRF-style transforms.json using collected frame metadata."""
        scene = bpy.context.scene
        cam = self.camera.data

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
# Multi-frame driver   (also incremental)
# --------------------------------------------------------------------------- #
class DatasetGenerator:
    """High-level driver that iterates over Blender timeline frames."""

    def __init__(
        self,
        camera_name: str,
        start_frame: int,
        end_frame: int,
        images_per_frame: int,
        radius: float,
        target: Tuple[float, float, float] | Vector = (0.0, 0.0, 0.0),
        sampling: SamplingStrategy | None = None,
    ) -> None:
        cam_obj = bpy.data.objects.get(camera_name)
        if cam_obj is None or cam_obj.type != "CAMERA":
            raise ValueError(f"No camera named {camera_name!r} found")
        self.camera = cam_obj

        self.start = start_frame
        self.end = end_frame
        self.images_per_frame = images_per_frame
        self.radius = radius
        self.target = Vector(target)
        self.strategy = sampling or FibonacciSphereSampling()

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #
    @property
    def total_images(self) -> int:  # total work units
        return (self.end - self.start + 1) * self.images_per_frame

    # --------------------------------------------------------------------- #
    # Public incremental generator
    # --------------------------------------------------------------------- #
    def iter_generate(self, output_dir: str | Path) -> Iterator[Tuple[int, int]]:
        """
        Incrementally generate the whole dataset.

        Yields
        ------
        (done, total) : Tuple[int,int]
            *done* images rendered so far and *total* images in the batch.
        """
        root_out = Path(bpath.abspath(str(output_dir))).resolve()
        root_out.mkdir(parents=True, exist_ok=True)

        done = 0
        total = self.total_images

        for frame_idx in range(self.start, self.end + 1):
            renderer = FrameDatasetRenderer(
                frame_idx=frame_idx,
                camera=self.camera,
                strategy=self.strategy,
                images_per_frame=self.images_per_frame,
                target=self.target,
                radius=self.radius,
            )
            for _ in renderer.iter_render(root_out):
                done += 1
                yield done, total  # report progress on every single image
