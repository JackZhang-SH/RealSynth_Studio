# operators.py
"""
Adds coloured mesh markers (sphere / torus / cone) to every RS-Studio camera so that
* split (train / valid / test)             → sphere colour
* camera-quality score  (good / warn / bad) → torus colour (hidden until evaluated)
* extra binary flag (e.g. occlusion)       → cone colour (hidden until flagged)

Mesh markers are hidden from final rendering but always visible in any viewport
shading because they rely on Object colour, not Overlay icon colours.

Palette (Okabe-Ito, colour-blind safe)
-------------------------------------
train  : #0072B2  → (0.00, 0.45, 0.70, 1.0)
valid  : #E69F00  → (0.90, 0.62, 0.00, 1.0)
test   : #D55E00  → (0.84, 0.37, 0.00, 1.0)
GOOD   : #009E73  → (0.00, 0.62, 0.45, 0.8)
WARN   : #F0E442  → (0.94, 0.89, 0.26, 0.8)
BAD    : #CC79A7  → (0.80, 0.47, 0.65, 0.8)
FLAG   : #CC79A7  → same as BAD
"""

from __future__ import annotations

import re
from pathlib import Path
import shutil
from typing import Optional, Dict, Tuple

import bpy
import bpy.path as bpath
from mathutils import Vector

from .core import (
    DatasetGenerator,
    ExportFormat,
    FibonacciSphereSampling,
    FibonacciHemisphereSampling,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
RS_CAMERA_PREFIX = "RS_Studio"                     # every RS Studio camera name starts here
_MARKER_SUFFIX   = "__marker"                      # child Empty name suffix

_SPLIT_SUFFIXES = ("train", "valid", "test")

# palette (RGBA 0-1)
SPLIT_COLORS = {
    "train": (0.29, 0.56, 0.89, 1.0),   # blue
    "valid": (0.96, 0.77, 0.09, 1.0),   # amber
    "test":  (0.91, 0.31, 0.22, 1.0),   # red
}

# collection names
_ROOT_COLLECTION = "RS_Studio_Camera"
_SUB_COLLECTIONS = {
    "train": "Train Cameras",
    "valid": "Valid Cameras",
    "test":  "Test Cameras",
}

# --------------------------------------------------------------------------- #
# Helpers: collection & marker management
# --------------------------------------------------------------------------- #
def _ensure_collections() -> Dict[str, bpy.types.Collection]:
    """Return a dict {split: collection} and guarantee they all exist."""
    # root
    root = bpy.data.collections.get(_ROOT_COLLECTION)
    if root is None:
        root = bpy.data.collections.new(_ROOT_COLLECTION)
        bpy.context.scene.collection.children.link(root)

    # sub-collections
    result: Dict[str, bpy.types.Collection] = {}
    for k, name in _SUB_COLLECTIONS.items():
        sub = root.children.get(name)
        if sub is None:
            sub = bpy.data.collections.new(name)
            root.children.link(sub)
        result[k] = sub
    return result


def _get_marker(cam: bpy.types.Object) -> Optional[bpy.types.Object]:
    """Return existing child Empty marker (if any)."""
    for child in cam.children:
        if child.name.endswith(_MARKER_SUFFIX):
            return child
    return None

# ---------------------------------------------------------------------------
# Helpers: create / update coloured marker
# ---------------------------------------------------------------------------
import bmesh

_SHARED_MESH_NAME = "RS_marker_mesh"      # 单例网格，所有小球共用
def _get_shared_marker_mesh() -> bpy.types.Mesh:
    """Return a small icosphere mesh (shared by all markers)."""
    mesh = bpy.data.meshes.get(_SHARED_MESH_NAME)
    if mesh is None:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.12)
        mesh = bpy.data.meshes.new(_SHARED_MESH_NAME)
        bm.to_mesh(mesh)
        bm.free()
    return mesh


def _ensure_marker(
    cam: bpy.types.Object,
    color: Tuple[float, float, float, float],
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """Attach / update a coloured icosphere (Mesh) as camera marker."""
    marker = _get_marker(cam)
    if marker is None:
        mesh = _get_shared_marker_mesh()
        marker = bpy.data.objects.new(f"{cam.name}{_MARKER_SUFFIX}", mesh)
        marker.hide_render = True                   # 不进入最终渲染
        marker.parent = cam
        marker.matrix_parent_inverse = cam.matrix_world.inverted()
        marker.scale = (1, 1, 1)        # 比较小，不挡视线
        collection.objects.link(marker)

    marker.color = color                           # 直接用 object.color
    return marker

def _move_between_split_collections(
    cam: bpy.types.Object,
    marker: bpy.types.Object | None,
    dest: bpy.types.Collection,
) -> None:
    """Unlink cam (and marker) from all RS Studio split collections and link to dest."""
    root = bpy.data.collections.get(_ROOT_COLLECTION)
    if root is None:
        dest.objects.link(cam)
        if marker:
            dest.objects.link(marker)
        return

    for col in root.children:
        if cam.name in col.objects:
            col.objects.unlink(cam)
        if marker and marker.name in col.objects:
            col.objects.unlink(marker)

    dest.objects.link(cam)
    if marker:
        dest.objects.link(marker)
# ------------------------------------------------------------------ Utils --#
def _force_object_color_shading() -> None:
    """Switch every 3D-view to use Object color so markers are visible."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.color_type = 'OBJECT'


# --------------------------------------------------------------------------- #
# Scene-level settings
# --------------------------------------------------------------------------- #
class RSDatasetSettings(bpy.types.PropertyGroup):
    output_dir: bpy.props.StringProperty(
        name="Output Directory", subtype="DIR_PATH", default="//nerf_dataset"
    )  # type: ignore
    start_frame: bpy.props.IntProperty(name="Start Frame", min=1, default=1)  # type: ignore
    end_frame: bpy.props.IntProperty(name="End Frame", min=1, default=1)  # type: ignore

    images_per_frame: bpy.props.IntProperty(
        name="Camera Count", min=1, default=60
    )  # type: ignore
    sampling_strategy: bpy.props.EnumProperty(  # type: ignore
        name="Sampling Strategy",
        items=[
            ("HEMI", "Fibonacci (Upper-Hemisphere)", "Uniformly sample cameras on z ≥ 0"),
            ("SPHERE", "Fibonacci (Full-Sphere)", "Uniformly sample the entire sphere"),
        ],
        default="HEMI",
    )
    radius: bpy.props.FloatProperty(name="Sphere Radius", min=0.1, default=10.0)  # type: ignore
    target_point: bpy.props.FloatVectorProperty(
        name="Target Point", subtype="TRANSLATION", default=(0.0, 0.0, 0.0)
    )  # type: ignore

    progress: bpy.props.FloatProperty(
        name="Progress", min=0.0, max=1.0, default=0.0, subtype="FACTOR"
    )  # type: ignore
    is_running: bpy.props.BoolProperty(name="Generating", default=False)  # type: ignore
    cameras_generated: bpy.props.BoolProperty(name="Cameras Generated", default=False)  # type: ignore

    camera_source: bpy.props.EnumProperty(  # type: ignore
        name="Camera Source",
        items=[
            ("DEFAULT", "Default Camera", "Use Blender default camera as template"),
            ("IMPORT", "Import JSON", "Load intrinsic & extrinsic from external json (coming soon)"),
        ],
        default="DEFAULT",
    )
    export_format: bpy.props.EnumProperty(# type: ignore
        name="Export Format",
        items=[
            ("NGP",        "Instant-NGP",      "Instant-NGP compatible"),
            ("NERF_SYNTH", "NeRF Synthetic",   "train/val/test + transforms_*.json"),
            ("TACV",       "TACV",          "frame_n/train + transforms.json + transforms_test.json"),
            ("COLMAP_POSES", "COLMAP (poses only)", "images/ + cameras/images .bin"),  
            ("COLMAP_3DGS", "3DGS (COLMAP)", "sparse/*.bin with point cloud"),
        ],
        default="NGP",
    )
    aabb_scale: bpy.props.IntProperty(                 # ← 放在其它参数后面即可
    name="AABB Scale",
    min=1, default=2,
    description="Axis-aligned bounding-box scale used by Instant-NGP"
    )  # type: ignore

# --------------------------------------------------------------------------- #
# Camera-split operator
# --------------------------------------------------------------------------- #
class RS_OT_SetCameraSplit(bpy.types.Operator):
    """Mark selected RS Studio cameras as train / valid / test."""

    bl_idname = "rs.set_camera_split"
    bl_label = "Set Camera Split"
    bl_options = {"REGISTER", "UNDO"}

    split: bpy.props.EnumProperty(  # type: ignore
        name="Split",
        items=[
            ("train", "Train", "Mark as train cameras"),
            ("valid", "Valid", "Mark as validation cameras"),
            ("test", "Test", "Mark as test cameras"),
        ],
        default="train",
    )

    # -------------------------------------------------------------- helpers #
    @staticmethod
    def _base_name(obj_name: str) -> str | None:
        m = re.match(rf"({RS_CAMERA_PREFIX}_[0-9]+)(?:_(?:{'|'.join(_SPLIT_SUFFIXES)}))?$", obj_name)
        return m.group(1) if m else None

    # ------------------------------------------------------------------ run #
    def execute(self, context):
        subcols = _ensure_collections()
        dest_col = subcols[self.split]
        color = SPLIT_COLORS[self.split]

        changed = 0
        for cam in context.selected_objects:
            if cam.type != "CAMERA" or not cam.name.startswith(RS_CAMERA_PREFIX):
                continue
            base = self._base_name(cam.name)
            if base is None:
                continue

            cam.name = f"{base}_{self.split}"

            # marker colour & collection move
            marker = _ensure_marker(cam, color, dest_col)
            _move_between_split_collections(cam, marker, dest_col)

            changed += 1

        if changed == 0:
            self.report({"WARNING"}, "No RS Studio cameras selected")
        else:
            self.report({"INFO"}, f"{changed} camera(s) set to {self.split}")
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Generate cameras
# --------------------------------------------------------------------------- #
class RS_OT_GenerateCameras(bpy.types.Operator):
    bl_idname = "rs.generate_cameras"
    bl_label = "Generate Cameras"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.rs_settings

        if s.camera_source != "DEFAULT":
            self.report({"INFO"}, "JSON import not implemented yet")
            return {"CANCELLED"}

        # ------------------------------------------------ template camera --
        created_template = False
        if context.scene.camera is None:
            cam_data = bpy.data.cameras.new("DefaultCam")
            template = bpy.data.objects.new("DefaultCam", cam_data)
            context.collection.objects.link(template)
            created_template = True
        else:
            template = context.scene.camera

        # ------------------------------------------------ sample positions --
        Sampler = FibonacciHemisphereSampling if s.sampling_strategy == "HEMI" else FibonacciSphereSampling
        positions = Sampler().sample(s.images_per_frame, s.radius)
        target = Vector(s.target_point)

        # ------------------------------------------------ idx bookkeeping --
        existing = [
            o
            for o in bpy.data.objects
            if o.type == "CAMERA" and o.name.startswith(RS_CAMERA_PREFIX)
        ]
        used_idx = {
            int(m.group(1))
            for o in existing
            if (m := re.match(rf"{RS_CAMERA_PREFIX}_(\d+)", o.name))
        }
        next_idx = max(used_idx) + 1 if used_idx else 0

        # ------------------------------------------------ ensure collections
        subcols = _ensure_collections()
        train_col = subcols["train"]

        # ------------------------------------------------ create cameras ---
        for off, pos in enumerate(positions):
            idx = next_idx + off
            name = f"{RS_CAMERA_PREFIX}_{idx}_train"

            cam = bpy.data.objects.get(name)
            if cam is None:
                cam = template.copy()
                cam.data = template.data.copy()
                cam.name = name
                train_col.objects.link(cam)

            cam.location = pos
            cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()

            _ensure_marker(cam, SPLIT_COLORS["train"], train_col)

        # ----------------------------------------- clean up temp template --
        if created_template:
            context.collection.objects.unlink(template)
            bpy.data.objects.remove(template, do_unlink=True)

        s.cameras_generated = True
        self.report(
            {"INFO"},
            f"Added {len(positions)} cameras (total {len(existing) + len(positions)})",
        ) 
        _force_object_color_shading()
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Clear cameras
# --------------------------------------------------------------------------- #
class RS_OT_ClearCameras(bpy.types.Operator):
    bl_idname = "rs.clear_cameras"
    bl_label = "Clear RS Studio Cameras"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        cams = [
            o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith(RS_CAMERA_PREFIX)
        ]
        markers = [_get_marker(c) for c in cams]
        for obj in [*cams, *[m for m in markers if m]]:
            bpy.data.objects.remove(obj, do_unlink=True)

        context.scene.rs_settings.cameras_generated = False
        self.report({"INFO"}, f"Removed {len(cams)} cameras")
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Dataset generation (unchanged)
# --------------------------------------------------------------------------- #
class RS_OT_GenerateDataset(bpy.types.Operator):
    """Generate dataset without blocking the UI (incremental rendering)."""

    bl_idname = "rs.generate_dataset"
    bl_label = "Generate Dataset"
    bl_options = {"REGISTER", "UNDO"}

    _timer: bpy.types.Timer | None = None
    _iterator = None
    _total_images: int = 0
    _cancel_requested: bool = False

    def execute(self, context):
        global _active_generator
        if _active_generator is not None:
            self.report({"WARNING"}, "A generation job is already running")
            return {"CANCELLED"}

        s = context.scene.rs_settings
        wm = context.window_manager
        if s.export_format == "COLMAP_3DGS" and not shutil.which("colmap"):
            self.report({"WARNING"},
                "未检测到 COLMAP，可执行文件 'colmap' 不在 PATH，已降级为 “COLMAP (poses only)”")
        rs_cams = [
            o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith(RS_CAMERA_PREFIX)
        ]
        if not rs_cams:
            self.report({"ERROR"}, "No RS Studio cameras found – generate cameras first")
            return {"CANCELLED"}

        generator = DatasetGenerator(
            cameras=rs_cams,
            start_frame=s.start_frame,
            end_frame=s.end_frame,
            export_fmt=ExportFormat[s.export_format],
        )
        self._total_images = generator.total_images

        output_dir = Path(bpath.abspath(s.output_dir)).resolve()
        self._iterator = generator.iter_generate(output_dir)

        wm.progress_begin(0, self._total_images)
        s.is_running, s.progress = True, 0.0
        self._cancel_requested = False
        _active_generator = self

        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    # ------------- modal loop / finish / cancel remain identical ---------- #
    def modal(self, context, event):
        s, wm = context.scene.rs_settings, context.window_manager
        if event.type == "ESC":
            self._cancel_requested = True
        if event.type == "TIMER":
            if self._cancel_requested:
                self._finish(wm, s, cancelled=True)
                self.report({"WARNING"}, "Dataset generation cancelled")
                return {"CANCELLED"}
            try:
                done, total = next(self._iterator)
            except StopIteration:
                self._finish(wm, s, cancelled=False)
                # ⬇️ 新增：强制刷新当前区域，立刻让进度条消失
                context.area.tag_redraw()

                self.report({"INFO"}, "Dataset generation finished ✔")
                return {"FINISHED"}
            except Exception as exc:
                self._finish(wm, s, cancelled=True)
                self.report({"ERROR"}, f"Generation failed: {exc}")
                return {"CANCELLED"}
            s.progress = done / total
            wm.progress_update(done)
            context.area.tag_redraw()
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _finish(self, wm, settings, *, cancelled: bool):
        global _active_generator
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()
        settings.is_running = False
        settings.progress = 0.0 if cancelled else 1.0
        _active_generator = None
        # ⬇️ 新增：刷新所有 3D-View，让 N-panel 立即重绘
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()
    def cancel(self, context):
        self._cancel_requested = True


# --------------------------------------------------------------------------- #
# Cancel button
# --------------------------------------------------------------------------- #
class RS_OT_CancelGeneration(bpy.types.Operator):
    bl_idname = "rs.cancel_generation"
    bl_label = "Cancel"

    def execute(self, context):
        global _active_generator
        if _active_generator is not None:
            _active_generator.cancel(context)
            return {"FINISHED"}
        self.report({"INFO"}, "No generation job is currently running")
        return {"CANCELLED"}


# --------------------------------------------------------------------------- #
# Global pointer to active generator
# --------------------------------------------------------------------------- #
_active_generator: Optional[RS_OT_GenerateDataset] = None


# --------------------------------------------------------------------------- #
# Registration list
# --------------------------------------------------------------------------- #
CLASSES = [
    RSDatasetSettings,
    RS_OT_GenerateDataset,
    RS_OT_CancelGeneration,
    RS_OT_GenerateCameras,
    RS_OT_ClearCameras,
    RS_OT_SetCameraSplit,
]
