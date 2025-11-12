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

import math
import re
from pathlib import Path
import shutil
from typing import Optional, Dict, Tuple
from mathutils import Vector, Matrix
import math

import bpy
import bpy.path as bpath
from mathutils import Vector, Quaternion, Matrix
from bpy.props import BoolProperty   
from .importers import IMPORTER_REGISTRY
from .core import (
    DatasetGenerator,
    ExportFormat,
    FibonacciSphereSampling,
    FibonacciHemisphereSampling,
    EllipticalRingSampling
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

def _next_rs_index() -> int:
    existing = [
        o for o in bpy.data.objects
        if o.type == "CAMERA" and o.name.startswith(RS_CAMERA_PREFIX + "_")
    ]
    used = {
        int(m.group(1))
        for o in existing
        if (m := re.match(rf"{RS_CAMERA_PREFIX}_(\d+)", o.name))
    }
    return (max(used) + 1) if used else 0

def _ensure_marker(
    cam: bpy.types.Object,
    color: Tuple[float, float, float, float],
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    """
    Attach / update a coloured icosphere (Mesh) as camera marker.
    Always keeps marker at the camera's world transform.
    """
    marker = _get_marker(cam)
    if marker is None:
        mesh = _get_shared_marker_mesh()
        marker = bpy.data.objects.new(f"{cam.name}{_MARKER_SUFFIX}", mesh)
        marker.hide_render = True
        collection.objects.link(marker)
        marker.parent = cam

    # ── 无论新建/已有，都同步到相机位姿 ────────────────────────
    marker.matrix_parent_inverse = cam.matrix_world.inverted()
    marker.matrix_world          = cam.matrix_world
    marker.scale = (1.0, 1.0, 1.0)
    marker.color = color
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
# settings 枚举动态生成
def _import_items(self, _context):
    return [(k, k, "") for k in IMPORTER_REGISTRY.keys()]

# ─────────────────────── Scene-level settings ───────────────────── #
class RSDatasetSettings(bpy.types.PropertyGroup):
    # ---------- camera generation ----------
    images_per_frame: bpy.props.IntProperty(          # 生成的相机数量
        name="Number of Cameras", min=1, default=60
    )  # type: ignore

    sampling_strategy: bpy.props.EnumProperty(        # 采样方法
        name="Sampling Method",
        items=[
            ("HEMI",   "Fibonacci (Upper-Hemisphere)", ""),
            ("SPHERE", "Fibonacci (Full-Sphere)",     ""),
            ("ELLIPSE","Ellipse (Ring, centered)",     ""),
        ],
        default="HEMI",
    )  # type: ignore

    radius: bpy.props.FloatProperty(                  # 球面半径
        name="Sampling Radius", min=0.1, default=10.0
    )  # type: ignore

    target_point: bpy.props.FloatVectorProperty(      # 视点
        name="Look-at Point", subtype="TRANSLATION", default=(0.0, 0.0, 0.0)
    )  # type: ignore
    camera_source: bpy.props.EnumProperty(
        name="Template",
        items=[
            ("DEFAULT", "Default Camera", "Use current scene camera"),
        ],
        default="DEFAULT",
    )# type: ignore

    # ---------- new: camera import ----------
    import_format: bpy.props.EnumProperty(
        name="Format", items=_import_items
    )# type: ignore
    import_dir: bpy.props.StringProperty(
        name="Import Path", subtype="FILE_PATH", default="//",
        description="Can be a folder (e.g. COLMAP model) or a calibration .json file"
    )# type: ignore

    # ---------- dataset generation ----------
    output_dir: bpy.props.StringProperty(
        name="Output Dir", subtype="DIR_PATH", default="//nerf_dataset"
    )# type: ignore
    start_frame: bpy.props.IntProperty(name="Start", min=1, default=1)# type: ignore
    end_frame: bpy.props.IntProperty(name="End", min=1, default=1)# type: ignore
    export_format: bpy.props.EnumProperty(
        name="Export",
        items=[
            ("NGP",          "Instant-NGP",        ""),
            ("NERF_SYNTH",   "NeRF Synthetic",     ""),
            ("TACV",         "TACV",               ""),
            ("COLMAP_POSES", "COLMAP (poses)",     ""),
            ("COLMAP_3DGS_MESH", "3DGS (COLMAP, Surface)", ""), 
            ("COLMAP_3DGS_DEPTH", "3DGS (COLMAP, Depth)",   ""),  
            ("DEPTH_PT", "Depth (.npy + optional .pt)", ""),  # saves .npy always; .pt if torch is available
        ],
        default="NGP",
    )# type: ignore
    aabb_scale: bpy.props.IntProperty(name="AABB Scale", min=1, default=2)# type: ignore
    scale: bpy.props.FloatProperty(# type: ignore
        name="Scale", min=0.0001, default=1.0,
        description="Multiply camera translation by this factor",
    )

    # ---------- runtime ----------
    progress: bpy.props.FloatProperty(name="Progress", min=0.0, max=1.0, default=0.0)# type: ignore
    is_running: bpy.props.BoolProperty(name="Generating", default=False)# type: ignore
    cameras_generated: bpy.props.BoolProperty(name="Cameras Ready", default=False)# type: ignore
    # ---------- import scaling ----------
    import_scale: bpy.props.FloatProperty(          # type: ignore
        name="Scale",
        min=0.0001,
        default=1.0,
        description="Multiply camera translation when importing",
    )
    scan_fix_scale: BoolProperty(# type: ignore
        name="Apply Scale",
        description="Apply object scale so that 1 BU = 1 m",
        default=True,
    )# type: ignore
    scan_fix_material: BoolProperty(# type: ignore
        name="Repair Material",
        description="Auto-build (or patch) Principled-BSDF with textures",
        default=True,
    )# type: ignore
    # ---- depth-fusion point cloud params (for COLMAP_3DGS export) ----
    depth_stride: bpy.props.IntProperty(            # type: ignore
        name="Depth Stride (px)",
        min=1, default=4,
        description="Sample every N pixels when back-projecting depth"
    )
    pointcloud_voxel: bpy.props.FloatProperty(      # type: ignore
        name="Voxel Size (m)",
        min=1e-5, default=0.01,
        description="Voxel size for voxel-grid fusion in meters"
    )
    pointcloud_max_points: bpy.props.IntProperty(   # type: ignore
        name="Max Points",
        min=1, default=2_000_000,
        description="Upper bound on fused points (controls memory/time)"
    )
    mesh_points_target: bpy.props.IntProperty(     # type: ignore
        name="Target Points",
        min=1, default=800_000,
        description="Target number of points after voxel merge"
    )
    mesh_voxel: bpy.props.FloatProperty(           # type: ignore
        name="Voxel Size (m)",
        min=1e-5, default=0.01,
        description="Voxel size for surface-sampling merge in meters"
    )
    mesh_visibility_filter: bpy.props.BoolProperty(  # type: ignore
        name="Visibility Filter",
        default=True,
        description="Keep points only if visible from cameras"
    )
    mesh_min_visible_cameras: bpy.props.IntProperty( # type: ignore
        name="Min Visible Cams",
        min=1, max=999, default=1,
        description="A point must be visible from at least this many cameras"
    )
    mesh_backface_cull: bpy.props.BoolProperty(      # type: ignore
        name="Backface Cull",
        default=True,
        description="Discard samples whose normal faces away from camera"
    )
    mesh_max_cameras_check: bpy.props.IntProperty(   # type: ignore
        name="Max Cams to Check",
        min=1, default=12,
        description="Upper bound of cameras tested for visibility (speed)"
    )
    mesh_color_mode: bpy.props.EnumProperty(         # type: ignore
        name="Color Source",
        items=[
            ("TEXTURE",  "Albedo from Texture (UV)", ""),
            ("VCOL",     "Vertex Color",              ""),
            ("MATERIAL", "Material Base Color",       ""),
            ("WHITE",    "White",                     ""),
        ],
        default="TEXTURE",
        description="How to derive per-point color"
    )
    # ---- AABB bounds (OpenCV/COLMAP world) ----
    mesh_bounds_enable: bpy.props.BoolProperty(  # type: ignore
        name="Limit to Bounds (CV world)",
        description="Keep only points within the axis-aligned bounds in OpenCV/COLMAP world coordinates",
        default=False,
    )

    mesh_bounds_min: bpy.props.FloatVectorProperty(  # type: ignore
        name="Bounds Min (x,y,z)",
        description="Lower corner of the AABB in OpenCV/COLMAP world coordinates",
        size=3, subtype='XYZ',
        default=(-10.0, -10.0, -10.0),
    )

    mesh_bounds_max: bpy.props.FloatVectorProperty(  # type: ignore
        name="Bounds Max (x,y,z)",
        description="Upper corner of the AABB in OpenCV/COLMAP world coordinates",
        size=3, subtype='XYZ',
        default=(10.0, 10.0, 10.0),
    )
    # ---- layered sampling (mitigate ground dominance) ----
    mesh_object_gamma: bpy.props.FloatProperty(      # type: ignore
        name="Object Weight Gamma",
        min=0.01, max=1.0, default=0.5,
        description="Exponent for per-object sampling weight = area^gamma (lower flattens dominance)"
    )
    mesh_object_cap_ratio: bpy.props.FloatProperty(  # type: ignore
        name="Object Cap Ratio",
        min=0.05, max=1.0, default=0.33,
        description="Max share of target points any single object can take"
    )

    # ---- micro thickness (thin shell for stability) ----
    mesh_thickness_eps: bpy.props.FloatProperty(     # type: ignore
        name="Micro Thickness (m)",
        min=0.0, soft_max=0.02, default=0.0,
        description="Duplicate points at ±epsilon along normal to create a thin shell"
    )
    mesh_thickness_two_sided: bpy.props.BoolProperty(# type: ignore
        name="Two-sided Thickness",
        default=True,
        description="If enabled, create both +epsilon and -epsilon copies; if off, only +epsilon"
    )
    ellipse_minor: bpy.props.FloatProperty(           # type: ignore
        name="Ellipse Minor Radius (b)",
        description="Semi-minor axis length for the ellipse (major axis = Sampling Radius)",
        min=0.1, default=8.0
    )
    ellipse_center: bpy.props.FloatVectorProperty(       # type: ignore
        name="Ellipse Center (x,y,z)",
        description="World-space center of the ellipse",
        size=3, subtype='TRANSLATION',
        default=(0.0, 0.0, 0.0),
    )
    # NEW — focal length control
    focal_length_mm: bpy.props.FloatProperty(         # type: ignore
        name="Focal Length (mm)",
        description="Camera focal length in millimeters (perspective cameras)",
        min=1.0, max=1000.0, default=35.0
    )   
    # ---------- Stadium (Ring + Tile) ----------
    pitch_length_m: bpy.props.FloatProperty(  # type: ignore
        name="Pitch Length (m)", min=10.0, default=105.0,
        description="Field length (X axis), e.g., 105 m"
    )
    pitch_width_m: bpy.props.FloatProperty(   # type: ignore
        name="Pitch Width (m)", min=10.0, default=68.0,
        description="Field width (Y axis), e.g., 68 m"
    )

    # Ring cameras
    ring_k: bpy.props.IntProperty(            # type: ignore
        name="Ring Count (k)", min=1, max=360, default=12,
        description="Number of panoramic cameras on the ring"
    )
    ring_radius_m: bpy.props.FloatProperty(   # type: ignore
        name="Ring Radius (m)", min=1.0, default=65.0
    )
    ring_height_m: bpy.props.FloatProperty(   # type: ignore
        name="Ring Height (m)", min=0.1, default=25.0
    )
    tile_row_stagger_ratio: bpy.props.FloatProperty(  # type: ignore
        name="Row Stagger X Ratio",
        description="Shift X per row to avoid overlaps; 0 disables row-stagger",
        min=0.0, max=1.0, default=0.33
    )

    # Tiles (Nx × Ny), m cameras per tile placed near sidelines
    tiles_nx: bpy.props.IntProperty(          # type: ignore
        name="Tiles X (Nx)", min=1, default=6
    )
    tiles_ny: bpy.props.IntProperty(          # type: ignore
        name="Tiles Y (Ny)", min=1, default=3
    )
    tile_cams_per_zone: bpy.props.IntProperty(  # type: ignore
        name="Cams per Tile (m)", min=1, max=16, default=2,
        description="Number of near long-lens cameras per tile"
    )
    tile_side_mode: bpy.props.EnumProperty(     # type: ignore
        name="Side Mode",
        description="How to place m cameras with respect to sidelines",
        items=[
            ("BOTH", "Both Sides (split L/R)", "Split m evenly on left/right"),
            ("NEAREST", "Nearest Sideline", "Place all m on the nearest sideline"),
        ],
        default="BOTH",
    )
    sideline_out_offset_m: bpy.props.FloatProperty(  # type: ignore
        name="Sideline Offset (m)", min=0.0, default=6.0,
        description="Offset outside sidelines for long-lens cameras"
    )
    tile_cam_height_m: bpy.props.FloatProperty(      # type: ignore
        name="Tile Cam Height (m)", min=0.1, default=2.5
    )
    tile_multi_dx_ratio: bpy.props.FloatProperty(    # type: ignore
        name="Multi-cam X Offset Ratio", min=0.0, max=1.0, default=0.25,
        description="Spread multiple cams along X inside one tile: offset = k * ratio * tile_dx"
    )
    ring_lens_mm: bpy.props.FloatProperty( # type: ignore
        name="Ring Lens (mm)",
        description="Focal length for ring cameras",
        min=1.0, max=1000.0, default=35.0,
    )

    tile_lens_mm: bpy.props.FloatProperty( # type: ignore
        name="Tile Lens (mm)", 
        description="Focal length for tile cameras",
        min=1.0, max=1000.0, default=102.0,  # ≈ 36mm sensor & ~20° FoVx
    )
    # Lens & clipping for both groups
    sensor_width_mm: bpy.props.FloatProperty(        # type: ignore
        name="Sensor Width (mm)", min=1.0, max=100.0, default=36.0
    )
    stadium_clip_start: bpy.props.FloatProperty(     # type: ignore
        name="Clip Start", min=0.001, default=0.1
    )
    stadium_clip_end: bpy.props.FloatProperty(       # type: ignore
        name="Clip End", min=10.0, default=1000.0
    )
    stadium_clear_old: bpy.props.BoolProperty(       # type: ignore
        name="Clear Old", default=True,
        description="Delete existing Cam_Ring_* / Cam_Tile_* before generating"
    )

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
        # ------------------------------------------------ sample positions --
        if s.sampling_strategy == "HEMI":
            positions = FibonacciHemisphereSampling().sample(s.images_per_frame, s.radius)
        elif s.sampling_strategy == "SPHERE":
            positions = FibonacciSphereSampling().sample(s.images_per_frame, s.radius)

        else:  # "ELLIPSE"
            Sampler = EllipticalRingSampling(
                a=s.radius,
                b=s.ellipse_minor,
                center=s.ellipse_center,   # NEW
            )
            positions = Sampler.sample(s.images_per_frame, s.radius)
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
            if cam.data is not None and getattr(cam.data, "type", None) == 'PERSP':
                cam.data.lens = s.focal_length_mm
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
    
class RS_OT_ClearStadiumCameras(bpy.types.Operator):
    bl_idname = "rs.clear_stadium_cameras"
    bl_label  = "Clear Stadium Cameras"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # remove RS_Studio cameras that were generated by Stadium layout
        cams = [
            o for o in bpy.data.objects
            if o.type == 'CAMERA'
            and o.name.startswith(RS_CAMERA_PREFIX + "_")
            and o.get("rs_group") == "stadium"
        ]
        n = 0
        for cam in cams:
            m = _get_marker(cam)
            if m:
                bpy.data.objects.remove(m, do_unlink=True)
            bpy.data.objects.remove(cam, do_unlink=True)
            n += 1

        # also remove helper empties (TileCenter)
        helpers = [o for o in bpy.data.objects if o.type == 'EMPTY' and o.name.startswith("TileCenter_")]
        for o in helpers:
            bpy.data.objects.remove(o, do_unlink=True)

        self.report({'INFO'}, f"Removed {n} stadium RS camera(s) and {len(helpers)} helper(s)")
        return {"FINISHED"}

# ───────────────────────── Import cameras ───────────────────────── #
class RS_OT_ImportCameras(bpy.types.Operator):
    """Import external cameras → RS-Studio cameras (default *train*)."""

    bl_idname = "rs.import_cameras"
    bl_label  = "Import Cameras"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s   = context.scene.rs_settings
        fmt = s.import_format
        importer = IMPORTER_REGISTRY.get(fmt)
        if importer is None:
            self.report({'ERROR'}, f"Unsupported format: {fmt}")
            return {'CANCELLED'}

        model_dir = Path(bpy.path.abspath(s.import_dir)).resolve()

        # ① collections
        subcols   = _ensure_collections()
        train_col = subcols["train"]

        # ② next camera index
        existing = [
            o for o in bpy.data.objects
            if o.type == "CAMERA" and o.name.startswith(RS_CAMERA_PREFIX)
        ]
        used = {
            int(m.group(1))
            for o in existing
            if (m := re.match(rf"{RS_CAMERA_PREFIX}_(\d+)", o.name))
        }
        next_idx = max(used) + 1 if used else 0

        # ③ call concrete importer
        try:
            created = importer(
                model_dir,
                name_prefix = RS_CAMERA_PREFIX,
                start_index = next_idx,
                collection  = train_col,
                color       = SPLIT_COLORS["train"],
            )
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}

        if created == 0:
            self.report({'WARNING'}, "No cameras imported (empty model?)")
            return {'CANCELLED'}

        # ④ make sure marker exists *and* sits at cam location
        for idx in range(next_idx, next_idx + created):
            cam_name = f"{RS_CAMERA_PREFIX}_{idx}_train"
            cam = bpy.data.objects.get(cam_name)
            if cam is None or cam.type != 'CAMERA':
                continue

            marker = _get_marker(cam)
            if marker is None:
                # create fresh marker
                marker = _ensure_marker(cam, SPLIT_COLORS["train"], train_col)
            else:
                # relocate / re-orient existing marker
                marker.location = cam.location
                marker.rotation_euler = cam.rotation_euler
                if marker.name not in train_col.objects:
                    train_col.objects.link(marker)

        s.cameras_generated = True
        _force_object_color_shading()

        self.report({'INFO'}, f"Imported {created} camera(s) from {fmt}")
        return {'FINISHED'}

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
            o for o in bpy.data.objects
            if o.type == "CAMERA" and (
                o.name.startswith(RS_CAMERA_PREFIX) or
                o.name.startswith("Cam_Ring_") or
                o.name.startswith("Cam_Tile_")
            )
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
# ---------------------------------- helper --------------------------------- #
def _has_paired_tex_to_principled(mat: bpy.types.Material) -> bool:
    """Return True if there is at least one ImageTexture linked to Principled"""
    if not mat.use_nodes:
        return False
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            for link in n.inputs["Base Color"].links:
                if link.from_node.type == "TEX_IMAGE":
                    return True
    return False


def _smart_pbr(mat: bpy.types.Material) -> None:
    """Only build/repair when材质缺失或未正确连线"""
    mat.use_nodes = True
    nt, nodes, links = mat.node_tree, mat.node_tree.nodes, mat.node_tree.links

    if _has_paired_tex_to_principled(mat):
        # 材质 already ok – nothing to do
        return

    # ========== rebuild or patch ==========
    tex_nodes = [n for n in nodes if n.type == "TEX_IMAGE"]
    principled = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)

    # ① 若无 Principled，则创建
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        principled.location = (0, 0)

    # ② 若未连贴图，则把第一张贴图接 base-color & alpha
    if tex_nodes:
        tex = tex_nodes[0]
        links.new(tex.outputs.get("Color"), principled.inputs["Base Color"])
        if tex.outputs.get("Alpha"):
            links.new(tex.outputs["Alpha"], principled.inputs["Alpha"])

    # ③ 保证 Material Output
    out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        out = nodes.new("ShaderNodeOutputMaterial")
        out.location = (300, 0)
    # ④ 连线
    if not principled.outputs["BSDF"].is_linked:
        links.new(principled.outputs["BSDF"], out.inputs["Surface"])

# --------------------------------------------------------------------------- #
# Organise imported scan model(s)                                              #
# --------------------------------------------------------------------------- #
class RS_OT_OrganizeScan(bpy.types.Operator):
    """Move selected scan objects to *Scans/Models*, apply unit-scale,
    and give every mesh a clean Principled-BSDF material."""
    bl_idname  = "rs.organize_scan"
    bl_label   = "Organize Scan Model"
    bl_options = {"REGISTER", "UNDO"}

    def _ensure_scan_collection(self) -> bpy.types.Collection:
        root = bpy.data.collections.get("Scans")
        if root is None:
            root = bpy.data.collections.new("Scans")
            bpy.context.scene.collection.children.link(root)

        models = root.children.get("Models")
        if models is None:
            models = bpy.data.collections.new("Models")
            root.children.link(models)
        return models

    # ------------------------------------------------------------------ run #
    def execute(self, context):
        models_col = self._ensure_scan_collection()
        selected   = [o for o in context.selected_objects]

        if not selected:
            self.report({"WARNING"}, "Nothing selected")
            return {"CANCELLED"}

        # Work in Object mode
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        # Backup current selection
        orig_sel = set(context.selected_objects)

        for obj in selected:
            # 1) link / relink to Scans/Models collection
            if models_col not in obj.users_collection:
                models_col.objects.link(obj)
            for c in obj.users_collection[:]:
                if c != models_col:
                    c.objects.unlink(obj)

            # 2) apply scale so that scale = (1,1,1)
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj
            if obj.type == 'MESH':
                bpy.ops.object.transform_apply(location=False,
                                               rotation=False,
                                               scale=True)
            obj.select_set(False)

            # 3) ensure Principled-BSDF material
            if obj.type == 'MESH':
                if not obj.data.materials:
                    mat = bpy.data.materials.new("ScanMaterial")
                    _smart_pbr(mat)
                    obj.data.materials.append(mat)
                else:
                    for mat in obj.data.materials:
                        if mat is None:
                            mat = bpy.data.materials.new("ScanMaterial")
                            obj.data.materials.append(mat)
                        _smart_pbr(mat)

        # restore previous selection
        for o in orig_sel:
            o.select_set(True)

        self.report({'INFO'}, f"Organised {len(selected)} object(s)")
        return {'FINISHED'}
# ---------- Stadium layout (Ring + Tiles) generator ----------


class RS_OT_GenerateStadiumCameras(bpy.types.Operator):
    """Generate stadium RS_Studio cameras (Train split only, with marker).
       All cameras live directly under:
         RS_Studio_Camera → Train Cameras
    """
    bl_idname = "rs.generate_stadium_cameras"
    bl_label  = "Generate Stadium Cameras"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        import math, re, bpy
        from mathutils import Vector

        s = context.scene.rs_settings

        # ---------------- helpers ----------------
        def set_look_at(cam_obj: bpy.types.Object, target: Vector) -> None:
            cam_obj.rotation_euler = (target - cam_obj.location).to_track_quat("-Z", "Y").to_euler()

        def next_rs_index(prefix: str) -> int:
            used = set()
            for o in bpy.data.objects:
                if o.type == 'CAMERA' and o.name.startswith(prefix + "_"):
                    m = re.match(rf"{re.escape(prefix)}_(\d+)", o.name)
                    if m:
                        used.add(int(m.group(1)))
            return (max(used) + 1) if used else 0

        def link_once(col: bpy.types.Collection, obj: bpy.types.Object) -> None:
            if col not in obj.users_collection:
                col.objects.link(obj)

        # RS prefix fallback
        try:
            rs_prefix = RS_CAMERA_PREFIX
        except NameError:
            rs_prefix = "RS_Studio"

        # RS Studio split collections (root/train/valid/test)
        subcols   = _ensure_collections()
        train_col = subcols["train"]  # <- ONLY destination collection

        # -------------- optional cleanup --------------
        if getattr(s, "stadium_clear_old", True):
            # delete last batch of stadium RS cameras (+ markers)
            to_remove = [
                o for o in bpy.data.objects
                if o.type == 'CAMERA'
                and o.name.startswith(rs_prefix + "_")
                and o.get("rs_group") == "stadium"
            ]
            for o in to_remove:
                m = _get_marker(o)          # do NOT create new marker here
                if m:
                    bpy.data.objects.remove(m, do_unlink=True)
                bpy.data.objects.remove(o, do_unlink=True)

            # helpers
            helpers = [o for o in bpy.data.objects if o.type == 'EMPTY' and o.name.startswith("TileCenter_")]
            for o in helpers:
                bpy.data.objects.remove(o, do_unlink=True)

            # legacy visual-only (top-level RingCams/TileCams) — just in case, purge contents
            for lc_name in ("RingCams", "TileCams"):
                lc = bpy.data.collections.get(lc_name)
                if lc and lc != train_col:
                    for o in list(lc.objects):
                        if o.name.startswith(("Cam_Ring_", "Cam_Tile_", "TileCenter_")):
                            bpy.data.objects.remove(o, do_unlink=True)

        # -------------- RS Studio camera factory --------------
        next_idx = next_rs_index(rs_prefix)

        def spawn_rs_cam(
            alias: str,
            location: Vector,
            target: Vector,
            lens_mm: float,
            stadium_sub: str,                     # "ring" | "tile"
            extra_meta: dict | None = None
        ) -> bpy.types.Object:
            """Create a train-split RS camera and its marker directly under Train collection."""
            nonlocal next_idx
            cam_name = f"{rs_prefix}_{next_idx}_train"

            cam_data = bpy.data.cameras.new(cam_name + "_data")
            cam_obj  = bpy.data.objects.new(cam_name, cam_data)

            # link ONLY to Train collection (no sub-collections)
            link_once(train_col, cam_obj)

            # optics by lens (mm)
            cam_data.sensor_fit   = 'HORIZONTAL'
            cam_data.sensor_width = s.sensor_width_mm
            cam_data.clip_start   = s.stadium_clip_start
            cam_data.clip_end     = s.stadium_clip_end
            cam_data.lens         = float(lens_mm)

            # pose
            cam_obj.location = Vector(location)
            set_look_at(cam_obj, target)

            # tags
            cam_obj["rs_group"] = "stadium"
            cam_obj["rs_sub"]   = stadium_sub
            cam_obj["rs_alias"] = alias
            if extra_meta:
                for k, v in extra_meta.items():
                    cam_obj[k] = v

            # marker lives in Train collection as well
            _ensure_marker(cam_obj, SPLIT_COLORS["train"], train_col)

            next_idx += 1
            return cam_obj

        # ==================== RING ====================
        k = max(1, int(s.ring_k))
        for i in range(k):
            ang = 2.0 * math.pi * (i / k)
            x = s.ring_radius_m * math.cos(ang)
            y = s.ring_radius_m * math.sin(ang)
            z = s.ring_height_m
            alias = f"Cam_Ring_{i:03d}"
            spawn_rs_cam(
                alias=alias,
                location=Vector((x, y, z)),
                target=Vector((0.0, 0.0, 0.0)),
                lens_mm=s.ring_lens_mm,
                stadium_sub="ring",
            )

        # ==================== TILES (row-stagger along sideline) ====================
        NX = max(1, int(s.tiles_nx))
        NY = max(1, int(s.tiles_ny))
        dx = s.pitch_length_m / NX
        dy = s.pitch_width_m  / NY
        x0 = -s.pitch_length_m * 0.5 + dx * 0.5
        y0 = -s.pitch_width_m  * 0.5 + dy * 0.5

        left_y  = -s.pitch_width_m * 0.5 - s.sideline_out_offset_m
        right_y =  s.pitch_width_m  * 0.5 + s.sideline_out_offset_m
        m_total = max(1, int(s.tile_cams_per_zone))

        # row-stagger to avoid overlaps across rows (0 disables)
        row_stagger_ratio = getattr(s, "tile_row_stagger_ratio", 0.33)

        for r in range(NY):
            for c in range(NX):
                cx = x0 + c * dx
                cy = y0 + r * dy
                center = Vector((cx, cy, 0.0))

                # helper empty also lives directly under Train
                empty = bpy.data.objects.new(f"TileCenter_r{r}_c{c}", None)
                link_once(train_col, empty)
                empty.location = center

                # base X with row-stagger, symmetric around middle row
                base_x = cx + ((r - (NY - 1) * 0.5) * (dx * row_stagger_ratio))

                def place_side(side: str, count: int, y_pos: float):
                    if count <= 0:
                        return
                    offsets = [(k - (count - 1) * 0.5) * (dx * s.tile_multi_dx_ratio) for k in range(count)]
                    for idx_local, off in enumerate(offsets, start=1):
                        suffix = "" if count == 1 else str(idx_local)
                        alias  = f"Cam_Tile_r{r}_c{c}_{side}{suffix}"
                        spawn_rs_cam(
                            alias=alias,
                            location=Vector((base_x + off, y_pos, s.tile_cam_height_m)),
                            target=center,
                            lens_mm=s.tile_lens_mm,
                            stadium_sub="tile",
                            extra_meta={"rs_tile_r": r, "rs_tile_c": c, "rs_side": side, "rs_idx_in_tile": idx_local},
                        )

                if s.tile_side_mode == "NEAREST":
                    distL = abs(cy - (-s.pitch_width_m * 0.5))
                    distR = abs(cy - ( s.pitch_width_m * 0.5))
                    if distL <= distR:
                        place_side("L", m_total, left_y)
                    else:
                        place_side("R", m_total, right_y)
                else:
                    left_count  = m_total // 2
                    right_count = m_total // 2
                    if m_total % 2 == 1:
                        distL = abs(cy - (-s.pitch_width_m * 0.5))
                        distR = abs(cy - ( s.pitch_width_m * 0.5))
                        if distL <= distR: left_count += 1
                        else:              right_count += 1
                    place_side("L", left_count,  left_y)
                    place_side("R", right_count, right_y)
        s.cameras_generated = True
        self.report({'INFO'}, "Stadium RS cameras generated under Train (no sub-collections)")
        return {"FINISHED"}






# --------------------------------------------------------------------------- #
# Register new operator                                                       #
# --------------------------------------------------------------------------- #

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
    RS_OT_ImportCameras,    
    RS_OT_OrganizeScan,
    RS_OT_GenerateStadiumCameras,
    RS_OT_ClearStadiumCameras, 
]
