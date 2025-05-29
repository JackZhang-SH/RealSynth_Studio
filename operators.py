# operators.py
"""
Modal operators and PropertyGroup definitions.  The modal operator renders the
dataset incrementally so the UI stays alive.  A companion Cancel operator lets
users stop the job instantly.

Fix 2025-05-29
---------------
* keep a module-level reference to the active generator so
  RS_OT_CancelGeneration can signal it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import bpy
import bpy.path as bpath
from mathutils import Vector

from .core import DatasetGenerator, FibonacciSphereSampling

# --------------------------------------------------------------------------- #
# Global pointer to the *currently running* generator operator
# --------------------------------------------------------------------------- #
_active_generator: Optional["RS_OT_GenerateDataset"] = None


# --------------------------------------------------------------------------- #
# Scene-level settings (unchanged)
# --------------------------------------------------------------------------- #
class RSDatasetSettings(bpy.types.PropertyGroup):
    output_dir: bpy.props.StringProperty(  # type: ignore
        name="Output Directory", subtype="DIR_PATH", default="//nerf_dataset"
    )

    start_frame: bpy.props.IntProperty(name="Start Frame", min=1, default=1)  # type: ignore
    end_frame: bpy.props.IntProperty(name="End Frame", min=1, default=1)  # type: ignore

    images_per_frame: bpy.props.IntProperty(  # type: ignore
        name="Images per Frame", min=1, default=60
    )
    radius: bpy.props.FloatProperty(name="Sphere Radius", min=0.1, default=10.0)  # type: ignore
    target_point: bpy.props.FloatVectorProperty(  # type: ignore
        name="Target Point", subtype="TRANSLATION", default=(0.0, 0.0, 0.0)
    )

    progress: bpy.props.FloatProperty(  # type: ignore
        name="Progress", min=0.0, max=1.0, default=0.0, subtype="FACTOR"
    )
    is_running: bpy.props.BoolProperty(  # type: ignore
        name="Generating", default=False
    )


# --------------------------------------------------------------------------- #
# Generate-dataset modal operator
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

    # --------------------------------------------------------------------- #
    # Execute – initialise generator & timer
    # --------------------------------------------------------------------- #
    def execute(self, context):
        global _active_generator  # noqa: PLW0603

        # prevent launching twice
        if _active_generator is not None:
            self.report({"WARNING"}, "A generation job is already running")
            return {"CANCELLED"}

        s = context.scene.rs_settings
        wm = context.window_manager

        # build generator -------------------------------------------------- #
        generator = DatasetGenerator(
            camera_name=context.scene.camera.name,
            start_frame=s.start_frame,
            end_frame=s.end_frame,
            images_per_frame=s.images_per_frame,
            radius=s.radius,
            target=Vector(s.target_point),
            sampling=FibonacciSphereSampling(),
        )
        self._total_images = generator.total_images

        output_dir = Path(bpath.abspath(s.output_dir)).resolve()
        self._iterator = generator.iter_generate(output_dir)

        # UI initialisation ------------------------------------------------ #
        wm.progress_begin(0, self._total_images)
        s.is_running, s.progress = True, 0.0
        self._cancel_requested = False

        # register as active
        _active_generator = self

        # timer triggers modal every 0.1 s
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    # --------------------------------------------------------------------- #
    # Modal loop – render ONE image per TIMER tick
    # --------------------------------------------------------------------- #
    def modal(self, context, event):
        s, wm = context.scene.rs_settings, context.window_manager

        if event.type == "ESC":
            self._cancel_requested = True

        if event.type == "TIMER":
            # cancellation check
            if self._cancel_requested:
                self._finish(wm, s, cancelled=True)
                self.report({"WARNING"}, "Dataset generation cancelled")
                return {"CANCELLED"}

            # advance generator by one image
            try:
                done, total = next(self._iterator)
            except StopIteration:
                self._finish(wm, s, cancelled=False)
                self.report({"INFO"}, "Dataset generation finished ✔")
                return {"FINISHED"}
            except Exception as exc:
                self._finish(wm, s, cancelled=True)
                self.report({"ERROR"}, f"Generation failed: {exc}")
                return {"CANCELLED"}

            # update progress
            s.progress = done / total
            wm.progress_update(done)
            context.area.tag_redraw()
            bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _finish(self, wm, settings, *, cancelled: bool):
        global _active_generator  # noqa: PLW0603

        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        wm.progress_end()

        settings.is_running = False
        settings.progress = 0.0 if cancelled else 1.0

        _active_generator = None  # clear global reference

    # called by Esc key AND by Cancel button via RS_OT_CancelGeneration
    def cancel(self, context):
        self._cancel_requested = True


# --------------------------------------------------------------------------- #
# Cancel button operator
# --------------------------------------------------------------------------- #
class RS_OT_CancelGeneration(bpy.types.Operator):
    bl_idname = "rs.cancel_generation"
    bl_label = "Cancel"

    def execute(self, context):
        global _active_generator  # noqa: PLW0603

        if _active_generator is not None:
            _active_generator.cancel(context)
            return {"FINISHED"}

        self.report({"INFO"}, "No generation job is currently running")
        return {"CANCELLED"}


# --------------------------------------------------------------------------- #
# Class list for registration
# --------------------------------------------------------------------------- #
CLASSES = [RSDatasetSettings, RS_OT_GenerateDataset, RS_OT_CancelGeneration]
