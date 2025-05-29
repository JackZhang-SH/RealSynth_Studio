# ui.py
"""
Draws the add-on UI in 3D-View > N-panel (“RS Studio” tab).
"""
from __future__ import annotations

import bpy

from .operators import RSDatasetSettings, RS_OT_CancelGeneration, RS_OT_GenerateDataset

# gather classes for __init__.py
CLASSES: list[type] = []


# --------------------------------------------------------------------------- #
# Main panel
# --------------------------------------------------------------------------- #
class RS_PT_MainPanel(bpy.types.Panel):
    bl_idname = "RS_PT_main"
    bl_label = "RealSynth Dataset Studio"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RS Studio"

    def draw(self, context):
        layout = self.layout
        s = context.scene.rs_settings

        # I/O + parameters ------------------------------------------------- #
        layout.prop(s, "output_dir")
        layout.prop(s, "start_frame")
        layout.prop(s, "end_frame")
        layout.prop(s, "images_per_frame")
        layout.prop(s, "radius")
        layout.prop(s, "target_point")

        if s.is_running:
            # Blender ≥ 4.0: real progress widget
            if hasattr(layout, "progress"):
                layout.progress(
                    factor=s.progress,
                    type="BAR",
                    text=f"{int(s.progress * 100):3d}%"
                )
            else:
                # Fallback for pre-4.0 builds: show the factor as a slider
                layout.prop(s, "progress", slider=True, text="Progress")

            layout.operator("rs.cancel_generation", icon="CANCEL")
        else:
            layout.operator("rs.generate_dataset", icon="RENDER_ANIMATION")


CLASSES.append(RS_PT_MainPanel)
