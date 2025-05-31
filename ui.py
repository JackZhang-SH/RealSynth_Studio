# ui.py
"""
Draws the add-on UI in 3D-View > N-panel (“RS Studio” tab).
"""
from __future__ import annotations

import bpy

from .operators import RSDatasetSettings, RS_OT_CancelGeneration, RS_OT_GenerateDataset,    RS_OT_GenerateCameras, RS_OT_ClearCameras

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

        # ① 摄像机管理区 --------------------------------------------------- #
        cam_box = layout.box()
        cam_box.label(text="Camera Rig", icon='CAMERA_DATA')
        cam_box.prop(s, "images_per_frame")            # Camera Count
        cam_box.prop(s, "radius")
        cam_box.prop(s, "target_point")
        cam_box.prop(s, "sampling_strategy")           # ← NEW
        cam_box.prop(s, "camera_source", text="Source")

        row = cam_box.row(align=True)
        row.operator("rs.generate_cameras", icon='CAMERA_DATA')
        row.operator("rs.clear_cameras",  icon='TRASH')

        layout.separator()

        # ② 数据集生成区 --------------------------------------------------- #
        ds_box = layout.box()
        ds_box.label(text="Dataset Generation", icon='RENDER_ANIMATION')
        ds_box.prop(s, "output_dir")
        ds_box.prop(s, "start_frame")
        ds_box.prop(s, "end_frame")
        if s.is_running:
            # Blender ≥ 4.0: real progress widget
            if hasattr(ds_box, "progress"):
                ds_box.progress(
                    factor=s.progress,
                    type="BAR",
                    text=f"{int(s.progress * 100):3d}%"
                )
            else:
                # Fallback for pre-4.0 builds: show the factor as a slider
                ds_box.prop(s, "progress", slider=True, text="Progress")

            ds_box.operator("rs.cancel_generation", icon="CANCEL")
        else:
            row = ds_box.row()
            row.enabled = s.cameras_generated  # ← 生成前禁用
            row.operator("rs.generate_dataset", icon="RENDER_ANIMATION")
            if not s.cameras_generated:
                row = ds_box.row()
                row.label(text="Generate cameras first", icon='INFO')


CLASSES.append(RS_PT_MainPanel)
