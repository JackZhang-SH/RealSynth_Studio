# ui.py
"""
Draws the add-on UI in 3D-View > N-panel (“RS Studio” tab).
"""
from __future__ import annotations

import bpy

from .operators import RSDatasetSettings, RS_OT_CancelGeneration, RS_OT_GenerateDataset,    RS_OT_GenerateCameras, RS_OT_ClearCameras, RS_OT_SetCameraSplit

# gather classes for __init__.py
CLASSES: list[type] = []


class RS_PT_MainPanel(bpy.types.Panel):
    bl_idname = "RS_PT_main"
    bl_label = "RealSynth Dataset Studio"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RS Studio"

    def draw(self, context):
        layout = self.layout
        s = context.scene.rs_settings

        # ─────────────────── Camera Rig ─────────────────── #
        cam_box = layout.box()
        cam_box.label(text="Camera Rig", icon='CAMERA_DATA')
        cam_box.prop(s, "images_per_frame")
        cam_box.prop(s, "radius")
        cam_box.prop(s, "target_point")
        cam_box.prop(s, "sampling_strategy")
        cam_box.prop(s, "camera_source", text="Source")

        row = cam_box.row(align=True)
        row.operator("rs.generate_cameras", icon='CAMERA_DATA')
        row.operator("rs.clear_cameras", icon='TRASH')

        # ---------- 新增：数据集划分切换按钮 ---------- #
        row = cam_box.row(align=True)
        row.label(text="Set split for selected:")
        btn_row = cam_box.row(align=True)
        op = btn_row.operator("rs.set_camera_split", text="Train", icon='HIDE_OFF')
        op.split = 'train'
        op = btn_row.operator("rs.set_camera_split", text="Valid", icon='EVENT_V')
        op.split = 'valid'
        op = btn_row.operator("rs.set_camera_split", text="Test", icon='EVENT_T')
        op.split = 'test'

        layout.separator()

        # ───────────────── Dataset Generation ───────────────── #
        ds_box = layout.box()
        ds_box.label(text="Dataset Generation", icon='RENDER_ANIMATION')
        ds_box.prop(s, "output_dir")
        ds_box.prop(s, "start_frame")
        ds_box.prop(s, "end_frame")
        if s.is_running:
            if hasattr(ds_box, "progress"):
                ds_box.progress(
                    factor=s.progress,
                    type="BAR",
                    text=f"{int(s.progress * 100):3d}%"
                )
            else:
                ds_box.prop(s, "progress", slider=True, text="Progress")

            ds_box.operator("rs.cancel_generation", icon="CANCEL")
        else:
            row = ds_box.row()
            row.enabled = s.cameras_generated
            row.operator("rs.generate_dataset", icon="RENDER_ANIMATION")
            if not s.cameras_generated:
                row = ds_box.row()
                row.label(text="Generate cameras first", icon='INFO')


CLASSES.append(RS_PT_MainPanel)