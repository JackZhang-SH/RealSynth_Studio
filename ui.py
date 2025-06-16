# ui.py – adds “Import Cameras” section above Camera Rig
from __future__ import annotations

import bpy

from .operators import (
    RSDatasetSettings,
    RS_OT_ImportCameras,
    RS_OT_GenerateCameras,
    RS_OT_ClearCameras,
    RS_OT_SetCameraSplit,
    RS_OT_GenerateDataset,
    RS_OT_CancelGeneration,
    RS_OT_OrganizeScan,  
)

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

        # ─────────────── Import Cameras ─────────────── #
        imp = layout.box()
        imp.label(text="Import Cameras", icon='IMPORT')

        # ↓ Show required file/folder hints based on selected format
        help_text = {
            "COLMAP"    : "Select the folder containing cameras.txt & images.txt",
            "NGP"       : "Select the folder containing transforms.json",
            "TACV"      : "Select the folder containing transforms.json & transforms_test.json",
            "NERF_SYNTH": "Select the folder containing transforms_train/val/test.json",
            "CMU_PANOPTIC"  : "Select either a calibration .json file or a folder containing it",
        }.get(s.import_format, "")
        if help_text:
            imp.label(text=help_text, icon='INFO')
        imp.prop(s, "import_format", text="Format")
        imp.prop(s, "import_dir")
        imp.prop(s, "import_scale", text="Scale")   # ★ new line
        imp.operator("rs.import_cameras", icon='IMPORT')

        layout.separator()
        # ─────────────── Lighting (new) ─────────────── #
        light = layout.box()
        light.label(text="Lighting", icon='LIGHT_SUN')

        light.prop(s2 := context.scene.rs_light, "location", expand=True)
        light.prop(s2, "weather")

        # weather-specific sliders
        if s2.weather in {"CLEAR", "SCATTERED", "OVERCAST"}:
            light.prop(s2, "cloudiness", slider=True)
        elif s2.weather == "RAIN":
            light.prop(s2, "rain_intensity", slider=True)
        elif s2.weather == "FOG":
            light.prop(s2, "fog_density", slider=True)

        light.separator()

        col = light.column(align=True)
        col.use_property_split = True           # 左右分栏：标签 / 控件
        col.prop(s2, "date",       text="Date")
        col.prop(s2, "hour",       text="Hour (0-24)")

        # 清晰分组经纬度
        col.label(text="Geographic Location")
        row = col.row(align=True)
        row.prop(s2, "latitude")
        row.prop(s2, "longitude")

        # ─────────────── Post-Import Organiser ─────────────── #
        org = layout.box()
        org.label(text="Scan Post-Import", icon='FILE_REFRESH')
        row = org.row(align=True)
        row.operator("rs.organize_scan", icon='FILE_REFRESH')
        row.prop(s, "scan_fix_scale", text="Scale ↔ 1 m")
        row.prop(s, "scan_fix_material", text="Fix Material")
        # ─────────────── Camera Rig ─────────────── #
        cam_box = layout.box()
        cam_box.label(text="Camera Generator", icon='CAMERA_DATA')
        cam_box.prop(s, "images_per_frame")
        cam_box.prop(s, "sampling_strategy")
        cam_box.prop(s, "radius")
        cam_box.prop(s, "target_point")

        row = cam_box.row(align=True)
        row.operator("rs.generate_cameras", icon='CAMERA_DATA')
        row.operator("rs.clear_cameras",   icon='TRASH')

        # split buttons
        row = cam_box.row(align=True)
        row.label(text="Set split for selected:")
        btn = cam_box.row(align=True)
        op = btn.operator("rs.set_camera_split", text="Train", icon='HIDE_OFF')
        op.split = 'train'
        op = btn.operator("rs.set_camera_split", text="Valid", icon='EVENT_V')
        op.split = 'valid'
        op = btn.operator("rs.set_camera_split", text="Test",  icon='EVENT_T')
        op.split = 'test'

        layout.separator()

        # ───────────── Dataset Generation ───────────── #
        ds_box = layout.box()
        ds_box.label(text="Dataset Generation", icon='RENDER_ANIMATION')
        ds_box.prop(s, "output_dir")
        ds_box.prop(s, "start_frame")
        ds_box.prop(s, "end_frame")
        ds_box.prop(s, "export_format")
        if s.export_format == "NGP":
            ds_box.prop(s, "aabb_scale")

        if s.is_running:
            if hasattr(ds_box, "progress"):
                ds_box.progress(factor=s.progress, type="BAR",
                                text=f"{int(s.progress*100):3d}%")
            else:
                ds_box.prop(s, "progress", slider=True, text="Progress")
            ds_box.operator("rs.cancel_generation", icon="CANCEL")
        else:
            row = ds_box.row()
            row.enabled = s.cameras_generated
            row.operator("rs.generate_dataset", icon="RENDER_ANIMATION")
            if not s.cameras_generated:
                ds_box.label(text="Generate / import cameras first",
                             icon='INFO')


CLASSES.append(RS_PT_MainPanel)
