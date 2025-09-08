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
        # Lighting 面板中的天气-相关控件
        if s2.weather == "SUNNY":
            light.prop(s2, "sun_intensity", slider=True)
        elif s2.weather == "RAIN":
            light.prop(s2, "rain_intensity", slider=True)
        elif s2.weather == "FOG":
            light.prop(s2, "fog_visibility", slider=True)
        light.separator()

        if s2.location == "OUTDOOR":
            col = light.column(align=True)
            col.use_property_split = True

            # 日期 & 小时（小时带滑条，步长 0.25 h）
            col.prop(s2, "date", text="Date")
            col.prop(s2, "hour", text="Local Time (h)", slider=True)

            # 经纬度两行分开，滑条 ±90 / ±180
            col.prop(s2, "latitude",  text="Latitude (°)",  slider=True)
            col.prop(s2, "longitude", text="Longitude (°)", slider=True)

            # 时区提示
            tz_off = s2.longitude / 15.0
            light.label(text=f"Local TZ ≈ UTC{tz_off:+.1f} h")
        light.operator("rs.apply_lighting", icon='CHECKMARK')
        light.operator("rs.clear_lighting", icon='TRASH')
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
        # Show depth-fusion point-cloud options when exporting 3DGS (COLMAP)
        if s.export_format == "COLMAP_3DGS_MESH":  # ← NEW
            sub = ds_box.box()
            sub.label(text="Surface-Sampling Point Cloud", icon='POINTCLOUD_DATA')
            col = sub.column(align=True); col.use_property_split = True
            col.prop(s, "mesh_points_target")
            col.prop(s, "mesh_voxel")
            col.prop(s, "mesh_color_mode")
            col.separator()
            row = col.row(align=True)
            # --- AABB bounds section (OpenCV/COLMAP world) ---
            sub2 = ds_box.box()
            sub2.label(text="AABB Bounds (OpenCV/COLMAP World)", icon='MESH_CUBE')
            col2 = sub2.column(align=True); col2.use_property_split = True
            col2.prop(s, "mesh_bounds_enable")
            row = sub2.row(align=True)
            row.enabled = s.mesh_bounds_enable
            cmin = row.column(align=True); cmin.use_property_split = True
            cmin.prop(s, "mesh_bounds_min")
            cmax = row.column(align=True); cmax.use_property_split = True
            cmax.prop(s, "mesh_bounds_max")
            col.separator()
            col.label(text="Layered Sampling")
            col.prop(s, "mesh_object_gamma", slider=True)
            col.prop(s, "mesh_object_cap_ratio", slider=True)

            col.separator()
            col.label(text="Micro Thickness")
            col.prop(s, "mesh_thickness_eps")
            col.prop(s, "mesh_thickness_two_sided")
        if s.export_format == "COLMAP_3DGS_DEPTH":
            sub = ds_box.box()
            sub.label(text="Depth Back-Projection Point Cloud", icon='POINTCLOUD_DATA')
            col = sub.column(align=True); col.use_property_split = True
            col.prop(s, "depth_stride")
            col.prop(s, "pointcloud_voxel")
            col.prop(s, "pointcloud_max_points")
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
