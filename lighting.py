"""
Lighting subsystem for RealSynth Studio
=======================================

* RSLightingSettings – data-only PropertyGroup kept on Scene
* RS_OT_ApplyLighting – one-click operator
* apply(scene, cfg)  – high-level entry point (currently stubbed)

Expanding later
---------------
Fill `_setup_sun`, `_setup_weather`, `_setup_fog` … with real code or
hook into Blender’s “Sun Position” add-on.  Every helper is a NO-OP now
so nothing crashes during import.
"""
from __future__ import annotations
from typing import List, Type
import bpy
from bpy.props import (
    EnumProperty, FloatProperty, StringProperty,
)

# ------------------------------------------------------------------ PG ---- #
class RSLightingSettings(bpy.types.PropertyGroup):
    # 1) Location (indoor ↔ outdoor)
    location: EnumProperty(
        name="Location",
        items=[("INDOOR", "Indoor (no sun)", ""),
               ("OUTDOOR", "Outdoor (sun & sky)", "")],
        default="OUTDOOR",
    )# type: ignore

    # 2) Weather
    weather: EnumProperty(
        name="Weather",
        items=[("CLEAR",      "Clear",      ""),
               ("SCATTERED",  "Scattered",  ""),
               ("OVERCAST",   "Overcast",   ""),
               ("RAIN",       "Rain",       ""),
               ("FOG",        "Fog / Haze", "")],
        default="CLEAR",
    )# type: ignore

    cloudiness:     FloatProperty(name="Cloud cover", min=0.0, max=1.0, default=0.2)# type: ignore
    rain_intensity: FloatProperty(name="Rain",        min=0.0, max=1.0, default=0.4)# type: ignore
    fog_density:    FloatProperty(name="Fog density", min=0.0, max=0.1, default=0.02)# type: ignore

    # 3) Explicit Date · Time · Latitude · Longitude
    date:       StringProperty(name="Date (YYYY-MM-DD)", default="2025-06-15")# type: ignore
    hour:       FloatProperty (name="Hour", min=0.0, max=23.99, default=12.0, subtype='TIME')# type: ignore
    latitude:   FloatProperty (name="Latitude (°)",  default=30.0)   # type: ignore
    longitude:  FloatProperty (name="Longitude (°)", default=120.0)  # type: ignore


# ---------------------------------------------------------------- Operator #
class RS_OT_ApplyLighting(bpy.types.Operator):
    """Apply current lighting settings to the scene"""
    bl_idname = "rs.apply_lighting"
    bl_label  = "Apply Lighting"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, ctx):
        from . import lighting                                       # self-import
        lighting.apply(ctx.scene, ctx.scene.rs_light)
        self.report({'INFO'}, "Lighting applied")
        return {'FINISHED'}


# ---------------------------------------------------------------- Impl --- #
def apply(scene: bpy.types.Scene, cfg: RSLightingSettings) -> None:
    """High-level façade – dispatch to helper stubs (safe no-ops)."""
    if cfg.location == 'OUTDOOR':
        _setup_sky(scene, cfg.cloudiness)
        _setup_sun(scene, cfg)
    else:
        _clear_sun_sky(scene)
        _setup_indoor(scene)

    if cfg.weather == 'RAIN':
        _setup_rain(scene, cfg.rain_intensity)
    else:
        _clear_rain(scene)

    if cfg.weather == 'FOG':
        _setup_fog(scene, cfg.fog_density)
    else:
        _clear_fog(scene)

# ----------------------- helper stubs (expand later) --------------------- #
def _setup_sky(scene, cloudiness):      pass
def _setup_sun(scene, cfg):             pass
def _clear_sun_sky(scene):              pass
def _setup_indoor(scene):               pass
def _setup_rain(scene, intensity):      pass
def _clear_rain(scene):                 pass
def _setup_fog(scene, density):         pass
def _clear_fog(scene):                  pass

# ---------------------------------------------------------------- Reglist #
CLASSES: List[Type] = [
    RSLightingSettings,
    RS_OT_ApplyLighting,
]
