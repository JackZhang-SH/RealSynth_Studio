"""
Lighting subsystem for RealSynth Studio
--------------------------------------

* PropertyGroup: RSLightingSettings   (already used by the UI)
* Operator    : RS_OT_ApplyLighting   (called by UI button)
* apply()     : high-level entry – decides which helpers to run

Design goals
------------
1. Switching “Indoor ↔ Outdoor” never leaves duplicate lamps/sky.
2. Helpers are small & composable, so future weather logic can graft on.
"""

from __future__ import annotations
from typing import List, Type
import bpy
from math import radians
from bpy.props import EnumProperty, FloatProperty, StringProperty
from typing import Iterable
import datetime as _dt
import math
from mathutils import Vector
try:
    from pysolar.solar import get_altitude, get_azimuth
    _HAS_PYSOLAR = True
    print("using pysolar ")
except ModuleNotFoundError:
    _HAS_PYSOLAR = False
 # ----------------------------------------------------------------- Helpers #
# ---- viewport shading ---------------------------------------------------- #
def _switch_viewport_to_rendered() -> None:
    """
    Force every 3D-view into 'RENDERED' shading and enable
    scene lights / scene world so the new lighting is visible.
    """
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type != 'VIEW_3D':
                    continue
                space.shading.type = 'RENDERED'
                space.shading.use_scene_lights = True
                space.shading.use_scene_world  = True
# ---- render-engine guard (NEW) ------------------------------------------- #
from bpy.types import RenderSettings               # 放在文件顶部已有 import 后

def _ensure_preview_engine(scene: bpy.types.Scene) -> None:
    """
    If the current engine is Workbench, switch to Eevee/Eevee-Next so
    Rendered 视图能立刻看到灯光。
    """
    if scene.render.engine != 'BLENDER_WORKBENCH':
        return                                      # 用户本来就用 Eevee/Cycles

    # 优先用 Eevee-Next；若旧版 Blender 没有，则退回 Eevee
    engines = {e.identifier
               for e in RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT"
        if "BLENDER_EEVEE_NEXT" in engines
        else "BLENDER_EEVEE"
    )
# ------------------------------------------------------------------- PG --- #
class RSLightingSettings(bpy.types.PropertyGroup):
    # 1) Location (indoor ↔ outdoor)
    location: EnumProperty(
        name="Location",
        items=[("INDOOR", "Indoor (no sun)", ""),
               ("OUTDOOR", "Outdoor (sun & sky)", "")],
        default="OUTDOOR",
    )  # type: ignore

    # 2) Weather
    weather: EnumProperty(
        name="Weather",
        items=[("CLEAR", "Clear", ""),
               ("OVERCAST", "Overcast", ""),
               ("RAIN", "Rain", ""),
               ("FOG", "Fog / Haze", "")],
        default="CLEAR",
    )  # type: ignore

    cloudiness:     FloatProperty(name="Cloud cover", min=0.0, max=1.0, default=0.2)  # type: ignore
    rain_intensity: FloatProperty(name="Rain",        min=0.0, max=1.0, default=0.4)  # type: ignore
    fog_density:    FloatProperty(name="Fog density", min=0.0, max=0.1, default=0.02) # type: ignore

    # 3) Explicit Date · Time · Lat · Lon  (sun angles -- TODO)
    date:      StringProperty(name="Date (YYYY-MM-DD)", default="2025-06-15")             # type: ignore
    hour:      FloatProperty(name="Hour", min=0.0, max=23.99, default=12.0,
                             subtype='TIME')                                              # type: ignore
    latitude:  FloatProperty(
        name="Latitude (°)",  default=30.0,
        min=-90.0, max=90.0            # ⬅️ 新增
    )  # type: ignore
    longitude: FloatProperty(
        name="Longitude (°)", default=120.0,
        min=-180.0, max=180.0          # ⬅️ 新增
    )  # type: ignore

# ----------------------------------------------------------------- Helpers #
# ---- object management --------------------------------------------------- #
def _ensure_light(name: str, kind: str) -> bpy.types.Object:
    """
    Get existing light object *name*, or create a new one of *kind*
    ('SUN' | 'AREA' | 'POINT' …).  Returned object is always linked to
    scene.collection (top-level) so user can move it at will.
    """
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'LIGHT':
        return obj

    lite_data = bpy.data.lights.new(name, type=kind)
    obj = bpy.data.objects.new(name, lite_data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _remove_light(name: str) -> None:
    """Delete light object *name* (and its data) if present."""
    obj = bpy.data.objects.get(name)
    if obj and obj.type == 'LIGHT':
        bpy.data.objects.remove(obj, do_unlink=True)


# ---- world / sky --------------------------------------------------------- #
def _ensure_world(scene: bpy.types.Scene) -> bpy.types.World:
    if scene.world is None:
        scene.world = bpy.data.worlds.new("RS_World")
    scene.world.use_nodes = True
    return scene.world


def _clear_world_nodes(world: bpy.types.World) -> None:
    nt = world.node_tree
    nt.nodes.clear()
    nt.links.clear()

# ---------- helper: high-accuracy solar alt/az -------------
def _alt_az_pysolar(dt_utc, lat, lon):
    """Return altitude & azimuth in *radians* using pysolar"""
    alt_deg = get_altitude(lat, lon, dt_utc)
    az_deg  = get_azimuth (lat, lon, dt_utc)      # 0 ° = 北，顺时针正
    return math.radians(alt_deg), math.radians(az_deg)


def _alt_az_noaa(dt_utc, lat, lon):
    """Fallback: NOAA simplified formula, incl. Equation-of-Time."""
    n = dt_utc.timetuple().tm_yday
    B = math.radians((360/365)*(n - 81))
    eot = 9.87*math.sin(2*B) - 7.53*math.cos(B) - 1.5*math.sin(B)  # in minutes

    # local standard time → true solar time (min)
    lst_min = dt_utc.hour*60 + dt_utc.minute + dt_utc.second/60
    lst_min += lon*4                                   # 1° → 4 min
    tst = lst_min + eot
    H  = math.radians((tst/4) - 180)                   # hour angle rad

    decl = math.radians(23.44) * math.sin(math.radians(360*(n-81)/365))
    lat  = math.radians(lat)

    sin_alt = (math.sin(lat)*math.sin(decl) +
               math.cos(lat)*math.cos(decl)*math.cos(H))
    alt = math.asin(max(-1, min(1, sin_alt)))

    az = math.atan2(math.sin(H),
                    math.cos(H)*math.sin(lat) - math.tan(decl)*math.cos(lat))
    return alt, az


def _calc_sun_direction(date_str: str, hour_f: float,
                        lat_deg: float, lon_deg: float):
    """
    Returns (dir_vector, altitude_rad).
    dir_vector == None  →  sun below horizon.
    """
    # ---------- build local datetime then convert to UTC ----------
    # ① 解析本地日期时间 --------------------------------------------------
    h = int(hour_f)
    m = int((hour_f - h) * 60)
    s = int(round(((hour_f - h) * 60 - m) * 60))

    # → 本地时区 offset = 经度(°) × 4 min
    offset_min   = int(round(lon_deg * 4))
    tz_local     = _dt.timezone(_dt.timedelta(minutes=offset_min))
    dt_local     = _dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=h, minute=m, second=s, tzinfo=tz_local)

    # ② 转成 UTC（带 tzinfo）---------------------------------------------
    dt_utc = dt_local.astimezone(_dt.timezone.utc)

    # ③ 计算高度角 / 方位角 ------------------------------------------------
    if _HAS_PYSOLAR:
        alt, az = _alt_az_pysolar(dt_utc, lat_deg, lon_deg)
    else:
        alt, az = _alt_az_noaa  (dt_utc, lat_deg, lon_deg)

    # 夜晚：直接返回 None
    if alt <= math.radians(-0.5):
        return None, alt

    # 转为 Blender 向量 (+X = 东, +Y = 北, +Z = 上)
    x = math.cos(alt) * math.sin(az)
    y = math.cos(alt) * math.cos(az)
    z = math.sin(alt)
    return Vector((x, y, z)).normalized(), alt


# -------------------------- update _setup_sun ------------------------------
def _setup_sun(scene: bpy.types.Scene, cfg: RSLightingSettings):
    sun = _ensure_light("RS_Sun", "SUN")

    res = _calc_sun_direction(cfg.date, cfg.hour,
                              cfg.latitude, cfg.longitude)
    dir_vec, alt = res if isinstance(res, tuple) else (None, None)

    # ---- 夜晚 ----------------------------------------------------------
    if dir_vec is None:
        sun.hide_render   = True
        sun.hide_viewport = True
        sun.data.energy   = 0.0
        return

    sun.hide_render   = False
    sun.hide_viewport = False

    # energy curve：正午 ≈ 5 kLux，低空渐暗
    sun.data.energy = 5000 * max(0.0, math.sin(alt))
    sun.data.angle  = math.radians(0.53)   # solar disc 半角≈0.25°, 取直径 0.53°

    # 让 -Z 指向场景中心
    sun.rotation_euler = (-dir_vec).to_track_quat('-Z', 'Y').to_euler()
    sun.location       = dir_vec * 10.0          # 保持远离原点即可
    # 清除室内灯
    _remove_light("RS_In_AreaCeiling")
    _remove_light("RS_In_PointFill")
# ---------------------------------------------------------------- SET-UPS --
def _setup_sky(scene: bpy.types.Scene, cloudiness: float) -> None:
    """
    Build simple physical sky using Nishita model.
    cloudiness ∈ [0,1] remaps to SkyTexture.turbidity  ≈ (2 → 10).
    """
    world = _ensure_world(scene)
    _clear_world_nodes(world)
    nt = world.node_tree
    sky  = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = 'NISHITA'
    sky.turbidity = 2.0 + cloudiness * 8.0

    bg   = nt.nodes.new("ShaderNodeBackground")
    out  = nt.nodes.new("ShaderNodeOutputWorld")

    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def _clear_sky(scene: bpy.types.Scene) -> None:
    world = _ensure_world(scene)
    _clear_world_nodes(world)
    # Flat neutral grey background
    nt = world.node_tree
    bg  = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

def _setup_indoor(scene: bpy.types.Scene) -> None:
    """
    Basic two-light indoor rig:
        * RS_In_AreaCeiling – big soft box overhead
        * RS_In_PointFill   – small omni fill light
    """
    # remove sun / sky first
    _remove_light("RS_Sun")
    _clear_sky(scene)

    # Area light – ceiling panel
    area = _ensure_light("RS_In_AreaCeiling", "AREA")
    area.location = (0.0, 0.0, 5.0)
    area.rotation_euler = (radians(180.0), 0.0, 0.0)   # face downward
    area.data.shape = 'RECTANGLE'
    area.data.size = 6.0
    area.data.size_y = 4.0
    area.data.energy = 800.0

    # Point light – gentle fill
    pt = _ensure_light("RS_In_PointFill", "POINT")
    pt.location = (0.0, 0.0, 2.0)
    pt.data.energy = 300.0


# ---- Weather stubs (future work) ---------------------------------------- #
def _setup_rain(scene, intensity):   pass   # TODO
def _clear_rain(scene):              pass
def _setup_fog(scene, density):      pass   # TODO
def _clear_fog(scene):               pass

# ---------------------------------------------------------------- APPLY ---- #
def apply(scene: bpy.types.Scene, cfg: RSLightingSettings) -> None:
    """Entry point called by UI operator."""
    if cfg.location == "OUTDOOR":
        _setup_sky(scene, cfg.cloudiness)
        _setup_sun(scene, cfg)
    else:
        _setup_indoor(scene)

    # Weather hooks (not yet implemented)
    if cfg.weather == "RAIN":
        _setup_rain(scene, cfg.rain_intensity)
    else:
        _clear_rain(scene)

    if cfg.weather == "FOG":
        _setup_fog(scene, cfg.fog_density)
    else:
        _clear_fog(scene)


# ---------------------------------------------------------------- Operator #
class RS_OT_ApplyLighting(bpy.types.Operator):
    bl_idname = "rs.apply_lighting"
    bl_label  = "Apply Lighting"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, ctx):
        _ensure_preview_engine(ctx.scene)  
        apply(ctx.scene, ctx.scene.rs_light)
        _switch_viewport_to_rendered() 
        self.report({'INFO'}, "Lighting applied")
        return {'FINISHED'}


# ---------------------------------------------------------------- Register #
CLASSES: List[Type] = [RSLightingSettings, RS_OT_ApplyLighting]
