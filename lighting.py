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
import bmesh
import random

# ===== collection helper =============================
_LIGHTS_COLLECTION = "RS_Lighting"       

def _ensure_lights_collection() -> bpy.types.Collection:
    """Return the dedicated collection that stores every light created by RS-Studio."""
    return _ensure_collection(_LIGHTS_COLLECTION)
def _clear_lights_collection() -> None:
    """
    Remove every object inside RS_Lighting collection
    (called each time before new lights are built).
    """
    col = bpy.data.collections.get(_LIGHTS_COLLECTION)
    if not col:
        return
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

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
from bpy.types import RenderSettings             

def _ensure_preview_engine(scene: bpy.types.Scene) -> None:
    """
    If the current engine is Workbench, switch to Eevee/Eevee-Next so
    Rendered viewport works correctly.
    """
    if scene.render.engine != 'BLENDER_WORKBENCH':
        return                                     


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
    # ───────── Indoor controls ─────────
    indoor_key_strength: FloatProperty(      # type: ignore  
        name="Key",
        min=0.0, max=2.0, default=1.0,
        description="Key-light multiplier"
    )
    indoor_fill_strength: FloatProperty(     # type: ignore 
        name="Fill",
        min=0.0, max=1.0, default=0.4,
        description="Fill-light multiplier"
    )
    indoor_rim_strength: FloatProperty(     # type: ignore  
        name="Rim",
        min=0.0, max=1.5, default=0.6,
        description="Rim-light multiplier"
    )
    # 2) Weather
    weather: EnumProperty(
        name="Weather",
        items=[                                 
            ("SUNNY", "Sunny (Clear Sky)",  ""),
            ("RAIN",  "Rain",                ""),
            ("FOG",   "Fog / Haze",          ""),
        ],
        default="SUNNY",
    )  # type: ignore
    sun_intensity: FloatProperty(                              # type: ignore
        name="Sun Intensity",
        min=0.0, max=2.0, default=1.0,
        description="Multiply physical sunlight brightness",
    )
    cloudiness: FloatProperty(options={"HIDDEN"})              # type: ignore
    rain_intensity: FloatProperty(name="Rain",        min=0.0, max=1.0, default=0.4)  # type: ignore
    # visibility (meteorological range, metres)
    fog_visibility: FloatProperty(
        name="Visibility (m)",
        min=10.0, max=5_000.0, default=500.0,
        description="Human-eye visual range; density is solved from it",
    )  # type: ignore
    # keep the old field as a hidden legacy fallback
    fog_density: FloatProperty(options={"HIDDEN"}) # type: ignore

    # 3) Explicit Date · Time · Lat · Lon  (sun angles -- TODO)
    date:      StringProperty(name="Date (YYYY-MM-DD)", default="2025-06-15")             # type: ignore
    hour:      FloatProperty(name="Hour", min=0.0, max=23.99, default=12.0,
                             subtype='TIME')                                              # type: ignore
    latitude:  FloatProperty(
        name="Latitude (°)",  default=30.0,
        min=-90.0, max=90.0         
    )  # type: ignore
    longitude: FloatProperty(
        name="Longitude (°)", default=120.0,
        min=-180.0, max=180.0         
    )  # type: ignore

# ----------------------------------------------------------------- Helpers #
# ---- object management --------------------------------------------------- #
def _ensure_light(name: str, kind: str) -> bpy.types.Object:
    """
    Get (or create) a Light named *name* of Blender type *kind* and make sure
    it lives in the dedicated RS_Lighting collection (only once).
    """
    obj = bpy.data.objects.get(name)

    if obj is None or obj.type != 'LIGHT':
        # -------- create fresh light --------
        data = bpy.data.lights.new(name, type=kind)
        obj  = bpy.data.objects.new(name, data)

    # -------- guarantee membership --------
    col = _ensure_lights_collection()
    if col not in obj.users_collection:
        col.objects.link(obj)

    # unlink † this light from any other collection to avoid duplicates
    for c in list(obj.users_collection):
        if c != col:
            c.objects.unlink(obj)

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
    az_deg  = get_azimuth (lat, lon, dt_utc)     
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
    # ---------- build local datetime then convert to UTC -------------
    h = int(hour_f)
    m = int((hour_f - h) * 60)
    s = int(round(((hour_f - h) * 60 - m) * 60))


    offset_min   = int(round(lon_deg * 4))
    tz_local     = _dt.timezone(_dt.timedelta(minutes=offset_min))
    dt_local     = _dt.datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=h, minute=m, second=s, tzinfo=tz_local)


    dt_utc = dt_local.astimezone(_dt.timezone.utc)


    if _HAS_PYSOLAR:
        alt, az = _alt_az_pysolar(dt_utc, lat_deg, lon_deg)
    else:
        alt, az = _alt_az_noaa  (dt_utc, lat_deg, lon_deg)


    if alt <= math.radians(-0.5):
        return None, alt


    x = math.cos(alt) * math.sin(az)
    y = math.cos(alt) * math.cos(az)
    z = math.sin(alt)
    return Vector((x, y, z)).normalized(), alt

# === Collection helper =====================================================
def _ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col
# -------------------------- update _setup_sun ------------------------------

def _setup_sun(scene: bpy.types.Scene, cfg: RSLightingSettings):
    sun = _ensure_light("RS_Sun", "SUN")
    dir_vec, alt = _calc_sun_direction(
        cfg.date, cfg.hour, cfg.latitude, cfg.longitude
    )

    if dir_vec is None:                       
        sun.hide_render = sun.hide_viewport = True
        sun.data.energy = 0.0
        return

    sun.hide_render = sun.hide_viewport = False

    base_energy = 100 * max(0.0, math.sin(alt))   # ≈ 5 klx @ zenith

    # γ-curve: perceptually more linear (γ = 2.0)
    gamma       = 1.0
    scale       = cfg.sun_intensity ** gamma
    sun.data.energy = base_energy * scale

    if cfg.weather == "RAIN":                       
        sun.data.energy *= (1.0 - 0.8 * cfg.rain_intensity)

    sun.data.angle = math.radians(0.53)
    sun.rotation_euler = (-dir_vec).to_track_quat('-Z', 'Y').to_euler()
    sun.location = dir_vec * 10.0

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
    # heavy cloud → darker sky
    bg.inputs["Strength"].default_value = 1.0 - 0.5 * (cloudiness ** 1.5)

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
# ──────────────────────────────────────────────────────────────
def _aim_at(obj: bpy.types.Object, target: Vector) -> None:
    """
    Rotate *obj* so its −Z 轴（Blender 灯光默认前向）指向 *target*。
    """
    direction = target - obj.location
    if direction.length_squared == 0:
        return                                              # already at target
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
# ──────────────────────────────────────────────────────────────
def _setup_indoor(scene: bpy.types.Scene, cfg: RSLightingSettings):

    for n in ("RS_Sun", "RS_In_AreaCeiling", "RS_In_PointFill"):
        _remove_light(n)
    _clear_sky(scene)
    _clear_rain(scene)
    _clear_fog(scene)
    target = Vector((0.0, 0.0, 0.0))

    world = _ensure_world(scene)
    _clear_world_nodes(world)
    nt = world.node_tree
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.04, 0.04, 0.04, 1.0)
    bg.inputs["Strength"].default_value = 0.8    
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # ───────── Key ─────────
    key = _ensure_light("RS_In_Key", "AREA")
    key.data.shape = 'RECTANGLE'
    key.data.size  = 2.5
    key.data.size_y = 3.5
    key.location = (-4.0, -3.0, 3.0)
    _aim_at(key, target)  
    key.data.color = (1.0, 0.92, 0.88)          
    key.data.energy = 600.0 * cfg.indoor_key_strength

    # ───────── Fill  ─────────
    fill = _ensure_light("RS_In_Fill", "AREA")
    fill.data.shape = 'RECTANGLE'
    fill.data.size = 2.0
    fill.data.size_y = 2.0
    fill.location = (3.0, -2.5, 2.2)
    _aim_at(fill, target)  
    fill.data.color = (0.85, 0.90, 1.0)          
    fill.data.energy = 300.0 * cfg.indoor_fill_strength

    # ───────── Rim / Back  ─────────
    rim = _ensure_light("RS_In_Rim", "SPOT")
    rim.location = (2.5, 3.5, 3.5)
    _aim_at(rim, target)  
    rim.data.spot_size = radians(50)
    rim.data.shadow_soft_size = 0.4
    rim.data.energy = 500.0 * cfg.indoor_rim_strength


# ---- Weather stubs (future work) ---------------------------------------- #
# ───────── Fog helpers ─────────────────────────────────────────────────────
_KOSCHMIEDER = 3.912        # MOR ≈ 3.912 / β    (β = extinction coefficient)

def _clear_fog(scene):
    world = _ensure_world(scene)
    nt = world.node_tree
    for n in list(nt.nodes):
        if n.name.startswith("RS_Fog"):
            nt.nodes.remove(n)
    for l in list(nt.links):
        if l.is_valid and l.to_socket.identifier == "Volume":
            nt.links.remove(l)

def _setup_fog(scene, cfg: RSLightingSettings):
    """Volumetric haze:
       • Density solved from visibility (MOR) via Koschmieder
       • Exponentially decays with height
       • Subtle 3-D noise for realism"""
    vis = max(cfg.fog_visibility, 1.0)          # avoid ÷0
    base_density = _KOSCHMIEDER / vis          # m-¹   (works fine for BU≈m)

    _clear_fog(scene)
    if base_density < 1e-4:                     # almost clear – skip
        return

    world = _ensure_world(scene)
    nt = world.node_tree

    # -------------------------------- nodes --------------------------------
    # Value ▸ base_density  ┐
    val = nt.nodes.new("ShaderNodeValue");         val.name = "RS_FogBase"
    val.outputs[0].default_value = base_density

    # Texture Coordinate ▸ Generated ▸ Separate XYZ (take Z) ▸ Map Range
    tex = nt.nodes.new("ShaderNodeTexCoord");      tex.name = "RS_FogTC"
    sep = nt.nodes.new("ShaderNodeSeparateXYZ");   sep.name = "RS_FogSep"
    mrg = nt.nodes.new("ShaderNodeMapRange");      mrg.name = "RS_FogHeight"
    mrg.inputs["From Min"].default_value = 0.0     # ground
    mrg.inputs["From Max"].default_value = 20.0    # 20 m ⇒ fully clear
    mrg.inputs["To Min"].default_value   = 1.0
    mrg.inputs["To Max"].default_value   = 0.0
    nt.links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"],        mrg.inputs[0])

    # 3-D Noise
    # Blender 4.5+ has ShaderNodeVolumeNoise; earlier builds do not.
    try:
        noise = nt.nodes.new("ShaderNodeVolumeNoise")   # preferred
    except (RuntimeError, TypeError, ValueError):
        # Node type not found → fall back to regular 3-D TexNoise
        noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.name = "RS_FogNoise"
    noise.inputs["Scale"].default_value   = 3.0
    noise.inputs["Detail"].default_value  = 2.0

    # Math Multiply ×3  (base × height × noise) → Density
    mul1 = nt.nodes.new("ShaderNodeMath");  mul1.operation = 'MULTIPLY'
    mul2 = nt.nodes.new("ShaderNodeMath");  mul2.operation = 'MULTIPLY'
    nt.links.new(val.outputs[0],    mul1.inputs[0])
    nt.links.new(mrg.outputs[0],    mul1.inputs[1])
    nt.links.new(mul1.outputs[0],   mul2.inputs[0])
    nt.links.new(noise.outputs.get("Fac", 0), mul2.inputs[1])

    # Principled Volume (colour slightly bluish-grey)
    fog = nt.nodes.new("ShaderNodeVolumePrincipled"); fog.name = "RS_Fog"
    fog.inputs["Color"].default_value   = (0.6, 0.7, 0.8, 1.0)
    nt.links.new(mul2.outputs[0], fog.inputs["Density"])

    # World Output – Volume socket
    out = next((n for n in nt.nodes if n.type == 'OUTPUT_WORLD'), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(fog.outputs["Volume"], out.inputs["Volume"])


# === Particle-based rain ===================================================
def _clear_rain(scene):
    col = bpy.data.collections.get("RS_Rain")
    if col:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(col, do_unlink=True)
    for obj_name in ("RS_Raindrop", "RS_RainEmitter"):
        obj = bpy.data.objects.get(obj_name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)


def _setup_rain(scene, intensity: float):
    """
    Creates a simple but convincing rain system:
    * A hidden raindrop instance object (elongated UV-sphere, glass material)
    * A large overhead emitter plane with a particle system
    """
    if intensity <= 0.0:
        _clear_rain(scene)
        return

    # 1. ensure collection
    col = _ensure_collection("RS_Rain")

    # 2. raindrop object (instanced by particles)
    drop = bpy.data.objects.get("RS_Raindrop")
    if drop is None:
        mesh = bpy.data.meshes.new("RS_RaindropMesh")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=6, v_segments=3, radius=0.02)
        bm.to_mesh(mesh)
        bm.free()

        drop = bpy.data.objects.new("RS_Raindrop", mesh)
        drop.scale = (0.3, 0.3, 2.0)                      # streak shape
        col.objects.link(drop)

        # water material
        mat = bpy.data.materials.new("RS_RainWater")
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        # Blender 4.4: Transmission / Roughness sockets renamed.
        tx_sock = (bsdf.inputs.get("Transmission")
                   or bsdf.inputs.get("Transmission Weight"))
        if tx_sock:
            tx_sock.default_value = 1.0

        rough_sock = (bsdf.inputs.get("Roughness")
                      or bsdf.inputs.get("Roughness Factor"))
        if rough_sock:
            rough_sock.default_value = 0.05
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        drop.data.materials.append(mat)

        drop.hide_render   = False          # **must be False or nothing shows up**
        drop.hide_viewport = True           # still invisible in viewport

    else:
        if col not in drop.users_collection:
            col.objects.link(drop)


    # 3. emitter plane
    emitter = bpy.data.objects.get("RS_RainEmitter")
    if emitter is None:
        mesh = bpy.data.meshes.new("RS_RainEmitterMesh")
        mesh.from_pydata(
            [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)]
        )
        emitter = bpy.data.objects.new("RS_RainEmitter", mesh)
        scene.collection.objects.link(emitter)
    emitter.scale = (30, 30, 1)
    emitter.location = (0, 0, 20)
    if col not in emitter.users_collection:
        col.objects.link(emitter)

    # 4. particle system (reuse if already present)
    if emitter.particle_systems:
        ps = emitter.particle_systems[0]
    else:
        ps = emitter.modifiers.new("RS_RainPS", type='PARTICLE_SYSTEM').particle_system

    part = ps.settings
    part.count = int(40_000 * intensity)               # denser = more obvious
    part.frame_start = 1
    part.frame_end = 250
    part.lifetime = 250
    part.emit_from = 'FACE'
    part.physics_type = 'NEWTON'
    part.normal_factor = 0.0
    # ---------------------------------------------------------------
    # Hide emitter plane in renders (API changed in 4.4)
    if hasattr(part, "use_render_emitter"):       # ≤4.3
        part.use_render_emitter = False
    elif hasattr(ps, "show_emitter"):             # ≥4.4 (property on ParticleSystem)
        ps.show_emitter = False                   # hide in both viewport & render
    if hasattr(part, "render_type"):
        part.render_type = 'OBJECT'
    else:
        part.render_as   = 'OBJECT'
    part.instance_object = drop
    part.particle_size = 0.05
    if hasattr(part, "mass"):
        part.mass = 0.01
    # --- gravity ---------------------------------------------------------
    if hasattr(part, "gravity"):
        # Blender ≤4.3
        part.gravity = (0, 0, -9.81)
    else:
        # Blender 4.4 → 通过 effector 权重
        part.effector_weights.gravity = 1.0

    # --- removed in 4.4 ---------------------------------------------------
    if hasattr(part, "timestep"):
        part.timestep = 0.04          # <=4.3 only

    # --- random velocity --------------------------------------------------
    if hasattr(part, "velocity_factor_random"):
        part.velocity_factor_random = 0.3
    elif hasattr(part, "factor_random"):          # 4.4 rename
        part.factor_random = 0.3

    part.use_rotations = True                          # orient to velocity
    if hasattr(part, "rotation_mode"):                 # ≤ 4.3
        part.rotation_mode = 'VEL'
    elif hasattr(part, "angular_velocity_mode"):       # ≥ 4.4 rename
        part.angular_velocity_mode = 'VELOCITY'


# === Dispatcher – apply() ==================================================
def apply(scene: bpy.types.Scene, cfg: RSLightingSettings):
    _ensure_lights_collection()   
    _clear_lights_collection()  
    if cfg.location == "INDOOR":
        _setup_indoor(scene, cfg)
        _clear_rain(scene); _clear_fog(scene)
        return   
    if cfg.location == "OUTDOOR":
        sky_cloud = 0.0 if cfg.weather == "SUNNY" else 0.9
        _setup_sky(scene, sky_cloud)
        _setup_sun(scene, cfg)
    else:
        _setup_indoor(scene)

    if cfg.weather == "RAIN":
        _setup_rain(scene, cfg.rain_intensity)
    else:
        _clear_rain(scene)

    if cfg.weather == "FOG":
        _setup_fog(scene, cfg)
    else:
        _clear_fog(scene)
# ---------------------------------------------------------------- Operator #
class RS_OT_ClearLighting(bpy.types.Operator):
    """Remove every RS-Studio light object, sky, rain/fog, and reset World."""
    bl_idname = "rs.clear_lighting"
    bl_label  = "Clear RS Lighting"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, ctx):
        scene = ctx.scene

        col = bpy.data.collections.get(_LIGHTS_COLLECTION)
        if col:
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            # unlink collection from parents before removing
            for parent in bpy.data.collections:
                if col.name in parent.children:
                    parent.children.unlink(col)
            bpy.data.collections.remove(col)

        for obj in list(bpy.data.objects):
            if obj.type == 'LIGHT' and obj.name.startswith("RS_"):
                bpy.data.objects.remove(obj, do_unlink=True)


        _clear_rain(scene)
        _clear_fog(scene)
        _clear_sky(scene)                     

        self.report({'INFO'}, "RS-Studio lighting cleared")
        return {'FINISHED'}

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
CLASSES: List[Type] = [RSLightingSettings, RS_OT_ApplyLighting,RS_OT_ClearLighting]
