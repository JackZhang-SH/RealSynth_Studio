# __init__.py
"""
Blender add-on entry point – registers all classes and provides hot-reload for
faster development.
"""
from __future__ import annotations

import importlib
import logging
from typing import List, Type

import bpy
import os, sys as _sys
_libs = os.path.join(os.path.dirname(__file__), "libs")
if _libs not in _sys.path:
    _sys.path.append(_libs)
print("RealSynth Dataset Studio – modules reloaded")

# --------------------------------------------------------------------------- #
# Add-on metadata
# --------------------------------------------------------------------------- #
bl_info = {
    "name": "RealSynth Dataset Studio",
    "author": "Yunxiao Zhang (Jack)",
    "version": (2, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D View ▸ N-panel ▸ RS Studio",
    "description": "Generate NeRF / 3DGS datasets with incremental rendering",
    "category": "Object",
}

# --------------------------------------------------------------------------- #
# Logging helper
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Import sub-modules (hot reload friendly)
# --------------------------------------------------------------------------- #
SUBMODULES = ("core", "operators", "ui", "lighting")
for mod_name in SUBMODULES:
    if (mod := globals().get(mod_name)) is not None:
        importlib.reload(mod)
    else:
        globals()[mod_name] = importlib.import_module(f"{__name__}.{mod_name}")

from .operators import CLASSES as OPERATOR_CLASSES, RSDatasetSettings  # noqa: E402
from .ui import CLASSES as UI_CLASSES  # noqa: E402
from .lighting import CLASSES as LIGHTING_CLASSES    
_ALL_CLASSES: List[Type] = [*OPERATOR_CLASSES, *UI_CLASSES, *LIGHTING_CLASSES]

# --------------------------------------------------------------------------- #
# Register / Unregister
# --------------------------------------------------------------------------- #
def register() -> None:
    for cls in _ALL_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.rs_settings = bpy.props.PointerProperty(type=RSDatasetSettings)
    bpy.types.Scene.rs_light    = bpy.props.PointerProperty(type=LIGHTING_CLASSES[0])
    logger.info("RealSynth Dataset Studio registered")


def unregister() -> None:
    if hasattr(bpy.types.Scene, "rs_settings"):
        del bpy.types.Scene.rs_settings
    for cls in reversed(_ALL_CLASSES):
        bpy.utils.unregister_class(cls)
    logger.info("RealSynth Dataset Studio un-registered")
