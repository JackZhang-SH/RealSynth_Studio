# importers.py  (放在 addon 根目录或 core 子包)
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import bpy
from mathutils import Matrix, Quaternion, Vector


__all__ = ["IMPORTER_REGISTRY", "BaseImporter"]


# ───────────────────── Base class ───────────────────── #
class BaseImporter(ABC):
    """负责把 *某种* 外部相机格式 → Blender 相机"""

    # 子类需重载的枚举名称，用于 UI 下拉框
    format_name: str

    # ——— 主入口 ——— #
    def __call__(
        self,
        model_dir: Path,
        *,
        name_prefix: str,
        start_index: int,
        collection: bpy.types.Collection,
        color: Tuple[float, float, float, float],
    ) -> int:
        """
        解析并将相机实例化到场景  
        返回创建的相机数量
        """
        cams = self.parse(model_dir)
        self.instantiate(
            cams, name_prefix, start_index, collection, color
        )
        return len(cams)

    # ——— 子类必须实现 ——— #
    @abstractmethod
    def parse(self, model_dir: Path) -> List[Dict]:
        """读磁盘 -> 返回 [{'R':(4,), 't':(3,), 'w':W, 'h':H, 'fx':, …}, ...]"""
        ...

    @staticmethod
    def _cv_to_blender_matrix(q: Quaternion, t: Vector) -> Matrix:
        """COLMAP world→camera 外参 [R|t]  ➜ Blender camera→world 矩阵"""
        blender2cv = Matrix((
            (1, 0, 0, 0),
            (0,-1, 0, 0),   # flip  Y
            (0, 0,-1, 0),   # flip  Z
            (0, 0, 0, 1),
        ))
        # 正确顺序：T @ R  →  [R | t]
        M_w2cv = Matrix.Translation(t) @ q.to_matrix().to_4x4()
        # 取逆得到 camera→world (CV)，再转 Blender 坐标系
        return M_w2cv.inverted() @ blender2cv

    def instantiate(
        self,
        cams: List[Dict],
        name_prefix: str,
        start_index: int,
        collection: bpy.types.Collection,
        color: Tuple[float, float, float, float],
    ):
        from .operators import _ensure_marker  # 延迟导入避免循环

        for idx, info in enumerate(cams):
            cam_name = f"{name_prefix}_{start_index + idx}_train"
            data = bpy.data.cameras.new(cam_name)
            data.type = "PERSP"
            angle_x = 2 * math.atan(info["w"] / (2 * info["fx"]))
            data.angle = angle_x
            data["fl_x"], data["fl_y"] = info["fx"], info["fy"]

            obj = bpy.data.objects.new(cam_name, data)
            q = Quaternion(info["q"])  # (w,x,y,z)
            t = Vector(info["t"])      # (x,y,z)
            obj.matrix_world = self._cv_to_blender_matrix(q, t)

            collection.objects.link(obj)
            _ensure_marker(obj, color, collection)


# ───────────────────── COLMAP 实现 ───────────────────── #
class ColmapImporter(BaseImporter):
    format_name = "COLMAP"

    # --------------- helpers --------------- #
    def _parse_cameras_txt(self, path: Path) -> Dict[int, Dict]:
        cams: Dict[int, Dict] = {}
        with path.open("r", encoding="utf8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                t = line.split()
                cid, model, w, h = int(t[0]), t[1], int(t[2]), int(t[3])
                if model == "PINHOLE":
                    fx, fy, cx, cy = map(float, t[4:8])
                elif model == "SIMPLE_PINHOLE":
                    fx = fy = float(t[4])
                    cx, cy = map(float, t[5:7])
                else:  # 粗放兜底
                    fx, fy = map(float, t[4:6])
                    cx = cy = 0.5
                cams[cid] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
        return cams

    # --------------- main --------------- #
    def parse(self, model_dir: Path) -> List[Dict]:
        if (model_dir / "cameras.txt").exists():
            root = model_dir
        elif (model_dir / "0" / "cameras.txt").exists():
            root = model_dir / "0"
        else:
            raise FileNotFoundError("cameras.txt not found")

        cams_intr = self._parse_cameras_txt(root / "cameras.txt")
        imgs_txt  = root / "images.txt"
        if not imgs_txt.exists():
            raise FileNotFoundError("images.txt not found")

        out: List[Dict] = []
        for raw in imgs_txt.read_text().splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue                     # 跳过空行 / 注释
            t = raw.split()
            if len(t) < 10:                 # 第二行 (x y X Y id ...) 或异常行
               continue                    # 直接忽略
            q = tuple(map(float, t[1:5]))  # w,x,y,z
            trans = tuple(map(float, t[5:8]))
            cam_id = int(t[8])
            intr = cams_intr.get(cam_id)
            if intr is None:
                continue
            out.append(
                dict(q=q, t=trans, **intr)
            )
        return out


# ───────────────────── Registry ───────────────────── #
IMPORTER_REGISTRY: Dict[str, BaseImporter] = {}

def _register_importer(cls: type[BaseImporter]):
    IMPORTER_REGISTRY[cls.format_name] = cls()  # 单例
    return cls

# 把 COLMAP 注册进去
_register_importer(ColmapImporter)
