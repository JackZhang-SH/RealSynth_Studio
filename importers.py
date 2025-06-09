# importers.py  (放在 addon 根目录或 core 子包)
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import bpy
from mathutils import Matrix, Quaternion, Vector


__all__ = ["IMPORTER_REGISTRY", "BaseImporter"]

# ──────────────────────────  Intrinsics helper  ────────────────────────── #
def _apply_intrinsics(
    camdat: bpy.types.Camera,
    *,
    w: int, h: int,
    fx: float, fy: float,
    cx: float, cy: float,
    dist: dict[str, float] | None = None,
    sensor_width: float = 36.0,          # 常用全幅等效
):
    """Write (fx, fy, cx, cy) ⟹ Blender camera, **preserving** values loss-lessly."""
    camdat.type = 'PERSP'
    camdat.sensor_fit = 'HORIZONTAL'          # 锁水平以免 Blender 自动缩放纵向
    camdat.sensor_width = sensor_width
    camdat.lens = fx * sensor_width / w       # 焦距(mm)

    # 校正纵向焦距
    camdat.sensor_height = sensor_width * h / w * fx / fy

    # 主点偏移 → shift
    camdat.shift_x = (cx / w) - 0.5           # +右  -左
    camdat.shift_y = 0.5 - (cy / h)           # +上  -下（OpenCV⇄Blender）

    # 持久化到自定义属性，便于导出阶段直接读取
    camdat["fl_x"], camdat["fl_y"] = fx, fy
    camdat["cx"],  camdat["cy"]    = cx, cy
    if dist:
        for k, v in dist.items():
            camdat[k] = float(v)

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
        # 1) COLMAP world → camera  (4×4)
        M_w2cv = Matrix.Translation(t) @ q.to_matrix().to_4x4()

        # 2) camera → world  (still in OpenCV axes)
        M_cv2w = M_w2cv.inverted()

        # 3) OpenCV world → Blender world  (A)
        A = Matrix((
            (1, 0,  0, 0),
            (0, 0,  1, 0),
            (0,-1,  0, 0),
            (0, 0,  0, 1),
        ))

        # 4) Blender local → OpenCV local  (T)
        T = Matrix((
            (1, 0,  0, 0),
            (0,-1,  0, 0),
            (0, 0,-1, 0),
            (0, 0,  0, 1),
        ))

        # 5) Compose:  Blender matrix_world
        return A @ M_cv2w @ T

    def instantiate(
        self,
        cams: List[Dict],
        name_prefix: str,
        start_index: int,
        collection: bpy.types.Collection,
        color: Tuple[float, float, float, float],
    ):
        from .operators import _ensure_marker                   # late-import

        s = bpy.context.scene.rs_settings
        scale = getattr(s, "import_scale", 1.0)

        for idx, info in enumerate(cams):
            name = f"{name_prefix}_{start_index+idx}_train"
            camdat = bpy.data.cameras.new(name)

            # --- 写入内参 ----------------------------------------------------
            w, h  = info["w"],  info["h"]
            fx,fy = info["fx"], info["fy"]
            cx    = info.get("cx", w*0.5)
            cy    = info.get("cy", h*0.5)
            dist  = {k: info.get(k, 0.0) for k in ("k1","k2","p1","p2")}
            _apply_intrinsics(camdat, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
                              dist=dist)

            obj = bpy.data.objects.new(name, camdat)
            q   = Quaternion(info["q"])
            t   = Vector(info["t"]) * scale
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

def _matrix_from_list(m):
    from mathutils import Matrix
    return Matrix(m)

# ───────────────────── NGP ───────────────────── #
class NGPImporter(BaseImporter):
    format_name = "NGP"

    # ---------- 解析 ---------- #
    def parse(self, model_dir: Path):
        import json
        tf_files = list(model_dir.rglob("transforms*.json"))
        if not tf_files:
            raise FileNotFoundError("No transforms*.json found")

        cams = []
        for f in tf_files:
            data = json.loads(f.read_text())
            w, h   = data["w"],   data["h"]
            fx, fy = data["fl_x"], data["fl_y"]
            cx     = data.get("cx", w * 0.5)          # ≤ v4.4 作者输出里就有 cx/cy
            cy     = data.get("cy", h * 0.5)

            split  = ("test" if "test" in f.stem else
                      "valid" if "val"  in f.stem else "train")

            for fr in data["frames"]:
                cams.append(dict(
                    mat   = fr["transform_matrix"],
                    w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
                    split = split,
                ))
        return cams

    # ---------- 实例化 ---------- #
    def instantiate(self, cams, name_prefix, start_index, *_):
        from .operators import _ensure_collections, _ensure_marker, SPLIT_COLORS
        subcols = _ensure_collections()
        scale   = bpy.context.scene.rs_settings.import_scale

        for idx, info in enumerate(cams):
            split  = info.get("split", "train")
            col    = subcols[split]
            color  = SPLIT_COLORS[split]
            name   = f"{name_prefix}_{start_index+idx}_{split}"

            camdat = bpy.data.cameras.new(name)
            _apply_intrinsics(
                camdat,
                w = info["w"],   h = info["h"],
                fx= info["fx"], fy= info["fy"],
                cx= info["cx"], cy= info["cy"],
            )

            obj = bpy.data.objects.new(name, camdat)
            mat = _matrix_from_list(info["mat"])
            if scale != 1.0:
                mat.translation *= scale
            obj.matrix_world = mat

            col.objects.link(obj)
            _ensure_marker(obj, color, col)

        return len(cams)


class TACVImporter(BaseImporter):
    format_name = "TACV"

    # ---------- 解析 ---------- #
    def parse(self, model_dir: Path):
        import json
        cams = []

        def _add(fpath, split):
            if not fpath.exists():
                return
            for item in json.loads(fpath.read_text()):
                w, h   = item["w"],   item["h"]
                fx, fy = item["fl_x"], item["fl_y"]
                cx     = item.get("cx", w * 0.5)
                cy     = item.get("cy", h * 0.5)

                cams.append(dict(
                    mat   = item["transform_matrix"],
                    w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy,
                    split = split,
                ))

        _add(model_dir / "transforms.json",       "train")
        _add(model_dir / "transforms_valid.json", "valid")
        _add(model_dir / "transforms_test.json",  "test")
        if not cams:
            raise FileNotFoundError("No transforms*.json found")
        return cams

    # ---------- 实例化 ---------- #
    instantiate = NGPImporter.instantiate      # 逻辑完全一致

# ───────────────────── NeRF Synthetic ───────────────────── #
class NeRFSynthImporter(BaseImporter):
    format_name = "NERF_SYNTH"

    def parse(self, model_dir: Path):
        import json, re
        cams = []
        for tf in model_dir.rglob("transforms_*.json"):
            m = re.match(r"^transforms_(train|val|test)\.json$", tf.name)
            if not m:
                continue
            split = {"train":"train", "val":"valid", "test":"test"}[m.group(1)]
            data  = json.loads(tf.read_text())
            w, h = data["w"], data["h"]
            fx, fy = data["fl_x"], data["fl_y"]
            for fr in data["frames"]:
                cams.append(dict(
                    mat   = fr["transform_matrix"],
                    w=w, h=h, fx=fx, fy=fy,
                    split = split,
                ))
        if not cams:
            raise FileNotFoundError("No transforms_*.json found")
        return cams

    def instantiate(self, cams, name_prefix, start_index, *_):
        from .operators import _ensure_collections, _ensure_marker, SPLIT_COLORS
        subcols = _ensure_collections()
        scale   = bpy.context.scene.rs_settings.import_scale
        for idx, info in enumerate(cams):
            split = info["split"]      # train / valid / test
            col   = subcols[split]
            color = SPLIT_COLORS[split]
            name  = f"{name_prefix}_{start_index+idx}_{split}"

            data = bpy.data.cameras.new(name)
            data.type = 'PERSP'
            data.angle = 2 * math.atan(info["w"] / (2 * info["fx"]))
            data["fl_x"], data["fl_y"] = info["fx"], info["fy"]

            obj = bpy.data.objects.new(name, data)
            mat = _matrix_from_list(info["mat"])
            if scale != 1.0:
                mat.translation *= scale
            obj.matrix_world = mat

            col.objects.link(obj)
            _ensure_marker(obj, color, col)
            
class CMUPanopticImporter(BaseImporter):
    format_name = "CMU_PANOPTIC"

    def parse(self, model_dir: Path):
        import json
        from mathutils import Matrix, Vector

        # --- 找到 json ---
        if model_dir.is_file():
            json_path = model_dir
        else:
            cand = [p for p in model_dir.iterdir() if p.suffix == ".json"]
            if not cand:
                raise FileNotFoundError("No *.json calibration file found")
            json_path = cand[0]

        calib = json.loads(json_path.read_text())

        cams = []
        for cam in calib["cameras"]:
            w, h = cam["resolution"]
            K    = cam["K"]                        # 3×3
            fx, fy = K[0][0], K[1][1]
            cx, cy = K[0][2], K[1][2]

            R_mat  = Matrix(cam["R"])
            q      = R_mat.to_quaternion()         # (w,x,y,z)

            t_arr  = cam["t"]                      # [[x],[y],[z]]
            t_vec  = Vector((t_arr[0][0], t_arr[1][0], t_arr[2][0]))

            cams.append(dict(
                q=q, t=t_vec,
                w=w, h=h,
                fx=fx, fy=fy, cx=cx, cy=cy,
            ))
        if not cams:
            raise ValueError("No camera entries found in calibration JSON")
        return cams

# ───────────────────── Registry ───────────────────── #
IMPORTER_REGISTRY: Dict[str, BaseImporter] = {}

def _register_importer(cls: type[BaseImporter]):
    IMPORTER_REGISTRY[cls.format_name] = cls()  # 单例
    return cls

# 把 COLMAP 注册进去
_register_importer(ColmapImporter)
_register_importer(NeRFSynthImporter)
_register_importer(NGPImporter)
_register_importer(TACVImporter)
_register_importer(CMUPanopticImporter)