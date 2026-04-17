# File: config_types_utils.py
# Summary:
#   Dataclasses (Camera*, CanonicalModel, CameraDetection, ObjectTrack, PoseEstimate),
#   intrinsics/extrinsics/frame loaders, GLB normalize+decimate (target subset only),
#   geometry/transform/projection utils, mesh raster + silhouette overlay,
#   comparison grid, JSON/NPZ/text/posed-GLB exporters.
#   CPU-only safe imports (open3d/scipy/torch optional).
# Run:
#   (직접 실행 X) — pipeline_core.py / cli.py 에서 import.
# ---------------------------------------------------------------
# 좌표계:
#   World/Base   = cam0 (T_base<-cam0 = I)
#   Camera (CV)  = +X right, +Y down, +Z forward
#   Object       = GLB canonical (origin centered after normalize)
#   T_base_cam_i = pose of cam_i in base
#   T_base_obj   = pose of object in base
#   Isaac frame  = +X fwd, +Z up  (export 시 변환)
# ---------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import trimesh

# ---- optional deps (safe import) ----------------------------------
try:
    import open3d as o3d  # noqa
    HAVE_O3D = True
except Exception:
    HAVE_O3D = False

try:
    from scipy.spatial.transform import Rotation as _Rot  # noqa
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ===================================================================
#                            DATACLASSES
# ===================================================================

@dataclass
class CameraIntrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    serial: Optional[str] = None

    @property
    def fx(self): return float(self.K[0, 0])
    @property
    def fy(self): return float(self.K[1, 1])
    @property
    def cx(self): return float(self.K[0, 2])
    @property
    def cy(self): return float(self.K[1, 2])


@dataclass
class CameraFrame:
    cam_id: int
    rgb: np.ndarray                  # HxWx3 uint8 (BGR)
    depth: np.ndarray                # HxW   float32 meters
    intr: CameraIntrinsics
    T_base_cam: np.ndarray           # 4x4 base<-cam_i
    frame_id: int = 0


@dataclass
class CanonicalModel:
    name: str                        # GLB stem (예: 'object_002')
    glb_path: Path
    mesh: trimesh.Trimesh            # high-res, origin-centered, meters
    diameter: float
    extents: np.ndarray              # (3,) bbox extents
    color_hint_hsv: Optional[Tuple[int, int, int]] = None
    mesh_low: Optional[trimesh.Trimesh] = None   # decimated for raster


@dataclass
class CameraDetection:
    cam_id: int
    label: str                       # MUST be one of target object names
    score: float
    mask: np.ndarray                 # HxW bool
    bbox_xyxy: Tuple[int, int, int, int]
    centroid_cam: Optional[np.ndarray] = None
    centroid_base: Optional[np.ndarray] = None


@dataclass
class ObjectTrack:
    track_id: int
    label: str
    detections: List[CameraDetection] = field(default_factory=list)
    fused_points_base: Optional[np.ndarray] = None
    fused_colors: Optional[np.ndarray] = None


@dataclass
class PoseEstimate:
    track_id: int
    label: str
    T_base_obj: np.ndarray           # 4x4
    confidence: float = 0.0
    source: str = "fallback"


# ===================================================================
#                       INTRINSICS / EXTRINSICS
# ===================================================================

def load_intrinsics(intr_dir: Path, num_cams: int = 3) -> List[CameraIntrinsics]:
    """cam{i}.npz. 키 (K,dist,width,height) 또는 (color_K,color_D,color_w,color_h)."""
    intr_dir = Path(intr_dir)
    out = []
    for i in range(num_cams):
        d = np.load(intr_dir / f"cam{i}.npz", allow_pickle=True)
        keys = set(d.files)
        K_k = "K" if "K" in keys else "color_K"
        D_k = "dist" if "dist" in keys else ("color_D" if "color_D" in keys else None)
        W_k = "width" if "width" in keys else ("color_w" if "color_w" in keys else None)
        H_k = "height" if "height" in keys else ("color_h" if "color_h" in keys else None)
        S_k = "serial" if "serial" in keys else None
        K = np.asarray(d[K_k], dtype=np.float64)
        dist = np.asarray(d[D_k], dtype=np.float64).reshape(-1) if D_k \
               else np.zeros(5)
        w = int(d[W_k]) if W_k else 0
        h = int(d[H_k]) if H_k else 0
        out.append(CameraIntrinsics(K=K, dist=dist, width=w, height=h,
                                    serial=str(d[S_k]) if S_k else None))
    return out


def load_extrinsics(extr_dir: Path, num_cams: int = 3) -> List[np.ndarray]:
    Ts = [np.eye(4)]
    for i in range(1, num_cams):
        Ts.append(np.asarray(np.load(Path(extr_dir) / f"T_C0_C{i}.npy"),
                             dtype=np.float64))
    return Ts


def load_capture_frame(capture_dir: Path, cam_id: int, frame_id: int,
                       intr: CameraIntrinsics, T_base_cam: np.ndarray,
                       depth_scale: float = 1e-3) -> CameraFrame:
    cdir = Path(capture_dir) / f"cam{cam_id}"
    rgb = cv2.imread(str(cdir / f"rgb_{frame_id:06d}.jpg"), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(cdir / f"depth_{frame_id:06d}.png"),
                           cv2.IMREAD_UNCHANGED)
    if rgb is None or depth_raw is None:
        raise FileNotFoundError(f"frame {frame_id} cam{cam_id} not found in {cdir}")
    depth = depth_raw.astype(np.float32) * depth_scale
    if intr.width == 0 or intr.height == 0:
        intr.height, intr.width = rgb.shape[:2]
    return CameraFrame(cam_id=cam_id, rgb=rgb, depth=depth,
                       intr=intr, T_base_cam=T_base_cam, frame_id=frame_id)


# ===================================================================
#                  GLB LIBRARY (target subset only)
# ===================================================================

def _to_single_mesh(loaded) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        geos = list(loaded.geometry.values())
        if not geos:
            raise ValueError("empty GLB scene")
        return trimesh.util.concatenate(geos) if len(geos) > 1 else geos[0]
    raise ValueError(f"unsupported GLB content: {type(loaded)}")


def _decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if mesh.faces.shape[0] <= target_faces:
        return mesh.copy()
    if HAVE_O3D:
        try:
            mo = o3d.geometry.TriangleMesh()
            mo.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            mo.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            mo2 = mo.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            V = np.asarray(mo2.vertices); F = np.asarray(mo2.triangles)
            if F.shape[0] > 0:
                return trimesh.Trimesh(vertices=V, faces=F, process=False)
        except Exception:
            pass
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        return mesh


def load_canonical_from_glb(glb_path: Path, target_low_faces: int = 5000,
                            color_hint_hsv: Optional[Tuple[int, int, int]] = None
                            ) -> CanonicalModel:
    glb_path = Path(glb_path)
    mesh = _to_single_mesh(trimesh.load(str(glb_path), force="scene"))
    mesh.apply_translation(-mesh.bounding_box.centroid)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    diameter = float(np.linalg.norm(extents))
    return CanonicalModel(name=glb_path.stem, glb_path=glb_path, mesh=mesh,
                          diameter=diameter, extents=extents,
                          color_hint_hsv=color_hint_hsv,
                          mesh_low=_decimate_mesh(mesh, target_low_faces))


def load_target_library(target_glb_paths: List[Path],
                        color_priors_hsv: Optional[Dict[str, Tuple[int, int, int]]] = None,
                        target_low_faces: int = 5000
                        ) -> Dict[str, CanonicalModel]:
    """target_glb_paths 에 명시된 GLB 만 로드 (subset-only). 빈 list 면 ValueError."""
    if not target_glb_paths:
        raise ValueError("target_glb_paths is empty — selective pipeline requires explicit targets.")
    lib: Dict[str, CanonicalModel] = {}
    cp = color_priors_hsv or {}
    for p in target_glb_paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"target GLB not found: {p}")
        m = load_canonical_from_glb(p, target_low_faces=target_low_faces,
                                    color_hint_hsv=cp.get(p.stem))
        lib[m.name] = m
    return lib


# ===================================================================
#                       COLOR PRIORS (재사용성)
# ===================================================================

DEMO_COLOR_PRIORS_HSV: Dict[str, Tuple[int, int, int]] = {
    "object_001": (0,   200, 160),
    "object_002": (25,  200, 200),
    "object_003": (110, 200, 80),
    "object_004": (85,  130, 170),
}


def load_color_priors_json(path: Path) -> Dict[str, Tuple[int, int, int]]:
    return {k: tuple(int(x) for x in v)
            for k, v in json.loads(Path(path).read_text()).items()}


def save_color_priors_json(priors: Dict[str, Tuple[int, int, int]], path: Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: list(v) for k, v in priors.items()}, indent=2))
    return p


def hue_distance(h1: np.ndarray, h2: int) -> np.ndarray:
    d = np.abs(h1.astype(np.int32) - int(h2))
    return np.minimum(d, 180 - d).astype(np.float32)


def hsv_to_bgr(hsv: Tuple[int, int, int]) -> Tuple[int, int, int]:
    bgr = cv2.cvtColor(np.uint8([[list(hsv)]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ===================================================================
#                         GEOMETRY UTILS
# ===================================================================

def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    return (Ph @ T.T)[:, :3]


def invert_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def backproject_masked(depth: np.ndarray, mask: np.ndarray,
                       intr: CameraIntrinsics, max_depth: float = 3.0) -> np.ndarray:
    H, W = depth.shape
    ys, xs = np.where(mask & (depth > 1e-3) & (depth < max_depth))
    if xs.size == 0:
        return np.zeros((0, 3))
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - intr.cx) * z / intr.fx
    y = (ys.astype(np.float64) - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=1)


def project_points(P_cam: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    z = np.maximum(P_cam[:, 2], 1e-6)
    u = intr.fx * P_cam[:, 0] / z + intr.cx
    v = intr.fy * P_cam[:, 1] / z + intr.cy
    return np.stack([u, v], axis=1)


# ===================================================================
#                          MESH RASTER
# ===================================================================

def rasterize_mesh_mask(mesh: trimesh.Trimesh, T_cam_obj: np.ndarray,
                        intr: CameraIntrinsics, H: int, W: int,
                        max_faces: int = 5000) -> np.ndarray:
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
    if F.shape[0] > max_faces:
        idx = np.random.RandomState(0).choice(F.shape[0], max_faces, replace=False)
        F = F[idx]
    V_cam = transform_points(T_cam_obj, V)
    mask = np.zeros((H, W), dtype=np.uint8)
    valid_v = V_cam[:, 2] > 1e-3
    F = F[valid_v[F].all(axis=1)]
    if F.shape[0] == 0:
        return mask
    uv = project_points(V_cam, intr).astype(np.int32)
    fu = uv[F]
    in_screen = ~(((fu[:, :, 0] >= W).all(1)) | ((fu[:, :, 0] < 0).all(1))
                  | ((fu[:, :, 1] >= H).all(1)) | ((fu[:, :, 1] < 0).all(1)))
    for tri in fu[in_screen]:
        cv2.fillConvexPoly(mask, tri, 1)
    return mask


def render_mesh_silhouette(rgb: np.ndarray, frame: CameraFrame,
                           model: CanonicalModel, T_base_obj: np.ndarray,
                           color_bgr: Tuple[int, int, int],
                           alpha: float = 0.45) -> np.ndarray:
    H, W = rgb.shape[:2]
    T_cam_obj = invert_se3(frame.T_base_cam) @ T_base_obj
    mesh = model.mesh_low if model.mesh_low is not None else model.mesh
    sil = rasterize_mesh_mask(mesh, T_cam_obj, frame.intr, H, W)
    out = rgb.copy()
    if sil.sum() > 0:
        layer = np.full_like(rgb, 0); layer[:] = color_bgr
        m = sil.astype(bool)
        out[m] = (alpha * layer[m] + (1.0 - alpha) * rgb[m]).astype(np.uint8)
        contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color_bgr, 2, lineType=cv2.LINE_AA)
    corners = trimesh.bounds.corners(model.mesh.bounding_box.bounds)
    Cc = transform_points(T_cam_obj, corners)
    if (Cc[:, 2] > 1e-3).all():
        cuv = project_points(Cc, frame.intr).astype(np.int32)
        edges = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for a, b in edges:
            cv2.line(out, tuple(cuv[a]), tuple(cuv[b]), color_bgr, 1, cv2.LINE_AA)
    return out


def _put_label(img: np.ndarray, text: str, org=(8, 22),
               fg=(255, 255, 255), bg=(0, 0, 0)) -> np.ndarray:
    cv2.rectangle(img, (org[0]-4, org[1]-18), (org[0]+8*len(text), org[1]+6), bg, -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)
    return img


def make_comparison_grid(frames: List[CameraFrame],
                         library: Dict[str, CanonicalModel],
                         estimates: List[PoseEstimate],
                         label_color_bgr: Dict[str, Tuple[int, int, int]],
                         frame_id: int) -> np.ndarray:
    top, bot = [], []
    for f in frames:
        t = f.rgb.copy(); _put_label(t, f"cam{f.cam_id} Original (frame {frame_id:06d})")
        top.append(t)
        b = f.rgb.copy()
        labels_drawn = []
        for e in estimates:
            if e.label not in library:
                continue
            color = label_color_bgr.get(e.label, (0, 255, 0))
            b = render_mesh_silhouette(b, f, library[e.label], e.T_base_obj, color)
            labels_drawn.append(e.label)
        _put_label(b, f"cam{f.cam_id} Pose Overlay")
        if labels_drawn:
            _put_label(b, " | ".join(labels_drawn), org=(8, b.shape[0] - 8))
        bot.append(b)
    return np.concatenate(
        [np.concatenate(top, axis=1), np.concatenate(bot, axis=1)], axis=0)


# ===================================================================
#                              EXPORT
# ===================================================================

T_CV_TO_ISAAC = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64)


def export_posed_glb(model: CanonicalModel, T_base_obj: np.ndarray,
                     out_path: Path, frame: str = "opencv") -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    m = model.mesh.copy()
    T = T_CV_TO_ISAAC @ T_base_obj if frame == "isaac" else T_base_obj
    m.apply_transform(T)
    trimesh.Scene(m).export(str(out_path))
    return out_path


def _quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    if HAVE_SCIPY:
        from scipy.spatial.transform import Rotation as Rot
        q = Rot.from_matrix(R).as_quat()
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = float(np.sqrt(tr + 1) * 2)
        return ((R[2,1]-R[1,2])/S, (R[0,2]-R[2,0])/S, (R[1,0]-R[0,1])/S, 0.25*S)
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = float(np.sqrt(1+R[0,0]-R[1,1]-R[2,2]) * 2)
        return (0.25*S, (R[0,1]+R[1,0])/S, (R[0,2]+R[2,0])/S, (R[2,1]-R[1,2])/S)
    if R[1,1] > R[2,2]:
        S = float(np.sqrt(1+R[1,1]-R[0,0]-R[2,2]) * 2)
        return ((R[0,1]+R[1,0])/S, 0.25*S, (R[1,2]+R[2,1])/S, (R[0,2]-R[2,0])/S)
    S = float(np.sqrt(1+R[2,2]-R[0,0]-R[1,1]) * 2)
    return ((R[0,2]+R[2,0])/S, (R[1,2]+R[2,1])/S, 0.25*S, (R[1,0]-R[0,1])/S)


def _euler_xyz_deg(R: np.ndarray) -> Tuple[float, float, float]:
    if HAVE_SCIPY:
        from scipy.spatial.transform import Rotation as Rot
        return tuple(float(x) for x in Rot.from_matrix(R).as_euler("xyz", degrees=True))
    sy = float(np.sqrt(R[0,0]**2 + R[1,0]**2))
    if sy > 1e-6:
        return (float(np.degrees(np.arctan2(R[2,1], R[2,2]))),
                float(np.degrees(np.arctan2(-R[2,0], sy))),
                float(np.degrees(np.arctan2(R[1,0], R[0,0]))))
    return (float(np.degrees(np.arctan2(-R[1,2], R[1,1]))),
            float(np.degrees(np.arctan2(-R[2,0], sy))), 0.0)


def export_results_json(estimates: List[PoseEstimate], tracks: List[ObjectTrack],
                        library: Dict[str, CanonicalModel],
                        posed_glb_paths: Dict[int, str],
                        out_path: Path) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    track_by_id = {t.track_id: t for t in tracks}
    payload = {"estimates": []}
    for e in estimates:
        t = track_by_id.get(e.track_id)
        model = library.get(e.label)
        T = np.asarray(e.T_base_obj)
        qx, qy, qz, qw = _quat_xyzw(T[:3, :3])
        size_mm = (np.asarray(model.extents) * 1000.0).tolist() if model else None
        payload["estimates"].append({
            "object_name": e.label,
            "glb_path": str(model.glb_path) if model else None,
            "track_id": e.track_id,
            "confidence": float(e.confidence),
            "source": e.source,
            "T_base_obj": T.tolist(),
            "translation_mm": (T[:3, 3] * 1000.0).tolist(),
            "quaternion_xyzw": [qx, qy, qz, qw],
            "size_mm": size_mm,
            "cams": [d.cam_id for d in (t.detections if t else [])],
            "posed_glb": posed_glb_paths.get(e.track_id),
        })
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


def export_results_npz(estimates: List[PoseEstimate], out_path: Path) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    if not estimates:
        np.savez(out_path, labels=np.array([]), track_ids=np.array([]),
                 poses=np.zeros((0, 4, 4)))
    else:
        np.savez(out_path,
                 labels=np.array([e.label for e in estimates]),
                 track_ids=np.array([e.track_id for e in estimates]),
                 confidences=np.array([e.confidence for e in estimates]),
                 poses=np.stack([e.T_base_obj for e in estimates], axis=0))
    return out_path


def _fmt_signed(x: float, prec: int = 1) -> str:
    return f"{x:+.{prec}f}".replace("-", "−")


DEFAULT_OBJECT_DESCRIPTIONS = {
    "object_001": "빨강 아치", "object_002": "노랑 실린더",
    "object_003": "곤색 직사각형", "object_004": "민트 실린더",
}


def load_object_descriptions(path: Path) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    return {k: str(v) for k, v in json.loads(p.read_text()).items()}


def export_results_text(estimates: List[PoseEstimate],
                        library: Dict[str, CanonicalModel],
                        out_path: Path, frame_id: int,
                        descriptions: Optional[Dict[str, str]] = None) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    desc = dict(DEFAULT_OBJECT_DESCRIPTIONS)
    if descriptions:
        desc.update(descriptions)
    lines = [f"Frame {frame_id:06d}"]
    for e in estimates:
        m = library.get(e.label)
        size_mm = (np.asarray(m.extents) * 1000.0) if m else np.zeros(3)
        T = np.asarray(e.T_base_obj); t_mm = T[:3, 3] * 1000.0
        rx, ry, rz = _euler_xyz_deg(T[:3, :3])
        qx, qy, qz, qw = _quat_xyzw(T[:3, :3])
        lines += ["", f"[{e.label}] {desc.get(e.label, '')}".rstrip(), "",
                  f"크기 (mm): X={size_mm[0]:.1f} Y={size_mm[1]:.1f} Z={size_mm[2]:.1f}",
                  "위치 (mm): "
                  f"X={_fmt_signed(t_mm[0])} Y={_fmt_signed(t_mm[1])} Z={_fmt_signed(t_mm[2])}",
                  "회전 Euler (°): "
                  f"rx={_fmt_signed(rx,2)} ry={_fmt_signed(ry,2)} rz={_fmt_signed(rz,2)}",
                  "쿼터니언 (xyzw): "
                  f"[{_fmt_signed(qx,4)}, {_fmt_signed(qy,4)}, "
                  f"{_fmt_signed(qz,4)}, {_fmt_signed(qw,4)}]",
                  f"신뢰도: {e.confidence:.3f}  (source={e.source})"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
