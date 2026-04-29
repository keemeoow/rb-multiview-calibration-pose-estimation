#!/usr/bin/env python3
"""
Calibration-Aware Multi-View RGB-D CAD Pose Estimation Pipeline

멀티뷰 RGB-D + 캘리브레이션 + GLB → 물체별 6D pose (멀티 오브젝트 지원)

실행:
  # 단일 프레임 (프레임→GLB 고정 매핑)
  python3 src/pose_pipeline.py --frame_id 000000

  # 특정 GLB 수동 지정
  python3 src/pose_pipeline.py --frame_id 000000 --glb src/data/object_004.glb

  # 멀티 오브젝트 모드
  python3 src/pose_pipeline.py --frame_id 000000 --multi

  # 전체 배치
  python3 src/pose_pipeline.py --batch
"""

# ═══════════════════════════════════════════════════════════
# 0. 좌표계 정의
# ═══════════════════════════════════════════════════════════
#
# ┌───────────────────────────────────────────────────────┐
# │ Frame     │ Description                               │
# ├───────────────────────────────────────────────────────┤
# │ cam_i     │ 카메라 i 광학 좌표계 (OpenCV)                  │
# │           │ x: right, y: down, z: forward             │
# │ base      │ 기준 = cam0 (hand-eye 없을 때)               │
# │ obj       │ GLB canonical frame (mesh centroid 기준)   │
# │ sim/isaac │ Isaac Sim (X-fwd, Y-left, Z-up)           │
# └───────────────────────────────────────────────────────┘
#
# T_A_B = "A ← B" → p_A = T_A_B @ p_B
# 단위: meter | 쿼터니언: (x,y,z,w) SciPy
# ═══════════════════════════════════════════════════════════

import json
import argparse
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import cv2
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

# Isaac Sim 좌표 변환: OpenCV → Isaac
T_ISAAC_CV = np.array([
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0,  0,  1]], dtype=np.float64)


# ═══════════════════════════════════════════════════════════
# 1. 데이터 구조
# ═══════════════════════════════════════════════════════════

@dataclass
class CameraIntrinsics:
    K: np.ndarray; D: np.ndarray; depth_scale: float
    width: int = 640; height: int = 480

@dataclass
class CameraFrame:
    cam_id: int; intrinsics: CameraIntrinsics
    T_base_cam: np.ndarray; color_bgr: np.ndarray; depth_u16: np.ndarray

@dataclass
class CanonicalModel:
    name: str; mesh: trimesh.Trimesh; center: np.ndarray
    extents_m: np.ndarray; is_watertight: bool
    mesh_low: Optional[trimesh.Trimesh] = None  # 렌더용 decimated mesh

@dataclass
class PoseEstimate:
    T_base_obj: np.ndarray; position_m: np.ndarray
    quaternion_xyzw: np.ndarray; euler_xyz_deg: np.ndarray
    scale: float = 1.0; confidence: float = 0.0
    fitness: float = 0.0; rmse: float = 0.0
    depth_score: float = 0.0; coverage: float = 0.0
    silhouette_score: float = 0.0; rgb_score: float = 0.0

@dataclass
class CameraDetection:
    cam_id: int
    det_id: str
    mask_u8: np.ndarray
    bbox_xyxy: np.ndarray
    area_px: int
    score: float
    centroid_uv: np.ndarray
    color_hist: np.ndarray
    mean_hsv: np.ndarray
    centroid_base: Optional[np.ndarray] = None
    extents_base: Optional[np.ndarray] = None
    hint_model_name: Optional[str] = None
    hint_score: float = 0.0

@dataclass
class ObjectTrack:
    track_id: str
    detections: Dict[int, CameraDetection]
    observed_masks: List[np.ndarray] = field(default_factory=list)
    centroids_base: List[np.ndarray] = field(default_factory=list)
    centroid_base: Optional[np.ndarray] = None
    fused_points_base: Optional[np.ndarray] = None
    candidate_scores: List[Tuple[str, float]] = field(default_factory=list)
    best_pose: Optional[PoseEstimate] = None
    best_model_name: Optional[str] = None
    ambiguity: List[Tuple[str, float]] = field(default_factory=list)
    hint_model_name: Optional[str] = None

DEPTH_POLICY = {
    "min_depth_m": 0.10, "max_depth_m": 1.20, "voxel_size_m": 0.002,
    "table_dist_thresh_m": 0.008,
    "object_min_height_m": 0.005, "object_max_height_m": 0.15,
    "cluster_eps_m": 0.006, "cluster_min_pts": 50,
    "cluster_min_extent_m": 0.012, "cluster_max_extent_m": 0.25,
}

# 프레임→GLB 고정 매핑 (단일 오브젝트 캡처)
FRAME_TO_GLB = {
    **{i: "object_004" for i in range(0, 5)},   # 민트 실린더
    **{i: "object_003" for i in range(5, 10)},   # 곤색 직사각형
    **{i: "object_002" for i in range(10, 15)},  # 노랑 실린더
    **{i: "object_001" for i in range(15, 17)},  # 빨강 아치
}

OBJECT_LABELS = {
    "object_001": "빨강 아치",    "object_002": "노랑 실린더",
    "object_003": "곤색 직사각형", "object_004": "민트 실린더",
}

MODEL_NAME_ALIASES = {
    "object_001": "object_001",
    "red_arch": "object_001",
    "red-arch": "object_001",
    "빨강_아치": "object_001",
    "빨강아치": "object_001",
    "object_002": "object_002",
    "yellow_cylinder": "object_002",
    "yellow-cylinder": "object_002",
    "노랑_실린더": "object_002",
    "노랑실린더": "object_002",
    "object_003": "object_003",
    "navy_block": "object_003",
    "navy-block": "object_003",
    "rect_block": "object_003",
    "곤색_직사각형": "object_003",
    "곤색직사각형": "object_003",
    "object_004": "object_004",
    "mint_cylinder": "object_004",
    "mint-cylinder": "object_004",
    "민트_실린더": "object_004",
    "민트실린더": "object_004",
}

# 물체 라이브러리 appearance prior.
# 새 물체는 이 metadata를 별도 파일로 빼는 것이 맞지만, 현재 파이프라인은
# library metadata가 없을 때 shape-only fallback이 가능하도록 구성한다.
OBJECT_COLOR_PRIORS_HSV = {
    "object_001": np.array([0.0, 185.0, 150.0], dtype=np.float32),
    "object_002": np.array([28.0, 185.0, 190.0], dtype=np.float32),
    "object_003": np.array([108.0, 110.0, 70.0], dtype=np.float32),
    "object_004": np.array([72.0, 80.0, 175.0], dtype=np.float32),
}

# 물체별 대칭 타입: "none" = 비대칭, "yaw" = 축 대칭 (실린더 등)
OBJECT_SYMMETRY = {
    "object_001": "none",
    "object_002": "yaw",
    "object_003": "none",
    "object_004": "yaw",
}

# 와이어프레임 색상 (BGR)
COLORS = [(0,255,0), (0,165,255), (255,0,255), (0,255,255),
          (255,128,0), (128,255,0), (255,0,128), (0,128,255)]


# ═══════════════════════════════════════════════════════════
# 2. 데이터 로딩
# ═══════════════════════════════════════════════════════════

def load_calibration(data_dir: Path, intrinsics_dir: Path,
                     num_cams: Optional[int] = None):
    """카메라 intrinsic + extrinsic 로드.

    num_cams=None 이면 `intrinsics_dir/cam*.npz` 개수에서 자동 감지.
    extrinsic 은 `data_dir/cube_session_01/calib_out_cube/T_C0_C{ci}.npy`
    형식으로 cam0 기준 (cam0은 항상 identity).
    """
    intrinsics_dir = Path(intrinsics_dir)
    if num_cams is None:
        cams_found = sorted(intrinsics_dir.glob("cam*.npz"))
        num_cams = len(cams_found)
        if num_cams == 0:
            raise FileNotFoundError(f"intrinsics not found in {intrinsics_dir}")
    intrinsics = []
    for ci in range(num_cams):
        npz = np.load(str(intrinsics_dir / f"cam{ci}.npz"), allow_pickle=True)
        intrinsics.append(CameraIntrinsics(
            K=npz["color_K"].astype(np.float64),
            D=npz["color_D"].astype(np.float64),
            depth_scale=float(npz["depth_scale_m_per_unit"]),
            width=int(npz["color_w"]), height=int(npz["color_h"])))

    ext_dir = Path(data_dir) / "cube_session_01" / "calib_out_cube"
    extrinsics = {0: np.eye(4)}
    for ci in range(1, num_cams):
        ext_file = ext_dir / f"T_C0_C{ci}.npy"
        if not ext_file.exists():
            raise FileNotFoundError(f"missing extrinsic: {ext_file}")
        extrinsics[ci] = np.load(str(ext_file)).astype(np.float64)
    return intrinsics, extrinsics


def load_frame(data_dir: Path, frame_id: str, intrinsics, extrinsics,
               capture_subdir: str = "object_capture",
               num_cams: Optional[int] = None):
    """frame_id 의 RGB-D 를 모든 카메라에서 로드.

    num_cams=None 이면 intrinsics 길이를 사용.
    """
    if num_cams is None:
        num_cams = len(intrinsics)
    img_dir = Path(data_dir) / capture_subdir
    frames = []
    for ci in range(num_cams):
        c = cv2.imread(str(img_dir / f"cam{ci}" / f"rgb_{frame_id}.jpg"))
        d = cv2.imread(str(img_dir / f"cam{ci}" / f"depth_{frame_id}.png"),
                       cv2.IMREAD_UNCHANGED)
        if c is None or d is None:
            raise FileNotFoundError(f"{capture_subdir}/cam{ci}/frame_{frame_id}")
        frames.append(CameraFrame(ci, intrinsics[ci], extrinsics[ci], c, d))
    return frames


class RGBHeuristicSegmentationModel:
    """Fallback instance segmenter.

    실제 배포에서는 SAM / YOLOv8-seg / Mask2Former 같은 RGB segmentation
    backend로 대체해야 한다. 현재 구현은 파이프라인 구조를 segmentation-first로
    유지하기 위한 임시 RGB-only fallback이다.
    """

    def __init__(self, score_thresh: float = 0.5, min_area_px: int = 200,
                 max_area_ratio: float = 0.25):
        self.score_thresh = score_thresh
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio

    def __call__(self, image_bgr: np.ndarray):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 배경(회색 테이블/벽) 대비 채색 물체를 우선 분리하고,
        # 어두운 곤색 블록은 저채도-저명도 예외로 통과시킨다.
        fg = ((sat > 28) | ((sat > 12) & (val < 120) & (val > 20))).astype(np.uint8) * 255

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k5)
        fg = cv2.dilate(fg, k3, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        max_area_px = int(image_bgr.shape[0] * image_bgr.shape[1] * self.max_area_ratio)
        results = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area_px or area > max_area_px:
                continue
            comp = np.zeros_like(fg)
            comp[labels == i] = 255
            for sub_mask in maybe_split_mask_by_appearance(image_bgr, comp):
                sub_area = int(np.count_nonzero(sub_mask))
                if sub_area < self.min_area_px or sub_area > max_area_px:
                    continue
                score = 1.0
                results.append({"mask": sub_mask > 0, "score": score})
        return results


class KnownBlockColorSegmentationModel:
    """CPU-friendly known-object segmenter using HSV color priors.

    현재 데이터처럼 색이 뚜렷한 블록 세트에서는 generic foreground보다
    object-wise mask를 훨씬 안정적으로 만든다.
    """

    def __init__(self, color_priors: Dict[str, np.ndarray],
                 min_area_px: int = 180, max_area_ratio: float = 0.12):
        self.color_priors = color_priors
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio

    def __call__(self, image_bgr: np.ndarray):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        valid = (s > 10) | ((v > 20) & (v < 150))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        max_area_px = int(image_bgr.shape[0] * image_bgr.shape[1] * self.max_area_ratio)

        score_maps = []
        names = []
        for name, prior in self.color_priors.items():
            hue_diff = np.abs(h - float(prior[0]))
            hue_diff = np.minimum(hue_diff, 180.0 - hue_diff) / 90.0
            hue_sim = 1.0 - np.clip(hue_diff, 0.0, 1.0)
            sat_sim = 1.0 - np.clip(np.abs(s - float(prior[1])) / 160.0, 0.0, 1.0)
            val_sim = 1.0 - np.clip(np.abs(v - float(prior[2])) / 160.0, 0.0, 1.0)
            score = 0.60 * hue_sim + 0.25 * sat_sim + 0.15 * val_sim
            score *= valid.astype(np.float32)
            score_maps.append(score)
            names.append(name)

        if not score_maps:
            return []

        score_stack = np.stack(score_maps, axis=0)
        best_idx = np.argmax(score_stack, axis=0)
        best_score = np.max(score_stack, axis=0)
        results = []

        for idx, name in enumerate(names):
            thresh = 0.58 if name in {"object_001", "object_002", "object_004"} else 0.48
            mask = ((best_idx == idx) & (best_score >= thresh)).astype(np.uint8) * 255
            if int(np.count_nonzero(mask)) < self.min_area_px:
                continue

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            mask = cv2.dilate(mask, k3, iterations=1)

            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            components = []
            for li in range(1, num):
                area = int(stats[li, cv2.CC_STAT_AREA])
                if area < self.min_area_px or area > max_area_px:
                    continue
                comp = np.zeros_like(mask)
                comp[labels == li] = 255
                ys, xs = np.where(comp > 0)
                mean_score = float(best_score[ys, xs].mean()) if len(xs) > 0 else 0.0
                components.append({
                    "mask": comp > 0,
                    "score": mean_score,
                    "model_name": name,
                    "_rank": area * mean_score,
                })

            if components:
                components.sort(key=lambda x: x["_rank"], reverse=True)
                best = components[0]
                best.pop("_rank", None)
                results.append(best)

        return results


class CombinedSegmentationModel:
    """Known-color segmenter 우선, 부족하면 generic heuristic로 보충."""

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback

    def __call__(self, image_bgr: np.ndarray):
        primary_results = self.primary(image_bgr) if self.primary is not None else []
        if primary_results or self.fallback is None:
            return primary_results

        fallback_results = self.fallback(image_bgr)
        return primary_results + fallback_results


def normalize_hint_model_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    key = str(name).strip().lower().replace(" ", "_")
    return MODEL_NAME_ALIASES.get(key, key if key.startswith("object_") else None)


class UltralyticsSegmentationModel:
    """Optional YOLOv8-seg backend.

    학습된 weights가 있으면 heuristic 대신 이 backend를 우선 사용한다.
    """

    def __init__(self, weights_path: Path, device: str = "cpu",
                 conf: float = 0.35, iou: float = 0.50):
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.device = device
        self.conf = conf
        self.iou = iou
        self.backend_name = f"ultralytics:{weights_path.name}"

    def __call__(self, image_bgr: np.ndarray):
        preds = self.model.predict(
            source=image_bgr,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        if not preds:
            return []

        pred = preds[0]
        if pred.masks is None or pred.boxes is None:
            return []

        masks = pred.masks.data.cpu().numpy()
        confs = pred.boxes.conf.cpu().numpy()
        classes = pred.boxes.cls.cpu().numpy().astype(int)
        names = pred.names

        results = []
        for mask, conf, cls_id in zip(masks, confs, classes):
            label = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else names[int(cls_id)]
            results.append({
                "mask": mask > 0.5,
                "score": float(conf),
                "model_name": normalize_hint_model_name(label),
            })
        return results


def load_segmentation_model(seg_backend: str = "auto",
                            seg_model_path: Optional[str] = None,
                            seg_device: str = "cpu",
                            seg_conf: float = 0.35,
                            seg_iou: float = 0.50):
    """RGB instance segmentation backend loader.

    실제 segmentation model(SAM/YOLOv8-seg 등)이 연결되지 않은 환경에서는
    fallback RGB heuristic backend를 사용한다.
    """
    candidate_paths = []
    if seg_model_path:
        candidate_paths.append(Path(seg_model_path))
    candidate_paths.extend([
        Path("src/models/blocks_yolov8n_seg.pt"),
        Path("src/models/blocks_seg.pt"),
        Path("models/blocks_yolov8n_seg.pt"),
    ])

    if seg_backend in {"auto", "yolo"}:
        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                return UltralyticsSegmentationModel(
                    path,
                    device=seg_device,
                    conf=seg_conf,
                    iou=seg_iou,
                )
            except Exception as exc:
                if seg_backend == "yolo":
                    raise RuntimeError(f"YOLO segmentation backend 로드 실패: {exc}") from exc
                print(f"  [WARN] YOLO segmentation backend 로드 실패: {exc}")
                break
        if seg_backend == "yolo":
            raise RuntimeError("YOLO segmentation backend 요청됨, 그러나 weights 파일을 찾지 못함")

    primary = KnownBlockColorSegmentationModel(OBJECT_COLOR_PRIORS_HSV)
    fallback = RGBHeuristicSegmentationModel()
    model = CombinedSegmentationModel(primary, fallback)
    model.backend_name = "known-color+heuristic"
    return model


def normalize_glb(glb_path: Path) -> CanonicalModel:
    scene = trimesh.load(str(glb_path))
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) \
        if isinstance(scene, trimesh.Scene) else scene
    # 렌더용 경량 mesh 생성 (face 수 2000 이하로)
    mesh_low = mesh
    if len(mesh.faces) > 2000:
        try:
            mesh_low = mesh.simplify_quadric_decimation(2000)
        except Exception:
            mesh_low = mesh
    return CanonicalModel(
        name=glb_path.stem, mesh=mesh, center=mesh.centroid.copy(),
        extents_m=mesh.bounding_box.extents.copy(), is_watertight=mesh.is_watertight,
        mesh_low=mesh_low)


def sample_model_points(model: CanonicalModel, n=20000) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(model.mesh, n)
    return (pts - model.center).astype(np.float64)


# ═══════════════════════════════════════════════════════════
# 3. 점군 유틸리티
# ═══════════════════════════════════════════════════════════

def transform_points(pts, T):
    return (T @ np.hstack([pts, np.ones((len(pts), 1))]).T)[:3].T

def backproject_depth(cam: CameraFrame, mask=None):
    h, w = cam.depth_u16.shape
    K, ds = cam.intrinsics.K, cam.intrinsics.depth_scale
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = cam.depth_u16.astype(np.float64) * ds
    ok = (z > DEPTH_POLICY["min_depth_m"]) & (z < DEPTH_POLICY["max_depth_m"])
    if mask is not None: ok &= (mask > 0)
    z = z[ok]
    return np.stack([(uu[ok]-K[0,2])*z/K[0,0], (vv[ok]-K[1,2])*z/K[1,1], z], -1)


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0, 0, 0, 0], dtype=np.int32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int32)


def normalized_extent(ext: np.ndarray) -> np.ndarray:
    mx = float(np.max(ext)) if len(ext) else 0.0
    if mx < 1e-8:
        return np.zeros_like(ext, dtype=np.float64)
    return np.sort(ext.astype(np.float64)) / mx


def circular_hue_distance(h1: float, h2: float) -> float:
    dh = abs(float(h1) - float(h2))
    return min(dh, 180.0 - dh) / 90.0


def compute_mask_color_hist(color_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [12], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [4], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [4], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-8
    return hist


def compute_mask_mean_hsv(color_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[mask > 0]
    if len(pix) == 0:
        return np.zeros(3, dtype=np.float32)
    h = pix[:, 0].astype(np.float32)
    ang = h / 180.0 * (2.0 * np.pi)
    h_mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    if h_mean < 0:
        h_mean += 2.0 * np.pi
    hue = h_mean / (2.0 * np.pi) * 180.0
    sat = pix[:, 1].astype(np.float32).mean()
    val = pix[:, 2].astype(np.float32).mean()
    return np.array([hue, sat, val], dtype=np.float32)


def maybe_split_mask_by_appearance(color_bgr: np.ndarray, component_mask: np.ndarray):
    area = int(np.count_nonzero(component_mask))
    ys, xs = np.where(component_mask > 0)
    if area < 1200 or len(xs) < 600:
        return [component_mask]

    x_span = int(xs.max() - xs.min() + 1)
    y_span = int(ys.max() - ys.min() + 1)
    if max(x_span, y_span) < 70:
        return [component_mask]

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    feat = np.stack([
        ((xs - xs.mean()) / max(float(x_span), 1.0)) * 0.8,
        ((ys - ys.mean()) / max(float(y_span), 1.0)) * 0.8,
        hsv[ys, xs, 0].astype(np.float32) / 180.0 * 1.8,
        hsv[ys, xs, 1].astype(np.float32) / 255.0,
        hsv[ys, xs, 2].astype(np.float32) / 255.0 * 0.5,
    ], axis=1).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.2,
    )
    try:
        _, labels, centers = cv2.kmeans(
            feat,
            2,
            None,
            criteria,
            4,
            cv2.KMEANS_PP_CENTERS,
        )
    except cv2.error:
        return [component_mask]

    labels = labels.reshape(-1)
    color_gap = np.linalg.norm(centers[0, 2:] - centers[1, 2:])
    spatial_gap = np.linalg.norm(centers[0, :2] - centers[1, :2])
    if color_gap < 0.30 and spatial_gap < 0.30:
        return [component_mask]

    masks = []
    min_child_area = max(150, int(area * 0.15))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for k in range(2):
        sub_mask = np.zeros_like(component_mask)
        sel = labels == k
        if sel.sum() < min_child_area:
            continue
        sub_mask[ys[sel], xs[sel]] = 255
        sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel)
        sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_CLOSE, kernel)
        if int(np.count_nonzero(sub_mask)) >= min_child_area:
            masks.append(sub_mask)

    if len(masks) < 2:
        return [component_mask]
    return masks


def create_camera_detection(cam: CameraFrame, mask: np.ndarray, det_id: str,
                            hint_model_name: Optional[str] = None,
                            hint_score: float = 0.0):
    area_px = int(np.count_nonzero(mask))
    if area_px == 0:
        return None

    bbox_xyxy = mask_bbox(mask)
    ys, xs = np.where(mask > 0)
    centroid_uv = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    color_hist = compute_mask_color_hist(cam.color_bgr, mask)
    mean_hsv = compute_mask_mean_hsv(cam.color_bgr, mask)

    pts_cam = backproject_depth(cam, mask=mask)
    centroid_base = None
    extents_base = None
    if len(pts_cam) > 0:
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        centroid_base = pts_base.mean(axis=0)
        extents_base = pts_base.max(axis=0) - pts_base.min(axis=0)

    return CameraDetection(
        cam_id=cam.cam_id,
        det_id=det_id,
        mask_u8=mask,
        bbox_xyxy=bbox_xyxy,
        area_px=area_px,
        score=min(1.0, area_px / float(mask.size)),
        centroid_uv=centroid_uv,
        color_hist=color_hist,
        mean_hsv=mean_hsv,
        centroid_base=centroid_base,
        extents_base=extents_base,
        hint_model_name=hint_model_name,
        hint_score=hint_score,
    )


def build_camera_detections(cam: CameraFrame, masks: List[np.ndarray]):
    detections = []
    for di, mask in enumerate(masks):
        det = create_camera_detection(cam, mask, f"cam{cam.cam_id}_{di}")
        if det is None or det.area_px < 150:
            continue
        detections.append(det)
    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections


def build_camera_instances(cam: CameraFrame, segmenter, min_score=0.5, min_area=150):
    """RGB instance segmentation → CameraDetection 리스트.

    segmentation model 출력(mask+score)을 직접 CameraDetection으로 변환한다.
    build_observed_mask + build_camera_detections 2단계를 하나로 합친 함수.
    """
    results = segmenter(cam.color_bgr)
    detections = []
    det_idx = 0
    for obj in results:
        if isinstance(obj, dict):
            mask = obj.get("mask")
            score = float(obj.get("score", 1.0))
            model_name = obj.get("model_name")
        else:
            mask = obj
            score = 1.0
            model_name = None

        if mask is None or score < min_score:
            continue

        mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
        area = int(np.count_nonzero(mask_u8))
        if area < min_area:
            continue

        det = create_camera_detection(
            cam,
            mask_u8,
            f"cam{cam.cam_id}_seg_{det_idx}",
            hint_model_name=model_name,
            hint_score=score,
        )
        if det is None:
            continue

        detections.append(det)
        det_idx += 1

    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections


def compute_3d_centroid(cam: CameraFrame, mask: np.ndarray):
    pts_cam = backproject_depth(cam, mask=mask)
    if len(pts_cam) == 0:
        return None
    pts_base = transform_points(pts_cam, cam.T_base_cam)
    return pts_base.mean(axis=0)


def combine_instance_masks(masks: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=np.uint8)
    out = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        out = cv2.bitwise_or(out, mask.astype(np.uint8))
    return out


def save_camera_detections(detections_by_cam, frames, frame_id, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in frames:
        vis = cam.color_bgr.copy()
        detections = detections_by_cam.get(cam.cam_id, [])
        for di, det in enumerate(detections):
            x0, y0, x1, y1 = det.bbox_xyxy.tolist()
            color = COLORS[di % len(COLORS)]
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), color, 2)
            cv2.putText(
                vis,
                f"d{di}",
                (x0, max(20, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.imwrite(str(out_dir / f"detections_cam{cam.cam_id}_{frame_id}.png"), vis)


def color_hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def extent_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.5
    return float(1.0 / (1.0 + 4.0 * np.linalg.norm(normalized_extent(a) - normalized_extent(b))))


def detection_pair_score(a: CameraDetection, b: CameraDetection) -> float:
    pos_score = 0.0
    if a.centroid_base is not None and b.centroid_base is not None:
        dist = np.linalg.norm(a.centroid_base - b.centroid_base)
        pos_score = float(np.exp(-((dist / 0.06) ** 2)))
    app_score = color_hist_similarity(a.color_hist, b.color_hist)
    size_score = extent_similarity(a.extents_base, b.extents_base)
    return 0.50 * pos_score + 0.35 * app_score + 0.15 * size_score


def reprojection_center_error(det_a: CameraDetection, det_b: CameraDetection,
                              cam_a: CameraFrame, cam_b: CameraFrame) -> float:
    """det_a의 3D centroid를 cam_b로 투영했을 때 det_b 중심과의 픽셀 거리."""
    if det_a.centroid_base is None or det_b.centroid_uv is None:
        return 1e9

    T_cam_b_base = np.linalg.inv(cam_b.T_base_cam)
    p_cam = (T_cam_b_base @ np.append(det_a.centroid_base, 1.0))[:3]

    if p_cam[2] <= 0.05:
        return 1e9

    K = cam_b.intrinsics.K
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]

    pred = np.array([u, v], dtype=np.float64)
    return float(np.linalg.norm(pred - det_b.centroid_uv))


def detection_pair_score_v2(det_a: CameraDetection, det_b: CameraDetection,
                            cam_a: CameraFrame, cam_b: CameraFrame) -> float:
    """Geometry-first detection pair scoring.

    centroid 거리 + reprojection consistency를 주 신호로 사용하고
    appearance/size는 tie-breaker로만 사용한다.
    """
    # 3D centroid 거리
    if det_a.centroid_base is not None and det_b.centroid_base is not None:
        dist = np.linalg.norm(det_a.centroid_base - det_b.centroid_base)
        pos_score = float(np.exp(-((dist / 0.04) ** 2)))
    else:
        pos_score = 0.0

    # reprojection consistency (양방향 중 최소)
    reproj_ab = reprojection_center_error(det_a, det_b, cam_a, cam_b)
    reproj_ba = reprojection_center_error(det_b, det_a, cam_b, cam_a)
    reproj = min(reproj_ab, reproj_ba)
    reproj_score = float(np.exp(-((reproj / 25.0) ** 2)))

    # appearance (tie-breaker)
    app_score = color_hist_similarity(det_a.color_hist, det_b.color_hist)
    size_score = extent_similarity(det_a.extents_base, det_b.extents_base)

    return 0.40 * pos_score + 0.35 * reproj_score + 0.15 * app_score + 0.10 * size_score


def build_track_observed_masks(track: ObjectTrack, frames):
    masks = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cam in frames:
        det = track.detections.get(cam.cam_id)
        if det is None:
            masks.append(np.zeros((cam.intrinsics.height, cam.intrinsics.width), dtype=np.uint8))
            continue
        mask = cv2.morphologyEx(det.mask_u8, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k, iterations=1)
        masks.append(mask)
    return masks


def infer_track_model_hint(track: ObjectTrack) -> Optional[str]:
    votes: Dict[str, float] = {}
    for det in track.detections.values():
        if not det.hint_model_name:
            continue
        votes[det.hint_model_name] = votes.get(det.hint_model_name, 0.0) + max(det.hint_score, 0.5)
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def track_hint_votes(detections: Dict[int, CameraDetection]) -> Dict[str, float]:
    votes: Dict[str, float] = {}
    for det in detections.values():
        if not det.hint_model_name:
            continue
        votes[det.hint_model_name] = votes.get(det.hint_model_name, 0.0) + max(det.hint_score, 0.5)
    return votes


def associate_detections_across_views(frames, detections_by_cam, dist_thresh_m=0.05):
    """멀티뷰 geometric association.

    같은 물체라면 base frame에서 3D centroid가 일관되어야 한다는 가정으로
    detection을 track으로 묶는다. appearance는 association의 핵심 신호에서
    제거하고 geometry를 우선 사용한다.
    """
    tracks: List[ObjectTrack] = []
    track_idx = 0

    for cam in frames:
        detections = detections_by_cam.get(cam.cam_id, [])
        detections = sorted(detections, key=lambda d: d.area_px, reverse=True)

        for det in detections:
            centroid = det.centroid_base
            best_track = None
            best_dist = np.inf

            if centroid is not None:
                for track in tracks:
                    if cam.cam_id in track.detections:
                        continue
                    if track.centroid_base is None:
                        continue
                    dist = np.linalg.norm(centroid - track.centroid_base)
                    if dist < dist_thresh_m and dist < best_dist:
                        best_dist = dist
                        best_track = track

            if best_track is None:
                track = ObjectTrack(
                    track_id=f"track_{track_idx:02d}",
                    detections={cam.cam_id: det},
                    centroids_base=[centroid] if centroid is not None else [],
                    centroid_base=centroid.copy() if centroid is not None else None,
                )
                tracks.append(track)
                track_idx += 1
                continue

            best_track.detections[cam.cam_id] = det
            if centroid is not None:
                best_track.centroids_base.append(centroid)
                best_track.centroid_base = np.mean(best_track.centroids_base, axis=0)

    for track in tracks:
        track.observed_masks = build_track_observed_masks(track, frames)
        track.hint_model_name = infer_track_model_hint(track)

    tracks.sort(
        key=lambda t: sum(det.area_px for det in t.detections.values()),
        reverse=True,
    )
    return tracks


def associate_detections_across_views_v2(frames, detections_by_cam):
    """Geometry-first 멀티뷰 association.

    centroid 거리 + reprojection consistency를 주 신호로 사용하고
    appearance는 tie-breaker로만 사용한다.
    """
    frame_by_id = {cam.cam_id: cam for cam in frames}

    all_dets = []
    for cam_id in sorted(detections_by_cam):
        all_dets.extend(detections_by_cam[cam_id])
    all_dets.sort(key=lambda d: d.area_px, reverse=True)

    used = set()
    tracks: List[ObjectTrack] = []
    track_idx = 0

    for det in all_dets:
        if det.det_id in used:
            continue

        members = {det.cam_id: det}
        used.add(det.det_id)

        for cam in frames:
            if cam.cam_id in members:
                continue

            best_det = None
            best_score = -1.0
            for cand in detections_by_cam.get(cam.cam_id, []):
                if cand.det_id in used:
                    continue
                votes = track_hint_votes(members)
                majority_hint = max(votes.items(), key=lambda kv: kv[1])[0] if votes else None
                if (
                    cand.hint_model_name is not None
                    and majority_hint is not None
                    and cand.hint_model_name != majority_hint
                ):
                    continue

                pair_scores = []
                for m in members.values():
                    if (
                        cand.hint_model_name is not None
                        and m.hint_model_name is not None
                        and cand.hint_model_name != m.hint_model_name
                    ):
                        continue
                    s = detection_pair_score_v2(
                        cand, m,
                        frame_by_id[cand.cam_id],
                        frame_by_id[m.cam_id],
                    )
                    pair_scores.append(s)

                score = float(np.mean(pair_scores)) if pair_scores else 0.0
                if score > best_score:
                    best_score = score
                    best_det = cand

            if best_det is not None and best_score >= 0.35:
                members[cam.cam_id] = best_det
                used.add(best_det.det_id)

        centroids = [
            det.centroid_base for det in members.values()
            if det.centroid_base is not None
        ]
        track = ObjectTrack(
            track_id=f"track_{track_idx:02d}",
            detections=members,
            centroids_base=centroids,
            centroid_base=np.mean(centroids, axis=0) if centroids else None,
        )
        track.observed_masks = build_track_observed_masks(track, frames)
        track.hint_model_name = infer_track_model_hint(track)
        tracks.append(track)
        track_idx += 1

    tracks.sort(
        key=lambda t: sum(det.area_px for det in t.detections.values()),
        reverse=True,
    )
    return tracks


def fuse_track_points(frames, track: ObjectTrack):
    return fuse_masked_points(frames, track.observed_masks)


def aggregate_track_mean_hsv(track: ObjectTrack):
    detections = list(track.detections.values())
    if not detections:
        return None

    weights = np.array([max(det.area_px, 1) for det in detections], dtype=np.float64)
    angles = np.array([det.mean_hsv[0] / 180.0 * (2.0 * np.pi) for det in detections], dtype=np.float64)
    hue = np.arctan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights))
    if hue < 0:
        hue += 2.0 * np.pi
    sat = float(np.average([det.mean_hsv[1] for det in detections], weights=weights))
    val = float(np.average([det.mean_hsv[2] for det in detections], weights=weights))
    return np.array([hue / (2.0 * np.pi) * 180.0, sat, val], dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# 4. 테이블 추정 + foreground 생성 + 멀티 오브젝트 클러스터링
# ═══════════════════════════════════════════════════════════

def estimate_table_plane(frames):
    """
    전체 depth를 base frame으로 합쳐서 테이블 평면 추정.

    Returns:
        plane_n: (3,) unit normal
        plane_d: float  -> signed distance plane: n^T x + d = 0
        table_center: (3,) 테이블 inlier 점군의 중심
        table_radius: float 테이블 inlier 점군의 수평 반경
    """
    all_pts = []
    for cam in frames:
        pts = backproject_depth(cam)
        if len(pts) == 0:
            continue
        all_pts.append(transform_points(pts, cam.T_base_cam))

    if not all_pts:
        raise RuntimeError("테이블 평면 추정 실패: 유효한 depth 없음")

    merged = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    pcd = pcd.voxel_down_sample(DEPTH_POLICY["voxel_size_m"])
    if len(pcd.points) < 3:
        raise RuntimeError("테이블 평면 추정 실패: 유효 점군 부족")

    plane, inlier_idx = pcd.segment_plane(
        distance_threshold=DEPTH_POLICY["table_dist_thresh_m"],
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        raise RuntimeError("테이블 평면 추정 실패: normal norm too small")
    n /= n_norm
    d /= n_norm

    # 테이블 inlier 점의 수평(XZ) 중심 · 반경 계산 → 배경 제거용
    table_pts = np.asarray(pcd.points)[inlier_idx]
    table_center = table_pts.mean(axis=0)
    horiz = np.array([table_pts[:, 0], table_pts[:, 2]]).T
    tc_horiz = np.array([table_center[0], table_center[2]])
    dists = np.linalg.norm(horiz - tc_horiz, axis=1)
    table_radius = np.percentile(dists, 90) * 1.1  # 90th percentile + 10% 여유

    return n, d, table_center, table_radius


def build_observed_mask(cam: CameraFrame, segmentation_model, score_thresh=0.5, min_area_px=200):
    """RGB instance segmentation 기반 물체별 mask 생성.

    Returns:
        List[np.ndarray]: object instance 별 binary mask
    """
    image = cam.color_bgr
    results = segmentation_model(image)

    masks: List[np.ndarray] = []
    for obj in results:
        if isinstance(obj, dict):
            mask = obj.get("mask")
            score = float(obj.get("score", 1.0))
        else:
            mask = obj
            score = 1.0

        if mask is None or score < score_thresh:
            continue

        mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8) * 255
        if int(np.count_nonzero(mask_u8)) < min_area_px:
            continue
        masks.append(mask_u8)

    return masks


def save_observed_masks(observed_masks_by_cam, frames, frame_id, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ci, (cam, masks) in enumerate(zip(frames, observed_masks_by_cam)):
        vis = cam.color_bgr.copy()
        for mi, mask in enumerate(masks):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = COLORS[mi % len(COLORS)]
            cv2.drawContours(vis, cnts, -1, color, 2)
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cv2.putText(
                    vis,
                    f"m{mi}",
                    (int(xs.mean()), max(20, int(ys.min()) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        cv2.putText(
            vis,
            f"cam{ci} observed masks ({len(masks)})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(out_dir / f"observed_mask_cam{ci}_{frame_id}.png"), vis)


def get_above_table_points(frames, table_n, table_d, table_center, table_radius):
    """전체 depth에서 테이블 위 물체 영역 점군만 추출."""
    dp = DEPTH_POLICY
    all_above = []
    for cam in frames:
        pts_cam = backproject_depth(cam)
        if len(pts_cam) == 0:
            continue
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        heights = -(np.dot(pts_base, table_n) + table_d)
        valid = ((heights > dp["object_min_height_m"])
                 & (heights < dp["object_max_height_m"]))
        horiz = np.linalg.norm(
            pts_base[:, [0, 2]] - table_center[[0, 2]], axis=1)
        valid &= horiz < table_radius
        if valid.sum() > 0:
            all_above.append(pts_base[valid])
    if not all_above:
        return np.zeros((0, 3), dtype=np.float64)
    merged = np.vstack(all_above)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    pcd = pcd.voxel_down_sample(dp["voxel_size_m"])
    return np.asarray(pcd.points)


def filter_detections_by_table(detections_by_cam, table_n, table_d,
                               table_center, table_radius):
    """Detection 중 테이블 위 물체 영역에 해당하는 것만 남긴다."""
    dp = DEPTH_POLICY
    filtered = {}
    for cam_id, dets in detections_by_cam.items():
        kept = []
        for det in dets:
            if det.centroid_base is None:
                continue
            h = -(np.dot(table_n, det.centroid_base) + table_d)
            if h < dp["object_min_height_m"] or h > dp["object_max_height_m"]:
                continue
            horiz = np.linalg.norm(
                det.centroid_base[[0, 2]] - table_center[[0, 2]])
            if horiz > table_radius:
                continue
            kept.append(det)
        filtered[cam_id] = kept
    return filtered


def supplement_with_depth_clusters(detections_by_cam, frames, above_pts,
                                   merge_dist_m=0.03):
    """Depth 클러스터링으로 RGB segmentation이 놓친 물체를 보충한다.

    above_pts에서 DBSCAN 클러스터를 찾고, 기존 detection과 겹치지 않는
    클러스터를 새 detection으로 추가한다.
    """
    clusters = find_all_clusters(above_pts)
    if not clusters:
        return detections_by_cam

    existing_centroids = []
    for dets in detections_by_cam.values():
        for det in dets:
            if det.centroid_base is not None:
                existing_centroids.append(det.centroid_base)

    new_by_cam = {cam_id: list(dets) for cam_id, dets in detections_by_cam.items()}

    for ci, (cpts, centroid) in enumerate(clusters):
        already = False
        for ec in existing_centroids:
            if np.linalg.norm(centroid - ec) < merge_dist_m:
                already = True
                break
        if already:
            continue

        for cam_id in new_by_cam:
            cam = None
            for f in frames:
                if f.cam_id == cam_id:
                    cam = f
                    break
            if cam is None:
                continue
            mask = project_points_to_mask(cpts, cam)
            if np.count_nonzero(mask) < 150:
                continue
            det = create_camera_detection(
                cam, mask, f"cam{cam_id}_depth_{ci}")
            if det is not None:
                new_by_cam[cam_id].append(det)
        existing_centroids.append(centroid)

    return new_by_cam


def max_model_extent(models: Dict[str, CanonicalModel]) -> float:
    """전체 GLB 모델 중 가장 큰 단일 축 크기 반환."""
    return max(m.extents_m.max() for m in models.values()) if models else 0.15


def try_split_cluster(cluster_pts, max_single_extent, min_pts=50):
    """Oversized 클러스터를 분할 시도.

    여러 물체가 한 클러스터에 합쳐진 경우 DBSCAN(작은 eps) 또는
    k-means로 분리한다.

    Returns: list of (sub_pts, sub_centroid)
    """
    ext = (cluster_pts.max(0) - cluster_pts.min(0)).max()

    # 분할 필요 없음
    if ext <= 1.20 * max_single_extent:
        return [(cluster_pts, cluster_pts.mean(0))]

    dp = DEPTH_POLICY
    target_max = 1.15 * max_single_extent

    def finalize_parts(parts):
        out = []
        for pts in parts:
            if len(pts) < min_pts:
                continue
            out.append((pts, pts.mean(0)))
        return out

    def valid_parts(parts):
        if len(parts) < 2:
            return False
        for pts in parts:
            if len(pts) < min_pts:
                return False
            sub_ext = (pts.max(0) - pts.min(0)).max()
            if sub_ext > target_max:
                return False
        return True

    # 시도 1: DBSCAN with progressively smaller eps
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_pts)
    for mult in [0.75, 0.60, 0.50, 0.40]:
        labels = np.array(pcd.cluster_dbscan(
            eps=max(dp["cluster_eps_m"] * mult, 0.0025),
            min_points=10,
            print_progress=False,
        ))
        if labels.max() < 1:
            continue
        parts = []
        for lbl in range(labels.max() + 1):
            mask = labels == lbl
            if mask.sum() < min_pts:
                continue
            parts.append(cluster_pts[mask])
        if valid_parts(parts):
            return finalize_parts(parts)

    # 시도 2: k-means (k=2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3)
    for axes in [(0, 2), (0, 1, 2)]:
        data = cluster_pts[:, axes].astype(np.float32)
        for k in [2, 3, 4]:
            if len(cluster_pts) < k * min_pts:
                continue
            try:
                _, km_labels, _ = cv2.kmeans(
                    data, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
            except cv2.error:
                continue
            km_labels = km_labels.reshape(-1)
            parts = []
            for ki in range(k):
                mask = km_labels == ki
                if mask.sum() < min_pts:
                    continue
                parts.append(cluster_pts[mask])
            if valid_parts(parts):
                return finalize_parts(parts)

    # 시도 3: 주축 기준 median split
    centered = cluster_pts - cluster_pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(cluster_pts), 1)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    proj = centered @ axis
    median = np.median(proj)
    left = cluster_pts[proj <= median]
    right = cluster_pts[proj > median]
    parts = [left, right]
    if valid_parts(parts):
        return finalize_parts(parts)

    return [(cluster_pts, cluster_pts.mean(0))]


def build_cluster_observed_masks(cluster_pts, frames, segmented_masks_by_cam=None):
    """클러스터 support mask를 RGB segmentation과 교차해 관측 mask를 만든다."""
    support_masks = project_cluster_to_support_masks(cluster_pts, frames)
    if segmented_masks_by_cam is None:
        return support_masks

    refined = []
    for ci, support in enumerate(support_masks):
        seg_masks = segmented_masks_by_cam[ci] if ci < len(segmented_masks_by_cam) else []
        if np.count_nonzero(support) == 0 or not seg_masks:
            refined.append(support)
            continue

        best_mask = None
        best_score = 0.0
        support_bin = support > 0
        support_area = max(int(support_bin.sum()), 1)

        for seg in seg_masks:
            seg_u8 = (seg > 0).astype(np.uint8) * 255
            inter = np.logical_and(support_bin, seg_u8 > 0).sum()
            if inter == 0:
                continue
            union = np.logical_or(support_bin, seg_u8 > 0).sum()
            iou = inter / max(union, 1)
            cover = inter / support_area
            score = 0.65 * cover + 0.35 * iou
            if score > best_score:
                best_score = score
                best_mask = seg_u8

        if best_mask is not None and best_score >= 0.18:
            refined.append(best_mask)
        else:
            refined.append(support)

    return refined


def glb_extent_compatible(cluster_pts, model: CanonicalModel,
                         scale_range=(0.7, 1.4)) -> bool:
    """클러스터 크기가 GLB 모델과 호환 가능한지 빠르게 검사."""
    c_ext = np.sort(cluster_pts.max(0) - cluster_pts.min(0))[::-1]
    m_ext = np.sort(model.extents_m)[::-1]
    c_ext_n = normalized_extent(c_ext)
    m_ext_n = normalized_extent(m_ext)
    if np.linalg.norm(c_ext_n - m_ext_n) > 0.95:
        return False
    for s in [0.85, 1.0, 1.15, scale_range[0], scale_range[1]]:
        ratios = c_ext / (m_ext * s + 1e-8)
        if np.all(ratios > 0.45) and np.all(ratios < 2.1):
            return True
    return False


def detect_objects_glb_first(frames, all_models: Dict[str, CanonicalModel],
                             table_n, table_d, table_center, table_radius,
                             runtime_symmetry=None,
                             segmented_masks_by_cam=None):
    """GLB-first 물체 인식.

    RGB segmentation 대신 depth clustering → GLB 형상 검증으로 물체를 찾는다.
    삼각대, 카메라, 가방 등 GLB에 없는 물체는 자동 제거된다.

    Returns:
        list of (cluster_pts, model_name, coarse_score, observed_masks)
    """
    # 1. 테이블 위 점군 추출 + 클러스터링
    above_pts = get_above_table_points(
        frames, table_n, table_d, table_center, table_radius)
    if len(above_pts) == 0:
        return []

    raw_clusters = find_all_clusters(above_pts)
    # Step 3: oversized 클러스터 분할
    max_ext = max_model_extent(all_models)
    clusters = []
    for cpts, cen in raw_clusters:
        clusters.extend(try_split_cluster(cpts, max_ext))
    print(f"  depth clusters: {len(raw_clusters)} → after split: {len(clusters)}")
    for ci, (cpts, cen) in enumerate(clusters):
        ext = cpts.max(0) - cpts.min(0)
        print(
            f"    cluster#{ci}: {len(cpts)} pts "
            f"extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm "
            f"center=[{cen[0]:+.3f},{cen[1]:+.3f},{cen[2]:+.3f}]"
        )

    # 2. 각 클러스터 × 각 GLB: extent 호환성 + coarse init + render score
    #    GLB와 형상이 안 맞는 클러스터(카메라, 삼각대 등)는 여기서 탈락
    candidates = []
    used_models = set()

    for ci, (cpts, centroid) in enumerate(clusters):
        # 클러스터별 observed mask 생성 (RGB segmentation과 교차)
        obs_masks = build_cluster_observed_masks(
            cpts, frames, segmented_masks_by_cam=segmented_masks_by_cam)

        best_name = None
        best_score = -1.0
        best_all_scores = []

        for name, model in all_models.items():
            # 빠른 크기 호환성 검사
            if not glb_extent_compatible(cpts, model):
                continue

            # coarse init → RGB render + depth score
            pose_init = coarse_init(cpts, model)
            render_sc = rgb_compare_score(
                model, pose_init.T_base_obj, frames, obs_masks,
                scale=pose_init.scale)
            model_pts = model.mesh.vertices - model.center
            aligned = transform_points(model_pts * pose_init.scale, pose_init.T_base_obj)
            depth_sc = mv_depth_score(aligned, frames, obs_masks)

            # extent 비교
            c_ext_n = normalized_extent(cpts.max(0) - cpts.min(0))
            m_ext_n = normalized_extent(model.extents_m)
            ext_sc = 1.0 / (1.0 + 4.0 * np.linalg.norm(c_ext_n - m_ext_n))

            score = 0.40 * render_sc + 0.35 * depth_sc + 0.25 * ext_sc
            best_all_scores.append((name, score, render_sc, depth_sc, ext_sc))

            if score > best_score:
                best_score = score
                best_name = name

        # GLB match 점수가 너무 낮으면 이 클러스터는 블록이 아님 → 건너뜀
        MIN_GLB_MATCH = 0.10
        if best_name is None or best_score < MIN_GLB_MATCH:
            print(f"    cluster#{ci}: NO GLB match (score={best_score:.3f}) → skip (카메라/삼각대 등)")
            continue

        best_all_scores.sort(key=lambda x: x[1], reverse=True)
        top = best_all_scores[0]
        print(
            f"    cluster#{ci}: best={top[0]} score={top[1]:.3f} "
            f"(render={top[2]:.3f} depth={top[3]:.3f} ext={top[4]:.3f})"
        )
        if len(best_all_scores) > 1:
            r2 = best_all_scores[1]
            print(f"      runner-up={r2[0]} score={r2[1]:.3f}")

        candidates.append((cpts, best_all_scores, obs_masks))

    # 3. greedy assignment: score 순으로 정렬 → 모델 중복 방지
    all_assignments = []
    for cpts, scores, obs_masks in candidates:
        for name, score, _, _, _ in scores:
            all_assignments.append((score, cpts, name, obs_masks))

    all_assignments.sort(key=lambda x: x[0], reverse=True)

    results = []
    used_models = set()
    used_clusters = set()

    for score, cpts, name, obs_masks in all_assignments:
        cid = id(cpts)
        if name in used_models or cid in used_clusters:
            continue
        results.append((cpts, name, score, obs_masks))
        used_models.add(name)
        used_clusters.add(cid)

    return results


def fuse_masked_points(frames, observed_masks):
    """cam별 observed mask 내부 depth만 3D화해서 base frame으로 합친다."""
    all_pts = []
    for cam, mask in zip(frames, observed_masks):
        pts = backproject_depth(cam, mask=mask)
        if len(pts) == 0:
            continue
        pts_base = transform_points(pts, cam.T_base_cam)
        all_pts.append(pts_base)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float64)

    merged = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    pcd = pcd.voxel_down_sample(DEPTH_POLICY["voxel_size_m"])

    if len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return np.asarray(pcd.points)


def find_all_clusters(above_pts):
    """
    테이블 위 점군 → DBSCAN → 유효 크기의 모든 클러스터 반환.

    Returns: list of (cluster_pts, centroid)
    """
    dp = DEPTH_POLICY
    if len(above_pts) < dp["cluster_min_pts"]:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(above_pts)
    labels = np.array(pcd.cluster_dbscan(
        eps=dp["cluster_eps_m"], min_points=10, print_progress=False))

    if labels.max() < 0:
        return []

    clusters = []
    for lbl in range(labels.max() + 1):
        mask = labels == lbl
        n = mask.sum()
        if n < dp["cluster_min_pts"]:
            continue
        pts = above_pts[mask]
        ext = pts.max(0) - pts.min(0)
        max_ext = ext.max()
        min_ext = ext.min()
        if max_ext < dp["cluster_min_extent_m"] or max_ext > dp["cluster_max_extent_m"]:
            continue
        if min_ext < 0.005:
            continue

        # 아웃라이어 제거
        c_pcd = o3d.geometry.PointCloud()
        c_pcd.points = o3d.utility.Vector3dVector(pts)
        if len(pts) > 50:
            c_pcd, _ = c_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pts = np.asarray(c_pcd.points)

        clusters.append((pts, pts.mean(axis=0)))

    # 크기(점 수) 순 정렬
    clusters.sort(key=lambda x: len(x[0]), reverse=True)
    return clusters


def find_single_object(above_pts):
    """단일 물체: 밀도 시드 기반 가장 밀집된 클러스터 1개 반환."""
    dp = DEPTH_POLICY
    if len(above_pts) < 30:
        return above_pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(above_pts)
    kd = o3d.geometry.KDTreeFlann(pcd)

    rng = np.random.default_rng(42)
    n_sample = min(len(above_pts), 3000)
    sample_idx = rng.choice(len(above_pts), n_sample, replace=False)

    best_d, seed_pt = 0, above_pts[0]
    for si in sample_idx:
        [_, idx, _] = kd.search_radius_vector_3d(above_pts[si], 0.03)
        if len(idx) > best_d:
            best_d = len(idx); seed_pt = above_pts[si]

    nearby = above_pts[np.linalg.norm(above_pts - seed_pt, axis=1) < 0.10]
    if len(nearby) < 30:
        return nearby

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(nearby)
    labels = np.array(pcd2.cluster_dbscan(eps=dp["cluster_eps_m"], min_points=10, print_progress=False))
    if labels.max() < 0:
        return nearby

    sl = np.argmin(np.linalg.norm(nearby - seed_pt, axis=1))
    seed_label = labels[sl]
    if seed_label >= 0:
        pts = nearby[labels == seed_label]
    else:
        u, c = np.unique(labels[labels>=0], return_counts=True)
        pts = nearby[labels == u[np.argmax(c)]] if len(u) > 0 else nearby

    if len(pts) > 50:
        p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(pts)
        p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pts = np.asarray(p.points)
    return pts


def project_cluster_to_support_masks(obj_pts, frames):
    masks = []
    for cam in frames:
        h, w = cam.intrinsics.height, cam.intrinsics.width
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(obj_pts) == 0:
            masks.append(mask)
            continue
        T_inv = np.linalg.inv(cam.T_base_cam)
        p = transform_points(obj_pts, T_inv)
        front = p[:, 2] > 0.05
        p = p[front]
        if len(p) == 0:
            masks.append(mask)
            continue
        K = cam.intrinsics.K
        u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
        v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        mask[v[ok], u[ok]] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        masks.append(mask)
    return masks


def refine_eval_masks_with_support(observed_masks, support_masks):
    """Track mask와 3D support mask를 교차해 정합용 mask를 만든다.

    track 관측 mask가 너무 넓게 잡힌 경우 support로 줄이고,
    support가 빈 경우에는 원래 observed mask를 유지한다.
    """
    refined = []
    for obs_mask, sup_mask in zip(observed_masks, support_masks):
        obs_area = int(np.count_nonzero(obs_mask))
        sup_area = int(np.count_nonzero(sup_mask))
        if obs_area == 0:
            refined.append(np.zeros_like(obs_mask))
            continue
        if sup_area == 0:
            refined.append(obs_mask.copy())
            continue

        inter = cv2.bitwise_and(obs_mask, sup_mask)
        inter_area = int(np.count_nonzero(inter))
        min_keep = max(80, int(0.08 * min(obs_area, sup_area)))
        refined.append(inter if inter_area >= min_keep else obs_mask.copy())
    return refined


def build_object_tracks(frames, segmentation_model,
                        table_n, table_d, table_center, table_radius,
                        out_dir: Path, frame_id: str):
    """Instance separation → cross-view association → track 생성."""
    detections_by_cam = {
        cam.cam_id: build_camera_instances(cam, segmentation_model)
        for cam in frames
    }
    segmented_masks_by_cam = [
        [det.mask_u8 for det in detections_by_cam.get(cam.cam_id, [])]
        for cam in frames
    ]
    save_observed_masks(segmented_masks_by_cam, frames, frame_id, out_dir)
    detections_by_cam = filter_detections_by_table(
        detections_by_cam, table_n, table_d, table_center, table_radius
    )

    above_pts = get_above_table_points(frames, table_n, table_d, table_center, table_radius)
    rgb_det_counts = [len(detections_by_cam.get(cam.cam_id, [])) for cam in frames]
    rgb_detection_is_sufficient = sum(c >= 2 for c in rgb_det_counts) >= 2
    if not rgb_detection_is_sufficient:
        detections_by_cam = supplement_with_depth_clusters(
            detections_by_cam, frames, above_pts
        )
        detections_by_cam = filter_detections_by_table(
            detections_by_cam, table_n, table_d, table_center, table_radius
        )
    save_camera_detections(detections_by_cam, frames, frame_id, out_dir)

    tracks = associate_detections_across_views_v2(frames, detections_by_cam)
    return segmented_masks_by_cam, detections_by_cam, tracks, above_pts


def build_single_object_masks(frames, segmented_masks_by_cam, single_pts):
    combined_masks = [
        combine_instance_masks(masks, (cam.intrinsics.height, cam.intrinsics.width))
        for cam, masks in zip(frames, segmented_masks_by_cam)
    ]
    support_masks = project_cluster_to_support_masks(single_pts, frames)
    eval_masks = []
    for obs_mask, sup_mask in zip(combined_masks, support_masks):
        obs_area = int(np.count_nonzero(obs_mask))
        sup_area = int(np.count_nonzero(sup_mask))
        if sup_area == 0 and obs_area == 0:
            eval_masks.append(np.zeros_like(obs_mask))
            continue
        if sup_area == 0:
            eval_masks.append(obs_mask.copy())
            continue
        if obs_area == 0:
            eval_masks.append(sup_mask.copy())
            continue

        inter = cv2.bitwise_and(obs_mask, sup_mask)
        inter_area = int(np.count_nonzero(inter))
        if inter_area >= max(60, int(0.10 * sup_area)):
            eval_masks.append(inter)
        elif obs_area > 2 * sup_area:
            eval_masks.append(sup_mask.copy())
        else:
            eval_masks.append(obs_mask.copy())
    return eval_masks


def rank_single_object_points(above_pts, masked_pts, model: CanonicalModel):
    """단일 물체 점군 후보를 평가.

    3가지 소스:
    1. masked: RGB segmentation mask 기반 (채도 높은 물체에 유리)
    2. above: 테이블 위 전체에서 밀도 기반 추출
    3. depth_cluster: DBSCAN 클러스터 중 GLB 크기에 가장 맞는 것 (색 무관, 민트 등에 강건)
    """
    candidates = []

    def score_pts(pts, source_name):
        if len(pts) == 0:
            return
        ext = pts.max(0) - pts.min(0)
        ext_score = 1.0 / (1.0 + 3.0 * np.linalg.norm(
            normalized_extent(ext) - normalized_extent(model.extents_m)
        ))
        scale_ratio = float(np.max(ext) / (np.max(model.extents_m) + 1e-8))
        scale_penalty = np.exp(-((scale_ratio - 1.0) / 0.45) ** 2)
        point_bonus = min(1.0, len(pts) / 2500.0)
        total = 0.55 * ext_score + 0.30 * scale_penalty + 0.15 * point_bonus
        candidates.append((total, source_name, pts))

    # 소스 1: RGB masked
    if len(masked_pts) > 0:
        score_pts(find_single_object(masked_pts), "masked")

    # 소스 2: 테이블 위 전체 밀도 기반
    if len(above_pts) > 0:
        score_pts(find_single_object(above_pts), "above")

    # 소스 3: depth-only DBSCAN — GLB 크기와 가장 잘 맞는 클러스터 직접 선택
    #   RGB segmentation과 무관하므로 민트 등 저채도 물체에 강건
    #   부분 관측(한 면만 보임)을 고려해서 per-axis 비율로 평가
    if len(above_pts) > 50:
        clusters = find_all_clusters(above_pts)
        model_ext_sorted = np.sort(model.extents_m)[::-1]
        best_cluster_score = -1.0
        best_cluster = None
        for cpts, _ in clusters:
            if len(cpts) < 80:
                continue
            ext = cpts.max(0) - cpts.min(0)
            ext_sorted = np.sort(ext)[::-1]
            # 가장 큰 축이 GLB 가장 큰 축의 1.2배를 넘으면 oversized
            if ext_sorted[0] > model_ext_sorted[0] * 1.2:
                continue
            ext_n = normalized_extent(ext)
            mod_n = normalized_extent(model.extents_m)
            cs = 1.0 / (1.0 + 3.0 * np.linalg.norm(ext_n - mod_n))
            pt_bonus = min(1.0, len(cpts) / 2000.0)
            sc = 0.7 * cs + 0.3 * pt_bonus
            if sc > best_cluster_score:
                best_cluster_score = sc
                best_cluster = cpts
        if best_cluster is not None:
            score_pts(best_cluster, "depth_cluster")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def split_track_points(track_pts, max_single_extent):
    """Track별 점군을 object-sized subcluster로 분리."""
    if len(track_pts) == 0:
        return []
    subclusters = try_split_cluster(track_pts, max_single_extent, min_pts=40)
    if not subclusters:
        return [(track_pts, track_pts.mean(axis=0))]
    subclusters.sort(key=lambda x: len(x[0]), reverse=True)
    return subclusters


def make_pose_estimate(T_base_obj: np.ndarray, scale: float = 1.0, confidence: float = 0.0):
    U, _, Vt = np.linalg.svd(T_base_obj[:3, :3])
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    T_base_obj = T_base_obj.copy()
    T_base_obj[:3, :3] = R_clean
    r_ = Rot.from_matrix(R_clean)
    return PoseEstimate(
        T_base_obj=T_base_obj.copy(),
        position_m=T_base_obj[:3, 3].copy(),
        quaternion_xyzw=r_.as_quat(),
        euler_xyz_deg=r_.as_euler("xyz", degrees=True),
        scale=scale,
        confidence=confidence,
    )


def coarse_init(cluster_pts, model: CanonicalModel):
    model_pts = model.mesh.vertices - model.center
    cluster_center = cluster_pts.mean(axis=0)
    cluster_ext_n = normalized_extent(cluster_pts.max(0) - cluster_pts.min(0))
    src_mean = model_pts.mean(axis=0)

    rots = rotation_candidates(model_pts, cluster_pts)
    if not rots:
        rots = [np.eye(3)]

    best_R = rots[0]
    best_err = np.inf
    for R in rots[:8]:
        rp = (R @ model_pts.T).T
        err = np.linalg.norm(normalized_extent(rp.max(0) - rp.min(0)) - cluster_ext_n)
        if err < best_err:
            best_err = err
            best_R = R

    T = np.eye(4)
    T[:3, :3] = best_R
    T[:3, 3] = cluster_center - best_R @ src_mean
    return make_pose_estimate(T, scale=1.0)


def render_model_to_mask(model: CanonicalModel, pose: PoseEstimate, cam: CameraFrame):
    h, w = cam.intrinsics.height, cam.intrinsics.width
    mask = np.zeros((h, w), dtype=np.uint8)

    mesh = model.mesh_low if model.mesh_low is not None else model.mesh
    verts = (mesh.vertices - model.center) * pose.scale
    verts_base = transform_points(verts, pose.T_base_obj)
    verts_cam = transform_points(verts_base, np.linalg.inv(cam.T_base_cam))

    K = cam.intrinsics.K
    z = verts_cam[:, 2]
    uv = np.zeros((len(verts_cam), 2), dtype=np.float64)
    front = z > 0.05
    uv[front, 0] = K[0, 0] * verts_cam[front, 0] / z[front] + K[0, 2]
    uv[front, 1] = K[1, 1] * verts_cam[front, 1] / z[front] + K[1, 2]

    faces = np.asarray(mesh.faces)
    for tri in faces:
        if not np.all(front[tri]):
            continue
        poly = np.round(uv[tri]).astype(np.int32)
        if np.any(poly[:, 0] < -w) or np.any(poly[:, 0] > 2 * w):
            continue
        if np.any(poly[:, 1] < -h) or np.any(poly[:, 1] > 2 * h):
            continue
        cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def render_silhouette_score(model: CanonicalModel, pose: PoseEstimate, frames, observed_masks):
    total = 0.0
    count = 0
    for cam, obs_mask in zip(frames, observed_masks):
        if np.count_nonzero(obs_mask) == 0:
            continue
        rendered = render_model_to_mask(model, pose, cam) > 0
        observed = obs_mask > 0
        union = np.logical_or(rendered, observed).sum()
        if union == 0:
            continue
        inter = np.logical_and(rendered, observed).sum()
        total += inter / union
        count += 1
    return total / max(count, 1)


def render_compare_score(model: CanonicalModel, T_base_obj: np.ndarray,
                         frames, observed_masks, scale: float = 1.0) -> float:
    """GLB mesh를 T_base_obj로 투영해서 observed mask와의 IoU를 계산.

    PoseEstimate 없이 raw transform만으로 GLB-aware score를 평가할 수 있어
    shortlist 단계에서 coarse init 직후 바로 사용 가능하다.
    """
    mesh = model.mesh_low if model.mesh_low is not None else model.mesh
    verts = (mesh.vertices - model.center) * scale
    verts_base = transform_points(verts, T_base_obj)

    scores = []
    faces = np.asarray(mesh.faces)
    for cam, obs_mask in zip(frames, observed_masks):
        h, w = cam.intrinsics.height, cam.intrinsics.width
        mask = np.zeros((h, w), dtype=np.uint8)

        verts_cam = transform_points(verts_base, np.linalg.inv(cam.T_base_cam))
        z = verts_cam[:, 2]
        front = z > 0.05
        if front.sum() < 3:
            continue

        K = cam.intrinsics.K
        uv = np.zeros((len(verts_cam), 2), dtype=np.float64)
        uv[front, 0] = K[0, 0] * verts_cam[front, 0] / z[front] + K[0, 2]
        uv[front, 1] = K[1, 1] * verts_cam[front, 1] / z[front] + K[1, 2]

        for tri in faces:
            if not np.all(front[tri]):
                continue
            poly = np.round(uv[tri]).astype(np.int32)
            if np.any(poly[:, 0] < -w) or np.any(poly[:, 0] > 2 * w):
                continue
            if np.any(poly[:, 1] < -h) or np.any(poly[:, 1] > 2 * h):
                continue
            cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)

        pred = mask > 0
        obs = obs_mask > 0
        union = np.logical_or(pred, obs).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred, obs).sum()
        scores.append(float(inter / union))

    return float(np.mean(scores)) if scores else 0.0


def render_model_to_rgb(model: CanonicalModel, T_base_obj: np.ndarray,
                        cam: CameraFrame, scale: float = 1.0):
    """GLB mesh를 카메라 이미지 평면에 렌더링.

    face loop 대신 vertex color + convex hull로 빠르게 렌더링.
    Returns: (rendered_bgr, rendered_mask)
    """
    h, w = cam.intrinsics.height, cam.intrinsics.width
    rendered_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    rendered_mask = np.zeros((h, w), dtype=np.uint8)

    mesh = model.mesh_low if model.mesh_low is not None else model.mesh
    verts = (mesh.vertices - model.center) * scale
    verts_base = transform_points(verts, T_base_obj)
    verts_cam = transform_points(verts_base, np.linalg.inv(cam.T_base_cam))

    K = cam.intrinsics.K
    z = verts_cam[:, 2]
    front = z > 0.05
    if front.sum() < 3:
        return rendered_bgr, rendered_mask

    u = (K[0, 0] * verts_cam[front, 0] / z[front] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * verts_cam[front, 1] / z[front] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    if ok.sum() < 3:
        return rendered_bgr, rendered_mask

    # convex hull mask
    pts_2d = np.stack([u[ok], v[ok]], axis=1)
    hull = cv2.convexHull(pts_2d)
    cv2.fillConvexPoly(rendered_mask, hull, 255)

    # vertex color → 평균 색상으로 mask 영역 채우기
    try:
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3]  # (N, 3) RGB
        front_colors = vc[front][ok]
        mean_rgb = front_colors.mean(axis=0).astype(np.uint8)
        mean_bgr = (int(mean_rgb[2]), int(mean_rgb[1]), int(mean_rgb[0]))
    except Exception:
        mean_bgr = (128, 128, 128)

    rendered_bgr[rendered_mask > 0] = mean_bgr

    return rendered_bgr, rendered_mask


def rgb_compare_score(model: CanonicalModel, T_base_obj: np.ndarray,
                      frames, observed_masks, scale: float = 1.0,
                      best_cam_only: bool = False) -> float:
    """GLB를 렌더링한 RGB와 실제 카메라 RGB를 비교.

    mask IoU + 색상 유사도를 결합한 점수를 반환한다.
    best_cam_only=True이면 가장 큰 mask의 카메라 1대만 사용 (속도 3배).
    """
    cam_mask_pairs = list(zip(frames, observed_masks))
    if best_cam_only and len(cam_mask_pairs) > 1:
        best_idx = max(range(len(cam_mask_pairs)),
                       key=lambda i: np.count_nonzero(cam_mask_pairs[i][1]))
        cam_mask_pairs = [cam_mask_pairs[best_idx]]

    scores = []
    for cam, obs_mask in cam_mask_pairs:
        rendered_bgr, rendered_mask = render_model_to_rgb(
            model, T_base_obj, cam, scale)

        pred = rendered_mask > 0
        obs = obs_mask > 0
        pred_area = pred.sum()
        if pred_area < 10:
            continue

        # mask IoU
        union = np.logical_or(pred, obs).sum()
        mask_iou = float(np.logical_and(pred, obs).sum() / max(union, 1))

        # 색상 유사도: 렌더링 영역 내에서 실제 RGB와 비교
        overlap = pred  # 렌더링된 영역에서 비교
        n_overlap = overlap.sum()
        if n_overlap < 10:
            scores.append(0.3 * mask_iou)
            continue

        # HSV로 변환해서 색상 비교
        rendered_hsv = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2HSV)
        actual_hsv = cv2.cvtColor(cam.color_bgr, cv2.COLOR_BGR2HSV)

        r_pix = rendered_hsv[overlap].astype(np.float32)
        a_pix = actual_hsv[overlap].astype(np.float32)

        # hue 거리 (circular, 0-180 range)
        hue_diff = np.abs(r_pix[:, 0] - a_pix[:, 0])
        hue_diff = np.minimum(hue_diff, 180.0 - hue_diff) / 90.0
        hue_sim = float(1.0 - np.mean(hue_diff))

        # saturation + value 거리
        sv_diff = np.abs(r_pix[:, 1:] - a_pix[:, 1:]) / 255.0
        sv_sim = float(1.0 - np.mean(sv_diff))

        color_sim = 0.6 * hue_sim + 0.4 * sv_sim
        color_sim = max(0.0, min(1.0, color_sim))

        scores.append(0.4 * mask_iou + 0.6 * color_sim)

    return float(np.mean(scores)) if scores else 0.0


# ═══════════════════════════════════════════════════════════
# 5. GLB 형상 매칭 (멀티 오브젝트용)
# ═══════════════════════════════════════════════════════════

def pca_descriptor(pts):
    c = pts - pts.mean(0)
    vals = np.sqrt(np.maximum(np.linalg.eigvalsh(c.T @ c / len(pts)), 0))
    vals = np.sort(vals)[::-1]
    return vals / vals[0] if vals[0] > 0 else vals

def model_appearance_score(model_name: str, track_mean_hsv: Optional[np.ndarray],
                           color_priors: Optional[Dict] = None) -> float:
    """색상 prior가 있으면 사용, 없으면 0.5(neutral) 반환.

    color_priors가 주어지면 해당 dict에서 조회하고,
    None이면 글로벌 OBJECT_COLOR_PRIORS_HSV를 fallback으로 사용한다.
    새 물체(prior 미등록)는 자동으로 shape-only 평가로 전환된다.
    """
    priors = color_priors if color_priors is not None else OBJECT_COLOR_PRIORS_HSV
    prior = priors.get(model_name)
    if prior is None or track_mean_hsv is None:
        return 0.5
    hue_score = 1.0 - min(1.0, circular_hue_distance(track_mean_hsv[0], prior[0]))
    sat_score = 1.0 - min(1.0, abs(float(track_mean_hsv[1]) - float(prior[1])) / 180.0)
    val_score = 1.0 - min(1.0, abs(float(track_mean_hsv[2]) - float(prior[2])) / 180.0)
    return float(np.clip(0.60 * hue_score + 0.25 * sat_score + 0.15 * val_score, 0.0, 1.0))


def shortlist_glb_candidates(cluster_pts, models: Dict[str, CanonicalModel], frames,
                             observed_masks, top_k=3):
    """GLB-aware shortlist.

    coarse init 후 실제 GLB silhouette/depth 일관성을 평가해서 후보를 고른다.
    """
    obj_ext = cluster_pts.max(0) - cluster_pts.min(0)
    obj_ext_n = normalized_extent(obj_ext)

    results = []
    for name, model in models.items():
        pose_init = coarse_init(cluster_pts, model)
        model_pts = model.mesh.vertices - model.center
        aligned_init = transform_points(model_pts * pose_init.scale, pose_init.T_base_obj)

        sil_score = render_silhouette_score(model, pose_init, frames, observed_masks)
        depth_score = mv_depth_score(aligned_init, frames, observed_masks)
        ext_score = 1.0 / (
            1.0 + 4.0 * np.linalg.norm(obj_ext_n - normalized_extent(model.extents_m))
        )

        coarse_score = 0.50 * sil_score + 0.35 * depth_score + 0.15 * ext_score
        results.append((name, coarse_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def shortlist_glb_candidates_v2(cluster_pts, models: Dict[str, CanonicalModel],
                                frames, observed_masks, top_k=3,
                                track_mean_hsv=None,
                                color_priors: Optional[Dict] = None):
    """Render-aware GLB shortlist.

    coarse init → 실제 GLB silhouette 투영 + depth + shape + appearance를 종합.
    color_priors가 없는 모델은 shape/render만으로 평가 (새 물체 대응).
    """
    results = []

    for name, model in models.items():
        # Shape descriptor 비교
        shape_score = 0.0
        try:
            obj_desc = pca_descriptor(cluster_pts)
            mod_desc = pca_descriptor(model.mesh.vertices - model.center)
            desc_dist = np.linalg.norm(obj_desc - mod_desc)

            obj_ext = cluster_pts.max(0) - cluster_pts.min(0)
            obj_ext_n = normalized_extent(obj_ext)
            mod_ext_n = normalized_extent(model.extents_m)
            ext_dist = np.linalg.norm(obj_ext_n - mod_ext_n)

            shape_score = 1.0 / (1.0 + 8.0 * desc_dist + 4.0 * ext_dist)
        except Exception:
            pass

        # Appearance (색 prior가 있으면 사용, 없으면 shape-only)
        appearance_score = model_appearance_score(name, track_mean_hsv,
                                                 color_priors=color_priors)

        # Coarse init → render silhouette IoU
        pose_init = coarse_init(cluster_pts, model)
        render_score = render_compare_score(
            model, pose_init.T_base_obj, frames, observed_masks,
            scale=pose_init.scale,
        )

        # Depth consistency
        model_pts = model.mesh.vertices - model.center
        aligned_init = transform_points(model_pts * pose_init.scale, pose_init.T_base_obj)
        depth_score = mv_depth_score(aligned_init, frames, observed_masks)

        # 종합 (render가 가장 직접적인 신호)
        coarse = (0.35 * render_score
                  + 0.25 * depth_score
                  + 0.25 * shape_score
                  + 0.15 * appearance_score)
        results.append((name, coarse))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ═══════════════════════════════════════════════════════════
# 6. PCA 초기화 + ICP 정합
# ═══════════════════════════════════════════════════════════

def pca_axes(pts):
    c = pts - pts.mean(0)
    vals, vecs = np.linalg.eigh(c.T @ c / len(pts))
    order = np.argsort(vals)[::-1]
    return vecs[:, order], np.sqrt(np.maximum(vals[order], 0))

def rotation_candidates(src_pts, tgt_pts):
    R_s, _ = pca_axes(src_pts); R_t, _ = pca_axes(tgt_pts)
    cands = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1,-1], repeat=3):
            R = np.zeros((3,3))
            for i in range(3): R[:,i] = signs[i]*R_s[:,perm[i]]
            if np.linalg.det(R) < 0: continue
            cands.append(R_t @ R.T)
    unique = [cands[0]]
    for R in cands[1:]:
        if all(np.arccos(np.clip((np.trace(R@Ru.T)-1)/2,-1,1)) > np.radians(5) for Ru in unique):
            unique.append(R)
    return unique


def combine_pose_scores(model_name: str, depth_score: float,
                        coverage: float, silhouette: float,
                        rgb_sc: float = 0.0,
                        symmetry_map: Optional[Dict[str, str]] = None) -> float:
    """Symmetry-aware pose score 합산 (RGB score 포함).

    rgb_sc가 유효하면 silhouette 대신 RGB score를 사용한다.
    RGB는 색상+형상을 동시에 검증하므로 silhouette보다 강한 신호.
    """
    sym_dict = symmetry_map if symmetry_map is not None else OBJECT_SYMMETRY
    sym = sym_dict.get(model_name, "none")
    if rgb_sc > 0.01:
        if sym == "yaw":
            return 0.35 * depth_score + 0.15 * coverage + 0.50 * rgb_sc
        return 0.30 * depth_score + 0.25 * coverage + 0.45 * rgb_sc
    if sym == "yaw":
        return 0.50 * depth_score + 0.20 * coverage + 0.30 * silhouette
    return 0.45 * depth_score + 0.35 * coverage + 0.20 * silhouette


def infer_symmetry_from_mesh(model: CanonicalModel) -> str:
    """PCA eigenvalue ratio로 축 대칭(실린더 등) 여부를 추정."""
    pts = model.mesh.vertices - model.center
    cov = pts.T @ pts / len(pts)
    vals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    if vals[0] < 1e-12:
        return "none"
    ratio = vals[1] / vals[0]
    if ratio > 0.85:
        return "yaw"
    return "none"


def generate_pose_perturbations(T_base_obj: np.ndarray,
                                n_candidates: int = 48,
                                trans_range_m: float = 0.02,
                                rot_range_deg: float = 15.0) -> List[np.ndarray]:
    """초기 pose 주변에서 translation/rotation을 살짝 변형한 후보를 생성."""
    rng = np.random.default_rng(42)
    candidates = [T_base_obj.copy()]
    for _ in range(n_candidates):
        dt = rng.uniform(-trans_range_m, trans_range_m, 3)
        dr = rng.uniform(-rot_range_deg, rot_range_deg, 3)
        dR = Rot.from_euler("xyz", dr, degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = dR @ T_base_obj[:3, :3]
        T[:3, 3] = T_base_obj[:3, 3] + dt
        candidates.append(T)
    return candidates


def local_pose_search(model: CanonicalModel, frames, observed_masks,
                      T_init: np.ndarray, scale: float = 1.0,
                      n_candidates: int = 16, top_k: int = 3) -> List[np.ndarray]:
    """Coarse init 주변에서 perturbation → RGB score(1-cam fast) → 상위 top_k.

    ICP 전에 더 나은 초기값을 찾아주는 역할. 속도를 위해 1-cam만 사용.
    """
    candidates = generate_pose_perturbations(
        T_init, n_candidates=n_candidates)
    scored = []
    for T_cand in candidates:
        sc = rgb_compare_score(model, T_cand, frames, observed_masks, scale,
                               best_cam_only=True)
        scored.append((sc, T_cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [T for _, T in scored[:top_k]]


def register_table_grounded(model: CanonicalModel, model_pts: np.ndarray,
                            obj_pts: np.ndarray, frames,
                            observed_masks, table_n: np.ndarray, table_d: float,
                            model_name: str = "", symmetry_map=None):
    """GLB를 테이블 위에 세워놓고 yaw만 탐색.

    1. 점군 centroid의 XZ (수평) 위치 사용
    2. GLB 바닥을 테이블 면 높이에 맞춤
    3. GLB Z축(높이)을 테이블 법선 방향에 정렬
    4. yaw 360° 탐색 → depth 비교 → best → ICP fine-tune
    """
    # --- 좌표 정리 ---
    up = -table_n  # 테이블 위 방향
    up = up / (np.linalg.norm(up) + 1e-8)

    # GLB에서 높이 축 = Z, 바닥 Z값
    verts_centered = model.mesh.vertices - model.center
    glb_z_min = float(verts_centered[:, 2].min())  # 바닥

    # 점군의 수평 centroid
    centroid = obj_pts.mean(axis=0)

    # 테이블 면 위 높이: n·p + d = 0 → 테이블 면의 점
    # 물체 바닥 = 테이블 면 높이
    table_height_at_centroid = -(np.dot(table_n, centroid) + table_d)
    # 테이블 면 위의 점 = centroid를 테이블 면에 투영
    table_point = centroid - (np.dot(table_n, centroid) + table_d) * table_n

    # --- rotation: GLB Z축 → base up 방향 ---
    glb_up = np.array([0, 0, 1], dtype=np.float64)
    # rotation from glb_up to base up
    v = np.cross(glb_up, up)
    c = np.dot(glb_up, up)
    if np.linalg.norm(v) < 1e-6:
        R_align = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx / (1 + c)

    # --- translation: GLB 바닥을 테이블 면에 맞춤 ---
    # GLB center가 R_align 적용 후 base 좌표계에서의 위치
    # 바닥 offset = R_align @ [0, 0, glb_z_min]
    bottom_offset = R_align @ np.array([0, 0, glb_z_min])

    # --- yaw sweep ---
    n_yaw = 72  # 5° 간격
    scales = [0.88, 0.94, 1.0, 1.06, 1.12]
    all_candidates = []

    print(f"    table-grounded search: {n_yaw} yaw × {len(scales)} scales = {n_yaw * len(scales)} candidates")

    for scale in scales:
        for yi in range(n_yaw):
            yaw_rad = yi * 2 * np.pi / n_yaw
            R_yaw = Rot.from_rotvec(yaw_rad * up).as_matrix()

            R = R_yaw @ R_align
            # translation: 물체 bottom을 table_point에 놓음
            t = table_point - R @ (model.center * scale) - R_yaw @ (bottom_offset * scale)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            score = render_depth_score(model, T, frames, scale=scale)
            all_candidates.append((score, T, scale))

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    top = all_candidates[:10]
    print(f"    top depth scores: {[f'{c[0]:.3f}' for c in top[:5]]}")

    # --- ICP fine-tune on top 5 ---
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(obj_pts)
    best = None

    for rank, (search_score, T_init, scale) in enumerate(top[:5]):
        scaled = model_pts * scale
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(scaled)

        mod_max = (scaled.max(0) - scaled.min(0)).max()
        mc = mod_max * 0.25
        nr = max(mc, 0.006)

        tn = o3d.geometry.PointCloud(tgt)
        tn.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(nr, 30))

        sc = o3d.geometry.PointCloud(src)
        sc.transform(T_init)
        sc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(nr, 30))

        cT = np.eye(4)
        for mf in [1.0, 0.5, 0.3]:
            try:
                icp = o3d.pipelines.registration.registration_icp(
                    sc, tn, mc * mf, cT,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(1e-8, 1e-8, 80))
                cT = icp.transformation
            except Exception:
                break

        fT = cT @ T_init

        # score
        aligned = transform_points(scaled, fT)
        ds = mv_depth_score(aligned, frames, observed_masks)
        cvs = cov_score(aligned, tgt, max(mod_max * 0.08, 0.003))
        ss = silhouette_iou_score(aligned, frames, observed_masks)
        rs = render_depth_score(model, fT, frames, scale=scale)
        score = 0.35 * rs + 0.25 * ds + 0.25 * cvs + 0.15 * ss

        print(
            f"    rank#{rank}: search={search_score:.3f} → final={score:.3f} "
            f"(rdepth={rs:.3f} depth={ds:.3f} cov={cvs:.3f} sil={ss:.3f}) scale={scale:.2f}")

        if best is None or score > best.confidence:
            U, _, Vt = np.linalg.svd(fT[:3, :3])
            R_clean = U @ Vt
            if np.linalg.det(R_clean) < 0:
                U[:, -1] *= -1
                R_clean = U @ Vt
            fT[:3, :3] = R_clean
            r_ = Rot.from_matrix(R_clean)
            best = PoseEstimate(
                T_base_obj=fT, position_m=fT[:3, 3],
                quaternion_xyzw=r_.as_quat(),
                euler_xyz_deg=r_.as_euler('xyz', degrees=True),
                scale=scale, confidence=score,
                fitness=getattr(icp, 'fitness', 0.0),
                rmse=getattr(icp, 'inlier_rmse', 0.0),
                depth_score=ds, coverage=cvs, silhouette_score=ss,
                rgb_score=rs)

    if best is None:
        raise RuntimeError("table-grounded 정합 실패")
    return best


def render_depth_score(model: CanonicalModel, T_base_obj: np.ndarray,
                       frames, scale: float = 1.0, tol: float = 0.012,
                       max_pts: int = 3000) -> float:
    """GLB를 각 카메라로 투영해서 실제 depth와 직접 비교.

    렌더된 z-depth와 실제 depth 센서 값의 일치율을 반환.
    RGB나 mask 없이 depth만으로 평가하므로 색 무관.
    max_pts로 vertex 수를 제한해서 속도를 확보한다.
    """
    all_verts = (model.mesh.vertices - model.center) * scale
    if len(all_verts) > max_pts:
        idx = np.random.default_rng(0).choice(len(all_verts), max_pts, replace=False)
        all_verts = all_verts[idx]
    verts_base = transform_points(all_verts, T_base_obj)

    total_ok, total_valid = 0, 0
    for cam in frames:
        h, w = cam.intrinsics.height, cam.intrinsics.width
        K = cam.intrinsics.K
        verts_cam = transform_points(verts_base, np.linalg.inv(cam.T_base_cam))
        z_model = verts_cam[:, 2]
        front = z_model > 0.05

        if front.sum() < 10:
            continue

        u = (K[0, 0] * verts_cam[front, 0] / z_model[front] + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * verts_cam[front, 1] / z_model[front] + K[1, 2]).astype(np.int32)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        if ok.sum() < 5:
            continue

        z_pred = z_model[front][ok]
        z_real = cam.depth_u16[v[ok], u[ok]].astype(np.float64) * cam.intrinsics.depth_scale
        has_depth = z_real > 0.05

        if has_depth.sum() < 5:
            continue

        matches = np.abs(z_pred[has_depth] - z_real[has_depth]) < tol
        total_ok += matches.sum()
        total_valid += has_depth.sum()

    return total_ok / max(total_valid, 1)


def exhaustive_pose_search(model: CanonicalModel, obj_pts: np.ndarray,
                           frames, table_n: np.ndarray,
                           n_yaw: int = 72, n_tilt: int = 5,
                           scale_factors=(0.88, 0.94, 1.0, 1.06, 1.12)):
    """Render-and-compare exhaustive search.

    테이블 법선으로 수직축을 고정하고, yaw를 360° 탐색해서
    3대 카메라 depth와 가장 잘 맞는 pose를 찾는다.

    Returns: list of (score, T_base_obj, scale) top candidates
    """
    # 1. 위치 = 점군 centroid (depth에서 직접 얻으므로 정확함)
    centroid = obj_pts.mean(axis=0)

    # 2. 수직축 = 테이블 법선 (물체가 테이블 위에 서 있음)
    up = -table_n  # 테이블 법선 반대 방향이 위쪽
    if up[1] > 0:  # OpenCV 좌표계에서 y-down이므로
        up = -up
    up = up / (np.linalg.norm(up) + 1e-8)

    # 3. yaw × tilt × scale sweep
    model_center = model.center
    yaw_angles = np.linspace(0, 360, n_yaw, endpoint=False)
    tilt_angles = np.linspace(-10, 10, n_tilt)  # ±10° tilt

    all_candidates = []

    for scale in scale_factors:
        for yaw_deg in yaw_angles:
            for tilt_deg in tilt_angles:
                # rotation: 수직축 기준 yaw + 작은 tilt
                R_yaw = Rot.from_rotvec(np.radians(yaw_deg) * up).as_matrix()
                # tilt: yaw 적용 후 수평축 기준 작은 기울기
                horiz = np.cross(up, np.array([1, 0, 0]))
                if np.linalg.norm(horiz) < 0.1:
                    horiz = np.cross(up, np.array([0, 0, 1]))
                horiz = horiz / (np.linalg.norm(horiz) + 1e-8)
                R_tilt = Rot.from_rotvec(np.radians(tilt_deg) * horiz).as_matrix()

                R = R_tilt @ R_yaw
                t = centroid - R @ model_center * scale

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t

                score = render_depth_score(model, T, frames, scale=scale)
                all_candidates.append((score, T, scale))

    # 상위 후보 반환
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    return all_candidates[:20]


def register_model_exhaustive(model: CanonicalModel, model_pts: np.ndarray,
                              obj_pts: np.ndarray, frames,
                              observed_masks, table_n: np.ndarray,
                              model_name: str = "",
                              symmetry_map=None):
    """Exhaustive search + ICP fine-tune.

    1. exhaustive_pose_search로 depth 기반 best pose 찾기
    2. 상위 5개에 대해 ICP fine-tune
    3. 최종 multi-view depth+coverage+silhouette score로 best 선택
    """
    print("    exhaustive search: yaw sweep × tilt × scale...")
    candidates = exhaustive_pose_search(model, obj_pts, frames, table_n)

    if not candidates:
        raise RuntimeError("exhaustive search 실패: 유효한 후보 없음")

    top5 = candidates[:5]
    print(f"    top-5 depth scores: {[f'{c[0]:.3f}' for c in top5]}")

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(obj_pts)
    obj_center = obj_pts.mean(0)

    best = None

    for rank, (search_score, T_init, scale) in enumerate(top5):
        scaled = model_pts * scale
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(scaled)

        # ICP fine-tune (coarse → fine)
        mod_max = (scaled.max(0) - scaled.min(0)).max()
        mc = mod_max * 0.25
        nr = max(mc, 0.006)

        tn = o3d.geometry.PointCloud(tgt)
        tn.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(nr, 30))

        sc = o3d.geometry.PointCloud(src)
        sc.transform(T_init)
        sc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(nr, 30))

        cT = np.eye(4)
        for m_factor in [1.0, 0.5, 0.3]:
            try:
                icp = o3d.pipelines.registration.registration_icp(
                    sc, tn, mc * m_factor, cT,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(1e-8, 1e-8, 80))
                cT = icp.transformation
            except Exception:
                break

        fT = cT @ T_init

        # 최종 score (3대 카메라 전부)
        aligned = transform_points(scaled, fT)
        ds = mv_depth_score(aligned, frames, observed_masks)
        cs = cov_score(aligned, tgt, max(mod_max * 0.08, 0.003))
        ss = silhouette_iou_score(aligned, frames, observed_masks)
        rs = render_depth_score(model, fT, frames, scale=scale)
        score = 0.35 * rs + 0.25 * ds + 0.25 * cs + 0.15 * ss

        if best is None or score > best.confidence:
            U, _, Vt = np.linalg.svd(fT[:3, :3])
            R_clean = U @ Vt
            if np.linalg.det(R_clean) < 0:
                U[:, -1] *= -1
                R_clean = U @ Vt
            fT[:3, :3] = R_clean
            r_ = Rot.from_matrix(R_clean)
            best = PoseEstimate(
                T_base_obj=fT, position_m=fT[:3, 3],
                quaternion_xyzw=r_.as_quat(),
                euler_xyz_deg=r_.as_euler('xyz', degrees=True),
                scale=scale, confidence=score,
                fitness=icp.fitness if 'icp' in dir() else 0.0,
                rmse=icp.inlier_rmse if 'icp' in dir() else 0.0,
                depth_score=ds, coverage=cs, silhouette_score=ss,
                rgb_score=rs)
            print(
                f"    rank#{rank}: search={search_score:.3f} → final={score:.3f} "
                f"(render_depth={rs:.3f} depth={ds:.3f} cov={cs:.3f} sil={ss:.3f}) "
                f"scale={scale:.2f}")

    if best is None:
        raise RuntimeError("exhaustive register 실패")
    return best


def register_model(model_pts, obj_pts, frames, observed_masks,
                   model_name: str = "", symmetry_map: Optional[Dict[str, str]] = None,
                   model_ref: Optional[CanonicalModel] = None):
    """CAD → 관측 점군 정합. 실물 크기(scale≈1.0) 중심 탐색.

    GLB 모델이 실물 크기(m)이므로 scale=1.0이 정답.
    뎁스 카메라의 부분 관측을 고려해 ±20% 범위 탐색.
    model_ref가 있으면 RGB render score를 최종 평가에 반영한다.
    """
    obj_center = obj_pts.mean(0)
    obj_max = (obj_pts.max(0)-obj_pts.min(0)).max()
    mod_max = (model_pts.max(0)-model_pts.min(0)).max()

    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(obj_pts)
    best = None

    # scale=1.0 중심 탐색 (GLB = 실물 크기이므로)
    for sf in [0.88, 0.94, 1.0, 1.06, 1.12]:
        scale = sf                    # 실물 크기 기준 직접 탐색
        scaled = model_pts * scale
        src_c = scaled.mean(0)
        rs = mod_max * scale
        mc = rs * 0.25

        src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(scaled)

        # FPFH
        vx = max(0.002, rs*0.08)
        sd = src.voxel_down_sample(vx); td = tgt.voxel_down_sample(vx)
        for p in [sd, td]:
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(vx*3, 30))
        fpfh_T = None
        try:
            sf_ = o3d.pipelines.registration.compute_fpfh_feature(sd, o3d.geometry.KDTreeSearchParamHybrid(vx*5,100))
            tf_ = o3d.pipelines.registration.compute_fpfh_feature(td, o3d.geometry.KDTreeSearchParamHybrid(vx*5,100))
            r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                sd, td, sf_, tf_, True, vx*3,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vx*3)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(20000, 0.995))
            if r.fitness > 0.05: fpfh_T = r.transformation
        except: pass

        # 초기 후보
        rots = rotation_candidates(scaled, obj_pts)
        inits = []
        if fpfh_T is not None: inits.append(fpfh_T)
        for R in rots[:8]:
            T = np.eye(4); T[:3,:3]=R; T[:3,3]=obj_center-R@src_c; inits.append(T)
        if not inits:
            T=np.eye(4); T[:3,3]=obj_center-src_c; inits.append(T)

        # Step 2: RGB-guided local pose search (fast: 1-cam only for speed)
        if model_ref is not None and len(inits) > 0:
            # 기존 init 중 상위 3개를 RGB score로 선별 (1-cam fast)
            init_scored = [
                (rgb_compare_score(model_ref, Ti, frames, observed_masks, scale,
                                   best_cam_only=True), Ti)
                for Ti in inits
            ]
            init_scored.sort(key=lambda x: x[0], reverse=True)
            top_inits = [Ti for _, Ti in init_scored[:3]]
            # 각 상위 init 주변에서 perturbation (적은 횟수, 1-cam)
            expanded = []
            for Ti in top_inits:
                best_local = local_pose_search(
                    model_ref, frames, observed_masks, Ti,
                    scale=scale, n_candidates=16, top_k=2)
                expanded.extend(best_local)
            inits = top_inits + expanded

        nr = max(mc, 0.006)
        tn = o3d.geometry.PointCloud(tgt)
        tn.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(nr, 30))

        bl_T, bl_f = None, -1
        for Ti in inits:
            sc = o3d.geometry.PointCloud(src); sc.transform(Ti)
            sc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(nr, 30))
            icp = o3d.pipelines.registration.registration_icp(
                sc, tn, mc, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(1e-7,1e-7,50))
            tT = icp.transformation @ Ti
            ac = (tT @ np.append(src_c,1))[:3]
            if np.linalg.norm(ac-obj_center) > rs*0.8: continue
            if icp.fitness > bl_f: bl_f = icp.fitness; bl_T = tT

        if bl_T is None or bl_f < 0.05: continue

        # Fine ICP
        sr = o3d.geometry.PointCloud(src); sr.transform(bl_T)
        sr.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(nr, 30))
        cT = np.eye(4)
        for m in [1.0, 0.5, 0.3]:
            icp = o3d.pipelines.registration.registration_icp(
                sr, tn, mc*m, cT,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(1e-8,1e-8,120))
            cT = icp.transformation

        fT = cT @ bl_T
        fc = (fT @ np.append(src_c,1))[:3]
        if np.linalg.norm(fc-obj_center) > rs*0.5: continue

        aligned = transform_points(scaled, fT)
        ds = mv_depth_score(aligned, frames, observed_masks)
        cs = cov_score(aligned, tgt, max(rs*0.08, 0.003))
        ss = silhouette_iou_score(aligned, frames, observed_masks)
        rs_ = 0.0
        if model_ref is not None:
            rs_ = rgb_compare_score(model_ref, fT, frames, observed_masks, scale)
        score = combine_pose_scores(model_name, ds, cs, ss,
                                    rgb_sc=rs_, symmetry_map=symmetry_map)

        if best is None or score > best.confidence:
            # 회전 행렬 정규화 (ICP 수치 오차 보정)
            U, _, Vt = np.linalg.svd(fT[:3,:3])
            R_clean = U @ Vt
            if np.linalg.det(R_clean) < 0:
                U[:, -1] *= -1
                R_clean = U @ Vt
            fT[:3,:3] = R_clean
            r_ = Rot.from_matrix(R_clean)
            best = PoseEstimate(
                T_base_obj=fT, position_m=fT[:3,3],
                quaternion_xyzw=r_.as_quat(), euler_xyz_deg=r_.as_euler('xyz',degrees=True),
                scale=scale, confidence=score, fitness=icp.fitness,
                rmse=icp.inlier_rmse, depth_score=ds, coverage=cs,
                silhouette_score=ss, rgb_score=rs_)

    if best is None:
        raise RuntimeError("정합 실패")
    return best


# ═══════════════════════════════════════════════════════════
# 7. 멀티뷰 검증
# ═══════════════════════════════════════════════════════════

def mv_depth_score(aligned, frames, observed_masks, tol=0.015):
    tok, tv = 0, 0
    for cam, mask in zip(frames, observed_masks):
        h, w = cam.intrinsics.height, cam.intrinsics.width
        K = cam.intrinsics.K
        p = transform_points(aligned, np.linalg.inv(cam.T_base_cam))
        front = p[:, 2] > 0.05
        p = p[front]
        if len(p) == 0:
            continue

        u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
        v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ok.sum() == 0:
            continue

        inside = mask[v[ok], u[ok]] > 0
        if inside.sum() == 0:
            continue

        zm = p[ok, 2][inside]
        zr = cam.depth_u16[v[ok], u[ok]][inside].astype(np.float64) * cam.intrinsics.depth_scale
        hd = zr > 0.05
        if hd.sum() == 0:
            continue

        tok += (np.abs(zm[hd] - zr[hd]) < tol).sum()
        tv += hd.sum()

    return tok / max(tv, 1)


def project_points_to_mask(obj_pts, cam: CameraFrame):
    h, w = cam.intrinsics.height, cam.intrinsics.width
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(obj_pts) == 0:
        return mask

    p = transform_points(obj_pts, np.linalg.inv(cam.T_base_cam))
    front = p[:, 2] > 0.05
    p = p[front]
    if len(p) == 0:
        return mask

    K = cam.intrinsics.K
    u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
    v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask[v[ok], u[ok]] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def silhouette_iou_score(aligned, frames, observed_masks):
    total = 0.0
    count = 0
    for cam, obs_mask in zip(frames, observed_masks):
        pred_mask = project_points_to_mask(aligned, cam)
        pred = pred_mask > 0
        obs = obs_mask > 0
        union = np.logical_or(pred, obs).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred, obs).sum()
        total += inter / union
        count += 1
    return total / max(count, 1)

def cov_score(aligned, obj_pcd, radius=0.005):
    obj_pts = np.asarray(obj_pcd.points)
    mp = o3d.geometry.PointCloud(); mp.points=o3d.utility.Vector3dVector(aligned)
    fd = np.asarray(mp.compute_point_cloud_distance(obj_pcd))
    fwd = (fd<radius).sum()/max(len(fd),1)
    m=radius*5; lo=aligned.min(0)-m; hi=aligned.max(0)+m
    ib=np.all((obj_pts>=lo)&(obj_pts<=hi),1)
    if ib.sum()<10: return fwd*0.01
    ip=o3d.geometry.PointCloud(); ip.points=o3d.utility.Vector3dVector(obj_pts[ib])
    rd=np.asarray(ip.compute_point_cloud_distance(mp))
    return fwd*((rd<radius).sum()/max(len(rd),1))


# ═══════════════════════════════════════════════════════════
# 8. 출력: JSON + NPZ + Posed GLB + Overlay
# ═══════════════════════════════════════════════════════════

def export_result(pose, model, frame_id, out_dir, glb_src_path):
    """한 물체의 모든 산출물 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = OBJECT_LABELS.get(model.name, model.name)

    result = {
        "frame_id": frame_id, "object_name": model.name, "label": label,
        "coordinate_frame": "base (= cam0)", "unit": "meter",
        "position_m": pose.position_m.tolist(),
        "quaternion_xyzw": pose.quaternion_xyzw.tolist(),
        "euler_xyz_deg": pose.euler_xyz_deg.tolist(),
        "T_base_obj": pose.T_base_obj.tolist(),
        "rotation_matrix": pose.T_base_obj[:3,:3].tolist(),
        "scale": pose.scale,
        "real_size_m": {
            "x": float(model.extents_m[0]*pose.scale),
            "y": float(model.extents_m[1]*pose.scale),
            "z": float(model.extents_m[2]*pose.scale),
        },
        "confidence": pose.confidence, "fitness": pose.fitness,
        "rmse": pose.rmse, "depth_score": pose.depth_score,
        "coverage": pose.coverage, "silhouette_score": pose.silhouette_score,
    }

    # JSON + NPZ
    jp = out_dir / f"pose_{model.name}_{frame_id}.json"
    np.savez(out_dir / f"pose_{model.name}_{frame_id}.npz",
             T_base_obj=pose.T_base_obj, position_m=pose.position_m,
             quaternion_xyzw=pose.quaternion_xyzw, scale=pose.scale)

    # Posed GLB (cam0 + isaac 2개)
    for coord in ["opencv", "isaac"]:
        scene = trimesh.load(str(glb_src_path))
        mesh = trimesh.util.concatenate(list(scene.geometry.values())) \
            if isinstance(scene, trimesh.Scene) else scene.copy()
        v = (mesh.vertices - model.center) * pose.scale
        vh = np.hstack([v, np.ones((len(v),1))])
        vp = (pose.T_base_obj @ vh.T)[:3].T
        if coord == "isaac":
            vp = (T_ISAAC_CV @ np.hstack([vp, np.ones((len(vp),1))]).T)[:3].T
        mesh.vertices = vp
        suffix = "" if coord == "opencv" else "_isaac"
        gp = out_dir / f"{model.name}_posed_{frame_id}{suffix}.glb"
        mesh.export(str(gp), file_type="glb")
        result[f"posed_glb_{coord}"] = str(gp)

    # JSON 최종 저장 (GLB 경로 포함)
    with open(jp, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def render_wireframe(mesh, center, pose, cam, color=(0,255,0), thickness=2):
    """메시 와이어프레임 → 카메라 이미지에 투영."""
    img = cam.color_bgr.copy()
    h, w = img.shape[:2]; K = cam.intrinsics.K
    v = (mesh.vertices - center) * pose.scale
    vb = transform_points(v, pose.T_base_obj)
    vc = transform_points(vb, np.linalg.inv(cam.T_base_cam))
    z = vc[:,2]; ok = z>0.05
    pu = np.full(len(v),-1.0); pv = np.full(len(v),-1.0)
    pu[ok] = K[0,0]*vc[ok,0]/z[ok]+K[0,2]
    pv[ok] = K[1,1]*vc[ok,1]/z[ok]+K[1,2]
    for e0, e1 in mesh.edges_unique:
        if not (ok[e0] and ok[e1]): continue
        p0=(int(pu[e0]),int(pv[e0])); p1=(int(pu[e1]),int(pv[e1]))
        if abs(p0[0])>2*w or abs(p0[1])>2*h or abs(p1[0])>2*w or abs(p1[1])>2*h: continue
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


def save_combined_overlay(all_poses, all_models, all_masks, frames, frame_id, out_dir):
    """모든 물체의 와이어프레임을 한 이미지에 합성."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_imgs = [cam.color_bgr.copy() for cam in frames]

    for obj_idx, (pose, model, masks) in enumerate(zip(all_poses, all_models, all_masks)):
        color = COLORS[obj_idx % len(COLORS)]
        for ci, cam in enumerate(frames):
            wire = render_wireframe(model.mesh, model.center, pose, cam, color)
            # 와이어프레임만 추출하여 합성
            diff = cv2.absdiff(wire, cam.color_bgr)
            wire_mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 10
            base_imgs[ci][wire_mask] = wire[wire_mask]
            # 마스크 윤곽
            cnts, _ = cv2.findContours(masks[ci], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base_imgs[ci], cnts, -1, color, 1)
            # 라벨
            label = OBJECT_LABELS.get(model.name, model.name)
            centroid_2d = None
            if masks[ci].any():
                ys, xs = np.where(masks[ci] > 0)
                centroid_2d = (int(xs.mean()), int(ys.min()) - 5)
            if centroid_2d:
                cv2.putText(base_imgs[ci], label, centroid_2d,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    for ci in range(len(frames)):
        cv2.putText(base_imgs[ci], f"cam{ci}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack(base_imgs)
    cv2.imwrite(str(out_dir / f"overlay_{frame_id}.png"), combined)


# ═══════════════════════════════════════════════════════════
# 메인 파이프라인
# ═══════════════════════════════════════════════════════════

def run_pipeline(data_dir, intrinsics_dir, frame_id,
                 glb_path=None, output_dir="src/output/pose_pipeline",
                 multi_object=False,
                 seg_backend="auto",
                 seg_model_path=None,
                 seg_device="cpu",
                 seg_conf=0.35,
                 seg_iou=0.50,
                 capture_subdir="object_capture"):
    data_dir = Path(data_dir)
    intrinsics_dir = Path(intrinsics_dir)
    out = Path(output_dir)

    print("=" * 60)
    print(f" Pose Estimation — Frame {frame_id} {'[MULTI]' if multi_object else '[SINGLE]'}")
    print("=" * 60)

    # 1. 캘리브레이션 + GLB (자동 탐색)
    intrinsics, extrinsics = load_calibration(data_dir, intrinsics_dir)
    glb_paths: Dict[str, Path] = {}
    all_models: Dict[str, CanonicalModel] = {}
    if glb_path:
        gp = Path(glb_path)
        m = normalize_glb(gp)
        all_models[m.name] = m
        glb_paths[m.name] = gp
    else:
        for p in sorted(data_dir.glob("*.glb")):
            m = normalize_glb(p)
            all_models[m.name] = m
            glb_paths[m.name] = p

    # 런타임 symmetry map: 등록된 prior가 없으면 mesh PCA로 추정
    runtime_symmetry: Dict[str, str] = {}
    for n, m in all_models.items():
        if n in OBJECT_SYMMETRY:
            runtime_symmetry[n] = OBJECT_SYMMETRY[n]
        else:
            runtime_symmetry[n] = infer_symmetry_from_mesh(m)

    for n, m in all_models.items():
        label = OBJECT_LABELS.get(n, n)
        sym = runtime_symmetry.get(n, "none")
        print(
            f"  {n} ({label}): "
            f"[{m.extents_m[0]*100:.1f}, {m.extents_m[1]*100:.1f}, {m.extents_m[2]*100:.1f}]cm "
            f"sym={sym}"
        )

    # 2. RGB-D 로드
    frames = load_frame(data_dir, frame_id, intrinsics, extrinsics, capture_subdir=capture_subdir)
    segmentation_model = load_segmentation_model(
        seg_backend=seg_backend,
        seg_model_path=seg_model_path,
        seg_device=seg_device,
        seg_conf=seg_conf,
        seg_iou=seg_iou,
    )
    print(f"  segmentation_backend={getattr(segmentation_model, 'backend_name', type(segmentation_model).__name__)}")
    model_points_cache = {
        name: sample_model_points(model) for name, model in all_models.items()
    }

    # 3. 테이블 평면 추정
    print("\n[1] 테이블 평면 추정")
    table_n, table_d, table_center, table_radius = estimate_table_plane(frames)
    print(f"  table center=[{table_center[0]:+.3f}, {table_center[1]:+.3f}, {table_center[2]:+.3f}]")
    print(f"  table radius={table_radius:.3f}m")

    # 4. RGB instance separation → cross-view association
    print("\n[2] Instance Separation + Cross-view Association")
    segmented_masks_by_cam, detections_by_cam, tracks, above_pts = build_object_tracks(
        frames, segmentation_model,
        table_n, table_d, table_center, table_radius,
        out, frame_id,
    )

    for cam in frames:
        n_seg = len(segmented_masks_by_cam[cam.cam_id])
        n_det = len(detections_by_cam.get(cam.cam_id, []))
        print(f"  cam{cam.cam_id}: segmented={n_seg} detections={n_det}")

    print(f"  associated tracks={len(tracks)}")
    for ti, track in enumerate(tracks):
        cams = ",".join(f"cam{cid}" for cid in sorted(track.detections.keys()))
        area = sum(det.area_px for det in track.detections.values())
        hint = track.hint_model_name or "-"
        print(
            f"    track#{ti} {track.track_id}: views={len(track.detections)} "
            f"[{cams}] area={area} hint={hint}"
        )

    # 단일 물체 프레임은 object-centric multi-track logic를 우회하고
    # segmentation 전체를 하나의 관측으로 묶어 바로 정합한다.
    if not multi_object:
        if glb_path:
            model_name = Path(glb_path).stem
            if model_name not in all_models:
                raise RuntimeError(f"{model_name} 모델 없음")
        else:
            model_name = FRAME_TO_GLB.get(int(frame_id))
            if model_name is None:
                raise RuntimeError(f"프레임 {frame_id}에 대한 GLB 매핑 없음")
            if model_name not in all_models:
                raise RuntimeError(f"{model_name} 모델 파일 없음")

        model = all_models[model_name]
        model_pts = model_points_cache[model_name]
        print(f"\n[3] Single-Object Direct Registration")
        print(f"  매핑: {model_name} ({OBJECT_LABELS.get(model_name, '')})")

        combined_masks = [
            combine_instance_masks(masks, (cam.intrinsics.height, cam.intrinsics.width))
            for cam, masks in zip(frames, segmented_masks_by_cam)
        ]
        masked_pts = fuse_masked_points(frames, combined_masks)
        point_candidates = rank_single_object_points(above_pts, masked_pts, model)
        if not point_candidates:
            raise RuntimeError("단일 물체 점군 추출 실패")
        for cand_score, source_name, cand_pts in point_candidates:
            obj_ext = cand_pts.max(0) - cand_pts.min(0)
            print(
                f"  candidate source={source_name} score={cand_score:.3f} "
                f"pts={len(cand_pts)} "
                f"extent=[{obj_ext[0]*100:.1f},{obj_ext[1]*100:.1f},{obj_ext[2]*100:.1f}]cm"
            )

        best_single_pose = None
        best_single_masks = None
        best_single_pts = None
        best_source_name = None

        for _, source_name, cand_pts in point_candidates[:3]:
            eval_masks = build_single_object_masks(frames, segmented_masks_by_cam, cand_pts)
            try:
                pose = register_table_grounded(
                    model,
                    model_pts,
                    cand_pts,
                    frames,
                    eval_masks,
                    table_n,
                    table_d,
                    model_name=model_name,
                    symmetry_map=runtime_symmetry,
                )
            except RuntimeError:
                continue
            if best_single_pose is None or pose.confidence > best_single_pose.confidence:
                best_single_pose = pose
                best_single_masks = eval_masks
                best_single_pts = cand_pts
                best_source_name = source_name

        if best_single_pose is None or best_single_masks is None or best_single_pts is None:
            raise RuntimeError("단일 물체 정합 실패")

        pose = best_single_pose
        eval_masks = best_single_masks
        single_pts = best_single_pts
        print(f"  selected source={best_source_name}")

        print(f"  ── {OBJECT_LABELS.get(model_name, model_name)} ──")
        print(
            f"  position:   [{pose.position_m[0]:+.4f}, "
            f"{pose.position_m[1]:+.4f}, {pose.position_m[2]:+.4f}] m"
        )
        print(
            f"  quaternion: [{pose.quaternion_xyzw[0]:+.4f}, {pose.quaternion_xyzw[1]:+.4f}, "
            f"{pose.quaternion_xyzw[2]:+.4f}, {pose.quaternion_xyzw[3]:+.4f}]"
        )
        print(
            f"  scale={pose.scale:.4f}  confidence={pose.confidence:.4f}  "
            f"fitness={pose.fitness:.4f}  silhouette={pose.silhouette_score:.4f}"
        )

        result = export_result(pose, model, frame_id, out, glb_paths[model_name])
        save_combined_overlay([pose], [model], [eval_masks], frames, frame_id, out)
        print(f"\n  overlay: {out / f'overlay_{frame_id}.png'}")
        print(f"  {model_name}: JSON=pose_{model_name}_{frame_id}.json")
        print(f"    GLB(cam0)  = {result['posed_glb_opencv']}")
        print(f"    GLB(isaac) = {result['posed_glb_isaac']}")
        return [result]

    # 5. Per-object reconstruction + GLB shortlist + ICP refinement
    MIN_CONFIDENCE = 0.03
    MIN_FITNESS = 0.10
    MAX_SCALE_DEV = 0.25

    all_results = []
    all_poses = []
    all_model_objs = []
    all_masks_list = []
    used_glbs = set()
    max_single_extent = max_model_extent(all_models)

    if multi_object:
        active_tracks = [t for t in tracks if len(t.detections) >= 2]
        if not active_tracks:
            active_tracks = tracks[:1]
    else:
        active_tracks = tracks[:1]
        if not active_tracks and len(above_pts) > 0:
            fallback_masks = [
                combine_instance_masks(masks, (cam.intrinsics.height, cam.intrinsics.width))
                for cam, masks in zip(frames, segmented_masks_by_cam)
            ]
            active_tracks = [
                ObjectTrack(
                    track_id="track_fallback_00",
                    detections={},
                    observed_masks=fallback_masks,
                    fused_points_base=find_single_object(above_pts),
                )
            ]

    for ti, track in enumerate(active_tracks):
        if multi_object and len(used_glbs) >= len(all_models):
            break

        print(f"\n[3-{ti}] Object Track {track.track_id}")
        if track.fused_points_base is None:
            track_pts = fuse_track_points(frames, track)
        else:
            track_pts = track.fused_points_base

        if len(track_pts) == 0:
            print("  [FAIL] fused point cloud 비어 있음")
            continue

        track.fused_points_base = track_pts
        track_ext = track_pts.max(0) - track_pts.min(0)
        print(
            f"  fused pts={len(track_pts)} "
            f"extent=[{track_ext[0]*100:.1f},{track_ext[1]*100:.1f},{track_ext[2]*100:.1f}]cm"
        )

        subclusters = split_track_points(track_pts, max_single_extent)
        print(f"  subclusters={len(subclusters)}")
        for si, (cpts, _) in enumerate(subclusters[:3]):
            ext = cpts.max(0) - cpts.min(0)
            print(
                f"    sub#{si}: {len(cpts)} pts "
                f"extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm"
            )

        if multi_object and (
            track_ext.max() > 2.2 * max_single_extent or len(subclusters) > 4
        ):
            print("  [SKIP] mixed track로 판단 → recovery 단계에서 처리")
            continue

        if glb_path:
            model_name = Path(glb_path).stem
            if model_name not in all_models:
                print(f"  [WARN] {model_name} 모델 없음 → 스킵")
                continue
            shortlist = [(model_name, 1.0)]
        elif not multi_object:
            model_name = FRAME_TO_GLB.get(int(frame_id))
            if model_name is None:
                raise RuntimeError(f"프레임 {frame_id}에 대한 GLB 매핑 없음")
            if model_name not in all_models:
                print(f"  [WARN] {model_name} 모델 없음 → 스킵")
                continue
            print(f"  매핑: {model_name} ({OBJECT_LABELS.get(model_name, '')})")
            shortlist = [(model_name, 1.0)]
        else:
            candidate_models = {
                k: v for k, v in all_models.items()
                if k not in used_glbs
            }
            if not candidate_models:
                break
            hint_name = track.hint_model_name
            if hint_name in candidate_models:
                fallback_shortlist = shortlist_glb_candidates_v2(
                    track_pts,
                    candidate_models,
                    frames,
                    track.observed_masks,
                    top_k=min(2, len(candidate_models)),
                    track_mean_hsv=aggregate_track_mean_hsv(track),
                    color_priors=OBJECT_COLOR_PRIORS_HSV,
                )
                shortlist = [(hint_name, 0.98)]
                shortlist.extend([(n, s) for n, s in fallback_shortlist if n != hint_name])
            else:
                shortlist = shortlist_glb_candidates_v2(
                    track_pts,
                    candidate_models,
                    frames,
                    track.observed_masks,
                    top_k=min(3, len(candidate_models)),
                    track_mean_hsv=aggregate_track_mean_hsv(track),
                    color_priors=OBJECT_COLOR_PRIORS_HSV,
                )

        if not shortlist:
            print("  [FAIL] GLB 후보 없음")
            continue

        best_pose = None
        best_model_name = None
        best_masks = None
        best_score = -1.0
        best_sub_idx = -1

        for cand_name, coarse_score in shortlist:
            model = all_models[cand_name]
            model_pts = model_points_cache[cand_name]

            for sub_idx, (cpts, _) in enumerate(subclusters[:3]):
                if not glb_extent_compatible(cpts, model):
                    continue

                support_masks = project_cluster_to_support_masks(cpts, frames)
                eval_masks = refine_eval_masks_with_support(track.observed_masks, support_masks)

                try:
                    pose_tmp = register_model(
                        model_pts, cpts, frames, eval_masks,
                        model_name=cand_name, symmetry_map=runtime_symmetry,
                        model_ref=model,
                    )
                except RuntimeError:
                    continue

                if abs(pose_tmp.scale - 1.0) > MAX_SCALE_DEV:
                    continue

                cluster_weight = len(cpts) / max(len(track_pts), 1)
                view_bonus = len(track.detections) / max(len(frames), 1)
                final_score = (
                    0.20 * coarse_score
                    + 0.65 * pose_tmp.confidence
                    + 0.10 * cluster_weight
                    + 0.05 * view_bonus
                )
                print(
                    f"    candidate={cand_name} sub={sub_idx} coarse={coarse_score:.3f} "
                    f"pose_conf={pose_tmp.confidence:.3f} fit={pose_tmp.fitness:.3f} "
                    f"rgb={pose_tmp.rgb_score:.3f} sil={pose_tmp.silhouette_score:.3f} "
                    f"scale={pose_tmp.scale:.3f} final={final_score:.3f}"
                )

                if final_score > best_score:
                    best_score = final_score
                    best_pose = pose_tmp
                    best_model_name = cand_name
                    best_masks = eval_masks
                    best_sub_idx = sub_idx

        if best_model_name is None or best_pose is None or best_masks is None:
            print("  [FAIL] 모든 후보 정합 실패")
            continue
        if best_pose.confidence < MIN_CONFIDENCE:
            print(f"  [FAIL] confidence={best_pose.confidence:.4f} < {MIN_CONFIDENCE}")
            continue
        if best_pose.fitness < MIN_FITNESS:
            print(f"  [FAIL] fitness={best_pose.fitness:.4f} < {MIN_FITNESS}")
            continue

        model = all_models[best_model_name]
        used_glbs.add(best_model_name)
        label = OBJECT_LABELS.get(best_model_name, best_model_name)
        print(f"  최종 선택: {best_model_name} sub={best_sub_idx} score={best_score:.3f}")
        print(f"  ── {label} ──")
        print(
            f"  position:   [{best_pose.position_m[0]:+.4f}, "
            f"{best_pose.position_m[1]:+.4f}, {best_pose.position_m[2]:+.4f}] m"
        )
        print(
            f"  quaternion: [{best_pose.quaternion_xyzw[0]:+.4f}, {best_pose.quaternion_xyzw[1]:+.4f}, "
            f"{best_pose.quaternion_xyzw[2]:+.4f}, {best_pose.quaternion_xyzw[3]:+.4f}]"
        )
        print(
            f"  scale={best_pose.scale:.4f}  confidence={best_pose.confidence:.4f}  "
            f"fitness={best_pose.fitness:.4f}  silhouette={best_pose.silhouette_score:.4f}"
        )

        result = export_result(best_pose, model, frame_id, out, glb_paths[best_model_name])
        all_results.append(result)
        all_poses.append(best_pose)
        all_model_objs.append(model)
        all_masks_list.append(best_masks)

    # 6. recovery: object-centric 단계에서 누락된 물체는 GLB-first로 보충
    matched_glbs = {r["object_name"] for r in all_results}
    unmatched = set(all_models.keys()) - matched_glbs
    if unmatched:
        print(f"\n[4] Recovery for unmatched GLBs: {', '.join(sorted(unmatched))}")
        recovery_models = {k: all_models[k] for k in sorted(unmatched)}
        glb_detections = detect_objects_glb_first(
            frames, recovery_models, table_n, table_d, table_center, table_radius,
            runtime_symmetry=runtime_symmetry,
            segmented_masks_by_cam=segmented_masks_by_cam,
        )
        for gi, (cluster_pts, model_name, coarse_score, obs_masks) in enumerate(glb_detections):
            if model_name in matched_glbs:
                continue
            model = all_models[model_name]
            label = OBJECT_LABELS.get(model_name, model_name)
            print(f"  recovery#{gi}: {model_name} ({label}) coarse={coarse_score:.3f}")
            try:
                best_pose = register_model(
                    model_points_cache[model_name],
                    cluster_pts,
                    frames,
                    obs_masks,
                    model_name=model_name,
                    symmetry_map=runtime_symmetry,
                    model_ref=model,
                )
            except RuntimeError:
                print("    [FAIL] ICP 정합 실패")
                continue

            if abs(best_pose.scale - 1.0) > MAX_SCALE_DEV:
                print(f"    [FAIL] scale={best_pose.scale:.3f} (편차 초과)")
                continue
            if best_pose.confidence < MIN_CONFIDENCE or best_pose.fitness < MIN_FITNESS:
                print(
                    f"    [FAIL] confidence={best_pose.confidence:.3f} "
                    f"fitness={best_pose.fitness:.3f}"
                )
                continue

            result = export_result(best_pose, model, frame_id, out, glb_paths[model_name])
            all_results.append(result)
            all_poses.append(best_pose)
            all_model_objs.append(model)
            all_masks_list.append(obs_masks)
            matched_glbs.add(model_name)

    unmatched = set(all_models.keys()) - matched_glbs
    if unmatched and multi_object:
        print(f"\n  [WARN] 미매칭 GLB: {', '.join(sorted(unmatched))}")

    # 6. overlay
    if all_poses:
        save_combined_overlay(all_poses, all_model_objs, all_masks_list, frames, frame_id, out)
        print(f"\n  overlay: {out / f'overlay_{frame_id}.png'}")

    for r in all_results:
        obj_name = r["object_name"]
        print(f"  {obj_name}: JSON=pose_{obj_name}_{frame_id}.json")
        print(f"    GLB(cam0)  = {r['posed_glb_opencv']}")
        print(f"    GLB(isaac) = {r['posed_glb_isaac']}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-View CAD Pose Estimation")
    parser.add_argument("--data_dir", default="src/data")
    parser.add_argument("--intrinsics_dir", default="src/intrinsics")
    parser.add_argument("--frame_id", default=None)
    parser.add_argument("--glb", default=None)
    parser.add_argument("--output_dir", default="src/output/pose_pipeline")
    parser.add_argument("--multi", action="store_true", help="멀티 오브젝트 모드")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--seg_backend", default="auto", choices=["auto", "yolo", "heuristic"])
    parser.add_argument("--seg_model", default=None)
    parser.add_argument("--seg_device", default="cpu")
    parser.add_argument("--seg_conf", type=float, default=0.35)
    parser.add_argument("--seg_iou", type=float, default=0.50)
    parser.add_argument("--capture_subdir", default="object_capture")
    args = parser.parse_args()

    if args.batch:
        cam0_dir = Path(args.data_dir) / args.capture_subdir / "cam0"
        fids = sorted(f.stem.replace("rgb_","") for f in cam0_dir.glob("rgb_*.jpg"))
        print(f"배치: {len(fids)} 프레임\n")
        all_r = []
        for fid in fids:
            try:
                r = run_pipeline(args.data_dir, args.intrinsics_dir, fid,
                                 args.glb, args.output_dir, args.multi,
                                 args.seg_backend, args.seg_model,
                                 args.seg_device, args.seg_conf, args.seg_iou,
                                 args.capture_subdir)
                all_r.extend(r)
            except Exception as e:
                print(f"  [ERROR] {fid}: {e}")
                all_r.append({"frame_id": fid, "error": str(e)})
        s = Path(args.output_dir) / "batch_summary.json"
        with open(s,"w") as f: json.dump(all_r, f, indent=2, ensure_ascii=False)
        print(f"\n배치 완료: {s}")
    else:
        fid = args.frame_id or "000000"
        run_pipeline(args.data_dir, args.intrinsics_dir, fid,
                     args.glb, args.output_dir, args.multi,
                     args.seg_backend, args.seg_model,
                     args.seg_device, args.seg_conf, args.seg_iou,
                     args.capture_subdir)


if __name__ == "__main__":
    main()
