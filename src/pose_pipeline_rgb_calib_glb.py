#!/usr/bin/env python3
"""
RGB-first + calibration-aware multi-view CAD pose pipeline.

목표:
- RGB에서 물체를 instance 단위로 분리
- calibration으로 서로 다른 카메라에서 같은 물체를 association
- object-wise depth fusion
- GLB render-and-compare + CAD alignment로 6D pose 추정
- posed GLB export

기본값은 threshold 기반 segmenter지만,
실전에서는 YOLO-seg 같은 instance segmentation 모델 사용을 권장.

예시:
  python3 pose_pipeline_rgb_calib_glb.py --frame_id 000000 --multi
  python3 pose_pipeline_rgb_calib_glb.py --frame_id 000000 --multi --segmenter yolo --yolo_model src/models/blocks_yolov8n_seg.pt
  python3 pose_pipeline_rgb_calib_glb.py --batch --multi
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from sklearn.cluster import DBSCAN


# -----------------------------------------------------------------------------
# 좌표계 / metadata
# -----------------------------------------------------------------------------
T_ISAAC_CV = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

DEPTH_POLICY = {
    "min_depth_m": 0.10,
    "max_depth_m": 1.20,
    "table_dist_thresh_m": 0.008,
    "object_min_height_m": 0.004,
    "object_max_height_m": 0.18,
    "voxel_size_m": 0.002,
    "cluster_eps_m": 0.010,
    "cluster_min_pts": 80,
    "crop_pad_px": 8,
}

FRAME_TO_GLB = {
    **{i: "object_004" for i in range(0, 5)},
    **{i: "object_003" for i in range(5, 10)},
    **{i: "object_002" for i in range(10, 15)},
    **{i: "object_001" for i in range(15, 17)},
}

OBJECT_LABELS = {
    "object_001": "빨강 아치",
    "object_002": "노랑 실린더",
    "object_003": "곤색 직사각형",
    "object_004": "민트 실린더",
}

OBJECT_COLOR_PRIORS_HSV = {
    "object_001": np.array([0.0, 185.0, 150.0], dtype=np.float32),
    "object_002": np.array([28.0, 185.0, 190.0], dtype=np.float32),
    "object_003": np.array([108.0, 110.0, 70.0], dtype=np.float32),
    "object_004": np.array([72.0, 80.0, 175.0], dtype=np.float32),
}

OBJECT_SYMMETRY = {
    "object_001": "none",
    "object_002": "yaw",
    "object_003": "none",
    "object_004": "yaw",
}

COLORS = [
    (0, 255, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255),
    (255, 128, 0), (128, 255, 0), (255, 0, 128), (0, 128, 255),
]


# -----------------------------------------------------------------------------
# 데이터 구조
# -----------------------------------------------------------------------------
@dataclass
class CameraIntrinsics:
    K: np.ndarray
    D: np.ndarray
    depth_scale: float
    width: int
    height: int


@dataclass
class CameraFrame:
    cam_id: int
    intrinsics: CameraIntrinsics
    T_base_cam: np.ndarray
    color_bgr: np.ndarray
    depth_u16: np.ndarray


@dataclass
class CanonicalModel:
    name: str
    mesh: trimesh.Trimesh
    center: np.ndarray
    extents_m: np.ndarray
    is_watertight: bool


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


@dataclass
class ObjectTrack:
    track_id: str
    detections: Dict[int, CameraDetection]
    observed_masks: List[np.ndarray] = field(default_factory=list)
    fused_points_base: Optional[np.ndarray] = None


@dataclass
class PoseEstimate:
    T_base_obj: np.ndarray
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    euler_xyz_deg: np.ndarray
    scale: float = 1.0
    confidence: float = 0.0
    fitness: float = 0.0
    rmse: float = 0.0
    depth_score: float = 0.0
    coverage: float = 0.0
    silhouette_score: float = 0.0
    model_name: str = ""


# -----------------------------------------------------------------------------
# 공통 유틸
# -----------------------------------------------------------------------------
def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 3)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    return (T @ pts_h.T)[:3].T


def backproject_depth(cam: CameraFrame, mask: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = cam.depth_u16.shape
    K = cam.intrinsics.K
    z = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    valid = (z > DEPTH_POLICY["min_depth_m"]) & (z < DEPTH_POLICY["max_depth_m"])
    if mask is not None:
        valid &= (mask > 0)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float64)

    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = z[valid]
    x = (uu[valid] - K[0, 2]) * z / K[0, 0]
    y = (vv[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    if len(pts) == 0:
        return pts
    q = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return pts[np.sort(idx)]


def mask_bbox(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0, 0, 0, 0], dtype=np.int32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.int32)


def normalized_extent(ext: np.ndarray) -> np.ndarray:
    ext = np.asarray(ext, dtype=np.float64)
    if ext.size == 0:
        return np.zeros(3, dtype=np.float64)
    mx = np.max(ext)
    return np.sort(ext) / (mx + 1e-8)


def circular_hue_distance(h1: float, h2: float) -> float:
    dh = abs(float(h1) - float(h2))
    return min(dh, 180.0 - dh) / 90.0


def compute_mask_color_hist(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], mask, [12], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], mask, [4], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], mask, [4], [0, 256]).flatten()
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-8)
    return hist


def compute_mask_mean_hsv(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[mask > 0]
    if len(pix) == 0:
        return np.zeros(3, dtype=np.float32)
    h = pix[:, 0].astype(np.float32)
    ang = h / 180.0 * (2.0 * np.pi)
    hue = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    if hue < 0:
        hue += 2.0 * np.pi
    sat = pix[:, 1].astype(np.float32).mean()
    val = pix[:, 2].astype(np.float32).mean()
    return np.array([hue / (2.0 * np.pi) * 180.0, sat, val], dtype=np.float32)


# -----------------------------------------------------------------------------
# 로딩
# -----------------------------------------------------------------------------
def load_calibration(data_dir: Path, intrinsics_dir: Path):
    intrinsics: List[CameraIntrinsics] = []
    for ci in range(3):
        npz = np.load(str(intrinsics_dir / f"cam{ci}.npz"), allow_pickle=True)
        intrinsics.append(CameraIntrinsics(
            K=npz["color_K"].astype(np.float64),
            D=npz["color_D"].astype(np.float64),
            depth_scale=float(npz["depth_scale_m_per_unit"]),
            width=int(npz["color_w"]),
            height=int(npz["color_h"]),
        ))

    ext_dir = data_dir / "cube_session_01" / "calib_out_cube"
    extrinsics = {0: np.eye(4, dtype=np.float64)}
    for ci in [1, 2]:
        extrinsics[ci] = np.load(str(ext_dir / f"T_C0_C{ci}.npy")).astype(np.float64)
    return intrinsics, extrinsics


def load_frame(data_dir: Path, frame_id: str, intrinsics, extrinsics, capture_subdir: str = "object_capture"):
    img_dir = data_dir / capture_subdir
    frames: List[CameraFrame] = []
    for ci in range(3):
        rgb_path = img_dir / f"cam{ci}" / f"rgb_{frame_id}.jpg"
        depth_path = img_dir / f"cam{ci}" / f"depth_{frame_id}.png"
        c = cv2.imread(str(rgb_path))
        d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if c is None or d is None:
            raise FileNotFoundError(f"missing frame: cam{ci} {frame_id}")
        frames.append(CameraFrame(ci, intrinsics[ci], extrinsics[ci], c, d))
    return frames


def normalize_glb(glb_path: Path) -> CanonicalModel:
    scene = trimesh.load(str(glb_path))
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) if isinstance(scene, trimesh.Scene) else scene
    return CanonicalModel(
        name=glb_path.stem,
        mesh=mesh,
        center=mesh.centroid.copy(),
        extents_m=mesh.bounding_box.extents.copy(),
        is_watertight=bool(mesh.is_watertight),
    )


def sample_model_points(model: CanonicalModel, n: int = 12000) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(model.mesh, n)
    return (pts - model.center).astype(np.float64)


# -----------------------------------------------------------------------------
# 테이블 추정 / fallback segmentation support
# -----------------------------------------------------------------------------
def fit_plane_from_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return None
    n = n / norm
    d = -np.dot(n, p0)
    return n, d


def estimate_table_plane(frames: List[CameraFrame], max_points: int = 12000):
    pts_all = []
    for cam in frames:
        pts_cam = backproject_depth(cam)
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        if len(pts_base) > 0:
            pts_all.append(pts_base)
    if not pts_all:
        raise RuntimeError("테이블 평면 추정 실패: depth 없음")
    pts = np.vstack(pts_all)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]

    best_inliers = np.array([], dtype=np.int64)
    thr = DEPTH_POLICY["table_dist_thresh_m"]
    rng = np.random.default_rng(42)
    n_pts = len(pts)

    for _ in range(800):
        ids = rng.choice(n_pts, 3, replace=False)
        fit = fit_plane_from_points(pts[ids[0]], pts[ids[1]], pts[ids[2]])
        if fit is None:
            continue
        n, d = fit
        dist = np.abs(pts @ n + d)
        inliers = np.where(dist < thr)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 100:
        raise RuntimeError("테이블 평면 추정 실패: inlier 부족")

    inlier_pts = pts[best_inliers]
    c = inlier_pts.mean(axis=0)
    uu, ss, vv = np.linalg.svd(inlier_pts - c, full_matrices=False)
    n = vv[-1]
    n = n / (np.linalg.norm(n) + 1e-8)
    d = -np.dot(n, c)

    horiz = np.stack([inlier_pts[:, 0], inlier_pts[:, 2]], axis=1)
    c_h = np.array([c[0], c[2]])
    table_radius = np.percentile(np.linalg.norm(horiz - c_h, axis=1), 90) * 1.15
    return n, d, c, float(table_radius)


# -----------------------------------------------------------------------------
# Segmenter interface
# -----------------------------------------------------------------------------
class BaseSegmenter:
    def predict(self, image_bgr: np.ndarray, cam: Optional[CameraFrame] = None, aux: Optional[dict] = None) -> List[dict]:
        raise NotImplementedError


class ThresholdFallbackSegmenter(BaseSegmenter):
    """모델이 없을 때 동작하는 fallback. 실전용보다는 구조 검증용."""
    def predict(self, image_bgr: np.ndarray, cam: Optional[CameraFrame] = None, aux: Optional[dict] = None) -> List[dict]:
        if cam is None or aux is None or "plane_n" not in aux:
            raise ValueError("ThresholdFallbackSegmenter는 cam과 aux(plane)가 필요합니다.")
        plane_n = aux["plane_n"]
        plane_d = aux["plane_d"]
        table_center = aux.get("table_center")
        table_radius = aux.get("table_radius")

        h, w = cam.depth_u16.shape
        z = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
        valid = (z > DEPTH_POLICY["min_depth_m"]) & (z < DEPTH_POLICY["max_depth_m"])
        if valid.sum() == 0:
            return []

        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        u_valid = uu[valid]
        v_valid = vv[valid]
        z_valid = z[valid]
        K = cam.intrinsics.K
        x = (u_valid - K[0, 2]) * z_valid / K[0, 0]
        y = (v_valid - K[1, 2]) * z_valid / K[1, 1]
        pts_cam = np.stack([x, y, z_valid], axis=1)
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        signed = pts_base @ plane_n + plane_d
        if np.sum(signed > DEPTH_POLICY["object_min_height_m"]) < np.sum(signed < -DEPTH_POLICY["object_min_height_m"]):
            signed = -signed

        keep = (signed > DEPTH_POLICY["object_min_height_m"]) & (signed < DEPTH_POLICY["object_max_height_m"])
        if table_center is not None and table_radius is not None:
            horiz = np.stack([pts_base[:, 0], pts_base[:, 2]], axis=1)
            c_h = np.array([table_center[0], table_center[2]])
            keep &= (np.linalg.norm(horiz - c_h, axis=1) < table_radius)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[v_valid[keep], u_valid[keep]] = 255

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        color_mask = ((sat > 30) | ((sat > 15) & (val < 120) & (val > 20))).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, color_mask)

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
        mask = cv2.dilate(mask, k3, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        results = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 150:
                continue
            obj_mask = np.zeros_like(mask)
            obj_mask[labels == i] = 255
            results.append({"mask": obj_mask, "score": min(1.0, area / float(mask.size)), "label": None})
        return results


class YoloSegSegmenter(BaseSegmenter):
    def __init__(self, model_path: str, conf: float = 0.35):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("ultralytics가 설치되어 있지 않습니다. pip install ultralytics 후 사용하세요.") from e
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, image_bgr: np.ndarray, cam: Optional[CameraFrame] = None, aux: Optional[dict] = None) -> List[dict]:
        results = self.model(image_bgr, conf=self.conf, verbose=False)
        outputs = []
        for r in results:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            for i, mask in enumerate(masks):
                mask_u8 = (cv2.resize(mask.astype(np.uint8), (image_bgr.shape[1], image_bgr.shape[0])) > 0).astype(np.uint8) * 255
                outputs.append({
                    "mask": mask_u8,
                    "score": float(scores[i]),
                    "label": str(labels[i]),
                })
        return outputs


def build_segmenter(name: str, yolo_model: Optional[str] = None) -> BaseSegmenter:
    if name == "threshold":
        return ThresholdFallbackSegmenter()
    if name == "yolo":
        if not yolo_model:
            raise ValueError("--segmenter yolo 사용 시 --yolo_model 필요")
        return YoloSegSegmenter(yolo_model)
    raise ValueError(f"unknown segmenter: {name}")


# -----------------------------------------------------------------------------
# Detection 생성
# -----------------------------------------------------------------------------
def create_camera_detection(cam: CameraFrame, mask: np.ndarray, det_id: str, score: float = 1.0) -> Optional[CameraDetection]:
    area_px = int(np.count_nonzero(mask))
    if area_px == 0:
        return None
    bbox_xyxy = mask_bbox(mask)
    ys, xs = np.where(mask > 0)
    centroid_uv = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    color_hist = compute_mask_color_hist(cam.color_bgr, mask)
    mean_hsv = compute_mask_mean_hsv(cam.color_bgr, mask)

    pts_cam = backproject_depth(cam, mask)
    centroid_base = None
    extents_base = None
    if len(pts_cam) > 20:
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        centroid_base = pts_base.mean(axis=0)
        extents_base = pts_base.max(axis=0) - pts_base.min(axis=0)

    return CameraDetection(
        cam_id=cam.cam_id,
        det_id=det_id,
        mask_u8=mask,
        bbox_xyxy=bbox_xyxy,
        area_px=area_px,
        score=float(score),
        centroid_uv=centroid_uv,
        color_hist=color_hist,
        mean_hsv=mean_hsv,
        centroid_base=centroid_base,
        extents_base=extents_base,
    )


def build_camera_instances(cam: CameraFrame, segmenter: BaseSegmenter, aux: Optional[dict] = None, min_score: float = 0.3, min_area: int = 120):
    preds = segmenter.predict(cam.color_bgr, cam=cam, aux=aux)
    detections: List[CameraDetection] = []
    det_idx = 0
    for pred in preds:
        mask = pred["mask"].astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        score = float(pred.get("score", 1.0))
        if score < min_score:
            continue
        if int(np.count_nonzero(mask)) < min_area:
            continue
        det = create_camera_detection(cam, mask, f"cam{cam.cam_id}_seg_{det_idx}", score=score)
        if det is not None:
            detections.append(det)
            det_idx += 1
    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections


def save_camera_detections(detections_by_cam: Dict[int, List[CameraDetection]], frames: List[CameraFrame], frame_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in frames:
        vis = cam.color_bgr.copy()
        for di, det in enumerate(detections_by_cam.get(cam.cam_id, [])):
            color = COLORS[di % len(COLORS)]
            x0, y0, x1, y1 = det.bbox_xyxy.tolist()
            cnts, _ = cv2.findContours(det.mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, color, 2)
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), color, 1)
            cv2.putText(vis, f"d{di}:{det.score:.2f}", (x0, max(18, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imwrite(str(out_dir / f"detections_cam{cam.cam_id}_{frame_id}.png"), vis)


# -----------------------------------------------------------------------------
# 멀티뷰 association
# -----------------------------------------------------------------------------
def color_hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def extent_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.5
    return float(1.0 / (1.0 + 4.0 * np.linalg.norm(normalized_extent(a) - normalized_extent(b))))


def reprojection_center_error(det_a: CameraDetection, det_b: CameraDetection, cam_b: CameraFrame) -> float:
    if det_a.centroid_base is None:
        return 1e9
    p_cam = (np.linalg.inv(cam_b.T_base_cam) @ np.append(det_a.centroid_base, 1.0))[:3]
    if p_cam[2] <= 0.05:
        return 1e9
    K = cam_b.intrinsics.K
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    pred_uv = np.array([u, v], dtype=np.float64)
    return float(np.linalg.norm(pred_uv - det_b.centroid_uv))


def detection_pair_score(a: CameraDetection, b: CameraDetection, frame_by_id: Dict[int, CameraFrame]) -> float:
    pos_score = 0.0
    if a.centroid_base is not None and b.centroid_base is not None:
        dist = np.linalg.norm(a.centroid_base - b.centroid_base)
        pos_score = float(np.exp(-((dist / 0.045) ** 2)))

    reproj_ab = reprojection_center_error(a, b, frame_by_id[b.cam_id])
    reproj_ba = reprojection_center_error(b, a, frame_by_id[a.cam_id])
    reproj = min(reproj_ab, reproj_ba)
    reproj_score = float(np.exp(-((reproj / 24.0) ** 2)))

    app_score = color_hist_similarity(a.color_hist, b.color_hist)
    size_score = extent_similarity(a.extents_base, b.extents_base)
    return 0.40 * pos_score + 0.35 * reproj_score + 0.15 * app_score + 0.10 * size_score


def build_track_observed_masks(track: ObjectTrack, frames: List[CameraFrame]) -> List[np.ndarray]:
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


def associate_detections_across_views(frames: List[CameraFrame], detections_by_cam: Dict[int, List[CameraDetection]], threshold: float = 0.50):
    frame_by_id = {cam.cam_id: cam for cam in frames}
    all_dets: List[CameraDetection] = []
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
                scores = [detection_pair_score(cand, m, frame_by_id) for m in members.values()]
                score = float(np.mean(scores)) if scores else 0.0
                if score > best_score:
                    best_score = score
                    best_det = cand
            if best_det is not None and best_score >= threshold:
                members[cam.cam_id] = best_det
                used.add(best_det.det_id)

        track = ObjectTrack(track_id=f"track_{track_idx:02d}", detections=members)
        track.observed_masks = build_track_observed_masks(track, frames)
        tracks.append(track)
        track_idx += 1

    tracks.sort(key=lambda t: sum(det.area_px for det in t.detections.values()), reverse=True)
    return tracks


def fuse_track_points(frames: List[CameraFrame], track: ObjectTrack) -> np.ndarray:
    all_pts = []
    for cam, mask in zip(frames, track.observed_masks):
        if np.count_nonzero(mask) == 0:
            continue
        pts = backproject_depth(cam, mask)
        if len(pts) == 0:
            continue
        all_pts.append(transform_points(pts, cam.T_base_cam))
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.vstack(all_pts)
    pts = voxel_downsample(pts, DEPTH_POLICY["voxel_size_m"])
    return pts


# -----------------------------------------------------------------------------
# GLB-aware shortlist / rendering
# -----------------------------------------------------------------------------
def pca_descriptor(pts: np.ndarray) -> np.ndarray:
    if len(pts) < 8:
        return np.zeros(3, dtype=np.float64)
    c = pts - pts.mean(axis=0)
    vals = np.sqrt(np.maximum(np.linalg.eigvalsh(c.T @ c / max(len(pts), 1)), 0.0))
    vals = np.sort(vals)[::-1]
    return vals / (vals[0] + 1e-8)


def model_appearance_score(model_name: str, track_mean_hsv: Optional[np.ndarray]) -> float:
    prior = OBJECT_COLOR_PRIORS_HSV.get(model_name)
    if prior is None or track_mean_hsv is None:
        return 0.5
    hue_score = 1.0 - min(1.0, circular_hue_distance(track_mean_hsv[0], prior[0]))
    sat_score = 1.0 - min(1.0, abs(float(track_mean_hsv[1]) - float(prior[1])) / 180.0)
    val_score = 1.0 - min(1.0, abs(float(track_mean_hsv[2]) - float(prior[2])) / 180.0)
    return float(np.clip(0.60 * hue_score + 0.25 * sat_score + 0.15 * val_score, 0.0, 1.0))


def aggregate_track_mean_hsv(track: ObjectTrack) -> Optional[np.ndarray]:
    dets = list(track.detections.values())
    if not dets:
        return None
    weights = np.array([max(d.area_px, 1) for d in dets], dtype=np.float64)
    angles = np.array([d.mean_hsv[0] / 180.0 * 2.0 * np.pi for d in dets], dtype=np.float64)
    hue = np.arctan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights))
    if hue < 0:
        hue += 2.0 * np.pi
    sat = float(np.average([d.mean_hsv[1] for d in dets], weights=weights))
    val = float(np.average([d.mean_hsv[2] for d in dets], weights=weights))
    return np.array([hue / (2.0 * np.pi) * 180.0, sat, val], dtype=np.float32)


def render_model_to_mask(model: CanonicalModel, T_base_obj: np.ndarray, cam: CameraFrame, scale: float = 1.0) -> np.ndarray:
    h, w = cam.intrinsics.height, cam.intrinsics.width
    mask = np.zeros((h, w), dtype=np.uint8)
    verts = (model.mesh.vertices - model.center) * scale
    verts_base = transform_points(verts, T_base_obj)
    verts_cam = transform_points(verts_base, np.linalg.inv(cam.T_base_cam))
    z = verts_cam[:, 2]
    valid = z > 0.05
    verts_cam = verts_cam[valid]
    if len(verts_cam) < 3:
        return mask
    K = cam.intrinsics.K
    u = (K[0, 0] * verts_cam[:, 0] / verts_cam[:, 2] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * verts_cam[:, 1] / verts_cam[:, 2] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pts = np.stack([u[ok], v[ok]], axis=1)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def render_compare_score(model: CanonicalModel, T_base_obj: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray], scale: float = 1.0) -> float:
    vals = []
    for cam, obs in zip(frames, observed_masks):
        pred = render_model_to_mask(model, T_base_obj, cam, scale=scale) > 0
        obs = obs > 0
        union = np.logical_or(pred, obs).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred, obs).sum()
        vals.append(inter / union)
    return float(np.mean(vals)) if vals else 0.0


def coarse_pose_init_from_track(cluster_pts: np.ndarray, model: CanonicalModel) -> np.ndarray:
    T = np.eye(4)
    if len(cluster_pts) == 0:
        return T
    T[:3, 3] = cluster_pts.mean(axis=0)
    return T


def shortlist_glb_candidates(track_pts: np.ndarray, track: ObjectTrack, models: Dict[str, CanonicalModel], frames: List[CameraFrame], top_k: int = 3):
    obj_desc = pca_descriptor(track_pts) if len(track_pts) else np.zeros(3)
    obj_ext = track_pts.max(axis=0) - track_pts.min(axis=0) if len(track_pts) else np.zeros(3)
    track_mean_hsv = aggregate_track_mean_hsv(track)

    results = []
    for name, model in models.items():
        mod_desc = pca_descriptor(model.mesh.vertices - model.center)
        desc_dist = np.linalg.norm(obj_desc - mod_desc)
        ext_dist = np.linalg.norm(normalized_extent(obj_ext) - normalized_extent(model.extents_m))
        shape_score = 1.0 / (1.0 + 8.0 * desc_dist + 4.0 * ext_dist)
        app_score = model_appearance_score(name, track_mean_hsv)
        T0 = coarse_pose_init_from_track(track_pts, model)
        render_score = render_compare_score(model, T0, frames, track.observed_masks, scale=1.0)
        coarse = 0.35 * shape_score + 0.25 * app_score + 0.40 * render_score
        results.append((name, float(coarse)))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# -----------------------------------------------------------------------------
# Local search + ICP refinement
# -----------------------------------------------------------------------------
def pca_axes(pts: np.ndarray):
    if len(pts) < 8:
        return np.eye(3), np.ones(3)
    c = pts - pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(c.T @ c / max(len(pts), 1))
    order = np.argsort(vals)[::-1]
    return vecs[:, order], np.sqrt(np.maximum(vals[order], 0))


def rotation_candidates(src_pts: np.ndarray, tgt_pts: np.ndarray):
    R_s, _ = pca_axes(src_pts)
    R_t, _ = pca_axes(tgt_pts)
    cands = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i in range(3):
                R[:, i] = signs[i] * R_s[:, perm[i]]
            if np.linalg.det(R) < 0:
                continue
            cands.append(R_t @ R.T)
    uniq = []
    for R in cands:
        if not uniq:
            uniq.append(R)
            continue
        angs = [np.arccos(np.clip((np.trace(R @ U.T) - 1) / 2, -1, 1)) for U in uniq]
        if min(angs) > np.radians(8):
            uniq.append(R)
    return uniq[:12] if uniq else [np.eye(3)]


def best_fit_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    AA = A - ca
    BB = B - cb
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cb - R @ ca
    return R, t


def icp_point_to_point(src_pts: np.ndarray, tgt_pts: np.ndarray, init_T: np.ndarray, max_iter: int = 30, max_corr: float = 0.02):
    if len(src_pts) == 0 or len(tgt_pts) == 0:
        return init_T.copy(), 0.0, np.inf
    T = init_T.copy()
    tree = cKDTree(tgt_pts)
    fitness = 0.0
    rmse = np.inf
    for _ in range(max_iter):
        src_tf = transform_points(src_pts, T)
        dists, idx = tree.query(src_tf, k=1)
        valid = dists < max_corr
        if valid.sum() < 20:
            break
        A = src_tf[valid]
        B = tgt_pts[idx[valid]]
        R, t = best_fit_transform(A, B)
        T_up = np.eye(4)
        T_up[:3, :3] = R
        T_up[:3, 3] = t
        T = T_up @ T
        fitness = float(valid.mean())
        rmse = float(np.sqrt((dists[valid] ** 2).mean()))
        if rmse < 1e-4:
            break
    return T, fitness, rmse


def project_points_to_mask(obj_pts: np.ndarray, cam: CameraFrame) -> np.ndarray:
    h, w = cam.intrinsics.height, cam.intrinsics.width
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(obj_pts) == 0:
        return mask
    p = transform_points(obj_pts, np.linalg.inv(cam.T_base_cam))
    valid = p[:, 2] > 0.05
    p = p[valid]
    if len(p) == 0:
        return mask
    K = cam.intrinsics.K
    u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() < 3:
        return mask
    pts2 = np.stack([u[ok], v[ok]], axis=1)
    hull = cv2.convexHull(pts2)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def mv_depth_score(aligned_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray], tol: float = 0.015) -> float:
    total_ok = 0
    total_valid = 0
    for cam, mask in zip(frames, observed_masks):
        p = transform_points(aligned_pts, np.linalg.inv(cam.T_base_cam))
        valid = p[:, 2] > 0.05
        p = p[valid]
        if len(p) == 0:
            continue
        K = cam.intrinsics.K
        u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
        ok = (u >= 0) & (u < cam.intrinsics.width) & (v >= 0) & (v < cam.intrinsics.height)
        if ok.sum() == 0:
            continue
        in_mask = mask[v[ok], u[ok]] > 0
        if in_mask.sum() == 0:
            continue
        zm = p[ok, 2][in_mask]
        zr = cam.depth_u16[v[ok], u[ok]][in_mask].astype(np.float64) * cam.intrinsics.depth_scale
        hd = zr > 0.05
        if hd.sum() == 0:
            continue
        total_ok += int((np.abs(zm[hd] - zr[hd]) < tol).sum())
        total_valid += int(hd.sum())
    return total_ok / max(total_valid, 1)


def coverage_score(aligned_pts: np.ndarray, obs_pts: np.ndarray, radius: float = 0.006) -> float:
    if len(aligned_pts) == 0 or len(obs_pts) == 0:
        return 0.0
    tree_obs = cKDTree(obs_pts)
    d1, _ = tree_obs.query(aligned_pts, k=1)
    fwd = float((d1 < radius).mean())
    tree_aln = cKDTree(aligned_pts)
    d2, _ = tree_aln.query(obs_pts, k=1)
    bwd = float((d2 < radius).mean())
    return 0.5 * (fwd + bwd)


def silhouette_iou_score(aligned_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray]) -> float:
    vals = []
    for cam, obs in zip(frames, observed_masks):
        pred = project_points_to_mask(aligned_pts, cam) > 0
        obs = obs > 0
        union = np.logical_or(pred, obs).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred, obs).sum()
        vals.append(inter / union)
    return float(np.mean(vals)) if vals else 0.0


def combine_pose_scores(model_name: str, depth_score: float, coverage: float, silhouette: float) -> float:
    if OBJECT_SYMMETRY.get(model_name, "none") == "yaw":
        return 0.50 * depth_score + 0.20 * coverage + 0.30 * silhouette
    return 0.45 * depth_score + 0.35 * coverage + 0.20 * silhouette


def coarse_pose_search(model: CanonicalModel, model_pts: np.ndarray, track_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray]):
    if len(track_pts) == 0:
        return [(np.eye(4), 0.0, 1.0)]
    center = track_pts.mean(axis=0)
    rots = rotation_candidates(model_pts, track_pts)
    trans_xy = [-0.03, 0.0, 0.03]
    trans_z = [-0.015, 0.0, 0.015]
    yaw_offsets = [0.0, np.radians(-20), np.radians(20)]
    scales = [0.94, 1.0, 1.06]

    ranked = []
    for scale in scales:
        scaled = model_pts * scale
        src_center = scaled.mean(axis=0)
        for R0 in rots[:6]:
            for dyaw in yaw_offsets:
                R_yaw = Rot.from_euler('z', dyaw).as_matrix()
                R = R_yaw @ R0
                for dx in trans_xy:
                    for dy in trans_xy:
                        for dz in trans_z:
                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3] = center + np.array([dx, dy, dz]) - R @ src_center
                            render_score = render_compare_score(model, T, frames, observed_masks, scale=scale)
                            ranked.append((T, float(render_score), float(scale)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:12]


def register_model(model: CanonicalModel, model_pts: np.ndarray, track_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray]) -> PoseEstimate:
    if len(track_pts) < 40:
        raise RuntimeError("정합 실패: track points too small")

    best: Optional[PoseEstimate] = None
    coarse_candidates = coarse_pose_search(model, model_pts, track_pts, frames, observed_masks)
    max_dim = float(np.max(model.extents_m))
    max_corr = max(0.008, max_dim * 0.25)

    for T0, pre_score, scale in coarse_candidates:
        scaled = model_pts * scale
        T_icp, fitness, rmse = icp_point_to_point(scaled, track_pts, T0, max_iter=35, max_corr=max_corr)
        aligned = transform_points(scaled, T_icp)
        ds = mv_depth_score(aligned, frames, observed_masks)
        cs = coverage_score(aligned, track_pts, radius=max(0.004, max_dim * 0.08))
        ss = silhouette_iou_score(aligned, frames, observed_masks)
        conf = 0.15 * pre_score + 0.85 * combine_pose_scores(model.name, ds, cs, ss)

        R = T_icp[:3, :3]
        U, _, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1
            R_clean = U @ Vt
        T_clean = T_icp.copy()
        T_clean[:3, :3] = R_clean
        rot = Rot.from_matrix(R_clean)
        pose = PoseEstimate(
            T_base_obj=T_clean,
            position_m=T_clean[:3, 3].copy(),
            quaternion_xyzw=rot.as_quat(),
            euler_xyz_deg=rot.as_euler('xyz', degrees=True),
            scale=float(scale),
            confidence=float(conf),
            fitness=float(fitness),
            rmse=float(rmse),
            depth_score=float(ds),
            coverage=float(cs),
            silhouette_score=float(ss),
            model_name=model.name,
        )
        if best is None or pose.confidence > best.confidence:
            best = pose

    if best is None:
        raise RuntimeError("정합 실패")
    return best


# -----------------------------------------------------------------------------
# 출력
# -----------------------------------------------------------------------------
def export_result(pose: PoseEstimate, model: CanonicalModel, frame_id: str, out_dir: Path, glb_src_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    label = OBJECT_LABELS.get(model.name, model.name)
    result = {
        "frame_id": frame_id,
        "object_name": model.name,
        "label": label,
        "coordinate_frame": "base (= cam0)",
        "unit": "meter",
        "position_m": pose.position_m.tolist(),
        "quaternion_xyzw": pose.quaternion_xyzw.tolist(),
        "euler_xyz_deg": pose.euler_xyz_deg.tolist(),
        "T_base_obj": pose.T_base_obj.tolist(),
        "rotation_matrix": pose.T_base_obj[:3, :3].tolist(),
        "scale": pose.scale,
        "real_size_m": {
            "x": float(model.extents_m[0] * pose.scale),
            "y": float(model.extents_m[1] * pose.scale),
            "z": float(model.extents_m[2] * pose.scale),
        },
        "confidence": pose.confidence,
        "fitness": pose.fitness,
        "rmse": pose.rmse,
        "depth_score": pose.depth_score,
        "coverage": pose.coverage,
        "silhouette_score": pose.silhouette_score,
    }

    np.savez(out_dir / f"pose_{model.name}_{frame_id}.npz",
             T_base_obj=pose.T_base_obj,
             position_m=pose.position_m,
             quaternion_xyzw=pose.quaternion_xyzw,
             scale=pose.scale)

    jp = out_dir / f"pose_{model.name}_{frame_id}.json"
    for coord in ["opencv", "isaac"]:
        scene = trimesh.load(str(glb_src_path))
        mesh = trimesh.util.concatenate(list(scene.geometry.values())) if isinstance(scene, trimesh.Scene) else scene.copy()
        verts = (mesh.vertices - model.center) * pose.scale
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts_pose = (pose.T_base_obj @ verts_h.T)[:3].T
        if coord == "isaac":
            verts_pose = (T_ISAAC_CV @ np.hstack([verts_pose, np.ones((len(verts_pose), 1))]).T)[:3].T
        mesh.vertices = verts_pose
        suffix = "" if coord == "opencv" else "_isaac"
        gp = out_dir / f"{model.name}_posed_{frame_id}{suffix}.glb"
        mesh.export(str(gp), file_type="glb")
        result[f"posed_glb_{coord}"] = str(gp)

    with open(jp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def render_wireframe(mesh: trimesh.Trimesh, center: np.ndarray, pose: PoseEstimate, cam: CameraFrame, color=(0, 255, 0), thickness: int = 2):
    img = cam.color_bgr.copy()
    h, w = img.shape[:2]
    K = cam.intrinsics.K
    verts = (mesh.vertices - center) * pose.scale
    vb = transform_points(verts, pose.T_base_obj)
    vc = transform_points(vb, np.linalg.inv(cam.T_base_cam))
    z = vc[:, 2]
    ok = z > 0.05
    pu = np.full(len(verts), -1.0)
    pv = np.full(len(verts), -1.0)
    pu[ok] = K[0, 0] * vc[ok, 0] / z[ok] + K[0, 2]
    pv[ok] = K[1, 1] * vc[ok, 1] / z[ok] + K[1, 2]
    for e0, e1 in mesh.edges_unique:
        if not (ok[e0] and ok[e1]):
            continue
        p0 = (int(pu[e0]), int(pv[e0]))
        p1 = (int(pu[e1]), int(pv[e1]))
        if abs(p0[0]) > 2 * w or abs(p0[1]) > 2 * h or abs(p1[0]) > 2 * w or abs(p1[1]) > 2 * h:
            continue
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


def save_combined_overlay(all_poses: List[PoseEstimate], all_models: List[CanonicalModel], all_masks: List[List[np.ndarray]], frames: List[CameraFrame], frame_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    base_imgs = [cam.color_bgr.copy() for cam in frames]
    for obj_idx, (pose, model, masks) in enumerate(zip(all_poses, all_models, all_masks)):
        color = COLORS[obj_idx % len(COLORS)]
        for ci, cam in enumerate(frames):
            wire = render_wireframe(model.mesh, model.center, pose, cam, color=color, thickness=2)
            diff = cv2.absdiff(wire, cam.color_bgr)
            wire_mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 8
            base_imgs[ci][wire_mask] = wire[wire_mask]
            cnts, _ = cv2.findContours(masks[ci], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base_imgs[ci], cnts, -1, color, 1)
            if masks[ci].any():
                ys, xs = np.where(masks[ci] > 0)
                anchor = (int(xs.mean()), max(20, int(ys.min()) - 4))
                cv2.putText(base_imgs[ci], OBJECT_LABELS.get(model.name, model.name), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for ci in range(len(frames)):
        cv2.putText(base_imgs[ci], f"cam{ci}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    combined = np.hstack(base_imgs)
    cv2.imwrite(str(out_dir / f"overlay_{frame_id}.png"), combined)


# -----------------------------------------------------------------------------
# 메인 파이프라인
# -----------------------------------------------------------------------------
def run_pipeline(data_dir: str,
                 intrinsics_dir: str,
                 frame_id: str,
                 glb_path: Optional[str] = None,
                 output_dir: str = "src/output/pose_pipeline_v3",
                 multi_object: bool = False,
                 capture_subdir: str = "object_capture",
                 segmenter_name: str = "threshold",
                 yolo_model: Optional[str] = None):
    data_dir_p = Path(data_dir)
    intr_dir_p = Path(intrinsics_dir)
    out = Path(output_dir)

    print("=" * 64)
    print(f" RGB-first Multi-View CAD Pose — Frame {frame_id} {'[MULTI]' if multi_object else '[SINGLE]'}")
    print("=" * 64)

    intrinsics, extrinsics = load_calibration(data_dir_p, intr_dir_p)
    glb_paths: Dict[str, Path] = {}
    all_models: Dict[str, CanonicalModel] = {}
    if glb_path:
        gp = Path(glb_path)
        model = normalize_glb(gp)
        all_models[model.name] = model
        glb_paths[model.name] = gp
    else:
        for i in range(1, 5):
            p = data_dir_p / f"object_{i:03d}.glb"
            if p.exists():
                model = normalize_glb(p)
                all_models[model.name] = model
                glb_paths[model.name] = p

    frames = load_frame(data_dir_p, frame_id, intrinsics, extrinsics, capture_subdir=capture_subdir)

    # plane only for fallback segmenter auxiliary use
    plane_n, plane_d, table_center, table_radius = estimate_table_plane(frames)
    aux = {
        "plane_n": plane_n,
        "plane_d": plane_d,
        "table_center": table_center,
        "table_radius": table_radius,
    }
    segmenter = build_segmenter(segmenter_name, yolo_model=yolo_model)

    # 1) RGB segmentation / detections
    print("\n[1] RGB instance segmentation")
    detections_by_cam: Dict[int, List[CameraDetection]] = {}
    for cam in frames:
        detections_by_cam[cam.cam_id] = build_camera_instances(cam, segmenter, aux=aux)
        print(f"  cam{cam.cam_id}: detections={len(detections_by_cam[cam.cam_id])}")
    save_camera_detections(detections_by_cam, frames, frame_id, out)

    # 2) cross-view association
    print("\n[2] Calibration-aware association")
    tracks = associate_detections_across_views(frames, detections_by_cam, threshold=0.50)
    if not tracks:
        raise RuntimeError("association 실패: track이 0개")
    print(f"  tracks={len(tracks)}")
    for ti, track in enumerate(tracks):
        cams = ",".join([f"cam{cid}" for cid in sorted(track.detections.keys())])
        area_sum = sum(det.area_px for det in track.detections.values())
        print(f"    #{ti} {track.track_id}: views={len(track.detections)} [{cams}] area={area_sum}")

    # 3) per-object depth fusion + shortlist + pose
    print("\n[3] Depth fusion + GLB-aware shortlist + pose refinement")
    all_results = []
    all_poses: List[PoseEstimate] = []
    all_model_objs: List[CanonicalModel] = []
    all_masks_list: List[List[np.ndarray]] = []
    used_glbs = set()

    active_tracks = tracks if multi_object else tracks[:1]
    for ti, track in enumerate(active_tracks):
        if multi_object and len(used_glbs) >= len(all_models):
            break
        print(f"  - {track.track_id}")
        track_pts = fuse_track_points(frames, track)
        track.fused_points_base = track_pts
        if len(track_pts) < 60:
            print("    skip: fused points too small")
            continue
        track_pts = voxel_downsample(track_pts, DEPTH_POLICY["voxel_size_m"])
        ext = track_pts.max(axis=0) - track_pts.min(axis=0)
        print(f"    fused_pts={len(track_pts)} extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm")

        if glb_path:
            shortlist = [(Path(glb_path).stem, 1.0)]
        elif not multi_object:
            fixed_name = FRAME_TO_GLB.get(int(frame_id))
            if fixed_name is None:
                raise RuntimeError(f"frame {frame_id}에 대한 GLB mapping 없음")
            shortlist = [(fixed_name, 1.0)]
            print(f"    fixed mapping: {fixed_name} ({OBJECT_LABELS.get(fixed_name, '')})")
        else:
            candidate_models = {k: v for k, v in all_models.items() if k not in used_glbs}
            shortlist = shortlist_glb_candidates(track_pts, track, candidate_models, frames, top_k=min(3, len(candidate_models)))
            print("    shortlist:", ", ".join([f"{n}:{s:.3f}" for n, s in shortlist]))

        best_pose = None
        best_model = None
        best_score = -1.0
        for cand_name, coarse_score in shortlist:
            if cand_name not in all_models:
                continue
            model = all_models[cand_name]
            model_pts = sample_model_points(model)
            try:
                pose = register_model(model, model_pts, track_pts, frames, track.observed_masks)
            except RuntimeError:
                continue
            final_score = 0.20 * coarse_score + 0.80 * pose.confidence
            print(f"      candidate={cand_name} coarse={coarse_score:.3f} pose_conf={pose.confidence:.3f} fit={pose.fitness:.3f} sil={pose.silhouette_score:.3f} final={final_score:.3f}")
            if final_score > best_score:
                best_score = final_score
                best_pose = pose
                best_model = model

        if best_pose is None or best_model is None:
            print("    fail: no valid pose")
            continue
        if best_pose.confidence < 0.05:
            print(f"    fail: confidence too low ({best_pose.confidence:.3f})")
            continue

        used_glbs.add(best_model.name)
        print(f"    selected: {best_model.name} conf={best_pose.confidence:.3f} fit={best_pose.fitness:.3f} sil={best_pose.silhouette_score:.3f}")
        result = export_result(best_pose, best_model, frame_id, out, glb_paths[best_model.name])
        all_results.append(result)
        all_poses.append(best_pose)
        all_model_objs.append(best_model)
        all_masks_list.append(track.observed_masks)

    if all_poses:
        save_combined_overlay(all_poses, all_model_objs, all_masks_list, frames, frame_id, out)
        print(f"\n  overlay: {out / f'overlay_{frame_id}.png'}")
    else:
        print("\n  [WARN] export할 pose가 없습니다.")

    return all_results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RGB-first calibration-aware multi-view CAD pose pipeline")
    parser.add_argument("--data_dir", default="src/data")
    parser.add_argument("--intrinsics_dir", default="src/intrinsics")
    parser.add_argument("--capture_subdir", default="object_capture")
    parser.add_argument("--frame_id", default=None)
    parser.add_argument("--glb", default=None)
    parser.add_argument("--output_dir", default="src/output/pose_pipeline_v3")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--segmenter", default="threshold", choices=["threshold", "yolo"])
    parser.add_argument("--yolo_model", default=None)
    args = parser.parse_args()

    if args.batch:
        cam0_dir = Path(args.data_dir) / args.capture_subdir / "cam0"
        fids = sorted(f.stem.replace("rgb_", "") for f in cam0_dir.glob("rgb_*.jpg"))
        print(f"batch frames: {len(fids)}")
        all_results = []
        for fid in fids:
            try:
                res = run_pipeline(
                    data_dir=args.data_dir,
                    intrinsics_dir=args.intrinsics_dir,
                    frame_id=fid,
                    glb_path=args.glb,
                    output_dir=args.output_dir,
                    multi_object=args.multi,
                    capture_subdir=args.capture_subdir,
                    segmenter_name=args.segmenter,
                    yolo_model=args.yolo_model,
                )
                all_results.extend(res)
            except Exception as e:
                print(f"[ERROR] {fid}: {e}")
                all_results.append({"frame_id": fid, "error": str(e)})
        with open(Path(args.output_dir) / "batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    else:
        fid = args.frame_id or "000000"
        run_pipeline(
            data_dir=args.data_dir,
            intrinsics_dir=args.intrinsics_dir,
            frame_id=fid,
            glb_path=args.glb,
            output_dir=args.output_dir,
            multi_object=args.multi,
            capture_subdir=args.capture_subdir,
            segmenter_name=args.segmenter,
            yolo_model=args.yolo_model,
        )


if __name__ == "__main__":
    main()
