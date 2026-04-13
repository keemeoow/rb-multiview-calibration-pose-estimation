#!/usr/bin/env python3
"""
Non-YOLO multi-view RGB-D + calibration + GLB 6D pose pipeline.

핵심 구조
- RGB: coarse proposals + GrabCut (선택적으로 SAM2)
- Calibration: cross-view association / reprojection consistency
- Depth: object-wise point cloud fusion
- GLB: render-and-compare shortlist + ICP refinement + posed GLB export

실행 예시
  python3 pose_pipeline_no_yolo.py --data_dir src/data --intrinsics_dir src/intrinsics --frame_id 000000 --multi
  python3 pose_pipeline_no_yolo.py --data_dir src/data --intrinsics_dir src/intrinsics --frame_id 000000 --glb src/data/object_004.glb

선택 사항
  --segmenter sam2 --sam2_cfg <cfg> --sam2_ckpt <ckpt>
    SAM2가 설치되어 있으면 box prompt 기반으로 더 정확한 mask를 생성

기본값은 GrabCut 기반이라 YOLO 없이 바로 실행 가능.
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
from scipy.spatial.transform import Rotation as Rot

try:
    import open3d as o3d
except Exception as e:  # pragma: no cover
    raise RuntimeError("open3d가 필요합니다. 현재 코드의 ICP/plane fitting에 사용됩니다.") from e

# -----------------------------
# Constants / metadata
# -----------------------------

T_ISAAC_CV = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

DEPTH_POLICY = {
    "min_depth_m": 0.10,
    "max_depth_m": 1.20,
    "voxel_size_m": 0.002,
    "table_dist_thresh_m": 0.008,
    "object_min_height_m": 0.004,
    "object_max_height_m": 0.18,
    "cluster_eps_m": 0.006,
    "cluster_min_pts": 50,
    "cluster_min_extent_m": 0.012,
    "cluster_max_extent_m": 0.25,
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

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class CameraIntrinsics:
    K: np.ndarray
    D: np.ndarray
    depth_scale: float
    width: int = 640
    height: int = 480


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
    mean_bgr: np.ndarray
    mean_hsv: np.ndarray
    is_watertight: bool


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
    render_rgb_score: float = 0.0


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
    best_pose: Optional[PoseEstimate] = None
    best_model_name: Optional[str] = None
    candidate_scores: List[Tuple[str, float]] = field(default_factory=list)

# -----------------------------
# IO / loading
# -----------------------------

def load_calibration(data_dir: Path, intrinsics_dir: Path):
    intrinsics = []
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
    extrinsics = {0: np.eye(4)}
    for ci in [1, 2]:
        extrinsics[ci] = np.load(str(ext_dir / f"T_C0_C{ci}.npy")).astype(np.float64)
    return intrinsics, extrinsics


def load_frame(data_dir: Path, frame_id: str, intrinsics, extrinsics, capture_subdir: str = "object_capture"):
    img_dir = data_dir / capture_subdir
    frames = []
    for ci in range(3):
        c = cv2.imread(str(img_dir / f"cam{ci}" / f"rgb_{frame_id}.jpg"))
        d = cv2.imread(str(img_dir / f"cam{ci}" / f"depth_{frame_id}.png"), cv2.IMREAD_UNCHANGED)
        if c is None or d is None:
            raise FileNotFoundError(f"cam{ci}/frame_{frame_id}")
        frames.append(CameraFrame(ci, intrinsics[ci], extrinsics[ci], c, d))
    return frames


def normalize_glb(glb_path: Path) -> CanonicalModel:
    scene = trimesh.load(str(glb_path))
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) if isinstance(scene, trimesh.Scene) else scene
    try:
        rgb = np.asarray(mesh.visual.vertex_colors)[:, :3].mean(axis=0).astype(np.uint8)
    except Exception:
        rgb = np.array([128, 128, 128], dtype=np.uint8)
    hsv = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
    return CanonicalModel(
        name=glb_path.stem,
        mesh=mesh,
        center=mesh.centroid.copy(),
        extents_m=mesh.bounding_box.extents.copy(),
        mean_bgr=rgb[::-1].copy(),
        mean_hsv=hsv,
        is_watertight=mesh.is_watertight,
    )


def sample_model_points(model: CanonicalModel, n=20000) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(model.mesh, n)
    return (pts - model.center).astype(np.float64)

# -----------------------------
# Geometry helpers
# -----------------------------

def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.copy()
    return (T @ np.hstack([pts, np.ones((len(pts), 1))]).T)[:3].T


def backproject_depth(cam: CameraFrame, mask: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = cam.depth_u16.shape
    K = cam.intrinsics.K
    ds = cam.intrinsics.depth_scale
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = cam.depth_u16.astype(np.float64) * ds
    ok = (z > DEPTH_POLICY["min_depth_m"]) & (z < DEPTH_POLICY["max_depth_m"])
    if mask is not None:
        ok &= (mask > 0)
    z = z[ok]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([
        (uu[ok] - K[0, 2]) * z / K[0, 0],
        (vv[ok] - K[1, 2]) * z / K[1, 1],
        z,
    ], axis=-1)


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


def estimate_table_plane(frames: List[CameraFrame]):
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
    plane, inlier_idx = pcd.segment_plane(
        distance_threshold=DEPTH_POLICY["table_dist_thresh_m"],
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    d /= np.linalg.norm(np.array([a, b, c])) + 1e-12
    table_pts = np.asarray(pcd.points)[inlier_idx]
    center = table_pts.mean(axis=0)
    horiz = table_pts[:, [0, 2]]
    tc = center[[0, 2]]
    radius = np.percentile(np.linalg.norm(horiz - tc, axis=1), 90) * 1.1
    return n, d, center, radius

# -----------------------------
# Segmentation backends (no YOLO)
# -----------------------------

class BaseSegmenter:
    def predict(self, cam: CameraFrame, coarse_boxes: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError


class GrabCutSegmenter(BaseSegmenter):
    def predict(self, cam: CameraFrame, coarse_boxes: List[np.ndarray]) -> List[np.ndarray]:
        image = cam.color_bgr
        masks: List[np.ndarray] = []
        for box in coarse_boxes:
            x0, y0, x1, y1 = [int(v) for v in box]
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(image.shape[1], x1); y1 = min(image.shape[0], y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            rect = (x0, y0, x1 - x0, y1 - y0)
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(image, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
            except cv2.error:
                continue
            out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            # keep largest contour only
            cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(cnt) < 120:
                continue
            keep = np.zeros_like(out)
            cv2.drawContours(keep, [cnt], -1, 255, cv2.FILLED)
            masks.append(keep)
        return nms_masks(masks, iou_thr=0.65)


class SAM2BoxSegmenter(BaseSegmenter):
    def __init__(self, model_cfg: str, checkpoint: str):
        self.available = False
        self.predictor = None
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam = build_sam2(model_cfg, checkpoint)
            self.predictor = SAM2ImagePredictor(sam)
            self.available = True
        except Exception:
            self.available = False

    def predict(self, cam: CameraFrame, coarse_boxes: List[np.ndarray]) -> List[np.ndarray]:
        if not self.available:
            raise RuntimeError("SAM2가 설치되어 있지 않거나 checkpoint 로드에 실패했습니다.")
        image_rgb = cv2.cvtColor(cam.color_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        masks: List[np.ndarray] = []
        for box in coarse_boxes:
            x0, y0, x1, y1 = [float(v) for v in box]
            box_np = np.array([x0, y0, x1, y1], dtype=np.float32)
            pred_masks, scores, _ = self.predictor.predict(box=box_np[None, :], multimask_output=True)
            if pred_masks is None or len(pred_masks) == 0:
                continue
            idx = int(np.argmax(scores))
            m = (pred_masks[idx].astype(np.uint8) * 255)
            masks.append(m)
        return nms_masks(masks, iou_thr=0.65)

# -----------------------------
# Coarse RGB/depth proposals (no YOLO)
# -----------------------------

def build_coarse_foreground_mask(cam: CameraFrame, plane_n, plane_d, table_center, table_radius) -> np.ndarray:
    h, w = cam.depth_u16.shape
    z = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    valid = (z > DEPTH_POLICY["min_depth_m"]) & (z < DEPTH_POLICY["max_depth_m"])
    if valid.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    pts_cam = np.stack([
        (uu[valid] - cam.intrinsics.K[0, 2]) * z[valid] / cam.intrinsics.K[0, 0],
        (vv[valid] - cam.intrinsics.K[1, 2]) * z[valid] / cam.intrinsics.K[1, 1],
        z[valid],
    ], axis=1)
    pts_base = transform_points(pts_cam, cam.T_base_cam)
    signed = pts_base @ plane_n + plane_d
    if np.sum(signed > DEPTH_POLICY["object_min_height_m"]) < np.sum(signed < -DEPTH_POLICY["object_min_height_m"]):
        signed = -signed
    keep = (signed > DEPTH_POLICY["object_min_height_m"]) & (signed < DEPTH_POLICY["object_max_height_m"])
    horiz = pts_base[:, [0, 2]]
    tc = table_center[[0, 2]]
    keep &= (np.linalg.norm(horiz - tc, axis=1) < table_radius)

    mask = np.zeros((h, w), dtype=np.uint8)
    valid_flat = valid.reshape(-1)
    uu_flat = uu.reshape(-1)[valid_flat]
    vv_flat = vv.reshape(-1)[valid_flat]
    mask[vv_flat[keep], uu_flat[keep]] = 255

    hsv = cv2.cvtColor(cam.color_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_fg = ((sat > 18) & (val > 15)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, color_fg)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    mask = cv2.dilate(mask, k3, iterations=1)
    return mask


def extract_coarse_boxes(mask: np.ndarray, min_area: int = 120) -> List[np.ndarray]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = [int(v) for v in stats[i, :4]]
        pad_x = max(6, int(w * 0.10))
        pad_y = max(6, int(h * 0.10))
        boxes.append(np.array([x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y], dtype=np.int32))
    return boxes


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union > 0 else 0.0


def nms_masks(masks: List[np.ndarray], iou_thr: float = 0.65) -> List[np.ndarray]:
    keep: List[np.ndarray] = []
    for m in sorted(masks, key=lambda x: int(np.count_nonzero(x)), reverse=True):
        if np.count_nonzero(m) < 100:
            continue
        if all(mask_iou(m, k) < iou_thr for k in keep):
            keep.append(m)
    return keep

# -----------------------------
# Detection creation / association
# -----------------------------

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


def create_camera_detection(cam: CameraFrame, mask: np.ndarray, det_id: str) -> Optional[CameraDetection]:
    area_px = int(np.count_nonzero(mask))
    if area_px < 100:
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
    )


def build_camera_detections(cam: CameraFrame, masks: List[np.ndarray]) -> List[CameraDetection]:
    out: List[CameraDetection] = []
    for i, mask in enumerate(masks):
        det = create_camera_detection(cam, mask, f"cam{cam.cam_id}_det_{i}")
        if det is not None:
            out.append(det)
    out.sort(key=lambda d: d.area_px, reverse=True)
    return out


def save_camera_detections(detections_by_cam, frames, frame_id, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam in frames:
        vis = cam.color_bgr.copy()
        for di, det in enumerate(detections_by_cam.get(cam.cam_id, [])):
            x0, y0, x1, y1 = det.bbox_xyxy.tolist()
            color = COLORS[di % len(COLORS)]
            cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), color, 2)
            cv2.putText(vis, f"d{di}", (x0, max(20, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(out_dir / f"detections_cam{cam.cam_id}_{frame_id}.png"), vis)


def color_hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def extent_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.5
    return float(1.0 / (1.0 + 4.0 * np.linalg.norm(normalized_extent(a) - normalized_extent(b))))


def reprojection_center_error(det_a: CameraDetection, det_b: CameraDetection, cam_a: CameraFrame, cam_b: CameraFrame) -> float:
    if det_a.centroid_base is None:
        return 1e9
    p_cam = (np.linalg.inv(cam_b.T_base_cam) @ np.append(det_a.centroid_base, 1.0))[:3]
    if p_cam[2] <= 0.05:
        return 1e9
    K = cam_b.intrinsics.K
    pred = np.array([
        K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
        K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2],
    ], dtype=np.float64)
    return float(np.linalg.norm(pred - det_b.centroid_uv))


def detection_pair_score(a: CameraDetection, b: CameraDetection, cam_a: CameraFrame, cam_b: CameraFrame) -> float:
    pos_score = 0.0
    if a.centroid_base is not None and b.centroid_base is not None:
        dist = np.linalg.norm(a.centroid_base - b.centroid_base)
        pos_score = float(np.exp(-((dist / 0.04) ** 2)))
    reproj = min(reprojection_center_error(a, b, cam_a, cam_b), reprojection_center_error(b, a, cam_b, cam_a))
    reproj_score = float(np.exp(-((reproj / 25.0) ** 2)))
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
        else:
            m = cv2.morphologyEx(det.mask_u8, cv2.MORPH_CLOSE, k)
            m = cv2.dilate(m, k, iterations=1)
            masks.append(m)
    return masks


def associate_detections_across_views(frames: List[CameraFrame], detections_by_cam: Dict[int, List[CameraDetection]]) -> List[ObjectTrack]:
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
                score = float(np.mean([
                    detection_pair_score(cand, m, frame_by_id[cand.cam_id], frame_by_id[m.cam_id])
                    for m in members.values()
                ]))
                if score > best_score:
                    best_score = score
                    best_det = cand
            if best_det is not None and best_score >= 0.50:
                members[cam.cam_id] = best_det
                used.add(best_det.det_id)
        track = ObjectTrack(track_id=f"track_{track_idx:02d}", detections=members)
        track.observed_masks = build_track_observed_masks(track, frames)
        tracks.append(track)
        track_idx += 1
    tracks.sort(key=lambda t: sum(d.area_px for d in t.detections.values()), reverse=True)
    return tracks


def fuse_masked_points(frames: List[CameraFrame], masks: List[np.ndarray]) -> np.ndarray:
    all_pts = []
    for cam, mask in zip(frames, masks):
        pts = backproject_depth(cam, mask)
        if len(pts) == 0:
            continue
        all_pts.append(transform_points(pts, cam.T_base_cam))
    if not all_pts:
        return np.zeros((0, 3), dtype=np.float64)
    merged = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    pcd = pcd.voxel_down_sample(DEPTH_POLICY["voxel_size_m"])
    if len(pcd.points) > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pcd.points)


def fuse_track_points(frames: List[CameraFrame], track: ObjectTrack) -> np.ndarray:
    return fuse_masked_points(frames, track.observed_masks)


def aggregate_track_mean_hsv(track: ObjectTrack) -> Optional[np.ndarray]:
    dets = list(track.detections.values())
    if not dets:
        return None
    weights = np.array([max(d.area_px, 1) for d in dets], dtype=np.float64)
    angles = np.array([d.mean_hsv[0] / 180.0 * (2 * np.pi) for d in dets])
    hue = np.arctan2(np.sum(np.sin(angles) * weights), np.sum(np.cos(angles) * weights))
    if hue < 0:
        hue += 2 * np.pi
    sat = float(np.average([d.mean_hsv[1] for d in dets], weights=weights))
    val = float(np.average([d.mean_hsv[2] for d in dets], weights=weights))
    return np.array([hue / (2 * np.pi) * 180.0, sat, val], dtype=np.float32)

# -----------------------------
# GLB rendering / scoring
# -----------------------------

def pca_axes(pts: np.ndarray):
    c = pts - pts.mean(0)
    vals, vecs = np.linalg.eigh(c.T @ c / max(len(pts), 1))
    order = np.argsort(vals)[::-1]
    return vecs[:, order], np.sqrt(np.maximum(vals[order], 0))


def pca_descriptor(pts: np.ndarray) -> np.ndarray:
    c = pts - pts.mean(0)
    vals = np.sqrt(np.maximum(np.linalg.eigvalsh(c.T @ c / max(len(pts), 1)), 0))
    vals = np.sort(vals)[::-1]
    return vals / vals[0] if vals[0] > 1e-8 else vals


def rotation_candidates(src_pts: np.ndarray, tgt_pts: np.ndarray) -> List[np.ndarray]:
    R_s, _ = pca_axes(src_pts)
    R_t, _ = pca_axes(tgt_pts)
    cands: List[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3), dtype=np.float64)
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
        ang_ok = all(np.arccos(np.clip((np.trace(R @ U.T) - 1) / 2, -1, 1)) > np.radians(8) for U in uniq)
        if ang_ok:
            uniq.append(R)
        if len(uniq) >= 8:
            break
    return uniq


def render_model_to_mask(model: CanonicalModel, T_base_obj: np.ndarray, cam: CameraFrame, scale: float = 1.0) -> np.ndarray:
    h, w = cam.intrinsics.height, cam.intrinsics.width
    mask = np.zeros((h, w), dtype=np.uint8)
    verts = (model.mesh.vertices - model.center) * scale
    vb = transform_points(verts, T_base_obj)
    vc = transform_points(vb, np.linalg.inv(cam.T_base_cam))
    z = vc[:, 2]
    valid = z > 0.05
    vc = vc[valid]
    if len(vc) < 3:
        return mask
    K = cam.intrinsics.K
    u = (K[0, 0] * vc[:, 0] / vc[:, 2] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * vc[:, 1] / vc[:, 2] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pts = np.stack([u[ok], v[ok]], axis=1)
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def render_model_to_overlay(model: CanonicalModel, T_base_obj: np.ndarray, cam: CameraFrame, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    mask = render_model_to_mask(model, T_base_obj, cam, scale)
    if np.count_nonzero(mask) == 0:
        return mask, np.zeros_like(cam.color_bgr)
    color_img = np.zeros_like(cam.color_bgr)
    color = model.mean_bgr.astype(np.uint8)
    color_img[mask > 0] = color
    return mask, color_img


def render_compare_score(model: CanonicalModel, T_base_obj: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray], scale: float = 1.0) -> Tuple[float, float]:
    sil_scores = []
    rgb_scores = []
    for cam, obs_mask in zip(frames, observed_masks):
        pred_mask, pred_rgb = render_model_to_overlay(model, T_base_obj, cam, scale)
        pred = pred_mask > 0
        obs = obs_mask > 0
        union = np.logical_or(pred, obs).sum()
        if union > 0:
            inter = np.logical_and(pred, obs).sum()
            sil_scores.append(inter / union)
        if pred.sum() > 0 and obs.sum() > 0:
            overlap = np.logical_and(pred, obs)
            if overlap.sum() > 30:
                img = cam.color_bgr.astype(np.float32)
                pred_col = pred_rgb.astype(np.float32)
                diff = np.mean(np.abs(img[overlap] - pred_col[overlap]))
                rgb_scores.append(max(0.0, 1.0 - diff / 128.0))
    sil = float(np.mean(sil_scores)) if sil_scores else 0.0
    rgb = float(np.mean(rgb_scores)) if rgb_scores else 0.0
    return sil, rgb


def model_appearance_score(model: CanonicalModel, track_mean_hsv: Optional[np.ndarray]) -> float:
    if track_mean_hsv is None:
        return 0.5
    hue_score = 1.0 - min(1.0, circular_hue_distance(track_mean_hsv[0], model.mean_hsv[0]))
    sat_score = 1.0 - min(1.0, abs(float(track_mean_hsv[1]) - float(model.mean_hsv[1])) / 180.0)
    val_score = 1.0 - min(1.0, abs(float(track_mean_hsv[2]) - float(model.mean_hsv[2])) / 180.0)
    return float(np.clip(0.60 * hue_score + 0.25 * sat_score + 0.15 * val_score, 0.0, 1.0))


def coarse_pose_init(cluster_pts: np.ndarray, model: CanonicalModel) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = cluster_pts.mean(axis=0)
    return T


def shortlist_glb_candidates(cluster_pts: np.ndarray, models: Dict[str, CanonicalModel], frames: List[CameraFrame], observed_masks: List[np.ndarray], track_mean_hsv: Optional[np.ndarray], top_k: int = 3):
    obj_desc = pca_descriptor(cluster_pts)
    obj_ext = cluster_pts.max(0) - cluster_pts.min(0)
    results = []
    for name, model in models.items():
        mod_desc = pca_descriptor(model.mesh.vertices - model.center)
        desc_dist = np.linalg.norm(obj_desc - mod_desc)
        obj_ext_n = np.sort(obj_ext) / (np.max(obj_ext) + 1e-8)
        mod_ext_n = np.sort(model.extents_m) / (np.max(model.extents_m) + 1e-8)
        ext_dist = np.linalg.norm(obj_ext_n - mod_ext_n)
        shape_score = 1.0 / (1.0 + 8.0 * desc_dist + 4.0 * ext_dist)
        app_score = model_appearance_score(model, track_mean_hsv)
        T0 = coarse_pose_init(cluster_pts, model)
        sil, rgb = render_compare_score(model, T0, frames, observed_masks, scale=1.0)
        coarse = 0.25 * shape_score + 0.20 * app_score + 0.35 * sil + 0.20 * rgb
        results.append((name, coarse))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# -----------------------------
# Pose search / refinement
# -----------------------------

def project_points_to_mask(obj_pts: np.ndarray, cam: CameraFrame) -> np.ndarray:
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
    u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask[v[ok], u[ok]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def mv_depth_score(aligned: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray], tol: float = 0.015) -> float:
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


def silhouette_iou_score(aligned: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray]) -> float:
    vals = []
    for cam, obs_mask in zip(frames, observed_masks):
        pred_mask = project_points_to_mask(aligned, cam)
        union = np.logical_or(pred_mask > 0, obs_mask > 0).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_mask > 0, obs_mask > 0).sum()
        vals.append(inter / union)
    return float(np.mean(vals)) if vals else 0.0


def cov_score(aligned: np.ndarray, tgt_pcd: o3d.geometry.PointCloud, radius: float = 0.005) -> float:
    obj_pts = np.asarray(tgt_pcd.points)
    mp = o3d.geometry.PointCloud(); mp.points = o3d.utility.Vector3dVector(aligned)
    fd = np.asarray(mp.compute_point_cloud_distance(tgt_pcd))
    fwd = (fd < radius).sum() / max(len(fd), 1)
    m = radius * 5
    lo = aligned.min(0) - m; hi = aligned.max(0) + m
    ib = np.all((obj_pts >= lo) & (obj_pts <= hi), axis=1)
    if ib.sum() < 10:
        return fwd * 0.01
    ip = o3d.geometry.PointCloud(); ip.points = o3d.utility.Vector3dVector(obj_pts[ib])
    rd = np.asarray(ip.compute_point_cloud_distance(mp))
    return fwd * ((rd < radius).sum() / max(len(rd), 1))


def combine_pose_scores(model_name: str, depth_score: float, coverage: float, silhouette: float, render_rgb: float) -> float:
    if OBJECT_SYMMETRY.get(model_name, "none") == "yaw":
        return 0.38 * depth_score + 0.17 * coverage + 0.25 * silhouette + 0.20 * render_rgb
    return 0.36 * depth_score + 0.24 * coverage + 0.24 * silhouette + 0.16 * render_rgb


def coarse_pose_search(model: CanonicalModel, model_pts: np.ndarray, obj_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray]) -> List[np.ndarray]:
    center = obj_pts.mean(axis=0)
    Rs = rotation_candidates(model_pts, obj_pts)
    offsets_xy = [-0.02, 0.0, 0.02]
    offsets_z = [-0.01, 0.0, 0.01]
    yaw_offsets = [-20, -10, 0, 10, 20]
    cands: List[Tuple[float, np.ndarray]] = []
    for R0 in Rs[:4]:
        for yaw_deg in yaw_offsets:
            Ry = Rot.from_euler('z', yaw_deg, degrees=True).as_matrix()
            R = Ry @ R0
            for dx in offsets_xy:
                for dy in offsets_xy:
                    for dz in offsets_z:
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = center + np.array([dx, dy, dz], dtype=np.float64)
                        sil, rgb = render_compare_score(model, T, frames, observed_masks, scale=1.0)
                        cands.append((0.65 * sil + 0.35 * rgb, T))
    cands.sort(key=lambda x: x[0], reverse=True)
    out = [T for _, T in cands[:8]]
    if not out:
        T = np.eye(4)
        T[:3, 3] = center
        out = [T]
    return out


def register_model(model: CanonicalModel, model_pts: np.ndarray, obj_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray]) -> PoseEstimate:
    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(obj_pts)
    model_max = float((model_pts.max(0) - model_pts.min(0)).max())
    obj_center = obj_pts.mean(0)
    best: Optional[PoseEstimate] = None

    for scale in [0.94, 1.0, 1.06]:
        scaled = model_pts * scale
        src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(scaled)
        rs = model_max * scale
        max_corr = max(0.006, rs * 0.25)
        tgt_norm = o3d.geometry.PointCloud(tgt)
        tgt_norm.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr, max_nn=30))

        init_candidates = coarse_pose_search(model, scaled, obj_pts, frames, observed_masks)
        for Ti in init_candidates:
            # center correction to keep model near observed cluster
            Ti = Ti.copy()
            Ti[:3, 3] += (obj_center - Ti[:3, 3]) * 0.2
            src_i = o3d.geometry.PointCloud(src)
            src_i.transform(Ti)
            src_i.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_corr, max_nn=30))
            icp = o3d.pipelines.registration.registration_icp(
                src_i, tgt_norm, max_corr, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=120),
            )
            Tf = icp.transformation @ Ti
            aligned = transform_points(scaled, Tf)
            ds = mv_depth_score(aligned, frames, observed_masks)
            cs = cov_score(aligned, tgt, max(model.extents_m.max() * 0.08, 0.003))
            ss = silhouette_iou_score(aligned, frames, observed_masks)
            sil, rgb = render_compare_score(model, Tf, frames, observed_masks, scale=scale)
            pose_conf = combine_pose_scores(model.name, ds, cs, max(ss, sil), rgb)
            if best is None or pose_conf > best.confidence:
                U, _, Vt = np.linalg.svd(Tf[:3, :3])
                R_clean = U @ Vt
                if np.linalg.det(R_clean) < 0:
                    U[:, -1] *= -1
                    R_clean = U @ Vt
                Tf[:3, :3] = R_clean
                r = Rot.from_matrix(R_clean)
                best = PoseEstimate(
                    T_base_obj=Tf,
                    position_m=Tf[:3, 3].copy(),
                    quaternion_xyzw=r.as_quat(),
                    euler_xyz_deg=r.as_euler('xyz', degrees=True),
                    scale=scale,
                    confidence=pose_conf,
                    fitness=float(icp.fitness),
                    rmse=float(icp.inlier_rmse),
                    depth_score=ds,
                    coverage=cs,
                    silhouette_score=max(ss, sil),
                    render_rgb_score=rgb,
                )

    if best is None:
        raise RuntimeError("정합 실패")
    return best

# -----------------------------
# Export / visualization
# -----------------------------

def export_result(pose: PoseEstimate, model: CanonicalModel, frame_id: str, out_dir: Path, glb_src_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "frame_id": frame_id,
        "object_name": model.name,
        "label": OBJECT_LABELS.get(model.name, model.name),
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
        "render_rgb_score": pose.render_rgb_score,
    }
    jp = out_dir / f"pose_{model.name}_{frame_id}.json"
    np.savez(out_dir / f"pose_{model.name}_{frame_id}.npz", T_base_obj=pose.T_base_obj, position_m=pose.position_m, quaternion_xyzw=pose.quaternion_xyzw, scale=pose.scale)
    for coord in ["opencv", "isaac"]:
        scene = trimesh.load(str(glb_src_path))
        mesh = trimesh.util.concatenate(list(scene.geometry.values())) if isinstance(scene, trimesh.Scene) else scene.copy()
        v = (mesh.vertices - model.center) * pose.scale
        vp = transform_points(v, pose.T_base_obj)
        if coord == "isaac":
            vp = transform_points(vp, T_ISAAC_CV)
        mesh.vertices = vp
        suffix = "" if coord == "opencv" else "_isaac"
        gp = out_dir / f"{model.name}_posed_{frame_id}{suffix}.glb"
        mesh.export(str(gp), file_type="glb")
        result[f"posed_glb_{coord}"] = str(gp)
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def render_wireframe(model: CanonicalModel, pose: PoseEstimate, cam: CameraFrame, color=(0, 255, 0), thickness=2):
    img = cam.color_bgr.copy()
    h, w = img.shape[:2]
    K = cam.intrinsics.K
    v = (model.mesh.vertices - model.center) * pose.scale
    vb = transform_points(v, pose.T_base_obj)
    vc = transform_points(vb, np.linalg.inv(cam.T_base_cam))
    z = vc[:, 2]
    ok = z > 0.05
    pu = np.full(len(v), -1.0)
    pv = np.full(len(v), -1.0)
    pu[ok] = K[0, 0] * vc[ok, 0] / z[ok] + K[0, 2]
    pv[ok] = K[1, 1] * vc[ok, 1] / z[ok] + K[1, 2]
    for e0, e1 in model.mesh.edges_unique:
        if not (ok[e0] and ok[e1]):
            continue
        p0 = (int(pu[e0]), int(pv[e0]))
        p1 = (int(pu[e1]), int(pv[e1]))
        if abs(p0[0]) > 2 * w or abs(p0[1]) > 2 * h or abs(p1[0]) > 2 * w or abs(p1[1]) > 2 * h:
            continue
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


def save_combined_overlay(all_poses, all_models, all_masks, frames, frame_id, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    base_imgs = [cam.color_bgr.copy() for cam in frames]
    for obj_idx, (pose, model, masks) in enumerate(zip(all_poses, all_models, all_masks)):
        color = COLORS[obj_idx % len(COLORS)]
        for ci, cam in enumerate(frames):
            wire = render_wireframe(model, pose, cam, color)
            diff = cv2.absdiff(wire, cam.color_bgr)
            wire_mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 10
            base_imgs[ci][wire_mask] = wire[wire_mask]
            cnts, _ = cv2.findContours(masks[ci], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base_imgs[ci], cnts, -1, color, 1)
            if masks[ci].any():
                ys, xs = np.where(masks[ci] > 0)
                anchor = (int(xs.mean()), int(max(12, ys.min() - 5)))
                cv2.putText(base_imgs[ci], OBJECT_LABELS.get(model.name, model.name), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for ci in range(len(frames)):
        cv2.putText(base_imgs[ci], f"cam{ci}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    combined = np.hstack(base_imgs)
    cv2.imwrite(str(out_dir / f"overlay_{frame_id}.png"), combined)

# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(data_dir: str, intrinsics_dir: str, frame_id: str, glb_path: Optional[str] = None,
                 output_dir: str = "src/output/pose_pipeline_no_yolo", multi_object: bool = False,
                 capture_subdir: str = "object_capture", segmenter_name: str = "grabcut",
                 sam2_cfg: Optional[str] = None, sam2_ckpt: Optional[str] = None):
    data_dir = Path(data_dir)
    intrinsics_dir = Path(intrinsics_dir)
    out = Path(output_dir)

    print("=" * 68)
    print(f" Non-YOLO Pose Pipeline — Frame {frame_id} {'[MULTI]' if multi_object else '[SINGLE]'}")
    print("=" * 68)

    intrinsics, extrinsics = load_calibration(data_dir, intrinsics_dir)
    glb_paths: Dict[str, Path] = {}
    all_models: Dict[str, CanonicalModel] = {}
    if glb_path:
        gp = Path(glb_path)
        m = normalize_glb(gp)
        all_models[m.name] = m
        glb_paths[m.name] = gp
    else:
        for i in range(1, 5):
            p = data_dir / f"object_{i:03d}.glb"
            if p.exists():
                m = normalize_glb(p)
                all_models[m.name] = m
                glb_paths[m.name] = p

    frames = load_frame(data_dir, frame_id, intrinsics, extrinsics, capture_subdir=capture_subdir)

    plane_n, plane_d, table_center, table_radius = estimate_table_plane(frames)
    print(f"table center: [{table_center[0]:+.3f}, {table_center[1]:+.3f}, {table_center[2]:+.3f}] m")
    print(f"table radius: {table_radius:.3f} m")

    coarse_fg_masks = [build_coarse_foreground_mask(cam, plane_n, plane_d, table_center, table_radius) for cam in frames]

    if segmenter_name == "sam2":
        segmenter: BaseSegmenter = SAM2BoxSegmenter(sam2_cfg or "", sam2_ckpt or "")
    else:
        segmenter = GrabCutSegmenter()

    detections_by_cam: Dict[int, List[CameraDetection]] = {}
    for cam, coarse_fg in zip(frames, coarse_fg_masks):
        boxes = extract_coarse_boxes(coarse_fg)
        masks = segmenter.predict(cam, boxes)
        detections_by_cam[cam.cam_id] = build_camera_detections(cam, masks)
    save_camera_detections(detections_by_cam, frames, frame_id, out)

    for cam in frames:
        print(f"cam{cam.cam_id}: detections={len(detections_by_cam.get(cam.cam_id, []))}")

    tracks = associate_detections_across_views(frames, detections_by_cam)
    if not tracks:
        raise RuntimeError("cross-view association 결과가 비어 있습니다.")

    print(f"associated tracks: {len(tracks)}")
    for i, tr in enumerate(tracks):
        cams = ','.join([f'cam{cid}' for cid in sorted(tr.detections.keys())])
        area = sum(det.area_px for det in tr.detections.values())
        print(f"  #{i} {tr.track_id}: views={len(tr.detections)} [{cams}] area={area}")

    active_tracks = tracks[: max(1, len(all_models))] if multi_object else tracks[:1]
    used_glbs = set()
    all_results = []
    all_poses = []
    all_model_objs = []
    all_masks_list = []

    for ti, track in enumerate(active_tracks):
        print(f"\n[track {track.track_id}] processing")
        track_pts = fuse_track_points(frames, track)
        if len(track_pts) == 0:
            print("  [skip] fused points empty")
            continue
        track.fused_points_base = track_pts
        ext = track_pts.max(0) - track_pts.min(0)
        print(f"  fused pts={len(track_pts)} extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm")

        track_mean_hsv = aggregate_track_mean_hsv(track)
        if glb_path:
            shortlist = [(Path(glb_path).stem, 1.0)]
        elif not multi_object:
            model_name = FRAME_TO_GLB.get(int(frame_id))
            if model_name is None:
                raise RuntimeError(f"프레임 {frame_id}에 대한 GLB 매핑 없음")
            shortlist = [(model_name, 1.0)]
        else:
            candidate_models = {k: v for k, v in all_models.items() if k not in used_glbs}
            shortlist = shortlist_glb_candidates(track_pts, candidate_models, frames, track.observed_masks, track_mean_hsv, top_k=min(3, len(candidate_models)))

        if not shortlist:
            print("  [skip] shortlist empty")
            continue
        print("  shortlist:", ", ".join([f"{n}:{s:.3f}" for n, s in shortlist]))

        best_pose = None
        best_model_name = None
        best_score = -1.0
        for cand_name, coarse_score in shortlist:
            model = all_models[cand_name]
            model_pts = sample_model_points(model)
            try:
                pose = register_model(model, model_pts, track_pts, frames, track.observed_masks)
            except RuntimeError:
                continue
            final = 0.20 * coarse_score + 0.80 * pose.confidence
            print(f"    candidate={cand_name} coarse={coarse_score:.3f} conf={pose.confidence:.3f} fit={pose.fitness:.3f} sil={pose.silhouette_score:.3f} rgb={pose.render_rgb_score:.3f} final={final:.3f}")
            if final > best_score:
                best_score = final
                best_pose = pose
                best_model_name = cand_name
        if best_pose is None or best_model_name is None:
            print("  [skip] no valid pose")
            continue
        used_glbs.add(best_model_name)
        model = all_models[best_model_name]
        track.best_pose = best_pose
        track.best_model_name = best_model_name

        print(f"  final: {best_model_name} conf={best_pose.confidence:.3f} fitness={best_pose.fitness:.3f}")
        result = export_result(best_pose, model, frame_id, out, glb_paths[best_model_name])
        all_results.append(result)
        all_poses.append(best_pose)
        all_model_objs.append(model)
        all_masks_list.append(track.observed_masks)

    if all_poses:
        save_combined_overlay(all_poses, all_model_objs, all_masks_list, frames, frame_id, out)
        print(f"overlay: {out / f'overlay_{frame_id}.png'}")

    unmatched = set(all_models.keys()) - used_glbs
    if unmatched and multi_object:
        print(f"unmatched GLB: {', '.join(sorted(unmatched))}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Non-YOLO multi-view RGB-D GLB pose pipeline")
    parser.add_argument("--data_dir", default="src/data")
    parser.add_argument("--intrinsics_dir", default="src/intrinsics")
    parser.add_argument("--capture_subdir", default="object_capture")
    parser.add_argument("--frame_id", default=None)
    parser.add_argument("--glb", default=None)
    parser.add_argument("--output_dir", default="src/output/pose_pipeline_no_yolo")
    parser.add_argument("--segmenter", default="grabcut", choices=["grabcut", "sam2"])
    parser.add_argument("--sam2_cfg", default=None)
    parser.add_argument("--sam2_ckpt", default=None)
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()

    if args.batch:
        cam0_dir = Path(args.data_dir) / args.capture_subdir / "cam0"
        fids = sorted(f.stem.replace("rgb_", "") for f in cam0_dir.glob("rgb_*.jpg"))
        print(f"배치: {len(fids)} 프레임")
        all_r = []
        for fid in fids:
            try:
                r = run_pipeline(
                    data_dir=args.data_dir,
                    intrinsics_dir=args.intrinsics_dir,
                    frame_id=fid,
                    glb_path=args.glb,
                    output_dir=args.output_dir,
                    multi_object=args.multi,
                    capture_subdir=args.capture_subdir,
                    segmenter_name=args.segmenter,
                    sam2_cfg=args.sam2_cfg,
                    sam2_ckpt=args.sam2_ckpt,
                )
                all_r.extend(r)
            except Exception as e:
                print(f"  [ERROR] {fid}: {e}")
                all_r.append({"frame_id": fid, "error": str(e)})
        summary = Path(args.output_dir) / "batch_summary.json"
        with open(summary, "w", encoding="utf-8") as f:
            json.dump(all_r, f, indent=2, ensure_ascii=False)
        print(f"배치 완료: {summary}")
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
            sam2_cfg=args.sam2_cfg,
            sam2_ckpt=args.sam2_ckpt,
        )


if __name__ == "__main__":
    main()
