#!/usr/bin/env python3
"""
Shared data, geometry, loading, detection, association, and rendering utilities
for the CNOS + FoundationPose multi-view CAD pose pipeline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot


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
    mesh_low: trimesh.Trimesh
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
    label: Optional[str]
    centroid_uv: np.ndarray
    centroid_base: Optional[np.ndarray] = None
    extents_base: Optional[np.ndarray] = None


@dataclass
class TrackConsistency:
    pair_count: int = 0
    centroid_spread_m: float = 0.0
    mean_reprojection_error_px: float = 0.0
    mean_bbox_iou: float = 0.0
    mean_mask_iou: float = 0.0
    passed: bool = True


@dataclass
class ObjectTrack:
    track_id: str
    detections: Dict[int, CameraDetection]
    label: Optional[str] = None
    observed_masks: List[np.ndarray] = field(default_factory=list)
    fused_points_base: Optional[np.ndarray] = None
    consistency: TrackConsistency = field(default_factory=TrackConsistency)


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
    init_score: float = 0.0
    init_cam_id: Optional[int] = None


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


def coerce_matrix4x4(value: Any, name: str) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise RuntimeError(f"{name} shape invalid: expected (4,4), got {mat.shape}")
    return mat


def cam_pose_to_base(T_base_cam: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    return T_base_cam @ T_cam_obj


def load_glb_library(data_dir: Path, glb_path: Optional[str] = None) -> Tuple[Dict[str, Path], Dict[str, CanonicalModel]]:
    glb_paths: Dict[str, Path] = {}
    all_models: Dict[str, CanonicalModel] = {}
    if glb_path:
        gp = Path(glb_path)
        model = normalize_glb(gp)
        glb_paths[model.name] = gp
        all_models[model.name] = model
        return glb_paths, all_models

    for i in range(1, 5):
        gp = data_dir / f"object_{i:03d}.glb"
        if not gp.exists():
            continue
        model = normalize_glb(gp)
        glb_paths[model.name] = gp
        all_models[model.name] = model
    return glb_paths, all_models


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
    mesh_low = mesh
    try:
        if hasattr(mesh, "simplify_quadric_decimation") and len(mesh.faces) > 1200:
            mesh_low = mesh.simplify_quadric_decimation(1200)
    except Exception:
        mesh_low = mesh
    return CanonicalModel(
        name=glb_path.stem,
        mesh=mesh,
        mesh_low=mesh_low,
        center=mesh.centroid.copy(),
        extents_m=mesh.bounding_box.extents.copy(),
        is_watertight=bool(mesh.is_watertight),
    )


def sample_model_points(model: CanonicalModel, n: int = 12000) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(model.mesh, n)
    return (pts - model.center).astype(np.float64)


def make_virtual_camera(image_size: int = 224, focal_px: float = 220.0) -> CameraFrame:
    K = np.array([
        [focal_px, 0.0, image_size / 2.0],
        [0.0, focal_px, image_size / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    intr = CameraIntrinsics(
        K=K,
        D=np.zeros(5, dtype=np.float64),
        depth_scale=1.0,
        width=image_size,
        height=image_size,
    )
    return CameraFrame(
        cam_id=-1,
        intrinsics=intr,
        T_base_cam=np.eye(4, dtype=np.float64),
        color_bgr=np.zeros((image_size, image_size, 3), dtype=np.uint8),
        depth_u16=np.zeros((image_size, image_size), dtype=np.uint16),
    )


def create_camera_detection(
    cam: CameraFrame,
    mask: np.ndarray,
    det_id: str,
    score: float = 1.0,
    label: Optional[str] = None,
) -> Optional[CameraDetection]:
    area_px = int(np.count_nonzero(mask))
    if area_px == 0:
        return None
    bbox_xyxy = mask_bbox(mask)
    ys, xs = np.where(mask > 0)
    centroid_uv = np.array([xs.mean(), ys.mean()], dtype=np.float64)

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
        label=label,
        centroid_uv=centroid_uv,
        centroid_base=centroid_base,
        extents_base=extents_base,
    )


def build_camera_detections_from_predictions(
    cam: CameraFrame,
    preds: List[dict],
    det_prefix: str,
    min_score: float = 0.3,
    min_area: int = 120,
):
    detections: List[CameraDetection] = []
    det_idx = 0
    for pred in preds:
        mask = pred["mask"].astype(np.uint8)
        if mask.max() <= 1:
            mask = mask * 255
        score = float(pred.get("score", 1.0))
        if score < min_score or int(np.count_nonzero(mask)) < min_area:
            continue
        label = pred.get("label")
        label_tag = str(label) if label is not None else f"{det_prefix}_{det_idx}"
        det_id = f"cam{cam.cam_id}_{label_tag}_{det_idx}"
        det = create_camera_detection(cam, mask, det_id, score=score, label=str(label) if label is not None else None)
        if det is not None:
            detections.append(det)
            det_idx += 1
    detections.sort(key=lambda d: d.area_px, reverse=True)
    return detections


def build_camera_detections_from_cnos(cam: CameraFrame, cnos_segmenter: Any, min_score: float = 0.2, min_area: int = 120):
    return build_camera_detections_from_predictions(cam, cnos_segmenter.predict(cam.color_bgr), det_prefix="cnos", min_score=min_score, min_area=min_area)


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
            cv2.putText(vis, f"{det.label or f'd{di}'}:{det.score:.2f}", (x0, max(18, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.imwrite(str(out_dir / f"detections_cam{cam.cam_id}_{frame_id}.png"), vis)


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = np.asarray(box_a, dtype=np.float64).tolist()
    bx0, by0, bx1, by1 = np.asarray(box_b, dtype=np.float64).tolist()
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def binary_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a) > 0
    b = np.asarray(mask_b) > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def project_base_points_to_scatter_mask(points_base: np.ndarray, cam: CameraFrame, dilate_px: int = 5) -> np.ndarray:
    h, w = cam.intrinsics.height, cam.intrinsics.width
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points_base) == 0:
        return mask

    pts_cam = transform_points(points_base, np.linalg.inv(cam.T_base_cam))
    pts_cam = pts_cam[pts_cam[:, 2] > 0.05]
    if len(pts_cam) == 0:
        return mask

    K = cam.intrinsics.K
    u = np.round(K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2] + K[0, 2]).astype(np.int32)
    v = np.round(K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() == 0:
        return mask

    mask[v[ok], u[ok]] = 255
    ksize = max(3, int(dilate_px) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def reproject_detection_mask(det_src: CameraDetection, frame_by_id: Dict[int, CameraFrame], cam_dst_id: int) -> np.ndarray:
    cam_src = frame_by_id[det_src.cam_id]
    cam_dst = frame_by_id[cam_dst_id]
    pts_src = backproject_depth(cam_src, det_src.mask_u8)
    if len(pts_src) == 0:
        return np.zeros((cam_dst.intrinsics.height, cam_dst.intrinsics.width), dtype=np.uint8)
    return project_base_points_to_scatter_mask(transform_points(pts_src, cam_src.T_base_cam), cam_dst)


def reprojection_center_error(det_a: CameraDetection, det_b: CameraDetection, cam_b: CameraFrame) -> float:
    if det_a.centroid_base is None:
        return 1e9
    p_cam = (np.linalg.inv(cam_b.T_base_cam) @ np.append(det_a.centroid_base, 1.0))[:3]
    if p_cam[2] <= 0.05:
        return 1e9
    K = cam_b.intrinsics.K
    pred_uv = np.array([
        K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
        K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2],
    ], dtype=np.float64)
    return float(np.linalg.norm(pred_uv - det_b.centroid_uv))


def majority_label(track: ObjectTrack) -> Optional[str]:
    labels = [det.label for det in track.detections.values() if det.label]
    return None if not labels else Counter(labels).most_common(1)[0][0]


def centroid_distance_m(a: CameraDetection, b: CameraDetection) -> float:
    if a.centroid_base is None or b.centroid_base is None:
        return np.inf
    return float(np.linalg.norm(a.centroid_base - b.centroid_base))


def geometry_pair_score(
    a: CameraDetection,
    b: CameraDetection,
    frame_by_id: Dict[int, CameraFrame],
    centroid_thresh_m: float = 0.05,
    reproj_thresh_px: float = 25.0,
) -> float:
    dist = centroid_distance_m(a, b)
    reproj = min(
        reprojection_center_error(a, b, frame_by_id[b.cam_id]),
        reprojection_center_error(b, a, frame_by_id[a.cam_id]),
    )
    reproj_mask_ab = reproject_detection_mask(a, frame_by_id, b.cam_id)
    reproj_mask_ba = reproject_detection_mask(b, frame_by_id, a.cam_id)
    mask_iou_val = 0.5 * (
        binary_mask_iou(reproj_mask_ab, b.mask_u8) +
        binary_mask_iou(reproj_mask_ba, a.mask_u8)
    )
    bbox_iou_val = 0.5 * (
        bbox_iou(mask_bbox(reproj_mask_ab), b.bbox_xyxy) +
        bbox_iou(mask_bbox(reproj_mask_ba), a.bbox_xyxy)
    )
    pos_score = float(np.exp(-((dist / max(centroid_thresh_m, 1e-6)) ** 2))) if np.isfinite(dist) else 0.0
    reproj_score = float(np.exp(-((reproj / max(reproj_thresh_px, 1e-6)) ** 2))) if np.isfinite(reproj) else 0.0
    return 0.30 * pos_score + 0.20 * reproj_score + 0.20 * bbox_iou_val + 0.30 * mask_iou_val


def geometry_pair_is_consistent(
    a: CameraDetection,
    b: CameraDetection,
    frame_by_id: Dict[int, CameraFrame],
    centroid_thresh_m: float = 0.05,
    reproj_thresh_px: float = 25.0,
) -> bool:
    dist = centroid_distance_m(a, b)
    if not np.isfinite(dist) or dist > centroid_thresh_m:
        return False
    reproj = min(
        reprojection_center_error(a, b, frame_by_id[b.cam_id]),
        reprojection_center_error(b, a, frame_by_id[a.cam_id]),
    )
    if not (np.isfinite(reproj) and reproj <= reproj_thresh_px):
        return False
    reproj_mask_ab = reproject_detection_mask(a, frame_by_id, b.cam_id)
    reproj_mask_ba = reproject_detection_mask(b, frame_by_id, a.cam_id)
    mask_iou_val = 0.5 * (
        binary_mask_iou(reproj_mask_ab, b.mask_u8) +
        binary_mask_iou(reproj_mask_ba, a.mask_u8)
    )
    bbox_iou_val = 0.5 * (
        bbox_iou(mask_bbox(reproj_mask_ab), b.bbox_xyxy) +
        bbox_iou(mask_bbox(reproj_mask_ba), a.bbox_xyxy)
    )
    return bool(mask_iou_val >= 0.05 or bbox_iou_val >= 0.10)


def build_track_observed_masks(track: ObjectTrack, frames: List[CameraFrame]) -> List[np.ndarray]:
    masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cam in frames:
        det = track.detections.get(cam.cam_id)
        if det is None:
            masks.append(np.zeros((cam.intrinsics.height, cam.intrinsics.width), dtype=np.uint8))
            continue
        mask = cv2.morphologyEx(det.mask_u8, cv2.MORPH_CLOSE, kernel)
        masks.append(cv2.dilate(mask, kernel, iterations=1))
    return masks


def validate_track_consistency(
    track: ObjectTrack,
    frames: List[CameraFrame],
    centroid_thresh_m: float = 0.05,
    reproj_thresh_px: float = 25.0,
) -> TrackConsistency:
    dets = list(track.detections.values())
    if len(dets) < 2:
        return TrackConsistency(passed=True)

    frame_by_id = {cam.cam_id: cam for cam in frames}
    centroid_dists: List[float] = []
    reproj_errors: List[float] = []
    bbox_ious: List[float] = []
    mask_ious: List[float] = []
    pair_count = 0
    for idx_a in range(len(dets)):
        for idx_b in range(idx_a + 1, len(dets)):
            a = dets[idx_a]
            b = dets[idx_b]
            pair_count += 1
            dist = centroid_distance_m(a, b)
            if np.isfinite(dist):
                centroid_dists.append(dist)
            reproj = min(
                reprojection_center_error(a, b, frame_by_id[b.cam_id]),
                reprojection_center_error(b, a, frame_by_id[a.cam_id]),
            )
            if np.isfinite(reproj) and reproj < 1e8:
                reproj_errors.append(reproj)
            reproj_mask_ab = reproject_detection_mask(a, frame_by_id, b.cam_id)
            reproj_mask_ba = reproject_detection_mask(b, frame_by_id, a.cam_id)
            bbox_ious.append(0.5 * (
                bbox_iou(mask_bbox(reproj_mask_ab), b.bbox_xyxy) +
                bbox_iou(mask_bbox(reproj_mask_ba), a.bbox_xyxy)
            ))
            mask_ious.append(0.5 * (
                binary_mask_iou(reproj_mask_ab, b.mask_u8) +
                binary_mask_iou(reproj_mask_ba, a.mask_u8)
            ))

    if not centroid_dists or not reproj_errors:
        return TrackConsistency(
            pair_count=pair_count,
            centroid_spread_m=float(np.max(centroid_dists)) if centroid_dists else float("inf"),
            mean_reprojection_error_px=float(np.mean(reproj_errors)) if reproj_errors else float("inf"),
            mean_bbox_iou=float(np.mean(bbox_ious)) if bbox_ious else 0.0,
            mean_mask_iou=float(np.mean(mask_ious)) if mask_ious else 0.0,
            passed=False,
        )

    centroid_spread = float(np.max(centroid_dists))
    mean_reproj = float(np.mean(reproj_errors))
    mean_bbox = float(np.mean(bbox_ious)) if bbox_ious else 0.0
    mean_mask = float(np.mean(mask_ious)) if mask_ious else 0.0
    return TrackConsistency(
        pair_count=pair_count,
        centroid_spread_m=centroid_spread,
        mean_reprojection_error_px=mean_reproj,
        mean_bbox_iou=mean_bbox,
        mean_mask_iou=mean_mask,
        passed=bool(
            centroid_spread <= centroid_thresh_m and
            mean_reproj <= reproj_thresh_px and
            (mean_mask >= 0.05 or mean_bbox >= 0.10)
        ),
    )


def associate_by_label_and_geometry(
    frames: List[CameraFrame],
    detections_by_cam: Dict[int, List[CameraDetection]],
    centroid_thresh_m: float = 0.05,
    reproj_thresh_px: float = 25.0,
) -> List[ObjectTrack]:
    frame_by_id = {cam.cam_id: cam for cam in frames}
    tracks: List[ObjectTrack] = []
    used = set()
    track_idx = 0

    labels = sorted({
        det.label
        for dets in detections_by_cam.values()
        for det in dets
        if det.label
    })

    for label in labels:
        candidates = [
            det
            for cam_id in sorted(detections_by_cam)
            for det in detections_by_cam[cam_id]
            if det.label == label
        ]
        candidates.sort(key=lambda d: (d.score, d.area_px), reverse=True)
        for seed in candidates:
            if seed.det_id in used:
                continue
            members = {seed.cam_id: seed}
            used.add(seed.det_id)

            for cam in frames:
                if cam.cam_id in members:
                    continue
                best_det = None
                best_score = -1.0
                for cand in detections_by_cam.get(cam.cam_id, []):
                    if cand.det_id in used or cand.label != label:
                        continue
                    if not all(
                        geometry_pair_is_consistent(cand, member, frame_by_id, centroid_thresh_m=centroid_thresh_m, reproj_thresh_px=reproj_thresh_px)
                        for member in members.values()
                    ):
                        continue
                    score = float(np.mean([
                        geometry_pair_score(cand, member, frame_by_id, centroid_thresh_m=centroid_thresh_m, reproj_thresh_px=reproj_thresh_px)
                        for member in members.values()
                    ]))
                    if score > best_score:
                        best_det = cand
                        best_score = score
                if best_det is not None:
                    members[cam.cam_id] = best_det
                    used.add(best_det.det_id)

            track = ObjectTrack(track_id=f"track_{track_idx:02d}", detections=members, label=label)
            track.observed_masks = build_track_observed_masks(track, frames)
            track.consistency = validate_track_consistency(track, frames, centroid_thresh_m=centroid_thresh_m, reproj_thresh_px=reproj_thresh_px)
            tracks.append(track)
            track_idx += 1

    for cam_id in sorted(detections_by_cam):
        for det in detections_by_cam[cam_id]:
            if det.det_id in used:
                continue
            track = ObjectTrack(track_id=f"track_{track_idx:02d}", detections={cam_id: det}, label=det.label)
            track.observed_masks = build_track_observed_masks(track, frames)
            track.consistency = validate_track_consistency(track, frames)
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
    return voxel_downsample(np.vstack(all_pts), DEPTH_POLICY["voxel_size_m"])


def extract_mesh_face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        face_colors = np.asarray(mesh.visual.face_colors)
        if len(face_colors) == len(mesh.faces):
            return face_colors[:, :3][:, ::-1].astype(np.uint8)
    except Exception:
        pass
    try:
        vertex_colors = np.asarray(mesh.visual.vertex_colors)
        if len(vertex_colors) == len(mesh.vertices):
            return vertex_colors[np.asarray(mesh.faces)].mean(axis=1)[:, :3][:, ::-1].astype(np.uint8)
    except Exception:
        pass
    return np.full((len(mesh.faces), 3), 180, dtype=np.uint8)


def rasterize_mesh(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    image_size: Tuple[int, int],
    face_colors_bgr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_size
    rendered_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    z_buffer = np.full((h, w), np.inf, dtype=np.float32)

    z = verts_cam[:, 2]
    valid = z > 0.05
    if valid.sum() < 3:
        return rendered_bgr, mask, z_buffer

    uv = np.zeros((len(verts_cam), 2), dtype=np.float64)
    uv[valid, 0] = K[0, 0] * verts_cam[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * verts_cam[valid, 1] / z[valid] + K[1, 2]

    faces = np.asarray(faces, dtype=np.int32)
    valid_face_mask = np.all(valid[faces], axis=1)
    valid_faces = faces[valid_face_mask]
    if len(valid_faces) == 0:
        return rendered_bgr, mask, z_buffer
    face_colors = face_colors_bgr[valid_face_mask] if face_colors_bgr is not None else None

    for face_idx, tri in enumerate(valid_faces):
        pts = uv[tri]
        tri_z = z[tri]
        min_x = max(0, int(np.floor(pts[:, 0].min())))
        min_y = max(0, int(np.floor(pts[:, 1].min())))
        max_x = min(w - 1, int(np.ceil(pts[:, 0].max())))
        max_y = min(h - 1, int(np.ceil(pts[:, 1].max())))
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = pts[0]
        x1, y1 = pts[1]
        x2, y2 = pts[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue

        xs, ys = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5,
            np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5,
        )
        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue

        tri_depth = w0 * tri_z[0] + w1 * tri_z[1] + w2 * tri_z[2]
        region_depth = z_buffer[min_y:max_y + 1, min_x:max_x + 1]
        update = inside & (tri_depth < region_depth)
        if not update.any():
            continue

        region_depth[update] = tri_depth[update]
        z_buffer[min_y:max_y + 1, min_x:max_x + 1] = region_depth
        region_mask = mask[min_y:max_y + 1, min_x:max_x + 1]
        region_mask[update] = 255
        mask[min_y:max_y + 1, min_x:max_x + 1] = region_mask
        if face_colors is not None:
            region_rgb = rendered_bgr[min_y:max_y + 1, min_x:max_x + 1]
            region_rgb[update] = face_colors[face_idx]
            rendered_bgr[min_y:max_y + 1, min_x:max_x + 1] = region_rgb

    return rendered_bgr, mask, z_buffer


def render_model_to_image(
    model: CanonicalModel,
    T_base_obj: np.ndarray,
    cam: CameraFrame,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = model.mesh_low if model.mesh_low is not None else model.mesh
    verts = (mesh.vertices - model.center) * scale
    verts_base = transform_points(verts, T_base_obj)
    verts_cam = transform_points(verts_base, np.linalg.inv(cam.T_base_cam))
    return rasterize_mesh(
        verts_cam=verts_cam,
        faces=np.asarray(mesh.faces),
        K=cam.intrinsics.K,
        image_size=(cam.intrinsics.height, cam.intrinsics.width),
        face_colors_bgr=extract_mesh_face_colors(mesh),
    )


def render_model_to_mask(model: CanonicalModel, T_base_obj: np.ndarray, cam: CameraFrame, scale: float = 1.0) -> np.ndarray:
    _, mask, _ = render_model_to_image(model, T_base_obj, cam, scale=scale)
    return mask


def render_compare_score(model: CanonicalModel, T_base_obj: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray], scale: float = 1.0) -> float:
    vals = []
    for cam, obs in zip(frames, observed_masks):
        pred = render_model_to_mask(model, T_base_obj, cam, scale=scale) > 0
        obs = obs > 0
        union = np.logical_or(pred, obs).sum()
        if union == 0:
            continue
        vals.append(np.logical_and(pred, obs).sum() / union)
    return float(np.mean(vals)) if vals else 0.0


def best_fit_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ca = A.mean(axis=0)
    cb = B.mean(axis=0)
    H = (A - ca).T @ (B - cb)
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
        R, t = best_fit_transform(src_tf[valid], tgt_pts[idx[valid]])
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
    p = p[p[:, 2] > 0.05]
    if len(p) == 0:
        return mask
    K = cam.intrinsics.K
    u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() < 3:
        return mask
    cv2.fillConvexPoly(mask, cv2.convexHull(np.stack([u[ok], v[ok]], axis=1)), 255)
    return mask


def mv_depth_score(aligned_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray], tol: float = 0.015) -> float:
    total_ok = 0
    total_valid = 0
    for cam, mask in zip(frames, observed_masks):
        p = transform_points(aligned_pts, np.linalg.inv(cam.T_base_cam))
        p = p[p[:, 2] > 0.05]
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
    d1, _ = cKDTree(obs_pts).query(aligned_pts, k=1)
    d2, _ = cKDTree(aligned_pts).query(obs_pts, k=1)
    return 0.5 * (float((d1 < radius).mean()) + float((d2 < radius).mean()))


def silhouette_iou_score(aligned_pts: np.ndarray, frames: List[CameraFrame], observed_masks: List[np.ndarray]) -> float:
    vals = []
    for cam, obs in zip(frames, observed_masks):
        pred = project_points_to_mask(aligned_pts, cam) > 0
        obs = obs > 0
        union = np.logical_or(pred, obs).sum()
        if union == 0:
            continue
        vals.append(np.logical_and(pred, obs).sum() / union)
    return float(np.mean(vals)) if vals else 0.0


def combine_pose_scores(model_name: str, depth_score: float, coverage: float, silhouette: float) -> float:
    if OBJECT_SYMMETRY.get(model_name, "none") == "yaw":
        return 0.50 * depth_score + 0.20 * coverage + 0.30 * silhouette
    return 0.45 * depth_score + 0.35 * coverage + 0.20 * silhouette


def perturb_pose(T_base_obj: np.ndarray, translation_m: Tuple[float, float, float] = (0.0, 0.0, 0.0), euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    T = T_base_obj.copy()
    T[:3, :3] = Rot.from_euler("xyz", euler_deg, degrees=True).as_matrix() @ T[:3, :3]
    T[:3, 3] = T[:3, 3] + np.asarray(translation_m, dtype=np.float64)
    return T


def local_pose_candidates(T_init_base_obj: np.ndarray, model_name: str) -> List[np.ndarray]:
    translations = [
        (0.0, 0.0, 0.0),
        (0.006, 0.0, 0.0), (-0.006, 0.0, 0.0),
        (0.0, 0.006, 0.0), (0.0, -0.006, 0.0),
        (0.0, 0.0, 0.004), (0.0, 0.0, -0.004),
    ]
    rotations = [
        (0.0, 0.0, 0.0),
        (6.0, 0.0, 0.0), (-6.0, 0.0, 0.0),
        (0.0, 6.0, 0.0), (0.0, -6.0, 0.0),
        (0.0, 0.0, 8.0), (0.0, 0.0, -8.0),
    ]
    if OBJECT_SYMMETRY.get(model_name, "none") == "yaw":
        rotations.extend([(0.0, 0.0, 18.0), (0.0, 0.0, -18.0), (0.0, 0.0, 30.0), (0.0, 0.0, -30.0)])

    candidates: List[np.ndarray] = []
    seen = set()
    for t in translations:
        for r in rotations:
            key = tuple(np.round(np.asarray(t + r, dtype=np.float64), 6).tolist())
            if key in seen:
                continue
            seen.add(key)
            candidates.append(perturb_pose(T_init_base_obj, translation_m=t, euler_deg=r))
    return candidates


__all__ = [
    "COLORS",
    "DEPTH_POLICY",
    "OBJECT_LABELS",
    "OBJECT_SYMMETRY",
    "T_ISAAC_CV",
    "CameraDetection",
    "CameraFrame",
    "CameraIntrinsics",
    "CanonicalModel",
    "ObjectTrack",
    "PoseEstimate",
    "TrackConsistency",
    "associate_by_label_and_geometry",
    "backproject_depth",
    "bbox_iou",
    "binary_mask_iou",
    "build_camera_detections_from_cnos",
    "build_camera_detections_from_predictions",
    "build_track_observed_masks",
    "cam_pose_to_base",
    "centroid_distance_m",
    "coerce_matrix4x4",
    "create_camera_detection",
    "extract_mesh_face_colors",
    "fuse_track_points",
    "geometry_pair_is_consistent",
    "geometry_pair_score",
    "icp_point_to_point",
    "load_calibration",
    "load_frame",
    "load_glb_library",
    "local_pose_candidates",
    "majority_label",
    "make_virtual_camera",
    "mask_bbox",
    "mv_depth_score",
    "normalize_glb",
    "perturb_pose",
    "project_points_to_mask",
    "project_base_points_to_scatter_mask",
    "rasterize_mesh",
    "render_compare_score",
    "render_model_to_image",
    "render_model_to_mask",
    "reproject_detection_mask",
    "reprojection_center_error",
    "sample_model_points",
    "save_camera_detections",
    "silhouette_iou_score",
    "transform_points",
    "validate_track_consistency",
    "voxel_downsample",
    "best_fit_transform",
    "combine_pose_scores",
    "coverage_score",
]
