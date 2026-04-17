#!/usr/bin/env python3
"""
Cluster-first multi-view 6D pose estimator.

구조 전환:
  (legacy) 물체 이름 → color mask → point fusion → ICP
  (new)    table foreground → 3D DBSCAN → cluster별 CAD 매칭 → 공간 규칙 선택

핵심 prior는 유지: 테이블 평면, 테이블 위 foreground, vertical-axis constrained
pose. 색상 정보는 optional helper로만 남김.

출력:
  src/output/pose_per_object/frame_{frame}/
    cluster_{id:02d}__{model_name}__pose.json
    cluster_{id:02d}__{model_name}__posed.glb (+ _isaac)
    debug_masks.png
    summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (  # type: ignore
    CameraFrame,
    CanonicalModel,
    DEPTH_POLICY,
    OBJECT_LABELS,
    OBJECT_SYMMETRY,
    PoseEstimate,
    T_ISAAC_CV,
    backproject_depth,
    estimate_table_plane,
    load_calibration,
    load_frame,
    mv_depth_score,
    normalize_glb,
    sample_model_points,
    silhouette_iou_score,
    transform_points,
)
from scipy.spatial.transform import Rotation as Rot

import open3d as o3d

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
OUT_DIR = SCRIPT_DIR / "output" / "pose_per_object"


# ═══════════════════════════════════════════════════════════
# Cluster-first 파이프라인 설정
# ═══════════════════════════════════════════════════════════

CLUSTER_CFG: Dict[str, float] = {
    # 테이블 위 foreground 추출
    "height_min_m": 0.008,          # 테이블 표면 위 최소 높이 (8mm) - 테이블 먼지/에지 노이즈 컷
    "height_max_m": 0.22,           # 22cm 이상은 로봇 팔/사람 컷오프
    "radius_margin_m": 0.03,        # 테이블 반경 외부 여유
    # point cloud 정리
    "voxel_size_m": 0.003,
    "stat_outlier_nb": 20,
    "stat_outlier_std": 2.5,
    # DBSCAN
    "dbscan_eps_m": 0.012,
    "dbscan_min_pts": 30,
    # 클러스터 필터
    "min_cluster_pts": 120,         # 너무 작은 cluster는 노이즈
    "min_cluster_extent_m": 0.025,  # 2.5cm 이하는 노이즈
    "max_cluster_extent_m": 0.18,   # 18cm 이상은 robot arm/두 물체 merge
    "min_height_max_m": 0.015,      # 클러스터 vertical extent가 1.5cm 이상이어야 (floor-hugging 노이즈 컷)
    # 재투영 마스크
    "reproject_dilate_px": 5,
}


# Legacy color prior (optional). 새 파이프라인은 색상을 사용하지 않음.
# 사용자가 명시적으로 원할 때만 external config로 읽는 구조가 이상적.
COLOR_REF_HSV = {
    "object_001": (2.0, 215, 165),
    "object_002": (19.0, 240, 205),
    "object_003": (103.0, 120, 60),
    "object_004": (90.0, 120, 170),
}
COLOR_THRESHOLDS = {
    "object_001": {"hue": 10.0, "s_min": 120, "v_min": 80, "v_max": 240},
    "object_002": {"hue": 10.0, "s_min": 180, "v_min": 140, "v_max": 250},
    "object_003": {"hue": 14.0, "s_min": 65, "v_min": 25, "v_max": 110},
    "object_004": {"hue": 14.0, "s_min": 60, "v_min": 120, "v_max": 230},
}


# ═══════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════

@dataclass
class ClusterCandidate:
    """테이블 위 3D 인스턴스 후보."""
    cluster_id: int
    points_base: np.ndarray
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extents: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        if len(self.points_base) > 0:
            self.centroid = self.points_base.mean(axis=0).copy()
            self.bbox_min = self.points_base.min(axis=0).copy()
            self.bbox_max = self.points_base.max(axis=0).copy()
            self.extents = self.bbox_max - self.bbox_min


@dataclass
class ModelEntry:
    """CAD 후보 하나의 정적 메타데이터."""
    name: str
    glb_path: Path
    model: CanonicalModel
    symmetry: str            # "none" | "yaw"
    label: str


@dataclass
class PoseHypothesis:
    cluster_id: int
    entry: ModelEntry
    pose: PoseEstimate
    masks: List[np.ndarray]
    score: float


# ═══════════════════════════════════════════════════════════
# 기하 유틸 (기존 유지)
# ═══════════════════════════════════════════════════════════

def horizontal_axes(up_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    up = up_vec / (np.linalg.norm(up_vec) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(up, ref); a /= np.linalg.norm(a) + 1e-12
    b = np.cross(up, a)
    return a, b


def horizontal_max_extent(pts: np.ndarray, up_vec: np.ndarray) -> float:
    if len(pts) == 0:
        return 0.0
    a, b = horizontal_axes(up_vec)
    xs = pts @ a; ys = pts @ b
    return float(max(xs.max() - xs.min(), ys.max() - ys.min()))


def estimate_scale_auto(obj_pts: np.ndarray, model_pts: np.ndarray,
                        R_align: np.ndarray, up_world: np.ndarray) -> float:
    model_rot = (R_align @ model_pts.T).T
    obs_h = horizontal_max_extent(obj_pts, up_world)
    mod_h = horizontal_max_extent(model_rot, up_world)
    if mod_h < 1e-6:
        return 1.0
    return float(np.clip(obs_h / mod_h, 0.2, 2.5))


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis) + 1e-12
        return Rot.from_rotvec(axis * np.pi).as_matrix()
    v = np.cross(a, b); s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def reproject_to_vertical(R: np.ndarray, target_up: np.ndarray) -> np.ndarray:
    target_up = target_up / (np.linalg.norm(target_up) + 1e-12)
    dots = R.T @ target_up
    idx = int(np.argmax(np.abs(dots)))
    sign = 1.0 if dots[idx] > 0 else -1.0
    cur_up = R[:, idx] * sign
    R_fix = rotation_between(cur_up, target_up)
    return R_fix @ R


def snap_to_table(T: np.ndarray, scaled_model_pts: np.ndarray,
                  table_n: np.ndarray, table_d: float,
                  margin_m: float = 0.002) -> np.ndarray:
    aligned = (T[:3, :3] @ scaled_model_pts.T).T + T[:3, 3]
    heights = -(aligned @ table_n + table_d)
    min_h = float(heights.min())
    delta = margin_m - min_h
    if abs(delta) < 1e-6:
        return T
    T_out = T.copy()
    T_out[:3, 3] = T[:3, 3] + (-table_n) * delta
    return T_out


# ═══════════════════════════════════════════════════════════
# 색상 helper (optional, 기본 파이프라인에서 호출 안 함)
# ═══════════════════════════════════════════════════════════

def hsv_circular_distance(h_img: np.ndarray, h_ref: float) -> np.ndarray:
    d = np.abs(h_img - h_ref)
    return np.minimum(d, 180.0 - d)


def color_mask_for_object(bgr: np.ndarray, object_name: str) -> np.ndarray:
    """Legacy color prior 기반 mask. 새 파이프라인은 cluster mask를 사용."""
    if object_name not in COLOR_REF_HSV:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    ref_h = COLOR_REF_HSV[object_name][0]
    thr = COLOR_THRESHOLDS[object_name]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32); s = hsv[:, :, 1]; v = hsv[:, :, 2]
    hue_ok = hsv_circular_distance(h, float(ref_h)) <= thr["hue"]
    sv_ok = (s >= thr["s_min"]) & (v >= thr["v_min"]) & (v <= thr["v_max"])
    mask = (hue_ok & sv_ok).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    return mask


# ═══════════════════════════════════════════════════════════
# 새 cluster-first 파이프라인
# ═══════════════════════════════════════════════════════════

def extract_tabletop_foreground_points(
    frames: List[CameraFrame],
    table_info,
    cfg: Dict[str, float] = CLUSTER_CFG,
) -> np.ndarray:
    """모든 카메라 depth에서 테이블 위 foreground 점을 base 프레임으로 통합.

    색상/마스크 없이 오직 depth + 테이블 평면만 사용.
    """
    tn, td, tc, tr = table_info
    max_r = max(tr, 0.30) + cfg["radius_margin_m"]
    all_pts: List[np.ndarray] = []
    for cam in frames:
        pts_cam = backproject_depth(cam)
        if len(pts_cam) == 0:
            continue
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        heights = -(pts_base @ tn + td)
        on_top = (heights > cfg["height_min_m"]) & (heights < cfg["height_max_m"])
        horiz = np.linalg.norm(pts_base[:, [0, 2]] - tc[[0, 2]], axis=1)
        within = horiz < max_r
        keep = on_top & within
        if keep.any():
            all_pts.append(pts_base[keep])
    if not all_pts:
        return np.zeros((0, 3))
    return np.vstack(all_pts)


def cluster_tabletop_objects(
    pts_all: np.ndarray,
    cfg: Dict[str, float] = CLUSTER_CFG,
    table_n: Optional[np.ndarray] = None,
    table_d: Optional[float] = None,
) -> List[ClusterCandidate]:
    """Tabletop foreground를 3D DBSCAN으로 인스턴스 cluster로 분리.

    table_n, table_d 주어지면 각 cluster의 vertical extent도 계산해 필터링.
    """
    if len(pts_all) < cfg["min_cluster_pts"]:
        return []

    if len(pts_all) > 300_000:
        idx = np.random.choice(len(pts_all), 300_000, replace=False)
        pts_all = pts_all[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)
    pcd = pcd.voxel_down_sample(cfg["voxel_size_m"])
    if len(pcd.points) < cfg["min_cluster_pts"]:
        return []
    if len(pcd.points) > 80:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=int(cfg["stat_outlier_nb"]),
            std_ratio=float(cfg["stat_outlier_std"]),
        )
    pts = np.asarray(pcd.points)
    if len(pts) < cfg["min_cluster_pts"]:
        return []

    labels = np.array(
        pcd.cluster_dbscan(
            eps=float(cfg["dbscan_eps_m"]),
            min_points=int(cfg["dbscan_min_pts"]),
            print_progress=False,
        )
    )
    if labels.max() < 0:
        return []

    clusters: List[ClusterCandidate] = []
    min_ext = float(cfg["min_cluster_extent_m"])
    max_ext = float(cfg["max_cluster_extent_m"])
    min_pts = int(cfg["min_cluster_pts"])
    min_vext = float(cfg.get("min_height_max_m", 0.0))
    next_id = 0
    for lbl in np.unique(labels[labels >= 0]):
        sub = pts[labels == lbl]
        if len(sub) < min_pts:
            continue
        ext = float((sub.max(0) - sub.min(0)).max())
        if ext < min_ext or ext > max_ext:
            continue
        if table_n is not None and table_d is not None and min_vext > 0:
            heights = -(sub @ table_n + table_d)
            if float(heights.max() - heights.min()) < min_vext:
                continue
        clusters.append(ClusterCandidate(cluster_id=next_id, points_base=sub))
        next_id += 1

    clusters.sort(key=lambda c: c.centroid[0])
    for i, c in enumerate(clusters):
        c.cluster_id = i
    return clusters


def project_cluster_to_masks(
    cluster: ClusterCandidate,
    frames: List[CameraFrame],
    dilate_px: int = 5,
) -> List[np.ndarray]:
    """3D cluster → 각 카메라 2D mask 재투영."""
    masks: List[np.ndarray] = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate_px) | 1,) * 2)
    pts = cluster.points_base
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    for cam in frames:
        h, w = cam.intrinsics.height, cam.intrinsics.width
        pts_cam = (np.linalg.inv(cam.T_base_cam) @ pts_h.T)[:3].T
        z = pts_cam[:, 2]
        ok = z > 0.05
        if not ok.any():
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue
        K = cam.intrinsics.K
        u = np.round(K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]).astype(np.int32)
        v = np.round(K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]).astype(np.int32)
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        m = np.zeros((h, w), dtype=np.uint8)
        m[v[valid], u[valid]] = 255
        m = cv2.dilate(m, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        masks.append(m)
    return masks


# ═══════════════════════════════════════════════════════════
# Model DB & symmetry 자동 검출
# ═══════════════════════════════════════════════════════════

def detect_model_symmetry(model: CanonicalModel, thresh: float = 0.85) -> str:
    """GLB extents 비율로 yaw-symmetry 자동 추정.

    두 축이 비슷하고(짧은/긴 비율 ≥ thresh), 나머지가 다르면 yaw 대칭.
    명시적 OBJECT_SYMMETRY가 있으면 그 값을 우선 사용.
    """
    hinted = OBJECT_SYMMETRY.get(model.name)
    if hinted:
        return hinted
    ext = np.sort(np.asarray(model.extents_m))
    if ext[0] < 1e-6:
        return "none"
    short_long = ext[0] / ext[2]
    two_short = ext[0] / ext[1]
    if two_short >= thresh and short_long < thresh:
        return "yaw"
    return "none"


def build_model_db(
    data_dir: Path,
    only: Optional[List[str]] = None,
) -> List[ModelEntry]:
    """data_dir 내 *.glb를 스캔해 ModelEntry 리스트 생성."""
    if only:
        glb_paths = [data_dir / f"{n}.glb" for n in only]
        glb_paths = [p for p in glb_paths if p.exists()]
    else:
        glb_paths = sorted(data_dir.glob("*.glb"))
    entries: List[ModelEntry] = []
    for gp in glb_paths:
        model = normalize_glb(gp)
        symmetry = detect_model_symmetry(model)
        label = OBJECT_LABELS.get(model.name, model.name)
        entries.append(ModelEntry(
            name=model.name, glb_path=gp, model=model,
            symmetry=symmetry, label=label,
        ))
    return entries


# ═══════════════════════════════════════════════════════════
# Pose scoring & registration (기존 유지, symmetry 파라미터화)
# ═══════════════════════════════════════════════════════════

def coverage_score_two_sided(
    aligned: np.ndarray, obj_pts: np.ndarray, radius: float,
) -> Tuple[float, float]:
    if len(aligned) == 0 or len(obj_pts) == 0:
        return 0.0, 0.0
    from scipy.spatial import cKDTree
    tree_obs = cKDTree(obj_pts); tree_mod = cKDTree(aligned)
    sample_mod = aligned if len(aligned) <= 2000 else aligned[
        np.random.choice(len(aligned), 2000, replace=False)
    ]
    d1, _ = tree_obs.query(sample_mod, k=1, distance_upper_bound=radius)
    forward = float(np.sum(np.isfinite(d1))) / float(len(sample_mod))
    sample_obs = obj_pts if len(obj_pts) <= 2000 else obj_pts[
        np.random.choice(len(obj_pts), 2000, replace=False)
    ]
    d2, _ = tree_mod.query(sample_obs, k=1, distance_upper_bound=radius)
    backward = float(np.sum(np.isfinite(d2))) / float(len(sample_obs))
    return forward, backward


def _extent_match_score(aligned: np.ndarray, obj_pts: np.ndarray) -> float:
    if len(aligned) < 10 or len(obj_pts) < 10:
        return 0.0
    mod_ext = aligned.max(0) - aligned.min(0)
    obs_ext = obj_pts.max(0) - obj_pts.min(0)
    ratios = []
    for i in range(3):
        me, oe = float(mod_ext[i]), float(obs_ext[i])
        if me < 1e-6 or oe < 1e-6:
            continue
        ratios.append(min(me / oe, oe / me))
    return float(np.mean(ratios)) if ratios else 0.0


def _score_pose(
    T_final: np.ndarray, scaled: np.ndarray, obj_pts: np.ndarray,
    tgt: o3d.geometry.PointCloud, frames: List[CameraFrame],
    masks: List[np.ndarray], symmetry: str,
) -> Tuple[float, float, float, float, float, float, float]:
    aligned = transform_points(scaled, T_final)
    mod_max_world = float((aligned.max(0) - aligned.min(0)).max())
    try:
        eval_res = o3d.pipelines.registration.evaluate_registration(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned)),
            tgt, max(mod_max_world * 0.06, 0.005),
        )
        fitness = float(eval_res.fitness); rmse = float(eval_res.inlier_rmse)
    except Exception:
        fitness, rmse = 0.0, 0.0
    ds = mv_depth_score(aligned, frames, masks)
    radius = max(mod_max_world * 0.05, 0.004)
    cov_fwd, cov_bwd = coverage_score_two_sided(aligned, obj_pts, radius)
    ss = silhouette_iou_score(aligned, frames, masks)
    em = _extent_match_score(aligned, obj_pts)
    if symmetry == "yaw":
        conf = (0.12 * ds + 0.10 * cov_fwd + 0.12 * cov_bwd
                + 0.26 * ss + 0.22 * em + 0.18 * fitness)
    else:
        conf = (0.10 * ds + 0.10 * cov_fwd + 0.12 * cov_bwd
                + 0.28 * ss + 0.22 * em + 0.18 * fitness)
    return float(conf), fitness, rmse, ds, float(cov_fwd), float(cov_bwd), ss


def model_up_candidates(
    model_extents_m: np.ndarray, symmetry: str,
) -> List[np.ndarray]:
    if symmetry == "yaw":
        axis_idx = int(np.argmax(np.asarray(model_extents_m)))
        axis = np.zeros(3, dtype=np.float64); axis[axis_idx] = 1.0
        return [axis, -axis]
    out = []
    for idx in range(3):
        for sign in (1.0, -1.0):
            vec = np.zeros(3, dtype=np.float64); vec[idx] = sign
            out.append(vec)
    return out


def vertical_constrained_inits(
    table_n: np.ndarray, model_pts: np.ndarray, model_extents_m: np.ndarray,
    obj_pts: np.ndarray, yaw_steps: int = 18, symmetry: str = "none",
) -> List[Tuple[np.ndarray, float]]:
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    steps = 1 if symmetry == "yaw" else yaw_steps
    ups = model_up_candidates(model_extents_m, symmetry)
    out: List[Tuple[np.ndarray, float]] = []
    for model_up in ups:
        R_align = rotation_between(model_up, target_up)
        scale_auto = estimate_scale_auto(obj_pts, model_pts, R_align, target_up)
        for yaw_deg in np.linspace(0, 360, steps, endpoint=False):
            R_yaw = Rot.from_rotvec(target_up * np.radians(yaw_deg)).as_matrix()
            out.append((R_yaw @ R_align, scale_auto))
    return out


def _run_icp_stack(
    scaled: np.ndarray, T0: np.ndarray, tgt: o3d.geometry.PointCloud,
    mod_max_world: float,
) -> np.ndarray:
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(scaled)
    src.transform(T0)
    cur = np.eye(4)
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(1e-9, 1e-9, 80)
    for mc in (mod_max_world * 0.35, mod_max_world * 0.15,
               mod_max_world * 0.07, mod_max_world * 0.035):
        try:
            res = o3d.pipelines.registration.registration_icp(
                src, tgt, max(mc, 0.004), cur, p2p, crit,
            )
        except Exception:
            break
        cur = res.transformation
    try:
        if not tgt.has_normals():
            tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(mod_max_world * 0.1, 0.01), max_nn=30))
        p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, max(mod_max_world * 0.04, 0.005), cur, p2l, crit,
        )
        cur = res.transformation
    except Exception:
        pass
    return cur @ T0


def register_direct(
    symmetry: str,
    model: CanonicalModel,
    model_pts: np.ndarray,
    obj_pts: np.ndarray,
    frames: List[CameraFrame],
    masks: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    verbose: bool = True,
    fast: bool = True,
) -> Optional[PoseEstimate]:
    """Vertical-axis constrained pose estimation.

    symmetry는 model 기준 속성. 외부에서 주입.
    fast=True: cluster-first 용. yaw/scale sweep 축소(3×↓). 다수 cluster×model
    pair에 대한 합리적 해결책.
    """
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    obj_center = obj_pts.mean(axis=0)
    mod_ext_unit = model_pts.max(0) - model_pts.min(0)
    mod_max_unit = float(mod_ext_unit.max())

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(obj_pts)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))

    if fast:
        yaw_steps = 1 if symmetry == "yaw" else 12
        scale_factors = [0.75, 1.00, 1.35]
    else:
        yaw_steps = 1 if symmetry == "yaw" else 24
        scale_factors = [0.75, 0.88, 1.00, 1.15, 1.35]
    inits = vertical_constrained_inits(
        table_n, model_pts, np.asarray(model.extents_m),
        obj_pts, yaw_steps=yaw_steps, symmetry=symmetry,
    )
    if not inits:
        return None
    if verbose:
        print(f"      [init] scale_auto={inits[0][1]:.3f} rot_candidates={len(inits)} "
              f"scale_sweep={len(scale_factors)}")

    stats = {"iter": 0, "rejected_center": 0, "reprojected": 0, "rejected_extent": 0}
    candidates: List[Tuple[float, np.ndarray, float, np.ndarray]] = []
    TOP_K = 4

    def _eval_and_store(R_init: np.ndarray, scale: float):
        scaled = model_pts * scale
        src_center = scaled.mean(0)
        stats["iter"] += 1
        T0 = np.eye(4); T0[:3, :3] = R_init
        T0[:3, 3] = obj_center - R_init @ src_center
        mod_max_world = mod_max_unit * scale
        T_final = _run_icp_stack(scaled, T0, tgt, mod_max_world)
        R_final = T_final[:3, :3]
        cos_to_up = float(np.max(np.abs(R_final.T @ target_up)))
        if cos_to_up < 0.985:
            stats["reprojected"] += 1
            R_fix = reproject_to_vertical(R_final, target_up)
            T_final[:3, :3] = R_fix
            try:
                src2 = o3d.geometry.PointCloud()
                src2.points = o3d.utility.Vector3dVector(transform_points(scaled, T_final))
                res2 = o3d.pipelines.registration.registration_icp(
                    src2, tgt, max(mod_max_world * 0.05, 0.006), np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(1e-9, 1e-9, 40),
                )
                T_final = res2.transformation @ T_final
                R_final2 = T_final[:3, :3]
                if float(np.max(np.abs(R_final2.T @ target_up))) < 0.999:
                    T_final[:3, :3] = reproject_to_vertical(R_final2, target_up)
            except Exception:
                pass
        T_final = snap_to_table(T_final, scaled, table_n, table_d, margin_m=0.0005)
        fc = (T_final @ np.append(src_center, 1))[:3]
        if np.linalg.norm(fc - obj_center) > mod_max_world * 0.85:
            stats["rejected_center"] += 1
            return
        aligned_pre = transform_points(scaled, T_final)
        if _extent_match_score(aligned_pre, obj_pts) < 0.55:
            stats["rejected_extent"] += 1
            return
        conf, fitness, rmse, ds, cov_fwd, cov_bwd, ss = _score_pose(
            T_final, scaled, obj_pts, tgt, frames, masks, symmetry
        )
        candidates.append((conf, T_final, scale,
                           np.array([fitness, rmse, ds, cov_fwd, cov_bwd, ss])))

    for R_init, scale_auto in inits:
        for sf in scale_factors:
            _eval_and_store(R_init, scale_auto * sf)

    if not candidates:
        if verbose:
            print(f"      [reg] iters={stats['iter']} all rejected")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:TOP_K]
    if verbose:
        print(f"      [coarse] iters={stats['iter']} reproj={stats['reprojected']} "
              f"rej_ctr={stats['rejected_center']} rej_ext={stats['rejected_extent']} "
              f"top1_conf={top[0][0]:.3f}")

    fine_candidates: List[Tuple[float, np.ndarray, float, np.ndarray]] = []
    if fast:
        fine_yaws = [0.0] if symmetry == "yaw" else [-8.0, -4.0, 0.0, 4.0, 8.0]
        fine_sfs = [0.95, 1.00, 1.05]
    else:
        fine_yaws = [0.0] if symmetry == "yaw" else list(np.arange(-12.0, 12.01, 2.0))
        fine_sfs = [0.94, 0.97, 1.00, 1.03, 1.06, 1.10]
    for _, T0_fine, scale0, _ in top:
        R0 = T0_fine[:3, :3]
        for dyaw in fine_yaws:
            R_yaw = Rot.from_rotvec(target_up * np.radians(dyaw)).as_matrix()
            R_try = R_yaw @ R0
            for sf in fine_sfs:
                scale = scale0 * sf
                scaled = model_pts * scale
                src_center = scaled.mean(0)
                T0 = np.eye(4); T0[:3, :3] = R_try
                T0[:3, 3] = obj_center - R_try @ src_center
                mod_max_world = mod_max_unit * scale
                T_final = _run_icp_stack(scaled, T0, tgt, mod_max_world)
                R_final = T_final[:3, :3]
                cos_to_up = float(np.max(np.abs(R_final.T @ target_up)))
                if cos_to_up < 0.995:
                    T_final[:3, :3] = reproject_to_vertical(R_final, target_up)
                T_final = snap_to_table(T_final, scaled, table_n, table_d, 0.0005)
                fc = (T_final @ np.append(src_center, 1))[:3]
                if np.linalg.norm(fc - obj_center) > mod_max_world * 0.85:
                    continue
                aligned_pre = transform_points(scaled, T_final)
                if _extent_match_score(aligned_pre, obj_pts) < 0.55:
                    continue
                conf, fitness, rmse, ds, cov_fwd, cov_bwd, ss = _score_pose(
                    T_final, scaled, obj_pts, tgt, frames, masks, symmetry
                )
                fine_candidates.append((
                    conf, T_final, scale,
                    np.array([fitness, rmse, ds, cov_fwd, cov_bwd, ss])))

    pool = list(top) + fine_candidates
    pool.sort(key=lambda x: x[0], reverse=True)
    conf_best, T_best, scale_best, metrics = pool[0]
    fitness, rmse, ds, cov_fwd, cov_bwd, ss = metrics.tolist()

    Rf = T_best[:3, :3]
    U, _, Vt = np.linalg.svd(Rf)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1; R_clean = U @ Vt
    T_clean = T_best.copy(); T_clean[:3, :3] = R_clean
    rot = Rot.from_matrix(R_clean)
    best = PoseEstimate(
        T_base_obj=T_clean,
        position_m=T_clean[:3, 3].copy(),
        quaternion_xyzw=rot.as_quat(),
        euler_xyz_deg=rot.as_euler("xyz", degrees=True),
        scale=float(scale_best),
        confidence=float(conf_best),
        fitness=float(fitness), rmse=float(rmse),
        depth_score=float(ds),
        coverage=float(0.5 * (cov_fwd + cov_bwd)),
        silhouette_score=float(ss),
    )
    if verbose:
        print(f"      [fine] n={len(fine_candidates)} best_conf={best.confidence:.3f} "
              f"scale={best.scale:.3f} ss={best.silhouette_score:.3f}")
    return best


def refine_anisotropic_scale(
    T_base_obj: np.ndarray, scale_uniform: float,
    model_pts: np.ndarray, obj_pts: np.ndarray,
    symmetry: str = "none", coverage_ratio_min: float = 0.75,
) -> np.ndarray:
    R = T_base_obj[:3, :3]; t = T_base_obj[:3, 3]
    obs_aligned = (R.T @ (obj_pts - t).T).T
    mod_ext = model_pts.max(0) - model_pts.min(0)
    obs_ext = obs_aligned.max(0) - obs_aligned.min(0)
    posed_ext = mod_ext * scale_uniform
    coverage = obs_ext / (posed_ext + 1e-9)
    per_axis = np.full(3, scale_uniform, dtype=np.float64)

    if symmetry == "yaw":
        major = int(np.argmax(mod_ext))
        minors = [i for i in range(3) if i != major]
        w = np.array([max(coverage[minors[0]], 0.0), max(coverage[minors[1]], 0.0)])
        if w.sum() > 1e-6 and w.max() >= coverage_ratio_min:
            r_obs = (w[0] * obs_ext[minors[0]] / (mod_ext[minors[0]] + 1e-9)
                     + w[1] * obs_ext[minors[1]] / (mod_ext[minors[1]] + 1e-9)
                     ) / w.sum()
            per_axis[minors[0]] = float(r_obs); per_axis[minors[1]] = float(r_obs)
        if coverage[major] >= coverage_ratio_min:
            per_axis[major] = float(obs_ext[major] / (mod_ext[major] + 1e-9))
    else:
        for i in range(3):
            if coverage[i] >= coverage_ratio_min:
                per_axis[i] = float(obs_ext[i] / (mod_ext[i] + 1e-9))
    per_axis = np.clip(per_axis, scale_uniform * 0.7, scale_uniform * 1.3)
    return per_axis


# ═══════════════════════════════════════════════════════════
# Render-based refinement (기존 유지; symmetry 파라미터화)
# ═══════════════════════════════════════════════════════════

def render_posed_mesh_mask(
    mesh: "trimesh.Trimesh", T_base_obj: np.ndarray,
    scale_per_axis: np.ndarray, model_center: np.ndarray,
    cam: CameraFrame, downscale: int = 1,
) -> np.ndarray:
    h_full, w_full = cam.intrinsics.height, cam.intrinsics.width
    h, w = h_full // downscale, w_full // downscale
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    if len(V) == 0 or len(F) == 0:
        return np.zeros((h_full, w_full), dtype=np.uint8)
    V_obj = (V - model_center) * scale_per_axis
    Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
    V_base = (T_base_obj @ Vh.T)[:3].T
    T_cam_base = np.linalg.inv(cam.T_base_cam)
    V_cam = (T_cam_base @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
    z = V_cam[:, 2]; ok_v = z > 0.05
    K = cam.intrinsics.K
    u = (K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]) / downscale
    v = (K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]) / downscale
    mask = np.zeros((h, w), dtype=np.uint8)
    f_ok = ok_v[F[:, 0]] & ok_v[F[:, 1]] & ok_v[F[:, 2]]
    if f_ok.sum() == 0:
        return np.zeros((h_full, w_full), dtype=np.uint8)
    F_valid = F[f_ok]
    tri = np.stack([
        np.stack([u[F_valid[:, 0]], v[F_valid[:, 0]]], axis=-1),
        np.stack([u[F_valid[:, 1]], v[F_valid[:, 1]]], axis=-1),
        np.stack([u[F_valid[:, 2]], v[F_valid[:, 2]]], axis=-1),
    ], axis=1).astype(np.int32)
    lim = 4 * max(w, h)
    inside = np.abs(tri).reshape(len(tri), -1).max(axis=1) <= lim
    for face in tri[inside]:
        cv2.fillConvexPoly(mask, face, 255)
    if downscale > 1:
        mask = cv2.resize(mask, (w_full, h_full), interpolation=cv2.INTER_NEAREST)
    return mask


def render_and_compare_refine(
    pose: PoseEstimate, model: CanonicalModel, glb_src_path: Path,
    frames: List[CameraFrame], masks_obs: List[np.ndarray],
    table_n: np.ndarray, table_d: float, max_iter: int = 150,
    verbose: bool = True, pitch_roll_dof: bool = False,
) -> PoseEstimate:
    from scipy.optimize import minimize
    import time as _time

    scene = trimesh.load(str(glb_src_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene.copy())
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)

    obs_small = [m > 0 for m in masks_obs]
    obs_areas = [int(b.sum()) for b in obs_small]
    valid_cams = [ci for ci, a in enumerate(obs_areas) if a >= 50]
    if len(valid_cams) == 0:
        return pose
    if verbose:
        print(f"      [render-refine] faces={len(mesh.faces)} valid_cams={valid_cams}")

    R0 = pose.T_base_obj[:3, :3].copy()
    t0 = pose.T_base_obj[:3, 3].copy()
    s0 = float(pose.scale)
    h1_ax, h2_ax = horizontal_axes(target_up)

    def _apply(params):
        if pitch_roll_dof:
            dx, dy, dz, dyaw, dpitch, droll, dls = params
        else:
            dx, dy, dz, dyaw, dls = params
            dpitch = droll = 0.0
        scale = s0 * float(np.exp(dls))
        R_yaw = Rot.from_rotvec(target_up * dyaw).as_matrix()
        R = R_yaw @ R0
        if pitch_roll_dof and (abs(dpitch) > 0 or abs(droll) > 0):
            R_pitch = Rot.from_rotvec(h1_ax * dpitch).as_matrix()
            R_roll = Rot.from_rotvec(h2_ax * droll).as_matrix()
            R = R_roll @ R_pitch @ R
        t = t0 + np.array([dx, dy, dz])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T, scale

    def _render(T, scale, ci):
        cam = frames[ci]
        return render_posed_mesh_mask(
            mesh, T, np.full(3, scale), model.center, cam, downscale=1,
        ) > 0

    step = {"n": 0, "best": 1.0, "t0": _time.time()}

    def _loss(params):
        T, scale = _apply(params)
        iou_sum = 0.0
        for ci in valid_cams:
            rnd = _render(T, scale, ci); obs = obs_small[ci]
            inter = int(np.logical_and(obs, rnd).sum())
            union = int(np.logical_or(obs, rnd).sum())
            iou_sum += inter / union if union > 0 else 0.0
        L = 1.0 - iou_sum / len(valid_cams)
        step["n"] += 1
        if L < step["best"] - 1e-4:
            step["best"] = L
        return L

    n_dof = 7 if pitch_roll_dof else 5
    loss0 = _loss(np.zeros(n_dof))
    init_simplex = np.zeros((n_dof + 1, n_dof))
    init_simplex[1][0] = 0.030
    init_simplex[2][1] = 0.030
    init_simplex[3][2] = 0.030
    init_simplex[4][3] = np.radians(10.0)
    if pitch_roll_dof:
        init_simplex[5][4] = np.radians(6.0)
        init_simplex[6][5] = np.radians(6.0)
        init_simplex[7][6] = np.log(1.25)
    else:
        init_simplex[5][4] = np.log(1.25)
    try:
        res = minimize(
            _loss, np.zeros(n_dof), method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4,
                     "initial_simplex": init_simplex, "disp": False},
        )
    except Exception as exc:
        if verbose:
            print(f"      [render-refine] 실패 ({exc})")
        return pose
    if res.fun >= loss0 - 1e-4:
        if verbose:
            print(f"      [render-refine] no improvement {loss0:.3f}→{res.fun:.3f}")
        return pose
    T_new, scale_new = _apply(res.x)
    R_new = T_new[:3, :3]
    if float(np.max(np.abs(R_new.T @ target_up))) < 0.999:
        T_new[:3, :3] = reproject_to_vertical(R_new, target_up)
    model_pts = sample_model_points(model, n=2000)
    T_new = snap_to_table(T_new, model_pts * scale_new, table_n, table_d, 0.0005)
    rot = Rot.from_matrix(T_new[:3, :3])
    if verbose:
        print(f"      [render-refine] IoU {1-loss0:.3f}→{1-res.fun:.3f} iters={res.nit}")
    return PoseEstimate(
        T_base_obj=T_new, position_m=T_new[:3, 3].copy(),
        quaternion_xyzw=rot.as_quat(),
        euler_xyz_deg=rot.as_euler("xyz", degrees=True),
        scale=float(scale_new),
        confidence=float(1.0 - res.fun),
        fitness=pose.fitness, rmse=pose.rmse,
        depth_score=pose.depth_score, coverage=pose.coverage,
        silhouette_score=float(1.0 - res.fun),
    )


# ═══════════════════════════════════════════════════════════
# Export
# ═══════════════════════════════════════════════════════════

def export_posed_anisotropic(
    pose: PoseEstimate, per_axis_scale: np.ndarray,
    model: CanonicalModel, glb_src_path: Path, out_prefix: str,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = per_axis_scale.tolist()
    R = pose.T_base_obj[:3, :3]; t = pose.T_base_obj[:3, 3]
    paths: Dict[str, str] = {}
    for coord in ["opencv", "isaac"]:
        scene = trimesh.load(str(glb_src_path))
        mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
                if isinstance(scene, trimesh.Scene) else scene.copy())
        verts = (mesh.vertices - model.center) * np.array([sx, sy, sz])
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        T_out = np.eye(4); T_out[:3, :3] = R; T_out[:3, 3] = t
        verts_pose = (T_out @ verts_h.T)[:3].T
        if coord == "isaac":
            verts_pose = (T_ISAAC_CV @ np.hstack(
                [verts_pose, np.ones((len(verts_pose), 1))]
            ).T)[:3].T
        mesh.vertices = verts_pose
        suffix = "" if coord == "opencv" else "_isaac"
        gp = out_dir / f"{out_prefix}_posed{suffix}.glb"
        mesh.export(str(gp), file_type="glb")
        paths[f"posed_glb_{coord}"] = str(gp)
    return paths


# ═══════════════════════════════════════════════════════════
# Cluster-first core: 각 cluster에 model 후보 매칭
# ═══════════════════════════════════════════════════════════

def estimate_cluster_pose(
    cluster: ClusterCandidate,
    entry: ModelEntry,
    frames: List[CameraFrame],
    masks: List[np.ndarray],
    table_info,
    render_refine: bool = False,
    pitch_roll_dof: bool = False,
    verbose: bool = True,
) -> Optional[PoseEstimate]:
    """단일 cluster × 단일 model pose 추정."""
    table_n, table_d, _, _ = table_info
    obj_pts = cluster.points_base
    if len(obj_pts) < 40:
        return None

    model_pts = sample_model_points(entry.model, n=6000)
    try:
        pose = register_direct(
            entry.symmetry, entry.model, model_pts, obj_pts,
            frames, masks, table_n, table_d, verbose=verbose,
        )
    except Exception as exc:
        if verbose:
            print(f"      [FAIL] {entry.name}: 정합 실패 ({exc})")
        return None
    if pose is None:
        return None

    if render_refine:
        try:
            pose = render_and_compare_refine(
                pose, entry.model, entry.glb_path, frames, masks,
                table_n, table_d, max_iter=150, verbose=verbose,
                pitch_roll_dof=pitch_roll_dof,
            )
        except Exception as exc:
            if verbose:
                print(f"      [render-refine] 예외 ({exc}) → skip")

    return pose


def _extent_compatible(
    cluster: ClusterCandidate,
    entry: ModelEntry,
    under_ratio: float = 2.5,
    over_ratio: float = 1.4,
) -> bool:
    """Cluster max extent와 model max extent의 호환성 체크.

    부분 관측 (occlusion, self-occlusion) 때문에 cluster가 모델보다 작은 건 허용
    (under_ratio까지). cluster가 모델보다 크면 오매칭 가능성이 높으므로 타이트
    (over_ratio).
    """
    cluster_max = float(cluster.extents.max())
    model_max = float(np.asarray(entry.model.extents_m).max())
    if model_max < 1e-6:
        return False
    ratio = cluster_max / model_max
    return (1.0 / under_ratio) <= ratio <= over_ratio


def match_cluster_to_models(
    cluster: ClusterCandidate,
    frames: List[CameraFrame],
    model_db: List[ModelEntry],
    table_info,
    render_refine: bool = False,
    pitch_roll_dof: bool = False,
    verbose: bool = True,
) -> List[PoseHypothesis]:
    """cluster에 대해 모든 CAD를 시도하고 hypothesis 리스트 반환.

    Extent pre-filter로 명백히 크기가 안 맞는 model은 제외하여 ICP 수를 줄임.
    반환: score 내림차순. 동일 형상 물체 여러 개가 있을 때 unique-assignment
    (Hungarian) 단계에서 후순위 가설을 활용하기 위해 전부 반환.
    """
    masks = project_cluster_to_masks(
        cluster, frames, dilate_px=int(CLUSTER_CFG["reproject_dilate_px"])
    )

    if verbose:
        ext = cluster.extents
        print(f"\n  === cluster_{cluster.cluster_id:02d}  "
              f"pts={len(cluster.points_base)} "
              f"ctr=[{cluster.centroid[0]:+.3f},{cluster.centroid[1]:+.3f},"
              f"{cluster.centroid[2]:+.3f}] "
              f"ext=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm ===")

    hypotheses: List[PoseHypothesis] = []
    for entry in model_db:
        if not _extent_compatible(cluster, entry):
            if verbose:
                model_ext = np.asarray(entry.model.extents_m).max() * 100
                print(f"    skip {entry.name}: extent mismatch "
                      f"(cluster_max={cluster.extents.max()*100:.1f}cm vs "
                      f"model_max={model_ext:.1f}cm)")
            continue
        if verbose:
            print(f"    --- model {entry.name} ({entry.label}, "
                  f"sym={entry.symmetry}) ---")
        pose = estimate_cluster_pose(
            cluster, entry, frames, masks, table_info,
            render_refine=render_refine, pitch_roll_dof=pitch_roll_dof,
            verbose=verbose,
        )
        if pose is None:
            continue
        hyp = PoseHypothesis(
            cluster_id=cluster.cluster_id, entry=entry, pose=pose,
            masks=masks, score=float(pose.confidence),
        )
        if verbose:
            print(f"    [hyp] {entry.name}: conf={pose.confidence:.3f} "
                  f"silh={pose.silhouette_score:.3f} scale={pose.scale:.3f}")
        hypotheses.append(hyp)

    hypotheses.sort(key=lambda h: h.score, reverse=True)
    if verbose and hypotheses:
        print(f"  → cluster_{cluster.cluster_id:02d} top: "
              + ", ".join(f"{h.entry.name}={h.score:.3f}" for h in hypotheses[:3]))
    return hypotheses


def assign_clusters_to_models_unique(
    all_hypotheses: List[PoseHypothesis],
    conf_threshold: float = 0.30,
    verbose: bool = True,
) -> List[PoseHypothesis]:
    """Hungarian assignment: 각 model을 최대 1개 cluster에만 매칭.

    단일 인스턴스 inventory 가정 (현재 dataset). Multi-instance 장면에선
    unrestricted 모드(best hypothesis per cluster) 사용.
    """
    from scipy.optimize import linear_sum_assignment

    candidates = [h for h in all_hypotheses if h.pose.confidence >= conf_threshold]
    if not candidates:
        return []

    cluster_ids = sorted({h.cluster_id for h in candidates})
    model_names = sorted({h.entry.name for h in candidates})
    if not cluster_ids or not model_names:
        return []

    n_rows = len(cluster_ids)
    n_cols = len(model_names)
    # linear_sum_assignment 는 정방이 아니어도 동작 (rectangular).
    # 미매칭 허용을 위해 padding (매우 낮은 "no-assign" 비용)은 불필요.
    INF = 1e6
    cost = np.full((n_rows, n_cols), INF)
    lookup: Dict[Tuple[int, int], PoseHypothesis] = {}
    for h in candidates:
        r = cluster_ids.index(h.cluster_id)
        c = model_names.index(h.entry.name)
        # 매칭 cost = (1 - confidence) 로 설계하여 고신뢰 매칭을 선호
        cost[r, c] = 1.0 - h.pose.confidence
        lookup[(r, c)] = h

    row_ind, col_ind = linear_sum_assignment(cost)

    selected: List[PoseHypothesis] = []
    if verbose:
        print(f"\n  [assignment] Hungarian: {n_rows} clusters × {n_cols} models")
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= INF - 1:
            continue  # 실제 hypothesis가 없는 pair
        hyp = lookup[(r, c)]
        selected.append(hyp)
        if verbose:
            print(f"    cluster_{hyp.cluster_id:02d} ↔ {hyp.entry.name} "
                  f"conf={hyp.pose.confidence:.3f}")
    return selected


def select_best_per_cluster(
    all_hypotheses: List[PoseHypothesis],
    conf_threshold: float = 0.0,
) -> List[PoseHypothesis]:
    """Unrestricted: 각 cluster 최고 hypothesis만 선택 (동일 CAD 중복 허용)."""
    best_per_cluster: Dict[int, PoseHypothesis] = {}
    for h in all_hypotheses:
        if h.pose.confidence < conf_threshold:
            continue
        cur = best_per_cluster.get(h.cluster_id)
        if cur is None or h.score > cur.score:
            best_per_cluster[h.cluster_id] = h
    return list(best_per_cluster.values())


# ═══════════════════════════════════════════════════════════
# Debug & output
# ═══════════════════════════════════════════════════════════

def draw_cluster_debug_tiles(
    cluster: ClusterCandidate, entry: ModelEntry,
    frames: List[CameraFrame], masks: List[np.ndarray],
) -> List[np.ndarray]:
    tiles: List[np.ndarray] = []
    for ci, cam in enumerate(frames):
        vis = cam.color_bgr.copy()
        cnts, _ = cv2.findContours(masks[ci], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        label = (f"cluster_{cluster.cluster_id:02d}:{entry.name} cam{ci} "
                 f"area={int(np.count_nonzero(masks[ci]))}")
        cv2.putText(vis, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(vis)
    return tiles


# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════

def run(
    frame_id: str,
    only: Optional[List[str]] = None,
    render_refine: bool = False,
    pitch_roll_dof: bool = False,
    assignment_mode: str = "unique",
) -> List[dict]:
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    print("=" * 64)
    print(f" Cluster-first Pose Estimation — Frame {frame_id}")
    print("=" * 64)

    table_info = estimate_table_plane(frames)
    tn, td, tc, tr = table_info
    print(f"[plane] center=[{tc[0]:+.3f},{tc[1]:+.3f},{tc[2]:+.3f}] "
          f"radius={tr:.3f}")

    model_db = build_model_db(DATA_DIR, only=only)
    print(f"[model_db] {len(model_db)} CAD(s): "
          + ", ".join(f"{e.name}(sym={e.symmetry})" for e in model_db))
    if not model_db:
        print("[abort] no GLB files in data_dir")
        return []

    pts_all = extract_tabletop_foreground_points(frames, table_info, CLUSTER_CFG)
    print(f"[foreground] tabletop pts = {len(pts_all)}")
    clusters = cluster_tabletop_objects(pts_all, CLUSTER_CFG, table_n=tn, table_d=td)
    print(f"[clusters] {len(clusters)} instance candidate(s):")
    for c in clusters:
        print(f"  cluster_{c.cluster_id:02d}  pts={len(c.points_base):5d}  "
              f"ctr=[{c.centroid[0]:+.3f},{c.centroid[1]:+.3f},{c.centroid[2]:+.3f}]  "
              f"ext=[{c.extents[0]*100:.1f},{c.extents[1]*100:.1f},"
              f"{c.extents[2]*100:.1f}]cm")
    if not clusters:
        print("[abort] no valid clusters")
        return []

    frame_dir = OUT_DIR / f"frame_{frame_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # 1단계: 각 cluster에 대해 모든 model hypothesis 수집
    all_hyps: List[PoseHypothesis] = []
    cluster_by_id: Dict[int, ClusterCandidate] = {}
    for cluster in clusters:
        cluster_by_id[cluster.cluster_id] = cluster
        hyps = match_cluster_to_models(
            cluster, frames, model_db, table_info,
            render_refine=render_refine, pitch_roll_dof=pitch_roll_dof,
        )
        if not hyps:
            print(f"  [SKIP] cluster_{cluster.cluster_id:02d}: no valid model match")
            continue
        all_hyps.extend(hyps)

    # 2단계: assignment. "unique" = Hungarian (동일 CAD 중복 금지),
    # "unrestricted" = cluster별 최고 가설(동일 CAD 다중 허용)
    if assignment_mode == "unique":
        selected_hyps = assign_clusters_to_models_unique(all_hyps, conf_threshold=0.30)
    else:
        selected_hyps = select_best_per_cluster(all_hyps, conf_threshold=0.0)

    results: List[dict] = []
    tiles_per_cluster: List[List[np.ndarray]] = []
    for hyp in sorted(selected_hyps, key=lambda h: h.cluster_id):
        cluster = cluster_by_id[hyp.cluster_id]
        pose = hyp.pose
        entry = hyp.entry
        model_pts = sample_model_points(entry.model, n=6000)
        per_axis = refine_anisotropic_scale(
            pose.T_base_obj, pose.scale, model_pts, cluster.points_base,
            symmetry=entry.symmetry, coverage_ratio_min=0.75,
        )
        prefix = f"cluster_{cluster.cluster_id:02d}__{entry.name}"
        aniso_paths = export_posed_anisotropic(
            pose, per_axis, entry.model, entry.glb_path, prefix, frame_dir,
        )
        mod_ext_unit = model_pts.max(0) - model_pts.min(0)
        real_per_axis_m = per_axis * mod_ext_unit
        rot = Rot.from_matrix(pose.T_base_obj[:3, :3])
        result = {
            "frame_id": frame_id,
            "cluster_id": int(cluster.cluster_id),
            "model_name": entry.name,
            "label": entry.label,
            "symmetry": entry.symmetry,
            "coordinate_frame": "base (= cam0)",
            "unit": "meter",
            "position_m": pose.position_m.tolist(),
            "quaternion_xyzw": rot.as_quat().tolist(),
            "euler_xyz_deg": rot.as_euler("xyz", degrees=True).tolist(),
            "T_base_obj": pose.T_base_obj.tolist(),
            "rotation_matrix": pose.T_base_obj[:3, :3].tolist(),
            "scale": float(pose.scale),
            "anisotropic_scale_xyz": per_axis.tolist(),
            "real_size_m": {
                "x": float(real_per_axis_m[0]),
                "y": float(real_per_axis_m[1]),
                "z": float(real_per_axis_m[2]),
            },
            "confidence": float(pose.confidence),
            "fitness": float(pose.fitness),
            "rmse": float(pose.rmse),
            "depth_score": float(pose.depth_score),
            "coverage": float(pose.coverage),
            "silhouette_score": float(pose.silhouette_score),
            "cluster_centroid_m": cluster.centroid.tolist(),
            "cluster_extents_m": cluster.extents.tolist(),
            **aniso_paths,
        }
        json_path = frame_dir / f"{prefix}__pose.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"    [OK] {prefix} conf={pose.confidence:.3f} "
              f"silh={pose.silhouette_score:.3f} scale={pose.scale:.3f}")
        results.append(result)
        tiles_per_cluster.append(draw_cluster_debug_tiles(cluster, entry, frames, hyp.masks))

    if tiles_per_cluster:
        rows = [np.hstack(tiles) for tiles in tiles_per_cluster]
        grid = np.vstack(rows)
        cv2.imwrite(str(frame_dir / "debug_masks.png"), grid)

    summary_path = frame_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  summary: {summary_path}  "
          f"({len(results)}/{len(clusters)} clusters matched)")

    # Target selection: 각 model별 highest-conf cluster만 선택 + 저신뢰 컷오프.
    # compare_overlay.py가 `pose_{model}.json` 포맷을 읽으므로 backward-compat로도 저장.
    select_and_export_targets(results, frame_dir, frame_id)
    return results


def select_and_export_targets(
    results: List[dict],
    frame_dir: Path,
    frame_id: str,
    conf_threshold: float = 0.40,
) -> List[dict]:
    """conf>=threshold 가설 중 model별 최고점만 남기고 pose_{model}.json으로 저장.

    compare_overlay.py 호환 레이어.
    """
    confident = [r for r in results if r["confidence"] >= conf_threshold]
    best_by_model: Dict[str, dict] = {}
    for r in confident:
        m = r["model_name"]
        if m not in best_by_model or r["confidence"] > best_by_model[m]["confidence"]:
            best_by_model[m] = r

    if not best_by_model:
        print(f"  [targets] no hypothesis passes conf>={conf_threshold}")
        return []

    print(f"  [targets] {len(best_by_model)} object(s) selected (conf>={conf_threshold}):")
    for m, r in sorted(best_by_model.items()):
        compat = dict(r)
        compat["object_name"] = m
        # Legacy alias for compare_overlay.py 호환
        out_path = frame_dir / f"pose_{m}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(compat, f, indent=2, ensure_ascii=False)
        print(f"    pose_{m}.json ← cluster_{r['cluster_id']:02d} "
              f"conf={r['confidence']:.3f} silh={r['silhouette_score']:.3f}")
    return list(best_by_model.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only", default=None,
                    help="콤마로 구분된 model_name. GLB DB를 해당 이름만으로 제한.")
    ap.add_argument("--render_refine", action="store_true")
    ap.add_argument("--pitch_roll_dof", action="store_true")
    ap.add_argument("--assignment_mode", default="unique",
                    choices=["unique", "unrestricted"],
                    help="unique: 각 CAD 최대 1 cluster (Hungarian). "
                         "unrestricted: 동일 CAD 다중 cluster 허용.")
    args = ap.parse_args()

    only = args.only.split(",") if args.only else None
    kw = dict(render_refine=args.render_refine,
              pitch_roll_dof=args.pitch_roll_dof,
              assignment_mode=args.assignment_mode)
    if args.all:
        cam0_dir = DATA_DIR / "object_capture" / "cam0"
        fids = sorted(f.stem.replace("rgb_", "") for f in cam0_dir.glob("rgb_*.jpg"))
        for fid in fids:
            run(fid, only=only, **kw)
    else:
        run(args.frame_id, only=only, **kw)


if __name__ == "__main__":
    main()
