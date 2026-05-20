#!/usr/bin/env python3
"""
Improved Multi-cam InstantMesh Scale Pipeline
===========================================

목적
----
1. 멀티 RGB-D 카메라 + SAM/SAM2 mask + calibration으로 metric object point cloud 생성
2. InstantMesh 결과 mesh의 scale을 robust하게 metric scale로 정합
3. FoundationPose 입력에 바로 사용할 수 있는 scaled mesh(GLB)와 scale report 저장

핵심 개선점
-----------
- SAM mask erosion으로 boundary depth contamination 감소
- view별 object cloud 생성 및 view별 scale 후보 계산
- bbox scale 단독 사용 금지: bbox / robust pairwise distance / nearest-neighbor Chamfer 후보 비교
- MAD 기반 view scale outlier rejection
- optional Sim(3)-style ICP refinement: scale, rotation, translation을 반복 정합하되 최종적으로 scale만 신뢰
- scale 품질 리포트 JSON 저장

입력 폴더 예시
-------------
data_dir/
  cam0_rgb.png
  cam0_depth.png
  cam0_K.txt
  cam0_T_cam_to_world.txt
  cam1_rgb.png
  cam1_depth.png
  cam1_K.txt
  cam1_T_cam_to_world.txt

mask_dir/
  cam0_mask.png
  cam1_mask.png
  ...
또는 다중 물체:
  cam0_obj1_mask.png
  cam1_obj1_mask.png
  cam0_obj2_mask.png
  cam1_obj2_mask.png

실행 예시
--------
python improved_instantmesh_pose.py \
  --data_dir ./capture \
  --mask_dir ./masks \
  --instantmesh_mesh ./instantmesh_results/object.glb \
  --out_dir ./outputs \
  --depth_scale 0.001 \
  --mask_erode_px 5 \
  --scale_method auto \
  --refine_sim3_icp

출력
----
outputs/
  obj0_input.png
  obj0_cloud_clean.ply
  obj0_scaled.glb
  obj0_scale_report.json
  objects_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.neighbors import LocalOutlierFactor


# ============================================================
# Data structure
# ============================================================

@dataclass
class CameraPacket:
    cam_id: str
    rgb: np.ndarray             # H x W x 3, uint8, RGB order
    depth: np.ndarray           # H x W, float32, meter
    K: np.ndarray               # 3 x 3
    T_cam_to_world: np.ndarray  # 4 x 4, camera frame -> world/base frame


@dataclass
class ViewCloud:
    cam_id: str
    points: np.ndarray
    colors: Optional[np.ndarray]
    raw_count: int
    clean_count: int


@dataclass
class ScaleCandidate:
    name: str
    scale: float
    score: float
    details: dict


# ============================================================
# I/O utilities
# ============================================================

def load_matrix(path: str | Path, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"Matrix shape mismatch: {path}, expected={shape}, got={arr.shape}")
    return arr


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path: str | Path, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    depth = depth_raw.astype(np.float32) * depth_scale
    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_cameras_from_folder(data_dir: str | Path, depth_scale: float = 0.001) -> Dict[str, CameraPacket]:
    data_dir = Path(data_dir)
    cameras: Dict[str, CameraPacket] = {}

    rgb_files = sorted(data_dir.glob("cam*_rgb.*"))
    if not rgb_files:
        raise FileNotFoundError(f"No cam*_rgb.* files found in {data_dir}")

    for rgb_path in rgb_files:
        cam_id = rgb_path.stem.replace("_rgb", "")
        depth_candidates = sorted(data_dir.glob(f"{cam_id}_depth.*"))
        if not depth_candidates:
            raise FileNotFoundError(f"Depth file missing for {cam_id}")

        K_path = data_dir / f"{cam_id}_K.txt"
        T_path = data_dir / f"{cam_id}_T_cam_to_world.txt"
        if not K_path.exists():
            raise FileNotFoundError(f"K file missing: {K_path}")
        if not T_path.exists():
            raise FileNotFoundError(f"T file missing: {T_path}")

        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_candidates[0], depth_scale=depth_scale)
        K = load_matrix(K_path, (3, 3))
        T = load_matrix(T_path, (4, 4))

        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(f"RGB/depth size mismatch in {cam_id}: rgb={rgb.shape}, depth={depth.shape}")

        cameras[cam_id] = CameraPacket(cam_id=cam_id, rgb=rgb, depth=depth, K=K, T_cam_to_world=T)

    return cameras


def load_mesh_any(path: str | Path) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(str(path), force="scene")
    if isinstance(mesh_or_scene, trimesh.Scene):
        geoms = [g for g in mesh_or_scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise TypeError(f"No Trimesh geometry in scene: {path}")
        mesh = trimesh.util.concatenate(tuple(geoms))
    elif isinstance(mesh_or_scene, trimesh.Trimesh):
        mesh = mesh_or_scene
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh_or_scene)}")

    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise RuntimeError(f"Mesh has no vertices: {path}")
    return mesh


# ============================================================
# Mask utilities
# ============================================================

def erode_mask(mask: np.ndarray, erode_px: int = 5, min_pixels_after: int = 200) -> np.ndarray:
    """SAM boundary에 섞인 배경 depth를 줄이기 위한 erosion."""
    mask = mask.astype(bool)
    if erode_px <= 0:
        return mask
    kernel = np.ones((erode_px, erode_px), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if eroded.sum() < min_pixels_after:
        print(f"[WARN] erosion made mask too small ({eroded.sum()} px). Use original mask instead.")
        return mask
    return eroded


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return labels == largest


def load_masks(mask_dir: str | Path, cameras: Dict[str, CameraPacket]) -> Dict[str, np.ndarray]:
    mask_dir = Path(mask_dir)
    masks: Dict[str, np.ndarray] = {}
    for cam_id, cam in cameras.items():
        mask_path = mask_dir / f"{cam_id}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if mask.shape != cam.depth.shape:
            raise ValueError(f"Mask/depth size mismatch in {cam_id}: mask={mask.shape}, depth={cam.depth.shape}")
        masks[cam_id] = mask > 0
    return masks


def load_masks_per_object(
    mask_dir: str | Path,
    cameras: Dict[str, CameraPacket],
    obj_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    mask_dir = Path(mask_dir)
    masks_by_obj: Dict[str, Dict[str, np.ndarray]] = {}

    pat = re.compile(r"^(?P<cam>cam[^_]+)_obj(?P<obj>[^_]+)_mask\.[A-Za-z0-9]+$")
    multi_files = []
    for fp in mask_dir.iterdir():
        if not fp.is_file():
            continue
        m = pat.match(fp.name)
        if m:
            multi_files.append((m.group("cam"), m.group("obj"), fp))

    if multi_files:
        for cam_id, obj_id, fp in multi_files:
            if cam_id not in cameras:
                print(f"[WARN] mask file references unknown cam_id={cam_id}: {fp.name}")
                continue
            if obj_ids is not None and obj_id not in obj_ids:
                continue
            mk = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if mk is None:
                raise FileNotFoundError(f"Mask not readable: {fp}")
            if mk.shape != cameras[cam_id].depth.shape:
                raise ValueError(f"Mask/depth size mismatch in {cam_id}/obj{obj_id}: {mk.shape} vs {cameras[cam_id].depth.shape}")
            masks_by_obj.setdefault(obj_id, {})[cam_id] = mk > 0
    else:
        if obj_ids is not None and "0" not in obj_ids:
            raise FileNotFoundError("No cam*_obj*_mask.* files and fallback obj_id='0' was not requested.")
        masks_by_obj["0"] = load_masks(mask_dir, cameras)

    if not masks_by_obj:
        raise FileNotFoundError(f"No usable object masks found in {mask_dir}.")
    return masks_by_obj


def preprocess_masks(
    masks: Dict[str, np.ndarray],
    erode_px: int,
    largest_cc: bool,
    min_pixels_after: int = 200,
) -> Dict[str, np.ndarray]:
    out = {}
    for cam_id, mask in masks.items():
        m = keep_largest_connected_component(mask) if largest_cc else mask.astype(bool)
        m = erode_mask(m, erode_px=erode_px, min_pixels_after=min_pixels_after)
        out[cam_id] = m
        print(f"[{cam_id}] mask pixels after preprocess: {int(m.sum())}")
    return out


# ============================================================
# Point cloud construction
# ============================================================

def depth_mask_to_world_points(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    min_depth: float = 0.05,
    max_depth: float = 2.0,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if mask.dtype != bool:
        mask = mask.astype(bool)

    if stride > 1:
        sampled = np.zeros_like(mask, dtype=bool)
        sampled[::stride, ::stride] = mask[::stride, ::stride]
        mask = sampled

    v, u = np.where(mask)
    z = depth[v, u]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    u, v, z = u[valid], v[valid], z[valid]
    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float64), None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z.astype(np.float64) / fx
    y = (v.astype(np.float64) - cy) * z.astype(np.float64) / fy

    pts_cam_h = np.stack([x, y, z.astype(np.float64), np.ones_like(z, dtype=np.float64)], axis=1)
    pts_world = (T_cam_to_world @ pts_cam_h.T).T[:, :3]

    colors = None
    if rgb is not None:
        colors = rgb[v, u].astype(np.float64) / 255.0
    return pts_world, colors


def filter_cloud_open3d(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.002,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
    radius: float = 0.01,
    min_points: int = 8,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(points) == 0:
        return points, colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd.points) >= nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(pcd.points) >= min_points:
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

    clean_points = np.asarray(pcd.points)
    clean_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return clean_points, clean_colors


def filter_cloud_lof(points: np.ndarray, n_neighbors: int = 30, contamination: float = 0.03) -> np.ndarray:
    if len(points) < n_neighbors + 1 or contamination <= 0:
        return points
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(points)
    return points[labels == 1]


def build_view_clouds(
    cameras: Dict[str, CameraPacket],
    masks: Dict[str, np.ndarray],
    min_depth: float,
    max_depth: float,
    stride: int,
    voxel_size: float,
    radius: float,
    lof_contamination: float,
) -> List[ViewCloud]:
    view_clouds: List[ViewCloud] = []
    for cam_id, cam in cameras.items():
        if cam_id not in masks:
            continue
        raw_pts, raw_cols = depth_mask_to_world_points(
            depth=cam.depth,
            mask=masks[cam_id],
            K=cam.K,
            T_cam_to_world=cam.T_cam_to_world,
            rgb=cam.rgb,
            min_depth=min_depth,
            max_depth=max_depth,
            stride=stride,
        )
        clean_pts, clean_cols = filter_cloud_open3d(
            raw_pts,
            raw_cols,
            voxel_size=voxel_size,
            nb_neighbors=30,
            std_ratio=2.0,
            radius=radius,
            min_points=8,
        )
        clean_pts = filter_cloud_lof(clean_pts, n_neighbors=30, contamination=lof_contamination)
        view_clouds.append(ViewCloud(cam_id, clean_pts, clean_cols, len(raw_pts), len(clean_pts)))
        print(f"[{cam_id}] raw={len(raw_pts)}, clean={len(clean_pts)}")
    return view_clouds


def merge_view_clouds(view_clouds: List[ViewCloud]) -> np.ndarray:
    pts = [vc.points for vc in view_clouds if len(vc.points) > 0]
    if not pts:
        raise RuntimeError("No valid points from any camera after filtering.")
    return np.concatenate(pts, axis=0)


def save_cloud_ply(path: str | Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)


# ============================================================
# Geometry helpers
# ============================================================

def sample_points(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(points) == 0:
        return points
    if len(points) <= n:
        return points.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def sample_mesh_surface(mesh: trimesh.Trimesh, n: int = 20000, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts, dtype=np.float64)


def center_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    c = np.median(points, axis=0)
    return points - c, c


def robust_extent(points: np.ndarray, q_low: float = 2.0, q_high: float = 98.0) -> np.ndarray:
    lo = np.percentile(points, q_low, axis=0)
    hi = np.percentile(points, q_high, axis=0)
    return np.maximum(hi - lo, 1e-9)


def estimate_bbox_info(points: np.ndarray, use_oriented_bbox: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 10:
        raise RuntimeError("Too few points to estimate bbox.")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if use_oriented_bbox:
        bbox = pcd.get_oriented_bounding_box(robust=True)
        return np.asarray(bbox.center), np.asarray(bbox.extent), np.asarray(bbox.R)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    return 0.5 * (bbox_min + bbox_max), bbox_max - bbox_min, np.eye(3)


def median_pairwise_distance(points: np.ndarray, n_pairs: int = 5000, seed: int = 0) -> float:
    if len(points) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(points), size=n_pairs)
    j = rng.integers(0, len(points), size=n_pairs)
    d = np.linalg.norm(points[i] - points[j], axis=1)
    d = d[d > 1e-9]
    if len(d) == 0:
        return 0.0
    return float(np.median(d))


def mad_filter(values: List[float], z_thresh: float = 2.5) -> Tuple[List[float], List[bool]]:
    if not values:
        return [], []
    arr = np.asarray(values, dtype=np.float64)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-12:
        mask = np.ones(len(arr), dtype=bool)
    else:
        robust_z = 0.6745 * np.abs(arr - med) / mad
        mask = robust_z <= z_thresh
    return arr[mask].tolist(), mask.tolist()


def chamfer_score(
    mesh_pts_centered: np.ndarray,
    cloud_pts_centered: np.ndarray,
    scale: float,
    max_points: int = 12000,
    seed: int = 0,
) -> float:
    """낮을수록 좋음. scale만 적용하고 중심은 각 point cloud median으로 제거한 상태에서 비교."""
    if scale <= 0 or not np.isfinite(scale):
        return float("inf")
    mp = sample_points(mesh_pts_centered, max_points, seed=seed) * scale
    cp = sample_points(cloud_pts_centered, max_points, seed=seed + 1)
    if len(mp) == 0 or len(cp) == 0:
        return float("inf")
    tree_c = cKDTree(cp)
    tree_m = cKDTree(mp)
    d_m2c, _ = tree_c.query(mp, k=1, workers=-1)
    d_c2m, _ = tree_m.query(cp, k=1, workers=-1)
    # 평균 대신 trimmed median/percentile 조합: outlier에 덜 민감
    return float(0.5 * np.median(d_m2c) + 0.5 * np.median(d_c2m))


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """dst ≈ s R src + t. correspondence가 이미 맞춰져 있다고 가정."""
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("src/dst must have same length >= 3")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    xs = src - mu_src
    yd = dst - mu_dst
    cov = (yd.T @ xs) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    if estimate_scale:
        var_src = np.mean(np.sum(xs * xs, axis=1))
        scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-12))
    else:
        scale = 1.0
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def sim3_icp_refine(
    mesh_pts: np.ndarray,
    cloud_pts: np.ndarray,
    init_scale: float,
    max_iter: int = 20,
    max_correspondence_dist: float = 0.03,
    trim_quantile: float = 0.85,
    seed: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """
    단순 Sim(3) ICP. InstantMesh shape가 완전하지 않으므로 final pose로 쓰기보다 scale refinement용으로 사용.
    반환: scale, R, t, report. cloud ≈ s R mesh + t
    """
    src = sample_points(mesh_pts, 15000, seed=seed)
    dst = sample_points(cloud_pts, 15000, seed=seed + 1)
    _, src_center = center_points(src)
    _, dst_center = center_points(dst)

    s = float(init_scale)
    R = np.eye(3)
    t = dst_center - s * src_center
    tree = cKDTree(dst)
    prev_err = float("inf")
    used = 0

    for it in range(max_iter):
        transformed = (s * (R @ src.T)).T + t
        d, idx = tree.query(transformed, k=1, workers=-1)
        valid = d < max_correspondence_dist
        if valid.sum() < 20:
            # 거리 threshold가 너무 빡빡할 때 trimmed nearest만 사용
            q = np.quantile(d, min(max(trim_quantile, 0.1), 0.95))
            valid = d <= q
        src_corr = src[valid]
        dst_corr = dst[idx[valid]]
        if len(src_corr) < 20:
            break
        # 추가 trim
        d_valid = d[valid]
        keep_th = np.quantile(d_valid, min(max(trim_quantile, 0.1), 0.95))
        keep = d_valid <= keep_th
        src_corr = src_corr[keep]
        dst_corr = dst_corr[keep]
        used = len(src_corr)
        if used < 20:
            break
        s_new, R_new, t_new = umeyama_similarity(src_corr, dst_corr, estimate_scale=True)
        if not np.isfinite(s_new) or s_new <= 0:
            break
        transformed_new = (s_new * (R_new @ src_corr.T)).T + t_new
        err = float(np.median(np.linalg.norm(transformed_new - dst_corr, axis=1)))
        s, R, t = s_new, R_new, t_new
        if abs(prev_err - err) < 1e-6:
            prev_err = err
            break
        prev_err = err

    report = {
        "iterations": it + 1 if 'it' in locals() else 0,
        "median_nn_error_m": prev_err,
        "used_correspondences": used,
        "init_scale": init_scale,
        "refined_scale": s,
    }
    return float(s), R, t, report


# ============================================================
# Scale estimation
# ============================================================

def make_scale_candidates(
    mesh_pts: np.ndarray,
    cloud_pts: np.ndarray,
    view_clouds: List[ViewCloud],
    scale_mode: str,
    seed: int = 0,
) -> List[ScaleCandidate]:
    mesh_centered, _ = center_points(mesh_pts)
    cloud_centered, _ = center_points(cloud_pts)
    candidates: List[ScaleCandidate] = []

    mesh_extent = robust_extent(mesh_centered)
    cloud_extent = robust_extent(cloud_centered)
    axis_ratios = cloud_extent / np.maximum(mesh_extent, 1e-9)
    valid = np.isfinite(axis_ratios) & (axis_ratios > 0)
    if valid.any():
        if scale_mode == "mean":
            bbox_s = float(np.mean(axis_ratios[valid]))
        elif scale_mode == "max":
            bbox_s = float(np.max(axis_ratios[valid]))
        else:
            bbox_s = float(np.median(axis_ratios[valid]))
        candidates.append(ScaleCandidate("global_robust_bbox", bbox_s, float("inf"), {
            "axis_ratios": axis_ratios.tolist(),
            "mesh_extent": mesh_extent.tolist(),
            "cloud_extent": cloud_extent.tolist(),
        }))

    mesh_pair = median_pairwise_distance(mesh_centered, seed=seed)
    cloud_pair = median_pairwise_distance(cloud_centered, seed=seed + 11)
    if mesh_pair > 0 and cloud_pair > 0:
        candidates.append(ScaleCandidate("global_pairwise_median", cloud_pair / mesh_pair, float("inf"), {
            "mesh_pairwise_median": mesh_pair,
            "cloud_pairwise_median": cloud_pair,
        }))

    view_scales = []
    view_details = []
    for k, vc in enumerate(view_clouds):
        if len(vc.points) < 50:
            continue
        vc_centered, _ = center_points(vc.points)
        vc_extent = robust_extent(vc_centered)
        ratios = vc_extent / np.maximum(mesh_extent, 1e-9)
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if len(ratios) == 0:
            continue
        view_bbox_s = float(np.median(ratios))
        vc_pair = median_pairwise_distance(vc_centered, seed=seed + 100 + k)
        if mesh_pair > 0 and vc_pair > 0:
            view_pair_s = float(vc_pair / mesh_pair)
            # partial view는 pairwise가 더 안정적인 경우가 많지만, 너무 작게 나올 수 있어 bbox와 median 조합
            view_s = float(np.median([view_bbox_s, view_pair_s]))
        else:
            view_pair_s = None
            view_s = view_bbox_s
        view_scales.append(view_s)
        view_details.append({
            "cam_id": vc.cam_id,
            "scale": view_s,
            "bbox_scale": view_bbox_s,
            "pairwise_scale": view_pair_s,
            "clean_points": len(vc.points),
        })

    kept_scales, keep_mask = mad_filter(view_scales, z_thresh=2.5)
    if kept_scales:
        candidates.append(ScaleCandidate("view_voting_mad_median", float(np.median(kept_scales)), float("inf"), {
            "all_view_scales": view_details,
            "keep_mask": keep_mask,
            "kept_scales": kept_scales,
        }))

    # 후보 주변을 약간 grid search해서 Chamfer가 낮은 scale도 추가
    base_scales = [c.scale for c in candidates if c.scale > 0 and np.isfinite(c.scale)]
    grid = []
    for s in base_scales:
        for f in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]:
            grid.append(s * f)
    for s in sorted(set(round(x, 10) for x in grid)):
        if s > 0:
            candidates.append(ScaleCandidate("grid_chamfer_probe", float(s), float("inf"), {}))

    for c in candidates:
        c.score = chamfer_score(mesh_centered, cloud_centered, c.scale, seed=seed)

    # 중복 scale 후보 정리
    unique: Dict[str, ScaleCandidate] = {}
    for c in sorted(candidates, key=lambda x: x.score):
        key = f"{c.name}:{c.scale:.8f}"
        unique.setdefault(key, c)
    return sorted(unique.values(), key=lambda x: x.score)


def estimate_final_scale(
    mesh: trimesh.Trimesh,
    cloud_pts: np.ndarray,
    view_clouds: List[ViewCloud],
    scale_method: str,
    scale_mode: str,
    refine_sim3_icp: bool,
    icp_max_iter: int,
    icp_max_corr: float,
    seed: int = 0,
) -> Tuple[float, dict, Optional[Tuple[np.ndarray, np.ndarray]]]:
    mesh_pts = sample_mesh_surface(mesh, n=30000, seed=seed)
    candidates = make_scale_candidates(mesh_pts, cloud_pts, view_clouds, scale_mode=scale_mode, seed=seed)
    if not candidates:
        raise RuntimeError("No valid scale candidates were generated.")

    if scale_method == "bbox":
        selected = next((c for c in candidates if c.name == "global_robust_bbox"), candidates[0])
    elif scale_method == "pairwise":
        selected = next((c for c in candidates if c.name == "global_pairwise_median"), candidates[0])
    elif scale_method == "view_voting":
        selected = next((c for c in candidates if c.name == "view_voting_mad_median"), candidates[0])
    else:
        selected = candidates[0]

    final_scale = selected.scale
    sim3_pose = None
    icp_report = None
    if refine_sim3_icp:
        refined_scale, R, t, icp_report = sim3_icp_refine(
            mesh_pts=mesh_pts,
            cloud_pts=cloud_pts,
            init_scale=final_scale,
            max_iter=icp_max_iter,
            max_correspondence_dist=icp_max_corr,
            seed=seed,
        )
        # ICP가 과도하게 튀면 후보 scale 유지
        ratio = refined_scale / max(final_scale, 1e-12)
        if np.isfinite(refined_scale) and 0.75 <= ratio <= 1.33:
            final_scale = refined_scale
            sim3_pose = (R, t)
        else:
            icp_report["warning"] = "Rejected refined scale because it changed too much from selected candidate."

    report = {
        "scale_method": scale_method,
        "scale_mode": scale_mode,
        "selected_candidate": {
            "name": selected.name,
            "scale": selected.scale,
            "score_m": selected.score,
            "details": selected.details,
        },
        "final_scale": final_scale,
        "candidates_top10": [
            {"name": c.name, "scale": c.scale, "score_m": c.score, "details": c.details}
            for c in candidates[:10]
        ],
        "sim3_icp_report": icp_report,
    }
    return final_scale, report, sim3_pose


def export_scaled_mesh(
    mesh_path: str | Path,
    out_glb_path: str | Path,
    scale: float,
    center_mesh: bool = True,
    apply_world_pose: bool = False,
    world_center: Optional[np.ndarray] = None,
    world_R: Optional[np.ndarray] = None,
) -> trimesh.Trimesh:
    mesh = load_mesh_any(mesh_path)
    mesh.apply_scale(scale)
    if center_mesh:
        mesh.apply_translation(-mesh.bounding_box.centroid)
    if apply_world_pose:
        if world_center is None:
            raise ValueError("world_center is required when apply_world_pose=True")
        if world_R is not None:
            T = np.eye(4)
            T[:3, :3] = world_R
            mesh.apply_transform(T)
        mesh.apply_translation(world_center)
    mesh.export(str(out_glb_path))
    return mesh


# ============================================================
# InstantMesh input view selection and runner
# ============================================================

def mask_bbox_area(mask: np.ndarray) -> int:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def choose_best_view(cameras: Dict[str, CameraPacket], masks: Dict[str, np.ndarray], min_mask_pixels: int = 500) -> str:
    best_cam = None
    best_score = -np.inf
    for cam_id, cam in cameras.items():
        mask = masks.get(cam_id)
        if mask is None:
            continue
        area = int(mask.sum())
        if area < min_mask_pixels:
            print(f"[{cam_id}] skip best-view: mask too small: {area}")
            continue
        ys, xs = np.where(mask)
        h, w = mask.shape
        margin = min(xs.min(), ys.min(), w - 1 - xs.max(), h - 1 - ys.max())
        clipped_penalty = 0.5 if margin < 5 else 1.0
        bb_area = max(mask_bbox_area(mask), 1)
        compactness = area / bb_area
        # depth valid ratio도 반영
        valid_depth = np.isfinite(cam.depth[mask]) & (cam.depth[mask] > 0)
        depth_ratio = float(valid_depth.mean()) if len(valid_depth) else 0.0
        score = area * clipped_penalty * (0.5 + compactness) * (0.5 + depth_ratio)
        print(f"[{cam_id}] view score={score:.1f}, mask_pixels={area}, compactness={compactness:.3f}, depth_ratio={depth_ratio:.3f}")
        if score > best_score:
            best_score = score
            best_cam = cam_id
    if best_cam is None:
        raise RuntimeError("No valid camera view for InstantMesh input.")
    return best_cam


def save_rgba_input(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    crop: bool = True,
    padding: int = 30,
    output_size: Optional[int] = 512,
) -> None:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    rgba = np.dstack([rgb, mask_u8])
    if crop:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise RuntimeError("Cannot crop empty mask.")
        h, w = mask.shape
        x0 = max(xs.min() - padding, 0)
        x1 = min(xs.max() + padding + 1, w)
        y0 = max(ys.min() - padding, 0)
        y1 = min(ys.max() + padding + 1, h)
        rgba = rgba[y0:y1, x0:x1]
    if output_size is not None:
        ih, iw = rgba.shape[:2]
        scale = output_size / max(ih, iw)
        nh, nw = int(round(ih * scale)), int(round(iw * scale))
        resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        y0 = (output_size - nh) // 2
        x0 = (output_size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        rgba = canvas
    Image.fromarray(rgba).save(out_path)
    print(f"Saved InstantMesh RGBA input: {out_path}")


def run_instantmesh(
    instantmesh_root: str | Path,
    config_path: str | Path,
    input_png: str | Path,
    output_dir: str | Path,
    python_bin: str = "python",
    no_rembg: bool = True,
    export_texmap: bool = True,
) -> None:
    instantmesh_root = Path(instantmesh_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python_bin, "run.py", str(config_path), str(input_png), "--output_path", str(output_dir)]
    if no_rembg:
        cmd.append("--no_rembg")
    if export_texmap:
        cmd.append("--export_texmap")
    print("Running InstantMesh:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(instantmesh_root), check=True)


def resolve_mesh_path(mesh_arg: Optional[Path], obj_tag: str, num_objects: int) -> Optional[Path]:
    if mesh_arg is None:
        return None
    if mesh_arg.is_dir():
        for ext in (".glb", ".obj", ".ply", ".stl"):
            for name in (f"{obj_tag}{ext}", f"{obj_tag}_mesh{ext}", f"mesh{ext}"):
                cand = mesh_arg / name
                if cand.exists():
                    return cand
        return None
    if num_objects == 1:
        if not mesh_arg.exists():
            raise FileNotFoundError(f"InstantMesh mesh not found: {mesh_arg}")
        return mesh_arg
    print(f"[{obj_tag}] Multi-object mode requires --instantmesh_mesh to be a directory.")
    return None


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out_dir", default="outputs_multicam_instantmesh")

    parser.add_argument("--depth_scale", type=float, default=0.001, help="RealSense uint16 mm -> meter: 0.001")
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--mask_erode_px", type=int, default=5)
    parser.add_argument("--keep_largest_cc", action="store_true", help="Keep largest connected component per mask before erosion")
    parser.add_argument("--voxel_size", type=float, default=0.002, help="meter. 0.002 = 2 mm")
    parser.add_argument("--lof_contamination", type=float, default=0.03)
    parser.add_argument("--use_oriented_bbox", action="store_true")

    parser.add_argument("--obj_ids", default="", help="Comma-separated object ids. Empty = all detected. Single-object fallback id is 0.")

    parser.add_argument("--run_instantmesh", action="store_true")
    parser.add_argument("--instantmesh_root", default="InstantMesh")
    parser.add_argument("--instantmesh_config", default="configs/instant-mesh-large.yaml")
    parser.add_argument("--instantmesh_python", default="python")
    parser.add_argument("--instantmesh_mesh", default="", help="Pre-generated mesh file or directory containing obj{id}.glb/.obj/.ply")

    parser.add_argument("--scale_method", choices=["auto", "bbox", "pairwise", "view_voting"], default="auto")
    parser.add_argument("--scale_mode", choices=["mean", "median", "max"], default="median", help="Used for bbox axis ratio aggregation")
    parser.add_argument("--refine_sim3_icp", action="store_true")
    parser.add_argument("--icp_max_iter", type=int, default=20)
    parser.add_argument("--icp_max_corr", type=float, default=0.03, help="meter")
    parser.add_argument("--apply_world_pose", action="store_true", help="Usually keep False for FoundationPose mesh input")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = load_cameras_from_folder(args.data_dir, depth_scale=args.depth_scale)
    print(f"Loaded cameras: {list(cameras.keys())}")

    selected_obj_ids = [s.strip() for s in args.obj_ids.split(",") if s.strip()] or None
    masks_by_obj_raw = load_masks_per_object(args.mask_dir, cameras, obj_ids=selected_obj_ids)
    print(f"Objects to process: {sorted(masks_by_obj_raw.keys())}")

    mesh_arg = Path(args.instantmesh_mesh) if args.instantmesh_mesh else None
    results_summary: Dict[str, dict] = {}

    for obj_id in sorted(masks_by_obj_raw.keys()):
        obj_tag = f"obj{obj_id}"
        print(f"\n=== Processing {obj_tag} ===")

        raw_masks = masks_by_obj_raw[obj_id]
        masks = preprocess_masks(raw_masks, erode_px=args.mask_erode_px, largest_cc=args.keep_largest_cc)
        obj_cams = {cid: cameras[cid] for cid in masks.keys() if cid in cameras}

        radius = max(args.voxel_size * 4.0, 0.006)
        view_clouds = build_view_clouds(
            cameras=obj_cams,
            masks=masks,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            stride=args.stride,
            voxel_size=args.voxel_size,
            radius=radius,
            lof_contamination=args.lof_contamination,
        )
        cloud_pts = merge_view_clouds(view_clouds)
        if len(cloud_pts) < 50:
            print(f"[{obj_tag}] Skip: too few clean points ({len(cloud_pts)}).")
            continue
        print(f"[{obj_tag}] merged clean cloud: {len(cloud_pts)} points")
        save_cloud_ply(out_dir / f"{obj_tag}_cloud_clean.ply", cloud_pts)

        center_world, bbox_extents_m, R_bbox_to_world = estimate_bbox_info(cloud_pts, use_oriented_bbox=args.use_oriented_bbox)
        bbox_info = {
            "obj_id": obj_id,
            "center_world_m": center_world.tolist(),
            "bbox_extents_m": bbox_extents_m.tolist(),
            "R_bbox_to_world": R_bbox_to_world.tolist(),
            "use_oriented_bbox": bool(args.use_oriented_bbox),
            "merged_clean_points": int(len(cloud_pts)),
            "view_clouds": [
                {"cam_id": vc.cam_id, "raw_count": vc.raw_count, "clean_count": vc.clean_count}
                for vc in view_clouds
            ],
        }
        with open(out_dir / f"{obj_tag}_bbox_metric.json", "w", encoding="utf-8") as f:
            json.dump(bbox_info, f, indent=2)
        print(f"[{obj_tag}] bbox_extents_m={bbox_extents_m}, center={center_world}")

        best_cam = choose_best_view(obj_cams, masks)
        input_png = out_dir / f"{obj_tag}_input.png"
        save_rgba_input(obj_cams[best_cam].rgb, masks[best_cam], input_png, crop=True, padding=30, output_size=512)

        if args.run_instantmesh:
            im_out = out_dir / "instantmesh_output" / obj_tag
            run_instantmesh(
                instantmesh_root=args.instantmesh_root,
                config_path=args.instantmesh_config,
                input_png=input_png.resolve(),
                output_dir=im_out,
                python_bin=args.instantmesh_python,
                no_rembg=True,
                export_texmap=True,
            )

        mesh_path = resolve_mesh_path(mesh_arg, obj_tag=obj_tag, num_objects=len(masks_by_obj_raw))
        if mesh_path is None:
            print(f"[{obj_tag}] Skip mesh scaling: no mesh provided or no matching mesh file.")
            continue

        mesh = load_mesh_any(mesh_path)
        final_scale, scale_report, sim3_pose = estimate_final_scale(
            mesh=mesh,
            cloud_pts=cloud_pts,
            view_clouds=view_clouds,
            scale_method=args.scale_method,
            scale_mode=args.scale_mode,
            refine_sim3_icp=args.refine_sim3_icp,
            icp_max_iter=args.icp_max_iter,
            icp_max_corr=args.icp_max_corr,
            seed=args.seed,
        )

        out_glb = out_dir / f"{obj_tag}_scaled.glb"
        export_scaled_mesh(
            mesh_path=mesh_path,
            out_glb_path=out_glb,
            scale=final_scale,
            center_mesh=True,
            apply_world_pose=args.apply_world_pose,
            world_center=center_world,
            world_R=R_bbox_to_world,
        )

        scale_report.update({
            "mesh_path": str(mesh_path),
            "scaled_glb": str(out_glb),
            "bbox_info": bbox_info,
            "best_cam_for_instantmesh": best_cam,
            "foundationpose_note": "For FoundationPose, usually use obj*_scaled.glb centered at origin and provide RGB-D/mask for pose estimation. Do not enable --apply_world_pose unless you explicitly need world-placed visualization mesh.",
        })
        if sim3_pose is not None:
            R_sim3, t_sim3 = sim3_pose
            scale_report["sim3_pose_mesh_to_world"] = {
                "R": R_sim3.tolist(),
                "t_m": t_sim3.tolist(),
                "warning": "This pose is for scale refinement/debug only. Use FoundationPose for final R,t."
            }

        report_path = out_dir / f"{obj_tag}_scale_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(scale_report, f, indent=2)

        print(f"[{obj_tag}] final_scale={final_scale:.8f}")
        print(f"[{obj_tag}] saved scaled mesh: {out_glb}")
        print(f"[{obj_tag}] saved scale report: {report_path}")

        results_summary[obj_id] = {
            "scaled_glb": str(out_glb),
            "scale": final_scale,
            "scale_report": str(report_path),
            "cloud_clean_ply": str(out_dir / f"{obj_tag}_cloud_clean.ply"),
            "input_png": str(input_png),
            "bbox_extents_m": bbox_extents_m.tolist(),
            "center_world_m": center_world.tolist(),
        }

    if results_summary:
        summary_path = out_dir / "objects_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSaved summary: {summary_path}")
    else:
        print("\nNo object produced scaled mesh. Check masks, depth, and --instantmesh_mesh.")


if __name__ == "__main__":
    main()
