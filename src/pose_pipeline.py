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
from dataclasses import dataclass
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

@dataclass
class PoseEstimate:
    T_base_obj: np.ndarray; position_m: np.ndarray
    quaternion_xyzw: np.ndarray; euler_xyz_deg: np.ndarray
    scale: float = 1.0; confidence: float = 0.0
    fitness: float = 0.0; rmse: float = 0.0
    depth_score: float = 0.0; coverage: float = 0.0

DEPTH_POLICY = {
    "min_depth_m": 0.10, "max_depth_m": 1.20, "voxel_size_m": 0.002,
    "table_dist_thresh_m": 0.008,
    "object_min_height_m": 0.005, "object_max_height_m": 0.15,
    "cluster_eps_m": 0.012, "cluster_min_pts": 100,
    "cluster_min_extent_m": 0.015, "cluster_max_extent_m": 0.18,
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

# 와이어프레임 색상 (BGR)
COLORS = [(0,255,0), (0,165,255), (255,0,255), (0,255,255),
          (255,128,0), (128,255,0), (255,0,128), (0,128,255)]


# ═══════════════════════════════════════════════════════════
# 2. 데이터 로딩
# ═══════════════════════════════════════════════════════════

def load_calibration(data_dir: Path, intrinsics_dir: Path):
    intrinsics = []
    for ci in range(3):
        npz = np.load(str(intrinsics_dir / f"cam{ci}.npz"), allow_pickle=True)
        intrinsics.append(CameraIntrinsics(
            K=npz["color_K"].astype(np.float64),
            D=npz["color_D"].astype(np.float64),
            depth_scale=float(npz["depth_scale_m_per_unit"]),
            width=int(npz["color_w"]), height=int(npz["color_h"])))

    ext_dir = data_dir / "cube_session_01" / "calib_out_cube"
    extrinsics = {0: np.eye(4)}
    for ci in [1, 2]:
        extrinsics[ci] = np.load(str(ext_dir / f"T_C0_C{ci}.npy")).astype(np.float64)
    return intrinsics, extrinsics


def load_frame(data_dir: Path, frame_id: str, intrinsics, extrinsics):
    img_dir = data_dir / "object_capture"
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
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) \
        if isinstance(scene, trimesh.Scene) else scene
    return CanonicalModel(
        name=glb_path.stem, mesh=mesh, center=mesh.centroid.copy(),
        extents_m=mesh.bounding_box.extents.copy(), is_watertight=mesh.is_watertight)


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


def build_observed_mask(cam: CameraFrame, plane_n, plane_d, table_center=None, table_radius=None):
    """
    RGB-D 기반 실제 관측 마스크 생성.
    - 유효 depth
    - 테이블 위 점만 유지
    - 테이블 영역(수평 XZ) 내부만 허용 → 배경 노이즈 제거
    - 색상 채도 필터 (회색 테이블 vs 채색 물체)
    - morphology + connected components
    """
    h, w = cam.depth_u16.shape
    z = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    valid = (z > DEPTH_POLICY["min_depth_m"]) & (z < DEPTH_POLICY["max_depth_m"])

    if valid.sum() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    uu_flat = uu.reshape(-1)
    vv_flat = vv.reshape(-1)
    valid_flat = valid.reshape(-1)

    z_valid = z.reshape(-1)[valid_flat]
    u_valid = uu_flat[valid_flat]
    v_valid = vv_flat[valid_flat]

    K = cam.intrinsics.K
    x = (u_valid - K[0, 2]) * z_valid / K[0, 0]
    y = (v_valid - K[1, 2]) * z_valid / K[1, 1]
    pts_cam = np.stack([x, y, z_valid], axis=1)
    pts_base = transform_points(pts_cam, cam.T_base_cam)

    signed = pts_base @ plane_n + plane_d

    # 테이블 위쪽 물체가 +signed가 되도록 방향을 맞춘다.
    if np.sum(signed > DEPTH_POLICY["object_min_height_m"]) < np.sum(
        signed < -DEPTH_POLICY["object_min_height_m"]
    ):
        signed = -signed

    keep = (
        (signed > DEPTH_POLICY["object_min_height_m"])
        & (signed < DEPTH_POLICY["object_max_height_m"])
    )

    # 테이블 수평 영역(XZ) 내부만 허용 → 배경 노이즈 차단
    if table_center is not None and table_radius is not None:
        horiz = np.array([pts_base[:, 0], pts_base[:, 2]]).T
        tc = np.array([table_center[0], table_center[2]])
        dist = np.linalg.norm(horiz - tc, axis=1)
        keep &= (dist < table_radius)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[v_valid[keep], u_valid[keep]] = 255

    # 색상 필터: 채색 물체 vs 무채색 배경(회색 테이블, 흰 벽)
    hsv = cv2.cvtColor(cam.color_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # sat > 30: 대부분 채색 물체 (빨강/노랑/민트/곤색)
    # 곤색은 어두워서 val이 낮고 sat은 중간 → sat > 25 & val < 120 으로 허용
    rgb_fg = ((sat > 30) | ((sat > 15) & (val < 120) & (val > 20))).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, rgb_fg)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    mask = cv2.dilate(mask, k3, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 200:
            out[labels == i] = 255
    return out


def save_observed_masks(observed_masks, frames, frame_id, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ci, (cam, mask) in enumerate(zip(frames, observed_masks)):
        vis = cam.color_bgr.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"cam{ci} observed mask",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(str(out_dir / f"observed_mask_cam{ci}_{frame_id}.png"), vis)


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


# ═══════════════════════════════════════════════════════════
# 5. GLB 형상 매칭 (멀티 오브젝트용)
# ═══════════════════════════════════════════════════════════

def pca_descriptor(pts):
    c = pts - pts.mean(0)
    vals = np.sqrt(np.maximum(np.linalg.eigvalsh(c.T @ c / len(pts)), 0))
    vals = np.sort(vals)[::-1]
    return vals / vals[0] if vals[0] > 0 else vals

def shortlist_glb_candidates(cluster_pts, models: Dict[str, CanonicalModel], top_k=3):
    """PCA + extent 기준으로 빠르게 후보 축소하고 최종 선택은 정합 점수로 한다."""
    obj_desc = pca_descriptor(cluster_pts)
    obj_ext = cluster_pts.max(0) - cluster_pts.min(0)

    results = []
    for name, model in models.items():
        mod_pts = model.mesh.vertices - model.center
        mod_desc = pca_descriptor(mod_pts)

        desc_dist = np.linalg.norm(obj_desc - mod_desc)

        mod_ext = model.extents_m
        obj_ext_n = np.sort(obj_ext) / (np.max(obj_ext) + 1e-8)
        mod_ext_n = np.sort(mod_ext) / (np.max(mod_ext) + 1e-8)
        ext_dist = np.linalg.norm(obj_ext_n - mod_ext_n)

        coarse_score = 1.0 / (1.0 + 8.0 * desc_dist + 4.0 * ext_dist)
        results.append((name, coarse_score))

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


def register_model(model_pts, obj_pts, frames, observed_masks):
    """CAD → 관측 점군 정합. 실물 크기(scale≈1.0) 중심 탐색.

    GLB 모델이 실물 크기(m)이므로 scale=1.0이 정답.
    뎁스 카메라의 부분 관측을 고려해 ±20% 범위 탐색.
    """
    obj_center = obj_pts.mean(0)
    obj_max = (obj_pts.max(0)-obj_pts.min(0)).max()
    mod_max = (model_pts.max(0)-model_pts.min(0)).max()

    tgt = o3d.geometry.PointCloud(); tgt.points = o3d.utility.Vector3dVector(obj_pts)
    best = None

    # scale=1.0 중심 탐색 (GLB = 실물 크기이므로)
    for sf in [0.80, 0.88, 0.94, 1.0, 1.06, 1.12, 1.20]:
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
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
            if r.fitness > 0.05: fpfh_T = r.transformation
        except: pass

        # 초기 후보
        rots = rotation_candidates(scaled, obj_pts)
        inits = []
        if fpfh_T is not None: inits.append(fpfh_T)
        for R in rots[:16]:
            T = np.eye(4); T[:3,:3]=R; T[:3,3]=obj_center-R@src_c; inits.append(T)
        if not inits:
            T=np.eye(4); T[:3,3]=obj_center-src_c; inits.append(T)

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
                o3d.pipelines.registration.ICPConvergenceCriteria(1e-7,1e-7,80))
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
                o3d.pipelines.registration.ICPConvergenceCriteria(1e-8,1e-8,200))
            cT = icp.transformation

        fT = cT @ bl_T
        fc = (fT @ np.append(src_c,1))[:3]
        if np.linalg.norm(fc-obj_center) > rs*0.5: continue

        aligned = transform_points(scaled, fT)
        ds = mv_depth_score(aligned, frames, observed_masks)
        cs = cov_score(aligned, tgt, max(rs*0.08, 0.003))
        score = ds * cs

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
                rmse=icp.inlier_rmse, depth_score=ds, coverage=cs)

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
        "rmse": pose.rmse, "depth_score": pose.depth_score, "coverage": pose.coverage,
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
                 multi_object=False):
    data_dir = Path(data_dir); intrinsics_dir = Path(intrinsics_dir); out = Path(output_dir)

    print("=" * 60)
    print(f" Pose Estimation — Frame {frame_id} {'[MULTI]' if multi_object else '[SINGLE]'}")
    print("=" * 60)

    # 1. 캘리브레이션 + GLB
    intrinsics, extrinsics = load_calibration(data_dir, intrinsics_dir)
    glb_paths: Dict[str, Path] = {}
    all_models: Dict[str, CanonicalModel] = {}
    if glb_path:
        gp = Path(glb_path)
        m = normalize_glb(gp); all_models[m.name] = m; glb_paths[m.name] = gp
    else:
        for i in range(1, 5):
            p = data_dir / f"object_{i:03d}.glb"
            if p.exists():
                m = normalize_glb(p); all_models[m.name] = m; glb_paths[m.name] = p

    for n, m in all_models.items():
        print(f"  {n} ({OBJECT_LABELS.get(n,'')}): "
              f"[{m.extents_m[0]*100:.1f}, {m.extents_m[1]*100:.1f}, {m.extents_m[2]*100:.1f}]cm")

    # 2. RGB-D 로드
    frames = load_frame(data_dir, frame_id, intrinsics, extrinsics)

    # 3. 테이블 추정 + 실제 관측 마스크 생성 + mask 기반 점군 융합
    print("\n[1] 테이블 추정 + observed mask 생성")
    plane_n, plane_d, table_center, table_radius = estimate_table_plane(frames)
    print(f"  table center: [{table_center[0]:.3f}, {table_center[1]:.3f}, {table_center[2]:.3f}] m")
    print(f"  table radius: {table_radius:.3f} m")
    observed_masks = [build_observed_mask(cam, plane_n, plane_d, table_center, table_radius) for cam in frames]
    save_observed_masks(observed_masks, frames, frame_id, out)

    above_pts = fuse_masked_points(frames, observed_masks)
    print(f"  observed foreground pts: {len(above_pts)}")
    if len(above_pts) == 0:
        raise RuntimeError("foreground point cloud가 비어 있음")

    # 4. 물체 분리
    if multi_object:
        print("\n[2] 멀티 오브젝트 클러스터링")
        clusters = find_all_clusters(above_pts)
        print(f"  발견된 클러스터: {len(clusters)}개")
        for i, (cpts, cent) in enumerate(clusters):
            ext = cpts.max(0)-cpts.min(0)
            print(f"    #{i}: {len(cpts)} pts, "
                  f"extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm")
    else:
        print("\n[2] 단일 물체 추출")
        single_pts = find_single_object(above_pts)
        if len(single_pts) == 0:
            raise RuntimeError("단일 물체 추출 실패: foreground point cloud가 비어 있음")
        clusters = [(single_pts, single_pts.mean(axis=0))]
        ext = single_pts.max(0)-single_pts.min(0)
        print(f"  물체: {len(single_pts)} pts, "
              f"extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm")

    # 5~7. 각 클러스터별 GLB 매칭 + 정합
    MIN_CONFIDENCE = 0.03     # 이 미만이면 거부
    MIN_FITNESS    = 0.10     # ICP fitness 최소 하한
    MAX_SCALE_DEV  = 0.25     # |scale - 1.0| 허용 편차

    all_results = []
    all_poses = []
    all_model_objs = []
    all_masks_list = []

    used_glbs = set()

    if multi_object:
        # ── 글로벌 최적 할당: 모든 (클러스터, GLB) 조합 점수 계산 ──
        print("\n[3] 글로벌 최적 할당 시작")
        n_glb = len(all_models)
        # 상위 클러스터만 시도 (물체 수 × 2 또는 최대 12개)
        max_clusters = min(len(clusters), n_glb * 3, 12)
        score_table = {}  # (ci, model_name) → (score, pose, support_masks)

        for ci in range(max_clusters):
            cpts, cent = clusters[ci]
            print(f"\n[3-{ci}] 클러스터 #{ci} 처리 ({len(cpts)} pts)")
            support_masks = project_cluster_to_support_masks(cpts, frames)

            shortlist = shortlist_glb_candidates(cpts, all_models, top_k=min(n_glb, 3))
            if not shortlist:
                print("  매칭 가능한 GLB 없음 → 스킵")
                continue

            for cand_name, coarse_score in shortlist:
                model = all_models[cand_name]
                model_pts = sample_model_points(model)
                try:
                    pose_tmp = register_model(model_pts, cpts, frames, observed_masks)
                except RuntimeError:
                    continue

                # scale 편차 검사
                if abs(pose_tmp.scale - 1.0) > MAX_SCALE_DEV:
                    print(f"    candidate={cand_name} SKIP scale={pose_tmp.scale:.3f} (편차 초과)")
                    continue

                final_score = 0.2 * coarse_score + 0.8 * pose_tmp.confidence
                print(
                    f"    candidate={cand_name} coarse={coarse_score:.3f} "
                    f"pose_conf={pose_tmp.confidence:.3f} scale={pose_tmp.scale:.3f} "
                    f"fitness={pose_tmp.fitness:.3f} final={final_score:.3f}"
                )

                prev = score_table.get((ci, cand_name))
                if prev is None or final_score > prev[0]:
                    score_table[(ci, cand_name)] = (final_score, pose_tmp, support_masks)

        # ── 탐욕적이 아닌 최적 할당 (GLB당 최고 점수 클러스터) ──
        print("\n[4] 최적 할당")
        assigned = {}  # model_name → (ci, score, pose, masks)
        used_clusters = set()

        # 전체 점수 내림차순 정렬
        candidates = sorted(score_table.items(), key=lambda x: x[1][0], reverse=True)

        for (ci, mname), (score, pose, masks) in candidates:
            if mname in assigned or ci in used_clusters:
                continue
            # 신뢰도 · fitness 하한 검사
            if pose.confidence < MIN_CONFIDENCE:
                print(f"  {mname} 거부: confidence={pose.confidence:.4f} < {MIN_CONFIDENCE}")
                continue
            if pose.fitness < MIN_FITNESS:
                print(f"  {mname} 거부: fitness={pose.fitness:.4f} < {MIN_FITNESS}")
                continue
            assigned[mname] = (ci, score, pose, masks)
            used_clusters.add(ci)
            label = OBJECT_LABELS.get(mname, mname)
            print(f"  {label} → 클러스터 #{ci} score={score:.3f} "
                  f"conf={pose.confidence:.3f} scale={pose.scale:.3f}")

        # 할당된 결과 처리
        for model_name, (ci, score, pose, support_masks) in sorted(assigned.items()):
            model = all_models[model_name]
            used_glbs.add(model_name)

            label = OBJECT_LABELS.get(model_name, model_name)
            print(f"\n  ── {label} (클러스터 #{ci}) ──")
            print(f"  position:   [{pose.position_m[0]:+.4f}, {pose.position_m[1]:+.4f}, {pose.position_m[2]:+.4f}] m")
            print(f"  quaternion: [{pose.quaternion_xyzw[0]:+.4f}, {pose.quaternion_xyzw[1]:+.4f}, "
                  f"{pose.quaternion_xyzw[2]:+.4f}, {pose.quaternion_xyzw[3]:+.4f}]")
            print(f"  scale={pose.scale:.4f}  confidence={pose.confidence:.4f}  fitness={pose.fitness:.4f}")

            result = export_result(pose, model, frame_id, out, glb_paths[model_name])
            all_results.append(result)
            all_poses.append(pose)
            all_model_objs.append(model)
            all_masks_list.append(support_masks)

        unmatched = set(all_models.keys()) - used_glbs
        if unmatched:
            print(f"\n  [WARN] 미매칭 GLB: {', '.join(sorted(unmatched))}")

    else:
        # ── 단일 / 수동 GLB 모드 (기존 로직) ──
        for ci, (cpts, cent) in enumerate(clusters):
            print(f"\n[3-{ci}] 클러스터 #{ci} 처리 ({len(cpts)} pts)")

            support_masks = project_cluster_to_support_masks(cpts, frames)

            if glb_path:
                model_name = Path(glb_path).stem
                if model_name not in all_models:
                    print(f"  [WARN] {model_name} 모델 없음 → 스킵")
                    continue
            else:
                model_name = FRAME_TO_GLB.get(int(frame_id))
                if model_name is None:
                    raise RuntimeError(f"프레임 {frame_id}에 대한 GLB 매핑 없음")
                print(f"  매핑: {model_name} ({OBJECT_LABELS.get(model_name,'')})")
                if model_name not in all_models:
                    print(f"  [WARN] {model_name} 모델 없음 → 스킵")
                    continue

            model = all_models[model_name]
            model_pts = sample_model_points(model)
            try:
                pose = register_model(model_pts, cpts, frames, observed_masks)
            except RuntimeError as e:
                print(f"  [FAIL] {e}")
                continue

            used_glbs.add(model_name)

            label = OBJECT_LABELS.get(model_name, model_name)
            print(f"  ── {label} ──")
            print(f"  position:   [{pose.position_m[0]:+.4f}, {pose.position_m[1]:+.4f}, {pose.position_m[2]:+.4f}] m")
            print(f"  quaternion: [{pose.quaternion_xyzw[0]:+.4f}, {pose.quaternion_xyzw[1]:+.4f}, "
                  f"{pose.quaternion_xyzw[2]:+.4f}, {pose.quaternion_xyzw[3]:+.4f}]")
            print(f"  scale={pose.scale:.4f}  confidence={pose.confidence:.4f}")

            result = export_result(pose, model, frame_id, out, glb_paths[model_name])
            all_results.append(result)
            all_poses.append(pose)
            all_model_objs.append(model)
            all_masks_list.append(support_masks)

    # 8. 합산 오버레이
    if all_poses:
        save_combined_overlay(all_poses, all_model_objs, all_masks_list, frames, frame_id, out)
        print(f"\n  overlay: {out / f'overlay_{frame_id}.png'}")

    for r in all_results:
        obj_name = r['object_name']
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
    args = parser.parse_args()

    if args.batch:
        cam0_dir = Path(args.data_dir) / "object_capture" / "cam0"
        fids = sorted(f.stem.replace("rgb_","") for f in cam0_dir.glob("rgb_*.jpg"))
        print(f"배치: {len(fids)} 프레임\n")
        all_r = []
        for fid in fids:
            try:
                r = run_pipeline(args.data_dir, args.intrinsics_dir, fid,
                                 args.glb, args.output_dir, args.multi)
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
                     args.glb, args.output_dir, args.multi)


if __name__ == "__main__":
    main()
