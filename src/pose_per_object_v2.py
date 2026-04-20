#!/usr/bin/env python3
"""
CPU-friendly per-object 6D pose estimator.

색상 prior로 각 물체 마스크를 독립적으로 추출 → 멀티뷰 점군 융합 →
GLB ICP 정합. 기존 pose_pipeline.py의 association/recovery 단계에서
발생하던 track mixing / GLB 오매칭 문제를 우회.

출력:
  src/output/pose_per_object_v2/
    object_XXX_posed_{frame}.glb (+ _isaac)
    pose_object_XXX_{frame}.json
    debug_mask_cam{0,1,2}_object_XXX_{frame}.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
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
    OBJECT_COLOR_PRIORS_HSV,
    OBJECT_LABELS,
    OBJECT_SYMMETRY,
    PoseEstimate,
    T_ISAAC_CV,
    backproject_depth,
    combine_pose_scores,
    estimate_table_plane,
    export_result,
    load_calibration,
    load_frame,
    mv_depth_score,
    normalize_glb,
    rotation_candidates,
    sample_model_points,
    silhouette_iou_score,
    transform_points,
)
from scipy.spatial.transform import Rotation as Rot

import open3d as o3d

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
OUT_DIR = SCRIPT_DIR / "output" / "pose_per_object_v2"

# 실제 이미지에서 측정한 HSV (cam0/1/2 평균)
#   red   H≈1-3,  S≈215, V≈165
#   yellow H≈19,  S≈245, V≈210
#   mint   H≈90,  S≈120, V≈170
#   navy   H≈103, S≈120, V≈60
COLOR_REF_HSV = {
    "object_001": (2.0,   215, 165),
    "object_002": (19.0,  240, 205),
    "object_003": (103.0, 120,  60),
    "object_004": (90.0,  120, 170),
}
# hue 반경 / saturation 하한 / value 구간
# 실측 HSV 분포 기반 조정 (카메라별 하이라이트/그림자 커버):
#   빨강: cam0 H가 넓어 hue 유지, S/V는 실측 범위 커버
#   노랑: cam0 S 하한 낮춤 (181→160), V 하한 낮춤 (154→130)
#   곤색: cam1 hue 넓어짐 (98-116), S 하한 낮춤, V 상한 높임
#   민트: cam2 S 범위 매우 넓어 (79-240), V 하한 낮춤
COLOR_THRESHOLDS = {
    "object_001": {"hue": 12.0, "s_min": 100, "v_min": 70,  "v_max": 245},  # 빨강
    "object_002": {"hue": 12.0, "s_min": 160, "v_min": 130, "v_max": 255},  # 노랑
    "object_003": {"hue": 18.0, "s_min": 55,  "v_min": 20,  "v_max": 120},  # 곤색
    "object_004": {"hue": 16.0, "s_min": 50,  "v_min": 110, "v_max": 240},  # 민트
}

RELAXED_RECOVER_THR = {"hue": 22.0, "s_min": 25, "v_min": 15, "v_max": 255}

def horizontal_axes(up_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """수직축(up)에 직교하는 두 정규화 수평 기저 벡터."""
    up = up_vec / (np.linalg.norm(up_vec) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(up, ref)
    a /= np.linalg.norm(a) + 1e-12
    b = np.cross(up, a)
    return a, b


def horizontal_max_extent(pts: np.ndarray, up_vec: np.ndarray) -> float:
    """점군의 수평 평면 방향 max bounding-box extent."""
    if len(pts) == 0:
        return 0.0
    a, b = horizontal_axes(up_vec)
    xs = pts @ a
    ys = pts @ b
    return float(max(xs.max() - xs.min(), ys.max() - ys.min()))


def estimate_scale_auto(
    obj_pts: np.ndarray,
    model_pts: np.ndarray,
    R_align: np.ndarray,
    up_world: np.ndarray,
) -> float:
    """관측/모델 수평 extent 비율로 scale 자동 추정.

    탑뷰 계열 관측에서 수평 extent는 수직 관측 정도에 무관하게 실측에 가깝고,
    3대 카메라 cross-view로 대체로 정확한 XY 길이를 복원한다.
    """
    model_rot = (R_align @ model_pts.T).T
    obs_h = horizontal_max_extent(obj_pts, up_world)
    mod_h = horizontal_max_extent(model_rot, up_world)
    if mod_h < 1e-6:
        return 1.0
    return float(np.clip(obs_h / mod_h, 0.2, 2.5))


def hsv_circular_distance(h_img: np.ndarray, h_ref: float) -> np.ndarray:
    d = np.abs(h_img - h_ref)
    return np.minimum(d, 180.0 - d)


def depth_segment_all_objects(
    frames: List[CameraFrame],
    table_n: np.ndarray, table_d: float,
    table_center: np.ndarray, table_radius: float,
) -> Dict[str, List[np.ndarray]]:
    """Depth 기반 물체 세그멘테이션 (모든 카메라 + 모든 물체 한 번에).

    HSV 색상 임계 대신 depth map에서 '테이블 위 15mm~15cm 돌출 영역'을
    connected component로 분리한 뒤, 각 component의 평균 HSV로 물체에 할당.
    경계가 depth edge 기준이라 색상 번짐/조명 변화에 강건.
    """
    object_names = list(COLOR_REF_HSV.keys())
    result: Dict[str, List[np.ndarray]] = {n: [] for n in object_names}

    for ci, cam in enumerate(frames):
        depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
        h, w = cam.intrinsics.height, cam.intrinsics.width
        K = cam.intrinsics.K
        u_px, v_px = np.meshgrid(np.arange(w), np.arange(h))
        valid = depth > 0.05
        x_cam = (u_px - K[0, 2]) * depth / K[0, 0]
        y_cam = (v_px - K[1, 2]) * depth / K[1, 1]
        pts_cam_flat = np.stack([x_cam.ravel(), y_cam.ravel(), depth.ravel()], axis=-1)
        R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
        pts_base = (R @ pts_cam_flat.T).T + t
        heights = -(pts_base @ table_n + table_d).reshape(h, w)
        horiz = np.linalg.norm(
            pts_base[:, [0, 2]].reshape(h, w, 2) - table_center[[0, 2]], axis=-1
        )

        above = valid & (heights > 0.015) & (heights < 0.15) & (horiz < max(table_radius, 0.30))
        above_u8 = above.astype(np.uint8) * 255
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        above_u8 = cv2.morphologyEx(above_u8, cv2.MORPH_OPEN, k5)
        above_u8 = cv2.morphologyEx(above_u8, cv2.MORPH_CLOSE, k5)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(above_u8)
        hsv_img = cv2.cvtColor(cam.color_bgr, cv2.COLOR_BGR2HSV)

        # 각 component의 평균 HSV 수집
        comps = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 300:
                continue
            comp_mask = labels == i
            mh = float(hsv_img[:, :, 0][comp_mask].mean())
            ms = float(hsv_img[:, :, 1][comp_mask].mean())
            mv = float(hsv_img[:, :, 2][comp_mask].mean())
            comps.append((i, area, mh, ms, mv))

        # Greedy 할당: 각 물체에 대해 최소 색상 거리인 component 할당
        # 이미 할당된 component는 재사용 불가 (1:1 매칭)
        assigned_comps: set = set()
        for obj_name in object_names:
            ref_h, ref_s, ref_v = COLOR_REF_HSV[obj_name]
            best_idx = -1
            best_dist = 999.0
            for idx, (comp_id, area, mh, ms, mv) in enumerate(comps):
                if idx in assigned_comps:
                    continue
                hd = min(abs(mh - ref_h), 180.0 - abs(mh - ref_h))
                dist = hd + 0.2 * abs(ms - ref_s) + 0.1 * abs(mv - ref_v)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx >= 0 and best_dist < 60.0:
                comp_id = comps[best_idx][0]
                mask = (labels == comp_id).astype(np.uint8) * 255
                result[obj_name].append(mask)
                assigned_comps.add(best_idx)
            else:
                result[obj_name].append(np.zeros((h, w), dtype=np.uint8))

    return result


def color_mask_for_object(bgr: np.ndarray, object_name: str) -> np.ndarray:
    ref_h = COLOR_REF_HSV[object_name][0]
    thr = COLOR_THRESHOLDS[object_name]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hue_ok = hsv_circular_distance(h, float(ref_h)) <= thr["hue"]
    sv_ok = (s >= thr["s_min"]) & (v >= thr["v_min"]) & (v <= thr["v_max"])
    mask = (hue_ok & sv_ok).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)       # 작은 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k9)       # 인접 조각 병합 (5→9)
    mask = cv2.dilate(mask, k7, iterations=1)                 # 경계 확장
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)        # 내부 구멍 채움
    return mask


def relaxed_color_mask_for_object(
    bgr: np.ndarray,
    object_name: str,
    hue_radius: Optional[float] = None,
) -> np.ndarray:
    if object_name not in COLOR_REF_HSV:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    ref_h = COLOR_REF_HSV[object_name][0]
    hue_thr = RELAXED_RECOVER_THR["hue"] if hue_radius is None else float(hue_radius)
    hue_ok = hsv_circular_distance(h, float(ref_h)) <= hue_thr
    sv_ok = ((s >= RELAXED_RECOVER_THR["s_min"])
             & (v >= RELAXED_RECOVER_THR["v_min"])
             & (v <= RELAXED_RECOVER_THR["v_max"]))
    mask = (hue_ok & sv_ok).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    return mask


def best_connected_component(mask: np.ndarray, min_area: int = 150) -> np.ndarray:
    if int(np.count_nonzero(mask)) < min_area:
        return np.zeros_like(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if int(areas[best - 1]) < min_area:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def connected_components_sorted(
    mask: np.ndarray,
    min_area: int = 150,
    top_k: Optional[int] = None,
) -> List[np.ndarray]:
    if int(np.count_nonzero(mask)) < min_area:
        return []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps: List[Tuple[int, np.ndarray]] = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = np.zeros_like(mask)
        comp[labels == idx] = 255
        comps.append((area, comp))
    comps.sort(key=lambda x: x[0], reverse=True)
    masks = [m for _, m in comps]
    return masks[:top_k] if top_k is not None else masks


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def mask_overlap_metrics(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    ab = a > 0
    bb = b > 0
    inter = int(np.logical_and(ab, bb).sum())
    union = int(np.logical_or(ab, bb).sum())
    area_a = int(ab.sum())
    area_b = int(bb.sum())
    iou = inter / union if union > 0 else 0.0
    precision = inter / area_a if area_a > 0 else 0.0
    recall = inter / area_b if area_b > 0 else 0.0
    return float(iou), float(precision), float(recall)


def project_points_to_mask_local(
    obj_pts: np.ndarray,
    cam: CameraFrame,
    dilate_px: int = 5,
) -> np.ndarray:
    h, w = cam.intrinsics.height, cam.intrinsics.width
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(obj_pts) == 0:
        return mask
    pts_cam = (np.linalg.inv(cam.T_base_cam) @ np.hstack(
        [obj_pts, np.ones((len(obj_pts), 1))]
    ).T)[:3].T
    z = pts_cam[:, 2]
    ok = z > 0.05
    if not ok.any():
        return mask
    K = cam.intrinsics.K
    u = np.round(K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]).astype(np.int32)
    v = np.round(K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]).astype(np.int32)
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not valid.any():
        return mask
    mask[v[valid], u[valid]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate_px) | 1,) * 2)
    mask = cv2.dilate(mask, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def recover_mask_from_projection(
    cam: CameraFrame,
    proj_mask: np.ndarray,
    object_name: Optional[str],
    morph_kernel: np.ndarray,
    pad_kernel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    proj_dilated = cv2.dilate(proj_mask, pad_kernel, iterations=2)
    proj_dilated = cv2.morphologyEx(proj_dilated, cv2.MORPH_CLOSE, pad_kernel)
    if object_name not in COLOR_REF_HSV:
        return proj_dilated.copy(), proj_dilated
    color_ok = relaxed_color_mask_for_object(cam.color_bgr, object_name)
    recovered = cv2.bitwise_and(color_ok, proj_dilated)
    recovered = cv2.morphologyEx(recovered, cv2.MORPH_OPEN, morph_kernel)
    recovered = cv2.morphologyEx(recovered, cv2.MORPH_CLOSE, morph_kernel)
    return recovered, proj_dilated


def select_multiview_seed_masks(
    frames: List[CameraFrame],
    candidate_masks: List[List[np.ndarray]],
    table_info,
    top_zero: bool = True,
) -> List[np.ndarray]:
    tn, td, tc, tr = table_info
    options_by_cam: List[List[np.ndarray]] = []
    for ci, opts in enumerate(candidate_masks):
        base = np.zeros((frames[ci].intrinsics.height, frames[ci].intrinsics.width), dtype=np.uint8)
        cam_opts = [base] if top_zero else []
        cam_opts.extend([m.copy() for m in opts])
        if not cam_opts:
            cam_opts = [base]
        options_by_cam.append(cam_opts[:4])

    best_score = -1e18
    best_combo = [opts[0].copy() for opts in options_by_cam]
    for combo in itertools.product(*options_by_cam):
        combo_masks = [m.copy() for m in combo]
        valid_areas = [mask_area(m) for m in combo_masks]
        n_valid = sum(a >= 80 for a in valid_areas)
        if n_valid == 0:
            continue
        obj_pts = fuse_object_points(frames, combo_masks, tn, td, tc, tr)
        if len(obj_pts) < 40:
            continue
        ext = obj_pts.max(0) - obj_pts.min(0)
        ext_max = float(ext.max())
        overlap_terms = []
        proj_terms = 0
        for cam, raw in zip(frames, combo_masks):
            proj = project_points_to_mask_local(obj_pts, cam, dilate_px=7)
            raw_area = mask_area(raw)
            proj_area = mask_area(proj)
            if proj_area >= 60:
                proj_terms += 1
            if raw_area >= 50 and proj_area >= 50:
                iou, precision, recall = mask_overlap_metrics(raw, proj)
                overlap_terms.append(0.50 * iou + 0.30 * precision + 0.20 * recall)
            elif raw_area < 50 and proj_area >= 80:
                overlap_terms.append(0.10)
            elif raw_area >= 50 and proj_area < 50:
                overlap_terms.append(-0.20)
        if not overlap_terms:
            continue
        extent_penalty = 0.0
        if ext_max > 0.18:
            extent_penalty -= 4.0 * (ext_max - 0.18) / 0.05
        if ext_max < 0.008:
            extent_penalty -= 1.0
        score = (
            2.4 * float(np.mean(overlap_terms))
            + 0.0008 * float(min(len(obj_pts), 4000))
            + 0.35 * n_valid
            + 0.10 * proj_terms
            + extent_penalty
        )
        if n_valid == 1:
            score -= 0.25
        if score > best_score:
            best_score = score
            best_combo = combo_masks
    return best_combo


def fuse_object_points(
    frames: List[CameraFrame],
    masks: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    table_center: np.ndarray,
    table_radius: float,
) -> np.ndarray:
    all_pts = []
    for cam, mask in zip(frames, masks):
        if int(np.count_nonzero(mask)) < 100:
            continue
        pts_cam = backproject_depth(cam, mask)
        if len(pts_cam) == 0:
            continue
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        all_pts.append(pts_base)
    if not all_pts:
        return np.zeros((0, 3))

    pts = np.vstack(all_pts)

    # 테이블 위, 반경 안쪽만. get_above_table_points와 동일 관례 사용.
    heights = -(pts @ table_n + table_d)
    on_top = heights > 0.005  # 5mm 위
    horiz = np.linalg.norm(pts[:, [0, 2]] - table_center[[0, 2]], axis=1)
    within = horiz < max(table_radius, 0.30)
    before = len(pts)
    pts = pts[on_top & within]
    if len(pts) < 30:
        print(f"      [debug] fused={before} after_table_filter={len(pts)}")
        return np.zeros((0, 3))

    # 보클 다운샘플 (큰 마스크 안전 캡: pre-voxel 20만 포인트 이하로 랜덤 샘플)
    import open3d as o3d
    if len(pts) > 200_000:
        idx = np.random.choice(len(pts), 200_000, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(DEPTH_POLICY["voxel_size_m"])

    # 통계적 아웃라이어 제거
    if len(pcd.points) > 80:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # DBSCAN + 크기 필터로 목표 물체 클러스터 선택
    pts_np = np.asarray(pcd.points)
    if len(pts_np) < 30:
        return pts_np
    labels = np.array(
        pcd.cluster_dbscan(
            eps=DEPTH_POLICY.get("cluster_eps_m", 0.012),
            min_points=10,
            print_progress=False,
        )
    )
    if labels.max() < 0:
        return pts_np
    u, c = np.unique(labels[labels >= 0], return_counts=True)
    # 후보: 점 개수 ≥ 30, bbox 최대 extent < 15cm (손크기 물체 범위)
    candidates = []
    for lbl, cnt in zip(u, c):
        if cnt < 30:
            continue
        sub = pts_np[labels == lbl]
        ext = float((sub.max(0) - sub.min(0)).max())
        if ext > 0.15:
            continue
        candidates.append((cnt, sub))
    if not candidates:
        # fallback: 원래 최대 클러스터
        return pts_np[labels == u[int(np.argmax(c))]]
    # 가장 큰 "유효" 클러스터 선택
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis) + 1e-12
        return Rot.from_rotvec(axis * np.pi).as_matrix()
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def model_up_candidates(
    model_extents_m: np.ndarray, symmetry: str
) -> List[np.ndarray]:
    """모델의 '수직축' 후보.

    yaw-대칭: GLB bbox 최장축만(실린더 심볼상 Z축).
    일반: ±X/±Y/±Z 모두.
    """
    if symmetry == "yaw":
        axis_idx = int(np.argmax(np.asarray(model_extents_m)))
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_idx] = 1.0
        return [axis, -axis]
    out = []
    for idx in range(3):
        for sign in (1.0, -1.0):
            vec = np.zeros(3, dtype=np.float64)
            vec[idx] = sign
            out.append(vec)
    return out


def vertical_constrained_inits(
    table_n: np.ndarray,
    model_pts: np.ndarray,
    model_extents_m: np.ndarray,
    obj_pts: np.ndarray,
    yaw_steps: int = 18,
    symmetry: str = "none",
) -> List[Tuple[np.ndarray, float]]:
    """model up × yaw 스윕 + auto-scale.

    반환: [(R_init, scale_auto), ...]
    """
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


def reproject_to_vertical(R: np.ndarray, target_up: np.ndarray) -> np.ndarray:
    """R의 수직축을 target_up에 강제 정렬. yaw는 최대한 유지.

    ICP 후 수직 정렬이 깨질 경우 reject 대신 재투영으로 복구.
    """
    target_up = target_up / (np.linalg.norm(target_up) + 1e-12)
    dots = R.T @ target_up
    idx = int(np.argmax(np.abs(dots)))
    sign = 1.0 if dots[idx] > 0 else -1.0
    cur_up = R[:, idx] * sign
    R_fix = rotation_between(cur_up, target_up)
    return R_fix @ R


def snap_to_table(
    T: np.ndarray,
    scaled_model_pts: np.ndarray,
    table_n: np.ndarray,
    table_d: float,
    margin_m: float = 0.002,
) -> np.ndarray:
    """포즈된 모델의 최저점이 테이블 표면에 닿도록 수직 평행이동.

    table plane: n·p + d = 0, height = -(n·p + d).
    최저 height (= 가장 테이블에 가까운 점)이 margin과 같아지도록 보정.
    """
    aligned = (T[:3, :3] @ scaled_model_pts.T).T + T[:3, 3]
    heights = -(aligned @ table_n + table_d)
    min_h = float(heights.min())
    delta = margin_m - min_h  # delta만큼 위로 올림 (수직 방향 = -table_n)
    if abs(delta) < 1e-6:
        return T
    T_out = T.copy()
    T_out[:3, 3] = T[:3, 3] + (-table_n) * delta
    return T_out


def _run_icp_stack(
    scaled: np.ndarray,
    T0: np.ndarray,
    tgt: o3d.geometry.PointCloud,
    mod_max_world: float,
) -> np.ndarray:
    """Coarse → fine point-to-point + fine point-to-plane ICP.

    mod_max_world: 스케일이 적용된 model extent(월드 단위). correspondence
    threshold가 실제 물체 크기에 맞춰 축소되어야 ICP가 지역 minima에 수렴.
    """
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
    # Point-to-plane fine refine (taget에 normal 필요)
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


def _extent_match_score(aligned: np.ndarray, obj_pts: np.ndarray) -> float:
    """모델과 관측의 3D bounding-box extent 일치도 (0~1).

    작은 model이 큰 관측 안에 '쏙 들어가' bwd coverage로 높은 점수를
    받는 편향을 교정한다. 축별 min(obs/mod, mod/obs)의 평균.
    """
    if len(aligned) < 10 or len(obj_pts) < 10:
        return 0.0
    mod_ext = aligned.max(0) - aligned.min(0)
    obs_ext = obj_pts.max(0) - obj_pts.min(0)
    ratios = []
    for i in range(3):
        me = float(mod_ext[i])
        oe = float(obs_ext[i])
        if me < 1e-6 or oe < 1e-6:
            continue
        ratios.append(min(me / oe, oe / me))
    return float(np.mean(ratios)) if ratios else 0.0


def _score_pose(
    T_final: np.ndarray,
    scaled: np.ndarray,
    obj_pts: np.ndarray,
    tgt: o3d.geometry.PointCloud,
    frames: List[CameraFrame],
    masks: List[np.ndarray],
    symmetry: str,
) -> Tuple[float, float, float, float, float, float, float]:
    aligned = transform_points(scaled, T_final)
    mod_max_world = float((aligned.max(0) - aligned.min(0)).max())
    try:
        eval_res = o3d.pipelines.registration.evaluate_registration(
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned)),
            tgt, max(mod_max_world * 0.06, 0.005),
        )
        fitness = float(eval_res.fitness)
        rmse = float(eval_res.inlier_rmse)
    except Exception:
        fitness, rmse = 0.0, 0.0
    ds = mv_depth_score(aligned, frames, masks)
    radius = max(mod_max_world * 0.05, 0.004)
    cov_fwd, cov_bwd = coverage_score_two_sided(aligned, obj_pts, radius)
    ss = silhouette_iou_score(aligned, frames, masks)
    em = _extent_match_score(aligned, obj_pts)
    # 가중치: extent_match를 강하게(0.22) 반영하여 올바른 크기의 모델이
    # 작은/큰 모델을 이기도록 함. cov_fwd는 작은 모델도 높게 나오므로 낮춤.
    if symmetry == "yaw":
        conf = (0.12 * ds + 0.10 * cov_fwd + 0.12 * cov_bwd
                + 0.26 * ss + 0.22 * em + 0.18 * fitness)
    else:
        conf = (0.10 * ds + 0.10 * cov_fwd + 0.12 * cov_bwd
                + 0.28 * ss + 0.22 * em + 0.18 * fitness)
    return float(conf), fitness, rmse, ds, float(cov_fwd), float(cov_bwd), ss


def register_direct(
    object_name: str,
    model: CanonicalModel,
    model_pts: np.ndarray,
    obj_pts: np.ndarray,
    frames: List[CameraFrame],
    masks: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    scale_override: Optional[float] = None,
) -> Optional[PoseEstimate]:
    """Vertical-axis constrained pose estimation.

    테이블 법선에 물체 "up"을 고정하고 yaw만 탐색. 부분 관측에서
    기울어진 ambiguous pose를 원천 차단.

    2-stage: coarse grid (yaw × scale × up) → top-K fine refine
    (yaw ±12°@2°, scale ±12%@3%).
    """
    symmetry = OBJECT_SYMMETRY.get(object_name, "none")
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    obj_center = obj_pts.mean(axis=0)
    mod_ext_unit = model_pts.max(0) - model_pts.min(0)
    mod_max_unit = float(mod_ext_unit.max())

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(obj_pts)
    # ICP correspondence용: target normals 미리 준비
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))

    yaw_steps = 1 if symmetry == "yaw" else 24  # 15° 간격 (속도 우선)
    inits = vertical_constrained_inits(
        table_n, model_pts, np.asarray(model.extents_m),
        obj_pts, yaw_steps=yaw_steps, symmetry=symmetry,
    )
    if not inits:
        return None

    # scale_override가 있으면 모든 init 후보의 auto_scale을 교체
    if scale_override is not None:
        inits = [(R, scale_override) for R, _ in inits]
        print(f"      [scale-override] auto_scale → {scale_override:.3f}")

    cyl_scale_lb = None
    scale_factors = [0.85, 0.92, 1.00, 1.08, 1.18]
    print(
        f"      [init] scale={inits[0][1]:.3f} rot_candidates={len(inits)} "
        f"scale_sweep={len(scale_factors)}"
    )

    stats = {"iter": 0, "rejected_center": 0, "reprojected": 0}
    # coarse: keep top-K for fine refinement
    candidates: List[Tuple[float, np.ndarray, float, np.ndarray]] = []
    TOP_K = 4

    def _eval_and_store(R_init: np.ndarray, scale: float):
        # Cylinder radius lower bound
        if cyl_scale_lb is not None and scale < cyl_scale_lb * 0.85:
            return
        scaled = model_pts * scale
        src_center = scaled.mean(0)
        stats["iter"] += 1
        T0 = np.eye(4)
        T0[:3, :3] = R_init
        T0[:3, 3] = obj_center - R_init @ src_center
        mod_max_world = mod_max_unit * scale
        T_final = _run_icp_stack(scaled, T0, tgt, mod_max_world)
        # vertical 재투영: reject 대신 강제 복구
        R_final = T_final[:3, :3]
        cos_to_up = float(np.max(np.abs(R_final.T @ target_up)))
        if cos_to_up < 0.985:  # 10° 이상 틀어지면 재투영 후 짧은 ICP 재정렬
            stats["reprojected"] += 1
            R_fix = reproject_to_vertical(R_final, target_up)
            T_final[:3, :3] = R_fix
            # 재투영으로 깨진 translation/잔여 yaw를 짧게 재정렬
            try:
                src2 = o3d.geometry.PointCloud()
                src2.points = o3d.utility.Vector3dVector(
                    transform_points(scaled, T_final)
                )
                res2 = o3d.pipelines.registration.registration_icp(
                    src2, tgt, max(mod_max_world * 0.05, 0.006), np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(1e-9, 1e-9, 40),
                )
                T_final = res2.transformation @ T_final
                # 다시 수직축 강제 (ICP가 다시 기울이지 않도록)
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
        # Hard extent filter: 모델/관측 3D bbox 축별 비율이 0.55 미만이면 기각.
        # 작은 모델이 관측 내부에 쏙 들어가는 편향을 원천 차단.
        aligned_pre = transform_points(scaled, T_final)
        em_pre = _extent_match_score(aligned_pre, obj_pts)
        if em_pre < 0.55:
            stats.setdefault("rejected_extent", 0)
            stats["rejected_extent"] += 1
            return
        conf, fitness, rmse, ds, cov_fwd, cov_bwd, ss = _score_pose(
            T_final, scaled, obj_pts, tgt, frames, masks, symmetry
        )
        candidates.append((conf, T_final, scale,
                           np.array([fitness, rmse, ds, cov_fwd, cov_bwd, ss])))

    # Coarse sweep
    for R_init, scale_auto in inits:
        for sf in scale_factors:
            _eval_and_store(R_init, scale_auto * sf)

    if not candidates:
        print(f"      [reg] iters={stats['iter']} all rejected")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:TOP_K]
    print(
        f"      [coarse] iters={stats['iter']} reproj={stats['reprojected']} "
        f"rej_ctr={stats['rejected_center']} top1_conf={top[0][0]:.3f}"
    )

    # Fine refinement: 각 top에서 yaw/scale 국소 그리드
    fine_candidates: List[Tuple[float, np.ndarray, float, np.ndarray]] = []
    fine_yaws = [0.0] if symmetry == "yaw" else list(np.arange(-12.0, 12.01, 2.0))
    fine_sfs = [0.94, 0.97, 1.00, 1.03, 1.06, 1.10]
    for _, T0_fine, scale0, _ in top:
        R0 = T0_fine[:3, :3]
        for dyaw in fine_yaws:
            R_yaw = Rot.from_rotvec(target_up * np.radians(dyaw)).as_matrix()
            R_try = R_yaw @ R0
            for sf in fine_sfs:
                scale = scale0 * sf
                if cyl_scale_lb is not None and scale < cyl_scale_lb * 0.85:
                    continue
                scaled = model_pts * scale
                src_center = scaled.mean(0)
                T0 = np.eye(4); T0[:3, :3] = R_try
                T0[:3, 3] = obj_center - R_try @ src_center
                mod_max_world = mod_max_unit * scale
                T_final = _run_icp_stack(scaled, T0, tgt, mod_max_world)
                R_final = T_final[:3, :3]
                cos_to_up = float(np.max(np.abs(R_final.T @ target_up)))
                if cos_to_up < 0.995:
                    R_fix = reproject_to_vertical(R_final, target_up)
                    T_final[:3, :3] = R_fix
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

    # orthonormalize
    Rf = T_best[:3, :3]
    U, _, Vt = np.linalg.svd(Rf)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    T_clean = T_best.copy(); T_clean[:3, :3] = R_clean
    rot = Rot.from_matrix(R_clean)
    best = PoseEstimate(
        T_base_obj=T_clean,
        position_m=T_clean[:3, 3].copy(),
        quaternion_xyzw=rot.as_quat(),
        euler_xyz_deg=rot.as_euler("xyz", degrees=True),
        scale=float(scale_best),
        confidence=float(conf_best),
        fitness=float(fitness),
        rmse=float(rmse),
        depth_score=float(ds),
        coverage=float(0.5 * (cov_fwd + cov_bwd)),
        silhouette_score=float(ss),
    )
    print(
        f"      [fine] n={len(fine_candidates)} best_conf={best.confidence:.3f} "
        f"scale={best.scale:.3f} ss={best.silhouette_score:.3f}"
    )
    return best


def refine_anisotropic_scale(
    T_base_obj: np.ndarray,
    scale_uniform: float,
    model_pts: np.ndarray,
    obj_pts: np.ndarray,
    symmetry: str = "none",
    coverage_ratio_min: float = 0.75,
) -> np.ndarray:
    """축별 독립 scale 추정 (보수적).

    기본은 uniform scale을 유지. 해당 축의 관측 커버리지가 매우 높을
    때만(≥ 0.75) obs 기반 scale로 교체. 이전 0.55 임계값은 pose 오차를
    anisotropic scale로 흡수해 '찌그러진 모델'을 만들었음.

    yaw-symmetric: 수직축이 모델 최장축(내부 함수 model_up_candidates 참조).
    수평 xy 두 축은 같은 반경이므로 두 축을 묶어 동일 scale 적용.
    """
    R = T_base_obj[:3, :3]
    t = T_base_obj[:3, 3]
    obs_aligned = (R.T @ (obj_pts - t).T).T
    mod_ext = model_pts.max(0) - model_pts.min(0)
    obs_ext = obs_aligned.max(0) - obs_aligned.min(0)
    posed_ext = mod_ext * scale_uniform
    coverage = obs_ext / (posed_ext + 1e-9)
    per_axis = np.full(3, scale_uniform, dtype=np.float64)

    if symmetry == "yaw":
        # 모델 최장축 = 수직축. 나머지 두 축은 radial → 동일 scale.
        major = int(np.argmax(mod_ext))
        minors = [i for i in range(3) if i != major]
        # radial scale: 두 minor 축 관측 중 더 작은 uncertainty(더 가까운 1.0)을
        # 사용하는 대신, 두 축 커버리지 가중 평균으로 안정화.
        w = np.array([max(coverage[minors[0]], 0.0),
                      max(coverage[minors[1]], 0.0)])
        if w.sum() > 1e-6 and w.max() >= coverage_ratio_min:
            r_obs = (w[0] * obs_ext[minors[0]] / (mod_ext[minors[0]] + 1e-9)
                     + w[1] * obs_ext[minors[1]] / (mod_ext[minors[1]] + 1e-9)
                     ) / w.sum()
            per_axis[minors[0]] = float(r_obs)
            per_axis[minors[1]] = float(r_obs)
        if coverage[major] >= coverage_ratio_min:
            per_axis[major] = float(obs_ext[major] / (mod_ext[major] + 1e-9))
    else:
        for i in range(3):
            if coverage[i] >= coverage_ratio_min:
                per_axis[i] = float(obs_ext[i] / (mod_ext[i] + 1e-9))

    # 안전망: uniform의 0.7~1.3배 범위 (pose가 정확할 때만 허용하는 소폭 보정)
    per_axis = np.clip(per_axis, scale_uniform * 0.7, scale_uniform * 1.3)
    return per_axis


def mask_refine_with_depth_cluster(
    frames: List[CameraFrame],
    masks: List[np.ndarray],
    table_info,
    dilate_px: int = 4,
    object_name: Optional[str] = None,
) -> List[np.ndarray]:
    """색상 마스크 → depth 클러스터 → 재투영으로 인접 오염 제거 + 빈 마스크 복원.

    1) 기존 색상 마스크로 3D 점 수집 (≥1대 카메라만 있어도 진행)
    2) 테이블 위 최대 클러스터만 유지
    3) 각 카메라에 재투영:
       - 원본 마스크가 충분(≥50px): AND로 오염 제거
       - 원본 마스크가 비어있음: 투영 영역을 색상 느슨한 검증 후 **복원**으로 채움
    """
    tn, td, tc, tr = table_info
    pts = fuse_object_points(frames, masks, tn, td, tc, tr)
    if len(pts) < 30:
        return masks
    import cv2
    refined: List[np.ndarray] = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate_px) | 1,) * 2)
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for cam, mask in zip(frames, masks):
        h, w = cam.intrinsics.height, cam.intrinsics.width
        pts_cam = (np.linalg.inv(cam.T_base_cam) @ np.hstack(
            [pts, np.ones((len(pts), 1))]
        ).T)[:3].T
        z = pts_cam[:, 2]
        ok = z > 0.05
        if not ok.any():
            refined.append(mask)
            continue
        K = cam.intrinsics.K
        u = np.round(K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]).astype(np.int32)
        v = np.round(K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]).astype(np.int32)
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        proj_mask = np.zeros((h, w), dtype=np.uint8)
        proj_mask[v[valid], u[valid]] = 255
        proj_mask = cv2.dilate(proj_mask, k, iterations=1)
        proj_mask = cv2.morphologyEx(proj_mask, cv2.MORPH_CLOSE, k)
        recovered, proj_dilated = recover_mask_from_projection(
            cam, proj_mask, object_name, morph_kernel=k, pad_kernel=k_big,
        )

        raw_area = mask_area(mask)
        proj_area = mask_area(proj_mask)
        rec_area = mask_area(recovered)
        img_cap = int(h * w * 0.12)

        if raw_area < 50:
            if 150 <= rec_area <= img_cap:
                refined.append(recovered)
            elif 120 <= proj_area <= img_cap:
                refined.append(proj_mask)
            else:
                refined.append(np.zeros_like(mask))
            continue

        if proj_area < 60:
            refined.append(mask)
            continue

        iou, precision, recall = mask_overlap_metrics(mask, proj_mask)
        base = cv2.bitwise_and(mask, proj_mask)
        base_area = mask_area(base)

        # 큰 raw mask가 projection과 거의 안 맞으면 raw를 버리고 교체한다.
        if (iou < 0.08 and precision < 0.18) or raw_area > proj_area * 3.0:
            if 120 <= rec_area <= img_cap:
                refined.append(recovered)
            elif proj_area <= img_cap:
                refined.append(proj_mask)
            else:
                refined.append(np.zeros_like(mask))
            continue

        # raw가 projection의 일부분만 덮는 경우는 projection 내부 relaxed-color로 확장.
        if recall < 0.45:
            if rec_area >= max(120, int(base_area * 1.15)) and rec_area <= img_cap:
                refined.append(recovered)
            elif base_area >= 80:
                refined.append(base)
            else:
                refined.append(proj_mask if proj_area <= img_cap else base)
            continue

        if base_area >= 80:
            refined.append(base)
        elif 120 <= rec_area <= img_cap:
            refined.append(recovered)
        else:
            refined.append(base)
    return refined


def coverage_score_simple(aligned: np.ndarray, obj_pts: np.ndarray, radius: float) -> float:
    if len(aligned) == 0 or len(obj_pts) == 0:
        return 0.0
    from scipy.spatial import cKDTree
    tree = cKDTree(obj_pts)
    if len(aligned) > 2000:
        idx = np.random.choice(len(aligned), 2000, replace=False)
        sample = aligned[idx]
    else:
        sample = aligned
    dist, _ = tree.query(sample, k=1, distance_upper_bound=radius)
    return float(np.sum(np.isfinite(dist))) / float(len(sample))


def coverage_score_two_sided(
    aligned: np.ndarray, obj_pts: np.ndarray, radius: float
) -> Tuple[float, float]:
    """양방향 coverage.

    forward: 모델→관측 (모델의 visible surface 중 관측점에 닿는 비율).
    backward: 관측→모델 (관측점 중 모델 근처인 비율; 부분 관측이라도 모든
    관측점이 모델 표면에 닿아야 함).
    부분 관측에서는 forward가 낮을 수 있으나 backward는 높아야 함 → yaw/roll
    모호성을 분해.
    """
    if len(aligned) == 0 or len(obj_pts) == 0:
        return 0.0, 0.0
    from scipy.spatial import cKDTree
    tree_obs = cKDTree(obj_pts)
    tree_mod = cKDTree(aligned)

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


def export_posed_anisotropic(
    pose: PoseEstimate,
    per_axis_scale: np.ndarray,
    model: CanonicalModel,
    glb_src_path: Path,
    frame_id: str,
    out_dir: Path,
) -> Dict[str, str]:
    """축별 scale 적용한 posed GLB export + 크기 재산출."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = per_axis_scale.tolist()
    R = pose.T_base_obj[:3, :3]
    t = pose.T_base_obj[:3, 3]
    # centroid 보정: uniform scale → anisotropic로 바꿀 때 centroid shift 상쇄
    mean_unit = np.zeros(3)  # sample_model_points는 이미 centroid 제거
    t_adj = t + R @ (mean_unit * (np.array([sx, sy, sz]) - pose.scale))

    paths: Dict[str, str] = {}
    for coord in ["opencv", "isaac"]:
        scene = trimesh.load(str(glb_src_path))
        mesh = (
            trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene)
            else scene.copy()
        )
        verts = (mesh.vertices - model.center) * np.array([sx, sy, sz])
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        T_out = np.eye(4); T_out[:3, :3] = R; T_out[:3, 3] = t_adj
        verts_pose = (T_out @ verts_h.T)[:3].T
        if coord == "isaac":
            verts_pose = (
                T_ISAAC_CV @ np.hstack(
                    [verts_pose, np.ones((len(verts_pose), 1))]
                ).T
            )[:3].T
        mesh.vertices = verts_pose
        suffix = "" if coord == "opencv" else "_isaac"
        gp = out_dir / f"{model.name}_posed{suffix}.glb"
        mesh.export(str(gp), file_type="glb")
        paths[f"posed_glb_{coord}"] = str(gp)
    return paths


def render_posed_mesh_mask(
    mesh: "trimesh.Trimesh",
    T_base_obj: np.ndarray,
    scale_per_axis: np.ndarray,
    model_center: np.ndarray,
    cam: CameraFrame,
    downscale: int = 1,
) -> np.ndarray:
    """포즈된 mesh의 face-fill 실루엣을 카메라 이미지에 rasterize.

    downscale>1 이면 해상도를 축소해서 rasterize 후 업샘플 (빠른 loss 평가용).
    """
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
    z = V_cam[:, 2]
    ok_v = z > 0.05
    K = cam.intrinsics.K
    u = (K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]) / downscale
    v = (K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]) / downscale
    mask = np.zeros((h, w), dtype=np.uint8)
    f_ok = ok_v[F[:, 0]] & ok_v[F[:, 1]] & ok_v[F[:, 2]]
    if f_ok.sum() == 0:
        return np.zeros((h_full, w_full), dtype=np.uint8)
    # 벡터화된 좌표 수집: (N, 3, 2)
    F_valid = F[f_ok]
    tri = np.stack([
        np.stack([u[F_valid[:, 0]], v[F_valid[:, 0]]], axis=-1),
        np.stack([u[F_valid[:, 1]], v[F_valid[:, 1]]], axis=-1),
        np.stack([u[F_valid[:, 2]], v[F_valid[:, 2]]], axis=-1),
    ], axis=1).astype(np.int32)
    lim = 4 * max(w, h)
    # reject triangles way outside frame
    inside = np.all(np.abs(tri).max(axis=(1, 2)) <= lim, axis=-1) if False else (
        np.abs(tri).reshape(len(tri), -1).max(axis=1) <= lim
    )
    tri = tri[inside]
    for face in tri:
        cv2.fillConvexPoly(mask, face, 255)
    if downscale > 1:
        mask = cv2.resize(mask, (w_full, h_full), interpolation=cv2.INTER_NEAREST)
    return mask


def mask_bbox_stats(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
    size = np.array([x1 - x0 + 1.0, y1 - y0 + 1.0], dtype=np.float64)
    return center, size, int(len(xs))


def evaluate_render_alignment(
    mesh: "trimesh.Trimesh",
    model_center: np.ndarray,
    T_base_obj: np.ndarray,
    scale: float,
    frames: List[CameraFrame],
    masks_obs: List[np.ndarray],
    downscale: int = 1,
) -> Dict[str, object]:
    per_cam = []
    score_terms = []
    scale_per_axis = np.full(3, scale, dtype=np.float64)
    for ci, (cam, obs_mask) in enumerate(zip(frames, masks_obs)):
        if mask_area(obs_mask) < 50:
            continue
        rnd = render_posed_mesh_mask(
            mesh, T_base_obj, scale_per_axis, model_center, cam, downscale=downscale
        )
        obs = obs_mask
        if downscale > 1:
            h, w = obs_mask.shape
            obs = cv2.resize(
                obs_mask, (w // downscale, h // downscale),
                interpolation=cv2.INTER_NEAREST,
            )
        iou, precision, recall = mask_overlap_metrics(rnd, obs)
        obs_stats = mask_bbox_stats(obs)
        rnd_stats = mask_bbox_stats(rnd)
        if obs_stats is None or rnd_stats is None:
            center_err = 1.0
            size_err = 1.0
        else:
            obs_center, obs_size, _ = obs_stats
            rnd_center, rnd_size, _ = rnd_stats
            diag = float(np.linalg.norm(obs_size) + 1e-6)
            center_err = float(np.linalg.norm(rnd_center - obs_center) / diag)
            size_err = 0.5 * (
                abs(np.log((rnd_size[0] + 1e-6) / (obs_size[0] + 1e-6)))
                + abs(np.log((rnd_size[1] + 1e-6) / (obs_size[1] + 1e-6)))
            )
        score = (
            1.00 * iou
            + 0.12 * precision
            + 0.08 * recall
            - 0.22 * center_err
            - 0.18 * size_err
        )
        per_cam.append({
            "cam_id": ci,
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "center_err": float(center_err),
            "size_err": float(size_err),
            "score": float(score),
        })
        score_terms.append(score)

    if not per_cam:
        return {
            "score": -1.0,
            "mean_iou": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_center_err": 1.0,
            "mean_size_err": 1.0,
            "valid_cams": [],
            "per_cam": [],
        }
    return {
        "score": float(np.mean(score_terms)),
        "mean_iou": float(np.mean([m["iou"] for m in per_cam])),
        "mean_precision": float(np.mean([m["precision"] for m in per_cam])),
        "mean_recall": float(np.mean([m["recall"] for m in per_cam])),
        "mean_center_err": float(np.mean([m["center_err"] for m in per_cam])),
        "mean_size_err": float(np.mean([m["size_err"] for m in per_cam])),
        "valid_cams": [int(m["cam_id"]) for m in per_cam],
        "per_cam": per_cam,
    }


def clone_pose_with_transform(
    pose: PoseEstimate,
    T_new: np.ndarray,
    scale: Optional[float] = None,
) -> PoseEstimate:
    rot_tmp = Rot.from_matrix(T_new[:3, :3])
    return PoseEstimate(
        T_base_obj=T_new,
        position_m=T_new[:3, 3].copy(),
        quaternion_xyzw=rot_tmp.as_quat(),
        euler_xyz_deg=rot_tmp.as_euler("xyz", degrees=True),
        scale=float(pose.scale if scale is None else scale),
        confidence=pose.confidence,
        fitness=pose.fitness,
        rmse=pose.rmse,
        depth_score=pose.depth_score,
        coverage=pose.coverage,
        silhouette_score=pose.silhouette_score,
        rgb_score=pose.rgb_score,
    )


def _simplify_mesh_for_render(mesh: "trimesh.Trimesh", target_faces: int = 600):
    """렌더 속도용 mesh 간소화. open3d quadric decimation."""
    try:
        if len(mesh.faces) <= target_faces:
            return mesh
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        m = m.simplify_quadric_decimation(target_faces)
        simplified = trimesh.Trimesh(
            vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles),
            process=False,
        )
        return simplified if len(simplified.faces) > 10 else mesh
    except Exception:
        return mesh


def depth_render_compare_refine(
    pose: PoseEstimate,
    model: CanonicalModel,
    frames: List[CameraFrame],
    masks_obs: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    max_iter: int = 200,
    n_pts: int = 5000,
    verbose: bool = True,
) -> PoseEstimate:
    """Depth 기반 render-and-compare 자동 정제.

    5000개 모델 표면 포인트를 현재 pose로 각 카메라에 투영 →
    관측 depth map과 비교. loss = mean(|z_model - z_obs|) / tol + (1-IoU).
    Silhouette(2D)와 달리 depth(3D)가 scale을 직접 제약하므로 모델이
    줄어드는 편향을 근본적으로 해결.
    """
    from scipy.optimize import minimize
    import time as _time

    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    h1_ax, h2_ax = horizontal_axes(target_up)
    model_pts = sample_model_points(model, n=n_pts)

    R0 = pose.T_base_obj[:3, :3].copy()
    t0 = pose.T_base_obj[:3, 3].copy()
    s0 = float(pose.scale)

    # 관측 depth (미터) + inverse extrinsics 미리 캐시
    obs_depths = []
    T_cam_bases = []
    valid_cams = []
    for ci, (cam, mask) in enumerate(zip(frames, masks_obs)):
        d = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
        obs_depths.append(d)
        T_cam_bases.append(np.linalg.inv(cam.T_base_cam))
        if int(np.count_nonzero(mask)) >= 100:
            valid_cams.append(ci)
    if not valid_cams:
        return pose

    depth_tol = 0.025  # 2.5cm 정규화 기준

    def _apply(params):
        dx, dy, dz, dyaw, dls = params
        scale = s0 * float(np.exp(dls))
        R_yaw = Rot.from_rotvec(target_up * dyaw).as_matrix()
        R = R_yaw @ R0
        t = t0 + np.array([dx, dy, dz])
        return R, t, scale

    step = {"n": 0, "best": 999.0, "t0": _time.time()}

    def _loss(params):
        R, t, scale = _apply(params)
        scaled = model_pts * scale
        pts_base = (R @ scaled.T).T + t

        depth_sum = 0.0
        sil_sum = 0.0
        n_cam = 0

        for ci in valid_cams:
            cam = frames[ci]
            mask = masks_obs[ci]
            depth_obs = obs_depths[ci]
            T_cb = T_cam_bases[ci]
            K = cam.intrinsics.K
            h, w = cam.intrinsics.height, cam.intrinsics.width

            pts_cam = (T_cb[:3, :3] @ pts_base.T).T + T_cb[:3, 3]
            z = pts_cam[:, 2]
            ok = z > 0.05
            if ok.sum() < 50:
                continue

            u = (K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]).astype(np.int32)
            v = (K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]).astype(np.int32)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if inside.sum() < 30:
                continue

            u_v, v_v = u[inside], v[inside]
            z_model = z[ok][inside]
            z_obs = depth_obs[v_v, u_v]
            in_mask = mask[v_v, u_v] > 0
            z_ok = (z_obs > 0.05) & in_mask
            if z_ok.sum() < 10:
                continue

            # Depth loss: mean abs diff 정규화
            depth_err = float(np.mean(np.abs(z_model[z_ok] - z_obs[z_ok])))
            depth_sum += depth_err / depth_tol

            # Silhouette IoU (point-based, 빠름)
            proj_mask = np.zeros((h, w), dtype=bool)
            proj_mask[v_v, u_v] = True
            obs_bool = mask > 0
            inter = np.logical_and(proj_mask, obs_bool).sum()
            union = np.logical_or(proj_mask, obs_bool).sum()
            iou = inter / max(union, 1)
            sil_sum += (1.0 - iou)
            n_cam += 1

        if n_cam == 0:
            return 10.0
        # Depth 60% + Silhouette 40% (depth가 주 신호)
        L = (0.6 * depth_sum + 0.4 * sil_sum) / n_cam
        step["n"] += 1
        if L < step["best"] - 1e-4:
            step["best"] = L
            if verbose:
                dt = _time.time() - step["t0"]
                ds = depth_sum / n_cam * depth_tol * 1000  # mm 단위
                si = 1.0 - sil_sum / n_cam
                print(f"        [depth-iter {step['n']:3d}] loss={L:.4f} "
                      f"depth_err={ds:.1f}mm IoU={si:.3f} "
                      f"dx={params[0]*100:+.1f}cm dy={params[1]*100:+.1f}cm "
                      f"dz={params[2]*100:+.1f}cm dyaw={np.degrees(params[3]):+.1f}° "
                      f"scale×{np.exp(params[4]):.3f}  t={dt:.1f}s", flush=True)
        return L

    loss0 = _loss(np.zeros(5))
    init_simplex = np.zeros((6, 5))
    init_simplex[1][0] = 0.030
    init_simplex[2][1] = 0.030
    init_simplex[3][2] = 0.030
    init_simplex[4][3] = np.radians(10.0)
    init_simplex[5][4] = np.log(1.25)
    try:
        res = minimize(
            _loss, np.zeros(5), method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-5,
                     "initial_simplex": init_simplex, "disp": False},
        )
    except Exception as exc:
        if verbose:
            print(f"      [depth-refine] 실패 ({exc})")
        return pose

    if res.fun >= loss0 - 1e-4:
        if verbose:
            print(f"      [depth-refine] no improvement")
        return pose

    R_new, t_new, scale_new = _apply(res.x)
    T_new = np.eye(4); T_new[:3, :3] = R_new; T_new[:3, 3] = t_new
    # 수직 재투영
    if float(np.max(np.abs(R_new.T @ target_up))) < 0.999:
        T_new[:3, :3] = reproject_to_vertical(R_new, target_up)
    scaled = model_pts * scale_new
    T_new = snap_to_table(T_new, scaled, table_n, table_d, 0.0005)

    # IoU 계산 (silhouette 기반, 보고용)
    final_iou = 0.0
    n_c = 0
    for ci in valid_cams:
        cam = frames[ci]
        mask = masks_obs[ci]
        pts_base = (T_new[:3, :3] @ scaled.T).T + T_new[:3, 3]
        T_cb = T_cam_bases[ci]
        K = cam.intrinsics.K
        h, w = cam.intrinsics.height, cam.intrinsics.width
        pts_cam = (T_cb[:3, :3] @ pts_base.T).T + T_cb[:3, 3]
        z = pts_cam[:, 2]; ok = z > 0.05
        if ok.sum() < 30: continue
        u = (K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]).astype(np.int32)
        v = (K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        proj = np.zeros((h, w), dtype=bool)
        proj[v[inside], u[inside]] = True
        obs_b = mask > 0
        inter = np.logical_and(proj, obs_b).sum()
        union = np.logical_or(proj, obs_b).sum()
        final_iou += inter / max(union, 1)
        n_c += 1
    final_iou = final_iou / max(n_c, 1)

    rot = Rot.from_matrix(T_new[:3, :3])
    new_pose = PoseEstimate(
        T_base_obj=T_new,
        position_m=T_new[:3, 3].copy(),
        quaternion_xyzw=rot.as_quat(),
        euler_xyz_deg=rot.as_euler("xyz", degrees=True),
        scale=float(scale_new),
        confidence=float(final_iou),
        fitness=pose.fitness,
        rmse=float(res.fun),
        depth_score=float(1.0 - res.fun),
        coverage=pose.coverage,
        silhouette_score=float(final_iou),
    )
    if verbose:
        print(f"      [depth-refine] loss {loss0:.4f}→{res.fun:.4f} "
              f"IoU={final_iou:.3f} iters={res.nit} "
              f"scale {s0:.3f}→{scale_new:.3f}")
    return new_pose


def depth_refine_with_flips(
    pose: PoseEstimate,
    model: CanonicalModel,
    frames: List[CameraFrame],
    masks_obs: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    max_iter: int = 200,
) -> PoseEstimate:
    """Flip 후보 포함 depth-render-compare refinement."""
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    h1, h2 = horizontal_axes(target_up)
    candidates: List[Tuple[str, PoseEstimate]] = [("original", pose)]
    for axis, tag in [(h1, "flip_h1"), (h2, "flip_h2")]:
        R_flip = Rot.from_rotvec(axis * np.pi).as_matrix()
        T_new = pose.T_base_obj.copy()
        T_new[:3, :3] = R_flip @ pose.T_base_obj[:3, :3]
        model_pts_tmp = sample_model_points(model, n=2000)
        scaled_tmp = model_pts_tmp * pose.scale
        T_new = snap_to_table(T_new, scaled_tmp, table_n, table_d, 0.0005)
        rot_tmp = Rot.from_matrix(T_new[:3, :3])
        flipped = PoseEstimate(
            T_base_obj=T_new, position_m=T_new[:3, 3].copy(),
            quaternion_xyzw=rot_tmp.as_quat(),
            euler_xyz_deg=rot_tmp.as_euler("xyz", degrees=True),
            scale=pose.scale, confidence=0, fitness=0, rmse=0,
            depth_score=0, coverage=0, silhouette_score=0,
        )
        candidates.append((tag, flipped))
    results = []
    for tag, cand in candidates:
        print(f"      [depth-flip] refining {tag}")
        try:
            refined = depth_render_compare_refine(
                cand, model, frames, masks_obs, table_n, table_d,
                max_iter=max_iter,
            )
        except Exception as exc:
            print(f"      [depth-flip] {tag} 실패 ({exc})")
            refined = cand
        results.append((tag, refined))
    results.sort(key=lambda x: x[1].silhouette_score, reverse=True)
    best_tag, best = results[0]
    print(f"      [depth-flip] best={best_tag} IoU={best.silhouette_score:.3f}")
    return best


def _compute_target_scale_simple(
    model: CanonicalModel,
    glb_src_path: Path,
    frames: List[CameraFrame],
    object_name: str,
    seed_masks: Optional[List[np.ndarray]] = None,
) -> Optional[float]:
    """HSV 마스크 bbox의 **실세계 크기**에서 target scale 계산 (pose 불필요).

    각 카메라에서 HSV 마스크의 bbox를 depth 기반으로 실세계 미터로 변환 후,
    GLB 모델의 unscaled extent와 비교하여 scale 추정.
    """
    scene = trimesh.load(str(glb_src_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene.copy())
    verts = np.asarray(mesh.vertices) - model.center
    mod_ext = verts.max(0) - verts.min(0)  # unscaled model extent (3D)
    mod_max = float(mod_ext.max())

    targets = []
    for ci, cam in enumerate(frames):
        if seed_masks is not None and ci < len(seed_masks) and mask_area(seed_masks[ci]) >= 80:
            obs_mask = seed_masks[ci]
        else:
            obs_mask = best_connected_component(color_mask_for_object(cam.color_bgr, object_name))
        obs_ys, obs_xs = np.where(obs_mask > 0)
        if len(obs_xs) < 100:
            continue
        cx, cy = int(obs_xs.mean()), int(obs_ys.mean())
        depth = cam.depth_u16.astype(float) * cam.intrinsics.depth_scale
        # bbox 근처 depth 중앙값
        y1, y2 = max(0, cy - 20), min(cam.intrinsics.height, cy + 20)
        x1, x2 = max(0, cx - 20), min(cam.intrinsics.width, cx + 20)
        d_patch = depth[y1:y2, x1:x2]
        d_valid = d_patch[(d_patch > 0.05) & (d_patch < 1.5)]
        if len(d_valid) < 10:
            continue
        z_m = float(np.median(d_valid))
        # bbox pixel 크기 → 실세계 미터
        ow_px = float(obs_xs.max() - obs_xs.min())
        oh_px = float(obs_ys.max() - obs_ys.min())
        K = cam.intrinsics.K
        ow_m = ow_px * z_m / K[0, 0]
        oh_m = oh_px * z_m / K[1, 1]
        obs_max_m = max(ow_m, oh_m)
        if mod_max > 1e-6:
            s = obs_max_m / mod_max
            targets.append(float(s))
    if len(targets) < 1:
        return None
    return float(np.median(targets))


def _compute_target_scale(
    pose: PoseEstimate,
    model: CanonicalModel,
    glb_src_path: Path,
    frames: List[CameraFrame],
    masks_obs: List[np.ndarray],
    object_name: Optional[str] = None,
) -> Optional[float]:
    """HSV 색상 마스크의 bbox 크기에서 목표 scale을 역산.

    depth-seg 마스크는 물체 상단만 캡처해 bbox가 작으므로, 색상 마스크
    (실제 물체 전체 영역)를 기준으로 scale을 계산.
    """
    scene = trimesh.load(str(glb_src_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene.copy())
    cur_scale = pose.scale
    targets = []
    for ci, cam in enumerate(frames):
        if ci < len(masks_obs) and mask_area(masks_obs[ci]) >= 80:
            obs_mask = masks_obs[ci]
        elif object_name is not None and object_name in COLOR_REF_HSV:
            obs_mask = best_connected_component(color_mask_for_object(cam.color_bgr, object_name))
        else:
            obs_mask = masks_obs[ci]
        obs_ys, obs_xs = np.where(obs_mask > 0)
        if len(obs_xs) < 50:
            print(f"        [bbox-scale] cam{ci}: color mask too small ({len(obs_xs)}px)")
            continue
        ow = obs_xs.max() - obs_xs.min()
        oh = obs_ys.max() - obs_ys.min()
        if ow < 10 or oh < 10:
            continue
        rnd = render_posed_mesh_mask(
            mesh, pose.T_base_obj, np.full(3, cur_scale),
            model.center, cam,
        )
        rnd_ys, rnd_xs = np.where(rnd > 0)
        if len(rnd_xs) < 10:
            continue
        rw = rnd_xs.max() - rnd_xs.min()
        rh = rnd_ys.max() - rnd_ys.min()
        if rw < 5 or rh < 5:
            continue
        sw = cur_scale * (ow / rw)
        sh = cur_scale * (oh / rh)
        targets.append((sw + sh) / 2.0)
    print(f"        [bbox-scale] targets={[f'{t:.3f}' for t in targets]} "
          f"n_valid={len(targets)}")
    if len(targets) < 1:
        return None
    return float(np.median(targets))


def render_and_compare_refine(
    pose: PoseEstimate,
    model: CanonicalModel,
    glb_src_path: Path,
    frames: List[CameraFrame],
    masks_obs: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    max_iter: int = 150,
    verbose: bool = True,
    downscale: int = 1,          # 원본 해상도
    simplify_faces: int = 2000,   # render-refine 속도↑ (500K→2K faces, 정확도 유지)
    try_flips: bool = False,
    pitch_roll_dof: bool = False,
    min_scale: Optional[float] = None,
) -> PoseEstimate:
    """Analysis-by-synthesis 자동 정제.

    loss = 1 - mean_IoU(rendered_silhouette, observed_mask).
    파라미터 [dx, dy, dz, dyaw, dscale_log] 5D delta. 원본 해상도 + 전체 mesh.
    """
    from scipy.optimize import minimize
    import time as _time

    scene = trimesh.load(str(glb_src_path))
    mesh_full = (trimesh.util.concatenate(list(scene.geometry.values()))
                 if isinstance(scene, trimesh.Scene) else scene.copy())
    mesh = (_simplify_mesh_for_render(mesh_full, target_faces=simplify_faces)
            if simplify_faces > 0 else mesh_full)
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)

    obs_small = []
    for m in masks_obs:
        if downscale > 1:
            h, w = m.shape
            m_small = cv2.resize(
                m, (w // downscale, h // downscale),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            m_small = m
        obs_small.append(m_small > 0)
    obs_areas = [int(b.sum()) for b in obs_small]
    valid_cams = [ci for ci, a in enumerate(obs_areas) if a >= 50]
    if len(valid_cams) == 0:
        return pose
    if verbose:
        print(f"      [render-refine] start: faces={len(mesh.faces)} "
              f"valid_cams={valid_cams} obs_areas={obs_areas}")

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
        scale = s0 * float(np.exp(np.clip(dls, np.log(0.90), np.log(1.10))))
        if min_scale is not None and scale < min_scale:
            scale = min_scale
        R_yaw = Rot.from_rotvec(target_up * dyaw).as_matrix()
        R = R_yaw @ R0
        if pitch_roll_dof and (abs(dpitch) > 0 or abs(droll) > 0):
            R_pitch = Rot.from_rotvec(h1_ax * dpitch).as_matrix()
            R_roll = Rot.from_rotvec(h2_ax * droll).as_matrix()
            R = R_roll @ R_pitch @ R
        t = t0 + np.array([dx, dy, dz])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T, scale

    def _render_small(T, scale, ci):
        cam = frames[ci]
        scale_per_axis = np.full(3, scale, dtype=np.float64)
        # render_posed_mesh_mask의 downscale 경로 활용 + 업샘플 생략
        h_full, w_full = cam.intrinsics.height, cam.intrinsics.width
        h, w = h_full // downscale, w_full // downscale
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int32)
        V_obj = (V - model.center) * scale_per_axis
        Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
        V_base = (T @ Vh.T)[:3].T
        T_cam_base = np.linalg.inv(cam.T_base_cam)
        V_cam = (T_cam_base @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
        z = V_cam[:, 2]
        ok_v = z > 0.05
        K = cam.intrinsics.K
        u = (K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]) / downscale
        v = (K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]) / downscale
        mask = np.zeros((h, w), dtype=np.uint8)
        f_ok = ok_v[F[:, 0]] & ok_v[F[:, 1]] & ok_v[F[:, 2]]
        if f_ok.sum() == 0:
            return mask > 0
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
        return mask > 0

    step = {"n": 0, "best": 1.0, "t0": _time.time()}

    def _loss(params):
        T, scale = _apply(params)
        loss_sum = 0.0
        for ci in valid_cams:
            rnd = _render_small(T, scale, ci)
            obs = obs_small[ci]
            iou, precision, recall = mask_overlap_metrics(
                rnd.astype(np.uint8) * 255, obs.astype(np.uint8) * 255
            )
            obs_stats = mask_bbox_stats(obs.astype(np.uint8) * 255)
            rnd_stats = mask_bbox_stats(rnd.astype(np.uint8) * 255)
            if obs_stats is None or rnd_stats is None:
                center_err = 1.0
                size_err = 1.0
            else:
                obs_center, obs_size, _ = obs_stats
                rnd_center, rnd_size, _ = rnd_stats
                diag = float(np.linalg.norm(obs_size) + 1e-6)
                center_err = float(np.linalg.norm(rnd_center - obs_center) / diag)
                size_err = 0.5 * (
                    abs(np.log((rnd_size[0] + 1e-6) / (obs_size[0] + 1e-6)))
                    + abs(np.log((rnd_size[1] + 1e-6) / (obs_size[1] + 1e-6)))
                )
            cam_loss = (
                (1.0 - iou)
                + 0.18 * (1.0 - precision)
                + 0.10 * (1.0 - recall)
                + 0.30 * center_err
                + 0.22 * size_err
            )
            loss_sum += cam_loss
        L = loss_sum / len(valid_cams)
        step["n"] += 1
        if L < step["best"] - 1e-4:
            step["best"] = L
            if verbose:
                dt = _time.time() - step["t0"]
                if pitch_roll_dof:
                    extra = (f" pitch={np.degrees(params[4]):+.1f}° "
                             f"roll={np.degrees(params[5]):+.1f}°")
                    scale_idx = 6
                else:
                    extra = ""
                    scale_idx = 4
                score_preview = evaluate_render_alignment(
                    mesh, model.center, T, scale, frames, masks_obs, downscale=downscale
                )
                print(f"        [iter {step['n']:3d}] "
                      f"IoU={score_preview['mean_iou']:.3f} "
                      f"bbox={score_preview['mean_center_err']:.3f}/"
                      f"{score_preview['mean_size_err']:.3f} "
                      f"dx={params[0]*100:+.1f}cm dy={params[1]*100:+.1f}cm "
                      f"dz={params[2]*100:+.1f}cm dyaw={np.degrees(params[3]):+.1f}°"
                      f"{extra} scale×{np.exp(params[scale_idx]):.3f}  "
                      f"t={dt:.0f}s", flush=True)
        return L

    n_dof = 7 if pitch_roll_dof else 5
    loss0 = _loss(np.zeros(n_dof))
    init_simplex = np.zeros((n_dof + 1, n_dof))
    # 공통 5 DOF: 위치 3cm, yaw 10°, scale ±25%
    init_simplex[1][0] = 0.030        # dx
    init_simplex[2][1] = 0.030        # dy
    init_simplex[3][2] = 0.030        # dz
    init_simplex[4][3] = np.radians(10.0)   # dyaw
    if pitch_roll_dof:
        init_simplex[5][4] = np.radians(6.0)   # dpitch
        init_simplex[6][5] = np.radians(6.0)   # droll
        init_simplex[7][6] = np.log(1.05)      # dls (bbox-scale 보정 후 tight)
    else:
        init_simplex[5][4] = np.log(1.05)      # dls (bbox-scale 보정 후 tight)
    try:
        res = minimize(
            _loss, np.zeros(n_dof), method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4,
                     "initial_simplex": init_simplex, "disp": False},
        )
    except Exception as exc:
        if verbose:
            print(f"      [render-refine] 실패 ({exc}) → 유지")
        return pose
    if res.fun >= loss0 - 1e-4:
        if verbose:
            print(f"      [render-refine] no improvement (loss {loss0:.3f}→{res.fun:.3f})")
        return pose
    T_new, scale_new = _apply(res.x)
    # 테이블 스냅 + 수직 재투영
    R_new = T_new[:3, :3]
    cos_up = float(np.max(np.abs(R_new.T @ target_up)))
    if cos_up < 0.999:
        T_new[:3, :3] = reproject_to_vertical(R_new, target_up)
    model_pts = sample_model_points(model, n=2000)
    scaled = model_pts * scale_new
    T_new = snap_to_table(T_new, scaled, table_n, table_d, margin_m=0.0005)
    rot = Rot.from_matrix(T_new[:3, :3])
    final_metrics = evaluate_render_alignment(
        mesh, model.center, T_new, scale_new, frames, masks_obs, downscale=1
    )
    new_pose = PoseEstimate(
        T_base_obj=T_new,
        position_m=T_new[:3, 3].copy(),
        quaternion_xyzw=rot.as_quat(),
        euler_xyz_deg=rot.as_euler("xyz", degrees=True),
        scale=float(scale_new),
        confidence=float(max(pose.confidence, final_metrics["score"])),
        fitness=pose.fitness,
        rmse=pose.rmse,
        depth_score=pose.depth_score,
        coverage=pose.coverage,
        silhouette_score=float(final_metrics["mean_iou"]),
        rgb_score=float(final_metrics["score"]),
    )
    if verbose:
        print(f"      [render-refine] mean_IoU {pose.silhouette_score:.3f}→"
              f"{final_metrics['mean_iou']:.3f} "
              f"score {pose.rgb_score:.3f}→{final_metrics['score']:.3f} "
              f"bbox={final_metrics['mean_center_err']:.3f}/"
              f"{final_metrics['mean_size_err']:.3f} "
              f"iters={res.nit} scale {s0:.3f}→{scale_new:.3f}")
    return new_pose


def render_and_compare_with_flips(
    pose: PoseEstimate,
    model: CanonicalModel,
    glb_src_path: Path,
    frames: List[CameraFrame],
    masks_obs: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    max_iter: int = 150,
    multi_yaw: bool = False,
    pitch_roll_dof: bool = False,
    include_axis_turns: bool = False,
    shortlist_k: int = 6,
) -> PoseEstimate:
    """Flip + (옵션) multi-initial-yaw 포함 render-refine.

    원본 + h1/h2 flip 3후보 × (multi_yaw=True면 yaw 0°/90°/180°/270° 4초기)
    = 최대 12 Nelder-Mead 실행. 최고 IoU 채택.
    """
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    h1, h2 = horizontal_axes(target_up)
    scene = trimesh.load(str(glb_src_path))
    mesh_full = (
        trimesh.util.concatenate(list(scene.geometry.values()))
        if isinstance(scene, trimesh.Scene) else scene.copy()
    )
    # flip-search scoring용 메시 간소화 (2000 faces): 48 후보 scoring이
    # 500K faces 메시로는 매우 느림. 실루엣은 2K faces로도 정확.
    mesh = _simplify_mesh_for_render(mesh_full, target_faces=2000)
    model_pts_tmp = sample_model_points(model, n=2000)
    scaled_tmp = model_pts_tmp * pose.scale

    def _make_candidate(tag: str, R_delta: np.ndarray, base_pose: PoseEstimate) -> Tuple[str, PoseEstimate]:
        T_new = base_pose.T_base_obj.copy()
        T_new[:3, :3] = R_delta @ base_pose.T_base_obj[:3, :3]
        T_new = snap_to_table(T_new, scaled_tmp, table_n, table_d, 0.0005)
        return tag, clone_pose_with_transform(base_pose, T_new)

    base_candidates: List[Tuple[str, PoseEstimate]] = [("original", pose)]
    for axis, tag in [(h1, "flip_h1"), (h2, "flip_h2")]:
        R_flip = Rot.from_rotvec(axis * np.pi).as_matrix()
        base_candidates.append(_make_candidate(tag, R_flip, pose))

    yaw_offsets = [0.0, 90.0, 180.0, 270.0] if multi_yaw else [0.0]

    if include_axis_turns:
        local_axes = [
            pose.T_base_obj[:3, 0],
            pose.T_base_obj[:3, 1],
            pose.T_base_obj[:3, 2],
        ]
        for ai, axis in enumerate(local_axes):
            axis_n = axis / (np.linalg.norm(axis) + 1e-12)
            for ang in (-90.0, 90.0, 180.0):
                R_delta = Rot.from_rotvec(axis_n * np.radians(ang)).as_matrix()
                base_candidates.append(_make_candidate(
                    f"axis{ai}_{int(ang):+d}", R_delta, pose
                ))

    all_candidates: List[Tuple[str, PoseEstimate]] = []
    for tag, p in base_candidates:
        for yaw_deg in yaw_offsets:
            if yaw_deg == 0.0:
                all_candidates.append((tag, p))
                continue
            R_yaw = Rot.from_rotvec(target_up * np.radians(yaw_deg)).as_matrix()
            all_candidates.append(_make_candidate(f"{tag}_yaw{int(yaw_deg)}", R_yaw, p))

    print(f"      [flip-search] candidates: {len(all_candidates)} "
          f"(base×{len(base_candidates)}, yaw×{len(yaw_offsets)})")

    scored_candidates = []
    for tag, cand in all_candidates:
        metrics = evaluate_render_alignment(
            mesh, model.center, cand.T_base_obj, cand.scale, frames, masks_obs, downscale=1
        )
        scored_candidates.append((float(metrics["score"]), float(metrics["mean_iou"]), tag, cand))
    scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    keep_n = max(1, min(shortlist_k, len(scored_candidates)))
    shortlist = scored_candidates[:keep_n]
    preview = ", ".join(f"{tag}={score:.3f}/{iou:.3f}" for score, iou, tag, _ in shortlist[:5])
    print(f"      [flip-search] shortlist={keep_n} top={preview}")

    results: List[Tuple[str, PoseEstimate, float]] = []
    for i, (_, _, tag, cand) in enumerate(shortlist):
        print(f"      [flip-search] refining {i+1}/{len(shortlist)} ({tag})")
        try:
            refined = render_and_compare_refine(
                cand, model, glb_src_path, frames, masks_obs,
                table_n, table_d, max_iter=max_iter, verbose=True,
                pitch_roll_dof=pitch_roll_dof,
            )
        except Exception as exc:
            print(f"      [flip-search] {tag} 실패 ({exc})")
            refined = cand
        final_metrics = evaluate_render_alignment(
            mesh, model.center, refined.T_base_obj, refined.scale, frames, masks_obs, downscale=1
        )
        refined.rgb_score = float(final_metrics["score"])
        refined.silhouette_score = float(final_metrics["mean_iou"])
        results.append((tag, refined, float(final_metrics["score"])))

    results.sort(key=lambda x: (x[2], x[1].silhouette_score), reverse=True)
    best_tag, best, best_score = results[0]
    summary = ", ".join(
        f"{t}={s:.3f}/{p.silhouette_score:.3f}" for t, p, s in results[:5]
    )
    print(f"      [flip-search] best={best_tag} score={best_score:.3f} "
          f"IoU={best.silhouette_score:.3f} "
          f"(top5: {summary})")
    return best


def auto_refine_pose_until_converged(
    pose: PoseEstimate,
    model: CanonicalModel,
    glb_src_path: Path,
    frames: List[CameraFrame],
    masks_obs: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    multi_yaw: bool = True,
    pitch_roll_dof: bool = True,
    max_rounds: int = 6,
    target_iou: float = 0.86,
    target_center_err: float = 0.055,
    target_size_err: float = 0.085,
) -> Tuple[PoseEstimate, List[np.ndarray]]:
    scene = trimesh.load(str(glb_src_path))
    mesh_full = (
        trimesh.util.concatenate(list(scene.geometry.values()))
        if isinstance(scene, trimesh.Scene) else scene.copy()
    )
    # auto_converge 루프의 모든 scoring/eval에 간소화 mesh 사용 (속도 ~100x)
    mesh = _simplify_mesh_for_render(mesh_full, target_faces=2000)
    cur_pose = pose
    cur_masks = [m.copy() for m in masks_obs]
    cur_metrics = evaluate_render_alignment(
        mesh, model.center, cur_pose.T_base_obj, cur_pose.scale, frames, cur_masks, downscale=1
    )
    cur_pose.rgb_score = float(cur_metrics["score"])
    cur_pose.silhouette_score = float(cur_metrics["mean_iou"])
    print(
        f"      [auto-converge] start score={cur_metrics['score']:.3f} "
        f"IoU={cur_metrics['mean_iou']:.3f} "
        f"bbox={cur_metrics['mean_center_err']:.3f}/{cur_metrics['mean_size_err']:.3f}"
    )

    for round_i in range(max_rounds):
        if (cur_metrics["mean_iou"] >= target_iou
                and cur_metrics["mean_center_err"] <= target_center_err
                and cur_metrics["mean_size_err"] <= target_size_err):
            print(
                f"      [auto-converge] target reached at round {round_i} "
                f"(IoU={cur_metrics['mean_iou']:.3f}, "
                f"bbox={cur_metrics['mean_center_err']:.3f}/"
                f"{cur_metrics['mean_size_err']:.3f})"
            )
            break

        print(f"      [auto-converge] round {round_i + 1}/{max_rounds}")
        best_pose = cur_pose
        best_masks = cur_masks
        best_metrics = cur_metrics

        try:
            cand_pose = render_and_compare_with_flips(
                cur_pose, model, glb_src_path, frames, cur_masks,
                table_n, table_d, max_iter=120,
                multi_yaw=multi_yaw, pitch_roll_dof=pitch_roll_dof,
                include_axis_turns=True, shortlist_k=8,
            )
            cand_metrics = evaluate_render_alignment(
                mesh, model.center, cand_pose.T_base_obj, cand_pose.scale,
                frames, cur_masks, downscale=1
            )
            cand_pose.rgb_score = float(cand_metrics["score"])
            cand_pose.silhouette_score = float(cand_metrics["mean_iou"])
            if cand_metrics["score"] > best_metrics["score"] + 1e-3:
                best_pose = cand_pose
                best_metrics = cand_metrics
        except Exception as exc:
            print(f"      [auto-converge] candidate refine 실패 ({exc})")

        try:
            masks_new = refine_masks_with_pose(
                frames, cur_masks, best_pose, model, glb_src_path, dilate_px=6
            )
            cand_pose2 = render_and_compare_with_flips(
                best_pose, model, glb_src_path, frames, masks_new,
                table_n, table_d, max_iter=100,
                multi_yaw=multi_yaw, pitch_roll_dof=pitch_roll_dof,
                include_axis_turns=True, shortlist_k=6,
            )
            cand_metrics2 = evaluate_render_alignment(
                mesh, model.center, cand_pose2.T_base_obj, cand_pose2.scale,
                frames, masks_new, downscale=1
            )
            cand_pose2.rgb_score = float(cand_metrics2["score"])
            cand_pose2.silhouette_score = float(cand_metrics2["mean_iou"])
            if cand_metrics2["score"] > best_metrics["score"] + 1e-3:
                best_pose = cand_pose2
                best_masks = masks_new
                best_metrics = cand_metrics2
        except Exception as exc:
            print(f"      [auto-converge] mask-guided refine 실패 ({exc})")

        if best_metrics["score"] <= cur_metrics["score"] + 1e-3:
            print(
                f"      [auto-converge] stop: no improvement "
                f"({cur_metrics['score']:.3f}→{best_metrics['score']:.3f})"
            )
            break

        print(
            f"      [auto-converge] accept "
            f"score {cur_metrics['score']:.3f}→{best_metrics['score']:.3f} "
            f"IoU {cur_metrics['mean_iou']:.3f}→{best_metrics['mean_iou']:.3f} "
            f"bbox {cur_metrics['mean_center_err']:.3f}/"
            f"{cur_metrics['mean_size_err']:.3f}→"
            f"{best_metrics['mean_center_err']:.3f}/"
            f"{best_metrics['mean_size_err']:.3f}"
        )
        cur_pose = best_pose
        cur_masks = best_masks
        cur_metrics = best_metrics

    return cur_pose, cur_masks


def refine_masks_with_pose(
    frames: List[CameraFrame],
    masks_color: List[np.ndarray],
    pose: PoseEstimate,
    model: CanonicalModel,
    glb_src_path: Path,
    dilate_px: int = 6,
) -> List[np.ndarray]:
    """현재 pose로 mesh silhouette을 투영 → 색상 마스크와 교집합으로 정제.

    색상 마스크에 포함된 배경/잡음(다른 동색 물체)을 pose 기반으로 걸러내고,
    색상 마스크가 비어있던 카메라에선 silhouette만으로 마스크를 재구성.
    """
    scene = trimesh.load(str(glb_src_path))
    mesh_full = (
        trimesh.util.concatenate(list(scene.geometry.values()))
        if isinstance(scene, trimesh.Scene) else scene.copy()
    )
    mesh = _simplify_mesh_for_render(mesh_full, target_faces=2000)
    scale_per_axis = np.full(3, pose.scale, dtype=np.float64)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, dilate_px) | 1,) * 2)
    refined: List[np.ndarray] = []
    for cam, mcol in zip(frames, masks_color):
        silh = render_posed_mesh_mask(
            mesh, pose.T_base_obj, scale_per_axis, model.center, cam,
        )
        silh_d = cv2.dilate(silh, k, iterations=1)
        silh_d = cv2.morphologyEx(silh_d, cv2.MORPH_CLOSE, k)
        if int(np.count_nonzero(mcol)) >= 100:
            iou, precision, recall = mask_overlap_metrics(mcol, silh_d)
            base = cv2.bitwise_and(mcol, silh_d)
            base_area = mask_area(base)
            silh_area = mask_area(silh_d)
            raw_area = mask_area(mcol)
            if (iou < 0.08 and precision < 0.18) or raw_area > silh_area * 3.0:
                refined.append(silh_d)
            elif recall < 0.45 and silh_area >= max(120, int(raw_area * 1.2)):
                refined.append(silh_d)
            elif base_area >= 80:
                refined.append(base)
            else:
                refined.append(silh_d)
        else:
            # 색상 마스크가 거의 없으면 silhouette 단독 사용
            refined.append(silh_d)
    return refined


def estimate_one(
    object_name: str,
    frames: List[CameraFrame],
    model: CanonicalModel,
    glb_src_path: Path,
    table_info,
    out_dir: Path,
    frame_id: str,
    pose_refine_iters: int = 0,
    render_refine: bool = False,
    multi_yaw: bool = False,
    mask_iter_loop: bool = False,
    pitch_roll_dof: bool = False,
    depth_seg_masks: Optional[List[np.ndarray]] = None,
    auto_converge: bool = False,
) -> Tuple[Optional[dict], List[np.ndarray]]:
    if depth_seg_masks is not None:
        raw_candidates = [
            connected_components_sorted(m.copy(), min_area=80, top_k=3) or [m.copy()]
            for m in depth_seg_masks
        ]
    else:
        raw_candidates = []
        for cam in frames:
            raw = color_mask_for_object(cam.color_bgr, object_name)
            comps = connected_components_sorted(raw, min_area=80, top_k=3)
            raw_candidates.append(comps)

    masks_raw = select_multiview_seed_masks(frames, raw_candidates, table_info)
    areas = [int(np.count_nonzero(m)) for m in masks_raw]
    print(f"      [seed_mask] areas cam0={areas[0]} cam1={areas[1]} cam2={areas[2]}")

    # 깊이 클러스터 기반 마스크 정제: 인접 동색상 오염 제거 + cross-cam 복원
    masks = mask_refine_with_depth_cluster(
        frames, masks_raw, table_info, object_name=object_name
    )
    # 디버그: 정제된 마스크 시각화 타일 생성 (run에서 한 장으로 합침)
    debug_tiles: List[np.ndarray] = []
    for ci, cam in enumerate(frames):
        vis = cam.color_bgr.copy()
        cnts, _ = cv2.findContours(
            masks[ci], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{object_name} cam{ci} area={int(np.count_nonzero(masks[ci]))}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        debug_tiles.append(vis)

    n_valid = sum(1 for m in masks if int(np.count_nonzero(m)) > 200)
    if n_valid < 1:
        print(f"    [SKIP] {object_name}: 유효 마스크 없음")
        return None, debug_tiles

    table_n, table_d, table_center, table_radius = table_info
    obj_pts = fuse_object_points(
        frames, masks, table_n, table_d, table_center, table_radius
    )
    if len(obj_pts) < 60:
        print(f"    [SKIP] {object_name}: 융합 점군 부족 ({len(obj_pts)}pts)")
        return None, debug_tiles
    ext = obj_pts.max(0) - obj_pts.min(0)
    print(
        f"    {object_name} valid_masks={n_valid}/3 pts={len(obj_pts)} "
        f"extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm"
    )

    model_pts = sample_model_points(model, n=6000)
    table_n, table_d, _, _ = table_info

    # bbox 기반 target scale 사전 계산 → register_direct에 전달
    _pre_scale = _compute_target_scale_simple(
        model, glb_src_path, frames, object_name, seed_masks=masks_raw,
    )
    if _pre_scale is not None:
        print(f"      [pre-bbox-scale] target={_pre_scale:.3f}")

    try:
        pose = register_direct(
            object_name, model, model_pts, obj_pts, frames, masks, table_n, table_d,
            scale_override=_pre_scale,
        )
    except Exception as exc:
        print(f"    [WARN] {object_name}: scale_override 정합 실패 ({exc}), fallback")
        pose = None
    if pose is None and _pre_scale is not None:
        print(f"    [FALLBACK] scale_override 없이 재시도")
        try:
            pose = register_direct(
                object_name, model, model_pts, obj_pts, frames, masks, table_n, table_d,
            )
        except Exception as exc:
            print(f"    [FAIL] {object_name}: fallback도 실패 ({exc})")
            return None, debug_tiles
    if pose is None:
        print(f"    [FAIL] {object_name}: 정합 실패")
        return None, debug_tiles

    # Iterative pose-guided mask refinement:
    #   (1) 현재 pose의 mesh silhouette 투영 → 색상마스크 ∩ silhouette_dilated로 tight 마스크
    #   (2) 재-fuse 후 re-register. 점수 개선 시에만 채택.
    for ri in range(pose_refine_iters):
        try:
            masks_tight = refine_masks_with_pose(
                frames, masks, pose, model, glb_src_path, dilate_px=8,
            )
        except Exception as exc:
            print(f"      [iter{ri}] mask refine 실패: {exc}")
            break
        obj_pts_tight = fuse_object_points(
            frames, masks_tight, table_n, table_d, table_center, table_radius
        )
        if len(obj_pts_tight) < 60:
            print(f"      [iter{ri}] tight pts 부족 ({len(obj_pts_tight)}) → skip")
            break
        try:
            pose_new = register_direct(
                object_name, model, model_pts, obj_pts_tight,
                frames, masks_tight, table_n, table_d,
            )
        except Exception as exc:
            print(f"      [iter{ri}] re-register 실패: {exc}")
            break
        scale_ratio = pose_new.scale / pose.scale if pose_new else 0.0
        if (pose_new is not None
                and pose_new.confidence > pose.confidence + 0.005
                and 0.75 < scale_ratio < 1.35):
            print(
                f"      [iter{ri}] pose 개선 conf {pose.confidence:.3f} → "
                f"{pose_new.confidence:.3f} (pts {len(obj_pts)}→{len(obj_pts_tight)})"
            )
            pose = pose_new
            obj_pts = obj_pts_tight
            masks = masks_tight
        else:
            new_conf = pose_new.confidence if pose_new is not None else 0.0
            print(
                f"      [iter{ri}] 개선 없음 conf {pose.confidence:.3f} vs "
                f"{new_conf:.3f} → 유지"
            )
            break

    # Render-and-compare 자동 정제 (analysis-by-synthesis):
    #   실루엣 IoU를 목적함수로 Nelder-Mead 최적화. 육안 일치도 직접 개선.
    #   비대칭 물체는 상하 flip 후보까지 탐색 (180° 뒤집힘 방지).
    if render_refine:
        symmetry_sel = OBJECT_SYMMETRY.get(object_name, "none")
        # bbox 기반 scale 보정: 관측 마스크 bbox 크기에서 목표 scale 계산
        target_scale = _compute_target_scale(
            pose, model, glb_src_path, frames, masks,
            object_name=object_name,
        )
        if target_scale is not None and abs(target_scale / pose.scale - 1.0) > 0.15:
            print(f"      [bbox-scale] {pose.scale:.3f} → {target_scale:.3f} "
                  f"({target_scale/pose.scale:.2f}×)")
            pose = PoseEstimate(
                T_base_obj=pose.T_base_obj, position_m=pose.position_m,
                quaternion_xyzw=pose.quaternion_xyzw,
                euler_xyz_deg=pose.euler_xyz_deg,
                scale=float(target_scale),
                confidence=pose.confidence, fitness=pose.fitness,
                rmse=pose.rmse, depth_score=pose.depth_score,
                coverage=pose.coverage, silhouette_score=pose.silhouette_score,
            )

        # Silhouette IoU 기반 render-compare (scale 고정, 위치/yaw만 4D)
        try:
            if auto_converge:
                pose, masks = auto_refine_pose_until_converged(
                    pose, model, glb_src_path, frames, masks,
                    table_n, table_d,
                    multi_yaw=True if symmetry_sel == "none" else multi_yaw,
                    pitch_roll_dof=True if symmetry_sel == "none" else pitch_roll_dof,
                    max_rounds=6,
                )
            else:
                if symmetry_sel == "none":
                    pose = render_and_compare_with_flips(
                        pose, model, glb_src_path, frames, masks,
                        table_n, table_d, max_iter=100, multi_yaw=multi_yaw,
                        pitch_roll_dof=pitch_roll_dof,
                        include_axis_turns=True,
                        shortlist_k=6,
                    )
                else:
                    pose = render_and_compare_refine(
                        pose, model, glb_src_path, frames, masks,
                        table_n, table_d, max_iter=150,
                        pitch_roll_dof=pitch_roll_dof,
                    )
            if mask_iter_loop and not auto_converge:
                target_iou = 0.85
                max_loops = 8
                dilate_schedule = [14, 10, 8, 6, 5, 4, 3, 3]
                for loop_i in range(max_loops):
                    if pose.silhouette_score >= target_iou:
                        print(f"      [auto-loop] target reached "
                              f"IoU={pose.silhouette_score:.3f}≥{target_iou}")
                        break
                    try:
                        dilate_px = dilate_schedule[min(loop_i, len(dilate_schedule)-1)]
                        masks_new = refine_masks_with_pose(
                            frames, masks, pose, model, glb_src_path,
                            dilate_px=dilate_px,
                        )
                        pose_new = render_and_compare_refine(
                            pose, model, glb_src_path, frames, masks_new,
                            table_n, table_d, max_iter=80,
                        )
                        if pose_new.silhouette_score > pose.silhouette_score + 1e-3:
                            print(f"      [auto-loop {loop_i} dilate={dilate_px}] IoU "
                                  f"{pose.silhouette_score:.3f}→"
                                  f"{pose_new.silhouette_score:.3f}")
                            pose = pose_new
                            masks = masks_new
                        else:
                            print(f"      [auto-loop {loop_i}] no improvement, stop "
                                  f"(IoU={pose.silhouette_score:.3f})")
                            break
                    except Exception as exc:
                        print(f"      [auto-loop] 예외 ({exc}) → skip")
                        break
        except Exception as exc:
            print(f"      [render-refine] 예외 ({exc}) → skip")

    # 비균일 scale 정제 (pose가 정확할 때만 소폭 보정)
    symmetry = OBJECT_SYMMETRY.get(object_name, "none")
    per_axis = refine_anisotropic_scale(
        pose.T_base_obj, pose.scale, model_pts, obj_pts,
        symmetry=symmetry, coverage_ratio_min=0.75,
    )
    aniso_paths = export_posed_anisotropic(
        pose, per_axis, model, glb_src_path, frame_id, out_dir
    )
    # pose JSON export: 축별 크기를 기록
    mod_ext_unit = model_pts.max(0) - model_pts.min(0)
    real_per_axis_m = per_axis * mod_ext_unit
    rot = Rot.from_matrix(pose.T_base_obj[:3, :3])
    result = {
        "frame_id": frame_id,
        "object_name": model.name,
        "label": OBJECT_LABELS.get(model.name, model.name),
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
        **aniso_paths,
    }
    json_path = out_dir / f"pose_{model.name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(
        f"    [OK] {object_name} conf={pose.confidence:.3f} "
        f"fit={pose.fitness:.3f} rmse={pose.rmse*1000:.1f}mm "
        f"scale_per_axis=[{per_axis[0]:.3f},{per_axis[1]:.3f},{per_axis[2]:.3f}]"
    )
    return result, debug_tiles


def run(frame_id: str, only: Optional[List[str]] = None,
        pose_refine_iters: int = 0, render_refine: bool = False,
        multi_yaw: bool = False, mask_iter_loop: bool = False,
        pitch_roll_dof: bool = False, depth_seg: bool = False,
        auto_converge: bool = False,
        sam_masks_dir: Optional[str] = None):
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    print("=" * 64)
    print(f" Per-object Pose Estimation — Frame {frame_id}")
    print("=" * 64)

    table_info = estimate_table_plane(frames)
    tn, td, tc, tr = table_info
    print(
        f"[plane] center=[{tc[0]:+.3f},{tc[1]:+.3f},{tc[2]:+.3f}] radius={tr:.3f}"
    )

    # 프레임별 전용 폴더에 산출물 저장 (타임스탬프로 기존 결과 보존)
    import datetime as _dt
    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    _suffix = "_sam" if sam_masks_dir else ""
    frame_dir = OUT_DIR / f"run_{_ts}{_suffix}_frame_{frame_id}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    # 최신 결과는 frame_{id}/ 심볼릭 링크도 남김 (compare_overlay 호환)
    latest_link = OUT_DIR / f"frame_{frame_id}"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
        latest_link.symlink_to(frame_dir.name)
    except Exception:
        pass

    # Depth-based segmentation (옵션)
    depth_seg_all: Optional[Dict[str, List[np.ndarray]]] = None
    if depth_seg:
        depth_seg_all = depth_segment_all_objects(
            frames, tn, td, tc, tr,
        )
        print("[depth-seg] 물체별 마스크 생성 완료")

    # SAM 사전 계산 마스크 로드 (선택)
    sam_masks_all: Optional[Dict[str, List[np.ndarray]]] = None
    if sam_masks_dir is not None:
        sam_dir = Path(sam_masks_dir)
        if sam_dir.is_dir():
            sam_masks_all = {}
            for name in [f"object_{i:03d}" for i in range(1, 5)]:
                masks_for_obj = []
                for ci in range(3):
                    p = sam_dir / f"{name}_cam{ci}.png"
                    if p.exists():
                        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                        masks_for_obj.append(m if m is not None else np.zeros((480, 640), dtype=np.uint8))
                    else:
                        masks_for_obj.append(np.zeros((480, 640), dtype=np.uint8))
                sam_masks_all[name] = masks_for_obj
            print(f"[sam] 마스크 로드 완료: {sam_dir}")
        else:
            print(f"[WARN] --sam_masks_dir 경로 없음: {sam_dir}")

    results = []
    tiles_by_obj: Dict[str, List[np.ndarray]] = {}
    object_names = only or [f"object_{i:03d}" for i in range(1, 5)]
    for name in object_names:
        glb = DATA_DIR / f"{name}.glb"
        if not glb.exists():
            print(f"  [SKIP] {name}: GLB 없음 {glb}")
            continue
        print(f"\n  === {name} ({OBJECT_LABELS.get(name, name)}) ===")
        model = normalize_glb(glb)
        # 우선순위: SAM > depth-seg > color mask
        if sam_masks_all is not None and name in sam_masks_all:
            pre_masks = sam_masks_all[name]
        else:
            pre_masks = depth_seg_all.get(name) if depth_seg_all else None
        r, tiles = estimate_one(
            name, frames, model, glb, table_info, frame_dir, frame_id,
            pose_refine_iters=pose_refine_iters,
            render_refine=render_refine,
            multi_yaw=multi_yaw,
            mask_iter_loop=mask_iter_loop,
            pitch_roll_dof=pitch_roll_dof,
            depth_seg_masks=pre_masks,
            auto_converge=auto_converge,
        )
        if tiles:
            tiles_by_obj[name] = tiles
        if r is not None:
            results.append(r)

    # 레거시 flat-layout 잔여 파일 제거 (이 frame 관련)
    for pat in (f"debug_mask_cam*_object_*_{frame_id}.png",
                f"debug_masks_{frame_id}.png",
                f"pose_object_*_{frame_id}.json",
                f"pose_object_*_{frame_id}.npz",
                f"object_*_posed_{frame_id}.glb",
                f"object_*_posed_{frame_id}_isaac.glb",
                f"summary_{frame_id}.json"):
        for legacy in OUT_DIR.glob(pat):
            try: legacy.unlink()
            except Exception: pass
    legacy_cmp = OUT_DIR / f"comparison_{frame_id}"
    if legacy_cmp.is_dir():
        try:
            for p in legacy_cmp.glob("*"):
                p.unlink()
            legacy_cmp.rmdir()
        except Exception:
            pass

    # 모든 물체의 마스크 타일을 하나의 grid 이미지로 저장 (행=물체, 열=카메라)
    if tiles_by_obj:
        ordered = [n for n in object_names if n in tiles_by_obj]
        rows = [np.hstack(tiles_by_obj[n]) for n in ordered]
        grid = np.vstack(rows)
        grid_path = frame_dir / "debug_masks.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"  debug_masks grid: {grid_path}")

    summary_path = frame_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  summary: {summary_path} ({len(results)}/{len(object_names)} 성공)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--all", action="store_true", help="모든 프레임 처리")
    ap.add_argument(
        "--only", default=None, help="콤마로 구분된 object_001..object_004"
    )
    ap.add_argument(
        "--pose_refine_iters", type=int, default=0,
        help="pose-guided mask refinement 반복 횟수 (0=off).",
    )
    ap.add_argument(
        "--render_refine", action="store_true",
        help="render-and-compare (IoU Nelder-Mead) 자동 정제 활성화.",
    )
    ap.add_argument("--multi_yaw", action="store_true",
                    help="각 flip 후보에 yaw 0/90/180/270 4초기 탐색 추가.")
    ap.add_argument("--mask_iter_loop", action="store_true",
                    help="pose로 마스크 재정제 → re-refine 루프 (최대 2회).")
    ap.add_argument("--pitch_roll_dof", action="store_true",
                    help="Nelder-Mead에 pitch/roll 자유도 추가 (수직축 가까이).")
    ap.add_argument("--depth_seg", action="store_true",
                    help="HSV 색상 대신 depth 기반 세그멘테이션 사용.")
    ap.add_argument(
        "--auto_converge", action="store_true",
        help="orientation 후보 탐색 + mask 재정제 + render refine를 수렴할 때까지 반복.",
    )
    ap.add_argument("--sam_masks_dir", default=None,
                    help="SAM으로 사전 계산한 마스크 디렉토리 (object_XXX_camY.png).")
    args = ap.parse_args()

    only = args.only.split(",") if args.only else None

    kw = dict(pose_refine_iters=args.pose_refine_iters,
              render_refine=args.render_refine,
              multi_yaw=args.multi_yaw,
              mask_iter_loop=args.mask_iter_loop,
              pitch_roll_dof=args.pitch_roll_dof,
              depth_seg=args.depth_seg,
              auto_converge=args.auto_converge,
              sam_masks_dir=args.sam_masks_dir)
    if args.all:
        cam0_dir = DATA_DIR / "object_capture" / "cam0"
        fids = sorted(f.stem.replace("rgb_", "") for f in cam0_dir.glob("rgb_*.jpg"))
        for fid in fids:
            run(fid, only=only, **kw)
    else:
        run(args.frame_id, only=only, **kw)


if __name__ == "__main__":
    main()
