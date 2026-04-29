#!/usr/bin/env python3
"""Object-profile driven multi-view pose estimation pipeline core.

ObjectProfile (JSON) 한 파일이 한 물체의 SAM-mask + pose 동작을 모두 정의.
어떤 물체든 profile 만 추가하면 같은 코드로 추정 가능.

이 모듈은 라이브러리이며, CLI는 `run_pipeline.py` 가 담당.

자체 포함:
  - SAM 마스크 helper: glb_extent, mask_3d_info, project_bbox_from_3d,
    cylinder_axis_points, run_sam, keep_nearest_component, auto_refine_mask,
    evaluate_mask_quality, _navy_top_face_mask 등 모두 본 파일에 정의.
  - Pose 추정: ICP fitness + render_compare (silhouette IoU + Nelder-Mead).
  - 외부 의존: pose_pipeline (load_calibration, load_frame, normalize_glb,
    estimate_table_plane), mobile_sam (SamPredictor, sam_model_registry).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

from pose_pipeline import (
    CameraIntrinsics, CameraFrame,
    load_calibration, load_frame,
    normalize_glb, sample_model_points,
    estimate_table_plane,
)
from mobile_sam import SamPredictor, sam_model_registry


# ═══════════════════════════════════════════════════════════
# 0. SAM 마스크 helper 함수들 (이전 make_final_sam_masks.py 에서 흡수)
# ═══════════════════════════════════════════════════════════

def glb_extent(glb_path: Path) -> np.ndarray:
    """GLB unscaled 3D extent (width, height, depth)."""
    scene = trimesh.load(str(glb_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene)
    V = np.asarray(mesh.vertices)
    return V.max(0) - V.min(0)


def mask_3d_info(mask: np.ndarray, cam) -> Optional[dict]:
    """mask → base frame 3D centroid + extent."""
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    K = cam.intrinsics.K
    m = (mask > 0) & (depth > 0.05) & (depth < 1.5)
    if m.sum() < 30:
        return None
    ys, xs = np.where(m)
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t
    centroid = np.median(pts_base, axis=0)
    lo = np.percentile(pts_base, 5, axis=0)
    hi = np.percentile(pts_base, 95, axis=0)
    ext = hi - lo
    return {"centroid": centroid, "extent": ext,
            "max_extent": float(ext.max()), "n_pts": int(m.sum())}


def project_bbox_from_3d(centroid: np.ndarray, extent: np.ndarray, cam,
                         padding_m: float = 0.015) -> Optional[tuple]:
    """3D 중심 + extent의 8 corners를 카메라에 투영 → bbox."""
    half = extent / 2 + padding_m
    corners = np.array([
        centroid + half * np.array([sx, sy, sz])
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ])
    T_cb = np.linalg.inv(cam.T_base_cam)
    pts_cam = (T_cb[:3, :3] @ corners.T).T + T_cb[:3, 3]
    z = pts_cam[:, 2]
    ok = z > 0.05
    if ok.sum() < 4:
        return None
    K = cam.intrinsics.K
    u = K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2]
    v = K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2]
    H, W = cam.intrinsics.height, cam.intrinsics.width
    x1 = max(0, int(u.min())); x2 = min(W - 1, int(u.max()))
    y1 = max(0, int(v.min())); y2 = min(H - 1, int(v.max()))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return (x1, y1, x2, y2)


def cylinder_axis_points(seed: np.ndarray) -> np.ndarray:
    """실린더용 3점 prompt (위/중간/아래)."""
    ys, xs = np.where(seed > 0)
    if len(xs) < 30:
        return np.array([[int(xs.mean()), int(ys.mean())]])
    y_min, y_max = ys.min(), ys.max()
    pts = []
    for y_t in [y_min + (y_max - y_min)//5, (y_min + y_max)//2,
                y_max - (y_max - y_min)//5]:
        row_xs = xs[np.abs(ys - y_t) < 3]
        if len(row_xs) > 0:
            pts.append([int(np.median(row_xs)), int(y_t)])
    return np.array(pts) if pts else np.array([[int(xs.mean()), int(ys.mean())]])


def run_sam(predictor: SamPredictor, bgr: np.ndarray,
            bbox: tuple, points: Optional[np.ndarray] = None) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    kwargs = dict(box=np.array(bbox), multimask_output=True)
    if points is not None and len(points) > 0:
        kwargs["point_coords"] = points
        kwargs["point_labels"] = np.ones(len(points), dtype=np.int32)
    masks, scores, _ = predictor.predict(**kwargs)
    best = int(np.argmax(scores))
    out = masks[best].astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k5)


def _largest_cc(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return np.where(labels == best, 255, 0).astype(np.uint8)


def keep_nearest_component(mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """bbox 중심에 가장 가까운 component만 유지."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    best_i, best_d = -1, 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            continue
        d = (cents[i][0] - cx)**2 + (cents[i][1] - cy)**2
        if d < best_d:
            best_d = d; best_i = i
    if best_i < 0:
        return mask
    return np.where(labels == best_i, 255, 0).astype(np.uint8)


def _color_homogeneity(mask: np.ndarray, bgr: np.ndarray, ref_h: float) -> float:
    if int((mask > 0).sum()) < 30:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    h = hsv[ys, xs, 0].astype(np.float32)
    diff = np.minimum(np.abs(h - ref_h), 180.0 - np.abs(h - ref_h))
    return float(np.clip(1.0 - np.mean(diff) / 30.0, 0.0, 1.0))


def _compactness(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(cnt)
    if a < 30:
        return 0.0
    hull = cv2.convexHull(cnt)
    ah = cv2.contourArea(hull)
    if ah < 1:
        return 0.0
    return float(min(1.0, a / ah))


def evaluate_mask_quality(mask: np.ndarray, bgr: np.ndarray, cam,
                          obj_name: str, glb_ext: np.ndarray,
                          ref_h: float) -> dict:
    """3D extent + 색상 균질성 + compactness 가중 평균."""
    info = mask_3d_info(mask, cam)
    extent_score = 0.0; extent_axes = None
    over_axis = False; under_axis = False
    if info is not None:
        ax = np.sort(info["extent"])[::-1]
        ge = np.sort(glb_ext)[::-1]
        extent_axes = ax
        ratios = []
        for a, g in zip(ax, ge):
            if g < 1e-6:
                continue
            r = a / g
            ratios.append(min(r, 1.0 / r))
            if r > 1.6:
                over_axis = True
            if r < 0.35:
                under_axis = True
        if ratios:
            extent_score = float(np.mean(ratios))
    color_score = _color_homogeneity(mask, bgr, ref_h)
    compact = _compactness(mask)
    area = int((mask > 0).sum())
    if area < 80:
        total = 0.0
    else:
        total = 0.5 * extent_score + 0.3 * color_score + 0.2 * compact
    return {"score": total, "extent_score": extent_score,
            "color_score": color_score, "compactness": compact,
            "area": area,
            "extent_axes_m": extent_axes.tolist() if extent_axes is not None else None,
            "over_axis": over_axis, "under_axis": under_axis}


def _tighten_with_grabcut(mask, bgr, shrink_px=5,
                           extra_fgd: Optional[np.ndarray] = None) -> np.ndarray:
    if int((mask > 0).sum()) < 100:
        return mask
    H, W = mask.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (shrink_px * 2 + 1, shrink_px * 2 + 1))
    eroded = cv2.erode(mask, k, iterations=1)
    dilated = cv2.dilate(mask, k, iterations=2)
    gc_mask = np.full((H, W), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[dilated > 0] = cv2.GC_PR_BGD
    gc_mask[mask > 0] = cv2.GC_PR_FGD
    gc_mask[eroded > 0] = cv2.GC_FGD
    if extra_fgd is not None:
        gc_mask[extra_fgd > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                    255, 0).astype(np.uint8)
    if int((out > 0).sum()) < 100:
        return mask
    return _largest_cc(out)


def _halo_strip(mask, erode_px=5, dilate_back_px=3) -> np.ndarray:
    if int((mask > 0).sum()) < 100:
        return mask
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (erode_px * 2 + 1, erode_px * 2 + 1))
    eroded = cv2.erode(mask, ke, iterations=1)
    eroded = _largest_cc(eroded)
    if int((eroded > 0).sum()) < 40:
        return mask
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (dilate_back_px * 2 + 1,
                                     dilate_back_px * 2 + 1))
    out = cv2.dilate(eroded, kd, iterations=1)
    out = cv2.bitwise_and(out, mask)
    return _largest_cc(out)


def _open_largest(mask, k_size=5) -> np.ndarray:
    if int((mask > 0).sum()) < 80:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    op = _largest_cc(op)
    if int((op > 0).sum()) < 80:
        return mask
    return op


def _fill_holes(mask) -> np.ndarray:
    if int((mask > 0).sum()) < 80:
        return mask
    h, w = mask.shape
    ff = mask.copy()
    mk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)


def _convex_hull_fill(mask) -> np.ndarray:
    if int((mask > 0).sum()) < 80:
        return mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [hull], -1, 255, thickness=cv2.FILLED)
    return out


def _navy_top_face_mask(bgr, bbox) -> np.ndarray:
    """곤색 블록의 어두운 윗면(저채도/저명도 청회색)을 bbox 내부에서만 추출."""
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]; v = hsv[:, :, 2]
    hue_ok = (h >= 100) & (h <= 130)
    sv_ok = (s >= 20) & (v >= 10) & (v <= 75)
    m = (hue_ok & sv_ok).astype(np.uint8) * 255
    x1, y1, x2, y2 = bbox
    pad = 4
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    out = np.zeros_like(m)
    out[y1:y2, x1:x2] = m[y1:y2, x1:x2]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, k3)


def _edge_snap(mask: np.ndarray, bgr: np.ndarray,
               search_px: int = 5, canny_lo: int = 40,
               canny_hi: int = 120) -> np.ndarray:
    """이미지 gradient(edge)에 마스크 경계 snap.

    grabcut 이 색상 GMM 으로 분할한다면, edge_snap 은 순수 gradient 만 사용.
    물체와 배경이 비슷한 색이지만 강한 edge 가 있는 경우 (금속, 텍스쳐 같음)
    에 유용. 동작:
      1) Canny → strong edges
      2) 마스크 경계 ±search_px 의 boundary band
      3) 확실 interior(erode) ∪ (band ∩ dilated_edges) 만 유지
    """
    if int((mask > 0).sum()) < 100:
        return mask
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    k_e = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (search_px * 2 + 1, search_px * 2 + 1))
    interior = cv2.erode(mask, k_e, iterations=1)
    if int((interior > 0).sum()) < 50:
        return mask
    exterior = cv2.dilate(mask, k_e, iterations=1)
    boundary_band = cv2.bitwise_and(exterior, cv2.bitwise_not(interior))
    edges_dil = cv2.dilate(edges, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    band_with_edges = cv2.bitwise_and(boundary_band, edges_dil)
    refined = cv2.bitwise_or(interior, band_with_edges)
    refined = _largest_cc(refined)
    if int((refined > 0).sum()) < 80:
        return mask
    # close 작게: edge band 가 끊어졌어도 fill
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k3)
    return refined


def auto_refine_mask(initial: np.ndarray, bgr: np.ndarray, cam, obj_name: str,
                     glb_ext: np.ndarray, ref_h: float,
                     max_iter: int = 10, verbose: bool = True) -> tuple:
    """반복 정제: 후보 액션 (close/halo/grabcut/erode/dilate/fill_holes/hull_fill)
    중 점수 가장 좋은 것 채택. 초기 영역의 60% 미만으로는 줄지 않음.
    """
    cur = initial.copy()
    best = initial.copy()
    best_q = evaluate_mask_quality(best, bgr, cam, obj_name, glb_ext, ref_h)
    initial_area = best_q["area"]
    abs_min_area = max(150, int(initial_area * 0.6))
    if verbose:
        print(f"      [auto] iter0 score={best_q['score']:.3f} "
              f"ext={best_q['extent_score']:.2f} col={best_q['color_score']:.2f} "
              f"cmp={best_q['compactness']:.2f} area={best_q['area']} "
              f"(min_floor={abs_min_area})")
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    no_improve = 0
    used_grabcut = 0
    used_halo = 0
    used_edge_snap = 0
    for it in range(1, max_iter + 1):
        q = evaluate_mask_quality(cur, bgr, cam, obj_name, glb_ext, ref_h)
        if q["score"] >= 0.88:
            break
        candidates = []
        n, _, _, _ = cv2.connectedComponentsWithStats(cur, connectivity=8)
        if n - 1 >= 2:
            c = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, k7)
            candidates.append(("close+largest", _largest_cc(c)))
        if q["compactness"] < 0.6:
            candidates.append(("fill_holes", _fill_holes(cur)))
            candidates.append(("hull_fill", _convex_hull_fill(cur)))
        if q["compactness"] < 0.85 and used_halo < 3:
            candidates.append(("halo_strip4",
                                _halo_strip(cur, erode_px=4, dilate_back_px=3)))
            candidates.append(("halo_strip6",
                                _halo_strip(cur, erode_px=6, dilate_back_px=4)))
            candidates.append(("open5", _open_largest(cur, k_size=5)))
            candidates.append(("open7", _open_largest(cur, k_size=7)))
        if (q["color_score"] < 0.9 or q["compactness"] < 0.9) \
                and used_grabcut < 2:
            extra_fgd = None
            if obj_name == "object_003":
                ys, xs = np.where(cur > 0)
                if len(xs) > 0:
                    bb = (int(xs.min()), int(ys.min()),
                          int(xs.max()), int(ys.max()))
                    extra_fgd = _navy_top_face_mask(bgr, bb)
            c = _tighten_with_grabcut(cur, bgr, shrink_px=4, extra_fgd=extra_fgd)
            candidates.append(("grabcut", c))
        # edge_snap: gradient 기반 boundary 정렬 (grabcut 보완)
        if (q["compactness"] < 0.9 or q["color_score"] < 0.9) \
                and used_edge_snap < 2:
            candidates.append(("edge_snap",
                                _edge_snap(cur, bgr, search_px=5)))
        if q["over_axis"]:
            c = cv2.erode(cur, k3, iterations=1)
            candidates.append(("erode", _largest_cc(c)))
        if q["under_axis"] and it <= 2:
            candidates.append(("dilate", cv2.dilate(cur, k3, iterations=1)))

        if q["compactness"] < 0.4:
            ratio = 0.5
        elif q["compactness"] < 0.6:
            ratio = 0.6
        else:
            ratio = 0.7
        min_area_keep = max(abs_min_area, int(q["area"] * ratio))
        best_action = None; best_action_q = None; best_action_mask = None
        for name, m in candidates:
            a = int((m > 0).sum())
            if a < min_area_keep:
                continue
            qm = evaluate_mask_quality(m, bgr, cam, obj_name, glb_ext, ref_h)
            if best_action_q is None or qm["score"] > best_action_q["score"]:
                best_action_q = qm; best_action = name; best_action_mask = m
        if best_action is None:
            break
        if verbose:
            print(f"      [auto] iter{it} action={best_action} → "
                  f"score={best_action_q['score']:.3f} "
                  f"ext={best_action_q['extent_score']:.2f} "
                  f"col={best_action_q['color_score']:.2f} "
                  f"cmp={best_action_q['compactness']:.2f} "
                  f"area={best_action_q['area']}")
        if "halo" in best_action:
            used_halo += 1
        if best_action == "grabcut":
            used_grabcut += 1
        if best_action == "edge_snap":
            used_edge_snap += 1
        if best_action_q["score"] > best_q["score"] + 1e-3:
            best = best_action_mask.copy(); best_q = best_action_q
            cur = best_action_mask; no_improve = 0
        else:
            no_improve += 1; cur = best_action_mask
        if no_improve >= 2:
            break
    if verbose:
        print(f"      [auto] best score={best_q['score']:.3f} "
              f"area={best_q['area']}")
    return best, best_q


# ═══════════════════════════════════════════════════════════
# 1. ObjectProfile dataclass + JSON loader
# ═══════════════════════════════════════════════════════════

@dataclass
class ColorPriorConfig:
    enabled: bool = True
    hue_ref: float = 0.0
    hue_radius: float = 12.0
    s_min: int = 100
    v_min: int = 70
    v_max: int = 245
    relaxed_hue_radius: float = 22.0
    relaxed_s_min: int = 25
    relaxed_v_min: int = 15
    relaxed_v_max: int = 255
    # 검정/흰 배경 조합 등에서 배경 노이즈 차단:
    # true 면 V > white_v_min 인 가장 큰 영역(흰 종이)을 찾고,
    # 그 영역 + dilation 안의 픽셀만 색상 후보로 인정
    background_white_assist: bool = False
    white_v_min: int = 200
    white_dilate_px: int = 40


@dataclass
class SamConfig:
    bbox_pad_ratio: float = 0.05
    # centroid | color_axis_3pt | cylinder_axis | mask_skeleton
    prompt_strategy: str = "centroid"
    post_color_intersect: bool = True
    auto_refine: str = "full"            # full | extent_only | off
    reliability_threshold: float = 0.30
    own_bbox_max_image_ratio: float = 0.7  # this ratio 넘으면 cross-cam fallback
    # 3D-size filter scale_range (GLB max extent 대비 허용 비율)
    scale_range_min: float = 0.20
    scale_range_max: float = 1.20
    # own bbox + GLB-projected bbox 결합 방식
    # union: navy 윗면 등 own 색상 mask 가 빠뜨린 영역 포함 (블록류)
    # intersect: own 위치 + GLB 크기 cap (knife 등 다른 노란 물체 흡수 차단)
    bbox_combine: str = "union"   # union | intersect


@dataclass
class ShapeConfig:
    symmetry: str = "none"               # none | yaw
    init_orientation: str = "auto"       # auto | upright | lying_flat
    yaw_steps: int = 24
    anisotropic_scale: bool = False
    horizontal_constrain: bool = False
    extent_match_min: float = 0.30       # GLB 대비 3D extent 비율 하한 (auto-detect 용)


@dataclass
class PoseConfig:
    method: str = "icp_fitness"          # icp_fitness | render_compare
    render_refine: bool = False
    flip_signs: Tuple[int, ...] = (1, -1)
    icp_voxel: float = 0.005
    icp_max_iter: int = 80
    # render_compare 옵션
    render_topk: int = 3                  # coarse 상위 K개를 fine으로
    render_max_iter: int = 100            # Nelder-Mead 반복 한도
    render_simplify_faces: int = 2000     # 렌더용 mesh 간소화 (속도)
    render_orientation_grid: int = 24     # coarse yaw 후보 수 (대칭이면 무시)
    render_cam_weights: Tuple[float, ...] = (1.0, 1.0, 1.0)
    # pitch/roll DOF: table plane 추정이 부정확하거나 물체가 살짝 기울어진
    # 경우. yaw + pitch + roll 합쳐 7DOF (isotropic) / 9DOF (anisotropic).
    refine_tilt_dof: bool = False
    # 평면 추정에 흰 종이 영역만 사용 (color_prior.background_white_assist 활성 시)
    table_plane_use_white_assist: bool = False
    # Isaac Sim 호환: posed_isaac.glb 저장 시 추가 180° 축 회전.
    # IsaacSim 에서 import 시 위/아래 / 앞/뒤 / 좌/우가 뒤집혀 보일 때 토글.
    isaac_rot_x_180: bool = True
    isaac_rot_y_180: bool = True
    isaac_rot_z_180: bool = False


@dataclass
class ObjectProfile:
    name: str
    glb: str
    label: str = ""
    overlay_color_bgr: Tuple[int, int, int] = (60, 200, 60)
    color_prior: ColorPriorConfig = field(default_factory=ColorPriorConfig)
    multicolor: bool = False
    sam: SamConfig = field(default_factory=SamConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)


def _build_dc(cls, src: Optional[dict]):
    if src is None:
        return cls()
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in src.items() if k in fields})


def load_profile(path: str | Path) -> ObjectProfile:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        # cwd 상대로 안 보이면 src/configs/objects/ 또는 src/ 에서 시도
        here = Path(__file__).resolve().parent
        for cand in (here / "configs" / "objects" / p,
                     here / "configs" / "objects" / p.name,
                     here / p):
            if cand.exists():
                p = cand; break
    data = json.loads(p.read_text(encoding="utf-8"))
    return ObjectProfile(
        name=data["name"],
        glb=data["glb"],
        label=data.get("label", ""),
        overlay_color_bgr=tuple(data.get("overlay_color_bgr", [60, 200, 60])),
        color_prior=_build_dc(ColorPriorConfig, data.get("color_prior")),
        multicolor=bool(data.get("multicolor", False)),
        sam=_build_dc(SamConfig, data.get("sam")),
        shape=_build_dc(ShapeConfig, data.get("shape")),
        pose=_build_dc(PoseConfig, data.get("pose")),
    )


def profile_to_dict(p: ObjectProfile) -> dict:
    return asdict(p)


def auto_detect_profile(name: str, glb_path: str | Path,
                         frames: Optional[List[CameraFrame]] = None,
                         hue_seed: Optional[float] = None) -> ObjectProfile:
    """profile JSON 없이 GLB + (optional) frames 만으로 합리적 default 추정."""
    glb = Path(glb_path)
    ext = glb_extent(glb)
    ext_sorted = np.sort(ext)[::-1]
    ratio = float(ext_sorted[0] / max(ext_sorted[-1], 1e-6))

    # init_orientation: 가장 긴 축이 짧은 축의 3배 이상이면 누워있는 자세 가정
    init_ori = "lying_flat" if ratio > 3.0 else "auto"

    # symmetry: 가운데 두 축의 비율이 ~1 이면 회전 대칭(yaw)
    yaw_sym = abs(ext_sorted[0] / max(ext_sorted[1], 1e-6) - 1.0) < 0.10 \
              and ratio > 1.5

    # multicolor: 색상 prior 없으면 multicolor 가정 (안전)
    multicolor = hue_seed is None

    # GLB 가 1m 같은 너무 큰 단위면 실제 스케일이 작을 가능성이 큼 → scale_range 완화
    glb_max = float(ext_sorted[0])
    if glb_max > 0.5:
        sr_min, sr_max = 0.05, 1.5
    else:
        sr_min, sr_max = 0.20, 1.20

    cp = ColorPriorConfig(enabled=hue_seed is not None,
                           hue_ref=float(hue_seed) if hue_seed else 0.0,
                           hue_radius=18.0)
    return ObjectProfile(
        name=name, glb=str(glb),
        label=name,
        color_prior=cp,
        multicolor=multicolor,
        sam=SamConfig(
            bbox_pad_ratio=0.30 if multicolor else 0.05,
            prompt_strategy="cylinder_axis" if yaw_sym else "centroid",
            post_color_intersect=not multicolor,
            auto_refine="off" if multicolor else "full",
            scale_range_min=sr_min,
            scale_range_max=sr_max,
            bbox_combine="intersect" if multicolor else "union",
            reliability_threshold=0.10 if multicolor else 0.30,
        ),
        shape=ShapeConfig(
            symmetry="yaw" if yaw_sym else "none",
            init_orientation=init_ori,
        ),
    )


# ═══════════════════════════════════════════════════════════
# 2. HSV color mask (profile-driven)
# ═══════════════════════════════════════════════════════════

def _hue_dist(h: np.ndarray, ref: float) -> np.ndarray:
    return np.minimum(np.abs(h - ref), 180.0 - np.abs(h - ref))


def _estimate_table_plane_in_region(frames, cp: ColorPriorConfig):
    """흰 종이 영역 내부의 depth 점만 사용해서 평면 추정 (RANSAC).

    배경 walls / 책상 가장자리 / 케이블 등 노이즈가 평면 fit 에 끼지 않게.
    Returns: (n, d) — n: 단위 normal, plane: n·x + d = 0
    """
    all_pts = []
    for cam in frames:
        paper = _white_paper_region(cam.color_bgr, cp)
        depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
        K = cam.intrinsics.K
        ys, xs = np.where((paper > 0) & (depth > 0.1) & (depth < 1.5))
        if len(xs) > 8000:
            idx = np.random.choice(len(xs), 8000, replace=False)
            ys, xs = ys[idx], xs[idx]
        if len(xs) < 200:
            continue
        z = depth[ys, xs]
        x_cam = (xs - K[0, 2]) * z / K[0, 0]
        y_cam = (ys - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
        R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
        all_pts.append((R @ pts_cam.T).T + t)
    if not all_pts:
        # fallback to pose_pipeline 의 전체 평면 추정
        n, d, *_ = estimate_table_plane(frames)
        return n, d
    P = np.concatenate(all_pts, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd_ds = pcd.voxel_down_sample(0.003)
    plane, _ = pcd_ds.segment_plane(distance_threshold=0.005,
                                     ransac_n=3, num_iterations=2000)
    n = np.array(plane[:3]); d = float(plane[3])
    n = n / (np.linalg.norm(n) + 1e-12)
    return n, d


def _white_paper_region(bgr: np.ndarray, cp: ColorPriorConfig) -> np.ndarray:
    """V>white_v_min 의 가장 큰 component → dilate 한 영역 (검색 범위)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    white = (v >= cp.white_v_min).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k5)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k5)
    if int((white > 0).sum()) < 500:
        return np.full_like(white, 255)  # 흰 영역 못 찾으면 전체 허용
    n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n <= 1:
        return np.full_like(white, 255)
    areas = st[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    paper = np.where(lab == big, 255, 0).astype(np.uint8)
    pad = max(1, int(cp.white_dilate_px))
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    return cv2.dilate(paper, kk, iterations=1)


def color_mask_strict(bgr: np.ndarray, cp: ColorPriorConfig) -> np.ndarray:
    if not cp.enabled:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]; v = hsv[:, :, 2]
    hue_ok = _hue_dist(h, cp.hue_ref) <= cp.hue_radius
    sv_ok = (s >= cp.s_min) & (v >= cp.v_min) & (v <= cp.v_max)
    m = (hue_ok & sv_ok).astype(np.uint8) * 255
    if cp.background_white_assist:
        m = cv2.bitwise_and(m, _white_paper_region(bgr, cp))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k9)
    m = cv2.dilate(m, k7, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7)
    return m


def color_mask_relaxed(bgr: np.ndarray, cp: ColorPriorConfig) -> np.ndarray:
    if not cp.enabled:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]; v = hsv[:, :, 2]
    hue_ok = _hue_dist(h, cp.hue_ref) <= cp.relaxed_hue_radius
    sv_ok = ((s >= cp.relaxed_s_min) & (v >= cp.relaxed_v_min)
             & (v <= cp.relaxed_v_max))
    m = (hue_ok & sv_ok).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    return m


def best_components_sorted(mask: np.ndarray, min_area: int = 200,
                           top_k: int = 5) -> List[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        bm = (labels == i).astype(np.uint8) * 255
        out.append((a, bm))
    out.sort(key=lambda x: -x[0])
    return [b for _, b in out[:top_k]]


def filter_by_3d_size(candidates, cam, glb_ext,
                      scale_range=(0.20, 1.20), aspect_max=3.0):
    glb_max = float(glb_ext.max())
    glb_sorted = np.sort(glb_ext)[::-1]
    min_m = glb_max * scale_range[0]
    max_m = glb_max * scale_range[1]
    scored = []
    for m in candidates:
        if int((m > 0).sum()) < 200:
            continue
        info = mask_3d_info(m, cam)
        if info is None or info["max_extent"] < 1e-4:
            continue
        cand_max = info["max_extent"]
        if cand_max < min_m or cand_max > max_m:
            continue
        ext_sorted = np.sort(info["extent"])[::-1]
        if ext_sorted[-1] > glb_sorted[-1] * aspect_max:
            continue
        ratios = [min(a/b, b/a) for a, b in zip(ext_sorted, glb_sorted)
                  if a > 1e-6 and b > 1e-6]
        if not ratios:
            continue
        score = float(np.mean(ratios))
        scored.append((m, info, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════
# 3. SAM prompt strategies
# ═══════════════════════════════════════════════════════════

def _skeleton_points(mask: np.ndarray, n_pts: int = 3) -> Optional[np.ndarray]:
    """Distance-transform 기반 medial axis 샘플링.

    cylinder_axis(세로축 3점) 보다 일반적: PCA로 mask 의 주축을 찾고,
    그 축을 따라 n_pts 구간 분할 → 각 구간에서 distance transform 값 최대인
    픽셀 (즉 medial axis 위 가장 깊은 점) 을 prompt 로.
    L자 / 휘어진 / 비대칭 형태에 유용.
    """
    if int((mask > 0).sum()) < 30:
        return None
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ys, xs = np.where(dt > 0)
    if len(xs) < n_pts:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, -1]   # largest eigenvalue
    proj = (pts - centroid) @ main_axis
    pmin, pmax = float(proj.min()), float(proj.max())
    if pmax - pmin < 1.0:
        return None
    out = []
    for k in range(n_pts):
        lo = pmin + (pmax - pmin) * k / n_pts
        hi = pmin + (pmax - pmin) * (k + 1) / n_pts
        sel = (proj >= lo) & (proj <= hi)
        if not sel.any():
            continue
        d_in = dt[ys[sel], xs[sel]]
        b = int(np.argmax(d_in))
        out.append([int(xs[sel][b]), int(ys[sel][b])])
    return np.array(out) if out else None


def _sam_points(strategy: str, color_blob: Optional[np.ndarray],
                bbox: Tuple[int, int, int, int]) -> np.ndarray:
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    if strategy == "centroid" or color_blob is None:
        return np.array([[cx, cy]])
    if strategy == "cylinder_axis":
        return cylinder_axis_points(color_blob)
    if strategy == "color_axis_3pt":
        ys, xs = np.where(color_blob > 0)
        if len(xs) < 30:
            return np.array([[cx, cy]])
        order = np.argsort(ys)
        pts = []
        for fr in [0.2, 0.5, 0.8]:
            k = int(len(order) * fr)
            pts.append([int(xs[order[k]]), int(ys[order[k]])])
        return np.array(pts)
    if strategy == "mask_skeleton":
        sk = _skeleton_points(color_blob, n_pts=3)
        if sk is not None and len(sk) > 0:
            return sk
        return np.array([[cx, cy]])
    return np.array([[cx, cy]])


# ═══════════════════════════════════════════════════════════
# 4. SAM mask generation per object × per camera
# ═══════════════════════════════════════════════════════════

def _intersect_with_color(mask: np.ndarray, bgr: np.ndarray,
                           cp: ColorPriorConfig,
                           bbox: Tuple[int, int, int, int]) -> np.ndarray:
    if not cp.enabled:
        return mask
    H, W = mask.shape
    color_m = color_mask_relaxed(bgr, cp)
    x1, y1, x2, y2 = bbox
    pad = 5
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(W, x2 + pad), min(H, y2 + pad)
    cb = np.zeros_like(color_m)
    cb[y1p:y2p, x1p:x2p] = color_m[y1p:y2p, x1p:x2p]
    if int((cb > 0).sum()) < 50:
        return mask
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cb_dil = cv2.dilate(cb, k5, iterations=2)
    inter = cv2.bitwise_and(mask, cb_dil)
    if int((inter > 0).sum()) < 50:
        return mask
    return _largest_cc(inter)


def generate_sam_masks(profile: ObjectProfile,
                       frames: List[CameraFrame],
                       predictor: SamPredictor,
                       anchor_cam: int = 1,
                       verbose: bool = True) -> List[np.ndarray]:
    """단일 물체에 대해 3-cam SAM 마스크 생성. profile 의 모든 옵션 반영."""
    glb_ext = glb_extent(Path(profile.glb))
    cp = profile.color_prior
    sam_cfg = profile.sam

    # 1. 각 카메라 색상 후보 → 3D-size 필터 (top-K 보존)
    per_cam_topk = []   # 각 cam 의 모든 scored 후보
    sr = (sam_cfg.scale_range_min, sam_cfg.scale_range_max)
    for ci, cam in enumerate(frames):
        if cp.enabled:
            cm = color_mask_strict(cam.color_bgr, cp)
            cands = best_components_sorted(cm, min_area=200, top_k=5)
            scored = filter_by_3d_size(cands, cam, glb_ext, scale_range=sr)
            per_cam_topk.append(scored)
            if verbose:
                if scored:
                    s, c = scored[0][2], scored[0][1]["centroid"]
                    print(f"    cam{ci}: top1 score={s:.3f} centroid={c} "
                          f"(of {len(scored)})")
                else:
                    print(f"    cam{ci}: no valid 3D candidate")
        else:
            per_cam_topk.append([])

    # 1b. cross-cam consistency: 모든 cam의 top-1 centroid 중 두 cam 이상이
    # 가까이 있는 그룹의 centroid 를 consensus 로 잡고, 각 cam 에서 consensus
    # 가까운 후보를 채택. peg 처럼 1위가 잘못된 어두운 영역인 경우 보정.
    per_cam = [(s[0] if s else None) for s in per_cam_topk]
    valid_top1 = [(i, x[1]["centroid"])
                  for i, x in enumerate(per_cam) if x is not None]
    if len(valid_top1) >= 2:
        # pairwise 거리 계산해 가장 가까운 페어 찾기
        best_pair = None; best_d = 1e9
        for i in range(len(valid_top1)):
            for j in range(i + 1, len(valid_top1)):
                d = float(np.linalg.norm(
                    valid_top1[i][1] - valid_top1[j][1]))
                if d < best_d:
                    best_d = d; best_pair = (valid_top1[i], valid_top1[j])
        if best_pair is not None and best_d < 0.10:  # 10cm 안에 있으면 합의
            consensus = (best_pair[0][1] + best_pair[1][1]) / 2.0
            if verbose:
                print(f"    consensus centroid={consensus} "
                      f"(pair dist={best_d*100:.1f}cm)")
            for ci in range(len(per_cam)):
                if not per_cam_topk[ci]:
                    continue
                # 해당 cam top-K 중 consensus 와 가장 가까운 후보로 교체
                best_cand = None; best_cand_d = 1e9
                for cand in per_cam_topk[ci]:
                    cd = float(np.linalg.norm(
                        cand[1]["centroid"] - consensus))
                    if cd < best_cand_d:
                        best_cand_d = cd; best_cand = cand
                if best_cand is not None and best_cand_d < 0.15:
                    if best_cand is not per_cam[ci]:
                        if verbose:
                            print(f"    cam{ci}: switch to consensus-near "
                                  f"candidate (d={best_cand_d*100:.1f}cm)")
                        per_cam[ci] = best_cand
                else:
                    # consensus 와 너무 멀면 cam 자체 후보 사용 안함
                    if verbose:
                        print(f"    cam{ci}: no candidate near consensus "
                              f"(min d={best_cand_d*100:.1f}cm) → drop")
                    per_cam[ci] = None

    # 2. anchor 결정
    anchor = per_cam[anchor_cam] if anchor_cam < len(per_cam) else None
    if anchor is None:
        valid = [(i, x) for i, x in enumerate(per_cam) if x is not None]
        if not valid:
            return [np.zeros(c.color_bgr.shape[:2], dtype=np.uint8)
                    for c in frames]
        _, anchor = max(valid, key=lambda x: x[1][2])
    centroid_anchor = anchor[1]["centroid"]
    if verbose:
        print(f"    anchor 3D: center={centroid_anchor}")

    # 3. 카메라별 SAM
    POS_CONSISTENT_M = 0.10
    final_masks: List[np.ndarray] = []
    for ci, cam in enumerate(frames):
        entry = per_cam[ci]
        H, W = cam.color_bgr.shape[:2]
        use_own = False
        if entry is not None:
            dist = float(np.linalg.norm(entry[1]["centroid"] - centroid_anchor))
            if dist < POS_CONSISTENT_M and entry[2] >= sam_cfg.reliability_threshold:
                use_own = True

        if use_own:
            mask, _, _ = entry
            ys, xs = np.where(mask > 0)
            own_bb = (int(xs.min()), int(ys.min()),
                      int(xs.max()), int(ys.max()))
            bw = own_bb[2] - own_bb[0]; bh = own_bb[3] - own_bb[1]
            if bw > W * sam_cfg.own_bbox_max_image_ratio \
                    or bh > H * sam_cfg.own_bbox_max_image_ratio:
                use_own = False
                if verbose:
                    print(f"    cam{ci}: own bbox 너무 큼 → cross-cam")

        if use_own:
            anchor_ext = anchor[1]["extent"]
            # union/intersect 양쪽 모두 GLB 가 너무 커서 bbox 폭주하지 않도록
            # scale_range_max 로 cap.
            ext_cap_hi = glb_ext * sam_cfg.scale_range_max
            if sam_cfg.bbox_combine == "intersect":
                ext_for_proj = np.maximum(anchor_ext, glb_ext * 0.20)
            else:
                ext_for_proj = np.maximum(anchor_ext, glb_ext * 0.30)
            ext_for_proj = np.minimum(ext_for_proj, ext_cap_hi)
            proj_bb = project_bbox_from_3d(
                centroid_anchor, ext_for_proj, cam, padding_m=0.012)
            if proj_bb is not None:
                if sam_cfg.bbox_combine == "intersect":
                    bbox = (max(own_bb[0], proj_bb[0]),
                            max(own_bb[1], proj_bb[1]),
                            min(own_bb[2], proj_bb[2]),
                            min(own_bb[3], proj_bb[3]))
                    if bbox[2] - bbox[0] < 20 or bbox[3] - bbox[1] < 20:
                        bbox = own_bb
                else:
                    bbox = (min(own_bb[0], proj_bb[0]),
                            min(own_bb[1], proj_bb[1]),
                            max(own_bb[2], proj_bb[2]),
                            max(own_bb[3], proj_bb[3]))
            else:
                bbox = own_bb
            source = "own"
        else:
            anchor_ext = anchor[1]["extent"]
            # cap range는 scale_range로 결정 (GLB 가 매우 클 때 bbox 폭주 방지)
            lo = max(0.02, sam_cfg.scale_range_min * 0.5)  # 작은 값 floor
            hi = sam_cfg.scale_range_max
            ext_for_proj = np.maximum(anchor_ext, glb_ext * lo)
            ext_for_proj = np.minimum(ext_for_proj, glb_ext * hi)
            bbox = project_bbox_from_3d(centroid_anchor, ext_for_proj, cam,
                                        padding_m=0.012)
            if bbox is None:
                final_masks.append(np.zeros((H, W), dtype=np.uint8))
                continue
            source = "cross-cam"

        # navy block 윗면 보강용 horizontal_constrain (own bbox 일 때)
        if profile.shape.horizontal_constrain and source == "own":
            x1, y1, x2, y2 = bbox
            h = y2 - y1; cy_b = (y1 + y2) // 2
            new_h = int(h * 0.7)
            y1 = max(0, cy_b - new_h // 2)
            y2 = min(H - 1, cy_b + new_h // 2)
            bbox = (x1, y1, x2, y2)

        # multicolor 등: bbox 패딩 확장
        if sam_cfg.bbox_pad_ratio > 0:
            x1, y1, x2, y2 = bbox
            bw = x2 - x1; bh = y2 - y1
            px = int(bw * sam_cfg.bbox_pad_ratio)
            py = int(bh * sam_cfg.bbox_pad_ratio)
            bbox = (max(0, x1 - px), max(0, y1 - py),
                    min(W - 1, x2 + px), min(H - 1, y2 + py))

        # SAM prompt
        color_blob = entry[0] if entry is not None else None
        points = _sam_points(sam_cfg.prompt_strategy, color_blob, bbox)
        with torch.no_grad():
            m = run_sam(predictor, cam.color_bgr, bbox, points)
        m = keep_nearest_component(m, bbox)
        area_raw = int((m > 0).sum())

        # post-color intersect
        if sam_cfg.post_color_intersect and cp.enabled:
            m = _intersect_with_color(m, cam.color_bgr, cp, bbox)

        # auto-refine
        if sam_cfg.auto_refine != "off":
            ref_h = cp.hue_ref if cp.enabled else 0.0
            m, _q = auto_refine_mask(
                m, cam.color_bgr, cam,
                obj_name=profile.name, glb_ext=glb_ext, ref_h=ref_h,
                verbose=False,
            )

        # navy 윗면 강제 union (object_003 호환)
        if profile.name == "object_003":
            ys2, xs2 = np.where(m > 0)
            if len(xs2) > 0:
                bb_now = (int(xs2.min()), int(ys2.min()),
                           int(xs2.max()), int(ys2.max()))
                top = _navy_top_face_mask(cam.color_bgr, bb_now)
                kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                m_dil = cv2.dilate(m, kk, iterations=1)
                top = cv2.bitwise_and(top, m_dil)
                m = cv2.bitwise_or(m, top)
                kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
                m = _largest_cc(m)

        # final close + largest CC
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
        m = _largest_cc(m)
        area = int((m > 0).sum())
        if verbose:
            print(f"    cam{ci}: {source} bbox={bbox} → "
                  f"SAM={area_raw} → final={area}")
        final_masks.append(m)
    return final_masks


# ═══════════════════════════════════════════════════════════
# 5. Pose estimation (profile-aware ICP)
# ═══════════════════════════════════════════════════════════

def _backproject_mask(mask, cam, max_depth=2.0, sample=8000):
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    K = cam.intrinsics.K
    ys, xs = np.where((mask > 0) & (depth > 0.05) & (depth < max_depth))
    if len(xs) == 0:
        return np.zeros((0, 3))
    if len(xs) > sample:
        idx = np.random.choice(len(xs), sample, replace=False)
        ys, xs = ys[idx], xs[idx]
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    return (R @ pts_cam.T).T + t


def _fuse_object_pts(masks, frames, table_n, table_d,
                      above_table_min_m=0.002, voxel=0.003):
    pts_all = []
    for m, cam in zip(masks, frames):
        if int((m > 0).sum()) < 50:
            continue
        p = _backproject_mask(m, cam)
        if len(p) == 0:
            continue
        pts_all.append(p)
    if not pts_all:
        return np.zeros((0, 3))
    P = np.concatenate(pts_all, axis=0)
    # 부호 확인: 물체 mask 평균이 위쪽에 와야 함. (p @ n) + d 의 평균이
    # 음수면 plane 부호를 flip.
    sd = (P @ table_n) + table_d
    if np.mean(sd) < 0:
        table_n = -table_n; table_d = -table_d
        sd = -sd
    pts = []
    for p in pts_all:
        d = (p @ table_n) + table_d
        keep = d > above_table_min_m
        pts.append(p[keep])
    if not pts:
        return np.zeros((0, 3))
    P = np.concatenate(pts, axis=0)
    if len(P) > 30:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P)
        pcd_ds = pcd.voxel_down_sample(voxel)
        cl, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        P = np.asarray(cl.points)
    return P


def _rotation_align(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.dot(a, b)); s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))


def _icp_register(model_pts, obj_pts, init_T, voxel=0.005, max_iter=80):
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(model_pts)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(obj_pts)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.015, max_nn=30))
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=voxel * 5, init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter))
    return res.transformation, float(res.fitness), float(res.inlier_rmse)


def _render_silhouette_fast(mesh, T, scale_xyz, model_center, cam) -> np.ndarray:
    """render_silhouette와 같지만 boolean mask 반환 (속도용)."""
    return render_silhouette(mesh, T, float(scale_xyz[0]), model_center, cam,
                              aniso=np.asarray(scale_xyz)) > 0


def _silhouette_iou(rendered: np.ndarray, observed: np.ndarray) -> float:
    inter = int((rendered & observed).sum())
    union = int((rendered | observed).sum())
    if union == 0:
        return 0.0
    return inter / union


def _silhouette_iou_avg(mesh, T, scale, model_center, frames, masks_obs,
                        cam_weights=None) -> float:
    if cam_weights is None:
        cam_weights = [1.0] * len(frames)
    aniso = np.array([scale, scale, scale])
    total = 0.0; w_sum = 0.0
    for ci, cam in enumerate(frames):
        if int((masks_obs[ci] > 0).sum()) < 50:
            continue
        rnd = _render_silhouette_fast(mesh, T, aniso, model_center, cam)
        obs = masks_obs[ci] > 0
        iou = _silhouette_iou(rnd, obs)
        total += cam_weights[ci] * iou
        w_sum += cam_weights[ci]
    return total / max(w_sum, 1e-9)


def _simplify_mesh(mesh, target_faces=2000):
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        return mesh


def _refine_render_nelder_mead(T0, scale0, mesh, model_center, frames,
                                masks_obs, table_n, cam_weights,
                                max_iter=100, anisotropic=False,
                                tilt_dof=False):
    """Nelder-Mead 리파인.

    isotropic, no tilt (5DOF):     dx, dy, dz, dyaw, dscale_log
    isotropic + tilt (7DOF):        + dpitch, droll
    anisotropic, no tilt (7DOF):    dx, dy, dz, dyaw, dlsx, dlsy, dlsz
    anisotropic + tilt (9DOF):      + dpitch, droll

    pitch/roll axes 는 target_up 에 직교하는 두 horizontal 축.
    Returns: (T_new, scale_xyz_array, iou)
    """
    from scipy.optimize import minimize
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    # horizontal axes 두 개 (pitch / roll)
    if abs(target_up[0]) < 0.9:
        h1 = np.cross(target_up, np.array([1.0, 0.0, 0.0]))
    else:
        h1 = np.cross(target_up, np.array([0.0, 1.0, 0.0]))
    h1 = h1 / (np.linalg.norm(h1) + 1e-12)
    h2 = np.cross(target_up, h1)
    h2 = h2 / (np.linalg.norm(h2) + 1e-12)

    R0 = T0[:3, :3].copy(); t0 = T0[:3, 3].copy()
    masks_bool = [(m > 0) for m in masks_obs]
    valid_cams = [ci for ci, m in enumerate(masks_bool) if int(m.sum()) >= 50]
    scale_init = (np.array(scale0, dtype=np.float64).reshape(-1)
                  if hasattr(scale0, "__len__")
                  else np.array([float(scale0)] * 3))
    if scale_init.size == 1:
        scale_init = np.full(3, float(scale_init[0]))
    if not valid_cams:
        return T0, scale_init, 0.0

    def _apply(params):
        dx, dy, dz, dyaw = params[0], params[1], params[2], params[3]
        idx = 4
        if anisotropic:
            dls = np.array(params[idx:idx + 3])
            scale_xyz = scale_init * np.exp(np.clip(dls,
                                                     np.log(0.70),
                                                     np.log(1.30)))
            idx += 3
        else:
            dls = float(params[idx])
            scale_xyz = scale_init * float(
                np.exp(np.clip(dls, np.log(0.70), np.log(1.30))))
            idx += 1
        if tilt_dof:
            dpitch, droll = params[idx], params[idx + 1]
        else:
            dpitch = droll = 0.0
        R_yaw = Rot.from_rotvec(target_up * dyaw).as_matrix()
        R = R_yaw @ R0
        if tilt_dof and (abs(dpitch) > 0 or abs(droll) > 0):
            R_pitch = Rot.from_rotvec(h1 * dpitch).as_matrix()
            R_roll = Rot.from_rotvec(h2 * droll).as_matrix()
            R = R_roll @ R_pitch @ R
        t = t0 + np.array([dx, dy, dz])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T, scale_xyz

    def _loss(params):
        T, scale_xyz = _apply(params)
        loss_sum = 0.0; w_sum = 0.0
        for ci in valid_cams:
            rnd = _render_silhouette_fast(mesh, T, scale_xyz, model_center,
                                            frames[ci])
            iou = _silhouette_iou(rnd, masks_bool[ci])
            w = cam_weights[ci]
            loss_sum += w * (1.0 - iou)
            w_sum += w
        return loss_sum / max(w_sum, 1e-9)

    base_dof = 7 if anisotropic else 5
    n_dof = base_dof + (2 if tilt_dof else 0)
    init_simplex = np.zeros((n_dof + 1, n_dof))
    init_simplex[1][0] = 0.020          # dx 2cm
    init_simplex[2][1] = 0.020          # dy 2cm
    init_simplex[3][2] = 0.020          # dz 2cm
    init_simplex[4][3] = np.radians(10.0)
    if anisotropic:
        init_simplex[5][4] = np.log(1.05)
        init_simplex[6][5] = np.log(1.05)
        init_simplex[7][6] = np.log(1.05)
    else:
        init_simplex[5][4] = np.log(1.05)
    if tilt_dof:
        init_simplex[base_dof + 1][base_dof] = np.radians(8.0)   # dpitch
        init_simplex[base_dof + 2][base_dof + 1] = np.radians(8.0)  # droll
    try:
        res = minimize(_loss, np.zeros(n_dof), method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-4,
                                "fatol": 1e-4, "initial_simplex": init_simplex,
                                "disp": False})
        T_new, scale_xyz = _apply(res.x)
        return T_new, scale_xyz, 1.0 - float(res.fun)
    except Exception:
        return T0, scale_init, 0.0


def _coarse_orientation_search(profile, mesh, model_center, model_pts,
                                obj_center, table_n, scale,
                                frames, masks_obs):
    """yaw × flip 그리드 → silhouette IoU top-K 반환."""
    target_up = -table_n / (np.linalg.norm(table_n) + 1e-12)
    sorted_idx = np.argsort(np.asarray(mesh.bounding_box.extents))
    init_ori = profile.shape.init_orientation
    if init_ori == "auto":
        ext = np.asarray(mesh.bounding_box.extents)
        ratio = float(ext[sorted_idx[2]] / max(ext[sorted_idx[0]], 1e-6))
        init_ori = "lying_flat" if ratio > 3.0 else "upright"
    align_axis = (np.eye(3)[sorted_idx[0]] if init_ori == "lying_flat"
                   else np.eye(3)[sorted_idx[2]])
    yaw_steps = 1 if profile.shape.symmetry == "yaw" \
                else max(1, profile.pose.render_orientation_grid)
    cam_weights = list(profile.pose.render_cam_weights)
    if len(cam_weights) < len(frames):
        cam_weights = cam_weights + [1.0] * (len(frames) - len(cam_weights))

    candidates = []
    for sign in profile.pose.flip_signs:
        R_align = _rotation_align(align_axis * float(sign), target_up)
        for yaw_deg in np.linspace(0, 360, yaw_steps, endpoint=False):
            R_yaw = Rot.from_rotvec(target_up * np.radians(yaw_deg)).as_matrix()
            R_init = R_yaw @ R_align
            T_init = np.eye(4)
            T_init[:3, :3] = R_init
            T_init[:3, 3] = obj_center
            iou = _silhouette_iou_avg(mesh, T_init, scale, model_center,
                                       frames, masks_obs,
                                       cam_weights=cam_weights)
            candidates.append((iou, T_init, scale, sign, yaw_deg))
    candidates.sort(key=lambda x: -x[0])
    return candidates[:profile.pose.render_topk], cam_weights


def estimate_pose(profile: ObjectProfile, masks: List[np.ndarray],
                  frames: List[CameraFrame],
                  table_n: np.ndarray, table_d: float,
                  verbose: bool = True) -> Optional[Dict[str, Any]]:
    # plane 부호 보정: 물체 마스크 평균이 plane 위쪽이 되도록
    raw_pts_all = []
    for m, cam in zip(masks, frames):
        if int((m > 0).sum()) < 50:
            continue
        p = _backproject_mask(m, cam)
        if len(p):
            raw_pts_all.append(p)
    if raw_pts_all:
        P = np.concatenate(raw_pts_all, axis=0)
        sd = (P @ table_n) + table_d
        if np.mean(sd) < 0:
            if verbose:
                print(f"  [pose] plane sign flipped (mean sd={np.mean(sd):+.3f})")
            table_n = -table_n; table_d = -table_d
    obj_pts = _fuse_object_pts(masks, frames, table_n, table_d)
    if verbose:
        print(f"  [pose] fused obj_pts={len(obj_pts)}")
    if len(obj_pts) < 50:
        return None

    glb = Path(profile.glb)
    model = normalize_glb(glb)            # CanonicalModel
    model_pts = sample_model_points(model, n=8000)   # already centered
    model_ext = np.asarray(model.extents_m)

    obj_ext = obj_pts.max(0) - obj_pts.min(0)
    obj_long = float(np.max(obj_ext))
    glb_long = float(np.max(model_ext))
    scale = float(np.clip(obj_long / glb_long, 0.05, 2.0))
    if verbose:
        print(f"  [pose] auto_scale={scale:.4f} obj_extent={obj_ext}")

    obj_center = obj_pts.mean(axis=0)
    model_pts_s = model_pts * scale
    target_up = -table_n

    # init_orientation: 어떤 모델축을 table normal에 정렬할지
    sorted_idx = np.argsort(model_ext)  # asc: [shortest, mid, longest]
    init_ori = profile.shape.init_orientation
    if init_ori == "auto":
        # 가장 긴 축이 짧은 축의 3배 이상 → lying_flat hint
        ratio = float(model_ext[sorted_idx[2]] /
                      max(model_ext[sorted_idx[0]], 1e-6))
        init_ori = "lying_flat" if ratio > 3.0 else "upright"
    if init_ori == "lying_flat":
        align_axis = np.eye(3)[sorted_idx[0]]   # shortest
    else:
        align_axis = np.eye(3)[sorted_idx[2]]   # longest

    yaw_steps = 1 if profile.shape.symmetry == "yaw" \
                else max(1, profile.shape.yaw_steps)

    method = profile.pose.method
    final_T = None; final_scale_xyz = np.array([scale, scale, scale])
    final_fit = 0.0; final_rmse = 0.0; final_iou = 0.0

    if method == "render_compare":
        # 1) coarse: orientation 그리드 → silhouette IoU 상위 K개
        topk, cam_weights = _coarse_orientation_search(
            profile, model.mesh, model.center, model_pts, obj_center,
            table_n, scale, frames, masks)
        if verbose:
            tag = " (aniso)" if profile.shape.anisotropic_scale else ""
            print(f"  [pose] coarse top-{len(topk)} IoUs{tag}: " +
                  ", ".join(f"{c[0]:.3f}" for c in topk))
        # 2) ICP 1회 + Nelder-Mead 1회 (각 후보)
        best_iou = -1.0
        mesh_render = _simplify_mesh(model.mesh,
                                       profile.pose.render_simplify_faces)
        for iou0, T_init, sc, sign, yaw_deg in topk:
            T_icp, fit, rmse = _icp_register(
                model_pts_s, obj_pts, T_init,
                voxel=profile.pose.icp_voxel,
                max_iter=profile.pose.icp_max_iter)
            T_ref, sc_ref, iou = _refine_render_nelder_mead(
                T_icp, sc, mesh_render, model.center, frames, masks,
                table_n, cam_weights,
                max_iter=profile.pose.render_max_iter,
                anisotropic=profile.shape.anisotropic_scale,
                tilt_dof=profile.pose.refine_tilt_dof)
            if iou > best_iou:
                best_iou = iou
                final_T = T_ref
                final_scale_xyz = (np.asarray(sc_ref).reshape(-1)
                                    if hasattr(sc_ref, "__len__")
                                    else np.array([sc_ref] * 3))
                if final_scale_xyz.size == 1:
                    final_scale_xyz = np.full(3, float(final_scale_xyz[0]))
                final_fit = fit; final_rmse = rmse; final_iou = iou
        if verbose:
            print(f"  [pose] render_compare best IoU={final_iou:.3f} "
                  f"(ICP fit={final_fit:.3f} rmse={final_rmse*1000:.1f}mm) "
                  f"scale_xyz=[{final_scale_xyz[0]:.3f},"
                  f"{final_scale_xyz[1]:.3f},{final_scale_xyz[2]:.3f}]")
    else:
        # icp_fitness (단순 ICP 베스트)
        best = None
        for sign in profile.pose.flip_signs:
            R_align = _rotation_align(align_axis * float(sign), target_up)
            for yaw_deg in np.linspace(0, 360, yaw_steps, endpoint=False):
                R_yaw = Rot.from_rotvec(
                    target_up * np.radians(yaw_deg)).as_matrix()
                R_init = R_yaw @ R_align
                T_init = np.eye(4)
                T_init[:3, :3] = R_init
                T_init[:3, 3] = obj_center
                T_icp, fit, rmse = _icp_register(
                    model_pts_s, obj_pts, T_init,
                    voxel=profile.pose.icp_voxel,
                    max_iter=profile.pose.icp_max_iter)
                s = fit - rmse * 8
                if best is None or s > best[0]:
                    best = (s, T_icp, fit, rmse)
        final_T, final_fit, final_rmse = best[1], best[2], best[3]
        final_scale_xyz = np.array([scale, scale, scale])
        if verbose:
            print(f"  [pose] icp best fit={final_fit:.3f} "
                  f"rmse={final_rmse*1000:.1f}mm")

    rot = final_T[:3, :3]
    eul = Rot.from_matrix(rot).as_euler("xyz", degrees=True)
    quat = Rot.from_matrix(rot).as_quat()
    final_scale_xyz = np.asarray(final_scale_xyz).reshape(-1)
    if final_scale_xyz.size == 1:
        final_scale_xyz = np.full(3, float(final_scale_xyz[0]))
    iso_scale = float(np.mean(final_scale_xyz))
    return {
        "object_name": profile.name,
        "label": profile.label or profile.name,
        "T_base_obj": final_T,
        "position_m": final_T[:3, 3],
        "quaternion_xyzw": quat,
        "euler_xyz_deg": eul,
        "scale": iso_scale,
        "anisotropic_scale_xyz": [float(x) for x in final_scale_xyz],
        "fitness": final_fit,
        "rmse": final_rmse,
        "silhouette_iou": final_iou,
        "model_center": model.center,
        "model_extents": model_ext,
        "mesh": model.mesh,
        "real_size_m": (model_ext * final_scale_xyz),
    }


# ═══════════════════════════════════════════════════════════
# 6. Visualization (single + multi-object)
# ═══════════════════════════════════════════════════════════

def render_silhouette(mesh, T_base_obj, scale, model_center, cam,
                      aniso=None) -> np.ndarray:
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    h, w = cam.intrinsics.height, cam.intrinsics.width
    if aniso is None:
        aniso = np.array([scale, scale, scale])
    V_obj = (V - model_center) * aniso
    Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
    V_base = (T_base_obj @ Vh.T)[:3].T
    T_cb = np.linalg.inv(cam.T_base_cam)
    V_cam = (T_cb @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
    z = V_cam[:, 2]; ok_v = z > 0.05
    K = cam.intrinsics.K
    u = K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]
    v = K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]
    mask = np.zeros((h, w), dtype=np.uint8)
    f_ok = ok_v[F[:, 0]] & ok_v[F[:, 1]] & ok_v[F[:, 2]]
    if f_ok.sum() == 0:
        return mask
    F_v = F[f_ok]
    tri = np.stack([
        np.stack([u[F_v[:, 0]], v[F_v[:, 0]]], axis=-1),
        np.stack([u[F_v[:, 1]], v[F_v[:, 1]]], axis=-1),
        np.stack([u[F_v[:, 2]], v[F_v[:, 2]]], axis=-1),
    ], axis=1).astype(np.int32)
    lim = 4 * max(w, h)
    inside = np.abs(tri).reshape(len(tri), -1).max(axis=1) <= lim
    for face in tri[inside]:
        cv2.fillConvexPoly(mask, face, 255)
    return mask


def overlay_color(img, mask, color, alpha=0.55):
    out = img.copy()
    if mask.max() == 0:
        return out
    cl = np.zeros_like(img); cl[:] = color
    m3 = mask > 0
    blended = cv2.addWeighted(img, 1 - alpha, cl, alpha, 0)
    out[m3] = blended[m3]
    return out


def draw_label(img, txt, org=(10, 25), scale=0.6):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def save_sam_comparison(profiles, masks_per_obj, frames, out_dir: Path):
    """Top: raw, Bottom: 모든 물체 SAM mask overlay (color 별)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_panels = []; ovl_panels = []
    for ci, cam in enumerate(frames):
        raw = cam.color_bgr.copy()
        ovl = cam.color_bgr.copy()
        for prof in profiles:
            ovl = overlay_color(ovl, masks_per_obj[prof.name][ci],
                                  tuple(prof.overlay_color_bgr), 0.5)
        draw_label(raw, f"cam{ci} raw")
        draw_label(ovl, f"cam{ci} SAM masks")
        raw_panels.append(raw); ovl_panels.append(ovl)
    grid = np.vstack([np.hstack(raw_panels), np.hstack(ovl_panels)])
    cv2.imwrite(str(out_dir / "comparison.png"), grid)


def save_pose_comparison(profiles, poses, frames, out_dir: Path):
    """Top: raw, Bottom: posed GLB silhouette overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_panels = []; ovl_panels = []
    for ci, cam in enumerate(frames):
        raw = cam.color_bgr.copy()
        ovl = cam.color_bgr.copy()
        for prof in profiles:
            pose = poses.get(prof.name)
            if pose is None:
                continue
            sil = render_silhouette(pose["mesh"], pose["T_base_obj"],
                                     pose["scale"], pose["model_center"], cam,
                                     aniso=np.asarray(pose["anisotropic_scale_xyz"]))
            ovl = overlay_color(ovl, sil, tuple(prof.overlay_color_bgr), 0.55)
        draw_label(raw, f"cam{ci} raw")
        draw_label(ovl, f"cam{ci} GLB pose")
        raw_panels.append(raw); ovl_panels.append(ovl)
    grid = np.vstack([np.hstack(raw_panels), np.hstack(ovl_panels)])
    cv2.imwrite(str(out_dir / "comparison.png"), grid)


def save_per_object_view(prof, mask, pose, frames, out_dir: Path,
                          frame_id: str):
    """v12 스타일 단일 물체 비교 (top: raw, bottom: 해당 물체 silhouette
    + 화살표 + 색상 텍스트박스)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    color = tuple(prof.overlay_color_bgr)
    raw_panels = []; ovl_panels = []
    for ci, cam in enumerate(frames):
        raw = cam.color_bgr.copy()
        draw_label(raw, f"cam{ci}  raw  (frame {frame_id})")
        raw_panels.append(raw)
        if pose is not None:
            sil = render_silhouette(
                pose["mesh"], pose["T_base_obj"], pose["scale"],
                pose["model_center"], cam,
                aniso=np.asarray(pose["anisotropic_scale_xyz"]))
        else:
            sil = mask[ci]
        img = overlay_color(cam.color_bgr.copy(), sil, color, 0.55)
        draw_label(img, f"cam{ci}  {prof.name}  (frame {frame_id})")
        ph, pw = img.shape[:2]
        ys, xs = np.where(sil > 0)
        if len(xs) > 5:
            tx, ty = int(np.mean(xs)), int(np.mean(ys))
        else:
            tx, ty = pw // 2, ph // 2
        if pose is not None:
            pos = pose["position_m"]; eul = pose["euler_xyz_deg"]
            aniso = pose["anisotropic_scale_xyz"]
            conf = pose.get("fitness", 0.0)
            lines = [
                f"{prof.name}",
                f"pos(m)=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}]",
                f"euler(deg)=[{eul[0]:+.1f},{eul[1]:+.1f},{eul[2]:+.1f}]",
                f"scale=[{aniso[0]:.3f},{aniso[1]:.3f},{aniso[2]:.3f}]",
                f"conf={conf:.3f}",
            ]
        else:
            lines = [f"{prof.name}", "(pose not available)"]
        _draw_text_box_arrow(img, lines, (tx, ty), pw, ph, color)
        ovl_panels.append(img)
    grid = np.vstack([np.hstack(raw_panels), np.hstack(ovl_panels)])
    cv2.imwrite(str(out_dir / f"comparison_{prof.name}.png"), grid)


def _draw_text_box_arrow(img, lines, target, pw, ph, color):
    font = cv2.FONT_HERSHEY_SIMPLEX; scale = 0.42; thick = 1
    pad = 6; line_h = 16
    sizes = [cv2.getTextSize(s, font, scale, thick)[0] for s in lines]
    box_w = max(sz[0] for sz in sizes) + 2 * pad
    box_h = len(lines) * line_h + 2 * pad
    tx, ty = target
    left = tx < pw / 2; top = ty < ph / 2
    x0 = (pw - box_w - 4) if left else 4
    y0 = (ph - box_h - 4) if top else 4
    x1, y1 = x0 + box_w, y0 + box_h
    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x0 + pad, y0 + pad + (i + 1) * line_h - 4),
                    font, scale, color, thick, cv2.LINE_AA)
    ax = max(x0, min(x1, tx)); ay = max(y0, min(y1, ty))
    if ax == tx and ay == ty:
        ax, ay = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.arrowedLine(img, (ax, ay), (tx, ty), color, 2, tipLength=0.15)


# ═══════════════════════════════════════════════════════════
# 7. Save pose JSON + posed GLB
# ═══════════════════════════════════════════════════════════

T_ISAAC_CV = np.array([[0, 0, 1, 0],
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]], dtype=np.float64)


def save_pose_outputs(profile: ObjectProfile, pose: Dict[str, Any],
                      frame_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if pose is None:
        return
    pos = pose["position_m"]
    quat = pose["quaternion_xyzw"]
    eul = pose["euler_xyz_deg"]
    T = pose["T_base_obj"]
    aniso = np.asarray(pose["anisotropic_scale_xyz"])
    real_size = np.asarray(pose["real_size_m"])
    j = {
        "frame_id": frame_id,
        "object_name": profile.name,
        "label": profile.label or profile.name,
        "coordinate_frame": "base (= cam0)",
        "unit": "meter",
        "position_m": [float(x) for x in pos],
        "quaternion_xyzw": [float(x) for x in quat],
        "euler_xyz_deg": [float(x) for x in eul],
        "T_base_obj": T.tolist(),
        "rotation_matrix": T[:3, :3].tolist(),
        "scale": float(pose["scale"]),
        "anisotropic_scale_xyz": [float(x) for x in aniso],
        "real_size_m": {"x": float(real_size[0]),
                          "y": float(real_size[1]),
                          "z": float(real_size[2])},
        "fitness": float(pose["fitness"]),
        "rmse": float(pose["rmse"]),
    }
    (out_dir / f"pose_{profile.name}.json").write_text(
        json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")

    # posed GLB (OpenCV + Isaac)
    mesh = pose["mesh"]
    V = np.asarray(mesh.vertices, dtype=np.float64)
    V_obj = (V - pose["model_center"]) * aniso
    Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
    V_base = (T @ Vh.T)[:3].T
    posed = trimesh.Trimesh(vertices=V_base, faces=mesh.faces, process=False)
    posed.export(str(out_dir / f"{profile.name}_posed.glb"))

    # Isaac 좌표계 + IsaacSim 호환 축 보정
    T_isaac = T_ISAAC_CV.copy()
    if profile.pose.isaac_rot_x_180:
        Rx = np.eye(4); Rx[1, 1] = -1; Rx[2, 2] = -1
        T_isaac = Rx @ T_isaac
    if profile.pose.isaac_rot_y_180:
        Ry = np.eye(4); Ry[0, 0] = -1; Ry[2, 2] = -1
        T_isaac = Ry @ T_isaac
    if profile.pose.isaac_rot_z_180:
        Rz = np.eye(4); Rz[0, 0] = -1; Rz[1, 1] = -1
        T_isaac = Rz @ T_isaac
    V_isaac = (T_isaac @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
    posed_isaac = trimesh.Trimesh(vertices=V_isaac, faces=mesh.faces,
                                   process=False)
    posed_isaac.export(str(out_dir / f"{profile.name}_posed_isaac.glb"))


# ═══════════════════════════════════════════════════════════
# 8. SAM predictor convenience
# ═══════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAM_WEIGHTS = SCRIPT_DIR / "weights" / "mobile_sam.pt"


def load_sam_predictor(weights: Optional[Path] = None) -> SamPredictor:
    weights = weights or DEFAULT_SAM_WEIGHTS
    sam = sam_model_registry["vit_t"](checkpoint=str(weights))
    sam.to("cpu"); sam.eval()
    return SamPredictor(sam)


# ═══════════════════════════════════════════════════════════
# 9. End-to-end orchestration
# ═══════════════════════════════════════════════════════════

def run_pipeline(profiles: List[ObjectProfile],
                  data_dir: Path, intr_dir: Path, frame_id: str,
                  out_root: Path,
                  predictor: Optional[SamPredictor] = None,
                  capture_subdir: str = "object_capture",
                  verbose: bool = True) -> Dict[str, Any]:
    """profiles 리스트를 받아 SAM masks + pose + viz 까지 일괄 수행."""
    if predictor is None:
        if verbose:
            print("Loading MobileSAM...")
        predictor = load_sam_predictor()

    if verbose:
        print("=" * 64)
        print(f" Multi-object pipeline — frame {frame_id}")
        print("=" * 64)

    intrinsics, extrinsics = load_calibration(data_dir, intr_dir)
    frames = load_frame(data_dir, frame_id, intrinsics, extrinsics,
                         capture_subdir=capture_subdir)

    masks_per_obj: Dict[str, List[np.ndarray]] = {}
    for prof in profiles:
        if verbose:
            print(f"\n=== SAM mask: {prof.name} ===")
        masks = generate_sam_masks(prof, frames, predictor, verbose=verbose)
        masks_per_obj[prof.name] = masks

    # save SAM masks
    sam_dir = out_root / "sam_masks" / f"{frame_id}"
    sam_dir.mkdir(parents=True, exist_ok=True)
    for prof in profiles:
        for ci, m in enumerate(masks_per_obj[prof.name]):
            cv2.imwrite(str(sam_dir / f"{prof.name}_cam{ci}.png"), m)
    save_sam_comparison(profiles, masks_per_obj, frames, sam_dir)
    if verbose:
        print(f"\n[SAM] saved: {sam_dir}")

    # pose
    if verbose:
        print(f"\n=== Pose estimation ===")
    # 어떤 profile 이라도 background_white_assist 가 활성이고
    # pose.table_plane_use_white_assist 이면, 종이 영역 안의 점만으로 평면 추정.
    use_paper_plane = any(
        p.color_prior.background_white_assist
        and p.pose.table_plane_use_white_assist
        for p in profiles)
    if use_paper_plane:
        # 가장 첫 번째 white-assist profile 의 cp 사용 (모든 profile 의 paper 영역
        # 대체로 동일).
        cp_for_paper = next(p.color_prior for p in profiles
                             if p.color_prior.background_white_assist)
        table_n, table_d = _estimate_table_plane_in_region(
            frames, cp_for_paper)
        if verbose:
            print(f"  [plane-paper] n={table_n} d={table_d:.3f}")
    else:
        table_n, table_d, *_ = estimate_table_plane(frames)
        if verbose:
            print(f"  [plane] n={table_n} d={table_d:.3f}")

    pose_dir = out_root / "pose" / f"{frame_id}"
    pose_dir.mkdir(parents=True, exist_ok=True)
    poses: Dict[str, Any] = {}
    for prof in profiles:
        if verbose:
            print(f"\n--- {prof.name} ---")
        pose = estimate_pose(prof, masks_per_obj[prof.name], frames,
                              table_n, table_d, verbose=verbose)
        poses[prof.name] = pose
        if pose is not None:
            save_pose_outputs(prof, pose, frame_id, pose_dir)
            save_per_object_view(prof, masks_per_obj[prof.name],
                                  pose, frames, pose_dir, frame_id)
    save_pose_comparison(profiles, poses, frames, pose_dir)
    summary = []
    for prof in profiles:
        pose = poses.get(prof.name)
        if pose is None:
            continue
        summary.append({
            "object_name": prof.name,
            "label": prof.label or prof.name,
            "position_m": [float(x) for x in pose["position_m"]],
            "euler_xyz_deg": [float(x) for x in pose["euler_xyz_deg"]],
            "scale": float(pose["scale"]),
            "fitness": float(pose["fitness"]),
            "rmse": float(pose["rmse"]),
        })
    (pose_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"\n[POSE] saved: {pose_dir}")
    return {"sam_dir": sam_dir, "pose_dir": pose_dir,
            "masks": masks_per_obj, "poses": poses, "frames": frames}
