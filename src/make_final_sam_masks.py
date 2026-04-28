#!/usr/bin/env python3
"""최종 통합 SAM 마스크 생성 스크립트.

하나의 스크립트로 final_20260420_152347 수준 결과 재현:
- HSV color mask + 3D size filtering (GLB extent 대비)
- 카메라별 신뢰도 평가 → 신뢰 카메라에서 3D centroid 추출
- 신뢰 낮은 카메라는 cross-camera bbox projection
- 물체별 특수 제약 (navy block 수평 제한, 실린더 3-point axis prompt)
- MobileSAM으로 최종 정밀 마스크

저장:
  src/output/sam_masks/final_{timestamp}_frame_{id}/{object}_cam{ci}.png
  src/output/ideal_overlay/final_{timestamp}_frame_{id}/comparison*.png
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import trimesh

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import load_calibration, load_frame, OBJECT_SYMMETRY
from pose_per_object_v2 import (
    COLOR_REF_HSV,
    color_mask_for_object,
    connected_components_sorted,
    relaxed_color_mask_for_object,
)
from mobile_sam import SamPredictor, sam_model_registry

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
SAM_WEIGHTS = SCRIPT_DIR / "weights" / "mobile_sam.pt"

# 물체별 overlay 색상 (BGR)
OBJ_COLORS_BGR = {
    "object_001": (60, 60, 255),     # 빨강
    "object_002": (80, 230, 255),    # 노랑
    "object_003": (255, 140, 40),    # 곤색
    "object_004": (210, 240, 130),   # 민트
}

# 물체별 특수 제약 (shape hint)
OBJECT_SHAPE_CONSTRAINTS = {
    "object_003": {"horizontal_constrain": True},  # navy block 가로형
}


# ═══════════════════════════════════════════════════════════
# 3D 계산 유틸
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


def filter_by_3d_size(candidates: list, cam, glb_ext: np.ndarray,
                      scale_range=(0.3, 1.1),
                      aspect_max=3.0) -> list:
    """GLB 크기 대비 3D extent가 합리적인 candidate만 반환.

    Returns: [(mask, info, score), ...] sorted by score desc.
    """
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
        # 가장 얇은 축이 GLB 얇은 축의 3배 넘으면 reject (얇은 block vs 덩어리)
        if ext_sorted[-1] > glb_sorted[-1] * aspect_max:
            continue
        # 3축 size matching score
        ratios = [min(a/b, b/a) for a, b in zip(ext_sorted, glb_sorted)
                  if a > 1e-6 and b > 1e-6]
        if not ratios:
            continue
        score = float(np.mean(ratios))
        scored.append((m, info, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


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


# ═══════════════════════════════════════════════════════════
# SAM 호출
# ═══════════════════════════════════════════════════════════

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


def _mask_3d_extent_axes(mask: np.ndarray, cam) -> Optional[np.ndarray]:
    """mask 픽셀을 base 좌표로 backproject → 3D extent (sorted desc)."""
    info = mask_3d_info(mask, cam)
    if info is None:
        return None
    return np.sort(info["extent"])[::-1]


def _color_homogeneity(mask: np.ndarray, bgr: np.ndarray, ref_h: float) -> float:
    """mask 내부 hue가 ref_h 와 가까울수록 1, 멀수록 0."""
    if int((mask > 0).sum()) < 30:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    h = hsv[ys, xs, 0].astype(np.float32)
    diff = np.minimum(np.abs(h - ref_h), 180.0 - np.abs(h - ref_h))
    # 평균 hue 거리 → 0~30 을 1~0 으로 매핑
    score = float(np.clip(1.0 - np.mean(diff) / 30.0, 0.0, 1.0))
    return score


def _compactness(mask: np.ndarray) -> float:
    """area / convex_hull_area (1.0 에 가까울수록 단단한 block-like 형태)."""
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
    """마스크 품질 점수 (3D extent + 색상 균질성 + compactness)."""
    info = mask_3d_info(mask, cam)
    extent_score = 0.0
    extent_axes = None
    over_axis = False
    under_axis = False
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
        # 가중 평균: 3D extent 50%, 색 30%, compactness 20%
        total = 0.5 * extent_score + 0.3 * color_score + 0.2 * compact
    return {
        "score": total,
        "extent_score": extent_score,
        "color_score": color_score,
        "compactness": compact,
        "area": area,
        "extent_axes_m": extent_axes.tolist() if extent_axes is not None else None,
        "over_axis": over_axis,
        "under_axis": under_axis,
    }


def _tighten_with_grabcut(mask: np.ndarray, bgr: np.ndarray,
                          shrink_px: int = 5,
                          extra_fgd: Optional[np.ndarray] = None) -> np.ndarray:
    """GrabCut으로 마스크를 image edge에 맞춰 tightening.

    - 마스크 erode → PR_FGD seed
    - 마스크 ↔ dilated 사이 → PR_BGD seed
    - dilated 외부 → BGD
    - extra_fgd: 추가로 강제 FGD로 마킹할 영역 (예: navy 윗면 색상 mask)
    """
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
        # 강제 FGD: 색상 mask로 표시된 어두운 윗면 등 grabcut 이 놓치는 영역
        gc_mask[extra_fgd > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return mask
    out = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    if int((out > 0).sum()) < 100:
        return mask
    return _largest_cc(out)


def _halo_strip(mask: np.ndarray, erode_px: int = 5,
                dilate_back_px: int = 3) -> np.ndarray:
    """가장자리 halo(블록 주변 색상 같은 잡티) + tail 제거.

    erode → 가장 큰 CC → 약간 작은 dilate. 본체는 유지하면서 주변/꼬리
    픽셀만 떼어냄.
    """
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
    # 원래 mask 영역 안에서만 유지 → 본체 영역에서 tail/halo만 제거
    out = cv2.bitwise_and(out, mask)
    out = _largest_cc(out)
    return out


def _open_largest(mask: np.ndarray, k_size: int = 5) -> np.ndarray:
    """morph_open + 가장 큰 CC: 작은 fragments/tail 일괄 제거."""
    if int((mask > 0).sum()) < 80:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    op = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    op = _largest_cc(op)
    if int((op > 0).sum()) < 80:
        return mask
    return op


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """floodfill 기반 닫힌 hole 채움."""
    if int((mask > 0).sum()) < 80:
        return mask
    h, w = mask.shape
    ff = mask.copy()
    mk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)


def _convex_hull_fill(mask: np.ndarray) -> np.ndarray:
    """가장 큰 contour의 convex hull로 마스크 채움 (열린 U자형, 분산형 해결)."""
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


def auto_refine_mask(initial: np.ndarray, bgr: np.ndarray, cam, obj_name: str,
                     glb_ext: np.ndarray, ref_h: float,
                     max_iter: int = 10,
                     verbose: bool = True) -> tuple:
    """반복 정제: 3D extent + 색 + compactness 점수가 plateau 될 때까지.

    행동 풀 (현재 점수와 약점에 따라 그때그때 선택):
      - close+largest : 분리된 fragments 합치기 (다중 component)
      - halo_strip    : block 주변 halo 제거 (compactness 낮은 단일 CC)
      - grabcut       : image edge 에 snap (color_score 낮음 또는 미세 over)
      - erode         : extent over (3D 너무 큼)
      - dilate        : extent under (3D 너무 작음)

    각 후보 액션의 결과 점수를 비교, 가장 좋은 것을 채택.
    """
    cur = initial.copy()
    best = initial.copy()
    best_q = evaluate_mask_quality(best, bgr, cam, obj_name, glb_ext, ref_h)
    initial_area = best_q["area"]
    # 초기 영역의 60% 미만으로는 어떤 경우에도 줄이지 않음 (점수 fooling 방지)
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

    for it in range(1, max_iter + 1):
        q = evaluate_mask_quality(cur, bgr, cam, obj_name, glb_ext, ref_h)
        if q["score"] >= 0.88:
            break
        # 후보 액션들 모두 시도 → 점수가 올라간 것 중 최고 채택
        candidates = []
        # multi-component → close
        n, _, _, _ = cv2.connectedComponentsWithStats(cur, connectivity=8)
        if n - 1 >= 2:
            c = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, k7)
            c = _largest_cc(c)
            candidates.append(("close+largest", c))
        # 속 빈 형태 (compactness 매우 낮음) → 구멍 채움 + convex hull fill
        if q["compactness"] < 0.6:
            candidates.append(("fill_holes", _fill_holes(cur)))
            candidates.append(("hull_fill", _convex_hull_fill(cur)))
        # 단일 CC지만 compactness 낮음 → halo strip + open_largest 후보
        if q["compactness"] < 0.85 and used_halo < 3:
            candidates.append(("halo_strip4",
                                _halo_strip(cur, erode_px=4, dilate_back_px=3)))
            candidates.append(("halo_strip6",
                                _halo_strip(cur, erode_px=6, dilate_back_px=4)))
            candidates.append(("open5", _open_largest(cur, k_size=5)))
            candidates.append(("open7", _open_largest(cur, k_size=7)))
        # color/edge alignment 약함 → grabcut
        if (q["color_score"] < 0.9 or q["compactness"] < 0.9) and used_grabcut < 2:
            extra_fgd = None
            if obj_name == "object_003":
                # navy 윗면이 어두워 grabcut 이 BG로 분류하지 않게 강제 FGD seed
                ys, xs = np.where(cur > 0)
                if len(xs) > 0:
                    bb = (int(xs.min()), int(ys.min()),
                          int(xs.max()), int(ys.max()))
                    extra_fgd = _navy_top_face_mask(bgr, bb)
            c = _tighten_with_grabcut(cur, bgr, shrink_px=4,
                                      extra_fgd=extra_fgd)
            candidates.append(("grabcut", c))
        # extent over
        if q["over_axis"]:
            c = cv2.erode(cur, k3, iterations=1)
            c = _largest_cc(c)
            candidates.append(("erode", c))
        # extent under
        if q["under_axis"] and it <= 2:
            c = cv2.dilate(cur, k3, iterations=1)
            candidates.append(("dilate", c))

        # area 가드: 절대 하한 (initial × 0.6) 와 현재 비율 floor 중 큰 값.
        # compactness 가 매우 낮으면(<0.4) 비율 floor 만 살짝 낮춤.
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
        if best_action_q["score"] > best_q["score"] + 1e-3:
            best = best_action_mask.copy()
            best_q = best_action_q
            cur = best_action_mask
            no_improve = 0
        else:
            no_improve += 1
            cur = best_action_mask
        if no_improve >= 2:
            break

    if verbose:
        print(f"      [auto] best score={best_q['score']:.3f} "
              f"area={best_q['area']}")
    return best, best_q


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


def _navy_top_face_mask(bgr: np.ndarray, bbox: tuple) -> np.ndarray:
    """곤색 블록의 어두운 윗면(저채도/저명도 청회색)을 bbox 내부에서만 추출.

    중요: hue 범위를 mint(~90)와 분리하기 위해 hue 100~130 으로 제한,
    v_max 도 75로 강화 (mint v_min=110 보다 한참 아래).
    """
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
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
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3)
    return out


def filter_by_color(mask: np.ndarray, bgr: np.ndarray, obj_name: str,
                    bbox: tuple,
                    hue_radius: float = 25.0,
                    tighten: bool = False) -> np.ndarray:
    """SAM 결과에서 색상이 안 맞는 픽셀을 제거.

    - bbox 내부 relaxed color mask와 AND → 인접 다른 색 노이즈 컷
    - tighten=True 면 작은 open(3x3) 으로 경계 잡음 제거 (실린더용, erode 는 안함)
    - 곤색(object_003)은 윗면(저명도 100~130 hue)만 추가로 합집합
    - 결과는 SAM 마스크의 dilated 영역을 절대 넘지 않도록 cap (인접 다른 물체 차단)
    """
    if mask.max() == 0:
        return mask
    H, W = mask.shape[:2]
    color_m = relaxed_color_mask_for_object(bgr, obj_name, hue_radius=hue_radius)
    x1, y1, x2, y2 = bbox
    pad = 4
    x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
    x2p = min(W, x2 + pad); y2p = min(H, y2 + pad)
    color_in_bbox = np.zeros_like(color_m)
    color_in_bbox[y1p:y2p, x1p:x2p] = color_m[y1p:y2p, x1p:x2p]
    if obj_name == "object_003":
        # mint hue 와 분리되는 어두운 청회색만 추가
        top = _navy_top_face_mask(bgr, bbox)
        color_in_bbox = cv2.bitwise_or(color_in_bbox, top)
    if int((color_in_bbox > 0).sum()) < 50:
        return mask
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_dil = cv2.dilate(color_in_bbox, k5, iterations=2)
    inter = cv2.bitwise_and(mask, color_dil)
    if int((inter > 0).sum()) < 50:
        # SAM이 색상 픽셀을 거의 담지 못함 → bbox 내 color mask를 사용,
        # 단 SAM 마스크의 dilated 영역으로만 제한하여 인접 다른 물체로 새지 않게.
        sam_dil = cv2.dilate(mask, k5, iterations=2)
        out = cv2.bitwise_and(color_in_bbox, sam_dil)
    else:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(inter, connectivity=8)
        if n <= 1:
            out = inter
        else:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best = 1 + int(np.argmax(areas))
            out = np.where(labels == best, 255, 0).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if tighten:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k3)
    if obj_name == "object_003":
        # SAM 마스크 + dilation 으로 cap → 윗면 색상만 사용해 외부로 새는 거 차단.
        sam_dil = cv2.dilate(mask, k5, iterations=3)
        union_src = cv2.bitwise_or(out, color_in_bbox)
        union_src = cv2.bitwise_and(union_src, sam_dil)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        union = cv2.morphologyEx(union_src, cv2.MORPH_CLOSE, k7)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
        if n2 > 1:
            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            best_i, best_d = -1, 1e9
            for i in range(1, n2):
                if st2[i, cv2.CC_STAT_AREA] < 200:
                    continue
                cxi = st2[i, cv2.CC_STAT_LEFT] + st2[i, cv2.CC_STAT_WIDTH] / 2
                cyi = st2[i, cv2.CC_STAT_TOP] + st2[i, cv2.CC_STAT_HEIGHT] / 2
                d = (cxi - cx) ** 2 + (cyi - cy) ** 2
                if d < best_d:
                    best_d = d; best_i = i
            if best_i >= 0:
                out = np.where(lab2 == best_i, 255, 0).astype(np.uint8)
        else:
            out = union
    return out


# ═══════════════════════════════════════════════════════════
# 물체별 처리 (핵심 로직)
# ═══════════════════════════════════════════════════════════

def process_object(obj_name: str, frames, glb_path: Path,
                   predictor: SamPredictor, reliability_threshold: float = 0.35,
                   anchor_cam: int = 1):
    """1 물체 × 3 카메라 → 3 마스크.

    anchor_cam: cross-cam projection 시 centroid anchor로 우선 사용할 카메라 (기본 cam1).
    cam1이 중앙 시야라 대부분 경우 가장 신뢰 가능.
    """
    glb_ext = glb_extent(glb_path)
    is_cyl = OBJECT_SYMMETRY.get(obj_name, "none") == "yaw"
    constraints = OBJECT_SHAPE_CONSTRAINTS.get(obj_name, {})

    # 1) 각 카메라: HSV candidates + 3D size filter → 신뢰도 순 정렬
    per_cam_reliable = []
    for ci, cam in enumerate(frames):
        color_m = color_mask_for_object(cam.color_bgr, obj_name)
        cands = connected_components_sorted(color_m, min_area=200, top_k=5)
        scored = filter_by_3d_size(cands, cam, glb_ext)
        per_cam_reliable.append(scored[0] if scored else None)
        if scored:
            mask, info, score = scored[0]
            print(f"    cam{ci}: score={score:.3f} centroid={info['centroid']}")
        else:
            print(f"    cam{ci}: no valid 3D candidate")

    # 2) Anchor centroid: anchor_cam 우선, 없으면 다른 cam 중 max score
    anchor_entry = per_cam_reliable[anchor_cam] if anchor_cam < len(per_cam_reliable) else None
    if anchor_entry is None:
        valid = [(i, x) for i, x in enumerate(per_cam_reliable) if x is not None]
        if not valid:
            return [np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8) for cam in frames]
        _, anchor_entry = max(valid, key=lambda x: x[1][2])

    centroid_anchor = anchor_entry[1]["centroid"]
    print(f"    anchor 3D: center={centroid_anchor}")

    # 3) 각 카메라:
    #   - candidate 3D centroid가 anchor centroid와 너무 멀면(>10cm) reject → cross-cam
    #   - score 낮아도 위치 일치하면 "own" 유지
    POS_CONSISTENT_M = 0.10
    final_masks = []
    for ci, cam in enumerate(frames):
        entry = per_cam_reliable[ci]
        use_own = False
        if entry is not None:
            dist = float(np.linalg.norm(entry[1]["centroid"] - centroid_anchor))
            if dist < POS_CONSISTENT_M and entry[2] >= reliability_threshold:
                use_own = True
            else:
                print(f"    cam{ci}: own rejected (dist={dist*100:.1f}cm, "
                      f"score={entry[2]:.2f})")

        if use_own:
            mask, info, score = entry
            ys, xs = np.where(mask > 0)
            own_bb = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            # own bbox ∪ anchor 3D centroid + GLB extent projection.
            # 색상 mask가 어두운 윗면이나 음영을 빠뜨려 잘렸을 때 보강.
            proj_bb = project_bbox_from_3d(
                centroid_anchor, glb_ext, cam, padding_m=0.012
            )
            if proj_bb is not None:
                bbox = (
                    min(own_bb[0], proj_bb[0]),
                    min(own_bb[1], proj_bb[1]),
                    max(own_bb[2], proj_bb[2]),
                    max(own_bb[3], proj_bb[3]),
                )
            else:
                bbox = own_bb
            source = "own"
        else:
            # cross-cam projection: anchor가 본 실제 3D extent로 projection
            # (anchor의 mask_3d_info → 픽셀-깊이 기반 percentile extent).
            # GLB 전체 사용시 over-extension, 0.7× 사용시 navy 같은 긴 형상 잘림.
            anchor_ext = anchor_entry[1]["extent"]
            ext_for_proj = np.maximum(anchor_ext, glb_ext * 0.45)
            ext_for_proj = np.minimum(ext_for_proj, glb_ext * 1.05)
            bbox = project_bbox_from_3d(
                centroid_anchor, ext_for_proj, cam, padding_m=0.012
            )
            if bbox is None:
                final_masks.append(np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8))
                print(f"    cam{ci}: bbox projection 실패 → empty")
                continue
            source = "cross-cam"

        # navy 블록 horizontal_constrain은 own_bb + projected_bb union에서는 비활성.
        # (윗면 어두운 부분이 잘렸던 원인이라 그대로 두고 SAM이 판단).

        # SAM prompt 구성
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        if is_cyl and source == "reliable":
            # 실린더: axis 3점
            points = cylinder_axis_points(entry[0])
        else:
            points = np.array([[cx, cy]])

        with torch.no_grad():
            m = run_sam(predictor, cam.color_bgr, bbox, points)
        m = keep_nearest_component(m, bbox)
        area_raw = int((m > 0).sum())
        tighten = is_cyl
        m = filter_by_color(m, cam.color_bgr, obj_name, bbox, tighten=tighten)
        area_cf = int((m > 0).sum())
        # auto-refine: 3D extent + 색 + compactness 점수가 plateau 될 때까지
        ref_h = COLOR_REF_HSV.get(obj_name, (0.0,))[0]
        m, q = auto_refine_mask(m, cam.color_bgr, cam, obj_name, glb_ext, ref_h)
        # navy: auto-refine 이 윗면 어두운 부분을 잘라내는 경향 → 마무리에 강제 union
        if obj_name == "object_003":
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                bb_now = (int(xs.min()), int(ys.min()),
                          int(xs.max()), int(ys.max()))
                top = _navy_top_face_mask(cam.color_bgr, bb_now)
                # 마스크 dilation 영역 안의 top 만 OR (외부로 새지 않게)
                kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                m_dil = cv2.dilate(m, kk, iterations=1)
                top = cv2.bitwise_and(top, m_dil)
                m = cv2.bitwise_or(m, top)
                kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
                m = _largest_cc(m)
        area = int((m > 0).sum())
        print(f"    cam{ci}: {source} bbox={bbox} → "
              f"SAM={area_raw} → color={area_cf} → auto={area} "
              f"(score={q['score']:.2f})")
        final_masks.append(m)

    return final_masks


# ═══════════════════════════════════════════════════════════
# Overlay 시각화
# ═══════════════════════════════════════════════════════════

def make_overlay(img, mask, color, alpha=0.55):
    out = img.copy(); cl = np.zeros_like(img); cl[:] = color
    m = mask > 0
    out[m] = cv2.addWeighted(img, 1-alpha, cl, alpha, 0)[m]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
    return out


def annotate(img, txt):
    out = img.copy()
    cv2.putText(out, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2, cv2.LINE_AA)
    return out


def label_bar(w):
    h = 24
    bar = np.full((h, w, 3), 50, dtype=np.uint8)
    part = w // 4
    for i, (n, c) in enumerate(OBJ_COLORS_BGR.items()):
        x = i * part
        cv2.rectangle(bar, (x, 0), (x + part, h), (80, 80, 80), -1)
        cv2.rectangle(bar, (x+4, 4), (x+16, h-4), c, -1)
        cv2.putText(bar, n, (x+22, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def render_overlays(per_obj_masks, frames, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]

    # 전체 overlay
    all_ov = [cam.color_bgr.copy() for cam in frames]
    for obj, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj]
        for ci in range(3):
            all_ov[ci] = make_overlay(all_ov[ci], masks[ci], color)
    top = np.hstack([annotate(originals[ci], f"cam{ci} Original") for ci in range(3)])
    bot = np.hstack([annotate(all_ov[ci], f"cam{ci} Final SAM") for ci in range(3)])
    grid = np.vstack([top, bot, label_bar(top.shape[1])])
    cv2.imwrite(str(out_dir / "comparison.png"), grid)
    print(f"saved: {out_dir/'comparison.png'}")

    # 물체별 overlay
    for obj, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj]
        obj_ov = []
        for ci, cam in enumerate(frames):
            ov = make_overlay(cam.color_bgr.copy(), masks[ci], color)
            a = int((masks[ci] > 0).sum())
            cv2.putText(ov, f"{obj} area={a}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            obj_ov.append(annotate(ov, f"cam{ci} {obj}"))
        topr = np.hstack([annotate(originals[ci], f"cam{ci}") for ci in range(3)])
        botr = np.hstack(obj_ov)
        cv2.imwrite(str(out_dir / f"comparison_{obj}.png"),
                    np.vstack([topr, botr, label_bar(topr.shape[1])]))
    print(f"per-object overlays saved: {out_dir}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def build(frame_id: str, mask_dir: Path, overlay_dir: Path,
          reliability: float = 0.6):
    print("Loading MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        print(f"\n=== {obj_name} ===")
        glb = DATA_DIR / f"{obj_name}.glb"
        masks = process_object(obj_name, frames, glb, predictor,
                               reliability_threshold=reliability)
        per_obj_masks[obj_name] = masks

    # 마스크 저장
    mask_dir.mkdir(parents=True, exist_ok=True)
    for obj, masks in per_obj_masks.items():
        for ci, m in enumerate(masks):
            cv2.imwrite(str(mask_dir / f"{obj}_cam{ci}.png"), m)
    (mask_dir / "meta.json").write_text(json.dumps({
        "frame_id": frame_id,
        "method": "unified: HSV + 3D size filter + cross-cam projection + SAM",
        "reliability_threshold": reliability,
        "constraints": OBJECT_SHAPE_CONSTRAINTS,
    }, indent=2))
    print(f"\nmasks saved: {mask_dir}")

    # Overlay
    render_overlays(per_obj_masks, frames, overlay_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--reliability", type=float, default=0.35,
                    help="3D size matching score threshold for 'reliable' camera.")
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_dir = Path("src/output/sam_masks") / f"final_{ts}_frame_{args.frame_id}"
    overlay_dir = Path("src/output/ideal_overlay") / f"final_{ts}_frame_{args.frame_id}"
    build(args.frame_id, mask_dir, overlay_dir, args.reliability)


if __name__ == "__main__":
    main()
