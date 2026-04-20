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
    color_mask_for_object,
    connected_components_sorted,
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
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            source = "own"
        else:
            ext_for_proj = glb_ext * 0.7
            bbox = project_bbox_from_3d(
                centroid_anchor, ext_for_proj, cam, padding_m=0.015
            )
            if bbox is None:
                final_masks.append(np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8))
                print(f"    cam{ci}: bbox projection 실패 → empty")
                continue
            source = "cross-cam"

        # 특수 제약: navy block은 수평형이므로 bbox 높이를 50%로 제한
        if constraints.get("horizontal_constrain"):
            x1, y1, x2, y2 = bbox
            h = y2 - y1; cy_b = (y1 + y2) // 2
            new_h = int(h * 0.5)
            y1 = max(0, cy_b - new_h // 2)
            y2 = min(cam.intrinsics.height - 1, cy_b + new_h // 2)
            bbox = (x1, y1, x2, y2)

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
        area = int((m > 0).sum())
        print(f"    cam{ci}: {source} bbox={bbox} → SAM area={area}")
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
