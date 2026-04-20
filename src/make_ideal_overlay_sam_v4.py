#!/usr/bin/env python3
"""SAM + 3D 크기 필터링: GLB 실물 크기와 일치하는 candidate만 선택.

v3의 한계: 나무상자 등 큰 물체가 shape matching을 통과.
v4 해결: 각 candidate를 depth로 3D backproject → 3D bbox 크기 계산 →
GLB unscaled extent와 비교. 크기가 크게 다르면(>2x 또는 <0.5x) reject.

저장: src/output/sam_masks/v4_{timestamp}_frame_{id}/
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (
    load_calibration, load_frame, estimate_table_plane,
    normalize_glb, OBJECT_SYMMETRY,
)
from pose_per_object_v2 import (
    color_mask_for_object,
    connected_components_sorted,
)
from mobile_sam import SamPredictor, sam_model_registry

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
SAM_WEIGHTS = SCRIPT_DIR / "weights" / "mobile_sam.pt"

OBJ_COLORS_BGR = {
    "object_001": (60, 60, 255),
    "object_002": (80, 230, 255),
    "object_003": (255, 140, 40),
    "object_004": (210, 240, 130),
}


def glb_extent(glb_path: Path) -> np.ndarray:
    """GLB unscaled 3D extent (width, height, depth in GLB units)."""
    scene = trimesh.load(str(glb_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene)
    V = np.asarray(mesh.vertices)
    return V.max(0) - V.min(0)


def candidate_3d_size(mask: np.ndarray, cam) -> dict:
    """마스크를 depth로 3D backproject → base frame에서 3D bbox 크기.

    return: {"max_extent_m": float, "volume_m3": float, "n_pts": int}
    """
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    H, W = depth.shape
    K = cam.intrinsics.K
    m = (mask > 0) & (depth > 0.05) & (depth < 1.5)
    if m.sum() < 30:
        return {"max_extent_m": 0.0, "volume_m3": 0.0, "n_pts": 0}
    ys, xs = np.where(m)
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t
    # Robust extent: 5-95 percentile
    lo = np.percentile(pts_base, 5, axis=0)
    hi = np.percentile(pts_base, 95, axis=0)
    ext = hi - lo
    return {
        "max_extent_m": float(ext.max()),
        "min_extent_m": float(ext.min()),
        "extent_m": ext.tolist(),
        "volume_m3": float(ext[0] * ext[1] * ext[2]),
        "n_pts": int(m.sum()),
    }


def select_best_candidate_3d(candidates: list, cam, glb_ext: np.ndarray,
                             scale_prior: tuple = (0.3, 1.1)) -> np.ndarray:
    """GLB 3D extent와 가장 가까운 크기의 candidate 선택.

    scale_prior: (min_scale, max_scale) — 실제 물체 크기는 GLB의
      30~110% 범위로 가정. 이 범위 밖은 reject.
    Score: 크기 매칭만 사용 (area 가중치 제거). 나무상자 같은
    다른 큰 물체는 scale > 1.5이라 자동 탈락.
    """
    if not candidates:
        return None
    glb_max = float(glb_ext.max())
    min_m = glb_max * scale_prior[0]
    max_m = glb_max * scale_prior[1]

    # Sorted GLB extent for 3-axis matching (largest to smallest)
    glb_sorted = np.sort(glb_ext)[::-1]

    scored = []
    for m in candidates:
        area = int((m > 0).sum())
        if area < 200:
            continue
        sz = candidate_3d_size(m, cam)
        if sz["n_pts"] < 30 or sz["max_extent_m"] < 1e-4:
            continue
        cand_max = sz["max_extent_m"]
        if cand_max < min_m or cand_max > max_m:
            continue

        ext = np.array(sz["extent_m"])
        ext_sorted = np.sort(ext)[::-1]
        # 축별 매칭: 가장 큰 축부터 비교
        # 각 축의 size ratio 계산 후 평균
        ratios = []
        for a, b in zip(ext_sorted, glb_sorted):
            if a > 1e-6 and b > 1e-6:
                ratios.append(min(a/b, b/a))
        if not ratios:
            continue
        # 3축 평균 size ratio (나무상자처럼 한 축이 크게 다르면 점수 하락)
        size_match = float(np.mean(ratios))
        # 작은 축이 GLB 작은 축의 2배 이상이면 reject (얇은 물체와 덩어리 구분)
        if ext_sorted[-1] > glb_sorted[-1] * 3.0:
            continue
        scored.append((size_match, m, sz))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# ───────────────────────────── SAM ───────────────────────────────────────────
def sam_refine(predictor, bgr, seed, is_cylinder):
    if int((seed > 0).sum()) < 50:
        return np.zeros_like(seed)
    ys, xs = np.where(seed > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    H, W = seed.shape
    x1 = max(0, x1 - 5); y1 = max(0, y1 - 5)
    x2 = min(W - 1, x2 + 5); y2 = min(H - 1, y2 + 5)
    if is_cylinder:
        y_min, y_max = ys.min(), ys.max()
        pts = []
        for y_t in [y_min + (y_max - y_min)//5, (y_min + y_max)//2,
                    y_max - (y_max - y_min)//5]:
            row_xs = xs[np.abs(ys - y_t) < 3]
            if len(row_xs) > 0:
                pts.append([int(np.median(row_xs)), int(y_t)])
        if not pts:
            pts = [[int(xs.mean()), int(ys.mean())]]
        points = np.array(pts)
    else:
        points = np.array([[int(xs.mean()), int(ys.mean())]])

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=np.ones(len(points), dtype=np.int32),
        box=np.array([x1, y1, x2, y2]), multimask_output=True,
    )
    best = int(np.argmax(scores))
    out = masks[best].astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n > 2:
        best_i, best_ov = -1, -1
        for i in range(1, n):
            ov = int(((labels == i) & (seed > 0)).sum())
            if ov > best_ov:
                best_ov = ov; best_i = i
        if best_i > 0:
            out = np.where(labels == best_i, 255, 0).astype(np.uint8)
    return out


# Viz
def make_overlay(img, mask, color, alpha=0.55):
    out = img.copy()
    color_layer = np.zeros_like(img); color_layer[:] = color
    m = mask > 0
    out[m] = cv2.addWeighted(img, 1.0 - alpha, color_layer, alpha, 0)[m]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
    return out

def label_bar(w):
    h = 24; bar = np.full((h, w, 3), 50, dtype=np.uint8); part = w // 4
    for i, (name, color) in enumerate(OBJ_COLORS_BGR.items()):
        x = i * part
        cv2.rectangle(bar, (x, 0), (x + part, h), (80, 80, 80), -1)
        cv2.rectangle(bar, (x + 4, 4), (x + 16, h - 4), color, -1)
        cv2.putText(bar, name, (x + 22, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return bar

def compose(top, bot, title_bot="SAM+3D size"):
    rows = []
    for row, t in [(top, "Original"), (bot, title_bot)]:
        labeled = [cv2.putText(img.copy(), f"cam{ci} {t}", (10, 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                   for ci, img in enumerate(row)]
        rows.append(np.hstack(labeled))
    return np.vstack([np.vstack(rows), label_bar(rows[0].shape[1])])


def build(frame_id: str, out_dir: Path, mask_dir: Path):
    print("Loading MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)
    print("SAM loaded")

    # GLB 크기 prior
    ext_by_obj = {}
    for name in OBJ_COLORS_BGR:
        ext_by_obj[name] = glb_extent(DATA_DIR / f"{name}.glb")
        print(f"  {name} GLB extent: {ext_by_obj[name]}")

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        print(f"\nProcessing {obj_name}...")
        is_cyl = OBJECT_SYMMETRY.get(obj_name, "none") == "yaw"
        glb_ext = ext_by_obj[obj_name]
        final = []
        for ci, cam in enumerate(frames):
            color_m = color_mask_for_object(cam.color_bgr, obj_name)
            comps = connected_components_sorted(color_m, min_area=200, top_k=5)
            # 각 candidate의 3D 크기 계산 + 필터
            print(f"  cam{ci}: {len(comps)} candidates")
            for i, c in enumerate(comps):
                sz = candidate_3d_size(c, cam)
                area = int((c > 0).sum())
                print(f"    cand{i} area={area} 3d_max={sz.get('max_extent_m', 0):.3f}m "
                      f"ext={[f'{v:.3f}' for v in sz.get('extent_m', [0,0,0])]}")
            seed = select_best_candidate_3d(comps, cam, glb_ext,
                                            scale_prior=(0.25, 1.2))
            if seed is None:
                final.append(np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8))
                print(f"  cam{ci}: no valid candidate after 3D filter")
                continue
            with torch.no_grad():
                m = sam_refine(predictor, cam.color_bgr, seed, is_cyl)
            final.append(m)
            sz = candidate_3d_size(m, cam)
            print(f"  cam{ci}: seed={int((seed>0).sum())} → sam={int((m>0).sum())} "
                  f"3d_ext={sz['max_extent_m']:.3f}m")
        per_obj_masks[obj_name] = final

    mask_dir.mkdir(parents=True, exist_ok=True)
    for name, masks in per_obj_masks.items():
        for ci, m in enumerate(masks):
            cv2.imwrite(str(mask_dir / f"{name}_cam{ci}.png"), m)
    (mask_dir / "meta.json").write_text(json.dumps(
        {"frame_id": frame_id, "method": "HSV + 3D size filter + SAM"}, indent=2))
    print(f"\nmasks saved: {mask_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
    all_ov = [cam.color_bgr.copy() for cam in frames]
    for name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[name]
        for ci in range(3):
            all_ov[ci] = make_overlay(all_ov[ci], masks[ci], color)
    cv2.imwrite(str(out_dir / "comparison.png"), compose(originals, all_ov))

    for name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[name]
        ovs = []
        for ci, cam in enumerate(frames):
            ov = make_overlay(cam.color_bgr.copy(), masks[ci], color)
            a = int((masks[ci] > 0).sum())
            cv2.putText(ov, f"{name} area={a}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            ovs.append(ov)
        cv2.imwrite(str(out_dir / f"comparison_{name}.png"),
                    compose(originals, ovs, title_bot=f"{name}"))
    print(f"overlays saved: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("src/output/ideal_overlay") / f"sam_v4_{ts}_frame_{args.frame_id}"
    mask_dir = Path("src/output/sam_masks") / f"v4_{ts}_frame_{args.frame_id}"
    build(args.frame_id, out_dir, mask_dir)


if __name__ == "__main__":
    main()
