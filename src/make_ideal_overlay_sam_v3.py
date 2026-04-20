#!/usr/bin/env python3
"""SAM + GLB 실루엣 기반 shape matching.

HSV만으로는 cam0/cam2의 유사 색상 배경(노란 나무상자, 민트 스티커 등)이
잘못 잡힘. 해결:
  1. HSV 색상 mask에서 top-K connected components 추출
  2. GLB를 여러 yaw에서 렌더해 "실루엣 템플릿" 세트 생성
  3. 각 candidate의 shape descriptor(Hu moments, aspect, solidity)를
     GLB 템플릿과 비교 → 가장 유사한 component 선택
  4. 선택된 seed에 SAM 적용 → 정밀 mask

저장: src/output/sam_masks/v3_{timestamp}_frame_{id}/
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


# ───────────────────────────── Shape descriptors ──────────────────────────────
def mask_shape_desc(mask: np.ndarray) -> dict:
    """Compact shape descriptor for a binary mask."""
    if int(np.count_nonzero(mask)) < 30:
        return {"area": 0}
    ys, xs = np.where(mask > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    area = int(mask.sum() > 0) and int((mask > 0).sum())
    bbox_area = w * h
    solidity_bbox = area / bbox_area
    aspect = h / w  # >1: tall, <1: wide
    # Hu moments
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"area": 0}
    cnt = max(cnts, key=cv2.contourArea)
    m = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()
    # log-transform for comparability
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    # Convex hull solidity
    hull = cv2.convexHull(cnt)
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity_hull = area / hull_area
    return {
        "area": area, "w": w, "h": h, "aspect": aspect,
        "solidity_bbox": solidity_bbox, "solidity_hull": solidity_hull,
        "hu": hu, "cnt": cnt, "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
    }


def render_glb_silhouette_templates(glb_path: Path, n_views: int = 12) -> list:
    """GLB를 여러 yaw에서 렌더해 실루엣 템플릿 descriptor 생성.

    orthographic projection으로 다양한 각도의 실루엣 모양 확보.
    """
    scene = trimesh.load(str(glb_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene)
    V = np.asarray(mesh.vertices)
    V -= V.mean(0)

    templates = []
    for i in range(n_views):
        yaw = 2 * np.pi * i / n_views
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
        pts = V @ R.T
        # project to xy plane (front view)
        xs, ys = pts[:, 0], pts[:, 1]
        # normalize to 100x100 canvas
        x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
        if (x1 - x0) < 1e-6 or (y1 - y0) < 1e-6:
            continue
        scale = 80 / max(x1 - x0, y1 - y0)
        px = ((xs - (x0 + x1)/2) * scale + 50).astype(np.int32)
        py = ((ys - (y0 + y1)/2) * scale + 50).astype(np.int32)
        canvas = np.zeros((100, 100), dtype=np.uint8)
        for (x, y) in zip(px, py):
            if 0 <= x < 100 and 0 <= y < 100:
                canvas[y, x] = 255
        # fill holes
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k5)
        cv2.floodFill(canvas, None, (0, 0), 128)
        canvas = ((canvas == 255) | (canvas == 0)).astype(np.uint8) * 255
        # Canvas flipped Y (image coords)
        canvas = np.flipud(canvas)
        desc = mask_shape_desc(canvas)
        if desc.get("area", 0) > 200:
            templates.append(desc)
    return templates


def shape_distance(cand: dict, templates: list) -> float:
    """Hu moments + aspect + solidity 비교 (낮을수록 유사)."""
    if cand.get("area", 0) == 0 or not templates:
        return 1e9
    best = 1e9
    for t in templates:
        # Hu moment 거리
        hu_d = float(np.sum(np.abs(cand["hu"] - t["hu"])))
        # aspect 비율 (log 차이)
        asp_d = abs(np.log(cand["aspect"] / t["aspect"]))
        # solidity 차이
        sol_d = abs(cand["solidity_hull"] - t["solidity_hull"])
        dist = 0.5 * hu_d + 1.5 * asp_d + 2.0 * sol_d
        if dist < best:
            best = dist
    return best


def select_best_candidate(candidates: list, templates: list) -> np.ndarray:
    """Candidate 중 GLB 실루엣 템플릿에 가장 가까운 것 선택.

    candidates: list of mask arrays (HSV top-K components)
    """
    if not candidates:
        return None
    scored = []
    for m in candidates:
        desc = mask_shape_desc(m)
        if desc.get("area", 0) < 200:
            continue
        d = shape_distance(desc, templates)
        scored.append((d, m, desc))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


# ───────────────────────────── SAM ───────────────────────────────────────────
def sam_refine(predictor: SamPredictor, bgr, seed, is_cylinder):
    if int(np.count_nonzero(seed)) < 50:
        return np.zeros_like(seed)
    ys, xs = np.where(seed > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    H, W = seed.shape
    x1 = max(0, x1 - 5); y1 = max(0, y1 - 5)
    x2 = min(W - 1, x2 + 5); y2 = min(H - 1, y2 + 5)

    # point prompt
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
    # Keep largest connected component overlapping seed
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


# ───────────────────────────── Visualization ─────────────────────────────────
def make_overlay(img, mask, color, alpha=0.55):
    out = img.copy()
    color_layer = np.zeros_like(img); color_layer[:] = color
    m = mask > 0
    out[m] = cv2.addWeighted(img, 1.0 - alpha, color_layer, alpha, 0)[m]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
    return out


def label_bar(w):
    h = 24
    bar = np.full((h, w, 3), 50, dtype=np.uint8)
    part = w // 4
    for i, (name, color) in enumerate(OBJ_COLORS_BGR.items()):
        x = i * part
        cv2.rectangle(bar, (x, 0), (x + part, h), (80, 80, 80), -1)
        cv2.rectangle(bar, (x + 4, 4), (x + 16, h - 4), color, -1)
        cv2.putText(bar, name, (x + 22, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def compose(top, bot, title_bot="SAM+Shape"):
    rows = []
    for row, t in [(top, "Original"), (bot, title_bot)]:
        labeled = [cv2.putText(img.copy(), f"cam{ci} {t}", (10, 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                   for ci, img in enumerate(row)]
        rows.append(np.hstack(labeled))
    return np.vstack([np.vstack(rows), label_bar(rows[0].shape[1])])


# ───────────────────────────── Main ──────────────────────────────────────────
def build(frame_id: str, out_dir: Path, mask_dir: Path):
    print("Loading MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)
    print("SAM loaded")

    # Pre-compute GLB silhouette templates
    templates_by_obj = {}
    for obj_name in OBJ_COLORS_BGR:
        glb = DATA_DIR / f"{obj_name}.glb"
        templates_by_obj[obj_name] = render_glb_silhouette_templates(glb)
        print(f"  {obj_name}: {len(templates_by_obj[obj_name])} shape templates")

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        print(f"\nProcessing {obj_name}...")
        is_cyl = OBJECT_SYMMETRY.get(obj_name, "none") == "yaw"
        templates = templates_by_obj[obj_name]
        final = []
        for ci, cam in enumerate(frames):
            # HSV top-K components
            color_m = color_mask_for_object(cam.color_bgr, obj_name)
            comps = connected_components_sorted(color_m, min_area=200, top_k=5)
            # Shape-based selection
            seed = select_best_candidate(comps, templates)
            if seed is None:
                final.append(np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8))
                print(f"  cam{ci}: no valid candidate")
                continue
            desc = mask_shape_desc(seed)
            sd = shape_distance(desc, templates)
            print(f"  cam{ci}: seed area={desc['area']} asp={desc['aspect']:.2f} "
                  f"sol={desc['solidity_hull']:.2f} shape_dist={sd:.2f}")
            # SAM refine
            with torch.no_grad():
                m = sam_refine(predictor, cam.color_bgr, seed, is_cyl)
            final.append(m)
            print(f"         sam area={int((m>0).sum())}")
        per_obj_masks[obj_name] = final

    # Save masks
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name, masks in per_obj_masks.items():
        for ci, m in enumerate(masks):
            cv2.imwrite(str(mask_dir / f"{name}_cam{ci}.png"), m)
    (mask_dir / "meta.json").write_text(json.dumps(
        {"frame_id": frame_id, "method": "HSV + shape template + SAM"}, indent=2))
    print(f"\nmasks saved: {mask_dir}")

    # Overlay
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
            area = int(np.count_nonzero(masks[ci]))
            cv2.putText(ov, f"{name} area={area}", (10, 50),
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
    out_dir = Path("src/output/ideal_overlay") / f"sam_v3_{ts}_frame_{args.frame_id}"
    mask_dir = Path("src/output/sam_masks") / f"v3_{ts}_frame_{args.frame_id}"
    build(args.frame_id, out_dir, mask_dir)


if __name__ == "__main__":
    main()
