#!/usr/bin/env python3
"""SAM v5: multi-view 3D consistency + 3D size filter + SAM.

v4 문제: 각 카메라 독립 선택 → cam0 빨강이 엉뚱한 빨간 영역, cam2 노랑이
왼쪽 위 잡음 선택.
v5 해결: select_multiview_seed_masks로 카메라간 3D 일관성 있는 candidate
조합을 강제. cam0 candidate를 cam1/cam2 candidate와 3D 거리 비교 → 맞는 것만 선택.

순서:
1. 각 카메라에서 HSV candidate top-K
2. 3D size filter (GLB extent 대비)로 후보 축소
3. select_multiview_seed_masks로 3D 일관성 조합 선택
4. 선택된 seed에 SAM 적용

저장: src/output/sam_masks/v5_{timestamp}_frame_{id}/
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
    select_multiview_seed_masks,
    mask_refine_with_depth_cluster,
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
    scene = trimesh.load(str(glb_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene)
    V = np.asarray(mesh.vertices)
    return V.max(0) - V.min(0)


def candidate_3d_size(mask: np.ndarray, cam) -> dict:
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    H, W = depth.shape
    K = cam.intrinsics.K
    m = (mask > 0) & (depth > 0.05) & (depth < 1.5)
    if m.sum() < 30:
        return {"extent_m": [0,0,0], "max_extent_m": 0.0, "n_pts": 0}
    ys, xs = np.where(m)
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t
    lo = np.percentile(pts_base, 5, axis=0)
    hi = np.percentile(pts_base, 95, axis=0)
    ext = hi - lo
    return {
        "extent_m": ext.tolist(),
        "max_extent_m": float(ext.max()),
        "n_pts": int(m.sum()),
    }


def filter_by_3d_size(candidates: list, cam, glb_ext: np.ndarray,
                      scale_prior=(0.3, 1.1)) -> list:
    """3D 크기와 GLB 비교해서 통과한 candidate만 반환."""
    glb_max = float(glb_ext.max())
    glb_sorted = np.sort(glb_ext)[::-1]
    min_m = glb_max * scale_prior[0]
    max_m = glb_max * scale_prior[1]
    out = []
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
        ext_sorted = np.sort(np.array(sz["extent_m"]))[::-1]
        # 3축 매칭
        ratios = [min(a/b, b/a) for a, b in zip(ext_sorted, glb_sorted)
                  if a > 1e-6 and b > 1e-6]
        if not ratios:
            continue
        size_match = float(np.mean(ratios))
        if ext_sorted[-1] > glb_sorted[-1] * 3.0:
            continue
        # 통과한 candidate만 유지
        out.append(m)
    return out


# SAM
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
    cl = np.zeros_like(img); cl[:] = color
    m = mask > 0
    out[m] = cv2.addWeighted(img, 1-alpha, cl, alpha, 0)[m]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
    return out

def label_bar(w):
    h = 24; bar = np.full((h, w, 3), 50, dtype=np.uint8); part = w // 4
    for i, (n, c) in enumerate(OBJ_COLORS_BGR.items()):
        x = i * part
        cv2.rectangle(bar, (x, 0), (x + part, h), (80, 80, 80), -1)
        cv2.rectangle(bar, (x+4, 4), (x+16, h-4), c, -1)
        cv2.putText(bar, n, (x+22, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220,220,220), 1, cv2.LINE_AA)
    return bar

def compose(top, bot, title_bot="SAM v5"):
    rows = []
    for row, t in [(top, "Original"), (bot, title_bot)]:
        labeled = [cv2.putText(img.copy(), f"cam{ci} {t}", (10, 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)
                   for ci, img in enumerate(row)]
        rows.append(np.hstack(labeled))
    return np.vstack([np.vstack(rows), label_bar(rows[0].shape[1])])


def build(frame_id: str, out_dir: Path, mask_dir: Path):
    print("Loading SAM...")
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    ext_by_obj = {n: glb_extent(DATA_DIR / f"{n}.glb") for n in OBJ_COLORS_BGR}
    for n, e in ext_by_obj.items():
        print(f"  {n} GLB extent: {e}")

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)
    table_info = estimate_table_plane(frames)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        print(f"\nProcessing {obj_name}...")
        is_cyl = OBJECT_SYMMETRY.get(obj_name, "none") == "yaw"
        glb_ext = ext_by_obj[obj_name]

        # 1) 각 카메라 HSV top-5
        per_cam_cands = []
        for ci, cam in enumerate(frames):
            color_m = color_mask_for_object(cam.color_bgr, obj_name)
            comps = connected_components_sorted(color_m, min_area=200, top_k=5)
            # 2) 3D size filter
            filtered = filter_by_3d_size(comps, cam, glb_ext)
            print(f"  cam{ci}: {len(comps)} → {len(filtered)} (3D size ok)")
            # empty면 전체 candidates 전달 (fallback)
            per_cam_cands.append(filtered if filtered else comps[:2])

        # 3) Multi-view consistency selection (v2 함수)
        seeds = select_multiview_seed_masks(frames, per_cam_cands, table_info)
        # 4) depth cluster 재투영 정제 (cross-cam 복원 포함)
        seeds = mask_refine_with_depth_cluster(
            frames, seeds, table_info, object_name=obj_name
        )
        for ci in range(3):
            a = int((seeds[ci] > 0).sum())
            print(f"  cam{ci} seed after consistency: area={a}")

        # 5) SAM refine
        final = []
        for ci, cam in enumerate(frames):
            if int((seeds[ci] > 0).sum()) < 100:
                final.append(np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8))
                print(f"  cam{ci}: seed too small, skip SAM")
                continue
            with torch.no_grad():
                m = sam_refine(predictor, cam.color_bgr, seeds[ci], is_cyl)
            final.append(m)
            print(f"  cam{ci}: sam area={int((m>0).sum())}")
        per_obj_masks[obj_name] = final

    mask_dir.mkdir(parents=True, exist_ok=True)
    for n, ms in per_obj_masks.items():
        for ci, m in enumerate(ms):
            cv2.imwrite(str(mask_dir / f"{n}_cam{ci}.png"), m)
    (mask_dir / "meta.json").write_text(json.dumps(
        {"frame_id": frame_id, "method": "HSV + 3D size + multi-view consistency + SAM"},
        indent=2))
    print(f"\nmasks saved: {mask_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
    all_ov = [cam.color_bgr.copy() for cam in frames]
    for n, ms in per_obj_masks.items():
        c = OBJ_COLORS_BGR[n]
        for ci in range(3):
            all_ov[ci] = make_overlay(all_ov[ci], ms[ci], c)
    cv2.imwrite(str(out_dir / "comparison.png"), compose(originals, all_ov))
    for n, ms in per_obj_masks.items():
        c = OBJ_COLORS_BGR[n]
        ovs = []
        for ci, cam in enumerate(frames):
            ov = make_overlay(cam.color_bgr.copy(), ms[ci], c)
            a = int((ms[ci] > 0).sum())
            cv2.putText(ov, f"{n} area={a}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
            ovs.append(ov)
        cv2.imwrite(str(out_dir / f"comparison_{n}.png"),
                    compose(originals, ovs, title_bot=f"{n}"))
    print(f"overlays saved: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("src/output/ideal_overlay") / f"sam_v5_{ts}_frame_{args.frame_id}"
    mask_dir = Path("src/output/sam_masks") / f"v5_{ts}_frame_{args.frame_id}"
    build(args.frame_id, out_dir, mask_dir)


if __name__ == "__main__":
    main()
