#!/usr/bin/env python3
"""SAM 기반 마스크 생성 (실린더 특화 + fallback).

개선사항 (v2):
- 빈 seed 시 color mask top-1 blob을 fallback seed로 사용
- 실린더(yaw-sym)는 3 point prompt (top/center/bottom) + bbox
- bbox padding 5px
- 마스크 disk 저장 → pose 추정 파이프라인 연동 가능

저장: src/output/sam_masks/{timestamp}_frame_{id}/{object}_cam{ci}.png
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

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import load_calibration, load_frame, estimate_table_plane, OBJECT_SYMMETRY
from pose_per_object_v2 import (
    color_mask_for_object,
    connected_components_sorted,
    select_multiview_seed_masks,
    mask_refine_with_depth_cluster,
    best_connected_component,
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


def get_seed_with_fallback(frames, obj_name, table_info):
    """seed 획득: v2 multi-view selection → 실패 시 color mask top-1 fallback."""
    color_masks = [color_mask_for_object(cam.color_bgr, obj_name) for cam in frames]
    cands = [connected_components_sorted(m, top_k=3) for m in color_masks]
    seed_masks = select_multiview_seed_masks(frames, cands, table_info)
    seed_masks = mask_refine_with_depth_cluster(
        frames, seed_masks, table_info, object_name=obj_name
    )

    # Fallback: 비어있는 카메라는 color mask 최대 blob 사용
    for ci in range(len(frames)):
        if int(np.count_nonzero(seed_masks[ci])) < 100:
            fallback = best_connected_component(color_masks[ci], min_area=200)
            if int(np.count_nonzero(fallback)) >= 200:
                print(f"    [fallback] {obj_name} cam{ci}: color-mask top-1 사용")
                seed_masks[ci] = fallback
    return seed_masks


def cylinder_point_prompts(seed: np.ndarray) -> np.ndarray:
    """실린더 세로축을 따라 3점 추출 (top/center/bottom)."""
    ys, xs = np.where(seed > 0)
    if len(xs) < 30:
        return np.array([[int(xs.mean()), int(ys.mean())]])
    y_min, y_max = ys.min(), ys.max()
    y_ctr = (y_min + y_max) // 2
    pts = []
    for y_t in [y_min + (y_max - y_min)//5, y_ctr, y_max - (y_max - y_min)//5]:
        row_xs = xs[np.abs(ys - y_t) < 3]
        if len(row_xs) > 0:
            pts.append([int(np.median(row_xs)), int(y_t)])
    return np.array(pts) if pts else np.array([[int(xs.mean()), int(ys.mean())]])


def sam_refine(predictor: SamPredictor, bgr: np.ndarray,
               seed: np.ndarray, is_cylinder: bool) -> np.ndarray:
    if int(np.count_nonzero(seed)) < 50:
        return np.zeros_like(seed)

    ys, xs = np.where(seed > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    # bbox padding 5px
    H, W = seed.shape
    x1 = max(0, x1 - 5); y1 = max(0, y1 - 5)
    x2 = min(W - 1, x2 + 5); y2 = min(H - 1, y2 + 5)

    # point prompt
    if is_cylinder:
        points = cylinder_point_prompts(seed)
    else:
        points = np.array([[int(xs.mean()), int(ys.mean())]])
    point_labels = np.ones(len(points), dtype=np.int32)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=point_labels,
        box=np.array([x1, y1, x2, y2]),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    out = (masks[best].astype(np.uint8)) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k5)

    # seed와 가장 많이 겹치는 단일 component만 유지
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


def compose(top, bot, title_bot="SAM Overlay"):
    rows = []
    for row, t in [(top, "Original"), (bot, title_bot)]:
        labeled = [cv2.putText(img.copy(), f"cam{ci} {t}", (10, 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
                   for ci, img in enumerate(row)]
        rows.append(np.hstack(labeled))
    return np.vstack([np.vstack(rows), label_bar(rows[0].shape[1])])


def build(frame_id: str, out_dir: Path, mask_save_dir: Path):
    print("Loading MobileSAM...")
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)
    print("SAM loaded")

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)
    table_info = estimate_table_plane(frames)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        print(f"\nProcessing {obj_name}...")
        is_cyl = OBJECT_SYMMETRY.get(obj_name, "none") == "yaw"
        seeds = get_seed_with_fallback(frames, obj_name, table_info)
        final = []
        for ci, cam in enumerate(frames):
            if int(np.count_nonzero(seeds[ci])) < 100:
                final.append(np.zeros_like(seeds[ci]))
                print(f"  cam{ci}: no seed (skip)")
                continue
            with torch.no_grad():
                m = sam_refine(predictor, cam.color_bgr, seeds[ci], is_cyl)
            final.append(m)
            print(f"  cam{ci}: seed={int((seeds[ci]>0).sum())} → sam={int((m>0).sum())}"
                  f"{' [cyl]' if is_cyl else ''}")
        per_obj_masks[obj_name] = final

    # 마스크 파일 저장 (pose 추정 재사용)
    mask_save_dir.mkdir(parents=True, exist_ok=True)
    for obj_name, masks in per_obj_masks.items():
        for ci, m in enumerate(masks):
            cv2.imwrite(str(mask_save_dir / f"{obj_name}_cam{ci}.png"), m)
    meta = {"frame_id": frame_id, "objects": list(OBJ_COLORS_BGR.keys()),
            "cameras": 3, "note": "SAM + fallback color mask"}
    (mask_save_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmasks saved: {mask_save_dir}")

    # overlay 이미지
    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
    all_ov = [cam.color_bgr.copy() for cam in frames]
    for obj_name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj_name]
        for ci in range(3):
            all_ov[ci] = make_overlay(all_ov[ci], masks[ci], color, alpha=0.55)
    cv2.imwrite(str(out_dir / "comparison.png"), compose(originals, all_ov))
    print(f"saved: {out_dir/'comparison.png'}")

    for obj_name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj_name]
        ovs = []
        for ci, cam in enumerate(frames):
            ov = make_overlay(cam.color_bgr.copy(), masks[ci], color, alpha=0.55)
            a = int(np.count_nonzero(masks[ci]))
            cv2.putText(ov, f"{obj_name} area={a}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            ovs.append(ov)
        cv2.imwrite(str(out_dir / f"comparison_{obj_name}.png"),
                    compose(originals, ovs, title_bot=f"{obj_name} SAM"))
    print(f"overlays saved: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("src/output/ideal_overlay") / f"sam_v2_{ts}_frame_{args.frame_id}"
    mask_dir = Path("src/output/sam_masks") / f"{ts}_frame_{args.frame_id}"
    build(args.frame_id, out_dir, mask_dir)


if __name__ == "__main__":
    main()
