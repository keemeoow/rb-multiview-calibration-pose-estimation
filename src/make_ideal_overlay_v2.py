#!/usr/bin/env python3
"""GrabCut 기반 정밀 overlay: HSV seed + GrabCut으로 실제 물체 경계 추출.

처리 흐름:
1. HSV + depth로 초기 seed mask
2. seed를 foreground, 주변 dilated 영역을 probable foreground,
   멀리는 sure background로 설정
3. GrabCut 반복 → GMM 기반 color 모델로 정확한 경계 추출
4. 결과 마스크를 각 물체 색으로 overlay

저장: src/output/ideal_overlay/run_{timestamp}_frame_{id}/
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import load_calibration, load_frame, estimate_table_plane
from pose_per_object_v2 import (
    color_mask_for_object,
    connected_components_sorted,
    select_multiview_seed_masks,
    mask_refine_with_depth_cluster,
)

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"

OBJ_COLORS_BGR = {
    "object_001": (60, 60, 255),
    "object_002": (80, 230, 255),
    "object_003": (255, 140, 40),
    "object_004": (210, 240, 130),
}


def grabcut_refine(bgr: np.ndarray, seed_mask: np.ndarray,
                   fg_dilate: int = 3, bg_dilate: int = 30) -> np.ndarray:
    """seed mask를 기준으로 GrabCut 실행 → 정밀 마스크.

    - seed_mask>0 → SURE_FG
    - dilate(seed, fg_dilate) \ seed → PR_FG (probable)
    - outside dilate(seed, bg_dilate) → SURE_BG
    - 나머지 (ring) → PR_BG
    """
    if int(np.count_nonzero(seed_mask)) < 50:
        return np.zeros_like(seed_mask)

    gc_mask = np.full(seed_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    fg = seed_mask > 0
    k_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_dilate*2+1, fg_dilate*2+1))
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilate*2+1, bg_dilate*2+1))
    pr_fg_region = cv2.dilate(fg.astype(np.uint8), k_fg, iterations=1) > 0
    far_region = cv2.dilate(fg.astype(np.uint8), k_bg, iterations=1) > 0

    gc_mask[~far_region] = cv2.GC_BGD          # far → sure background
    gc_mask[pr_fg_region & ~fg] = cv2.GC_PR_FGD
    gc_mask[fg] = cv2.GC_FGD                    # seed → sure foreground

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_MASK)
    except Exception as exc:
        print(f"  [grabcut] 실패 ({exc}), seed 그대로 사용")
        return seed_mask.copy()

    result = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 잔여 노이즈 제거
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, k5)
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k5)
    # seed와 겹치는 최대 component만 유지 (튐 방지)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    if num > 1:
        best_id = -1; best_score = -1
        for i in range(1, num):
            a = stats[i, cv2.CC_STAT_AREA]
            if a < 100:
                continue
            overlap = int(((labels == i) & fg).sum())
            if overlap > best_score:
                best_score = overlap
                best_id = i
        if best_id > 0:
            result = np.where(labels == best_id, 255, 0).astype(np.uint8)
    return result


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
    items = list(OBJ_COLORS_BGR.items())
    part = w // len(items)
    for i, (name, color) in enumerate(items):
        x = i * part
        cv2.rectangle(bar, (x, 0), (x + part, h), (80, 80, 80), -1)
        cv2.rectangle(bar, (x + 4, 4), (x + 16, h - 4), color, -1)
        cv2.putText(bar, name, (x + 22, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return bar


def compose(top, bot, title_top="Original", title_bot="GrabCut Overlay"):
    out_rows = []
    for row, title in [(top, title_top), (bot, title_bot)]:
        labeled = []
        for ci, img in enumerate(row):
            out = img.copy()
            cv2.putText(out, f"cam{ci} {title}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
            labeled.append(out)
        out_rows.append(np.hstack(labeled))
    canvas = np.vstack(out_rows)
    bar = label_bar(canvas.shape[1])
    return np.vstack([canvas, bar])


def build(frame_id: str, out_dir: Path):
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)
    table_info = estimate_table_plane(frames)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        # 1) color mask → top 3 components
        color_masks = [color_mask_for_object(cam.color_bgr, obj_name) for cam in frames]
        cands = [connected_components_sorted(m, top_k=3) for m in color_masks]
        # 2) multi-view seed selection (3D 일관성)
        seed_masks = select_multiview_seed_masks(frames, cands, table_info)
        # 3) depth cluster 재투영 정제
        refined_seed = mask_refine_with_depth_cluster(
            frames, seed_masks, table_info, object_name=obj_name
        )
        # 4) GrabCut으로 실제 물체 경계까지 확장
        final_masks = []
        for ci, cam in enumerate(frames):
            seed = refined_seed[ci]
            if int(np.count_nonzero(seed)) < 100:
                final_masks.append(np.zeros_like(seed))
                continue
            m = grabcut_refine(cam.color_bgr, seed, fg_dilate=3, bg_dilate=30)
            final_masks.append(m)
            print(f"  {obj_name} cam{ci}: seed={int(seed.sum()>0 and (seed>0).sum()) or 0} "
                  f"→ grabcut={int((m>0).sum())}")
        per_obj_masks[obj_name] = final_masks

    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
    per_cam_all = [cam.color_bgr.copy() for cam in frames]
    for obj_name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj_name]
        for ci in range(3):
            per_cam_all[ci] = make_overlay(per_cam_all[ci], masks[ci], color, alpha=0.55)

    cv2.imwrite(str(out_dir / "comparison.png"), compose(originals, per_cam_all))
    print(f"saved: {out_dir/'comparison.png'}")

    for obj_name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj_name]
        overlays = []
        for ci, cam in enumerate(frames):
            ov = make_overlay(cam.color_bgr.copy(), masks[ci], color, alpha=0.55)
            area = int(np.count_nonzero(masks[ci]))
            cv2.putText(ov, f"{obj_name} area={area}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            overlays.append(ov)
        cv2.imwrite(str(out_dir / f"comparison_{obj_name}.png"),
                    compose(originals, overlays, title_bot=f"{obj_name} GrabCut"))
        print(f"saved: {out_dir/f'comparison_{obj_name}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("src/output/ideal_overlay") / f"grabcut_{ts}_frame_{args.frame_id}"
    else:
        out_dir = Path(args.out_dir)
    build(args.frame_id, out_dir)


if __name__ == "__main__":
    main()
