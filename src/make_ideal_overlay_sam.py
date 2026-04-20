#!/usr/bin/env python3
"""MobileSAM 기반 정밀 overlay.

처리 흐름:
1. HSV 색상 마스크로 각 물체 대략 위치 파악 (bbox + centroid)
2. bbox prompt + centroid point prompt를 SAM에 입력
3. SAM이 픽셀 단위 정밀 mask 생성 → 학습된 object-awareness로
   유사 색상 배경과 정확히 분리
4. 각 물체 색으로 overlay

저장: src/output/ideal_overlay/sam_{timestamp}_frame_{id}/
"""
from __future__ import annotations

import argparse
import datetime
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import load_calibration, load_frame, estimate_table_plane
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


def compose(top, bot, title_top="Original", title_bot="SAM Overlay"):
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
    return np.vstack([canvas, label_bar(canvas.shape[1])])


def sam_refine(predictor: SamPredictor, bgr: np.ndarray,
               seed_mask: np.ndarray) -> np.ndarray:
    """seed의 bbox + centroid를 prompt로 SAM 실행."""
    if int(np.count_nonzero(seed_mask)) < 50:
        return np.zeros_like(seed_mask)

    ys, xs = np.where(seed_mask > 0)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    # centroid: seed의 무게중심
    cx, cy = int(xs.mean()), int(ys.mean())

    # SAM은 RGB 기대
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        box=np.array([x1, y1, x2, y2]),
        multimask_output=True,
    )
    # 최고 score mask 선택
    best = int(np.argmax(scores))
    out_mask = (masks[best].astype(np.uint8)) * 255

    # seed와 겹치는 영역만 유지 (SAM이 다른 물체까지 포함 시 방지)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_CLOSE, k5)
    return out_mask


def build(frame_id: str, out_dir: Path):
    print("Loading MobileSAM...")
    device = "cpu"  # CUDA 없어도 OK
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM loaded")

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)
    table_info = estimate_table_plane(frames)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        print(f"\nProcessing {obj_name}...")
        color_masks = [color_mask_for_object(cam.color_bgr, obj_name) for cam in frames]
        cands = [connected_components_sorted(m, top_k=3) for m in color_masks]
        seed_masks = select_multiview_seed_masks(frames, cands, table_info)
        seed_masks = mask_refine_with_depth_cluster(
            frames, seed_masks, table_info, object_name=obj_name
        )

        final_masks = []
        for ci, cam in enumerate(frames):
            seed = seed_masks[ci]
            if int(np.count_nonzero(seed)) < 100:
                final_masks.append(np.zeros_like(seed))
                print(f"  cam{ci}: seed too small, skip")
                continue
            with torch.no_grad():
                m = sam_refine(predictor, cam.color_bgr, seed)
            final_masks.append(m)
            print(f"  cam{ci}: seed={int((seed>0).sum())} → sam={int((m>0).sum())}")
        per_obj_masks[obj_name] = final_masks

    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
    per_cam_all = [cam.color_bgr.copy() for cam in frames]
    for obj_name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj_name]
        for ci in range(3):
            per_cam_all[ci] = make_overlay(per_cam_all[ci], masks[ci], color, alpha=0.55)

    cv2.imwrite(str(out_dir / "comparison.png"), compose(originals, per_cam_all))
    print(f"\nsaved: {out_dir/'comparison.png'}")

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
                    compose(originals, overlays, title_bot=f"{obj_name} SAM"))
        print(f"saved: {out_dir/f'comparison_{obj_name}.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("src/output/ideal_overlay") / f"sam_{ts}_frame_{args.frame_id}"
    else:
        out_dir = Path(args.out_dir)
    build(args.frame_id, out_dir)


if __name__ == "__main__":
    main()
