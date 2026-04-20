#!/usr/bin/env python3
"""이상적 overlay: v2의 multi-view seed selection으로 클린 마스크 확보.

pose 추정 없이:
1. HSV 임계로 color mask 생성
2. v2의 connected_components_sorted + select_multiview_seed_masks로
   카메라간 3D 일관성 있는 최적 마스크 조합 선택
3. depth-cluster cross-cam 재투영으로 정제
"""
from __future__ import annotations

import argparse
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
    OBJECT_LABELS as _LBL,
)

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"

OBJ_COLORS_BGR = {
    "object_001": (60, 60, 255),     # 빨강
    "object_002": (80, 230, 255),    # 노랑
    "object_003": (255, 140, 40),    # 곤색
    "object_004": (210, 240, 130),   # 민트
}


def make_overlay(img, mask, color, alpha=0.55):
    out = img.copy()
    color_layer = np.zeros_like(img)
    color_layer[:] = color
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


def compose(top, bot, title_top="Original", title_bot="Pose Overlay"):
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
        # 각 카메라에서 color mask → 상위 3개 components
        color_masks = [color_mask_for_object(cam.color_bgr, obj_name) for cam in frames]
        candidates = [connected_components_sorted(m, top_k=3) for m in color_masks]
        # 3D 일관성 있는 조합 선택 (select_multiview_seed_masks)
        masks_raw = select_multiview_seed_masks(frames, candidates, table_info)
        # depth 클러스터 재투영 기반 정제
        masks = mask_refine_with_depth_cluster(
            frames, masks_raw, table_info, object_name=obj_name
        )
        per_obj_masks[obj_name] = masks

    # 개별 overlay
    per_cam_all = [cam.color_bgr.copy() for cam in frames]
    for obj_name, masks in per_obj_masks.items():
        color = OBJ_COLORS_BGR[obj_name]
        for ci in range(len(frames)):
            per_cam_all[ci] = make_overlay(per_cam_all[ci], masks[ci], color, alpha=0.55)

    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
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
                    compose(originals, overlays, title_bot=f"{obj_name}"))
        print(f"saved: {out_dir/f'comparison_{obj_name}.png'}")


def main():
    import datetime
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--out_dir", default=None,
                    help="생략 시 ideal_overlay/{timestamp}/ 에 저장 (기존 파일 보존).")
    args = ap.parse_args()
    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("src/output/ideal_overlay") / f"run_{ts}_frame_{args.frame_id}"
    else:
        out_dir = Path(args.out_dir)
    build(args.frame_id, out_dir)


if __name__ == "__main__":
    main()
