#!/usr/bin/env python3
"""
InstantMesh 입력용 RGBA 이미지 자동 생성.

[실행 명령어]
python src/Obj_Step4-1_prepare_instantmesh_input.py \
  --data_dir ./capture_obj_set2 \
  --mask_dir ./masks_set2 \
  --out_dir ./instantmesh_input_set2 \
  --target_size 512 \
  --padding_pct 0.10


각 객체별로:
  1) 3대 카메라 중 마스크 픽셀이 가장 많은 view를 best view로 선택
  2) 해당 cam의 RGB에 mask 적용 → RGBA (mask 밖 alpha=0)
  3) 객체 bbox + padding으로 square crop
  4) target_size (기본 512) 로 리사이즈
  5) instantmesh_input/<obj>/object_input.png 저장
  6) best_view.txt 에 어떤 cam을 골랐는지 기록

사용자가 InstantMesh 웹 데모에 이 이미지를 업로드 → GLB 다운로드 →
  instantmesh_results/<obj>.glb 에 저장 후 improved_instantmesh_pose.py 실행
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def clean_mask(mask: np.ndarray, close_px: int = 5, erode_px: int = 1) -> np.ndarray:
    """largest CC만 남기고, morphological close로 구멍 메우고, 살짝 erode로 경계 노이즈 제거."""
    bm = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bm, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        bm = (labels == keep).astype(np.uint8)
    if close_px > 0:
        k = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, k)
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        bm = cv2.erode(bm, k)
    return (bm * 255).astype(np.uint8)


def view_quality_score(mask_bin: np.ndarray, depth: np.ndarray | None) -> tuple[float, dict]:
    """area * compactness * (0.5 + depth_valid_ratio) — 픽셀 수와 마스크 품질 모두 반영."""
    ys, xs = np.where(mask_bin)
    n_pix = int(mask_bin.sum())
    if n_pix == 0:
        return 0.0, {"n_pix": 0, "compactness": 0.0, "depth_valid_ratio": 0.0}
    bb = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    compactness = n_pix / max(bb, 1)
    if depth is not None:
        d = depth[mask_bin]
        depth_valid_ratio = float((np.isfinite(d) & (d > 0)).mean())
    else:
        depth_valid_ratio = 1.0
    score = n_pix * compactness * (0.5 + depth_valid_ratio)
    return score, {"n_pix": n_pix, "compactness": float(compactness), "depth_valid_ratio": depth_valid_ratio}


def crop_to_square_rgba(rgb: np.ndarray, mask: np.ndarray, target_size: int, padding_pct: float):
    H, W = mask.shape
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        raise ValueError("mask is empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    side = max(bw, bh)
    pad = int(side * padding_pct)
    side_pad = side + 2 * pad
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    half = side_pad // 2
    sx0, sy0 = cx - half, cy - half
    sx1, sy1 = sx0 + side_pad, sy0 + side_pad

    pad_l = max(0, -sx0)
    pad_t = max(0, -sy0)
    pad_r = max(0, sx1 - W)
    pad_b = max(0, sy1 - H)
    sx0c, sy0c = max(0, sx0), max(0, sy0)
    sx1c, sy1c = min(W, sx1), min(H, sy1)

    rgb_crop = rgb[sy0c:sy1c, sx0c:sx1c]
    mask_crop = mask[sy0c:sy1c, sx0c:sx1c]

    rgb_crop = np.pad(rgb_crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), constant_values=0)
    mask_crop = np.pad(mask_crop, ((pad_t, pad_b), (pad_l, pad_r)), constant_values=0)

    alpha = (mask_crop > 127).astype(np.uint8) * 255
    rgba = np.dstack([rgb_crop, alpha])
    return Image.fromarray(rgba).resize((target_size, target_size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./capture_obj")
    ap.add_argument("--mask_dir", default="./masks")
    ap.add_argument("--out_dir", default="./instantmesh_input")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--padding_pct", type=float, default=0.10,
                    help="객체 bbox 주변 padding 비율 (0.10 = 10%)")
    ap.add_argument("--mask_close_px", type=int, default=5,
                    help="마스크 morphological close (구멍 메우기) 반경")
    ap.add_argument("--mask_erode_px", type=int, default=1,
                    help="cleanup 후 경계 erode (boundary 노이즈 제거)")
    ap.add_argument("--force_cam", default="",
                    help="형식 'obj01:cam0,obj03:cam2' — 특정 객체의 best_view를 수동 지정")
    args = ap.parse_args()

    force_map: dict[str, str] = {}
    if args.force_cam:
        for pair in args.force_cam.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                force_map[k.strip()] = v.strip()

    data = Path(args.data_dir)
    mask_root = Path(args.mask_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    obj_dirs = sorted([d for d in mask_root.iterdir() if d.is_dir()])
    if not obj_dirs:
        raise FileNotFoundError(f"No object dirs in {mask_root}")
    print(f"[INFO] {len(obj_dirs)} objects found: {[d.name for d in obj_dirs]}")

    summary = []
    for obj_dir in obj_dirs:
        name = obj_dir.name
        masks_clean, scores, infos = {}, {}, {}
        for ci in ["cam0", "cam1", "cam2"]:
            mp = obj_dir / f"{ci}_mask.png"
            if not mp.exists():
                continue
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            m_clean = clean_mask(m, close_px=args.mask_close_px, erode_px=args.mask_erode_px)
            masks_clean[ci] = m_clean
            # depth는 view 선택용 점수 계산에만 사용
            dpath = data / f"{ci}_depth.png"
            depth = None
            if dpath.exists():
                d_raw = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
                if d_raw is not None:
                    depth = d_raw.astype(np.float32)
            s, info = view_quality_score(m_clean > 127, depth)
            scores[ci] = s
            infos[ci] = info

        if not masks_clean:
            print(f"[SKIP] {name}: no mask found")
            continue

        if name in force_map and force_map[name] in masks_clean:
            best = force_map[name]
            print(f"[{name}] forced best={best}")
        else:
            best = max(scores, key=scores.get)

        rgb_bgr = cv2.imread(str(data / f"{best}_rgb.png"))
        if rgb_bgr is None:
            print(f"[SKIP] {name}: {best} RGB not found")
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        try:
            rgba = crop_to_square_rgba(rgb, masks_clean[best], args.target_size, args.padding_pct)
        except ValueError as e:
            print(f"[SKIP] {name}: {e}")
            continue

        out_obj = out_root / name
        out_obj.mkdir(parents=True, exist_ok=True)
        out_path = out_obj / "object_input.png"
        rgba.save(out_path)
        # cleaned mask도 저장 (디버그/재사용)
        cv2.imwrite(str(out_obj / f"{best}_mask_clean.png"), masks_clean[best])
        with open(out_obj / "best_view.json", "w") as f:
            json.dump({"best_cam": best, "scores": scores, "per_cam_info": infos}, f, indent=2)

        info_best = infos[best]
        print(f"[{name}] best={best}  area={info_best['n_pix']} px  "
              f"compact={info_best['compactness']:.3f} depth_valid={info_best['depth_valid_ratio']:.3f}  "
              f"score={scores[best]:.1f}  -> {out_path}")
        summary.append((name, best, info_best['n_pix']))

    print("\n=== Summary ===")
    print(f"{'object':<10s} {'best_cam':<10s} {'area_px':>10s}")
    for name, best, n in summary:
        print(f"{name:<10s} {best:<10s} {n:>10d}")

    print(f"\n다음 단계:")
    print(f"  1. https://huggingface.co/spaces/TencentARC/InstantMesh 접속")
    print(f"  2. 각 {out_root}/<obj>/object_input.png 를 업로드 → GLB 다운로드")
    print(f"  3. instantmesh_results/<obj>.glb 위치에 저장")
    print(f"  4. improved_instantmesh_pose.py 객체별 1회씩 실행")


if __name__ == "__main__":
    main()
