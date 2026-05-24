#!/usr/bin/env python3
"""
InstantMesh 입력 RGBA v1: soft alpha matting + premultiplied resize.

원본 Step4-1 대비 변경점 (오직 두 가지):
  (1) closed-form alpha matting 으로 마스크 경계를 부드러운(soft) alpha 로 정제.
      → 1-bit 경계 / 머리카락·얇은 부분 잘림 제거.
  (2) Premultiplied alpha 로 LANCZOS resize → 경계 검정 halo / 색 번짐 제거.

best view 선택은 하지 않고, **각 카메라(cam0/cam1/cam2) 각각**에 대해 입력 이미지를 만든다.

[실행 예]
python src/Obj_Step4-1_prepare_instantmesh_input_v1.py \
  --data_dir ./capture_obj_set1 \
  --mask_dir ./masks_set1 \
  --out_dir ./instantmesh_input_v1_set1 \
  --target_size 512 --padding_pct 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pymatting import estimate_alpha_cf, estimate_foreground_ml


def clean_binary_mask(mask: np.ndarray, close_px: int = 5) -> np.ndarray:
    bm = (mask > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bm, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + int(np.argmax(areas))
        bm = (labels == keep).astype(np.uint8)
    if close_px > 0:
        k = np.ones((close_px * 2 + 1, close_px * 2 + 1), np.uint8)
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, k)
    return bm


def build_trimap(mask_bin: np.ndarray, fg_erode: int, bg_dilate: int) -> np.ndarray:
    k_fg = np.ones((fg_erode * 2 + 1, fg_erode * 2 + 1), np.uint8)
    k_bg = np.ones((bg_dilate * 2 + 1, bg_dilate * 2 + 1), np.uint8)
    fg = cv2.erode(mask_bin, k_fg)
    bg_keep = cv2.dilate(mask_bin, k_bg)
    trimap = np.full(mask_bin.shape, 0.5, dtype=np.float32)
    trimap[bg_keep == 0] = 0.0
    trimap[fg == 1] = 1.0
    return trimap


def matte(rgb_u8: np.ndarray, mask_u8: np.ndarray,
          fg_erode: int = 6, bg_dilate: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """closed-form matting → (foreground RGB float[0,1], alpha float[0,1]).
    pymatting numba 커널이 float64 를 기대하므로 내부적으로 float64 로 처리."""
    img = (rgb_u8.astype(np.float64) / 255.0)
    bm = clean_binary_mask(mask_u8)
    trimap = build_trimap(bm, fg_erode=fg_erode, bg_dilate=bg_dilate).astype(np.float64)
    alpha = np.clip(estimate_alpha_cf(img, trimap), 0.0, 1.0)
    fg = np.clip(estimate_foreground_ml(img, alpha), 0.0, 1.0)
    return fg.astype(np.float32), alpha.astype(np.float32)


def crop_square_premultiplied(fg_rgb_f: np.ndarray, alpha_f: np.ndarray,
                              target_size: int, padding_pct: float,
                              alpha_thresh: float = 0.05) -> Image.Image:
    """alpha bbox 기준 정사각 crop → premultiplied resize → un-premultiply → RGBA."""
    H, W = alpha_f.shape
    bm = alpha_f > alpha_thresh
    ys, xs = np.where(bm)
    if len(xs) == 0:
        raise ValueError("alpha empty")
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

    rgb_crop = fg_rgb_f[sy0c:sy1c, sx0c:sx1c]
    a_crop = alpha_f[sy0c:sy1c, sx0c:sx1c]
    rgb_crop = np.pad(rgb_crop, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)))
    a_crop = np.pad(a_crop, ((pad_t, pad_b), (pad_l, pad_r)))

    rgb_pm = rgb_crop * a_crop[..., None]
    rgb_pm_r = cv2.resize(rgb_pm, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    a_r = cv2.resize(a_crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    a_r = np.clip(a_r, 0.0, 1.0)

    a_safe = np.maximum(a_r, 1e-4)
    rgb_r = np.clip(rgb_pm_r / a_safe[..., None], 0.0, 1.0)
    rgb_r[a_r < 1e-3] = 0.0

    rgba_u8 = np.concatenate(
        [(rgb_r * 255.0).astype(np.uint8), (a_r * 255.0).astype(np.uint8)[..., None]],
        axis=-1,
    )
    return Image.fromarray(rgba_u8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./capture_obj")
    ap.add_argument("--mask_dir", default="./masks")
    ap.add_argument("--out_dir", default="./instantmesh_input_v1")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--padding_pct", type=float, default=0.10)
    ap.add_argument("--fg_erode", type=int, default=6,
                    help="trimap FG 영역으로 쓸 erode 반경 (px)")
    ap.add_argument("--bg_dilate", type=int, default=10,
                    help="trimap BG 영역 결정용 dilate 반경 (px)")
    ap.add_argument("--cams", default="cam0,cam1,cam2",
                    help="처리할 카메라 목록 (콤마 구분)")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    data = Path(args.data_dir)
    mask_root = Path(args.mask_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    obj_dirs = sorted([d for d in mask_root.iterdir() if d.is_dir()])
    if not obj_dirs:
        raise FileNotFoundError(f"No object dirs in {mask_root}")
    print(f"[v1] {len(obj_dirs)} objects: {[d.name for d in obj_dirs]}  cams={cams}")

    summary = []
    for obj_dir in obj_dirs:
        name = obj_dir.name
        out_obj = out_root / name
        out_obj.mkdir(parents=True, exist_ok=True)
        rec = {"object": name, "cams": {}}
        for ci in cams:
            mp = obj_dir / f"{ci}_mask.png"
            rp = data / f"{ci}_rgb.png"
            if not mp.exists() or not rp.exists():
                print(f"  [SKIP] {name}/{ci}: missing files")
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
            if int((mask > 127).sum()) < 100:
                print(f"  [SKIP] {name}/{ci}: mask too small ({(mask>127).sum()} px)")
                continue
            try:
                fg_f, alpha_f = matte(rgb, mask,
                                      fg_erode=args.fg_erode, bg_dilate=args.bg_dilate)
                rgba = crop_square_premultiplied(fg_f, alpha_f,
                                                 args.target_size, args.padding_pct)
            except Exception as e:
                print(f"  [ERR ] {name}/{ci}: {e}")
                continue
            out_path = out_obj / f"{ci}_input.png"
            rgba.save(out_path)
            area = int((alpha_f > 0.5).sum())
            rec["cams"][ci] = {"area_px": area, "out": out_path.name}
            print(f"  [{name}/{ci}] matted area={area} px -> {out_path}")
        with open(out_obj / "v1_meta.json", "w") as f:
            json.dump(rec, f, indent=2)
        summary.append(rec)

    print("\n=== v1 Summary ===")
    for r in summary:
        cams_done = ",".join(sorted(r["cams"].keys()))
        print(f"  {r['object']:<8s}  cams=[{cams_done}]")


if __name__ == "__main__":
    main()
