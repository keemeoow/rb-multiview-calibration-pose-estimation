#!/usr/bin/env python3
"""
InstantMesh 입력 RGBA v2: 전체 품질 향상 파이프라인.

개선 항목:
  (1) closed-form alpha matting (soft alpha)
  (2) Premultiplied alpha LANCZOS resize (halo / color fringe 제거)
  (3) View quality score = area * compactness * (0.5 + depth_valid)
        * sharpness (Laplacian variance, 마스크 내부 정규화)
        * boundary penalty (객체가 프레임 경계에 닿으면 가중치↓)
  (4) Foreground estimation (estimate_foreground_ml) 으로 RGB 의 배경색 번짐 제거
  (5) Gray-world white balance + 약한 percentile stretch 로 톤 정규화
  (6) Larger default padding (0.15) + mask centroid 중심 정렬

best view 선택은 하지 않고, **각 카메라(cam0/cam1/cam2)** 에 대해 입력 이미지를
모두 만든다. (점수는 메타에 기록 — 디버깅/후처리용)

[실행 예]
python src/Obj_Step4-1_prepare_instantmesh_input_v2.py \
  --data_dir ./capture_obj_set1 \
  --mask_dir ./masks_set1 \
  --out_dir ./instantmesh_input_v2_set1 \
  --target_size 512 --padding_pct 0.15
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
    img = rgb_u8.astype(np.float64) / 255.0
    bm = clean_binary_mask(mask_u8)
    trimap = build_trimap(bm, fg_erode=fg_erode, bg_dilate=bg_dilate).astype(np.float64)
    alpha = np.clip(estimate_alpha_cf(img, trimap), 0.0, 1.0)
    fg = np.clip(estimate_foreground_ml(img, alpha), 0.0, 1.0)
    return fg.astype(np.float32), alpha.astype(np.float32)


def gray_world_wb(rgb_u8: np.ndarray) -> np.ndarray:
    """전체 이미지에 대한 단순 gray-world WB. 객체 색이 한쪽으로 치우치는 것을 막기 위해
    객체 영역이 아닌 *전체 프레임* 평균을 사용한다."""
    img = rgb_u8.astype(np.float32)
    avg = img.reshape(-1, 3).mean(0) + 1e-6
    gray = avg.mean()
    gains = gray / avg
    out = img * gains[None, None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def percentile_stretch(rgb_u8: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """약한 percentile 기반 톤 정규화 (clipping 방지로 lo/hi 1%만 자름)."""
    img = rgb_u8.astype(np.float32)
    p_lo = np.percentile(img, lo)
    p_hi = np.percentile(img, hi)
    if p_hi - p_lo < 1.0:
        return rgb_u8
    out = (img - p_lo) * (255.0 / (p_hi - p_lo))
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def view_quality(mask_bin: np.ndarray, rgb_u8: np.ndarray,
                 depth: np.ndarray | None) -> tuple[float, dict]:
    """(3) view quality score with sharpness + boundary penalty.
    선택에 쓰진 않지만 점수를 메타에 남긴다."""
    H, W = mask_bin.shape
    ys, xs = np.where(mask_bin)
    n_pix = int(mask_bin.sum())
    if n_pix == 0:
        return 0.0, {"n_pix": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bb = (x1 - x0 + 1) * (y1 - y0 + 1)
    compact = n_pix / max(bb, 1)
    if depth is not None:
        d = depth[mask_bin]
        depth_valid = float((np.isfinite(d) & (d > 0)).mean())
    else:
        depth_valid = 1.0
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap[mask_bin].var())
    sharp = lap_var / (lap_var + 200.0)  # 0~1 정규화 (200 = 부드러운 sigmoid 중간점)
    touch = (x0 <= 1) or (x1 >= W - 2) or (y0 <= 1) or (y1 >= H - 2)
    boundary_pen = 0.3 if touch else 1.0
    score = float(n_pix * compact * (0.5 + depth_valid) * (0.3 + 0.7 * sharp) * boundary_pen)
    return score, {
        "n_pix": n_pix, "compactness": float(compact),
        "depth_valid": depth_valid, "sharpness": sharp,
        "lap_var": lap_var, "touches_boundary": bool(touch),
        "score": score,
    }


def crop_square_premultiplied(fg_rgb_f: np.ndarray, alpha_f: np.ndarray,
                              target_size: int, padding_pct: float,
                              use_centroid: bool = True,
                              alpha_thresh: float = 0.05) -> Image.Image:
    H, W = alpha_f.shape
    bm = (alpha_f > alpha_thresh).astype(np.uint8)
    ys, xs = np.where(bm)
    if len(xs) == 0:
        raise ValueError("alpha empty")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    side = max(bw, bh)
    pad = int(side * padding_pct)
    side_pad = side + 2 * pad

    if use_centroid:
        M = cv2.moments(bm.astype(np.uint8))
        if M["m00"] > 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))
        else:
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
    else:
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
    a_r = np.clip(cv2.resize(a_crop, (target_size, target_size),
                             interpolation=cv2.INTER_LANCZOS4), 0.0, 1.0)
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
    ap.add_argument("--out_dir", default="./instantmesh_input_v2")
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--padding_pct", type=float, default=0.15)
    ap.add_argument("--fg_erode", type=int, default=6)
    ap.add_argument("--bg_dilate", type=int, default=10)
    ap.add_argument("--no_wb", action="store_true", help="gray-world WB 비활성화")
    ap.add_argument("--no_stretch", action="store_true", help="percentile stretch 비활성화")
    ap.add_argument("--no_centroid", action="store_true", help="중심정렬을 bbox center 로 폴백")
    ap.add_argument("--cams", default="cam0,cam1,cam2")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    data = Path(args.data_dir)
    mask_root = Path(args.mask_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    obj_dirs = sorted([d for d in mask_root.iterdir() if d.is_dir()])
    if not obj_dirs:
        raise FileNotFoundError(f"No object dirs in {mask_root}")
    print(f"[v2] {len(obj_dirs)} objects: {[d.name for d in obj_dirs]}  cams={cams}")

    summary = []
    for obj_dir in obj_dirs:
        name = obj_dir.name
        out_obj = out_root / name
        out_obj.mkdir(parents=True, exist_ok=True)
        rec = {"object": name, "cams": {}}
        for ci in cams:
            mp = obj_dir / f"{ci}_mask.png"
            rp = data / f"{ci}_rgb.png"
            dp = data / f"{ci}_depth.png"
            if not mp.exists() or not rp.exists():
                print(f"  [SKIP] {name}/{ci}: missing files")
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            rgb_raw = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
            if int((mask > 127).sum()) < 100:
                print(f"  [SKIP] {name}/{ci}: mask too small ({(mask>127).sum()} px)")
                continue
            depth = None
            if dp.exists():
                d_raw = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
                if d_raw is not None:
                    depth = d_raw.astype(np.float32)

            # (5) WB + 톤 정규화 (matting 전에 적용 → fg 추정도 정규화된 색 기준)
            rgb = rgb_raw
            if not args.no_wb:
                rgb = gray_world_wb(rgb)
            if not args.no_stretch:
                rgb = percentile_stretch(rgb)

            # (3) 점수는 정규화된 RGB 로 계산 (sharpness 가 더 안정적)
            score, info = view_quality(clean_binary_mask(mask).astype(bool), rgb, depth)

            try:
                # (1)(4) soft alpha + foreground 추정
                fg_f, alpha_f = matte(rgb, mask,
                                      fg_erode=args.fg_erode, bg_dilate=args.bg_dilate)
                # (2)(6) premultiplied resize + centroid centering + 큰 padding
                rgba = crop_square_premultiplied(
                    fg_f, alpha_f,
                    target_size=args.target_size,
                    padding_pct=args.padding_pct,
                    use_centroid=not args.no_centroid,
                )
            except Exception as e:
                print(f"  [ERR ] {name}/{ci}: {e}")
                continue

            out_path = out_obj / f"{ci}_input.png"
            rgba.save(out_path)
            rec["cams"][ci] = {**info, "out": out_path.name}
            print(f"  [{name}/{ci}] area={info['n_pix']} sharp={info['sharpness']:.2f} "
                  f"touch={info['touches_boundary']} score={score:.1f} -> {out_path.name}")

        # 카메라 간 비교용 best 표기 (선택은 안 함, 메타만)
        if rec["cams"]:
            best = max(rec["cams"].items(), key=lambda kv: kv[1].get("score", 0.0))
            rec["best_cam"] = best[0]
        with open(out_obj / "v2_meta.json", "w") as f:
            json.dump(rec, f, indent=2)
        summary.append(rec)

    print("\n=== v2 Summary ===")
    print(f"  {'object':<8s} {'best':<6s} cams")
    for r in summary:
        scores = " ".join(f"{c}:{r['cams'][c]['score']:.0f}" for c in sorted(r["cams"].keys()))
        print(f"  {r['object']:<8s} {r.get('best_cam',''):<6s} {scores}")


if __name__ == "__main__":
    main()
