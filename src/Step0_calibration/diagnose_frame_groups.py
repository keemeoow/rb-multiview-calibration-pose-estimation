"""
diagnose_frame_groups.py
frames 37~76 vs 나머지의 PnP 품질 / 큐브 위치 / 카메라 페어별 triangulation 비교.

가설: frames 37~76는 PnP는 OK인데 외부 파라미터가 일반화 안 된 카메라 공간 영역에 큐브가 위치.

python diagnose_frame_groups.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from _aruco_cube import CubeConfig, ArucoCubeTarget, rodrigues_to_Rt, inv_T


def load_intrinsics(intr_dir, ci):
    p = os.path.join(intr_dir, f"cam{ci}.npz")
    z = np.load(p, allow_pickle=True)
    return z["color_K"].astype(np.float64), z["color_D"].astype(np.float64).reshape(-1)


def load_T_C0_Ci(calib_dir, ref_ci, ci):
    if ci == ref_ci:
        return np.eye(4, dtype=np.float64)
    return np.load(os.path.join(calib_dir, f"T_C{ref_ci}_C{ci}.npy")).astype(np.float64)


def triangulate_n_views(pixels_n, P_n):
    rows = []
    for (x, y), P in zip(pixels_n, P_n):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return np.full(3, np.nan)
    return Xh[:3] / Xh[3]


def ray_perp_distance(X, origin, direction):
    d = direction / (np.linalg.norm(direction) + 1e-12)
    v = X - origin
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def group_of(fid: int) -> str:
    if 37 <= fid <= 76:
        return "B (37-76)"
    if fid < 37:
        return "A (0-36)"
    return "C (77-105)"


def fmt(x, w=8, p=3):
    return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'nan':>{w}}"


def summarize(values: List[float]) -> str:
    if not values:
        return "n=0"
    arr = np.array(values)
    return (f"n={len(arr):4d} mean={fmt(arr.mean())} median={fmt(np.median(arr))} "
            f"p90={fmt(np.percentile(arr, 90))} max={fmt(arr.max())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_folder", required=True)
    ap.add_argument("--intrinsics_dir", required=True)
    ap.add_argument("--ref_cam_idx", type=int, default=0)
    ap.add_argument("--calib_dir", default=None)
    args = ap.parse_args()

    root = args.root_folder
    calib_dir = args.calib_dir or os.path.join(root, "calib_out_cube")

    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)
    captures = meta["captures"]

    cam_indices = sorted({int(k) for cap in captures for k in cap["cams"].keys()})
    ref_ci = args.ref_cam_idx
    print(f"[INFO] cams={cam_indices}  ref=cam{ref_ci}  frames={len(captures)}")

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    K_map, D_map, T_C0_Ci_map = {}, {}, {}
    P_i_map, cam_center_C0, R_C0_Ci_map = {}, {}, {}
    img_size = {}
    for ci in cam_indices:
        K_map[ci], D_map[ci] = load_intrinsics(args.intrinsics_dir, ci)
        T_C0_Ci = load_T_C0_Ci(calib_dir, ref_ci, ci)
        T_C0_Ci_map[ci] = T_C0_Ci
        T_Ci_C0 = np.linalg.inv(T_C0_Ci)
        P_i_map[ci] = T_Ci_C0[:3, :]
        cam_center_C0[ci] = T_C0_Ci[:3, 3].copy()
        R_C0_Ci_map[ci] = T_C0_Ci[:3, :3].copy()

    # ============================================================
    # 프레임별 metric 수집
    # ============================================================
    # per_frame[fid] = {
    #   "pnp":  {ci: {"reproj_mean":, "n_markers":, "rvec":, "tvec":}}
    #   "cube_in_cam":  {ci: (x,y,z)} cam_i 좌표계에서 cube center
    #   "cube_in_img":  {ci: (u,v, frac_x, frac_y)} 이미지 픽셀 + (0..1 정규화)
    #   "tri_per_pair": {(ci,cj): [perp_mm,...]}
    #   "tri_all":      [perp_mm,...]
    #   "edge_err":     [mm,...]
    # }
    per_frame = {}

    for cap in captures:
        fid = int(cap["event_id"])
        cams = cap["cams"]
        rec = {
            "pnp": {},
            "cube_in_cam": {},
            "cube_in_img": {},
            "tri_per_pair": defaultdict(list),
            "tri_all": [],
            "edge_err": [],
        }

        per_cam_markers_und: Dict[int, Dict[int, np.ndarray]] = {}
        per_cam_markers_px: Dict[int, Dict[int, np.ndarray]] = {}

        for ci_str, info in cams.items():
            ci = int(ci_str)
            if not info.get("saved", False):
                continue
            rgb_path = info.get("rgb_path")
            img_path = rgb_path if os.path.isabs(rgb_path) else os.path.join(root, rgb_path)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            img_size[ci] = (W, H)

            corners_list, ids = cube.detect(img)
            if ids is None or len(corners_list) == 0:
                continue

            entry_und, entry_px = {}, {}
            for c, mid in zip(corners_list, ids):
                mid = int(mid)
                if mid not in cfg.id_to_face:
                    continue
                px = c.reshape(4, 2).astype(np.float64)
                und = cv2.undistortPoints(px.reshape(-1, 1, 2),
                                          K_map[ci], D_map[ci]).reshape(-1, 2)
                entry_px[mid] = px
                entry_und[mid] = und
            if entry_und:
                per_cam_markers_und[ci] = entry_und
                per_cam_markers_px[ci] = entry_px

            # ── PnP (Step3와 동일 모드) ──
            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img, K_map[ci], D_map[ci],
                use_ransac=False, min_markers=1,
                reproj_thr_mean_px=5.0,
                single_marker_only=False,
                return_reproj=True,
            )
            if ok and reproj is not None:
                T_Ci_O = rodrigues_to_Rt(rvec, tvec)
                cube_center_in_Ci = T_Ci_O[:3, 3]
                rec["pnp"][ci] = {
                    "reproj_mean": float(reproj["err_mean"]),
                    "reproj_p90":  float(reproj["err_p90"]),
                    "n_markers":   int(len(used)),
                    "n_points":    int(reproj["n_points"]),
                    "used":        [int(x) for x in used],
                    "cube_in_Ci":  cube_center_in_Ci,
                }
                rec["cube_in_cam"][ci] = cube_center_in_Ci

                # 이미지 픽셀 위치 (cube center 투영)
                proj, _ = cv2.projectPoints(
                    np.zeros((1, 3)), rvec, tvec, K_map[ci], D_map[ci]
                )
                u, v = float(proj[0, 0, 0]), float(proj[0, 0, 1])
                rec["cube_in_img"][ci] = (u, v, u / W, v / H)

        # 카메라 페어별 triangulation
        if len(per_cam_markers_und) >= 2:
            marker_to_cams: Dict[int, List[int]] = defaultdict(list)
            for ci, mids in per_cam_markers_und.items():
                for mid in mids:
                    marker_to_cams[mid].append(ci)

            for mid, ci_list in marker_to_cams.items():
                ci_list = sorted(set(ci_list))
                if len(ci_list) < 2:
                    continue
                tri_corners = np.zeros((4, 3))
                ok = True
                for k in range(4):
                    pixels = np.stack(
                        [per_cam_markers_und[ci][mid][k] for ci in ci_list], axis=0
                    )
                    P_list = np.stack([P_i_map[ci] for ci in ci_list], axis=0)
                    X = triangulate_n_views(pixels, P_list)
                    if not np.all(np.isfinite(X)):
                        ok = False
                        break
                    tri_corners[k] = X

                    # per-ray perp residual
                    ray_res_by_cam = {}
                    for ci, px in zip(ci_list, pixels):
                        o = cam_center_C0[ci]
                        d = R_C0_Ci_map[ci] @ np.array([px[0], px[1], 1.0])
                        perp_mm = ray_perp_distance(X, o, d) * 1000.0
                        ray_res_by_cam[ci] = perp_mm
                        rec["tri_all"].append(perp_mm)
                    # 페어별: cam pair (i,j) 양쪽 ray 잔차의 평균
                    for i in range(len(ci_list)):
                        for j in range(i + 1, len(ci_list)):
                            ci_a, ci_b = ci_list[i], ci_list[j]
                            pair = (ci_a, ci_b)
                            rec["tri_per_pair"][pair].append(
                                0.5 * (ray_res_by_cam[ci_a] + ray_res_by_cam[ci_b])
                            )

                if not ok:
                    continue
                edge_expected = cfg.marker_size_m
                for ia, ib in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    L = np.linalg.norm(tri_corners[ia] - tri_corners[ib])
                    rec["edge_err"].append(abs(L - edge_expected) * 1000.0)

        per_frame[fid] = rec

    # ============================================================
    # 그룹별 통계 출력
    # ============================================================
    groups = ["A (0-36)", "B (37-76)", "C (77-105)"]

    # 1. PnP 품질 per group per cam
    print("\n" + "=" * 80)
    print(" [1] PnP reproj quality — frames 37-76은 진짜 PnP가 나쁜가?")
    print("=" * 80)
    for g in groups:
        print(f"\n  Group {g}:")
        for ci in cam_indices:
            vals = [per_frame[fid]["pnp"][ci]["reproj_mean"]
                    for fid in per_frame
                    if group_of(fid) == g and ci in per_frame[fid]["pnp"]]
            n_markers = [per_frame[fid]["pnp"][ci]["n_markers"]
                         for fid in per_frame
                         if group_of(fid) == g and ci in per_frame[fid]["pnp"]]
            print(f"    cam{ci}  reproj_px : {summarize(vals)}")
            if n_markers:
                arr = np.array(n_markers)
                print(f"           n_markers : n={len(arr)} mean={arr.mean():.2f} "
                      f"(1 marker frames={int((arr==1).sum())})")

    # 2. Cube 위치 (cam0 frame): X/Y/Z 평균
    print("\n" + "=" * 80)
    print(" [2] Cube 3D position in cam0 frame (m) — 공간상 어디에 있나?")
    print("=" * 80)
    for g in groups:
        xs, ys, zs, dists = [], [], [], []
        for fid in per_frame:
            if group_of(fid) != g:
                continue
            if ref_ci not in per_frame[fid]["cube_in_cam"]:
                continue
            c = per_frame[fid]["cube_in_cam"][ref_ci]
            xs.append(float(c[0])); ys.append(float(c[1])); zs.append(float(c[2]))
            dists.append(float(np.linalg.norm(c)))
        if not xs:
            print(f"\n  Group {g}: n=0")
            continue
        xs, ys, zs, dists = map(np.array, (xs, ys, zs, dists))
        print(f"\n  Group {g}:  n={len(xs)}")
        print(f"    X (좌-우)   mean={xs.mean():+7.3f} std={xs.std():.3f} "
              f"range=[{xs.min():+7.3f}, {xs.max():+7.3f}]")
        print(f"    Y (위-아래) mean={ys.mean():+7.3f} std={ys.std():.3f} "
              f"range=[{ys.min():+7.3f}, {ys.max():+7.3f}]")
        print(f"    Z (깊이)    mean={zs.mean():+7.3f} std={zs.std():.3f} "
              f"range=[{zs.min():+7.3f}, {zs.max():+7.3f}]")
        print(f"    cam→cube dist mean={dists.mean():.3f}m")

    # 3. Cube 이미지 위치 (정규화 픽셀: 0=좌상단, 1=우하단)
    print("\n" + "=" * 80)
    print(" [3] Cube 이미지 픽셀 위치 (정규화: 0~1) — 카메라 화면 어디 보이나?")
    print("=" * 80)
    for g in groups:
        print(f"\n  Group {g}:")
        for ci in cam_indices:
            fx, fy = [], []
            for fid in per_frame:
                if group_of(fid) != g:
                    continue
                if ci not in per_frame[fid]["cube_in_img"]:
                    continue
                _, _, nx, ny = per_frame[fid]["cube_in_img"][ci]
                fx.append(nx); fy.append(ny)
            if not fx:
                print(f"    cam{ci}: n=0")
                continue
            fx, fy = np.array(fx), np.array(fy)
            ext_mark_x = " <-edge!" if (fx.min() < 0.15 or fx.max() > 0.85) else ""
            ext_mark_y = " <-edge!" if (fy.min() < 0.15 or fy.max() > 0.85) else ""
            print(f"    cam{ci}: x mean={fx.mean():.2f} range=[{fx.min():.2f}, "
                  f"{fx.max():.2f}]{ext_mark_x}  "
                  f"y mean={fy.mean():.2f} range=[{fy.min():.2f}, "
                  f"{fy.max():.2f}]{ext_mark_y}")

    # 4. 카메라-페어별 triangulation residual
    print("\n" + "=" * 80)
    print(" [4] 카메라 페어별 triangulation residual (mm)")
    print("=" * 80)
    pair_keys = []
    for i in range(len(cam_indices)):
        for j in range(i + 1, len(cam_indices)):
            pair_keys.append((cam_indices[i], cam_indices[j]))

    for g in groups:
        print(f"\n  Group {g}:")
        for pair in pair_keys:
            vals = []
            for fid in per_frame:
                if group_of(fid) != g:
                    continue
                if pair in per_frame[fid]["tri_per_pair"]:
                    vals.extend(per_frame[fid]["tri_per_pair"][pair])
            print(f"    cam{pair[0]}↔cam{pair[1]}: {summarize(vals)}")
        # 전체
        vals_all = []
        for fid in per_frame:
            if group_of(fid) != g:
                continue
            vals_all.extend(per_frame[fid]["tri_all"])
        print(f"    ALL rays    : {summarize(vals_all)}")

    # 5. Edge err per group
    print("\n" + "=" * 80)
    print(" [5] Edge recovery error (mm) — metric scale 직접 영향")
    print("=" * 80)
    for g in groups:
        vals = []
        for fid in per_frame:
            if group_of(fid) != g:
                continue
            vals.extend(per_frame[fid]["edge_err"])
        print(f"  Group {g}: {summarize(vals)}")

    # 6. 결론 도출용 핵심 비교
    print("\n" + "=" * 80)
    print(" [6] 결론 도출 — 가설 검증")
    print("=" * 80)
    pnp_by_g = {g: [] for g in groups}
    tri_by_g = {g: [] for g in groups}
    for fid, rec in per_frame.items():
        g = group_of(fid)
        for ci, info in rec["pnp"].items():
            pnp_by_g[g].append(info["reproj_mean"])
        tri_by_g[g].extend(rec["tri_all"])

    print(f"{'':16s}  {'PnP reproj mean (px)':>22s}  {'Tri residual mean (mm)':>26s}")
    for g in groups:
        pnp_arr = np.array(pnp_by_g[g]) if pnp_by_g[g] else np.array([np.nan])
        tri_arr = np.array(tri_by_g[g]) if tri_by_g[g] else np.array([np.nan])
        print(f"  {g:16s}  {pnp_arr.mean():>14.3f}  ({len(pnp_arr):4d})   "
              f"{tri_arr.mean():>14.3f}  ({len(tri_arr):4d})")

    print("\n  해석:")
    print("    - PnP 평균이 그룹별 비슷 → PnP 자체는 OK")
    print("    - Tri 평균이 B에서만 큼 → 외부 파라미터(T_C0_Ci)가 B 큐브 위치에서만 어긋남")
    print("    - 위 [2]/[3]에서 B 그룹의 cube 위치/이미지 영역이 다른 그룹과 다르면 가설 확정.")


if __name__ == "__main__":
    main()
