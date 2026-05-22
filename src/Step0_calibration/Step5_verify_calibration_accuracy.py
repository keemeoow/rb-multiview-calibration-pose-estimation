# Step5_verify_calibration_accuracy.py
# 정량적 캘리브레이션 정확도 점검
#   (1) Cross-view triangulation 잔차 (mm)
#       - >=2 cam에서 본 같은 마커 코너를 N-view DLT로 C0 좌표계에 삼각측량
#       - 각 카메라 ray에 대한 perpendicular 3D residual의 mean/median/RMS/p90 (mm)
#   (2) Marker edge / diagonal 회복 오차 (mm)
#       - 삼각측량된 4개 코너로 만든 정사각형의 변/대각 길이가
#         실제 marker 크기와 얼마나 일치하는지 → metric scale 직접 검증
#
# Step3가 끝나서 calib_out_cube/T_C0_C{i}.npy 가 있는 세션이면 그대로 동작.
# Depth가 없는 세션(cube_session_01 같은)에서도 RGB만 있으면 검증 가능.
"""
python Step5_verify_calibration_accuracy.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0
"""

import os
import sys
import json
import argparse
from typing import Dict, List

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from _aruco_cube import CubeConfig, ArucoCubeTarget


def load_intrinsics(intr_dir: str, ci: int):
    p = os.path.join(intr_dir, f"cam{ci}.npz")
    z = np.load(p, allow_pickle=True)
    K = z["color_K"].astype(np.float64)
    D = z["color_D"].astype(np.float64).reshape(-1)
    return K, D


def load_T_C0_Ci(calib_dir: str, ref_cam_idx: int, ci: int) -> np.ndarray:
    if ci == ref_cam_idx:
        return np.eye(4, dtype=np.float64)
    p = os.path.join(calib_dir, f"T_C{ref_cam_idx}_C{ci}.npy")
    return np.load(p).astype(np.float64)


def triangulate_n_views(pixels_n: np.ndarray, P_n: np.ndarray) -> np.ndarray:
    """DLT triangulation. pixels_n:(N,2) normalized; P_n:(N,3,4) maps C0->cam_i."""
    rows = []
    for (x, y), P in zip(pixels_n, P_n):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.stack(rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return Xh[:3] / Xh[3]


def ray_perp_distance(X, origin, direction) -> float:
    d = direction / (np.linalg.norm(direction) + 1e-12)
    v = X - origin
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--ref_cam_idx", type=int, default=0)
    parser.add_argument("--calib_dir", default=None,
                        help="Default: <root>/calib_out_cube")
    parser.add_argument("--marker_size_m", type=float, default=None,
                        help="Override CubeConfig.marker_size_m (m)")
    parser.add_argument("--out_csv", default=None,
                        help="Optional: write summary metrics to CSV")
    args = parser.parse_args()

    root = args.root_folder
    calib_dir = args.calib_dir or os.path.join(root, "calib_out_cube")
    meta_path = os.path.join(root, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    captures = meta["captures"]
    print(f"[INFO] Loaded {len(captures)} captures from {meta_path}")

    cfg = CubeConfig()
    if args.marker_size_m is not None:
        cfg.marker_size_m = float(args.marker_size_m)
    cube = ArucoCubeTarget(cfg)

    edge_expected = cfg.marker_size_m
    diag_expected = edge_expected * np.sqrt(2.0)
    print(f"[INFO] Expected marker edge = {edge_expected*1000:.2f} mm  "
          f"(diagonal = {diag_expected*1000:.2f} mm)")

    cam_indices = sorted({int(k) for cap in captures for k in cap["cams"].keys()})
    print(f"[INFO] Cameras: {cam_indices}, ref=cam{args.ref_cam_idx}")

    K_map, D_map = {}, {}
    P_i_map = {}        # 3x4: C0 -> normalized cam_i
    cam_center_C0 = {}
    R_C0_Ci_map = {}
    for ci in cam_indices:
        K_map[ci], D_map[ci] = load_intrinsics(args.intrinsics_dir, ci)
        T_C0_Ci = load_T_C0_Ci(calib_dir, args.ref_cam_idx, ci)
        T_Ci_C0 = np.linalg.inv(T_C0_Ci)
        P_i_map[ci] = T_Ci_C0[:3, :]
        cam_center_C0[ci] = T_C0_Ci[:3, 3].copy()
        R_C0_Ci_map[ci] = T_C0_Ci[:3, :3].copy()

    tri_residuals_mm = []
    edge_errors_mm = []
    diag_errors_mm = []
    per_frame = []
    per_marker_seen = {}  # mid -> count

    for cap in captures:
        fid = int(cap["event_id"])
        cams = cap["cams"]

        per_cam_markers: Dict[int, Dict[int, np.ndarray]] = {}
        for ci_str, info in cams.items():
            ci = int(ci_str)
            if not info.get("saved", False):
                continue
            rgb_path = info.get("rgb_path")
            if not rgb_path:
                continue
            img_path = rgb_path if os.path.isabs(rgb_path) else os.path.join(root, rgb_path)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            corners_list, ids = cube.detect(img)
            if ids is None:
                continue
            entry = {}
            for c, mid in zip(corners_list, ids):
                mid = int(mid)
                if mid not in cfg.id_to_face:
                    continue
                px = c.reshape(4, 2).astype(np.float64)
                und = cv2.undistortPoints(
                    px.reshape(-1, 1, 2), K_map[ci], D_map[ci]
                ).reshape(-1, 2)
                entry[mid] = und
            if entry:
                per_cam_markers[ci] = entry

        if len(per_cam_markers) < 2:
            continue

        marker_to_cams: Dict[int, List[int]] = {}
        for ci, mids in per_cam_markers.items():
            for mid in mids:
                marker_to_cams.setdefault(mid, []).append(ci)

        frame_tri = []
        frame_edge = []
        for mid, ci_list in marker_to_cams.items():
            ci_list = sorted(ci_list)
            if len(ci_list) < 2:
                continue
            per_marker_seen[mid] = per_marker_seen.get(mid, 0) + 1

            tri_corners = np.zeros((4, 3), dtype=np.float64)
            ok_marker = True
            for k in range(4):
                pixels = np.stack(
                    [per_cam_markers[ci][mid][k] for ci in ci_list], axis=0
                )
                P_list = np.stack([P_i_map[ci] for ci in ci_list], axis=0)
                X = triangulate_n_views(pixels, P_list)
                if not np.all(np.isfinite(X)):
                    ok_marker = False
                    break
                tri_corners[k] = X
                for ci, px in zip(ci_list, pixels):
                    o = cam_center_C0[ci]
                    d = R_C0_Ci_map[ci] @ np.array([px[0], px[1], 1.0])
                    perp_mm = ray_perp_distance(X, o, d) * 1000.0
                    tri_residuals_mm.append(perp_mm)
                    frame_tri.append(perp_mm)

            if not ok_marker:
                continue

            for i_a, i_b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                length = np.linalg.norm(tri_corners[i_a] - tri_corners[i_b])
                err_mm = abs(length - edge_expected) * 1000.0
                edge_errors_mm.append(err_mm)
                frame_edge.append(err_mm)
            for i_a, i_b in [(0, 2), (1, 3)]:
                length = np.linalg.norm(tri_corners[i_a] - tri_corners[i_b])
                diag_errors_mm.append(abs(length - diag_expected) * 1000.0)

        if frame_tri:
            per_frame.append({
                "frame": fid,
                "n_obs": len(frame_tri),
                "tri_mean": float(np.mean(frame_tri)),
                "tri_p90": float(np.percentile(frame_tri, 90)),
                "edge_mean": float(np.mean(frame_edge)) if frame_edge else float("nan"),
            })

    if not tri_residuals_mm:
        print("[WARN] No multi-view marker observations found.")
        return

    arr_tri = np.array(tri_residuals_mm)
    arr_edge = np.array(edge_errors_mm)
    arr_diag = np.array(diag_errors_mm)

    print()
    print("=" * 72)
    print(" CALIBRATION ACCURACY REPORT")
    print("=" * 72)
    print(f"Frames analyzed:                       {len(per_frame)} / {len(captures)}")
    print(f"Multi-view corner observations (ray):  {len(arr_tri)}")
    print(f"Marker observations (triangulated):    {len(arr_edge)//4}")
    print(f"Markers seen in >=2 cams (count):      "
          + ", ".join(f"id{mid}:{n}" for mid, n in sorted(per_marker_seen.items())))
    print()
    print("[1] Cross-view triangulation residual  (3D perp dist to rays, mm)")
    print(f"      mean   = {arr_tri.mean():8.4f}")
    print(f"      median = {np.median(arr_tri):8.4f}")
    print(f"      RMS    = {np.sqrt(np.mean(arr_tri**2)):8.4f}")
    print(f"      p90    = {np.percentile(arr_tri, 90):8.4f}")
    print(f"      max    = {arr_tri.max():8.4f}")
    print()
    print(f"[2] Marker edge recovery error         (expected {edge_expected*1000:.2f} mm)")
    print(f"      mean abs = {arr_edge.mean():8.4f} mm")
    print(f"      median   = {np.median(arr_edge):8.4f} mm")
    print(f"      RMS      = {np.sqrt(np.mean(arr_edge**2)):8.4f} mm")
    print(f"      p90      = {np.percentile(arr_edge, 90):8.4f} mm")
    print(f"      max      = {arr_edge.max():8.4f} mm")
    print(f"      relative = {arr_edge.mean()/(edge_expected*1000.0)*100:.2f}%")
    print()
    print(f"[3] Marker diagonal recovery error     (expected {diag_expected*1000:.2f} mm)")
    print(f"      mean abs = {arr_diag.mean():8.4f} mm")
    print(f"      median   = {np.median(arr_diag):8.4f} mm")
    print()

    print("Per-frame summary:")
    print(f"  {'frame':>5} {'n_rays':>6} {'tri_mean':>10} {'tri_p90':>10} {'edge_mean':>10}")
    for r in per_frame:
        print(f"  {r['frame']:>5d} {r['n_obs']:>6d} "
              f"{r['tri_mean']:>9.4f}mm {r['tri_p90']:>9.4f}mm "
              f"{r['edge_mean']:>9.4f}mm")

    print()
    print("Interpretation guide:")
    print("  - 3D residual << 1mm + edge err << 0.5mm  ->  mesh metric scale 신뢰 OK")
    print("  - 3D residual 1~3mm or edge err 0.5~2mm   ->  보통, 큰 물체에 적합")
    print("  - 그 이상                                  ->  재캘리브 권장")

    if args.out_csv:
        out_dir = os.path.dirname(args.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["tri_residual_mean_mm", float(arr_tri.mean())])
            w.writerow(["tri_residual_median_mm", float(np.median(arr_tri))])
            w.writerow(["tri_residual_rms_mm", float(np.sqrt(np.mean(arr_tri**2)))])
            w.writerow(["tri_residual_p90_mm", float(np.percentile(arr_tri, 90))])
            w.writerow(["tri_residual_max_mm", float(arr_tri.max())])
            w.writerow(["edge_err_mean_mm", float(arr_edge.mean())])
            w.writerow(["edge_err_median_mm", float(np.median(arr_edge))])
            w.writerow(["edge_err_rms_mm", float(np.sqrt(np.mean(arr_edge**2)))])
            w.writerow(["edge_err_p90_mm", float(np.percentile(arr_edge, 90))])
            w.writerow(["edge_err_relative_pct",
                        float(arr_edge.mean() / (edge_expected * 1000.0) * 100.0)])
            w.writerow(["diag_err_mean_mm", float(arr_diag.mean())])
            w.writerow(["expected_edge_mm", float(edge_expected * 1000.0)])
            w.writerow(["frames_analyzed", len(per_frame)])
            w.writerow(["multi_view_obs", len(arr_tri)])
        print(f"\n[SAVE] {args.out_csv}")


if __name__ == "__main__":
    main()
