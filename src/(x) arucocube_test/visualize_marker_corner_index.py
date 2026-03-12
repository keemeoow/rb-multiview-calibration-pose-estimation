"""
Save visualization images with marker corner indices (0,1,2,3) for one frame.

Usage:
  python visualize_marker_corner_indices.py \
    --root_folder ./data/cube_session_01 \
    --frame_idx 0
"""

import os
import glob
import argparse

import cv2
import numpy as np

from src3._aruco_cube import CubeConfig, ArucoCubeTarget


CORNER_COLORS = [
    (0, 255, 255),   # 0
    (0, 200, 0),     # 1
    (255, 180, 0),   # 2
    (255, 0, 255),   # 3
]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def discover_cams(root_folder: str):
    cam_idxs = []
    for name in os.listdir(root_folder):
        if not name.startswith("cam"):
            continue
        try:
            idx = int(name.replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(os.path.join(root_folder, name, "rgb_*.jpg")):
            cam_idxs.append(idx)
    return sorted(cam_idxs)


def draw_marker_corner_indices(image_bgr, corners_list, ids):
    vis = image_bgr.copy()

    if ids is None or len(ids) == 0:
        return vis, 0

    try:
        cv2.aruco.drawDetectedMarkers(vis, corners_list, ids.reshape(-1, 1))
    except Exception:
        pass

    n_markers = 0
    for corners, mid in zip(corners_list, ids):
        pts = corners.reshape(4, 2)
        n_markers += 1

        center = pts.mean(axis=0)
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.putText(
            vis,
            f"id={int(mid)}",
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"id={int(mid)}",
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        for i, (x, y) in enumerate(pts):
            xi, yi = int(round(x)), int(round(y))
            color = CORNER_COLORS[i % len(CORNER_COLORS)]
            cv2.circle(vis, (xi, yi), 6, color, -1)
            cv2.putText(
                vis,
                str(i),
                (xi + 6, yi - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                str(i),
                (xi + 6, yi - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    legend = "corner idx colors: 0(yellow), 1(green), 2(orange), 3(magenta)"
    cv2.putText(vis, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return vis, n_markers


def main():
    parser = argparse.ArgumentParser(description="Visualize ArUco marker corner index order for one frame.")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="default: <root_folder>/corner_index_frameXXXXX",
    )
    args = parser.parse_args()

    cfg = CubeConfig()
    detector = ArucoCubeTarget(cfg)

    cam_idxs = discover_cams(args.root_folder)
    if not cam_idxs:
        raise RuntimeError(f"No camera folders found under {args.root_folder}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(args.root_folder, f"corner_index_frame{args.frame_idx:05d}")
    ensure_dir(out_dir)

    print(f"[INFO] cams: {cam_idxs}")
    print(f"[INFO] frame_idx: {args.frame_idx}")
    print(f"[INFO] save dir: {out_dir}")

    for ci in cam_idxs:
        rgb_path = os.path.join(args.root_folder, f"cam{ci}", f"rgb_{args.frame_idx:05d}.jpg")
        if not os.path.exists(rgb_path):
            print(f"[WARN] cam{ci}: missing {rgb_path}")
            continue

        img = cv2.imread(rgb_path)
        if img is None:
            print(f"[WARN] cam{ci}: failed to read image")
            continue

        corners_list, ids = detector.detect(img)
        if ids is None:
            vis = img.copy()
            cv2.putText(vis, "No markers detected", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            n_markers = 0
        else:
            vis, n_markers = draw_marker_corner_indices(img, corners_list, ids)

        save_path = os.path.join(out_dir, f"cam{ci}_frame{args.frame_idx:05d}_corners.jpg")
        cv2.imwrite(save_path, vis)
        print(f"[SAVE] cam{ci}: markers={n_markers} -> {save_path}")

    print("[INFO] done.")


if __name__ == "__main__":
    main()
