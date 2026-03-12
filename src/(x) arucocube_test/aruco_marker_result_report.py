# diagnose_detection.py
"""
각 카메라의 모든 프레임에서 ArUco 마커 감지 상태를 진단합니다.

실행:
  python diagnose_detection.py \
    --root_folder ./data/cube_session_01 \
    --intrinsics_dir ./intrinsics
"""

import os
import glob
import argparse

import cv2
import numpy as np

from src3._aruco_cube import CubeConfig, ArucoCubeTarget


def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    data = np.load(p)
    return data["color_K"].astype(np.float64), data["color_D"].astype(np.float64)


def discover_cams(root_folder: str):
    idxs = []
    for name in os.listdir(root_folder):
        if not name.startswith("cam"):
            continue
        try:
            idx = int(name.replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(os.path.join(root_folder, name, "rgb_*.jpg")):
            idxs.append(idx)
    return sorted(idxs)


def diagnose(root_folder, intrinsics_dir, reproj_max_px):
    cfg  = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    cam_idxs = discover_cams(root_folder)
    print(f"[INFO] 카메라: {cam_idxs}")
    print(f"[INFO] reproj_max_px = {reproj_max_px}px\n")

    total_no_det   = []   # (cam, frame) 마커 아예 미감지
    total_high_err = []   # (cam, frame, err) 감지됐지만 오차 초과
    total_ok       = []   # (cam, frame, err) 성공

    for ci in cam_idxs:
        try:
            K, D = load_intrinsics(intrinsics_dir, ci)
        except FileNotFoundError:
            print(f"[WARN] cam{ci}: intrinsics 없음, skip")
            continue

        cam_dir   = os.path.join(root_folder, f"cam{ci}")
        rgb_files = sorted(glob.glob(os.path.join(cam_dir, "rgb_*.jpg")))

        no_det   = []
        high_err = []
        ok_frames = []

        for rgb_path in rgb_files:
            frame_idx = int(
                os.path.basename(rgb_path).split("_")[-1].split(".")[0]
            )
            img = cv2.imread(rgb_path)
            if img is None:
                no_det.append(frame_idx)
                continue

            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img, K, D,
                use_ransac=False,
                min_markers=1,
                reproj_thr_mean_px=reproj_max_px,
                return_reproj=True,
            )

            if reproj is None:
                # 마커 감지 자체 실패 또는 solvePnP 실패
                # 감지 여부 재확인
                corners_list, ids = cube.detect(img)
                if ids is None or len(ids) == 0:
                    no_det.append(frame_idx)
                else:
                    # 마커는 보이지만 solvePnP 자체 실패 (드문 경우)
                    detected_ids = [int(i) for i in ids if int(i) in cfg.id_to_face]
                    high_err.append((frame_idx, float("inf"), detected_ids))
            else:
                err_mean = reproj["err_mean"]
                if not ok:
                    high_err.append((frame_idx, err_mean, [int(x) for x in used]))
                else:
                    ok_frames.append((frame_idx, err_mean, [int(x) for x in used]))

        # ---- 카메라별 요약 ----
        n_total = len(rgb_files)
        print(f"{'='*60}")
        print(f"cam{ci}:  전체 {n_total}프레임")
        print(f"  성공         : {len(ok_frames):3d}프레임")
        print(f"  고오차(>{reproj_max_px}px): {len(high_err):3d}프레임")
        print(f"  마커 미감지  : {len(no_det):3d}프레임")

        if no_det:
            print(f"\n  [미감지 프레임] {no_det}")

        if high_err:
            print(f"\n  [고오차 프레임]")
            for fid, err, uids in sorted(high_err):
                err_str = f"{err:.2f}px" if err != float("inf") else "PnP실패"
                print(f"    frame {fid:5d}  err={err_str}  used={uids}")

        if ok_frames:
            errs = [e for _, e, _ in ok_frames]
            print(f"\n  [성공 프레임]  reproj: "
                  f"min={min(errs):.3f}  mean={np.mean(errs):.3f}  max={max(errs):.3f} px")
            for fid, err, uids in sorted(ok_frames):
                print(f"    frame {fid:5d}  err={err:.2f}px  used={uids}")

        total_no_det   += [(ci, f) for f in no_det]
        total_high_err += [(ci, f, e, u) for f, e, u in high_err]
        total_ok       += [(ci, f, e, u) for f, e, u in ok_frames]

        print()

    # ---- 전체 요약 ----
    print(f"{'='*60}")
    print(f"[전체 요약]")
    print(f"  성공         : {len(total_ok)}건")
    print(f"  고오차       : {len(total_high_err)}건")
    print(f"  마커 미감지  : {len(total_no_det)}건")


def main():
    parser = argparse.ArgumentParser(description="마커 감지 진단")
    parser.add_argument("--root_folder",    required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--reproj_max_px",  type=float, default=5.0)
    args = parser.parse_args()

    diagnose(args.root_folder, args.intrinsics_dir, args.reproj_max_px)


if __name__ == "__main__":
    main()
