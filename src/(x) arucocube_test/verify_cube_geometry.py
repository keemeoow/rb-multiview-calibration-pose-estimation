# verify_cube_geometry.py
# 큐브 면 정의(id_to_face)가 올바른지 시각화로 확인
#
# 동작:
#   각 카메라의 지정 프레임에서 ArUco를 감지하고,
#   감지된 마커 1개 이상으로 큐브 pose를 추정한 뒤,
#   나머지 모든 마커의 예측 위치를 이미지에 오버레이한다.
#
#   - 실선 사각형: PnP에 사용된 마커 (감지됨)
#   - 점선 사각형: 예측만 된 마커 (감지 안 됨)
#   id_to_face가 올바르다면 예측 위치가 실제 마커와 겹쳐야 한다.
#
# 실행 예:
"""   
python verify_cube_geometry.py \
     --root_folder ./data/cube_session_02 \
     --intrinsics_dir ./intrinsics \
     --frame_idx 0
"""

import os
import argparse
import glob

import cv2
import numpy as np

from src3._aruco_cube import CubeConfig, ArucoCubeTarget


# 마커 ID별 색상 (BGR)
MARKER_COLORS = {
    0: (0,   220,   0),    # 초록
    1: (0,    80, 255),    # 주황
    2: (255,   0,   0),    # 파랑
    3: (0,   200, 255),    # 노랑
    4: (180,   0, 255),    # 보라
}


def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"intrinsics 없음: {p}")
    data = np.load(p)
    return data["color_K"].astype(np.float64), data["color_D"].astype(np.float64)


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


def draw_quad(img, pts4x2, color, thickness=2, dashed=False):
    """4점을 연결하는 사각형을 그린다. dashed=True면 점선."""
    pts = pts4x2.astype(int)
    h, w = img.shape[:2]
    for i in range(4):
        p1 = tuple(np.clip(pts[i],     [0, 0], [w-1, h-1]))
        p2 = tuple(np.clip(pts[(i+1) % 4], [0, 0], [w-1, h-1]))
        if dashed:
            # 점선: 선분을 10px 간격으로 끊어서 그림
            pt1 = np.array(p1, dtype=float)
            pt2 = np.array(p2, dtype=float)
            length = np.linalg.norm(pt2 - pt1)
            if length < 1:
                continue
            n_seg = max(1, int(length / 10))
            for s in range(n_seg):
                if s % 2 == 0:
                    a = pt1 + (pt2 - pt1) * (s / n_seg)
                    b = pt1 + (pt2 - pt1) * ((s + 1) / n_seg)
                    cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness)
        else:
            cv2.line(img, p1, p2, color, thickness)


def draw_corners(img, pts4x2, color, radius=5, filled=True):
    pts = pts4x2.astype(int)
    h, w = img.shape[:2]
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(img, (x, y), radius, color, -1 if filled else 1)


def put_label(img, text_lines, pos_xy, color, font_scale=0.5):
    x, y = int(pos_xy[0]), int(pos_xy[1])
    h, w = img.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return
    for i, txt in enumerate(text_lines):
        yy = y + i * 18
        cv2.putText(img, txt, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)


def visualize_camera(img, cube, K, D, reproj_max_px=50.0):
    """
    한 장의 이미지에서 큐브 pose를 추정하고 모든 마커를 오버레이한 이미지를 반환.
    """
    vis = img.copy()
    h, w = vis.shape[:2]
    cfg = cube.cfg

    corners_list, ids = cube.detect(img)

    if ids is None or len(ids) == 0:
        cv2.putText(vis, "NO MARKERS DETECTED", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis, False

    # ArUco 감지 결과 표시 (흰 테두리)
    cv2.aruco.drawDetectedMarkers(vis, corners_list,
                                  ids.reshape(-1, 1).astype(np.int32))

    # PnP (임계값 넉넉하게 → 시각화용)
    ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
        img, K, D,
        use_ransac=False,
        min_markers=1,
        reproj_thr_mean_px=reproj_max_px,
        return_reproj=True,
    )

    if reproj is None:
        cv2.putText(vis, "PnP FAILED (no solution)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis, False

    err_mean = reproj["err_mean"]
    ok_str = "OK" if ok else f"HIGH_ERR({err_mean:.1f}px)"

    # 모든 마커 투영
    projected = cube.project_all_markers(rvec, tvec, K, D)

    for mid, pts in projected.items():
        face = cfg.id_to_face[mid]
        color = MARKER_COLORS.get(mid, (200, 200, 200))
        is_used = mid in used

        # 사각형: 사용됨 → 실선, 예측만 → 점선
        draw_quad(vis, pts, color, thickness=2, dashed=not is_used)
        draw_corners(vis, pts, color, radius=4, filled=is_used)

        # 레이블
        center = pts.mean(axis=0)
        status = "DETECTED" if is_used else "predicted"
        put_label(vis, [f"id{mid}: {face}", status], center + np.array([-30, -12]), color)

    # 큐브 원점 + 축 그리기
    axis_len = cfg.cube_side_m * 1.5
    axis_pts_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len],
    ])
    axis_2d, _ = cv2.projectPoints(axis_pts_3d, rvec, tvec, K, D)
    axis_2d = axis_2d.reshape(-1, 2).astype(int)

    orig = tuple(np.clip(axis_2d[0], [0, 0], [w-1, h-1]))
    for i, (col, lbl) in enumerate(zip([(0,0,255),(0,255,0),(255,0,0)], ["+X","+Y","+Z"])):
        ep = tuple(np.clip(axis_2d[i+1], [0, 0], [w-1, h-1]))
        cv2.arrowedLine(vis, orig, ep, col, 2, tipLength=0.15)
        cv2.putText(vis, lbl, ep, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(vis, lbl, ep, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)

    # 상단 정보
    header = (f"PnP {ok_str}  used={used}  reproj_mean={err_mean:.2f}px  "
              f"detected_ids={list(ids)}")
    cv2.putText(vis, header, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(vis, header, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return vis, ok


def main():
    parser = argparse.ArgumentParser(
        description="큐브 id_to_face 정의 검증 시각화"
    )
    parser.add_argument("--root_folder",    required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--frame_idx",      type=int,   default=0)
    parser.add_argument("--reproj_max_px",  type=float, default=50.0,
                        help="PnP 통과 임계값 (시각화용이라 넉넉하게)")
    args = parser.parse_args()

    cfg  = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    cam_idxs = discover_cams(args.root_folder)
    if not cam_idxs:
        raise RuntimeError(f"카메라 폴더 없음: {args.root_folder}")

    out_dir = os.path.join(args.root_folder, "geometry_verify")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[INFO] id_to_face 매핑:")
    for mid, face in cfg.id_to_face.items():
        print(f"       마커 {mid} → {face} face")
    print()

    for ci in cam_idxs:
        rgb_path = os.path.join(
            args.root_folder, f"cam{ci}", f"rgb_{args.frame_idx:05d}.jpg"
        )
        if not os.path.exists(rgb_path):
            print(f"[WARN] cam{ci}: rgb_{args.frame_idx:05d}.jpg 없음, skip")
            continue

        img = cv2.imread(rgb_path)
        if img is None:
            print(f"[WARN] cam{ci}: 이미지 로드 실패")
            continue

        try:
            K, D = load_intrinsics(args.intrinsics_dir, ci)
        except FileNotFoundError as e:
            print(f"[WARN] cam{ci}: {e}")
            continue

        vis, ok = visualize_camera(img, cube, K, D,
                                   reproj_max_px=args.reproj_max_px)

        out_path = os.path.join(out_dir, f"cam{ci}_frame{args.frame_idx:05d}_verify.jpg")
        cv2.imwrite(out_path, vis)
        status = "OK" if ok else "PnP_ERR or HIGH_REPROJ"
        print(f"[SAVE] cam{ci} → {out_path}  [{status}]")

    print(f"\n[완료] 결과 이미지: {out_dir}/")
    print("  - 실선 사각형 = PnP에 사용된 마커 (감지됨)")
    print("  - 점선 사각형 = 큐브 pose로 예측한 마커 (미감지)")
    print("  - 예측 위치가 실제 마커와 겹치면 id_to_face 정의 OK")


if __name__ == "__main__":
    main()
