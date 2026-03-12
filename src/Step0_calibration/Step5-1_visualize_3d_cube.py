# visualize_3d.py
"""
Matplotlib 3D로 시각화:
  - 큐브 형상 + 각 면의 마커 ID
  - 각 카메라의 위치 및 방향 (PnP로 추정)

실행:
  python visualize_3d.py \
    --root_folder ./data/cube_session_01 \
    --intrinsics_dir ./data/_intrinsics
"""

import os
import glob
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src3._aruco_cube import CubeConfig, ArucoCubeModel, ArucoCubeTarget


CAM_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

FACE_COLORS = {
    "+X": "#FF6666", "-X": "#FFAAAA",
    "+Y": "#66BB66", "-Y": "#AADDAA",
    "+Z": "#6699FF", "-Z": "#AACCFF",
}

# 각 face의 법선 방향 (outward)
FACE_NORMALS = {
    "+X": np.array([1, 0, 0]),
    "-X": np.array([-1, 0, 0]),
    "+Y": np.array([0, 1, 0]),
    "-Y": np.array([0, -1, 0]),
    "+Z": np.array([0, 0, 1]),
    "-Z": np.array([0, 0, -1]),
}


# ------------------------------------------------------------------ #
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


def get_best_pnp(root_folder: str, intrinsics_dir: str,
                 cam_idx: int, cube: ArucoCubeTarget,
                 max_frames: int = 60):
    """
    카메라 cam_idx의 프레임들에서 가장 낮은 reproj 오차의 PnP 결과를 반환.
    Returns: dict(R, t, err, used, frame) or None
    """
    try:
        K, D = load_intrinsics(intrinsics_dir, cam_idx)
    except FileNotFoundError:
        return None

    cam_dir = os.path.join(root_folder, f"cam{cam_idx}")
    rgb_files = sorted(glob.glob(os.path.join(cam_dir, "rgb_*.jpg")))

    best = None
    best_score = float("inf")

    for rgb_path in rgb_files[:max_frames]:
        img = cv2.imread(rgb_path)
        if img is None:
            continue

        ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
            img, K, D,
            use_ransac=False,
            min_markers=1,
            reproj_thr_mean_px=50.0,   # 시각화용: 넉넉하게 (마커 1개도 통과)
            return_reproj=True,
        )
        if reproj is None:
            continue

        # 마커 수가 많을수록, 오차가 낮을수록 좋음
        score = reproj["err_mean"] / max(len(used), 1)
        if score < best_score:
            best_score = score
            R, _ = cv2.Rodrigues(rvec)
            frame_idx = int(
                os.path.basename(rgb_path).split("_")[-1].split(".")[0]
            )
            best = dict(R=R, t=tvec.flatten(),
                        err=reproj["err_mean"], used=used,
                        frame=frame_idx, K=K, D=D)

    return best


# ------------------------------------------------------------------ #
def face_poly_3d(face_name: str, d: float):
    """face_name에 해당하는 정사각형 꼭짓점 4개 반환 (순서: CCW from outside)."""
    polys = {
        "+X": np.array([(d, -d, -d), (d,  d, -d), (d,  d,  d), (d, -d,  d)]),
        "-X": np.array([(-d, -d, -d), (-d, -d,  d), (-d,  d,  d), (-d,  d, -d)]),
        "+Y": np.array([(-d, d, -d), (d,  d, -d), (d,  d,  d), (-d,  d,  d)]),
        "-Y": np.array([(-d, -d, -d), (-d, -d,  d), (d, -d,  d), (d, -d, -d)]),
        "+Z": np.array([(-d, -d,  d), (d, -d,  d), (d,  d,  d), (-d,  d,  d)]),
        "-Z": np.array([(-d, -d, -d), (-d,  d, -d), (d,  d, -d), (d, -d, -d)]),
    }
    return polys[face_name]


def draw_cube(ax: plt.Axes, cfg: CubeConfig):
    """큐브 면(반투명) + 마커 테두리 + 마커 ID 레이블 그리기."""
    d = cfg.cube_side_m / 2.0
    model = ArucoCubeModel(cfg)
    face_to_id = {v: k for k, v in cfg.id_to_face.items()}

    for face_name, color in FACE_COLORS.items():
        poly_verts = face_poly_3d(face_name, d)
        mid = face_to_id.get(face_name)

        # 반투명 면
        fc = Poly3DCollection(
            [poly_verts.tolist()],
            alpha=0.18, facecolor=color, edgecolor="gray", linewidth=0.8
        )
        ax.add_collection3d(fc)

        # 마커 테두리 (굵은 선)
        if mid is not None:
            mc = model.marker_corners_in_rig(mid)
            loop = np.vstack([mc, mc[0]])          # 닫힌 루프
            ax.plot3D(loop[:, 0], loop[:, 1], loop[:, 2],
                      color=color, linewidth=2.5, solid_capstyle="round")

        # 레이블: 면 바깥쪽에 마커 ID + 면 이름
        center = poly_verts.mean(axis=0)
        normal = FACE_NORMALS[face_name]
        label_pos = center + normal * d * 0.55    # 면에서 약간 바깥으로
        txt = f"id {mid}\n({face_name})" if mid is not None else face_name
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                txt, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec=color, alpha=0.75))

    # 큐브 중심 좌표계 (작은 RGB 화살표)
    arrow_len = d * 0.8
    for direction, color, lbl in [
        ([arrow_len, 0, 0], "red",   "+X"),
        ([0, arrow_len, 0], "green", "+Y"),
        ([0, 0, arrow_len], "blue",  "+Z"),
    ]:
        ax.quiver(0, 0, 0, *direction,
                  color=color, arrow_length_ratio=0.25,
                  linewidth=2, alpha=0.9)
        ax.text(direction[0] * 1.15, direction[1] * 1.15, direction[2] * 1.15,
                lbl, color=color, fontsize=9, fontweight="bold")


def draw_camera(ax: plt.Axes, R_CO: np.ndarray, t_CO: np.ndarray,
                color: str, label: str, scale: float = 0.05):
    """
    카메라 위치와 방향을 큐브 좌표계에서 그린다.
    R_CO, t_CO: PnP 결과 (Obj -> Cam 변환)
    카메라 원점 in cube frame = -R_CO.T @ t_CO
    """
    t_CO = t_CO.flatten()
    cam_pos = -R_CO.T @ t_CO        # 큐브 좌표계에서의 카메라 위치

    # 카메라 마커 (삼각형)
    ax.scatter(*cam_pos, s=120, c=color, marker="^", zorder=6, edgecolors="k")
    ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + scale * 0.4,
            label, color=color, fontsize=10, fontweight="bold", ha="center")

    # 카메라 좌표축 (큐브 프레임에서 표현)
    R_OC = R_CO.T                   # 카메라 → 큐브 방향 변환
    axis_labels = ["+X", "+Y", "+Z"]
    axis_colors = ["#CC3333", "#33AA33", "#3333CC"]
    for i in range(3):
        direction = R_OC[:, i] * scale
        ax.quiver(*cam_pos, *direction,
                  color=axis_colors[i], arrow_length_ratio=0.35,
                  linewidth=1.5, alpha=0.8)

    # 카메라 → 큐브 원점 연결선 (점선)
    ax.plot3D([cam_pos[0], 0], [cam_pos[1], 0], [cam_pos[2], 0],
              "--", color=color, alpha=0.25, linewidth=1)


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="큐브 + 카메라 3D 시각화")
    parser.add_argument("--root_folder",    required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--max_frames",     type=int, default=60,
                        help="카메라 당 PnP 탐색 최대 프레임 수")
    parser.add_argument("--save",           action="store_true",
                        help="화면 표시 대신 PNG로 저장")
    args = parser.parse_args()

    cfg  = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    print(f"\n[INFO] 큐브 설정:")
    print(f"  cube_side_m   = {cfg.cube_side_m * 100:.1f} cm")
    print(f"  marker_size_m = {cfg.marker_size_m * 100:.1f} cm")
    print(f"  id_to_face:")
    for mid, face in sorted(cfg.id_to_face.items()):
        print(f"    마커 {mid} → {face} 면")

    cam_idxs = discover_cams(args.root_folder)
    print(f"\n[INFO] 발견된 카메라: {cam_idxs}")

    # 각 카메라의 best PnP 추정
    cam_poses = {}
    for ci in cam_idxs:
        result = get_best_pnp(args.root_folder, args.intrinsics_dir,
                              ci, cube, max_frames=args.max_frames)
        if result:
            cam_poses[ci] = result
            print(f"[INFO] cam{ci}: frame {result['frame']}  "
                  f"used={result['used']}  reproj={result['err']:.2f}px")
        else:
            print(f"[WARN] cam{ci}: PnP 실패 → 위치 표시 안 됨")

    # ---- 3D Plot ----
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    draw_cube(ax, cfg)

    for i, (ci, result) in enumerate(sorted(cam_poses.items())):
        color = CAM_COLORS[i % len(CAM_COLORS)]
        draw_camera(ax, result["R"], result["t"],
                    color=color,
                    label=f"cam{ci}\n(f{result['frame']}, {result['err']:.1f}px)",
                    scale=cfg.cube_side_m * 0.7)

    # ---- 정육면체 비율 강제 (equal aspect ratio) ----
    # 모든 점(큐브 꼭짓점 + 카메라 위치)을 모아 bounding box 계산
    d = cfg.cube_side_m / 2.0
    all_pts = [face_poly_3d(f, d) for f in FACE_NORMALS]
    for result in cam_poses.values():
        cam_pos = -result["R"].T @ result["t"]
        all_pts.append(cam_pos.reshape(1, 3))
    all_pts = np.concatenate(all_pts, axis=0)

    half_range = np.ptp(all_pts, axis=0).max() / 2.0 * 1.15   # 15% 여백
    mid_pt     = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0

    ax.set_xlim(mid_pt[0] - half_range, mid_pt[0] + half_range)
    ax.set_ylim(mid_pt[1] - half_range, mid_pt[1] + half_range)
    ax.set_zlim(mid_pt[2] - half_range, mid_pt[2] + half_range)
    ax.set_box_aspect([1, 1, 1])    # 정육면체 비율 강제

    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)
    ax.set_title("ArUco Cube — 면별 마커 ID & 카메라 위치\n"
                 "(삼각형=카메라, 굵은 사각형=마커, 숫자=마커 ID)",
                 fontsize=11)

    # 범례
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=CAM_COLORS[i],
               markeredgecolor="k", markersize=10,
               label=f"cam{ci}  (frame {cam_poses[ci]['frame']}, "
                     f"err={cam_poses[ci]['err']:.1f}px)")
        for i, ci in enumerate(sorted(cam_poses))
    ]
    if legend_items:
        ax.legend(handles=legend_items, loc="upper left", fontsize=9)

    plt.tight_layout()

    if args.save:
        out_path = os.path.join(args.root_folder, "cube_3d_visualization.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n[SAVE] {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
