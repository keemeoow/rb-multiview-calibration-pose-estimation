# visualize_calibration.py
"""
Step3에서 저장된 T_C0_Ci 행렬을 로드하여 카메라 배치를 3D 시각화합니다.
기준 카메라(cam0)를 원점에 두고, 나머지 카메라의 위치와 방향을 표시합니다.

실행:
  python visualize_calibration.py \
    --root_folder ./data/cube_session_01 \
    --intrinsics_dir ./data/_intrinsics \
    --ref_cam_idx 0
"""

import os
import json
import argparse
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src3._aruco_cube import CubeConfig, ArucoCubeModel, ArucoCubeTarget


CAM_COLORS  = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
FACE_COLORS = {
    "+X": "#FF6666", "-X": "#FFAAAA",
    "+Y": "#66BB66", "-Y": "#AADDAA",
    "+Z": "#6699FF", "-Z": "#AACCFF",
}
FACE_NORMALS = {
    "+X": np.array([1,0,0]), "-X": np.array([-1,0,0]),
    "+Y": np.array([0,1,0]), "-Y": np.array([0,-1,0]),
    "+Z": np.array([0,0,1]), "-Z": np.array([0,0,-1]),
}


def load_intrinsics(intrinsics_dir, cam_idx):
    d = np.load(os.path.join(intrinsics_dir, f"cam{cam_idx}.npz"))
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def face_poly_3d(face_name, d):
    polys = {
        "+X": np.array([(d,-d,-d),(d, d,-d),(d, d, d),(d,-d, d)]),
        "-X": np.array([(-d,-d,-d),(-d,-d, d),(-d, d, d),(-d, d,-d)]),
        "+Y": np.array([(-d,d,-d),(d,d,-d),(d,d, d),(-d,d, d)]),
        "-Y": np.array([(-d,-d,-d),(-d,-d, d),(d,-d, d),(d,-d,-d)]),
        "+Z": np.array([(-d,-d,d),(d,-d,d),(d, d,d),(-d, d,d)]),
        "-Z": np.array([(-d,-d,-d),(-d,d,-d),(d,d,-d),(d,-d,-d)]),
    }
    return polys[face_name]


def draw_cube_in_cam0(ax, T_C0_O, cfg):
    """큐브를 cam0 좌표계에서 그린다."""
    d     = cfg.cube_side_m / 2.0
    model = ArucoCubeModel(cfg)
    face_to_id = {v: k for k, v in cfg.id_to_face.items()}
    R, t = T_C0_O[:3, :3], T_C0_O[:3, 3]

    def to_cam0(pts):
        return (R @ pts.T).T + t

    for face_name, color in FACE_COLORS.items():
        verts_obj = face_poly_3d(face_name, d)
        verts_c0  = to_cam0(verts_obj)
        mid       = face_to_id.get(face_name)

        ax.add_collection3d(Poly3DCollection(
            [verts_c0.tolist()], alpha=0.12,
            facecolor=color, edgecolor="gray", linewidth=0.6
        ))

        if mid is not None:
            mc   = to_cam0(model.marker_corners_in_rig(mid))
            loop = np.vstack([mc, mc[0]])
            ax.plot3D(loop[:,0], loop[:,1], loop[:,2], color=color, linewidth=2.0)

        center   = verts_c0.mean(axis=0)
        normal_c0 = R @ FACE_NORMALS[face_name]
        lpos     = center + normal_c0 * d * 0.55
        txt      = f"id {mid}\n({face_name})" if mid is not None else face_name
        ax.text(*lpos, txt, ha="center", va="center",
                fontsize=7, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))

    # 큐브 좌표축 (cam0 프레임에서)
    origin = t
    al = d * 0.8
    for vec_obj, col, lbl in [([al,0,0],"red","+X"),([0,al,0],"green","+Y"),([0,0,al],"blue","+Z")]:
        v = R @ np.array(vec_obj)
        ax.quiver(*origin, *v, color=col, arrow_length_ratio=0.25, linewidth=1.5, alpha=0.7)
        ax.text(*(origin + v * 1.2), lbl, color=col, fontsize=7, fontweight="bold")


def draw_camera(ax, pos, R_C0_Ci, color, label, scale=0.1):
    """카메라 위치 + 좌표축 화살표."""
    ax.scatter(*pos, c=color, s=180, marker="^", edgecolors="k", zorder=6)
    ax.text(pos[0], pos[1], pos[2] + scale * 0.5, label,
            color=color, fontsize=10, fontweight="bold", ha="center")
    # cam_i의 각 축 방향 = R_C0_Ci의 각 열 (cam0 프레임에서 표현)
    for i, c in enumerate(["#CC3333", "#33AA33", "#3333CC"]):
        ax.quiver(*pos, *(R_C0_Ci[:, i] * scale),
                  color=c, arrow_length_ratio=0.3, linewidth=1.5, alpha=0.85)


def get_avg_cube_pose(root_folder, intrinsics_dir, ref_cam_idx, cfg):
    """ref 카메라의 모든 유효 프레임 PnP로 평균 T_C0_O 계산."""
    cube = ArucoCubeTarget(cfg)
    try:
        K, D = load_intrinsics(intrinsics_dir, ref_cam_idx)
    except FileNotFoundError:
        return None

    cam_dir   = os.path.join(root_folder, f"cam{ref_cam_idx}")
    rgb_files = sorted(glob.glob(os.path.join(cam_dir, "rgb_*.jpg")))

    T_list, weights = [], []
    for rgb_path in rgb_files:
        img = cv2.imread(rgb_path)
        if img is None:
            continue
        ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
            img, K, D, use_ransac=False, min_markers=1,
            reproj_thr_mean_px=5.0, return_reproj=True,
        )
        if not ok or reproj is None:
            continue
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3,:3] = R; T[:3,3] = tvec.flatten()
        T_list.append(T)
        weights.append(1.0 / max(reproj["err_mean"], 1e-9))

    if not T_list:
        return None

    # 가중 SE(3) 평균
    w = np.array(weights); w /= w.sum()
    t_mean = sum(wi * T[:3,3] for wi, T in zip(w, T_list))
    R_sum  = sum(wi * T[:3,:3] for wi, T in zip(w, T_list))
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:,-1] *= -1; R_mean = U @ Vt

    T_avg = np.eye(4); T_avg[:3,:3] = R_mean; T_avg[:3,3] = t_mean
    return T_avg


def main():
    parser = argparse.ArgumentParser(description="캘리브레이션 결과 3D 시각화")
    parser.add_argument("--root_folder",    required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--ref_cam_idx",    type=int, default=0)
    parser.add_argument("--axis_scale",     type=float, default=0.0,
                        help="카메라 축 화살표 크기(m). 0=자동")
    parser.add_argument("--save",           action="store_true")
    args = parser.parse_args()

    # T_C0_Ci_all.json 찾기 (transforms 하위 또는 calib_out_cube 직접)
    json_path = os.path.join(
        args.root_folder, "calib_out_cube", "transforms",
        f"T_C{args.ref_cam_idx}_Ci_all.json"
    )
    if not os.path.exists(json_path):
        json_path = os.path.join(
            args.root_folder, "calib_out_cube",
            f"T_C{args.ref_cam_idx}_Ci_all.json"
        )
    if not os.path.exists(json_path):
        json_path = os.path.join(
            args.root_folder, "calib_out_cube", "T_C0_Ci_all.json"
        )

    with open(json_path) as f:
        data = json.load(f)

    print(f"[INFO] {json_path}")
    print(f"[INFO] 기준 카메라: cam{data['ref_cam_idx']}\n")

    cam_poses = {}
    for ci_str, flat in data["T_Cref_Ci"].items():
        ci = int(ci_str)
        T  = np.array(flat, dtype=np.float64).reshape(4, 4)
        cam_poses[ci] = T

    # 카메라 간 거리 출력
    ref_ci  = data["ref_cam_idx"]
    ref_pos = cam_poses[ref_ci][:3, 3]
    for ci, T in sorted(cam_poses.items()):
        pos  = T[:3, 3]
        dist = np.linalg.norm(pos - ref_pos)
        print(f"cam{ci}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] m  "
              f"dist_from_cam{ref_ci}={dist:.4f} m")

    # 큐브 평균 pose (cam0 기준)
    cfg      = CubeConfig()
    T_C0_O   = get_avg_cube_pose(args.root_folder, args.intrinsics_dir, ref_ci, cfg)
    if T_C0_O is not None:
        cube_pos = T_C0_O[:3, 3]
        print(f"\n[INFO] 큐브 위치 (cam{ref_ci} 기준): "
              f"[{cube_pos[0]:+.4f}, {cube_pos[1]:+.4f}, {cube_pos[2]:+.4f}] m")

    # ---- 3D Plot ----
    fig = plt.figure(figsize=(10, 9))
    ax  = fig.add_subplot(111, projection="3d")

    all_pts = []

    # 큐브 그리기
    if T_C0_O is not None:
        draw_cube_in_cam0(ax, T_C0_O, cfg)
        d = cfg.cube_side_m / 2.0
        for fn in FACE_NORMALS:
            all_pts.append(face_poly_3d(fn, d) @ T_C0_O[:3,:3].T + T_C0_O[:3,3])

    # 카메라 그리기
    positions = np.array([T[:3,3] for T in cam_poses.values()])
    span = np.ptp(positions, axis=0).max()
    scale = args.axis_scale if args.axis_scale > 0 else max(span * 0.12, 0.05)

    for i, (ci, T) in enumerate(sorted(cam_poses.items())):
        color = CAM_COLORS[i % len(CAM_COLORS)]
        pos   = T[:3, 3]
        R     = T[:3, :3]
        draw_camera(ax, pos, R, color, f"cam{ci}", scale=scale)
        all_pts.append(pos.reshape(1, 3))

        if ci != ref_ci:
            ax.plot3D([ref_pos[0], pos[0]], [ref_pos[1], pos[1]], [ref_pos[2], pos[2]],
                      "--", color=color, alpha=0.25, linewidth=1)

    # Equal aspect ratio
    all_pts = np.concatenate([p.reshape(-1, 3) for p in all_pts], axis=0)
    half    = np.ptp(all_pts, axis=0).max() / 2.0 * 1.2
    mid     = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0]-half, mid[0]+half)
    ax.set_ylim(mid[1]-half, mid[1]+half)
    ax.set_zlim(mid[2]-half, mid[2]+half)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)
    ax.set_title(
        f"캘리브레이션 결과 — 카메라 배치 (cam{ref_ci} 기준)\n"
        "삼각형=카메라, 화살표=카메라 좌표축(R/G/B = X/Y/Z)",
        fontsize=11
    )

    plt.tight_layout()

    if args.save:
        out = os.path.join(args.root_folder, "calib_visualization.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n[SAVE] {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
