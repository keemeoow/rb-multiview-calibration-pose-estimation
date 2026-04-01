# visualize_3d_all_frames.py
"""
전체 프레임에서 PnP를 수행하고, 결과를 3D로 시각화합니다.

큐브를 원점(고정)에 놓고,
각 카메라가 각 프레임에서 추정한 위치를 모두 scatter plot으로 그립니다.

- 점(작은 원): 개별 프레임의 카메라 위치 추정값
- 삼각형(큰): 해당 카메라의 최적 프레임 (가중 평균 위치)
- 점들이 한 곳에 모일수록 → 캘리브레이션 데이터가 일관됨

실행:
  python visualize_3d_all_frames.py \
    --root_folder ./data/cube_session_01 \
    --intrinsics_dir ./data/_intrinsics
"""

import os
import glob
import argparse
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from _aruco_cube import CubeConfig, ArucoCubeModel, ArucoCubeTarget


CAM_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

FACE_COLORS = {
    "+X": "#FF6666", "-X": "#FFAAAA",
    "+Y": "#66BB66", "-Y": "#AADDAA",
    "+Z": "#6699FF", "-Z": "#AACCFF",
}
FACE_NORMALS = {
    "+X": np.array([1, 0, 0]), "-X": np.array([-1, 0, 0]),
    "+Y": np.array([0, 1, 0]), "-Y": np.array([0, -1, 0]),
    "+Z": np.array([0, 0, 1]), "-Z": np.array([0, 0, -1]),
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


def collect_all_pnp(root_folder: str, intrinsics_dir: str,
                    cam_idx: int, cube: ArucoCubeTarget,
                    reproj_max_px: float = 5.0):
    """
    한 카메라의 모든 프레임에서 PnP 결과 수집.
    Returns: list of dict(cam_pos, R, t, err, used, frame)
    """
    try:
        K, D = load_intrinsics(intrinsics_dir, cam_idx)
    except FileNotFoundError:
        return []

    cam_dir   = os.path.join(root_folder, f"cam{cam_idx}")
    rgb_files = sorted(glob.glob(os.path.join(cam_dir, "rgb_*.jpg")))

    results = []
    for rgb_path in rgb_files:
        img = cv2.imread(rgb_path)
        if img is None:
            continue

        ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
            img, K, D,
            use_ransac=False,
            min_markers=1,
            reproj_thr_mean_px=reproj_max_px,
            return_reproj=True,
        )
        if not ok or reproj is None:
            continue

        R, _ = cv2.Rodrigues(rvec)
        cam_pos = -R.T @ tvec.flatten()   # 큐브 좌표계에서의 카메라 위치
        frame_idx = int(
            os.path.basename(rgb_path).split("_")[-1].split(".")[0]
        )
        results.append(dict(
            cam_pos=cam_pos,
            R=R, t=tvec.flatten(),
            err=reproj["err_mean"],
            used=used,
            frame=frame_idx,
        ))

    return results


# ------------------------------------------------------------------ #
def face_poly_3d(face_name: str, d: float) -> np.ndarray:
    polys = {
        "+X": np.array([(d,-d,-d),(d, d,-d),(d, d, d),(d,-d, d)]),
        "-X": np.array([(-d,-d,-d),(-d,-d, d),(-d, d, d),(-d, d,-d)]),
        "+Y": np.array([(-d,d,-d),(d,d,-d),(d,d, d),(-d,d, d)]),
        "-Y": np.array([(-d,-d,-d),(-d,-d, d),(d,-d, d),(d,-d,-d)]),
        "+Z": np.array([(-d,-d,d),(d,-d,d),(d, d,d),(-d, d,d)]),
        "-Z": np.array([(-d,-d,-d),(-d,d,-d),(d,d,-d),(d,-d,-d)]),
    }
    return polys[face_name]


def draw_cube(ax, cfg: CubeConfig):
    d     = cfg.cube_side_m / 2.0
    model = ArucoCubeModel(cfg)
    face_to_id = {v: k for k, v in cfg.id_to_face.items()}

    for face_name, color in FACE_COLORS.items():
        verts = face_poly_3d(face_name, d)
        mid   = face_to_id.get(face_name)

        ax.add_collection3d(Poly3DCollection(
            [verts.tolist()], alpha=0.12,
            facecolor=color, edgecolor="gray", linewidth=0.6
        ))

        if mid is not None:
            mc   = model.marker_corners_in_rig(mid)
            loop = np.vstack([mc, mc[0]])
            ax.plot3D(loop[:,0], loop[:,1], loop[:,2],
                      color=color, linewidth=2.0)

        center   = verts.mean(axis=0)
        normal   = FACE_NORMALS[face_name]
        lpos     = center + normal * d * 0.55
        txt      = f"id {mid}\n({face_name})" if mid is not None else face_name
        ax.text(*lpos, txt, ha="center", va="center",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))

    # 큐브 좌표축
    al = d * 0.9
    for vec, col, lbl in [([al,0,0],"red","+X"),([0,al,0],"green","+Y"),([0,0,al],"blue","+Z")]:
        ax.quiver(0,0,0,*vec, color=col, arrow_length_ratio=0.25, linewidth=1.5, alpha=0.85)
        ax.text(vec[0]*1.2, vec[1]*1.2, vec[2]*1.2, lbl,
                color=col, fontsize=8, fontweight="bold")


def draw_camera_axes(ax, R_CO, t_CO, color, scale=0.04, alpha=0.5):
    """카메라 위치에 좌표축 화살표만 그린다 (레이블 없음)."""
    cam_pos = -R_CO.T @ t_CO.flatten()
    R_OC    = R_CO.T
    for i, c in enumerate(["#CC3333","#33AA33","#3333CC"]):
        d = R_OC[:, i] * scale
        ax.quiver(*cam_pos, *d, color=c, arrow_length_ratio=0.4,
                  linewidth=1.0, alpha=alpha)


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="전체 프레임 3D 시각화 (큐브 + 카메라 분포)"
    )
    parser.add_argument("--root_folder",    required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--reproj_max_px",  type=float, default=5.0,
                        help="PnP 허용 reproj 오차 (Step3와 동일값 권장)")
    parser.add_argument("--save",           action="store_true",
                        help="PNG 저장 (화면 표시 없음)")
    args = parser.parse_args()

    cfg  = CubeConfig()
    cube = ArucoCubeTarget(cfg)
    d    = cfg.cube_side_m / 2.0

    print(f"\n[INFO] reproj_max_px = {args.reproj_max_px}px")
    print(f"[INFO] id_to_face:")
    for mid, face in sorted(cfg.id_to_face.items()):
        print(f"  마커 {mid} → {face}")

    cam_idxs = discover_cams(args.root_folder)
    print(f"[INFO] 카메라: {cam_idxs}\n")

    # ---- 전체 프레임 PnP 수집 ----
    all_data = {}      # cam_idx → list of result dicts
    for ci in cam_idxs:
        results = collect_all_pnp(
            args.root_folder, args.intrinsics_dir,
            ci, cube, reproj_max_px=args.reproj_max_px,
        )
        all_data[ci] = results
        if results:
            errs = [r["err"] for r in results]
            print(f"[INFO] cam{ci}: {len(results)}프레임 성공  "
                  f"reproj: min={min(errs):.3f}  "
                  f"mean={np.mean(errs):.3f}  "
                  f"max={max(errs):.3f} px")
        else:
            print(f"[WARN] cam{ci}: 유효 프레임 없음")

    # ---- 3D Plot ----
    fig = plt.figure(figsize=(11, 9))
    ax  = fig.add_subplot(111, projection="3d")

    draw_cube(ax, cfg)

    all_cam_pos = []
    for i, ci in enumerate(cam_idxs):
        results = all_data.get(ci, [])
        if not results:
            continue

        color = CAM_COLORS[i % len(CAM_COLORS)]
        positions = np.array([r["cam_pos"] for r in results])   # (N, 3)
        errs      = np.array([r["err"]     for r in results])

        # 모든 프레임 위치 → 작은 점
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   c=color, s=20, alpha=0.45, zorder=3,
                   label=f"cam{ci} ({len(results)}frames, "
                         f"mean={errs.mean():.2f}px)")

        # reproj 오차로 색조(명도) 구분: 오차 낮을수록 진하게
        # (선택적: 오차가 낮은 상위 30% 프레임은 조금 크게)
        low_err_mask = errs <= np.percentile(errs, 30)
        if low_err_mask.any():
            ax.scatter(positions[low_err_mask, 0],
                       positions[low_err_mask, 1],
                       positions[low_err_mask, 2],
                       c=color, s=60, alpha=0.85,
                       edgecolors="k", linewidths=0.5, zorder=4)

        # 가중 평균 위치 (best estimate) → 큰 삼각형 + 축
        weights  = 1.0 / np.maximum(errs, 1e-9)
        w_norm   = weights / weights.sum()
        mean_pos = (w_norm[:, None] * positions).sum(axis=0)

        # 가중 평균 R (대표 자세)
        best_idx = int(np.argmin(errs))
        best     = results[best_idx]

        ax.scatter(*mean_pos, c=color, s=180, marker="^",
                   edgecolors="k", linewidths=1.0, zorder=6)
        draw_camera_axes(ax, best["R"], best["t"], color,
                         scale=cfg.cube_side_m * 0.6, alpha=0.9)
        ax.text(mean_pos[0], mean_pos[1], mean_pos[2] + cfg.cube_side_m * 0.15,
                f"cam{ci}", color=color, fontsize=10, fontweight="bold", ha="center")

        # 평균 위치 → 큐브 원점 연결선
        ax.plot3D([mean_pos[0], 0], [mean_pos[1], 0], [mean_pos[2], 0],
                  "--", color=color, alpha=0.2, linewidth=1)

        all_cam_pos.append(positions)

    # ---- Equal aspect ratio ----
    cube_verts = np.concatenate([face_poly_3d(f, d) for f in FACE_NORMALS])
    if all_cam_pos:
        pts = np.concatenate([cube_verts] + all_cam_pos, axis=0)
    else:
        pts = cube_verts

    half  = np.ptp(pts, axis=0).max() / 2.0 * 1.15
    mid_p = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    ax.set_xlim(mid_p[0]-half, mid_p[0]+half)
    ax.set_ylim(mid_p[1]-half, mid_p[1]+half)
    ax.set_zlim(mid_p[2]-half, mid_p[2]+half)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)
    ax.set_title(
        f"전체 프레임 카메라 위치 분포 (reproj ≤ {args.reproj_max_px}px)\n"
        "작은 점=각 프레임, 진한 점=하위 30% 오차, 삼각형=가중평균 위치",
        fontsize=10
    )
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    if args.save:
        out = os.path.join(args.root_folder, "cube_3d_all_frames.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n[SAVE] {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
