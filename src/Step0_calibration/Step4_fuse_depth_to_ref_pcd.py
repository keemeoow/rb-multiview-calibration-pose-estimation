# Step4_fuse_depth_to_ref_pcd.py
# 전체 프레임 기반 검증/통합 + 단일 프레임 시각화
"""
[결과만 잘나오게]
python Step4_fuse_depth_to_ref_pcd.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0 \
  --frame_idx 0 \
  --auto_best_frame \
  --use_depth \
  --save_overlay \
  --depth_pose_mode frame \
  --depth_cube_roi_margin_m 0.05 \
  --depth_z_min 0.2 \
  --depth_z_max 1.5 \
  --depth_auto_z_window_m 0.10 \
  --depth_stride 2
"""
"""
[roi 없이]
python Step4_fuse_depth_to_ref_pcd.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0 \
  --frame_idx 0 \
  --auto_best_frame \
  --use_depth \
  --save_overlay \
  --depth_pose_mode frame \
  --depth_dense_no_roi \
  --depth_vis_max_points 120000
"""
"""
python Step4_fuse_depth_to_ref_pcd.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0 \
  --frame_idx 0 \
  --use_depth \
  --save_overlay \
  --depth_cube_roi_margin_m 0.06
"""
"""
python Step4_fuse_depth_to_ref_pcd.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0 \
  --frame_idx 0 \
  --save_overlay

설명:
- 계산: 전체 프레임 사용 (PnP / cross-camera reprojection / 전역 큐브 pose 추정)
- 시각화: --frame_idx 한 프레임만 사용 (기본 0)
"""
"""
[전체 프레임 depth fusion (멀티프레임)]
python Step4_fuse_depth_to_ref_pcd.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0 \
  --frame_idx 0 \
  --auto_best_frame \
  --use_depth \
  --depth_all_frames \
  --save_overlay \
  --depth_pose_mode frame \
  --depth_stride 2 \
  --depth_voxel_size_m 0.002 \
  --depth_z_min 0.2 \
  --depth_z_max 1.5
"""

import os
import glob
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# open3d 는 선택적 의존성; 없어도 PLY 저장 + matplotlib로 대체

from _aruco_cube import CubeConfig, ArucoCubeModel, ArucoCubeTarget, rodrigues_to_Rt

try:
    from _utils_pose import robust_se3_average
except Exception:
    robust_se3_average = None


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def parse_frame_id_from_rgb_path(rgb_path: str) -> int:
    return int(os.path.basename(rgb_path).split("_")[-1].split(".")[0])


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra @ Rb.T
    c = (np.trace(dR) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def se3_avg_weighted(T_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    n = len(T_list)
    if n == 0:
        raise ValueError("T_list is empty")
    if weights is None:
        weights = [1.0] * n

    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    ts = np.asarray([T[:3, 3] for T in T_list], dtype=np.float64)
    t_mean = (w[:, None] * ts).sum(axis=0)

    Rs = np.asarray([T[:3, :3] for T in T_list], dtype=np.float64)
    R_mean = (w[:, None, None] * Rs).sum(axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t_mean
    return out


def load_device_map(intrinsics_dir: str):
    map_path = os.path.join(intrinsics_dir, "device_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"device_map.json not found: {map_path}")
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    if not serial_to_idx:
        raise RuntimeError("device_map.json has empty serial_to_idx.")
    return serial_to_idx, map_path


def load_intrinsics_cam_npz(intrinsics_dir: str, cam_idx: int):
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing intrinsics npz: {p}")
    data = np.load(p, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64)
    depth_scale = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else None
    return K, D, depth_scale


def discover_cam_indices_from_data(root_folder: str) -> List[int]:
    cam_idxs = []
    for name in os.listdir(root_folder):
        if not name.startswith("cam"):
            continue
        try:
            idx = int(name.replace("cam", ""))
        except Exception:
            continue
        if len(glob.glob(os.path.join(root_folder, name, "rgb_*.jpg"))) > 0:
            cam_idxs.append(idx)
    cam_idxs = sorted(cam_idxs)
    if len(cam_idxs) == 0:
        raise RuntimeError(f"No cam folders with rgb images found in {root_folder}")
    return cam_idxs


def discover_frame_indices(root_folder: str, cam_indices: List[int]) -> List[int]:
    frame_set = set()
    for ci in cam_indices:
        pattern = os.path.join(root_folder, f"cam{ci}", "rgb_*.jpg")
        for p in glob.glob(pattern):
            try:
                frame_set.add(parse_frame_id_from_rgb_path(p))
            except Exception:
                continue
    frame_ids = sorted(frame_set)
    if len(frame_ids) == 0:
        raise RuntimeError(f"No rgb_*.jpg found under {root_folder}/cam*/")
    return frame_ids


def collect_pnp_all_frames(
    root_folder: str,
    cam_indices: List[int],
    frame_ids: List[int],
    cube: ArucoCubeTarget,
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    reproj_max_px: float,
):
    pnp_by_frame = {}
    cam_stats = {
        ci: {"attempts": 0, "ok": 0, "errs": []}
        for ci in cam_indices
    }

    print(
        f"\n[STEP4-1] 전체 프레임 PnP 수집 "
        f"(frames={len(frame_ids)}, reproj_max={reproj_max_px:.1f}px)"
    )

    for fid in frame_ids:
        frame_results = {}
        for ci in cam_indices:
            rgb_path = os.path.join(root_folder, f"cam{ci}", f"rgb_{fid:05d}.jpg")
            if not os.path.exists(rgb_path):
                continue

            img = cv2.imread(rgb_path)
            if img is None:
                continue

            cam_stats[ci]["attempts"] += 1
            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img,
                K_map[ci],
                D_map[ci],
                min_markers=1,
                reproj_thr_mean_px=reproj_max_px,
                single_marker_only=True,
                return_reproj=True,
            )
            if not ok or reproj is None:
                continue

            cam_stats[ci]["ok"] += 1
            cam_stats[ci]["errs"].append(float(reproj["err_mean"]))
            frame_results[ci] = {
                "rvec": rvec,
                "tvec": tvec,
                "reproj": reproj,
                "used": [int(x) for x in used],
            }

        if frame_results:
            pnp_by_frame[fid] = frame_results

    for ci in cam_indices:
        st = cam_stats[ci]
        mean_err = float(np.mean(st["errs"])) if st["errs"] else float("nan")
        print(
            f"[INFO] cam{ci}: PnP {st['ok']}/{st['attempts']} 성공  "
            f"reproj_mean={mean_err:.3f}px"
        )

    return pnp_by_frame


def build_ref_pose_observations(
    pnp_by_frame: Dict[int, Dict[int, dict]],
    T_ref: Dict[int, np.ndarray],
):
    obs_by_frame = {}
    all_obs = []

    for fid, per_cam in sorted(pnp_by_frame.items()):
        frame_obs = []
        for ci, res in sorted(per_cam.items()):
            if ci not in T_ref:
                continue
            T_Ci_O = rodrigues_to_Rt(res["rvec"], res["tvec"])
            T_Cref_O = T_ref[ci] @ T_Ci_O
            err = float(res["reproj"]["err_mean"])
            ob = {
                "frame_id": int(fid),
                "cam_idx": int(ci),
                "T_Cref_O": T_Cref_O,
                "err_mean": err,
                "weight": 1.0 / max(err, 1e-9),
                "used": res["used"],
            }
            frame_obs.append(ob)
            all_obs.append(ob)
        if frame_obs:
            obs_by_frame[fid] = frame_obs
    return obs_by_frame, all_obs


def estimate_global_cube_pose(all_obs: List[dict]) -> np.ndarray:
    if len(all_obs) == 0:
        raise RuntimeError("No valid cube pose observations.")
    Ts = [o["T_Cref_O"] for o in all_obs]
    ws = [o["weight"] for o in all_obs]
    if robust_se3_average is not None and len(Ts) >= 3:
        return robust_se3_average(Ts)
    return se3_avg_weighted(Ts, ws)


def summarize_cube_consistency(obs_by_frame: Dict[int, List[dict]], T_global: np.ndarray):
    frame_trans_mm, frame_rot_deg = [], []
    global_trans_mm, global_rot_deg = [], []

    for _, obs in sorted(obs_by_frame.items()):
        Ts = [o["T_Cref_O"] for o in obs]
        ws = [o["weight"] for o in obs]
        if len(Ts) >= 2:
            if robust_se3_average is not None and len(Ts) >= 3:
                T_frame = robust_se3_average(Ts)
            else:
                T_frame = se3_avg_weighted(Ts, ws)

            for T in Ts:
                frame_trans_mm.append(np.linalg.norm(T[:3, 3] - T_frame[:3, 3]) * 1000.0)
                frame_rot_deg.append(rotation_angle_deg(T[:3, :3], T_frame[:3, :3]))

        for T in Ts:
            global_trans_mm.append(np.linalg.norm(T[:3, 3] - T_global[:3, 3]) * 1000.0)
            global_rot_deg.append(rotation_angle_deg(T[:3, :3], T_global[:3, :3]))

    if frame_trans_mm:
        print(
            "\n[EVAL-ALL 3D] 프레임 내 카메라 간 큐브 합치기 편차: "
            f"trans mean={np.mean(frame_trans_mm):.2f}mm max={np.max(frame_trans_mm):.2f}mm, "
            f"rot mean={np.mean(frame_rot_deg):.2f}deg max={np.max(frame_rot_deg):.2f}deg"
        )
    else:
        print("\n[EVAL-ALL 3D] 프레임 내 다중 카메라 비교 가능한 샘플이 부족합니다.")

    print(
        "[EVAL-ALL 3D] 전역 큐브(전체 프레임 통합) 대비 편차: "
        f"trans mean={np.mean(global_trans_mm):.2f}mm max={np.max(global_trans_mm):.2f}mm, "
        f"rot mean={np.mean(global_rot_deg):.2f}deg max={np.max(global_rot_deg):.2f}deg"
    )


def estimate_frame_cube_pose(obs_by_frame: Dict[int, List[dict]], frame_id: int) -> Optional[np.ndarray]:
    obs = obs_by_frame.get(int(frame_id), [])
    if len(obs) == 0:
        return None
    Ts = [o["T_Cref_O"] for o in obs]
    ws = [o["weight"] for o in obs]
    if robust_se3_average is not None and len(Ts) >= 3:
        return robust_se3_average(Ts)
    return se3_avg_weighted(Ts, ws)


def select_best_frame_for_depth(
    pnp_by_frame: Dict[int, Dict[int, dict]],
    ref_cam_idx: int,
) -> Optional[int]:
    """
    단일 depth fusion용으로 '잘 맞는' 프레임을 고른다.
    우선순위:
    - ref cam 포함
    - PnP 성공 카메라 수 많음
    - 사용 마커 수(합) 많음
    - 평균 reproj 오차 작음
    - ref cam reproj 오차 작음
    """
    best_fid = None
    best_key = None

    for fid, frame_res in sorted(pnp_by_frame.items()):
        if len(frame_res) == 0:
            continue
        has_ref = 1 if ref_cam_idx in frame_res else 0
        n_cams = len(frame_res)
        total_markers = int(sum(len(r.get("used", [])) for r in frame_res.values()))
        errs = [float(r["reproj"]["err_mean"]) for r in frame_res.values() if "reproj" in r]
        mean_err = float(np.mean(errs)) if errs else float("inf")
        ref_err = float(frame_res[ref_cam_idx]["reproj"]["err_mean"]) if has_ref else float("inf")

        # max metrics are positive, min-error metrics are negated
        key = (
            has_ref,
            n_cams,
            total_markers,
            -mean_err,
            -ref_err,
            -int(fid),  # tie-break: earlier frame preferred
        )
        if best_key is None or key > best_key:
            best_key = key
            best_fid = int(fid)

    return best_fid


def evaluate_cross_reproj_all_frames(
    cam_indices: List[int],
    pnp_by_frame: Dict[int, Dict[int, dict]],
    T_ref: Dict[int, np.ndarray],
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    ref_cam_idx: int,
):
    if ref_cam_idx not in K_map:
        print("[WARN] ref cam intrinsics 없음, 전체 cross reprojection 평가 skip")
        return

    K_ref = K_map[ref_cam_idx]
    D_ref = D_map[ref_cam_idx]

    stats = {
        ci: {"frames": 0, "points": 0, "all_dists": []}
        for ci in cam_indices
        if ci != ref_cam_idx
    }

    for _, frame_res in sorted(pnp_by_frame.items()):
        if ref_cam_idx not in frame_res:
            continue
        ref_res = frame_res[ref_cam_idx]
        obj_pts_ref = ref_res["reproj"]["obj_pts"].reshape(-1, 3)
        ref_img_pts = ref_res["reproj"]["img_pts"].reshape(-1, 2)

        for ci in sorted(stats.keys()):
            if ci not in frame_res or ci not in T_ref:
                continue

            res_i = frame_res[ci]
            T_Ci_O = rodrigues_to_Rt(res_i["rvec"], res_i["tvec"])
            T_Cref_O = T_ref[ci] @ T_Ci_O
            R_pred = T_Cref_O[:3, :3]
            t_pred = T_Cref_O[:3, 3]
            rvec_pred, _ = cv2.Rodrigues(R_pred)

            proj, _ = cv2.projectPoints(obj_pts_ref, rvec_pred, t_pred, K_ref, D_ref)
            proj = proj.reshape(-1, 2)
            dists = np.linalg.norm(proj - ref_img_pts, axis=1)
            if dists.size == 0:
                continue

            st = stats[ci]
            st["frames"] += 1
            st["points"] += int(dists.size)
            st["all_dists"].append(dists)

    print(f"\n[EVAL-ALL] Cross-camera reprojection → ref cam{ref_cam_idx}:")
    for ci in sorted(stats.keys()):
        st = stats[ci]
        if st["frames"] == 0:
            print(f"[EVAL-ALL] cam{ci}: 비교 가능한 프레임 없음")
            continue
        all_d = np.concatenate(st["all_dists"], axis=0)
        print(
            f"[EVAL-ALL] cam{ci} → ref cam{ref_cam_idx}: "
            f"frames={st['frames']}  n={st['points']}  "
            f"mean={all_d.mean():.2f}px  p90={np.percentile(all_d, 90):.2f}px  "
            f"max={all_d.max():.2f}px"
        )


CAM_COLORS_CV = [(0, 255, 0), (0, 0, 255), (255, 128, 0), (0, 255, 255), (255, 0, 255)]
CAM_COLORS_MPL = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

FACE_COLORS = {
    "+X": "#FF6666", "-X": "#FFAAAA",
    "+Y": "#66BB66", "-Y": "#AADDAA",
    "+Z": "#6699FF", "-Z": "#AACCFF",
}


def face_poly_3d(face_name: str, d: float) -> np.ndarray:
    polys = {
        "+X": np.array([(d, -d, -d), (d, d, -d), (d, d, d), (d, -d, d)]),
        "-X": np.array([(-d, -d, -d), (-d, -d, d), (-d, d, d), (-d, d, -d)]),
        "+Y": np.array([(-d, d, -d), (d, d, -d), (d, d, d), (-d, d, d)]),
        "-Y": np.array([(-d, -d, -d), (-d, -d, d), (d, -d, d), (d, -d, -d)]),
        "+Z": np.array([(-d, -d, d), (d, -d, d), (d, d, d), (-d, d, d)]),
        "-Z": np.array([(-d, -d, -d), (-d, d, -d), (d, d, -d), (d, -d, -d)]),
    }
    return polys[face_name]


def draw_cube_at_T(ax, T_Cref_O: np.ndarray, cfg: CubeConfig, alpha: float = 0.18):
    d = cfg.cube_side_m / 2.0
    model = ArucoCubeModel(cfg)
    R, t = T_Cref_O[:3, :3], T_Cref_O[:3, 3]
    face_to_id = {v: k for k, v in cfg.id_to_face.items()}

    def to_ref(pts):
        return (R @ pts.T).T + t

    for face_name, color in FACE_COLORS.items():
        verts = to_ref(face_poly_3d(face_name, d))
        mid = face_to_id.get(face_name)
        ax.add_collection3d(Poly3DCollection(
            [verts.tolist()],
            alpha=alpha,
            facecolor=color,
            edgecolor="gray",
            linewidth=0.7
        ))
        if mid is not None:
            mc = to_ref(model.marker_corners_in_rig(mid))
            loop = np.vstack([mc, mc[0]])
            ax.plot3D(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linewidth=1.5)

    al = d * 0.9
    for vec_obj, col in [([al, 0, 0], "red"), ([0, al, 0], "green"), ([0, 0, al], "blue")]:
        v = R @ np.array(vec_obj)
        ax.quiver(*t, *v, color=col, arrow_length_ratio=0.3, linewidth=1.2, alpha=0.8)


def visualize_global_scene(
    T_cube_global: np.ndarray,
    T_ref: Dict[int, np.ndarray],
    cfg: CubeConfig,
    ref_cam_idx: int,
    viz_frame_idx: int,
    pnp_viz: Dict[int, dict],
    save_path: Optional[str] = None,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    draw_cube_at_T(ax, T_cube_global, cfg, alpha=0.18)
    cube_center = T_cube_global[:3, 3]

    all_pts = [cube_center.reshape(1, 3)]
    for i, ci in enumerate(sorted(T_ref.keys())):
        color = CAM_COLORS_MPL[i % len(CAM_COLORS_MPL)]
        if ci == ref_cam_idx:
            cam_pos = np.zeros(3, dtype=np.float64)
        else:
            cam_pos = T_ref[ci][:3, 3]
        all_pts.append(cam_pos.reshape(1, 3))

        ax.scatter(*cam_pos, c=color, s=170, marker="^", edgecolors="k", zorder=8)
        ax.plot3D(
            [cam_pos[0], cube_center[0]],
            [cam_pos[1], cube_center[1]],
            [cam_pos[2], cube_center[2]],
            "--",
            color=color,
            alpha=0.45,
            linewidth=1.2,
        )

        label = f"cam{ci}"
        if ci in pnp_viz:
            label += f"\n(frame{viz_frame_idx}: {pnp_viz[ci]['reproj']['err_mean']:.2f}px)"
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2] + cfg.cube_side_m * 0.3, label,
                color=color, fontsize=9, ha="center")

    ax.scatter(*cube_center, c="k", s=80, marker="o", zorder=9)
    ax.text(cube_center[0], cube_center[1], cube_center[2] + cfg.cube_side_m * 0.25,
            "cube (global)", color="k", fontsize=9, ha="center")

    all_pts = np.concatenate(all_pts, axis=0)
    half = max(np.ptp(all_pts, axis=0).max() * 0.65, cfg.cube_side_m * 1.5)
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Global Calibration Result (all frames), ref=cam{ref_cam_idx}\n"
        f"Single cube + camera rig | visualization frame={viz_frame_idx}",
        fontsize=10,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    else:
        plt.show()


def load_images_for_frame(root_folder: str, cam_indices: List[int], frame_idx: int) -> Dict[int, np.ndarray]:
    images = {}
    for ci in cam_indices:
        rgb_path = os.path.join(root_folder, f"cam{ci}", f"rgb_{frame_idx:05d}.jpg")
        if not os.path.exists(rgb_path):
            continue
        img = cv2.imread(rgb_path)
        if img is not None:
            images[ci] = img
    return images


def validate_cross_reproj_single_frame(
    cam_indices: List[int],
    T_ref: Dict[int, np.ndarray],
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray],
    images: Dict[int, np.ndarray],
    pnp_results: Dict[int, dict],
    ref_cam_idx: int,
):
    if ref_cam_idx not in images:
        print("[WARN] ref cam image not found, single-frame cross-reprojection skip")
        return None
    if ref_cam_idx not in pnp_results:
        print("[WARN] ref cam PnP not found, single-frame cross-reprojection skip")
        return None

    ref_vis = images[ref_cam_idx].copy()
    K_ref = K_map[ref_cam_idx]
    D_ref = D_map[ref_cam_idx]

    ref_img_pts = pnp_results[ref_cam_idx]["reproj"]["img_pts"].reshape(-1, 2)
    for pt in ref_img_pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(ref_vis, (x, y), 6, (255, 255, 255), 2)

    print(f"\n[EVAL-FRAME {pnp_results[ref_cam_idx].get('frame_id', 'N/A')}] Cross-camera reprojection:")
    for i, ci in enumerate(sorted(cam_indices)):
        if ci == ref_cam_idx:
            continue
        if ci not in pnp_results:
            print(f"[EVAL-FRAME] cam{ci}: PnP 없음, skip")
            continue
        if ci not in T_ref:
            print(f"[EVAL-FRAME] cam{ci}: transform 없음, skip")
            continue

        rvec_i = pnp_results[ci]["rvec"]
        tvec_i = pnp_results[ci]["tvec"]
        T_Ci_O = rodrigues_to_Rt(rvec_i, tvec_i)
        T_Cref_O = T_ref[ci] @ T_Ci_O
        R_pred = T_Cref_O[:3, :3]
        t_pred = T_Cref_O[:3, 3]
        rvec_pred, _ = cv2.Rodrigues(R_pred)

        obj_pts_ref = pnp_results[ref_cam_idx]["reproj"]["obj_pts"].reshape(-1, 3)
        proj, _ = cv2.projectPoints(obj_pts_ref, rvec_pred, t_pred, K_ref, D_ref)
        proj = proj.reshape(-1, 2)
        dists = np.linalg.norm(proj - ref_img_pts, axis=1)
        print(
            f"[EVAL-FRAME] cam{ci} → ref cam{ref_cam_idx}: "
            f"mean={dists.mean():.2f}px  max={dists.max():.2f}px  n={len(dists)}"
        )

        color = CAM_COLORS_CV[i % len(CAM_COLORS_CV)]
        for pt in proj:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < ref_vis.shape[1] and 0 <= y < ref_vis.shape[0]:
                cv2.circle(ref_vis, (x, y), 5, color, -1)
        if len(proj) > 0:
            lx, ly = int(proj[0, 0]), int(proj[0, 1])
            cv2.putText(ref_vis, f"cam{ci}", (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(
        ref_vis,
        f"ref=cam{ref_cam_idx} | white=detected | colored=cross-projected",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return ref_vis


# ------------------------------------------------------------------ #
# depth 관련 함수 (--use_depth 플래그 사용 시에만 호출)
# ------------------------------------------------------------------ #
def _save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Colored point cloud을 ASCII PLY 파일로 저장 (open3d 불필요)."""
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        lines = [
            f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
            f"{rgb_u8[i, 0]} {rgb_u8[i, 1]} {rgb_u8[i, 2]}"
            for i in range(n)
        ]
        f.write("\n".join(lines))
        if n > 0:
            f.write("\n")


def depth_to_points_cam(depth_u16, K, depth_scale, z_min, z_max, stride=4):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth_u16.shape[:2]
    pts, pix = [], []
    for v in range(0, h, stride):
        for u in range(0, w, stride):
            d = int(depth_u16[v, u])
            if d == 0:
                continue
            z = float(d) * float(depth_scale)
            if z < z_min or z > z_max:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts.append([x, y, z])
            pix.append((v, u))
    if len(pts) == 0:
        return np.empty((0, 3), np.float64), []
    return np.asarray(pts, np.float64), pix


def transform_points_rowvec(points, T_ref_cam):
    R = T_ref_cam[:3, :3]
    t = T_ref_cam[:3, 3]
    return points @ R.T + t.reshape(1, 3)


def cube_roi_mask_in_ref(points_ref: np.ndarray, T_ref_cube: np.ndarray, half_extents_xyz: np.ndarray) -> np.ndarray:
    """
    ref 좌표계 점들을 큐브 좌표계(local)로 변환한 뒤 oriented box ROI 마스크를 만든다.
    T_ref_cube: cube/object -> ref transform
    """
    if points_ref.size == 0:
        return np.zeros((0,), dtype=bool)
    R = T_ref_cube[:3, :3]
    t = T_ref_cube[:3, 3]
    pts_local = (points_ref - t.reshape(1, 3)) @ R
    he = np.asarray(half_extents_xyz, dtype=np.float64).reshape(1, 3)
    return np.all(np.abs(pts_local) <= he, axis=1)


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy-only voxel grid downsampling. 같은 voxel 내 점들의 위치/색상 평균."""
    if points.shape[0] == 0 or voxel_size <= 0:
        return points, colors
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    mins = voxel_indices.min(axis=0)
    shifted = voxel_indices - mins
    dims = shifted.max(axis=0) + 1
    keys = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_voxels = len(unique_keys)
    sum_pts = np.zeros((n_voxels, 3), dtype=np.float64)
    sum_cols = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sum_pts, inverse, points)
    np.add.at(sum_cols, inverse, colors)
    return sum_pts / counts[:, None], sum_cols / counts[:, None]


def _fuse_depth_single_frame(
    root_folder: str,
    frame_idx: int,
    cam_indices: List[int],
    T_ref: Dict[int, np.ndarray],
    K_map: Dict[int, np.ndarray],
    intrinsics_dir: str,
    z_min: float,
    z_max: float,
    stride: int,
    depth_cube_roi: bool,
    roi_half: Optional[np.ndarray],
    T_cube_depth: np.ndarray,
    verbose: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """단일 프레임의 모든 카메라 depth를 ref 좌표계로 변환하여 반환."""
    frame_pts: List[np.ndarray] = []
    frame_cols: List[np.ndarray] = []
    for ci in cam_indices:
        if ci not in T_ref:
            continue
        rgb_path = os.path.join(root_folder, f"cam{ci}", f"rgb_{frame_idx:05d}.jpg")
        depth_path = os.path.join(root_folder, f"cam{ci}", f"depth_{frame_idx:05d}.png")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            if verbose:
                print(f"[WARN] depth 없음: cam{ci} frame {frame_idx} (skip)")
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_u16 is None:
            if verbose:
                print(f"[WARN] 이미지 로드 실패: cam{ci} frame {frame_idx} (skip)")
            continue

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        _, _, ds = load_intrinsics_cam_npz(intrinsics_dir, ci)
        if ds is None or np.isnan(ds):
            ds = 0.001

        pts_cam, pix = depth_to_points_cam(depth_u16, K_map[ci], ds, z_min, z_max, stride)
        if pts_cam.shape[0] == 0:
            continue

        raw_n = int(pts_cam.shape[0])
        pix_arr = np.asarray(pix, dtype=np.int32)
        pts_ref = transform_points_rowvec(pts_cam, T_ref[ci])

        if depth_cube_roi and roi_half is not None:
            mask = cube_roi_mask_in_ref(pts_ref, T_cube_depth, roi_half)
            kept = int(np.count_nonzero(mask))
            if kept == 0:
                continue
            pts_ref = pts_ref[mask]
            pix_arr = pix_arr[mask]
            if verbose:
                print(f"[INFO] cam{ci}: raw={raw_n}  roi={kept} points fused")
        else:
            if verbose:
                print(f"[INFO] cam{ci}: {raw_n} points fused")

        cols = rgb[pix_arr[:, 0], pix_arr[:, 1]].astype(np.float64)
        frame_pts.append(pts_ref)
        frame_cols.append(cols)

    return frame_pts, frame_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--intrinsics_dir", type=str, required=True)
    parser.add_argument("--calib_dir", type=str, default=None)
    parser.add_argument("--ref_cam_idx", type=int, default=0)
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="시각화/단일 프레임 overlay 기준 프레임 (계산은 전체 프레임 사용)",
    )
    parser.add_argument("--reproj_max_px", type=float, default=5.0,
                        help="PnP 허용 reproj 오차 (Step3와 동일 값 권장)")
    parser.add_argument("--save_overlay", action="store_true")

    parser.add_argument("--use_depth", action="store_true",
                        help="depth 이미지가 있을 때 frame_idx 한 프레임 fusion 수행")
    parser.add_argument("--auto_best_frame", action="store_true",
                        help="depth/시각화용 frame_idx를 PnP 품질 기준으로 자동 선택")
    parser.add_argument("--depth_dense_no_roi", action="store_true",
                        help="결과를 빽빽하게 보기 위한 프리셋: ROI OFF + stride=1 + auto-z-window OFF + vis points 증가")
    parser.add_argument("--depth_pose_mode", type=str, default="frame", choices=["global", "frame"],
                        help="depth ROI/시각화에 사용할 큐브 pose 기준 (default: frame)")
    parser.add_argument("--depth_stride", type=int, default=4,
                        help="depth subsampling stride (default: 4)")
    parser.add_argument("--depth_z_min", type=float, default=0.2,
                        help="depth min range in meters")
    parser.add_argument("--depth_z_max", type=float, default=1.5,
                        help="depth max range in meters")
    parser.add_argument("--depth_auto_z_window_m", type=float, default=0.12,
                        help="선택된 큐브 중심 z 기준 +/- window로 depth z-range를 추가로 좁힘 (0=비활성)")
    parser.add_argument("--depth_cube_roi", dest="depth_cube_roi", action="store_true",
                        help="전역 큐브 pose 기준 ROI 박스로 depth를 crop (default: on)")
    parser.add_argument("--no_depth_cube_roi", dest="depth_cube_roi", action="store_false",
                        help="큐브 ROI crop 비활성화 (기존 동작)")
    parser.add_argument("--depth_cube_roi_margin_m", type=float, default=0.06,
                        help="큐브 반경계에 추가할 ROI margin (m), default=0.06")
    parser.add_argument("--depth_vis_max_points", type=int, default=8000,
                        help="depth PNG 시각화 시 표시할 최대 점 수 (PLY 저장 점 수와는 별개)")
    parser.add_argument("--depth_all_frames", action="store_true",
                        help="모든 프레임 depth fusion (--use_depth와 함께 사용)")
    parser.add_argument("--depth_voxel_size_m", type=float, default=0.0,
                        help="voxel downsampling grid size in meters (0=disable, 권장: 0.001~0.003)")
    parser.add_argument("--depth_frame_skip", type=int, default=1,
                        help="multi-frame 시 N번째 프레임마다 사용 (메모리 절약, default: 1=all)")
    parser.set_defaults(depth_cube_roi=True)
    args = parser.parse_args()

    if args.depth_dense_no_roi:
        args.depth_cube_roi = False
        args.depth_stride = 1
        args.depth_auto_z_window_m = 0.0
        if args.depth_vis_max_points == 8000:
            args.depth_vis_max_points = 50000
        print(
            "[INFO] depth_dense_no_roi preset applied: "
            "depth_cube_roi=OFF, depth_stride=1, depth_auto_z_window_m=0, "
            f"depth_vis_max_points={args.depth_vis_max_points}"
        )

    try:
        _, map_path = load_device_map(args.intrinsics_dir)
        print(f"[INFO] Using device map: {map_path}")
    except Exception as e:
        print(f"[WARN] device_map.json unavailable: {e}")

    cam_indices = discover_cam_indices_from_data(args.root_folder)
    frame_ids = discover_frame_indices(args.root_folder, cam_indices)
    calib_dir = args.calib_dir or os.path.join(args.root_folder, "calib_out_cube")
    if not os.path.exists(calib_dir):
        raise FileNotFoundError(f"calib_dir not found: {calib_dir}")
    if args.ref_cam_idx not in cam_indices:
        raise RuntimeError(f"ref_cam_idx={args.ref_cam_idx} not in data cams={cam_indices}")

    print(f"[INFO] Data cams found: {cam_indices}")
    print(f"[INFO] Frames found: {len(frame_ids)} (min={frame_ids[0]}, max={frame_ids[-1]})")
    print(f"[INFO] Calib dir: {calib_dir}")

    T_ref = {}
    for ci in cam_indices:
        if ci == args.ref_cam_idx:
            T_ref[ci] = np.eye(4, dtype=np.float64)
            continue
        p = os.path.join(calib_dir, f"T_C{args.ref_cam_idx}_C{ci}.npy")
        if not os.path.exists(p):
            print(f"[WARN] Missing transform for cam{ci}: {p} (skip)")
            continue
        T_ref[ci] = np.load(p).astype(np.float64)
        print(f"[INFO] Loaded T_C{args.ref_cam_idx}_C{ci}")

    K_map, D_map = {}, {}
    for ci in cam_indices:
        K_map[ci], D_map[ci], _ = load_intrinsics_cam_npz(args.intrinsics_dir, ci)

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)
    print("[INFO] Cube face_roll_deg:", {int(k): float(v) for k, v in sorted(cfg.face_roll_deg.items())})
    print("[INFO] PnP mode: single_marker_only=True (best marker per frame)")

    pnp_by_frame = collect_pnp_all_frames(
        args.root_folder, cam_indices, frame_ids, cube, K_map, D_map, args.reproj_max_px
    )
    if len(pnp_by_frame) == 0:
        raise RuntimeError("No valid PnP observations collected from any frame.")

    obs_by_frame, all_obs = build_ref_pose_observations(pnp_by_frame, T_ref)
    if len(all_obs) == 0:
        raise RuntimeError("No valid pose observations after applying T_ref transforms.")

    print(f"\n[STEP4-2] 전역 큐브 pose 추정 (all observations={len(all_obs)})")
    T_cube_global = estimate_global_cube_pose(all_obs)
    c = T_cube_global[:3, 3]
    print(f"[INFO] Global cube center in ref cam frame: x={c[0]:.4f} y={c[1]:.4f} z={c[2]:.4f} (m)")

    summarize_cube_consistency(obs_by_frame, T_cube_global)
    evaluate_cross_reproj_all_frames(
        cam_indices, pnp_by_frame, T_ref, K_map, D_map, args.ref_cam_idx
    )

    if args.auto_best_frame:
        best_fid = select_best_frame_for_depth(pnp_by_frame, args.ref_cam_idx)
        if best_fid is None:
            print("[WARN] auto_best_frame: suitable frame not found, keeping --frame_idx")
        else:
            old_fid = int(args.frame_idx)
            args.frame_idx = int(best_fid)
            print(f"[INFO] auto_best_frame: frame_idx {old_fid} -> {args.frame_idx}")

    print(f"\n[STEP4-3] 시각화 프레임: {args.frame_idx}")
    pnp_viz = pnp_by_frame.get(args.frame_idx, {})
    if len(pnp_viz) == 0:
        print(f"[WARN] frame {args.frame_idx}에서 PnP 성공 카메라가 없습니다.")

    images_viz = load_images_for_frame(args.root_folder, cam_indices, args.frame_idx)
    for ci, rec in pnp_viz.items():
        rec["frame_id"] = int(args.frame_idx)
        print(
            f"[INFO] cam{ci}: frame {args.frame_idx} PnP err_mean={rec['reproj']['err_mean']:.2f}px  "
            f"used={rec['used']}"
        )

    ref_overlay = validate_cross_reproj_single_frame(
        cam_indices, T_ref, K_map, D_map, images_viz, pnp_viz, args.ref_cam_idx
    )
    if args.save_overlay and ref_overlay is not None:
        p = os.path.join(
            args.root_folder,
            f"cross_reproj_ref{args.ref_cam_idx}_frame{args.frame_idx:05d}.jpg"
        )
        cv2.imwrite(p, ref_overlay)
        print(f"[SAVE] {p}")

    vis_path = None
    if args.save_overlay:
        vis_path = os.path.join(
            args.root_folder,
            f"calib_3d_global_frame{args.frame_idx:05d}.png"
        )
    visualize_global_scene(
        T_cube_global, T_ref, cfg, args.ref_cam_idx, args.frame_idx, pnp_viz, save_path=vis_path
    )

    if args.use_depth:
        # ── 공통: z-range, stride, ROI 설정 ─────────────────────────
        # z-range auto-centering 용 pose (multi-frame은 global 사용)
        T_cube_for_z = T_cube_global
        if not args.depth_all_frames and args.depth_pose_mode == "frame":
            T_frame = estimate_frame_cube_pose(obs_by_frame, args.frame_idx)
            if T_frame is not None:
                T_cube_for_z = T_frame
                cdf = T_cube_for_z[:3, 3]
                print(
                    f"[INFO] Depth pose mode: frame  (cube center z={cdf[2]:.4f}m for frame {args.frame_idx})"
                )
            else:
                print("[WARN] Depth pose mode=frame but frame pose unavailable, fallback to global")
                cdf = T_cube_for_z[:3, 3]
                print(f"[INFO] Depth pose mode: global (cube center z={cdf[2]:.4f}m)")
        else:
            cdf = T_cube_for_z[:3, 3]
            print(f"[INFO] Depth pose mode: {'global (all-frames)' if args.depth_all_frames else 'global'} "
                  f"(cube center z={cdf[2]:.4f}m)")

        stride = max(1, int(args.depth_stride))
        z_min = float(args.depth_z_min)
        z_max = float(args.depth_z_max)
        if args.depth_auto_z_window_m > 0:
            zc = float(T_cube_for_z[2, 3])
            w = float(args.depth_auto_z_window_m)
            z_min_auto = zc - w
            z_max_auto = zc + w
            z_min = max(z_min, z_min_auto)
            z_max = min(z_max, z_max_auto)
            print(
                f"[INFO] Depth z-range: base=({args.depth_z_min:.3f}, {args.depth_z_max:.3f}) "
                f"auto_centered=({z_min_auto:.3f}, {z_max_auto:.3f}) -> effective=({z_min:.3f}, {z_max:.3f})"
            )
        else:
            print(f"[INFO] Depth z-range: effective=({z_min:.3f}, {z_max:.3f})")
        if z_max <= z_min:
            raise RuntimeError(
                f"Invalid depth z-range after auto-centering: z_min={z_min:.3f}, z_max={z_max:.3f}. "
                "Adjust --depth_auto_z_window_m or base z range."
            )

        roi_half = None
        if args.depth_cube_roi:
            roi_half_scalar = (float(cfg.cube_side_m) * 0.5) + float(args.depth_cube_roi_margin_m)
            roi_half = np.array([roi_half_scalar, roi_half_scalar, roi_half_scalar], dtype=np.float64)
            print(
                f"[INFO] Depth ROI(cube OBB): ON  half_extents="
                f"({roi_half[0]:.3f}, {roi_half[1]:.3f}, {roi_half[2]:.3f})m"
            )
        else:
            print("[INFO] Depth ROI(cube OBB): OFF")

        # ── 분기: 멀티프레임 vs 단일프레임 ───────────────────────────
        all_pts, all_cols = [], []

        if args.depth_all_frames:
            # === MULTI-FRAME FUSION ===
            fusion_frame_ids = sorted(pnp_by_frame.keys())
            if args.depth_frame_skip > 1:
                fusion_frame_ids = fusion_frame_ids[::args.depth_frame_skip]
            print(
                f"\n[DEPTH-MULTIFRAME] Fusing {len(fusion_frame_ids)} frames "
                f"x {len(cam_indices)} cams (skip={args.depth_frame_skip})"
            )
            frames_used = 0
            for fid in fusion_frame_ids:
                if args.depth_pose_mode == "frame":
                    T_cube_frame = estimate_frame_cube_pose(obs_by_frame, fid)
                    if T_cube_frame is None:
                        T_cube_frame = T_cube_global
                else:
                    T_cube_frame = T_cube_global

                f_pts, f_cols = _fuse_depth_single_frame(
                    root_folder=args.root_folder,
                    frame_idx=fid,
                    cam_indices=cam_indices,
                    T_ref=T_ref,
                    K_map=K_map,
                    intrinsics_dir=args.intrinsics_dir,
                    z_min=z_min, z_max=z_max, stride=stride,
                    depth_cube_roi=args.depth_cube_roi,
                    roi_half=roi_half,
                    T_cube_depth=T_cube_frame,
                    verbose=False,
                )
                if f_pts:
                    n_frame = sum(p.shape[0] for p in f_pts)
                    all_pts.extend(f_pts)
                    all_cols.extend(f_cols)
                    frames_used += 1
                    print(
                        f"[DEPTH-MULTIFRAME] frame {fid:5d}: {n_frame:>8d} points "
                        f"({len(f_pts)} cams)"
                    )
            print(f"[DEPTH-MULTIFRAME] Frames used: {frames_used}/{len(fusion_frame_ids)}")
            T_cube_viz = T_cube_global
            ply_name = "depth_fusion_allframes.ply"
            fig_name = "depth_fusion_allframes.png"
            title_extra = f"all frames ({frames_used} frames)"
        else:
            # === SINGLE-FRAME FUSION (기존 동작) ===
            T_cube_depth = T_cube_for_z
            all_pts, all_cols = _fuse_depth_single_frame(
                root_folder=args.root_folder,
                frame_idx=args.frame_idx,
                cam_indices=cam_indices,
                T_ref=T_ref,
                K_map=K_map,
                intrinsics_dir=args.intrinsics_dir,
                z_min=z_min, z_max=z_max, stride=stride,
                depth_cube_roi=args.depth_cube_roi,
                roi_half=roi_half,
                T_cube_depth=T_cube_depth,
            )
            T_cube_viz = T_cube_depth
            ply_name = f"depth_fusion_frame{args.frame_idx:05d}.ply"
            fig_name = f"depth_fusion_frame{args.frame_idx:05d}.png"
            title_extra = f"frame {args.frame_idx}"

        # ── 공통 후처리: voxel downsample → PLY 저장 → 시각화 ────────
        if all_pts:
            P = np.concatenate(all_pts, axis=0)
            C = np.concatenate(all_cols, axis=0)
            print(f"[INFO] 전체 fused points: {len(P)}")

            if args.depth_voxel_size_m > 0:
                n_before = len(P)
                P, C = voxel_downsample(P, C, args.depth_voxel_size_m)
                print(
                    f"[INFO] Voxel downsample ({args.depth_voxel_size_m*1000:.1f}mm): "
                    f"{n_before} -> {len(P)} points ({100*len(P)/max(n_before,1):.1f}%)"
                )

            ply_path = os.path.join(args.root_folder, ply_name)
            _save_ply(ply_path, P, C)
            print(f"[SAVE] {ply_path}")

            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(111, projection="3d")
            max_vis_points = max(1, int(args.depth_vis_max_points))
            vis_stride = max(1, len(P) // max_vis_points)
            ax.scatter(
                P[::vis_stride, 0], P[::vis_stride, 1], P[::vis_stride, 2],
                c=C[::vis_stride], s=1, alpha=0.6, linewidths=0,
            )
            draw_cube_at_T(ax, T_cube_viz, cfg, alpha=0.25)

            for i, ci in enumerate(sorted(T_ref.keys())):
                cam_pos = np.zeros(3) if ci == args.ref_cam_idx else T_ref[ci][:3, 3]
                ax.scatter(*cam_pos, c=CAM_COLORS_MPL[i % len(CAM_COLORS_MPL)],
                           s=120, marker="^", edgecolors="k", zorder=8)
                ax.text(*cam_pos, f" cam{ci}", fontsize=8,
                        color=CAM_COLORS_MPL[i % len(CAM_COLORS_MPL)])

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(
                f"Depth Fusion – ref cam{args.ref_cam_idx}, {title_extra}\n"
                f"{len(P)} points total  (PLY: {os.path.basename(ply_path)})",
                fontsize=9,
            )
            ax.set_box_aspect([1, 1, 1])
            plt.tight_layout()
            if args.save_overlay:
                fig_path = os.path.join(args.root_folder, fig_name)
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                print(f"[SAVE] {fig_path}")
            plt.show()
        else:
            print("[WARN] depth fusion: 유효 점 없음 (depth 이미지 확인 또는 --save_depth 재캡처 필요)")

    print("[INFO] Step4 finished.")


if __name__ == "__main__":
    main()
