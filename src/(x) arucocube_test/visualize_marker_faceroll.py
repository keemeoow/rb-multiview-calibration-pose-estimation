"""
Estimate face_roll_deg from a single frame using fixed id_to_face, then visualize
how markers are attached on the cube.

Usage:
  python visualize_marker_attachment_frame.py \
    --root_folder ./data/cube_session_01 \
    --intrinsics_dir ./intrinsics \
    --frame_idx 0 \
    --save
"""

import os
import glob
import json
import argparse
from itertools import product

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src3._aruco_cube import CubeConfig, ArucoCubeTarget, ArucoCubeModel, rot_axis_angle


ROLL_CANDIDATES = [0.0, 90.0, 180.0, 270.0]
FACE_COLORS = {
    "+X": "#FF6666", "-X": "#FFAAAA",
    "+Y": "#66BB66", "-Y": "#AADDAA",
    "+Z": "#6699FF", "-Z": "#AACCFF",
}
FACE_NORMALS = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}

# User-provided corner-contact constraints around marker id=0.
# Each group means those three marker corners should correspond to the same cube vertex neighborhood.
ID0_CORNER_CONTACT_GROUPS = [
    [(0, 0), (4, 1), (1, 0)],
    [(0, 1), (3, 1), (4, 0)],
    [(0, 2), (3, 0), (2, 1)],
    [(0, 3), (1, 1), (2, 0)],
]

# user-corner-index -> model-corner-index mapping candidates
# (D4 symmetries of square corner indexing)
SQUARE_CORNER_REMAPS = [
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
    (0, 3, 2, 1),
    (3, 2, 1, 0),
    (2, 1, 0, 3),
    (1, 0, 3, 2),
]


def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    d = np.load(p)
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


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


def evaluate_corner_contact_groups(cfg: CubeConfig, groups, corner_remap=None):
    """
    Geometry-only score for manual corner-contact constraints.
    Lower is better.
    Returns:
      cost_mm_sum, detail_list
    """
    model = ArucoCubeModel(cfg)
    details = []
    cost_mm_sum = 0.0

    for gi, group in enumerate(groups):
        pts = []
        valid = True
        for mid, cidx in group:
            if mid not in cfg.id_to_face:
                valid = False
                break
            if cidx < 0 or cidx > 3:
                valid = False
                break
            model_cidx = int(cidx)
            if corner_remap is not None:
                model_cidx = int(corner_remap[model_cidx])
            if model_cidx < 0 or model_cidx > 3:
                valid = False
                break
            pts.append(model.marker_corners_in_rig(int(mid))[model_cidx])

        if not valid:
            details.append(
                {
                    "group_index": int(gi),
                    "group": [[int(a), int(b)] for a, b in group],
                    "valid": False,
                    "pair_dists_mm": [],
                    "mean_mm": None,
                }
            )
            cost_mm_sum += 1e6
            continue

        pair_d = []
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d_mm = float(np.linalg.norm(pts[i] - pts[j]) * 1000.0)
                pair_d.append(d_mm)
                cost_mm_sum += d_mm

        details.append(
            {
                "group_index": int(gi),
                "group": [[int(a), int(b)] for a, b in group],
                "valid": True,
                "pair_dists_mm": [float(x) for x in pair_d],
                "mean_mm": float(np.mean(pair_d)),
                "corner_remap_user_to_model": (
                    [int(x) for x in corner_remap] if corner_remap is not None else None
                ),
            }
        )

    return float(cost_mm_sum), details


def contact_marker_ids(groups):
    ids = set()
    for group in groups:
        for mid, _ in group:
            ids.add(int(mid))
    return sorted(ids)


def parse_manual_corner_remap(spec: str):
    s = str(spec).strip().lower()
    if s == "auto":
        return "auto"
    vals = [int(x.strip()) for x in str(spec).split(",") if x.strip() != ""]
    if len(vals) != 4 or sorted(vals) != [0, 1, 2, 3]:
        raise ValueError(
            f"manual_corner_remap must be 'auto' or a permutation like 0,1,2,3. got: {spec}"
        )
    return tuple(vals)


def collect_frame_detections(root_folder: str, frame_idx: int, cam_idxs, detector: ArucoCubeTarget):
    images = {}
    detections = {}
    for ci in cam_idxs:
        rgb_path = os.path.join(root_folder, f"cam{ci}", f"rgb_{frame_idx:05d}.jpg")
        if not os.path.exists(rgb_path):
            continue
        img = cv2.imread(rgb_path)
        if img is None:
            continue
        images[ci] = img

        corners_list, ids = detector.detect(img)
        if ids is None:
            continue

        dets = {}
        for corners, mid in zip(corners_list, ids):
            mid = int(mid)
            if mid not in detector.cfg.id_to_face:
                continue
            dets[mid] = corners.reshape(4, 2).astype(np.float64)
        if dets:
            detections[ci] = dets
    return images, detections


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    dR = Ra @ Rb.T
    c = (np.trace(dR) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti


def _se3_relation_error(Ta: np.ndarray, Tb: np.ndarray):
    trans_mm = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000.0)
    rot_deg = _rot_angle_deg(Ta[:3, :3], Tb[:3, :3])
    term = trans_mm + 5.0 * rot_deg
    return trans_mm, rot_deg, term


def solve_single_marker_pose(mid: int, corners, K, D, model: ArucoCubeModel):
    obj = model.marker_corners_in_rig(int(mid)).reshape(-1, 1, 3).astype(np.float64)
    img = corners.reshape(-1, 1, 2).astype(np.float64)

    retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj, img, K, D, flags=cv2.SOLVEPNP_IPPE
    )
    if retval == 0 or len(rvecs) == 0:
        return None

    best = None
    for rv, tv in zip(rvecs, tvecs):
        rv = np.asarray(rv, dtype=np.float64).reshape(3, 1)
        tv = np.asarray(tv, dtype=np.float64).reshape(3, 1)
        if float(tv[2]) <= 0:
            continue
        proj, _ = cv2.projectPoints(obj.reshape(-1, 3), rv, tv, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1).astype(np.float64)
        err_mean = float(np.mean(err))
        R, _ = cv2.Rodrigues(rv)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tv.reshape(3)
        rec = {
            "mid": int(mid),
            "rvec": rv,
            "tvec": tv,
            "R": R,
            "T_C_O": T,
            "err_mean": err_mean,
            "err_max": float(np.max(err)),
        }
        if best is None or rec["err_mean"] < best["err_mean"]:
            best = rec

    if best is None:
        return None
    return best


def solve_pose_from_detections(dets, K, D, model: ArucoCubeModel, min_points: int = 8):
    obj_list, img_list = [], []
    for mid, corners in dets.items():
        obj = model.marker_corners_in_rig(int(mid))
        obj_list.append(obj.reshape(-1, 1, 3))
        img_list.append(corners.reshape(-1, 1, 2))

    if not obj_list:
        return None

    obj_pts = np.concatenate(obj_list, axis=0).astype(np.float64)
    img_pts = np.concatenate(img_list, axis=0).astype(np.float64)
    n_points = int(obj_pts.shape[0])
    if n_points < int(min_points):
        return None

    if n_points >= 8:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
    else:
        return None

    proj, _ = cv2.projectPoints(obj_pts.reshape(-1, 3), rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - img_pts.reshape(-1, 2), axis=1).astype(np.float64)

    R, _ = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec.reshape(3)
    return {
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        "R": R,
        "cam_pos": cam_pos,
        "obj_pts": obj_pts,
        "img_pts": img_pts,
        "proj_pts": proj.reshape(-1, 2),
        "err": err,
        "err_mean": float(np.mean(err)),
        "err_max": float(np.max(err)),
        "n_points": n_points,
    }


def evaluate_roll_map(
    roll_map,
    detections,
    K_map,
    D_map,
    id_to_face,
    min_markers_per_cam: int,
    use_manual_contacts: bool,
    manual_contact_weight: float,
    manual_corner_remap,
):
    cfg = CubeConfig()
    cfg.id_to_face = dict(id_to_face)
    cfg.face_roll_deg = {int(mid): float(roll_map.get(mid, 0.0)) for mid in cfg.id_to_face.keys()}
    model = ArucoCubeModel(cfg)

    manual_contact_cost_mm = 0.0
    manual_contact_details = []
    best_manual_corner_remap = None
    manual_remap_costs = []
    if use_manual_contacts:
        remap_candidates = (
            SQUARE_CORNER_REMAPS
            if manual_corner_remap == "auto"
            else [tuple(manual_corner_remap)]
        )
        best_cost = None
        best_details = None
        best_remap = None
        for remap in remap_candidates:
            cst, det = evaluate_corner_contact_groups(
                cfg, ID0_CORNER_CONTACT_GROUPS, corner_remap=remap
            )
            manual_remap_costs.append(
                {
                    "corner_remap_user_to_model": [int(x) for x in remap],
                    "cost_mm_sum": float(cst),
                }
            )
            if best_cost is None or cst < best_cost:
                best_cost = float(cst)
                best_details = det
                best_remap = tuple(remap)
        manual_contact_cost_mm = float(best_cost)
        manual_contact_details = best_details if best_details is not None else []
        best_manual_corner_remap = best_remap

    # 1) camera별 marker 단일 PnP
    single = {}
    for ci, dets in detections.items():
        p = {}
        for mid, corners in dets.items():
            rec = solve_single_marker_pose(int(mid), corners, K_map[ci], D_map[ci], model)
            if rec is not None:
                p[int(mid)] = rec
        single[int(ci)] = p

    relations = {
        "same_camera": [],
        "cross_camera": [],
        "overlap_ids": {},
    }

    score = 0.0
    relation_terms = 0

    # 2) 같은 이미지 내 marker간 일관성 (같은 큐브 pose여야 함)
    for ci in sorted(single.keys()):
        mids = sorted(single[ci].keys())
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                ma, mb = mids[i], mids[j]
                Ta = single[ci][ma]["T_C_O"]
                Tb = single[ci][mb]["T_C_O"]
                trans_mm, rot_deg, term = _se3_relation_error(Ta, Tb)
                relations["same_camera"].append(
                    {
                        "cam_idx": int(ci),
                        "marker_a": int(ma),
                        "marker_b": int(mb),
                        "trans_mm": trans_mm,
                        "rot_deg": rot_deg,
                        "term": term,
                    }
                )
                score += term
                relation_terms += 1

    # 3) 카메라 간 overlap marker 기반 일관성
    cam_list = sorted(single.keys())
    for i in range(len(cam_list)):
        for j in range(i + 1, len(cam_list)):
            ca, cb = cam_list[i], cam_list[j]
            shared = sorted(set(single[ca].keys()).intersection(single[cb].keys()))
            relations["overlap_ids"][f"{ca}-{cb}"] = [int(x) for x in shared]

            if len(shared) < 2:
                continue

            pair_T = []
            for mid in shared:
                Ta = single[ca][mid]["T_C_O"]
                Tb = single[cb][mid]["T_C_O"]
                pair_T.append((int(mid), Ta @ _inv_T(Tb)))

            for a in range(len(pair_T)):
                for b in range(a + 1, len(pair_T)):
                    ma, Tab_a = pair_T[a]
                    mb, Tab_b = pair_T[b]
                    trans_mm, rot_deg, term = _se3_relation_error(Tab_a, Tab_b)
                    relations["cross_camera"].append(
                        {
                            "cam_a": int(ca),
                            "cam_b": int(cb),
                            "marker_a": int(ma),
                            "marker_b": int(mb),
                            "trans_mm": trans_mm,
                            "rot_deg": rot_deg,
                            "term": term,
                        }
                    )
                    score += term
                    relation_terms += 1

    # 4) 다중 마커 solvePnP (검증/오버레이용)
    per_cam = {}
    err_all = []
    for ci, dets in detections.items():
        res = solve_pose_from_detections(
            dets,
            K_map[ci],
            D_map[ci],
            model,
            min_points=max(8, int(min_markers_per_cam) * 4),
        )
        if res is None:
            continue
        per_cam[ci] = res
        err_all.append(res["err"])

    expected_pose_count = sum(len(v) for v in detections.values())
    valid_pose_count = sum(len(v) for v in single.values())
    miss_marker_penalty = float(max(0, expected_pose_count - valid_pose_count) * 300.0)
    low_marker_cam_penalty = 0.0
    for ci, dets in detections.items():
        if len(single.get(ci, {})) < int(min_markers_per_cam):
            low_marker_cam_penalty += 800.0

    # 다중마커 reproj는 tie-breaker로 약하게 사용
    reproj_term = 0.0
    if err_all:
        reproj_term = float(np.mean(np.concatenate(err_all, axis=0))) * 0.2

    metrics = {
        "relation_terms": int(relation_terms),
        "miss_marker_penalty": float(miss_marker_penalty),
        "low_marker_cam_penalty": float(low_marker_cam_penalty),
        "reproj_term": float(reproj_term),
        "manual_contact_cost_mm": float(manual_contact_cost_mm),
        "manual_contact_weight": float(manual_contact_weight),
        "manual_contact_details": manual_contact_details,
        "manual_corner_remap_mode": "auto" if manual_corner_remap == "auto" else "fixed",
        "manual_corner_remap_best": (
            [int(x) for x in best_manual_corner_remap]
            if best_manual_corner_remap is not None
            else None
        ),
        "manual_corner_remap_costs": manual_remap_costs,
    }
    # 관계식이 전혀 없어도 manual contact 제약만으로 평가 가능하게 허용
    if relation_terms == 0 and not use_manual_contacts:
        metrics["invalid_reason"] = "no_relation_terms_and_manual_contacts_off"
        return float("inf"), per_cam, cfg, relations, metrics

    score = score + miss_marker_penalty + low_marker_cam_penalty + reproj_term
    score += float(manual_contact_weight) * float(manual_contact_cost_mm)
    return score, per_cam, cfg, relations, metrics


def search_best_rolls(
    detections,
    K_map,
    D_map,
    base_cfg: CubeConfig,
    top_k: int,
    min_markers_per_cam: int,
    use_manual_contacts: bool,
    manual_contact_weight: float,
    manual_corner_remap,
):
    observed_ids = sorted({int(mid) for dets in detections.values() for mid in dets.keys()})
    if not observed_ids and not use_manual_contacts:
        raise RuntimeError("No markers detected in this frame.")

    search_ids = list(observed_ids)
    if use_manual_contacts:
        search_ids = sorted(set(search_ids).union(contact_marker_ids(ID0_CORNER_CONTACT_GROUPS)))

    print(f"[INFO] observed marker ids at this frame: {observed_ids}")
    print(f"[INFO] roll optimization ids: {search_ids}")

    all_combos = list(product(ROLL_CANDIDATES, repeat=len(search_ids)))
    print(f"[INFO] searching face_roll combinations: {len(all_combos)} cases")

    top_results = []
    best = None
    for i, combo in enumerate(all_combos):
        roll_map = {int(mid): float(base_cfg.face_roll_deg.get(mid, 0.0)) for mid in base_cfg.id_to_face.keys()}
        for mid, roll in zip(search_ids, combo):
            roll_map[int(mid)] = float(roll)

        score, per_cam, cfg_eval, relations, metrics = evaluate_roll_map(
            roll_map,
            detections,
            K_map,
            D_map,
            base_cfg.id_to_face,
            min_markers_per_cam,
            use_manual_contacts,
            manual_contact_weight,
            manual_corner_remap,
        )
        item = {
            "score": score,
            "roll_map": roll_map,
            "per_cam": per_cam,
            "cfg": cfg_eval,
            "relations": relations,
            "metrics": metrics,
        }

        if best is None or score < best["score"]:
            best = item

        top_results.append(item)
        if (i + 1) % 128 == 0:
            print(f"  {i+1}/{len(all_combos)} searched... best_score={best['score']:.4f}")

    top_results.sort(key=lambda x: x["score"])
    return best, top_results[:max(1, int(top_k))], observed_ids, search_ids


def visualize_cube_attachment(best, frame_idx: int, save_path: str = None):
    cfg = best["cfg"]
    model = ArucoCubeModel(cfg)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    d = cfg.cube_side_m / 2.0
    face_to_id = {v: k for k, v in cfg.id_to_face.items()}

    for face_name, color in FACE_COLORS.items():
        verts = face_poly_3d(face_name, d)
        ax.add_collection3d(Poly3DCollection(
            [verts.tolist()], alpha=0.12, facecolor=color, edgecolor="gray", linewidth=0.6
        ))

        mid = face_to_id.get(face_name)
        if mid is None:
            continue
        c_face, u_ref, _, n = model.face_defs[face_name]
        mc = model.marker_corners_in_rig(mid)
        loop = np.vstack([mc, mc[0]])
        ax.plot3D(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linewidth=2.2)
        ax.scatter(mc[:, 0], mc[:, 1], mc[:, 2], c=color, s=20, depthshade=False)

        # 마커 corner 순서(0,1,2,3) 라벨
        for corner_idx, p in enumerate(mc):
            p_txt = p + n * (d * 0.06)
            ax.text(
                p_txt[0], p_txt[1], p_txt[2],
                f"{corner_idx}",
                fontsize=7,
                ha="center",
                va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8),
            )

        center = c_face
        roll = float(cfg.face_roll_deg.get(mid, 0.0))

        # 공통 기준 시각화:
        # - 검정 실선: roll=0deg 기준 방향(u_ref)
        # - 자홍 실선: 추정 roll 적용 후 방향(u_roll)
        r = rot_axis_angle(n, np.deg2rad(roll))
        u_roll = (r @ u_ref.reshape(3, 1)).reshape(3)
        vec_len = d * 0.42
        p_ref = center + u_ref * vec_len
        p_roll = center + u_roll * vec_len
        ax.plot3D(
            [center[0], p_ref[0]], [center[1], p_ref[1]], [center[2], p_ref[2]],
            color="black", linewidth=1.7
        )
        ax.plot3D(
            [center[0], p_roll[0]], [center[1], p_roll[1]], [center[2], p_roll[2]],
            color="#CC00CC", linewidth=1.7
        )
        ax.text(
            p_ref[0], p_ref[1], p_ref[2],
            "0deg", fontsize=7, color="black", ha="center", va="center"
        )
        ax.text(
            p_roll[0], p_roll[1], p_roll[2],
            "est", fontsize=7, color="#CC00CC", ha="center", va="center"
        )

        lpos = center + n * (d * 0.45)
        ax.text(
            lpos[0], lpos[1], lpos[2],
            f"id {mid}\n{face_name}\nroll={roll:.0f} deg",
            fontsize=8,
            ha="center",
            va="center",
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.75),
        )

    axis_len = d * 0.9
    for vec, col, lbl in [([axis_len, 0, 0], "red", "+X"), ([0, axis_len, 0], "green", "+Y"), ([0, 0, axis_len], "blue", "+Z")]:
        ax.quiver(0, 0, 0, *vec, color=col, arrow_length_ratio=0.25, linewidth=1.6)
        ax.text(vec[0] * 1.2, vec[1] * 1.2, vec[2] * 1.2, lbl, color=col, fontsize=8, fontweight="bold")
    ax.text(0.0, 0.0, -d * 0.55, "common frame origin", fontsize=8, color="black", ha="center")

    all_pts = np.concatenate([face_poly_3d(f, d) for f in FACE_NORMALS], axis=0)
    half = max(np.ptp(all_pts, axis=0).max() * 0.6, cfg.cube_side_m * 1.1)
    mid = np.zeros(3, dtype=np.float64)
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Marker Attachment Visualization (frame {frame_idx})\n"
        "common frame shown: cube axes + per-face 0deg(ref) vs est + corner idx(0..3)",
        fontsize=11
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    else:
        plt.show()


def save_roll_overlay_images(images, detections, best, out_dir: str, frame_idx: int):
    cfg = best["cfg"]
    model = ArucoCubeModel(cfg)
    per_cam = best["per_cam"]
    os.makedirs(out_dir, exist_ok=True)

    for ci, img in images.items():
        vis = img.copy()
        dets = detections.get(ci, {})
        res = per_cam.get(ci, None)

        # detected corners
        for mid, corners in dets.items():
            pts = corners.reshape(4, 2)
            for k, (x, y) in enumerate(pts):
                xi, yi = int(round(x)), int(round(y))
                cv2.circle(vis, (xi, yi), 4, (255, 255, 255), -1)
                cv2.putText(vis, f"d{k}", (xi + 4, yi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, f"d{k}", (xi + 4, yi - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            c = pts.mean(axis=0)
            cv2.putText(vis, f"id{mid}", (int(c[0]) + 4, int(c[1]) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"id{mid}", (int(c[0]) + 4, int(c[1]) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # projected corners from best roll + solved pose (cached in res["proj_pts"])
        if res is not None:
            # res['proj_pts'] corresponds to concatenated detected markers
            offset = 0
            for mid, corners in dets.items():
                n = 4
                if offset + n <= len(res["proj_pts"]):
                    p = res["proj_pts"][offset:offset + n]
                    for k, (x, y) in enumerate(p):
                        xi, yi = int(round(x)), int(round(y))
                        cv2.circle(vis, (xi, yi), 4, (0, 0, 255), -1)
                        cv2.putText(vis, f"p{k}", (xi + 4, yi + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(vis, f"p{k}", (xi + 4, yi + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    offset += n

        header1 = f"frame={frame_idx} cam{ci} | white=detected(d0..d3) red=projected(p0..p3)"
        header2 = f"best_roll_score={best['score']:.4f}"
        cv2.putText(vis, header1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, header1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, header2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, header2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"cam{ci}_frame{frame_idx:05d}_roll_overlay.jpg")
        cv2.imwrite(out_path, vis)
        print(f"[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Estimate face_roll_deg from one frame and visualize marker attachment.")
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--min_markers_per_cam",
        type=int,
        default=2,
        help="roll 추정에 사용할 카메라 최소 검출 마커 수 (default: 2)",
    )
    parser.add_argument(
        "--use_manual_id0_contacts",
        action="store_true",
        help="사용자 제공 id0 코너-접촉 제약을 score에 반영",
    )
    parser.add_argument(
        "--manual_contact_weight",
        type=float,
        default=30.0,
        help="manual contact cost(mm)에 곱할 가중치 (default: 30.0)",
    )
    parser.add_argument(
        "--manual_corner_remap",
        type=str,
        default="auto",
        help="manual contact에서 user corner idx -> model corner idx 매핑. "
             "auto 또는 예: 0,1,2,3",
    )
    parser.add_argument(
        "--manual_only",
        action="store_true",
        help="이미지 관측 관계식 없이 manual corner-contact 제약만으로 roll 탐색",
    )
    parser.add_argument("--save", action="store_true", help="save outputs to files instead of only showing plot")
    args = parser.parse_args()

    manual_corner_remap = parse_manual_corner_remap(args.manual_corner_remap)

    base_cfg = CubeConfig()
    cam_idxs = discover_cams(args.root_folder)
    if not cam_idxs:
        raise RuntimeError(f"No camera folders found: {args.root_folder}")

    K_map, D_map = {}, {}
    images, detections = {}, {}
    if args.manual_only:
        print("[INFO] manual_only: skip image detection and intrinsics-based constraints")
    else:
        detector = ArucoCubeTarget(base_cfg)
        for ci in cam_idxs:
            try:
                K_map[ci], D_map[ci] = load_intrinsics(args.intrinsics_dir, ci)
            except Exception as e:
                print(f"[WARN] cam{ci} intrinsics unavailable: {e}")
        images, detections = collect_frame_detections(args.root_folder, args.frame_idx, cam_idxs, detector)
        if not detections:
            raise RuntimeError(f"No detections found at frame {args.frame_idx}.")
        detections = {
            ci: dets for ci, dets in detections.items()
            if len(dets) >= int(args.min_markers_per_cam)
        }
        images = {ci: images[ci] for ci in detections.keys() if ci in images}
        if not detections and not args.use_manual_id0_contacts:
            raise RuntimeError(
                f"No cameras satisfy min_markers_per_cam={args.min_markers_per_cam} at frame {args.frame_idx}."
            )

    print("[INFO] id_to_face (fixed):")
    for mid, face in sorted(base_cfg.id_to_face.items()):
        print(f"  marker {mid} -> {face}")
    print("\n[INFO] detections at frame:")
    if detections:
        for ci in sorted(detections):
            print(f"  cam{ci}: ids={sorted(detections[ci].keys())}")
    else:
        print("  (no detection constraints used)")
    if args.manual_only:
        print("[INFO] roll search mode: manual_only")
    else:
        print(
            f"[INFO] roll search uses cameras with >= {args.min_markers_per_cam} markers "
            "(multi-marker only)"
        )
    if args.use_manual_id0_contacts:
        print(
            f"[INFO] manual id0 corner-contact constraints ON "
            f"(weight={args.manual_contact_weight})"
        )
        if manual_corner_remap == "auto":
            print("[INFO] manual corner remap: auto search (8 square symmetries)")
        else:
            print(f"[INFO] manual corner remap (fixed user->model): {list(manual_corner_remap)}")
    else:
        print("[INFO] manual id0 corner-contact constraints OFF")

    best, top_list, observed_ids, search_ids = search_best_rolls(
        detections,
        K_map,
        D_map,
        base_cfg,
        top_k=args.top_k,
        min_markers_per_cam=int(args.min_markers_per_cam),
        use_manual_contacts=bool(args.use_manual_id0_contacts),
        manual_contact_weight=float(args.manual_contact_weight),
        manual_corner_remap=manual_corner_remap,
    )

    print("\n[RESULT] best face_roll_deg:")
    for mid in sorted(base_cfg.id_to_face.keys()):
        roll = float(best["roll_map"].get(mid, 0.0))
        if mid in observed_ids:
            note = ""
        elif mid in search_ids:
            note = " (from manual constraints)"
        else:
            note = " (not constrained)"
        print(f"  marker {mid} ({base_cfg.id_to_face[mid]}): {roll:.0f} deg{note}")
    print(f"[RESULT] best score: {best['score']:.4f}")
    if "metrics" in best:
        m = best["metrics"]
        print(
            "[RESULT] score terms: "
            f"manual_contact_cost_mm={m.get('manual_contact_cost_mm', 0.0):.3f}, "
            f"relation_terms={m.get('relation_terms', 0)}"
        )
        if m.get("manual_corner_remap_best") is not None:
            print(
                "[RESULT] manual corner remap best (user->model): "
                f"{m.get('manual_corner_remap_best')}"
            )
        if m.get("manual_contact_details"):
            print("[RESULT] manual contact group distances (mm):")
            for g in m["manual_contact_details"]:
                if not g.get("valid", False):
                    print(f"  group#{g['group_index']}: invalid")
                    continue
                d = ", ".join(f"{x:.3f}" for x in g.get("pair_dists_mm", []))
                print(f"  group#{g['group_index']}: [{d}] mean={g['mean_mm']:.3f}")

    print("\n[RESULT] top candidates:")
    for rank, item in enumerate(top_list, start=1):
        rolls = ", ".join(f"{mid}:{int(item['roll_map'][mid])}" for mid in sorted(search_ids))
        print(f"  #{rank} score={item['score']:.4f}  rolls[{rolls}]")

    out_json = None
    out_img = None
    out_overlay_dir = None
    if args.save:
        out_json = os.path.join(
            args.root_folder, f"face_roll_est_frame{args.frame_idx:05d}.json"
        )
        out_img = os.path.join(
            args.root_folder, f"marker_attachment_frame{args.frame_idx:05d}.png"
        )
        out_overlay_dir = os.path.join(
            args.root_folder, f"face_roll_overlay_frame{args.frame_idx:05d}"
        )
        with open(out_json, "w") as f:
            json.dump(
                {
                    "frame_idx": int(args.frame_idx),
                    "id_to_face": {str(k): v for k, v in base_cfg.id_to_face.items()},
                    "face_roll_deg": {str(k): float(v) for k, v in best["roll_map"].items()},
                    "observed_marker_ids": [int(x) for x in observed_ids],
                    "score": float(best["score"]),
                    "relations": best.get("relations", {}),
                    "metrics": best.get("metrics", {}),
                },
                f,
                indent=2,
            )
        print(f"[SAVE] {out_json}")
        if images and detections and best.get("per_cam"):
            save_roll_overlay_images(images, detections, best, out_overlay_dir, args.frame_idx)

    visualize_cube_attachment(best, args.frame_idx, save_path=out_img)


if __name__ == "__main__":
    main()
