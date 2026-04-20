#!/usr/bin/env python3
"""Post-hoc pose 보정 v3.

사용자 피드백 반영:
  - object_001 빨강: "눕혀져 있음" → 90° orientation 전수 탐색 (6축 + 기본 3 flip)
  - cam2 cylinders: 회전값 차이 → cylinder는 xy scale lock (실린더는 radial 대칭)

변경사항 vs v2:
  1. 비대칭 물체: 24개 orientation 후보 (6 basic flips × 4 yaw, 90° 회전 포함)
  2. Cylinder: 7 DOF (tx/ty/tz/yaw/radial_scale/z_scale/yaw_align)
  3. 비대칭 물체는 여전히 9 DOF anisotropic

저장: src/output/pose_fitted/v3_{timestamp}_frame_{id}/
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (
    load_calibration, load_frame, estimate_table_plane,
    OBJECT_SYMMETRY, T_ISAAC_CV,
)

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"


def simplify_mesh(mesh_full, target=1500):
    import open3d as o3d
    if len(mesh_full.faces) <= target:
        return mesh_full
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_full.vertices))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_full.faces))
    m = m.simplify_quadric_decimation(target)
    return trimesh.Trimesh(vertices=np.asarray(m.vertices),
                           faces=np.asarray(m.triangles), process=False)


def sam_centroid_3d(mask, cam):
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    K = cam.intrinsics.K
    m = (mask > 0) & (depth > 0.05) & (depth < 1.5)
    if m.sum() < 30:
        return None, None
    ys, xs = np.where(m)
    z = depth[ys, xs]
    xc = (xs - K[0, 2]) * z / K[0, 0]
    yc = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([xc, yc, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t
    return np.median(pts_base, axis=0), pts_base.max(0) - pts_base.min(0)


def gen_orientation_candidates(is_sym: bool):
    """전수 orientation 후보 생성.

    non-symmetric:
      - 3축(X,Y,Z) 각각 {0°, 90°, 180°, 270°} 회전 → 24개 rotation symmetry 후보
      - 중복 제거 후 유니크한 회전만 유지
    symmetric (yaw-sym cylinder):
      - 단일 원본 (yaw는 ICP/Nelder-Mead에서 자유)
    """
    if is_sym:
        return [("original", np.eye(3))]

    # Euler (X, Y, Z) 각도 쌍 탐색
    angles = [0, 90, 180, 270]
    candidates = {}
    for ax, ay, az in itertools.product(angles, angles, angles):
        R = Rot.from_euler("XYZ", [ax, ay, az], degrees=True).as_matrix()
        # quantize to detect unique
        key = tuple(np.round(R, 4).flatten().tolist())
        if key not in candidates:
            tag = f"rot_{ax}_{ay}_{az}"
            candidates[key] = (tag, R)
    return list(candidates.values())


def fit_pose(pose_json, sam_masks, frames, glb_path: Path, obj_name: str,
             max_iter: int = 250, table_n=None):
    scene = trimesh.load(str(glb_path))
    mesh_full = (trimesh.util.concatenate(list(scene.geometry.values()))
                 if isinstance(scene, trimesh.Scene) else scene.copy())
    mesh = simplify_mesh(mesh_full, 1500)
    V_full = np.asarray(mesh_full.vertices)
    model_center = (V_full.max(0) + V_full.min(0)) / 2

    # SAM centroid 강제
    cents = []
    for cam, m in zip(frames, sam_masks):
        if m is None or (m > 0).sum() < 100:
            continue
        c, _ = sam_centroid_3d(m, cam)
        if c is not None:
            cents.append(c)
    target_t = np.median(np.array(cents), axis=0) if cents else None

    T0 = np.array(pose_json["T_base_obj"])
    R0_base = T0[:3, :3].copy()
    s0 = float(pose_json["scale"])
    t0 = target_t if target_t is not None else T0[:3, 3].copy()
    if target_t is not None:
        print(f"  target centroid: {target_t}")

    is_sym = OBJECT_SYMMETRY.get(obj_name, "none") == "yaw"
    ori_candidates = gen_orientation_candidates(is_sym)
    print(f"  orientation candidates: {len(ori_candidates)}")

    ds = 2
    H0, W0 = frames[0].color_bgr.shape[:2]
    obs_small = []
    for m in sam_masks:
        if m is None:
            obs_small.append(None); continue
        ms = cv2.resize(m, (W0 // ds, H0 // ds), interpolation=cv2.INTER_NEAREST)
        obs_small.append(ms > 0)

    tp = -table_n / (np.linalg.norm(table_n) + 1e-12) if table_n is not None else np.array([0, -1, 0])
    ref = np.array([1, 0, 0]) if abs(tp[0]) < 0.9 else np.array([0, 1, 0])
    h1_ax = np.cross(tp, ref); h1_ax /= np.linalg.norm(h1_ax) + 1e-12
    h2_ax = np.cross(tp, h1_ax)

    def apply_cyl(params, R_init):
        # cylinder: tx,ty,tz, yaw, pitch, roll, log_radial, log_z
        dx, dy, dz, dyaw, dpitch, droll, dls_r, dls_z = params
        sr = s0 * float(np.exp(np.clip(dls_r, np.log(0.70), np.log(1.40))))
        sz = s0 * float(np.exp(np.clip(dls_z, np.log(0.70), np.log(1.40))))
        scale_axis = np.array([sr, sr, sz])  # xy lock
        R_yaw = Rot.from_rotvec(tp * dyaw).as_matrix()
        R_pitch = Rot.from_rotvec(h1_ax * dpitch).as_matrix()
        R_roll = Rot.from_rotvec(h2_ax * droll).as_matrix()
        R = R_roll @ R_pitch @ R_yaw @ R_init @ R0_base
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t0 + np.array([dx, dy, dz])
        return T, scale_axis

    def apply_gen(params, R_init):
        # general: 9 DOF
        dx, dy, dz, dyaw, dpitch, droll, dls_x, dls_y, dls_z = params
        sx = s0 * float(np.exp(np.clip(dls_x, np.log(0.70), np.log(1.40))))
        sy = s0 * float(np.exp(np.clip(dls_y, np.log(0.70), np.log(1.40))))
        sz = s0 * float(np.exp(np.clip(dls_z, np.log(0.70), np.log(1.40))))
        scale_axis = np.array([sx, sy, sz])
        R_yaw = Rot.from_rotvec(tp * dyaw).as_matrix()
        R_pitch = Rot.from_rotvec(h1_ax * dpitch).as_matrix()
        R_roll = Rot.from_rotvec(h2_ax * droll).as_matrix()
        R = R_roll @ R_pitch @ R_yaw @ R_init @ R0_base
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t0 + np.array([dx, dy, dz])
        return T, scale_axis

    apply_fn = apply_cyl if is_sym else apply_gen
    n_dof = 8 if is_sym else 9

    def loss(params, R_init):
        T, scale_axis = apply_fn(params, R_init)
        iou_sum = 0; n = 0
        for cam, obs in zip(frames, obs_small):
            if obs is None:
                continue
            h, w = obs.shape
            V = np.asarray(mesh.vertices, dtype=np.float64)
            V_obj = (V - model_center) * scale_axis
            Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
            V_base = (T @ Vh.T)[:3].T
            T_cb = np.linalg.inv(cam.T_base_cam)
            V_cam = (T_cb @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
            z = V_cam[:, 2]
            ok = z > 0.05
            K = cam.intrinsics.K
            u = (K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]) / ds
            v = (K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]) / ds
            rnd = np.zeros((h, w), dtype=np.uint8)
            F = np.asarray(mesh.faces, dtype=np.int32)
            f_ok = ok[F[:, 0]] & ok[F[:, 1]] & ok[F[:, 2]]
            if f_ok.sum() == 0:
                continue
            F_valid = F[f_ok]
            tri = np.stack([
                np.stack([u[F_valid[:, 0]], v[F_valid[:, 0]]], axis=-1),
                np.stack([u[F_valid[:, 1]], v[F_valid[:, 1]]], axis=-1),
                np.stack([u[F_valid[:, 2]], v[F_valid[:, 2]]], axis=-1),
            ], axis=1).astype(np.int32)
            lim = 4 * max(w, h)
            inside = np.abs(tri).reshape(len(tri), -1).max(axis=1) <= lim
            for face in tri[inside]:
                cv2.fillConvexPoly(rnd, face, 255)
            rnd_b = rnd > 0
            inter = int(np.logical_and(rnd_b, obs).sum())
            union = int(np.logical_or(rnd_b, obs).sum())
            iou_sum += inter / max(union, 1)
            n += 1
        return 1.0 - iou_sum / max(n, 1)

    # init simplex
    init_simplex = np.zeros((n_dof + 1, n_dof))
    init_simplex[1][0] = 0.04
    init_simplex[2][1] = 0.04
    init_simplex[3][2] = 0.04
    init_simplex[4][3] = np.radians(15)
    init_simplex[5][4] = np.radians(10)
    init_simplex[6][5] = np.radians(10)
    if is_sym:
        init_simplex[7][6] = np.log(1.15)  # radial
        init_simplex[8][7] = np.log(1.15)  # z
    else:
        init_simplex[7][6] = np.log(1.15)
        init_simplex[8][7] = np.log(1.15)
        init_simplex[9][8] = np.log(1.15)

    # 1단계: 모든 orientation에 대해 빠른 평가 (Nelder-Mead 짧게)
    quick_iter = 60 if len(ori_candidates) > 4 else max_iter
    scored = []
    for tag, R_init in ori_candidates:
        l0 = loss(np.zeros(n_dof), R_init)
        if l0 > 0.85:  # 초기 loss 너무 높으면 skip (분명 wrong orientation)
            continue
        res = minimize(lambda p: loss(p, R_init), np.zeros(n_dof),
                       method="Nelder-Mead",
                       options={"maxiter": quick_iter, "xatol": 1e-3, "fatol": 1e-3,
                                "initial_simplex": init_simplex, "disp": False})
        scored.append((res.fun, tag, R_init, res.x))

    if not scored:
        print(f"  [FAIL] all orientations failed")
        return pose_json

    scored.sort(key=lambda x: x[0])
    top_n = min(3, len(scored))
    print(f"  top{top_n} orientations: " +
          ", ".join(f"{tag}={1-l:.3f}" for l, tag, _, _ in scored[:top_n]))

    # 2단계: top-3에 대해 더 긴 Nelder-Mead
    best = None
    for l_init, tag, R_init, x_init in scored[:top_n]:
        res = minimize(lambda p: loss(p, R_init), x_init, method="Nelder-Mead",
                       options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4,
                                "disp": False})
        if best is None or res.fun < best["loss"]:
            best = {"tag": tag, "R_init": R_init, "x": res.x, "loss": res.fun}

    print(f"  => best: {best['tag']} IoU={1-best['loss']:.3f}")

    T_new, scale_axis = apply_fn(best["x"], best["R_init"])
    rot = Rot.from_matrix(T_new[:3, :3])
    new_pose = dict(pose_json)
    new_pose["T_base_obj"] = T_new.tolist()
    new_pose["rotation_matrix"] = T_new[:3, :3].tolist()
    new_pose["position_m"] = T_new[:3, 3].tolist()
    new_pose["quaternion_xyzw"] = rot.as_quat().tolist()
    new_pose["euler_xyz_deg"] = rot.as_euler("xyz", degrees=True).tolist()
    new_pose["scale"] = float(scale_axis.mean())
    new_pose["anisotropic_scale_xyz"] = scale_axis.tolist()
    new_pose["confidence"] = float(1 - best["loss"])
    new_pose["silhouette_score"] = float(1 - best["loss"])
    new_pose["orientation_tag"] = best["tag"]
    return new_pose


def save_posed_glb(pose_json, glb_src, out_path, coord="opencv"):
    scene = trimesh.load(str(glb_src))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene.copy())
    V = np.asarray(mesh.vertices)
    center = (V.max(0) + V.min(0)) / 2
    scale = np.array(pose_json["anisotropic_scale_xyz"])
    T = np.array(pose_json["T_base_obj"])
    verts = (V - center) * scale
    vh = np.hstack([verts, np.ones((len(verts), 1))])
    vp = (T @ vh.T)[:3].T
    if coord == "isaac":
        vp = (T_ISAAC_CV @ np.hstack([vp, np.ones((len(vp), 1))]).T)[:3].T
    mesh.vertices = vp
    mesh.export(str(out_path), file_type="glb")


def build(frame_id, pose_dir, mask_dir, out_dir, max_iter):
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)
    table_info = estimate_table_plane(frames)
    table_n = table_info[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for obj_name in ["object_001", "object_002", "object_003", "object_004"]:
        pose_path = pose_dir / f"pose_{obj_name}.json"
        if not pose_path.exists():
            continue
        pose_json = json.loads(pose_path.read_text())
        masks = []
        for ci in range(3):
            p = mask_dir / f"{obj_name}_cam{ci}.png"
            masks.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None)
        print(f"\n=== {obj_name} ===")
        glb_path = DATA_DIR / f"{obj_name}.glb"
        new_pose = fit_pose(pose_json, masks, frames, glb_path, obj_name,
                            max_iter=max_iter, table_n=table_n)
        (out_dir / f"pose_{obj_name}.json").write_text(
            json.dumps(new_pose, indent=2, ensure_ascii=False))
        save_posed_glb(new_pose, glb_path, out_dir / f"{obj_name}_posed.glb", "opencv")
        save_posed_glb(new_pose, glb_path, out_dir / f"{obj_name}_posed_isaac.glb", "isaac")
        results.append(new_pose)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--pose_dir",
                    default="src/output/pose_per_object_v2/run_20260420_154121_sam_frame_000000")
    ap.add_argument("--mask_dir",
                    default="src/output/sam_masks/fixed_20260420_151150_frame_000000")
    ap.add_argument("--max_iter", type=int, default=200)
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("src/output/pose_fitted") / f"v3_{ts}_frame_{args.frame_id}"
    build(args.frame_id, Path(args.pose_dir), Path(args.mask_dir), out_dir, args.max_iter)


if __name__ == "__main__":
    main()
