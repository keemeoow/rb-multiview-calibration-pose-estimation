#!/usr/bin/env python3
"""결과 우선 post-hoc pose 보정.

기존 pose 결과 + SAM 마스크 → 각 물체별로:
  1. SAM 마스크 3D centroid를 pose translation으로 강제
  2. SAM 마스크 bbox 크기로 scale 보정
  3. IoU(rendered silhouette, SAM mask)를 목적함수로 Nelder-Mead 추가 정제
     (7 DOF: tx, ty, tz, yaw, pitch, roll, log_scale)

저장: src/output/pose_fitted/run_{timestamp}_frame_{id}/
"""
from __future__ import annotations

import argparse
import datetime
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
    normalize_glb, OBJECT_LABELS, T_ISAAC_CV,
)

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"

OBJ_COLORS_BGR = {
    "object_001": (60, 60, 255),
    "object_002": (80, 230, 255),
    "object_003": (255, 140, 40),
    "object_004": (210, 240, 130),
}


def sam_mask_centroid_3d(mask: np.ndarray, cam) -> np.ndarray:
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    K = cam.intrinsics.K
    m = (mask > 0) & (depth > 0.05) & (depth < 1.5)
    if m.sum() < 30:
        return None
    ys, xs = np.where(m)
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t
    return np.median(pts_base, axis=0), pts_base.max(0) - pts_base.min(0)


def render_silhouette(mesh, T, scale_per_axis, model_center, cam):
    h, w = cam.intrinsics.height, cam.intrinsics.width
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    if len(V) == 0 or len(F) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    V_obj = (V - model_center) * scale_per_axis
    Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
    V_base = (T @ Vh.T)[:3].T
    T_cb = np.linalg.inv(cam.T_base_cam)
    V_cam = (T_cb @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
    z = V_cam[:, 2]
    ok = z > 0.05
    K = cam.intrinsics.K
    u = K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]
    v = K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]
    mask = np.zeros((h, w), dtype=np.uint8)
    f_ok = ok[F[:, 0]] & ok[F[:, 1]] & ok[F[:, 2]]
    if f_ok.sum() == 0:
        return mask
    F_valid = F[f_ok]
    tri = np.stack([
        np.stack([u[F_valid[:, 0]], v[F_valid[:, 0]]], axis=-1),
        np.stack([u[F_valid[:, 1]], v[F_valid[:, 1]]], axis=-1),
        np.stack([u[F_valid[:, 2]], v[F_valid[:, 2]]], axis=-1),
    ], axis=1).astype(np.int32)
    lim = 4 * max(w, h)
    inside = np.abs(tri).reshape(len(tri), -1).max(axis=1) <= lim
    for face in tri[inside]:
        cv2.fillConvexPoly(mask, face, 255)
    return mask


def simplify_mesh(mesh_full, target_faces=1500):
    import open3d as o3d
    if len(mesh_full.faces) <= target_faces:
        return mesh_full
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_full.vertices))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_full.faces))
    m = m.simplify_quadric_decimation(target_faces)
    return trimesh.Trimesh(vertices=np.asarray(m.vertices),
                            faces=np.asarray(m.triangles), process=False)


def fit_pose(pose_json: dict, sam_masks: list, frames, glb_path: Path,
             max_iter: int = 200) -> dict:
    """기존 pose를 SAM 마스크에 맞춰 정제."""
    scene = trimesh.load(str(glb_path))
    mesh_full = (trimesh.util.concatenate(list(scene.geometry.values()))
                 if isinstance(scene, trimesh.Scene) else scene.copy())
    mesh = simplify_mesh(mesh_full, target_faces=1500)
    model_center = np.asarray(pose_json.get("model_center", [0, 0, 0]))
    if "model_center" not in pose_json:
        V = np.asarray(mesh_full.vertices)
        model_center = (V.max(0) + V.min(0)) / 2

    # 초기 pose
    T0 = np.array(pose_json["T_base_obj"])
    s0 = float(pose_json["scale"])
    aniso0 = np.array(pose_json.get("anisotropic_scale_xyz", [s0, s0, s0]))

    # SAM centroid로 translation 강제 (중앙값)
    centroids = []
    extents = []
    for ci, (cam, m) in enumerate(zip(frames, sam_masks)):
        if m is None or (m > 0).sum() < 100:
            continue
        ret = sam_mask_centroid_3d(m, cam)
        if ret is None:
            continue
        c, e = ret
        centroids.append(c)
        extents.append(e)
    if centroids:
        target_t = np.median(np.array(centroids), axis=0)
        print(f"  target centroid: {target_t}")
        T0[:3, 3] = target_t

    # SAM bool masks (downscaled for speed)
    ds = 2
    H, W = frames[0].color_bgr.shape[:2]
    obs_small = []
    for m in sam_masks:
        if m is None:
            obs_small.append(None); continue
        ms = cv2.resize(m, (W // ds, H // ds), interpolation=cv2.INTER_NEAREST)
        obs_small.append(ms > 0)

    R0 = T0[:3, :3].copy()
    t0 = T0[:3, 3].copy()

    table_n = (-R0[:, 2])  # approximate
    tp = np.array([0, -1, 0])  # world up approximation
    tp = tp / (np.linalg.norm(tp) + 1e-12)

    def apply(params):
        dx, dy, dz, dyaw, dpitch, droll, dls = params
        scale = s0 * float(np.exp(np.clip(dls, np.log(0.85), np.log(1.15))))
        R_yaw = Rot.from_rotvec(tp * dyaw).as_matrix()
        R_pitch = Rot.from_rotvec(np.array([1, 0, 0]) * dpitch).as_matrix()
        R_roll = Rot.from_rotvec(np.array([0, 0, 1]) * droll).as_matrix()
        R = R_roll @ R_pitch @ R_yaw @ R0
        t = t0 + np.array([dx, dy, dz])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T, scale

    def loss(params):
        T, scale = apply(params)
        scale_axis = np.full(3, scale, dtype=np.float64)
        iou_sum = 0.0
        n = 0
        for ci, (cam, obs) in enumerate(zip(frames, obs_small)):
            if obs is None:
                continue
            # render at downscaled resolution
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
            iou = inter / max(union, 1)
            iou_sum += iou
            n += 1
        return 1.0 - (iou_sum / max(n, 1))

    init_simplex = np.zeros((8, 7))
    init_simplex[1][0] = 0.02          # dx 2cm
    init_simplex[2][1] = 0.02
    init_simplex[3][2] = 0.02
    init_simplex[4][3] = np.radians(8)  # yaw
    init_simplex[5][4] = np.radians(5)  # pitch
    init_simplex[6][5] = np.radians(5)  # roll
    init_simplex[7][6] = np.log(1.08)

    print(f"  initial loss={loss(np.zeros(7)):.4f}")
    res = minimize(loss, np.zeros(7), method="Nelder-Mead",
                   options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4,
                            "initial_simplex": init_simplex, "disp": False})
    print(f"  final loss={res.fun:.4f} IoU={1-res.fun:.3f} iters={res.nit}")

    T_new, scale_new = apply(res.x)
    rot = Rot.from_matrix(T_new[:3, :3])
    new_pose = dict(pose_json)
    new_pose["T_base_obj"] = T_new.tolist()
    new_pose["rotation_matrix"] = T_new[:3, :3].tolist()
    new_pose["position_m"] = T_new[:3, 3].tolist()
    new_pose["quaternion_xyzw"] = rot.as_quat().tolist()
    new_pose["euler_xyz_deg"] = rot.as_euler("xyz", degrees=True).tolist()
    new_pose["scale"] = float(scale_new)
    new_pose["anisotropic_scale_xyz"] = [float(scale_new)] * 3
    new_pose["confidence"] = float(1 - res.fun)
    new_pose["silhouette_score"] = float(1 - res.fun)
    return new_pose


def save_posed_glb(pose_json: dict, glb_src: Path, out_path: Path,
                   coord: str = "opencv"):
    scene = trimesh.load(str(glb_src))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene.copy())
    V = np.asarray(mesh.vertices)
    center = (V.max(0) + V.min(0)) / 2
    scale = np.array(pose_json["anisotropic_scale_xyz"])
    T = np.array(pose_json["T_base_obj"])
    verts = (V - center) * scale
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_pose = (T @ verts_h.T)[:3].T
    if coord == "isaac":
        verts_pose = (T_ISAAC_CV @ np.hstack(
            [verts_pose, np.ones((len(verts_pose), 1))]).T)[:3].T
    mesh.vertices = verts_pose
    mesh.export(str(out_path), file_type="glb")


def build(frame_id: str, pose_dir: Path, mask_dir: Path,
          out_dir: Path, max_iter: int):
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy posed GLB to new dir for later overlay
    # 원본 pose json 및 SAM 마스크 로드
    results = []
    for obj_name in ["object_001", "object_002", "object_003", "object_004"]:
        pose_path = pose_dir / f"pose_{obj_name}.json"
        if not pose_path.exists():
            print(f"[SKIP] {obj_name}: pose json 없음")
            continue
        pose_json = json.loads(pose_path.read_text())

        masks = []
        for ci in range(3):
            p = mask_dir / f"{obj_name}_cam{ci}.png"
            masks.append(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None)

        print(f"\n=== {obj_name} ===")
        glb_path = DATA_DIR / f"{obj_name}.glb"
        new_pose = fit_pose(pose_json, masks, frames, glb_path, max_iter=max_iter)

        # 저장
        (out_dir / f"pose_{obj_name}.json").write_text(
            json.dumps(new_pose, indent=2, ensure_ascii=False))
        save_posed_glb(new_pose, glb_path, out_dir / f"{obj_name}_posed.glb", "opencv")
        save_posed_glb(new_pose, glb_path, out_dir / f"{obj_name}_posed_isaac.glb", "isaac")
        results.append(new_pose)

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved to: {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--pose_dir",
                    default="src/output/pose_per_object_v2/run_20260420_154121_sam_frame_000000")
    ap.add_argument("--mask_dir",
                    default="src/output/sam_masks/fixed_20260420_151150_frame_000000")
    ap.add_argument("--max_iter", type=int, default=250)
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("src/output/pose_fitted") / f"run_{ts}_frame_{args.frame_id}"
    build(args.frame_id, Path(args.pose_dir), Path(args.mask_dir), out_dir,
          args.max_iter)


if __name__ == "__main__":
    main()
