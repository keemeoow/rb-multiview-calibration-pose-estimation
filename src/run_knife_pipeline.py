#!/usr/bin/env python3
"""Single-object pipeline (knife) — SAM mask + ICP pose estimation.

기존 4-object pipeline (`make_final_sam_masks.py` + `pose_per_object_v2.py`) 의
single-object 변형. data_dir, intrinsics, GLB 경로를 명시적으로 받음.

Usage:
    python3 src/run_knife_pipeline.py \
        --data_dir src/data_knife \
        --intr_dir src/data_knife/_intrinsics \
        --glb     src/data_knife/reference_knife.glb \
        --frame_id 000004 \
        --hue_ref 24 --hue_radius 14 --s_min 110 --v_min 100
"""
from __future__ import annotations
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import CameraIntrinsics, CameraFrame
from make_final_sam_masks import (
    auto_refine_mask, _largest_cc, mask_3d_info, glb_extent,
    project_bbox_from_3d, run_sam, keep_nearest_component,
)
from mobile_sam import SamPredictor, sam_model_registry


# ═══════════════════════════════════════════════════════════
# 1. Calibration / Frame loading (custom data_dir)
# ═══════════════════════════════════════════════════════════

def load_calibration(data_dir: Path, intr_dir: Path):
    intrinsics = []
    for ci in range(3):
        npz = np.load(str(intr_dir / f"cam{ci}.npz"), allow_pickle=True)
        intrinsics.append(CameraIntrinsics(
            K=npz["color_K"].astype(np.float64),
            D=npz["color_D"].astype(np.float64),
            depth_scale=float(npz["depth_scale_m_per_unit"]),
            width=int(npz["color_w"]), height=int(npz["color_h"]),
        ))
    ext_dir = data_dir / "cube_session_01" / "calib_out_cube"
    extrinsics = {0: np.eye(4)}
    for ci in [1, 2]:
        extrinsics[ci] = np.load(str(ext_dir / f"T_C0_C{ci}.npy")).astype(np.float64)
    return intrinsics, extrinsics


def load_frame(data_dir: Path, frame_id: str, intrinsics, extrinsics):
    img_dir = data_dir / "object_capture"
    frames = []
    for ci in range(3):
        c = cv2.imread(str(img_dir / f"cam{ci}" / f"rgb_{frame_id}.jpg"))
        d = cv2.imread(str(img_dir / f"cam{ci}" / f"depth_{frame_id}.png"),
                       cv2.IMREAD_UNCHANGED)
        if c is None or d is None:
            raise FileNotFoundError(f"cam{ci}/{frame_id}")
        frames.append(CameraFrame(ci, intrinsics[ci], extrinsics[ci], c, d))
    return frames


# ═══════════════════════════════════════════════════════════
# 2. HSV color mask (knife = yellow)
# ═══════════════════════════════════════════════════════════

def hsv_circular_distance(h: np.ndarray, ref: float) -> np.ndarray:
    return np.minimum(np.abs(h - ref), 180.0 - np.abs(h - ref))


def color_mask(bgr, hue_ref, hue_radius=14.0, s_min=110, v_min=100, v_max=255):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]; v = hsv[:, :, 2]
    hue_ok = hsv_circular_distance(h, hue_ref) <= hue_radius
    sv_ok = (s >= s_min) & (v >= v_min) & (v <= v_max)
    m = (hue_ok & sv_ok).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
    return m


def best_components_sorted(mask, min_area=200, top_k=5):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        bm = (labels == i).astype(np.uint8) * 255
        out.append((a, bm))
    out.sort(key=lambda x: -x[0])
    return [b for _, b in out[:top_k]]


# ═══════════════════════════════════════════════════════════
# 3. 3D size filter
# ═══════════════════════════════════════════════════════════

def filter_by_3d_size(candidates, cam, glb_ext, scale_range=(0.05, 1.5),
                      aspect_max=4.0):
    glb_max = float(glb_ext.max())
    glb_sorted = np.sort(glb_ext)[::-1]
    min_m = glb_max * scale_range[0]
    max_m = glb_max * scale_range[1]
    scored = []
    for m in candidates:
        if int((m > 0).sum()) < 200:
            continue
        info = mask_3d_info(m, cam)
        if info is None or info["max_extent"] < 1e-4:
            continue
        cand_max = info["max_extent"]
        if cand_max < min_m or cand_max > max_m:
            continue
        ext_sorted = np.sort(info["extent"])[::-1]
        if ext_sorted[-1] > glb_sorted[-1] * aspect_max:
            continue
        ratios = [min(a/b, b/a) for a, b in zip(ext_sorted, glb_sorted)
                  if a > 1e-6 and b > 1e-6]
        if not ratios:
            continue
        score = float(np.mean(ratios))
        scored.append((m, info, score))
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ═══════════════════════════════════════════════════════════
# 4. SAM mask generation per camera
# ═══════════════════════════════════════════════════════════

def generate_sam_masks(frames, glb_path, predictor,
                       hue_ref, hue_radius, s_min, v_min,
                       reliability=0.10, anchor_cam=1):
    glb_ext = glb_extent(glb_path)
    print(f"[glb] extent (raw)={glb_ext}, max={glb_ext.max():.3f}m")

    # 1) per-cam HSV candidates → 3D size filter
    per_cam = []
    for ci, cam in enumerate(frames):
        cm = color_mask(cam.color_bgr, hue_ref, hue_radius, s_min, v_min)
        cands = best_components_sorted(cm, min_area=200, top_k=5)
        scored = filter_by_3d_size(cands, cam, glb_ext)
        per_cam.append(scored[0] if scored else None)
        if scored:
            print(f"  cam{ci}: score={scored[0][2]:.3f} centroid={scored[0][1]['centroid']}")
        else:
            print(f"  cam{ci}: no valid 3D candidate")

    # 2) anchor centroid: anchor_cam 우선
    anchor = per_cam[anchor_cam] if anchor_cam < len(per_cam) else None
    if anchor is None:
        valid = [(i, x) for i, x in enumerate(per_cam) if x is not None]
        if not valid:
            return [np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8)
                    for cam in frames]
        _, anchor = max(valid, key=lambda x: x[1][2])
    centroid_anchor = anchor[1]["centroid"]
    print(f"  anchor 3D: center={centroid_anchor}")

    # 3) per-cam SAM with bbox + auto-refine
    POS_CONSISTENT_M = 0.15
    final_masks = []
    for ci, cam in enumerate(frames):
        entry = per_cam[ci]
        use_own = False
        if entry is not None:
            dist = float(np.linalg.norm(entry[1]["centroid"] - centroid_anchor))
            if dist < POS_CONSISTENT_M and entry[2] >= reliability:
                use_own = True
            else:
                print(f"  cam{ci}: own rejected (dist={dist*100:.1f}cm "
                      f"score={entry[2]:.2f})")

        if use_own:
            mask, info, score = entry
            ys, xs = np.where(mask > 0)
            own_bb = (int(xs.min()), int(ys.min()),
                      int(xs.max()), int(ys.max()))
            H, W = cam.color_bgr.shape[:2]
            bw = own_bb[2] - own_bb[0]; bh = own_bb[3] - own_bb[1]
            # bbox 가 이미지의 50% 이상 차지 → 다른 노란 물체 흡수, cross-cam 사용
            if bw > W * 0.7 or bh > H * 0.7:
                use_own = False
                print(f"  cam{ci}: own bbox 너무 큼 ({bw}x{bh}) → cross-cam 전환")

        if use_own:
            anchor_ext = anchor[1]["extent"]
            ext_for_proj = np.maximum(anchor_ext, glb_ext * 0.20)
            proj_bb = project_bbox_from_3d(centroid_anchor, ext_for_proj, cam,
                                            padding_m=0.012)
            if proj_bb is not None:
                bbox = (max(own_bb[0], proj_bb[0]), max(own_bb[1], proj_bb[1]),
                        min(own_bb[2], proj_bb[2]), min(own_bb[3], proj_bb[3]))
                if bbox[2] - bbox[0] < 20 or bbox[3] - bbox[1] < 20:
                    bbox = own_bb
            else:
                bbox = own_bb
            # knife처럼 색상 다중인 물체: SAM bbox를 30% 확장해서 비-노란 부분(grip, blade)까지 포함
            H, W = cam.color_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            bw = x2 - x1; bh = y2 - y1
            pad_x = int(bw * 0.30); pad_y = int(bh * 0.30)
            bbox = (max(0, x1 - pad_x), max(0, y1 - pad_y),
                    min(W - 1, x2 + pad_x), min(H - 1, y2 + pad_y))
            source = "own_expanded"
        else:
            anchor_ext = anchor[1]["extent"]
            ext_for_proj = np.maximum(anchor_ext, glb_ext * 0.45)
            ext_for_proj = np.minimum(ext_for_proj, glb_ext * 1.05)
            bbox = project_bbox_from_3d(centroid_anchor, ext_for_proj, cam,
                                        padding_m=0.012)
            if bbox is None:
                final_masks.append(
                    np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8))
                print(f"  cam{ci}: bbox proj 실패 → empty")
                continue
            source = "cross-cam"

        # SAM에 노란 부분 중심 + 노란 영역 sampling 3점을 prompt로 줌
        # (knife는 multicolor이므로 SAM이 전체 knife 영역을 추론하도록 도움)
        ys_c, xs_c = np.where(entry[0] > 0) if entry is not None else (None, None)
        if xs_c is not None and len(xs_c) > 30:
            # 노란 영역의 (top, mid, bottom) 3개 포인트
            order = np.argsort(ys_c)
            pts = []
            for fr in [0.2, 0.5, 0.8]:
                k = int(len(order) * fr)
                pts.append([int(xs_c[order[k]]), int(ys_c[order[k]])])
            points = np.array(pts)
        else:
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            points = np.array([[cx, cy]])
        with torch.no_grad():
            m = run_sam(predictor, cam.color_bgr, bbox, points)
        m = keep_nearest_component(m, bbox)
        area_raw = int((m > 0).sum())

        # knife는 multicolor → 색상 intersection / auto-refine 색상 가중치 모두 skip
        # 단순히 close + largest CC + 작은 노이즈 제거
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5)
        m = _largest_cc(m)
        area = int((m > 0).sum())
        print(f"  cam{ci}: {source} bbox={bbox} → SAM={area_raw} → "
              f"final={area}")
        final_masks.append(m)

    return final_masks


# ═══════════════════════════════════════════════════════════
# 5. Pose estimation (ICP)
# ═══════════════════════════════════════════════════════════

def backproject_mask_to_base(mask, cam, max_depth=2.0, sample=8000):
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    K = cam.intrinsics.K
    ys, xs = np.where((mask > 0) & (depth > 0.05) & (depth < max_depth))
    if len(xs) == 0:
        return np.zeros((0, 3))
    if len(xs) > sample:
        idx = np.random.choice(len(xs), sample, replace=False)
        ys, xs = ys[idx], xs[idx]
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    return (R @ pts_cam.T).T + t


def estimate_table_plane(frames, max_pts=6000):
    """Fuse all depth points and fit the dominant plane → table normal."""
    all_pts = []
    for cam in frames:
        depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
        K = cam.intrinsics.K
        ys, xs = np.where((depth > 0.1) & (depth < 1.5))
        if len(xs) > max_pts:
            idx = np.random.choice(len(xs), max_pts, replace=False)
            ys, xs = ys[idx], xs[idx]
        z = depth[ys, xs]
        x_cam = (xs - K[0, 2]) * z / K[0, 0]
        y_cam = (ys - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
        R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
        all_pts.append((R @ pts_cam.T).T + t)
    P = np.concatenate(all_pts, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    plane, _ = pcd.segment_plane(distance_threshold=0.008,
                                  ransac_n=3, num_iterations=1000)
    n = np.array(plane[:3]); d = float(plane[3])
    n = n / (np.linalg.norm(n) + 1e-12)
    # 카메라 위 = -y in base, table normal pointing up
    if n[1] > 0:
        n = -n; d = -d
    return n, d


def fuse_object_pts(masks, frames, table_n, table_d,
                     above_table_min_m=0.002, max_dist_m=0.30):
    """3-cam mask → base 좌표 fused 점군 (table 위, anchor 주변만)."""
    pts = []
    for m, cam in zip(masks, frames):
        if int((m > 0).sum()) < 50:
            continue
        p = backproject_mask_to_base(m, cam)
        if len(p) == 0:
            continue
        # table 위 dist
        dist_above = (p @ table_n) + table_d
        keep = dist_above > above_table_min_m
        pts.append(p[keep])
    if not pts:
        return np.zeros((0, 3))
    P = np.concatenate(pts, axis=0)
    if len(P) > 30:
        # remove outliers via 2-stage radius filter
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P)
        pcd_ds = pcd.voxel_down_sample(0.003)
        cl, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=20,
                                                   std_ratio=2.0)
        P = np.asarray(cl.points)
    return P


def normalize_glb(glb_path: Path):
    scene = trimesh.load(str(glb_path))
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) \
        if isinstance(scene, trimesh.Scene) else scene
    return mesh, mesh.centroid.copy(), mesh.bounding_box.extents.copy()


def sample_model_points(mesh, n=20000):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.asarray(pts).astype(np.float64)


def icp_register(model_pts_centered, obj_pts, init_T,
                 voxel=0.004, max_iter=80):
    """Open3D point-to-plane ICP."""
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(model_pts_centered)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(obj_pts)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.015, max_nn=30))
    res = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=voxel * 5,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter),
    )
    return res.transformation, float(res.fitness), float(res.inlier_rmse)


def estimate_pose(masks, frames, glb_path, model_extents,
                   table_n, table_d):
    """누워있는 자세 (lying-flat) init + ICP yaw sweep + 3-axis flip."""
    obj_pts = fuse_object_pts(masks, frames, table_n, table_d)
    print(f"[pose] fused obj_pts={len(obj_pts)}")
    if len(obj_pts) < 50:
        return None
    mesh, model_center, model_ext = normalize_glb(glb_path)
    model_pts = sample_model_points(mesh, 8000) - model_center

    # auto-scale: GLB max extent vs obj_pts longest axis
    obj_ext = obj_pts.max(0) - obj_pts.min(0)
    obj_long = float(np.max(obj_ext))
    glb_long = float(np.max(model_ext))
    scale = float(np.clip(obj_long / glb_long, 0.05, 2.0))
    print(f"[pose] auto_scale={scale:.4f} obj_extent={obj_ext}")

    obj_center = obj_pts.mean(axis=0)
    model_pts_s = model_pts * scale

    # init candidates: model의 SHORTEST axis (= 두께)를 table normal에 정렬,
    # 즉 knife가 누워있는 자세. 그 후 yaw 24개 sweep.
    target_up = -table_n
    sorted_idx = np.argsort(model_ext)  # asc: [shortest, mid, longest]
    short_ax_unit = np.eye(3)[sorted_idx[0]]
    long_ax_unit = np.eye(3)[sorted_idx[2]]

    best = None
    # 누워있을 때는 3가지 rotation 가능: 짧은 축이 +up, -up
    # (2 sign options × 24 yaws = 48 candidates)
    for sign in [+1.0, -1.0]:
        R_align = _rotation_align(short_ax_unit * sign, target_up)
        for yaw_deg in np.linspace(0, 360, 24, endpoint=False):
            R_yaw = Rot.from_rotvec(target_up * np.radians(yaw_deg)).as_matrix()
            R_init = R_yaw @ R_align
            T_init = np.eye(4)
            T_init[:3, :3] = R_init
            T_init[:3, 3] = obj_center
            T_icp, fit, rmse = icp_register(model_pts_s, obj_pts, T_init,
                                             voxel=0.005, max_iter=80)
            score = fit - rmse * 8
            if best is None or score > best[0]:
                best = (score, T_icp, fit, rmse, scale)
    print(f"[pose] best fit={best[2]:.3f} rmse={best[3]*1000:.1f}mm")
    return {
        "T_base_obj": best[1],
        "scale": best[4],
        "fitness": best[2],
        "rmse": best[3],
        "model_center": model_center,
        "model_extents": model_ext,
        "mesh": mesh,
    }


def _rotation_align(a, b):
    """Rotation aligning unit vector a → unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))


# ═══════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════

def render_silhouette(mesh, T_base_obj, scale, model_center, cam):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    h, w = cam.intrinsics.height, cam.intrinsics.width
    V_obj = (V - model_center) * scale
    Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
    V_base = (T_base_obj @ Vh.T)[:3].T
    T_cam_base = np.linalg.inv(cam.T_base_cam)
    V_cam = (T_cam_base @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
    z = V_cam[:, 2]
    ok_v = z > 0.05
    K = cam.intrinsics.K
    u = K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]
    v = K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]
    mask = np.zeros((h, w), dtype=np.uint8)
    f_ok = ok_v[F[:, 0]] & ok_v[F[:, 1]] & ok_v[F[:, 2]]
    if f_ok.sum() == 0:
        return mask
    F_v = F[f_ok]
    tri = np.stack([
        np.stack([u[F_v[:, 0]], v[F_v[:, 0]]], axis=-1),
        np.stack([u[F_v[:, 1]], v[F_v[:, 1]]], axis=-1),
        np.stack([u[F_v[:, 2]], v[F_v[:, 2]]], axis=-1),
    ], axis=1).astype(np.int32)
    lim = 4 * max(w, h)
    inside = np.abs(tri).reshape(len(tri), -1).max(axis=1) <= lim
    for face in tri[inside]:
        cv2.fillConvexPoly(mask, face, 255)
    return mask


def overlay_color(img, mask, color, alpha=0.55):
    out = img.copy()
    if mask.max() == 0:
        return out
    cl = np.zeros_like(img); cl[:, :] = color
    m3 = mask > 0
    blended = cv2.addWeighted(img, 1 - alpha, cl, alpha, 0)
    out[m3] = blended[m3]
    return out


def render_overlays(masks, frames, mesh, pose_dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sam_color = (40, 220, 220)   # yellow-ish for SAM mask
    pose_color = (60, 200, 60)    # green for posed GLB silhouette
    # Top: raw, Mid: SAM mask, Bottom: posed GLB silhouette
    rows = [[], [], []]
    for ci, cam in enumerate(frames):
        rows[0].append(cam.color_bgr.copy())
        sam_ov = overlay_color(cam.color_bgr.copy(), masks[ci], sam_color, 0.5)
        rows[1].append(sam_ov)
        if pose_dict is not None:
            sil = render_silhouette(mesh, pose_dict["T_base_obj"],
                                     pose_dict["scale"],
                                     pose_dict["model_center"], cam)
            pose_ov = overlay_color(cam.color_bgr.copy(), sil, pose_color, 0.6)
        else:
            pose_ov = cam.color_bgr.copy()
        rows[2].append(pose_ov)
    # labels
    for ci in range(3):
        for r, lbl in enumerate(["raw", "SAM mask", "GLB pose"]):
            txt = f"cam{ci}  {lbl}"
            for img in [rows[r][ci]]:
                cv2.putText(img, txt, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                            3, cv2.LINE_AA)
                cv2.putText(img, txt, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                            1, cv2.LINE_AA)
    grid = np.vstack([np.hstack(r) for r in rows])
    cv2.imwrite(str(out_dir / "comparison.png"), grid)


# ═══════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--intr_dir", required=True)
    ap.add_argument("--glb", required=True)
    ap.add_argument("--frame_id", required=True)
    ap.add_argument("--hue_ref", type=float, default=24.0,
                    help="HSV hue (yellow ~ 22-28)")
    ap.add_argument("--hue_radius", type=float, default=14.0)
    ap.add_argument("--s_min", type=int, default=110)
    ap.add_argument("--v_min", type=int, default=100)
    ap.add_argument("--out_root", default="src/output")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    intr_dir = Path(args.intr_dir)
    glb_path = Path(args.glb)
    frame_id = args.frame_id

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sam_dir = Path(args.out_root) / "sam_masks" / f"knife_{ts}_frame_{frame_id}"
    pose_dir = Path(args.out_root) / "pose_knife" / f"{ts}_frame_{frame_id}"
    sam_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f" Knife pipeline — frame {frame_id}")
    print("=" * 64)

    print("Loading MobileSAM...")
    sam_weights = SCRIPT_DIR / "weights" / "mobile_sam.pt"
    sam = sam_model_registry["vit_t"](checkpoint=str(sam_weights))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    intrinsics, extrinsics = load_calibration(data_dir, intr_dir)
    frames = load_frame(data_dir, frame_id, intrinsics, extrinsics)

    print("\n[1/3] SAM mask 생성")
    masks = generate_sam_masks(frames, glb_path, predictor,
                                args.hue_ref, args.hue_radius,
                                args.s_min, args.v_min)
    for ci, m in enumerate(masks):
        cv2.imwrite(str(sam_dir / f"knife_cam{ci}.png"), m)
    # SAM mask overlay comparison: top=raw, bottom=SAM mask
    raw_panels = []; ovl_panels = []
    for ci, cam in enumerate(frames):
        raw = cam.color_bgr.copy()
        ovl = overlay_color(cam.color_bgr.copy(), masks[ci],
                             (40, 220, 220), 0.6)
        cnts, _ = cv2.findContours(masks[ci], cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ovl, cnts, -1, (0, 255, 0), 2)
        a = int((masks[ci] > 0).sum())
        for img, lbl in [(raw, f"cam{ci} raw"),
                          (ovl, f"cam{ci} SAM area={a}")]:
            cv2.putText(img, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lbl, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
        raw_panels.append(raw); ovl_panels.append(ovl)
    sam_grid = np.vstack([np.hstack(raw_panels), np.hstack(ovl_panels)])
    cv2.imwrite(str(sam_dir / "comparison.png"), sam_grid)
    print(f"masks + comparison saved: {sam_dir}")

    print("\n[2/3] Pose 추정")
    table_n, table_d = estimate_table_plane(frames)
    print(f"[plane] n={table_n} d={table_d:.3f}")
    mesh, model_center, model_ext = normalize_glb(glb_path)
    pose = estimate_pose(masks, frames, glb_path, model_ext, table_n, table_d)

    print("\n[3/3] 결과 저장")
    if pose is not None:
        T = pose["T_base_obj"].tolist()
        rot = pose["T_base_obj"][:3, :3]
        eul = Rot.from_matrix(rot).as_euler("xyz", degrees=True)
        quat = Rot.from_matrix(rot).as_quat()
        result = {
            "frame_id": frame_id,
            "object_name": "knife",
            "coordinate_frame": "base (= cam0)",
            "unit": "meter",
            "position_m": pose["T_base_obj"][:3, 3].tolist(),
            "quaternion_xyzw": quat.tolist(),
            "euler_xyz_deg": eul.tolist(),
            "T_base_obj": T,
            "scale": float(pose["scale"]),
            "fitness": float(pose["fitness"]),
            "rmse": float(pose["rmse"]),
            "real_size_m": (np.array(model_ext) * pose["scale"]).tolist(),
        }
        (pose_dir / "pose_knife.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
        # save posed GLB
        V = np.asarray(mesh.vertices, dtype=np.float64)
        V_obj = (V - model_center) * pose["scale"]
        Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
        V_base = (pose["T_base_obj"] @ Vh.T)[:3].T
        posed_mesh = trimesh.Trimesh(vertices=V_base, faces=mesh.faces,
                                      process=False)
        posed_mesh.export(str(pose_dir / "knife_posed.glb"))
        print(f"  pose JSON + GLB saved: {pose_dir}")
        print(f"  pos={result['position_m']}")
        print(f"  euler_deg={result['euler_xyz_deg']}")
        print(f"  scale={result['scale']:.4f}")
    else:
        print("  pose estimation 실패")

    render_overlays(masks, frames, mesh, pose, pose_dir)
    print(f"  comparison.png saved: {pose_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
