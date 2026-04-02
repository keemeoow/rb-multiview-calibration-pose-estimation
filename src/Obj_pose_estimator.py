#!/usr/bin/env python3
"""
멀티뷰 RGB-D 기반 GLB 물체 포즈 추정 v2

전략:
  1) 3카메라 점군 통합 → 테이블 평면 제거
  2) GLB를 여러 스케일로 시도 → FPFH 글로벌 정합 + ICP 정밀 정합
  3) 멀티뷰 depth 일치도로 최종 스케일/포즈 선택
  4) 와이어프레임 오버레이 시각화

실행:
  python3 src/Obj_pose_estimator.py \
    --data_dir src/data \
    --glb_path src/data/Hole.glb \
    --frame_id 000000
"""

import os
import json
import argparse
import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import trimesh
import open3d as o3d


# ─────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────

@dataclass
class CamData:
    cam_id: int
    K: np.ndarray            # 3×3
    D: np.ndarray
    depth_scale: float
    color: np.ndarray        # BGR
    depth: np.ndarray        # uint16
    T_to_cam0: np.ndarray    # 4×4


@dataclass
class PoseResult:
    T_obj_in_cam0: np.ndarray   # 4×4
    scale: float
    score: float
    fitness: float
    rmse: float


# ─────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────

def load_cameras(data_dir: str, frame_id: str, num_cams: int = 3,
                  capture_dir: Optional[str] = None) -> List[CamData]:
    intr_dir = os.path.join(data_dir, "_intrinsics")
    ext_dir = os.path.join(data_dir, "cube_session_01", "calib_out_cube")

    if capture_dir:
        img_dir = capture_dir
    else:
        img_dir = None
        for name in sorted(os.listdir(data_dir)):
            d = os.path.join(data_dir, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "cam0", f"rgb_{frame_id}.jpg")):
                img_dir = d
                break
        if img_dir is None:
            raise FileNotFoundError(f"frame {frame_id} 이미지를 찾을 수 없습니다.")
    print(f"  이미지 폴더: {img_dir}")

    cams = []
    for ci in range(num_cams):
        npz = np.load(os.path.join(intr_dir, f"cam{ci}.npz"), allow_pickle=True)
        K = npz["color_K"].astype(np.float64)
        D = npz["color_D"].astype(np.float64)
        ds = float(npz["depth_scale_m_per_unit"])
        T = np.eye(4) if ci == 0 else np.load(
            os.path.join(ext_dir, f"T_C0_C{ci}.npy")).astype(np.float64)
        color = cv2.imread(os.path.join(img_dir, f"cam{ci}", f"rgb_{frame_id}.jpg"))
        depth = cv2.imread(os.path.join(img_dir, f"cam{ci}", f"depth_{frame_id}.png"),
                           cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise FileNotFoundError(f"cam{ci} frame {frame_id}")
        cams.append(CamData(ci, K, D, ds, color, depth, T))
        print(f"  cam{ci}: loaded")
    return cams


def load_glb(glb_path: str) -> trimesh.Trimesh:
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(list(scene.geometry.values()))
    return scene


# ─────────────────────────────────────────────────────────
# Point cloud generation
# ─────────────────────────────────────────────────────────

def depth_to_pcd(depth, K, ds, min_d=0.1, max_d=1.0):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) * ds
    valid = (z > min_d) & (z < max_d)
    z = z[valid]
    x = (u[valid] - K[0, 2]) * z / K[0, 0]
    y = (v[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1)


def merge_point_clouds(cams, voxel_size=0.002):
    all_pts = []
    for cam in cams:
        pts = depth_to_pcd(cam.depth, cam.K, cam.depth_scale)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam0 = (cam.T_to_cam0 @ pts_h.T)[:3].T
        all_pts.append(pts_cam0)
        print(f"  cam{cam.cam_id}: {len(pts)} pts")
    pts = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"  통합: {len(pts)} → {len(pcd.points)} pts (voxel={voxel_size}m)")
    return pcd


def remove_table(pcd, dist_thresh=0.008, min_h=0.003, max_h=0.12):
    plane, _ = pcd.segment_plane(dist_thresh, 3, 1000)
    a, b, c, d = plane
    norm = np.sqrt(a*a + b*b + c*c)
    n = np.array([a, b, c]) / norm
    pd = d / norm
    pts = np.asarray(pcd.points)
    signed = pts @ n + pd

    # 물체가 더 많은 쪽을 "위"로
    if np.sum(signed < -min_h) > np.sum(signed > min_h):
        signed = -signed

    keep = (signed > min_h) & (signed < max_h)
    obj = pcd.select_by_index(np.where(keep)[0])
    print(f"  평면 제거: {len(pts)} → {len(obj.points)} pts")
    return obj


# ─────────────────────────────────────────────────────────
# GLB → 점군
# ─────────────────────────────────────────────────────────

def glb_to_pcd(mesh, n_samples=20000):
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pts = pts - pts.mean(axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ─────────────────────────────────────────────────────────
# 멀티뷰 depth 검증 (핵심 스코어)
# ─────────────────────────────────────────────────────────

def multiview_depth_score(model_pts_cam0, cams, tol=0.015):
    """정합된 모델 점 → 각 카메라 depth와 비교 → 일치 비율"""
    total_ok = 0
    total_valid = 0
    for cam in cams:
        h, w = cam.depth.shape
        T_inv = np.linalg.inv(cam.T_to_cam0)
        pts_h = np.hstack([model_pts_cam0, np.ones((len(model_pts_cam0), 1))])
        pts_ci = (T_inv @ pts_h.T)[:3].T
        front = pts_ci[:, 2] > 0.05
        if front.sum() == 0:
            continue
        p = pts_ci[front]
        u = (cam.K[0, 0] * p[:, 0] / p[:, 2] + cam.K[0, 2]).astype(int)
        v = (cam.K[1, 1] * p[:, 1] / p[:, 2] + cam.K[1, 2]).astype(int)
        ok_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ok_uv.sum() == 0:
            continue
        z_model = p[ok_uv, 2]
        z_real = cam.depth[v[ok_uv], u[ok_uv]].astype(np.float64) * cam.depth_scale
        has_d = z_real > 0.05
        if has_d.sum() == 0:
            continue
        diff = np.abs(z_model[has_d] - z_real[has_d])
        total_ok += (diff < tol).sum()
        total_valid += has_d.sum()
    return total_ok / max(total_valid, 1)


def coverage_score(model_pts_cam0, obj_pcd, radius=0.005):
    """
    양방향 커버리지:
    1) forward: 모델 점 중 실제 점과 가까운 비율 (ICP fitness와 유사)
    2) reverse: 모델 bbox 내 실제 점 중 모델과 가까운 비율
       → 모델이 너무 작으면 bbox 안의 점을 설명 못함 → reverse가 낮음

    최종 score = forward * reverse
    """
    obj_pts = np.asarray(obj_pcd.points)

    # Forward: 모델→점군
    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(model_pts_cam0)
    fwd_dists = np.asarray(model_pcd.compute_point_cloud_distance(obj_pcd))
    forward = (fwd_dists < radius).sum() / max(len(fwd_dists), 1)

    # Reverse: 모델 bbox 근처의 점군 중 모델과 가까운 비율
    margin = radius * 5
    lo = model_pts_cam0.min(axis=0) - margin
    hi = model_pts_cam0.max(axis=0) + margin
    in_box = np.all((obj_pts >= lo) & (obj_pts <= hi), axis=1)
    n_inbox = in_box.sum()

    if n_inbox < 10:
        return forward * 0.01  # bbox에 점이 거의 없으면 패널티

    inbox_pcd = o3d.geometry.PointCloud()
    inbox_pcd.points = o3d.utility.Vector3dVector(obj_pts[in_box])
    rev_dists = np.asarray(inbox_pcd.compute_point_cloud_distance(model_pcd))
    reverse = (rev_dists < radius).sum() / max(len(rev_dists), 1)

    return forward * reverse


# ─────────────────────────────────────────────────────────
# PCA 회전 후보 생성
# ─────────────────────────────────────────────────────────

def pca_axes(pts):
    c = pts - pts.mean(axis=0)
    cov = c.T @ c / len(pts)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vecs[:, order]


def pca_rotation_candidates(src_pts, tgt_pts):
    """PCA 축 정렬 24 후보 → 중복 제거"""
    R_s = pca_axes(src_pts)
    R_t = pca_axes(tgt_pts)
    cands = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R_p = np.zeros((3, 3))
            for i in range(3):
                R_p[:, i] = signs[i] * R_s[:, perm[i]]
            if np.linalg.det(R_p) < 0:
                continue
            cands.append(R_t @ R_p.T)
    cands.append(np.eye(3))
    # 중복 제거
    unique = [cands[0]]
    for R in cands[1:]:
        dup = False
        for Ru in unique:
            ang = np.arccos(np.clip((np.trace(R @ Ru.T) - 1) / 2, -1, 1))
            if ang < np.radians(5):
                dup = True
                break
        if not dup:
            unique.append(R)
    return unique


# ─────────────────────────────────────────────────────────
# 핵심: 멀티스케일 FPFH + ICP 정합
# ─────────────────────────────────────────────────────────

def estimate_pose(
    obj_pcd: o3d.geometry.PointCloud,
    glb_mesh: trimesh.Trimesh,
    cams: List[CamData],
    voxel_size: float = 0.002,
    real_size_mm: Optional[float] = None,
) -> PoseResult:
    """
    전략:
      1) 물체 점군의 extent 추정 → GLB 스케일 후보 생성
      2) 각 스케일에서 FPFH 글로벌 정합 + PCA 후보 → ICP
      3) 멀티뷰 depth 스코어로 최종 선택

    real_size_mm: 실제 물체의 longest extent (mm). 지정 시 해당 크기만 탐색.
    """
    glb_pcd = glb_to_pcd(glb_mesh, 20000)
    glb_pts = np.asarray(glb_pcd.points)
    glb_extent = (glb_pts.max(0) - glb_pts.min(0)).max()
    print(f"  GLB: {len(glb_pts)} pts, extent={glb_extent:.4f}m")

    obj_pts = np.asarray(obj_pcd.points)
    obj_extent = (obj_pts.max(0) - obj_pts.min(0)).max()
    print(f"  물체 점군: {len(obj_pts)} pts, extent={obj_extent:.4f}m")

    # 스케일 후보 생성
    scale_candidates = []
    if real_size_mm is not None:
        # 고정 크기: 정확한 스케일만 사용
        real_m = real_size_mm / 1000.0
        s_exact = real_m / glb_extent
        scale_candidates = [s_exact]
        print(f"  고정 크기: {real_size_mm:.1f}mm (scale={s_exact:.6f})")
    else:
        for real_cm in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
            s = (real_cm / 100.0) / glb_extent
            scale_candidates.append(s)

    print(f"  스케일 후보: {len(scale_candidates)}개")

    best_score = -1
    best_result = None

    for scale in scale_candidates:
        scaled_pts = glb_pts * scale
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(scaled_pts)
        real_size = glb_extent * scale

        # FPFH 특징 기반 글로벌 정합
        vx = max(voxel_size, real_size * 0.08)
        src_down = src.voxel_down_sample(vx)
        tgt_down = obj_pcd.voxel_down_sample(vx)

        for p in [src_down, tgt_down]:
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vx * 3, max_nn=30))

        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=vx * 5, max_nn=100))
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=vx * 5, max_nn=100))

        # RANSAC 글로벌 정합
        try:
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down, tgt_down, src_fpfh, tgt_fpfh,
                mutual_filter=True,
                max_correspondence_distance=vx * 3,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vx * 3),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
            )
            fpfh_T = ransac.transformation
            fpfh_ok = ransac.fitness > 0.05
        except Exception:
            fpfh_T = np.eye(4)
            fpfh_ok = False

        # PCA 후보도 생성
        pca_rots = pca_rotation_candidates(scaled_pts, obj_pts)

        # 모든 초기화 후보
        init_candidates = []
        if fpfh_ok:
            init_candidates.append(fpfh_T)

        # 물체 점군의 centroid를 초기 위치로
        tgt_center = obj_pts.mean(axis=0)
        src_center = scaled_pts.mean(axis=0)

        for R in pca_rots[:12]:  # 상위 12개만
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tgt_center - R @ src_center
            init_candidates.append(T)

        # 각 초기화에서 ICP
        max_dist = real_size * 0.5
        tgt_n = o3d.geometry.PointCloud(obj_pcd)
        tgt_n.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30))

        best_local_fitness = -1
        best_local_T = np.eye(4)

        for T_init in init_candidates:
            src_c = o3d.geometry.PointCloud(src)
            src_c.transform(T_init)
            src_c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30))

            icp = o3d.pipelines.registration.registration_icp(
                src_c, tgt_n,
                max_correspondence_distance=max_dist,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=80),
            )
            total_T = icp.transformation @ T_init
            if icp.fitness > best_local_fitness:
                best_local_fitness = icp.fitness
                best_local_T = total_T

        if best_local_fitness < 0.1:
            print(f"    scale={scale:.4f} ({real_size*100:.1f}cm): fitness={best_local_fitness:.3f} → skip")
            continue

        # 정밀 ICP
        src_r = o3d.geometry.PointCloud(src)
        src_r.transform(best_local_T)
        src_r.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30))

        cur_T = np.eye(4)
        for icp_s in [3.0, 1.5, 0.8]:
            icp = o3d.pipelines.registration.registration_icp(
                src_r, tgt_n,
                max_correspondence_distance=max_dist * icp_s,
                init=cur_T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=200),
            )
            cur_T = icp.transformation

        final_T = cur_T @ best_local_T
        final_fitness = icp.fitness
        final_rmse = icp.inlier_rmse

        # 멀티뷰 depth 스코어
        aligned = (final_T @ np.hstack([scaled_pts, np.ones((len(scaled_pts), 1))]).T)[:3].T
        depth_sc = multiview_depth_score(aligned, cams)

        # 커버리지 스코어 (모델이 실제 점군을 얼마나 커버하는가)
        cov_sc = coverage_score(aligned, obj_pcd, radius=max(real_size * 0.1, 0.003))

        combined = depth_sc * cov_sc
        print(f"    scale={scale:.4f} ({real_size*100:.1f}cm): "
              f"fitness={final_fitness:.3f}  depth={depth_sc:.3f}  "
              f"cov={cov_sc:.3f}  → score={combined:.4f}")

        if combined > best_score:
            best_score = combined
            best_result = PoseResult(
                T_obj_in_cam0=final_T,
                scale=scale,
                score=combined,
                fitness=final_fitness,
                rmse=final_rmse,
            )

    if best_result is None:
        raise RuntimeError("정합 실패 — 물체를 찾지 못했습니다.")
    return best_result


# ─────────────────────────────────────────────────────────
# 와이어프레임 오버레이
# ─────────────────────────────────────────────────────────

def project_wireframe(mesh, T, scale, cam, color=(0, 255, 0), thickness=1):
    img = cam.color.copy()
    h, w = img.shape[:2]
    verts = (mesh.vertices - mesh.vertices.mean(axis=0)) * scale
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_cam0 = (T @ verts_h.T)[:3].T
    T_inv = np.linalg.inv(cam.T_to_cam0)
    verts_ci = (T_inv @ np.hstack([verts_cam0, np.ones((len(verts_cam0), 1))]).T)[:3].T
    z = verts_ci[:, 2]
    ok = z > 0.05
    u = np.full(len(verts), -1.0)
    v = np.full(len(verts), -1.0)
    u[ok] = cam.K[0, 0] * verts_ci[ok, 0] / z[ok] + cam.K[0, 2]
    v[ok] = cam.K[1, 1] * verts_ci[ok, 1] / z[ok] + cam.K[1, 2]
    for e0, e1 in mesh.edges_unique:
        if not (ok[e0] and ok[e1]):
            continue
        p0 = (int(u[e0]), int(v[e0]))
        p1 = (int(u[e1]), int(v[e1]))
        if abs(p0[0]) > 2*w or abs(p0[1]) > 2*h or abs(p1[0]) > 2*w or abs(p1[1]) > 2*h:
            continue
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────────────────

def save_results(result, mesh, cams, output_dir, frame_id):
    os.makedirs(output_dir, exist_ok=True)
    overlays = []
    for cam in cams:
        ov = project_wireframe(mesh, result.T_obj_in_cam0, result.scale, cam)
        cv2.putText(ov, f"cam{cam.cam_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        overlays.append(ov)
        cv2.imwrite(os.path.join(output_dir, f"overlay_cam{cam.cam_id}_{frame_id}.png"), ov)
    cv2.imwrite(os.path.join(output_dir, f"overlay_all_{frame_id}.png"), np.hstack(overlays))

    T = result.T_obj_in_cam0
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    pose = {
        "frame_id": frame_id,
        "T_obj_in_cam0": T.tolist(),
        "rotation_matrix": T[:3, :3].tolist(),
        "translation_m": T[:3, 3].tolist(),
        "rvec_rad": rvec.flatten().tolist(),
        "euler_xyz_deg": (rvec.flatten() * 180 / np.pi).tolist(),
        "scale_glb_to_real": result.scale,
        "real_size_m": float(
            (np.asarray(mesh.bounding_box.extents)).max() * result.scale),
        "score": result.score,
        "fitness": result.fitness,
        "rmse": result.rmse,
    }
    json_path = os.path.join(output_dir, f"pose_{frame_id}.json")
    with open(json_path, "w") as f:
        json.dump(pose, f, indent=2)
    print(f"  오버레이: {output_dir}/overlay_all_{frame_id}.png")
    print(f"  포즈 JSON: {json_path}")


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def run(args):
    print("=" * 60)
    print(" 멀티뷰 RGB-D GLB 포즈 추정 v2")
    print("=" * 60)

    print("\n[1/4] 데이터 로드")
    cams = load_cameras(args.data_dir, args.frame_id, args.num_cameras,
                         capture_dir=args.capture_dir)

    mesh = load_glb(args.glb_path)
    ext = mesh.bounding_box.extents
    print(f"  GLB: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
          f"extent={ext.max():.3f}m")

    print("\n[2/4] 점군 통합 + 테이블 제거")
    pcd = merge_point_clouds(cams, voxel_size=args.voxel_size)
    obj_pcd = remove_table(pcd)

    print("\n[3/4] 포즈 추정 (FPFH + ICP)")
    real_size_mm = getattr(args, 'real_size_mm', None)
    result = estimate_pose(obj_pcd, mesh, cams, voxel_size=args.voxel_size,
                           real_size_mm=real_size_mm)

    T = result.T_obj_in_cam0
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    real_size = ext.max() * result.scale
    print(f"\n  결과:")
    print(f"    위치: [{T[0,3]:+.4f}, {T[1,3]:+.4f}, {T[2,3]:+.4f}] m")
    print(f"    회전: [{rvec[0,0]:+.3f}, {rvec[1,0]:+.3f}, {rvec[2,0]:+.3f}] rad")
    print(f"    실제 크기: {real_size*100:.1f} cm  (scale={result.scale:.4f})")
    print(f"    품질: fitness={result.fitness:.4f}  depth_score={result.score/max(result.fitness,1e-9):.4f}")

    print("\n[4/4] 결과 저장")
    save_results(result, mesh, cams, args.output_dir, args.frame_id)
    return result


def main():
    parser = argparse.ArgumentParser(description="멀티뷰 GLB 포즈 추정 v2")
    parser.add_argument("--data_dir", default="src/data")
    parser.add_argument("--glb_path", required=True)
    parser.add_argument("--frame_id", default="000000")
    parser.add_argument("--capture_dir", default=None)
    parser.add_argument("--output_dir", default="src/output/pose_result")
    parser.add_argument("--num_cameras", type=int, default=3)
    parser.add_argument("--voxel_size", type=float, default=0.002)
    parser.add_argument("--real_size_mm", type=float, default=None,
                        help="실제 물체의 longest extent (mm). 지정 시 해당 크기로 고정")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
