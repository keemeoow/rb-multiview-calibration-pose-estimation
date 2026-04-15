#!/usr/bin/env python3
"""
CPU-friendly per-object 6D pose estimator.

색상 prior로 각 물체 마스크를 독립적으로 추출 → 멀티뷰 점군 융합 →
GLB ICP 정합. 기존 pose_pipeline.py의 association/recovery 단계에서
발생하던 track mixing / GLB 오매칭 문제를 우회.

출력:
  src/output/pose_per_object/
    object_XXX_posed_{frame}.glb (+ _isaac)
    pose_object_XXX_{frame}.json
    debug_mask_cam{0,1,2}_object_XXX_{frame}.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import (  # type: ignore
    CameraFrame,
    CanonicalModel,
    DEPTH_POLICY,
    OBJECT_COLOR_PRIORS_HSV,
    OBJECT_LABELS,
    OBJECT_SYMMETRY,
    PoseEstimate,
    T_ISAAC_CV,
    backproject_depth,
    combine_pose_scores,
    estimate_table_plane,
    export_result,
    load_calibration,
    load_frame,
    mv_depth_score,
    normalize_glb,
    rotation_candidates,
    sample_model_points,
    silhouette_iou_score,
    transform_points,
)
from scipy.spatial.transform import Rotation as Rot

import open3d as o3d

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
OUT_DIR = SCRIPT_DIR / "output" / "pose_per_object"

# 실제 이미지에서 측정한 HSV (cam0/1/2 평균)
#   red   H≈1-3,  S≈215, V≈165
#   yellow H≈19,  S≈245, V≈210
#   mint   H≈90,  S≈120, V≈170
#   navy   H≈103, S≈120, V≈60
COLOR_REF_HSV = {
    "object_001": (2.0,   215, 165),
    "object_002": (19.0,  240, 205),
    "object_003": (103.0, 120,  60),
    "object_004": (90.0,  120, 170),
}
# hue 반경 / saturation 하한 / value 구간
COLOR_THRESHOLDS = {
    "object_001": {"hue": 10.0, "s_min": 120, "v_min": 80,  "v_max": 240},  # 빨강
    "object_002": {"hue": 10.0, "s_min": 180, "v_min": 140, "v_max": 250},  # 노랑
    "object_003": {"hue": 14.0, "s_min": 65,  "v_min": 25,  "v_max": 110},  # 곤색
    "object_004": {"hue": 14.0, "s_min": 60,  "v_min": 120, "v_max": 230},  # 민트
}


def hsv_circular_distance(h_img: np.ndarray, h_ref: float) -> np.ndarray:
    d = np.abs(h_img - h_ref)
    return np.minimum(d, 180.0 - d)


def color_mask_for_object(bgr: np.ndarray, object_name: str) -> np.ndarray:
    ref_h = COLOR_REF_HSV[object_name][0]
    thr = COLOR_THRESHOLDS[object_name]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    hue_ok = hsv_circular_distance(h, float(ref_h)) <= thr["hue"]
    sv_ok = (s >= thr["s_min"]) & (v >= thr["v_min"]) & (v <= thr["v_max"])
    mask = (hue_ok & sv_ok).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    return mask


def best_connected_component(mask: np.ndarray, min_area: int = 150) -> np.ndarray:
    if int(np.count_nonzero(mask)) < min_area:
        return np.zeros_like(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if int(areas[best - 1]) < min_area:
        return np.zeros_like(mask)
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def fuse_object_points(
    frames: List[CameraFrame],
    masks: List[np.ndarray],
    table_n: np.ndarray,
    table_d: float,
    table_center: np.ndarray,
    table_radius: float,
) -> np.ndarray:
    all_pts = []
    for cam, mask in zip(frames, masks):
        if int(np.count_nonzero(mask)) < 100:
            continue
        pts_cam = backproject_depth(cam, mask)
        if len(pts_cam) == 0:
            continue
        pts_base = transform_points(pts_cam, cam.T_base_cam)
        all_pts.append(pts_base)
    if not all_pts:
        return np.zeros((0, 3))

    pts = np.vstack(all_pts)

    # 테이블 위, 반경 안쪽만. get_above_table_points와 동일 관례 사용.
    heights = -(pts @ table_n + table_d)
    on_top = heights > 0.005  # 5mm 위
    horiz = np.linalg.norm(pts[:, [0, 2]] - table_center[[0, 2]], axis=1)
    within = horiz < max(table_radius, 0.30)
    before = len(pts)
    pts = pts[on_top & within]
    if len(pts) < 30:
        print(f"      [debug] fused={before} after_table_filter={len(pts)}")
        return np.zeros((0, 3))

    # 보클 다운샘플
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(DEPTH_POLICY["voxel_size_m"])

    # 통계적 아웃라이어 제거
    if len(pcd.points) > 80:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # DBSCAN으로 최대 클러스터 선택
    pts_np = np.asarray(pcd.points)
    if len(pts_np) < 30:
        return pts_np
    labels = np.array(
        pcd.cluster_dbscan(
            eps=DEPTH_POLICY.get("cluster_eps_m", 0.012),
            min_points=10,
            print_progress=False,
        )
    )
    if labels.max() < 0:
        return pts_np
    u, c = np.unique(labels[labels >= 0], return_counts=True)
    best = u[int(np.argmax(c))]
    return pts_np[labels == best]


def register_direct(
    object_name: str,
    model: CanonicalModel,
    model_pts: np.ndarray,
    obj_pts: np.ndarray,
    frames: List[CameraFrame],
    masks: List[np.ndarray],
) -> Optional[PoseEstimate]:
    """Centroid + PCA rotation sweep + point-to-plane ICP.

    FPFH 없이 동작하므로 희박한 부분 관측에서도 안정.
    yaw-symmetric 객체는 yaw sweep만.
    """
    symmetry = OBJECT_SYMMETRY.get(object_name, "none")
    obj_center = obj_pts.mean(axis=0)
    mod_max = float((model_pts.max(0) - model_pts.min(0)).max())

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(obj_pts)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.02, 30))

    best: Optional[PoseEstimate] = None
    all_stats = {"iter": 0, "rejected_center": 0, "icp_fail": 0}

    # 모델 실크기는 GLB 단위(m). 부분 관측이 많아도 scale=1.0 근방으로 제한.
    scale_grid = [1.0, 0.95, 1.05, 0.90, 1.10]

    for scale in scale_grid:
        scaled = model_pts * scale
        src_center = scaled.mean(0)

        # 초기 회전 후보: PCA 기반 + yaw sweep
        rots = list(rotation_candidates(scaled, obj_pts))[:10]
        if symmetry == "yaw":
            yaw_extra = []
            for base in rots[:3]:
                for yaw_deg in np.linspace(0, 360, 12, endpoint=False):
                    Ry = Rot.from_euler("y", yaw_deg, degrees=True).as_matrix()
                    yaw_extra.append(Ry @ base)
            rots = rots + yaw_extra

        for R in rots:
            all_stats["iter"] += 1
            T0 = np.eye(4)
            T0[:3, :3] = R
            T0[:3, 3] = obj_center - R @ src_center

            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(scaled)
            src.transform(T0)
            src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.02, 30))

            # coarse → fine ICP (point-to-point이 희박 관측에 더 안정적)
            cur = np.eye(4)
            good = True
            for mc in [mod_max * 0.30, mod_max * 0.15, mod_max * 0.08]:
                try:
                    res = o3d.pipelines.registration.registration_icp(
                        src, tgt, max(mc, 0.006), cur,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(1e-8, 1e-8, 80),
                    )
                except Exception:
                    good = False
                    break
                cur = res.transformation
            if not good:
                all_stats["icp_fail"] += 1
                continue

            T_final = cur @ T0
            # 중심 이탈 체크 (모델 크기 전체까지 허용 — 부분 관측 고려)
            fc = (T_final @ np.append(src_center, 1))[:3]
            if np.linalg.norm(fc - obj_center) > mod_max * scale * 1.0:
                all_stats["rejected_center"] += 1
                continue

            aligned = transform_points(scaled, T_final)
            # 평가: 점 대응률 + depth/silhouette
            try:
                eval_res = o3d.pipelines.registration.evaluate_registration(
                    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(aligned)),
                    tgt, mod_max * 0.15,
                )
                fitness = float(eval_res.fitness)
                rmse = float(eval_res.inlier_rmse)
            except Exception:
                fitness, rmse = 0.0, 0.0

            ds = mv_depth_score(aligned, frames, masks)
            cov = coverage_score_simple(aligned, obj_pts, max(mod_max * 0.08, 0.004))
            ss = silhouette_iou_score(aligned, frames, masks)
            conf = combine_pose_scores(
                object_name, ds, cov, ss, 0.0, symmetry_map=OBJECT_SYMMETRY
            )

            if best is None or conf > best.confidence:
                Rf = T_final[:3, :3]
                # orthonormalize
                U, _, Vt = np.linalg.svd(Rf)
                R_clean = U @ Vt
                if np.linalg.det(R_clean) < 0:
                    U[:, -1] *= -1
                    R_clean = U @ Vt
                T_clean = T_final.copy()
                T_clean[:3, :3] = R_clean
                rot = Rot.from_matrix(R_clean)
                best = PoseEstimate(
                    T_base_obj=T_clean,
                    position_m=T_clean[:3, 3].copy(),
                    quaternion_xyzw=rot.as_quat(),
                    euler_xyz_deg=rot.as_euler("xyz", degrees=True),
                    scale=float(scale),
                    confidence=float(conf),
                    fitness=float(fitness),
                    rmse=float(rmse),
                    depth_score=float(ds),
                    coverage=float(cov),
                    silhouette_score=float(ss),
                )
    print(
        f"      [reg] iters={all_stats['iter']} icp_fail={all_stats['icp_fail']} "
        f"rejected_center={all_stats['rejected_center']} "
        f"best={'None' if best is None else f'conf={best.confidence:.3f}'}"
    )
    return best


def coverage_score_simple(aligned: np.ndarray, obj_pts: np.ndarray, radius: float) -> float:
    if len(aligned) == 0 or len(obj_pts) == 0:
        return 0.0
    tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
    )
    covered = 0
    for p in aligned:
        k, _, _ = tree.search_radius_vector_3d(p, radius)
        if k > 0:
            covered += 1
    return float(covered) / float(len(aligned))


def estimate_one(
    object_name: str,
    frames: List[CameraFrame],
    model: CanonicalModel,
    glb_src_path: Path,
    table_info,
    out_dir: Path,
    frame_id: str,
) -> Optional[dict]:
    masks = []
    for ci, cam in enumerate(frames):
        raw = color_mask_for_object(cam.color_bgr, object_name)
        best = best_connected_component(raw)
        masks.append(best)
        vis = cam.color_bgr.copy()
        cnts, _ = cv2.findContours(best, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{object_name} cam{ci} area={int(np.count_nonzero(best))}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(out_dir / f"debug_mask_cam{ci}_{object_name}_{frame_id}.png"), vis
        )

    n_valid = sum(1 for m in masks if int(np.count_nonzero(m)) > 200)
    if n_valid < 1:
        print(f"    [SKIP] {object_name}: 유효 마스크 없음")
        return None

    table_n, table_d, table_center, table_radius = table_info
    obj_pts = fuse_object_points(
        frames, masks, table_n, table_d, table_center, table_radius
    )
    if len(obj_pts) < 60:
        print(f"    [SKIP] {object_name}: 융합 점군 부족 ({len(obj_pts)}pts)")
        return None
    ext = obj_pts.max(0) - obj_pts.min(0)
    print(
        f"    {object_name} valid_masks={n_valid}/3 pts={len(obj_pts)} "
        f"extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm"
    )

    model_pts = sample_model_points(model)
    try:
        pose = register_direct(object_name, model, model_pts, obj_pts, frames, masks)
    except Exception as exc:
        print(f"    [FAIL] {object_name}: 정합 실패 ({exc})")
        return None
    if pose is None:
        print(f"    [FAIL] {object_name}: 정합 실패")
        return None

    result = export_result(pose, model, frame_id, out_dir, glb_src_path)
    print(
        f"    [OK] {object_name} conf={pose.confidence:.3f} "
        f"fit={pose.fitness:.3f} rmse={pose.rmse*1000:.1f}mm scale={pose.scale:.3f}"
    )
    return result


def run(frame_id: str, only: Optional[List[str]] = None):
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    print("=" * 64)
    print(f" Per-object Pose Estimation — Frame {frame_id}")
    print("=" * 64)

    table_info = estimate_table_plane(frames)
    tn, td, tc, tr = table_info
    print(
        f"[plane] center=[{tc[0]:+.3f},{tc[1]:+.3f},{tc[2]:+.3f}] radius={tr:.3f}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    object_names = only or [f"object_{i:03d}" for i in range(1, 5)]
    for name in object_names:
        glb = DATA_DIR / f"{name}.glb"
        if not glb.exists():
            print(f"  [SKIP] {name}: GLB 없음 {glb}")
            continue
        print(f"\n  === {name} ({OBJECT_LABELS.get(name, name)}) ===")
        model = normalize_glb(glb)
        r = estimate_one(name, frames, model, glb, table_info, OUT_DIR, frame_id)
        if r is not None:
            results.append(r)

    summary_path = OUT_DIR / f"summary_{frame_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  summary: {summary_path} ({len(results)}/{len(object_names)} 성공)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--all", action="store_true", help="모든 프레임 처리")
    ap.add_argument(
        "--only", default=None, help="콤마로 구분된 object_001..object_004"
    )
    args = ap.parse_args()

    only = args.only.split(",") if args.only else None

    if args.all:
        cam0_dir = DATA_DIR / "object_capture" / "cam0"
        fids = sorted(f.stem.replace("rgb_", "") for f in cam0_dir.glob("rgb_*.jpg"))
        for fid in fids:
            run(fid, only=only)
    else:
        run(args.frame_id, only=only)


if __name__ == "__main__":
    main()
