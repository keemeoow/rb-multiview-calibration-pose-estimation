#!/usr/bin/env python3
"""
mask + depth + calibration 만으로 물체 크기(bbox)를 mm 단위로 측정.

mesh 만들 필요 없이, 사용자가 들인 캘리브레이션 노력의 가장 직접적 보상:
  1) cam별 mask 안의 픽셀을 depth로 3D 점으로 backproject
  2) T_cam_to_world 로 world(=cam0) 좌표계로 통합
  3) Multi-view consensus: 다른 시점 마스크/깊이에 의해 검증된 점만 유지
  4) (옵션) RANSAC 으로 테이블 평면 제거
  5) DBSCAN 으로 가장 큰 클러스터만 유지
  6) 통계 outlier 제거
  7) AABB / OBB extent 를 mm 로 출력

[실행 예]
PYTHONWARNINGS=ignore python3 src/Obj_Step4_measure_size_from_masks.py \
  --data_dir ./capture_obj \
  --mask_dir ./masks \
  --out ./outputs/size_measurements.json \
  --voxel_m 0.002 \
  --min_views 2 \
  --erode_px 2 \
  --cluster_eps_m 0.01 \
  --depth_tol_m 0.08 \
  --save_ply_dir ./outputs/size_clouds \
  --save_vis_dir ./outputs/size_vis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

# 8 corners of a unit cube in [-0.5, +0.5]^3 — ordered by bit (x,y,z)
_UNIT_CORNERS = np.array([
    [s0, s1, s2] for s0 in (-0.5, 0.5) for s1 in (-0.5, 0.5) for s2 in (-0.5, 0.5)
], dtype=np.float64)
# 12 edges (pairs whose binary indices differ in exactly one bit)
_OBB_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def load_K(p: Path) -> np.ndarray:
    return np.loadtxt(p, dtype=np.float64).reshape(3, 3)


def load_T(p: Path) -> np.ndarray:
    return np.loadtxt(p, dtype=np.float64).reshape(4, 4)


def load_cam(data_dir: Path, mask_obj_dir: Path, cam_info: dict, erode_px: int) -> dict | None:
    cam_id = f"cam{cam_info['cam_idx']}"
    mask_path = mask_obj_dir / f"{cam_id}_mask.png"
    depth_path = data_dir / f"{cam_id}_depth.png"
    if not (mask_path.exists() and depth_path.exists()):
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        mask = cv2.erode(mask, k)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    K = load_K(data_dir / cam_info["K_file"])
    T_c2w = load_T(data_dir / cam_info["T_file"])
    rgb_path = data_dir / f"{cam_id}_rgb.png"
    return {
        "cam_id": cam_id,
        "K": K,
        "T_c2w": T_c2w,
        "T_w2c": np.linalg.inv(T_c2w),
        "depth_scale": float(cam_info["depth_scale_m_per_unit"]),
        "mask": mask,
        "depth": depth,
        "rgb_path": rgb_path if rgb_path.exists() else None,
        "H": mask.shape[0],
        "W": mask.shape[1],
    }


def backproject_mask(cam: dict, z_min: float = 0.1, z_max: float = 3.0) -> np.ndarray:
    """cam 의 mask 안 픽셀들을 depth 로 backproject → cam frame 3D 점 (N x 3, meters)."""
    ys, xs = np.where(cam["mask"] > 0)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = cam["depth"][ys, xs].astype(np.float64) * cam["depth_scale"]
    valid = (z > z_min) & (z < z_max)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)
    K = cam["K"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    return np.stack([X, Y, z], axis=1)


def transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    Ph = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ Ph.T).T[:, :3]


def consensus_filter(
    pts_world: np.ndarray,
    cams: list[dict],
    min_views: int = 2,
    depth_tol_m: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """각 world 점을 모든 카메라에 projection 해서 mask 안 + depth 일치 검증한 vote 합을 반환."""
    if len(pts_world) == 0:
        return pts_world, np.zeros(0, dtype=np.int32)

    agree = np.zeros(len(pts_world), dtype=np.int32)
    Ph = np.hstack([pts_world, np.ones((len(pts_world), 1))])  # N x 4

    for cam in cams:
        Pc = (cam["T_w2c"] @ Ph.T).T[:, :3]
        z = Pc[:, 2]
        in_front = z > 0.05
        z_safe = np.where(in_front, z, 1.0)
        K = cam["K"]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        u = (fx * Pc[:, 0] / z_safe + cx).astype(np.int32)
        v = (fy * Pc[:, 1] / z_safe + cy).astype(np.int32)
        in_img = in_front & (u >= 0) & (u < cam["W"]) & (v >= 0) & (v < cam["H"])

        u_s = np.clip(u, 0, cam["W"] - 1)
        v_s = np.clip(v, 0, cam["H"] - 1)
        mask_hit = cam["mask"][v_s, u_s] > 0
        depth_val = cam["depth"][v_s, u_s].astype(np.float64) * cam["depth_scale"]
        depth_valid = depth_val > 0.05
        # 깊이가 유효하면 |z - depth_val| < tol 일 때만 OK (앞에 다른 표면이 있으면 가린 거 → vote 안함)
        depth_ok = (~depth_valid) | (np.abs(z - depth_val) < depth_tol_m)

        agree += (in_img & mask_hit & depth_ok).astype(np.int32)

    keep = agree >= min_views
    return pts_world[keep], agree


def largest_cluster(pts: np.ndarray, eps_m: float = 0.01, min_pts: int = 30) -> np.ndarray:
    if len(pts) < min_pts:
        return pts
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.array(pcd.cluster_dbscan(eps=eps_m, min_points=min_pts, print_progress=False))
    if labels.max() < 0:
        return pts
    # noise = -1, ignore
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    best = unique[counts.argmax()]
    return pts[labels == best]


def remove_table_plane(
    pts: np.ndarray, dist_thresh: float = 0.005, min_inliers_ratio: float = 0.3
) -> tuple[np.ndarray, dict]:
    """RANSAC plane 검출. inlier 비율이 충분히 클 때만 제거."""
    if len(pts) < 100:
        return pts, {"removed": False, "reason": "too_few_pts"}
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    try:
        plane, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=300)
    except Exception as e:
        return pts, {"removed": False, "reason": f"ransac_failed: {e}"}
    ratio = len(inliers) / len(pts)
    if ratio < min_inliers_ratio:
        return pts, {"removed": False, "reason": f"inlier_ratio={ratio:.2f}<thr"}
    mask = np.ones(len(pts), dtype=bool)
    mask[inliers] = False
    return pts[mask], {"removed": True, "plane_coeff": [float(x) for x in plane],
                        "inlier_ratio": float(ratio)}


def obb_corners_world(center: np.ndarray, R: np.ndarray, extent: np.ndarray) -> np.ndarray:
    """OBB 의 8개 꼭짓점(world 좌표)."""
    local = _UNIT_CORNERS * np.asarray(extent)
    return center + (R @ local.T).T


def project_world_to_pixel(pts_world: np.ndarray, K: np.ndarray, T_w2c: np.ndarray):
    Ph = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    Pc = (T_w2c @ Ph.T).T[:, :3]
    z = Pc[:, 2]
    valid = z > 0.05
    z_safe = np.where(valid, z, 1.0)
    u = K[0, 0] * Pc[:, 0] / z_safe + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z_safe + K[1, 2]
    return np.stack([u, v], axis=1), valid


def draw_cam_panel(rgb_bgr: np.ndarray, mask: np.ndarray,
                    pts_world: np.ndarray, corners_world: np.ndarray,
                    K: np.ndarray, T_w2c: np.ndarray) -> np.ndarray:
    """원본 RGB(BGR) 위에 mask 윤곽(녹), 측정점(파), OBB 모서리(빨) 오버레이."""
    img = rgb_bgr.copy()
    # GT mask outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 200, 0), 1)

    # measured points
    if len(pts_world):
        pts2d, valid = project_world_to_pixel(pts_world, K, T_w2c)
        for (u, v), ok in zip(pts2d, valid):
            if ok and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (int(u), int(v)), 1, (255, 80, 80), -1)

    # OBB edges
    corners2d, cvalid = project_world_to_pixel(corners_world, K, T_w2c)
    for a, b in _OBB_EDGES:
        if cvalid[a] and cvalid[b]:
            pa = tuple(np.round(corners2d[a]).astype(int))
            pb = tuple(np.round(corners2d[b]).astype(int))
            cv2.line(img, pa, pb, (0, 0, 230), 2, cv2.LINE_AA)
    return img


def save_obj_visualization(obj_name: str, cams: list[dict], pts_world: np.ndarray,
                            obb_center: np.ndarray, obb_R: np.ndarray, obb_extent_raw: np.ndarray,
                            obb_ext_sorted_mm: np.ndarray, out_path: Path):
    """obj 하나에 대해 cam0/cam1/cam2 reprojection + 3D scatter 를 한 PNG 로 저장."""
    corners = obb_corners_world(obb_center, obb_R, obb_extent_raw)

    n_cams = len(cams)
    fig = plt.figure(figsize=(4.5 * (n_cams + 1), 4.5))

    for i, cam in enumerate(cams):
        rgb_bgr = cv2.imread(str(cam["rgb_path"])) if cam.get("rgb_path") else None
        if rgb_bgr is None:
            rgb_bgr = np.full((cam["H"], cam["W"], 3), 50, dtype=np.uint8)
        img_bgr = draw_cam_panel(rgb_bgr, cam["mask"], pts_world, corners, cam["K"], cam["T_w2c"])
        ax = fig.add_subplot(1, n_cams + 1, i + 1)
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{cam['cam_id']} (mask=green, pts=blue, OBB=red)", fontsize=9)
        ax.axis("off")

    ax3d = fig.add_subplot(1, n_cams + 1, n_cams + 1, projection="3d")
    sub = pts_world
    if len(sub) > 3000:
        idx = np.random.default_rng(0).choice(len(sub), 3000, replace=False)
        sub = sub[idx]
    ax3d.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="tab:blue", alpha=0.4)
    edges = [(corners[a], corners[b]) for a, b in _OBB_EDGES]
    ax3d.add_collection3d(Line3DCollection(edges, colors="red", linewidths=1.2))
    # 동일 스케일
    all_pts = np.vstack([sub, corners])
    mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0)
    ctr = (mins + maxs) / 2; r = max((maxs - mins).max() / 2, 0.05)
    ax3d.set_xlim(ctr[0] - r, ctr[0] + r)
    ax3d.set_ylim(ctr[1] - r, ctr[1] + r)
    ax3d.set_zlim(ctr[2] - r, ctr[2] + r)
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title(
        f"OBB (mm): {obb_ext_sorted_mm[0]:.1f} x {obb_ext_sorted_mm[1]:.1f} x "
        f"{obb_ext_sorted_mm[2]:.1f}",
        fontsize=9,
    )

    fig.suptitle(obj_name, fontsize=12, y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def remove_outliers(pts: np.ndarray, nb: int = 30, ratio: float = 2.0) -> np.ndarray:
    if len(pts) < nb + 1:
        return pts
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=ratio)
    return pts[idx]


def measure_one_obj(
    obj_name: str,
    mask_obj_dir: Path,
    data_dir: Path,
    calib: dict,
    voxel_m: float,
    min_views: int,
    erode_px: int,
    do_plane_removal: bool,
    cluster_eps_m: float,
    save_ply: Path | None,
    save_vis: Path | None,
    depth_tol_m: float = 0.03,
) -> dict | None:
    print(f"\n========== {obj_name} ==========")

    cams: list[dict] = []
    for cam_info in calib["cameras"]:
        cam = load_cam(data_dir, mask_obj_dir, cam_info, erode_px)
        if cam is None:
            print(f"  [SKIP] cam{cam_info['cam_idx']}: missing mask/depth")
            continue
        cams.append(cam)

    if not cams:
        print("  [SKIP] no cams loaded")
        return None

    # 1) backproject each cam → world
    pts_world_list = []
    pts_per_cam: dict[str, int] = {}
    for cam in cams:
        pts_cam = backproject_mask(cam)
        pts_w = transform_pts(pts_cam, cam["T_c2w"])
        pts_per_cam[cam["cam_id"]] = int(len(pts_w))
        if len(pts_w):
            pts_world_list.append(pts_w)
        print(f"  {cam['cam_id']}: {len(pts_w):6d} pts (raw)")

    if not pts_world_list:
        return None

    pts = np.concatenate(pts_world_list, axis=0)
    print(f"  total raw: {len(pts)}")

    # 2) multi-view consensus
    pts_filt, agree = consensus_filter(pts, cams, min_views=min_views, depth_tol_m=depth_tol_m)
    print(f"  after consensus(min_views={min_views}): {len(pts_filt)}  "
          f"(views agreement: mean={agree.mean():.2f} max={int(agree.max())})")
    if len(pts_filt) < 50:
        print("  [SKIP] too few points after consensus")
        return None
    pts = pts_filt

    # 3) optional plane removal
    plane_info: dict = {"removed": False}
    if do_plane_removal:
        pts, plane_info = remove_table_plane(pts)
        print(f"  plane removal: {plane_info}")
        if len(pts) < 50:
            print("  [SKIP] too few after plane removal")
            return None

    # 4) largest cluster
    n_before = len(pts)
    pts = largest_cluster(pts, eps_m=cluster_eps_m)
    print(f"  largest cluster: {len(pts)} / {n_before}")
    if len(pts) < 50:
        print("  [SKIP] cluster too small")
        return None

    # 5) statistical outlier
    pts = remove_outliers(pts)

    # 6) optional voxel downsample
    if voxel_m > 0:
        pcd_d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd_d = pcd_d.voxel_down_sample(voxel_size=voxel_m)
        pts = np.asarray(pcd_d.points)
    print(f"  final: {len(pts)} pts")

    if len(pts) < 50:
        print("  [SKIP] too few after final clean")
        return None

    # AABB
    aabb_min = pts.min(axis=0)
    aabb_max = pts.max(axis=0)
    aabb_ext = aabb_max - aabb_min

    # OBB
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    try:
        obb = pcd.get_oriented_bounding_box()
        obb_ext_raw = np.asarray(obb.extent)
        obb_ext = np.sort(obb_ext_raw)[::-1]
        obb_center = np.asarray(obb.center)
        obb_R = np.asarray(obb.R)
    except Exception as e:
        print(f"  [WARN] OBB failed: {e}")
        obb_ext_raw = aabb_ext.copy()
        obb_ext = np.sort(obb_ext_raw)[::-1]
        obb_center = aabb_min + aabb_ext_raw / 2
        obb_R = np.eye(3)

    print(
        f"  AABB (mm): {aabb_ext[0] * 1000:7.1f} x {aabb_ext[1] * 1000:7.1f} x "
        f"{aabb_ext[2] * 1000:7.1f}"
    )
    print(
        f"  OBB  (mm): {obb_ext[0] * 1000:7.1f} (long) x {obb_ext[1] * 1000:7.1f} (med) "
        f"x {obb_ext[2] * 1000:7.1f} (short)"
    )
    print(f"  center(world m): {obb_center}")

    if save_ply is not None:
        save_ply.mkdir(parents=True, exist_ok=True)
        out_ply = save_ply / f"{obj_name}_world_pts.ply"
        o3d.io.write_point_cloud(str(out_ply), pcd)
        print(f"  [SAVE] {out_ply}")

    if save_vis is not None:
        out_png = save_vis / f"{obj_name}_vis.png"
        save_obj_visualization(
            obj_name=obj_name,
            cams=cams,
            pts_world=pts,
            obb_center=obb_center,
            obb_R=obb_R,
            obb_extent_raw=obb_ext_raw,
            obb_ext_sorted_mm=obb_ext * 1000,
            out_path=out_png,
        )
        print(f"  [SAVE] {out_png}")

    return {
        "obj": obj_name,
        "n_pts": int(len(pts)),
        "pts_per_cam_raw": pts_per_cam,
        "aabb_extents_mm": [float(v * 1000) for v in aabb_ext],
        "obb_extents_mm_sorted": [float(v * 1000) for v in obb_ext],
        "obb_center_world_m": [float(v) for v in obb_center],
        "obb_R_world": [[float(v) for v in row] for row in obb_R],
        "plane_removal": plane_info,
        "min_views": min_views,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./capture_obj",
                    help="cam*_K.txt / cam*_T_cam_to_world.txt / cam*_depth.png / calib_info.json")
    ap.add_argument("--mask_dir", default="./masks", help="masks/<obj>/cam*_mask.png")
    ap.add_argument("--out", default="./outputs/size_measurements.json")
    ap.add_argument("--voxel_m", type=float, default=0.002,
                    help="voxel downsample size in meters (0=skip)")
    ap.add_argument("--min_views", type=int, default=2,
                    help="consensus: 점을 유지하기 위해 마스크에 들어가야 하는 시점 수")
    ap.add_argument("--erode_px", type=int, default=2, help="마스크 erosion 픽셀 수")
    ap.add_argument("--cluster_eps_m", type=float, default=0.01,
                    help="DBSCAN 이웃 거리(m)")
    ap.add_argument("--depth_tol_m", type=float, default=0.03,
                    help="consensus: reprojected z vs measured depth 허용 차이(m). "
                         "두꺼운/둥근 물체는 0.05~0.10 권장")
    ap.add_argument("--remove_plane", action="store_true",
                    help="RANSAC 으로 큰 평면(테이블) 제거 — 평면적 물체엔 끄세요")
    ap.add_argument("--save_ply_dir", default="./outputs/size_clouds",
                    help="저장하지 않으려면 빈 문자열")
    ap.add_argument("--save_vis_dir", default="./outputs/size_vis",
                    help="cam reprojection + 3D OBB 시각화 PNG. 끄려면 빈 문자열")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    mask_root = Path(args.mask_dir)
    calib = json.loads((data_dir / "calib_info.json").read_text())
    save_ply = Path(args.save_ply_dir) if args.save_ply_dir else None
    save_vis = Path(args.save_vis_dir) if args.save_vis_dir else None
    if save_vis is not None:
        save_vis.mkdir(parents=True, exist_ok=True)

    results = []
    for obj_dir in sorted(p for p in mask_root.iterdir() if p.is_dir()):
        r = measure_one_obj(
            obj_name=obj_dir.name,
            mask_obj_dir=obj_dir,
            data_dir=data_dir,
            calib=calib,
            voxel_m=args.voxel_m,
            min_views=args.min_views,
            erode_px=args.erode_px,
            do_plane_removal=args.remove_plane,
            cluster_eps_m=args.cluster_eps_m,
            save_ply=save_ply,
            save_vis=save_vis,
            depth_tol_m=args.depth_tol_m,
        )
        if r:
            results.append(r)

    print("\n=========== SUMMARY OBB (mm) ===========")
    print(f"{'obj':<8}{'long':>9}{'med':>9}{'short':>9}   pts")
    for r in results:
        e = r["obb_extents_mm_sorted"]
        print(f"{r['obj']:<8}{e[0]:>9.1f}{e[1]:>9.1f}{e[2]:>9.1f}   {r['n_pts']}")

    print("\n=========== SUMMARY AABB (mm, sorted) ===========")
    print(f"{'obj':<8}{'long':>9}{'med':>9}{'short':>9}   pts")
    for r in results:
        a = sorted(r["aabb_extents_mm"], reverse=True)
        print(f"{r['obj']:<8}{a[0]:>9.1f}{a[1]:>9.1f}{a[2]:>9.1f}   {r['n_pts']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[SAVE] {out_path}")


if __name__ == "__main__":
    main()
