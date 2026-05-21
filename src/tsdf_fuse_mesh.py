#!/usr/bin/env python3
"""
Open3D TSDF Fusion: 멀티뷰 RGB-D + 마스크 → metric mesh (GLB)

흐름:
  - 각 객체별로 3대 카메라 RGB-D 통합
  - 마스크 영역만 TSDF에 integrate (mask 밖 픽셀의 depth는 0으로 zero-out → TSDF가 무시)
  - 마칭 큐브로 triangle mesh 추출
  - trimesh로 .glb 저장 (metric scale 자동 — depth 단위 그대로 유지)

전제:
  --data_dir: flat 형식 (capture_obj/cam{0,1,2}_rgb.png, _depth.png, _K.txt, _T_cam_to_world.txt)
  --mask_dir: 객체별 폴더 (masks/<obj>/cam{0,1,2}_mask.png)
  world frame = cam0 (cam0_T_cam_to_world == I)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh


def load_K(p: Path) -> np.ndarray:
    return np.loadtxt(p, dtype=np.float64).reshape(3, 3)


def load_T(p: Path) -> np.ndarray:
    return np.loadtxt(p, dtype=np.float64).reshape(4, 4)


def _apply_mask(mask: np.ndarray, dilate_px: int, erode_px: int) -> np.ndarray:
    """양수 dilate 후 양수 erode 적용. erode가 크면 객체 안쪽만 keep → 책상 침범 방지."""
    out = mask.copy()
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        out = cv2.dilate(out, np.ones((k, k), np.uint8))
    if erode_px > 0:
        k = erode_px * 2 + 1
        out = cv2.erode(out, np.ones((k, k), np.uint8))
    return out


def _remove_dominant_plane(depth_m: np.ndarray, mask_bool: np.ndarray, K: np.ndarray,
                            plane_thresh_m: float) -> np.ndarray:
    """마스크 안 픽셀 중 RANSAC 평면(=책상)에 속하는 점을 제거한 새 mask 반환."""
    ys, xs = np.where(mask_bool & (depth_m > 0))
    if len(xs) < 30:
        return mask_bool
    z = depth_m[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    try:
        plane_model, inliers = pcd.segment_plane(distance_threshold=plane_thresh_m,
                                                  ransac_n=3, num_iterations=200)
    except Exception:
        return mask_bool
    if len(inliers) < 0.3 * len(pts):
        # 평면이 객체보다 작으면 평면이 아닐 가능성 → 그대로 둠
        return mask_bool
    keep = np.ones(len(pts), dtype=bool)
    keep[inliers] = False
    new_mask = np.zeros_like(mask_bool)
    new_mask[ys[keep], xs[keep]] = True
    return new_mask


def fuse_object(
    obj_name: str,
    cam_ids,
    rgbs, depths_u16, Ks, Ts, Hs, Ws,
    mask_obj_dir: Path,
    voxel_size: float,
    sdf_trunc: float,
    depth_scale_m: float,
    depth_trunc: float,
    mask_dilate_px: int,
    mask_erode_px: int,
    remove_plane: bool,
    plane_thresh_m: float,
):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    n_ok = 0
    for ci in cam_ids:
        mp = mask_obj_dir / f"{ci}_mask.png"
        if not mp.exists():
            print(f"  [SKIP] {ci}: no mask")
            continue
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = _apply_mask(mask, mask_dilate_px, mask_erode_px)
        mask_bool = mask >= 128

        depth_masked = depths_u16[ci].copy()
        depth_masked[~mask_bool] = 0
        valid_n = int((depth_masked > 0).sum())
        if valid_n == 0:
            print(f"  [SKIP] {ci}: 마스크 안 depth==0")
            continue

        if remove_plane:
            depth_m = depth_masked.astype(np.float32) * depth_scale_m
            kept = _remove_dominant_plane(depth_m, mask_bool, Ks[ci], plane_thresh_m)
            removed = int(mask_bool.sum() - kept.sum())
            depth_masked[~kept] = 0
            valid_n = int((depth_masked > 0).sum())
            print(f"  [{ci}] plane RANSAC removed {removed} px (book platform), keep={valid_n}")

        color_img = o3d.geometry.Image(rgbs[ci].astype(np.uint8))
        depth_img = o3d.geometry.Image(depth_masked.astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1.0 / depth_scale_m,  # mm→m이면 1000
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=Ws[ci], height=Hs[ci],
            fx=Ks[ci][0, 0], fy=Ks[ci][1, 1],
            cx=Ks[ci][0, 2], cy=Ks[ci][1, 2],
        )
        extrinsic = np.linalg.inv(Ts[ci])  # T_cam→world  →  T_world→cam (Open3D 규약)
        volume.integrate(rgbd, intrinsic, extrinsic)
        n_ok += 1
        print(f"  [{ci}] integrated  valid_depth_px={valid_n}")

    if n_ok == 0:
        return None

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def save_glb(mesh_o3d: o3d.geometry.TriangleMesh, out_path: Path):
    verts = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    vc = np.asarray(mesh_o3d.vertex_colors)
    if len(vc) == len(verts):
        vc_u8 = (np.clip(vc, 0, 1) * 255).astype(np.uint8)
        vc_rgba = np.hstack([vc_u8, np.full((len(vc_u8), 1), 255, dtype=np.uint8)])
    else:
        vc_rgba = None
    tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vc_rgba, process=False)
    tri.export(str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./capture_obj")
    ap.add_argument("--mask_dir", default="./masks")
    ap.add_argument("--out_dir",  default="./outputs_tsdf")
    ap.add_argument("--voxel_size", type=float, default=0.002,
                    help="TSDF voxel 한 변 (m). 객체가 1cm 이하면 0.001, 일반은 0.002")
    ap.add_argument("--sdf_trunc", type=float, default=0.01,
                    help="SDF truncation 거리 (m). 보통 voxel_size의 5배")
    ap.add_argument("--depth_scale", type=float, default=0.001,
                    help="depth unit → m (D415 uint16 mm이면 0.001)")
    ap.add_argument("--depth_trunc", type=float, default=1.5,
                    help="이 거리 이상은 TSDF에 안 넣음 (m)")
    ap.add_argument("--mask_dilate_px", type=int, default=0,
                    help="마스크 부풀리기 (책상 침범 위험). 보통 0")
    ap.add_argument("--mask_erode_px", type=int, default=3,
                    help="마스크 안쪽으로 erode (객체 경계의 책상/배경 제거). 보통 2-5")
    ap.add_argument("--remove_plane", action="store_true",
                    help="마스크 안 RANSAC 평면(=책상 일부) 자동 제거")
    ap.add_argument("--plane_thresh_m", type=float, default=0.003,
                    help="RANSAC 평면 inlier 거리 (m). D415 noise 고려 3-5mm")
    ap.add_argument("--simplify_target_faces", type=int, default=0,
                    help="0=원본 유지. 줄이려면 5000~20000 권장")
    ap.add_argument("--clean_mesh", action="store_true",
                    help="작은 connected component 제거 + 비매끄러운 부분 정리")
    ap.add_argument("--smooth_iter", type=int, default=0,
                    help="Laplacian smoothing 반복 횟수 (mesh 노이즈 완화)")
    args = ap.parse_args()

    data = Path(args.data_dir)
    mask_root = Path(args.mask_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cam_ids = [p.stem.replace("_rgb", "") for p in sorted(data.glob("cam*_rgb.png"))]
    if not cam_ids:
        raise FileNotFoundError(f"No cam*_rgb.png in {data}")
    print(f"[INFO] cams: {cam_ids}")

    rgbs, depths_u16, Ks, Ts, Hs, Ws = {}, {}, {}, {}, {}, {}
    for ci in cam_ids:
        bgr = cv2.imread(str(data / f"{ci}_rgb.png"))
        rgbs[ci] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depths_u16[ci] = cv2.imread(str(data / f"{ci}_depth.png"), cv2.IMREAD_UNCHANGED)
        Ks[ci] = load_K(data / f"{ci}_K.txt")
        Ts[ci] = load_T(data / f"{ci}_T_cam_to_world.txt")
        Hs[ci], Ws[ci] = rgbs[ci].shape[:2]

    obj_dirs = sorted([d for d in mask_root.iterdir() if d.is_dir()])
    if not obj_dirs:
        raise FileNotFoundError(f"No object dirs in {mask_root}")
    print(f"[INFO] objects: {[d.name for d in obj_dirs]}")
    print(f"[INFO] voxel={args.voxel_size*1000:.1f}mm  sdf_trunc={args.sdf_trunc*1000:.1f}mm  "
          f"depth_trunc={args.depth_trunc:.2f}m  mask_dilate={args.mask_dilate_px}px")

    for obj_dir in obj_dirs:
        name = obj_dir.name
        print(f"\n=== Fusing {name} ===")
        mesh = fuse_object(
            obj_name=name, cam_ids=cam_ids,
            rgbs=rgbs, depths_u16=depths_u16, Ks=Ks, Ts=Ts, Hs=Hs, Ws=Ws,
            mask_obj_dir=obj_dir,
            voxel_size=args.voxel_size,
            sdf_trunc=args.sdf_trunc,
            depth_scale_m=args.depth_scale,
            depth_trunc=args.depth_trunc,
            mask_dilate_px=args.mask_dilate_px,
            mask_erode_px=args.mask_erode_px,
            remove_plane=args.remove_plane,
            plane_thresh_m=args.plane_thresh_m,
        )
        if mesh is None:
            print(f"  [WARN] no view integrated, skip")
            continue

        n_v, n_t = len(mesh.vertices), len(mesh.triangles)
        bbox = mesh.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        print(f"  mesh: {n_v} verts, {n_t} tris  bbox={ext[0]*1000:.1f}x{ext[1]*1000:.1f}x{ext[2]*1000:.1f}mm")

        if args.clean_mesh:
            tri_clusters, n_per_cluster, _ = mesh.cluster_connected_triangles()
            tri_clusters = np.asarray(tri_clusters)
            n_per_cluster = np.asarray(n_per_cluster)
            if len(n_per_cluster) > 1:
                largest = int(np.argmax(n_per_cluster))
                keep = tri_clusters == largest
                mesh.remove_triangles_by_mask(~keep)
                mesh.remove_unreferenced_vertices()
                print(f"  cleaned: kept largest cluster ({n_per_cluster[largest]}/{n_per_cluster.sum()} tris)")

        if args.simplify_target_faces > 0 and len(mesh.triangles) > args.simplify_target_faces:
            mesh = mesh.simplify_quadric_decimation(args.simplify_target_faces)
            mesh.compute_vertex_normals()
            print(f"  simplified to {len(mesh.triangles)} tris")

        if args.smooth_iter > 0:
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=args.smooth_iter)
            mesh.compute_vertex_normals()
            print(f"  laplacian smoothed x{args.smooth_iter}")

        out_path = out_root / f"{name}.glb"
        save_glb(mesh, out_path)
        # PLY도 함께 저장 (디버깅 시각화용)
        o3d.io.write_triangle_mesh(str(out_root / f"{name}.ply"), mesh)
        print(f"  [SAVE] {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
