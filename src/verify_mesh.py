#!/usr/bin/env python3
"""
Scaled mesh 정확도 검증.

[실행 명령어]
python src/verify_mesh.py --outputs_dir ./outputs --data_dir ./capture_obj --mask_dir ./masks


각 객체 폴더(outputs/<obj>/)에 대해:
  1) bbox dimensions 출력 (자로 잰 실측과 비교 가능)
  2) mesh ↔ point cloud Chamfer distance
  3) 카메라별 silhouette IoU (mesh를 cam pose로 렌더 → 마스크 비교)
  4) mesh+cloud overlay PLY 저장 (Open3D/MeshLab로 시각 확인)
  5) 카메라별 projection overlay PNG 저장 (RGB에 mesh silhouette 빨강 오버레이)

사용 예:
  python src/verify_mesh.py \
    --outputs_dir ./outputs \
    --data_dir ./capture_obj \
    --mask_dir ./masks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree


def load_K(p: Path):
    return np.loadtxt(p, dtype=np.float64).reshape(3, 3)


def load_T(p: Path):
    return np.loadtxt(p, dtype=np.float64).reshape(4, 4)


def _find_newest(obj_out_dir: Path, pattern: str):
    """obj_out_dir 아래(재귀)에서 pattern과 매칭되는 가장 최신 파일."""
    cand = list(obj_out_dir.rglob(pattern))
    if not cand:
        return None
    return max(cand, key=lambda p: p.stat().st_mtime)


def find_scaled_glb(obj_out_dir: Path):
    return _find_newest(obj_out_dir, "*_scaled.glb")


def find_cloud_ply(obj_out_dir: Path):
    return _find_newest(obj_out_dir, "*_cloud_clean.ply")


def find_bbox_json(obj_out_dir: Path):
    return _find_newest(obj_out_dir, "*_bbox_metric.json")


def find_scale_report(obj_out_dir: Path):
    return _find_newest(obj_out_dir, "*_scale_report.json")


def load_mesh_as_trimesh(glb_path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene, trimesh.Scene):
        geoms = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        return trimesh.util.concatenate(tuple(geoms)) if geoms else None
    return scene


def chamfer_mesh_to_cloud(mesh: trimesh.Trimesh, cloud_pts: np.ndarray,
                          mesh_sample: int = 30000, cloud_sample: int = 30000):
    """mesh가 world 좌표계에 있고 cloud도 world 좌표계라는 가정."""
    rng = np.random.default_rng(0)
    pts_m, _ = trimesh.sample.sample_surface(mesh, mesh_sample)
    pts_m = np.asarray(pts_m)
    if len(cloud_pts) > cloud_sample:
        idx = rng.choice(len(cloud_pts), cloud_sample, replace=False)
        pts_c = cloud_pts[idx]
    else:
        pts_c = cloud_pts
    tree_c = cKDTree(pts_c)
    tree_m = cKDTree(pts_m)
    d_m2c, _ = tree_c.query(pts_m, k=1, workers=-1)
    d_c2m, _ = tree_m.query(pts_c, k=1, workers=-1)
    return {
        "m2c_median": float(np.median(d_m2c)),
        "m2c_mean": float(np.mean(d_m2c)),
        "c2m_median": float(np.median(d_c2m)),
        "c2m_mean": float(np.mean(d_c2m)),
        "chamfer_symmetric_median": float(0.5 * (np.median(d_m2c) + np.median(d_c2m))),
    }


def project_mesh_silhouette(mesh: trimesh.Trimesh, K: np.ndarray, T_cam_to_world: np.ndarray,
                             H: int, W: int) -> np.ndarray:
    """mesh vertex를 cam frame으로 변환 → K로 픽셀 투영 → triangle을 채워 silhouette mask 생성."""
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    Vh = np.hstack([verts, np.ones((len(verts), 1))])
    Vc = (T_world_to_cam @ Vh.T).T[:, :3]  # N x 3 in cam frame
    z = Vc[:, 2]

    # invalid (behind cam) 처리: pixel을 화면 밖으로
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * Vc[:, 0] / np.where(z > 1e-6, z, 1e-6) + cx
    v = fy * Vc[:, 1] / np.where(z > 1e-6, z, 1e-6) + cy
    u[z <= 1e-6] = -1e6
    v[z <= 1e-6] = -1e6

    silh = np.zeros((H, W), dtype=np.uint8)
    pts2d = np.stack([u, v], axis=1).astype(np.int32)
    for tri in faces:
        p = pts2d[tri]
        cv2.fillConvexPoly(silh, p, 255)
    return silh > 0


def silhouette_iou(mask_gt: np.ndarray, mask_pred: np.ndarray):
    gt = mask_gt > 0
    pr = mask_pred > 0
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def overlay_silhouette(rgb: np.ndarray, mask_gt: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    """GT mask = green, predicted silhouette = red, intersection = yellow."""
    out = rgb.copy()
    gt = mask_gt > 0
    pr = mask_pred > 0
    # green for GT only
    out[gt & ~pr] = (0.5 * out[gt & ~pr] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
    # red for pred only
    out[pr & ~gt] = (0.5 * out[pr & ~gt] + 0.5 * np.array([200, 0, 0])).astype(np.uint8)
    # yellow for both
    out[gt & pr] = (0.5 * out[gt & pr] + 0.5 * np.array([200, 200, 0])).astype(np.uint8)
    return out


def verify_one_object(obj_dir: Path, data_dir: Path, mask_obj_dir: Path):
    print(f"\n========== {obj_dir.name} ==========")
    glb = find_scaled_glb(obj_dir)
    ply = find_cloud_ply(obj_dir)
    bbox_json = find_bbox_json(obj_dir)
    scale_report = find_scale_report(obj_dir)
    if glb is None or ply is None or bbox_json is None:
        print(f"[SKIP] missing files in {obj_dir}")
        return None

    mesh = load_mesh_as_trimesh(glb)
    if mesh is None:
        print(f"[SKIP] mesh load failed: {glb}")
        return None

    # --- bbox dimensions ---
    bbox_ext = mesh.bounding_box.extents  # axis-aligned (after centering)
    print(f"[1] mesh AABB extents (mm): "
          f"{bbox_ext[0]*1000:7.1f} x {bbox_ext[1]*1000:7.1f} x {bbox_ext[2]*1000:7.1f}")
    with open(bbox_json) as f:
        bbox_info = json.load(f)
    cloud_ext = np.array(bbox_info["bbox_extents_m"])
    print(f"    cloud bbox extents (mm): "
          f"{cloud_ext[0]*1000:7.1f} x {cloud_ext[1]*1000:7.1f} x {cloud_ext[2]*1000:7.1f}")

    # --- scale_report ---
    with open(scale_report) as f:
        sr = json.load(f)
    selected = sr.get("selected_candidate", {})
    icp = sr.get("sim3_icp_report") or {}
    print(f"[2] final_scale={sr.get('final_scale'):.6g}  "
          f"selected={selected.get('name')}  chamfer_score(scale_only)={selected.get('score_m'):.4f} m")
    if icp:
        print(f"    ICP refined_scale={icp.get('refined_scale'):.6g}  "
              f"median_nn_err={icp.get('median_nn_error_m'):.4f} m  "
              f"warning={icp.get('warning', 'OK')}")

    # --- Chamfer to point cloud (post-scaling, mesh이미 metric, but mesh는 centered at origin,
    # cloud는 world 좌표) ---
    # improved_instantmesh_pose는 export_scaled_mesh()에서 center_mesh=True로 mesh를 원점에 둠.
    # cloud는 world 좌표. 비교하려면 cloud도 자신의 centroid 빼서 정합 비교.
    cloud_pcd = o3d.io.read_point_cloud(str(ply))
    cloud_pts = np.asarray(cloud_pcd.points)
    if len(cloud_pts) < 50:
        print(f"[3] [SKIP] cloud too small: {len(cloud_pts)}")
        return None
    cloud_center = np.median(cloud_pts, axis=0)
    cloud_centered = cloud_pts - cloud_center
    cham = chamfer_mesh_to_cloud(mesh, cloud_centered)
    print(f"[3] Chamfer mesh-vs-cloud (median, after centering both to origin):")
    print(f"    mesh->cloud: {cham['m2c_median']*1000:6.2f} mm   "
          f"cloud->mesh: {cham['c2m_median']*1000:6.2f} mm   "
          f"symmetric: {cham['chamfer_symmetric_median']*1000:6.2f} mm")

    # --- per-cam silhouette IoU ---
    print(f"[4] Silhouette IoU (mesh projection vs GT mask):")
    cam_ids = sorted([p.stem.replace("_rgb", "") for p in data_dir.glob("cam*_rgb.png")])
    iou_list = []
    overlays_dir = obj_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)
    for ci in cam_ids:
        rgb_path = data_dir / f"{ci}_rgb.png"
        K_path = data_dir / f"{ci}_K.txt"
        T_path = data_dir / f"{ci}_T_cam_to_world.txt"
        mask_path = mask_obj_dir / f"{ci}_mask.png"
        if not (rgb_path.exists() and mask_path.exists()):
            print(f"    [SKIP] {ci}: missing rgb or mask")
            continue
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        K = load_K(K_path)
        T = load_T(T_path)
        mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        H, W = mask_gt.shape

        # mesh가 origin 중심이므로 cloud_center로 translate (실제 위치)
        mesh_world = mesh.copy()
        mesh_world.apply_translation(cloud_center)

        silh = project_mesh_silhouette(mesh_world, K, T, H, W)
        iou = silhouette_iou(mask_gt, silh.astype(np.uint8) * 255)
        iou_list.append(iou)
        print(f"    {ci}: IoU = {iou:.3f}")

        ov = overlay_silhouette(rgb, mask_gt, silh.astype(np.uint8) * 255)
        cv2.imwrite(str(overlays_dir / f"{ci}_overlay.png"),
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    iou_mean = float(np.mean(iou_list)) if iou_list else 0.0
    print(f"    => mean IoU = {iou_mean:.3f}")

    # --- mesh+cloud overlay PLY (시각화용) ---
    mesh_world = mesh.copy()
    mesh_world.apply_translation(cloud_center)
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh_world.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh_world.faces)),
    )
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.7, 0.7, 0.9])
    cloud_pcd.paint_uniform_color([1.0, 0.3, 0.3])
    o3d.io.write_triangle_mesh(str(obj_dir / "verify_mesh_in_world.ply"), mesh_o3d)
    o3d.io.write_point_cloud(str(obj_dir / "verify_cloud_in_world.ply"), cloud_pcd)

    return {
        "name": obj_dir.name,
        "mesh_bbox_mm": [float(x * 1000) for x in bbox_ext],
        "cloud_bbox_mm": [float(x * 1000) for x in cloud_ext],
        "final_scale": sr.get("final_scale"),
        "chamfer_symmetric_mm": cham["chamfer_symmetric_median"] * 1000,
        "icp_warning": icp.get("warning"),
        "silhouette_iou_per_cam": iou_list,
        "silhouette_iou_mean": iou_mean,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="./outputs")
    ap.add_argument("--data_dir", default="./capture_obj")
    ap.add_argument("--mask_dir", default="./masks")
    args = ap.parse_args()

    outputs = Path(args.outputs_dir)
    data = Path(args.data_dir)
    mask_root = Path(args.mask_dir)

    obj_dirs = sorted([d for d in outputs.iterdir() if d.is_dir()])
    summary = []
    for od in obj_dirs:
        mask_obj_dir = mask_root / od.name
        if not mask_obj_dir.exists():
            print(f"[SKIP] {od.name}: no mask dir {mask_obj_dir}")
            continue
        r = verify_one_object(od, data, mask_obj_dir)
        if r:
            summary.append(r)

    print("\n=========== SUMMARY ===========")
    print(f"{'obj':<8}{'final_scale':>12}{'chamfer(mm)':>13}{'mean_IoU':>10}  flag")
    for r in summary:
        flag = []
        if r["chamfer_symmetric_mm"] > 10:
            flag.append("HIGH_CHAMFER")
        if r["silhouette_iou_mean"] < 0.7:
            flag.append("LOW_IOU")
        if r["icp_warning"]:
            flag.append("ICP_REJECTED")
        print(f"{r['name']:<8}{r['final_scale']:>12.5g}"
              f"{r['chamfer_symmetric_mm']:>13.2f}{r['silhouette_iou_mean']:>10.3f}  "
              f"{', '.join(flag) if flag else 'OK'}")

    with open(outputs / "verify_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] {outputs / 'verify_summary.json'}")
    print(f"[INFO] overlay images: outputs/<obj>/overlays/cam*_overlay.png")
    print(f"[INFO] open with MeshLab: outputs/<obj>/verify_mesh_in_world.ply + verify_cloud_in_world.ply")


if __name__ == "__main__":
    main()
