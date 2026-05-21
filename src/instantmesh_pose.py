#!/usr/bin/env python3
"""
Multi-cam Calibration기반 InstantMesh 파이프라인

전제:
1. 3~4대 RGB-D 카메라 동시 촬영 완료
2. 각 카메라의 intrinsics K, extrinsic T_cam_to_world 보유
3. extrinsic은 반드시 camera frame -> world/base frame 변환 행렬이어야 함
4. depth 단위는 meter로 정규화되어 있어야 함. RealSense depth가 uint16 mm이면 depth_scale 적용 필요
5. SAM2 / InstantMesh는 별도 설치되어 있고, 이 스크립트에서는 호출부를 분리함

출력:
- object_input.png        : InstantMesh 입력용 RGBA 이미지
- object_cloud_clean.ply  : 멀티뷰 마스크 기반 metric point cloud
- object_scaled.glb       : metric scale 적용된 GLB
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import LocalOutlierFactor


# ============================================================
# Data structure
# ============================================================

@dataclass
class CameraPacket:
    cam_id: str
    rgb: np.ndarray             # H x W x 3, uint8, RGB order
    depth: np.ndarray           # H x W, float32, meter
    K: np.ndarray               # 3 x 3
    T_cam_to_world: np.ndarray  # 4 x 4


# ============================================================
# I/O utilities
# ============================================================

def load_matrix(path: str | Path, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"Matrix shape mismatch: {path}, expected={shape}, got={arr.shape}")
    return arr


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth(path: str | Path, depth_scale: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Depth image not found: {path}")
    return depth_raw.astype(np.float32) * depth_scale


def load_cameras_from_folder(data_dir: str | Path, depth_scale: float = 0.001) -> Dict[str, CameraPacket]:
    """
    예상 폴더 구조:

    data_dir/
      cam0_rgb.png
      cam0_depth.png
      cam0_K.txt
      cam0_T_cam_to_world.txt
      cam1_rgb.png
      cam1_depth.png
      cam1_K.txt
      cam1_T_cam_to_world.txt
      ...

    depth_scale:
      RealSense uint16 depth가 mm 단위이면 0.001
      이미 meter float depth를 저장했다면 1.0
    """
    data_dir = Path(data_dir)
    cameras: Dict[str, CameraPacket] = {}

    rgb_files = sorted(data_dir.glob("cam*_rgb.*"))
    if not rgb_files:
        raise FileNotFoundError(f"No cam*_rgb.* files found in {data_dir}")

    for rgb_path in rgb_files:
        cam_id = rgb_path.stem.replace("_rgb", "")
        depth_path_candidates = list(data_dir.glob(f"{cam_id}_depth.*"))
        if not depth_path_candidates:
            raise FileNotFoundError(f"Depth file missing for {cam_id}")
        depth_path = depth_path_candidates[0]

        K_path = data_dir / f"{cam_id}_K.txt"
        T_path = data_dir / f"{cam_id}_T_cam_to_world.txt"

        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path, depth_scale=depth_scale)
        K = load_matrix(K_path, (3, 3))
        T = load_matrix(T_path, (4, 4))

        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(f"RGB/depth size mismatch in {cam_id}: rgb={rgb.shape}, depth={depth.shape}")

        cameras[cam_id] = CameraPacket(
            cam_id=cam_id,
            rgb=rgb,
            depth=depth,
            K=K,
            T_cam_to_world=T,
        )

    return cameras


# ============================================================
# SAM2 segmentation adapter
# ============================================================

def sam2_segment_placeholder(rgb: np.ndarray, prompt: Optional[dict] = None) -> np.ndarray:
    """
    여기를 실제 SAM2 segmentation 함수로 교체.

    권장 방식:
    1. 대표 뷰에서 클릭/box prompt로 물체 mask 생성
    2. multi-view에서는 아래 둘 중 하나 사용
       - 각 뷰에서 독립 SAM2/SAM1 segmentation
       - 대표 뷰 mask를 3D로 올린 뒤 다른 뷰에 project하여 prompt 생성

    반환:
      mask: H x W bool
    """
    raise NotImplementedError(
        "sam2_segment_placeholder를 실제 SAM2 호출 함수로 교체하거나 "
        "--mask_dir 옵션으로 미리 생성된 mask를 입력하세요."
    )


def load_masks(mask_dir: str | Path, cameras: Dict[str, CameraPacket]) -> Dict[str, np.ndarray]:
    """
    예상:
      mask_dir/cam0_mask.png
      mask_dir/cam1_mask.png
      ...

    mask는 0/255 PNG 또는 bool-like 이미지 가능.
    """
    mask_dir = Path(mask_dir)
    masks: Dict[str, np.ndarray] = {}

    for cam_id, cam in cameras.items():
        mask_path = mask_dir / f"{cam_id}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if mask.shape != cam.depth.shape:
            raise ValueError(f"Mask/depth size mismatch in {cam_id}: mask={mask.shape}, depth={cam.depth.shape}")
        masks[cam_id] = mask > 0

    return masks


def load_masks_per_object(
    mask_dir: str | Path,
    cameras: Dict[str, CameraPacket],
    obj_ids: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    다중 인스턴스 마스크 로더.

    지원 파일 패턴 (둘 중 하나):
      1) mask_dir/{cam_id}_obj{obj_id}_mask.png   ← 다중 물체
      2) mask_dir/{cam_id}_mask.png               ← 단일 물체 (obj_id="0" 처리)

    멀티뷰 인스턴스 매칭(같은 obj_id가 카메라 간 같은 물체임을 보장)은 호출자(예: SAM2
    multi-view propagation) 책임이다. 이 함수는 동일한 obj_id를 같은 물체로 가정한다.

    obj_ids 미지정시 mask_dir에서 검출된 모든 obj_id 사용.

    반환: {obj_id: {cam_id: bool mask}}
    """
    mask_dir = Path(mask_dir)
    masks_by_obj: Dict[str, Dict[str, np.ndarray]] = {}

    pat = re.compile(r"^(?P<cam>cam[^_]+)_obj(?P<obj>[^_]+)_mask\.[A-Za-z0-9]+$")
    multi_files = []
    for fp in mask_dir.iterdir():
        if not fp.is_file():
            continue
        m = pat.match(fp.name)
        if m:
            multi_files.append((m.group("cam"), m.group("obj"), fp))

    if multi_files:
        for cam_id, obj_id, fp in multi_files:
            if cam_id not in cameras:
                print(f"[WARN] mask file references unknown cam_id={cam_id}: {fp.name}")
                continue
            if obj_ids is not None and obj_id not in obj_ids:
                continue
            mk = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if mk is None:
                raise FileNotFoundError(f"Mask not readable: {fp}")
            cam = cameras[cam_id]
            if mk.shape != cam.depth.shape:
                raise ValueError(
                    f"Mask/depth size mismatch in {cam_id}/obj{obj_id}: "
                    f"mask={mk.shape}, depth={cam.depth.shape}"
                )
            masks_by_obj.setdefault(obj_id, {})[cam_id] = mk > 0
    else:
        # 단일 물체 fallback: obj_id="0"으로 묶는다.
        if obj_ids is not None and "0" not in obj_ids:
            raise FileNotFoundError(
                f"No cam*_obj*_mask.* files in {mask_dir} and requested obj_ids={obj_ids} "
                f"do not include the fallback id '0'."
            )
        single = load_masks(mask_dir, cameras)
        masks_by_obj["0"] = single

    if not masks_by_obj:
        raise FileNotFoundError(
            f"No usable object masks found in {mask_dir} (obj_ids filter={obj_ids})."
        )

    return masks_by_obj


# ============================================================
# Point cloud construction
# ============================================================

def depth_mask_to_world_points(
    depth: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    min_depth: float = 0.05,
    max_depth: float = 2.0,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    depth + mask 영역을 camera frame point cloud로 만든 뒤 world frame으로 변환.

    RealSense 기준 camera frame:
      x: right
      y: down
      z: forward

    T_cam_to_world가 이 camera frame 정의에 맞게 calibration되어 있어야 함.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    sampled_mask = np.zeros_like(mask, dtype=bool)
    sampled_mask[::stride, ::stride] = mask[::stride, ::stride]

    v, u = np.where(sampled_mask)
    z = depth[v, u]

    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    u, v, z = u[valid], v[valid], z[valid]

    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float64), None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy

    pts_cam_h = np.stack([x, y, z, np.ones_like(z)], axis=1)
    pts_world = (T_cam_to_world @ pts_cam_h.T).T[:, :3]

    colors = None
    if rgb is not None:
        colors = rgb[v, u].astype(np.float64) / 255.0

    return pts_world, colors


def merge_multiview_cloud(
    cameras: Dict[str, CameraPacket],
    masks: Dict[str, np.ndarray],
    min_depth: float,
    max_depth: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    all_points = []
    all_colors = []

    for cam_id, cam in cameras.items():
        pts, colors = depth_mask_to_world_points(
            depth=cam.depth,
            mask=masks[cam_id],
            K=cam.K,
            T_cam_to_world=cam.T_cam_to_world,
            rgb=cam.rgb,
            min_depth=min_depth,
            max_depth=max_depth,
            stride=stride,
        )
        print(f"[{cam_id}] valid masked points: {len(pts)}")
        if len(pts) > 0:
            all_points.append(pts)
            if colors is not None:
                all_colors.append(colors)

    if not all_points:
        raise RuntimeError("No valid masked depth points from any camera.")

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0) if all_colors else np.zeros_like(points)
    return points, colors


# ============================================================
# Cloud filtering and metric bbox
# ============================================================

def filter_cloud_open3d(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    voxel_size: float = 0.002,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
    radius: float = 0.01,
    min_points: int = 8,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    권장 필터 순서:
    1. voxel downsample
    2. statistical outlier removal
    3. radius outlier removal
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if len(pcd.points) >= nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    if len(pcd.points) >= min_points:
        pcd, _ = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

    clean_points = np.asarray(pcd.points)
    clean_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return clean_points, clean_colors


def filter_cloud_lof(points: np.ndarray, n_neighbors: int = 30, contamination: float = 0.05) -> np.ndarray:
    if len(points) < n_neighbors + 1:
        return points
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(points)
    return points[labels == 1]


def estimate_metric_bbox(
    points: np.ndarray,
    use_oriented_bbox: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    반환:
      center_world: 3,
      extents: 3, meter
      R_bbox_to_world: 3 x 3

    use_oriented_bbox=True:
      물체가 world 축과 비스듬하면 oriented bounding box 기준으로 크기 추정.
      단, 얇거나 대칭 물체는 OBB 축이 불안정할 수 있음.
    """
    if len(points) < 10:
        raise RuntimeError("Too few points to estimate bbox.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if use_oriented_bbox:
        bbox = pcd.get_oriented_bounding_box(robust=True)
        center = np.asarray(bbox.center)
        extents = np.asarray(bbox.extent)
        R_bbox_to_world = np.asarray(bbox.R)
    else:
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        extents = bbox_max - bbox_min
        R_bbox_to_world = np.eye(3)

    return center, extents, R_bbox_to_world


def save_cloud_ply(path: str | Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)


# ============================================================
# InstantMesh input view selection
# ============================================================

def mask_bbox_area(mask: np.ndarray) -> int:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0
    return int((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))


def choose_best_view(
    cameras: Dict[str, CameraPacket],
    masks: Dict[str, np.ndarray],
    min_mask_pixels: int = 500,
) -> str:
    """
    기본 기준:
      - mask pixel 수가 크고
      - bbox가 이미지 밖에 잘리지 않은 뷰 선호

    단순 mask.sum()만 쓰면 카메라에 너무 가까운 oblique view가 선택될 수 있음.
    필요하면 아래 score에 view angle term을 추가할 것.
    """
    best_cam = None
    best_score = -np.inf

    for cam_id, cam in cameras.items():
        mask = masks[cam_id]
        area = int(mask.sum())
        if area < min_mask_pixels:
            print(f"[{cam_id}] skip: mask too small: {area}")
            continue

        ys, xs = np.where(mask)
        h, w = mask.shape
        margin = min(xs.min(), ys.min(), w - 1 - xs.max(), h - 1 - ys.max())
        clipped_penalty = 0.5 if margin < 5 else 1.0

        # compactness: mask area / bbox area
        bb_area = max(mask_bbox_area(mask), 1)
        compactness = area / bb_area

        score = area * clipped_penalty * (0.5 + compactness)
        print(f"[{cam_id}] view score={score:.1f}, mask_pixels={area}, compactness={compactness:.3f}")

        if score > best_score:
            best_score = score
            best_cam = cam_id

    if best_cam is None:
        raise RuntimeError("No valid camera view for InstantMesh input.")

    return best_cam


def save_rgba_input(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_path: str | Path,
    crop: bool = True,
    padding: int = 30,
    output_size: Optional[int] = 512,
) -> None:
    """
    InstantMesh 입력용 RGBA 생성.

    권장:
      - crop=True
      - alpha는 mask
      - background는 흰색 또는 투명 유지
    """
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)

    rgba = np.dstack([rgb, mask_u8])

    if crop:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise RuntimeError("Cannot crop empty mask.")
        h, w = mask.shape
        x0 = max(xs.min() - padding, 0)
        x1 = min(xs.max() + padding + 1, w)
        y0 = max(ys.min() - padding, 0)
        y1 = min(ys.max() + padding + 1, h)
        rgba = rgba[y0:y1, x0:x1]

    if output_size is not None:
        # square canvas에 중앙 배치
        ih, iw = rgba.shape[:2]
        scale = output_size / max(ih, iw)
        nh, nw = int(round(ih * scale)), int(round(iw * scale))
        resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        y0 = (output_size - nh) // 2
        x0 = (output_size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        rgba = canvas

    Image.fromarray(rgba).save(out_path)
    print(f"Saved InstantMesh RGBA input: {out_path}")


# ============================================================
# InstantMesh execution
# ============================================================

def run_instantmesh(
    instantmesh_root: str | Path,
    config_path: str | Path,
    input_png: str | Path,
    output_dir: str | Path,
    python_bin: str = "python",
    no_rembg: bool = True,
    export_texmap: bool = True,
) -> None:
    """
    InstantMesh repo의 run.py를 subprocess로 실행.

    예:
      python run.py configs/instant-mesh-large.yaml object_input.png --no_rembg --export_texmap
    """
    instantmesh_root = Path(instantmesh_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        "run.py",
        str(config_path),
        str(input_png),
        "--output_path",
        str(output_dir),
    ]
    if no_rembg:
        cmd.append("--no_rembg")
    if export_texmap:
        cmd.append("--export_texmap")

    print("Running InstantMesh:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(instantmesh_root), check=True)


# ============================================================
# Mesh scaling and alignment
# ============================================================

def load_mesh_any(path: str | Path) -> trimesh.Trimesh:
    mesh_or_scene = trimesh.load(str(path), force="scene")
    if isinstance(mesh_or_scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh_or_scene.geometry.values()))
    else:
        mesh = mesh_or_scene
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {type(mesh)}")
    return mesh


def scale_mesh_to_metric_bbox(
    mesh_path: str | Path,
    bbox_extents_m: np.ndarray,
    out_glb_path: str | Path,
    center_world: Optional[np.ndarray] = None,
    R_bbox_to_world: Optional[np.ndarray] = None,
    scale_mode: str = "median",
    apply_world_pose: bool = False,
) -> float:
    """
    InstantMesh 결과 mesh를 metric bbox 크기에 맞게 scale.

    scale_mode:
      - mean   : x,y,z 축 scale 평균
      - median : 세 축 scale median. outlier 축에 덜 민감하므로 기본 추천
      - max    : 최대 축 기준

    apply_world_pose=True이면 mesh를 world bbox center와 orientation에 배치.
    Isaac Sim에서 world/base 위치까지 같이 쓰려면 True를 고려.
    단, InstantMesh mesh coordinate의 front/up 정의가 실제 물체와 다를 수 있어
    orientation은 후처리 검증이 필요함.
    """
    mesh = load_mesh_any(mesh_path)
    mesh_extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64)

    eps = 1e-8
    valid = mesh_extents > eps
    if valid.sum() == 0:
        raise RuntimeError("Mesh bbox has zero extent.")

    scale_factors = bbox_extents_m[valid] / mesh_extents[valid]

    if scale_mode == "mean":
        scale = float(np.mean(scale_factors))
    elif scale_mode == "median":
        scale = float(np.median(scale_factors))
    elif scale_mode == "max":
        scale = float(np.max(scale_factors))
    else:
        raise ValueError(f"Unknown scale_mode: {scale_mode}")

    mesh.apply_scale(scale)

    # mesh 중심을 원점으로 정렬
    mesh_center = mesh.bounding_box.centroid
    mesh.apply_translation(-mesh_center)

    if apply_world_pose:
        if center_world is None:
            raise ValueError("center_world is required when apply_world_pose=True")
        if R_bbox_to_world is not None:
            T = np.eye(4)
            T[:3, :3] = R_bbox_to_world
            mesh.apply_transform(T)
        mesh.apply_translation(center_world)

    mesh.export(str(out_glb_path))
    print(f"Saved scaled GLB: {out_glb_path}")
    print(f"mesh_extents_before={mesh_extents}, bbox_extents_m={bbox_extents_m}, scale={scale:.6f}")
    return scale


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing cam*_rgb, cam*_depth, cam*_K, cam*_T_cam_to_world")
    parser.add_argument("--mask_dir", required=True, help="Folder containing cam*_mask.png. Use precomputed SAM2 masks.")
    parser.add_argument("--out_dir", default="outputs_multicam_instantmesh")

    parser.add_argument("--depth_scale", type=float, default=0.001, help="uint16 mm depth -> meter: 0.001")
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--max_depth", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--voxel_size", type=float, default=0.002, help="meter. 0.002 = 2 mm")
    parser.add_argument("--use_oriented_bbox", action="store_true")

    parser.add_argument("--obj_ids", default="",
                        help="Comma-separated object ids to process (e.g. '1,3,5'). "
                             "Empty = all detected from cam*_obj*_mask filenames. "
                             "단일 물체(cam*_mask.png)인 경우 '0' 사용.")

    parser.add_argument("--run_instantmesh", action="store_true")
    parser.add_argument("--instantmesh_root", default="InstantMesh")
    parser.add_argument("--instantmesh_config", default="configs/instant-mesh-large.yaml")
    parser.add_argument("--instantmesh_python", default="python")
    parser.add_argument("--instantmesh_mesh", default="",
                        help="Pre-generated InstantMesh 결과 경로. "
                             "단일 파일이면 단일 물체 모드에만 적용. "
                             "디렉토리면 그 안의 obj{id}.glb/.obj/.ply 를 obj id별로 매칭.")

    parser.add_argument("--scale_mode", choices=["mean", "median", "max"], default="median")
    parser.add_argument("--apply_world_pose", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load calibrated multi-cam RGB-D packets
    cameras = load_cameras_from_folder(args.data_dir, depth_scale=args.depth_scale)
    print(f"Loaded cameras: {list(cameras.keys())}")

    # 2. Load SAM/SAM2 masks per object
    selected_obj_ids = [s.strip() for s in args.obj_ids.split(",") if s.strip()] or None
    masks_by_obj = load_masks_per_object(args.mask_dir, cameras, obj_ids=selected_obj_ids)
    print(f"Objects to process: {sorted(masks_by_obj.keys())}")

    instantmesh_mesh_arg = Path(args.instantmesh_mesh) if args.instantmesh_mesh else None
    instantmesh_mesh_is_dir = bool(instantmesh_mesh_arg and instantmesh_mesh_arg.is_dir())

    results_summary: Dict[str, dict] = {}

    for obj_id in sorted(masks_by_obj.keys()):
        per_cam_masks = masks_by_obj[obj_id]
        obj_tag = f"obj{obj_id}"
        print(f"\n=== Processing {obj_tag} (cameras: {sorted(per_cam_masks.keys())}) ===")

        obj_cams = {cid: cameras[cid] for cid in per_cam_masks.keys()}

        # 3. Merge masked depth points into world frame
        points, colors = merge_multiview_cloud(
            cameras=obj_cams,
            masks=per_cam_masks,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            stride=args.stride,
        )
        print(f"[{obj_tag}] Merged raw cloud points: {len(points)}")

        # 4. Filter cloud and estimate metric bbox
        clean_points, _ = filter_cloud_open3d(
            points,
            colors,
            voxel_size=args.voxel_size,
            nb_neighbors=30,
            std_ratio=2.0,
            radius=max(args.voxel_size * 4.0, 0.006),
            min_points=8,
        )
        print(f"[{obj_tag}] Clean cloud after Open3D filter: {len(clean_points)}")

        clean_points_lof = filter_cloud_lof(clean_points, n_neighbors=30, contamination=0.03)
        print(f"[{obj_tag}] Clean cloud after LOF: {len(clean_points_lof)}")

        if len(clean_points_lof) < 10:
            print(f"[{obj_tag}] Skip: too few points after filtering.")
            continue

        save_cloud_ply(out_dir / f"{obj_tag}_cloud_clean.ply", clean_points_lof, None)

        center_world, bbox_extents_m, R_bbox_to_world = estimate_metric_bbox(
            clean_points_lof,
            use_oriented_bbox=args.use_oriented_bbox,
        )

        bbox_info = {
            "obj_id": obj_id,
            "center_world_m": center_world.tolist(),
            "bbox_extents_m": bbox_extents_m.tolist(),
            "R_bbox_to_world": R_bbox_to_world.tolist(),
            "use_oriented_bbox": bool(args.use_oriented_bbox),
        }
        with open(out_dir / f"{obj_tag}_bbox_metric.json", "w", encoding="utf-8") as f:
            json.dump(bbox_info, f, indent=2)
        print(f"[{obj_tag}] metric bbox:")
        print(json.dumps(bbox_info, indent=2))

        # 5. Choose best view and save RGBA for InstantMesh
        best_cam = choose_best_view(obj_cams, per_cam_masks)
        print(f"[{obj_tag}] Best view for InstantMesh: {best_cam}")

        input_png = out_dir / f"{obj_tag}_input.png"
        save_rgba_input(
            rgb=obj_cams[best_cam].rgb,
            mask=per_cam_masks[best_cam],
            out_path=input_png,
            crop=True,
            padding=30,
            output_size=512,
        )

        # 6. Run InstantMesh if requested (per object)
        if args.run_instantmesh:
            im_out = out_dir / "instantmesh_output" / obj_tag
            run_instantmesh(
                instantmesh_root=args.instantmesh_root,
                config_path=args.instantmesh_config,
                input_png=input_png.resolve(),
                output_dir=im_out,
                python_bin=args.instantmesh_python,
                no_rembg=True,
                export_texmap=True,
            )

        # 7. Scale InstantMesh mesh to metric GLB
        mesh_path: Optional[Path] = None
        if instantmesh_mesh_arg is not None:
            if instantmesh_mesh_is_dir:
                for ext in (".glb", ".obj", ".ply"):
                    cand = instantmesh_mesh_arg / f"{obj_tag}{ext}"
                    if cand.exists():
                        mesh_path = cand
                        break
                if mesh_path is None:
                    print(f"[{obj_tag}] No mesh file found in {instantmesh_mesh_arg} for this obj.")
            elif len(masks_by_obj) == 1:
                # 단일 물체 모드 + 단일 파일 인자: 그대로 사용
                if instantmesh_mesh_arg.exists():
                    mesh_path = instantmesh_mesh_arg
                else:
                    raise FileNotFoundError(f"InstantMesh mesh not found: {instantmesh_mesh_arg}")
            else:
                print(f"[{obj_tag}] --instantmesh_mesh 는 다중 물체에서 디렉토리여야 합니다. skip.")

        if mesh_path is not None:
            out_glb = out_dir / f"{obj_tag}_scaled.glb"
            scale = scale_mesh_to_metric_bbox(
                mesh_path=mesh_path,
                bbox_extents_m=bbox_extents_m,
                out_glb_path=out_glb,
                center_world=center_world,
                R_bbox_to_world=R_bbox_to_world,
                scale_mode=args.scale_mode,
                apply_world_pose=args.apply_world_pose,
            )
            results_summary[obj_id] = {
                "scaled_glb": str(out_glb),
                "scale": scale,
                "bbox_extents_m": bbox_extents_m.tolist(),
                "center_world_m": center_world.tolist(),
            }
        else:
            print(f"[{obj_tag}] Skip mesh scaling (no mesh provided for this object).")

    if results_summary:
        with open(out_dir / "objects_summary.json", "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSaved per-object summary: {out_dir / 'objects_summary.json'}")


if __name__ == "__main__":
    main()
