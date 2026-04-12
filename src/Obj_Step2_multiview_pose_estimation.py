#!/usr/bin/env python3
"""
멀티뷰 RGB-D 기반 물체 포즈 추정 파이프라인.

목표:
  - Step2의 편한 입출력/자동 proposal 흐름은 유지
  - 포즈 정합 백엔드는 Obj_pose_estimator.py 스타일의
    multi-scale FPFH + ICP + multiview depth / coverage 스코어로 단순화
  - 입력으로는 calibrated multiview RGB-D와 unscaled reference GLB만 사용
  - Isaac Sim에 바로 넣을 수 있는 aligned GLB / pose npz / json 출력

예시:
  python3 src/Obj_Step2_multiview_pose_estimation.py \
    --data_dir src/data \
    --output_dir src/output/hole_run \
    --glb_path src/data/Hole.glb \
    --frame_id 000000
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")

try:
    import open3d as o3d
except ImportError:
    sys.exit("[ERROR] open3d 설치 필요: pip install open3d")

try:
    import trimesh
except ImportError:
    sys.exit("[ERROR] trimesh 설치 필요: pip install trimesh")

from scipy.spatial.transform import Rotation


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class CameraIntrinsics:
    K: np.ndarray
    D: np.ndarray
    depth_scale: float
    cam_id: int = 0


@dataclass
class CameraData:
    intrinsics: CameraIntrinsics
    color_img: np.ndarray
    depth_img: np.ndarray
    T_to_cam0: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class PoseResult:
    translation: np.ndarray
    rotation_matrix: np.ndarray
    euler_xyz_deg: np.ndarray
    quaternion_xyzw: np.ndarray
    transform_4x4: np.ndarray
    scale: float = 1.0
    score: float = 0.0
    depth_score: float = 0.0
    coverage: float = 0.0
    fitness: float = 0.0
    rmse: float = 0.0
    method: str = "Reference Matching"


@dataclass
class DetectionProposal:
    proposal_id: int
    roi: Tuple[int, int, int, int]
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    area_px: int
    mean_height_m: float
    depth_m: float
    score: float


@dataclass
class PreparedReference:
    pcd: o3d.geometry.PointCloud
    mesh: Optional[trimesh.Trimesh]
    original_center: np.ndarray
    original_longest_m: float
    prep_scale: float
    estimated_size_m: Optional[float] = None


# =============================================================================
# Data loading
# =============================================================================


class DataLoader:
    def __init__(
        self,
        data_dir: str,
        frame_id: str = "000003",
        extrinsics_dir: Optional[str] = None,
        glb_path: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.intrinsics_dir = self.data_dir / "_intrinsics"
        if not self.intrinsics_dir.exists():
            fallback_intrinsics_dir = self.data_dir.parent / "intrinsics"
            if fallback_intrinsics_dir.exists():
                print(f"  [INFO] _intrinsics 없음 -> fallback 사용: {fallback_intrinsics_dir}")
                self.intrinsics_dir = fallback_intrinsics_dir
        self.image_dir = self.data_dir / "object_capture"
        self.frame_id = frame_id

        if extrinsics_dir:
            self.extrinsics_dir = Path(extrinsics_dir)
        else:
            self.extrinsics_dir = self.data_dir / "cube_session_01" / "calib_out_cube"

        if glb_path:
            self.glb_path = Path(glb_path)
        else:
            self.glb_path = self.data_dir / "reference_knife.glb"

    def load_intrinsics(self, cam_id: int) -> CameraIntrinsics:
        npz_path = self.intrinsics_dir / f"cam{cam_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"캘리브레이션 파일 없음: {npz_path}")

        data = np.load(str(npz_path), allow_pickle=True)
        K = self._get_key(data, ["color_K", "K", "camera_matrix"]).astype(np.float64)
        D = self._get_key(data, ["color_D", "D", "dist_coeffs"]).astype(np.float64)
        depth_scale = float(self._get_key(data, ["depth_scale_m_per_unit", "depth_scale"]))
        print(f"  [cam{cam_id}] K loaded, depth_scale={depth_scale:.6f}")
        return CameraIntrinsics(K=K, D=D, depth_scale=depth_scale, cam_id=cam_id)

    def load_extrinsics(self) -> Dict[str, np.ndarray]:
        extrinsics = {}
        for name in ["T_C0_C1", "T_C0_C2"]:
            path = self.extrinsics_dir / f"{name}.npy"
            if path.exists():
                T = np.load(str(path)).astype(np.float64)
                if T.shape != (4, 4):
                    raise ValueError(f"외부 파라미터 shape 오류: {path} -> {T.shape}")
                extrinsics[name] = T
                print(f"  [{name}] 로드 완료")
            else:
                print(f"  [WARNING] {path} 없음")
        return extrinsics

    def load_images(self, cam_id: int) -> Tuple[np.ndarray, np.ndarray]:
        cam_dir = self.image_dir / f"cam{cam_id}"
        color_path = cam_dir / f"rgb_{self.frame_id}.jpg"
        depth_path = cam_dir / f"depth_{self.frame_id}.png"

        if not color_path.exists():
            raise FileNotFoundError(f"컬러 이미지 없음: {color_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"깊이 이미지 없음: {depth_path}")

        color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_img is None:
            raise FileNotFoundError(f"이미지 로드 실패: cam{cam_id} frame {self.frame_id}")
        color_img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        print(f"  [cam{cam_id}] color={color_img.shape}, depth={depth_img.shape}")
        return color_img, depth_img

    def load_reference_pcd(self) -> o3d.geometry.PointCloud:
        ply_path = self.data_dir / "reference_knife.ply"
        if ply_path.exists():
            pcd = o3d.io.read_point_cloud(str(ply_path))
            print(f"  [REF] PLY 로드: {len(pcd.points)} pts")
            return pcd

        mesh = self.load_reference_mesh(print_info=False)
        if mesh is None:
            raise FileNotFoundError("레퍼런스 모델 없음 (PLY/GLB)")

        pts, face_idx = trimesh.sample.sample_surface(mesh, 30000, seed=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        try:
            face_colors = mesh.visual.face_colors
            if face_colors is not None and len(face_colors) > 0:
                colors = face_colors[face_idx][:, :3].astype(np.float64) / 255.0
                if np.std(colors) > 0.01:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
        except Exception:
            pass
        print(f"  [REF] GLB -> 점군: {len(pts)} pts")
        return pcd

    def load_reference_mesh(self, print_info: bool = True) -> Optional[trimesh.Trimesh]:
        if not self.glb_path.exists():
            return None

        scene_or_mesh = trimesh.load(str(self.glb_path), force="scene")
        if isinstance(scene_or_mesh, trimesh.Scene):
            geometries = [g.copy() for g in scene_or_mesh.geometry.values()]
            if not geometries:
                return None
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = scene_or_mesh

        if print_info:
            print(f"  [REF] GLB mesh 로드: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
        return mesh

    @staticmethod
    def _get_key(data, keys):
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"키를 찾을 수 없음: {keys}, 가능: {list(data.keys())}")


# =============================================================================
# Point cloud processing
# =============================================================================


class PointCloudProcessor:
    @staticmethod
    def depth_to_pointcloud(
        color_img: np.ndarray,
        depth_img: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        depth_scale: float,
        min_depth: float = 0.1,
        max_depth: float = 3.0,
    ) -> o3d.geometry.PointCloud:
        h, w = depth_img.shape[:2]

        if np.any(D != 0):
            color_img = cv2.undistort(color_img, K, D)
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
            depth_img = cv2.remap(depth_img, map1, map2, cv2.INTER_NEAREST)

        z = depth_img.astype(np.float64) * depth_scale
        valid = (z > min_depth) & (z < max_depth)

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (u[valid] - cx) * z[valid] / fx
        y = (v[valid] - cy) * z[valid] / fy
        points = np.stack([x, y, z[valid]], axis=-1)

        if color_img.ndim == 3:
            colors = color_img[valid].astype(np.float64) / 255.0
        else:
            gray = color_img[valid].astype(np.float64) / 255.0
            colors = np.stack([gray, gray, gray], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    @staticmethod
    def roi_3d_from_cam0(
        depth_img: np.ndarray,
        K: np.ndarray,
        depth_scale: float,
        roi: Tuple[int, int, int, int],
        margin: float = 0.03,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = roi
        h, w = depth_img.shape[:2]
        x1 = int(np.clip(x1, 0, w - 1))
        x2 = int(np.clip(x2, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        y2 = int(np.clip(y2, 0, h - 1))
        patch = depth_img[y1:y2, x1:x2].astype(np.float64) * depth_scale
        z_vals = patch[patch > 0.05].ravel()
        if len(z_vals) == 0:
            raise RuntimeError("ROI 내에 유효한 depth 값이 없습니다.")

        counts, edges = np.histogram(z_vals, bins=100)
        peak_bin = int(np.argmax(counts))
        z_peak = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
        z_min_fg = z_peak - 0.10
        z_max_fg = z_peak + 0.10

        ys, xs = np.where((patch > z_min_fg) & (patch < z_max_fg))
        if len(ys) == 0:
            raise RuntimeError("ROI foreground depth 추출 실패.")

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        z = patch[ys, xs]
        x3d = (xs + x1 - cx) * z / fx
        y3d = (ys + y1 - cy) * z / fy
        pts = np.stack([x3d, y3d, z], axis=-1)
        mn = pts.min(axis=0) - margin
        mx = pts.max(axis=0) + margin
        print(f"[ROI] depth peak={z_peak:.3f}m, 사용 범위={z_min_fg:.3f}~{z_max_fg:.3f}m")
        return mn, mx

    @staticmethod
    def filter_by_3d_box(
        pcd: o3d.geometry.PointCloud,
        min_xyz: np.ndarray,
        max_xyz: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        mask = np.all((pts >= min_xyz) & (pts <= max_xyz), axis=1)
        return pcd.select_by_index(np.where(mask)[0])

    @staticmethod
    def merge_pointclouds(
        camera_data_list: list,
        voxel_size: float = 0.002,
    ) -> o3d.geometry.PointCloud:
        merged = o3d.geometry.PointCloud()
        for cam_data in camera_data_list:
            intr = cam_data.intrinsics
            pcd = PointCloudProcessor.depth_to_pointcloud(
                cam_data.color_img,
                cam_data.depth_img,
                intr.K,
                intr.D,
                intr.depth_scale,
            )
            pcd.transform(cam_data.T_to_cam0)
            print(f"  [cam{intr.cam_id}] {len(pcd.points)} pts")
            merged += pcd

        n_before = len(merged.points)
        merged = merged.voxel_down_sample(voxel_size=voxel_size)
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  [통합] {n_before} -> {len(merged.points)} pts (voxel={voxel_size}m)")
        return merged

    @staticmethod
    def remove_table_plane(
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = 0.008,
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = plane_model
        print(f"  평면: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        objects_pcd = pcd.select_by_index(inliers, invert=True)
        objects_pcd, _ = objects_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        print(f"  평면 제거: {len(pcd.points)} -> {len(objects_pcd.points)} pts")

        plane_normal = np.array([a, b, c], dtype=np.float64)
        plane_point = -d * plane_normal / np.dot(plane_normal, plane_normal)
        if np.dot(plane_normal, -plane_point) < 0:
            plane_model = [-a, -b, -c, -d]
        return objects_pcd, np.array(plane_model, dtype=np.float64)

    @staticmethod
    def filter_by_plane_height(
        pcd: o3d.geometry.PointCloud,
        plane_model: np.ndarray,
        min_height: float = 0.002,
        max_height: float = 0.12,
    ) -> o3d.geometry.PointCloud:
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return pcd

        normal = np.asarray(plane_model[:3], dtype=np.float64)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-9:
            return pcd

        signed_dist = (pts @ normal + float(plane_model[3])) / normal_norm
        keep = (signed_dist >= min_height) & (signed_dist <= max_height)
        filtered = pcd.select_by_index(np.where(keep)[0])
        print(
            f"  높이 필터: {len(pcd.points)} -> {len(filtered.points)} pts  "
            f"({min_height*1000:.0f}~{max_height*1000:.0f} mm)"
        )
        return filtered


# =============================================================================
# Automatic object proposals
# =============================================================================


class AutomaticObjectDetector:
    @staticmethod
    def detect_from_cam0(
        camera_data: CameraData,
        plane_model: np.ndarray,
        min_height: float = 0.002,
        max_height: float = 0.12,
        roi_margin_m: float = 0.03,
        min_area_px: int = 1200,
        max_proposals: int = 6,
        output_dir: Optional[str] = None,
        frame_id: str = "",
    ) -> list:
        depth_img = camera_data.depth_img.astype(np.float64)
        K = camera_data.intrinsics.K.astype(np.float64)
        depth_scale = float(camera_data.intrinsics.depth_scale)

        h, w = depth_img.shape[:2]
        z = depth_img * depth_scale
        valid = z > 0.05
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        normal = np.asarray(plane_model[:3], dtype=np.float64)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-9:
            return []

        signed_dist = (normal[0] * x + normal[1] * y + normal[2] * z + float(plane_model[3])) / normal_norm
        mask = valid & (signed_dist >= min_height) & (signed_dist <= max_height)

        mask_u8 = (mask.astype(np.uint8) * 255)
        mask_u8 = cv2.medianBlur(mask_u8, 5)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        proposals = []
        proposal_id = 0

        for label in range(1, n_labels):
            x0, y0, bw, bh, area = stats[label]
            if area < min_area_px or bw < 20 or bh < 20:
                continue
            if area > 0.20 * h * w:
                continue
            if bw > int(0.8 * w) and bh > int(0.6 * h):
                continue

            comp_mask = labels == label
            comp_heights = signed_dist[comp_mask]
            comp_depth = z[comp_mask]
            if len(comp_depth) == 0:
                continue

            pad = max(10, int(0.08 * max(bw, bh)))
            x1 = max(0, x0 - pad)
            y1 = max(0, y0 - pad)
            x2 = min(w - 1, x0 + bw + pad)
            y2 = min(h - 1, y0 + bh + pad)

            try:
                min_xyz, max_xyz = PointCloudProcessor.roi_3d_from_cam0(
                    camera_data.depth_img,
                    camera_data.intrinsics.K,
                    camera_data.intrinsics.depth_scale,
                    (x1, y1, x2, y2),
                    margin=roi_margin_m,
                )
            except Exception:
                continue

            occupancy = float(area) / max(float(bw * bh), 1.0)
            border_penalty = 0.7 if (x0 <= 2 or y0 <= 2 or (x0 + bw) >= (w - 2) or (y0 + bh) >= (h - 2)) else 1.0
            score = float(area) * occupancy * border_penalty
            proposals.append(
                DetectionProposal(
                    proposal_id=proposal_id,
                    roi=(x1, y1, x2, y2),
                    min_xyz=min_xyz,
                    max_xyz=max_xyz,
                    area_px=int(area),
                    mean_height_m=float(np.median(comp_heights)),
                    depth_m=float(np.median(comp_depth)),
                    score=score,
                )
            )
            proposal_id += 1

        proposals.sort(key=lambda p: p.score, reverse=True)
        proposals = proposals[:max_proposals]

        if output_dir is not None:
            AutomaticObjectDetector._save_debug(
                camera_data.color_img,
                mask_u8,
                proposals,
                output_dir,
                frame_id,
            )

        print(f"  자동 detector proposal: {len(proposals)}개")
        for prop in proposals:
            x1, y1, x2, y2 = prop.roi
            print(
                f"    proposal {prop.proposal_id}: roi=({x1},{y1},{x2},{y2})  "
                f"area={prop.area_px}px  depth={prop.depth_m:.3f}m  "
                f"height={prop.mean_height_m*1000:.1f}mm  score={prop.score:.1f}"
            )
        return proposals

    @staticmethod
    def detect_from_clusters(
        objects_pcd: o3d.geometry.PointCloud,
        camera_data: CameraData,
        roi_margin_m: float = 0.03,
        min_points: int = 80,
        max_proposals: int = 8,
    ) -> list:
        pts = np.asarray(objects_pcd.points)
        if len(pts) == 0:
            return []

        labels = np.array(objects_pcd.cluster_dbscan(eps=0.012, min_points=30, print_progress=False))
        unique_labels = np.unique(labels[labels >= 0])
        K = camera_data.intrinsics.K.astype(np.float64)
        depth_scale = float(camera_data.intrinsics.depth_scale)
        depth_cam0 = camera_data.depth_img.astype(np.float64) * depth_scale
        h, w = camera_data.color_img.shape[:2]

        proposals = []
        proposal_id = 100
        for label in unique_labels:
            cl_pts = pts[labels == label]
            if len(cl_pts) < min_points:
                continue

            lo = np.percentile(cl_pts, 5, axis=0)
            hi = np.percentile(cl_pts, 95, axis=0)
            ext = hi - lo
            longest = float(np.max(ext))
            if longest < 0.015 or longest > 0.25:
                continue

            valid = cl_pts[:, 2] > 0.05
            if valid.sum() < min_points:
                continue
            p = cl_pts[valid]
            u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
            v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if ok.sum() < min_points:
                continue
            u_ok = u[ok]
            v_ok = v[ok]
            p_ok = p[ok]
            depth_ref = depth_cam0[v_ok, u_ok]
            consistent = (depth_ref > 0.05) & (np.abs(depth_ref - p_ok[:, 2]) < 0.04)
            if consistent.sum() >= min_points:
                u_ok = u_ok[consistent]
                v_ok = v_ok[consistent]
                p_ok = p_ok[consistent]

            x1 = max(0, int(u_ok.min()) - 10)
            y1 = max(0, int(v_ok.min()) - 10)
            x2 = min(w - 1, int(u_ok.max()) + 10)
            y2 = min(h - 1, int(v_ok.max()) + 10)

            min_xyz = p_ok.min(axis=0) - roi_margin_m
            max_xyz = p_ok.max(axis=0) + roi_margin_m
            roi_area = int(max(1, (x2 - x1) * (y2 - y1)))
            depth_m = float(np.median(p_ok[:, 2]))
            v_norm = float(np.mean(v_ok)) / max(float(h), 1.0)
            score = float(len(p_ok)) / max(depth_m ** 3, 1e-4)
            score *= (0.2 + 0.8 * np.clip(v_norm, 0.0, 1.0))
            proposals.append(
                DetectionProposal(
                    proposal_id=proposal_id,
                    roi=(x1, y1, x2, y2),
                    min_xyz=min_xyz,
                    max_xyz=max_xyz,
                    area_px=roi_area,
                    mean_height_m=float(np.median(p_ok[:, 1])),
                    depth_m=depth_m,
                    score=score,
                )
            )
            proposal_id += 1

        proposals.sort(key=lambda p: p.score, reverse=True)
        return proposals[:max_proposals]

    @staticmethod
    def merge_proposals(proposals: list, max_proposals: int = 8) -> list:
        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / float(area_a + area_b - inter)

        proposals = sorted(proposals, key=lambda p: p.score, reverse=True)
        merged = []
        for prop in proposals:
            if any(_iou(prop.roi, kept.roi) > 0.5 for kept in merged):
                continue
            merged.append(prop)
            if len(merged) >= max_proposals:
                break
        return merged

    @staticmethod
    def _save_debug(
        color_img: np.ndarray,
        mask_u8: np.ndarray,
        proposals: list,
        output_dir: str,
        frame_id: str,
    ):
        os.makedirs(output_dir, exist_ok=True)

        overlay = color_img.copy()
        for prop in proposals:
            x1, y1, x2, y2 = prop.roi
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (64, 255, 64), 2)
            cv2.putText(
                overlay,
                f"id={prop.proposal_id}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (64, 255, 64),
                2,
                cv2.LINE_AA,
            )

        overlay_path = os.path.join(output_dir, f"auto_detect_cam0_{frame_id}.png")
        mask_path = os.path.join(output_dir, f"auto_detect_mask_{frame_id}.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask_u8)
        print(f"  자동 detector 시각화 저장: {overlay_path}")
        print(f"  자동 detector 마스크 저장: {mask_path}")


# =============================================================================
# Reference preparation
# =============================================================================


class ReferencePreprocessor:
    @staticmethod
    def _robust_longest_extent(pts: np.ndarray, q_low: float = 15.0, q_high: float = 85.0) -> float:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) == 0:
            return 0.0
        lo = np.percentile(pts, q_low, axis=0)
        hi = np.percentile(pts, q_high, axis=0)
        return float(np.max(hi - lo))

    @staticmethod
    def estimate_object_size_m(
        objects_pcd: o3d.geometry.PointCloud,
        proposals: Optional[list] = None,
        q_low: float = 15.0,
        q_high: float = 85.0,
        top_k: int = 3,
        min_points: int = 150,
    ) -> Optional[float]:
        size_candidates = []

        if proposals:
            for prop in proposals[:top_k]:
                crop = PointCloudProcessor.filter_by_3d_box(objects_pcd, prop.min_xyz, prop.max_xyz)
                pts = np.asarray(crop.points)
                if len(pts) < min_points:
                    continue
                longest = ReferencePreprocessor._robust_longest_extent(pts, q_low=q_low, q_high=q_high)
                if 0.01 <= longest <= 0.25:
                    size_candidates.append(longest)

        if not size_candidates:
            pts = np.asarray(objects_pcd.points)
            if len(pts) >= min_points:
                longest = ReferencePreprocessor._robust_longest_extent(pts, q_low=20.0, q_high=80.0)
                if 0.01 <= longest <= 0.25:
                    size_candidates.append(longest)

        if not size_candidates:
            return None

        return float(np.median(size_candidates))

    @staticmethod
    def prepare_reference(
        ref_pcd: o3d.geometry.PointCloud,
        ref_mesh: Optional[trimesh.Trimesh],
        output_dir: Optional[str] = None,
        target_size_m: Optional[float] = None,
        save_glb: bool = True,
    ) -> PreparedReference:
        ref_pts = np.asarray(ref_pcd.points, dtype=np.float64)
        if len(ref_pts) == 0:
            raise RuntimeError("레퍼런스 점군이 비어 있어 자동 준비를 할 수 없습니다.")

        center = ref_pts.mean(axis=0)
        centered_pts = ref_pts - center
        longest = float((centered_pts.max(axis=0) - centered_pts.min(axis=0)).max())
        if longest < 1e-9:
            raise RuntimeError("레퍼런스 모델 extent가 너무 작아 자동 준비를 할 수 없습니다.")

        prep_scale = float(target_size_m / longest) if target_size_m is not None else 1.0

        prepared_pcd = o3d.geometry.PointCloud()
        prepared_pcd.points = o3d.utility.Vector3dVector(centered_pts * prep_scale)
        if ref_pcd.has_colors():
            prepared_pcd.colors = o3d.utility.Vector3dVector(np.asarray(ref_pcd.colors))

        prepared_mesh = None
        if ref_mesh is not None:
            prepared_mesh = ref_mesh.copy()
            center_T = np.eye(4)
            center_T[:3, 3] = -center
            scale_T = np.eye(4)
            scale_T[:3, :3] *= prep_scale
            prepared_mesh.apply_transform(scale_T @ center_T)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            info = {
                "estimated_size_m": None if target_size_m is None else float(target_size_m),
                "original_longest_m": float(longest),
                "prep_scale": float(prep_scale),
                "prepared_longest_m": float(longest * prep_scale),
                "center_shift_xyz_m": center.tolist(),
            }
            info_path = os.path.join(output_dir, "reference_prepared_info.json")
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            print(f"  reference 준비 정보 저장: {info_path}")

            if save_glb and prepared_mesh is not None:
                prepared_glb_path = os.path.join(output_dir, "reference_prepared.glb")
                trimesh.Scene(prepared_mesh).export(prepared_glb_path)
                print(f"  자동 준비된 reference GLB 저장: {prepared_glb_path}")

        return PreparedReference(
            pcd=prepared_pcd,
            mesh=prepared_mesh,
            original_center=center,
            original_longest_m=longest,
            prep_scale=prep_scale,
            estimated_size_m=target_size_m,
        )


# =============================================================================
# Pose estimation backend
# =============================================================================


class PoseEstimator:
    _ref_points_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    @staticmethod
    def _pca_axes(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 3:
            return np.eye(3), np.ones(3)
        c = pts - pts.mean(axis=0)
        cov = c.T @ c / max(len(pts), 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        R = vecs[:, order]
        vals = vals[order]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return R, vals

    @staticmethod
    def _pca_rotation_candidates(src_pts: np.ndarray, tgt_pts: np.ndarray) -> List[np.ndarray]:
        R_s, _ = PoseEstimator._pca_axes(src_pts)
        R_t, _ = PoseEstimator._pca_axes(tgt_pts)
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

        unique = [cands[0]]
        for R in cands[1:]:
            dup = False
            for Ru in unique:
                ang = np.arccos(np.clip((np.trace(R @ Ru.T) - 1.0) / 2.0, -1.0, 1.0))
                if ang < np.radians(5.0):
                    dup = True
                    break
            if not dup:
                unique.append(R)
        return unique

    @staticmethod
    def _pca_candidate_transforms(src_pts: np.ndarray, dst_pts: np.ndarray) -> List[np.ndarray]:
        src_center = np.asarray(src_pts, dtype=np.float64).mean(axis=0)
        dst_center = np.asarray(dst_pts, dtype=np.float64).mean(axis=0)
        transforms = []
        for R in PoseEstimator._pca_rotation_candidates(src_pts, dst_pts)[:4]:
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = dst_center - R @ src_center
            transforms.append(T)
        return transforms

    @staticmethod
    def _fpfh_global_registration(source, target, voxel_size):
        radius_normal = voxel_size * 3
        radius_feature = voxel_size * 5

        for pcd in [source, target]:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
            )

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
        )
        return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 3,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )

    @staticmethod
    def _find_target_cluster(
        objects_pcd: o3d.geometry.PointCloud,
        labels: np.ndarray,
        unique_labels: np.ndarray,
    ) -> Tuple[int, float]:
        if len(unique_labels) == 0:
            return -1, 0.0
        if not objects_pcd.has_colors():
            counts = [(int(label), int(np.sum(labels == label))) for label in unique_labels]
            counts.sort(key=lambda x: x[1], reverse=True)
            return counts[0][0], 0.0

        colors = np.asarray(objects_pcd.colors)
        best_label = int(unique_labels[0])
        best_ratio = 0.0
        best_score = -1.0
        for label in unique_labels:
            mask = labels == label
            cl_colors = colors[mask]
            if len(cl_colors) == 0:
                continue
            yellow = (cl_colors[:, 0] > 0.55) & (cl_colors[:, 1] > 0.55) & (cl_colors[:, 2] < 0.45)
            yellow_ratio = float(np.mean(yellow))
            size_bonus = min(np.sum(mask) / 5000.0, 1.0) * 0.2
            score = yellow_ratio + size_bonus
            if score > best_score:
                best_score = score
                best_ratio = yellow_ratio
                best_label = int(label)
        return best_label, best_ratio

    @staticmethod
    def _reference_points(
        ref_pcd: o3d.geometry.PointCloud,
        ref_mesh: Optional[trimesh.Trimesh],
        n_samples: int = 20000,
    ) -> np.ndarray:
        cache_key = (id(ref_pcd), id(ref_mesh), int(n_samples))
        cached = PoseEstimator._ref_points_cache.get(cache_key)
        if cached is not None:
            return cached

        pts = np.asarray(ref_pcd.points, dtype=np.float64)
        if len(pts) == 0 and ref_mesh is not None:
            pts, _ = trimesh.sample.sample_surface(ref_mesh, n_samples, seed=0)
            pts = np.asarray(pts, dtype=np.float64)

        if len(pts) > n_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), n_samples, replace=False)
            pts = pts[idx]

        PoseEstimator._ref_points_cache[cache_key] = pts
        return pts

    @staticmethod
    def _multiview_depth_score(
        model_pts_cam0: np.ndarray,
        camera_data_list: list,
        tol: float = 0.015,
    ) -> float:
        total_ok = 0
        total_valid = 0
        pts_h = np.hstack([model_pts_cam0, np.ones((len(model_pts_cam0), 1), dtype=np.float64)])

        for cam_data in camera_data_list:
            h, w = cam_data.depth_img.shape
            T_inv = np.linalg.inv(cam_data.T_to_cam0)
            pts_ci = (T_inv @ pts_h.T)[:3].T
            front = pts_ci[:, 2] > 0.05
            if front.sum() == 0:
                continue

            p = pts_ci[front]
            K = cam_data.intrinsics.K
            u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
            v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
            ok_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if ok_uv.sum() == 0:
                continue

            z_model = p[ok_uv, 2]
            z_real = cam_data.depth_img[v[ok_uv], u[ok_uv]].astype(np.float64) * cam_data.intrinsics.depth_scale
            has_d = z_real > 0.05
            if has_d.sum() == 0:
                continue

            diff = np.abs(z_model[has_d] - z_real[has_d])
            total_ok += int((diff < tol).sum())
            total_valid += int(has_d.sum())

        return float(total_ok) / max(float(total_valid), 1.0)

    @staticmethod
    def _coverage_score(
        model_pts_cam0: np.ndarray,
        obj_pcd: o3d.geometry.PointCloud,
        radius: float = 0.005,
    ) -> float:
        obj_pts = np.asarray(obj_pcd.points)
        if len(obj_pts) == 0 or len(model_pts_cam0) == 0:
            return 0.0

        model_pcd = o3d.geometry.PointCloud()
        model_pcd.points = o3d.utility.Vector3dVector(model_pts_cam0)
        fwd_dists = np.asarray(model_pcd.compute_point_cloud_distance(obj_pcd))
        forward = float((fwd_dists < radius).sum()) / max(float(len(fwd_dists)), 1.0)

        margin = radius * 5.0
        lo = model_pts_cam0.min(axis=0) - margin
        hi = model_pts_cam0.max(axis=0) + margin
        in_box = np.all((obj_pts >= lo) & (obj_pts <= hi), axis=1)
        n_inbox = int(in_box.sum())
        if n_inbox < 10:
            return forward * 0.01

        inbox_pcd = o3d.geometry.PointCloud()
        inbox_pcd.points = o3d.utility.Vector3dVector(obj_pts[in_box])
        rev_dists = np.asarray(inbox_pcd.compute_point_cloud_distance(model_pcd))
        reverse = float((rev_dists < radius).sum()) / max(float(len(rev_dists)), 1.0)
        return forward * reverse

    @staticmethod
    def _scale_candidates(
        ref_longest: float,
        obj_extent: float,
        object_size_m: Optional[float],
        max_candidates: int = 8,
        narrow_search: bool = False,
    ) -> List[float]:
        if ref_longest < 1e-9:
            raise RuntimeError("레퍼런스 모델 extent가 너무 작습니다.")

        if object_size_m is not None:
            exact = object_size_m / ref_longest
            return [float(exact)]

        anchor = obj_extent / ref_longest if obj_extent > 1e-9 else 1.0
        candidates = []
        if narrow_search:
            for mult in [0.75, 0.90, 1.00, 1.10, 1.25]:
                candidates.append(anchor * mult)
        else:
            for mult in [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.50]:
                candidates.append(anchor * mult)
            for real_cm in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
                candidates.append((real_cm / 100.0) / ref_longest)

        unique = []
        for s in sorted(candidates, key=lambda x: abs(np.log(max(x, 1e-9) / max(anchor, 1e-9)))):
            if s <= 0.001 or s > 100.0:
                continue
            if any(abs(np.log(s / u)) < 0.03 for u in unique):
                continue
            unique.append(float(s))
            if len(unique) >= max_candidates:
                break
        return unique

    @staticmethod
    def estimate_pose(
        objects_pcd: o3d.geometry.PointCloud,
        ref_pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.003,
        object_size_m: Optional[float] = None,
        camera_data_list: Optional[list] = None,
        ref_mesh: Optional[trimesh.Trimesh] = None,
        ref_sample_points: int = 12000,
        max_scale_candidates: int = 8,
        max_pca_candidates: int = 8,
        narrow_scale_search: bool = False,
    ) -> Tuple[PoseResult, o3d.geometry.PointCloud, float]:
        if camera_data_list is None or len(camera_data_list) == 0:
            raise RuntimeError("camera_data_list가 필요합니다.")

        obj_pts = np.asarray(objects_pcd.points, dtype=np.float64)
        if len(obj_pts) < 50:
            raise RuntimeError("포즈 추정을 위한 물체 점군이 너무 적습니다.")

        ref_pts = PoseEstimator._reference_points(ref_pcd, ref_mesh, n_samples=ref_sample_points)
        if len(ref_pts) < 50:
            raise RuntimeError("레퍼런스 점군이 너무 적습니다.")
        ref_center = ref_pts.mean(axis=0)
        ref_pts_local = ref_pts - ref_center
        ref_longest = float((ref_pts_local.max(axis=0) - ref_pts_local.min(axis=0)).max())
        obj_extent = float((obj_pts.max(axis=0) - obj_pts.min(axis=0)).max())

        print("\n" + "=" * 60)
        print("[포즈 추정] Obj_pose_estimator 스타일 backend")
        print("=" * 60)
        print(f"  REF pts={len(ref_pts_local)} longest={ref_longest:.4f}m")
        print(f"  OBJ pts={len(obj_pts)} longest={obj_extent:.4f}m")

        scale_candidates = PoseEstimator._scale_candidates(
            ref_longest,
            obj_extent,
            object_size_m,
            max_candidates=max_scale_candidates,
            narrow_search=narrow_scale_search,
        )
        if object_size_m is not None:
            print(f"  스케일 고정: {object_size_m*100:.1f}cm")
        print(f"  스케일 후보: {len(scale_candidates)}개")

        best = None
        best_score = -1.0

        for scale in scale_candidates:
            scaled_pts = ref_pts_local * scale
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(scaled_pts)
            real_size = max(ref_longest * scale, voxel_size * 4)
            vx = max(voxel_size, real_size * 0.08)

            src_down = src.voxel_down_sample(vx)
            tgt_down = objects_pcd.voxel_down_sample(vx)
            if len(src_down.points) < 30 or len(tgt_down.points) < 30:
                continue

            try:
                ransac = PoseEstimator._fpfh_global_registration(
                    o3d.geometry.PointCloud(src_down),
                    o3d.geometry.PointCloud(tgt_down),
                    vx,
                )
                fpfh_T = np.asarray(ransac.transformation)
                fpfh_ok = ransac.fitness > 0.05
            except Exception:
                fpfh_T = np.eye(4)
                fpfh_ok = False

            init_candidates = []
            if fpfh_ok:
                init_candidates.append(fpfh_T)

            tgt_center = obj_pts.mean(axis=0)
            src_center = scaled_pts.mean(axis=0)
            for R in PoseEstimator._pca_rotation_candidates(scaled_pts, obj_pts)[:max_pca_candidates]:
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tgt_center - R @ src_center
                init_candidates.append(T)

            max_dist = max(real_size * 0.5, voxel_size * 6.0)
            tgt_n = o3d.geometry.PointCloud(objects_pcd)
            tgt_n.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30)
            )

            best_local_fitness = -1.0
            best_local_T = np.eye(4)
            for T_init in init_candidates:
                src_c = o3d.geometry.PointCloud(src)
                src_c.transform(T_init)
                src_c.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30)
                )

                icp = o3d.pipelines.registration.registration_icp(
                    src_c,
                    tgt_n,
                    max_correspondence_distance=max_dist,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-7,
                        relative_rmse=1e-7,
                        max_iteration=80,
                    ),
                )
                total_T = np.asarray(icp.transformation) @ T_init
                if icp.fitness > best_local_fitness:
                    best_local_fitness = float(icp.fitness)
                    best_local_T = total_T

            if best_local_fitness < 0.1:
                print(
                    f"    scale={scale:.4f} ({real_size*100:.1f}cm): "
                    f"fitness={best_local_fitness:.3f} -> skip"
                )
                continue

            src_r = o3d.geometry.PointCloud(src)
            src_r.transform(best_local_T)
            src_r.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30)
            )

            cur_T = np.eye(4)
            icp = None
            for icp_scale in [3.0, 1.5, 0.8]:
                icp = o3d.pipelines.registration.registration_icp(
                    src_r,
                    tgt_n,
                    max_correspondence_distance=max_dist * icp_scale,
                    init=cur_T,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-8,
                        relative_rmse=1e-8,
                        max_iteration=200,
                    ),
                )
                cur_T = np.asarray(icp.transformation)

            final_T = cur_T @ best_local_T
            aligned = (final_T @ np.hstack([scaled_pts, np.ones((len(scaled_pts), 1))]).T)[:3].T
            depth_sc = PoseEstimator._multiview_depth_score(aligned, camera_data_list)
            coverage = PoseEstimator._coverage_score(
                aligned,
                objects_pcd,
                radius=max(real_size * 0.1, 0.003),
            )
            combined = depth_sc * coverage
            final_fitness = float(icp.fitness) if icp is not None else 0.0
            final_rmse = float(icp.inlier_rmse) if icp is not None else 0.0

            print(
                f"    scale={scale:.4f} ({real_size*100:.1f}cm): "
                f"fitness={final_fitness:.3f}  depth={depth_sc:.3f}  "
                f"cov={coverage:.3f}  -> score={combined:.4f}"
            )

            if combined > best_score:
                best_score = combined
                best = {
                    "transform": final_T,
                    "scale": float(scale),
                    "score": float(combined),
                    "depth_score": float(depth_sc),
                    "coverage": float(coverage),
                    "fitness": final_fitness,
                    "rmse": final_rmse,
                    "scaled_pts": scaled_pts,
                }

        if best is None:
            raise RuntimeError("정합 실패 - 물체를 찾지 못했습니다.")

        T_raw = np.asarray(best["transform"], dtype=np.float64)
        R_raw = T_raw[:3, :3]
        U, _, Vt = np.linalg.svd(R_raw)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        t = T_raw[:3, 3].copy()
        T_out = np.eye(4)
        T_out[:3, :3] = R
        T_out[:3, 3] = t

        model_aligned = o3d.geometry.PointCloud()
        model_aligned.points = o3d.utility.Vector3dVector(best["scaled_pts"])
        model_aligned.transform(T_out)
        model_aligned.paint_uniform_color([1.0, 0.0, 0.0])

        rot = Rotation.from_matrix(R)
        pose = PoseResult(
            translation=t,
            rotation_matrix=R,
            euler_xyz_deg=rot.as_euler("xyz", degrees=True),
            quaternion_xyzw=rot.as_quat(),
            transform_4x4=T_out,
            scale=best["scale"],
            score=best["score"],
            depth_score=best["depth_score"],
            coverage=best["coverage"],
            fitness=best["fitness"],
            rmse=best["rmse"],
            method="Reference Matching",
        )
        print(
            f"  위치: {np.round(t, 4)} m  "
            f"회전: {np.round(pose.euler_xyz_deg, 1)} deg"
        )
        return pose, model_aligned, best["scale"]


# =============================================================================
# Result validation / export
# =============================================================================


class PoseValidator:
    @staticmethod
    def reprojection_check(
        model_pcd: o3d.geometry.PointCloud,
        camera_data: CameraData,
        output_path: str,
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pts = np.asarray(model_pcd.points)
        if len(pts) == 0:
            return

        K = camera_data.intrinsics.K
        T_cam = np.linalg.inv(camera_data.T_to_cam0)
        pts_hom = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        pts_cam = (T_cam @ pts_hom.T)[:3]

        valid = pts_cam[2] > 0.05
        proj = K @ pts_cam[:, valid]
        u = (proj[0] / proj[2]).astype(np.int32)
        v = (proj[1] / proj[2]).astype(np.int32)

        img = camera_data.color_img.copy()
        h, w = img.shape[:2]
        mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        cam_id = camera_data.intrinsics.cam_id
        axes[0].imshow(img)
        axes[0].set_title(f"cam{cam_id} - 원본")
        axes[0].axis("off")
        axes[1].imshow(img)
        axes[1].scatter(u[mask], v[mask], c="lime", s=1, alpha=0.3)
        axes[1].set_title(f"cam{cam_id} - 재투영")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  재투영 저장: {output_path}")


def save_estimation_manifest(
    output_path: str,
    data_dir: str,
    capture_dir: str,
    glb_path: str,
    frame_id: str,
    camera_data_list: list,
    prepared_ref: Optional[PreparedReference],
    used_auto_detect: bool,
    used_roi: bool,
):
    manifest = {
        "goal": "Estimate real-world scale and 6DoF pose from calibrated multiview RGB-D and an unscaled reference GLB.",
        "inputs_used": {
            "data_dir": data_dir,
            "capture_dir": capture_dir,
            "glb_path": glb_path,
            "frame_id": frame_id,
            "camera_intrinsics": [f"cam{cam.intrinsics.cam_id}" for cam in camera_data_list],
            "camera_extrinsics": ["T_C0_C1", "T_C0_C2"],
            "rgbd_images": [f"cam{cam.intrinsics.cam_id}: rgb/depth" for cam in camera_data_list],
        },
        "manual_real_world_measurements_used": False,
        "inference_sources": [
            "Depth point cloud reconstructed from calibrated RGB-D images",
            "Camera intrinsics and fixed-camera extrinsics",
            "Reference GLB geometry sampled as a point cloud",
        ],
        "pipeline_modes": {
            "auto_detect": bool(used_auto_detect),
            "roi_used": bool(used_roi),
            "reference_auto_prepare": prepared_ref is not None,
        },
    }
    if prepared_ref is not None:
        manifest["reference_auto_prepare"] = {
            "estimated_object_longest_m": prepared_ref.estimated_size_m,
            "estimated_object_longest_mm": None if prepared_ref.estimated_size_m is None else float(prepared_ref.estimated_size_m * 1000.0),
            "original_reference_longest_m": float(prepared_ref.original_longest_m),
            "reference_prep_scale": float(prepared_ref.prep_scale),
            "reference_center_shift_m": prepared_ref.original_center.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  추정 입력 요약 저장: {output_path}")


def _output_unit_factor(output_unit: str) -> float:
    unit = str(output_unit).strip().lower()
    if unit == "mm":
        return 1000.0
    if unit == "m":
        return 1.0
    raise ValueError(f"지원하지 않는 output unit: {output_unit}")


def _transform_with_scaled_translation(T: np.ndarray, factor: float) -> np.ndarray:
    T_scaled = np.asarray(T, dtype=np.float64).copy()
    T_scaled[:3, 3] *= factor
    return T_scaled


def print_pose(result: PoseResult):
    print(f"\n{'='*60}")
    print(f"포즈 추정 결과 [{result.method}]")
    print(f"{'='*60}")
    t = result.translation
    e = result.euler_xyz_deg
    print(f"  위치: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
    print(f"  회전: ({e[0]:+.1f}°, {e[1]:+.1f}°, {e[2]:+.1f}°)")
    print(
        f"  품질: score={result.score:.4f}, depth={result.depth_score:.4f}, "
        f"coverage={result.coverage:.4f}, fitness={result.fitness:.4f}, "
        f"RMSE={result.rmse:.6f}m, scale={result.scale:.4f}"
    )


def save_pose(result: PoseResult, output_path: str):
    transform_4x4_mm = _transform_with_scaled_translation(result.transform_4x4, 1000.0)
    np.savez(
        output_path,
        translation=result.translation,
        translation_mm=result.translation * 1000.0,
        rotation_matrix=result.rotation_matrix,
        euler_xyz_deg=result.euler_xyz_deg,
        quaternion_xyzw=result.quaternion_xyzw,
        transform_4x4=result.transform_4x4,
        transform_4x4_mm=transform_4x4_mm,
        scale=result.scale,
        score=result.score,
        depth_score=result.depth_score,
        coverage=result.coverage,
        fitness=result.fitness,
        rmse=result.rmse,
        method=result.method,
    )
    print(f"  포즈 저장: {output_path}")


def save_sim_config(
    result: PoseResult,
    model_pcd: o3d.geometry.PointCloud,
    output_path: str,
    output_unit: str = "mm",
):
    factor = _output_unit_factor(output_unit)
    unit_name = "millimeter" if factor == 1000.0 else "meter"
    obb = model_pcd.get_oriented_bounding_box()
    extent = np.sort(obb.extent)[::-1]
    t = result.translation
    e = result.euler_xyz_deg
    q = result.quaternion_xyzw
    T_out = result.transform_4x4

    t_out = t * factor
    extent_out = extent * factor
    obb_center_out = obb.center * factor
    T_out_scaled = _transform_with_scaled_translation(T_out, factor)
    T_out_mm = _transform_with_scaled_translation(T_out, 1000.0)

    config = {
        "coordinate_frame": "OpenCV (cam0 ref): X-right, Y-down, Z-forward",
        "unit": f"{unit_name} / degree",
        "position": {
            "x": round(float(t_out[0]), 3 if factor == 1000.0 else 6),
            "y": round(float(t_out[1]), 3 if factor == 1000.0 else 6),
            "z": round(float(t_out[2]), 3 if factor == 1000.0 else 6),
        },
        "position_mm": {
            "x": round(float(t[0] * 1000.0), 3),
            "y": round(float(t[1] * 1000.0), 3),
            "z": round(float(t[2] * 1000.0), 3),
        },
        "position_m": {
            "x": round(float(t[0]), 6),
            "y": round(float(t[1]), 6),
            "z": round(float(t[2]), 6),
        },
        "rotation_euler_xyz_deg": {
            "rx": round(float(e[0]), 2),
            "ry": round(float(e[1]), 2),
            "rz": round(float(e[2]), 2),
        },
        "rotation_quaternion_xyzw": {
            "x": round(float(q[0]), 6),
            "y": round(float(q[1]), 6),
            "z": round(float(q[2]), 6),
            "w": round(float(q[3]), 6),
        },
        "rotation_matrix_3x3": result.rotation_matrix.tolist(),
        "transform_4x4": T_out_scaled.tolist(),
        "transform_4x4_mm": T_out_mm.tolist(),
        "transform_4x4_m": T_out.tolist(),
        "scale_uniform": float(result.scale),
        "size": {
            "length": round(float(extent_out[0]), 3 if factor == 1000.0 else 4),
            "width": round(float(extent_out[1]), 3 if factor == 1000.0 else 4),
            "height": round(float(extent_out[2]), 3 if factor == 1000.0 else 4),
        },
        "size_mm": {
            "length": round(float(extent[0] * 1000.0), 3),
            "width": round(float(extent[1] * 1000.0), 3),
            "height": round(float(extent[2] * 1000.0), 3),
        },
        "size_cm": {
            "length": round(float(extent[0] * 100), 2),
            "width": round(float(extent[1] * 100), 2),
            "height": round(float(extent[2] * 100), 2),
        },
        "size_m": {
            "length": round(float(extent[0]), 4),
            "width": round(float(extent[1]), 4),
            "height": round(float(extent[2]), 4),
        },
        "obb_center": {
            "x": round(float(obb_center_out[0]), 3 if factor == 1000.0 else 6),
            "y": round(float(obb_center_out[1]), 3 if factor == 1000.0 else 6),
            "z": round(float(obb_center_out[2]), 3 if factor == 1000.0 else 6),
        },
        "obb_center_mm": {
            "x": round(float(obb.center[0] * 1000.0), 3),
            "y": round(float(obb.center[1] * 1000.0), 3),
            "z": round(float(obb.center[2] * 1000.0), 3),
        },
        "score": round(float(result.score), 6),
        "depth_score": round(float(result.depth_score), 6),
        "coverage": round(float(result.coverage), 6),
        "fitness": round(float(result.fitness), 6),
        "rmse_m": round(float(result.rmse), 6),
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  SIM 설정 저장: {output_path}")


def save_aligned_glb(
    source_glb_path: str,
    ref_pcd: o3d.geometry.PointCloud,
    pose: PoseResult,
    scale: float,
    output_path: str,
    transform_mode: str = "bake",
    output_unit: str = "mm",
):
    src_path = Path(source_glb_path)
    if not src_path.exists():
        print(f"  [WARNING] 정합 GLB 저장 생략: 입력 GLB 없음 ({src_path})")
        return

    ref_pts = np.asarray(ref_pcd.points)
    if len(ref_pts) == 0:
        print("  [WARNING] 정합 GLB 저장 생략: 레퍼런스 점군이 비었습니다.")
        return

    scene = trimesh.load(str(src_path), force="scene")
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    scene = scene.copy()

    center_T = np.eye(4)
    center_T[:3, 3] = -ref_pts.mean(axis=0)

    scale_T = np.eye(4)
    scale_T[:3, :3] *= float(scale)

    pose_T = np.asarray(pose.transform_4x4, dtype=np.float64)
    unit_factor = _output_unit_factor(output_unit)
    unit_T = np.eye(4)
    unit_T[:3, :3] *= unit_factor
    export_T = unit_T @ pose_T @ scale_T @ center_T
    mode = str(transform_mode).lower().strip()
    if mode == "node":
        for node_name in list(scene.graph.nodes_geometry):
            node_T, _ = scene.graph.get(frame_to=node_name)
            scene.graph.update(frame_to=node_name, matrix=export_T @ node_T)
        scene.metadata = dict(scene.metadata or {})
        scene.metadata["rb_pose_cam0"] = {
            "transform_4x4": pose.transform_4x4.tolist(),
            "transform_4x4_mm": _transform_with_scaled_translation(pose.transform_4x4, 1000.0).tolist(),
            "scale_uniform": float(scale),
            "translation_xyz_m": pose.translation.tolist(),
            "translation_xyz_mm": (pose.translation * 1000.0).tolist(),
            "rotation_quaternion_xyzw": pose.quaternion_xyzw.tolist(),
            "rotation_euler_xyz_deg": pose.euler_xyz_deg.tolist(),
            "output_unit": output_unit,
        }
    else:
        scene.apply_transform(export_T)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scene.export(output_path)
    print(f"  정합 GLB 저장({mode}): {output_path}")


# =============================================================================
# Helpers
# =============================================================================


def list_available_frame_ids(data_dir: str) -> list:
    cam0_dir = Path(data_dir) / "object_capture" / "cam0"
    if not cam0_dir.exists():
        return []
    ids = []
    for p in sorted(cam0_dir.glob("rgb_*.jpg")):
        stem = p.stem
        if "_" not in stem:
            continue
        fid = stem.split("_", 1)[1]
        if fid.isdigit():
            ids.append(fid)
    return ids


def parse_frame_search_spec(spec: Optional[str], available_ids: list) -> list:
    if not available_ids:
        return []
    if spec is None or str(spec).strip() == "":
        return available_ids

    wanted_int = set()
    wanted_raw = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a = a.strip()
            b = b.strip()
            if (not a.isdigit()) or (not b.isdigit()):
                continue
            ia, ib = int(a), int(b)
            lo, hi = min(ia, ib), max(ia, ib)
            for i in range(lo, hi + 1):
                wanted_int.add(i)
        else:
            if token.isdigit():
                wanted_int.add(int(token))
            else:
                wanted_raw.add(token)

    filtered = []
    for fid in available_ids:
        if fid in wanted_raw:
            filtered.append(fid)
        elif fid.isdigit() and int(fid) in wanted_int:
            filtered.append(fid)
    return filtered


def pose_confidence_score(pose: PoseResult) -> float:
    return float(max(pose.score, 1e-6) * max(0.25, pose.fitness))


def identity_confidence_score(pose: PoseResult) -> float:
    return pose_confidence_score(pose)


def parse_roi_arg(roi_str: str) -> Tuple[int, int, int, int]:
    vals = [int(v.strip()) for v in roi_str.split(",")]
    if len(vals) != 4:
        raise ValueError("--roi 형식은 x1,y1,x2,y2 이어야 합니다.")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--roi 는 x2>x1, y2>y1 이어야 합니다.")
    return x1, y1, x2, y2


def select_roi_interactive(color_img: np.ndarray) -> Tuple[int, int, int, int]:
    bgr = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    x, y, w, h = cv2.selectROI("Select ROI (cam0)", bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI (cam0)")
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI 선택이 취소되었습니다.")
    return int(x), int(y), int(x + w), int(y + h)


def save_roi_debug(color_img: np.ndarray, roi: Tuple[int, int, int, int], output_path: str):
    vis = color_img.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 64, 64), 2)
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"  ROI 시각화 저장: {output_path}")


def estimate_pose_with_proposals(
    objects_pcd: o3d.geometry.PointCloud,
    ref_pcd: o3d.geometry.PointCloud,
    ref_mesh: Optional[trimesh.Trimesh],
    proposals: list,
    camera_data_list: list,
    args,
):
    best = None
    best_total = -1.0
    second_best_total = -1.0
    max_prop_score = max((p.score for p in proposals), default=1.0)

    for prop in proposals:
        proposal_pcd = PointCloudProcessor.filter_by_3d_box(objects_pcd, prop.min_xyz, prop.max_xyz)
        n_pts = len(proposal_pcd.points)
        print(f"  [proposal {prop.proposal_id}] crop pts={n_pts}")
        if n_pts < args.detector_min_points:
            print(f"    [skip] 점 수 부족 (< {args.detector_min_points})")
            continue

        try:
            pose_i, model_i, scale_i = PoseEstimator.estimate_pose(
                proposal_pcd,
                ref_pcd,
                voxel_size=args.voxel_size,
                object_size_m=getattr(args, "object_size_m", None),
                camera_data_list=camera_data_list,
                ref_mesh=ref_mesh,
                ref_sample_points=getattr(args, "ref_sample_points", 12000),
                max_scale_candidates=getattr(args, "max_scale_candidates", 8),
                max_pca_candidates=getattr(args, "max_pca_candidates", 8),
                narrow_scale_search=getattr(args, "narrow_scale_search", False),
            )
        except Exception as exc:
            print(f"    [skip] proposal 정합 실패: {exc}")
            continue

        det_norm = float(prop.score) / max(float(max_prop_score), 1e-6)
        total_i = float(pose_i.score * (0.90 + 0.10 * det_norm))
        print(
            f"    detector_score={det_norm:.3f}  "
            f"pose_score={pose_i.score:.4f} depth={pose_i.depth_score:.4f} "
            f"coverage={pose_i.coverage:.4f} fitness={pose_i.fitness:.4f}  "
            f"total={total_i:.4f}"
        )

        if total_i > best_total:
            second_best_total = best_total
            best_total = total_i
            best = {
                "pose": pose_i,
                "model": model_i,
                "scale": scale_i,
                "proposal": prop,
                "score": total_i,
                "cropped_pcd": proposal_pcd,
            }
        elif total_i > second_best_total:
            second_best_total = total_i

    if best is None:
        raise RuntimeError("자동 detector proposal 중 정합 가능한 후보가 없습니다.")
    best["second_score"] = second_best_total
    return best


# =============================================================================
# Main pipeline
# =============================================================================


def run_pipeline(args):
    print("=" * 60)
    print(" 멀티뷰 RGB-D 기반 물체 포즈 추정")
    print("=" * 60)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n[1/5] 데이터 로드")
    loader = DataLoader(
        data_dir=args.data_dir,
        frame_id=args.frame_id,
        extrinsics_dir=args.extrinsics_dir,
        glb_path=args.glb_path,
    )
    intrinsics = [loader.load_intrinsics(i) for i in range(args.num_cameras)]
    extrinsics = loader.load_extrinsics()

    camera_data_list = []
    for i in range(args.num_cameras):
        color, depth = loader.load_images(i)
        T_to_cam0 = np.eye(4)
        if i == 1 and "T_C0_C1" in extrinsics:
            T_to_cam0 = extrinsics["T_C0_C1"]
        elif i == 2 and "T_C0_C2" in extrinsics:
            T_to_cam0 = extrinsics["T_C0_C2"]
        camera_data_list.append(
            CameraData(
                intrinsics=intrinsics[i],
                color_img=color,
                depth_img=depth,
                T_to_cam0=T_to_cam0,
            )
        )

    roi = None
    if args.roi is not None:
        roi = parse_roi_arg(args.roi)
        print(f"  [ROI] 수동 입력: {roi}")
    elif args.roi_interactive:
        roi = select_roi_interactive(camera_data_list[0].color_img)
        print(f"  [ROI] 인터랙티브 선택: {roi}")
    if roi is not None:
        save_roi_debug(
            camera_data_list[0].color_img,
            roi,
            os.path.join(args.output_dir, f"roi_cam0_{args.frame_id}.png"),
        )

    print("\n[2/5] 점군 통합")
    merged_pcd = PointCloudProcessor.merge_pointclouds(camera_data_list, voxel_size=args.voxel_size)
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "scene_merged.ply"), merged_pcd)

    if roi is not None:
        print("\n[2.5/5] ROI 기반 3D crop")
        min_xyz, max_xyz = PointCloudProcessor.roi_3d_from_cam0(
            camera_data_list[0].depth_img,
            camera_data_list[0].intrinsics.K,
            camera_data_list[0].intrinsics.depth_scale,
            roi,
            margin=args.roi_margin_m,
        )
        merged_pcd = PointCloudProcessor.filter_by_3d_box(merged_pcd, min_xyz, max_xyz)
        if len(merged_pcd.points) == 0:
            raise RuntimeError("ROI crop 이후 점군이 비었습니다. ROI를 조금 넓혀주세요.")
        o3d.io.write_point_cloud(os.path.join(args.output_dir, "scene_merged_roi.ply"), merged_pcd)

    print("\n[3/5] 테이블 평면 제거")
    objects_pcd, table_plane = PointCloudProcessor.remove_table_plane(merged_pcd)
    objects_pcd = PointCloudProcessor.filter_by_plane_height(
        objects_pcd,
        table_plane,
        min_height=args.min_height_m,
        max_height=args.max_height_m,
    )
    if len(objects_pcd.points) == 0:
        raise RuntimeError("테이블 높이 필터 이후 남은 점이 없습니다. max_height_m 값을 늘려보세요.")
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "objects_no_table.ply"), objects_pcd)

    proposals = []
    if roi is None and not args.disable_auto_detect:
        print("\n[3.5/5] 자동 object detector")
        mask_proposals = AutomaticObjectDetector.detect_from_cam0(
            camera_data_list[0],
            table_plane,
            min_height=args.min_height_m,
            max_height=args.max_height_m,
            roi_margin_m=args.roi_margin_m,
            min_area_px=args.detector_min_area_px,
            max_proposals=args.detector_max_proposals,
            output_dir=args.output_dir,
            frame_id=args.frame_id,
        )
        cluster_proposals = AutomaticObjectDetector.detect_from_clusters(
            objects_pcd,
            camera_data_list[0],
            roi_margin_m=args.roi_margin_m,
            min_points=args.detector_min_points,
            max_proposals=args.detector_max_proposals,
        )
        proposals = AutomaticObjectDetector.merge_proposals(
            mask_proposals + cluster_proposals,
            max_proposals=args.detector_max_proposals,
        )
        print(f"  최종 proposal 병합: {len(proposals)}개")
        for prop in proposals:
            x1, y1, x2, y2 = prop.roi
            print(
                f"    merged proposal {prop.proposal_id}: "
                f"roi=({x1},{y1},{x2},{y2}) area={prop.area_px}px score={prop.score:.1f}"
            )

    print("\n[4/5] 포즈 추정")
    ref_pcd_orig = loader.load_reference_pcd()
    ref_mesh_orig = loader.load_reference_mesh(print_info=True)

    prepared_ref = None
    ref_pcd_match = ref_pcd_orig
    ref_mesh_match = ref_mesh_orig
    matching_object_size_m = None
    matching_narrow_scale_search = getattr(args, "narrow_scale_search", False)

    if not getattr(args, "disable_auto_prepare_ref", False):
        estimated_size_m = ReferencePreprocessor.estimate_object_size_m(
            objects_pcd,
            proposals=proposals,
            min_points=args.detector_min_points,
        )
        if estimated_size_m is not None:
            print(f"  [AUTO] 촬영 이미지 기반 object size 추정: {estimated_size_m*1000:.1f} mm")
        else:
            print("  [AUTO] object size 자동 추정 실패 -> center-only reference 준비")

        prepared_ref = ReferencePreprocessor.prepare_reference(
            ref_pcd_orig,
            ref_mesh_orig,
            output_dir=args.output_dir,
            target_size_m=estimated_size_m,
            save_glb=not getattr(args, "disable_save_prepared_ref", False),
        )
        ref_pcd_match = prepared_ref.pcd
        ref_mesh_match = prepared_ref.mesh
        matching_object_size_m = None
        if estimated_size_m is not None:
            matching_narrow_scale_search = True

    proposal_score = None
    second_score = None
    if proposals:
        try:
            best = estimate_pose_with_proposals(
                objects_pcd,
                ref_pcd_match,
                ref_mesh_match,
                proposals,
                camera_data_list,
                argparse.Namespace(
                    **{
                        **vars(args),
                        "object_size_m": matching_object_size_m,
                        "narrow_scale_search": matching_narrow_scale_search,
                    }
                ),
            )
            pose = best["pose"]
            model_aligned = best["model"]
            scale = best["scale"]
            proposal_score = best.get("score")
            second_score = best.get("second_score")
            prop = best["proposal"]
            x1, y1, x2, y2 = prop.roi
            print(
                f"  선택된 proposal {prop.proposal_id}: "
                f"roi=({x1},{y1},{x2},{y2})  total_score={best['score']:.4f}"
            )
            o3d.io.write_point_cloud(
                os.path.join(args.output_dir, "detector_selected_crop.ply"),
                best["cropped_pcd"],
            )
        except Exception as exc:
            print(f"  [WARNING] proposal 기반 정합 실패 -> 전체 object cloud로 fallback: {exc}")
            pose, model_aligned, scale = PoseEstimator.estimate_pose(
                objects_pcd,
                ref_pcd_match,
                voxel_size=args.voxel_size,
                object_size_m=matching_object_size_m,
                camera_data_list=camera_data_list,
                ref_mesh=ref_mesh_match,
                ref_sample_points=getattr(args, "ref_sample_points", 12000),
                max_scale_candidates=getattr(args, "max_scale_candidates", 8),
                max_pca_candidates=getattr(args, "max_pca_candidates", 8),
                narrow_scale_search=matching_narrow_scale_search,
            )
    else:
        if args.disable_auto_detect and roi is None:
            print("  [INFO] 자동 detector 비활성화 -> 전체 object cloud에 대해 직접 정합")
        pose, model_aligned, scale = PoseEstimator.estimate_pose(
            objects_pcd,
            ref_pcd_match,
            voxel_size=args.voxel_size,
            object_size_m=matching_object_size_m,
            camera_data_list=camera_data_list,
            ref_mesh=ref_mesh_match,
            ref_sample_points=getattr(args, "ref_sample_points", 12000),
            max_scale_candidates=getattr(args, "max_scale_candidates", 8),
            max_pca_candidates=getattr(args, "max_pca_candidates", 8),
            narrow_scale_search=matching_narrow_scale_search,
        )

    if prepared_ref is not None:
        scale *= prepared_ref.prep_scale
        pose.scale = float(scale)
    print_pose(pose)

    combined = merged_pcd + model_aligned
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "alignment_result.ply"), combined)
    o3d.io.write_point_cloud(os.path.join(args.output_dir, "object_pointcloud.ply"), model_aligned)

    print("\n[5/5] 검증 및 저장")
    save_pose(pose, os.path.join(args.output_dir, "pose_Reference_Matching.npz"))
    save_estimation_manifest(
        os.path.join(args.output_dir, "estimation_manifest.json"),
        data_dir=args.data_dir,
        capture_dir=str(loader.image_dir),
        glb_path=str(loader.glb_path),
        frame_id=args.frame_id,
        camera_data_list=camera_data_list,
        prepared_ref=prepared_ref,
        used_auto_detect=(roi is None and not args.disable_auto_detect),
        used_roi=(roi is not None),
    )
    save_sim_config(
        pose,
        model_aligned,
        os.path.join(args.output_dir, "object_pose_sim.json"),
        output_unit=args.output_unit,
    )
    save_aligned_glb(
        str(loader.glb_path),
        ref_pcd_orig,
        pose,
        scale,
        os.path.join(args.output_dir, "aligned_object.glb"),
        transform_mode="bake",
        output_unit=args.output_unit,
    )
    save_aligned_glb(
        str(loader.glb_path),
        ref_pcd_orig,
        pose,
        scale,
        os.path.join(args.output_dir, "aligned_object_node.glb"),
        transform_mode="node",
        output_unit=args.output_unit,
    )

    for cam_data in camera_data_list:
        cam_id = cam_data.intrinsics.cam_id
        PoseValidator.reprojection_check(
            model_aligned,
            cam_data,
            os.path.join(args.output_dir, f"reprojection_cam{cam_id}.png"),
        )

    if proposal_score is not None:
        summary = {
            "proposal_score": proposal_score,
            "second_score": second_score,
            "proposal_margin": float(proposal_score / second_score) if second_score and second_score > 0 else None,
        }
        with open(os.path.join(args.output_dir, "proposal_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  proposal 요약 저장: {os.path.join(args.output_dir, 'proposal_summary.json')}")

    print(f"\n  결과 저장 위치: {args.output_dir}/")

    if args.visualize:
        print("\n[6] 시각화")
        vis_script = os.path.join(os.path.dirname(__file__), "Obj_Step3_visualize_pose_result.py")
        subprocess.run([sys.executable, vis_script], check=False)

    print("=" * 60)
    return {
        "pose": pose,
        "scale": scale,
        "output_dir": args.output_dir,
        "frame_id": args.frame_id,
    }


def run_auto_best_frame(args):
    available_ids = list_available_frame_ids(args.data_dir)
    if not available_ids:
        raise RuntimeError("자동 프레임 탐색 실패: object_capture/cam0에서 프레임을 찾지 못했습니다.")

    target_ids = parse_frame_search_spec(args.frame_search, available_ids)
    if not target_ids:
        raise RuntimeError("자동 프레임 탐색 실패: --frame_search로 지정된 프레임이 없습니다.")

    print("=" * 60)
    print(" 자동 프레임 탐색 (auto_best)")
    print("=" * 60)
    print(f"  탐색 프레임: {target_ids}")

    base_output = args.output_dir
    per_frame_root = os.path.join(base_output, "frames")
    os.makedirs(per_frame_root, exist_ok=True)
    all_results = []

    for fid in target_ids:
        print("\n" + "-" * 60)
        print(f"[AUTO] frame {fid} 실행")
        print("-" * 60)
        sub_args = argparse.Namespace(**vars(args))
        sub_args.frame_mode = "single"
        sub_args.frame_id = fid
        sub_args.output_dir = os.path.join(per_frame_root, fid)
        try:
            res = run_pipeline(sub_args)
            pose = res["pose"]
            conf = pose_confidence_score(pose)
            all_results.append(
                {
                    "frame_id": fid,
                    "status": "ok",
                    "confidence": conf,
                    "pose": pose,
                    "output_dir": sub_args.output_dir,
                }
            )
            print(
                f"[AUTO] frame {fid} 성공  "
                f"conf={conf:.6f}  score={pose.score:.4f}  "
                f"depth={pose.depth_score:.4f} cov={pose.coverage:.4f} fit={pose.fitness:.4f}"
            )
            if getattr(args, "auto_stop_on_first", False):
                break
        except Exception as exc:
            all_results.append(
                {
                    "frame_id": fid,
                    "status": "fail",
                    "error": str(exc),
                    "output_dir": sub_args.output_dir,
                }
            )
            print(f"[AUTO] frame {fid} 실패: {exc}")

    ok_results = [r for r in all_results if r["status"] == "ok"]
    best = max(ok_results, key=lambda r: r["confidence"]) if ok_results else None
    final_dir = base_output
    os.makedirs(final_dir, exist_ok=True)

    if best is not None:
        best_dir = best["output_dir"]
        copy_names = [
            "aligned_object.glb",
            "aligned_object_node.glb",
            "object_pose_sim.json",
            "pose_Reference_Matching.npz",
            "alignment_result.ply",
            "object_pointcloud.ply",
            "objects_no_table.ply",
            "scene_merged.ply",
            "reprojection_cam0.png",
            "reprojection_cam1.png",
            "reprojection_cam2.png",
            "proposal_summary.json",
        ]
        for name in copy_names:
            src = os.path.join(best_dir, name)
            dst = os.path.join(final_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, dst)

    summary = {
        "mode": "auto_best",
        "selected_frame": best["frame_id"] if best is not None else None,
        "selected_confidence": best["confidence"] if best is not None else None,
        "status": "ok" if best is not None else "no_match",
        "results": [],
    }
    for r in all_results:
        item = {
            "frame_id": r["frame_id"],
            "status": r["status"],
            "output_dir": r["output_dir"],
        }
        if r["status"] == "ok":
            p = r["pose"]
            item.update(
                {
                    "confidence": r["confidence"],
                    "score": p.score,
                    "depth_score": p.depth_score,
                    "coverage": p.coverage,
                    "fitness": p.fitness,
                    "rmse": p.rmse,
                }
            )
        else:
            item["error"] = r.get("error", "")
        summary["results"].append(item)

    summary_path = os.path.join(final_dir, "auto_frame_selection.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("[AUTO] 실행 요약")
    print("=" * 60)
    if best is not None:
        print(f"  선택 프레임: {best['frame_id']}")
        print(f"  confidence: {best['confidence']:.6f}")
        print(f"  원본 결과 위치: {best['output_dir']}")
    else:
        print("  선택 프레임: 없음")
    print(f"  최종 결과 위치: {final_dir}")
    print(f"  요약 저장: {summary_path}")
    print("=" * 60)
    return {
        "selected_frame": best["frame_id"] if best is not None else None,
        "confidence": best["confidence"] if best is not None else None,
        "summary_path": summary_path,
        "output_dir": final_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_data = str(script_dir / "data")
    default_output = str(script_dir / "output")
    default_ext = str(script_dir / "data" / "cube_session_01" / "calib_out_cube")
    default_glb = str(script_dir / "data" / "reference_knife.glb")

    parser = argparse.ArgumentParser(description="멀티뷰 RGB-D 기반 물체 포즈 추정")
    parser.add_argument("--data_dir", default=default_data, help="데이터 디렉토리")
    parser.add_argument("--output_dir", default=default_output, help="결과 저장 디렉토리")
    parser.add_argument("--extrinsics_dir", default=default_ext, help="외부 파라미터 디렉토리")
    parser.add_argument("--glb_path", default=default_glb, help="GLB 모델 경로")
    parser.add_argument("--frame_id", default="000003", help="프레임 번호 (예: 000003)")
    parser.add_argument(
        "--frame_mode",
        choices=["single", "auto_best"],
        default="single",
        help="single: 단일 프레임 실행, auto_best: 여러 프레임 중 최고 신뢰 결과 선택",
    )
    parser.add_argument(
        "--frame_search",
        type=str,
        default=None,
        help="auto_best 탐색 프레임 (예: 000000-000005 또는 000000,000003,000005)",
    )
    parser.add_argument(
        "--auto_stop_on_first",
        action="store_true",
        help="auto_best에서 첫 성공 프레임 발견 즉시 중단",
    )
    parser.add_argument("--num_cameras", type=int, default=3, help="카메라 수")
    parser.add_argument("--voxel_size", type=float, default=0.003, help="복셀 크기 (m)")
    parser.add_argument(
        "--output_unit",
        choices=["m", "mm"],
        default="mm",
        help="시뮬레이션 산출물(JSON/GLB) 저장 단위",
    )
    parser.add_argument("--visualize", action="store_true", help="완료 후 시각화 자동 실행")
    parser.add_argument("--roi_interactive", action="store_true", help="cam0 이미지에서 마우스로 ROI 선택")
    parser.add_argument("--roi", type=str, default=None, help="ROI 픽셀 좌표 (cam0 기준): x1,y1,x2,y2")
    parser.add_argument("--roi_margin_m", type=float, default=0.03, help="ROI를 3D box로 확장할 때 추가 여유(m)")
    parser.add_argument(
        "--disable_auto_prepare_ref",
        action="store_true",
        help="촬영 이미지 기반 reference 중심/스케일 자동 준비를 끔",
    )
    parser.add_argument(
        "--disable_save_prepared_ref",
        action="store_true",
        help="자동 준비된 reference GLB/정보 파일 저장을 생략",
    )
    parser.add_argument(
        "--narrow_scale_search",
        action="store_true",
        help="scale 탐색 범위를 좁혀 더 빠르게 실행",
    )
    parser.add_argument(
        "--ref_sample_points",
        type=int,
        default=12000,
        help="레퍼런스 모델에서 사용할 점 수. 작을수록 빠르지만 정합 안정성은 다소 낮아질 수 있음",
    )
    parser.add_argument(
        "--max_scale_candidates",
        type=int,
        default=8,
        help="평가할 최대 scale 후보 수. 작을수록 빠름",
    )
    parser.add_argument(
        "--max_pca_candidates",
        type=int,
        default=8,
        help="scale마다 평가할 최대 PCA 회전 후보 수. 작을수록 빠름",
    )
    parser.add_argument("--min_height_m", type=float, default=0.002, help="테이블 평면으로부터 최소 높이(m)")
    parser.add_argument("--max_height_m", type=float, default=0.12, help="테이블 평면으로부터 최대 높이(m)")
    parser.add_argument("--disable_auto_detect", action="store_true", help="ROI가 없을 때 자동 object detector 단계를 끔")
    parser.add_argument("--detector_min_area_px", type=int, default=1200, help="자동 detector가 유지할 최소 2D 면적(px)")
    parser.add_argument("--detector_max_proposals", type=int, default=6, help="자동 detector가 평가할 최대 proposal 수")
    parser.add_argument("--detector_min_points", type=int, default=80, help="proposal crop이 가져야 하는 최소 점 수")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.frame_mode == "auto_best":
        run_auto_best_frame(args)
    else:
        run_pipeline(args)
