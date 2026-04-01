#!/usr/bin/env python3
"""
=============================================================================
멀티뷰 RGB-D 카메라 기반 물체 6DoF 포즈 추정
=============================================================================

[원리]
  1) 3대의 RGB-D 카메라(cam0/1/2)의 내부 파라미터(K, D, depth_scale)와
     외부 파라미터(T_C0_C1, T_C0_C2)를 이용해 각 카메라의 RGB-D 점군을
     cam0 좌표계로 통합(merge)한다.
  2) RANSAC으로 테이블 평면을 제거하여 물체 후보만 남긴다.
  3) DBSCAN 클러스터링 후, 물체 고유 색상(예: 노란색)을 기준으로
     대상 클러스터를 식별한다.
  4) 레퍼런스 3D 모델(PLY/GLB)과 식별된 클러스터 사이에서:
     - PCA 주축 정렬로 초기 자세를 추정하고
     - ICP(Point-to-Plane)로 정밀 정합하여 최종 6DoF 포즈를 얻는다.
  5) 정합 결과를 각 카메라 이미지에 재투영(reprojection)하여 시각 검증한다.

[데이터 구조]
  src3/data/
  ├── _intrinsics/
  │   ├── cam0.npz           # color_K, color_D, depth_scale_m_per_unit
  │   ├── cam1.npz
  │   └── cam2.npz
  ├── cube_session_01/
  │   └── calib_out_cube/
  │       ├── T_C0_C1.npy    # 4x4 cam1→cam0 변환행렬
  │       └── T_C0_C2.npy    # 4x4 cam2→cam0 변환행렬
  ├── object_capture/
  │   ├── cam0/              # rgb_NNNNNN.jpg, depth_NNNNNN.png
  │   ├── cam1/
  │   └── cam2/
  ├── reference_knife.ply    # 레퍼런스 3D 모델 (점군)
  └── reference_knife.glb    # 레퍼런스 3D 모델 (메시, PLY 없을 때 대체)

[설치]
  pip install numpy opencv-python open3d trimesh scipy matplotlib

[사용법]
  python3 src3/multiview_pose_estimation.py --visualize
  python3 src3/multiview_pose_estimation.py --frame_id 000005
  python3 src3/multiview_pose_estimation.py --data_dir ./custom_data
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

warnings.filterwarnings("ignore")

try:
    import open3d as o3d
except ImportError:
    sys.exit("[ERROR] open3d 설치 필요: pip install open3d")

try:
    import trimesh
except ImportError:
    sys.exit("[ERROR] trimesh 설치 필요: pip install trimesh")

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# =============================================================================
# 1. 데이터 클래스
# =============================================================================

@dataclass
class CameraIntrinsics:
    K: np.ndarray           # 3x3 내부 행렬
    D: np.ndarray           # 왜곡 계수
    depth_scale: float      # depth → 미터
    cam_id: int = 0


@dataclass
class CameraData:
    intrinsics: CameraIntrinsics
    color_img: np.ndarray
    depth_img: np.ndarray
    T_to_cam0: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class PoseResult:
    translation: np.ndarray     # [x, y, z] (m)
    rotation_matrix: np.ndarray # 3x3
    euler_xyz_deg: np.ndarray   # [rx, ry, rz] (deg)
    quaternion_xyzw: np.ndarray # [x, y, z, w]
    transform_4x4: np.ndarray  # 4x4
    fitness: float = 0.0
    rmse: float = 0.0
    method: str = ""


# =============================================================================
# 2. 데이터 로드
# =============================================================================

class DataLoader:
    def __init__(self, data_dir: str, frame_id: str = "000003",
                 extrinsics_dir: Optional[str] = None,
                 glb_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.intrinsics_dir = self.data_dir / "_intrinsics"
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
        K = self._get_key(data, ["color_K", "K", "camera_matrix"])
        D = self._get_key(data, ["color_D", "D", "dist_coeffs"])
        depth_scale = float(
            self._get_key(data, ["depth_scale_m_per_unit", "depth_scale"])
        )
        print(f"  [cam{cam_id}] K loaded, depth_scale={depth_scale:.6f}")
        return CameraIntrinsics(K=K, D=D, depth_scale=depth_scale, cam_id=cam_id)

    def load_extrinsics(self) -> Dict[str, np.ndarray]:
        extrinsics = {}
        for name in ["T_C0_C1", "T_C0_C2"]:
            path = self.extrinsics_dir / f"{name}.npy"
            if path.exists():
                T = np.load(str(path))
                assert T.shape == (4, 4)
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

        color_img = cv2.cvtColor(
            cv2.imread(str(color_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        print(f"  [cam{cam_id}] color={color_img.shape}, depth={depth_img.shape}")
        return color_img, depth_img

    def load_reference_pcd(self) -> o3d.geometry.PointCloud:
        ply_path = self.data_dir / "reference_knife.ply"
        if ply_path.exists():
            pcd = o3d.io.read_point_cloud(str(ply_path))
            print(f"  [REF] PLY 로드: {len(pcd.points)} pts")
            return pcd

        if self.glb_path.exists():
            scene_or_mesh = trimesh.load(str(self.glb_path))
            if isinstance(scene_or_mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(scene_or_mesh.dump())
            else:
                mesh = scene_or_mesh
            pts = mesh.sample(30000)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            print(f"  [REF] GLB → 점군: {len(pts)} pts")
            return pcd

        raise FileNotFoundError("레퍼런스 모델 없음 (PLY/GLB)")

    @staticmethod
    def _get_key(data, keys):
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"키를 찾을 수 없음: {keys}, 가능: {list(data.keys())}")


# =============================================================================
# 3. 점군 처리
# =============================================================================

class PointCloudProcessor:

    @staticmethod
    def depth_to_pointcloud(
        color_img: np.ndarray, depth_img: np.ndarray,
        K: np.ndarray, D: np.ndarray, depth_scale: float,
        min_depth: float = 0.1, max_depth: float = 3.0,
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

        if len(color_img.shape) == 3:
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
        depth_img: np.ndarray, K: np.ndarray, depth_scale: float,
        roi: Tuple[int, int, int, int], margin: float = 0.03,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """cam0 픽셀 ROI → 3D AABB. depth histogram으로 foreground peak만 사용."""
        x1, y1, x2, y2 = roi
        h, w = depth_img.shape[:2]
        x1, x2 = int(np.clip(x1, 0, w - 1)), int(np.clip(x2, 0, w - 1))
        y1, y2 = int(np.clip(y1, 0, h - 1)), int(np.clip(y2, 0, h - 1))
        patch = depth_img[y1:y2, x1:x2].astype(np.float64) * depth_scale
        z_vals = patch[patch > 0.05].ravel()
        if len(z_vals) == 0:
            raise RuntimeError("ROI 내에 유효한 depth 값이 없습니다.")

        # histogram으로 foreground depth peak 찾기
        counts, edges = np.histogram(z_vals, bins=100)
        peak_bin = np.argmax(counts)
        z_peak = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0
        # peak ± 10cm만 사용 (배경 제거)
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
        min_xyz: np.ndarray, max_xyz: np.ndarray,
    ) -> o3d.geometry.PointCloud:
        """3D AABB(cam0 좌표계)로 점군 필터링."""
        pts = np.asarray(pcd.points)
        mask = np.all((pts >= min_xyz) & (pts <= max_xyz), axis=1)
        return pcd.select_by_index(np.where(mask)[0])

    @staticmethod
    def merge_pointclouds(
        camera_data_list: list, voxel_size: float = 0.002,
    ) -> o3d.geometry.PointCloud:
        merged = o3d.geometry.PointCloud()

        for cam_data in camera_data_list:
            intr = cam_data.intrinsics
            pcd = PointCloudProcessor.depth_to_pointcloud(
                cam_data.color_img, cam_data.depth_img,
                intr.K, intr.D, intr.depth_scale,
            )
            pcd.transform(cam_data.T_to_cam0)
            print(f"  [cam{intr.cam_id}] {len(pcd.points)} pts")
            merged += pcd

        n_before = len(merged.points)
        merged = merged.voxel_down_sample(voxel_size=voxel_size)
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  [통합] {n_before} → {len(merged.points)} pts (voxel={voxel_size}m)")
        return merged

    @staticmethod
    def remove_table_plane(
        pcd: o3d.geometry.PointCloud, distance_threshold: float = 0.008,
    ) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
        """테이블 평면 제거. Returns: (objects_pcd, plane_model [a,b,c,d])
        plane_model: ax+by+cz+d=0, 법선 방향은 카메라(원점) 쪽."""
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
        )
        a, b, c, d = plane_model
        print(f"  평면: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        objects_pcd = pcd.select_by_index(inliers, invert=True)
        objects_pcd, _ = objects_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=1.5
        )
        print(f"  평면 제거: {len(pcd.points)} → {len(objects_pcd.points)} pts")

        # 법선이 카메라(원점) 쪽을 향하도록 방향 보정
        plane_normal = np.array([a, b, c])
        plane_point = -d * plane_normal / np.dot(plane_normal, plane_normal)
        if np.dot(plane_normal, -plane_point) < 0:
            plane_model = [-a, -b, -c, -d]
        return objects_pcd, np.array(plane_model)

    @staticmethod
    def filter_by_plane_height(
        pcd: o3d.geometry.PointCloud,
        plane_model: np.ndarray,
        min_height: float = 0.002,
        max_height: float = 0.12,
    ) -> o3d.geometry.PointCloud:
        """테이블 평면으로부터 일정 높이 범위의 점만 유지."""
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
            f"  높이 필터: {len(pcd.points)} → {len(filtered.points)} pts  "
            f"({min_height*1000:.0f}~{max_height*1000:.0f} mm)"
        )
        return filtered


# =============================================================================
# 4. 포즈 추정 (레퍼런스 모델 매칭)
# =============================================================================

class PoseEstimator:

    @staticmethod
    def _pca_axes(pts: np.ndarray):
        """PCA 주축 반환 (3x3 열벡터, 큰 고유값 순)"""
        cov = np.cov((pts - pts.mean(axis=0)).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        R = eigenvectors[:, idx]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return R, eigenvalues[idx]

    @staticmethod
    def _pca_candidate_transforms(src_pts, dst_pts):
        """
        src(모델)를 dst(클러스터)에 PCA 축 정렬.
        축 부호 모호성(4가지 조합)의 변환 행렬을 모두 반환.
        """
        R_src, _ = PoseEstimator._pca_axes(src_pts)
        R_dst, _ = PoseEstimator._pca_axes(dst_pts)
        src_center = src_pts.mean(axis=0)
        dst_center = dst_pts.mean(axis=0)

        candidates = []
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                R_flip = R_src.copy()
                R_flip[:, 0] *= s1
                R_flip[:, 1] *= s2
                R_flip[:, 2] = np.cross(R_flip[:, 0], R_flip[:, 1])

                R_align = R_dst @ R_flip.T
                t = dst_center - R_align @ src_center

                T = np.eye(4)
                T[:3, :3] = R_align
                T[:3, 3] = t
                candidates.append(T)

        return candidates

    @staticmethod
    def _fpfh_global_registration(source, target, voxel_size):
        """FPFH 특징 기반 RANSAC 글로벌 정합. PCA 축 모호성 없이 직접 정합."""
        radius_normal = voxel_size * 3
        radius_feature = voxel_size * 6

        # 법선 추정
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

        # FPFH 특징 추출
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )

        # RANSAC 글로벌 정합
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 3,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
        )
        return result

    @staticmethod
    def _find_target_cluster(objects_pcd, labels, unique_labels):
        """
        색상 기반 물체 식별: 노란색 비율이 가장 높은 클러스터를 반환.
        색상 정보가 없으면 -1 반환.
        """
        if not objects_pcd.has_colors():
            return -1, 0.0

        all_colors = np.asarray(objects_pcd.colors)
        best_label, best_ratio = -1, 0.0

        for label in unique_labels:
            idx = np.where(labels == label)[0]
            if len(idx) < 100:
                continue
            cl_colors = all_colors[idx]
            yellow = ((cl_colors[:, 0] > 0.4) &
                      (cl_colors[:, 1] > 0.3) &
                      (cl_colors[:, 2] < 0.35))
            ratio = yellow.sum() / len(cl_colors)
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = label

        return best_label, best_ratio

    @staticmethod
    def estimate_pose(
        objects_pcd: o3d.geometry.PointCloud,
        ref_pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.005,
        table_plane: Optional[np.ndarray] = None,
        object_size_m: Optional[float] = None,
        camera_data_list: Optional[list] = None,
        direct_match: bool = False,  # kept for API compat, ignored
    ) -> Tuple[PoseResult, o3d.geometry.PointCloud, float]:
        """
        1) GLB를 실제 크기로 스케일링 (object_size_m 또는 클러스터 자동 추정)
        2) 크기가 가장 비슷한 DBSCAN 클러스터 자동 선택
        3) Centroid 초기화 + PCA 후보 ICP
        4) 누적 변환 올바르게 적용
        """
        print("\n" + "=" * 60)
        print("[포즈 추정] 크기 기반 자동 클러스터 선택 + ICP")
        print("=" * 60)

        def _robust_extents(proj_pts, q_low=5.0, q_high=95.0):
            lo = np.percentile(proj_pts, q_low, axis=0)
            hi = np.percentile(proj_pts, q_high, axis=0)
            return hi - lo

        # GLB 크기 + 형상 비율 계산
        ref_pts = np.asarray(ref_pcd.points)
        R_ref, ref_eig = PoseEstimator._pca_axes(ref_pts)
        ref_proj = (ref_pts - ref_pts.mean(axis=0)) @ R_ref
        ref_extents = _robust_extents(ref_proj)
        ref_longest = ref_extents.max()
        # PCA 고유값 비율 (형상 지문) — 정육면체는 ≈(1,1,1), 원통은 ≈(1,1,0.6)
        ref_eig_sorted = np.sort(ref_eig / ref_eig.max())[::-1]
        print(f"  GLB: {len(ref_pts)} pts, longest={ref_longest:.3f}m  "
              f"PCA비율={np.round(ref_eig_sorted, 2)}")
        if object_size_m is None and ref_longest > 0.20:
            print("  [경고] GLB 단위가 실제보다 크게 보입니다. "
                  "--object_size_m 지정이 훨씬 안정적입니다.")

        # 실제 크기 기준 균일 스케일 팩터
        if object_size_m is not None:
            fixed_scale = object_size_m / ref_longest
            print(f"  스케일 고정: {fixed_scale:.4f}  ({object_size_m*100:.1f}cm)")
        else:
            fixed_scale = None

        # GLB를 원점 중심으로 정규화
        ref_center_local = ref_pts.mean(axis=0)
        ref_pts_local = ref_pts - ref_center_local  # 원점 기준

        def _make_scaled_pcd(s):
            """GLB를 원점 기준 균일 스케일링한 pcd 반환."""
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(ref_pts_local * s)
            if ref_pcd.has_colors():
                pcd.colors = ref_pcd.colors
            return pcd

        def _run_icp(src, tgt, init_T, max_dist):
            for p in [src, tgt]:
                if not p.has_normals():
                    p.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist * 2, max_nn=30)
                    )
            return o3d.pipelines.registration.registration_icp(
                src, tgt,
                max_correspondence_distance=max_dist,
                init=init_T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=200,
                ),
            )

        edge_context = None
        if camera_data_list is not None and len(camera_data_list) > 0:
            edge_context = PoseRefiner2D.build_edge_context(camera_data_list)

        # DBSCAN 클러스터링
        labels = np.array(objects_pcd.cluster_dbscan(
            eps=0.012, min_points=30, print_progress=False
        ))
        unique_labels = np.unique(labels[labels >= 0])
        print(f"  {len(unique_labels)}개 클러스터")

        best_score = -1
        best_result = None

        for label in unique_labels:
            cl_idx = np.where(labels == label)[0]
            if len(cl_idx) < 30:
                continue
            cluster = objects_pcd.select_by_index(cl_idx)
            cl_pts = np.asarray(cluster.points)

            # 클러스터 크기 + 형상 비율 계산
            R_cl, cl_eig = PoseEstimator._pca_axes(cl_pts)
            cl_proj = (cl_pts - cl_pts.mean(axis=0)) @ R_cl
            cl_extents = _robust_extents(cl_proj)
            cl_longest = cl_extents.max()
            cl_eig_sorted = np.sort(cl_eig / cl_eig.max())[::-1]
            # GLB와 클러스터 형상 유사도 (1=완벽, 0=완전 다름)
            shape_sim = float(1.0 - np.mean(np.abs(cl_eig_sorted - ref_eig_sorted)))

            # 스케일 결정
            if fixed_scale is not None:
                s = fixed_scale
                # 실제 깊이 노이즈와 멀티뷰 병합 오차를 고려해 허용 범위를 넓게 둔다.
                ratio = cl_longest / object_size_m
                if ratio < 0.3 or ratio > 2.2:
                    continue
            else:
                s = cl_longest / ref_longest
                if s < 0.02 or s > 0.8:
                    continue

            model_s = _make_scaled_pcd(s)
            model_pts_s = np.asarray(model_s.points)  # 원점 기준

            # centroid 초기화: 모델을 클러스터 중심으로 이동
            cl_center = cl_pts.mean(axis=0)
            T_init = np.eye(4)
            T_init[:3, 3] = cl_center  # 원점 → 클러스터 중심

            # PCA 4후보 (클러스터 중심 기준)
            cand_Ts = PoseEstimator._pca_candidate_transforms(model_pts_s, cl_pts)
            cand_Ts.append(T_init)  # centroid 후보도 포함

            cluster.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
            )
            max_dist = max(voxel_size * 4, object_size_m * 0.3 if object_size_m else 0.02)

            best_cand_score = -1
            best_cand_T = T_init
            best_cand_icp = None
            for cand_T in cand_Ts:
                mc = o3d.geometry.PointCloud(model_s)
                mc.transform(cand_T)
                res = _run_icp(mc, o3d.geometry.PointCloud(cluster), np.eye(4), max_dist)
                if res.fitness > best_cand_score:
                    best_cand_score = res.fitness
                    best_cand_T = cand_T
                    best_cand_icp = res

            # shape_sim이 낮으면 형상이 GLB와 다른 물체 → 강하게 패널티
            # ICP fitness만 보면 원통에도 높게 나오므로 shape_sim을 곱해서 필터링
            edge_cost = None
            edge_score = 1.0
            if edge_context is not None and best_cand_icp is not None:
                model_eval = o3d.geometry.PointCloud(model_s)
                model_eval.transform(np.array(best_cand_icp.transformation) @ best_cand_T)
                edge_cost, mean_vis = PoseRefiner2D.projection_edge_cost(
                    np.asarray(model_eval.points), edge_context, sample_size=1500
                )
                edge_score = float(np.exp(-edge_cost / 10.0) * np.clip(mean_vis / 0.10, 0.2, 1.0))

            combined_score = best_cand_score * (shape_sim ** 2) * edge_score

            edge_msg = ""
            if edge_cost is not None:
                edge_msg = f"  edge={edge_cost:.2f}px"
            print(f"  cluster {label}: {len(cl_pts)}pts  longest={cl_longest*100:.1f}cm  "
                  f"PCA비율={np.round(cl_eig_sorted,2)}  "
                  f"shape={shape_sim:.3f}  ICP={best_cand_score:.4f}{edge_msg}  "
                  f"score={combined_score:.4f}")

            if combined_score > best_score:
                best_score = combined_score
                best_result = {
                    "model_s": model_s,
                    "cand_T": best_cand_T,
                    "icp_res": best_cand_icp,
                    "cluster": cluster,
                    "scale": s,
                    "cl_center": cl_center,
                    "edge_cost": edge_cost,
                }

        if best_result is None:
            raise RuntimeError("매칭 가능한 클러스터를 찾지 못했습니다. "
                               "--object_size_m 범위를 확인하세요.")

        # 최적 클러스터로 정밀 ICP
        model_s = best_result["model_s"]
        cluster = best_result["cluster"]
        cand_T = best_result["cand_T"]
        icp0 = best_result["icp_res"]
        max_dist = max(voxel_size * 4, object_size_m * 0.3 if object_size_m else 0.02)

        model_after_cand = o3d.geometry.PointCloud(model_s)
        model_after_cand.transform(cand_T)
        fine = _run_icp(model_after_cand, o3d.geometry.PointCloud(cluster),
                        icp0.transformation, max_dist)
        print(f"\n  정밀 ICP: fitness={fine.fitness:.4f}  RMSE={fine.inlier_rmse:.6f}")

        # 누적 변환: T_fine @ T_cand (model_s 원점 기준)
        T_total = np.array(fine.transformation) @ cand_T

        # centroid fallback: fitness 너무 낮으면 centroid만 적용
        model_aligned = _make_scaled_pcd(best_result["scale"])
        if fine.fitness < 0.05:
            print("  [경고] ICP fitness 낮음 → centroid 정렬만 적용")
            T_total = np.eye(4)
            T_total[:3, 3] = best_result["cl_center"]
        model_aligned.transform(T_total)
        model_aligned.paint_uniform_color([1.0, 0.0, 0.0])

        if edge_context is not None:
            final_edge_cost, final_vis = PoseRefiner2D.projection_edge_cost(
                np.asarray(model_aligned.points), edge_context, sample_size=1500
            )
            print(f"  최종 멀티뷰 edge cost={final_edge_cost:.2f}px  visibility={final_vis:.3f}")

        # Pose 추출
        R = T_total[:3, :3].copy()
        U, _, Vt = np.linalg.svd(R)
        R = (U @ Vt).copy()
        t = np.asarray(model_aligned.points).mean(axis=0).copy()
        rot = Rotation.from_matrix(R)
        T_out = np.eye(4)
        T_out[:3, :3] = R
        T_out[:3, 3] = t

        pose = PoseResult(
            translation=t,
            rotation_matrix=R,
            euler_xyz_deg=rot.as_euler("xyz", degrees=True),
            quaternion_xyzw=rot.as_quat(),
            transform_4x4=T_out,
            fitness=fine.fitness,
            rmse=fine.inlier_rmse,
            method="Reference Matching",
        )
        print(f"  위치: {np.round(t, 4)} m  회전: {np.round(rot.as_euler('xyz', degrees=True), 1)} deg")
        return pose, model_aligned, best_result["scale"]


# =============================================================================
# 5. 2D 엣지 기반 포즈 정밀화
# =============================================================================

class PoseRefiner2D:
    """
    ICP 결과를 이미지 엣지에 맞춰 정밀화.
    - 위치 탐색 범위: ICP 결과 ±search_r (기본 3cm)
    - 회전 탐색 범위: ICP 결과 ±rot_deg (기본 30°)
    """

    @staticmethod
    def build_edge_context(camera_data_list: list) -> Dict[str, list]:
        dist_maps, inv_T_list, K_list, img_shapes = [], [], [], []
        for cam_data in camera_data_list:
            gray = cv2.cvtColor(cam_data.color_img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
            edges = cv2.Canny(blur, 30, 90)
            edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
            dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
            dist_maps.append(dist.astype(np.float32))
            inv_T_list.append(np.linalg.inv(cam_data.T_to_cam0).astype(np.float64))
            K_list.append(cam_data.intrinsics.K.astype(np.float64))
            img_shapes.append(cam_data.color_img.shape[:2])
        return {
            "dist_maps": dist_maps,
            "inv_T_list": inv_T_list,
            "K_list": K_list,
            "img_shapes": img_shapes,
        }

    @staticmethod
    def projection_edge_cost(
        pts_cam0: np.ndarray,
        edge_context: Dict[str, list],
        sample_size: Optional[int] = 1500,
        min_visible_ratio: float = 0.03,
    ) -> Tuple[float, float]:
        pts = np.asarray(pts_cam0, dtype=np.float64)
        if len(pts) == 0:
            return 1e6, 0.0

        if sample_size is not None and len(pts) > sample_size:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), sample_size, replace=False)
            pts = pts[idx]

        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        total_cost = 0.0
        visibilities = []

        for inv_T, K, dm, (h, w) in zip(
            edge_context["inv_T_list"],
            edge_context["K_list"],
            edge_context["dist_maps"],
            edge_context["img_shapes"],
        ):
            pc = (inv_T @ pts_h.T)[:3].T
            valid = pc[:, 2] > 0.05
            if valid.sum() == 0:
                total_cost += 1e3
                visibilities.append(0.0)
                continue

            p = pc[valid]
            u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
            v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            visible_ratio = float(ok.sum()) / float(len(pts))
            visibilities.append(visible_ratio)

            if ok.sum() == 0:
                total_cost += 1e3
                continue

            mean_dist = float(dm[v[ok], u[ok]].mean())
            visibility_penalty = 120.0 * max(0.0, min_visible_ratio - visible_ratio) / max(min_visible_ratio, 1e-6)
            total_cost += mean_dist + visibility_penalty

        return total_cost / max(len(edge_context["dist_maps"]), 1), float(np.mean(visibilities))

    @staticmethod
    def refine(
        model_pcd: o3d.geometry.PointCloud,
        pose_init: "PoseResult",
        camera_data_list: list,
        n_samples: int = 2000,
        n_iter: int = 800,
        search_r: float = 0.02,    # 위치 탐색 반경 (m)
        rot_deg: float = 20.0,     # 회전 탐색 범위 (deg)
        output_dir: Optional[str] = None,
    ) -> Tuple["PoseResult", o3d.geometry.PointCloud]:
        from scipy.optimize import minimize

        print("\n" + "=" * 60)
        print("[2D 정밀화] GLB → 이미지 엣지 직접 정합")
        print("=" * 60)

        if pose_init.fitness < 0.05:
            print("  [경고] 3D 정합 fitness가 너무 낮아 2D 정밀화를 생략합니다.")
            return pose_init, model_pcd

        all_pts = np.asarray(model_pcd.points).astype(np.float64)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(all_pts), min(n_samples, len(all_pts)), replace=False)
        pts = all_pts[idx]
        centroid0 = pts.mean(axis=0).copy()
        pts_local = pts - centroid0  # 크기·형상 고정, 원점 기준

        edge_context = PoseRefiner2D.build_edge_context(camera_data_list)

        rv0 = Rotation.from_matrix(pose_init.rotation_matrix).as_rotvec()
        rot_bound = np.deg2rad(rot_deg)

        # 파라미터: delta_rotvec(3) + delta_t(3) — ICP 결과 기준 상대값
        def _cost(delta):
            try:
                rv = rv0 + delta[:3]
                t  = centroid0 + delta[3:6]
                R  = Rotation.from_rotvec(rv).as_matrix()
                pts_c0 = pts_local @ R.T + t
                edge_cost, _ = PoseRefiner2D.projection_edge_cost(
                    pts_c0, edge_context, sample_size=None
                )
                reg_t = 1.5 * np.linalg.norm(delta[3:6]) / max(search_r, 1e-6)
                reg_r = 0.5 * np.linalg.norm(delta[:3]) / max(rot_bound, 1e-6)
                return edge_cost + reg_t + reg_r
            except Exception:
                return 1e6

        x0 = np.zeros(6)
        c0 = _cost(x0)
        c0_edge, c0_vis = PoseRefiner2D.projection_edge_cost(pts, edge_context, sample_size=None)
        print(f"  초기 cost={c0:.3f}  edge={c0_edge:.3f}px  vis={c0_vis:.3f}  centroid={np.round(centroid0, 3)}")

        # 탐색 범위 제한 (bounds)
        b_t = search_r
        b_r = rot_bound
        bounds = [(-b_r, b_r)] * 3 + [(-b_t, b_t)] * 3

        result = minimize(_cost, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": n_iter, "ftol": 1e-5, "gtol": 1e-5})
        c1 = result.fun
        rv_test = rv0 + result.x[:3]
        t_test = centroid0 + result.x[3:6]
        R_test = Rotation.from_rotvec(rv_test).as_matrix()
        pts_test = pts_local @ R_test.T + t_test
        c1_edge, c1_vis = PoseRefiner2D.projection_edge_cost(pts_test, edge_context, sample_size=None)
        print(f"  최종 cost={c1:.3f}  edge={c1_edge:.3f}px  vis={c1_vis:.3f}  (iter={result.nit})")

        hit_translation_bound = np.any(np.isclose(np.abs(result.x[3:6]), b_t, atol=max(1e-4, b_t * 0.05)))
        hit_rotation_bound = np.any(np.isclose(np.abs(result.x[:3]), b_r, atol=max(1e-3, b_r * 0.05)))

        if (
            (not result.success)
            or (c1 >= c0 * 0.97)
            or (c1_vis < max(0.05, c0_vis * 0.7))
            or hit_translation_bound
            or hit_rotation_bound
        ):
            print("  [경고] 유의미한 개선 없음 → ICP 결과 유지")
            return pose_init, model_pcd

        # 결과 적용
        rv_opt = rv0 + result.x[:3]
        t_opt  = centroid0 + result.x[3:6]
        R_raw  = Rotation.from_rotvec(rv_opt).as_matrix()
        U, _, Vt = np.linalg.svd(R_raw)
        R_opt  = (U @ Vt).copy()

        new_pts = pts_local @ R_opt.T + t_opt  # 샘플링된 점
        all_local = all_pts - centroid0
        all_new = all_local @ R_opt.T + t_opt  # 전체 점

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(all_new)
        new_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        rot_obj = Rotation.from_matrix(R_opt)
        t_final = all_new.mean(axis=0).copy()
        T_out = np.eye(4)
        T_out[:3, :3] = R_opt
        T_out[:3, 3] = t_final
        new_pose = PoseResult(
            translation=t_final,
            rotation_matrix=R_opt,
            euler_xyz_deg=rot_obj.as_euler("xyz", degrees=True),
            quaternion_xyzw=rot_obj.as_quat(),
            transform_4x4=T_out,
            fitness=pose_init.fitness,
            rmse=pose_init.rmse,
            method="Reference Matching + 2D Refinement",
        )
        print(f"  정밀화 위치: {np.round(t_final, 4)} m  "
              f"이동: {np.round(result.x[3:6]*100, 2)} cm  "
              f"회전 보정: {np.round(np.rad2deg(result.x[:3]), 1)} deg")

        if output_dir is not None:
            PoseRefiner2D._save_debug(
                all_new,
                camera_data_list,
                edge_context["dist_maps"],
                edge_context["inv_T_list"],
                edge_context["K_list"],
                output_dir,
            )
        return new_pose, new_pcd

    @staticmethod
    def _save_debug(pts_cam0, camera_data_list, dist_maps, inv_T_list, K_list, output_dir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pts_h = np.hstack([pts_cam0, np.ones((len(pts_cam0), 1))])
        for i, (cam_data, inv_T, K, dm) in enumerate(
                zip(camera_data_list, inv_T_list, K_list, dist_maps)):
            pc = (inv_T @ pts_h.T)[:3].T
            valid = pc[:, 2] > 0.05
            if valid.sum() == 0:
                continue
            p = pc[valid]
            u = (K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(np.int32)
            v = (K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(np.int32)
            h, w = cam_data.color_img.shape[:2]
            ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].imshow(cam_data.color_img)
            axes[0].scatter(u[ok], v[ok], c="lime", s=2, alpha=0.6)
            axes[0].set_title(f"cam{i} — 2D 정밀화")
            axes[0].axis("off")
            axes[1].imshow((dm < 5).astype(np.uint8) * 255, cmap="gray")
            axes[1].scatter(u[ok], v[ok], c="red", s=2, alpha=0.6)
            axes[1].set_title(f"cam{i} — 엣지맵")
            axes[1].axis("off")
            plt.tight_layout()
            out = os.path.join(output_dir, f"refine2d_cam{i}.png")
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"  [2D 디버그] {out}")


# =============================================================================
# 7. 재투영 검증
# =============================================================================

class PoseValidator:

    @staticmethod
    def reprojection_check(
        model_pcd: o3d.geometry.PointCloud,
        camera_data: CameraData,
        output_path: str,
    ):
        """정합된 모델(cam0 좌표계)을 이미지에 재투영"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        K = camera_data.intrinsics.K
        pts = np.asarray(model_pcd.points)

        # cam0 → cam_i 변환
        T_cam = np.linalg.inv(camera_data.T_to_cam0)
        pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam = (T_cam @ pts_hom.T)[:3]

        # 투영
        valid = pts_cam[2] > 0
        proj = K @ pts_cam[:, valid]
        u = (proj[0] / proj[2]).astype(int)
        v = (proj[1] / proj[2]).astype(int)

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


# =============================================================================
# 6. 결과 출력/저장
# =============================================================================

def print_pose(result: PoseResult):
    print(f"\n{'='*60}")
    print(f"포즈 추정 결과 [{result.method}]")
    print(f"{'='*60}")
    t = result.translation
    e = result.euler_xyz_deg
    print(f"  위치: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
    print(f"  회전: ({e[0]:+.1f}°, {e[1]:+.1f}°, {e[2]:+.1f}°)")
    print(f"  품질: fitness={result.fitness:.4f}, RMSE={result.rmse:.6f}m")


def save_pose(result: PoseResult, output_path: str):
    np.savez(
        output_path,
        translation=result.translation,
        rotation_matrix=result.rotation_matrix,
        euler_xyz_deg=result.euler_xyz_deg,
        quaternion_xyzw=result.quaternion_xyzw,
        transform_4x4=result.transform_4x4,
        fitness=result.fitness,
        rmse=result.rmse,
        method=result.method,
    )
    print(f"  포즈 저장: {output_path}")


def save_sim_config(result: PoseResult, model_pcd: o3d.geometry.PointCloud,
                    output_path: str):
    """시뮬레이션 프로그램용 JSON 파일 저장.
    OpenCV 좌표계 (cam0 기준): X→오른쪽, Y→아래, Z→앞.
    """
    import json

    obb = model_pcd.get_oriented_bounding_box()
    extent = np.sort(obb.extent)[::-1]  # [L, W, H] 내림차순

    t = result.translation
    e = result.euler_xyz_deg
    q = result.quaternion_xyzw  # [x, y, z, w]

    config = {
        "coordinate_frame": "OpenCV (cam0 ref): X-right, Y-down, Z-forward",
        "unit": "meter / degree",
        "position": {
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
        "transform_4x4": result.transform_4x4.tolist(),
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
            "x": round(float(obb.center[0]), 6),
            "y": round(float(obb.center[1]), 6),
            "z": round(float(obb.center[2]), 6),
        },
        "fitness": round(result.fitness, 4),
        "rmse_m": round(result.rmse, 6),
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  SIM 설정 저장: {output_path}")


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


# =============================================================================
# 7. 메인 파이프라인
# =============================================================================

def run_pipeline(args):
    print("=" * 60)
    print(" 멀티뷰 RGB-D 기반 물체 포즈 추정")
    print("=" * 60)

    # --- 데이터 로드 ---
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

        camera_data_list.append(CameraData(
            intrinsics=intrinsics[i],
            color_img=color,
            depth_img=depth,
            T_to_cam0=T_to_cam0,
        ))

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

    # --- 점군 통합 ---
    print("\n[2/5] 점군 통합")
    merged_pcd = PointCloudProcessor.merge_pointclouds(
        camera_data_list, voxel_size=args.voxel_size
    )
    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "scene_merged.ply"), merged_pcd
    )

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
        o3d.io.write_point_cloud(
            os.path.join(args.output_dir, "scene_merged_roi.ply"), merged_pcd
        )

    # --- 테이블 제거 ---
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
    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "objects_no_table.ply"), objects_pcd
    )

    # --- 포즈 추정 (크기 기반 자동 클러스터 선택 + ICP) ---
    print("\n[4/5] 포즈 추정")
    ref_pcd = loader.load_reference_pcd()
    pose, model_aligned, scale = PoseEstimator.estimate_pose(
        objects_pcd, ref_pcd, voxel_size=args.voxel_size,
        table_plane=table_plane,
        object_size_m=getattr(args, "object_size_m", None),
        camera_data_list=camera_data_list,
    )
    print_pose(pose)

    # --- 2D 엣지 정밀화 ---
    if not args.no_refine_2d:
        print("\n[4.5/5] 2D 엣지 정밀화")
        pose, model_aligned = PoseRefiner2D.refine(
            model_aligned, pose, camera_data_list,
            n_samples=2000,
            n_iter=getattr(args, "refine_2d_iter", 800),
            search_r=getattr(args, "refine_2d_search_r", 0.02),
            rot_deg=getattr(args, "refine_2d_rot_deg", 20.0),
            output_dir=args.output_dir,
        )
        print_pose(pose)
    else:
        print("\n[4.5/5] 2D 엣지 정밀화 생략 (--no_refine_2d)")

    # 정합 결과 저장
    combined = merged_pcd + model_aligned
    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "alignment_result.ply"), combined
    )
    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "object_pointcloud.ply"), model_aligned
    )

    # --- 검증 및 저장 ---
    print("\n[5/5] 검증 및 저장")
    os.makedirs(args.output_dir, exist_ok=True)
    save_pose(pose, os.path.join(args.output_dir, "pose_Reference_Matching.npz"))
    save_sim_config(pose, model_aligned,
                    os.path.join(args.output_dir, "object_pose_sim.json"))

    for cam_data in camera_data_list:
        cam_id = cam_data.intrinsics.cam_id
        PoseValidator.reprojection_check(
            model_aligned, cam_data,
            os.path.join(args.output_dir, f"reprojection_cam{cam_id}.png"),
        )

    print(f"\n  결과 저장 위치: {args.output_dir}/")

    # --- 시각화 ---
    if args.visualize:
        print("\n[6] 시각화")
        import subprocess
        vis_script = os.path.join(os.path.dirname(__file__), "Obj_Step3_visualize_pose_result.py")
        subprocess.run([sys.executable, vis_script])
    print("=" * 60)


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    _default_data = str(_script_dir / "data")
    _default_output = str(_script_dir / "output")
    _default_ext = str(_script_dir / "data" / "cube_session_01" / "calib_out_cube")
    _default_glb = str(_script_dir / "data" / "reference_knife.glb")

    parser = argparse.ArgumentParser(description="멀티뷰 RGB-D 기반 물체 포즈 추정")
    parser.add_argument("--data_dir", default=_default_data, help="데이터 디렉토리")
    parser.add_argument("--output_dir", default=_default_output, help="결과 저장 디렉토리")
    parser.add_argument("--extrinsics_dir", default=_default_ext, help="외부 파라미터 디렉토리")
    parser.add_argument("--glb_path", default=_default_glb, help="GLB 모델 경로")
    parser.add_argument("--frame_id", default="000003", help="프레임 번호 (예: 000003)")
    parser.add_argument("--num_cameras", type=int, default=3, help="카메라 수")
    parser.add_argument("--voxel_size", type=float, default=0.003, help="복셀 크기 (m)")
    parser.add_argument("--visualize", action="store_true", help="완료 후 시각화 자동 실행")
    parser.add_argument("--roi_interactive", action="store_true",
                        help="cam0 이미지에서 마우스로 ROI 선택")
    parser.add_argument("--roi", type=str, default=None,
                        help="ROI 픽셀 좌표 (cam0 기준): x1,y1,x2,y2")
    parser.add_argument("--roi_margin_m", type=float, default=0.03,
                        help="ROI를 3D box로 확장할 때 추가 여유(m)")
    parser.add_argument("--object_size_m", type=float, default=None,
                        help="물체 실제 최장변 길이(m). 지정 시 스케일 고정 (예: 0.05 = 5cm)")
    parser.add_argument("--min_height_m", type=float, default=0.002,
                        help="테이블 평면으로부터 최소 높이(m)")
    parser.add_argument("--max_height_m", type=float, default=0.12,
                        help="테이블 평면으로부터 최대 높이(m)")
    parser.add_argument("--no_refine_2d", action="store_true",
                        help="2D 엣지 정밀화를 생략")
    parser.add_argument("--refine_2d_iter", type=int, default=800,
                        help="2D 엣지 정밀화 최대 반복 횟수 (기본 800)")
    parser.add_argument("--refine_2d_search_r", type=float, default=0.02,
                        help="2D 정밀화 위치 탐색 반경(m)")
    parser.add_argument("--refine_2d_rot_deg", type=float, default=20.0,
                        help="2D 정밀화 회전 탐색 범위(deg)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_pipeline(args)
