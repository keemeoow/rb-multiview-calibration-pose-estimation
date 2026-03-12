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
    ) -> Tuple[PoseResult, o3d.geometry.PointCloud, float]:
        """
        색상 기반 클러스터 식별 → PCA 축 정렬 → ICP 정밀 정합.
        table_plane: [a,b,c,d] 테이블 평면 (법선은 카메라쪽). 모델 상하 판별에 사용.
        Returns: (PoseResult, aligned_model_pcd, scale)
        """
        print("\n" + "=" * 60)
        print("[포즈 추정] 색상 식별 + PCA 정렬 + ICP 정합")
        print("=" * 60)

        ref_pts = np.asarray(ref_pcd.points)
        R_ref, ref_eig = PoseEstimator._pca_axes(ref_pts)
        ref_ratios = np.sort(ref_eig / ref_eig.max())[::-1]
        # PCA 최장축 기준 범위 (AABB보다 정확)
        ref_centered = ref_pts - ref_pts.mean(axis=0)
        ref_proj = ref_centered @ R_ref
        ref_pca_extents = ref_proj.max(axis=0) - ref_proj.min(axis=0)
        ref_longest = ref_pca_extents.max()
        print(f"  레퍼런스: {len(ref_pts)} pts, PCA longest={ref_longest:.3f}m")

        # DBSCAN 클러스터링
        labels = np.array(objects_pcd.cluster_dbscan(
            eps=0.015, min_points=50, print_progress=False
        ))
        if len(labels) == 0 or labels.max() < 0:
            raise RuntimeError("클러스터를 찾을 수 없습니다")

        unique_labels = np.unique(labels[labels >= 0])
        print(f"  {len(unique_labels)}개 클러스터 발견")

        # 색상으로 대상 클러스터 식별
        target_label, yellow_ratio = PoseEstimator._find_target_cluster(
            objects_pcd, labels, unique_labels
        )

        if target_label >= 0 and yellow_ratio > 0.3:
            candidate_labels = [target_label]
            print(f"  색상 매칭: cluster {target_label} (yellow={yellow_ratio:.0%})")
        else:
            candidate_labels = [l for l in unique_labels
                                if np.sum(labels == l) >= 100]
            print(f"  색상 매칭 실패 → 전체 {len(candidate_labels)}개 탐색")

        # 각 후보 클러스터에 대해 PCA 정렬 + ICP 시도
        best_score = -1
        best_result = None

        for label in candidate_labels:
            cluster_idx = np.where(labels == label)[0]
            cluster = objects_pcd.select_by_index(cluster_idx)
            cluster_pts = np.asarray(cluster.points)
            R_cl, _ = PoseEstimator._pca_axes(cluster_pts)
            cl_centered = cluster_pts - cluster_pts.mean(axis=0)
            cl_proj = cl_centered @ R_cl
            cl_pca_extents = cl_proj.max(axis=0) - cl_proj.min(axis=0)
            cluster_longest = cl_pca_extents.max()

            # 스케일 추정 (PCA 최장축 기준)
            scale = cluster_longest / ref_longest
            if label == target_label:
                if scale < 0.05 or scale > 0.5:
                    continue
            else:
                if scale < 0.10 or scale > 0.30:
                    continue

            # 형상 유사도
            _, cl_eig = PoseEstimator._pca_axes(cluster_pts)
            cl_ratios = np.sort(cl_eig / cl_eig.max())[::-1]
            shape_sim = 1.0 - np.mean(np.abs(cl_ratios - ref_ratios))

            # 비균일 스케일링: PCA 축별로 클러스터에 맞춤
            # ref 모델을 PCA 공간에서 축별 스케일 적용 후 다시 원래 공간으로
            ref_center = ref_pts.mean(axis=0)
            axis_scales = cl_pca_extents[np.argsort(cl_pca_extents)[::-1]] / \
                          ref_pca_extents[np.argsort(ref_pca_extents)[::-1]]
            # PCA 정렬 순서: 큰 축 → 작은 축
            ref_sorted_idx = np.argsort(ref_pca_extents)[::-1]
            S_pca = np.diag(axis_scales)  # PCA 공간에서의 스케일 행렬
            # ref_pts → PCA → scale → back
            ref_in_pca = (ref_pts - ref_center) @ R_ref
            ref_scaled_pca = ref_in_pca @ S_pca
            ref_scaled_pts = ref_scaled_pca @ R_ref.T + ref_center

            model_scaled = o3d.geometry.PointCloud()
            model_scaled.points = o3d.utility.Vector3dVector(ref_scaled_pts)
            if ref_pcd.has_colors():
                model_scaled.colors = ref_pcd.colors
            print(f"      축별 스케일: ({axis_scales[0]:.4f}, {axis_scales[1]:.4f}, {axis_scales[2]:.4f})")

            # FPFH 글로벌 정합 + PCA 4가지 조합 → 총 5개 초기 자세 후보
            cluster.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
            )

            # FPFH 글로벌 정합 (PCA 축 모호성 없이 직접 매칭)
            fpfh_result = PoseEstimator._fpfh_global_registration(
                o3d.geometry.PointCloud(model_scaled), cluster, voxel_size
            )
            print(f"      FPFH: fitness={fpfh_result.fitness:.4f}, "
                  f"rmse={fpfh_result.inlier_rmse:.6f}")

            # PCA 4가지 + FPFH 1가지 = 5 후보
            candidate_Ts = PoseEstimator._pca_candidate_transforms(
                np.asarray(model_scaled.points), cluster_pts
            )
            # FPFH 결과를 추가 후보로
            candidate_Ts.append(fpfh_result.transformation)

            best_orient_score = -1
            best_orient_T = None
            best_orient_icp = None

            for ci, cand_T in enumerate(candidate_Ts):
                model_cand = o3d.geometry.PointCloud(model_scaled)
                model_cand.transform(cand_T)
                model_cand.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
                )

                icp_result = o3d.pipelines.registration.registration_icp(
                    model_cand, cluster,
                    max_correspondence_distance=voxel_size * 3,
                    init=np.eye(4),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=200,
                    ),
                )

                orient_score = icp_result.fitness
                tag = "FPFH" if ci == 4 else f"PCA{ci}"
                print(f"      {tag}: fitness={icp_result.fitness:.4f}, "
                      f"rmse={icp_result.inlier_rmse:.6f}")

                if orient_score > best_orient_score:
                    best_orient_score = orient_score
                    best_orient_T = cand_T
                    best_orient_icp = icp_result

            # 색상 보너스
            color_bonus = 0.0
            if objects_pcd.has_colors():
                cl_colors = np.asarray(objects_pcd.colors)[cluster_idx]
                yellow = ((cl_colors[:, 0] > 0.4) &
                          (cl_colors[:, 1] > 0.3) &
                          (cl_colors[:, 2] < 0.35))
                color_bonus = (yellow.sum() / len(cl_colors)) * 0.5

            score = best_orient_icp.fitness * shape_sim + color_bonus

            print(f"    cluster {label}: {len(cluster_pts)} pts, "
                  f"scale={scale:.3f}, shape={shape_sim:.3f}, "
                  f"ICP={best_orient_icp.fitness:.4f}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best_result = {
                    "icp_T": best_orient_icp.transformation,
                    "init_T": best_orient_T,
                    "scale": scale,
                    "model_scaled": model_scaled,
                    "cluster": cluster,
                    "label": label,
                }

        if best_result is None:
            raise RuntimeError("매칭 가능한 클러스터를 찾지 못했습니다")

        # 최적 클러스터에 정밀 ICP
        print(f"\n  최적: cluster {best_result['label']}, "
              f"scale={best_result['scale']:.4f}, score={best_score:.4f}")

        model_final = o3d.geometry.PointCloud(best_result["model_scaled"])
        model_final.transform(best_result["init_T"])

        for p in [model_final, best_result["cluster"]]:
            p.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )

        fine_result = o3d.pipelines.registration.registration_icp(
            model_final, best_result["cluster"],
            max_correspondence_distance=voxel_size * 3,
            init=best_result["icp_T"],
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=300,
            ),
        )
        print(f"  정밀 ICP: fitness={fine_result.fitness:.4f}, "
              f"RMSE={fine_result.inlier_rmse:.6f}")

        # 최종 정합 모델 생성
        model_aligned = o3d.geometry.PointCloud(model_final)
        model_aligned.transform(fine_result.transformation)

        # 테이블 위 보정: 모델 하단이 테이블 평면에 닿도록 이동
        if table_plane is not None:
            aligned_pts = np.asarray(model_aligned.points)
            normal = table_plane[:3]
            signed_dists = aligned_pts @ normal + table_plane[3]
            min_dist = signed_dists.min()
            if min_dist > 0.001:
                shift = -normal * min_dist
                aligned_pts += shift
                model_aligned.points = o3d.utility.Vector3dVector(aligned_pts)
                print(f"  테이블 보정: {min_dist:.4f}m → 하단을 테이블에 맞춤")

        model_aligned.paint_uniform_color([1.0, 0.0, 0.0])

        # PoseResult 생성
        T = fine_result.transformation
        R = T[:3, :3]
        t = T[:3, 3]
        rot = Rotation.from_matrix(R)

        pose = PoseResult(
            translation=t,
            rotation_matrix=R,
            euler_xyz_deg=rot.as_euler("xyz", degrees=True),
            quaternion_xyzw=rot.as_quat(),
            transform_4x4=T.copy(),
            fitness=fine_result.fitness,
            rmse=fine_result.inlier_rmse,
            method="Reference Matching",
        )
        return pose, model_aligned, best_result["scale"]


# =============================================================================
# 5. 재투영 검증
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

    # --- 점군 통합 ---
    print("\n[2/5] 점군 통합")
    merged_pcd = PointCloudProcessor.merge_pointclouds(
        camera_data_list, voxel_size=args.voxel_size
    )
    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "scene_merged.ply"), merged_pcd
    )

    # --- 테이블 제거 ---
    print("\n[3/5] 테이블 평면 제거")
    objects_pcd, table_plane = PointCloudProcessor.remove_table_plane(merged_pcd)
    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "objects_no_table.ply"), objects_pcd
    )

    # --- 레퍼런스 매칭 포즈 추정 ---
    print("\n[4/5] 포즈 추정")
    ref_pcd = loader.load_reference_pcd()
    pose, model_aligned, scale = PoseEstimator.estimate_pose(
        objects_pcd, ref_pcd, voxel_size=args.voxel_size,
        table_plane=table_plane,
    )
    print_pose(pose)

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

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_pipeline(args)
