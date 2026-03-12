#!/usr/bin/env python3
"""
파이프라인 각 단계별 중간 결과를 시각화하는 디버그 스크립트.
결과: src3/output/debug/ 에 단계별 이미지 저장.
"""
import sys, os, warnings
from pathlib import Path
import numpy as np
import cv2

warnings.filterwarnings("ignore")

import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from multiview_pose_estimation import (
        DataLoader, PointCloudProcessor, PoseEstimator, CameraData, PoseValidator
    )
except ModuleNotFoundError:
    # Fallback for repositories where the pipeline file keeps the Obj_ prefix.
    from Obj_Step2_multiview_pose_estimation import (
        DataLoader, PointCloudProcessor, PoseEstimator, CameraData, PoseValidator
    )
from scipy.spatial.transform import Rotation

DATA_DIR = str(SCRIPT_DIR / "data")
OUTPUT_DIR = SCRIPT_DIR / "output"
DEBUG_DIR = OUTPUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

EXTRINSICS_DIR = str(SCRIPT_DIR / "data" / "cube_session_01" / "calib_out_cube")
GLB_PATH = str(SCRIPT_DIR / "data" / "reference_knife.glb")


def plot_pcd_3views(pts, colors, title, save_name, extra_pts=None, extra_colors=None, extra_label=None):
    """점군을 3뷰(원근/탑/사이드)로 저장"""
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')

    views = [(25, -55, "Perspective"), (90, -90, "Top (XZ)"), (0, -90, "Side (XY)")]
    step = max(1, len(pts) // 5000)

    for idx, (elev, azim, vname) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_facecolor('#16213e')

        c = colors[::step] if colors is not None else 'silver'
        ax.scatter(pts[::step, 0], pts[::step, 2], pts[::step, 1],
                   c=c, s=0.5, alpha=0.4)

        if extra_pts is not None:
            step2 = max(1, len(extra_pts) // 3000)
            ec = extra_colors[::step2] if extra_colors is not None else 'red'
            ax.scatter(extra_pts[::step2, 0], extra_pts[::step2, 2], extra_pts[::step2, 1],
                       c=ec, s=1.5, alpha=0.8, label=extra_label)
            if extra_label:
                ax.legend(fontsize=7, loc='upper right')

        ax.set_xlabel("X", fontsize=7, color='white')
        ax.set_ylabel("Z", fontsize=7, color='white')
        ax.set_zlabel("Y", fontsize=7, color='white')
        ax.tick_params(labelsize=5, colors='gray')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(vname, fontsize=9, color='white')
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('gray')

    fig.suptitle(title, fontsize=14, fontweight='bold', color='#FF6B35', y=0.98)
    plt.savefig(str(DEBUG_DIR / save_name), dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  -> {save_name}")


# =============================================================================
print("=" * 60)
print(" Pipeline Debug Visualization")
print("=" * 60)

# ── Step 1: 데이터 로드 ──
print("\n[Step 1] 데이터 로드")
loader = DataLoader(data_dir=DATA_DIR, frame_id="000003",
                    extrinsics_dir=EXTRINSICS_DIR, glb_path=GLB_PATH)

intrinsics = [loader.load_intrinsics(i) for i in range(3)]
extrinsics = loader.load_extrinsics()

camera_data_list = []
for i in range(3):
    color, depth = loader.load_images(i)
    T = np.eye(4)
    if i == 1 and "T_C0_C1" in extrinsics:
        T = extrinsics["T_C0_C1"]
    elif i == 2 and "T_C0_C2" in extrinsics:
        T = extrinsics["T_C0_C2"]
    camera_data_list.append(CameraData(intrinsics=intrinsics[i],
                                        color_img=color, depth_img=depth,
                                        T_to_cam0=T))

# Step 1 시각화: RGB 이미지 + Depth
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor('#1a1a2e')
for i in range(3):
    axes[0, i].imshow(camera_data_list[i].color_img)
    axes[0, i].set_title(f"cam{i} RGB", color='white', fontsize=10)
    axes[0, i].axis("off")
    d = camera_data_list[i].depth_img.astype(float)
    d[d == 0] = np.nan
    axes[1, i].imshow(d, cmap='turbo')
    axes[1, i].set_title(f"cam{i} Depth", color='white', fontsize=10)
    axes[1, i].axis("off")
fig.suptitle("Step 1: Input RGB-D Images (3 cameras)",
             fontsize=14, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step1_input_images.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  -> step1_input_images.png")


# ── Step 2: 점군 통합 ──
print("\n[Step 2] 점군 통합 (3 cameras → cam0 frame)")

# 각 카메라별 점군도 개별 시각화
fig = plt.figure(figsize=(18, 5))
fig.patch.set_facecolor('#1a1a2e')
cam_colors_viz = ['green', 'blue', 'orange']

for i in range(3):
    intr = camera_data_list[i].intrinsics
    pcd_i = PointCloudProcessor.depth_to_pointcloud(
        camera_data_list[i].color_img, camera_data_list[i].depth_img,
        intr.K, intr.D, intr.depth_scale)
    pcd_i.transform(camera_data_list[i].T_to_cam0)
    pts_i = np.asarray(pcd_i.points)
    colors_i = np.asarray(pcd_i.colors) if pcd_i.has_colors() else None

    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    ax.set_facecolor('#16213e')
    step = max(1, len(pts_i) // 3000)
    c = colors_i[::step] if colors_i is not None else cam_colors_viz[i]
    ax.scatter(pts_i[::step, 0], pts_i[::step, 2], pts_i[::step, 1],
               c=c, s=0.3, alpha=0.4)
    ax.set_title(f"cam{i} → cam0 ({len(pts_i)} pts)", fontsize=9, color='white')
    ax.view_init(25, -55)
    ax.tick_params(labelsize=5, colors='gray')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('gray')

fig.suptitle("Step 2: Per-camera point clouds (transformed to cam0)",
             fontsize=14, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step2a_per_camera.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  -> step2a_per_camera.png")

merged_pcd = PointCloudProcessor.merge_pointclouds(camera_data_list, voxel_size=0.003)
merged_pts = np.asarray(merged_pcd.points)
merged_colors = np.asarray(merged_pcd.colors) if merged_pcd.has_colors() else None

plot_pcd_3views(merged_pts, merged_colors,
                f"Step 2: Merged Point Cloud ({len(merged_pts)} pts, voxel=3mm)",
                "step2b_merged.png")


# ── Step 3: 테이블 제거 ──
print("\n[Step 3] 테이블 평면 제거 (RANSAC)")
objects_pcd, table_plane = PointCloudProcessor.remove_table_plane(merged_pcd)
obj_pts = np.asarray(objects_pcd.points)
obj_colors = np.asarray(objects_pcd.colors) if objects_pcd.has_colors() else None

# 테이블 vs 물체 비교
table_idx = merged_pcd.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=1000)[1]
table_pcd = merged_pcd.select_by_index(table_idx)
table_pts = np.asarray(table_pcd.points)

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_facecolor('#16213e')
step_t = max(1, len(table_pts) // 3000)
ax1.scatter(table_pts[::step_t, 0], table_pts[::step_t, 2], table_pts[::step_t, 1],
            c='gray', s=0.3, alpha=0.3, label=f'Table ({len(table_pts)})')
step_o = max(1, len(obj_pts) // 3000)
oc = obj_colors[::step_o] if obj_colors is not None else 'red'
ax1.scatter(obj_pts[::step_o, 0], obj_pts[::step_o, 2], obj_pts[::step_o, 1],
            c=oc, s=0.8, alpha=0.6, label=f'Objects ({len(obj_pts)})')
ax1.legend(fontsize=7)
ax1.set_title("Table (gray) vs Objects (color)", fontsize=9, color='white')
ax1.view_init(25, -55)
ax1.tick_params(labelsize=5, colors='gray')
for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_facecolor('#16213e')
ax2.scatter(obj_pts[::step_o, 0], obj_pts[::step_o, 2], obj_pts[::step_o, 1],
            c=oc, s=0.8, alpha=0.6)
ax2.set_title(f"Objects only ({len(obj_pts)} pts)", fontsize=9, color='white')
ax2.view_init(25, -55)
ax2.tick_params(labelsize=5, colors='gray')
for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

a, b, c, d = table_plane
fig.suptitle(f"Step 3: Table Removal — plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0",
             fontsize=13, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step3_table_removal.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  -> step3_table_removal.png")


# ── Step 4a: DBSCAN 클러스터링 ──
print("\n[Step 4a] DBSCAN 클러스터링 + 색상 매칭")
labels = np.array(objects_pcd.cluster_dbscan(eps=0.015, min_points=50, print_progress=False))
unique_labels = np.unique(labels[labels >= 0])

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_facecolor('#16213e')

# 클러스터별 색상
cmap = plt.cm.get_cmap('tab20', len(unique_labels))
for li, label in enumerate(unique_labels[:20]):  # 상위 20개만
    idx = labels == label
    pts_l = obj_pts[idx]
    step_l = max(1, len(pts_l) // 500)
    ax1.scatter(pts_l[::step_l, 0], pts_l[::step_l, 2], pts_l[::step_l, 1],
                c=[cmap(li)], s=1, alpha=0.6)
ax1.set_title(f"DBSCAN: {len(unique_labels)} clusters", fontsize=9, color='white')
ax1.view_init(25, -55)
ax1.tick_params(labelsize=5, colors='gray')
for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

# 노란색 클러스터 하이라이트
target_label, yellow_ratio = PoseEstimator._find_target_cluster(
    objects_pcd, labels, unique_labels)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_facecolor('#16213e')

# 나머지 회색
other_idx = labels != target_label
step_oth = max(1, other_idx.sum() // 3000)
ax2.scatter(obj_pts[other_idx][::step_oth, 0],
            obj_pts[other_idx][::step_oth, 2],
            obj_pts[other_idx][::step_oth, 1],
            c='gray', s=0.3, alpha=0.2)

# 칼 클러스터 강조
knife_idx = labels == target_label
knife_pts = obj_pts[knife_idx]
knife_colors = obj_colors[knife_idx] if obj_colors is not None else None
kc = knife_colors if knife_colors is not None else 'yellow'
ax2.scatter(knife_pts[:, 0], knife_pts[:, 2], knife_pts[:, 1],
            c=kc, s=3, alpha=0.9)

knife_ext = knife_pts.max(0) - knife_pts.min(0)
ax2.set_title(f"Target: cluster {target_label} ({len(knife_pts)} pts, yellow={yellow_ratio:.0%})",
              fontsize=9, color='white')
ax2.view_init(25, -55)
ax2.tick_params(labelsize=5, colors='gray')
for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

fig.suptitle("Step 4a: Clustering + Color Matching (yellow knife)",
             fontsize=13, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step4a_clustering.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  -> step4a_clustering.png (target={target_label}, {len(knife_pts)} pts)")


# ── Step 4b: 레퍼런스 모델 + 스케일링 ──
print("\n[Step 4b] 레퍼런스 모델 스케일링")
ref_pcd = loader.load_reference_pcd()
ref_pts = np.asarray(ref_pcd.points)

R_ref, ref_eig = PoseEstimator._pca_axes(ref_pts)
ref_centered = ref_pts - ref_pts.mean(0)
ref_proj = ref_centered @ R_ref
ref_pca_extents = ref_proj.max(0) - ref_proj.min(0)

R_cl, _ = PoseEstimator._pca_axes(knife_pts)
cl_centered = knife_pts - knife_pts.mean(0)
cl_proj = cl_centered @ R_cl
cl_pca_extents = cl_proj.max(0) - cl_proj.min(0)

axis_scales = cl_pca_extents[np.argsort(cl_pca_extents)[::-1]] / \
              ref_pca_extents[np.argsort(ref_pca_extents)[::-1]]

fig = plt.figure(figsize=(16, 5))
fig.patch.set_facecolor('#1a1a2e')

# 원본 레퍼런스
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.set_facecolor('#16213e')
step_r = max(1, len(ref_pts) // 3000)
ax1.scatter(ref_pts[::step_r, 0], ref_pts[::step_r, 2], ref_pts[::step_r, 1],
            c='cyan', s=0.3, alpha=0.4)
re = ref_pca_extents
ax1.set_title(f"Reference (original)\n{re[0]:.3f} x {re[1]:.3f} x {re[2]:.3f} m",
              fontsize=8, color='white')
ax1.view_init(25, -55)
ax1.tick_params(labelsize=5, colors='gray')
for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

# 스케일된 레퍼런스
ref_center = ref_pts.mean(0)
S_pca = np.diag(axis_scales)
ref_in_pca = (ref_pts - ref_center) @ R_ref
ref_scaled_pca = ref_in_pca @ S_pca
ref_scaled_pts = ref_scaled_pca @ R_ref.T + ref_center

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.set_facecolor('#16213e')
ax2.scatter(ref_scaled_pts[::step_r, 0], ref_scaled_pts[::step_r, 2], ref_scaled_pts[::step_r, 1],
            c='lime', s=0.3, alpha=0.4)
se = cl_pca_extents[np.argsort(cl_pca_extents)[::-1]]
ax2.set_title(f"Scaled (anisotropic)\nscale: ({axis_scales[0]:.4f}, {axis_scales[1]:.4f}, {axis_scales[2]:.4f})",
              fontsize=8, color='white')
ax2.view_init(25, -55)
ax2.tick_params(labelsize=5, colors='gray')
for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

# 클러스터
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.set_facecolor('#16213e')
kc2 = knife_colors if knife_colors is not None else 'yellow'
ax3.scatter(knife_pts[:, 0], knife_pts[:, 2], knife_pts[:, 1],
            c=kc2, s=2, alpha=0.8)
ce = cl_pca_extents
ax3.set_title(f"Target cluster\n{ce[0]:.4f} x {ce[1]:.4f} x {ce[2]:.4f} m",
              fontsize=8, color='white')
ax3.view_init(25, -55)
ax3.tick_params(labelsize=5, colors='gray')
for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

fig.suptitle("Step 4b: Reference Model Scaling (PCA axis-wise)",
             fontsize=13, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step4b_scaling.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  -> step4b_scaling.png")


# ── Step 4c: 5가지 초기 정렬 후보 ──
print("\n[Step 4c] 초기 정렬 5후보 (PCA x4 + FPFH x1)")

model_scaled = o3d.geometry.PointCloud()
model_scaled.points = o3d.utility.Vector3dVector(ref_scaled_pts)

cluster_pcd = objects_pcd.select_by_index(np.where(labels == target_label)[0])
cluster_pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=0.009, max_nn=30))

fpfh_result = PoseEstimator._fpfh_global_registration(
    o3d.geometry.PointCloud(model_scaled), cluster_pcd, 0.003)

candidate_Ts = PoseEstimator._pca_candidate_transforms(
    np.asarray(model_scaled.points), knife_pts)
candidate_Ts.append(fpfh_result.transformation)

fig = plt.figure(figsize=(20, 4))
fig.patch.set_facecolor('#1a1a2e')
tags = ["PCA0", "PCA1", "PCA2", "PCA3", "FPFH"]
icp_results = []

for ci, (cand_T, tag) in enumerate(zip(candidate_Ts, tags)):
    model_cand = o3d.geometry.PointCloud(model_scaled)
    model_cand.transform(cand_T)
    model_cand.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.009, max_nn=30))

    icp_r = o3d.pipelines.registration.registration_icp(
        model_cand, cluster_pcd,
        max_correspondence_distance=0.009,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=200))
    icp_results.append(icp_r)

    # ICP 적용 후 모델
    model_after = o3d.geometry.PointCloud(model_cand)
    model_after.transform(icp_r.transformation)
    m_pts = np.asarray(model_after.points)

    ax = fig.add_subplot(1, 5, ci + 1, projection='3d')
    ax.set_facecolor('#16213e')
    ax.scatter(knife_pts[:, 0], knife_pts[:, 2], knife_pts[:, 1],
               c='yellow', s=1, alpha=0.4)
    step_m = max(1, len(m_pts) // 2000)
    ax.scatter(m_pts[::step_m, 0], m_pts[::step_m, 2], m_pts[::step_m, 1],
               c='red', s=0.5, alpha=0.5)

    is_best = icp_r.fitness == max(r.fitness for r in icp_results)
    color = '#00FF00' if icp_r.fitness >= max(r.fitness for r in icp_results) else 'white'
    ax.set_title(f"{tag}\nfit={icp_r.fitness:.4f} rmse={icp_r.inlier_rmse:.5f}",
                 fontsize=7, color=color)
    ax.view_init(25, -55)
    ax.tick_params(labelsize=4, colors='gray')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('gray')

best_ci = max(range(5), key=lambda i: icp_results[i].fitness)
fig.suptitle(f"Step 4c: 5 Initial Alignment Candidates → Best: {tags[best_ci]} (fitness={icp_results[best_ci].fitness:.4f})",
             fontsize=13, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step4c_candidates.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  -> step4c_candidates.png (best={tags[best_ci]})")


# ── Step 4d: 정밀 ICP + 테이블 보정 ──
print("\n[Step 4d] 정밀 ICP + 테이블 보정")

# Best candidate로 정밀 ICP
best_T_init = candidate_Ts[best_ci]
best_icp_T = icp_results[best_ci].transformation

model_final = o3d.geometry.PointCloud(model_scaled)
model_final.transform(best_T_init)
for p in [model_final, cluster_pcd]:
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.006, max_nn=30))

fine_result = o3d.pipelines.registration.registration_icp(
    model_final, cluster_pcd,
    max_correspondence_distance=0.009,
    init=best_icp_T,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=300))

model_aligned = o3d.geometry.PointCloud(model_final)
model_aligned.transform(fine_result.transformation)

# 테이블 보정
aligned_pts = np.asarray(model_aligned.points)
normal = table_plane[:3]
signed_dists = aligned_pts @ normal + table_plane[3]
min_dist = signed_dists.min()
table_shift = 0
if min_dist > 0.001:
    shift = -normal * min_dist
    aligned_pts += shift
    model_aligned.points = o3d.utility.Vector3dVector(aligned_pts)
    table_shift = min_dist

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#1a1a2e')

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_facecolor('#16213e')
ax1.scatter(knife_pts[:, 0], knife_pts[:, 2], knife_pts[:, 1],
            c='yellow', s=2, alpha=0.5, label='Cluster')
al_pts = np.asarray(model_aligned.points)
step_al = max(1, len(al_pts) // 3000)
ax1.scatter(al_pts[::step_al, 0], al_pts[::step_al, 2], al_pts[::step_al, 1],
            c='red', s=0.5, alpha=0.5, label='Aligned model')
ax1.legend(fontsize=7)
ax1.set_title(f"Fine ICP: fitness={fine_result.fitness:.4f}, RMSE={fine_result.inlier_rmse:.5f}m\n"
              f"Table shift: {table_shift*1000:.1f}mm",
              fontsize=8, color='white')
ax1.view_init(25, -55)
ax1.tick_params(labelsize=5, colors='gray')
for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor('gray')

# 최종 변환 정보
T_final = fine_result.transformation
R_final = T_final[:3, :3]
t_final = T_final[:3, 3]
rot = Rotation.from_matrix(R_final)
euler = rot.as_euler("xyz", degrees=True)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor('#16213e')
ax2.axis('off')
info_text = (
    f"{'='*40}\n"
    f"  Final Pose (OpenCV / cam0 ref)\n"
    f"{'='*40}\n\n"
    f"  Position:\n"
    f"    x = {t_final[0]:+.4f} m\n"
    f"    y = {t_final[1]:+.4f} m\n"
    f"    z = {t_final[2]:+.4f} m\n\n"
    f"  Rotation (Euler XYZ):\n"
    f"    rx = {euler[0]:+.1f} deg\n"
    f"    ry = {euler[1]:+.1f} deg\n"
    f"    rz = {euler[2]:+.1f} deg\n\n"
    f"  Size (OBB):\n"
)
obb = model_aligned.get_oriented_bounding_box()
ext = np.sort(obb.extent)[::-1]
info_text += (
    f"    L = {ext[0]*100:.1f} cm\n"
    f"    W = {ext[1]*100:.1f} cm\n"
    f"    H = {ext[2]*100:.1f} cm\n\n"
    f"  Fitness: {fine_result.fitness:.4f}\n"
    f"  RMSE:    {fine_result.inlier_rmse:.6f} m\n"
)
ax2.text(0.1, 0.95, info_text, transform=ax2.transAxes,
         fontsize=10, color='#cccccc', verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460', alpha=0.9))

fig.suptitle("Step 4d: Fine ICP + Table Correction → Final Pose",
             fontsize=13, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step4d_final_icp.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  -> step4d_final_icp.png")


# ── Step 5: 재투영 ──
print("\n[Step 5] 재투영 검증")
model_aligned.paint_uniform_color([1.0, 0.0, 0.0])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#1a1a2e')

for i, cam_data in enumerate(camera_data_list):
    K = cam_data.intrinsics.K
    pts = np.asarray(model_aligned.points)
    T_cam = np.linalg.inv(cam_data.T_to_cam0)
    pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
    pts_cam = (T_cam @ pts_hom.T)[:3]
    valid = pts_cam[2] > 0
    proj = K @ pts_cam[:, valid]
    u = (proj[0] / proj[2]).astype(int)
    v = (proj[1] / proj[2]).astype(int)
    img = cam_data.color_img.copy()
    h, w = img.shape[:2]
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    axes[0, i].imshow(img)
    axes[0, i].set_title(f"cam{i} original", color='white', fontsize=10)
    axes[0, i].axis("off")

    axes[1, i].imshow(img)
    axes[1, i].scatter(u[mask], v[mask], c='lime', s=0.5, alpha=0.3)
    axes[1, i].set_title(f"cam{i} + reprojection ({mask.sum()} pts)", color='white', fontsize=10)
    axes[1, i].axis("off")

fig.suptitle("Step 5: Reprojection Verification",
             fontsize=14, fontweight='bold', color='#FF6B35')
plt.savefig(str(DEBUG_DIR / "step5_reprojection.png"), dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("  -> step5_reprojection.png")


# ── 완료 ──
print(f"\n{'='*60}")
print(f"  모든 디버그 이미지 저장 완료: {DEBUG_DIR}/")
print(f"{'='*60}")
print(f"  step1_input_images.png     - 입력 RGB-D 이미지")
print(f"  step2a_per_camera.png      - 카메라별 점군")
print(f"  step2b_merged.png          - 통합 점군")
print(f"  step3_table_removal.png    - 테이블 제거")
print(f"  step4a_clustering.png      - 클러스터링 + 색상 매칭")
print(f"  step4b_scaling.png         - 레퍼런스 스케일링")
print(f"  step4c_candidates.png      - 5가지 정렬 후보 비교")
print(f"  step4d_final_icp.png       - 최종 ICP + 포즈 결과")
print(f"  step5_reprojection.png     - 재투영 검증")
