#!/usr/bin/env python3
"""
=============================================================================
포즈 추정 결과 시각화
=============================================================================

multiview_pose_estimation.py 실행 후 output/ 에 생성된 결과를 시각화.
  - matplotlib 정적 이미지 (물체 좌표축 + 크기 + 회전 + 카메라)
  - Open3D 인터랙티브 뷰어

[사용법]
  python visualize_pose_result.py              # 이미지 생성 + Open3D 뷰어
  python visualize_pose_result.py --no-viewer  # 이미지만 생성
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── 경로 설정 ──
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = SCRIPT_DIR / "data"
CALIB_DIR = DATA_DIR / "cube_session_01" / "calib_out_cube"


# =============================================================================
# 데이터 로드
# =============================================================================

def load_object_pcd():
    pcd = o3d.io.read_point_cloud(str(OUTPUT_DIR / "object_pointcloud.ply"))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    # OBB 크기
    obb = pcd.get_oriented_bounding_box()
    return pts, colors, pcd, obb


def load_cameras():
    cams = [{"id": 0, "T_world": np.eye(4), "pos": np.zeros(3)}]
    for i, name in enumerate(["T_C0_C1", "T_C0_C2"], start=1):
        path = CALIB_DIR / f"{name}.npy"
        if path.exists():
            T = np.load(str(path))
            cam_pose = np.eye(4)
            cam_pose[:3, :3] = T[:3, :3]
            cam_pose[:3, 3] = T[:3, 3]
            cams.append({"id": i, "T_world": cam_pose, "pos": T[:3, 3]})
    return cams


def load_pose():
    path = OUTPUT_DIR / "pose_Reference_Matching.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {
        "translation": data["translation"],
        "rotation_matrix": data["rotation_matrix"],
        "euler_xyz_deg": data["euler_xyz_deg"],
        "transform_4x4": data["transform_4x4"],
        "fitness": float(data["fitness"]),
        "rmse": float(data["rmse"]),
        "method": str(data["method"]),
    }


# =============================================================================
# matplotlib 그리기
# =============================================================================

CAM_COLORS = [[0.2, 0.8, 0.2], [0.2, 0.2, 1.0], [1.0, 0.8, 0.0]]
ACCENT = "#FF6B35"
AXIS_COLORS = ['#FF3333', '#33CC33', '#3366FF']
AXIS_NAMES = ['X', 'Y', 'Z']


def draw_cameras(ax, cams):
    labels = ["cam0 (ref)", "cam1", "cam2"]
    for cam, color, label in zip(cams, CAM_COLORS, labels):
        pos = cam["pos"]
        ax.scatter(pos[0], pos[2], pos[1],
                   c=[color], s=120, marker='^', edgecolors='white',
                   linewidths=0.8, zorder=10)
        ax.text(pos[0], pos[2], pos[1] + 0.025, label,
                fontsize=7, ha='center', color=color, fontweight='bold')
        z_dir = cam["T_world"][:3, 2] * 0.06
        ax.quiver(pos[0], pos[2], pos[1],
                  z_dir[0], z_dir[2], z_dir[1],
                  color=color, arrow_length_ratio=0.15, linewidth=1.2, alpha=0.7)


def draw_pointcloud(ax, pts, colors, alpha=0.3, size=0.8):
    step = max(1, len(pts) // 4000)
    pts_ds = pts[::step]
    c = colors[::step] if colors is not None else 'silver'
    ax.scatter(pts_ds[:, 0], pts_ds[:, 2], pts_ds[:, 1], c=c, s=size, alpha=alpha)


def draw_pose_axes(ax, obj_center, R, euler, size=0.05, linewidth=3.0):
    """물체 중심에 XYZ 축 + 회전값 텍스트"""
    # matplotlib 3D: X→X, Y→Z, Z→Y (swap)
    cx, cy, cz = obj_center[0], obj_center[2], obj_center[1]

    for i in range(3):
        d = R[:, i] * size
        ax.quiver(cx, cy, cz,
                  d[0], d[2], d[1],
                  color=AXIS_COLORS[i], arrow_length_ratio=0.2,
                  linewidth=linewidth, zorder=15)

        # 축 끝에 이름 + 회전값 표시
        end = obj_center + R[:, i] * size * 1.3
        label = f"{AXIS_NAMES[i]}: {euler[i]:+.1f}°"
        ax.text(end[0], end[2], end[1], label,
                fontsize=6, color=AXIS_COLORS[i], fontweight='bold',
                zorder=20,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.6, edgecolor=AXIS_COLORS[i], linewidth=0.5))


def draw_size_annotations(ax, obj_center, R, obb_extent, euler):
    """OBB 크기를 차원선으로 표시"""
    # OBB의 3축 = PCA 축 ≈ R 열벡터
    # extent는 큰 순서로 정렬
    ext_sorted = np.sort(obb_extent)[::-1]
    dim_labels = [
        f"L={ext_sorted[0]*100:.1f}cm",
        f"W={ext_sorted[1]*100:.1f}cm",
        f"H={ext_sorted[2]*100:.1f}cm",
    ]

    # 물체 중심에서 각 축 방향으로 반 범위의 양 끝점 표시
    for i in range(3):
        half = ext_sorted[i] / 2
        p1 = obj_center - R[:, i] * half
        p2 = obj_center + R[:, i] * half

        # 차원선 그리기 (matplotlib 좌표: X→X, Y→Z, Z→Y)
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]],
                '--', color=AXIS_COLORS[i], alpha=0.5, linewidth=1.0)

        # 중간 위치에 치수 텍스트
        mid = (p1 + p2) / 2
        offset = R[:, (i + 1) % 3] * 0.015  # 약간 옆으로
        tp = mid + offset
        ax.text(tp[0], tp[2], tp[1], dim_labels[i],
                fontsize=5.5, color='white', ha='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#333333',
                          alpha=0.7, edgecolor='gray', linewidth=0.3))


def draw_rotation_arcs(ax, obj_center, R, euler, radius=0.035):
    """각 축 주위로 회전 호(arc) 그리기"""
    cx, cy, cz = obj_center

    for i in range(3):
        angle_deg = euler[i]
        if abs(angle_deg) < 0.5:
            continue

        # 호를 그릴 두 축 (i가 아닌 나머지 두 축)
        j = (i + 1) % 3
        k = (i + 2) % 3

        n_pts = max(10, int(abs(angle_deg)))
        angles = np.linspace(0, np.radians(angle_deg), n_pts)

        arc_pts = []
        for a in angles:
            p = obj_center + R[:, j] * radius * np.cos(a) + R[:, k] * radius * np.sin(a)
            arc_pts.append(p)
        arc_pts = np.array(arc_pts)

        ax.plot(arc_pts[:, 0], arc_pts[:, 2], arc_pts[:, 1],
                color=AXIS_COLORS[i], linewidth=1.5, alpha=0.6)

        # 호 끝에 화살표 표시
        if len(arc_pts) >= 2:
            end = arc_pts[-1]
            direction = arc_pts[-1] - arc_pts[-2]
            direction = direction / (np.linalg.norm(direction) + 1e-9) * 0.005
            ax.quiver(end[0], end[2], end[1],
                      direction[0], direction[2], direction[1],
                      color=AXIS_COLORS[i], arrow_length_ratio=0.5,
                      linewidth=1.5, alpha=0.6)


def set_equal_aspect(ax, pts):
    center = pts.mean(axis=0)
    r = (pts.max(axis=0) - pts.min(axis=0)).max() * 0.6
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[2] - r, center[2] + r)
    ax.set_zlim(center[1] - r, center[1] + r)


def generate_image(pose, obj_pts, obj_colors, cams, obb_extent):
    t = pose["translation"]
    R = pose["rotation_matrix"]
    euler = pose["euler_xyz_deg"]
    obj_center = obj_pts.mean(axis=0)

    fig = plt.figure(figsize=(22, 8))
    fig.patch.set_facecolor('#1a1a2e')

    views = [
        (25, -55, "Perspective View"),
        (90, -90, "Top View (XZ)"),
        (0, -90, "Side View (XY)"),
    ]

    for idx, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_facecolor('#16213e')

        # 점군
        draw_pointcloud(ax, obj_pts, obj_colors, alpha=0.35, size=0.6)

        # 카메라
        draw_cameras(ax, cams)

        # 물체 중심에 좌표축 + 회전값
        draw_pose_axes(ax, obj_center, R, euler, size=0.05, linewidth=3.0)

        # 회전 호
        draw_rotation_arcs(ax, obj_center, R, euler, radius=0.03)

        # 크기 표시 (Perspective 뷰에만)
        if idx == 0:
            draw_size_annotations(ax, obj_center, R, obb_extent, euler)

        set_equal_aspect(ax, obj_pts)
        ax.set_xlabel("X (m)", fontsize=8, color='white', labelpad=2)
        ax.set_ylabel("Z (m)", fontsize=8, color='white', labelpad=2)
        ax.set_zlabel("Y (m)", fontsize=8, color='white', labelpad=2)
        ax.tick_params(labelsize=6, colors='gray')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10, color='white', pad=5)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('gray')

    fig.suptitle("Object 6DoF Pose — OpenCV Coordinate (cam0 ref)",
                 fontsize=16, fontweight='bold', color=ACCENT, y=0.98)

    ext_sorted = np.sort(obb_extent)[::-1]
    info = (f"Position: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m   |   "
            f"Rotation (XYZ): ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg   |   "
            f"Size: {ext_sorted[0]*100:.1f} x {ext_sorted[1]*100:.1f} x {ext_sorted[2]*100:.1f} cm   |   "
            f"Fitness: {pose['fitness']:.4f}")
    fig.text(0.5, 0.02, info, ha='center', fontsize=9, color='#cccccc',
             fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460',
                       alpha=0.8, edgecolor='#555555'))

    legend = ("Red pts = Aligned model (cam0/OpenCV frame)\n"
              "Triangles: green=cam0(ref), blue=cam1, yellow=cam2\n"
              "Axes: X=Red  Y=Green  Z=Blue  |  Arcs = Rotation")
    fig.text(0.01, 0.02, legend, fontsize=7, color='#aaaaaa',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                       alpha=0.9, edgecolor='#444444'))

    save_path = OUTPUT_DIR / "pose_visualization.png"
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return save_path


# =============================================================================
# Open3D 뷰어
# =============================================================================

def make_axes(T, size=0.05):
    origin = T[:3, 3]
    geoms = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for i in range(3):
        end = origin + T[:3, i] * size
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([origin, end])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([colors[i]])
        geoms.append(line)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.15)
        sphere.translate(end)
        sphere.paint_uniform_color(colors[i])
        geoms.append(sphere)
    return geoms


def make_frustum(T_world, color):
    R, t = T_world[:3, :3], T_world[:3, 3]
    s, d = 0.03, 0.06
    local = np.array([[0,0,0], [-s,-s,d], [s,-s,d], [s,s,d], [-s,s,d]])
    world = (R @ local.T).T + t
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
    sphere.translate(world[0])
    sphere.paint_uniform_color(color)
    return [frustum, sphere]


def make_obb_lineset(obb):
    """OBB를 LineSet으로 변환"""
    box_pts = np.asarray(obb.get_box_points())
    lines = [[0,1],[0,2],[0,3],[1,6],[1,7],[2,5],[2,7],[3,5],[3,6],[4,5],[4,6],[4,7]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(box_pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([[1, 1, 0]] * len(lines))
    return ls


def open_viewer(pose, obj_pcd, cams, obb):
    geoms = [obj_pcd]

    # 물체 중심에 좌표축
    obj_center = np.asarray(obj_pcd.points).mean(axis=0)
    R = pose["rotation_matrix"]
    T_obj = np.eye(4)
    T_obj[:3, :3] = R
    T_obj[:3, 3] = obj_center
    geoms.extend(make_axes(T_obj, size=0.06))

    # OBB 표시
    obb.color = (1, 1, 0)
    geoms.append(make_obb_lineset(obb))

    # 카메라
    for cam, color in zip(cams, CAM_COLORS):
        geoms.extend(make_frustum(cam["T_world"], color))
        geoms.extend(make_axes(cam["T_world"], size=0.04))

    # 원점 좌표프레임
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    euler = pose["euler_xyz_deg"]
    t = pose["translation"]
    obb_ext = np.sort(obb.extent)[::-1]
    print(f"\n  [Open3D] Viewer")
    print(f"    Position : ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
    print(f"    Rotation : ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg")
    print(f"    Size     : {obb_ext[0]*100:.1f} x {obb_ext[1]*100:.1f} x {obb_ext[2]*100:.1f} cm")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Pose Estimation — OpenCV Coord (cam0 ref)", width=1200, height=800)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.12])
    opt.point_size = 2.0
    vis.get_view_control().set_zoom(0.5)
    vis.run()
    vis.destroy_window()


# =============================================================================
# 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="포즈 추정 결과 시각화")
    parser.add_argument("--no-viewer", action="store_true", help="Open3D 뷰어 생략")
    args = parser.parse_args()

    print("=" * 55)
    print(" Pose Estimation Visualization")
    print("=" * 55)

    obj_pts, obj_colors, obj_pcd, obb = load_object_pcd()
    cams = load_cameras()
    pose = load_pose()

    if pose is None:
        print("[ERROR] pose_Reference_Matching.npz 없음. 먼저 포즈 추정을 실행하세요.")
        return

    t = pose["translation"]
    euler = pose["euler_xyz_deg"]
    obb_ext = np.sort(obb.extent)[::-1]
    print(f"  Object : {len(obj_pts)} pts")
    print(f"  Position: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m")
    print(f"  Rotation: ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg")
    print(f"  Size    : {obb_ext[0]*100:.1f} x {obb_ext[1]*100:.1f} x {obb_ext[2]*100:.1f} cm")
    print(f"  Fitness : {pose['fitness']:.4f}  RMSE: {pose['rmse']:.6f} m")

    # matplotlib 이미지
    path = generate_image(pose, obj_pts, obj_colors, cams, obb.extent)
    print(f"\n  -> {path.name}")

    # Open3D 뷰어
    if not args.no_viewer:
        print("\n" + "=" * 55)
        print(" Open3D Interactive Viewer")
        print("=" * 55)
        open_viewer(pose, obj_pcd, cams, obb)
        print("\nViewer closed.")


if __name__ == "__main__":
    main()
