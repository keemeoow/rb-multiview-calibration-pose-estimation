#!/usr/bin/env python3
"""
멀티 오브젝트 포즈 추정 결과 — 멀티뷰 재투영 검증 시각화

각 카메라 뷰에 추정된 GLB 메시를 와이어프레임으로 재투영하고,
물체별 6DoF 포즈 정보를 함께 표시합니다.

사용법:
  python3 src/visualize_pose_verification.py
  python3 src/visualize_pose_verification.py --frame_id 000000
"""

import json
import argparse
import numpy as np
import cv2
import trimesh
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# 한글 폰트 설정 (ttc 파일 직접 등록)
_KR_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_KR_FONT).exists():
    fm.fontManager.addfont(_KR_FONT)
    plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 ──
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INTRINSICS_DIR = SCRIPT_DIR / "intrinsics"
CALIB_DIR = DATA_DIR / "cube_session_01" / "calib_out_cube"

# 물체별 색상 (BGR / RGB)
OBJ_COLORS_BGR = {
    "object_001": (60, 60, 255),
    "object_002": (0, 220, 255),
    "object_003": (255, 140, 30),
    "object_004": (200, 220, 50),
}
OBJ_COLORS_RGB = {
    "object_001": "#FF3C3C",
    "object_002": "#FFDC00",
    "object_003": "#1E8CFF",
    "object_004": "#32DCC8",
}


# ═══════════════════════════════════════════════════════════
# 데이터 로드
# ═══════════════════════════════════════════════════════════

def load_intrinsics():
    cams = []
    for ci in range(3):
        npz = np.load(str(INTRINSICS_DIR / f"cam{ci}.npz"), allow_pickle=True)
        cams.append({
            "K": npz["color_K"].astype(np.float64),
            "w": int(npz["color_w"]),
            "h": int(npz["color_h"]),
        })
    return cams


def load_extrinsics():
    ext = {0: np.eye(4)}
    for ci in [1, 2]:
        ext[ci] = np.load(str(CALIB_DIR / f"T_C0_C{ci}.npy")).astype(np.float64)
    return ext


def load_rgb(frame_id):
    imgs = []
    for ci in range(3):
        p = DATA_DIR / "object_capture" / f"cam{ci}" / f"rgb_{frame_id}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"RGB 이미지 없음: {p}")
        imgs.append(img)
    return imgs


def load_poses(out_dir, frame_id):
    poses = {}
    for oi in range(1, 5):
        name = f"object_{oi:03d}"
        jp = out_dir / f"pose_{name}_{frame_id}.json"
        if jp.exists():
            with open(jp) as f:
                poses[name] = json.load(f)
    return poses


def load_glb(name):
    p = DATA_DIR / f"{name}.glb"
    if not p.exists():
        return None, None
    scene = trimesh.load(str(p))
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) \
        if isinstance(scene, trimesh.Scene) else scene
    return mesh, mesh.centroid.copy()


# ═══════════════════════════════════════════════════════════
# 재투영
# ═══════════════════════════════════════════════════════════

def project_mesh_wireframe(mesh, center, pose_json, K, T_base_cam, img_hw):
    """메시 엣지를 카메라 이미지 좌표로 투영."""
    T = np.array(pose_json["T_base_obj"])
    scale = pose_json["scale"]
    h, w = img_hw

    v = (mesh.vertices - center) * scale
    vh = np.hstack([v, np.ones((len(v), 1))])
    vb = (T @ vh.T)[:3].T                        # base frame
    vc = (np.linalg.inv(T_base_cam) @ np.hstack([vb, np.ones((len(vb), 1))]).T)[:3].T  # cam frame

    z = vc[:, 2]
    ok = z > 0.05
    pu = np.full(len(v), np.nan)
    pv = np.full(len(v), np.nan)
    pu[ok] = K[0, 0] * vc[ok, 0] / z[ok] + K[0, 2]
    pv[ok] = K[1, 1] * vc[ok, 1] / z[ok] + K[1, 2]

    edges_2d = []
    for e0, e1 in mesh.edges_unique:
        if not (ok[e0] and ok[e1]):
            continue
        x0, y0, x1, y1 = int(pu[e0]), int(pv[e0]), int(pu[e1]), int(pv[e1])
        if abs(x0) > 3 * w or abs(y0) > 3 * h or abs(x1) > 3 * w or abs(y1) > 3 * h:
            continue
        edges_2d.append(((x0, y0), (x1, y1)))

    # 중심 투영
    cb = (T @ np.append(center * 0 , 1))[:3]  # pose의 position이 중심
    cb = np.array(pose_json["position_m"])
    cc = (np.linalg.inv(T_base_cam) @ np.append(cb, 1))[:3]
    cx, cy = None, None
    if cc[2] > 0.05:
        cx = int(K[0, 0] * cc[0] / cc[2] + K[0, 2])
        cy = int(K[1, 1] * cc[1] / cc[2] + K[1, 2])

    return edges_2d, (cx, cy)


def draw_wireframe(img, edges, color, thickness=1):
    overlay = img.copy()
    for (x0, y0), (x1, y1) in edges:
        cv2.line(overlay, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)


# ═══════════════════════════════════════════════════════════
# 메인 시각화
# ═══════════════════════════════════════════════════════════

def generate(frame_id, out_dir):
    print("=" * 60)
    print(f" Pose Verification — Frame {frame_id}")
    print("=" * 60)

    intrinsics = load_intrinsics()
    extrinsics = load_extrinsics()
    rgb_imgs = load_rgb(frame_id)
    poses = load_poses(out_dir, frame_id)
    print(f"  물체 {len(poses)}개 포즈 로드")

    meshes = {}
    for name in poses:
        m, c = load_glb(name)
        if m is not None:
            meshes[name] = (m, c)

    # ── OpenCV 오버레이 이미지 (3 카메라) ──
    overlay_imgs = [img.copy() for img in rgb_imgs]

    for name, pj in poses.items():
        if name not in meshes:
            continue
        mesh, center = meshes[name]
        color = OBJ_COLORS_BGR.get(name, (0, 255, 0))

        for ci in range(3):
            K = intrinsics[ci]["K"]
            h, w = rgb_imgs[ci].shape[:2]
            edges, (cx, cy) = project_mesh_wireframe(
                mesh, center, pj, K, extrinsics[ci], (h, w))

            draw_wireframe(overlay_imgs[ci], edges, color, thickness=2)

            # 라벨
            label = pj.get("label", name)
            if cx is not None and 0 <= cx < w and 0 <= cy < h:
                cv2.putText(overlay_imgs[ci], label,
                            (cx - 30, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                cv2.circle(overlay_imgs[ci], (cx, cy), 4, color, -1, cv2.LINE_AA)

    # ── matplotlib 최종 이미지 ──
    obj_names = sorted(poses.keys())
    n_obj = len(obj_names)

    fig = plt.figure(figsize=(30, 24))
    fig.patch.set_facecolor('#111122')

    gs = GridSpec(3, 4, figure=fig, hspace=0.10, wspace=0.04,
                  height_ratios=[1, 1, 0.7],
                  width_ratios=[1, 1, 1, 0.02])  # 4열째는 여백

    cam_names = ["cam0 (Reference)", "cam1", "cam2"]

    for ci in range(3):
        # Row 0: 원본
        ax0 = fig.add_subplot(gs[0, ci])
        ax0.imshow(cv2.cvtColor(rgb_imgs[ci], cv2.COLOR_BGR2RGB))
        ax0.set_title(f"{cam_names[ci]} — Original", color='white',
                      fontsize=13, fontweight='bold', pad=6)
        ax0.axis('off')

        # Row 1: 오버레이
        ax1 = fig.add_subplot(gs[1, ci])
        ax1.imshow(cv2.cvtColor(overlay_imgs[ci], cv2.COLOR_BGR2RGB))
        ax1.set_title(f"{cam_names[ci]} — Pose Reprojection",
                      color='#00FF88', fontsize=13, fontweight='bold', pad=6)
        ax1.axis('off')

    # ── Row 2: 물체별 6DoF 정보 (4개를 3칸에 배치) ──
    # 4개 물체 → col0, col1에 각 1개, col2에 2개
    info_layout = []
    if n_obj <= 3:
        for i in range(n_obj):
            info_layout.append((i, 0.05, 0.95))  # col, x, y
    else:
        info_layout.append((0, 0.05, 0.95))
        info_layout.append((1, 0.05, 0.95))
        info_layout.append((2, 0.05, 0.98))
        info_layout.append((2, 0.05, 0.42))

    info_axes = {}
    for ci in range(3):
        ax = fig.add_subplot(gs[2, ci])
        ax.set_facecolor('#0a0a1e')
        ax.axis('off')
        info_axes[ci] = ax

    for idx, name in enumerate(obj_names):
        pj = poses[name]
        col, tx, ty = info_layout[idx]
        ax = info_axes[col]

        label = pj.get("label", name)
        pos = pj["position_m"]
        euler = pj["euler_xyz_deg"]
        quat = pj["quaternion_xyzw"]
        conf = pj.get("confidence", 0)
        fit = pj.get("fitness", 0)
        rmse = pj.get("rmse", 0)
        sc = pj.get("scale", 1.0)
        sz = pj.get("real_size_m", {})
        dscore = pj.get("depth_score", 0)
        cov = pj.get("coverage", 0)

        # 간결도 조절 (col2에 2개 들어가면 짧게)
        if n_obj == 4 and idx >= 2:
            txt = (
                f" {label} ({name})\n"
                f" {'─' * 32}\n"
                f" Pos(m): [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]\n"
                f" Euler:  [{euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}] deg\n"
                f" Quat:   [{quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f}]\n"
                f" Scale={sc:.3f} Fit={fit:.3f} Conf={conf:.3f}\n"
                f" Size: {sz.get('x',0)*100:.1f}x{sz.get('y',0)*100:.1f}"
                f"x{sz.get('z',0)*100:.1f}cm"
            )
            fs = 8
        else:
            txt = (
                f" {label} ({name})\n"
                f" {'─' * 34}\n"
                f" Position (m):\n"
                f"   x = {pos[0]:+.4f}\n"
                f"   y = {pos[1]:+.4f}\n"
                f"   z = {pos[2]:+.4f}\n\n"
                f" Rotation (Euler XYZ):\n"
                f"   rx = {euler[0]:+7.2f} deg\n"
                f"   ry = {euler[1]:+7.2f} deg\n"
                f"   rz = {euler[2]:+7.2f} deg\n\n"
                f" Quaternion (xyzw):\n"
                f"   [{quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f}]\n\n"
                f" Scale:      {sc:.4f}\n"
                f" Size:       {sz.get('x',0)*100:.1f} x {sz.get('y',0)*100:.1f}"
                f" x {sz.get('z',0)*100:.1f} cm\n"
                f" Fitness:    {fit:.4f}\n"
                f" RMSE:       {rmse:.6f} m\n"
                f" Depth:      {dscore:.4f}\n"
                f" Coverage:   {cov:.4f}\n"
                f" Confidence: {conf:.4f}"
            )
            fs = 9

        ec = OBJ_COLORS_RGB.get(name, "#FFFFFF")
        ax.text(tx, ty, txt, transform=ax.transAxes,
                fontsize=fs, color='#dddddd', verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#151530',
                          edgecolor=ec, linewidth=2.5, alpha=0.95))

    fig.suptitle(
        f"Multi-Object 6DoF Pose Verification — Frame {frame_id}\n"
        f"(Wireframe reprojection onto each camera view)",
        fontsize=18, fontweight='bold', color='#FF6B35', y=0.995)

    # 색상 범례
    legend_parts = []
    for n in obj_names:
        c = OBJ_COLORS_RGB.get(n, "#FFFFFF")
        legend_parts.append(f"{poses[n].get('label', n)}")
    fig.text(0.5, 0.003,
             "  |  ".join(legend_parts),
             ha='center', fontsize=11, color='#aaaaaa', fontstyle='italic')

    save_path = out_dir / f"pose_verification_{frame_id}.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"\n  Saved: {save_path}")
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_id", default="000000")
    parser.add_argument("--output_dir", default=None,
                        help="결과 디렉토리 (기본: src/output/pose_pipeline_{frame_id})")
    args = parser.parse_args()

    fid = args.frame_id
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = SCRIPT_DIR / "output" / f"pose_pipeline_{fid}"

    generate(fid, out)


if __name__ == "__main__":
    main()
