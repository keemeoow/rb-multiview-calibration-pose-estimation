#!/usr/bin/env python3
"""
멀티뷰 원본 vs 포즈 추정 실루엣 오버레이 비교 이미지.

출력:
  src/output/pose_pipeline/comparison_{frame}/comparison_{frame}.png
  (상단 3개: 원본 3뷰 / 하단 3개: posed GLB 실루엣 오버레이)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
CALIB_DIR = DATA_DIR / "cube_session_01" / "calib_out_cube"
POSE_DIR = SCRIPT_DIR / "output" / "pose_pipeline"
OUT_ROOT = SCRIPT_DIR / "output" / "pose_pipeline"

OBJ_COLORS_BGR = {
    "object_001": (80, 80, 255),    # 빨강
    "object_002": (80, 230, 255),   # 노랑
    "object_003": (255, 140, 40),   # 곤색
    "object_004": (200, 230, 90),   # 민트
}


def load_intrinsics():
    Ks, sizes = [], []
    for ci in range(3):
        npz = np.load(str(INTR_DIR / f"cam{ci}.npz"), allow_pickle=True)
        Ks.append(npz["color_K"].astype(np.float64))
        sizes.append((int(npz["color_w"]), int(npz["color_h"])))
    return Ks, sizes


def load_extrinsics():
    ext = {0: np.eye(4, dtype=np.float64)}
    for ci in (1, 2):
        ext[ci] = np.load(str(CALIB_DIR / f"T_C0_C{ci}.npy")).astype(np.float64)
    return ext


def load_rgbs(frame_id):
    imgs = []
    for ci in range(3):
        p = DATA_DIR / "object_capture" / f"cam{ci}" / f"rgb_{frame_id}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        imgs.append(img)
    return imgs


def load_posed_meshes(frame_id):
    """이미 base 좌표계로 변환된 posed GLB들을 로드."""
    items = []
    for oi in range(1, 5):
        name = f"object_{oi:03d}"
        glb = POSE_DIR / f"{name}_posed_{frame_id}.glb"
        if not glb.exists():
            continue
        scene = trimesh.load(str(glb))
        mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
                if isinstance(scene, trimesh.Scene) else scene)
        items.append((name, mesh))
    return items


def render_silhouette(mesh, K, T_base_cam, img_hw):
    """base 좌표계 posed mesh → 카메라 실루엣 마스크 (z-buffer 없이 face fill)."""
    h, w = img_hw
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    T_cam_base = np.linalg.inv(T_base_cam)
    Vh = np.hstack([V, np.ones((len(V), 1))])
    Vc = (T_cam_base @ Vh.T)[:3].T  # cam frame
    z = Vc[:, 2]
    u = K[0, 0] * Vc[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2]
    v = K[1, 1] * Vc[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2]

    mask = np.zeros((h, w), dtype=np.uint8)
    ok = z > 0.05
    if not ok.any() or len(F) == 0:
        return mask

    # face 필터: 3 vertex 모두 z>0 이고 projection 범위 합리적
    f_ok = ok[F[:, 0]] & ok[F[:, 1]] & ok[F[:, 2]]
    for face in F[f_ok]:
        pts = np.stack([
            [u[face[0]], v[face[0]]],
            [u[face[1]], v[face[1]]],
            [u[face[2]], v[face[2]]],
        ]).astype(np.int32)
        # 크게 벗어난 건 skip
        if np.any(np.abs(pts) > 4 * max(w, h)):
            continue
        cv2.fillConvexPoly(mask, pts, 255)
    return mask


def overlay(img, mask, color_bgr, alpha=0.55):
    out = img.copy()
    color_layer = np.zeros_like(img)
    color_layer[:] = color_bgr
    m = mask > 0
    out[m] = cv2.addWeighted(img, 1.0 - alpha, color_layer, alpha, 0)[m]
    # 외곽선
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color_bgr, 2, cv2.LINE_AA)
    return out


def build_comparison(frame_id):
    Ks, sizes = load_intrinsics()
    ext = load_extrinsics()
    rgbs = load_rgbs(frame_id)
    posed = load_posed_meshes(frame_id)
    if not posed:
        raise RuntimeError(f"posed GLB 없음 (frame {frame_id})")

    overlays = [img.copy() for img in rgbs]
    name_labels = []
    for name, mesh in posed:
        color = OBJ_COLORS_BGR.get(name, (0, 255, 0))
        name_labels.append((name, color))
        for ci in range(3):
            h, w = rgbs[ci].shape[:2]
            silh = render_silhouette(mesh, Ks[ci], ext[ci], (h, w))
            overlays[ci] = overlay(overlays[ci], silh, color, alpha=0.55)

    # 타이틀
    top = [img.copy() for img in rgbs]
    for ci, img in enumerate(top):
        cv2.putText(img, f"cam{ci} Original (frame {frame_id})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for ci, img in enumerate(overlays):
        cv2.putText(img, f"cam{ci} Pose Overlay", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)

    # 범례
    legend = " | ".join(n for n, _ in name_labels)
    for img in overlays:
        cv2.putText(img, legend, (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)

    row_top = np.hstack(top)
    row_bot = np.hstack(overlays)
    canvas = np.vstack([row_top, row_bot])

    out_dir = OUT_ROOT / f"comparison_{frame_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_{frame_id}.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"saved: {out_path}  (objects: {', '.join(n for n,_ in name_labels)})")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        fids = sorted({p.stem.split("_")[-1]
                       for p in POSE_DIR.glob("object_*_posed_*.glb")
                       if "_isaac" not in p.stem})
        for fid in fids:
            build_comparison(fid)
    else:
        build_comparison(args.frame_id or "000000")


if __name__ == "__main__":
    main()
