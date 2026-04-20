#!/usr/bin/env python3
"""실제 물체 vs 예측 silhouette 일치도 시각화.

각 카메라 이미지에 예측 silhouette을 반투명 색상으로 overlay하고
윤곽선을 강조 표시. 실제 물체와 얼마나 겹치는지 직관적으로 판단 가능.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import load_calibration, load_frame, normalize_glb

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"

OBJ_COLORS_BGR = {
    "object_001": (80, 80, 255),     # 빨강
    "object_002": (80, 230, 255),    # 노랑
    "object_003": (255, 140, 40),    # 곤색
    "object_004": (200, 230, 90),    # 민트
}


def render_silhouette(mesh, T_base_obj, scale_per_axis, model_center, cam):
    h, w = cam.intrinsics.height, cam.intrinsics.width
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    if len(V) == 0 or len(F) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    V_obj = (V - model_center) * scale_per_axis
    Vh = np.hstack([V_obj, np.ones((len(V_obj), 1))])
    V_base = (T_base_obj @ Vh.T)[:3].T
    T_cam_base = np.linalg.inv(cam.T_base_cam)
    V_cam = (T_cam_base @ np.hstack([V_base, np.ones((len(V_base), 1))]).T)[:3].T
    z = V_cam[:, 2]
    ok = z > 0.05
    K = cam.intrinsics.K
    u = (K[0, 0] * V_cam[:, 0] / np.where(z > 1e-6, z, 1e-6) + K[0, 2])
    v = (K[1, 1] * V_cam[:, 1] / np.where(z > 1e-6, z, 1e-6) + K[1, 2])
    mask = np.zeros((h, w), dtype=np.uint8)
    f_ok = ok[F[:, 0]] & ok[F[:, 1]] & ok[F[:, 2]]
    if f_ok.sum() == 0:
        return mask
    F_valid = F[f_ok]
    tri = np.stack([
        np.stack([u[F_valid[:, 0]], v[F_valid[:, 0]]], axis=-1),
        np.stack([u[F_valid[:, 1]], v[F_valid[:, 1]]], axis=-1),
        np.stack([u[F_valid[:, 2]], v[F_valid[:, 2]]], axis=-1),
    ], axis=1).astype(np.int32)
    lim = 4 * max(w, h)
    inside = np.abs(tri).reshape(len(tri), -1).max(axis=1) <= lim
    for face in tri[inside]:
        cv2.fillConvexPoly(mask, face, 255)
    return mask


def build_match_visualization(frame_id: str, pose_dir: Path, out_path: Path):
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    # 각 카메라 이미지에 모든 물체의 silhouette outline 그리기
    overlays = [cam.color_bgr.copy() for cam in frames]
    fill_layer = [np.zeros_like(cam.color_bgr) for cam in frames]

    for oi in range(1, 5):
        name = f"object_{oi:03d}"
        pose_json = pose_dir / f"pose_{name}.json"
        if not pose_json.exists():
            continue
        pose = json.loads(pose_json.read_text())
        T = np.array(pose["T_base_obj"])
        scale_per = np.array(pose["anisotropic_scale_xyz"])
        color = OBJ_COLORS_BGR[name]

        glb = DATA_DIR / f"{name}.glb"
        model = normalize_glb(glb)
        scene = trimesh.load(str(glb))
        mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
                if isinstance(scene, trimesh.Scene) else scene)

        for ci, cam in enumerate(frames):
            silh = render_silhouette(mesh, T, scale_per, model.center, cam)
            # 반투명 채움
            fill_mask = silh > 0
            fill_layer[ci][fill_mask] = color
            # 외곽선 굵게
            cnts, _ = cv2.findContours(silh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlays[ci], cnts, -1, color, 3, cv2.LINE_AA)

    # 반투명 채움 병합 (alpha=0.25)
    final = []
    for ci in range(3):
        blend = cv2.addWeighted(overlays[ci], 0.75, fill_layer[ci], 0.25, 0)
        # 외곽선 위에 다시 그리기 (흐려지지 않도록)
        fill_only_mask = np.any(fill_layer[ci] > 0, axis=2)
        cv2.putText(blend, f"cam{ci}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        final.append(blend)

    # 가로로 연결
    grid = np.hstack(final)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"saved: {out_path}")

    # 각 카메라별 큰 이미지도 저장
    per_cam_dir = out_path.parent / f"{out_path.stem}_per_cam"
    per_cam_dir.mkdir(exist_ok=True)
    for ci, img in enumerate(final):
        cv2.imwrite(str(per_cam_dir / f"cam{ci}.png"), img)
    print(f"per-cam images: {per_cam_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--pose_dir", default="src/output/pose_per_object_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pose_dir = Path(args.pose_dir) / f"frame_{args.frame_id}"
    out = Path(args.out) if args.out else pose_dir / f"match_overlay_{args.frame_id}.png"
    build_match_visualization(args.frame_id, pose_dir, out)


if __name__ == "__main__":
    main()
