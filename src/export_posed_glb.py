#!/usr/bin/env python3
"""
포즈 추정 결과(JSON)를 기반으로 GLB 모델에
크기, 위치, 회전을 적용한 새 GLB 파일을 내보냄.

사용법 (Isaac Sim용, mm, Z-up):
  python3 src/export_posed_glb.py \
    --glb_path src/data/Hole.glb \
    --pose_json src/output/pose_hole/pose_000001.json \
    --out_path src/output/pose_hole/Hole_posed.glb \
    --coord isaac

  python3 src/export_posed_glb.py \
    --glb_path src/data/Peg.glb \
    --pose_json src/output/pose_peg/pose_000003.json \
    --out_path src/output/pose_peg/Peg_posed.glb \
    --coord isaac
"""

import argparse
import json

import numpy as np
import trimesh


# OpenCV → Isaac Sim (Z-up, right-handed)
# OpenCV: X-right, Y-down, Z-forward
# Isaac:  X-forward, Y-left, Z-up
T_CV_TO_ISAAC = np.array([
    [ 0,  0,  1,  0],
    [-1,  0,  0,  0],
    [ 0, -1,  0,  0],
    [ 0,  0,  0,  1]], dtype=np.float64)


def load_glb(path: str) -> trimesh.Trimesh:
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(list(scene.geometry.values()))
    return scene


def main():
    parser = argparse.ArgumentParser(description="포즈 적용 GLB 내보내기")
    parser.add_argument("--glb_path", required=True, help="원본 GLB 파일")
    parser.add_argument("--pose_json", required=True, help="포즈 JSON 파일")
    parser.add_argument("--out_path", required=True, help="출력 GLB 경로")
    parser.add_argument("--unit", default="mm", choices=["m", "mm"],
                        help="출력 단위 (기본: mm)")
    parser.add_argument("--coord", default="isaac", choices=["opencv", "isaac"],
                        help="좌표계 (기본: isaac)")
    args = parser.parse_args()

    with open(args.pose_json) as f:
        pose = json.load(f)

    scale = pose["scale_glb_to_real"]
    T = np.array(pose["T_obj_in_cam0"])  # 4x4, OpenCV cam0 기준

    # 좌표계 변환
    if args.coord == "isaac":
        T = T_CV_TO_ISAAC @ T

    print(f"[INFO] 원본 GLB: {args.glb_path}")
    print(f"[INFO] 스케일: {scale:.6f}")
    print(f"[INFO] 실제 크기: {pose['real_size_m']*100:.1f} cm")

    mesh = load_glb(args.glb_path)
    print(f"[INFO] 메시: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # 변환: center → scale → T (포즈 추정과 동일 순서)
    center = mesh.vertices.mean(axis=0)
    verts = (mesh.vertices - center) * scale

    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_transformed = (T @ verts_h.T)[:3].T

    # 단위 변환
    unit_scale = 1000.0 if args.unit == "mm" else 1.0
    verts_transformed *= unit_scale

    mesh.vertices = verts_transformed
    mesh.export(args.out_path, file_type="glb")

    coord_label = "Isaac Sim Z-up (X-fwd, Y-left, Z-up)" if args.coord == "isaac" \
                  else "OpenCV cam0 (X-right, Y-down, Z-fwd)"
    t_mm = T[:3, 3] * unit_scale
    print(f"[OK] 저장: {args.out_path}")
    print(f"     좌표계: {coord_label}")
    print(f"     단위: {args.unit}")
    print(f"     위치 ({args.unit}): X={t_mm[0]:+.2f}, Y={t_mm[1]:+.2f}, Z={t_mm[2]:+.2f}")


if __name__ == "__main__":
    main()
