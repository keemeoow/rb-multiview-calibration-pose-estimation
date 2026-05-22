#!/usr/bin/env python3
"""
outputs/obj{N}/obj{N}_scaled.glb 의 AABB 를 Step4 OBB extents 에 강제 맞춤.

uniform scale 만으로는 InstantMesh 메시 aspect ratio 와 실제 물체가 달라서
1축만 일치하는 문제가 있음 → vertex 에 축별 비균등 scale 을 적용해
GLB AABB 가 실측 OBB extents 와 (rank-매칭으로) 일치하도록 보정.

[실행]
python3 src/Obj_Step5_force_obb_scale.py \
  --glb_root ./outputs \
  --size_json ./outputs/size_measurements.json \
  --suffix _obb

→ outputs/obj{N}/obj{N}_scaled_obb.glb 생성. (원본 보존)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def force_axis_scale(mesh_or_scene, target_extents_sorted_desc: np.ndarray) -> trimesh.Scene:
    """현재 AABB extents 와 target extents 를 크기순으로 매칭해 축별 scale 적용."""
    geoms = (mesh_or_scene.geometry.values() if isinstance(mesh_or_scene, trimesh.Scene)
             else [mesh_or_scene])
    # collect global AABB across geometries
    all_v = np.concatenate([g.vertices for g in geoms if hasattr(g, "vertices") and len(g.vertices)])
    cur_extents = all_v.max(axis=0) - all_v.min(axis=0)  # 3 floats, in mesh order (x,y,z)

    # rank-match: sort current extents descending, get permutation
    order_cur = np.argsort(-cur_extents)         # mesh axis order from largest to smallest
    target_sorted = np.sort(target_extents_sorted_desc)[::-1]  # ensure desc
    # build per-axis scale on mesh axis order
    scale_per_axis = np.zeros(3, dtype=np.float64)
    for rank, axis in enumerate(order_cur):
        scale_per_axis[axis] = target_sorted[rank] / cur_extents[axis]

    print(f"   current AABB (mm):   {cur_extents * 1000.0}")
    print(f"   target OBB sorted:   {target_sorted}")
    print(f"   per-axis scale:       {scale_per_axis}")

    # apply non-uniform scale to every geometry's vertices
    S = np.diag([scale_per_axis[0], scale_per_axis[1], scale_per_axis[2], 1.0])
    out_scene = trimesh.Scene()
    for name, g in (mesh_or_scene.geometry.items() if isinstance(mesh_or_scene, trimesh.Scene)
                    else [("geom_0", mesh_or_scene)]):
        if not hasattr(g, "vertices"):
            continue
        g2 = g.copy()
        g2.apply_transform(S)
        out_scene.add_geometry(g2, geom_name=name)

    # verify
    v2 = np.concatenate([g.vertices for g in out_scene.geometry.values()])
    new_ext = v2.max(axis=0) - v2.min(axis=0)
    print(f"   new AABB (mm):        {new_ext * 1000.0}")
    return out_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb_root", default="./outputs",
                    help="obj{N}/obj{N}_scaled.glb 가 있는 루트")
    ap.add_argument("--size_json", default="./outputs/size_measurements.json")
    ap.add_argument("--suffix", default="_obb",
                    help="새 GLB 파일명 suffix (예: '_obb' → obj1_scaled_obb.glb). "
                         "빈 문자열이면 원본 덮어쓰기.")
    args = ap.parse_args()

    with open(args.size_json) as f:
        size_list = json.load(f)
    size_by_obj = {it["obj"]: it for it in size_list}

    root = Path(args.glb_root)
    for obj_dir in sorted(root.glob("obj*")):
        if not obj_dir.is_dir():
            continue
        name = obj_dir.name
        glb_in = obj_dir / f"{name}_scaled.glb"
        if not glb_in.exists():
            print(f"[skip] {glb_in} not found")
            continue
        if name not in size_by_obj:
            print(f"[skip] {name} not in {args.size_json}")
            continue

        # OBB extents in mm → convert to meters (GLB unit)
        obb_mm = np.array(size_by_obj[name]["obb_extents_mm_sorted"], dtype=np.float64)
        target_m = obb_mm / 1000.0

        print(f"\n[{name}]")
        scene = trimesh.load(str(glb_in), force="scene")
        out = force_axis_scale(scene, target_m)

        glb_out = obj_dir / f"{name}_scaled{args.suffix}.glb"
        out.export(str(glb_out))
        print(f"   → wrote {glb_out}")


if __name__ == "__main__":
    main()
