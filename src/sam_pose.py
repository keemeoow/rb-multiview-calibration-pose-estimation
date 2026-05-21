#!/usr/bin/env python3
"""고정 cam 3대 캘리브레이션 → multi-object SAM mask → 최종 3개만 선택."""
import numpy as np
import cv2
from pathlib import Path

from fuse_multiframe_pose import (
    CAPTURE_DIR, load_intrinsics, load_static_transforms,
)
from sam_pose_estimation import (
    get_sam_auto_generator,
    sam_segment_cluster_one_view,   # SAM + 인접 face 병합 + 3D backproject
    group_clusters_by_centroid,     # 다중 cam mask → object 단위 그룹
    WORKSPACE_BBOX_MM,
)


# ───────── (1) 캘리브레이션 + intrinsics 로드 ─────────
intrinsics       = load_intrinsics()          # {cam_id: (K, dist, depth_scale)}
static_T_base_cam = load_static_transforms()  # {cam_id: T_base_cam (4x4)}

FIXED_CAMS = [0, 1, 3]    # 고정 cam 3대만 (cam2 = gripper, skip)
REF_FRAME  = 0
TOP_N      = 3            # 최종으로 남길 객체 수


# ───────── (2) 고정 cam 3대 view × multi-object SAM mask ─────────
sam_gen = get_sam_auto_generator()
all_clusters = []
for ci in FIXED_CAMS:
    fid = f"{REF_FRAME:06d}"
    rgb   = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"rgb_{fid}.jpg"))
    depth = cv2.imread(str(CAPTURE_DIR / f"cam{ci}" / f"depth_{fid}.png"),
                       cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        continue
    K, _, depth_scale = intrinsics[ci]
    T_base_cam        = static_T_base_cam[ci]      # ← 캘리브레이션 데이터

    # 한 프레임 안의 여러 객체를 각각 mask 로 추출 + 3D backproject
    cls = sam_segment_cluster_one_view(
        sam_gen, rgb, depth, K, depth_scale, T_base_cam,
        min_mask_area_px=400, max_mask_area_px=80000,
    )
    for c in cls:
        c["cam"], c["frame"] = ci, REF_FRAME
    all_clusters.extend(cls)
    print(f"  cam{ci}: {len(cls)} masks")


# ───────── (3) 3개 cam 의 mask 를 객체 단위로 묶기 ─────────
groups = group_clusters_by_centroid(all_clusters, eps_mm=30.0)
print(f"  {len(groups)} object group(s)")


# ───────── (4) 노이즈 제거 + 큰 객체 TOP_N (=3) 만 최종 mask ─────────
main_groups = [gs for gs in groups
               if sum(c["n_pts"] for c in gs) >= 5000]
main_groups.sort(key=lambda gs: -sum(c["n_pts"] for c in gs))
final_groups = main_groups[:TOP_N]                  # ← 원하는 3개만
print(f"  final objects kept: {len(final_groups)}")


# ───────── (5) 최종 객체별 mask 정리 (cam 별로 묶어서 출력) ─────────
final_masks = []   # [{obj_idx, cam, frame, mask_2d, centroid_mm}, ...]
for obj_idx, gs in enumerate(final_groups):
    centroid = np.mean([c["centroid_mm"] for c in gs], axis=0)
    print(f"  obj{obj_idx}: {len(gs)} masks "
          f"cams={sorted({c['cam'] for c in gs})} "
          f"center≈[{centroid[0]:.0f},{centroid[1]:.0f},{centroid[2]:.0f}]mm")
    for c in gs:
        final_masks.append({
            "obj_idx":     obj_idx,
            "cam":         c["cam"],
            "frame":       c["frame"],
            "mask_2d":     c["mask_2d"],     # cam 픽셀 공간의 binary mask
            "centroid_mm": c["centroid_mm"],
        })
