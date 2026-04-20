#!/usr/bin/env python3
"""문제 마스크 4개를 개별적으로 수정:

1. cam0 object_002 (노랑): cam1 centroid → cam0 bbox 투영 → SAM
2. cam1 object_003 (곤색 찌그러짐): bbox aspect 제한 + SAM 재실행
3. cam2 object_002 (노랑): cam1 centroid → cam2 bbox 투영 → SAM
4. cam2 object_004 (민트): cam1 centroid → cam2 bbox 투영 → SAM

입력: merged 마스크 디렉토리
출력: src/output/sam_masks/fixed_{timestamp}_frame_{id}/
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pose_pipeline import load_calibration, load_frame
from mobile_sam import SamPredictor, sam_model_registry

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
SAM_WEIGHTS = SCRIPT_DIR / "weights" / "mobile_sam.pt"


def glb_extent(glb_path: Path) -> np.ndarray:
    scene = trimesh.load(str(glb_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene)
    V = np.asarray(mesh.vertices)
    return V.max(0) - V.min(0)


def mask_3d_centroid(mask: np.ndarray, cam) -> np.ndarray:
    """마스크 픽셀의 base-frame 3D centroid."""
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    K = cam.intrinsics.K
    m = (mask > 0) & (depth > 0.05) & (depth < 1.5)
    if m.sum() < 20:
        return None
    ys, xs = np.where(m)
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t
    return np.median(pts_base, axis=0)


def project_3d_bbox_to_cam(centroid: np.ndarray, extent: np.ndarray, cam,
                           padding_m: float = 0.01) -> tuple:
    """3D 중심 + extent의 8 corners를 cam image에 투영 → bbox."""
    half = extent / 2 + padding_m
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(centroid + half * np.array([sx, sy, sz]))
    corners = np.array(corners)
    T_cb = np.linalg.inv(cam.T_base_cam)
    pts_cam = (T_cb[:3, :3] @ corners.T).T + T_cb[:3, 3]
    z = pts_cam[:, 2]
    ok = z > 0.05
    if ok.sum() < 4:
        return None
    K = cam.intrinsics.K
    u = (K[0, 0] * pts_cam[ok, 0] / z[ok] + K[0, 2])
    v = (K[1, 1] * pts_cam[ok, 1] / z[ok] + K[1, 2])
    H, W = cam.intrinsics.height, cam.intrinsics.width
    x1 = max(0, int(u.min())); x2 = min(W - 1, int(u.max()))
    y1 = max(0, int(v.min())); y2 = min(H - 1, int(v.max()))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return (x1, y1, x2, y2)


def sam_mask(predictor, bgr, bbox, point=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)
    kwargs = dict(box=np.array(bbox), multimask_output=True)
    if point is not None:
        kwargs["point_coords"] = np.array([point])
        kwargs["point_labels"] = np.array([1])
    masks, scores, _ = predictor.predict(**kwargs)
    best = int(np.argmax(scores))
    out = masks[best].astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k5)
    return out


def keep_largest_overlapping(sam_mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """bbox 중심에 가장 가까운 component만 유지 (SAM이 배경까지 먹는 경우 방지)."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(sam_mask, connectivity=8)
    if n <= 2:
        return sam_mask
    cx_box = (bbox[0] + bbox[2]) / 2
    cy_box = (bbox[1] + bbox[3]) / 2
    best_i, best_d = -1, 1e9
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 100:
            continue
        d = (cents[i][0] - cx_box)**2 + (cents[i][1] - cy_box)**2
        if d < best_d:
            best_d = d; best_i = i
    if best_i < 0:
        return sam_mask
    return np.where(labels == best_i, 255, 0).astype(np.uint8)


def fix_masks(merged_dir: Path, out_dir: Path, frame_id: str):
    print("Loading SAM...")
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    # 1) merged 마스크 복사
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in merged_dir.glob("*.png"):
        shutil.copy(f, out_dir / f.name)
    print(f"baseline copied: {merged_dir} → {out_dir}")

    # 2) 수정 대상 정의: (object, problem_cam, anchor_cam, shape_constraint)
    fixes = [
        # object_002 (노랑): cam1 → cam0, cam2
        {"obj": "object_002", "anchor": 1, "fix_cams": [0, 2], "padding": 0.015},
        # object_004 (민트): cam1 → cam2 (cam0은 이미 OK)
        {"obj": "object_004", "anchor": 1, "fix_cams": [2], "padding": 0.015},
        # object_003 (곤색) cam1: 수직 제한
        {"obj": "object_003", "anchor": 2, "fix_cams": [1], "padding": 0.01,
         "constrain_horizontal": True},
    ]

    glb_ext = {n: glb_extent(DATA_DIR / f"{n}.glb")
               for n in ["object_001","object_002","object_003","object_004"]}

    for fix in fixes:
        obj = fix["obj"]
        anchor_ci = fix["anchor"]
        anchor_mask_path = out_dir / f"{obj}_cam{anchor_ci}.png"
        anchor_mask = cv2.imread(str(anchor_mask_path), cv2.IMREAD_GRAYSCALE)
        if anchor_mask is None or (anchor_mask > 0).sum() < 200:
            print(f"[skip] {obj}: anchor cam{anchor_ci} mask too small")
            continue
        centroid = mask_3d_centroid(anchor_mask, frames[anchor_ci])
        if centroid is None:
            print(f"[skip] {obj}: centroid estimation failed")
            continue
        ext = glb_ext[obj] * 0.7  # 실제 물체는 GLB의 약 70% 크기 (경험적)
        print(f"\n[{obj}] anchor=cam{anchor_ci} centroid={centroid}")

        for fix_ci in fix["fix_cams"]:
            cam = frames[fix_ci]
            bbox = project_3d_bbox_to_cam(centroid, ext, cam, padding_m=fix["padding"])
            if bbox is None:
                print(f"  cam{fix_ci}: bbox projection 실패")
                continue
            print(f"  cam{fix_ci}: projected bbox={bbox}")

            # 특수 제약: cam1 object_003 → 수평 제한 (block은 가로형)
            if fix.get("constrain_horizontal"):
                x1, y1, x2, y2 = bbox
                # 수직 extent를 bbox 높이의 60%로 제한
                h = y2 - y1
                cy = (y1 + y2) // 2
                new_h = int(h * 0.5)
                y1 = cy - new_h // 2
                y2 = cy + new_h // 2
                bbox = (x1, y1, x2, y2)
                print(f"  cam{fix_ci}: horizontal-constrained bbox={bbox}")

            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            with torch.no_grad():
                m = sam_mask(predictor, cam.color_bgr, bbox, point=(cx, cy))
            m = keep_largest_overlapping(m, bbox)
            area = int((m > 0).sum())
            print(f"  cam{fix_ci}: SAM area={area}")
            # 저장 (기존 덮어쓰기)
            cv2.imwrite(str(out_dir / f"{obj}_cam{fix_ci}.png"), m)

    (out_dir / "meta.json").write_text(json.dumps(
        {"frame_id": frame_id, "baseline": str(merged_dir),
         "method": "targeted cross-cam bbox + SAM for 4 problem masks"},
        indent=2))
    print(f"\nfixed masks saved: {out_dir}")


def render_overlay(mask_dir: Path, out_dir: Path, frame_id: str):
    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)
    COLORS = {"object_001":(60,60,255),"object_002":(80,230,255),
              "object_003":(255,140,40),"object_004":(210,240,130)}

    def ov(img, m, c):
        out = img.copy(); cl = np.zeros_like(img); cl[:] = c
        mm = m > 0
        out[mm] = cv2.addWeighted(img, 0.45, cl, 0.55, 0)[mm]
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, c, 2, cv2.LINE_AA)
        return out

    def ann(img, txt):
        r = img.copy()
        cv2.putText(r, txt, (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0,255,0), 2, cv2.LINE_AA)
        return r

    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
    all_ov = [cam.color_bgr.copy() for cam in frames]
    for obj, color in COLORS.items():
        for ci in range(3):
            m = cv2.imread(str(mask_dir/f"{obj}_cam{ci}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                all_ov[ci] = ov(all_ov[ci], m, color)
    top = np.hstack([ann(originals[ci], f"cam{ci} Original") for ci in range(3)])
    bot = np.hstack([ann(all_ov[ci], f"cam{ci} Fixed") for ci in range(3)])
    cv2.imwrite(str(out_dir/"comparison.png"), np.vstack([top, bot]))
    print(f"overlay saved: {out_dir/'comparison.png'}")

    for obj, color in COLORS.items():
        obj_ov = []
        for ci in range(3):
            m = cv2.imread(str(mask_dir/f"{obj}_cam{ci}.png"), cv2.IMREAD_GRAYSCALE)
            img = originals[ci].copy()
            if m is not None:
                img = ov(img, m, color)
                a = int((m>0).sum())
                cv2.putText(img, f"{obj} area={a}", (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            obj_ov.append(ann(img, f"cam{ci} {obj}"))
        topr = np.hstack([ann(originals[ci], f"cam{ci} Original") for ci in range(3)])
        botr = np.hstack(obj_ov)
        cv2.imwrite(str(out_dir/f"comparison_{obj}.png"), np.vstack([topr, botr]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    ap.add_argument("--merged_dir",
                    default="src/output/sam_masks/merged_20260420_142735_frame_000000")
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mask_dir = Path("src/output/sam_masks") / f"fixed_{ts}_frame_{args.frame_id}"
    out_dir = Path("src/output/ideal_overlay") / f"fixed_{ts}_frame_{args.frame_id}"
    fix_masks(Path(args.merged_dir), mask_dir, args.frame_id)
    render_overlay(mask_dir, out_dir, args.frame_id)


if __name__ == "__main__":
    main()
