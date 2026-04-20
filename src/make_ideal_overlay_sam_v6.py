#!/usr/bin/env python3
"""SAM v6: cross-camera projection으로 문제 cam의 bbox 재확보.

처리 흐름:
1. v4 3D size filter로 각 cam에서 candidate 생성
2. candidate들 중 신뢰 가능한(GLB size와 일치) cam의 mask로 3D 중심/크기 추정
3. 신뢰 cam이 2개 이상이면 그것들로 3D 위치 fusion
4. 문제 cam에 대해 3D 위치를 역투영 → bbox 생성
5. 그 bbox를 prompt로 SAM 실행 → 정확한 마스크

저장: src/output/sam_masks/v6_{timestamp}_frame_{id}/
"""
from __future__ import annotations

import argparse
import datetime
import json
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

from pose_pipeline import (
    load_calibration, load_frame, estimate_table_plane,
    OBJECT_SYMMETRY,
)
from pose_per_object_v2 import (
    color_mask_for_object,
    connected_components_sorted,
)
from mobile_sam import SamPredictor, sam_model_registry

DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
SAM_WEIGHTS = SCRIPT_DIR / "weights" / "mobile_sam.pt"

OBJ_COLORS_BGR = {
    "object_001": (60, 60, 255),
    "object_002": (80, 230, 255),
    "object_003": (255, 140, 40),
    "object_004": (210, 240, 130),
}


def glb_extent(glb_path: Path) -> np.ndarray:
    scene = trimesh.load(str(glb_path))
    mesh = (trimesh.util.concatenate(list(scene.geometry.values()))
            if isinstance(scene, trimesh.Scene) else scene)
    V = np.asarray(mesh.vertices)
    return V.max(0) - V.min(0)


def mask_to_3d_info(mask: np.ndarray, cam) -> dict:
    """mask → 3D centroid + extent in base frame."""
    depth = cam.depth_u16.astype(np.float64) * cam.intrinsics.depth_scale
    H, W = depth.shape
    K = cam.intrinsics.K
    m = (mask > 0) & (depth > 0.05) & (depth < 1.5)
    if m.sum() < 30:
        return None
    ys, xs = np.where(m)
    z = depth[ys, xs]
    x_cam = (xs - K[0, 2]) * z / K[0, 0]
    y_cam = (ys - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    R = cam.T_base_cam[:3, :3]; t = cam.T_base_cam[:3, 3]
    pts_base = (R @ pts_cam.T).T + t
    centroid = np.median(pts_base, axis=0)
    lo = np.percentile(pts_base, 5, axis=0)
    hi = np.percentile(pts_base, 95, axis=0)
    ext = hi - lo
    return {"centroid": centroid, "extent": ext, "max_extent": float(ext.max()),
            "n_pts": int(m.sum())}


def filter_reliable_candidate(candidates: list, cam, glb_ext: np.ndarray):
    """GLB size와 가장 매칭되는 candidate 반환 (신뢰 가능)."""
    glb_max = float(glb_ext.max())
    glb_sorted = np.sort(glb_ext)[::-1]
    best_score = 0.0
    best_mask = None
    best_info = None
    for m in candidates:
        if int((m > 0).sum()) < 200:
            continue
        info = mask_to_3d_info(m, cam)
        if info is None or info["max_extent"] < glb_max * 0.25 \
                or info["max_extent"] > glb_max * 1.2:
            continue
        ext_sorted = np.sort(info["extent"])[::-1]
        ratios = [min(a/b, b/a) for a, b in zip(ext_sorted, glb_sorted)
                  if a > 1e-6 and b > 1e-6]
        if not ratios:
            continue
        score = float(np.mean(ratios))
        if ext_sorted[-1] > glb_sorted[-1] * 3.0:
            continue
        if score > best_score:
            best_score = score
            best_mask = m
            best_info = info
    return best_mask, best_info, best_score


def project_3d_to_cam_bbox(centroid_base: np.ndarray, extent: np.ndarray,
                           cam, padding_m: float = 0.02) -> tuple:
    """3D 중심 + extent를 주어진 카메라 이미지로 투영해 bbox 추정."""
    # 3D bbox의 8 corners를 정의
    half = extent / 2 + padding_m
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corners.append(centroid_base + half * np.array([sx, sy, sz]))
    corners = np.array(corners)
    # base → cam
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


def sam_with_bbox(predictor, bgr, bbox, point=None):
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
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k5)


# Viz
def make_overlay(img, mask, color, alpha=0.55):
    out = img.copy()
    cl = np.zeros_like(img); cl[:] = color
    m = mask > 0
    out[m] = cv2.addWeighted(img, 1-alpha, cl, alpha, 0)[m]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
    return out

def label_bar(w):
    h = 24; bar = np.full((h, w, 3), 50, dtype=np.uint8); part = w // 4
    for i, (n, c) in enumerate(OBJ_COLORS_BGR.items()):
        x = i * part
        cv2.rectangle(bar, (x, 0), (x + part, h), (80, 80, 80), -1)
        cv2.rectangle(bar, (x+4, 4), (x+16, h-4), c, -1)
        cv2.putText(bar, n, (x+22, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220,220,220), 1, cv2.LINE_AA)
    return bar

def compose(top, bot, title_bot="SAM v6"):
    rows = []
    for row, t in [(top, "Original"), (bot, title_bot)]:
        labeled = [cv2.putText(img.copy(), f"cam{ci} {t}", (10, 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)
                   for ci, img in enumerate(row)]
        rows.append(np.hstack(labeled))
    return np.vstack([np.vstack(rows), label_bar(rows[0].shape[1])])


def build(frame_id: str, out_dir: Path, mask_dir: Path):
    print("Loading SAM...")
    sam = sam_model_registry["vit_t"](checkpoint=str(SAM_WEIGHTS))
    sam.to("cpu"); sam.eval()
    predictor = SamPredictor(sam)

    ext_by_obj = {n: glb_extent(DATA_DIR / f"{n}.glb") for n in OBJ_COLORS_BGR}

    intrinsics, extrinsics = load_calibration(DATA_DIR, INTR_DIR)
    frames = load_frame(DATA_DIR, frame_id, intrinsics, extrinsics)

    per_obj_masks = {}
    for obj_name in OBJ_COLORS_BGR:
        print(f"\n=== {obj_name} ===")
        glb_ext = ext_by_obj[obj_name]
        is_cyl = OBJECT_SYMMETRY.get(obj_name, "none") == "yaw"

        # 1) 각 cam의 candidate 생성 + 신뢰도 평가
        per_cam_reliable = []
        for ci, cam in enumerate(frames):
            color_m = color_mask_for_object(cam.color_bgr, obj_name)
            cands = connected_components_sorted(color_m, min_area=200, top_k=5)
            mask, info, score = filter_reliable_candidate(cands, cam, glb_ext)
            per_cam_reliable.append((mask, info, score))
            if info is not None:
                print(f"  cam{ci}: reliable score={score:.3f} "
                      f"centroid={info['centroid']} max_ext={info['max_extent']:.3f}m")
            else:
                print(f"  cam{ci}: no reliable candidate")

        # 2) 신뢰 cam들의 3D centroid로 consensus 3D 위치 계산
        reliable = [(ci, info) for ci, (_, info, s) in enumerate(per_cam_reliable)
                    if info is not None and s > 0.6]
        if len(reliable) == 0:
            print(f"  [WARN] no reliable camera, use top candidate of each")
            # fallback: score 관계없이 top 사용
            reliable = [(ci, info) for ci, (_, info, s) in enumerate(per_cam_reliable)
                        if info is not None]

        if not reliable:
            print(f"  [FAIL] {obj_name}: all cameras rejected")
            per_obj_masks[obj_name] = [np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8)
                                       for cam in frames]
            continue

        # 3D centroid fusion (median across reliable cams)
        centroids = np.array([info["centroid"] for _, info in reliable])
        consensus_centroid = np.median(centroids, axis=0)
        consensus_extent = np.median(np.array([info["extent"] for _, info in reliable]),
                                      axis=0)
        # cam들이 일치하지 않으면(centroid 분산 > 5cm) 가장 score 높은 cam 기준
        if len(reliable) >= 2:
            spread = np.linalg.norm(centroids - consensus_centroid, axis=1).max()
            if spread > 0.05:
                best_ci = max(reliable, key=lambda x:
                              next(s for c,(_,_,s) in enumerate(per_cam_reliable) if c==x[0]))
                consensus_centroid = best_ci[1]["centroid"]
                consensus_extent = best_ci[1]["extent"]
                print(f"  [spread={spread*100:.1f}cm too big, use best cam{best_ci[0]}]")

        print(f"  consensus 3D: center={consensus_centroid} ext={consensus_extent}")

        # 3) 각 cam에 대해 최종 mask 결정
        final = []
        for ci, cam in enumerate(frames):
            reliable_mask, reliable_info, reliable_score = per_cam_reliable[ci]

            # cam0/cam2 처럼 신뢰 못하면 cross-cam bbox 사용
            if reliable_score < 0.6 or reliable_mask is None:
                bbox = project_3d_to_cam_bbox(consensus_centroid, consensus_extent,
                                              cam, padding_m=0.02)
                if bbox is None:
                    final.append(np.zeros(cam.color_bgr.shape[:2], dtype=np.uint8))
                    print(f"  cam{ci}: bbox projection 실패 → skip")
                    continue
                # bbox 중심을 point prompt로
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                print(f"  cam{ci}: cross-cam bbox={bbox} center=({cx},{cy})")
                with torch.no_grad():
                    m = sam_with_bbox(predictor, cam.color_bgr, bbox, point=(cx, cy))
                final.append(m)
                print(f"  cam{ci}: cross-cam SAM area={int((m>0).sum())}")
            else:
                # 신뢰 가능한 경우도 SAM 정제
                ys, xs = np.where(reliable_mask > 0)
                bbox = (xs.min(), ys.min(), xs.max(), ys.max())
                cx, cy = int(xs.mean()), int(ys.mean())
                with torch.no_grad():
                    m = sam_with_bbox(predictor, cam.color_bgr, bbox, point=(cx, cy))
                # keep largest overlap with reliable_mask
                n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
                if n > 2:
                    best_i, best_ov = -1, -1
                    for i in range(1, n):
                        ov = int(((labels == i) & (reliable_mask > 0)).sum())
                        if ov > best_ov:
                            best_ov = ov; best_i = i
                    if best_i > 0:
                        m = np.where(labels == best_i, 255, 0).astype(np.uint8)
                final.append(m)
                print(f"  cam{ci}: reliable SAM area={int((m>0).sum())}")

        per_obj_masks[obj_name] = final

    mask_dir.mkdir(parents=True, exist_ok=True)
    for n, ms in per_obj_masks.items():
        for ci, m in enumerate(ms):
            cv2.imwrite(str(mask_dir / f"{n}_cam{ci}.png"), m)
    (mask_dir / "meta.json").write_text(json.dumps(
        {"frame_id": frame_id, "method": "SAM + cross-cam bbox projection"}, indent=2))
    print(f"\nmasks saved: {mask_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    originals = [cam.color_bgr.copy() for cam in frames]
    all_ov = [cam.color_bgr.copy() for cam in frames]
    for n, ms in per_obj_masks.items():
        c = OBJ_COLORS_BGR[n]
        for ci in range(3):
            all_ov[ci] = make_overlay(all_ov[ci], ms[ci], c)
    cv2.imwrite(str(out_dir / "comparison.png"), compose(originals, all_ov))
    for n, ms in per_obj_masks.items():
        c = OBJ_COLORS_BGR[n]
        ovs = []
        for ci, cam in enumerate(frames):
            ov = make_overlay(cam.color_bgr.copy(), ms[ci], c)
            a = int((ms[ci] > 0).sum())
            cv2.putText(ov, f"{n} area={a}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
            ovs.append(ov)
        cv2.imwrite(str(out_dir / f"comparison_{n}.png"),
                    compose(originals, ovs, title_bot=f"{n}"))
    print(f"overlays saved: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default="000000")
    args = ap.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("src/output/ideal_overlay") / f"sam_v6_{ts}_frame_{args.frame_id}"
    mask_dir = Path("src/output/sam_masks") / f"v6_{ts}_frame_{args.frame_id}"
    build(args.frame_id, out_dir, mask_dir)


if __name__ == "__main__":
    main()
