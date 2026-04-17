#!/usr/bin/env python3
"""
멀티뷰 원본 vs 포즈 추정 실루엣 오버레이 비교 이미지.

출력:
  src/output/pose_per_object/comparison_{frame}/comparison_{frame}.png
  src/output/pose_per_object/comparison_{frame}/comparison_{frame}_object_XXX.png
  (기본 비교 1장 + 물체별 pose 텍스트 비교 이미지)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
INTR_DIR = SCRIPT_DIR / "intrinsics"
CALIB_DIR = DATA_DIR / "cube_session_01" / "calib_out_cube"
POSE_DIR = SCRIPT_DIR / "output" / "pose_per_object"
OUT_ROOT = SCRIPT_DIR / "output" / "pose_per_object"

OBJ_COLORS_BGR = {
    "object_001": (80, 80, 255),    # 빨강
    "object_002": (80, 230, 255),   # 노랑
    "object_003": (255, 140, 40),   # 곤색
    "object_004": (200, 230, 90),   # 민트
}


def frame_dir(frame_id):
    return POSE_DIR / f"frame_{frame_id}"


def load_pose_infos(frame_id):
    """frame_{id}/ 새 레이아웃 우선, 없으면 과거 flat 레이아웃 fallback."""
    poses = {}
    fd = frame_dir(frame_id)
    for oi in range(1, 5):
        name = f"object_{oi:03d}"
        candidates = [
            fd / f"pose_{name}.json",
            POSE_DIR / f"pose_{name}_{frame_id}.json",
        ]
        for pj in candidates:
            if pj.exists():
                poses[name] = json.loads(pj.read_text())
                break
    return poses


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
    fd = frame_dir(frame_id)
    for oi in range(1, 5):
        name = f"object_{oi:03d}"
        candidates = [
            fd / f"{name}_posed.glb",
            POSE_DIR / f"{name}_posed_{frame_id}.glb",
        ]
        glb = next((p for p in candidates if p.exists()), None)
        if glb is None:
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


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _fmt_vec(values, digits):
    if values is None or len(values) != 3:
        return "n/a"
    fmt = f"{{:+.{digits}f}}"
    return "[" + ", ".join(fmt.format(float(v)) for v in values) + "]"


def pose_lines(pose_info):
    scale = pose_info.get("scale")
    scale_txt = f"{float(scale):.4f}" if scale is not None else "n/a"
    position_txt = _fmt_vec(pose_info.get("position_m"), 4)
    euler_txt = _fmt_vec(pose_info.get("euler_xyz_deg"), 1)
    return [
        f"scale: {scale_txt}",
        f"position: {position_txt} m",
        f"euler_xyz_deg: {euler_txt}",
    ]


def boxes_overlap(a, b, pad=8):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad < bx0 or bx1 + pad < ax0 or ay1 + pad < by0 or by1 + pad < ay0)


def clamp_rect(x, y, box_w, box_h, img_w, img_h, margin_x=6, margin_top=40, margin_bottom=32):
    x = int(np.clip(x, margin_x, max(margin_x, img_w - box_w - margin_x)))
    y = int(np.clip(y, margin_top, max(margin_top, img_h - box_h - margin_bottom)))
    return x, y, x + box_w, y + box_h


def choose_text_box(bbox, box_w, box_h, img_w, img_h, occupied):
    x0, y0, x1, y1 = bbox
    gap = 6
    center_y = (y0 + y1) // 2
    candidates = [
        (x1 + gap, y0),
        (x1 + gap, center_y - box_h // 2),
        (x0 - box_w - gap, y0),
        (x0 - box_w - gap, center_y - box_h // 2),
        (x1 + gap, y1 - box_h),
        (x0 - box_w - gap, y1 - box_h),
        (x0, y0 - box_h - gap),
        (x0, y1 + gap),
    ]
    blockers = occupied + [bbox]
    for cx, cy in candidates:
        rect = clamp_rect(cx, cy, box_w, box_h, img_w, img_h)
        if all(not boxes_overlap(rect, other) for other in blockers):
            return rect

    rect = clamp_rect(x1 + gap, center_y - box_h // 2, box_w, box_h, img_w, img_h)
    step = box_h // 2 + gap
    for _ in range(20):
        if all(not boxes_overlap(rect, other) for other in blockers):
            return rect
        rect = clamp_rect(rect[0], rect[1] + step, box_w, box_h, img_w, img_h)
    return rect


def draw_pose_text(img, mask, pose_info, color_bgr, occupied):
    bbox = mask_bbox(mask)
    if bbox is None or not pose_info:
        return img

    lines = pose_lines(pose_info)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.47
    thickness = 1
    outline = 2
    padding = 2
    line_gap = 6
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_h = max(h for _, h in text_sizes)
    box_w = max(w for w, _ in text_sizes) + padding * 2
    box_h = len(lines) * text_h + (len(lines) - 1) * line_gap + padding * 2
    rect = choose_text_box(bbox, box_w, box_h, img.shape[1], img.shape[0], occupied)
    occupied.append(rect)

    x0, y0, x1, y1 = rect

    bx0, by0, bx1, by1 = bbox
    anchor_y = (by0 + by1) // 2
    if x0 >= bx1:
        anchor_start = (bx1, anchor_y)
        anchor_end = (x0 - 2, y0 + text_h // 2)
    elif x1 <= bx0:
        anchor_start = (bx0, anchor_y)
        anchor_end = (x1 + 2, y0 + text_h // 2)
    elif y0 >= by1:
        anchor_start = ((bx0 + bx1) // 2, by1)
        anchor_end = (x0 + box_w // 2, y0 - 2)
    else:
        anchor_start = ((bx0 + bx1) // 2, by0)
        anchor_end = (x0 + box_w // 2, y1 + 2)
    cv2.line(img, anchor_start, anchor_end, color_bgr, 1, cv2.LINE_AA)

    baseline_y = y0 + text_h
    for line in lines:
        org = (x0 + padding, baseline_y)
        cv2.putText(img, line, org, font, font_scale, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(img, line, org, font, font_scale, color_bgr, thickness, cv2.LINE_AA)
        baseline_y += text_h + line_gap
    return img


def make_top_panels(rgbs, frame_id):
    top = [img.copy() for img in rgbs]
    for ci, img in enumerate(top):
        cv2.putText(img, f"cam{ci} Original (frame {frame_id})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return top


def decorate_overlay_panels(overlays, name_labels):
    decorated = [img.copy() for img in overlays]
    for ci, img in enumerate(decorated):
        cv2.putText(img, f"cam{ci} Pose Overlay", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA)
    legend = " | ".join(n for n, _ in name_labels)
    for img in decorated:
        cv2.putText(img, legend, (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)
    return decorated


def compose_canvas(top_panels, bottom_panels):
    return np.vstack([np.hstack(top_panels), np.hstack(bottom_panels)])


def build_comparison(frame_id):
    Ks, sizes = load_intrinsics()
    ext = load_extrinsics()
    rgbs = load_rgbs(frame_id)
    posed = load_posed_meshes(frame_id)
    pose_infos = load_pose_infos(frame_id)
    if not posed:
        raise RuntimeError(f"posed GLB 없음 (frame {frame_id})")

    overlays = [img.copy() for img in rgbs]
    per_object_annotations = {}
    name_labels = []
    for name, mesh in posed:
        color = OBJ_COLORS_BGR.get(name, (0, 255, 0))
        name_labels.append((name, color))
        per_object_annotations[name] = {
            "color": color,
            "pose_info": pose_infos.get(name),
            "masks": [],
        }
        for ci in range(3):
            h, w = rgbs[ci].shape[:2]
            silh = render_silhouette(mesh, Ks[ci], ext[ci], (h, w))
            overlays[ci] = overlay(overlays[ci], silh, color, alpha=0.55)
            per_object_annotations[name]["masks"].append(silh)

    out_dir = frame_dir(frame_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.png"
    top_panels = make_top_panels(rgbs, frame_id)
    base_overlays = decorate_overlay_panels(overlays, name_labels)
    cv2.imwrite(str(out_path), compose_canvas(top_panels, base_overlays))

    saved_object_paths = []
    for name, ann in per_object_annotations.items():
        if ann["pose_info"] is None:
            continue
        object_overlays = [img.copy() for img in base_overlays]
        occupied_boxes = [[] for _ in object_overlays]
        for ci, silh in enumerate(ann["masks"]):
            object_overlays[ci] = draw_pose_text(
                object_overlays[ci], silh, ann["pose_info"], ann["color"], occupied_boxes[ci]
            )
        object_path = out_dir / f"comparison_{name}.png"
        cv2.imwrite(str(object_path), compose_canvas(top_panels, object_overlays))
        saved_object_paths.append(object_path)

    print(f"saved: {out_path}  (objects: {', '.join(n for n,_ in name_labels)})")
    for path in saved_object_paths:
        print(f"saved: {path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_id", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        # 새 레이아웃: POSE_DIR/frame_XXXXXX/
        fids = sorted({p.name.replace("frame_", "")
                       for p in POSE_DIR.glob("frame_*") if p.is_dir()})
        if not fids:
            # 레거시 fallback
            fids = sorted({p.stem.split("_")[-1]
                           for p in POSE_DIR.glob("object_*_posed_*.glb")
                           if "_isaac" not in p.stem})
        for fid in fids:
            build_comparison(fid)
    else:
        build_comparison(args.frame_id or "000000")


if __name__ == "__main__":
    main()
