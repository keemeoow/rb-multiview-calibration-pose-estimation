#!/usr/bin/env python3
"""GLB 기반 블록 탐지: Template Matching vs YOLOv8-seg 비교.

방법 A: GLB 렌더링 → multi-scale template matching → 2D bbox → depth backproject → pose
방법 B: YOLOv8-seg 자동 라벨링 → 학습 → 추론 → mask → depth backproject → pose

사용:
  # 방법 A: template matching (학습 불필요)
  python3 src/detect_blocks.py --method template --frame_id 000010

  # 방법 B: YOLO 학습 + 추론
  python3 src/detect_blocks.py --method yolo --frame_id 000010

  # 비교
  python3 src/detect_blocks.py --method compare --frame_id 000010
"""

import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

# ─── 공통: GLB 렌더링 ───

def load_glb_for_detection(glb_path: Path):
    """GLB → (mesh, center, extents, face_colors)"""
    scene = trimesh.load(str(glb_path))
    mesh = trimesh.util.concatenate(list(scene.geometry.values())) \
        if isinstance(scene, trimesh.Scene) else scene
    center = mesh.centroid.copy()
    extents = mesh.bounding_box.extents.copy()
    try:
        fc = np.asarray(mesh.visual.vertex_colors)[:, :3].mean(axis=0).astype(np.uint8)
    except Exception:
        fc = np.array([128, 128, 128], dtype=np.uint8)
    return mesh, center, extents, fc


def render_glb_template(mesh, center, K, img_size, R, t, scale=1.0):
    """GLB를 주어진 pose로 카메라 이미지에 렌더링 → (mask, bbox)"""
    h, w = img_size
    verts = (mesh.vertices - center) * scale
    verts_cam = (R @ verts.T).T + t

    z = verts_cam[:, 2]
    front = z > 0.05
    if front.sum() < 3:
        return None, None

    u = (K[0, 0] * verts_cam[front, 0] / z[front] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * verts_cam[front, 1] / z[front] + K[1, 2]).astype(np.int32)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() < 3:
        return None, None

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.stack([u[ok], v[ok]], axis=1)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return mask, bbox


# ─── 방법 A: Template Matching ───

def generate_templates(mesh, center, K, img_size, dist=0.5, n_yaw=36):
    """GLB를 여러 yaw로 렌더링해서 template 목록 생성."""
    templates = []
    for yi in range(n_yaw):
        yaw = yi * 2 * np.pi / n_yaw
        R = Rot.from_euler('y', yaw).as_matrix()
        t = np.array([0, 0, dist])
        mask, bbox = render_glb_template(mesh, center, K, img_size, R, t)
        if mask is not None and bbox is not None:
            x0, y0, x1, y1 = bbox
            if (x1 - x0) > 5 and (y1 - y0) > 5:
                crop = mask[y0:y1, x0:x1]
                templates.append({
                    'mask': crop, 'yaw': yaw, 'R': R, 't': t,
                    'bbox_wh': (x1 - x0, y1 - y0),
                })
    return templates


def detect_by_template(image_bgr, templates, mean_color_bgr,
                       scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    """Multi-scale template matching으로 물체 위치 찾기.

    GLB 평균 색상으로 color mask 만들고, template silhouette과 매칭.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # GLB 평균 색상의 HSV
    bgr_pixel = np.uint8([[mean_color_bgr]])
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
    h_target = float(hsv_pixel[0])

    # 색상 기반 전경 mask
    h_diff = np.abs(hsv[:, :, 0].astype(float) - h_target)
    h_diff = np.minimum(h_diff, 180 - h_diff)
    color_mask = ((h_diff < 25) & (hsv[:, :, 1] > 30)).astype(np.uint8) * 255

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    best_score = -1
    best_loc = None
    best_tmpl = None
    best_scale = 1.0

    for tmpl in templates:
        for s in scales:
            tw = int(tmpl['bbox_wh'][0] * s)
            th = int(tmpl['bbox_wh'][1] * s)
            if tw < 10 or th < 10 or tw > image_bgr.shape[1] or th > image_bgr.shape[0]:
                continue
            resized = cv2.resize(tmpl['mask'], (tw, th))

            # color mask와 template matching
            if color_mask.shape[0] >= th and color_mask.shape[1] >= tw:
                result = cv2.matchTemplate(color_mask, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_loc = max_loc
                    best_tmpl = tmpl
                    best_scale = s

    if best_loc is None or best_score < 0.1:
        return None

    tw = int(best_tmpl['bbox_wh'][0] * best_scale)
    th = int(best_tmpl['bbox_wh'][1] * best_scale)
    x0, y0 = best_loc
    return {
        'bbox': (x0, y0, x0 + tw, y0 + th),
        'score': best_score,
        'yaw': best_tmpl['yaw'],
        'scale': best_scale,
    }


def detect_all_template(frames_bgr, K_list, glb_info, dist=0.5):
    """3대 카메라에서 template matching으로 물체 탐지."""
    mesh, center, extents, mean_rgb = glb_info
    mean_bgr = np.array([mean_rgb[2], mean_rgb[1], mean_rgb[0]], dtype=np.uint8)

    results = []
    for ci, (img, K) in enumerate(zip(frames_bgr, K_list)):
        h, w = img.shape[:2]
        templates = generate_templates(mesh, center, K, (h, w), dist=dist)
        det = detect_by_template(img, templates, mean_bgr)
        if det is not None:
            det['cam_id'] = ci
            results.append(det)
    return results


# ─── 방법 B: YOLOv8-seg ───

def auto_label_for_yolo(data_dir: Path, capture_subdir: str, glb_paths: Dict[str, Path],
                        frame_glb_map: Dict[int, str], output_dir: Path):
    """GLB 렌더링 + depth로 자동 YOLO 학습 데이터 생성.

    각 프레임의 물체 영역을 GLB color mask로 근사해서 polygon annotation 생성.
    """
    from pose_pipeline import (load_calibration, load_frame, estimate_table_plane,
                               get_above_table_points, find_all_clusters,
                               normalize_glb, transform_points, backproject_depth,
                               DEPTH_POLICY)

    intrinsics_dir = data_dir.parent / 'intrinsics'
    intrinsics, extrinsics = load_calibration(data_dir, intrinsics_dir)

    images_dir = output_dir / 'images' / 'train'
    labels_dir = output_dir / 'labels' / 'train'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_names = ['object_001', 'object_002', 'object_003', 'object_004']

    for fid_int, model_name in frame_glb_map.items():
        fid = f"{fid_int:06d}"
        class_id = class_names.index(model_name) if model_name in class_names else -1
        if class_id < 0:
            continue

        try:
            frames = load_frame(data_dir, fid, intrinsics, extrinsics,
                               capture_subdir=capture_subdir)
        except Exception:
            continue

        glb_path = glb_paths.get(model_name)
        if glb_path is None:
            continue

        model_info = load_glb_for_detection(glb_path)
        mean_rgb = model_info[3]
        mean_hsv = cv2.cvtColor(np.uint8([[mean_rgb]]), cv2.COLOR_RGB2HSV)[0, 0]

        for cam in frames:
            img = cam.color_bgr
            h, w = img.shape[:2]
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # GLB 색상 기반 mask
            h_diff = np.abs(hsv[:, :, 0].astype(float) - float(mean_hsv[0]))
            h_diff = np.minimum(h_diff, 180 - h_diff)
            mask = ((h_diff < 20) & (hsv[:, :, 1] > 40)).astype(np.uint8) * 255

            # morphology
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # 가장 큰 contour = 물체
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            # YOLO seg format: class_id x1 y1 x2 y2 ... (normalized polygon)
            polygon = cnt.reshape(-1, 2).astype(float)
            polygon[:, 0] /= w
            polygon[:, 1] /= h

            # 이미지 저장
            img_name = f"cam{cam.cam_id}_{fid}.jpg"
            cv2.imwrite(str(images_dir / img_name), img)

            # 라벨 저장
            label_name = f"cam{cam.cam_id}_{fid}.txt"
            pts_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            (labels_dir / label_name).write_text(f"{class_id} {pts_str}\n")

    # data.yaml
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/train

names:
  0: object_001
  1: object_002
  2: object_003
  3: object_004
"""
    (output_dir / 'data.yaml').write_text(yaml_content)
    print(f"YOLO dataset: {output_dir}")
    print(f"  images: {len(list(images_dir.glob('*.jpg')))}")
    print(f"  labels: {len(list(labels_dir.glob('*.txt')))}")


def train_yolo(data_yaml: Path, epochs=50, imgsz=640):
    """YOLOv8n-seg 학습."""
    from ultralytics import YOLO
    model = YOLO('yolov8n-seg.pt')
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        project=str(data_yaml.parent / 'runs'),
        name='blocks_seg',
        exist_ok=True,
    )
    best_path = data_yaml.parent / 'runs' / 'blocks_seg' / 'weights' / 'best.pt'
    return best_path


def detect_yolo(model_path: Path, image_bgr, conf=0.3):
    """YOLOv8-seg 추론 → detections list."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    results = model(image_bgr, conf=conf, verbose=False)
    detections = []
    for r in results:
        if r.masks is None:
            continue
        for i, (box, mask) in enumerate(zip(r.boxes, r.masks)):
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            mask_np = mask.data[0].cpu().numpy().astype(np.uint8) * 255
            mask_np = cv2.resize(mask_np, (image_bgr.shape[1], image_bgr.shape[0]))
            x0, y0, x1, y1 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                'class_id': cls_id,
                'class_name': f"object_{cls_id+1:03d}",
                'score': score,
                'bbox': (int(x0), int(y0), int(x1), int(y1)),
                'mask': mask_np,
            })
    return detections


# ─── 공통: bbox → depth backproject → pose ───

def bbox_to_3d_position(bbox, depth_u16, K, depth_scale, T_base_cam):
    """2D bbox 중심의 depth → 3D 위치."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    # bbox 중심 주변 depth 중앙값
    r = 5
    patch = depth_u16[max(0, cy-r):cy+r, max(0, cx-r):cx+r].astype(float) * depth_scale
    valid = patch[patch > 0.05]
    if len(valid) == 0:
        return None
    z = float(np.median(valid))
    x = (cx - K[0, 2]) * z / K[0, 0]
    y = (cy - K[1, 2]) * z / K[1, 1]
    pt_cam = np.array([x, y, z, 1.0])
    pt_base = T_base_cam @ pt_cam
    return pt_base[:3]


# ─── 메인: 비교 실행 ───

def run_comparison(data_dir, intrinsics_dir, frame_id, capture_subdir, glb_dir):
    """Template matching vs YOLO 비교."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pose_pipeline import (load_calibration, load_frame, normalize_glb,
                               estimate_table_plane, get_above_table_points,
                               find_all_clusters, FRAME_TO_GLB, OBJECT_LABELS,
                               sample_model_points, register_table_grounded,
                               project_cluster_to_support_masks, make_pose_estimate,
                               render_wireframe, transform_points, backproject_depth,
                               DEPTH_POLICY)

    data_dir = Path(data_dir)
    intrinsics_dir = Path(intrinsics_dir)
    glb_dir = Path(glb_dir) if glb_dir else data_dir

    intrinsics, extrinsics = load_calibration(data_dir, intrinsics_dir)
    frames = load_frame(data_dir, frame_id, intrinsics, extrinsics,
                       capture_subdir=capture_subdir)

    # GLB
    model_name = FRAME_TO_GLB.get(int(frame_id))
    if model_name is None:
        print(f"프레임 {frame_id}에 대한 GLB 매핑 없음")
        return

    glb_path = glb_dir / f"{model_name}.glb"
    if not glb_path.exists():
        print(f"GLB 없음: {glb_path}")
        return

    print(f"프레임 {frame_id}: {model_name} ({OBJECT_LABELS.get(model_name, '')})")
    print(f"GLB: {glb_path}")
    print()

    glb_info = load_glb_for_detection(glb_path)
    frames_bgr = [cam.color_bgr for cam in frames]
    K_list = [cam.intrinsics.K for cam in frames]

    # ─── 방법 A: Template Matching ───
    print("=" * 50)
    print("방법 A: Template Matching")
    print("=" * 50)

    t0 = time.time()
    detections_A = detect_all_template(frames_bgr, K_list, glb_info, dist=0.5)
    t_A = time.time() - t0

    print(f"  시간: {t_A:.2f}초")
    print(f"  탐지: {len(detections_A)}개 카메라")

    for det in detections_A:
        print(f"    cam{det['cam_id']}: bbox={det['bbox']} score={det['score']:.3f}")
        # 3D 위치
        cam = frames[det['cam_id']]
        pos = bbox_to_3d_position(det['bbox'], cam.depth_u16, cam.intrinsics.K,
                                  cam.intrinsics.depth_scale, cam.T_base_cam)
        if pos is not None:
            print(f"      3D position: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]m")

    # 시각화
    vis_A = []
    for ci, cam in enumerate(frames):
        img = cam.color_bgr.copy()
        for det in detections_A:
            if det['cam_id'] == ci:
                x0, y0, x1, y1 = det['bbox']
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(img, f"{det['score']:.2f}", (x0, y0 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        vis_A.append(img)

    # ─── 방법 B: YOLO ───
    print()
    print("=" * 50)
    print("방법 B: YOLOv8-seg")
    print("=" * 50)

    yolo_weights = Path('src/models/blocks_yolov8n_seg.pt')
    yolo_dataset_dir = Path('/tmp/yolo_blocks_dataset')

    if not yolo_weights.exists():
        print("  YOLO weights 없음 → 자동 라벨링 + 학습 시작")

        # 자동 라벨링
        glb_paths = {}
        for i in range(1, 5):
            p = glb_dir / f"object_{i:03d}.glb"
            if p.exists():
                glb_paths[f"object_{i:03d}"] = p

        auto_label_for_yolo(
            data_dir, capture_subdir, glb_paths, FRAME_TO_GLB,
            yolo_dataset_dir,
        )

        # 학습
        t0_train = time.time()
        yolo_weights = train_yolo(yolo_dataset_dir / 'data.yaml', epochs=30, imgsz=640)
        t_train = time.time() - t0_train
        print(f"  학습 시간: {t_train:.1f}초")

        # 모델 복사
        Path('src/models').mkdir(exist_ok=True)
        import shutil
        shutil.copy2(str(yolo_weights), 'src/models/blocks_yolov8n_seg.pt')
        yolo_weights = Path('src/models/blocks_yolov8n_seg.pt')

    # 추론
    t0 = time.time()
    detections_B = []
    for ci, cam in enumerate(frames):
        dets = detect_yolo(yolo_weights, cam.color_bgr)
        for d in dets:
            d['cam_id'] = ci
        detections_B.extend(dets)
    t_B = time.time() - t0

    print(f"  추론 시간: {t_B:.2f}초")
    print(f"  탐지: {len(detections_B)}개")

    for det in detections_B:
        print(f"    cam{det['cam_id']}: {det['class_name']} score={det['score']:.3f} bbox={det['bbox']}")
        cam = frames[det['cam_id']]
        pos = bbox_to_3d_position(det['bbox'], cam.depth_u16, cam.intrinsics.K,
                                  cam.intrinsics.depth_scale, cam.T_base_cam)
        if pos is not None:
            print(f"      3D position: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]m")

    # 시각화
    vis_B = []
    for ci, cam in enumerate(frames):
        img = cam.color_bgr.copy()
        for det in detections_B:
            if det['cam_id'] == ci:
                x0, y0, x1, y1 = det['bbox']
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(img, f"{det['class_name']} {det['score']:.2f}",
                           (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        vis_B.append(img)

    # ─── 비교 이미지 생성 ───
    print()
    print("=" * 50)
    print("비교 결과")
    print("=" * 50)
    print(f"  Template Matching: {t_A:.2f}초, {len(detections_A)} detections")
    print(f"  YOLOv8-seg:        {t_B:.2f}초, {len(detections_B)} detections")

    out_dir = Path('src/output/detection_compare')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3행: 원본 / Template / YOLO
    orig_row = np.hstack(frames_bgr)
    tmpl_row = np.hstack(vis_A)
    yolo_row = np.hstack(vis_B)

    # 높이 맞추기
    target_w = orig_row.shape[1]
    for row_name, row in [('tmpl', tmpl_row), ('yolo', yolo_row)]:
        if row.shape[1] != target_w:
            s = target_w / row.shape[1]
            row = cv2.resize(row, (target_w, int(row.shape[0] * s)))

    cv2.putText(orig_row, f'Original RGB (frame {frame_id})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(tmpl_row, f'A: Template Matching ({t_A:.2f}s)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(yolo_row, f'B: YOLOv8-seg ({t_B:.2f}s)', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    combined = np.vstack([orig_row, tmpl_row, yolo_row])
    save_path = out_dir / f'compare_{frame_id}.png'
    cv2.imwrite(str(save_path), combined)
    print(f"\n  비교 이미지: {save_path}")

    return {
        'template': {'time': t_A, 'detections': len(detections_A)},
        'yolo': {'time': t_B, 'detections': len(detections_B)},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='src/data')
    parser.add_argument('--intrinsics_dir', default='src/intrinsics')
    parser.add_argument('--frame_id', default='000010')
    parser.add_argument('--capture_subdir', default='object_capture_blocks(1)')
    parser.add_argument('--glb_dir', default=None)
    parser.add_argument('--method', default='compare', choices=['template', 'yolo', 'compare'])
    args = parser.parse_args()

    glb_dir = args.glb_dir or args.data_dir
    run_comparison(args.data_dir, args.intrinsics_dir, args.frame_id,
                   args.capture_subdir, glb_dir)
