#!/usr/bin/env python3
"""
capture_obj/ flat layout 에서 카메라별 RGB 위에 클릭으로 물체 마스크 생성.
(기본 모드 = interactive. auto 모드는 fallback 으로 유지)

interactive 사용 예 (Recommended):
  python src/sam_pose.py \
    --capture_dir ./capture_obj --masks_dir ./masks \
    --sam_checkpoint src/weights/mobile_sam.pt \
    --num_objects 3

  조작:
    좌클릭   : positive point (객체에 속함)
    우클릭   : negative point (객체 아님 / boundary 정제)
    SPACE    : 현재 마스크 저장하고 다음 객체로
    n        : 이 (cam, obj) skip (해당 뷰에서 객체 안 보임)
    b        : 마지막 클릭 취소
    r        : 현재 객체 reset
    q / ESC  : 종료 (지금까지 저장한 건 유지)

auto 사용 예 (cluttered scene 에서는 정확도 떨어짐):
  python src/sam_pose.py --mode auto --top_n 3 --eps_mm 50

출력 (--masks_dir):
  cam{i}_obj{N}_mask.png   # improved_instantmesh_pose.py 가 읽는 포맷
  mask_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# ============================================================
# Calibration loading
# ============================================================

def load_calib(capture_dir: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    Ks: Dict[int, np.ndarray] = {}
    Ts: Dict[int, np.ndarray] = {}
    for p in sorted(capture_dir.glob("cam*_K.txt")):
        ci_str = p.stem.replace("cam", "").replace("_K", "")
        try:
            ci = int(ci_str)
        except ValueError:
            print(f"[WARN] skip unparsable {p.name}")
            continue
        K = np.loadtxt(p, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K shape mismatch in {p}: {K.shape}")
        T_path = capture_dir / f"cam{ci}_T_cam_to_world.txt"
        if not T_path.exists():
            raise FileNotFoundError(f"T_cam_to_world missing for cam{ci}: {T_path}")
        T = np.loadtxt(T_path, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"T shape mismatch in {T_path}: {T.shape}")
        Ks[ci], Ts[ci] = K, T
    if not Ks:
        raise FileNotFoundError(f"No cam*_K.txt found in {capture_dir}")
    return Ks, Ts


def load_depth_scale(capture_dir: Path) -> float:
    info_path = capture_dir / "calib_info.json"
    if not info_path.exists():
        return 0.001
    with open(info_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    scales = [c.get("depth_scale_m_per_unit")
              for c in d.get("cameras", []) if c.get("depth_scale_m_per_unit")]
    if not scales:
        return 0.001
    return float(scales[0])


# ============================================================
# Device + MobileSAM
# ============================================================

def pick_device(prefer: str = "cpu") -> str:
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends.mps, "is_available", lambda: False)():
        # NOTE: MPS hits float64 issue inside mobile_sam.AutomaticMaskGenerator.
        # SamPredictor sometimes also crashes; CPU is the safest default.
        return "mps"
    return "cpu"


def load_sam(ckpt: Path, model_type: str, device: str):
    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.to(device=device)
    sam.eval()
    return sam


# ============================================================
# Interactive mode (point-prompted SamPredictor)
# ============================================================

def _label_one_object(
    predictor: SamPredictor,
    rgb_bgr: np.ndarray,
    cam_id: int,
    obj_idx: int,
    masks_dir: Path,
    window_size: Tuple[int, int] = (1024, 768),
) -> str:
    """Returns one of: 'saved', 'skip', 'quit'."""
    state = {"clicks": [], "mask": None}
    window = f"cam{cam_id}  obj{obj_idx}"
    img_h, img_w = rgb_bgr.shape[:2]

    def _refresh():
        if not state["clicks"]:
            state["mask"] = None
        else:
            pts = np.array([[c[0], c[1]] for c in state["clicks"]], dtype=np.float32)
            labels = np.array([c[2] for c in state["clicks"]], dtype=np.int32)
            if (labels == 1).any():
                masks, _, _ = predictor.predict(
                    point_coords=pts,
                    point_labels=labels,
                    multimask_output=False,
                )
                state["mask"] = masks[0].astype(bool)
            else:
                state["mask"] = None
        _redraw()

    def _redraw():
        disp = rgb_bgr.copy()
        if state["mask"] is not None and state["mask"].any():
            overlay = disp.copy()
            overlay[state["mask"]] = (0, 200, 0)
            disp = cv2.addWeighted(disp, 0.55, overlay, 0.45, 0)
        for (x, y, lbl) in state["clicks"]:
            color = (0, 220, 0) if lbl == 1 else (0, 0, 220)
            cv2.circle(disp, (x, y), 6, color, -1)
            cv2.circle(disp, (x, y), 7, (255, 255, 255), 1)
        msg = (f"cam{cam_id} obj{obj_idx}  "
               f"L=+  R=-  SPACE=save  n=skip  b=undo  r=reset  q=quit  "
               f"({len(state['clicks'])} pts)")
        cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(disp, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(window, disp)

    def on_mouse(event, x, y, *_):
        # cv2 sends raw image coordinates when WINDOW_NORMAL + resized window — OK.
        if 0 <= x < img_w and 0 <= y < img_h:
            if event == cv2.EVENT_LBUTTONDOWN:
                state["clicks"].append((x, y, 1))
                _refresh()
            elif event == cv2.EVENT_RBUTTONDOWN:
                state["clicks"].append((x, y, 0))
                _refresh()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, *window_size)
    cv2.setMouseCallback(window, on_mouse)
    _redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyWindow(window)
            return "quit"
        if key == ord("n"):
            print(f"  [cam{cam_id} obj{obj_idx}] skipped (not visible)")
            cv2.destroyWindow(window)
            return "skip"
        if key == ord("b"):
            if state["clicks"]:
                state["clicks"].pop()
                _refresh()
        elif key == ord("r"):
            state["clicks"].clear()
            state["mask"] = None
            _redraw()
        elif key == 32:  # SPACE
            if state["mask"] is None or not state["mask"].any():
                print(f"  [cam{cam_id} obj{obj_idx}] no mask yet — left-click on the object first.")
                continue
            out = masks_dir / f"cam{cam_id}_obj{obj_idx}_mask.png"
            cv2.imwrite(str(out), (state["mask"].astype(np.uint8) * 255))
            print(f"  [cam{cam_id} obj{obj_idx}] saved {out.name} ({int(state['mask'].sum())} px)")
            cv2.destroyWindow(window)
            return "saved"


def run_interactive(args, capture_dir: Path, masks_dir: Path) -> None:
    Ks, _ = load_calib(capture_dir)
    cam_ids = sorted(Ks.keys())

    device = pick_device(args.device)
    print(f"[INFO] Loading MobileSAM on {device}...")
    sam = load_sam(Path(args.sam_checkpoint), args.model_type, device)
    predictor = SamPredictor(sam)

    print(f"\n[INFO] {len(cam_ids)} camera(s) × {args.num_objects} object(s) = "
          f"{len(cam_ids) * args.num_objects} mask slot(s)")
    print("Controls: L=+ R=- SPACE=save n=skip b=undo r=reset q=quit\n")

    summary = []
    aborted = False
    for ci in cam_ids:
        if aborted:
            break
        rgb_path = capture_dir / f"cam{ci}_rgb.png"
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            print(f"[WARN] cam{ci}: failed to load {rgb_path.name}")
            continue
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        print(f"[cam{ci}] computing image embedding ({rgb_bgr.shape[1]}x{rgb_bgr.shape[0]})...")
        predictor.set_image(rgb_rgb)

        for obj_idx in range(1, args.num_objects + 1):
            result = _label_one_object(predictor, rgb_bgr, ci, obj_idx, masks_dir)
            summary.append({"cam_id": ci, "obj_idx": obj_idx, "result": result})
            if result == "quit":
                aborted = True
                break

    cv2.destroyAllWindows()

    with open(masks_dir / "mask_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": "interactive",
            "capture_dir": str(capture_dir),
            "masks_dir": str(masks_dir),
            "num_objects": args.num_objects,
            "events": summary,
        }, f, indent=2)

    saved = [s for s in summary if s["result"] == "saved"]
    skipped = [s for s in summary if s["result"] == "skip"]
    print(f"\n[INFO] saved {len(saved)}, skipped {len(skipped)}"
          + (" (quit early)" if aborted else ""))

    # per-object coverage check
    obj_cams: Dict[int, List[int]] = {}
    for s in saved:
        obj_cams.setdefault(s["obj_idx"], []).append(s["cam_id"])
    for oi in range(1, args.num_objects + 1):
        cams = obj_cams.get(oi, [])
        warn = "  [WARN] <2 cams → metric scale unreliable in improved_instantmesh_pose.py" \
               if len(cams) < 2 else ""
        print(f"  obj{oi}: cams={cams}{warn}")

    print(f"\nNext: python improved_instantmesh_pose.py "
          f"--data_dir {capture_dir} --mask_dir {masks_dir} "
          f"--out_dir ./outputs --depth_scale {load_depth_scale(capture_dir)}")


# ============================================================
# Auto mode (legacy fallback)
# ============================================================

def build_mask_generator(
    ckpt: Path,
    model_type: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    min_mask_region_area: int,
) -> SamAutomaticMaskGenerator:
    sam = load_sam(ckpt, model_type, device)
    print(f"[INFO] MobileSAM AutomaticMaskGenerator on {device}")
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=0,
        min_mask_region_area=min_mask_region_area,
    )


def segment_one_view(
    mask_gen: SamAutomaticMaskGenerator,
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    depth_scale: float,
    min_mask_area_px: int,
    max_mask_area_px: int,
    min_depth_m: float,
    max_depth_m: float,
    workspace_bbox_m: Optional[Tuple[float, float, float, float, float, float]],
) -> List[dict]:
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_gen.generate(rgb_rgb)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)

    clusters: List[dict] = []
    for m in masks:
        mask = m["segmentation"]
        area = int(mask.sum())
        if area < min_mask_area_px or area > max_mask_area_px:
            continue
        v, u = np.where(mask)
        z = depth_m[v, u]
        valid = np.isfinite(z) & (z > min_depth_m) & (z < max_depth_m)
        if valid.sum() < 50:
            continue
        u, v, z = u[valid], v[valid], z[valid]
        x = (u.astype(np.float64) - cx) * z.astype(np.float64) / fx
        y = (v.astype(np.float64) - cy) * z.astype(np.float64) / fy
        pts_cam = np.stack([x, y, z.astype(np.float64), np.ones_like(z, dtype=np.float64)], axis=1)
        pts_world = (T_cam_to_world @ pts_cam.T).T[:, :3]
        if workspace_bbox_m is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = workspace_bbox_m
            in_bb = ((pts_world[:, 0] >= xmin) & (pts_world[:, 0] <= xmax) &
                     (pts_world[:, 1] >= ymin) & (pts_world[:, 1] <= ymax) &
                     (pts_world[:, 2] >= zmin) & (pts_world[:, 2] <= zmax))
            if in_bb.sum() < 50:
                continue
            pts_world = pts_world[in_bb]
        centroid_m = np.median(pts_world, axis=0)
        clusters.append({
            "mask_2d": mask.astype(bool),
            "centroid_mm": centroid_m * 1000.0,
            "n_pts": int(len(pts_world)),
            "area_px": area,
        })
    return clusters


def group_clusters_by_centroid(all_clusters: List[dict], eps_mm: float) -> List[List[dict]]:
    if not all_clusters:
        return []
    n = len(all_clusters)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    centers = np.array([c["centroid_mm"] for c in all_clusters])
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(centers[i] - centers[j])) <= eps_mm:
                union(i, j)
    groups: Dict[int, List[dict]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(all_clusters[i])
    return list(groups.values())


def save_object_masks_auto(final_groups: List[List[dict]], masks_dir: Path) -> List[dict]:
    summary: List[dict] = []
    for obj_idx, gs in enumerate(final_groups, start=1):
        by_cam: Dict[int, np.ndarray] = {}
        for c in gs:
            ci = int(c["cam"])
            m = c["mask_2d"].astype(bool)
            by_cam[ci] = m if ci not in by_cam else (by_cam[ci] | m)
        cams_seen = sorted(by_cam.keys())
        centroid = np.mean([c["centroid_mm"] for c in gs], axis=0)
        total_pts = int(sum(c["n_pts"] for c in gs))
        print(f"[obj{obj_idx}] cams={cams_seen}  pts={total_pts}  "
              f"center≈[{centroid[0]:.0f},{centroid[1]:.0f},{centroid[2]:.0f}]mm")
        if len(cams_seen) < 2:
            print(f"  [WARN] visible in only {len(cams_seen)} camera.")
        for ci, m in by_cam.items():
            out = masks_dir / f"cam{ci}_obj{obj_idx}_mask.png"
            cv2.imwrite(str(out), (m.astype(np.uint8) * 255))
            print(f"  -> {out.name} ({int(m.sum())} px)")
        summary.append({
            "obj_idx": obj_idx,
            "cams_seen": cams_seen,
            "total_n_pts_3d": total_pts,
            "centroid_mm": centroid.tolist(),
        })
    return summary


def run_auto(args, capture_dir: Path, masks_dir: Path) -> None:
    workspace = None
    if args.workspace_bbox_m:
        ws = [float(x) for x in args.workspace_bbox_m.split(",")]
        if len(ws) != 6:
            raise ValueError("--workspace_bbox_m must be 6 floats")
        workspace = tuple(ws)  # type: ignore

    Ks, Ts = load_calib(capture_dir)
    depth_scale = load_depth_scale(capture_dir)
    print(f"[INFO] cameras: {sorted(Ks.keys())}  depth_scale={depth_scale}")

    device = pick_device(args.device)
    mask_gen = build_mask_generator(
        Path(args.sam_checkpoint),
        model_type=args.model_type,
        device=device,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_area_px,
    )

    all_clusters: List[dict] = []
    for ci in sorted(Ks.keys()):
        rgb_path = capture_dir / f"cam{ci}_rgb.png"
        depth_path = capture_dir / f"cam{ci}_depth.png"
        if not rgb_path.exists() or not depth_path.exists():
            print(f"[WARN] cam{ci}: missing rgb/depth")
            continue
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        if rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(f"cam{ci}: rgb/depth size mismatch {rgb.shape[:2]} vs {depth.shape[:2]}")
        print(f"[cam{ci}] SAM auto-mask...")
        cls = segment_one_view(
            mask_gen=mask_gen, rgb_bgr=rgb, depth_u16=depth, K=Ks[ci],
            T_cam_to_world=Ts[ci], depth_scale=depth_scale,
            min_mask_area_px=args.min_mask_area_px,
            max_mask_area_px=args.max_mask_area_px,
            min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m,
            workspace_bbox_m=workspace,
        )
        for c in cls:
            c["cam"] = ci
        all_clusters.extend(cls)
        print(f"  cam{ci}: {len(cls)} clusters")

    if not all_clusters:
        raise RuntimeError("All views produced 0 valid clusters.")

    groups = group_clusters_by_centroid(all_clusters, eps_mm=args.eps_mm)
    print(f"[INFO] {len(groups)} group(s) after clustering (eps={args.eps_mm}mm)")
    main_groups = [gs for gs in groups
                   if sum(c["n_pts"] for c in gs) >= args.min_n_pts_per_object]
    main_groups.sort(key=lambda gs: -sum(c["n_pts"] for c in gs))
    final_groups = main_groups[:args.top_n]
    print(f"[INFO] final objects kept: {len(final_groups)} (top_n={args.top_n})")

    obj_summary = save_object_masks_auto(final_groups, masks_dir)
    with open(masks_dir / "mask_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "mode": "auto",
            "capture_dir": str(capture_dir), "masks_dir": str(masks_dir),
            "depth_scale_m_per_unit": depth_scale,
            "workspace_bbox_m": list(workspace) if workspace else None,
            "objects": obj_summary,
        }, f, indent=2)


# ============================================================
# Main
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["interactive", "auto"], default="interactive")
    p.add_argument("--capture_dir", default="capture_obj")
    p.add_argument("--masks_dir", default="masks")
    p.add_argument("--sam_checkpoint", default="src/weights/mobile_sam.pt")
    p.add_argument("--model_type", default="vit_t")
    p.add_argument("--device", default="cpu", choices=["auto", "cuda", "mps", "cpu"],
                   help="기본 cpu (MPS 는 mobile_sam float64 이슈로 폴백 권장)")

    # interactive
    p.add_argument("--num_objects", type=int, default=3, help="(interactive) 카메라당 라벨링할 객체 수")

    # auto
    p.add_argument("--top_n", type=int, default=3, help="(auto) 최종으로 남길 객체 수")
    p.add_argument("--eps_mm", type=float, default=50.0, help="(auto) cross-view centroid 그룹핑 임계")
    p.add_argument("--min_mask_area_px", type=int, default=400)
    p.add_argument("--max_mask_area_px", type=int, default=80000)
    p.add_argument("--min_n_pts_per_object", type=int, default=5000, help="(auto) 3D point 노이즈 컷오프")
    p.add_argument("--min_depth_m", type=float, default=0.1)
    p.add_argument("--max_depth_m", type=float, default=2.0)
    p.add_argument("--workspace_bbox_m", default="",
                   help="(auto) x_min,x_max,y_min,y_max,z_min,z_max in meters (world frame).")
    p.add_argument("--points_per_side", type=int, default=32)
    p.add_argument("--pred_iou_thresh", type=float, default=0.86)
    p.add_argument("--stability_score_thresh", type=float, default=0.92)

    args = p.parse_args()
    capture_dir = Path(args.capture_dir).resolve()
    masks_dir = Path(args.masks_dir).resolve()
    masks_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "interactive":
        run_interactive(args, capture_dir, masks_dir)
    else:
        run_auto(args, capture_dir, masks_dir)


if __name__ == "__main__":
    main()
