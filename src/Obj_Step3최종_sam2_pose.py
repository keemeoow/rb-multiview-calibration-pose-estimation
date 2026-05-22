#!/usr/bin/env python3
"""
SAM 2 기반 multi-cam mask 생성 (카메라별 × 물체별 직접 클릭).

[실행명령어]
conda activate sam2env

PYTHONWARNINGS=ignore python3 src/Obj_Step3최종_sam2_pose.py \
  --capture_dir ./capture_obj \
  --masks_dir ./masks \
  --sam_checkpoint ~/sam2_checkpoints/sam2_hiera_large.pt \
  --sam_config configs/sam2/sam2_hiera_l.yaml \
  --num_objects 3 --device cpu

[마스크도 새 폴더]
python src/Obj_Step3최종_sam2_pose.py \
  --capture_dir ./capture_obj_set2 --masks_dir ./masks_set2 \
  --sam_checkpoint ~/sam2_checkpoints/sam2_hiera_large.pt \
  --sam_config configs/sam2/sam2_hiera_l.yaml \
  --num_objects 3 --device cpu

UI: 한 창씩 (cam, obj) 슬롯 표시. cam0 → obj1, obj2, ... → cam1 → obj1, ... 순서.
  좌클릭   : 양성 포인트
  우클릭   : 음성 포인트 (boundary 정제)
  SPACE    : 현재 마스크 저장 → 다음 (cam, obj)
  n        : 이 (cam, obj) skip (해당 뷰에서 객체 안 보임)
  b        : 마지막 클릭 취소
  r        : 현재 객체 reset
  q / ESC  : 종료 (지금까지 저장한 건 유지)

출력 (--masks_dir):
  masks/obj{N}/cam{ci}_mask.png    (서브디렉터리 구조; 다운스트림 호환)
  mask_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch  # noqa: F401  (SAM2 가 cuda init 시 필요)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# Calibration loading (intrinsics 만; 전파 없으므로 extrinsics 불필요)
# ============================================================

def load_calib(capture_dir: Path) -> Dict[int, np.ndarray]:
    Ks: Dict[int, np.ndarray] = {}
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
        Ks[ci] = K
    if not Ks:
        raise FileNotFoundError(f"No cam*_K.txt found in {capture_dir}")
    return Ks


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
# Interactive labeling (point-prompted SAM2ImagePredictor)
# ============================================================

def _label_one_object(
    predictor: SAM2ImagePredictor,
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
            obj_dir = masks_dir / f"obj{obj_idx}"
            obj_dir.mkdir(parents=True, exist_ok=True)
            out = obj_dir / f"cam{cam_id}_mask.png"
            cv2.imwrite(str(out), (state["mask"].astype(np.uint8) * 255))
            print(f"  [cam{cam_id} obj{obj_idx}] saved {out} ({int(state['mask'].sum())} px)")
            cv2.destroyWindow(window)
            return "saved"


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--capture_dir", default="./capture_obj")
    ap.add_argument("--masks_dir", default="./masks")
    ap.add_argument(
        "--sam_checkpoint",
        default=str(Path.home() / "sam2_checkpoints/sam2_hiera_large.pt"),
    )
    ap.add_argument(
        "--sam_config",
        default="configs/sam2/sam2_hiera_l.yaml",
        help="hydra config (SAM 2 repo 내부 경로)",
    )
    ap.add_argument("--num_objects", type=int, default=3,
                    help="카메라당 라벨링할 객체 수")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir).resolve()
    masks_dir = Path(args.masks_dir).resolve()
    masks_dir.mkdir(parents=True, exist_ok=True)

    Ks = load_calib(capture_dir)
    cam_ids = sorted(Ks.keys())

    print(f"[INFO] Loading SAM 2 from {args.sam_checkpoint}  device={args.device}")
    model = build_sam2(args.sam_config, args.sam_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(model)

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
        warn = "  [WARN] <2 cams → metric scale unreliable" if len(cams) < 2 else ""
        print(f"  obj{oi}: cams={cams}{warn}")

    print(f"\nNext: python src/instantmesh_pose.py "
          f"--data_dir {capture_dir} --mask_dir {masks_dir} "
          f"--out_dir ./outputs --depth_scale {load_depth_scale(capture_dir)}")


if __name__ == "__main__":
    main()
