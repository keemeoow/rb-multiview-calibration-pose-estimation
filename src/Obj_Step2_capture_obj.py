#!/usr/bin/env python3
"""
물체 RGB-D 단일 동기 프레임을 improved_instantmesh_pose.py 의 flat layout 으로 직접 저장.

사용 예 (캘리브레이션 K/T 까지 한 번에 dump):
  python src/Obj_Step2_capture_obj.py \
    --out_dir ./capture_obj \
    --intrinsics_dir src/Step0_calibration/intrinsics \
    --transforms_json src/Step0_calibration/data/cube_session_01/calib_out_cube/transforms/T_C0_Ci_all.json \
    --depth_burst_n 10 \
    --show

[새 폴더로 캡처]
python src/Obj_Step2_capture_obj.py \
  --out_dir ./capture_obj_set4 \
  --intrinsics_dir src/Step0_calibration/intrinsics \
  --transforms_json src/Step0_calibration/data/cube_session_01/calib_out_cube/transforms/T_C0_Ci_all.json \
  --depth_burst_n 10 --show


저장 결과 (out_dir/):
  cam{i}_rgb.png
  cam{i}_depth.png            # uint16, color-aligned
  cam{i}_K.txt                # --transforms_json 제공 시
  cam{i}_T_cam_to_world.txt
  calib_info.json

이후:
  1) cam{i}_rgb.png 에 SAM/SAM2 돌려 masks/cam{i}_mask.png 생성
  2) (선택) best view 1장으로 InstantMesh 실행해 mesh.glb 생성
  3) python improved_instantmesh_pose.py --data_dir ./capture_obj --mask_dir ./masks \
       --instantmesh_mesh ./mesh.glb --out_dir ./outputs
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for _p in (str(REPO_ROOT), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src._camera import RealSenseCamera  # noqa: E402
from Obj_Step1_dump_calib_to_flat import dump_calib_to_flat  # noqa: E402


def capture_depth_burst(cam, n_frames: int, max_wait_ms: int = 1500) -> Optional[np.ndarray]:
    depths = []
    last_ts = None
    start_t = time.monotonic()
    while len(depths) < n_frames:
        if (time.monotonic() - start_t) * 1000.0 > max_wait_ms:
            break
        _, depth, ts_ms = cam.get_latest()
        if depth is None or ts_ms == last_ts:
            time.sleep(0.003)
            continue
        depths.append(depth)
        last_ts = ts_ms
    return np.stack(depths, axis=0) if depths else None


def median_depth_ignore_zero(stack: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if stack is None or stack.shape[0] == 0:
        return None
    f = stack.astype(np.float32)
    f = np.where(f > 0, f, np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(f, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return med.astype(np.uint16)


def load_device_map(intr_dir: Path) -> Optional[Dict[str, int]]:
    map_path = intr_dir / "device_map.json"
    if not map_path.exists():
        return None
    with open(map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return m.get("serial_to_idx")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--intrinsics_dir", required=True)
    p.add_argument("--transforms_json", default="",
                   help="제공 시 cam{i}_K.txt, cam{i}_T_cam_to_world.txt 도 dump")
    p.add_argument("--use_depth_K", action="store_true")

    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)

    p.add_argument("--depth_burst_n", type=int, default=10,
                   help="저장 시 N장 median 으로 depth 노이즈 감소. 1 = 단일 프레임.")
    p.add_argument("--depth_burst_max_wait_ms", type=int, default=1500)
    p.add_argument("--no_align_depth_to_color", action="store_true",
                   help="WARN: improved_instantmesh_pose.py 는 color-aligned depth 를 가정함.")

    p.add_argument("--warmup_seconds", type=float, default=2.0)
    p.add_argument("--auto_save", action="store_true",
                   help="warmup 끝나면 자동으로 한 번 저장하고 종료.")
    p.add_argument("--show", action="store_true",
                   help="프리뷰 창 표시. SPACE=저장, q/ESC=종료.")

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intr_dir = Path(args.intrinsics_dir)

    align_depth = not args.no_align_depth_to_color
    if not align_depth and not args.use_depth_K:
        print("[WARN] --no_align_depth_to_color 켰는데 --use_depth_K 가 꺼져있음. "
              "improved_instantmesh_pose.py 가 color_K 로 미-align depth 를 unproject 해서 ray 가 어긋남.")

    serial_to_idx = load_device_map(intr_dir)
    devs = RealSenseCamera.list_devices()
    if not devs:
        raise RuntimeError("RealSense 카메라를 찾을 수 없습니다.")

    if serial_to_idx is None:
        print("[WARN] device_map.json 없음 → serial 정렬 순서 사용.")
        idx_serial = [(i, s) for i, s in enumerate(sorted(devs.keys()))]
    else:
        idx_serial = []
        for s in devs.keys():
            if s in serial_to_idx:
                idx_serial.append((int(serial_to_idx[s]), s))
            else:
                print(f"[WARN] serial {s} not in device_map.json (skip)")
        idx_serial.sort(key=lambda x: x[0])
    if not idx_serial:
        raise RuntimeError("device_map 필터 후 사용 가능한 카메라 없음.")

    print("[INFO] camera mapping:")
    for ci, s in idx_serial:
        print(f"  cam{ci}: {s}  ({devs.get(s, '?')})")

    cams: Dict[int, RealSenseCamera] = {}
    for ci, s in idx_serial:
        cam = RealSenseCamera(
            serial=s,
            width=args.width, height=args.height, fps=args.fps,
            use_color=True, use_depth=True,
            align_depth_to_color=align_depth,
            warmup_frames=10,
            frame_timeout_ms=2000,
            log_timeouts=True,
            log_errors=True,
        )
        cam.start()
        cams[ci] = cam

    print(f"\n[INFO] Warming up for {args.warmup_seconds:.1f}s...")
    time.sleep(args.warmup_seconds)

    auto = args.auto_save
    if not args.show:
        if not auto:
            print("[INFO] --show off → auto_save 모드로 전환.")
        auto = True

    if args.show:
        print("\nControls:")
        print("  SPACE : save one frame")
        print("  ESC/q : quit\n")

    try:
        while True:
            imgs: Dict[int, dict] = {}
            all_valid = True
            for ci, cam in cams.items():
                color, depth, ts_ms = cam.get_latest()
                imgs[ci] = {"color": color, "depth": depth, "ts_ms": ts_ms}
                if color is None or depth is None:
                    all_valid = False

            if args.show:
                panels = []
                for ci in sorted(cams.keys()):
                    img = imgs[ci]["color"]
                    if img is None:
                        img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                        cv2.putText(img, f"cam{ci} NO FRAME", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        img = img.copy()
                        msg = f"cam{ci} SPACE=save  q=quit"
                        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    panels.append(img)
                h0, w0 = panels[0].shape[:2]
                for i in range(len(panels)):
                    if panels[i].shape[:2] != (h0, w0):
                        panels[i] = cv2.resize(panels[i], (w0, h0))
                while len(panels) < 4:
                    panels.append(np.zeros((h0, w0, 3), dtype=np.uint8))
                top = np.hstack(panels[0:2])
                bot = np.hstack(panels[2:4])
                cv2.imshow("capture_flat_for_im", np.vstack([top, bot]))

            key = cv2.waitKey(1) & 0xFF if args.show else 255
            if args.show and (key == 27 or key == ord("q")):
                print("[INFO] aborted by user.")
                return

            do_save = (key == 32) or (auto and all_valid)
            if not do_save:
                continue
            if not all_valid:
                print("[INFO] cannot save yet: some cameras have no frame.")
                continue

            for ci in sorted(imgs.keys()):
                rgb_path = out_dir / f"cam{ci}_rgb.png"
                depth_path = out_dir / f"cam{ci}_depth.png"
                cv2.imwrite(str(rgb_path), imgs[ci]["color"])

                if args.depth_burst_n > 1:
                    burst = capture_depth_burst(
                        cams[ci], args.depth_burst_n, args.depth_burst_max_wait_ms,
                    )
                    fused = median_depth_ignore_zero(burst)
                    if fused is None:
                        fused = imgs[ci]["depth"]
                    elif burst is not None and burst.shape[0] < args.depth_burst_n:
                        print(f"[WARN] cam{ci}: burst {burst.shape[0]}/{args.depth_burst_n} frames in "
                              f"{args.depth_burst_max_wait_ms}ms")
                else:
                    fused = imgs[ci]["depth"]
                cv2.imwrite(str(depth_path), fused)
                print(f"[cam{ci}] saved {rgb_path.name}, {depth_path.name}")

            if args.transforms_json:
                print("[INFO] dumping calibration to flat layout...")
                dump_calib_to_flat(
                    intrinsics_dir=intr_dir,
                    transforms_json=Path(args.transforms_json),
                    out_dir=out_dir,
                    cam_ids=[ci for ci, _ in idx_serial],
                    use_depth_K=args.use_depth_K,
                )

            print(f"\n[INFO] Done. Flat layout at: {out_dir.resolve()}")
            print("Next: SAM/SAM2 로 cam*_rgb.png → masks/cam{i}_mask.png 만들기, 그 후 InstantMesh.")
            return

    finally:
        for cam in cams.values():
            cam.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
