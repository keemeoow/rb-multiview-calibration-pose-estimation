# Step2_capture_multi_cam.py
# 멀티캠 캡처 (Step4까지 할 거면 depth 저장 필수)

"""
[자동 캡처 모드 - SPACE 없이 조건 충족 시 자동 저장]
  --auto_save --stable_frames 5 --cooldown_ms 1000 \

python Step2_capture_multi_cam.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --fps 15 --width 640 --height 480 \
  --min_markers 1 \
  --show
"""

import os
import json
import time
import argparse
from typing import Dict, Optional

import cv2
import numpy as np
import sys

# Allow running this script from inside src/Step0_calibration by adding the
# repository root (parent of `src`) to sys.path so `import src...` works.
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src._camera import RealSenseCamera
from _aruco_cube import CubeConfig, ArucoCubeTarget


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def capture_depth_burst(cam, n_frames: int, max_wait_ms: int = 1500):
    """카메라에서 ts_ms 가 다른 N장의 depth 프레임을 모은다.
    Returns (n, H, W) uint16 ndarray 또는 None."""
    depths = []
    last_ts = None
    start_t = time.monotonic()
    while len(depths) < n_frames:
        elapsed_ms = (time.monotonic() - start_t) * 1000.0
        if elapsed_ms > max_wait_ms:
            break
        _, depth, ts_ms = cam.get_latest()
        if depth is None or ts_ms == last_ts:
            time.sleep(0.003)
            continue
        depths.append(depth)
        last_ts = ts_ms
    if not depths:
        return None
    return np.stack(depths, axis=0)


def median_depth_ignore_zero(stack: Optional[np.ndarray]):
    """Per-pixel median of (N, H, W) uint16 depth stack, ignoring zeros."""
    if stack is None or stack.shape[0] == 0:
        return None
    f = stack.astype(np.float32)
    f = np.where(f > 0, f, np.nan)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(f, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    return med.astype(np.uint16)


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None, None
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    return serial_to_idx, map_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)

    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--min_markers", type=int, default=2)

    parser.add_argument("--auto_save", action="store_true")
    parser.add_argument("--stable_frames", type=int, default=3)
    parser.add_argument("--cooldown_ms", type=int, default=700)

    parser.add_argument("--save_depth", dest="save_depth", action="store_true", default=True,
                        help="Save aligned depth (default: ON).")
    parser.add_argument("--no_save_depth", dest="save_depth", action="store_false",
                        help="Disable depth saving.")
    parser.add_argument("--depth_burst_n", type=int, default=10,
                        help="저장 시점 직후 N장 depth burst median으로 노이즈 감소 (default: 10). "
                             "1로 두면 단일 프레임 저장.")
    parser.add_argument("--depth_burst_max_wait_ms", type=int, default=1500,
                        help="N장을 모으는 데 허용하는 최대 시간(ms).")
    parser.add_argument("--no_align_depth_to_color", action="store_true")
    parser.add_argument("--camera_frame_timeout_ms", type=int, default=2000)
    parser.add_argument("--log_cam_timeouts", action="store_true")
    parser.add_argument("--log_cam_errors", action="store_true")
    parser.add_argument("--log_cam_stats_sec", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    root = ensure_dir(args.root_folder)
    intr_dir = ensure_dir(args.intrinsics_dir)

    devs = RealSenseCamera.list_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found.")

    serial_to_idx, map_path = load_device_map(intr_dir)
    if serial_to_idx is None:
        print("[WARN] device_map.json not found. Falling back to sorted(serial). (권장X)")
        # fallback (not recommended)
        serials = sorted(devs.keys())
        idx_serial_pairs = [(i, s) for i, s in enumerate(serials)]
    else:
        # map 기반으로 현재 연결된 장치만 가져오고 idx로 정렬
        idx_serial_pairs = []
        for serial in devs.keys():
            if serial not in serial_to_idx:
                print(f"[WARN] serial not in device_map.json: {serial} (Step1 다시 실행 권장)")
                continue
            idx_serial_pairs.append((int(serial_to_idx[serial]), serial))
        idx_serial_pairs.sort(key=lambda x: x[0])

    if len(idx_serial_pairs) == 0:
        raise RuntimeError(
            "No usable cameras found after applying device_map.json. "
            "Run Step1_dump_intrinsics.py again or reconnect mapped devices."
        )

    print("[INFO] devices (cam_idx fixed):")
    for idx, s in idx_serial_pairs:
        print(f"  cam{idx}: {s}  ({devs.get(s,'?')})")

    if args.save_depth:
        print(
            "[INFO] depth align_to_color:",
            "OFF" if args.no_align_depth_to_color else "ON",
        )
        if args.depth_burst_n > 1:
            print(f"[INFO] depth temporal median: {args.depth_burst_n} frames per save "
                  f"(max wait {args.depth_burst_max_wait_ms}ms)")
        else:
            print("[INFO] depth temporal median: OFF (single frame)")

    cams = {}
    for ci, serial in idx_serial_pairs:
        cam = RealSenseCamera(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_color=True,
            use_depth=args.save_depth,
            align_depth_to_color=(not args.no_align_depth_to_color),
            warmup_frames=10,
            frame_timeout_ms=args.camera_frame_timeout_ms,
            log_timeouts=args.log_cam_timeouts,
            log_errors=args.log_cam_errors,
        )
        cam.start()
        cams[ci] = cam

    if len(cams) == 0:
        raise RuntimeError("No cameras started successfully.")

    # intrinsics 존재 체크
    for ci in cams:
        intr_path = os.path.join(intr_dir, f"cam{ci}.npz")
        if not os.path.exists(intr_path):
            print(f"[WARN] intrinsics not found: {intr_path} (Step1_dump_intrinsics.py 필요)")

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    meta = {"root_folder": os.path.abspath(root), "captures": []}
    meta_path = os.path.join(root, "meta.json")

    for ci in cams:
        ensure_dir(os.path.join(root, f"cam{ci}"))

    event_id = 0
    stable_cnt = {ci: 0 for ci in cams}
    last_save_t = 0.0
    last_stats_log_t = time.monotonic()
    prev_cam_stats = {ci: cam.get_stats() for ci, cam in cams.items()}

    print("\nControls:")
    print("  SPACE : save once (all cams must satisfy min_markers & stable_frames)")
    print("  ESC/q : quit\n")

    try:
        while True:
            frames: Dict[int, dict] = {}
            all_ok = True

            for ci, cam in cams.items():
                color, depth, ts_ms = cam.get_latest()
                if color is None:
                    all_ok = False
                    continue

                corners, ids = cube.detect(color)
                ok = (ids is not None) and (len(ids) >= args.min_markers)

                if ok:
                    stable_cnt[ci] += 1
                else:
                    stable_cnt[ci] = 0
                    all_ok = False

                frames[ci] = {
                    "color": color,
                    "depth": depth,
                    "ts_ms": ts_ms,
                    "ok": ok,
                    "ids": ([] if ids is None else [int(x) for x in ids]),
                    "corners": corners,
                    "ids_np": ids,
                }

            if args.show:
                panels = []
                for ci in sorted(cams.keys()):
                    if ci in frames and frames[ci]["color"] is not None:
                        img = frames[ci]["color"].copy()
                        ids_np = frames[ci]["ids_np"]
                        corners = frames[ci]["corners"]
                        if ids_np is not None:
                            try:
                                cv2.aruco.drawDetectedMarkers(img, corners, ids_np)
                            except Exception:
                                pass
                        txt = (f"cam{ci} ok={frames[ci]['ok']} "
                               f"stable={stable_cnt[ci]} ids={frames[ci]['ids']}")
                        color_ok = (0, 255, 0) if frames[ci]['ok'] else (0, 0, 255)
                        cv2.putText(img, txt, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(img, txt, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ok, 1)
                    else:
                        img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                        cv2.putText(img, f"cam{ci} NO FRAME", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    panels.append(img)
                if panels:
                    h0, w0 = panels[0].shape[:2]
                    for i in range(len(panels)):
                        if panels[i].shape[:2] != (h0, w0):
                            panels[i] = cv2.resize(panels[i], (w0, h0))
                    # 2x2 그리드: 위 2개, 아래 2개 (cell이 부족하면 검은 placeholder 채움)
                    while len(panels) < 4:
                        panels.append(np.zeros((h0, w0, 3), dtype=np.uint8))
                    top    = np.hstack(panels[0:2])
                    bottom = np.hstack(panels[2:4])
                    combined = np.vstack([top, bottom])
                    cv2.imshow("multi_cam", combined)

            if args.log_cam_stats_sec > 0.0:
                now_mono = time.monotonic()
                dt = now_mono - last_stats_log_t
                if dt >= args.log_cam_stats_sec:
                    print(f"[CAM_STATS] dt={dt:.2f}s")
                    for ci in sorted(cams.keys()):
                        s = cams[ci].get_stats()
                        p = prev_cam_stats.get(ci, {})
                        d_frames = int(s["frames_received"]) - int(p.get("frames_received", 0))
                        d_timeouts = int(s["wait_timeouts"]) - int(p.get("wait_timeouts", 0))
                        d_errors = int(s["loop_errors"]) - int(p.get("loop_errors", 0))
                        d_stale = int(s["stale_frames"]) - int(p.get("stale_frames", 0))
                        fps_est = (d_frames / dt) if dt > 0.0 else 0.0
                        print(
                            f"  cam{ci} fps~{fps_est:.1f} (+frames={d_frames}, +timeouts={d_timeouts}, "
                            f"+errors={d_errors}, +stale={d_stale})"
                        )
                        prev_cam_stats[ci] = s
                    last_stats_log_t = now_mono

            key = cv2.waitKey(1) & 0xFF
            now_ms = time.time() * 1000.0

            manual_trigger = (key == 32)  # SPACE
            quit_trigger = (key == 27) or (key == ord("q"))
            if quit_trigger:
                break

            if args.auto_save:
                if all_ok and all(stable_cnt[ci] >= args.stable_frames for ci in cams) and (now_ms - last_save_t) >= args.cooldown_ms:
                    manual_trigger = True

            if manual_trigger:
                if not (all_ok and all(stable_cnt[ci] >= args.stable_frames for ci in cams)):
                    print("[INFO] save blocked: not all cams stable/ok")
                    continue

                cap_rec = {"event_id": int(event_id), "cams": {}}
                fid = int(event_id)

                for ci in sorted(frames.keys()):
                    fr = frames[ci]
                    rgb_rel = f"cam{ci}/rgb_{fid:05d}.jpg"
                    rgb_abs = os.path.join(root, rgb_rel)
                    cv2.imwrite(rgb_abs, fr["color"])

                    depth_rel = None
                    if args.save_depth and (fr["depth"] is not None):
                        depth_rel = f"cam{ci}/depth_{fid:05d}.png"
                        depth_abs = os.path.join(root, depth_rel)
                        if args.depth_burst_n > 1:
                            burst = capture_depth_burst(
                                cams[ci],
                                args.depth_burst_n,
                                args.depth_burst_max_wait_ms,
                            )
                            depth_to_save = median_depth_ignore_zero(burst)
                            if depth_to_save is None:
                                depth_to_save = fr["depth"]
                            else:
                                got = int(burst.shape[0])
                                if got < args.depth_burst_n:
                                    print(f"[WARN] cam{ci}: burst {got}/{args.depth_burst_n} "
                                          f"frames in {args.depth_burst_max_wait_ms}ms")
                        else:
                            depth_to_save = fr["depth"]
                        cv2.imwrite(depth_abs, depth_to_save)

                    cap_rec["cams"][str(ci)] = {
                        "saved": True,
                        "ts_ms": fr["ts_ms"],
                        "rgb_path": rgb_rel,
                        "depth_path": depth_rel,
                        "ids": fr["ids"],
                    }

                meta["captures"].append(cap_rec)
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                print(f"[SAVE] event_id={event_id} -> meta.json updated ({len(meta['captures'])} captures)")
                event_id += 1
                last_save_t = now_ms

    finally:
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
