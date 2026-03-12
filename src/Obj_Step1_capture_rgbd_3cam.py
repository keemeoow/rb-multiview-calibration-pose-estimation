# capture_rgbd_3cam.py
# 3대 RealSense RGBD 동시 캡처 (640x480, 15fps)
#
# 사용법:
#   python capture_rgbd_3cam.py --save_dir ./data/object_capture
#   python capture_rgbd_3cam.py --save_dir ./data/object_capture --intrinsics_dir ./data/_intrinsics
#
# 조작:
#   SPACE : 현재 프레임 저장
#   s     : 연속 저장 모드 토글 (매 프레임 자동 저장)
#   ESC/q : 종료

import os
import json
import time
import argparse
from typing import Dict

import cv2
import numpy as np

from src3._camera import RealSenseCamera


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_device_map(intr_dir: str):
    map_path = os.path.join(intr_dir, "device_map.json")
    if not os.path.exists(map_path):
        return None
    with open(map_path, "r") as f:
        m = json.load(f)
    return m.get("serial_to_idx", {})


def main():
    parser = argparse.ArgumentParser(description="3-cam RealSense RGBD capture")
    parser.add_argument("--save_dir", required=True, help="저장 폴더")
    parser.add_argument("--intrinsics_dir", default="./data/_intrinsics", help="device_map.json 위치")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    save_dir = ensure_dir(args.save_dir)

    # --- 카메라 탐색 ---
    devs = RealSenseCamera.list_devices()
    if len(devs) == 0:
        raise RuntimeError("RealSense 카메라가 연결되어 있지 않습니다.")
    print(f"[INFO] 감지된 카메라 {len(devs)}대:")
    for s, n in devs.items():
        print(f"  {s}  ({n})")

    # device_map.json 기반 인덱스 매핑
    serial_to_idx = load_device_map(args.intrinsics_dir)
    if serial_to_idx is not None:
        idx_serial = []
        for serial in devs.keys():
            if serial in serial_to_idx:
                idx_serial.append((int(serial_to_idx[serial]), serial))
            else:
                print(f"[WARN] device_map.json에 없는 시리얼: {serial}")
        idx_serial.sort(key=lambda x: x[0])
    else:
        print("[WARN] device_map.json 없음 -> 시리얼 정렬 순서 사용")
        idx_serial = [(i, s) for i, s in enumerate(sorted(devs.keys()))]

    if len(idx_serial) == 0:
        raise RuntimeError("사용 가능한 카메라가 없습니다.")

    # --- 카메라 시작 (RGBD) ---
    cams: Dict[int, RealSenseCamera] = {}
    for ci, serial in idx_serial:
        cam = RealSenseCamera(
            serial=serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            use_color=True,
            use_depth=True,
            align_depth_to_color=True,
            warmup_frames=10,
            frame_timeout_ms=2000,
            log_timeouts=True,
            log_errors=True,
        )
        cam.start()
        cams[ci] = cam
        ensure_dir(os.path.join(save_dir, f"cam{ci}"))

    print(f"\n[INFO] {len(cams)}대 카메라 시작 완료 ({args.width}x{args.height} @ {args.fps}fps, RGBD)")
    print("\n조작:")
    print("  SPACE : 현재 프레임 저장")
    print("  s     : 연속 저장 모드 토글")
    print("  ESC/q : 종료\n")

    frame_idx = 0
    continuous = False

    try:
        while True:
            imgs = {}
            for ci, cam in cams.items():
                color, depth, ts_ms = cam.get_latest()
                imgs[ci] = {"color": color, "depth": depth, "ts_ms": ts_ms}

            # 미리보기
            for ci in sorted(imgs.keys()):
                color = imgs[ci]["color"]
                if color is None:
                    continue
                disp = color.copy()
                label = f"cam{ci} | frame {frame_idx}"
                if continuous:
                    label += " [REC]"
                cv2.putText(disp, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if continuous else (255, 255, 255), 2)
                cv2.imshow(f"cam{ci}", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
            if key == ord("s"):
                continuous = not continuous
                print(f"[INFO] 연속 저장 모드: {'ON' if continuous else 'OFF'}")

            do_save = (key == 32) or continuous  # SPACE or continuous

            if do_save:
                all_valid = all(imgs[ci]["color"] is not None and imgs[ci]["depth"] is not None for ci in imgs)
                if not all_valid:
                    continue

                for ci in sorted(imgs.keys()):
                    rgb_path = os.path.join(save_dir, f"cam{ci}", f"rgb_{frame_idx:06d}.jpg")
                    depth_path = os.path.join(save_dir, f"cam{ci}", f"depth_{frame_idx:06d}.png")
                    cv2.imwrite(rgb_path, imgs[ci]["color"])
                    cv2.imwrite(depth_path, imgs[ci]["depth"])

                print(f"[SAVE] frame {frame_idx:06d} ({len(imgs)}cams)")
                frame_idx += 1

    finally:
        for cam in cams.values():
            cam.stop()
        cv2.destroyAllWindows()
        print(f"\n[INFO] 총 {frame_idx}장 저장 완료 -> {os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()
