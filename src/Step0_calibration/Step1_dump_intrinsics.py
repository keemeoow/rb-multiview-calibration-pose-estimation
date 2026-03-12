# Step1_dump_intrinsics.py

"""
python Step1_dump_intrinsics.py

[Result]
- intrinsics/device_map.json (serial → cam_idx 고정 매핑)
- intrinsics/cam0.npz, cam1.npz, cam2.npz (각 카메라 K,D 등)

"""

import os
import json
import time
import numpy as np
import pyrealsense2 as rs

def _intr_to_KD(intr: rs.intrinsics):
    fx, fy = float(intr.fx), float(intr.fy)
    cx, cy = float(intr.ppx), float(intr.ppy)
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)  # [k1,k2,p1,p2,k3]
    return K, D

def _list_serials():
    ctx = rs.context()
    devs = ctx.query_devices()
    serials = [d.get_info(rs.camera_info.serial_number) for d in devs]
    return serials

def _load_existing_map(map_path):
    if not os.path.exists(map_path):
        return None
    with open(map_path, "r") as f:
        return json.load(f)

def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main(
    out_dir="intrinsics",
    color_w=640, color_h=480, color_fps=15,
    depth_w=640, depth_h=480, depth_fps=15,
    use_existing_map=True
):
    os.makedirs(out_dir, exist_ok=True)
    by_serial_dir = os.path.join(out_dir, "intrinsics_by_serial")
    os.makedirs(by_serial_dir, exist_ok=True)

    map_path = os.path.join(out_dir, "device_map.json")
    scales_path = os.path.join(out_dir, "depth_scales.json")

    # 1) 카메라 검색
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("[ERROR] No RealSense devices found.")
        return

    # 2) 검색된 카메라 serials 정보 수집
    detected = []
    for d in devs:
        serial = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "Unknown"
        detected.append({"serial": serial, "name": name})

    detected_serials = [x["serial"] for x in detected]
    print("[INFO] detected devices:")
    for x in detected:
        print(f"  serial={x['serial']}  name={x['name']}")

    # 3) device_map.json
    existing_map = _load_existing_map(map_path) if use_existing_map else None

    if existing_map is not None:
        # validate: keep existing idx for serials that exist; append new ones
        serial_to_idx = dict(existing_map.get("serial_to_idx", {}))
        next_idx = 0 if len(serial_to_idx) == 0 else (max(serial_to_idx.values()) + 1)

        for s in detected_serials:
            if s not in serial_to_idx:
                serial_to_idx[s] = next_idx
                next_idx += 1

        # also remove serials no longer connected (optional; here we keep them but mark disconnected)
        map_obj = {
            "created_at_epoch": existing_map.get("created_at_epoch", time.time()),
            "updated_at_epoch": time.time(),
            "serial_to_idx": serial_to_idx,
            "detected_now": detected
        }
        print("[INFO] Using existing device_map.json (preserving indices).")
    else:
        # fresh mapping: stable order by serial string
        detected_serials_sorted = sorted(detected_serials)
        serial_to_idx = {s: i for i, s in enumerate(detected_serials_sorted)}
        map_obj = {
            "created_at_epoch": time.time(),
            "updated_at_epoch": time.time(),
            "serial_to_idx": serial_to_idx,
            "detected_now": detected
        }
        print("[INFO] Creating new device_map.json (sorted by serial).")

    _save_json(map_path, map_obj)
    print(f"[SAVE] {map_path}")

    # 3) acquire intrinsics per device
    depth_scales = {
        "updated_at_epoch": time.time(),
        "serial_to_depth_scale_m_per_unit": {}
    }

    # build idx->serial list for connected devices
    idx_serial_pairs = []
    for s in detected_serials:
        idx_serial_pairs.append((serial_to_idx[s], s))
    idx_serial_pairs.sort(key=lambda x: x[0])

    print("[INFO] cam index assignment for currently connected devices:")
    for idx, s in idx_serial_pairs:
        print(f"  cam{idx}: serial={s}")

    for cam_idx, serial in idx_serial_pairs:
        # query depth scale directly from device handle
        dev = None
        for d in devs:
            if d.get_info(rs.camera_info.serial_number) == serial:
                dev = d
                break
        if dev is None:
            print(f"[WARN] device handle not found for serial={serial} (skip)")
            continue

        try:
            ds = float(dev.first_depth_sensor().get_depth_scale())
            depth_scales["serial_to_depth_scale_m_per_unit"][serial] = ds
        except Exception as e:
            print(f"[WARN] depth_scale read failed for serial={serial}: {e}")
            ds = None

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)

        # enable both streams so we can read both intrinsics
        config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
        config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)

        try:
            profile = pipeline.start(config)

            # color intrinsics
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_intr = color_stream.get_intrinsics()
            Kc, Dc = _intr_to_KD(color_intr)

            # depth intrinsics
            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_intr = depth_stream.get_intrinsics()
            Kd, Dd = _intr_to_KD(depth_intr)

            # depth->color extrinsics (optional but often useful)
            try:
                extr = depth_stream.get_extrinsics_to(color_stream)
                R_dc = np.array(extr.rotation, dtype=np.float64).reshape(3, 3)
                t_dc = np.array(extr.translation, dtype=np.float64).reshape(3, 1)
            except Exception:
                R_dc = np.eye(3, dtype=np.float64)
                t_dc = np.zeros((3, 1), dtype=np.float64)

            # save cam-indexed npz
            npz_path = os.path.join(out_dir, f"cam{cam_idx}.npz")
            np.savez(
                npz_path,
                serial=serial,
                color_K=Kc, color_D=Dc,
                depth_K=Kd, depth_D=Dd,
                depth_scale_m_per_unit=(ds if ds is not None else np.nan),
                color_w=color_w, color_h=color_h, color_fps=color_fps,
                depth_w=depth_w, depth_h=depth_h, depth_fps=depth_fps,
                R_depth_to_color=R_dc,
                t_depth_to_color=t_dc
            )
            print(f"[SAVE] {npz_path}")

            # also save by serial
            serial_npz = os.path.join(by_serial_dir, f"serial_{serial}.npz")
            np.savez(
                serial_npz,
                serial=serial,
                cam_idx=cam_idx,
                color_K=Kc, color_D=Dc,
                depth_K=Kd, depth_D=Dd,
                depth_scale_m_per_unit=(ds if ds is not None else np.nan),
                color_w=color_w, color_h=color_h, color_fps=color_fps,
                depth_w=depth_w, depth_h=depth_h, depth_fps=depth_fps,
                R_depth_to_color=R_dc,
                t_depth_to_color=t_dc
            )
            print(f"[SAVE] {serial_npz}")

            # print quick summary
            print(f"[INFO] cam{cam_idx} serial={serial}")
            print("       color K=\n", Kc)
            print("       color D=", Dc.reshape(-1))
            print("       depth K=\n", Kd)
            print("       depth D=", Dd.reshape(-1))
            print("       depth_scale(m/unit)=", ds)
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    _save_json(scales_path, depth_scales)
    print(f"[SAVE] {scales_path}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
