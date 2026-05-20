#!/usr/bin/env python3
"""
멀티캠 캘리브레이션 결과(K + T_C0_Ci)를 improved_instantmesh_pose.py 가 요구하는
flat 폴더 형식으로 dump.

사용 예:
  python src/Obj_Step2_dump_calib_to_flat.py \
    --intrinsics_dir src/Step0_calibration/intrinsics \
    --transforms_json src/Step0_calibration/data/cube_session_01/calib_out_cube/transforms/T_C0_Ci_all.json \
    --out_dir ./capture_obj

출력 (out_dir/):
  cam{i}_K.txt              # 3x3 color intrinsics
  cam{i}_T_cam_to_world.txt # 4x4, world = ref cam (T_Cref_Ci -> point_ref = T @ point_i)
  calib_info.json           # 출처/시리얼/depth_scale 메타데이터
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_serial_from_npz(d: np.lib.npyio.NpzFile) -> Optional[str]:
    if "serial" not in d.files:
        return None
    arr = d["serial"]
    val = arr.item() if hasattr(arr, "item") and arr.ndim == 0 else arr
    if isinstance(val, bytes):
        return val.decode()
    return str(val)


def load_transforms(transforms_json: Path) -> Tuple[int, Dict[int, np.ndarray]]:
    with open(transforms_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    ref = int(d["ref_cam_idx"])
    out: Dict[int, np.ndarray] = {}
    for k, v in d["T_Cref_Ci"].items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.size != 16:
            raise ValueError(f"T_Cref_Ci[{k}] is not 16 numbers (got {arr.size})")
        out[int(k)] = arr.reshape(4, 4)
    return ref, out


def load_K_from_npz(npz_path: Path, use_depth_K: bool) -> np.ndarray:
    d = np.load(npz_path)
    key = "depth_K" if use_depth_K else "color_K"
    if key not in d.files:
        raise KeyError(f"{key} not in {npz_path}")
    K = np.asarray(d[key], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"{key} shape={K.shape} (expected 3x3) in {npz_path}")
    return K


def maybe_load_depth_scale(intrinsics_dir: Path, serial: Optional[str]) -> Optional[float]:
    scales_path = intrinsics_dir / "depth_scales.json"
    if not scales_path.exists() or not serial:
        return None
    with open(scales_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("serial_to_depth_scale_m_per_unit", {}).get(serial)


def dump_calib_to_flat(
    intrinsics_dir: Path,
    transforms_json: Path,
    out_dir: Path,
    cam_ids: Optional[List[int]] = None,
    use_depth_K: bool = False,
) -> dict:
    intrinsics_dir = Path(intrinsics_dir)
    transforms_json = Path(transforms_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_cam, T_map = load_transforms(transforms_json)
    if ref_cam not in T_map:
        raise RuntimeError(f"ref_cam_idx={ref_cam} missing from T_Cref_Ci")

    if cam_ids is None:
        cam_ids = sorted(T_map.keys())

    provenance = {
        "intrinsics_dir": str(intrinsics_dir.resolve()),
        "transforms_json": str(transforms_json.resolve()),
        "ref_cam_idx": ref_cam,
        "world_frame": f"cam{ref_cam}",
        "K_source": "depth_K" if use_depth_K else "color_K",
        "cameras": [],
    }

    for ci in cam_ids:
        npz_path = intrinsics_dir / f"cam{ci}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"intrinsics npz missing: {npz_path}")
        if ci not in T_map:
            raise KeyError(f"T_Cref_C{ci} not in {transforms_json.name}")

        npz = np.load(npz_path)
        K = load_K_from_npz(npz_path, use_depth_K=use_depth_K)
        T = T_map[ci]
        serial = _read_serial_from_npz(npz)
        depth_scale = maybe_load_depth_scale(intrinsics_dir, serial)

        K_path = out_dir / f"cam{ci}_K.txt"
        T_path = out_dir / f"cam{ci}_T_cam_to_world.txt"
        np.savetxt(K_path, K, fmt="%.12g")
        np.savetxt(T_path, T, fmt="%.12g")

        provenance["cameras"].append({
            "cam_idx": ci,
            "serial": serial,
            "K_file": K_path.name,
            "T_file": T_path.name,
            "depth_scale_m_per_unit": depth_scale,
        })
        print(f"[cam{ci}] K -> {K_path.name}, T -> {T_path.name}")

    info_path = out_dir / "calib_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"Saved provenance: {info_path}")
    return provenance


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intrinsics_dir", required=True,
                   help="Step1 출력 폴더 (cam{i}.npz 들이 있는 곳)")
    p.add_argument("--transforms_json", required=True,
                   help="Step3 출력 T_C0_Ci_all.json (calib_out_cube/transforms/)")
    p.add_argument("--out_dir", required=True,
                   help="improved_instantmesh_pose.py 가 읽을 flat 폴더")
    p.add_argument("--cam_ids", default="",
                   help="Comma-separated cam idx. 비우면 transforms JSON 의 모든 카메라.")
    p.add_argument("--use_depth_K", action="store_true",
                   help="depth_K 사용 (depth 가 color 에 align 안 된 경우에만).")
    args = p.parse_args()

    cam_ids = [int(x) for x in args.cam_ids.split(",") if x.strip()] or None
    dump_calib_to_flat(
        intrinsics_dir=Path(args.intrinsics_dir),
        transforms_json=Path(args.transforms_json),
        out_dir=Path(args.out_dir),
        cam_ids=cam_ids,
        use_depth_K=args.use_depth_K,
    )


if __name__ == "__main__":
    main()
