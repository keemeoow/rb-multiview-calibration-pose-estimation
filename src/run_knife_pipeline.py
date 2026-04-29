#!/usr/bin/env python3
"""[Legacy thin wrapper] knife single-object pipeline.

이 스크립트는 호환용입니다. 실제 동작은 모두 `pipeline_core` +
`configs/objects/knife.json` profile로 위임됩니다. 신규 코드는 직접
`run_pipeline.py`를 사용하세요:

    python3 src/run_pipeline.py \\
      --config src/configs/objects/knife.json \\
      --data_dir src/data_knife --intr_dir src/data_knife/_intrinsics \\
      --frame_id 000004
"""
from __future__ import annotations
import argparse
import datetime
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pipeline_core import load_profile, run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="src/data_knife")
    ap.add_argument("--intr_dir", default="src/data_knife/_intrinsics")
    ap.add_argument("--glb",      default="src/data_knife/reference_knife.glb")
    ap.add_argument("--frame_id", default="000004")
    ap.add_argument("--config",   default=None,
                    help="profile JSON 경로 (default: configs/objects/knife.json)")
    ap.add_argument("--out_root", default=None)
    # 이전 옵션은 무시 (profile JSON 으로 대체)
    ap.add_argument("--hue_ref", type=float, default=None)
    ap.add_argument("--hue_radius", type=float, default=None)
    ap.add_argument("--s_min", type=int, default=None)
    ap.add_argument("--v_min", type=int, default=None)
    args = ap.parse_args()

    cfg_path = args.config or str(SCRIPT_DIR / "configs" / "objects" / "knife.json")
    prof = load_profile(cfg_path)
    if args.glb:
        prof.glb = args.glb
    # 호환용 옵션 override
    if args.hue_ref is not None:
        prof.color_prior.hue_ref = args.hue_ref
    if args.hue_radius is not None:
        prof.color_prior.hue_radius = args.hue_radius
    if args.s_min is not None:
        prof.color_prior.s_min = args.s_min
    if args.v_min is not None:
        prof.color_prior.v_min = args.v_min

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root \
               else Path("src/output") / f"pipeline_knife_{ts}_frame_{args.frame_id}"
    run_pipeline([prof], Path(args.data_dir), Path(args.intr_dir),
                  args.frame_id, out_root)


if __name__ == "__main__":
    main()
