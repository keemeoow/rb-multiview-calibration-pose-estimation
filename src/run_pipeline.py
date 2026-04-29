#!/usr/bin/env python3
"""Object-profile driven pipeline CLI.

세 가지 호출 방식:
  A. profile JSON 로 (1개 또는 콤마구분 다수)
     python3 src/run_pipeline.py \
       --config configs/objects/object_001.json,object_002.json \
       --data_dir src/data --intr_dir src/intrinsics --frame_id 000001

  B. 디렉토리 안의 모든 profile 사용
     python3 src/run_pipeline.py \
       --config_dir src/configs/objects \
       --data_dir src/data --intr_dir src/intrinsics --frame_id 000001

  C. config 없이 직접 옵션 (auto-detect)
     python3 src/run_pipeline.py \
       --glb path/to/new.glb \
       --hue_ref 60 --hue_radius 15 --multicolor \
       --init_orientation lying_flat \
       --data_dir src/data_knife --intr_dir src/data_knife/_intrinsics \
       --frame_id 000004
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import List

from pipeline_core import (
    ObjectProfile, ColorPriorConfig, SamConfig, ShapeConfig, PoseConfig,
    load_profile, auto_detect_profile, run_pipeline,
    load_sam_predictor,
)


def parse_profiles(args) -> List[ObjectProfile]:
    profiles: List[ObjectProfile] = []
    if args.config:
        for spec in args.config.split(","):
            spec = spec.strip()
            if spec:
                profiles.append(load_profile(spec))
    if args.config_dir:
        cdir = Path(args.config_dir)
        for f in sorted(cdir.glob("*.json")):
            profiles.append(load_profile(f))
    if not profiles and args.glb:
        # CLI flags → ad-hoc profile (auto-detect 보완)
        prof = auto_detect_profile(
            name=args.name or Path(args.glb).stem,
            glb_path=args.glb,
            hue_seed=args.hue_ref,
        )
        # 사용자 명시 옵션으로 override
        if args.hue_ref is not None:
            prof.color_prior.enabled = True
            prof.color_prior.hue_ref = args.hue_ref
            if args.hue_radius is not None:
                prof.color_prior.hue_radius = args.hue_radius
            if args.s_min is not None:
                prof.color_prior.s_min = args.s_min
            if args.v_min is not None:
                prof.color_prior.v_min = args.v_min
        if args.multicolor:
            prof.multicolor = True
            prof.sam.post_color_intersect = False
            prof.sam.auto_refine = "off"
            prof.sam.bbox_combine = "intersect"
            prof.sam.reliability_threshold = min(
                prof.sam.reliability_threshold, 0.10)
            if prof.sam.bbox_pad_ratio < 0.20:
                prof.sam.bbox_pad_ratio = 0.30
        if args.bbox_pad is not None:
            prof.sam.bbox_pad_ratio = args.bbox_pad
        if args.init_orientation:
            prof.shape.init_orientation = args.init_orientation
        if args.symmetry:
            prof.shape.symmetry = args.symmetry
        if args.label:
            prof.label = args.label
        if args.overlay_color:
            prof.overlay_color_bgr = tuple(
                int(x) for x in args.overlay_color.split(","))
        profiles.append(prof)
    return profiles


def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--intr_dir", required=True)
    ap.add_argument("--frame_id", required=True,
                    help="단일 (예: 000001), 콤마구분 (000000,000001), "
                         "범위 (0-5), 또는 'all'")
    ap.add_argument("--capture_subdir", default="object_capture",
                    help="data_dir 안의 캡처 폴더명 (기본 object_capture). "
                         "예: peg_capture, hole_capture")
    ap.add_argument("--out_root", default=None,
                    help="default: src/output/pipeline_<timestamp>")
    # profile sources (택 1 이상)
    ap.add_argument("--config", default=None,
                    help="콤마로 구분된 profile JSON 경로(들)")
    ap.add_argument("--config_dir", default=None,
                    help="profile JSON 들이 들어있는 폴더")
    # ad-hoc CLI options (config 없을 때)
    ap.add_argument("--glb", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--overlay_color", default=None,
                    help='"B,G,R" 0-255')
    ap.add_argument("--hue_ref", type=float, default=None)
    ap.add_argument("--hue_radius", type=float, default=None)
    ap.add_argument("--s_min", type=int, default=None)
    ap.add_argument("--v_min", type=int, default=None)
    ap.add_argument("--multicolor", action="store_true")
    ap.add_argument("--bbox_pad", type=float, default=None)
    ap.add_argument("--init_orientation",
                    choices=["auto", "upright", "lying_flat"], default=None)
    ap.add_argument("--symmetry", choices=["none", "yaw"], default=None)

    args = ap.parse_args()

    profiles = parse_profiles(args)
    if not profiles:
        ap.error("profile 이 하나도 없습니다. --config, --config_dir, "
                 "또는 --glb [+ 옵션] 중 하나는 지정하세요.")

    frame_ids = expand_frame_ids(args.frame_id, Path(args.data_dir),
                                  args.capture_subdir)
    print(f"profiles: {[p.name for p in profiles]}")
    print(f"frames: {frame_ids}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_root:
        base_root = Path(args.out_root)
    else:
        base_root = Path("src") / "output" / f"pipeline_{ts}"

    # SAM predictor 한 번만 로드해서 재사용 (frame 마다 가중치 재로드 방지)
    predictor = None
    if len(frame_ids) > 1:
        print("Pre-loading MobileSAM (will be reused across frames)...")
        predictor = load_sam_predictor()

    for fid in frame_ids:
        if len(frame_ids) > 1:
            out_root = base_root / f"frame_{fid}"
            print(f"\n{'='*64}\n FRAME {fid}\n{'='*64}")
        else:
            out_root = base_root if args.out_root \
                       else Path("src") / "output" / f"pipeline_{ts}_frame_{fid}"
        try:
            run_pipeline(
                profiles=profiles,
                data_dir=Path(args.data_dir),
                intr_dir=Path(args.intr_dir),
                frame_id=fid,
                out_root=out_root,
                predictor=predictor,
                capture_subdir=args.capture_subdir,
            )
        except FileNotFoundError as e:
            print(f"  [skip] frame {fid}: {e}")


def expand_frame_ids(spec: str, data_dir: Path,
                      capture_subdir: str = "object_capture") -> list:
    """frame_id 표현식을 리스트로 확장.
    - "000003"          → ["000003"]
    - "000000,000002"   → ["000000", "000002"]
    - "0-3"             → ["000000","000001","000002","000003"]
    - "all"             → data_dir/<capture_subdir>/cam0/rgb_*.jpg 전체
    """
    spec = spec.strip()
    if spec.lower() == "all":
        cam0 = data_dir / capture_subdir / "cam0"
        ids = sorted(p.stem.replace("rgb_", "") for p in cam0.glob("rgb_*.jpg"))
        return ids
    if "," in spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    if "-" in spec and not spec.startswith("-"):
        a, b = spec.split("-", 1)
        try:
            i0, i1 = int(a), int(b)
            return [f"{i:06d}" for i in range(i0, i1 + 1)]
        except ValueError:
            return [spec]
    return [spec]


if __name__ == "__main__":
    main()
