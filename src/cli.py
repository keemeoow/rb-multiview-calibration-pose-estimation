# File: cli.py
# Summary:
#   CLI entry. ★ --target_glb_paths (또는 --target_glb_paths_file) 필수.
#   Builds PipelineConfig and calls run_pipeline().
#   Default = CPU fallback. --use_mobilesam 으로 MobileSAM segmentation 활성화.
# Run:
#   python cli.py --data_dir data --intrinsics_dir intrinsics --frame_id 0 \
#                 --target_glb_paths data/object_002.glb data/object_004.glb \
#                 --output_dir output/pose_pipeline --use_mobilesam \
#                 --mobilesam_ckpt weights/mobile_sam.pt
# ---------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline_core import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Selective targeted multiview RGB-D 6D pose pipeline.")
    p.add_argument("--data_dir", type=Path, required=True)
    p.add_argument("--intrinsics_dir", type=Path, required=True)
    p.add_argument("--capture_subdir", type=str, default="object_capture")
    p.add_argument("--frame_id", type=int, default=0)
    p.add_argument("--num_cams", type=int, default=3)
    p.add_argument("--output_dir", type=Path, default=Path("output/pose_pipeline"))
    p.add_argument("--depth_scale", type=float, default=1e-3)

    # ★ TARGET SUBSET
    p.add_argument("--target_glb_paths", nargs="+", type=Path, default=None)
    p.add_argument("--target_glb_paths_file", type=Path, default=None,
                   help="한 줄에 하나씩 GLB 경로가 적힌 텍스트 파일.")

    # backends
    p.add_argument("--use_mobilesam", action="store_true",
                   help="MobileSAM segmentation 사용 (CPU, ~30-60s/cam)")
    p.add_argument("--mobilesam_ckpt", type=Path, default=Path("weights/mobile_sam.pt"))
    p.add_argument("--use_cnos", action="store_true")
    p.add_argument("--use_cnos_lite", action="store_true",
                   help="DINOv2 template matching 라벨 validation (CPU)")
    p.add_argument("--use_megapose", action="store_true")
    p.add_argument("--use_foundationpose", action="store_true")
    p.add_argument("--cnos_cfg", type=Path, default=None)
    p.add_argument("--cnos_ckpt", type=Path, default=None)
    p.add_argument("--dinov2_ckpt", type=Path,
                   default=Path("weights/dinov2_vits14_pretrain.pth"))
    p.add_argument("--cnos_lite_n_views", type=int, default=36)
    p.add_argument("--cnos_lite_min_sim", type=float, default=0.55)
    p.add_argument("--cnos_lite_no_label_swap", action="store_true",
                   help="라벨 mismatch 시 swap 대신 drop")
    p.add_argument("--megapose_dir", type=Path, default=None)
    p.add_argument("--foundationpose_dir", type=Path, default=None)

    # MobileSAM tuning
    p.add_argument("--sam_points_per_side", type=int, default=16)
    p.add_argument("--sam_pred_iou_thresh", type=float, default=0.86)
    p.add_argument("--sam_stability_score_thresh", type=float, default=0.9)
    p.add_argument("--sam_min_mask_region_area", type=int, default=400)
    p.add_argument("--sam_match_color_w", type=float, default=1.0)
    p.add_argument("--sam_match_shape_w", type=float, default=1.0,
                   help="shape cost = sum |sorted_extent_diff_mm|; weight 1.0 = mm scale")
    p.add_argument("--sam_match_max_cost", type=float, default=60.0)
    p.add_argument("--sam_topk_per_label", type=int, default=3)
    p.add_argument("--sam_max_extent_ratio", type=float, default=1.5,
                   help="3D extent > GLB max × ratio → 거부 (배경 큰 mask 차단)")
    p.add_argument("--sam_no_keep_largest_cc", action="store_true")

    # color priors
    p.add_argument("--object_colors_json", type=Path, default=None)
    p.add_argument("--auto_colors", action="store_true")
    p.add_argument("--no_save_discovered_colors", action="store_true")

    # color-fallback segmenter tuning
    p.add_argument("--seg_min_depth_m", type=float, default=0.15)
    p.add_argument("--seg_max_depth_m", type=float, default=0.85)
    p.add_argument("--seg_min_sat", type=int, default=60)
    p.add_argument("--seg_min_area_px", type=int, default=600)
    p.add_argument("--seg_hue_cost_thresh", type=float, default=25.0)
    p.add_argument("--seg_topk_per_label", type=int, default=3)

    # association
    p.add_argument("--centroid_thresh_m", type=float, default=0.06)
    p.add_argument("--reproj_thresh_px", type=float, default=60.0)
    p.add_argument("--allow_single_cam", action="store_true",
                   help="multi-view consensus 없이 single-cam track 도 허용 (정확도 ↓)")

    # refinement
    p.add_argument("--no_icp", action="store_true")
    p.add_argument("--no_silhouette", action="store_true")
    p.add_argument("--silhouette_iters", type=int, default=80)
    p.add_argument("--silhouette_max_faces", type=int, default=5000)
    p.add_argument("--chamfer_weight", type=float, default=1.5,
                   help="depth chamfer loss weight (0 = off, 1.5 = 강함)")
    p.add_argument("--per_cam_iou_min", type=float, default=0.25)
    p.add_argument("--iterative_outlier_passes", type=int, default=2)

    # extrinsic refinement
    p.add_argument("--refine_extrinsics", action="store_true",
                   help="depth ICP 로 cam_i 외부파라미터 한번 보정 후 저장")
    p.add_argument("--no_use_refined_extr", action="store_true",
                   help="refined/T_C0_Ci.npy 가 있어도 무시 (원본 사용)")

    # filter / export
    p.add_argument("--min_confidence", type=float, default=0.0)
    p.add_argument("--export_frame", choices=["opencv", "isaac"], default="opencv")
    p.add_argument("--no_overlay", action="store_true")

    return p.parse_args()


def _resolve_targets(a: argparse.Namespace) -> list:
    paths: list = []
    if a.target_glb_paths:
        paths.extend(a.target_glb_paths)
    if a.target_glb_paths_file:
        for line in Path(a.target_glb_paths_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    paths = [Path(p) for p in paths]
    if not paths:
        print("ERROR: --target_glb_paths (or --target_glb_paths_file) is required.",
              file=sys.stderr)
        sys.exit(2)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("ERROR: target GLB(s) not found:\n  - " + "\n  - ".join(missing),
              file=sys.stderr)
        sys.exit(2)
    return paths


def main() -> None:
    a = parse_args()
    target_paths = _resolve_targets(a)
    cfg = PipelineConfig(
        data_dir=a.data_dir, intrinsics_dir=a.intrinsics_dir,
        target_glb_paths=target_paths,
        capture_subdir=a.capture_subdir, frame_id=a.frame_id,
        num_cams=a.num_cams, output_dir=a.output_dir, depth_scale=a.depth_scale,

        use_mobilesam=a.use_mobilesam, mobilesam_ckpt=a.mobilesam_ckpt,
        use_cnos=a.use_cnos, use_megapose=a.use_megapose,
        use_foundationpose=a.use_foundationpose,
        use_cnos_lite=a.use_cnos_lite,
        cnos_cfg=a.cnos_cfg, cnos_ckpt=a.cnos_ckpt,
        dinov2_ckpt=a.dinov2_ckpt,
        cnos_lite_n_views=a.cnos_lite_n_views,
        cnos_lite_min_sim=a.cnos_lite_min_sim,
        cnos_lite_label_swap=not a.cnos_lite_no_label_swap,
        megapose_dir=a.megapose_dir, foundationpose_dir=a.foundationpose_dir,

        sam_points_per_side=a.sam_points_per_side,
        sam_pred_iou_thresh=a.sam_pred_iou_thresh,
        sam_stability_score_thresh=a.sam_stability_score_thresh,
        sam_min_mask_region_area=a.sam_min_mask_region_area,
        sam_match_color_w=a.sam_match_color_w,
        sam_match_shape_w=a.sam_match_shape_w,
        sam_match_max_cost=a.sam_match_max_cost,
        sam_topk_per_label=a.sam_topk_per_label,
        sam_max_extent_ratio=a.sam_max_extent_ratio,
        sam_keep_largest_cc=not a.sam_no_keep_largest_cc,

        object_colors_json=a.object_colors_json,
        auto_colors=a.auto_colors,
        save_discovered_colors=not a.no_save_discovered_colors,

        seg_min_depth_m=a.seg_min_depth_m, seg_max_depth_m=a.seg_max_depth_m,
        seg_min_sat=a.seg_min_sat, seg_min_area_px=a.seg_min_area_px,
        seg_hue_cost_thresh=a.seg_hue_cost_thresh,
        seg_topk_per_label=a.seg_topk_per_label,

        centroid_match_thresh_m=a.centroid_thresh_m,
        reproj_match_thresh_px=a.reproj_thresh_px,
        require_multiview_consensus=not a.allow_single_cam,

        refine_icp=not a.no_icp,
        refine_silhouette=not a.no_silhouette,
        silhouette_iters=a.silhouette_iters,
        silhouette_max_faces=a.silhouette_max_faces,
        chamfer_weight=a.chamfer_weight,
        per_cam_iou_min=a.per_cam_iou_min,
        iterative_outlier_passes=a.iterative_outlier_passes,
        refine_extrinsics=a.refine_extrinsics,
        use_refined_extr=not a.no_use_refined_extr,

        min_confidence=a.min_confidence,
        export_frame=a.export_frame,
        save_overlays=not a.no_overlay,
    )
    summary = run_pipeline(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
