# File: pipeline_core.py
# Summary:
#   Targeted known-object pose pipeline. Loads ONLY target_glb_paths (subset).
#   Backends:
#     - MobileSAMSegmenter (CPU; preferred)  → SAM masks + GLB matching
#     - CNOSSegmenter (skeleton)
#     - FallbackSegmenter (color-prior; CPU baseline)
#     - MegaPose / FoundationPose (skeletons)
#     - fallback initial pose (PCA + extent + sign-flip)
#     - refinement: open3d ICP + multi-view silhouette IoU (gravity-aware)
#   ★ All stages restricted to the target subset by construction.
# Run:
#   python cli.py --data_dir data --intrinsics_dir intrinsics --frame_id 0 \
#                 --target_glb_paths data/object_002.glb data/object_004.glb \
#                 --output_dir output/pose_pipeline --use_mobilesam \
#                 --mobilesam_ckpt weights/mobile_sam.pt
# ---------------------------------------------------------------

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from config_types_utils import (
    CameraIntrinsics, CameraFrame, CanonicalModel,
    CameraDetection, ObjectTrack, PoseEstimate,
    load_intrinsics, load_extrinsics, load_capture_frame,
    load_target_library,
    DEMO_COLOR_PRIORS_HSV, load_color_priors_json, save_color_priors_json,
    hue_distance, hsv_to_bgr,
    transform_points, invert_se3,
    backproject_masked, project_points,
    rasterize_mesh_mask, render_mesh_silhouette, make_comparison_grid,
    export_posed_glb, export_results_json, export_results_npz,
    export_results_text, load_object_descriptions,
    HAVE_O3D, HAVE_SCIPY,
)


# ===================================================================
#                          CONFIG
# ===================================================================

@dataclass
class PipelineConfig:
    data_dir: Path
    intrinsics_dir: Path
    target_glb_paths: List[Path]                # ★ subset; empty → error
    capture_subdir: str = "object_capture"
    frame_id: int = 0
    num_cams: int = 3
    output_dir: Path = Path("output/pose_pipeline")
    depth_scale: float = 1e-3

    # backends
    use_mobilesam: bool = False
    use_cnos: bool = False
    use_cnos_lite: bool = False                  # ★ DINOv2 template matching validator
    use_megapose: bool = False
    use_foundationpose: bool = False
    mobilesam_ckpt: Optional[Path] = None
    cnos_cfg: Optional[Path] = None
    cnos_ckpt: Optional[Path] = None
    dinov2_ckpt: Optional[Path] = None
    cnos_lite_n_views: int = 36                  # 라벨당 template 개수
    cnos_lite_min_sim: float = 0.55              # detection 채택 최소 similarity
    cnos_lite_label_swap: bool = True            # 더 나은 target 발견 시 라벨 교체
    megapose_dir: Optional[Path] = None
    foundationpose_dir: Optional[Path] = None

    # MobileSAM tuning
    sam_points_per_side: int = 16
    sam_pred_iou_thresh: float = 0.86
    sam_stability_score_thresh: float = 0.9
    sam_min_mask_region_area: int = 400
    sam_match_color_w: float = 1.0
    sam_match_shape_w: float = 1.0
    sam_match_max_cost: float = 60.0
    sam_topk_per_label: int = 3
    sam_max_extent_ratio: float = 1.5    # 3D extent 가 GLB 최대 extent × 이 값 초과 → 거부
    sam_keep_largest_cc: bool = True     # mask 안 가장 큰 connected component 만
    sam_cross_view_prompt: bool = True   # 한 cam 의 centroid 를 다른 cam 에 reproject → SAM point prompt

    # color priors (target subset 만 사용)
    object_colors_json: Optional[Path] = None
    auto_colors: bool = False
    save_discovered_colors: bool = True

    # fallback (color) segmenter
    seg_min_depth_m: float = 0.15
    seg_max_depth_m: float = 0.85
    seg_min_sat: int = 60
    seg_min_area_px: int = 600
    seg_hue_cost_thresh: float = 25.0
    seg_topk_per_label: int = 3

    # association
    centroid_match_thresh_m: float = 0.06
    reproj_match_thresh_px: float = 60.0
    require_multiview_consensus: bool = True   # ★ ≥2 cam 합의 없으면 track 폐기

    # refinement
    refine_icp: bool = True
    refine_silhouette: bool = True
    silhouette_iters: int = 80
    silhouette_max_faces: int = 5000
    chamfer_weight: float = 1.5
    per_cam_iou_min: float = 0.25     # ★ 이 IoU 미만인 cam detection 은 track에서 제거
    iterative_outlier_passes: int = 2 # 제거 → 재refine → 제거 ... 반복 횟수

    # extrinsic refinement
    refine_extrinsics: bool = False              # depth ICP 로 cam_i 외부파라미터 보정
    refined_extr_subdir: str = "refined"         # 보정 결과 저장 위치 (calib_out_cube/refined/)
    use_refined_extr: bool = True                # 존재하면 자동 로드

    # filter / export
    min_confidence: float = 0.0
    export_frame: str = "opencv"
    save_overlays: bool = True


# ===================================================================
#       SEGMENTATION (MobileSAM | CNOS | color fallback) — TARGET ONLY
# ===================================================================

def _mask_features(mask: np.ndarray, frame: CameraFrame,
                   max_depth_m: float = 1.5
                   ) -> Optional[Dict]:
    """detection mask → (mean HSV, 3D centroid base, sorted 3D extents) 추출."""
    if mask.sum() < 50:
        return None
    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2HSV)
    h_in = hsv[:, :, 0][mask]
    s_in = hsv[:, :, 1][mask]
    v_in = hsv[:, :, 2][mask]
    # circular hue mean
    rad = h_in.astype(np.float64) * (np.pi / 90.0)
    h_mean = float(np.degrees(np.arctan2(np.sin(rad).mean(),
                                         np.cos(rad).mean())) / 2.0)
    if h_mean < 0:
        h_mean += 180.0
    s_mean = float(s_in.mean()); v_mean = float(v_in.mean())

    P_cam = backproject_masked(frame.depth, mask, frame.intr, max_depth=max_depth_m)
    if P_cam.shape[0] < 30:
        return None
    cent_cam = P_cam.mean(axis=0)
    cent_base = transform_points(frame.T_base_cam, cent_cam[None])[0]

    Pc = P_cam - cent_cam
    cov = Pc.T @ Pc / max(Pc.shape[0] - 1, 1)
    w, V = np.linalg.eigh(cov)
    R = V[:, np.argsort(w)[::-1]]
    proj = Pc @ R
    ext = np.sort(proj.max(0) - proj.min(0))[::-1]   # 내림차순

    return {
        "hsv": (int(h_mean), int(s_mean), int(v_mean)),
        "cent_cam": cent_cam, "cent_base": cent_base,
        "extents_sorted": ext,
        "n_pts": int(P_cam.shape[0]),
    }


def _match_cost(mask_feat: Dict, target_hsv: Tuple[int, int, int],
                target_extents_sorted: np.ndarray,
                w_color: float, w_shape: float) -> float:
    """color (hue 거리) + shape (sorted extent L1) 합산 비용."""
    hd = float(hue_distance(np.array([mask_feat["hsv"][0]], dtype=np.uint8),
                            int(target_hsv[0]))[0])
    sd_mm = float(np.abs(mask_feat["extents_sorted"]
                         - target_extents_sorted).sum() * 1000.0)
    return w_color * hd + w_shape * sd_mm


class MobileSAMSegmenter:
    """MobileSAM AutomaticMaskGenerator → 모든 mask 후보 →
       타겟 GLB 별로 (color + shape) 비용 최소 mask 매칭 (top-K).
       라벨은 반드시 target subset 의 키로만 출력."""

    def __init__(self, ckpt: Path, target_models: Dict[str, CanonicalModel],
                 color_priors: Dict[str, Tuple[int, int, int]],
                 points_per_side: int = 16,
                 pred_iou_thresh: float = 0.86,
                 stability_score_thresh: float = 0.9,
                 min_mask_region_area: int = 400,
                 match_color_w: float = 1.0,
                 match_shape_w: float = 1.0,
                 match_max_cost: float = 60.0,
                 topk_per_label: int = 3,
                 max_extent_ratio: float = 1.5,
                 keep_largest_cc: bool = True,
                 cross_view_prompt: bool = True):
        self.targets = target_models
        self.priors = {k: v for k, v in color_priors.items() if k in target_models}
        if not self.priors:
            raise ValueError("MobileSAM: no color priors for any target — "
                             "JSON 또는 --auto_colors 사용.")
        self.target_extents_sorted = {
            n: np.sort(np.asarray(target_models[n].extents))[::-1]
            for n in target_models
        }
        # 거부 임계값: GLB 최장 extent × ratio (m)
        self.target_max_extent_m = {
            n: float(np.max(target_models[n].extents)) for n in target_models
        }
        self.match_color_w = match_color_w
        self.match_shape_w = match_shape_w
        self.match_max_cost = match_max_cost
        self.topk = topk_per_label
        self.max_extent_ratio = max_extent_ratio
        self.keep_largest_cc = keep_largest_cc
        self.cross_view_prompt = cross_view_prompt
        self._ok = False
        self._gen = None
        self._predictor = None
        self._sam = None
        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam = sam_model_registry["vit_t"](checkpoint=str(ckpt))
            sam.to("cpu").eval()
            self._sam = sam
            self._gen = SamAutomaticMaskGenerator(
                sam, points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=min_mask_region_area,
            )
            self._predictor = SamPredictor(sam)
            self._ok = True
        except Exception as e:
            print(f"[MobileSAM] init failed → fallback. ({e})")

    def available(self) -> bool:
        return self._ok

    def segment(self, frame: CameraFrame) -> List[CameraDetection]:
        if not self._ok:
            return []
        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)
        sam_masks = self._gen.generate(rgb)
        H, W = frame.depth.shape

        # 전체 mask 의 features 미리 계산
        candidates: List[Tuple[np.ndarray, Dict]] = []
        for m in sam_masks:
            seg = m["segmentation"].astype(bool)
            depth_valid = seg & (frame.depth > 0.05) & (frame.depth < 1.5)
            if depth_valid.sum() < 50:
                continue
            # 가장 큰 connected component 만 살리기 (배경 분리된 small 부분 제거)
            if self.keep_largest_cc:
                n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(
                    depth_valid.astype(np.uint8), 8)
                if n_cc <= 1:
                    continue
                i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
                depth_valid = (lbl == i)
                if depth_valid.sum() < 50:
                    continue
            f = _mask_features(depth_valid, frame)
            if f is None:
                continue
            candidates.append((depth_valid, f))

        # target 라벨별로 비용 행렬 구성 → top-K (3D extent 검증 포함)
        dets: List[CameraDetection] = []
        for name, model in self.targets.items():
            target_hsv = self.priors.get(name)
            if target_hsv is None:
                continue
            target_ext = self.target_extents_sorted[name]
            max_allowed = self.target_max_extent_m[name] * self.max_extent_ratio
            scored: List[Tuple[float, np.ndarray, Dict]] = []
            for mk, ft in candidates:
                # ★ 3D extent 가 GLB 최장 × ratio 초과면 거부 (배경 mask 차단)
                if float(ft["extents_sorted"][0]) > max_allowed:
                    continue
                cost = _match_cost(ft, target_hsv, target_ext,
                                   self.match_color_w, self.match_shape_w)
                if cost <= self.match_max_cost:
                    scored.append((cost, mk, ft))
            scored.sort(key=lambda x: x[0])
            for k, (cost, mk, ft) in enumerate(scored[: self.topk]):
                ys, xs = np.where(mk)
                if len(ys) == 0:
                    continue
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                dets.append(CameraDetection(
                    cam_id=frame.cam_id, label=name,
                    score=float(1.0 / (1.0 + cost)),
                    mask=mk, bbox_xyxy=bbox,
                    centroid_cam=ft["cent_cam"],
                    centroid_base=ft["cent_base"],
                ))
        return dets

    def segment_at_point(self, frame: CameraFrame, label: str,
                         pixel_uv: Tuple[int, int]
                         ) -> Optional[CameraDetection]:
        """SAM point-prompt: pixel_uv 위치 → 1개 mask 회수.
           cross-view validation 용 (다른 cam 이 본 centroid 를 reproject 한 곳)."""
        if not self._ok or self._predictor is None or label not in self.priors:
            return None
        u, v = int(pixel_uv[0]), int(pixel_uv[1])
        H, W = frame.depth.shape
        if not (0 <= u < W and 0 <= v < H):
            return None
        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2RGB)
        try:
            self._predictor.set_image(rgb)
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array([[u, v]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
        except Exception:
            return None
        # 가장 score 좋고 size 적당한 mask 선택
        target_max_m = self.target_max_extent_m[label] * self.max_extent_ratio
        target_hsv = self.priors[label]
        target_ext = self.target_extents_sorted[label]
        best = None
        for mk_b, sc in zip(masks, scores):
            mk_b = mk_b.astype(bool)
            dv = mk_b & (frame.depth > 0.05) & (frame.depth < 1.5)
            if dv.sum() < 50:
                continue
            if self.keep_largest_cc:
                n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(dv.astype(np.uint8), 8)
                if n_cc <= 1:
                    continue
                i = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
                dv = (lbl == i)
                if dv.sum() < 50:
                    continue
            ft = _mask_features(dv, frame)
            if ft is None:
                continue
            if float(ft["extents_sorted"][0]) > target_max_m:
                continue
            cost = _match_cost(ft, target_hsv, target_ext,
                               self.match_color_w, self.match_shape_w)
            # cross-view 는 약간 더 관대: max_cost × 2
            if cost > self.match_max_cost * 2:
                continue
            if best is None or cost < best[0]:
                best = (cost, dv, ft)
        if best is None:
            return None
        cost, mk, ft = best
        ys, xs = np.where(mk)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        return CameraDetection(
            cam_id=frame.cam_id, label=label,
            score=float(1.0 / (1.0 + cost)),
            mask=mk, bbox_xyxy=bbox,
            centroid_cam=ft["cent_cam"], centroid_base=ft["cent_base"],
        )


class CNOSSegmenter:
    """CNOS wrapper skeleton (CUDA 권장)."""
    def __init__(self, cfg, ckpt, target_models): self._ok = False
    def available(self): return self._ok
    def segment(self, frame): raise NotImplementedError


# ===================================================================
#       CNOS-Lite: DINOv2 template matching validator
# ===================================================================

class CNOSLiteValidator:
    """경량 CNOS 대안.
       - target GLB 를 색 prior 로 colorize → N viewpoints raster
       - DINOv2 ViT-S/14 (CPU) embedding 캐시
       - detection crop embedding ↔ 모든 target template embedding cosine
       - 라벨 검증 (best target = 가장 높은 max sim)
       - sim < threshold → 거부 (다른 물체)
       - sim 이 다른 target 에 더 높음 → 라벨 swap (옵션)"""

    def __init__(self, dinov2_ckpt: Path, target_models: Dict[str, CanonicalModel],
                 color_priors: Dict[str, Tuple[int, int, int]],
                 n_views: int = 36, intr_for_render: Optional[CameraIntrinsics] = None):
        self._ok = False
        self.target_names = list(target_models.keys())
        self.template_embeds: Dict[str, np.ndarray] = {}   # name -> (N, 384) L2-normalized
        try:
            import torch
            sys_path_added = False
            if "dinov2" not in sys.modules:
                import sys as _sys
                here = str(Path(__file__).resolve().parent)
                if here not in _sys.path:
                    _sys.path.insert(0, here); sys_path_added = True
            from dinov2.models.vision_transformer import vit_small
            self.torch = torch
            self.model = vit_small(patch_size=14, img_size=518,
                                   init_values=1.0, num_register_tokens=0,
                                   block_chunks=0)
            sd = torch.load(str(dinov2_ckpt), map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd, strict=True)
            self.model.eval()
            # ImageNet normalization
            self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            # render templates (using a dummy intr at fixed distance)
            intr = intr_for_render or CameraIntrinsics(
                K=np.array([[600., 0, 112], [0, 600., 112], [0, 0, 1]]),
                dist=np.zeros(5), width=224, height=224)
            for name, model in target_models.items():
                color_bgr = hsv_to_bgr(color_priors.get(name, (0, 0, 200)))
                imgs = self._render_target_views(model, color_bgr, intr,
                                                 n_views=n_views)
                self.template_embeds[name] = self._embed_batch(imgs)
            self._ok = True
            print(f"[CNOSLite] templates: " +
                  ", ".join(f"{n}={v.shape[0]}" for n, v in self.template_embeds.items()))
        except Exception as e:
            print(f"[CNOSLite] init failed → off. ({e})")

    def available(self) -> bool: return self._ok

    def _render_target_views(self, model: CanonicalModel,
                             color_bgr: Tuple[int, int, int],
                             intr: CameraIntrinsics,
                             n_views: int = 36) -> List[np.ndarray]:
        """gravity-aligned rotations 의 subset 으로 다양한 viewpoint 렌더 (224x224)."""
        H = W = 224
        # 6 axes × 12 yaws = 72 → uniform pick n_views
        all_R = _gravity_aligned_rotations(np.array([0., -1., 0.]), n_yaws=12)
        if len(all_R) > n_views:
            stride = len(all_R) / n_views
            pick = [all_R[int(i * stride)] for i in range(n_views)]
        else:
            pick = all_R
        imgs = []
        z_dist = max(model.diameter * 1.6, 0.25)
        for R in pick:
            T_cam_obj = np.eye(4)
            T_cam_obj[:3, :3] = R
            T_cam_obj[:3, 3] = [0, 0, z_dist]
            mesh = model.mesh_low if model.mesh_low is not None else model.mesh
            sil = rasterize_mesh_mask(mesh, T_cam_obj, intr, H, W)
            img = np.full((H, W, 3), 200, dtype=np.uint8)   # gray bg
            img[sil.astype(bool)] = color_bgr
            imgs.append(img)
        return imgs

    def _preprocess_batch(self, imgs_bgr: List[np.ndarray]):
        torch = self.torch
        # BGR → RGB, resize 224, normalize
        arr = []
        for im in imgs_bgr:
            if im.shape[:2] != (224, 224):
                im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            arr.append(rgb)
        x = torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2)
        x = (x - self._mean) / self._std
        return x

    def _embed_batch(self, imgs_bgr: List[np.ndarray]) -> np.ndarray:
        torch = self.torch
        x = self._preprocess_batch(imgs_bgr)
        with torch.no_grad():
            y = self.model(x)
        y = torch.nn.functional.normalize(y, dim=-1).cpu().numpy()
        return y

    def crop_for_detection(self, rgb: np.ndarray, mask: np.ndarray,
                           bbox_xyxy: Tuple[int, int, int, int],
                           pad: int = 8) -> Optional[np.ndarray]:
        """detection mask 영역을 잘라 background gray 로 채운 224x224 crop."""
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = bbox_xyxy
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        crop = rgb[y0:y1, x0:x1].copy()
        m = mask[y0:y1, x0:x1].astype(bool)
        # background → gray
        crop[~m] = 200
        # square pad
        h, w = crop.shape[:2]
        side = max(h, w)
        sq = np.full((side, side, 3), 200, dtype=np.uint8)
        oy = (side - h) // 2; ox = (side - w) // 2
        sq[oy:oy + h, ox:ox + w] = crop
        return sq

    def best_target(self, det_embedding: np.ndarray
                    ) -> Tuple[str, float, Dict[str, float]]:
        """detection 의 embedding (384,) → 모든 target 별 max cosine sim."""
        sims_per_target = {}
        for name, T in self.template_embeds.items():
            sims = (T @ det_embedding).max() if T.size else -1.0
            sims_per_target[name] = float(sims)
        best = max(sims_per_target.items(), key=lambda x: x[1])
        return best[0], best[1], sims_per_target


def apply_cnos_lite_validation(detections_per_cam: List[List[CameraDetection]],
                               frames: List[CameraFrame],
                               validator: CNOSLiteValidator,
                               min_sim: float, allow_label_swap: bool
                               ) -> List[List[CameraDetection]]:
    """detection 별로 DINOv2 best_target 을 계산 →
       sim < min_sim 거부, label_swap 옵션 시 더 나은 target 으로 라벨 교체."""
    out: List[List[CameraDetection]] = []
    for ci, dets in enumerate(detections_per_cam):
        rgb = frames[ci].rgb
        crops = []
        valid_idx = []
        for di, d in enumerate(dets):
            cr = validator.crop_for_detection(rgb, d.mask, d.bbox_xyxy)
            if cr is None:
                continue
            crops.append(cr)
            valid_idx.append(di)
        if not crops:
            out.append([]); continue
        embs = validator._embed_batch(crops)         # (N, 384), L2-normalized
        keep: List[CameraDetection] = []
        for emb, di in zip(embs, valid_idx):
            d = dets[di]
            best_name, best_sim, sims = validator.best_target(emb)
            if best_sim < min_sim:
                continue                              # 거부 — 어느 target 에도 안 닮음
            if best_name != d.label:
                if allow_label_swap:
                    d = CameraDetection(
                        cam_id=d.cam_id, label=best_name,
                        score=float(best_sim),
                        mask=d.mask, bbox_xyxy=d.bbox_xyxy,
                        centroid_cam=d.centroid_cam, centroid_base=d.centroid_base,
                    )
                else:
                    continue                          # 라벨 mismatch → drop
            else:
                d = CameraDetection(
                    cam_id=d.cam_id, label=d.label, score=float(best_sim),
                    mask=d.mask, bbox_xyxy=d.bbox_xyxy,
                    centroid_cam=d.centroid_cam, centroid_base=d.centroid_base,
                )
            keep.append(d)
        out.append(keep)
        print(f"[CNOSLite] cam{frames[ci].cam_id}: "
              f"{len(dets)} → {len(keep)} after DINOv2 validation")
    return out


class FallbackSegmenter:
    """CPU-only color-prior segmenter (target subset 만)."""
    def __init__(self, target_models, color_priors,
                 min_depth_m, max_depth_m, min_sat, min_area_px,
                 hue_cost_thresh, topk_per_label):
        self.targets = list(target_models.keys())
        self.priors = {k: v for k, v in color_priors.items() if k in self.targets}
        if not self.priors:
            raise ValueError("FallbackSegmenter: no color priors for any target.")
        self.min_depth = min_depth_m; self.max_depth = max_depth_m
        self.min_sat = min_sat; self.min_area = min_area_px
        self.hue_cost_thresh = hue_cost_thresh; self.topk = topk_per_label

    def segment(self, frame: CameraFrame) -> List[CameraDetection]:
        H, W = frame.depth.shape
        valid = (frame.depth > self.min_depth) & (frame.depth < self.max_depth)
        if int(valid.sum()) < self.min_area:
            return []
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2HSV)
        h_img, s_img = hsv[:, :, 0], hsv[:, :, 1]
        fg = valid & (s_img >= self.min_sat)
        if int(fg.sum()) < self.min_area:
            return []
        names = list(self.priors.keys())
        cost = np.stack([hue_distance(h_img, self.priors[n][0]) for n in names], axis=0)
        per_lbl = np.argmin(cost, axis=0)
        per_min = cost.min(axis=0)
        gated = fg & (per_min < self.hue_cost_thresh)

        kernel = np.ones((3, 3), np.uint8)
        dets: List[CameraDetection] = []
        for ki, name in enumerate(names):
            mk = gated & (per_lbl == ki)
            if int(mk.sum()) < self.min_area:
                continue
            mk = cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
            mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, kernel, iterations=2)
            n_cc, lbl, stats, _ = cv2.connectedComponentsWithStats(mk, 8)
            if n_cc <= 1:
                continue
            order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[: self.topk]
            for oi in order:
                i = int(oi) + 1
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < self.min_area:
                    continue
                comp = (lbl == i)
                x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
                P_cam = backproject_masked(frame.depth, comp, frame.intr)
                if P_cam.shape[0] < 30:
                    continue
                cent_cam = P_cam.mean(axis=0)
                cent_base = transform_points(frame.T_base_cam, cent_cam[None])[0]
                dets.append(CameraDetection(
                    cam_id=frame.cam_id, label=name,
                    score=float(area) / float(H * W),
                    mask=comp, bbox_xyxy=(x, y, x + w, y + h),
                    centroid_cam=cent_cam, centroid_base=cent_base,
                ))
        return dets


# ===================================================================
#               COLOR PRIOR RESOLUTION (target subset only)
# ===================================================================

def _circular_hue_mean(hues: np.ndarray) -> float:
    rad = hues.astype(np.float64) * (np.pi / 90.0)
    a = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    if a < 0:
        a += 2 * np.pi
    return float(a * (90.0 / np.pi))


def discover_color_priors(frames: List[CameraFrame], target_names: List[str],
                          cfg: PipelineConfig
                          ) -> Dict[str, Tuple[int, int, int]]:
    n = len(target_names)
    if n == 0:
        return {}
    pixels = []
    for f in frames:
        valid = (f.depth > cfg.seg_min_depth_m) & (f.depth < cfg.seg_max_depth_m)
        hsv = cv2.cvtColor(f.rgb, cv2.COLOR_BGR2HSV)
        m = valid & (hsv[:, :, 1] >= cfg.seg_min_sat)
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        if len(ys) > 6000:
            sel = np.random.RandomState(0).choice(len(ys), 6000, replace=False)
            ys, xs = ys[sel], xs[sel]
        pixels.append(hsv[ys, xs])
    if not pixels:
        return {}
    P = np.concatenate(pixels, axis=0).astype(np.float32)
    rad = P[:, 0] * (np.pi / 90.0)
    feat = np.stack([np.cos(rad)*90, np.sin(rad)*90, P[:, 1]*0.4, P[:, 2]*0.1], axis=1).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lbl, _ = cv2.kmeans(feat, n, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.flatten()
    clusters = []
    for k in range(n):
        sel = (lbl == k)
        if sel.sum() < 50:
            continue
        clusters.append((int(_circular_hue_mean(P[sel, 0])),
                         int(P[sel, 1].mean()), int(P[sel, 2].mean())))
    clusters.sort(key=lambda x: x[0])
    return dict(zip(sorted(target_names), clusters))


def resolve_color_priors(frames: List[CameraFrame], target_names: List[str],
                         cfg: PipelineConfig
                         ) -> Dict[str, Tuple[int, int, int]]:
    raw: Dict[str, Tuple[int, int, int]] = {}
    if cfg.object_colors_json and Path(cfg.object_colors_json).exists():
        raw = load_color_priors_json(cfg.object_colors_json)
    elif (Path(cfg.data_dir) / "object_colors.json").exists():
        raw = load_color_priors_json(Path(cfg.data_dir) / "object_colors.json")
    elif cfg.auto_colors or not all(n in DEMO_COLOR_PRIORS_HSV for n in target_names):
        raw = discover_color_priors(frames, target_names, cfg)
        if raw and cfg.save_discovered_colors:
            save_color_priors_json(raw, Path(cfg.data_dir) / "object_colors.json")
    if not raw:
        raw = DEMO_COLOR_PRIORS_HSV
    return {n: raw[n] for n in target_names if n in raw}


def run_segmentation(frames: List[CameraFrame],
                     target_models: Dict[str, CanonicalModel],
                     cfg: PipelineConfig,
                     color_priors: Dict[str, Tuple[int, int, int]]
                     ) -> List[List[CameraDetection]]:
    seg = None
    # 우선순위: CNOS > MobileSAM > Fallback
    if cfg.use_cnos:
        s = CNOSSegmenter(cfg.cnos_cfg, cfg.cnos_ckpt, target_models)
        if s.available():
            seg = s
    if seg is None and cfg.use_mobilesam:
        if cfg.mobilesam_ckpt is None or not Path(cfg.mobilesam_ckpt).exists():
            print("[MobileSAM] checkpoint missing → fallback")
        else:
            s = MobileSAMSegmenter(
                ckpt=Path(cfg.mobilesam_ckpt),
                target_models=target_models,
                color_priors=color_priors,
                points_per_side=cfg.sam_points_per_side,
                pred_iou_thresh=cfg.sam_pred_iou_thresh,
                stability_score_thresh=cfg.sam_stability_score_thresh,
                min_mask_region_area=cfg.sam_min_mask_region_area,
                match_color_w=cfg.sam_match_color_w,
                match_shape_w=cfg.sam_match_shape_w,
                match_max_cost=cfg.sam_match_max_cost,
                topk_per_label=cfg.sam_topk_per_label,
                max_extent_ratio=cfg.sam_max_extent_ratio,
                keep_largest_cc=cfg.sam_keep_largest_cc,
                cross_view_prompt=cfg.sam_cross_view_prompt,
            )
            if s.available():
                seg = s
                print(f"[seg] MobileSAM active")
    if seg is None:
        seg = FallbackSegmenter(
            target_models=target_models, color_priors=color_priors,
            min_depth_m=cfg.seg_min_depth_m, max_depth_m=cfg.seg_max_depth_m,
            min_sat=cfg.seg_min_sat, min_area_px=cfg.seg_min_area_px,
            hue_cost_thresh=cfg.seg_hue_cost_thresh,
            topk_per_label=cfg.seg_topk_per_label,
        )
        print(f"[seg] color-prior fallback active")
    out = []
    for f in frames:
        dets = seg.segment(f)
        out.append([d for d in dets if d.label in target_models])    # 안전망
        print(f"[seg] cam{f.cam_id}: {len(out[-1])} detections")

    # ★ Cross-view validation: cam_i 의 가장 score 높은 detection centroid
    #   를 cam_j 에 reproject → SAM point-prompt 로 cam_j mask 회수.
    #   이미 같은 라벨 detection 이 있어도 더 강한 후보가 되도록 추가.
    if isinstance(seg, MobileSAMSegmenter) and cfg.sam_cross_view_prompt:
        for label in target_models:
            # 라벨별 가장 강한 source detection 찾기
            all_for_label = [(ci, d) for ci, dl in enumerate(out)
                             for d in dl if d.label == label]
            if not all_for_label:
                continue
            ci_src, d_src = max(all_for_label, key=lambda x: x[1].score)
            cent_base = d_src.centroid_base
            if cent_base is None:
                continue
            for ci_tgt, f_tgt in enumerate(frames):
                if ci_tgt == ci_src:
                    continue
                P_cam = transform_points(invert_se3(f_tgt.T_base_cam), cent_base[None])
                if P_cam[0, 2] <= 0:
                    continue
                uv = project_points(P_cam, f_tgt.intr)[0]
                d_new = seg.segment_at_point(f_tgt, label, (uv[0], uv[1]))
                if d_new is None:
                    continue
                # 새 detection 이 기존 best 보다 strong 하면 추가 (중복 dedup)
                # 위치 비슷한 기존 detection 있으면 score 더 높은 것만 유지
                replaced = False
                for i, d_existing in enumerate(out[ci_tgt]):
                    if d_existing.label != label:
                        continue
                    if (d_existing.centroid_base is not None
                            and np.linalg.norm(d_existing.centroid_base - d_new.centroid_base) < 0.05):
                        if d_new.score > d_existing.score:
                            out[ci_tgt][i] = d_new
                        replaced = True
                        break
                if not replaced:
                    out[ci_tgt].append(d_new)
        print(f"[seg] cross-view: " +
              ", ".join(f"cam{f.cam_id}={len(out[i])}" for i, f in enumerate(frames)))
    return out


# ===================================================================
#               CALIBRATION-AWARE ASSOCIATION
# ===================================================================

def associate_detections(detections_per_cam: List[List[CameraDetection]],
                         frames: List[CameraFrame],
                         target_models: Dict[str, CanonicalModel],
                         cfg: PipelineConfig) -> List[ObjectTrack]:
    by_label: Dict[str, Dict[int, List[CameraDetection]]] = {}
    for ci, dets in enumerate(detections_per_cam):
        for d in dets:
            if d.label not in target_models:
                continue
            by_label.setdefault(d.label, {}).setdefault(ci, []).append(d)

    tracks: List[ObjectTrack] = []
    next_id = 0
    thresh = cfg.centroid_match_thresh_m

    for label in target_models:
        if label not in by_label:
            continue
        per_cam = by_label[label]
        cams = sorted(per_cam.keys())
        best_combo: Optional[List[CameraDetection]] = None
        best_spread = float("inf")

        def _enum(idx: int, picked: List[CameraDetection]):
            nonlocal best_combo, best_spread
            if idx == len(cams):
                if len(picked) < 2:
                    return
                cents = np.stack([p.centroid_base for p in picked], axis=0)
                ok = all(np.linalg.norm(a.centroid_base - b.centroid_base) <= thresh
                         for a, b in combinations(picked, 2))
                if not ok:
                    return
                spread = float(np.linalg.norm(cents.std(axis=0)))
                if spread < best_spread:
                    best_combo = list(picked); best_spread = spread
                return
            ci = cams[idx]
            _enum(idx + 1, picked)
            for d in per_cam[ci]:
                _enum(idx + 1, picked + [d])

        _enum(0, [])

        if best_combo is None:
            if cfg.require_multiview_consensus:
                # 단일 cam track 은 신뢰도 낮음 → 폐기
                print(f"[assoc] drop {label} (no multi-view consensus)")
                continue
            for ci in cams:
                d_best = max(per_cam[ci], key=lambda d: d.score)
                tracks.append(ObjectTrack(track_id=next_id, label=label,
                                          detections=[d_best]))
                next_id += 1
        else:
            tracks.append(ObjectTrack(track_id=next_id, label=label,
                                      detections=list(best_combo)))
            next_id += 1
    return tracks


# ===================================================================
#                      DEPTH FUSION (base frame)
# ===================================================================

def fuse_track_pointcloud(track: ObjectTrack,
                          frames: List[CameraFrame]) -> ObjectTrack:
    by_cam = {f.cam_id: f for f in frames}
    pts, cols = [], []
    for d in track.detections:
        f = by_cam[d.cam_id]
        P_cam = backproject_masked(f.depth, d.mask, f.intr)
        if P_cam.shape[0] == 0:
            continue
        P_base = transform_points(f.T_base_cam, P_cam)
        ys, xs = np.where(d.mask & (f.depth > 1e-3))
        n = min(len(ys), P_base.shape[0])
        pts.append(P_base[:n]); cols.append(f.rgb[ys[:n], xs[:n]][:, ::-1])
    track.fused_points_base = (np.concatenate(pts, axis=0)
                               if pts else np.zeros((0, 3)))
    track.fused_colors = (np.concatenate(cols, axis=0)
                          if cols else np.zeros((0, 3), dtype=np.uint8))
    return track


# ===================================================================
#       POSE INIT (FoundationPose / MegaPose wrapper + fallback)
# ===================================================================

class FoundationPoseEstimator:
    def __init__(self, fp_dir, target_models): self._ok = False
    def available(self): return self._ok
    def estimate(self, track, model, frames): raise NotImplementedError


class MegaPoseEstimator:
    def __init__(self, mp_dir, target_models): self._ok = False
    def available(self): return self._ok
    def estimate(self, track, model, frames): raise NotImplementedError


def _pca_axes(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = P.mean(axis=0); Pc = P - centroid
    cov = Pc.T @ Pc / max(Pc.shape[0] - 1, 1)
    w, V = np.linalg.eigh(cov)
    R_obs = V[:, np.argsort(w)[::-1]]
    proj = Pc @ R_obs
    return R_obs, centroid, proj.max(0) - proj.min(0)


def _score_inlier(T: np.ndarray, P: np.ndarray, model: CanonicalModel) -> float:
    Pi = transform_points(invert_se3(T), P)
    half = model.extents / 2.0 + 0.02
    return float(np.all(np.abs(Pi) < half, axis=1).mean()) if Pi.size else 0.0


def fallback_initial_pose(track: ObjectTrack, model: CanonicalModel) -> PoseEstimate:
    P = track.fused_points_base
    if P is None or P.shape[0] < 30:
        return PoseEstimate(track.track_id, track.label, np.eye(4), 0.0, "fallback")
    R_obs, centroid, obs_ext = _pca_axes(P)
    obs_order = np.argsort(-obs_ext)
    glb_order = np.argsort(-np.asarray(model.extents))
    R = np.zeros((3, 3))
    for k in range(3):
        R[:, glb_order[k]] = R_obs[:, obs_order[k]]
    if np.linalg.det(R) < 0:
        R[:, glb_order[2]] *= -1
    best_T, best_s = np.eye(4), -1.0
    for s0 in (+1, -1):
        for s1 in (+1, -1):
            Rt = R.copy()
            Rt[:, glb_order[0]] *= s0; Rt[:, glb_order[1]] *= s1
            if np.linalg.det(Rt) < 0:
                Rt[:, glb_order[2]] *= -1
            T = np.eye(4); T[:3, :3] = Rt; T[:3, 3] = centroid
            sc = _score_inlier(T, P, model)
            if sc > best_s:
                best_T, best_s = T, sc
    return PoseEstimate(track.track_id, track.label, best_T, float(best_s), "fallback")


def estimate_initial_pose(track: ObjectTrack, model: CanonicalModel,
                          frames: List[CameraFrame], cfg: PipelineConfig
                          ) -> PoseEstimate:
    if cfg.use_megapose:
        mp = MegaPoseEstimator(cfg.megapose_dir, {model.name: model})
        if mp.available():
            try: return mp.estimate(track, model, frames)
            except Exception: pass
    if cfg.use_foundationpose:
        fp = FoundationPoseEstimator(cfg.foundationpose_dir, {model.name: model})
        if fp.available():
            try: return fp.estimate(track, model, frames)
            except Exception: pass
    return fallback_initial_pose(track, model)


# ===================================================================
#                   POSE REFINEMENT (ICP + silhouette)
# ===================================================================

def _se3_from_params(p: np.ndarray, T0: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(p[:3]).reshape(3, 1))
    dT = np.eye(4); dT[:3, :3] = R; dT[:3, 3] = p[3:]
    return dT @ T0


def refine_pose_icp(T_init: np.ndarray, P_target_base: np.ndarray,
                    model: CanonicalModel, iters: int = 30
                    ) -> Tuple[np.ndarray, float]:
    if not HAVE_O3D or P_target_base.shape[0] < 50:
        return T_init, 0.0
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.asarray(model.mesh.vertices))
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(model.diameter * 0.05, 0.005), max_nn=30))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(P_target_base)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(model.diameter * 0.05, 0.005), max_nn=30))
    thresh = max(model.diameter * 0.1, 0.01)
    try:
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, thresh, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        return np.asarray(reg.transformation), float(reg.fitness)
    except Exception:
        return T_init, 0.0


def _coarse_rotation_candidates() -> List[np.ndarray]:
    Rs: List[np.ndarray] = []
    base = [
        np.eye(3),
        np.array([[1,0,0],[0,0,-1],[0,1,0]],float),
        np.array([[1,0,0],[0,-1,0],[0,0,-1]],float),
        np.array([[1,0,0],[0,0,1],[0,-1,0]],float),
        np.array([[0,0,1],[0,1,0],[-1,0,0]],float),
        np.array([[0,0,-1],[0,1,0],[1,0,0]],float),
    ]
    for B in base:
        for k in range(4):
            t = k * np.pi / 2; c, s = np.cos(t), np.sin(t)
            Rs.append(B @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    return Rs


_COARSE_ROTS = _coarse_rotation_candidates()


def _rot_align(a_obj: np.ndarray, b_world: np.ndarray) -> np.ndarray:
    a = a_obj / (np.linalg.norm(a_obj) + 1e-12)
    b = b_world / (np.linalg.norm(b_world) + 1e-12)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3) if c > 0 else (np.eye(3) - 2 * np.outer(a, a))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _gravity_aligned_rotations(up_base: np.ndarray, n_yaws: int = 12) -> List[np.ndarray]:
    Rs: List[np.ndarray] = []
    obj_axes = [np.array([1.,0,0]), np.array([-1.,0,0]),
                np.array([0,1.,0]), np.array([0,-1.,0]),
                np.array([0,0,1.]), np.array([0,0,-1.])]
    u = up_base / (np.linalg.norm(up_base) + 1e-12)
    for ax in obj_axes:
        R_align = _rot_align(ax, up_base)
        for k in range(n_yaws):
            t = 2 * np.pi * k / n_yaws; c, s = np.cos(t), np.sin(t)
            ux, uy, uz = u
            R_yaw = np.array([
                [c+ux*ux*(1-c), ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c), uy*uz*(1-c)-ux*s],
                [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)],
            ])
            Rs.append(R_yaw @ R_align)
    return Rs


def _build_pcd_in_cam_frame(frame: CameraFrame, max_depth: float = 1.5,
                            voxel: float = 0.005):
    """frame.depth → PCD in CAM frame (open3d)."""
    if not HAVE_O3D:
        return None
    import open3d as o3d
    H, W = frame.depth.shape
    m = (frame.depth > 0.05) & (frame.depth < max_depth)
    ys, xs = np.where(m)
    z = frame.depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - frame.intr.cx) * z / frame.intr.fx
    y = (ys.astype(np.float64) - frame.intr.cy) * z / frame.intr.fy
    P = np.stack([x, y, z], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    if frame.rgb is not None:
        cols = frame.rgb[ys, xs][:, ::-1].astype(np.float64) / 255.0   # BGR→RGB
        pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd.voxel_down_sample(voxel)


def refine_extrinsics_with_depth(frames: List[CameraFrame],
                                 extrs: List[np.ndarray],
                                 voxel: float = 0.01,
                                 max_correction_mm: float = 100.0,
                                 max_correction_deg: float = 15.0,
                                 ) -> Tuple[List[np.ndarray], List[Dict]]:
    """cam0 reference. 현재 calibration 에서 시작해 multi-scale point-to-plane ICP 로 미세 보정.
       sanity bound 초과 보정은 거부 (RANSAC global 은 본 데이터에서 wild matches → 미사용).
       반환: (refined_extrs, info_per_cam)"""
    if not HAVE_O3D:
        return extrs, [{"err": "no open3d"}] * len(extrs)
    import open3d as o3d

    pcd_base = []
    for i, f in enumerate(frames):
        pc = _build_pcd_in_cam_frame(f, voxel=voxel)
        pc.transform(extrs[i])
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 2, max_nn=30))
        pcd_base.append(pc)
    pcd_ref = pcd_base[0]

    refined = [extrs[0].copy()]
    info = [{"cam": 0, "delta_trans_mm": [0, 0, 0], "delta_rot_deg": 0.0,
             "fitness_pre": 1.0, "fitness_post": 1.0,
             "rmse_pre_mm": 0.0, "rmse_post_mm": 0.0, "applied": True}]

    for i in range(1, len(frames)):
        pcd_i = pcd_base[i]
        # pre-ICP fitness baseline
        ev_pre = o3d.pipelines.registration.evaluate_registration(
            pcd_i, pcd_ref, voxel * 3, np.eye(4))

        # multi-scale ICP (coarse → fine)
        T_curr = np.eye(4)
        for thr in (voxel * 8, voxel * 4, voxel * 2):
            icp = o3d.pipelines.registration.registration_icp(
                pcd_i, pcd_ref, thr, T_curr,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
            T_curr = np.asarray(icp.transformation)

        delta = T_curr
        t_mm = (delta[:3, 3] * 1000)
        R_d = delta[:3, :3]
        ang_deg = float(np.degrees(np.arccos(np.clip((np.trace(R_d) - 1) / 2, -1, 1))))
        ev_post = o3d.pipelines.registration.evaluate_registration(
            pcd_i, pcd_ref, voxel * 3, delta)

        applied = (np.linalg.norm(t_mm) <= max_correction_mm
                   and ang_deg <= max_correction_deg
                   and ev_post.fitness >= ev_pre.fitness)   # 개선될 때만
        new_T = delta @ extrs[i] if applied else extrs[i].copy()
        refined.append(new_T)
        info.append({
            "cam": i,
            "delta_trans_mm": t_mm.round(2).tolist(),
            "delta_rot_deg": round(ang_deg, 3),
            "fitness_pre": float(ev_pre.fitness),
            "fitness_post": float(ev_post.fitness),
            "rmse_pre_mm": float(ev_pre.inlier_rmse * 1000),
            "rmse_post_mm": float(ev_post.inlier_rmse * 1000),
            "applied": bool(applied),
        })
    return refined, info


def estimate_table_normal(frames: List[CameraFrame],
                          max_depth_m: float = 1.5) -> Optional[np.ndarray]:
    if not HAVE_O3D:
        return None
    pts = []
    for f in frames:
        H, W = f.depth.shape
        m = (f.depth > 0.05) & (f.depth < max_depth_m)
        ys, xs = np.where(m)
        if len(ys) > 20000:
            sel = np.random.RandomState(0).choice(len(ys), 20000, replace=False)
            ys, xs = ys[sel], xs[sel]
        z = f.depth[ys, xs].astype(np.float64)
        x = (xs.astype(np.float64) - f.intr.cx) * z / f.intr.fx
        y = (ys.astype(np.float64) - f.intr.cy) * z / f.intr.fy
        pts.append(transform_points(f.T_base_cam, np.stack([x, y, z], axis=1)))
    if not pts:
        return None
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
    try:
        plane, _ = pcd.segment_plane(distance_threshold=0.005, ransac_n=3,
                                     num_iterations=200)
        n = np.asarray(plane[:3], dtype=np.float64)
        n /= (np.linalg.norm(n) + 1e-12)
        if n[1] > 0:
            n = -n
        return n
    except Exception:
        return None


def _silhouette_iou_loss(T: np.ndarray,
                         cam_data: List[Tuple[CameraFrame, np.ndarray]],
                         model: CanonicalModel, max_faces: int) -> float:
    mesh = model.mesh_low if model.mesh_low is not None else model.mesh
    loss = 0.0
    for f, det_mask in cam_data:
        T_cam_obj = invert_se3(f.T_base_cam) @ T
        sil = rasterize_mesh_mask(mesh, T_cam_obj, f.intr,
                                  f.rgb.shape[0], f.rgb.shape[1], max_faces=max_faces)
        sb = sil.astype(bool)
        inter = int(np.logical_and(sb, det_mask).sum())
        union = int(np.logical_or(sb, det_mask).sum())
        loss += 1.0 - inter / max(union, 1)
    return loss


def _chamfer_loss(T: np.ndarray,
                  P_base: np.ndarray,
                  model_kdtree,
                  diameter: float) -> float:
    """fused PCD (base frame) → mesh nearest-vertex distance (정규화).
       T_obj_base 적용 후 model frame 에서 KDTree query (mesh vertices 가 fixed)."""
    if P_base is None or P_base.shape[0] == 0 or model_kdtree is None:
        return 0.0
    P_obj = transform_points(invert_se3(T), P_base)
    d, _ = model_kdtree.query(P_obj, k=1)
    return float(np.mean(d) / max(diameter, 1e-6))     # 0 (perfect) ~ 1+


def _combined_loss(T: np.ndarray, sil_loss_fn, chamfer_args, w_chamfer: float) -> float:
    sl = sil_loss_fn(T)
    cl = _chamfer_loss(T, *chamfer_args) if w_chamfer > 0 else 0.0
    return sl + w_chamfer * cl


def _pca_major_axis(P: np.ndarray) -> Optional[np.ndarray]:
    """fused point cloud 의 PCA 최대 분산 축 (= 물체의 elongation 방향)."""
    if P is None or P.shape[0] < 30:
        return None
    Pc = P - P.mean(axis=0, keepdims=True)
    cov = Pc.T @ Pc / max(Pc.shape[0] - 1, 1)
    w, V = np.linalg.eigh(cov)
    return V[:, int(np.argmax(w))]


def refine_pose_silhouette(T_init: np.ndarray, track: ObjectTrack,
                           model: CanonicalModel,
                           frames_by_cam: Dict[int, CameraFrame],
                           max_iters: int = 80, max_faces: int = 5000,
                           up_base: Optional[np.ndarray] = None,
                           chamfer_weight: float = 1.5
                           ) -> Tuple[np.ndarray, float]:
    """Multi-view silhouette IoU refinement.
       1) coarse grid:
          (a) cube-symmetry × R_init  (24)
          (b) gravity-aligned ⊥ table   (6×12 = 72)
          (c) ★ PCD-major-axis-aligned (6×24 = 144) — 실제 elongation 방향에 맞춘 align
       2) Nelder-Mead 6 DOF fine refine
       ★ initial pose 의 IoU 보다 나빠지면 채택하지 않음 (pose degradation 방지)."""
    if not HAVE_SCIPY:
        return T_init, 0.0
    from scipy.optimize import minimize
    cam_data = [(frames_by_cam[d.cam_id], d.mask.astype(bool))
                for d in track.detections if d.cam_id in frames_by_cam]
    if not cam_data:
        return T_init, 0.0

    centroid = T_init[:3, 3].copy()
    R_init = T_init[:3, :3]
    coarse_faces = max(1500, max_faces // 3)

    # chamfer 준비: PCD + KDTree on model vertices (mesh 좌표계)
    P_base = track.fused_points_base
    use_chamfer = (chamfer_weight > 0
                   and HAVE_SCIPY
                   and P_base is not None
                   and P_base.shape[0] >= 30)
    kdtree = None
    if use_chamfer:
        try:
            from scipy.spatial import cKDTree
            mesh_use = model.mesh_low if model.mesh_low is not None else model.mesh
            kdtree = cKDTree(np.asarray(mesh_use.vertices))
        except Exception:
            use_chamfer = False
    chamfer_args = (P_base, kdtree, model.diameter)
    w_ch = chamfer_weight if use_chamfer else 0.0

    def loss_at(T_in: np.ndarray, faces: int) -> float:
        sl = _silhouette_iou_loss(T_in, cam_data, model, faces)
        cl = _chamfer_loss(T_in, *chamfer_args) if w_ch > 0 else 0.0
        return sl + w_ch * cl

    # initial pose loss → baseline (degradation guard)
    init_loss = loss_at(T_init, coarse_faces)
    best_T = T_init.copy(); best_loss = init_loss

    # (a) cube-symmetry rotations
    for Rc in _COARSE_ROTS:
        Tc = np.eye(4); Tc[:3, :3] = R_init @ Rc; Tc[:3, 3] = centroid
        l = loss_at(Tc, coarse_faces)
        if l < best_loss:
            best_loss, best_T = l, Tc

    # (b) gravity-aligned (table normal)
    if up_base is not None:
        for Rg in _gravity_aligned_rotations(up_base, n_yaws=12):
            Tc = np.eye(4); Tc[:3, :3] = Rg; Tc[:3, 3] = centroid
            l = loss_at(Tc, coarse_faces)
            if l < best_loss:
                best_loss, best_T = l, Tc

    # (c) PCD major-axis aligned — elongated object 에 결정적
    pca_axis = _pca_major_axis(P_base)
    if pca_axis is not None:
        for Rp in _gravity_aligned_rotations(pca_axis, n_yaws=24):
            Tc = np.eye(4); Tc[:3, :3] = Rp; Tc[:3, 3] = centroid
            l = loss_at(Tc, coarse_faces)
            if l < best_loss:
                best_loss, best_T = l, Tc

    # 2) Nelder-Mead — best_T 기준
    def total(p):
        return loss_at(_se3_from_params(p, best_T), max_faces)
    init_simplex = np.zeros((7, 6))
    deltas = np.array([0.08, 0.08, 0.08, 0.008, 0.008, 0.008])
    for i in range(6):
        init_simplex[i + 1, i] = deltas[i]
    try:
        res = minimize(total, np.zeros(6), method="Nelder-Mead",
                       options={"maxiter": max_iters, "xatol": 5e-4,
                                "fatol": 5e-4, "initial_simplex": init_simplex,
                                "adaptive": True})
        T_fine = _se3_from_params(res.x, best_T)
        fine_loss = float(res.fun)
    except Exception:
        T_fine, fine_loss = best_T, best_loss

    # ★ initial 보다 나쁘면 initial 유지 (degradation guard)
    if fine_loss <= best_loss and fine_loss <= init_loss:
        T_out, loss_out = T_fine, fine_loss
    elif best_loss < init_loss:
        T_out, loss_out = best_T, best_loss
    else:
        T_out, loss_out = T_init, init_loss
    return T_out, float(1.0 - loss_out / max(len(cam_data), 1))


def refine_pose(estimate: PoseEstimate, track: ObjectTrack,
                model: CanonicalModel, frames: List[CameraFrame],
                cfg: PipelineConfig, up_base: Optional[np.ndarray] = None
                ) -> PoseEstimate:
    T = estimate.T_base_obj.copy()
    conf = estimate.confidence; src = estimate.source
    if cfg.refine_icp and track.fused_points_base is not None \
            and track.fused_points_base.shape[0] >= 50:
        T_icp, fit = refine_pose_icp(T, track.fused_points_base, model)
        if fit > 0:
            T, conf, src = T_icp, fit, "refined_icp"
    if cfg.refine_silhouette and track.detections:
        frames_by_cam = {f.cam_id: f for f in frames}
        cam_data = [(frames_by_cam[d.cam_id], d.mask.astype(bool))
                    for d in track.detections if d.cam_id in frames_by_cam]
        cur_loss = _silhouette_iou_loss(T, cam_data, model, cfg.silhouette_max_faces) \
                   if cam_data else float("inf")
        T_sil, sc = refine_pose_silhouette(
            T, track, model, frames_by_cam,
            max_iters=cfg.silhouette_iters, max_faces=cfg.silhouette_max_faces,
            up_base=up_base, chamfer_weight=cfg.chamfer_weight,
        )
        new_loss = _silhouette_iou_loss(T_sil, cam_data, model,
                                        cfg.silhouette_max_faces) \
                   if cam_data else float("inf")
        if new_loss < cur_loss:
            T, conf, src = T_sil, sc, "refined_silhouette"
    return PoseEstimate(estimate.track_id, estimate.label, T, conf, src)


# ===================================================================
#                          RUN PIPELINE
# ===================================================================

def run_pipeline(cfg: PipelineConfig) -> Dict:
    if not cfg.target_glb_paths:
        raise ValueError("target_glb_paths is empty — selective pipeline requires explicit targets.")
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    intrs = load_intrinsics(cfg.intrinsics_dir, cfg.num_cams)
    calib_dir = Path(cfg.data_dir) / "cube_session_01" / "calib_out_cube"
    refined_dir = calib_dir / cfg.refined_extr_subdir
    if cfg.use_refined_extr and refined_dir.exists() and \
            all((refined_dir / f"T_C0_C{i}.npy").exists() for i in range(1, cfg.num_cams)):
        extrs = load_extrinsics(refined_dir, cfg.num_cams)
        print(f"[extr] loaded REFINED extrinsics from {refined_dir}")
    else:
        extrs = load_extrinsics(calib_dir, cfg.num_cams)

    # ★ TARGET-ONLY GLB
    target_models = load_target_library([Path(p) for p in cfg.target_glb_paths])
    print(f"[targets] {list(target_models.keys())}")

    cap_dir = Path(cfg.data_dir) / cfg.capture_subdir
    frames = [load_capture_frame(cap_dir, ci, cfg.frame_id, intrs[ci], extrs[ci],
                                 depth_scale=cfg.depth_scale)
              for ci in range(cfg.num_cams)]

    # ★ extrinsic refinement (depth ICP) — 한 번 보정 후 저장
    if cfg.refine_extrinsics:
        print("[extr] refining via multi-scale point-to-plane ICP...")
        refined_extrs, info = refine_extrinsics_with_depth(frames, extrs)
        any_applied = False
        for entry in info:
            tag = "✓ applied" if entry["applied"] else "✗ rejected"
            print(f"  cam{entry['cam']}: Δt={entry['delta_trans_mm']}mm "
                  f"Δrot={entry['delta_rot_deg']:.2f}°  "
                  f"fit {entry['fitness_pre']:.3f}→{entry['fitness_post']:.3f}  "
                  f"rmse {entry['rmse_pre_mm']:.1f}→{entry['rmse_post_mm']:.1f}mm  [{tag}]")
            if entry["cam"] > 0 and entry["applied"]:
                any_applied = True
        if any_applied:
            refined_dir.mkdir(parents=True, exist_ok=True)
            for i in range(cfg.num_cams):
                np.save(refined_dir / f"T_C0_C{i}.npy", refined_extrs[i])
            print(f"[extr] saved → {refined_dir}")
            for i, f in enumerate(frames):
                f.T_base_cam = refined_extrs[i]
            extrs = refined_extrs
        else:
            print("[extr] no improvement → kept original")

    color_priors = resolve_color_priors(frames, list(target_models.keys()), cfg)
    print(f"[priors] {color_priors}")

    dets_per_cam = run_segmentation(frames, target_models, cfg, color_priors)

    # ★ CNOS-Lite validation: DINOv2 template matching → 라벨 검증/교체
    if cfg.use_cnos_lite:
        if cfg.dinov2_ckpt is None or not Path(cfg.dinov2_ckpt).exists():
            print("[CNOSLite] dinov2 weights missing → skip")
        else:
            validator = CNOSLiteValidator(
                Path(cfg.dinov2_ckpt), target_models, color_priors,
                n_views=cfg.cnos_lite_n_views)
            if validator.available():
                dets_per_cam = apply_cnos_lite_validation(
                    dets_per_cam, frames, validator,
                    min_sim=cfg.cnos_lite_min_sim,
                    allow_label_swap=cfg.cnos_lite_label_swap)

    tracks = associate_detections(dets_per_cam, frames, target_models, cfg)
    tracks = [fuse_track_pointcloud(t, frames) for t in tracks]

    up_base = estimate_table_normal(frames) if cfg.refine_silhouette else None
    if up_base is not None:
        print(f"[gravity] table normal in base: {up_base.round(3).tolist()}")

    estimates: List[PoseEstimate] = []
    frames_by_cam = {f.cam_id: f for f in frames}
    for t in tracks:
        if t.label not in target_models:
            continue
        model = target_models[t.label]
        est0 = estimate_initial_pose(t, model, frames, cfg)
        est = refine_pose(est0, t, model, frames, cfg, up_base=up_base)

        # ★ Iterative outlier rejection: cam 별 silhouette IoU 가 낮으면 제거 → 재refine
        for it in range(cfg.iterative_outlier_passes):
            if len(t.detections) <= 1:
                break
            keep = []
            removed_cams = []
            for d in t.detections:
                f = frames_by_cam.get(d.cam_id)
                if f is None:
                    continue
                T_cam_obj = invert_se3(f.T_base_cam) @ est.T_base_obj
                mesh_use = model.mesh_low if model.mesh_low is not None else model.mesh
                sil = rasterize_mesh_mask(mesh_use, T_cam_obj, f.intr,
                                          f.rgb.shape[0], f.rgb.shape[1],
                                          max_faces=cfg.silhouette_max_faces)
                sb = sil.astype(bool); db = d.mask.astype(bool)
                inter = int(np.logical_and(sb, db).sum())
                union = int(np.logical_or(sb, db).sum())
                iou = inter / max(union, 1)
                if iou >= cfg.per_cam_iou_min:
                    keep.append(d)
                else:
                    removed_cams.append((d.cam_id, iou))
            if len(keep) == len(t.detections):
                break    # 변화 없음
            if len(keep) < 1 or (cfg.require_multiview_consensus and len(keep) < 2):
                est = PoseEstimate(t.track_id, t.label, est.T_base_obj, 0.0, "rejected_outlier")
                break
            print(f"[outlier] {t.label}: dropped " +
                  ", ".join(f"cam{c}(IoU={i:.2f})" for c, i in removed_cams))
            t.detections = keep
            t = fuse_track_pointcloud(t, frames)
            est = refine_pose(estimate_initial_pose(t, model, frames, cfg),
                              t, model, frames, cfg, up_base=up_base)

        if est.confidence < cfg.min_confidence or est.source == "rejected_outlier":
            continue
        estimates.append(est)

    posed_glb_paths: Dict[int, str] = {}
    for e in estimates:
        p = export_posed_glb(target_models[e.label], e.T_base_obj,
                             cfg.output_dir / f"posed_{e.label}_track{e.track_id}.glb",
                             frame=cfg.export_frame)
        posed_glb_paths[e.track_id] = str(p)
    json_p = export_results_json(estimates, tracks, target_models, posed_glb_paths,
                                 cfg.output_dir / f"poses_{cfg.frame_id:06d}.json")
    npz_p = export_results_npz(estimates, cfg.output_dir / f"poses_{cfg.frame_id:06d}.npz")
    descs = load_object_descriptions(Path(cfg.data_dir) / "object_descriptions.json")
    txt_p = export_results_text(estimates, target_models,
                                cfg.output_dir / f"poses_{cfg.frame_id:06d}.txt",
                                cfg.frame_id, descriptions=descs)

    label_color_bgr = {n: hsv_to_bgr(hsv) for n, hsv in color_priors.items()}
    comp_path = None
    if cfg.save_overlays:
        ov = cfg.output_dir / "overlay"; ov.mkdir(parents=True, exist_ok=True)
        for f in frames:
            img = f.rgb.copy()
            for e in estimates:
                color = label_color_bgr.get(e.label, (0, 255, 0))
                img = render_mesh_silhouette(img, f, target_models[e.label],
                                             e.T_base_obj, color)
            cv2.imwrite(str(ov / f"cam{f.cam_id}_frame{f.frame_id:06d}.jpg"), img)
        comp_dir = cfg.output_dir / f"comparison_{cfg.frame_id:06d}"
        comp_dir.mkdir(parents=True, exist_ok=True)
        grid = make_comparison_grid(frames, target_models, estimates,
                                    label_color_bgr, cfg.frame_id)
        comp_path = comp_dir / f"comparison_{cfg.frame_id:06d}.png"
        cv2.imwrite(str(comp_path), grid)

    return {
        "targets": list(target_models.keys()),
        "n_estimates": len(estimates),
        "estimates": [{"object_name": e.label, "track_id": e.track_id,
                       "confidence": e.confidence, "source": e.source,
                       "posed_glb": posed_glb_paths.get(e.track_id)}
                      for e in estimates],
        "json": str(json_p), "npz": str(npz_p), "text": str(txt_p),
        "comparison": str(comp_path) if comp_path else None,
        "output_dir": str(cfg.output_dir),
    }
