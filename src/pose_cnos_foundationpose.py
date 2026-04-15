#!/usr/bin/env python3
"""
CNOS + FoundationPose multi-view CAD pose pipeline.

This file owns the model wrappers, pose refinement, export, and CLI.
Shared data/geometry/loading/rendering base lives in `pose_cnos_foundationpose_core.py`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from pose_cnos_foundationpose_core import (
    COLORS,
    DEPTH_POLICY,
    OBJECT_LABELS,
    T_ISAAC_CV,
    CameraDetection,
    CameraFrame,
    CanonicalModel,
    PoseEstimate,
    associate_by_label_and_geometry,
    build_camera_detections_from_cnos,
    cam_pose_to_base,
    combine_pose_scores,
    coerce_matrix4x4,
    coverage_score,
    fuse_track_points,
    icp_point_to_point,
    local_pose_candidates,
    load_calibration,
    load_frame,
    load_glb_library,
    majority_label,
    make_virtual_camera,
    mask_bbox,
    normalize_glb,
    render_model_to_image,
    render_compare_score,
    mv_depth_score,
    sample_model_points,
    save_camera_detections,
    silhouette_iou_score,
    transform_points,
    voxel_downsample,
)


def maybe_prepend_sys_path(path: Optional[Path]) -> None:
    if path is None:
        return
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class CNOSSegmenter:
    def __init__(
        self,
        cfg_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        runner: Optional[str] = None,
        repo_dir: Optional[str] = None,
        template_cache_dir: Optional[str] = None,
        conf_threshold: float = 0.45,
        stability_score_thresh: float = 0.5,
        num_max_dets: int = 12,
        template_yaw_steps: int = 18,
    ):
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.runner_cmd = shlex.split(runner) if runner else None
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else None
        self.template_cache_dir = Path(template_cache_dir).resolve() if template_cache_dir else Path(tempfile.gettempdir()) / "cnos_template_cache"
        self.conf_threshold = float(conf_threshold)
        self.stability_score_thresh = float(stability_score_thresh)
        self.num_max_dets = int(num_max_dets)
        self.template_yaw_steps = int(template_yaw_steps)
        self.object_library: Dict[str, str] = {}
        self.template_dirs: Dict[str, Path] = {}
        self._repo_state: Optional[dict] = None
        self._reference_cache: Dict[str, Any] = {}

    def build_object_library(self, glb_paths: Dict[str, Path]) -> None:
        self.object_library = {label: str(path) for label, path in sorted(glb_paths.items())}
        if self.repo_dir is not None and self.runner_cmd is None:
            self.template_dirs = self._prepare_template_library()

    def _resolve_repo_dir(self) -> Optional[Path]:
        if self.repo_dir is not None:
            return self.repo_dir
        env_dir = os.environ.get("CNOS_REPO_DIR")
        return Path(env_dir).resolve() if env_dir else None

    def _prepare_template_library(self) -> Dict[str, Path]:
        template_dirs: Dict[str, Path] = {}
        self.template_cache_dir.mkdir(parents=True, exist_ok=True)
        for label, glb_path_str in self.object_library.items():
            glb_path = Path(glb_path_str)
            out_dir = self.template_cache_dir / label
            if not sorted(out_dir.glob("*.png")):
                self._render_object_templates(label, glb_path, out_dir)
            template_dirs[label] = out_dir
        return template_dirs

    def _render_object_templates(self, label: str, glb_path: Path, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        model = normalize_glb(glb_path)
        cam = make_virtual_camera(image_size=224, focal_px=220.0)
        pitches = (-18.0, 0.0, 18.0)
        yaw_values = np.linspace(0.0, 360.0, num=max(self.template_yaw_steps, 6), endpoint=False)
        distance = max(float(np.max(model.extents_m)) * 2.4, 0.20)

        idx = 0
        for pitch in pitches:
            for yaw in yaw_values:
                T_obj = np.eye(4, dtype=np.float64)
                T_obj[:3, :3] = Rot.from_euler("yx", [yaw, pitch], degrees=True).as_matrix()
                T_obj[:3, 3] = np.array([0.0, 0.0, distance], dtype=np.float64)
                rendered_bgr, rendered_mask, _ = render_model_to_image(model, T_obj, cam, scale=1.0)
                if np.count_nonzero(rendered_mask) < 30:
                    continue
                x0, y0, x1, y1 = mask_bbox(rendered_mask).tolist()
                crop = rendered_bgr[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                cv2.imwrite(str(out_dir / f"{label}_{idx:03d}.png"), crop)
                idx += 1
        if idx == 0:
            raise RuntimeError(f"CNOS template rendering failed for {label}: {glb_path}")

    def _cnos_config_target(self) -> Tuple[Path, str]:
        repo_dir = self._resolve_repo_dir()
        if repo_dir is None:
            raise RuntimeError("CNOS repo directory is not set. --cnos_repo_dir 또는 CNOS_REPO_DIR 필요")
        if self.cfg_path:
            cfg_path = Path(self.cfg_path)
            if not cfg_path.is_absolute():
                cfg_path = (repo_dir / cfg_path).resolve()
            return cfg_path.parent, cfg_path.stem
        return (repo_dir / "configs").resolve(), "run_inference"

    def _ensure_repo_state(self) -> dict:
        if self._repo_state is not None:
            return self._repo_state

        repo_dir = self._resolve_repo_dir()
        if repo_dir is None:
            raise RuntimeError("CNOS repo directory is not configured")
        maybe_prepend_sys_path(repo_dir)

        try:
            torch = importlib.import_module("torch")
            hydra_mod = importlib.import_module("hydra")
            hydra_utils = importlib.import_module("hydra.utils")
            pil_image = importlib.import_module("PIL.Image")
            bbox_utils = importlib.import_module("src.utils.bbox_utils")
            loss_mod = importlib.import_module("src.model.loss")
            utils_mod = importlib.import_module("src.model.utils")
        except Exception as e:
            raise RuntimeError(f"CNOS direct API import failed from {repo_dir}: {e}") from e

        initialize_config_dir = getattr(hydra_mod, "initialize_config_dir")
        compose = getattr(hydra_mod, "compose")
        instantiate = getattr(hydra_utils, "instantiate")
        Image = pil_image
        CropResizePad = getattr(bbox_utils, "CropResizePad")
        Similarity = getattr(loss_mod, "Similarity")
        Detections = getattr(utils_mod, "Detections")

        config_dir, config_name = self._cnos_config_target()
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=config_name)

        cfg_segmentor = cfg.model.segmentor_model
        target_name = str(getattr(cfg_segmentor, "_target_", ""))
        if "fast_sam" not in target_name:
            cfg.model.segmentor_model.stability_score_thresh = self.stability_score_thresh

        model = instantiate(cfg.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.descriptor_model.model = model.descriptor_model.model.to(device)
        model.descriptor_model.model.device = device
        if hasattr(model.segmentor_model, "predictor"):
            model.segmentor_model.predictor.model = model.segmentor_model.predictor.model.to(device)
        elif hasattr(model.segmentor_model, "model") and hasattr(model.segmentor_model.model, "setup_model"):
            model.segmentor_model.model.setup_model(device=device, verbose=False)

        self._repo_state = {
            "torch": torch,
            "Image": Image,
            "CropResizePad": CropResizePad,
            "Similarity": Similarity,
            "Detections": Detections,
            "device": device,
            "model": model,
            "metric": Similarity(),
        }
        return self._repo_state

    def _reference_features(self, label: str):
        if label in self._reference_cache:
            return self._reference_cache[label]

        state = self._ensure_repo_state()
        torch = state["torch"]
        Image = state["Image"]
        CropResizePad = state["CropResizePad"]
        model = state["model"]
        device = state["device"]

        template_dir = self.template_dirs.get(label)
        if template_dir is None:
            raise RuntimeError(f"CNOS template directory missing for {label}")
        template_paths = sorted(template_dir.glob("*.png"))
        if not template_paths:
            raise RuntimeError(f"CNOS template PNGs missing for {label}: {template_dir}")

        boxes = []
        templates = []
        for path in template_paths:
            image = Image.open(path).convert("RGB")
            boxes.append(image.getbbox())
            templates.append(np.array(image).astype(np.float32) / 255.0)
        templates_t = torch.from_numpy(np.stack(templates, axis=0)).permute(0, 3, 1, 2)
        boxes_t = torch.tensor(np.array(boxes), dtype=torch.long)
        proposal_processor = CropResizePad(224)
        templates_t = proposal_processor(images=templates_t, boxes=boxes_t).to(device)
        ref_feats = model.descriptor_model.compute_features(templates_t, token_name="x_norm_clstoken")
        self._reference_cache[label] = ref_feats
        return ref_feats

    def _predict_via_repo_api(self, image_bgr: np.ndarray) -> List[dict]:
        state = self._ensure_repo_state()
        torch = state["torch"]
        Detections = state["Detections"]
        model = state["model"]
        metric = state["metric"]

        rgb_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        proposals = Detections(model.segmentor_model.generate_masks(rgb_rgb))
        if len(proposals) == 0:
            return []

        descriptors = model.descriptor_model.forward(rgb_rgb, proposals)
        labels = sorted(self.object_library)
        if not labels:
            return []

        per_label_scores = []
        for label in labels:
            ref_feats = self._reference_features(label)
            topk_k = min(5, int(ref_feats.shape[0]))
            sim = metric(descriptors[:, None, :], ref_feats[None, :, :])
            label_scores = torch.mean(torch.topk(sim, k=topk_k, dim=-1)[0], dim=-1)
            per_label_scores.append(label_scores)

        score_matrix = torch.stack(per_label_scores, dim=1)
        best_scores, best_label_ids = torch.max(score_matrix, dim=1)
        keep = torch.nonzero(best_scores >= self.conf_threshold).flatten()
        if keep.numel() == 0:
            return []

        selected = proposals.clone()
        selected.add_attribute("scores", best_scores)
        selected.add_attribute("object_ids", best_label_ids)
        selected.filter(keep)
        if len(selected) > self.num_max_dets:
            top_ids = torch.topk(selected.scores, k=self.num_max_dets).indices
            selected.filter(top_ids)
        if hasattr(selected, "apply_nms_per_object_id"):
            selected.apply_nms_per_object_id(nms_thresh=0.5)
        selected.to_numpy()

        outputs: List[dict] = []
        for idx in range(len(selected.boxes)):
            outputs.append({
                "label": labels[int(selected.object_ids[idx])],
                "score": float(selected.scores[idx]),
                "mask": (np.asarray(selected.masks[idx]) > 0).astype(np.uint8) * 255,
                "bbox_xyxy": np.asarray(selected.boxes[idx], dtype=np.int32),
            })
        return outputs

    def _load_mask(self, mask_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"CNOS mask load failed: {mask_path}")
        if mask.shape != image_shape:
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    def predict(self, image_bgr: np.ndarray) -> List[dict]:
        if not self.object_library:
            raise RuntimeError("CNOS object library is empty. build_object_library() 먼저 호출하세요.")
        if self.runner_cmd is None and self._resolve_repo_dir() is not None:
            return self._predict_via_repo_api(image_bgr)
        if not self.runner_cmd:
            raise RuntimeError("CNOS runner/repo가 설정되지 않았습니다. --cnos_repo_dir 또는 --cnos_runner 필요합니다.")

        with tempfile.TemporaryDirectory(prefix="cnos_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            image_path = tmpdir / "image.png"
            library_path = tmpdir / "library.json"
            output_path = tmpdir / "predictions.json"
            cv2.imwrite(str(image_path), image_bgr)
            with open(library_path, "w", encoding="utf-8") as f:
                json.dump(self.object_library, f, indent=2, ensure_ascii=False)

            cmd = self.runner_cmd + ["--image", str(image_path), "--library", str(library_path), "--output", str(output_path)]
            if self.cfg_path:
                cmd.extend(["--cfg", self.cfg_path])
            if self.ckpt_path:
                cmd.extend(["--ckpt", self.ckpt_path])
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"CNOS runner failed: {e.stderr.strip() or e.stdout.strip() or e}") from e

            with open(output_path, "r", encoding="utf-8") as f:
                raw_preds = json.load(f)

            outputs: List[dict] = []
            image_shape = image_bgr.shape[:2]
            for pred in raw_preds:
                mask_path = Path(pred["mask_path"])
                if not mask_path.is_absolute():
                    mask_path = tmpdir / mask_path
                mask_u8 = self._load_mask(mask_path, image_shape)
                bbox_xyxy = pred.get("bbox_xyxy")
                outputs.append({
                    "label": str(pred["label"]),
                    "score": float(pred.get("score", 1.0)),
                    "mask": mask_u8,
                    "bbox_xyxy": np.asarray(bbox_xyxy, dtype=np.int32) if bbox_xyxy is not None else mask_bbox(mask_u8),
                })
            return outputs


class FoundationPoseEstimator:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        runner: Optional[str] = None,
        repo_dir: Optional[str] = None,
        refine_iterations: int = 5,
    ):
        self.model_dir = model_dir
        self.runner_cmd = shlex.split(runner) if runner else None
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else None
        self.refine_iterations = int(refine_iterations)
        self._repo_state: Optional[dict] = None
        self._estimator_cache: Dict[str, Any] = {}

    def _resolve_repo_dir(self) -> Optional[Path]:
        if self.repo_dir is not None:
            return self.repo_dir
        env_dir = os.environ.get("FOUNDATIONPOSE_REPO_DIR")
        return Path(env_dir).resolve() if env_dir else None

    def _ensure_repo_state(self) -> dict:
        if self._repo_state is not None:
            return self._repo_state

        repo_dir = self._resolve_repo_dir()
        if repo_dir is None:
            raise RuntimeError("FoundationPose repo directory is not configured")
        maybe_prepend_sys_path(repo_dir)

        try:
            estimater_mod = importlib.import_module("estimater")
            utils_mod = importlib.import_module("Utils")
            torch = importlib.import_module("torch")
        except Exception as e:
            raise RuntimeError(f"FoundationPose direct API import failed from {repo_dir}: {e}") from e

        self._repo_state = {
            "estimater_mod": estimater_mod,
            "utils_mod": utils_mod,
            "torch": torch,
        }
        return self._repo_state

    def _build_estimator(self, glb_path: str):
        state = self._ensure_repo_state()
        estimater_mod = state["estimater_mod"]
        utils_mod = state["utils_mod"]

        mesh_scene = trimesh.load(str(glb_path))
        mesh = trimesh.util.concatenate(list(mesh_scene.geometry.values())) if isinstance(mesh_scene, trimesh.Scene) else mesh_scene
        mesh = mesh.copy()
        model_pts = np.asarray(mesh.vertices, dtype=np.float32)
        model_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        scorer = estimater_mod.ScorePredictor()
        refiner = estimater_mod.PoseRefinePredictor()
        glctx = utils_mod.dr.RasterizeCudaContext()
        return estimater_mod.FoundationPose(
            model_pts=model_pts,
            model_normals=model_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
        )

    def _estimate_via_repo_api(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        glb_path: str,
        depth_scale: Optional[float],
    ) -> dict:
        estimator = self._estimator_cache.get(glb_path)
        if estimator is None:
            estimator = self._build_estimator(glb_path)
            self._estimator_cache[glb_path] = estimator

        rgb_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth_m = depth.astype(np.float32) * float(depth_scale) if depth_scale is not None else depth.astype(np.float32)
        T_cam_obj = estimator.register(
            K=np.asarray(K, dtype=np.float32),
            rgb=rgb_rgb,
            depth=depth_m,
            ob_mask=(mask > 0),
            iteration=self.refine_iterations,
        )

        score = 0.0
        scores_attr = getattr(estimator, "scores", None)
        if scores_attr is not None:
            try:
                score = float(scores_attr[0].item() if hasattr(scores_attr[0], "item") else scores_attr[0])
            except Exception:
                score = float(np.asarray(scores_attr)[0])
        return {
            "T_cam_obj": coerce_matrix4x4(T_cam_obj, "FoundationPose T_cam_obj"),
            "score": score,
        }

    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        glb_path: str,
        depth_scale: Optional[float] = None,
    ) -> dict:
        if self.runner_cmd is None and self._resolve_repo_dir() is not None:
            return self._estimate_via_repo_api(rgb, depth, mask, K, glb_path, depth_scale)
        if not self.runner_cmd:
            raise RuntimeError("FoundationPose runner/repo가 설정되지 않았습니다. --fp_repo_dir 또는 --fp_runner 필요합니다.")

        with tempfile.TemporaryDirectory(prefix="foundationpose_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            rgb_path = tmpdir / "rgb.png"
            depth_path = tmpdir / "depth.png"
            mask_path = tmpdir / "mask.png"
            intrinsics_path = tmpdir / "intrinsics.json"
            output_path = tmpdir / "pose.json"
            cv2.imwrite(str(rgb_path), rgb)
            cv2.imwrite(str(depth_path), depth)
            cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
            with open(intrinsics_path, "w", encoding="utf-8") as f:
                json.dump({"K": np.asarray(K, dtype=np.float64).tolist()}, f, indent=2)

            cmd = self.runner_cmd + [
                "--rgb", str(rgb_path),
                "--depth", str(depth_path),
                "--mask", str(mask_path),
                "--glb", str(glb_path),
                "--intrinsics_json", str(intrinsics_path),
                "--output", str(output_path),
            ]
            if self.model_dir:
                cmd.extend(["--model_dir", self.model_dir])
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FoundationPose runner failed: {e.stderr.strip() or e.stdout.strip() or e}") from e

            with open(output_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {
                "T_cam_obj": coerce_matrix4x4(raw["T_cam_obj"], "FoundationPose T_cam_obj"),
                "score": float(raw.get("score", 0.0)),
            }


def evaluate_pose_candidate(
    model: CanonicalModel,
    model_pts: np.ndarray,
    track_pts: np.ndarray,
    frames: List[CameraFrame],
    observed_masks: List[np.ndarray],
    T0: np.ndarray,
    pre_score: float,
    scale: float,
) -> PoseEstimate:
    max_dim = float(np.max(model.extents_m))
    T_icp, fitness, rmse = icp_point_to_point(
        model_pts * scale,
        track_pts,
        T0,
        max_iter=35,
        max_corr=max(0.008, max_dim * 0.25),
    )
    aligned = transform_points(model_pts * scale, T_icp)
    ds = mv_depth_score(aligned, frames, observed_masks)
    cs = coverage_score(aligned, track_pts, radius=max(0.004, max_dim * 0.08))
    ss = 0.30 * silhouette_iou_score(aligned, frames, observed_masks) + 0.70 * render_compare_score(model, T_icp, frames, observed_masks, scale=scale)
    conf = 0.10 * pre_score + 0.90 * combine_pose_scores(model.name, ds, cs, ss)

    R = T_icp[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    T_clean = T_icp.copy()
    T_clean[:3, :3] = R_clean
    rot = Rot.from_matrix(R_clean)
    return PoseEstimate(
        T_base_obj=T_clean,
        position_m=T_clean[:3, 3].copy(),
        quaternion_xyzw=rot.as_quat(),
        euler_xyz_deg=rot.as_euler("xyz", degrees=True),
        scale=float(scale),
        confidence=float(conf),
        fitness=float(fitness),
        rmse=float(rmse),
        depth_score=float(ds),
        coverage=float(cs),
        silhouette_score=float(ss),
        model_name=model.name,
    )


def refine_pose_with_depth_and_render(
    model: CanonicalModel,
    model_pts: np.ndarray,
    fused_points_base: np.ndarray,
    T_init_base_obj: np.ndarray,
    frames: List[CameraFrame],
    observed_masks: List[np.ndarray],
    scale_candidates: Optional[List[float]] = None,
) -> PoseEstimate:
    if len(fused_points_base) < 40:
        raise RuntimeError("정합 실패: track points too small")

    best: Optional[PoseEstimate] = None
    candidate_poses = local_pose_candidates(T_init_base_obj, model.name)
    for scale in scale_candidates or [0.97, 1.0, 1.03]:
        ranked = []
        for T_candidate in candidate_poses:
            ranked.append((render_compare_score(model, T_candidate, frames, observed_masks, scale=scale), T_candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for pre_score, T_candidate in ranked[:10]:
            pose = evaluate_pose_candidate(model, model_pts, fused_points_base, frames, observed_masks, T_candidate, pre_score=float(pre_score), scale=scale)
            if best is None or pose.confidence > best.confidence:
                best = pose

    if best is None:
        raise RuntimeError("정합 실패")
    return best


def export_result(
    pose: PoseEstimate,
    model: CanonicalModel,
    frame_id: str,
    out_dir: Path,
    glb_src_path: Path,
    result_suffix: str = "",
    extra: Optional[dict] = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = result_suffix if not result_suffix or result_suffix.startswith("_") else f"_{result_suffix}"
    result = {
        "frame_id": frame_id,
        "object_name": model.name,
        "label": OBJECT_LABELS.get(model.name, model.name),
        "coordinate_frame": "base (= cam0)",
        "unit": "meter",
        "position_m": pose.position_m.tolist(),
        "quaternion_xyzw": pose.quaternion_xyzw.tolist(),
        "euler_xyz_deg": pose.euler_xyz_deg.tolist(),
        "T_base_obj": pose.T_base_obj.tolist(),
        "rotation_matrix": pose.T_base_obj[:3, :3].tolist(),
        "scale": pose.scale,
        "real_size_m": {
            "x": float(model.extents_m[0] * pose.scale),
            "y": float(model.extents_m[1] * pose.scale),
            "z": float(model.extents_m[2] * pose.scale),
        },
        "confidence": pose.confidence,
        "fitness": pose.fitness,
        "rmse": pose.rmse,
        "depth_score": pose.depth_score,
        "coverage": pose.coverage,
        "silhouette_score": pose.silhouette_score,
    }
    if pose.init_cam_id is not None:
        result["init_cam_id"] = pose.init_cam_id
    if pose.init_score > 0.0:
        result["init_score"] = pose.init_score
    if extra:
        result.update(extra)

    np.savez(
        out_dir / f"pose_{model.name}_{frame_id}{suffix}.npz",
        T_base_obj=pose.T_base_obj,
        position_m=pose.position_m,
        quaternion_xyzw=pose.quaternion_xyzw,
        scale=pose.scale,
    )

    for coord in ["opencv", "isaac"]:
        scene = trimesh.load(str(glb_src_path))
        mesh = trimesh.util.concatenate(list(scene.geometry.values())) if isinstance(scene, trimesh.Scene) else scene.copy()
        verts = (mesh.vertices - model.center) * pose.scale
        verts_h = np.hstack([verts, np.ones((len(verts), 1))])
        verts_pose = (pose.T_base_obj @ verts_h.T)[:3].T
        if coord == "isaac":
            verts_pose = (T_ISAAC_CV @ np.hstack([verts_pose, np.ones((len(verts_pose), 1))]).T)[:3].T
        mesh.vertices = verts_pose
        coord_suffix = "" if coord == "opencv" else "_isaac"
        gp = out_dir / f"{model.name}_posed_{frame_id}{suffix}{coord_suffix}.glb"
        mesh.export(str(gp), file_type="glb")
        result[f"posed_glb_{coord}"] = str(gp)

    json_path = out_dir / f"pose_{model.name}_{frame_id}{suffix}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def render_wireframe(mesh: trimesh.Trimesh, center: np.ndarray, pose: PoseEstimate, cam: CameraFrame, color=(0, 255, 0), thickness: int = 2):
    img = cam.color_bgr.copy()
    h, w = img.shape[:2]
    K = cam.intrinsics.K
    verts = (mesh.vertices - center) * pose.scale
    vc = transform_points(transform_points(verts, pose.T_base_obj), np.linalg.inv(cam.T_base_cam))
    z = vc[:, 2]
    ok = z > 0.05
    pu = np.full(len(verts), -1.0)
    pv = np.full(len(verts), -1.0)
    pu[ok] = K[0, 0] * vc[ok, 0] / z[ok] + K[0, 2]
    pv[ok] = K[1, 1] * vc[ok, 1] / z[ok] + K[1, 2]
    for e0, e1 in mesh.edges_unique:
        if not (ok[e0] and ok[e1]):
            continue
        p0 = (int(pu[e0]), int(pv[e0]))
        p1 = (int(pu[e1]), int(pv[e1]))
        if abs(p0[0]) > 2 * w or abs(p0[1]) > 2 * h or abs(p1[0]) > 2 * w or abs(p1[1]) > 2 * h:
            continue
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


def save_combined_overlay(all_poses: List[PoseEstimate], all_models: List[CanonicalModel], all_masks: List[List[np.ndarray]], frames: List[CameraFrame], frame_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    base_imgs = [cam.color_bgr.copy() for cam in frames]
    for obj_idx, (pose, model, masks) in enumerate(zip(all_poses, all_models, all_masks)):
        color = COLORS[obj_idx % len(COLORS)]
        for ci, cam in enumerate(frames):
            wire = render_wireframe(model.mesh, model.center, pose, cam, color=color, thickness=2)
            diff = cv2.absdiff(wire, cam.color_bgr)
            wire_mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) > 8
            base_imgs[ci][wire_mask] = wire[wire_mask]
            cnts, _ = cv2.findContours(masks[ci], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base_imgs[ci], cnts, -1, color, 1)
            if masks[ci].any():
                ys, xs = np.where(masks[ci] > 0)
                anchor = (int(xs.mean()), max(20, int(ys.min()) - 4))
                cv2.putText(base_imgs[ci], OBJECT_LABELS.get(model.name, model.name), anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for ci in range(len(frames)):
        cv2.putText(base_imgs[ci], f"cam{ci}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.imwrite(str(out_dir / f"overlay_{frame_id}.png"), np.hstack(base_imgs))


def run_pipeline(
    data_dir: str,
    intrinsics_dir: str,
    frame_id: str,
    glb_path: Optional[str] = None,
    output_dir: str = "src/output/pose_pipeline_cnos_fp",
    multi_object: bool = False,
    capture_subdir: str = "object_capture",
    cnos_cfg: Optional[str] = None,
    cnos_ckpt: Optional[str] = None,
    cnos_runner: Optional[str] = None,
    cnos_repo_dir: Optional[str] = None,
    fp_model_dir: Optional[str] = None,
    fp_runner: Optional[str] = None,
    fp_repo_dir: Optional[str] = None,
    association_dist_m: float = 0.05,
    association_reproj_px: float = 25.0,
):
    data_dir_p = Path(data_dir)
    intr_dir_p = Path(intrinsics_dir)
    out = Path(output_dir)

    print("=" * 64)
    print(f" CNOS + FoundationPose Multi-View CAD Pose — Frame {frame_id} {'[MULTI]' if multi_object else '[SINGLE]'}")
    print("=" * 64)

    print("\n[1] load_calibration()")
    intrinsics, extrinsics = load_calibration(data_dir_p, intr_dir_p)
    print("[2] load_frame()")
    frames = load_frame(data_dir_p, frame_id, intrinsics, extrinsics, capture_subdir=capture_subdir)
    print("[3] GLB library")
    glb_paths, all_models = load_glb_library(data_dir_p, glb_path)
    if not all_models:
        raise RuntimeError("GLB library가 비어 있습니다.")

    print("[4] CNOS segmentation")
    cnos = CNOSSegmenter(cfg_path=cnos_cfg, ckpt_path=cnos_ckpt, runner=cnos_runner, repo_dir=cnos_repo_dir)
    cnos.build_object_library(glb_paths)
    detections_by_cam: Dict[int, List[CameraDetection]] = {}
    for cam in frames:
        detections = build_camera_detections_from_cnos(cam, cnos)
        detections_by_cam[cam.cam_id] = detections
        labels = ", ".join(det.label or "unknown" for det in detections) or "-"
        print(f"  cam{cam.cam_id}: detections={len(detections)} labels=[{labels}]")
    save_camera_detections(detections_by_cam, frames, frame_id, out)

    print("[5] Cross-view association")
    tracks = associate_by_label_and_geometry(frames, detections_by_cam, centroid_thresh_m=association_dist_m, reproj_thresh_px=association_reproj_px)
    if not tracks:
        raise RuntimeError("association 실패: track이 0개")
    print(f"  tracks={len(tracks)}")
    for ti, track in enumerate(tracks):
        cams = ",".join(f"cam{cid}" for cid in sorted(track.detections.keys()))
        status = "ok" if track.consistency.passed else "fail"
        print(
            f"    #{ti} {track.track_id}: label={track.label or 'unknown'} views={len(track.detections)} [{cams}] "
            f"centroid={track.consistency.centroid_spread_m:.4f}m reproj={track.consistency.mean_reprojection_error_px:.1f}px "
            f"bbox_iou={track.consistency.mean_bbox_iou:.2f} mask_iou={track.consistency.mean_mask_iou:.2f} {status}"
        )

    print("[6] Depth fusion + FoundationPose init + refinement")
    fp = FoundationPoseEstimator(model_dir=fp_model_dir, runner=fp_runner, repo_dir=fp_repo_dir)
    all_results = []
    all_poses: List[PoseEstimate] = []
    all_model_objs: List[CanonicalModel] = []
    all_masks_list: List[List[np.ndarray]] = []

    for track in tracks:
        label = track.label or majority_label(track)
        if not label:
            print(f"  - {track.track_id}: skip, no CNOS label")
            continue
        if label not in all_models:
            print(f"  - {track.track_id}: skip, label {label} not in GLB library")
            continue
        if len(track.detections) > 1 and not track.consistency.passed:
            print(f"  - {track.track_id}: skip, association consistency failed")
            continue

        print(f"  - {track.track_id} ({label})")
        track_pts = fuse_track_points(frames, track)
        track.fused_points_base = track_pts
        if len(track_pts) < 60:
            print("    skip: fused points too small")
            continue
        track_pts = voxel_downsample(track_pts, DEPTH_POLICY["voxel_size_m"])
        ext = track_pts.max(axis=0) - track_pts.min(axis=0)
        print(f"    fused_pts={len(track_pts)} extent=[{ext[0]*100:.1f},{ext[1]*100:.1f},{ext[2]*100:.1f}]cm")

        model = all_models[label]
        glb_src_path = glb_paths[label]
        model_pts = sample_model_points(model)

        per_view_poses: List[Tuple[int, np.ndarray, float]] = []
        for cam in frames:
            det = track.detections.get(cam.cam_id)
            if det is None:
                continue
            out_pose = fp.estimate(
                rgb=cam.color_bgr,
                depth=cam.depth_u16,
                mask=det.mask_u8,
                K=cam.intrinsics.K,
                glb_path=str(glb_src_path),
                depth_scale=cam.intrinsics.depth_scale,
            )
            score = float(out_pose.get("score", 0.0))
            T_base_obj = cam_pose_to_base(cam.T_base_cam, coerce_matrix4x4(out_pose["T_cam_obj"], "FoundationPose output T_cam_obj"))
            per_view_poses.append((cam.cam_id, T_base_obj, score))
            print(f"    cam{cam.cam_id}: init_score={score:.3f}")

        if not per_view_poses:
            print("    skip: FoundationPose init failed on every view")
            continue

        best_cam_id, T_init, init_score = max(per_view_poses, key=lambda item: item[2])
        pose = refine_pose_with_depth_and_render(
            model=model,
            model_pts=model_pts,
            fused_points_base=track_pts,
            T_init_base_obj=T_init,
            frames=frames,
            observed_masks=track.observed_masks,
        )
        pose.init_cam_id = best_cam_id
        pose.init_score = init_score
        print(
            f"    selected init: cam{best_cam_id} score={init_score:.3f} "
            f"refined_conf={pose.confidence:.3f} fit={pose.fitness:.3f} sil={pose.silhouette_score:.3f}"
        )

        result = export_result(
            pose,
            model,
            frame_id,
            out,
            glb_src_path,
            result_suffix=track.track_id,
            extra={
                "track_id": track.track_id,
                "track_label": label,
                "association": {
                    "pair_count": track.consistency.pair_count,
                    "centroid_spread_m": track.consistency.centroid_spread_m,
                    "mean_reprojection_error_px": track.consistency.mean_reprojection_error_px,
                    "mean_bbox_iou": track.consistency.mean_bbox_iou,
                    "mean_mask_iou": track.consistency.mean_mask_iou,
                    "passed": track.consistency.passed,
                },
            },
        )
        all_results.append(result)
        all_poses.append(pose)
        all_model_objs.append(model)
        all_masks_list.append(track.observed_masks)
        if not multi_object:
            break

    if all_poses:
        save_combined_overlay(all_poses, all_model_objs, all_masks_list, frames, frame_id, out)
        print(f"\n  overlay: {out / f'overlay_{frame_id}.png'}")
    else:
        print("\n  [WARN] export할 pose가 없습니다.")

    return all_results


def run_from_args(args, frame_id: str):
    return run_pipeline(
        data_dir=args.data_dir,
        intrinsics_dir=args.intrinsics_dir,
        frame_id=frame_id,
        glb_path=args.glb,
        output_dir=args.output_dir,
        multi_object=args.multi,
        capture_subdir=args.capture_subdir,
        cnos_cfg=args.cnos_cfg,
        cnos_ckpt=args.cnos_ckpt,
        cnos_runner=args.cnos_runner,
        cnos_repo_dir=args.cnos_repo_dir,
        fp_model_dir=args.fp_model_dir,
        fp_runner=args.fp_runner,
        fp_repo_dir=args.fp_repo_dir,
        association_dist_m=args.association_dist_m,
        association_reproj_px=args.association_reproj_px,
    )


def main():
    parser = argparse.ArgumentParser(description="CNOS + FoundationPose calibration-aware multi-view CAD pose pipeline")
    parser.add_argument("--data_dir", default="src/data")
    parser.add_argument("--intrinsics_dir", default="src/intrinsics")
    parser.add_argument("--capture_subdir", default="object_capture")
    parser.add_argument("--frame_id", default=None)
    parser.add_argument("--glb", default=None)
    parser.add_argument("--output_dir", default="src/output/pose_pipeline_cnos_fp")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--cnos_cfg", default=None)
    parser.add_argument("--cnos_ckpt", default=None)
    parser.add_argument("--cnos_runner", default=None)
    parser.add_argument("--cnos_repo_dir", default=None)
    parser.add_argument("--fp_model_dir", default=None)
    parser.add_argument("--fp_runner", default=None)
    parser.add_argument("--fp_repo_dir", default=None)
    parser.add_argument("--association_dist_m", type=float, default=0.05)
    parser.add_argument("--association_reproj_px", type=float, default=25.0)
    args = parser.parse_args()

    if args.batch:
        cam0_dir = Path(args.data_dir) / args.capture_subdir / "cam0"
        fids = sorted(f.stem.replace("rgb_", "") for f in cam0_dir.glob("rgb_*.jpg"))
        print(f"batch frames: {len(fids)}")
        all_results = []
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for fid in fids:
            try:
                all_results.extend(run_from_args(args, fid))
            except Exception as e:
                print(f"[ERROR] {fid}: {e}")
                all_results.append({"frame_id": fid, "error": str(e)})
        with open(Path(args.output_dir) / "batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    else:
        run_from_args(args, args.frame_id or "000000")


__all__ = [
    "CNOSSegmenter",
    "FoundationPoseEstimator",
    "main",
    "refine_pose_with_depth_and_render",
    "run_from_args",
    "run_pipeline",
]


if __name__ == "__main__":
    main()
