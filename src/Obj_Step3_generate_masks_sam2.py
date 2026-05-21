#!/usr/bin/env python3
"""
SAM 2 기반 multi-cam mask 생성 (안정화 버전).

[실행명령어]
conda activate sam2env

python src/Obj_Step3_generate_masks_sam2.py \
  --data_dir ./capture_obj \
  --out_dir ./masks \
  --sam_checkpoint ~/sam2_checkpoints/sam2_hiera_large.pt \
  --sam_config configs/sam2/sam2_hiera_l.yaml \
  --device cpu

UI:
  창에 cam0/cam1/cam2 가 한꺼번에 표시됨.

  [idle] 모드에서 cam0 패널을 클릭 → 터미널에서 객체명 입력 → [refining_cam0] 진입

  [refining_cam0] (cam0 패널에서)
    좌클릭 = 양성 포인트 추가, 우클릭 = 음성 포인트 추가
    클릭마다 cam0 마스크가 실시간 업데이트 (postprocess: 클릭점 포함 CC + closing)
    Enter = 확정 → cam1/cam2 로 자동 전파
    Esc   = 현재 객체 취소

  [confirming]
    cam0/1/2 모든 마스크 미리보기
    cam1/cam2 패널 좌클릭 = 그 카메라 마스크 추가 양성 포인트, 우클릭 = 음성 포인트
    y = 저장,  n = 폐기,  r = cam0 다시 refine 모드로 복귀

전파시:
  - cam0 마스크 안 픽셀의 depth 중 median ± depth_tol_m 만 유지 → outlier 깊이로 인한 박스 확장 차단
  - 좁아진 박스 + 중심점 prompt 로 SAM2 호출
  - 결과는 마지막 클릭점 기준 largest connected component 만 유지

저장 위치:
  masks/<object>/cam{0,1,2}_mask.png  + prompt.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: F401  (SAM2 가 cuda init 시 필요)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------- IO helpers ----------

def load_K(p: Path) -> np.ndarray:
    return np.loadtxt(p, dtype=np.float64).reshape(3, 3)


def load_T(p: Path) -> np.ndarray:
    return np.loadtxt(p, dtype=np.float64).reshape(4, 4)


def load_depth(p: Path, scale: float) -> np.ndarray:
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(p)
    return d.astype(np.float32) * scale


# ---------- geometry ----------

def project_to_cam(X_world: np.ndarray, T_cam_to_world: np.ndarray, K_cam: np.ndarray):
    """world(=cam0) 좌표 점을 cam_i 픽셀로 reproject. (u,v,z) or None."""
    T_inv = np.linalg.inv(T_cam_to_world)
    Xh = np.array([X_world[0], X_world[1], X_world[2], 1.0])
    Xi = (T_inv @ Xh)[:3]
    if Xi[2] <= 1e-6:
        return None
    u = K_cam[0, 0] * Xi[0] / Xi[2] + K_cam[0, 2]
    v = K_cam[1, 1] * Xi[1] / Xi[2] + K_cam[1, 2]
    return float(u), float(v), float(Xi[2])


def reproject_mask_points_filtered(
    mask_ref: np.ndarray,
    depth_ref: np.ndarray,
    K_ref: np.ndarray,
    T_ref_to_world: np.ndarray,
    K_tgt: np.ndarray,
    T_tgt_to_world: np.ndarray,
    tgt_shape: tuple,
    depth_median_tol_m: float = 0.05,
    max_samples: int = 2000,
):
    """cam_ref 마스크 픽셀 중 depth median ± tol 만 유지 후 cam_tgt 픽셀로 reproject."""
    ys, xs = np.where(mask_ref)
    if len(ys) == 0:
        return None, None
    ds = depth_ref[ys, xs]
    valid = (ds > 0) & np.isfinite(ds)
    ys, xs, ds = ys[valid], xs[valid], ds[valid]
    if len(ys) < 5:
        return None, None

    med = float(np.median(ds))
    keep = np.abs(ds - med) < depth_median_tol_m
    ys, xs, ds = ys[keep], xs[keep], ds[keep]
    if len(ys) < 5:
        return None, None

    if len(ys) > max_samples:
        idx = np.random.choice(len(ys), max_samples, replace=False)
        ys, xs, ds = ys[idx], xs[idx], ds[idx]

    fx, fy = K_ref[0, 0], K_ref[1, 1]
    cx, cy = K_ref[0, 2], K_ref[1, 2]
    Xc = np.stack(
        [(xs - cx) * ds / fx, (ys - cy) * ds / fy, ds, np.ones_like(ds)], axis=0
    )
    Xw = T_ref_to_world @ Xc
    Xt = np.linalg.inv(T_tgt_to_world) @ Xw
    Xt = Xt[:3]
    keep2 = Xt[2] > 1e-6
    Xt = Xt[:, keep2]
    if Xt.shape[1] < 5:
        return None, None
    u = K_tgt[0, 0] * Xt[0] / Xt[2] + K_tgt[0, 2]
    v = K_tgt[1, 1] * Xt[1] / Xt[2] + K_tgt[1, 2]
    H, W = tgt_shape[:2]
    in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u[in_img], v[in_img]
    if len(u) < 5:
        return None, med
    return (u, v), med


# ---------- mask post-process ----------

def postprocess_mask(mask: np.ndarray, anchor_xy: tuple, close_kernel: int = 5) -> np.ndarray:
    """closing 으로 작은 hole 메우고, anchor 픽셀 포함 connected component 만 유지.

    anchor 가 마스크 밖이면 가장 큰 component 유지.
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return mask
    if close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n_lbl, labels = cv2.connectedComponents(m)
    if n_lbl <= 1:
        return mask
    H, W = labels.shape
    cx = int(np.clip(round(anchor_xy[0]), 0, W - 1))
    cy = int(np.clip(round(anchor_xy[1]), 0, H - 1))
    lbl = int(labels[cy, cx])
    if lbl == 0:
        sizes = np.bincount(labels.flatten())
        sizes[0] = 0
        if sizes.max() == 0:
            return mask
        lbl = int(sizes.argmax())
    return labels == lbl


# ---------- SAM2 wrappers ----------

def sam_predict_pts(
    predictor: SAM2ImagePredictor,
    rgb: np.ndarray,
    positive: list,
    negative: list,
    box: np.ndarray | None = None,
    multimask: bool = True,
):
    """positive/negative 픽셀 좌표 리스트 + 선택 box 로 SAM2 호출."""
    predictor.set_image(rgb)
    pts_list = list(positive) + list(negative)
    if not pts_list and box is None:
        return None, 0.0
    kw: dict = {"multimask_output": multimask}
    if pts_list:
        kw["point_coords"] = np.array(pts_list, dtype=np.float32)
        kw["point_labels"] = np.array(
            [1] * len(positive) + [0] * len(negative), dtype=np.int32
        )
    if box is not None:
        kw["box"] = np.asarray(box, dtype=np.float32)
        # box 만 단독일 때 multimask_output 은 보통 False 가 안정적
        if not pts_list:
            kw["multimask_output"] = False
    masks, scores, _ = predictor.predict(**kw)
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


# ---------- overlay ----------

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 200, 0), alpha=0.5) -> np.ndarray:
    out = rgb.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


# ---------- main UI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./capture_obj")
    ap.add_argument("--out_dir", default="./masks")
    ap.add_argument(
        "--sam_checkpoint",
        default=str(Path.home() / "sam2_checkpoints/sam2_hiera_large.pt"),
    )
    ap.add_argument(
        "--sam_config",
        default="configs/sam2/sam2_hiera_l.yaml",
        help="hydra config (SAM 2 repo 내부 경로)",
    )
    ap.add_argument("--depth_scale", type=float, default=0.001, help="depth unit → m")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--depth_tol_m", type=float, default=0.05,
                    help="cam0 마스크 픽셀 중 median depth 와의 허용 차이 (m)")
    args = ap.parse_args()

    data = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    cam_ids = [p.stem.replace("_rgb", "") for p in sorted(data.glob("cam*_rgb.png"))]
    if not cam_ids:
        raise FileNotFoundError(f"No cam*_rgb.png in {data}")
    print(f"[INFO] cams found: {cam_ids}")

    rgbs, depths, Ks, Ts = {}, {}, {}, {}
    for ci in cam_ids:
        bgr = cv2.imread(str(data / f"{ci}_rgb.png"), cv2.IMREAD_COLOR)
        rgbs[ci] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depths[ci] = load_depth(data / f"{ci}_depth.png", args.depth_scale)
        Ks[ci] = load_K(data / f"{ci}_K.txt")
        Ts[ci] = load_T(data / f"{ci}_T_cam_to_world.txt")

    ref = "cam0"
    if ref not in cam_ids:
        raise RuntimeError("cam0 not found (ref/world frame)")
    if not np.allclose(Ts[ref], np.eye(4), atol=1e-6):
        print("[WARN] cam0_T_cam_to_world is not identity — world frame may not be cam0.")

    print(f"[INFO] Loading SAM 2 from {args.sam_checkpoint}  device={args.device}")
    model = build_sam2(args.sam_config, args.sam_checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(model)

    fig, axes = plt.subplots(1, len(cam_ids), figsize=(5 * len(cam_ids), 5))
    if len(cam_ids) == 1:
        axes = [axes]
    fig.canvas.manager.set_window_title("SAM2 mask UI") if fig.canvas.manager else None

    state = {
        "mode": "idle",   # idle | refining_cam0 | confirming
        "n_done": 0,
        "current": None,
        "overlays": {ci: rgbs[ci].copy() for ci in cam_ids},  # confirmed objects accumulated
        "rng": np.random.default_rng(123),
    }

    def status_text() -> str:
        if state["mode"] == "idle":
            return (f"[idle] cam0 클릭으로 객체 추가  (저장={state['n_done']})  "
                    "창 닫으면 종료")
        if state["mode"] == "refining_cam0":
            cur = state["current"]
            return (f"[refining cam0] {cur['name']}  "
                    f"+={len(cur['positive'])} -={len(cur['negative'])}  "
                    "L=+ R=-  Enter=propagate  Esc=cancel")
        if state["mode"] == "confirming":
            cur = state["current"]
            return (f"[confirming] {cur['name']}  "
                    "cam1/cam2 클릭으로 refine  y=save  n=discard  r=back to cam0")
        return ""

    def cur_overlay(ci):
        img = state["overlays"][ci].copy()
        if state["current"] is not None:
            m = state["current"]["masks"].get(ci)
            if m is not None:
                img = overlay_mask(img, m, color=(255, 200, 0), alpha=0.5)
        return img

    def render():
        for ax, ci in zip(axes, cam_ids):
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(cur_overlay(ci))
            if state["current"] is not None:
                if ci == ref:
                    for (u, v) in state["current"]["positive"]:
                        ax.plot(u, v, "+", color="lime", markersize=12, markeredgewidth=2)
                    for (u, v) in state["current"]["negative"]:
                        ax.plot(u, v, "x", color="red", markersize=10, markeredgewidth=2)
                else:
                    for (u, v) in state["current"]["positive_extra"].get(ci, []):
                        ax.plot(u, v, "+", color="lime", markersize=10, markeredgewidth=2)
                    for (u, v) in state["current"]["negative_extra"].get(ci, []):
                        ax.plot(u, v, "x", color="red", markersize=10, markeredgewidth=2)
            ax.set_title(ci)
        fig.suptitle(status_text(), fontsize=10)
        fig.canvas.draw_idle()

    def run_sam_cam0():
        cur = state["current"]
        if not cur["positive"]:
            cur["masks"][ref] = None
            return
        mask, score = sam_predict_pts(predictor, rgbs[ref], cur["positive"], cur["negative"])
        if mask is None:
            return
        mask = postprocess_mask(mask, cur["positive"][0])
        cur["masks"][ref] = mask
        cur["scores"][ref] = score

    def propagate_to_others():
        cur = state["current"]
        m0 = cur["masks"].get(ref)
        if m0 is None or m0.sum() < 10:
            print("[WARN] cam0 mask empty; cannot propagate")
            return
        for ci in cam_ids:
            if ci == ref:
                continue
            reproj, median_d = reproject_mask_points_filtered(
                m0, depths[ref], Ks[ref], Ts[ref], Ks[ci], Ts[ci], rgbs[ci].shape,
                depth_median_tol_m=args.depth_tol_m,
            )
            if reproj is None:
                print(f"[WARN] {ci}: not enough valid filtered depth — skip propagation")
                cur["masks"][ci] = None
                continue
            u_arr, v_arr = reproj
            pad = 10
            Hi, Wi = rgbs[ci].shape[:2]
            x1 = max(0.0, float(u_arr.min()) - pad)
            y1 = max(0.0, float(v_arr.min()) - pad)
            x2 = min(Wi - 1.0, float(u_arr.max()) + pad)
            y2 = min(Hi - 1.0, float(v_arr.max()) + pad)
            box = [x1, y1, x2, y2]
            cu = float(np.median(u_arr))
            cv_ = float(np.median(v_arr))

            mask, score = sam_predict_pts(
                predictor, rgbs[ci], positive=[(cu, cv_)], negative=[],
                box=np.array(box, dtype=np.float32), multimask=False,
            )
            if mask is None:
                cur["masks"][ci] = None
                continue
            mask = postprocess_mask(mask, (cu, cv_))
            cur["masks"][ci] = mask
            cur["scores"][ci] = score
            cur["boxes"][ci] = box
            cur["centers"][ci] = (cu, cv_)
            cur["cam0_median_depth"] = median_d

    def refine_other_cam(ci: str, u: float, v: float, positive: bool):
        cur = state["current"]
        key = "positive_extra" if positive else "negative_extra"
        cur[key].setdefault(ci, []).append((u, v))
        box = cur["boxes"].get(ci)
        center = cur["centers"].get(ci)
        pos_list = [center] if center is not None else []
        pos_list += cur["positive_extra"].get(ci, [])
        neg_list = cur["negative_extra"].get(ci, [])
        mask, score = sam_predict_pts(
            predictor, rgbs[ci], positive=pos_list, negative=neg_list,
            box=np.array(box, dtype=np.float32) if box is not None else None,
            multimask=False,
        )
        if mask is None:
            return
        anchor = pos_list[-1] if pos_list else (u, v)
        mask = postprocess_mask(mask, anchor)
        cur["masks"][ci] = mask
        cur["scores"][ci] = score

    def save_current():
        cur = state["current"]
        name = cur["name"]
        obj_dir = out_root / name
        obj_dir.mkdir(parents=True, exist_ok=True)
        prompt_log: dict = {}
        for ci in cam_ids:
            m = cur["masks"].get(ci)
            if m is None:
                continue
            cv2.imwrite(str(obj_dir / f"{ci}_mask.png"), (m.astype(np.uint8) * 255))
            if ci == ref:
                pt0 = cur["positive"][0]
                prompt_log[ci] = {
                    "x": float(pt0[0]), "y": float(pt0[1]),
                    "score": cur["scores"].get(ci, 0.0),
                    "positive_pts": [[float(p[0]), float(p[1])] for p in cur["positive"]],
                    "negative_pts": [[float(p[0]), float(p[1])] for p in cur["negative"]],
                    "cam0_median_depth_m": cur.get("cam0_median_depth"),
                }
            else:
                cu, cv_ = cur["centers"][ci]
                prompt_log[ci] = {
                    "x": float(cu), "y": float(cv_),
                    "score": cur["scores"].get(ci, 0.0),
                    "box": [float(x) for x in cur["boxes"][ci]],
                    "positive_extra": [[float(p[0]), float(p[1])] for p in cur["positive_extra"].get(ci, [])],
                    "negative_extra": [[float(p[0]), float(p[1])] for p in cur["negative_extra"].get(ci, [])],
                }
        with open(obj_dir / "prompt.json", "w") as f:
            json.dump(prompt_log, f, indent=2)
        print(f"[SAVE] {obj_dir}/  (cams={list(prompt_log.keys())})")

        # accumulate into overlays for next click visibility
        color = tuple(int(x) for x in (state["rng"].random(3) * 200 + 55))
        for ci in cam_ids:
            m = cur["masks"].get(ci)
            if m is not None:
                state["overlays"][ci] = overlay_mask(state["overlays"][ci], m,
                                                     color=color, alpha=0.4)
        state["n_done"] += 1

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        # which cam panel?
        cam_idx = None
        for i, ax in enumerate(axes):
            if event.inaxes == ax:
                cam_idx = i
                break
        if cam_idx is None:
            return
        ci = cam_ids[cam_idx]
        u, v = float(event.xdata), float(event.ydata)
        Hc, Wc = rgbs[ci].shape[:2]
        if not (0 <= u < Wc and 0 <= v < Hc):
            return

        if state["mode"] == "idle":
            if ci != ref:
                print(f"[INFO] idle 모드에선 cam0 패널을 클릭하세요. (you clicked {ci})")
                return
            default_name = f"obj{state['n_done'] + 1:02d}"
            print(f"\n>>> click cam0=({u:.1f},{v:.1f}). 객체명 [Enter={default_name}, 'q'=skip]: ",
                  end="", flush=True)
            try:
                name = input().strip()
            except EOFError:
                name = ""
            if name.lower() == "q":
                print("   skip")
                return
            if not name:
                name = default_name
            state["current"] = {
                "name": name,
                "positive": [(u, v)],
                "negative": [],
                "positive_extra": {},
                "negative_extra": {},
                "masks": {},
                "scores": {},
                "boxes": {},
                "centers": {},
            }
            state["mode"] = "refining_cam0"
            run_sam_cam0()
            render()
            return

        if state["mode"] == "refining_cam0":
            if ci != ref:
                print(f"[INFO] refining_cam0 모드에선 cam0 패널만 클릭하세요. (you clicked {ci})")
                return
            if event.button == 1:
                state["current"]["positive"].append((u, v))
            elif event.button == 3:
                state["current"]["negative"].append((u, v))
            else:
                return
            run_sam_cam0()
            render()
            return

        if state["mode"] == "confirming":
            if ci == ref:
                print("[INFO] cam0 수정하려면 'r' 키로 refining 모드로 돌아가세요.")
                return
            if event.button == 1:
                refine_other_cam(ci, u, v, positive=True)
            elif event.button == 3:
                refine_other_cam(ci, u, v, positive=False)
            else:
                return
            render()
            return

    def on_key(event):
        key = (event.key or "").lower()
        if state["mode"] == "refining_cam0":
            if key == "enter":
                propagate_to_others()
                state["mode"] = "confirming"
                render()
            elif key == "escape":
                state["mode"] = "idle"
                state["current"] = None
                render()
        elif state["mode"] == "confirming":
            if key == "y":
                save_current()
                state["mode"] = "idle"
                state["current"] = None
                render()
            elif key == "n":
                state["mode"] = "idle"
                state["current"] = None
                render()
            elif key == "r":
                state["mode"] = "refining_cam0"
                cur = state["current"]
                for ci in cam_ids:
                    if ci != ref:
                        cur["masks"].pop(ci, None)
                cur["positive_extra"].clear()
                cur["negative_extra"].clear()
                cur["boxes"].clear()
                cur["centers"].clear()
                render()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()
    print("[DONE]")


if __name__ == "__main__":
    main()
