#!/usr/bin/env python3
"""Prepare a YOLOv8-seg dataset from RGB captures and Labelme annotations.

Expected annotation format:
- Labelme sidecar JSON next to each RGB image
- Supported sidecar names:
  - rgb_000000.json
  - rgb_000000.labelme.json

Supported labels:
- object_001 / object_002 / object_003 / object_004
- several English/Korean aliases normalized to those class ids

Typical flow:
1. Annotate RGB images with Labelme polygons
2. Run this script
3. Train YOLOv8-seg with the generated data.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional


CLASS_NAMES = [
    "object_001",
    "object_002",
    "object_003",
    "object_004",
]

CLASS_ID_BY_NAME = {name: idx for idx, name in enumerate(CLASS_NAMES)}

LABEL_ALIASES = {
    "object_001": "object_001",
    "red_arch": "object_001",
    "red-arch": "object_001",
    "빨강아치": "object_001",
    "빨강_아치": "object_001",
    "object_002": "object_002",
    "yellow_cylinder": "object_002",
    "yellow-cylinder": "object_002",
    "노랑실린더": "object_002",
    "노랑_실린더": "object_002",
    "object_003": "object_003",
    "navy_block": "object_003",
    "navy-block": "object_003",
    "rect_block": "object_003",
    "곤색직사각형": "object_003",
    "곤색_직사각형": "object_003",
    "object_004": "object_004",
    "mint_cylinder": "object_004",
    "mint-cylinder": "object_004",
    "민트실린더": "object_004",
    "민트_실린더": "object_004",
}


def normalize_label(label: str) -> Optional[str]:
    key = str(label).strip().lower().replace(" ", "_")
    return LABEL_ALIASES.get(key)


def candidate_annotation_paths(image_path: Path) -> Iterable[Path]:
    yield image_path.with_suffix(".json")
    yield image_path.with_name(f"{image_path.stem}.labelme.json")


def load_labelme_json(image_path: Path) -> Optional[dict]:
    for json_path in candidate_annotation_paths(image_path):
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
    return None


def image_hw_from_labelme(data: dict) -> tuple[int, int]:
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Labelme JSON missing imageWidth/imageHeight")
    return width, height


def rectangle_to_polygon(points):
    (x0, y0), (x1, y1) = points[:2]
    xmin, xmax = sorted([x0, x1])
    ymin, ymax = sorted([y0, y1])
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ]


def polygon_to_yolo_row(class_id: int, points, width: int, height: int) -> str:
    coords = []
    for x, y in points:
        x_n = min(max(float(x) / max(width, 1), 0.0), 1.0)
        y_n = min(max(float(y) / max(height, 1), 0.0), 1.0)
        coords.append(f"{x_n:.6f}")
        coords.append(f"{y_n:.6f}")
    return " ".join([str(class_id), *coords])


def convert_labelme_to_yolo_rows(data: dict) -> list[str]:
    width, height = image_hw_from_labelme(data)
    rows = []
    for shape in data.get("shapes", []):
        label = normalize_label(shape.get("label", ""))
        if label is None:
            continue
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "polygon")
        if shape_type == "rectangle":
            points = rectangle_to_polygon(points)
        if len(points) < 3:
            continue
        rows.append(polygon_to_yolo_row(CLASS_ID_BY_NAME[label], points, width, height))
    return rows


def split_name_from_path(image_path: Path, val_stride: int) -> str:
    try:
        frame_id = int(image_path.stem.split("_")[-1])
    except ValueError:
        frame_id = sum(ord(ch) for ch in image_path.stem)
    return "val" if (frame_id % max(val_stride, 2) == 0) else "train"


def ensure_link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)
    else:
        shutil.copy2(src, dst)


def collect_rgb_images(source_dirs: list[Path]) -> list[Path]:
    images = []
    for source_dir in source_dirs:
        images.extend(sorted(source_dir.glob("cam*/rgb_*.jpg")))
    return images


def build_dest_name(image_path: Path) -> str:
    cam = image_path.parent.name
    scene = image_path.parent.parent.name.replace(" ", "_")
    return f"{scene}_{cam}_{image_path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLOv8-seg dataset for block instances")
    parser.add_argument(
        "--source_dir",
        action="append",
        dest="source_dirs",
        default=None,
        help="Capture root containing cam0/cam1/cam2 folders; can be passed multiple times",
    )
    parser.add_argument(
        "--dataset_dir",
        default="src/training/blocks_yolo/dataset",
        help="Output YOLO dataset directory",
    )
    parser.add_argument("--val_stride", type=int, default=5)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()

    source_dirs = [Path(p) for p in (args.source_dirs or [
        "src/data/object_capture_blocks(1)",
        "src/data/object_capture",
    ])]
    dataset_dir = Path(args.dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    unlabeled_dir = dataset_dir / "unlabeled"
    unlabeled_manifest = dataset_dir / "unlabeled_manifest.txt"

    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    images = collect_rgb_images(source_dirs)
    converted = 0
    unlabeled = []
    empty = 0

    for image_path in images:
        data = load_labelme_json(image_path)
        if data is None:
            unlabeled.append(str(image_path))
            continue

        try:
            rows = convert_labelme_to_yolo_rows(data)
        except Exception as exc:
            unlabeled.append(f"{image_path}  # invalid annotation: {exc}")
            continue

        split = split_name_from_path(image_path, args.val_stride)
        dest_name = build_dest_name(image_path)
        image_dst = images_dir / split / dest_name
        label_dst = labels_dir / split / f"{Path(dest_name).stem}.txt"

        ensure_link_or_copy(image_path.resolve(), image_dst, symlink=not args.copy)
        label_dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        converted += 1
        if not rows:
            empty += 1

    unlabeled_manifest.write_text("\n".join(unlabeled) + ("\n" if unlabeled else ""), encoding="utf-8")

    print(f"images_found={len(images)}")
    print(f"images_converted={converted}")
    print(f"empty_label_files={empty}")
    print(f"unlabeled_images={len(unlabeled)}")
    print(f"dataset_dir={dataset_dir}")
    print(f"unlabeled_manifest={unlabeled_manifest}")


if __name__ == "__main__":
    main()
