# Blocks YOLOv8-Seg

이 디렉터리는 블록 4종 전용 `YOLOv8-seg` 학습용 scaffold입니다.

## 클래스

- `object_001`: 빨강 아치
- `object_002`: 노랑 실린더
- `object_003`: 곤색 직사각형
- `object_004`: 민트 실린더

## 권장 워크플로

1. RGB 이미지를 Labelme로 polygon annotation
2. `scripts/prepare_blocks_yolo_dataset.py` 실행
3. `ultralytics`로 `yolov8n-seg` 학습
4. 학습된 weights를 `src/models/blocks_yolov8n_seg.pt`에 배치
5. `src/pose_pipeline.py --seg_backend yolo --seg_device cpu`로 실행

## Labelme 라벨 이름

아래 중 하나로 저장하면 됩니다.

- `object_001`, `red_arch`, `빨강아치`
- `object_002`, `yellow_cylinder`, `노랑실린더`
- `object_003`, `navy_block`, `곤색직사각형`
- `object_004`, `mint_cylinder`, `민트실린더`

## 데이터 준비

예시:

```bash
python scripts/prepare_blocks_yolo_dataset.py \
  --source_dir "src/data/object_capture_blocks(1)" \
  --source_dir "src/data/object_capture"
```

생성 결과:

- `src/training/blocks_yolo/dataset/images/train`
- `src/training/blocks_yolo/dataset/images/val`
- `src/training/blocks_yolo/dataset/labels/train`
- `src/training/blocks_yolo/dataset/labels/val`
- `src/training/blocks_yolo/dataset/unlabeled_manifest.txt`

## 학습 예시

```bash
yolo task=segment mode=train \
  model=yolov8n-seg.pt \
  data=src/training/blocks_yolo/data.yaml \
  epochs=100 imgsz=640 batch=8 device=cpu
```

CPU에서는 느리므로, 학습은 GPU 환경에서 하고 추론만 CPU에서 쓰는 편이 현실적입니다.

## 추론 예시

```bash
python src/pose_pipeline.py \
  --data_dir src/data \
  --intrinsics_dir src/intrinsics \
  --frame_id 000000 \
  --multi \
  --seg_backend yolo \
  --seg_model src/models/blocks_yolov8n_seg.pt \
  --seg_device cpu
```
