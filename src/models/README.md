# Models

학습된 segmentation weights를 여기에 두면 `src/pose_pipeline.py`가 자동 탐색합니다.

권장 파일명:

- `src/models/blocks_yolov8n_seg.pt`

직접 지정도 가능합니다.

```bash
python src/pose_pipeline.py \
  --seg_backend yolo \
  --seg_model src/models/blocks_yolov8n_seg.pt \
  --seg_device cpu
```
