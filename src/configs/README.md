# Object profile JSONs

각 파일은 1 물체의 SAM-mask + pose 동작을 정의합니다. `run_pipeline.py`가 이
파일을 읽어 동작.

## 사용

### A. profile JSON 1개 (단일 물체)

```bash
python3 src/run_pipeline.py \
  --config src/configs/objects/knife.json \
  --data_dir src/data_knife --intr_dir src/data_knife/_intrinsics \
  --frame_id 000004
```

### B. 여러 profile 한꺼번에 (콤마 구분)

```bash
python3 src/run_pipeline.py \
  --config src/configs/objects/object_001.json,src/configs/objects/object_002.json,src/configs/objects/object_003.json,src/configs/objects/object_004.json \
  --data_dir src/data --intr_dir src/intrinsics --frame_id 000001
```

### C. 폴더 안 모든 profile

```bash
python3 src/run_pipeline.py \
  --config_dir src/configs/objects \
  --data_dir src/data --intr_dir src/intrinsics --frame_id 000001
```

(주의: 여러 데이터셋 profile이 섞여 있으면 안 됨. 데이터셋별 폴더 분리 권장)

### D. config 없이 ad-hoc (auto-detect)

```bash
python3 src/run_pipeline.py \
  --glb path/to/new.glb \
  --hue_ref 60 --hue_radius 15 --multicolor \
  --init_orientation lying_flat \
  --data_dir src/data --intr_dir src/intrinsics --frame_id 000000
```

`--glb` 만 주면 GLB extent로 init_orientation/symmetry 자동 추정.

## 새 물체 추가하는 가장 빠른 방법

1. `src/configs/objects/<name>.json` 복사 후 수정
2. 색상 prior 측정 (이미지에서 HSV 따기) → `hue_ref`, `s_min`, `v_min`
3. `multicolor: true` 면 `bbox_pad_ratio: 0.3`, `post_color_intersect: false`
4. 실린더면 `symmetry: "yaw"`, 비대칭이면 `"none"`
5. 누워있는 자세 (긴/짧 비율 > 3) → `init_orientation: "lying_flat"`

## 옵션 의미

| 필드 | 기본 | 설명 |
|------|------|------|
| `color_prior.enabled` | true | HSV color seed 사용 |
| `color_prior.hue_ref` | 0 | OpenCV HSV hue (0–179) |
| `color_prior.hue_radius` | 12 | hue 허용 반경 |
| `color_prior.s_min` `v_min` `v_max` | 100/70/245 | 채도/명도 범위 |
| `multicolor` | false | true면 post-color filter / auto-refine 색상 가중 비활성 |
| `sam.bbox_pad_ratio` | 0.05 | own bbox 확장 비율 (multicolor면 0.30 권장) |
| `sam.prompt_strategy` | centroid | centroid / color_axis_3pt / cylinder_axis / mask_skeleton |
| `sam.post_color_intersect` | true | SAM 결과를 color mask와 AND |
| `sam.auto_refine` | full | full / extent_only / off |
| `sam.reliability_threshold` | 0.30 | 3D-size score 하한 |
| `sam.own_bbox_max_image_ratio` | 0.7 | own bbox이 이미지의 이 비율 넘으면 cross-cam |
| `sam.scale_range_min/max` | 0.20/1.20 | GLB max extent 대비 허용 비율 |
| `sam.bbox_combine` | union | union (own ∪ proj) / intersect (own ∩ proj) |
| `shape.symmetry` | none | none / yaw |
| `shape.init_orientation` | auto | auto / upright / lying_flat |
| `shape.yaw_steps` | 24 | pose ICP yaw 후보 수 |
| `shape.anisotropic_scale` | false | 향후 확장 (현재 ICP는 isotropic) |
| `shape.horizontal_constrain` | false | own bbox 높이 제한 (navy 블록) |
| `pose.icp_voxel` | 0.005 | ICP voxel size |
| `pose.icp_max_iter` | 80 | ICP 반복 횟수 |
| `pose.flip_signs` | [1,-1] | init R_align 부호 후보 |

## 알려진 한계

- **ICP-only pose**: 단일 물체 (knife) 에서 잘 동작하지만, 점대칭이 강한 블록은
  silhouette-IoU 기반 render-and-compare 가 더 정확 → 추후 `pose.method =
  "render_compare"` 추가 예정. 4-블록 정밀 결과는 `pose_per_object_v2.py +
  fit_pose_to_mask_v4.py` 조합 권장.
