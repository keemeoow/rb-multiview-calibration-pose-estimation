# Multi-view RGB-D 6D Pose Estimation Pipeline

3대 RealSense RGB-D 카메라로 촬영한 멀티뷰 이미지에서, GLB 레퍼런스 모델을 가지는
임의 물체의 **6DoF 포즈 + 스케일**을 추정합니다.

핵심은 **profile JSON 한 파일**로 한 물체의 SAM 마스크 + 포즈 추정 동작을 모두
정의하는 것이며, 어떤 물체든 profile을 추가하면 동일한 코드가 적용됩니다.

---

## 0. 전체 흐름

```
[1] 멀티뷰 RGB-D 촬영        →   src/data*/object_capture/cam{0,1,2}/
       Obj_Step1_capture_rgbd_3cam.py

[2] 레퍼런스 3D 모델 추출      →   src/data*/<obj>.glb     (외부, SMA3D 등)
       (수동)

[3] 물체 profile JSON 작성    →   src/configs/objects/<obj>.json

[4] 파이프라인 실행            →   src/output/pipeline_<ts>_frame_<id>/
       run_pipeline.py
       (SAM mask → ICP/render-compare pose → posed GLB + viz)
```

---

## 1. 디렉토리 구조

```
src/
├── pipeline_core.py            # 모든 SAM + pose 로직 (라이브러리)
├── run_pipeline.py             # CLI 진입점 (메인)
├── run_knife_pipeline.py       # legacy thin wrapper (호환용)
├── pose_pipeline.py            # 기초 유틸 (load_calibration / load_frame /
│                                  normalize_glb / estimate_table_plane)
├── Obj_Step1_capture_rgbd_3cam.py  # 데이터 캡처
├── configs/objects/
│   ├── README.md                  # profile 옵션 상세 + 사용 예시
│   ├── object_001.json            # 빨강 아치
│   ├── object_002.json            # 노랑 실린더
│   ├── object_003.json            # 곤색 직사각형 (anisotropic + horizontal_constrain)
│   ├── object_004.json            # 민트 실린더
│   └── knife.json                 # 노란 박스커터 (multicolor + lying_flat)
└── weights/
    └── mobile_sam.pt              # MobileSAM ViT-T 가중치
```

데이터:
```
src/data/                          # 4-블록 데이터셋
├── object_capture/cam{0,1,2}/rgb_*.jpg, depth_*.png
├── intrinsics/cam{0,1,2}.npz
├── cube_session_01/calib_out_cube/T_C0_C{1,2}.npy
└── object_001.glb ... object_004.glb

src/data_knife/                    # knife 데이터셋
├── object_capture/cam{0,1,2}/rgb_*.jpg, depth_*.png
├── _intrinsics/cam{0,1,2}.npz
├── cube_session_01/calib_out_cube/T_C0_C{1,2}.npy
└── reference_knife.glb
```

---

## 2. Step 1 — RGB-D 캡처

### 코드
- 3대 RealSense를 동시에 열고 RGB+Depth 획득
- `device_map.json`이 있으면 `cam0/1/2` 인덱스 고정 매핑
- 키 입력으로 저장 제어:
  - `SPACE`: 1프레임 저장
  - `s`: 연속 저장 ON/OFF
  - `ESC` / `q`: 종료

### 실행

```bash
python src/Obj_Step1_capture_rgbd_3cam.py --save_dir ./data/object_capture
# 해상도/FPS/intrinsic 경로 지정
python src/Obj_Step1_capture_rgbd_3cam.py \
  --save_dir ./data/object_capture \
  --intrinsics_dir ./data/_intrinsics \
  --fps 15 --width 640 --height 480
```

### 결과
- `<save_dir>/cam{0,1,2}/rgb_NNNNNN.jpg`
- `<save_dir>/cam{0,1,2}/depth_NNNNNN.png`

---

## 3. Step 2 — 레퍼런스 3D 모델 (외부)

SMA3D 등 외부 도구로 GLB/PLY 추출 후 `src/data*/<obj>.glb`에 배치합니다.
좌표는 OpenCV cam0 기준(`X-right, Y-down, Z-forward`)으로 정합됩니다.

---

## 4. Step 3 — 파이프라인 실행 (`run_pipeline.py`)

### 호출 방식 3가지

#### A. profile JSON 1개 (단일 물체)

```bash
python3 src/run_pipeline.py \
  --config src/configs/objects/knife.json \
  --data_dir src/data_knife --intr_dir src/data_knife/_intrinsics \
  --frame_id 000004
```

#### B. profile 여러 개 (콤마 또는 폴더)

```bash
# 콤마 구분
python3 src/run_pipeline.py \
  --config src/configs/objects/object_001.json,src/configs/objects/object_002.json,src/configs/objects/object_003.json,src/configs/objects/object_004.json \
  --data_dir src/data --intr_dir src/intrinsics --frame_id 000001

# 폴더 안 모든 JSON
python3 src/run_pipeline.py \
  --config_dir src/configs/objects \
  --data_dir src/data --intr_dir src/intrinsics --frame_id 000001
```

#### C. config 없이 ad-hoc CLI (auto-detect)

```bash
python3 src/run_pipeline.py \
  --glb path/to/new.glb \
  --hue_ref 60 --hue_radius 15 --multicolor \
  --init_orientation lying_flat \
  --data_dir src/data --intr_dir src/intrinsics --frame_id 000000
```

`--glb` 만 주면 GLB extent로 `init_orientation` / `symmetry` / `scale_range`를
자동 추정 (`auto_detect_profile`). `--hue_ref` / `--multicolor` 등으로 부분 override.

### Multi-frame batch

`--frame_id`에 여러 형태 지원:

| 형태 | 의미 |
|------|------|
| `000004` | 단일 |
| `000000,000003,000005` | 콤마 리스트 |
| `0-5` | 범위 (000000~000005) |
| `all` | `data_dir/object_capture/cam0/rgb_*.jpg` 전체 |

```bash
# 0-2번 프레임 모두 처리
python3 src/run_pipeline.py --config src/configs/objects/knife.json \
  --data_dir src/data_knife --intr_dir src/data_knife/_intrinsics \
  --frame_id 0-2
```

multi-frame 시 MobileSAM 가중치는 1회만 로드되어 재사용됩니다.

### 결과

```
src/output/pipeline_<ts>_frame_<id>/
├── sam_masks/<frame_id>/
│   ├── <obj>_cam{0,1,2}.png        # SAM mask 이진 이미지
│   └── comparison.png               # raw + 모든 마스크 overlay (3-cam grid)
└── pose/<frame_id>/
    ├── pose_<obj>.json              # 포즈 (position/quat/euler/scale/fitness)
    ├── <obj>_posed.glb              # OpenCV 좌표계 posed GLB
    ├── <obj>_posed_isaac.glb        # Isaac Sim 좌표계 posed GLB
    ├── comparison.png               # raw + 모든 GLB silhouette (3-cam grid)
    ├── comparison_<obj>.png         # 단일 물체 + 화살표 + 색상 텍스트박스
    └── summary.json                 # 모든 물체 포즈 요약
```

---

## 5. Object profile JSON (한 물체 = 한 파일)

자세한 옵션 표는 [`src/configs/objects/README.md`](src/configs/objects/README.md) 참조.

### 핵심 개념

| 영역 | 옵션 | 의미 |
|------|------|------|
| `color_prior` | `enabled`, `hue_ref`, `s_min`, `v_min` | HSV 색상 seed |
| `multicolor` | true/false | true면 SAM 후 색상 필터링 비활성 (knife처럼 multi-color 물체) |
| `sam.bbox_pad_ratio` | 0.0~0.5 | own bbox 확장 (multicolor면 0.30 권장) |
| `sam.prompt_strategy` | `centroid` / `color_axis_3pt` / `cylinder_axis` / `mask_skeleton` | SAM 점 prompt (mask_skeleton: distance-transform medial axis 3점, 비대칭 / 휘어진 형태에 유용) |
| `sam.bbox_combine` | `union` (블록) / `intersect` (knife) | own ∪ proj 또는 own ∩ proj |
| `sam.auto_refine` | `full` / `extent_only` / `off` | SAM 후 반복 정제 강도 |
| `shape.symmetry` | `none` / `yaw` | yaw → 실린더 (yaw_steps=1) |
| `shape.init_orientation` | `auto` / `upright` / `lying_flat` | 모델 어떤 축을 table normal에 정렬 |
| `shape.anisotropic_scale` | true/false | true면 render_compare가 7DOF (x/y/z 독립 스케일) Nelder-Mead 사용 |
| `shape.horizontal_constrain` | true/false | navy 윗면 보강 등 |
| `pose.method` | `icp_fitness` / `render_compare` | 단순 ICP vs silhouette IoU + Nelder-Mead |
| `pose.render_topk` | int | render_compare에서 coarse top-K를 fine 단계로 |

### 새 물체 추가 (가장 빠른 방법)

```bash
# 1) 가장 비슷한 profile 복사
cp src/configs/objects/knife.json src/configs/objects/myobj.json

# 2) 편집: glb 경로, hue_ref, s_min, v_min, multicolor, init_orientation

# 3) 실행
python3 src/run_pipeline.py \
  --config src/configs/objects/myobj.json \
  --data_dir src/data_my --intr_dir src/data_my/_intrinsics \
  --frame_id 000000
```

---

## 6. 파이프라인 내부 동작

### SAM mask (per object × per cam)

1. **HSV color seed**: profile의 hue/sat/val 임계로 candidate들 추출
2. **3D-size filter**: 각 candidate를 depth로 backproject → GLB extent 대비 크기 매칭 점수
3. **Anchor 결정**: 최고 점수 cam의 3D centroid를 anchor로 (default: cam1 우선)
4. **bbox 결합**: own bbox ∪/∩ GLB-projected bbox (`bbox_combine`)
5. **SAM**: bbox + prompt point(들)로 MobileSAM 호출
6. **post-color filter** (optional): SAM 결과를 relaxed color mask와 AND
7. **auto-refine**: 3D extent + 색상 균질성 + compactness 점수 기반 반복 정제
   (close+largest / fill_holes / hull_fill / halo_strip / open / grabcut /
   erode / dilate 중 점수 최고 액션 채택, 초기 영역 60% 미만으로는 안 줄임)

### Pose 추정

**ICP fitness** (`pose.method = "icp_fitness"`):
- 마스크 → 3D 포인트 backproject + table 위만 통과
- `init_orientation`에 따라 GLB 가장 짧은/긴 축을 table normal에 정렬
- yaw 후보(symmetry='yaw'면 1개) × flip(±) 그리드 → ICP 후 fitness−8·rmse 최고 선택

**render_compare** (`pose.method = "render_compare"`, 4-블록 권장):
- Coarse: 24개 orientation × 2 flip → silhouette IoU 평가, top-K 선별
- 각 후보에 ICP 1회 + Nelder-Mead (5 DOF: dx, dy, dz, dyaw, dscale_log) 리파인
- Loss = 1 − cam-weighted silhouette IoU

테이블 평면은 RANSAC + 부호 보정(물체 평균이 plane 위쪽에 와야 함)으로 추정.

### 좌표계
- 기본: OpenCV cam0 기준 (`X-right, Y-down, Z-forward`)
- `<obj>_posed_isaac.glb`는 `T_ISAAC_CV` 변환 적용한 Isaac Sim 좌표계

---

## 7. 의존성 그래프

```
pose_pipeline.py
    ↑ (import)
pipeline_core.py        ← profile schema + SAM + pose 모두 포함
    ↑ (import)
    ├── run_pipeline.py        # 메인 CLI
    └── run_knife_pipeline.py  # legacy wrapper (knife.json 기본)
```

`pipeline_core.py`는 자체 포함 라이브러리로, 외부 의존은 `pose_pipeline.py`와
`mobile_sam`만 있습니다.

---

## 8. 한 번에 실행하는 권장 순서

```bash
# 1) RGB-D 촬영
python3 src/Obj_Step1_capture_rgbd_3cam.py --save_dir src/data_my/object_capture

# 2) SMA3D 등에서 3D 모델 추출 후 src/data_my/myobj.glb 배치 (수동)

# 3) profile 작성
cp src/configs/objects/knife.json src/configs/objects/myobj.json
# (편집)

# 4) 파이프라인 실행 (frame 0-5 일괄)
python3 src/run_pipeline.py \
  --config src/configs/objects/myobj.json \
  --data_dir src/data_my --intr_dir src/data_my/_intrinsics \
  --frame_id 0-5
```

---

## 9. 확장 포인트

- **`prompt_strategy`**: 현재 4종 (centroid / color_axis_3pt / cylinder_axis /
  mask_skeleton). 향후 plug-in 추가 가능 (text_prompt 등).
- **`auto_refine` 액션**: 현재 9종 (close+largest, fill_holes, hull_fill,
  halo_strip4/6, open5/7, grabcut, edge_snap, erode, dilate). 모두 score 기반
  자동 선택. 추가 가능 (`split_yaw` 등).
- 카메라 수: `pose_pipeline.load_calibration` / `load_frame` 의 `num_cams`
  파라미터로 자동 감지 또는 명시 (default: `intrinsics_dir/cam*.npz` 카운트)

## 10. method 선택 가이드

| 상황 | 권장 |
|------|------|
| 단일 물체, 비대칭, isotropic 스케일 (knife) | `pose.method = "icp_fitness"` (빠름) |
| 점대칭 강한 블록 / 정밀 정렬 필요 | `pose.method = "render_compare"` |
| GLB 와 실제 비율이 axis 별로 다름 (제조 변형 등) | `pose.method = "render_compare"` + `shape.anisotropic_scale = true` (7DOF) |
| 실린더 (yaw 대칭) | `shape.symmetry = "yaw"` (yaw 후보 1개로 단축) |
| 누워있는 자세 (긴 축이 짧은 축의 3배 이상) | `shape.init_orientation = "lying_flat"` |
