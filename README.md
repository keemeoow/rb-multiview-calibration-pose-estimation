# Object Capture → SMA3D → Pose Estimation Pipeline (src)

이 문서는 아래 4개 스크립트를 기준으로 파이프라인을 정리합니다.

- `Obj_Step1_capture_rgbd_3cam.py`
- `Obj_Step2_multiview_pose_estimation.py`
- `Obj_Step3_visualize_pose_result.py`
- `Obj_Step22_Pose_Estimation_pipeline_result.py`

---

## 0. 전체 파이프라인

1. 물체 RGB-D 촬영 (3대 RealSense)
2. SMA3D 웹사이트에서 물체 3D 모델 추출 (`.glb` 또는 `.ply`)
3. 멀티뷰 포즈 추정 (cam0 기준 6DoF)
4. 결과 시각화/디버그 시각화

권장 데이터 흐름:

```text
src3/data/object_capture/cam{0,1,2}/rgb_*.jpg, depth_*.png
  + src3/data/_intrinsics/cam{0,1,2}.npz
  + src3/data/cube_session_01/calib_out_cube/T_C0_C1.npy, T_C0_C2.npy
  + src3/data/reference_knife.glb (or reference_knife.ply)
```

---

## 1) `Obj_Step1_capture_rgbd_3cam.py`

### 1-1. 코드 설명

- 3대 RealSense를 동시에 열고 RGB+Depth를 획득합니다.
- `device_map.json`이 있으면 카메라 인덱스(`cam0/1/2`)를 고정 매핑합니다.
- 미리보기 창에서 키 입력으로 저장을 제어합니다.
  - `SPACE`: 현재 프레임 1회 저장
  - `s`: 연속 저장 모드 ON/OFF
  - `ESC` 또는 `q`: 종료
- 저장 형식:
  - `camX/rgb_000000.jpg`
  - `camX/depth_000000.png`

### 1-2. 실행 명령어 (모드 포함)

`src3` 기준:

```bash
python Obj_Step1_capture_rgbd_3cam.py --save_dir ./data/object_capture
```

해상도/FPS/인트린식 경로 지정:

```bash
python Obj_Step1_capture_rgbd_3cam.py \
  --save_dir ./data/object_capture \
  --intrinsics_dir ./data/_intrinsics \
  --fps 15 --width 640 --height 480
```

모드:

- 단발 저장 모드: `SPACE`로만 저장
- 연속 저장 모드: `s`를 눌러 토글

### 1-3. 결과값

- `--save_dir` 하위에 `cam0`, `cam1`, `cam2` 폴더 생성
- 각 폴더에 프레임 번호 기준 RGB/Depth 쌍 저장
- 종료 시 총 저장 프레임 수 출력

---

## 2) SMA3D 웹사이트 단계 (외부에서 직접하기)

### 2-1. 코드 연결 설명

- 이 단계는 별도 Python 스크립트가 아니라 외부 웹(SMA3D)에서 3D를 추출하는 단계입니다.
- 추출한 모델을 Step2가 읽을 수 있는 경로에 배치합니다.

권장 파일명/위치:

- `src3/data/reference_knife.glb` (기본)
- 또는 `src3/data/reference_knife.ply` (있으면 Step2에서 우선 사용)

### 2-2. 실행 명령어

- 웹 업로드/추출 단계라 CLI 명령어는 없습니다.

### 2-3. 결과값

- 물체 레퍼런스 3D 모델 (`.glb` 또는 `.ply`)
- 이후 Step2에서 정합 대상(Reference Model)로 사용

---

## 3) `Obj_Step2_multiview_pose_estimation.py`

### 3-1. 코드 설명

핵심 로직:

1. `cam0/1/2` RGB-D + intrinsics + extrinsics 로드
2. 각 카메라 depth를 점군화하고 `cam0` 좌표계로 통합
3. RANSAC으로 테이블 평면 제거
4. DBSCAN으로 클러스터링, 색상 기반(노란색) 타겟 클러스터 우선 선택
5. 레퍼런스 모델(Ply/Glb)을 PCA 축 기반 비균일 스케일링
6. 초기정렬 5후보(PCA 4개 + FPFH 1개) 생성 후 ICP
7. 정밀 ICP + 테이블 접지 보정으로 최종 6DoF 산출
8. 각 카메라로 재투영 이미지 생성

좌표계:

- OpenCV cam0 기준 (`X-right, Y-down, Z-forward`)

### 3-2. 실행 명령어 (모드 포함)

기본 실행:

```bash
python Obj_Step2_multiview_pose_estimation.py
```

프레임 변경:

```bash
python Obj_Step2_multiview_pose_estimation.py --frame_id 000005
```

커스텀 경로 지정:

```bash
python Obj_Step2_multiview_pose_estimation.py \
  --data_dir ./data \
  --output_dir ./output \
  --extrinsics_dir ./data/cube_session_01/calib_out_cube \
  --glb_path ./data/reference_knife.glb
```

시각화 자동 실행 모드 (Step3 자동 호출):

```bash
python Obj_Step2_multiview_pose_estimation.py --visualize
```

주요 옵션:

- `--num_cameras` (기본 3)
- `--voxel_size` (기본 `0.003` m)
- `--visualize`: 포즈 추정 완료 후 `Obj_Step3_visualize_pose_result.py` 자동 실행

### 3-3. 결과값

기본 출력 폴더: `src3/output/`

- `scene_merged.ply`: 3카메라 융합 점군
- `objects_no_table.ply`: 테이블 제거 후 점군
- `alignment_result.ply`: 장면 + 정합된 모델
- `object_pointcloud.ply`: 최종 물체 점군
- `pose_Reference_Matching.npz`: 포즈 수치(translation/rotation/quat/fitness/rmse)
- `object_pose_sim.json`: 시뮬레이터 연동용 JSON (위치/회전/크기/4x4 변환)
- `reprojection_cam0.png`, `reprojection_cam1.png`, `reprojection_cam2.png`: 재투영 검증

---

## 4) `Obj_Step3_visualize_pose_result.py`

### 4-1. 코드 설명

- Step2 실행 후 `output/`에 생성된 결과를 시각화합니다.
- matplotlib 정적 이미지 + Open3D 인터랙티브 뷰어 두 가지를 제공합니다.
- 표시 요소:
  - 물체 포인트클라우드 (정합된 모델, 빨간색)
  - OBB (Oriented Bounding Box, 노란색 와이어프레임)
  - 물체 중심에 XYZ 좌표축 + 각 축 회전값 (Euler XYZ)
  - 회전 호(arc) 표시
  - 크기 치수선 (L/W/H cm)
  - 카메라 0/1/2 위치 + 시선 방향 (cam0에 "ref" 라벨)
- 3가지 뷰 생성: Perspective / Top (XZ) / Side (XY)

### 4-2. 실행 명령어 (모드 포함)

이미지 생성 + Open3D 뷰어:

```bash
python Obj_Step3_visualize_pose_result.py
```

이미지만 생성 (뷰어 생략):

```bash
python Obj_Step3_visualize_pose_result.py --no-viewer
```

참고:

- Step2에서 `--visualize` 플래그를 사용하면 자동으로 호출되므로 별도 실행 불필요
- 단독 실행 시 `output/pose_Reference_Matching.npz`와 `output/object_pointcloud.ply` 필요

### 4-3. 결과값

- `src3/output/pose_visualization.png`: 3뷰 시각화 이미지 (축/회전/크기/카메라)
- Open3D 인터랙티브 뷰어 (마우스로 회전/줌 가능)

---

## 5) `Obj_Step22_Pose_Estimation_pipeline_result.py`

### 5-1. 코드 설명

- Step2 내부 과정을 단계별 그림으로 분해하는 디버그용 스크립트입니다.
- 고정된 데이터 경로/프레임을 사용합니다.
  - `DATA_DIR=src3/data`
  - `frame_id="000003"`
- Step1~Step5 중간 결과를 각각 PNG로 저장합니다.

### 5-2. 실행 명령어

```bash
python Obj_Step22_Pose_Estimation_pipeline_result.py
```

참고:

- 스크립트 내부 import는 `multiview_pose_estimation`와
  `Obj_Step2_multiview_pose_estimation` 둘 다 대응하도록 구성되어 있습니다.

### 5-3. 결과값

출력 폴더: `src3/output/debug/`

- `step1_input_images.png`: 입력 RGB/Depth
- `step2a_per_camera.png`: 카메라별 점군(cam0 기준 변환 후)
- `step2b_merged.png`: 통합 점군
- `step3_table_removal.png`: 테이블 제거 결과
- `step4a_clustering.png`: DBSCAN + 색상 매칭
- `step4b_scaling.png`: 레퍼런스 스케일링
- `step4c_candidates.png`: 초기정렬 5후보 비교
- `step4d_final_icp.png`: 정밀 ICP + 최종 포즈
- `step5_reprojection.png`: 3카메라 재투영 검증

---

## 6) 한 번에 실행하는 권장 순서

`src3`에서:

```bash
# 1) 촬영
python Obj_Step1_capture_rgbd_3cam.py --save_dir ./data/object_capture

# 2) SMA3D 웹에서 3D 추출 후 ./data/reference_knife.glb 로 저장 (수동)

# 3) 포즈 추정 + 시각화 (한 번에)
python Obj_Step2_multiview_pose_estimation.py --frame_id 000003 --visualize

# 4) 디버그 단계별 시각화 (선택)
python Obj_Step22_Pose_Estimation_pipeline_result.py
```
