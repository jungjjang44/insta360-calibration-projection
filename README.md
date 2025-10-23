# insta360-calibration-projection

## 주요 기능

### 1. 카메라 캘리브레이션 (Camera Calibration)

**파일:** `calibration/cameraCalib_ver1.3.py` 
이 모듈은 Insta360 X5 fisheye 카메라의 내부 파라미터(`K`, `D`)를 추정하고, 해당 값을 기반으로 왜곡을 보정합니다.

- **함수:** `calibrate_camera_fisheye(image_dir='frames', pattern='capture*.jpg', board_size=(10,7), square_size=1.0, visualize=True)`
  - 체커보드 패턴을 이용해 어안 카메라의 내부 파라미터를 계산합니다.
  - 계산된 카메라 행렬(`K`)과 왜곡 계수(`D`)를 반환합니다.

- **함수:** `undistort_one_image_fisheye(img, K, D)`
  - 단일 이미지를 fisheye 모델을 기준으로 왜곡 보정합니다.
  - 현재는 **`frame_4` 폴더 내 이미지**를 보정하도록 구성되어 있습니다.

---

### 2. 실린더 투영 (Cylindrical Projection)

**폴더:** `cylinderProjection/` 
이 모듈은 보정된 이미지를 **실린더(cylindrical)** 혹은 **구면(spherical)** 좌표계로 투영하여 
파노라마 또는 환경 맵핑용 영상으로 변환합니다.

- **파일:** `getCameraCylindericalLut.py`
  - 실린더 혹은 구면 투영용 LUT(Look-Up Table)를 계산하는 함수 포함.
  - 핵심 함수: 
    ```python
    get_camera_cylindrical_spherical_lut()
    ```
  - 참고 자료: 
    [Gaussian37 블로그 - 원통/구면 투영 개념](https://gaussian37.github.io/vision-concept-spherical_projection/#%EC%9B%90%ED%86%B5-%ED%88%AC%EC%98%81%EB%B2%95-%EC%A0%81%EC%9A%A9-%EB%B0%A9%EB%B2%95-1)

- **파일:** `main.py`
  - `get_camera_cylindrical_spherical_lut()` 함수를 호출하여 
    이미지 데이터를 실린더 평면으로 투영하는 메인 실행 스크립트.
  - **`intrinsics.json`**과 **`intrinsics_fisheye.json`** 중 하나를 불러와 내부 파라미터를 적용:
    - `intrinsics.json` → 일반 Pinhole 카메라용 
    - `intrinsics_fisheye.json` → Fisheye 카메라용 (왜곡 계수 포함)
