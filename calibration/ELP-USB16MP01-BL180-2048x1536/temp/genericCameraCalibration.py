# 실행 - python3 genericCameraCalibration.py --config ./generic_camera_calibration_config.yaml

import cv2
import yaml
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import os
import glob
import math
import random
import json
import pandas as pd
import argparse
import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--visualize', action="store_true")
args = parser.parse_args()

def display_images(images, max_num=8):
    images = images[:max_num]
    N = len(images)
    nrows, ncols = (N // 4, 4) if N % 4 == 0 else (N // 4 + 1, 4)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axes = np.atleast_1d(axes).flatten()
    for idx, image in enumerate(images):
        # BGR -> RGB로 보여주기만 변환
        show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        axes[idx].imshow(show)
        axes[idx].axis('off')
    for k in range(idx + 1, len(axes)):
        axes[k].axis('off')
    plt.tight_layout()
    plt.show()

def save_images(images, image_names, reprojection_errors, save_path, color_convert=False, extension='png', log=False):
    os.makedirs(save_path, exist_ok=True)
    for img, name, error in zip(images, image_names, reprojection_errors):
        name = name.rstrip(".png").rstrip(".jpg")
        full_path = os.path.join(save_path, f"{name}_{str(round(float(error), 4))}.{extension}")
        out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if (color_convert and img.ndim == 3) else img
        cv2.imwrite(full_path, out)
        if log: print(f"Saved: {full_path}")

def main():
    ######################### Step 1. Calibration Setting #########################
    print("\nStep 1. Calibration Setting\n")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    pprint.pprint(config)
    camera_name = config["camera_name"]
    base_path = config["base_path"]
    square_size = float(config["square_size"])
    HFoV = float(config["HFoV"])
    VFoV = float(config["VFoV"])
    num_checkboard_width_intersection = int(config["num_checkboard_width_intersection"])
    num_checkboard_height_intersection = int(config["num_checkboard_height_intersection"])
    extrinsic = bool(config["extrinsic"])

    # 입력 경로 확인
    intrinsic_dir = os.path.join(base_path, "Intrinsic")
    origin_dir = os.path.join(intrinsic_dir, "ORIGIN")
    assert os.path.exists(intrinsic_dir), f"Not found: {intrinsic_dir}"
    assert os.path.exists(origin_dir), f"Not found: {origin_dir}"

    if extrinsic:
        extr_dir = os.path.join(base_path, "Extrinsic")
        assert os.path.exists(extr_dir), f"Not found: {extr_dir}"
        assert os.path.exists(os.path.join(extr_dir, f"{camera_name}_EXTRINSIC.png"))
        assert os.path.exists(os.path.join(extr_dir, camera_name))
        assert os.path.exists(os.path.join(extr_dir, camera_name, "points.csv"))

    # 서브픽셀 종료 조건
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # 보정 플래그 (비트 OR) + 초기 K 사용
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
        | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )

    _img_shape = None

    # 이미지 가져오기
    image_paths = glob.glob(os.path.join(origin_dir, "*.png"))
    image_paths.sort()
    print(">>> The number of images to get intrinsic parameter: ", len(image_paths))
    assert len(image_paths) > 0, f"No images found in: {origin_dir}"

    # 첫 장 크기
    img0_bgr = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    assert img0_bgr is not None, f"Failed to read: {image_paths[0]}"
    image_height, image_width = img0_bgr.shape[:2]
    print(f">>> WIDTH: {image_width}, HEIGHT: {image_height}")

    ######################### Step 2. Setting Intrinsic Checkboard Images #########################
    print("\nStep 2. Setting Intrinsic Checkboard Images\n")
    objpoints, imgpoints = [], []
    valid_image_names, valid_images, valid_visual_images = [], [], []

    # 체커보드 탐지 플래그
    cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

    for image_path in image_paths:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None: continue

        if _img_shape is None:
            _img_shape = img_bgr.shape[:2]
        else:
            assert _img_shape == img_bgr.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        is_found = False

        # 다양한 크기로 시도 (가려짐/프레이밍 대비)
        for dx in [0, -1, -2, -3, -4, -5]:
            for dy in [0, -1, -2, -3, -4, -5]:
                checkboard = (num_checkboard_width_intersection + dx,
                              num_checkboard_height_intersection + dy)
                if checkboard[0] < 2 or checkboard[1] < 2:
                    continue

                objp = np.zeros((1, checkboard[0]*checkboard[1], 3), np.float32)
                objp[0, :, :2] = np.mgrid[0:checkboard[0], 0:checkboard[1]].T.reshape(-1, 2)
                objp *= square_size

                ret, corners = cv2.findChessboardCorners(gray, checkboard, cb_flags)
                if ret:
                    is_found = True
                    corners_pixel = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), subpix_criteria)

                    objpoints.append(objp)
                    imgpoints.append(corners_pixel)        # ★ 정밀화된 코너 사용

                    valid_images.append(img_bgr.copy())    # BGR 원본 저장(리프로젝션 그릴 대상)
                    visual_img = cv2.drawChessboardCorners(img_bgr.copy(), checkboard, corners_pixel, ret)
                    valid_visual_images.append(visual_img) # 코너 시각화 이미지
                    valid_image_names.append(os.path.basename(image_path))
                    break
            if is_found: break

        # 실패 프레임 삭제하지 않음(데이터 보존)
        # if not is_found: os.remove(image_path)

    # 사본 보관
    objpoints_copy = objpoints.copy()
    imgpoints_copy = imgpoints.copy()
    image_paths_copy = image_paths.copy()
    valid_images_copy = valid_images.copy()
    valid_visual_images_copy = valid_visual_images.copy()

    if args.visualize and len(valid_visual_images) > 0:
        display_images(valid_visual_images)

    ######################### Step 4. Calibrate Intrinsic #########################
    print("\nStep 4. Calibrate Intrinsic & Extrinsic Parameters\n")
    objpoints = objpoints_copy.copy()
    imgpoints = imgpoints_copy.copy()
    image_paths = image_paths_copy.copy()
    valid_images = valid_images_copy.copy()
    valid_visual_images = valid_visual_images_copy.copy()

    N_OK = len(objpoints)
    assert N_OK >= 8, f"Too few valid boards: {N_OK} (need >= 8)"

    indices = list(range(N_OK))
    random.shuffle(indices)
    objpoints = [objpoints[i] for i in indices]
    imgpoints = [imgpoints[i] for i in indices]
    image_paths = [image_paths[i] for i in indices]
    valid_images = [valid_images[i] for i in indices]
    valid_visual_images = [valid_visual_images[i] for i in indices]

    # 초기 내부 파라미터
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    init_fx = image_width  / (2 * math.tan(math.radians(HFoV/2 - 10)))
    init_fy = image_height / (2 * math.tan(math.radians(VFoV/2 - 10)))
    init_cx = image_width  / 2.0
    init_cy = image_height / 2.0

    K = np.zeros((3, 3), dtype=np.float64)
    K[0,0] = init_fx
    K[1,1] = init_fy
    K[0,2] = init_cx
    K[1,2] = init_cy
    K[2,2] = 1.0

    while True:
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                (image_width, image_height),
                K, D, rvecs, tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
            break
        except cv2.error as err:
            msg = str(err)
            # 에러 메시지에서 idx 파싱 시도
            idx = None
            try:
                idx = int(msg.split('array ')[1][0])
            except Exception:
                idx = len(objpoints) - 1  # fallback: 마지막 프레임 제거

            if idx < 0 or idx >= len(objpoints):
                print("Bad frame index parsed; aborting. Message:", msg)
                break

            # 문제 프레임 제거 후 재시도
            objpoints.pop(idx); imgpoints.pop(idx); image_paths.pop(idx)
            valid_images.pop(idx); valid_visual_images.pop(idx)
            rvecs.pop(idx); tvecs.pop(idx)
            print(f"Removed ill-conditioned image {idx}. Retry... {len(objpoints)} images remain")

            if len(objpoints) < 8:
                raise RuntimeError("Too few frames remain for a stable calibration.")

    print(">>> Found {} valid images for calibration".format(len(objpoints)))
    print(">>> DIM = ({}, {})".format(image_width, image_height))
    print(">>> K = \n", K)
    print(">>> D = \n", D)

    ######################### Step 5. Reprojection Quality #########################
    print("\nStep 5. Check Intrinsic Parameter Quality with Reprojection.\n")
    reprojection_errors = []
    for i in range(len(objpoints)):
        reproj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        reproj = reproj[0]
        err = cv2.norm(imgpoints[i][:, 0, :], reproj, cv2.NORM_L2) / len(reproj)
        reprojection_errors.append(float(err))

        # 리프로젝션 점을 BGR 이미지에 그리기
        for pt in reproj:
            cv2.circle(valid_images[i], (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

    total_err = sum(reprojection_errors) / max(1, len(reprojection_errors))
    print("Total error: ", total_err)

    if args.visualize and len(valid_images) > 0:
        display_images(valid_images)

    # 리프로젝션 점이 그려진 이미지를 저장
    save_images(
        valid_images,            # ★ 리프로젝션을 그린 이미지
        valid_image_names,
        reprojection_errors,
        os.path.join(intrinsic_dir, "REPROJECTION"),
        color_convert=False      # ★ BGR 그대로 저장
    )

    ######################### Step 6. Save Intrinsic #########################
    print("\nStep 6. Save Intrinsic Parameters.\n")
    intrinsic_result = {
        "intrinsic": np.round(K, 5).reshape(-1).tolist(),
        "distortion": [1.0] + np.round(D, 5).reshape(-1).tolist(),
        "reprojection_error": round(total_err, 5),
        "image_size": [int(image_width), int(image_height)],
    }
    out_json = os.path.join(intrinsic_dir, "intrinsic.json")
    with open(out_json, "w") as f:
        json.dump(intrinsic_result, f, indent=4, separators=(',', ': '))
    print(f">>> saved to {out_json}")

if __name__ == "__main__":
    main()
