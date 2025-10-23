import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os

K = [[607.50474652, 0.0, 957.4755321 ],
     [0.0, 607.8220804 , 541.39410119],
     [0.0, 0.0, 1.0]]
D = [-1.76193729e-01, 2.98670897e-02, -4.91100242e-04, 8.09710773e-05, -2.10814274e-03]
img_path = "frames/frame_0008.jpg"

def calibrate_camera_from_chessboard(image_dir='frames', pattern='capture*.jpg',
                                     board_size=(10,7), square_size=1.0, visualize=True):
    """
    체커보드 이미지를 이용해 카메라 내부 파라미터(K, D)를 계산하는 함수

    Parameters
    ----------
    image_dir : str
        캘리브레이션 이미지 폴더 경로
    pattern : str
        이미지 파일명 패턴 (예: 'GO*.jpg')
    board_size : tuple
        (cols, rows) 내부 교차점 개수
    square_size : float
        체커보드 한 칸의 실제 크기 (mm 단위로 설정 가능)
    visualize : bool
        True일 경우, 코너 검출 결과를 시각화

    Returns
    -------
    ret : float
        RMS reprojection error
    K : np.ndarray
        3x3 카메라 행렬
    D : np.ndarray
        왜곡 계수 (k1, k2, p1, p2, k3)
    rvecs, tvecs : list
        각 이미지의 회전/이동 벡터
    """

    # 3D 체커보드 포인트 정의
    objp = np.zeros((board_size[0]*board_size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2)
    objp *= square_size  # 실제 단위 반영

    objpoints = []  # 3D 포인트
    imgpoints = []  # 2D 포인트

    # 이미지 리스트 읽기
    images = glob.glob(f'{image_dir}/{pattern}')
    print(f"[INFO] Found {len(images)} images")

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)

            if visualize:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, board_size, corners2, ret)
                cv2.imshow('Corners', vis)
                cv2.waitKey(200)

    cv2.destroyAllWindows()

    # 내부 파라미터 계산
    if len(objpoints) < 5:
        raise ValueError(f"체커보드가 검출된 이미지가 너무 적습니다: {len(objpoints)}장")

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n[RESULT] Camera Calibration Completed")
    print(f"RMS Reprojection Error: {ret}")
    print(f"Camera Matrix (K):\n{K}")
    print(f"Distortion Coefficients (D):\n{D.ravel()}")

    # 리프로젝션 에러 계산
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        total_error += error
    mean_error = total_error/len(objpoints)
    print(f"Mean reprojection error: {mean_error:.4f}")

    return ret, K, D, rvecs, tvecs

# 왜곡 보정 이미지 출력

def undistort_one_image(img_path, K, D, save_path=None, alpha=0.0, scale=1.0):
    """
    단일 이미지 왜곡 보정 후 저장/반환

    Parameters
    ----------
    img_path : str
        입력 이미지 경로
    K : (3,3) or list
        카메라 내부행렬
    D : (5,) or (1,5) or list
        왜곡계수 (k1,k2,p1,p2,k3) - 일반 모델
    save_path : str or None
        저장할 경로 (None이면 'undist_' 접두어로 원본 폴더에 저장)
    alpha : float [0~1]
        0: 크롭(검은 영역 최소) / 1: 전체 시야 유지(검은 영역 많아짐)
    scale : float
        출력 이미지 리사이즈 스케일(1.0=원본)

    Returns
    -------
    undist : np.ndarray
        보정된 이미지 (BGR)
    out_path : str
        저장된 파일 경로
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]

    K = np.asarray(K, dtype=np.float64).reshape(3,3)
    D = np.asarray(D, dtype=np.float64).reshape(-1)

    # 새 카메라 행렬 계산 (크롭/시야 보존 정도를 alpha로 제어)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha, (w, h))

    # 맵 생성 후 리맵
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, R=None, newCameraMatrix=newK,
        size=(w, h), m1type=cv2.CV_16SC2
    )
    undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    # 필요 시 스케일 조절
    if scale != 1.0:
        undist = cv2.resize(undist, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 저장 경로
    if save_path is None:
        base, name = os.path.split(img_path)
        out_name = "undist_" + name
        save_path = os.path.join(base, out_name)

    cv2.imwrite(save_path, undist)
    return undist, save_path


# calibrate_camera_from_chessboard()

input_dir="frames_3"
output_dir = os.path.join(input_dir, "undist")
os.makedirs(output_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
print(f"[INFO] 총 {len(images)}장의 이미지를 보정합니다.")

for i, img_path in enumerate(images):
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    _, out_path = undistort_one_image(img_path, K, D, save_path=save_path, alpha=0.0)
    print(f"[{i+1}/{len(images)}] saved → {out_path}")

# _, out_path = undistort_one_image(img_path, K, D, save_path="frames/undist/frame_0008.jpg", alpha=0.0)
print("saved to:", output_dir)