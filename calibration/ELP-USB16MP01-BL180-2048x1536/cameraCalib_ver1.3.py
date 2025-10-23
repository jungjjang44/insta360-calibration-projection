import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import os
import json  # ✅ 추가

img_path = "frames/frame_0008.jpg"

# ===================== JSON에서 K, D 불러오기 =====================
json_path = "intrinsics_fisheye.json"  # 내부 파라미터 저장된 파일 경로
with open(json_path, "r") as f:
    calib = json.load(f)

K = np.array(calib["camera_matrix"]["data"]).reshape(
    calib["camera_matrix"]["rows"], calib["camera_matrix"]["cols"]
)
D = np.array(calib["distortion_coefficients"]["data"]).reshape(-1, 1)

print("[INFO] Loaded Intrinsics from JSON")
print("K =\n", K)
print("D =", D.ravel())
print()

# ================================================================


def calibrate_camera_fisheye(image_dir='frames', pattern='capture*.jpg',
                             board_size=(10,7), square_size=1.0, visualize=True):
    import glob, cv2, numpy as np, os

    cols, rows = board_size
    objp = np.zeros((1, cols*rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    objpoints, imgpoints = [], []
    images = glob.glob(os.path.join(image_dir, pattern))
    print(f"[INFO] Found {len(images)} images for calibration")

    for idx, f in enumerate(images):
        img  = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ok, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not ok:
            print(f"[WARN] Chessboard not found in: {f}")
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (5,5), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        )

        imgpoints.append(corners)
        objpoints.append(objp)

        if visualize:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners, ok)
            cv2.imshow("Detected Corners", vis)
            print(f"[INFO] {idx+1}/{len(images)} → {os.path.basename(f)}")
            key = cv2.waitKey(300)
            if key == 27:
                break

    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        raise ValueError(f"체커보드 검출된 이미지가 너무 적습니다 ({len(objpoints)}장).")

    h, w = gray.shape[:2]
    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
             cv2.fisheye.CALIB_CHECK_COND |
             cv2.fisheye.CALIB_FIX_SKEW)

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        image_size=(w, h),
        K=K, D=D,
        rvecs=None, tvecs=None,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print("\n[FISHEYE] RMS Error:", rms)
    print("[FISHEYE] K:\n", K)
    print("[FISHEYE] D:", D.ravel())
    return rms, K, D, rvecs, tvecs


# ===================== 왜곡 보정 실행 =====================
def undistort_one_image_fisheye(img_path, K, D, save_path=None, balance=0.0, dim_out=None):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape[:2]
    DIM = (w, h)

    if dim_out is None:
        dim_out = DIM

    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, DIM, np.eye(3), balance=balance, new_size=dim_out
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newK, dim_out, cv2.CV_16SC2
    )
    undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

    if save_path is None:
        base, name = os.path.split(img_path)
        save_path = os.path.join(base, "undist_fisheye_" + name)
    cv2.imwrite(save_path, undist)
    return undist, save_path


# ===================== 메인 루프 =====================
# calibrate_camera_fisheye()

input_dir = "frames_4"
output_dir = os.path.join(input_dir, "undist")
os.makedirs(output_dir, exist_ok=True)

images = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
print(f"[INFO] 총 {len(images)}장의 이미지를 보정합니다.")

for i, img_path in enumerate(images):
    filename = os.path.basename(img_path)
    save_path = os.path.join(output_dir, filename)
    _, out_path = undistort_one_image_fisheye(img_path, K, D, save_path=save_path, balance=0.0)
    print(f"[{i+1}/{len(images)}] saved → {out_path}")

print("saved to:", output_dir)
