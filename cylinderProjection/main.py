import getCameraCylindricalLut as utils
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt

image = cv2.cvtColor(cv2.imread("frames/capture_20251022_163813.jpg", -1), cv2.COLOR_BGR2RGB)
origin_height, origin_width, _ = image.shape
target_height, target_width  = origin_height, origin_width

# JSON 파일 읽기
with open("intrinsics_fisheye.json", "r") as f:
    calib = json.load(f)

# 내부 파라미터 (K) 불러오기
intrinsic = np.array(calib["camera_matrix"]["data"]).reshape(
    calib["camera_matrix"]["rows"], calib["camera_matrix"]["cols"]
)

# 이미지 크기 보정이 필요한 경우 (예: target 크기 변경)
origin_width, origin_height = calib["image_size"]
target_width, target_height = 1920, 1080  # 필요 시 변경

intrinsic[0, :] *= (target_width / origin_width)
intrinsic[1, :] *= (target_height / origin_height)

# 왜곡 계수 (D) 불러오기
distortion = np.array(calib["distortion_coefficients"]["data"])

map_x, map_y = utils.get_camera_cylindrical_spherical_lut(intrinsic, distortion, "cylindrical",
                                                          target_width, target_height, hfov_deg=180, vfov_deg=180,
                                                          roll_degree=0, pitch_degree=0, yaw_degree=0)

new_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

plt.figure(figsize=(10, 5))
plt.imshow(new_image)
plt.axis('off')
plt.title("Cylindrical Projection Result")
plt.show()