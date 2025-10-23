import numpy as np

def get_camera_cylindrical_spherical_lut(
    K, D, conversion_mode, target_width, target_height, hfov_deg, vfov_deg, roll_degree, pitch_degree, yaw_degree):
    '''
    - K : (3, 3) intrinsic matrix
    - D : (5, ) distortion coefficient
    - conversion_mode: "cylindrical", "spherical"
    - target_width, target_height: output image size
    - hfov_deg: 0 ~ 360
    - vfov_deg: 0 ~ 180
    - roll_degree: 0 ~ 360
    - pitch_degree: 0 ~ 360
    - yaw_degree: 0 ~ 360
    '''

    fx = K[0][0]
    skew = K[0][1]
    cx = K[0][2]
    
    fy = K[1][1]        
    cy = K[1][2]
    k0, k1, k2, k3, k4 = 1.0, D[0], D[1], D[2], D[3]

    print(f"fx:{fx}, fy:{fy}, cx:{cx}, cy:{cy}")

    # 원통/구면 투영 시 생성할 azimuth/elevetion 각도 범위
    # 원통/구면 투영 시, azimuth 사용
    # 구면 투영 시, elevation 사용
    hfov=np.deg2rad(hfov_deg)
    vfov=np.deg2rad(vfov_deg)
    
    x_angle = pitch_degree
    y_angle = yaw_degree
    z_angle = roll_degree
    
    # X 축 (Pitch) 회전 행렬 (좌표축 회전) 
    Rx_PASSIVE = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(x_angle)), -np.sin(np.radians(x_angle))],
        [0, np.sin(np.radians(x_angle)), np.cos(np.radians(x_angle))]])
    
    # Y 축 (Yaw) 회전 행렬 (좌표축 회전)
    Ry_PASSIVE = np.array([
        [np.cos(np.radians(y_angle)), 0, np.sin(np.radians(y_angle))],
        [0, 1, 0],
        [-np.sin(np.radians(y_angle)), 0, np.cos(np.radians(y_angle))]])
    
    # Z 축 (Roll) 회전 행렬 (좌표축 회전)
    Rz_PASSIVE = np.array([
        [np.cos(np.radians(z_angle)), -np.sin(np.radians(z_angle)), 0],
        [np.sin(np.radians(z_angle)), np.cos(np.radians(z_angle)), 0],
        [0, 0, 1]])
    
    # X, Y, Z 축 전체 회전을 반영한 회전 행렬 (좌표축 회전)
    # SRC: 어떤 회전이 반영되지 않은 카메라 좌표축
    # TARGET: Roll/Pitch/Yaw 회전이 반영된 카메라 좌표축    
    # new_R_RDF_SRC_RDF_TARGET_PASSIVE: SRC → TARGET의 좌표축 회전
    new_R_RDF_SRC_RDF_TARGET_PASSIVE = Ry_PASSIVE @ Rx_PASSIVE @ Rz_PASSIVE
    # new_R_RDF_SRC_RDF_TARGET_ACTIVE: SRC → TARGET의 좌표 회전
    new_R_RDF_SRC_RDF_TARGET_ACTIVE = new_R_RDF_SRC_RDF_TARGET_PASSIVE.T
    ##############################################################################################################
    
    # 원통/구면 투영 시, normalized → image 로 적용하기 위한 intrinsic 행렬렬
    new_K = np.array([
        [target_width/hfov,       0,                     target_width/2],
        [0,                       target_height/vfov,    target_height/2],
        [0,                       0,                     1]], dtype=np.float32)
    new_K_inv = np.linalg.inv(new_K)
    
    # Create pixel grid and compute a ray for every pixel
    # xv : (target_height, target_width), yv : (target_height, target_width)
    xv, yv = np.meshgrid(range(target_width), range(target_height), indexing='xy')
    
    # p.shape : (3, #target_height, #target_width)
    p = np.stack([xv, yv, np.ones_like(xv)])  # pixel homogeneous coordinates    
    # p.shape : (#target_height, #target_width, 3, 1)    
    p = p.transpose(1, 2, 0)[:, :, :, np.newaxis] # [u, v, 1]
    '''
    p.shape : (H, W, 3, 1)
    p[:, : 0, :] : 0, 1, 2, ..., W-1
    p[:, : 1, :] : 0, 1, 2, ..., H-1    
    p[:, : 2, :] : 1, 1, 1, ..., 1
    '''
    # p_norm.shape : (#target_height, #target_width, 3, 1)
    p_norm = new_K_inv @ p  # r is in normalized coordinate
    
    '''
    p_norm[:, :, 0, :]. phi (azimuthal angle. horizontal) : -hfov/2 ~ hov/2
    p_norm[:, :, 1, :]. theta (elevation angla. vertical) : -vfov/2 ~ vfov/2
    p_norm[:, :, 2, :]. 1.    
    '''

    # x, y, z : cartesian coordinate in camera coordinate system (RDF, Right-Down-Front)
    # hemisphere
    if conversion_mode == "cylindrical":
        # azimuthal angle
        phi = p_norm[:, :, 0, :]
        
        x = np.sin(phi)
        y = p_norm[:, :, 1, :]
        z = np.cos(phi)
        
    elif conversion_mode == "spherical":
        # azimuthal angle
        phi = p_norm[:, :, 0, :]
        # elevation angle
        theta = p_norm[:, :, 1, :] 
        
        x =np.cos(theta)*np.sin(phi) # -1 ~ 1
        y =np.sin(theta) # -1 ~ 1
        z =np.cos(theta)*np.cos(phi) # 0 ~ 1
    else:
        print("wrong conversion_mode: ", conversion_mode)
        exit()
    
    RDF_cartesian = np.zeros(p_norm.shape).astype(np.float32)
    RDF_cartesian[:,:,0,:]=x
    RDF_cartesian[:,:,1,:]=y
    RDF_cartesian[:,:,2,:]=z    
    
    # RDF_rotated_cartesian = Rz @ Ry @ Rx @ RDF_cartesian
    # SRC → TARGET의 좌표 회전울 통하여 생성된 좌표들을 회전함
    RDF_rotated_cartesian = new_R_RDF_SRC_RDF_TARGET_ACTIVE @ RDF_cartesian
            
    # compute incidence angle
    x_un = RDF_rotated_cartesian[:, :, [0], :]
    y_un = RDF_rotated_cartesian[:, :, [1], :]
    z_un = RDF_rotated_cartesian[:, :, [2], :]
    # theta = np.arccos(RDF_rotated_cartesian[:, :, [2], :] / np.linalg.norm(RDF_rotated_cartesian, axis=2, keepdims=True))
    theta = np.arccos(z_un / np.sqrt(x_un**2 + y_un**2 + z_un**2))
    
    mask = theta > np.pi/2
    mask = mask.squeeze(-1).squeeze(-1)
    # project the ray onto the fisheye image according to the fisheye model and intrinsic calibration
    r_dn = k0*theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9
    
    r_un = np.sqrt(x_un**2 + y_un**2)
    
    x_dn = r_dn * x_un / (r_un + 1e-6) # horizontal
    y_dn = r_dn * y_un / (r_un + 1e-6) # vertical    
    
    map_x_origin2new = fx*x_dn[:, :, 0, 0] + cx + skew*y_dn[:, :, 0, 0]
    map_y_origin2new = fy*y_dn[:, :, 0, 0] + cy
    
    DEFAULT_OUT_VALUE = -100
    map_x_origin2new[mask] = DEFAULT_OUT_VALUE
    map_y_origin2new[mask] = DEFAULT_OUT_VALUE
    
    map_x_origin2new = map_x_origin2new.astype(np.float32)
    map_y_origin2new = map_y_origin2new.astype(np.float32)
    return map_x_origin2new, map_y_origin2new