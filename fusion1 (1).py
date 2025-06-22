import numpy as np
import cv2
import os
import open3d as o3d
from scipy.spatial.transform import Rotation as R



Camera = np.array([
    [604.359, 0.0, 317.891],
    [0.0, 625.035, 243.115],
    [0.0, 0.0, 1.0]
])

coefficents = np.array([0.144, -1.353, 0.007, -0.0006, 5.304])


BOARD_SIZE = (6, 9)  
SQUARE_SIZE = 0.025       

BASE_DIR = r'C:\Users\Natalia\Downloads\Telegram Desktop\new\new'
cloud_path = os.path.join(BASE_DIR, 'cloud1.pcd')
image_path = os.path.join(BASE_DIR, 'new', 'chess_000.png')


def cluster_point_cloud(pcd, eps=0.03, min_points=20):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels


#Вычисляем матрицу преобразования доска->лидар
def find_matrix_to_lidar(points, labels):
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return None
    
    board_label = unique_labels[np.argmax(counts)]
    board_points = points[labels == board_label]
    center = np.mean(board_points, axis=0)
    cov_matrix = np.cov(board_points.T)
    temp, eigenvectors = np.linalg.eigh(cov_matrix)
    board_normal = eigenvectors[:, 0]
    
    # Корректируем направление нормали
    if np.dot(board_normal, center) >= 0:
        board_normal = -board_normal
    
    z_axis = board_normal / np.linalg.norm(board_normal)
    
    
    if abs(z_axis[0]) > 0.9:
        helper = np.array([0, 1, 0])
    else:
        helper = np.array([1, 0, 0])
        
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    T_board_to_lidar = np.eye(4)
    T_board_to_lidar[:3, :3] = rotation_matrix
    T_board_to_lidar[:3, 3] = center
    return T_board_to_lidar



def detect_board_on_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
    
    if not ret:
        return None
    
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    ret, rvec, tvec = cv2.solvePnP(objp, corners_refined, Camera, coefficents)
    
    if not ret:
        return None
    
    R_mat, _ = cv2.Rodrigues(rvec)
    T_board_cam = np.eye(4)
    T_board_cam[:3, :3] = R_mat
    T_board_cam[:3, 3] = tvec.flatten()
    return T_board_cam


#Вычисляем матрицу калибровки T_cam_lidar
def calibration_matrix(T_board_cam, T_board_lidar):
    T_lidar_board = np.linalg.inv(T_board_lidar)
    T_cam_lidar = T_board_cam @ T_lidar_board
    return T_cam_lidar

if __name__ == '__main__':
    
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения!")
        exit()
    
    
    pcd = o3d.io.read_point_cloud(cloud_path)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("Ошибка загрузки облака точек!")
        exit()
    
    # Ищем шахматную доску на изображении
    T_board_cam = detect_board_on_image(img)
    if T_board_cam is None:
        print("Не удалось найти шахматную доску на изображении!")
        exit()
    
    print("=" * 50)
    print("Матрица преобразования доска->камера:")
    print("=" * 50)
    print(T_board_cam)
    
    
    labels = cluster_point_cloud(pcd)
    
    
    T_board_lidar = find_matrix_to_lidar(points, labels)
    if T_board_lidar is None:
        print("Не удалось найти шахматную доску в облаке точек!")
        exit()
    
    print("\n" + "=" * 50)
    print("Матрица преобразования доска->лидар")
    print("=" * 50)
    print(T_board_lidar)
    
   
    T_cam_lidar = calibration_matrix(T_board_cam, T_board_lidar)
    
    print("\n" + "=" * 50)
    print("Матрица калибровки лидар->камера без поворота")
    print("=" * 50)
    print(T_cam_lidar)

    # Матрицы поворотов
    theta = np.radians(180)  
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
   
    R_x_180 = np.array([
        [1, 0, 0, 0],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [0, 0, 0, 1]
    ])
    
    delta = np.radians(-90)  
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    
    
    R_z_minus90 = np.array([
        [cos_delta, -sin_delta, 0, 0],
        [sin_delta, cos_delta, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Повороты
    T_board_to_lidar_rotated1 = T_board_lidar @ R_x_180
    T_board_to_lidar_rotated = T_board_to_lidar_rotated1 @ R_z_minus90
    
    print("\n" + "=" * 50)
    print("Повернутая матрица преобразования доска->лидар")
    print("=" * 50)
    print(T_board_to_lidar_rotated)
    
    # Итоговая матрица лидар->камера
    T_lidar_to_cam = T_board_cam @ np.linalg.inv(T_board_to_lidar_rotated)
    
    
    rotation_matrix = T_lidar_to_cam[:3, :3]
    det = np.linalg.det(rotation_matrix)
    
    
    camera_matrix_precise = np.array([
        [604.3592113438442, 0.0, 317.8905811942291],
        [0.0, 625.0352102165691, 243.11461937048753],
        [0.0, 0.0, 1.0]
    ])
    
    dist_coeffs_precise = np.array([
        0.14397408976529047, 
        -1.353220164683342,
        0.007119382000046576,
        -0.0006007964675851499,
        5.303894143339311
    ])
    
    
    np.savez(
        r'C:\Users\Natalia\Downloads\extrinsics_lidar_cam.npz',
        T_lidar_to_cam=T_lidar_to_cam,
        camera_matrix=camera_matrix_precise,
        dist_coeffs=dist_coeffs_precise
    )
    
    print("\n" + "=" * 50)
    print("Итоговая матрица преобразования лидар->камера")
    print("=" * 50)
    print(T_lidar_to_cam)  
    print("\nРезультаты сохранены в файл 'extrinsics_lidar_cam.npz'")