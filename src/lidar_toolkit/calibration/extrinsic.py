"""外参标定：LiDAR↔IMU/Camera 手眼标定、PnP、ICP"""

import numpy as np


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 轴角向量 (Rodrigues)"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if abs(theta) < 1e-10:
        return np.zeros(3)
    k = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * k / (2 * np.sin(theta) + 1e-15)


def _rotvec_to_matrix(r: np.ndarray) -> np.ndarray:
    """轴角向量 → 旋转矩阵 (Rodrigues)"""
    theta = np.linalg.norm(r)
    if theta < 1e-10:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def solve_hand_eye_ax_xb(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """手眼标定 AX = XB, 经典解法 (Tsai-Lenz)。

    Args:
        A: (N,4,4) 传感器 A 的运动 (IMU/Camera)
        B: (N,4,4) 传感器 B 的运动 (LiDAR)
    Returns:
        X: (4,4) 外参矩阵 T_A_to_B
    """
    n = len(A)
    # 提取旋转和平移
    R_a = A[:, :3, :3]
    t_a = A[:, :3, 3]
    R_b = B[:, :3, :3]
    t_b = B[:, :3, 3]

    # 旋转部分: R_a @ R_x = R_x @ R_b
    # 用轴角表示
    alpha = np.zeros((3 * n, 3))
    beta = np.zeros(3 * n)

    for i in range(n):
        ra = _matrix_to_rotvec(R_a[i])
        rb = _matrix_to_rotvec(R_b[i])
        skew_ra = _skew(ra)
        skew_rb = _skew(rb)
        alpha[3 * i : 3 * i + 3] = skew_ra - skew_rb
        beta[3 * i : 3 * i + 3] = ra - rb

    # 最小二乘求解旋转向量
    r_x_vec, _, _, _ = np.linalg.lstsq(alpha, beta, rcond=None)
    r_x_vec = r_x_vec / max(np.linalg.norm(r_x_vec), 1e-15)
    theta = np.linalg.norm(r_x_vec)
    if theta > 1e-10:
        r_x_vec = r_x_vec / theta * np.tan(theta / 2)

    R_x = _rotvec_to_matrix(r_x_vec)

    # 平移部分: (R_a - I) @ t_x = R_x @ t_b - t_a
    C = np.zeros((3 * n, 3))
    d = np.zeros(3 * n)
    for i in range(n):
        C[3 * i : 3 * i + 3] = R_a[i] - np.eye(3)
        d[3 * i : 3 * i + 3] = R_x @ t_b[i] - t_a[i]

    t_x, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    X = np.eye(4)
    X[:3, :3] = R_x
    X[:3, 3] = t_x
    return X


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def solve_pnp_lidar_camera(
    lidar_points: np.ndarray, image_points: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """PnP: LiDAR 点云 + 对应图像点 → 外参 T_cam_to_lidar。

    使用 SVD 最小二乘 (非迭代)，注意这不是完整 PnP，而是已知 3D-2D 对应时
    的线性解法 (DLT)。

    Args:
        lidar_points: (N,3) LiDAR 坐标系下的 3D 点
        image_points: (N,2) 像素坐标
        K: (3,3) 相机内参矩阵
    Returns:
        T_cam_to_lidar: (4,4)
    """
    n = len(lidar_points)
    A = np.zeros((2 * n, 12))

    for i in range(n):
        X, Y, Z = lidar_points[i]
        u, v = image_points[i]
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

    _, _, vh = np.linalg.svd(A)
    P = vh[-1].reshape(3, 4)

    # 分解 P = K [R|t]
    R_t = np.linalg.inv(K) @ P
    R_approx = R_t[:, :3]
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    scale = np.linalg.norm(R_approx) / np.linalg.norm(R)
    t = R_t[:, 3] / scale

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def icp(
    source: np.ndarray,
    target: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """ICP (Iterative Closest Point): 将 source 点云对齐到 target。

    Args:
        source, target: (N,3) 点云
    Returns:
        (R, t): 旋转和平移使 source → target
    """
    src = source.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)

    for _ in range(max_iters):
        # 最近邻关联 (暴力搜索，小规模场景)
        diffs = src[:, None, :] - target[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        indices = np.argmin(dists, axis=1)
        target_matched = target[indices]

        # SVD 求解最优旋转平移
        centroid_src = src.mean(axis=0)
        centroid_tgt = target_matched.mean(axis=0)
        H = (src - centroid_src).T @ (target_matched - centroid_tgt)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = centroid_tgt - R @ centroid_src

        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        if np.linalg.norm(t) < tol:
            break

    return R_total, t_total


class ExtrinsicCalibrator:
    """外参标定器。

    支持三种方法:
    1. 手眼标定 (LiDAR↔IMU): 多帧运动 AX=XB
    2. PnP (LiDAR↔Camera): 已知 3D-2D 对应
    3. ICP (LiDAR↔LiDAR): 点云配准
    """

    @staticmethod
    def calibrate_hand_eye(
        motions_a: list[np.ndarray],
        motions_b: list[np.ndarray],
    ) -> np.ndarray:
        """手眼标定: 需要至少 2 组以上运动 (N≥3 推荐)。

        Args:
            motions_a: 传感器 A 的帧间运动 T_A_prev_to_curr (4,4)
            motions_b: 传感器 B 的帧间运动 T_B_prev_to_curr (4,4)
        Returns:
            T_A_to_B: (4,4)
        """
        A = np.stack(motions_a)
        B = np.stack(motions_b)
        return solve_hand_eye_ax_xb(A, B)

    @staticmethod
    def calibrate_lidar_camera(
        lidar_points: np.ndarray,
        image_points: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> np.ndarray:
        """LiDAR→Camera 外参标定 (PnP)。"""
        return solve_pnp_lidar_camera(lidar_points, image_points, camera_intrinsics)

    @staticmethod
    def calibrate_lidar_lidar(
        source_cloud: np.ndarray,
        target_cloud: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """LiDAR→LiDAR 外参标定 (ICP)。"""
        return icp(source_cloud, target_cloud)
