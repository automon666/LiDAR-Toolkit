"""内参标定：LiDAR 通道间角度偏差、距离零偏估计 (平面靶标法)"""

import numpy as np


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """SVD 平面拟合: n·x = d, 返回 (normal, d)。"""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]  # 最小奇异值对应法向量
    if normal[2] < 0:
        normal = -normal  # 法向量朝上
    d = np.dot(normal, centroid)
    return normal, d


def point_to_plane_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """点到平面距离 (有符号): dist = n·p - d"""
    return np.dot(points, normal) - d


class IntrinsicCalibrator:
    """基于平面靶标的 LiDAR 内参标定。

    核心原理：将雷达置于已知平面（如墙面）前方多个位姿，理论上有噪声的点云
    应该落在同一平面上。通过对平面残差建模，估计每个通道的角度偏差和距离零偏。

    模型: θ_corrected[i] = θ_raw[i] + Δθ[i]
          d_corrected[i] = d_raw[i] + d_bias[i]

    Args:
        n_channels: 通道数
        theta_nominal: 名义水平/垂直角度 (rad), shape (n_channels,)
    """

    def __init__(self, n_channels: int, theta_nominal: np.ndarray):
        self.n_channels = n_channels
        self.theta_nominal = theta_nominal
        self.delta_theta: np.ndarray = np.zeros(n_channels)
        self.d_bias: np.ndarray = np.zeros(n_channels)

    def _points_from_pose(
        self, distances: np.ndarray, theta: np.ndarray, phi: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """距离+角度 → 世界坐标点云。

        Args:
            distances: (N,) 距离
            theta: (N,) 水平角
            phi: (N,) 垂直角
            R: (3,3) 传感器旋转
            t: (3,) 传感器位置
        """
        x = distances * np.cos(phi) * np.cos(theta)
        y = distances * np.cos(phi) * np.sin(theta)
        z = distances * np.sin(phi)
        local_points = np.stack([x, y, z], axis=-1)
        return local_points @ R.T + t

    def calibrate_delta_theta(
        self,
        measurements: list[dict],
        learning_rate: float = 1e-4,
        max_iters: int = 500,
    ) -> tuple[np.ndarray, list[float]]:
        """Gauss-Newton 迭代估计角度偏差 Δθ。

        measurements: [{distances, theta, phi, R, t}, ...] — 多个位姿的观测

        残差: r = n·P(θ+Δθ, d) - d_plane
        每个位姿拟合一个平面，Δθ 是所有位姿共享的。

        Returns:
            (delta_theta, loss_history)
        """
        delta = np.zeros(self.n_channels)
        loss_history = []

        for _ in range(max_iters):
            total_loss = 0.0
            grad = np.zeros(self.n_channels)

            for meas in measurements:
                d_meas = meas["distances"]
                theta_raw = meas["theta"]
                phi = meas["phi"]
                R, t = meas["R"], meas["t"]
                valid = d_meas >= 0

                theta_corr = theta_raw + delta
                points = self._points_from_pose(
                    d_meas[valid], theta_corr[valid], phi[valid], R, t
                )

                if len(points) < 3:
                    continue

                normal, d_plane = fit_plane(points)
                residuals = point_to_plane_distance(points, normal, d_plane)
                total_loss += np.sum(residuals**2)

                # 雅可比: dP/dΔθ (数值微分)
                eps = 1e-4
                J = np.zeros((len(points), self.n_channels))
                for ch in range(self.n_channels):
                    delta_p = delta.copy()
                    delta_p[ch] += eps
                    theta_pert = theta_raw + delta_p
                    points_pert = self._points_from_pose(
                        d_meas[valid], theta_pert[valid], phi[valid], R, t
                    )
                    res_pert = point_to_plane_distance(points_pert, normal, d_plane)
                    J[:, ch] = (res_pert - residuals) / eps

                grad += J.T @ residuals

            loss_history.append(total_loss)
            if len(measurements) > 0:
                grad /= len(measurements)
            delta -= learning_rate * grad

        self.delta_theta = delta
        return delta, loss_history

    def calibrate_distance_bias(
        self,
        measurements: list[dict],
        true_distances: np.ndarray,
    ) -> np.ndarray:
        """估计距离零偏: bias = mean(d_measured - d_true)。

        Args:
            measurements: 含 channel 索引和 distances
            true_distances: 每个通道的真值距离 (m)
        Returns:
            d_bias: (n_channels,)
        """
        bias_sum = np.zeros(self.n_channels)
        counts = np.zeros(self.n_channels)

        for meas, d_true in zip(measurements, true_distances):
            d_meas = meas["distances"]
            valid = d_meas >= 0
            bias_sum[valid] += d_meas[valid] - d_true
            counts[valid] += 1

        self.d_bias = np.where(counts > 0, bias_sum / counts, 0.0)
        return self.d_bias

    def apply_correction(
        self, theta: np.ndarray, distances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """对内参偏差进行校正"""
        return theta + self.delta_theta, distances + self.d_bias
