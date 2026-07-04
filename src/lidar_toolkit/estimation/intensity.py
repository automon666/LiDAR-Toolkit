"""强度→反射率转换：距离补偿、角度归一化、灰度板标定"""

import numpy as np


class IntensityCalibrator:
    """从 LiDAR 回波强度反推目标反射率。

    原始强度 I_raw ∝ ρ × cos(θ) / R² × η_atm × η_sys
    校准后反射率: ρ = I_corrected / K, 其中 I_corrected = I_raw × R² / (cos(θ) × η_atm)

    Args:
        system_gain: 系统增益因子 K, 需通过灰度板标定获得
    """

    def __init__(self, system_gain: float = 1.0):
        self.system_gain = system_gain

    def distance_compensate(self, intensity: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """距离平方补偿: I_corrected = I_raw × R²"""
        r = np.maximum(distances, 0.01)
        valid = (distances >= 0) & (intensity > 0)
        result = np.zeros_like(intensity)
        result[valid] = intensity[valid] * r[valid] ** 2
        return result

    def angle_compensate(self, intensity: np.ndarray, incident_angles: np.ndarray) -> np.ndarray:
        """入射角补偿: I_corrected = I_raw / cos(θ)"""
        cos_theta = np.cos(np.clip(incident_angles, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))
        valid = cos_theta > 0.01
        result = np.zeros_like(intensity)
        result[valid] = intensity[valid] / cos_theta[valid]
        return result

    def compensate(
        self, intensity: np.ndarray, distances: np.ndarray, incident_angles: np.ndarray
    ) -> np.ndarray:
        """完整的距离+角度补偿。"""
        return self.distance_compensate(
            self.angle_compensate(intensity, incident_angles), distances
        )

    def intensity_to_reflectivity(self, intensity_corrected: np.ndarray) -> np.ndarray:
        """补偿后强度 → 反射率"""
        return np.clip(intensity_corrected / self.system_gain, 0.0, 1.0)

    def fit_from_gray_board(
        self,
        intensities_corrected: list[np.ndarray],
        true_reflectivities: list[float],
    ) -> tuple[float, float]:
        """从多级灰度板数据拟合系统增益。

        Args:
            intensities_corrected: 各灰度板的补偿后强度列表
            true_reflectivities: 对应真实反射率列表, 如 [0.05, 0.25, 0.50, 0.80]
        Returns:
            (K, R²) 系统增益和拟合优度
        """
        x = np.concatenate([np.full_like(ic, rho) for ic, rho in zip(intensities_corrected, true_reflectivities)])
        y = np.concatenate(intensities_corrected)

        # 线性回归: y = K × x  (过原点)
        K = float(np.sum(x * y) / max(np.sum(x * x), 1e-15))

        # R²
        y_pred = K * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-15)

        return K, r_squared

    def estimate_max_range_vs_reflectivity(
        self, reflectivities: np.ndarray, min_intensity: float = 0.01, peak_power: float = 50.0
    ) -> np.ndarray:
        """估计不同反射率下的最大探测距离。

        I_min = P_t × ρ × A_r × η / (π × R_max² × E_photon) × pulse_width
        → R_max = sqrt(P_t × ρ × A_r × η × pulse_width / (π × E_photon × I_min))
        """
        from lidar_toolkit.signal_model.attenuation import AttenuationModel

        att = AttenuationModel(peak_power=peak_power)
        ranges = []
        for rho in reflectivities:
            r_max = att.max_detectable_range(
                reflectance=rho,
                noise_floor_photons=min_intensity,
            )
            ranges.append(r_max)
        return np.array(ranges)
