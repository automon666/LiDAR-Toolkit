"""衰减模型：激光雷达方程、距离平方衰减、大气消光"""

import numpy as np


def lidar_range_equation(
    pt: float,
    rho: np.ndarray,
    distances: np.ndarray,
    ar: float,
    eta_sys: float = 0.7,
) -> np.ndarray:
    """标准激光雷达方程: P_r = P_t × ρ × A_r × η / (π × R²)

    Args:
        pt: 发射峰值功率 (W)
        rho: 目标反射率, shape (N,)
        distances: 距离 (m), shape (N,)
        ar: 接收孔径面积 (m²)
        eta_sys: 系统光学效率
    Returns:
        接收功率 (W), shape (N,)
    """
    r = np.maximum(distances, 0.01)  # 避免除零
    return pt * rho * ar * eta_sys / (np.pi * r**2)


class AttenuationModel:
    """激光雷达系统衰减模型。

    Args:
        peak_power: 发射峰值功率 (W), 典型 25~200 W
        aperture_diameter: 接收孔径直径 (m), 典型 0.01~0.05
        optical_efficiency: 光学系统总效率, 典型 0.5~0.8
        atmospheric_coefficient: 大气消光系数 (/m), 905nm 晴天约 1e-4
        wavelength: 波长 (m), 用于光子能量计算
    """

    def __init__(
        self,
        peak_power: float = 50.0,
        aperture_diameter: float = 0.02,
        optical_efficiency: float = 0.7,
        atmospheric_coefficient: float = 1e-4,
        wavelength: float = 905e-9,
    ):
        self.peak_power = peak_power
        self.aperture_area = np.pi * (aperture_diameter / 2) ** 2
        self.optical_efficiency = optical_efficiency
        self.atmospheric_coefficient = atmospheric_coefficient
        self.wavelength = wavelength
        self.photon_energy = 6.626e-34 * 3e8 / wavelength

    def received_power(self, reflectance: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """计算接收光功率 (W), 含距离衰减和大气消光"""
        valid = distances >= 0
        pr = np.zeros_like(distances)
        pr[valid] = lidar_range_equation(
            self.peak_power,
            reflectance[valid],
            distances[valid],
            self.aperture_area,
            self.optical_efficiency,
        )
        # 大气消光: exp(-2αR)
        pr[valid] *= np.exp(-2 * self.atmospheric_coefficient * distances[valid])
        return pr

    def power_to_photons(self, power: np.ndarray, pulse_width: float) -> np.ndarray:
        """接收光功率 → 单脉冲光子数"""
        energy = power * pulse_width
        return energy / self.photon_energy

    def snr(
        self, n_signal_photons: np.ndarray, n_noise_photons: float = 0.0
    ) -> np.ndarray:
        """信噪比估计 (电压域 SNR, 散粒噪声限制)"""
        n_total = n_signal_photons + n_noise_photons
        snr = np.where(n_total > 0, n_signal_photons / np.sqrt(n_total), 0.0)
        return snr

    def max_detectable_range(
        self,
        reflectance: float,
        min_snr: float = 5.0,
        noise_floor_photons: float = 10.0,
        pulse_width: float = 5e-9,
    ) -> float:
        """估计给定反射率下的最大可探测距离。

        解: P_t × ρ × A_r × η × exp(-2αR) / (π × R² × E_photon) = min_snr × √noise_floor
        """
        numerator = (
            self.peak_power * reflectance * self.aperture_area * self.optical_efficiency * pulse_width
        )
        denominator = np.pi * self.photon_energy * min_snr * np.sqrt(noise_floor_photons)

        # 简化: 忽略大气消光做初步估计, 然后迭代修正
        r_est = np.sqrt(numerator / denominator)

        for _ in range(5):
            correction = np.exp(self.atmospheric_coefficient * r_est)
            r_est = np.sqrt(numerator * correction / denominator)

        return float(r_est)
