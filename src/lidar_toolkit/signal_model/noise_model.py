"""LiDAR 噪声模型：散粒噪声、热噪声、暗计数、背景光噪声、量化噪声"""

import numpy as np


class LidarNoiseModel:
    """LiDAR 探测器噪声模型，支持 APD 和 SPAD 两种探测器类型。

    Args:
        detector_type: 'APD' (线性模式) 或 'SPAD' (盖革模式)
        dark_count_rate: 暗计数率 (Hz), 典型值 1e3~1e5
        background_power: 背景光功率 (W), 典型值 1e-9~1e-6
        wavelength: 激光波长 (m), 默认 905nm
        quantum_efficiency: 量子效率, 0~1, 典型 0.1~0.3
        excess_noise_factor: APD 过剩噪声因子, 典型 2~5
        temperature: 探测器温度 (K), 影响热噪声, 默认 300K
        load_resistance: 跨阻放大器反馈电阻 (Ω), 默认 1e3
        bandwidth: 探测器带宽 (Hz), 默认 200e6
        adc_bits: ADC 量化位数, 默认 12
        adc_range: ADC 满量程范围, 默认 1.0
    """

    def __init__(
        self,
        detector_type: str = "APD",
        dark_count_rate: float = 1e4,
        background_power: float = 1e-9,
        wavelength: float = 905e-9,
        quantum_efficiency: float = 0.2,
        excess_noise_factor: float = 3.0,
        temperature: float = 300.0,
        load_resistance: float = 1e3,
        bandwidth: float = 200e6,
        adc_bits: int = 12,
        adc_range: float = 1.0,
    ):
        self.detector_type = detector_type
        self.dark_count_rate = dark_count_rate
        self.background_power = background_power
        self.wavelength = wavelength
        self.quantum_efficiency = quantum_efficiency
        self.excess_noise_factor = excess_noise_factor
        self.temperature = temperature
        self.load_resistance = load_resistance
        self.bandwidth = bandwidth
        self.adc_bits = adc_bits
        self.adc_range = adc_range

        self.h = 6.626e-34
        self.c = 3e8
        self.k_B = 1.38e-23
        self.photon_energy = self.h * self.c / wavelength

    def photon_to_electrons(self, n_photons: np.ndarray) -> np.ndarray:
        """光子数 → 光电子数"""
        return np.random.poisson(n_photons * self.quantum_efficiency)

    def dark_noise(self, t_gate: float, n_samples: int) -> np.ndarray:
        """暗计数噪声 (Poisson)"""
        return np.random.poisson(self.dark_count_rate * t_gate, n_samples)

    def background_noise(self, t_gate: float, n_samples: int) -> np.ndarray:
        """背景光产生的光电子 (Poisson)"""
        bg_photons = self.background_power * t_gate / self.photon_energy
        return self.photon_to_electrons(np.full(n_samples, bg_photons))

    def thermal_noise(self, n_samples: int) -> np.ndarray:
        """热噪声 (Johnson-Nyquist, Gaussian)"""
        sigma = np.sqrt(4 * self.k_B * self.temperature * self.bandwidth / self.load_resistance)
        return np.random.normal(0, sigma, n_samples)

    def apd_gain(self, n_electrons: np.ndarray) -> np.ndarray:
        """APD 倍增 (均值 M_gain, 过剩噪声 σ = sqrt(F_excess * M^2))"""
        M = 100.0  # 典型 APD 增益
        F = self.excess_noise_factor
        n_multiplied = n_electrons * M
        noise = np.random.normal(0, np.sqrt(n_electrons * F * M**2))
        return n_multiplied + noise

    def quantization_noise(self, signal: np.ndarray) -> np.ndarray:
        """ADC 量化噪声 (均匀分布 ±0.5 LSB)"""
        lsb = self.adc_range / (2**self.adc_bits)
        return signal + np.random.uniform(-0.5 * lsb, 0.5 * lsb, signal.shape)

    def apply(
        self,
        n_photons: np.ndarray,
        t_gate: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """对光子计数信号施加完整噪声链。

        Args:
            n_photons: 每个回波的信号光子数
            t_gate: 门控时间 (s), 即脉冲宽度
        Returns:
            含噪的电子信号 (APD: 电压, SPAD: 0/1 检测标志)
        """
        rng = rng or np.random.default_rng()

        # 信号光电转换
        n_signal_e = self.photon_to_electrons(n_photons)

        # 暗计数
        dark_e = self.dark_noise(t_gate, len(n_photons))

        # 背景光
        bg_e = self.background_noise(t_gate, len(n_photons))

        n_total = n_signal_e + dark_e + bg_e

        if self.detector_type == "SPAD":
            # SPAD: 盖革模式, 有光子就触发 (简化为 Bernoulli 检测)
            prob_detect = 1 - np.exp(-n_total * self.quantum_efficiency)
            prob_detect = np.clip(prob_detect, 0, 1)
            # 堆积效应: 高光下饱和
            prob_detect = 1 - np.exp(-prob_detect)
            return (rng.random(len(n_photons)) < prob_detect).astype(np.float64)

        # APD: 线性模式
        multiplied = self.apd_gain(n_total)
        # 热噪声 (电压域)
        thermal = self.thermal_noise(len(n_photons))
        signal = multiplied + thermal
        # ADC 量化
        return self.quantization_noise(signal)


def apply_range_noise(
    distances: np.ndarray,
    sigma: float = 0.02,
    bias: float = 0.0,
    drop_prob: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """对距离测量施加简单的高斯噪声 + 零偏 + 随机丢点。

    Args:
        distances: 真值距离 (m), -1 表示未命中
        sigma: 距离噪声标准差 (m), 典型 0.01~0.03
        bias: 距离零偏 (m)
        drop_prob: 随机丢点概率 (0~1)
    Returns:
        含噪距离
    """
    rng = rng or np.random.default_rng()
    valid = distances >= 0
    noisy = distances.copy()

    gauss = rng.normal(bias, sigma, distances.shape)
    noisy[valid] += gauss[valid]

    if drop_prob > 0:
        drop_mask = rng.random(len(distances)) < drop_prob
        noisy[valid & drop_mask] = -1

    return noisy
