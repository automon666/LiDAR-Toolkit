"""角度精度分析与编码器误差建模"""

import numpy as np


class AngleEstimator:
    """机械旋转式激光雷达的角度估计与误差建模。

    Args:
        encoder_resolution: 编码器分辨率 (线/圈), 典型 4096~65536
        gear_ratio: 传动比, 1.0 为直驱
        axis_misalignment: 轴偏差角 (rad), 安装误差
        bearing_wobble: 轴承跳动 (rad RMS)
    """

    def __init__(
        self,
        encoder_resolution: int = 16384,
        gear_ratio: float = 1.0,
        axis_misalignment: float = 0.0,
        bearing_wobble: float = 0.0,
    ):
        self.encoder_resolution = encoder_resolution
        self.gear_ratio = gear_ratio
        self.axis_misalignment = axis_misalignment
        self.bearing_wobble = bearing_wobble
        self._angle_per_tick = 2 * np.pi / (encoder_resolution * gear_ratio)

    def quantization_error(self) -> float:
        """编码器量化误差 RMS (均匀分布 ±0.5 tick)。"""
        return self._angle_per_tick / np.sqrt(12)

    def apply_errors(self, theta_true: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """对真实角度施加误差链，返回含噪角度。

        误差源: 量化 + 轴偏差 + 轴承跳动
        """
        rng = rng or np.random.default_rng()
        n = len(theta_true)

        # 量化
        quant = rng.uniform(-0.5, 0.5, n) * self._angle_per_tick

        # 轴偏差 (一阶谐波)
        misalign = self.axis_misalignment * np.sin(theta_true)

        # 轴承跳动
        wobble = rng.normal(0, self.bearing_wobble, n)

        return theta_true + quant + misalign + wobble

    def estimate_resolution(self, theta_measured: np.ndarray) -> float:
        """从角度序列估计实际分辨率。"""
        diffs = np.diff(np.sort(theta_measured))
        return float(np.mean(diffs[diffs > 0]))

    def fit_harmonic_error(
        self, theta_true: np.ndarray, theta_measured: np.ndarray, n_harmonics: int = 3
    ) -> np.ndarray:
        """用傅里叶级数拟合角度误差的谐波分量。

        Returns:
            谐波系数 (2*n_harmonics,) = [a1,b1, a2,b2, ...]
        """
        error = theta_measured - theta_true
        A = np.zeros((len(theta_true), 2 * n_harmonics))
        for k in range(n_harmonics):
            A[:, 2 * k] = np.cos((k + 1) * theta_true)
            A[:, 2 * k + 1] = np.sin((k + 1) * theta_true)
        coeffs, _, _, _ = np.linalg.lstsq(A, error, rcond=None)
        return coeffs

    def angular_precision(self, distance: float, range_precision: float) -> float:
        """由距离精度反推角度精度。对于给定的距离 R 和测距精度 σ_R，
        目标横向位移 σ_lat = R · σ_θ, 若 σ_lat ≈ σ_R, 则 σ_θ ≈ σ_R / R。
        """
        return range_precision / max(distance, 0.01)
