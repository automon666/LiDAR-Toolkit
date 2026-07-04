"""深度估计算法：CFD、峰值检测、重心法、前沿鉴别"""

import numpy as np


def leading_edge_tof(t: np.ndarray, waveform: np.ndarray, threshold: float) -> float | None:
    """前沿鉴别 (Leading Edge Discriminator)。
    找到波形首次超过阈值的位置，线性插值精确定位。
    """
    above = np.where(waveform > threshold)[0]
    if len(above) == 0:
        return None
    idx = above[0]
    if idx == 0:
        return float(t[0])
    y0, y1 = waveform[idx - 1], waveform[idx]
    if abs(y1 - y0) < 1e-15:
        return float(t[idx])
    return float(t[idx - 1] + (threshold - y0) * (t[idx] - t[idx - 1]) / (y1 - y0))


def cfd_tof(t: np.ndarray, waveform: np.ndarray, fraction: float = 0.5, delay: float = 1e-9) -> float | None:
    """恒比鉴别 (Constant Fraction Discriminator)。
    延迟+衰减信号与原信号比较，过零点 = 到达时刻。
    """
    dt = t[1] - t[0]
    delay_samples = max(int(delay / dt), 1)
    if len(waveform) <= delay_samples:
        return None

    delayed = np.zeros_like(waveform)
    delayed[delay_samples:] = waveform[:-delay_samples]
    cfd_signal = delayed - fraction * waveform

    # 找过零点
    signs = np.sign(cfd_signal)
    for i in range(len(signs) - 1):
        if signs[i] > 0 and signs[i + 1] <= 0:
            y0, y1 = cfd_signal[i], cfd_signal[i + 1]
            if abs(y0 - y1) < 1e-15:
                return float(t[i])
            return float(t[i] - y0 * (t[i + 1] - t[i]) / (y1 - y0))
    return None


def peak_tof(t: np.ndarray, waveform: np.ndarray) -> float | None:
    """峰值检测。"""
    if len(waveform) == 0:
        return None
    idx = int(np.argmax(waveform))

    # 三点抛物线插值
    if 1 <= idx < len(t) - 1:
        y0, y1, y2 = waveform[idx - 1], waveform[idx], waveform[idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-15:
            delta = (y0 - y2) / (2 * denom)
            return float(t[idx] + delta * (t[1] - t[0]))
    return float(t[idx])


def centroid_tof(t: np.ndarray, waveform: np.ndarray) -> float | None:
    """重心法 (Center of Mass) — 对多回波场景效果较差，适合单峰。"""
    w = np.maximum(waveform, 0)
    total = w.sum()
    if total < 1e-15:
        return None
    return float(np.sum(t * w) / total)


class DepthEstimator:
    """从含噪波形中估计飞行时间 (ToF)，支持多种鉴别方法。

    Args:
        method: 'cfd' | 'peak' | 'centroid' | 'leading_edge'
        cfd_fraction: CFD 衰减比例
        cfd_delay: CFD 延迟时间 (s)
        led_threshold: 前沿鉴别阈值
    """

    def __init__(
        self,
        method: str = "cfd",
        cfd_fraction: float = 0.5,
        cfd_delay: float = 1e-9,
        led_threshold: float = 0.1,
    ):
        self.method = method
        self.cfd_fraction = cfd_fraction
        self.cfd_delay = cfd_delay
        self.led_threshold = led_threshold

    def estimate(self, t: np.ndarray, waveform: np.ndarray) -> float | None:
        if self.method == "cfd":
            return cfd_tof(t, waveform, self.cfd_fraction, self.cfd_delay)
        if self.method == "peak":
            return peak_tof(t, waveform)
        if self.method == "centroid":
            return centroid_tof(t, waveform)
        if self.method == "leading_edge":
            return leading_edge_tof(t, waveform, self.led_threshold)
        raise ValueError(f"Unknown method: {self.method}")

    def tof_to_distance(self, tof: float) -> float:
        return tof * 3e8 / 2.0

    def distance_precision_vs_snr(
        self, snr_db: np.ndarray, pulse_width: float = 5e-9
    ) -> np.ndarray:
        """理论距离精度 vs SNR (Cramér-Rao 下界近似)。

        σ_R ≈ c · σ_t / 2,  σ_t ≈ τ_pulse / SNR_linear
        """
        snr_linear = 10 ** (snr_db / 20)
        sigma_t = pulse_width / (snr_linear + 1e-10)
        sigma_R = 3e8 * sigma_t / 2.0
        return sigma_R
