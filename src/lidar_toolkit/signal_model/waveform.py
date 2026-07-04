"""波形仿真：高斯脉冲、多回波叠加、ADC 数字化"""

import numpy as np


def gaussian_pulse(t: np.ndarray, amplitude: float, t0: float, sigma: float) -> np.ndarray:
    """高斯脉冲: A × exp(-(t - t0)² / (2σ²))

    Args:
        t: 时间轴 (s)
        amplitude: 脉冲幅度
        t0: 脉冲中心时刻 (s)
        sigma: 标准差 (s), FWHM ≈ 2.355σ
    """
    return amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


class WaveformModel:
    """激光雷达回波波形仿真器。

    模拟: 激光脉冲发射 → 目标反射 → 多回波叠加 → ADC 采样 → 数字化波形

    Args:
        pulse_width: 激光脉冲 FWHM (s), 典型 1~10 ns
        sampling_rate: ADC 采样率 (Hz), 典型 500e6~2e9
        adc_bits: ADC 量化位数, 默认 12
        adc_range: ADC 满量程范围 (V), 默认 1.0
        noise_floor: 本底噪声 RMS (V)
    """

    def __init__(
        self,
        pulse_width: float = 5e-9,
        sampling_rate: float = 1e9,
        adc_bits: int = 12,
        adc_range: float = 1.0,
        noise_floor: float = 1e-3,
    ):
        self.pulse_width = pulse_width
        self.pulse_sigma = pulse_width / 2.355  # FWHM → σ
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.adc_bits = adc_bits
        self.adc_range = adc_range
        self.lsb = adc_range / (2**adc_bits)
        self.noise_floor = noise_floor

    def _make_time_axis(self, t_center: float, pulse_span: float = 5.0) -> np.ndarray:
        """以 t_center 为中心生成时间轴, ±pulse_span 个脉冲宽度"""
        half_span = pulse_span * self.pulse_sigma
        n_samples = int(2 * half_span / self.dt) + 1
        return np.linspace(t_center - half_span, t_center + half_span, n_samples)

    def tof_to_t0(self, distance: float) -> float:
        """距离 → 光子飞行时间 (往返)"""
        return 2 * distance / 3e8

    def single_echo(
        self,
        distance: float,
        amplitude: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成单个回波波形。

        Args:
            distance: 目标距离 (m)
            amplitude: 回波幅度 (V), 由反射率+衰减决定
        Returns:
            (t, waveform): 时间轴和含噪数字化波形
        """
        rng = rng or np.random.default_rng()
        t0 = self.tof_to_t0(distance)
        t = self._make_time_axis(t0)

        wf = gaussian_pulse(t, amplitude, t0, self.pulse_sigma)

        # 加噪声
        wf += rng.normal(0, self.noise_floor, len(t))

        # ADC 量化
        wf_digital = np.round(wf / self.lsb) * self.lsb
        wf_digital = np.clip(wf_digital, 0, self.adc_range)

        return t, wf_digital

    def multi_echo(
        self,
        distances: np.ndarray,
        amplitudes: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """生成多回波叠加波形 (如一束光打到玻璃+墙)。

        Args:
            distances: 各目标距离 (m), shape (N,)
            amplitudes: 各回波幅度 (V), shape (N,)
        Returns:
            (t, waveform): 叠加后的含噪波形
        """
        rng = rng or np.random.default_rng()
        valid = (distances >= 0) & (amplitudes > 0)
        if not valid.any():
            t = np.arange(0, 10 * self.pulse_sigma, self.dt)
            return t, np.zeros_like(t)

        d_valid = distances[valid]
        a_valid = amplitudes[valid]

        t_min = self.tof_to_t0(d_valid.min()) - 5 * self.pulse_sigma
        t_max = self.tof_to_t0(d_valid.max()) + 5 * self.pulse_sigma

        # 统一时间轴
        n_samples = max(int((t_max - t_min) / self.dt) + 1, 10)
        t = np.linspace(t_min, t_max, n_samples)
        wf = np.zeros(n_samples)

        for d, a in zip(d_valid, a_valid):
            t0 = self.tof_to_t0(d)
            wf += gaussian_pulse(t, a, t0, self.pulse_sigma)

        # 噪声 + 量化
        wf += rng.normal(0, self.noise_floor, n_samples)
        wf = np.round(wf / self.lsb) * self.lsb
        wf = np.clip(wf, 0, self.adc_range)

        return t, wf

    def extract_tof_cfd(
        self,
        t: np.ndarray,
        waveform: np.ndarray,
        fraction: float = 0.5,
    ) -> float | None:
        """CFD (恒比鉴别) 提取飞行时间。

        延迟+衰减信号与原信号比较, 过零点即为时刻。

        Args:
            t: 时间轴
            waveform: 数字化波形
            fraction: CFD 比例, 典型 0.5
        Returns:
            ToF (s) 或 None
        """
        delay_samples = int(0.5 * self.pulse_sigma / self.dt)  # 约半个脉冲宽度延迟
        delay_samples = max(delay_samples, 1)

        if len(waveform) <= delay_samples:
            return None

        delayed = np.roll(waveform, delay_samples)
        delayed[:delay_samples] = 0

        # CFD = delayed - fraction * original
        cfd_signal = delayed - fraction * waveform

        # 找过零点 (正到负)
        signs = np.sign(cfd_signal)
        zero_crossings = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]

        if len(zero_crossings) == 0:
            return None

        # 线性插值精确定位
        idx = zero_crossings[0]
        y0, y1 = cfd_signal[idx], cfd_signal[idx + 1]
        if abs(y0 - y1) < 1e-15:
            return float(t[idx])
        t_cross = t[idx] - y0 * (t[idx + 1] - t[idx]) / (y1 - y0)
        return float(t_cross)

    def extract_tof_peak(self, t: np.ndarray, waveform: np.ndarray) -> float | None:
        """峰值检测提取飞行时间."""
        if len(waveform) == 0:
            return None
        idx = int(np.argmax(waveform))
        return float(t[idx])

    def tof_to_distance(self, tof: float) -> float:
        """飞行时间 → 距离"""
        return tof * 3e8 / 2.0
