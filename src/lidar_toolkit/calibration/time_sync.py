"""时间同步：多传感器时间戳对齐与偏移估计"""

import numpy as np


def cross_correlation_offset(signal_a: np.ndarray, signal_b: np.ndarray) -> int:
    """互相关法估计两信号之间的时间偏移 (采样点数)。

    Returns:
        offset: signal_b 相对于 signal_a 的延迟 (正=signal_b 滞后)
    """
    corr = np.correlate(signal_a - signal_a.mean(), signal_b - signal_b.mean(), mode="full")
    lag = np.argmax(corr) - (len(signal_b) - 1)
    return int(lag)


def interpolate_timestamps(
    timestamps: np.ndarray, times_new: np.ndarray
) -> np.ndarray:
    """将信号线性插值到新的时间轴。

    用于将不同传感器的数据对齐到统一时间基准。
    """
    if len(timestamps) < 2:
        return np.full(len(times_new), timestamps[0] if len(timestamps) > 0 else 0.0)
    return np.interp(times_new, timestamps, np.arange(len(timestamps)))


class TimeSynchronizer:
    """多传感器时间同步器。

    支持:
    1. 硬件脉冲同步: 通过已知同步脉冲估计偏移
    2. 数据驱动同步: 通过互相关/IMU 角速度峰值对齐
    3. 时间戳插值对齐

    Args:
        hardware_sync_period: 硬件同步脉冲周期 (s), 0=无硬件同步
        max_offset: 最大允许时间偏移 (s)
    """

    def __init__(self, hardware_sync_period: float = 0.0, max_offset: float = 0.1):
        self.hardware_sync_period = hardware_sync_period
        self.max_offset = max_offset
        self.offsets: dict[str, float] = {}

    def estimate_offset_from_sync_pulse(
        self, sync_timestamps: np.ndarray, sensor_timestamps: np.ndarray
    ) -> float:
        """从已知的硬件同步脉冲时间戳估计传感器时钟偏移。

        假设同步脉冲周期固定，找到最近邻脉冲对齐。
        """
        if len(sync_timestamps) == 0 or len(sensor_timestamps) == 0:
            return 0.0

        # 第一个脉冲对齐
        t0_sync = sync_timestamps[0]
        t0_sensor = sensor_timestamps[
            int(np.argmin(np.abs(sensor_timestamps - t0_sync)))
        ]
        return t0_sync - t0_sensor

    def estimate_offset_from_motion(
        self,
        imu_angular_velocity: np.ndarray,
        imu_timestamps: np.ndarray,
        lidar_angular_velocity: np.ndarray,
        lidar_timestamps: np.ndarray,
    ) -> float:
        """从 IMU 和 LiDAR 角速度信号估计时间偏移 (互相关)。

        LiDAR 角速度可以从点云配准帧间运动获得。
        """
        # 重采样到统一时间轴
        dt = min(np.diff(imu_timestamps).mean(), np.diff(lidar_timestamps).mean())
        t_min = max(imu_timestamps[0], lidar_timestamps[0])
        t_max = min(imu_timestamps[-1], lidar_timestamps[-1])
        t_uniform = np.arange(t_min, t_max, dt)

        imu_interp = np.interp(t_uniform, imu_timestamps, imu_angular_velocity)
        lidar_interp = np.interp(t_uniform, lidar_timestamps, lidar_angular_velocity)

        lag_samples = cross_correlation_offset(imu_interp, lidar_interp)
        return float(lag_samples) * dt

    def align(
        self,
        sensor_name: str,
        sensor_timestamps: np.ndarray,
        reference_timestamps: np.ndarray,
        offset: float | None = None,
    ) -> np.ndarray:
        """将传感器时间戳对齐到参考时钟。

        Args:
            sensor_name: 传感器标识
            sensor_timestamps: 传感器原始时间戳
            reference_timestamps: 参考时钟时间戳
            offset: 已知时间偏移 (s), None 则用先前估计值
        Returns:
            对齐后的传感器时间戳
        """
        if offset is None:
            offset = self.offsets.get(sensor_name, 0.0)

        self.offsets[sensor_name] = offset
        return sensor_timestamps + offset

    def get_aligned_data(
        self,
        timestamps_a: np.ndarray,
        data_a: np.ndarray,
        timestamps_b: np.ndarray,
        data_b: np.ndarray,
        offset_b: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将两组数据对齐到统一时间基准。

        Args:
            timestamps_a, data_a: 参考传感器数据
            timestamps_b, data_b: 待对齐传感器数据
            offset_b: 传感器 b 的时间偏移 (s)
        Returns:
            (t_common, data_a_aligned, data_b_aligned)
        """
        tb_corrected = timestamps_b + offset_b
        t_min = max(timestamps_a[0], tb_corrected[0])
        t_max = min(timestamps_a[-1], tb_corrected[-1])
        dt = min(np.diff(timestamps_a).mean(), np.diff(tb_corrected).mean())
        t_common = np.arange(t_min, t_max, dt)

        da = np.interp(t_common, timestamps_a, data_a)
        db = np.interp(t_common, tb_corrected, data_b)
        return t_common, da, db
