"""LiDAR 拟真仿真器 — 策略级统一接口。

将 MuJoCo 渲染 + 信号模型 + 估计 串联成一条完整的噪声管线，
输出与真实雷达一致的点云格式 (x, y, z, intensity, ring, time)。

用法:
    from lidar_toolkit import LidarSim, LivoxGenerator
    from mujoco_lidar import MjLidarWrapper

    sim = LidarSim(lidar, "mid360", downsample=4,
                   range_noise=0.02, drop_prob=0.02)
    pcd = sim.scan(mj_data)  # → np.ndarray (N, 6)
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lidar_toolkit import (
    AttenuationModel,
    DepthEstimator,
    IntensityCalibrator,
    LidarNoiseModel,
    LivoxGenerator,
    ReflectivityModel,
    WaveformModel,
    apply_range_noise,
)


@dataclass
class LidarSimConfig:
    """LiDAR 仿真参数，模拟真实雷达的噪声特性。"""
    # 扫描
    lidar_model: str = "mid360"
    downsample: int = 4

    # 距离噪声: 真实雷达 ~1-3cm (1σ)
    range_noise_sigma: float = 0.02     # 距离高斯噪声 σ (m)
    range_noise_bias: float = 0.002     # 距离零偏 (m)
    drop_prob: float = 0.01             # 随机丢点概率

    # 角度噪声
    angle_encoder_res: int = 16384      # 编码器线数
    angle_misalignment: float = 0.001   # 轴偏差 (rad)
    angle_bearing_wobble: float = 0.0005  # 轴承跳动 σ (rad)

    # 信号参数
    peak_power: float = 50.0            # 发射峰值功率 (W)
    aperture_diameter: float = 0.02     # 接收孔径 (m)
    detector_type: str = "APD"
    quantum_efficiency: float = 0.2
    dark_count_rate: float = 1e4        # Hz
    background_power: float = 1e-9      # W

    # 波形/深度估计
    pulse_width: float = 5e-9           # s (FWHM)
    sampling_rate: float = 2e9          # Hz
    depth_method: str = "peak"          # peak | cfd | leading_edge

    # 反射率
    specular_fraction: float = 0.05     # Phong 镜面分量

    # 输出格式
    output_frame: str = "sensor"        # sensor | world
    output_columns: tuple = ("x", "y", "z", "intensity", "ring", "time")


class LidarSim:
    """拟真 LiDAR 仿真器。

    封装完整管线: 扫描模式 → MuJoCo 渲染 → 噪声注入 → 强度计算 → 格式化输出。

    Args:
        lidar_wrapper: MjLidarWrapper 实例（已绑定 MuJoCo 场景）
        config: 仿真参数配置
        geom_rgba: {geom_name: rgba_array}，用于反射率估算（可选）
    """

    def __init__(
        self,
        lidar_wrapper: Any,  # MjLidarWrapper
        config: LidarSimConfig | None = None,
        geom_rgba: dict[str, np.ndarray] | None = None,
    ):
        self.lidar = lidar_wrapper
        self.cfg = config or LidarSimConfig()
        self.geom_rgba = geom_rgba or {}

        # 扫描生成器（一次性加载 .npy 文件）
        self._gen = LivoxGenerator(self.cfg.lidar_model)
        self._theta, self._phi = self._gen.sample_ray_angles(
            downsample=self.cfg.downsample
        )
        self.n_rays = len(self._theta)

        # 子模块（纯 numpy，无状态）
        self._atm = AttenuationModel(
            peak_power=self.cfg.peak_power,
            aperture_diameter=self.cfg.aperture_diameter,
        )
        self._noise = LidarNoiseModel(
            detector_type=self.cfg.detector_type,
            dark_count_rate=self.cfg.dark_count_rate,
            background_power=self.cfg.background_power,
            quantum_efficiency=self.cfg.quantum_efficiency,
        )
        self._refl = ReflectivityModel(
            specular_fraction=self.cfg.specular_fraction,
        )
        self._wf = WaveformModel(
            pulse_width=self.cfg.pulse_width,
            sampling_rate=self.cfg.sampling_rate,
        )
        self._depth = DepthEstimator(method=self.cfg.depth_method)
        self._intensity = IntensityCalibrator()

        # 角度噪声
        from lidar_toolkit.estimation import AngleEstimator
        self._ang_est = AngleEstimator(
            encoder_resolution=self.cfg.angle_encoder_res,
            axis_misalignment=self.cfg.angle_misalignment,
            bearing_wobble=self.cfg.angle_bearing_wobble,
        )

        # 预计算 ring 索引（每个 θ/φ 对应的通道号，模拟多线雷达）
        self._rings = self._compute_rings(self._phi)

        # 随机数生成器
        self._rng = np.random.default_rng()

    @staticmethod
    def _compute_rings(phi: np.ndarray) -> np.ndarray:
        """将垂直角映射到 ring 编号 (0~N-1)。"""
        unique = np.sort(np.unique(np.round(phi, decimals=4)))
        ring_map = {float(v): i for i, v in enumerate(unique)}
        return np.array([ring_map[float(np.round(p, 4))] for p in phi], dtype=np.int32)

    def scan(self, mj_data: Any) -> np.ndarray:
        """执行一帧拟真 LiDAR 扫描。

        Args:
            mj_data: mujoco.MjData（已 mj_forward / mj_step）

        Returns:
            np.ndarray shape=(N, 6), columns=[x, y, z, intensity, ring, time]
            未命中点标记为 x=y=z=0, intensity=-1
        """
        # ── 1. MuJoCo 渲染：获取真值距离 ──
        gt_dist = self.lidar.trace_rays(mj_data, self._theta, self._phi)
        hit_points = self.lidar.get_hit_points()  # 局部坐标系
        valid_gt = gt_dist >= 0

        # ── 2. 角度噪声 ──
        theta_noisy = self._ang_est.apply_errors(self._theta, rng=self._rng)
        phi_noisy = self._ang_est.apply_errors(self._phi, rng=self._rng)

        # ── 3. 距离噪声 + 丢点 ──
        noisy_dist = apply_range_noise(
            gt_dist,
            sigma=self.cfg.range_noise_sigma,
            bias=self.cfg.range_noise_bias,
            drop_prob=self.cfg.drop_prob,
            rng=self._rng,
        )
        valid = noisy_dist >= 0

        if valid.sum() == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # ── 4. 信号衰减 + 反射率 → 回波强度 ──
        rho = np.full(self.n_rays, 0.5)  # 默认反射率
        if self.geom_rgba:
            # 如果有 RGBA 信息，用命中点做简单近似
            all_rho = self._refl.rgba_to_reflectivity(
                np.array(list(self.geom_rgba.values()))
            )
            rho = np.full(self.n_rays, float(all_rho.mean()))

        # 接收光子数
        pr = self._atm.received_power(rho[valid], noisy_dist[valid])
        photons_signal = self._atm.power_to_photons(pr, self.cfg.pulse_width)

        # ── 5. 探测器噪声 → 电子信号 → 强度 ──
        noisy_electrons = self._noise.apply(
            photons_signal, t_gate=self.cfg.pulse_width, rng=self._rng
        )
        # 强度归一化: 电子信号 / 最大信号 × 255（模拟 8-bit 强度）
        max_signal = max(noisy_electrons.max(), 1.0)
        intensity_raw = np.zeros(self.n_rays, dtype=np.float32)
        intensity_raw[valid] = np.clip(
            noisy_electrons / max_signal * 255, 0, 255
        ).astype(np.float32)
        # 未命中设为 -1
        intensity_raw[~valid] = -1

        # ── 6. 组装点云 ──
        # 用噪声距离重建局部坐标
        local_x = noisy_dist * np.cos(phi_noisy) * np.cos(theta_noisy)
        local_y = noisy_dist * np.cos(phi_noisy) * np.sin(theta_noisy)
        local_z = noisy_dist * np.sin(phi_noisy)

        # 未命中点归零
        local_x[~valid] = 0
        local_y[~valid] = 0
        local_z[~valid] = 0

        # 时间偏移 (归一化到 [0, 1]，模拟扫描周期内的时间戳)
        time_offset = np.linspace(0, 1, self.n_rays, dtype=np.float32)

        pcd = np.column_stack([
            local_x.astype(np.float32),
            local_y.astype(np.float32),
            local_z.astype(np.float32),
            intensity_raw,
            self._rings.astype(np.float32),
            time_offset,
        ])

        return pcd

    def scan_world(self, mj_data: Any) -> np.ndarray:
        """同 scan()，但点云在世界坐标系下。"""
        pcd_local = self.scan(mj_data)
        if len(pcd_local) == 0:
            return pcd_local

        R = self.lidar.sensor_rotation
        t = self.lidar.sensor_position

        xyz = pcd_local[:, :3] @ R.T + t
        result = pcd_local.copy()
        result[:, :3] = xyz.astype(np.float32)
        return result

    @property
    def scan_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """返回当前扫描模式的角度 (theta, phi)。"""
        return self._theta.copy(), self._phi.copy()

    def reseed(self, seed: int) -> None:
        """重置随机种子（用于可复现性）。"""
        self._rng = np.random.default_rng(seed)
