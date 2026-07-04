"""深度估计算法 × MuJoCo 集成演示。

完整管线:
  MuJoCo 渲染真值距离 → 衰减模型求信号幅度
  → 波形仿真生成时域波形 → 4 种深度估计算法
  → 与 MuJoCo 真值对比误差

用法: python examples/depth_estimation_in_mujoco.py [--scene demo]
"""

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from mujoco_lidar import MjLidarWrapper
from lidar_toolkit import LivoxGenerator
from lidar_toolkit.signal_model import AttenuationModel, WaveformModel, LidarNoiseModel
from lidar_toolkit.estimation import (
    DepthEstimator,
    cfd_tof,
    peak_tof,
    leading_edge_tof,
    centroid_tof,
)

MODELS_DIR = pathlib.Path("/home/tino66/MuJoCo-LiDAR/models")

SCENES = {"demo": "demo.xml", "primitive": "scene_primitive.xml"}

# ═════════════════════════════════════════════════════════
# 波形 + 深度估计的核心管线
# ═════════════════════════════════════════════════════════

def generate_waveform(distance: float, amplitude: float,
                      wf_model: WaveformModel,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """从距离和幅度生成含噪数字化波形。

    MuJoCo 渲染距离 → 波形（模拟真实 ADC 输出）
    """
    return wf_model.single_echo(distance=distance, amplitude=amplitude, rng=rng)


def compute_amplitude(distance: float, reflectance: float,
                      atm: AttenuationModel,
                      noise_model: LidarNoiseModel,
                      rng: np.random.Generator,
                      pulse_width: float = 5e-9) -> float:
    """从距离和反射率计算回波信号幅度（V）。

    物理链路:
      P_r = P_t × ρ × A_r × η / (πR²) × exp(-2αR)  [接收功率]
      N_ph = P_r × τ / E_ph                         [光子数]
      V_out = NoiseModel(N_ph)                       [含噪电子信号]
    """
    pr = atm.received_power(np.array([reflectance]), np.array([distance]))
    photons = atm.power_to_photons(pr, pulse_width)
    noisy_signal = noise_model.apply(
        photons, t_gate=pulse_width, rng=rng
    )
    # 归一化到 [0.05, 1.0] V 的幅度范围
    amplitude = float(np.clip(np.abs(noisy_signal[0]) / 5000.0, 0.05, 1.0))
    return amplitude


def estimate_depth(t: np.ndarray, waveform: np.ndarray,
                   method: str, **kwargs) -> float | None:
    """从波形估计飞行时间 → 距离。"""
    if method == "peak":
        tof = peak_tof(t, waveform)
    elif method == "cfd":
        tof = cfd_tof(t, waveform, kwargs.get("fraction", 0.5),
                       kwargs.get("delay", 1e-9))
    elif method == "leading_edge":
        tof = leading_edge_tof(t, waveform, kwargs.get("threshold", 0.2))
    elif method == "centroid":
        tof = centroid_tof(t, waveform)
    else:
        raise ValueError(f"Unknown method: {method}")

    if tof is None:
        return None
    return tof * 3e8 / 2.0  # ToF → 距离


# ═════════════════════════════════════════════════════════
# 主程序
# ═════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="demo", choices=list(SCENES))
    parser.add_argument("--samples", type=int, default=50,
                        help="采样距离点数")
    parser.add_argument("--snr_test", action="store_true",
                        help="运行 SNR vs 精度理论测试")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # ── 1. 加载 MuJoCo 场景 ──
    print("=" * 60)
    print("深度估计算法 × MuJoCo 集成演示")
    print("=" * 60)

    path = MODELS_DIR / SCENES[args.scene]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)

    body_id = -1
    try:
        body_id = model.body("your_robot_name").id
    except Exception:
        pass

    lidar = MjLidarWrapper(
        model, "lidar_site", backend="cpu", cutoff_dist=100.0,
        args={"bodyexclude": body_id},
    )

    gen = LivoxGenerator("mid360")
    theta, phi = gen.sample_ray_angles(downsample=6)
    mujoco.mj_forward(model, data)
    gt_distances = lidar.trace_rays(data, theta, phi)
    valid = gt_distances >= 0

    print(f"\n场景: {args.scene}  |  射线: {len(gt_distances)}")
    print(f"命中: {valid.sum()}  |  距离: [{gt_distances[valid].min():.2f}, "
          f"{gt_distances[valid].max():.2f}]m")

    # ── 2. 初始化信号链 ──
    atm = AttenuationModel(peak_power=50.0, aperture_diameter=0.02)
    noise_model = LidarNoiseModel(detector_type="APD")
    wf_model = WaveformModel(pulse_width=5e-9, sampling_rate=2e9)

    # ── 3. 从 MuJoCo 渲染距离中采样 ──
    sample_distances = np.sort(gt_distances[valid])[::max(1, valid.sum() // args.samples)]
    sample_distances = sample_distances[:args.samples]
    n = len(sample_distances)

    print(f"\n采样 {n} 个距离点用于深度估计对比")
    print(f"距离范围: [{sample_distances[0]:.3f}, {sample_distances[-1]:.3f}]m\n")

    # ── 4. 对每个 MuJoCo 渲染距离运行完整管线 ──
    methods = ["peak", "cfd", "leading_edge", "centroid"]
    errors: dict[str, list[float]] = {m: [] for m in methods}
    times: dict[str, list[float]] = {m: [] for m in methods}

    for i, d_gt in enumerate(sample_distances):
        # 4a. 计算回波幅度（从物理模型）
        amplitude = compute_amplitude(
            float(d_gt), 0.5, atm, noise_model, rng
        )

        # 4b. 生成波形
        t0_wf = time.perf_counter()
        t_wf, wf = generate_waveform(float(d_gt), amplitude, wf_model, rng)
        wf_gen_time = (time.perf_counter() - t0_wf) * 1e6  # μs

        # 4c. 用 4 种方法估计距离
        for method in methods:
            t0 = time.perf_counter()
            d_est = estimate_depth(t_wf, wf, method)
            elapsed = (time.perf_counter() - t0) * 1e6  # μs

            if d_est is not None:
                errors[method].append(abs(d_est - d_gt))
                times[method].append(elapsed)

        # 打印前 5 个和前 3 个采样的详细对比
        if i < 5:
            print(f"  d_gt={d_gt:.3f}m  amp={amplitude:.2f}V  "
                  f"波形={len(t_wf)}点")
            for method in methods:
                d_est = estimate_depth(t_wf, wf, method)
                err_mm = abs(d_est - d_gt) * 1000 if d_est else float("nan")
                print(f"    {method:14s}: d_est={d_est:.4f}m  "
                      f"|Δ|={err_mm:.2f}mm")

    # ── 5. 统计结果 ──
    print("\n" + "=" * 60)
    print("深度估计 × MuJoCo 对比统计")
    print("=" * 60)
    print(f"{'方法':<14s} {'MAE(mm)':<10s} {'RMSE(mm)':<10s} "
          f"{'最大误差(mm)':<12s} {'耗时(μs)':<10s} {'成功率':<8s}")
    print("-" * 60)

    for method in methods:
        if len(errors[method]) == 0:
            continue
        errs = np.array(errors[method])
        mae = np.mean(errs) * 1000
        rmse = np.sqrt(np.mean(errs**2)) * 1000
        max_err = np.max(errs) * 1000
        avg_time = np.mean(times[method])
        success_rate = len(errors[method]) / n * 100
        print(f"{method:<14s} {mae:<10.2f} {rmse:<10.2f} "
              f"{max_err:<12.2f} {avg_time:<10.1f} {success_rate:<8.1f}%")

    # ── 6. SNR vs 精度 理论曲线 ──
    if args.snr_test:
        print("\n" + "=" * 60)
        print("理论: 距离精度 vs SNR (Cramér-Rao 下界)")
        print("=" * 60)
        de = DepthEstimator()
        snr_range = np.arange(5, 61, 5)
        sigma_r = de.distance_precision_vs_snr(snr_range, pulse_width=5e-9)
        print(f"{'SNR(dB)':<10s} {'σ_R(mm)':<10s}")
        for snr, sig in zip(snr_range, sigma_r):
            print(f"{snr:<10.0f} {sig*1000:<10.2f}")

    print("\n" + "=" * 60)
    best_method = min(methods, key=lambda m: np.mean(errors[m]) if errors[m] else float("inf"))
    print(f"最佳方法: {best_method} (MAE={np.mean(errors[best_method])*1000:.2f}mm)")
    print(f"MuJoCo 渲染距离 → 波形 → 深度估计 → 对比真值 → 完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
