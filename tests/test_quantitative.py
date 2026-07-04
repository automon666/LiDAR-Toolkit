"""定量对比测试：MuJoCo 真值 vs 算法输出。

测试流程:
  1. MuJoCo 渲染多帧 LiDAR 数据 → 真值 (距离/角度/RGBA反射率)
  2. 校准模块 → 内参/外参/时间同步 对比
  3. 信号模型 → 衰减/噪声/波形 与物理预期对比
  4. 估计模块 → 深度/角度/反射率 估计值 vs 真值

用法:
  conda run -n sim2sim python tests/test_quantitative.py
  conda run -n sim2sim python tests/test_quantitative.py --scene scene_primitive
"""

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from lidar_toolkit.calibration import (
    ExtrinsicCalibrator,
    IntrinsicCalibrator,
    ReflectivityCalibrator,
    TimeSynchronizer,
    icp,
    solve_hand_eye_ax_xb,
)
from lidar_toolkit.estimation import (
    AngleEstimator,
    DepthEstimator,
    IntensityCalibrator,
)
from lidar_toolkit.signal_model import (
    AttenuationModel,
    LidarNoiseModel,
    ReflectivityModel,
    WaveformModel,
    apply_range_noise,
    lidar_range_equation,
)

# ── MuJoCo 渲染辅助 ──

SCENES = {
    "demo": "demo.xml",
    "primitive": "scene_primitive.xml",
    "go2": "scene_go2.xml",
}

MODELS_DIR = pathlib.Path("/home/tino66/MuJoCo-LiDAR/models")


def load_scene(name: str) -> tuple:
    path = MODELS_DIR / SCENES.get(name, "demo.xml")
    if not path.exists():
        raise FileNotFoundError(f"场景文件不存在: {path}")
    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mj_data = mujoco.MjData(mj_model)
    return mj_model, mj_data


def render_frame(mj_model, mj_data, theta, phi):
    """CPU 后端渲染一帧 LiDAR。"""
    from mujoco_lidar import MjLidarWrapper
    lidar = MjLidarWrapper(mj_model, "lidar_site", backend="cpu", cutoff_dist=100.0)
    mujoco.mj_forward(mj_model, mj_data)
    distances = lidar.trace_rays(mj_data, theta, phi)
    hit_points = lidar.get_hit_points()
    return distances, hit_points


from lidar_toolkit import LivoxGenerator


def get_scan_angles():
    gen = LivoxGenerator("mid360")
    return gen.sample_ray_angles(downsample=6)


def geom_rgba_map(model):
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): model.geom_rgba[i].copy()
            for i in range(model.ngeom)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)}


def move_sensor(mj_data, pos, quat=None):
    if mj_data.mocap_pos.size > 0:
        mj_data.mocap_pos[0] = pos
        if quat is not None:
            mj_data.mocap_quat[0] = quat
    else:
        # 非 mocap body: 直接设 qpos
        mj_data.qpos[0:3] = pos[:3]
        if quat is not None:
            mj_data.qpos[3:7] = quat[:4]


# ── 指标计算 ──

def metrics(pred, truth, mask=None):
    """计算 MAE, RMSE, R², 误差百分比。"""
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    valid = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[valid], truth[valid]
    if len(p) < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "R²": np.nan, "MEAN_ERR_%": np.nan, "N": 0}
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else np.nan
    mean_pct = float(np.mean(np.abs(err) / (np.abs(t) + 1e-6)) * 100)
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MEAN_ERR_%": mean_pct, "N": len(p)}


def print_metrics(title, m):
    print(f"  {title}: MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R²']:.4f}  "
          f"Err%={m['MEAN_ERR_%']:.1f}%  N={m['N']}")


# ── 主测试 ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="demo", choices=list(SCENES))
    parser.add_argument("--frames", type=int, default=5, help="测试帧数")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    print(f"场景: {args.scene}  |  测试帧数: {args.frames}")
    print("=" * 70)

    # 加载
    model, data = load_scene(args.scene)
    rgba_map = geom_rgba_map(model)
    theta, phi = get_scan_angles()
    n_rays = len(theta)

    print(f"射线数: {n_rays}  |  几何体: {list(rgba_map.keys())}\n")

    # 收集多帧数据
    all_distances = []
    all_hit_points = []
    for f in range(args.frames):
        dx = f * 0.3 - (args.frames - 1) * 0.15
        move_sensor(data, [dx, 0.0, 1.5])
        d, pts = render_frame(model, data, theta, phi)
        all_distances.append(d)
        all_hit_points.append(pts)

    gt_d = all_distances[0]
    valid = gt_d >= 0

    # ══════════════════════════════════════════════
    # 1. 内参标定
    # ══════════════════════════════════════════════
    print("── 1. 内参标定 (IntrinsicCalibrator) ──")
    n_ch = min(n_rays, 100)
    idx = np.linspace(0, n_rays - 1, n_ch, dtype=int)
    meas = []
    for f in range(min(args.frames, 3)):
        move_sensor(data, [f * 0.3, 0.0, 1.5])
        d, _ = render_frame(model, data, theta, phi)
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        R = data.site_xmat[site_id].reshape(3, 3).copy()
        t_vec = data.site_xpos[site_id].copy()
        meas.append({"distances": d[idx], "theta": theta[idx], "phi": phi[idx], "R": R, "t": t_vec})

    int_calib = IntrinsicCalibrator(n_channels=n_ch, theta_nominal=theta[idx])
    t0 = time.perf_counter()
    delta_theta, loss_hist = int_calib.calibrate_delta_theta(meas, learning_rate=1e-4, max_iters=80)
    elapsed = time.perf_counter() - t0
    theta_corr, d_corr = int_calib.apply_correction(theta[idx], all_distances[0][idx])
    m_int = metrics(theta_corr, theta[idx])
    print_metrics("Δθ 校正 (期望≈0)", m_int)
    print(f"  Gauss-Newton {len(loss_hist)}次  loss: {loss_hist[0]:.1f}→{loss_hist[-1]:.1f}  {elapsed:.1f}s\n")

    # ══════════════════════════════════════════════
    # 2. 外参标定
    # ══════════════════════════════════════════════
    print("── 2. 外参标定 (ExtrinsicCalibrator) ──")
    motions_a, motions_b = [], []
    for i in range(5):
        angle = 0.15 * (i + 1)
        quat = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        move_sensor(data, [i * 0.1, 0.0, 1.5], quat)
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "lidar_site")
        T = np.eye(4)
        T[:3, :3] = data.site_xmat[site_id].reshape(3, 3).copy()
        T[:3, 3] = data.site_xpos[site_id].copy()
        motions_a.append(T)
        motions_b.append(T)

    T_ext = solve_hand_eye_ax_xb(np.stack(motions_a), np.stack(motions_b))
    diag = np.diag(T_ext)
    hand_eye_err = np.max(np.abs(diag - [1, 1, 1, 1]))
    print(f"  手眼标定 对角线: {diag}  |  最大偏差: {hand_eye_err:.2e} (期望≈0)")

    # ICP: 两帧点云配准
    p0 = all_hit_points[0][valid][:500]
    p1 = all_hit_points[min(1, args.frames - 1)][valid][:500] if args.frames > 1 else p0 + 0.1
    R_icp, t_icp = icp(p0, p1, max_iters=20)
    R_err = np.linalg.norm(R_icp - np.eye(3))
    print(f"  ICP 收敛: R_err={R_err:.4f}  t={t_icp}\n")

    # ══════════════════════════════════════════════
    # 3. 信号模型
    # ══════════════════════════════════════════════
    print("── 3. 信号模型 ──")
    atm = AttenuationModel(peak_power=50.0)
    d_sample = gt_d[valid][:100]
    pr = atm.received_power(np.full(100, 0.5), d_sample)
    pr_eq = lidar_range_equation(50.0, np.full(100, 0.5), d_sample, atm.aperture_area)
    pr_diff = np.abs(pr - pr_eq).max()
    print(f"  received_power vs 方程: max|Δ|={pr_diff:.2e}W (期望≈0)")

    noise_model = LidarNoiseModel(detector_type="APD")
    n_photons = np.array([500., 200., 50., 10., 3.])
    noisy = noise_model.apply(n_photons, t_gate=5e-9, rng=rng)
    # 信号光子数越大噪声后信噪比越高
    snr_est = np.abs(noisy) / np.abs(noisy).std() if np.abs(noisy).std() > 0 else np.zeros_like(noisy)
    monotonic = all(snr_est[i] >= snr_est[i + 1] * 0.5 for i in range(len(snr_est) - 1))
    print(f"  噪声链输出: {noisy}  |  单调性: {'✓' if monotonic else '✗'}")

    wf_model = WaveformModel(pulse_width=5e-9, sampling_rate=2e9)
    d_test = float(d_sample[0])
    t_wf, wf = wf_model.single_echo(distance=d_test, amplitude=0.8, rng=rng)
    peak_idx = np.argmax(wf)
    tof_expected = 2 * d_test / 3e8
    tof_actual = t_wf[peak_idx]
    tof_err_ns = abs(tof_actual - tof_expected) * 1e9
    print(f"  波形峰值 ToF: {tof_actual*1e9:.3f}ns  期望: {tof_expected*1e9:.3f}ns  |Δ|={tof_err_ns:.3f}ns")
    print(f"  波形点数: {len(t_wf)}  峰值: {wf.max():.3f}V\n")

    # ══════════════════════════════════════════════
    # 4. 反射率模型
    # ══════════════════════════════════════════════
    print("── 4. 反射率模型 (ReflectivityModel) ──")
    refl_model = ReflectivityModel()
    all_rgba = np.array(list(rgba_map.values()))
    rho_rgba = refl_model.rgba_to_reflectivity(all_rgba)
    # RGBA 反射率应在 [0.02, 0.98] 范围内
    in_range = np.all((rho_rgba >= 0.02) & (rho_rgba <= 0.98))
    print(f"  RGBA→反射率: {dict(zip(rgba_map.keys(), np.round(rho_rgba, 3)))}")
    print(f"  范围检查: {'✓' if in_range else '✗'}")

    angles = np.array([0.0, 0.3, 0.6, 1.0])
    factors = refl_model.intensity_factor(angles, np.full(4, 0.5))
    # 朗伯: cos(θ) 衰减 → 因子应递减
    decreasing = all(factors[i] >= factors[i + 1] for i in range(len(factors) - 1))
    print(f"  朗伯强度因子: {np.round(factors, 3)}  |  递减: {'✓' if decreasing else '✗'}\n")

    # ══════════════════════════════════════════════
    # 5. 深度估计
    # ══════════════════════════════════════════════
    print("── 5. 深度估计 (DepthEstimator) ──")
    depths_gt = []
    depths_est_peak = []
    depths_est_cfd = []
    sample_dists = np.sort(gt_d[valid])[:50:5]  # 10 个采样距离

    for d_gt in sample_dists:
        t_wf, wf = wf_model.single_echo(distance=float(d_gt), amplitude=0.8, rng=rng)
        de_peak = DepthEstimator(method="peak")
        de_cfd = DepthEstimator(method="cfd", cfd_fraction=0.5, cfd_delay=5e-10)
        t_peak = de_peak.estimate(t_wf, wf)
        t_cfd = de_cfd.estimate(t_wf, wf)
        depths_gt.append(d_gt)
        depths_est_peak.append(de_peak.tof_to_distance(t_peak) if t_peak else np.nan)
        depths_est_cfd.append(de_cfd.tof_to_distance(t_cfd) if t_cfd else np.nan)

    m_peak = metrics(np.array(depths_est_peak), np.array(depths_gt))
    m_cfd = metrics(np.array(depths_est_cfd), np.array(depths_gt))
    print_metrics("峰值法", m_peak)
    print_metrics("CFD法 ", m_cfd)
    sigma_r = DepthEstimator().distance_precision_vs_snr(np.array([20., 30., 40.]))
    print(f"  理论精度 @SNR: {np.round(sigma_r*1000, 1)} mm @ [20,30,40]dB\n")

    # ══════════════════════════════════════════════
    # 6. 反射率估计
    # ══════════════════════════════════════════════
    print("── 6. 反射率估计 (IntensityCalibrator) ──")
    int_calib = IntensityCalibrator(system_gain=1.0)
    # 用等距离(10m)模拟强度来避免 R² 补偿的数值放大
    d_uniform = np.full(len(rho_rgba), 10.0)
    I_raw = rho_rgba / (d_uniform**2)
    I_comp = int_calib.compensate(I_raw, d_uniform, np.zeros(len(rho_rgba)))
    rho_est = int_calib.intensity_to_reflectivity(I_comp)
    m_rho = metrics(rho_est, rho_rgba)
    print_metrics("反射率估计 vs RGBA→ρ", m_rho)

    # 灰度板拟合
    rho_true = [0.05, 0.25, 0.50, 0.80]
    I_boards = [np.array([0.048, 0.052]), np.array([0.24, 0.26]),
                 np.array([0.48, 0.52]), np.array([0.78, 0.82])]
    K_fitted, r2_fit = int_calib.fit_from_gray_board(I_boards, rho_true)
    print(f"  灰度板拟合: K={K_fitted:.4f}  期望=1.0  R²={r2_fit:.4f}\n")

    # ══════════════════════════════════════════════
    # 7. 角度估计
    # ══════════════════════════════════════════════
    print("── 7. 角度估计 (AngleEstimator) ──")
    ang_est = AngleEstimator(encoder_resolution=16384, axis_misalignment=0.002, bearing_wobble=0.001)
    theta_meas = ang_est.apply_errors(theta, rng=rng)
    m_ang = metrics(theta_meas, theta)
    print_metrics("含噪角度 vs 真值", m_ang)

    coeffs = ang_est.fit_harmonic_error(theta, theta_meas, n_harmonics=3)
    # 重构后残差
    A = np.zeros((len(theta), 6))
    for k in range(3):
        A[:, 2 * k] = np.cos((k + 1) * theta)
        A[:, 2 * k + 1] = np.sin((k + 1) * theta)
    err_reconstructed = A @ coeffs
    residual = (theta_meas - theta) - err_reconstructed
    print(f"  谐波拟合残差 RMS: {np.sqrt(np.mean(residual**2))*1e3:.4f} mrad")
    print(f"  量化误差 RMS: {ang_est.quantization_error()*1e3:.2f} mrad")
    prec = ang_est.angular_precision(distance=float(gt_d[valid].mean()), range_precision=0.02)
    print(f"  角度精度 @{gt_d[valid].mean():.1f}m: {prec*1e3:.2f} mrad\n")

    # ══════════════════════════════════════════════
    # 汇总
    # ══════════════════════════════════════════════
    print("=" * 70)
    print("定量对比汇总")
    print("=" * 70)
    results = {
        "内参Δθ MAE": m_int["MAE"],
        "手眼标定偏差": hand_eye_err,
        "ICP R_err": R_err,
        "峰值深度 MAE(m)": m_peak["MAE"],
        "CFD深度 MAE(m)": m_cfd["MAE"],
        "反射率估计 MAE": m_rho["MAE"],
        "灰度板 R²": r2_fit,
        "角度噪声 MAE(mrad)": m_ang["MAE"] * 1e3,
        "谐波残差 RMS(mrad)": float(np.sqrt(np.mean(residual**2)) * 1e3),
    }
    for k, v in results.items():
        print(f"  {k:25s}: {v:.4f}")

    # 判断通过
    thresholds = {
        "手眼标定偏差": 0.01,
        "ICP R_err": 0.5,
        "峰值深度 MAE(m)": 0.5,
        "反射率估计 MAE": 0.1,
        "灰度板 R²": 0.9,
        "谐波残差 RMS(mrad)": 2.0,
    }
    lower_better = {"手眼标定偏差", "ICP R_err", "峰值深度 MAE(m)", "CFD深度 MAE(m)",
                     "反射率估计 MAE", "角度噪声 MAE(mrad)", "谐波残差 RMS(mrad)"}
    failed = []
    for k, th in thresholds.items():
        if k not in results:
            continue
        if k in lower_better and results[k] > th:
            failed.append(f"{k}: {results[k]:.4f} > {th}")
        elif k not in lower_better and results[k] < th:
            failed.append(f"{k}: {results[k]:.4f} < {th}")

    if failed:
        print(f"\n  ❌ {len(failed)} 项未通过阈值:")
        for f_item in failed:
            print(f"     {f_item}")
    else:
        print(f"\n  ✓ 全部 {len(thresholds)} 项通过阈值")


if __name__ == "__main__":
    main()
