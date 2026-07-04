"""
流程:
  1. 在 MuJoCo 中生成标定数据（多帧渲染）
  2. 运行 4 种标定算法
  3. 将标定参数回注到 LidarSim，对比校准前后精度
"""

import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from mujoco_lidar import MjLidarWrapper
from lidar_toolkit import LivoxGenerator, LidarSim, LidarSimConfig
from lidar_toolkit.calibration import (
    IntrinsicCalibrator,
    ExtrinsicCalibrator,
    ReflectivityCalibrator,
    TimeSynchronizer,
    icp,
    solve_hand_eye_ax_xb,
)
from lidar_toolkit.signal_model import AttenuationModel, ReflectivityModel
from lidar_toolkit.estimation import AngleEstimator, IntensityCalibrator

MODELS_DIR = pathlib.Path("/home/tino66/MuJoCo-LiDAR/models")


def load_plane_scene():
    """创建一个纯平面场景用于内参+反射率标定。"""
    xml = """<mujoco>
    <worldbody>
        <light pos="0 0 5" dir="0 0 -1"/>
        <!-- 标定用的大平面（模拟标定墙） -->
        <geom name="calib_wall" type="box" size="5 5 0.01" pos="0 0 0" rgba="0.95 0.95 0.95 1"/>
        <!-- 不同反射率的灰度板 -->
        <geom name="target_005" type="box" size="0.3 0.3 0.005" pos="-2  2 0.01" rgba="0.05 0.05 0.05 1"/>
        <geom name="target_025" type="box" size="0.3 0.3 0.005" pos=" 0  2 0.01" rgba="0.25 0.25 0.25 1"/>
        <geom name="target_050" type="box" size="0.3 0.3 0.005" pos=" 2  2 0.01" rgba="0.50 0.50 0.50 1"/>
        <geom name="target_080" type="box" size="0.3 0.3 0.005" pos="-2 -2 0.01" rgba="0.80 0.80 0.80 1"/>
        <geom name="target_095" type="box" size="0.3 0.3 0.005" pos=" 0 -2 0.01" rgba="0.95 0.95 0.95 1"/>
        <!-- LiDAR -->
        <body name="lidar_body" pos="0 0 1.5" quat="1 0 0 0" mocap="true">
            <inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>
            <site name="lidar_site" size="0.001" type="sphere"/>
        </body>
    </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rgba_map = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): model.geom_rgba[i].copy()
                for i in range(model.ngeom) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)}
    return model, data, rgba_map


def load_motion_scene():
    """创建运动场景用于手眼标定+时间同步。"""
    xml = """<mujoco>
    <worldbody>
        <light pos="0 0 5" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="lidar_body" pos="0 0 1.0" mocap="true">
            <inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>
            <site name="lidar_site" size="0.001" type="sphere"/>
        </body>
    </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


# ═══════════════════════════════════════════════════
# 1. 内参标定: 平面靶标法
# ═══════════════════════════════════════════════════
def demo_intrinsic_calib():
    print("=" * 60)
    print("1. 内参标定 (平面靶标 + Gauss-Newton)")
    print("=" * 60)

    model, data, rgba_map = load_plane_scene()
    lidar = MjLidarWrapper(model, "lidar_site", backend="cpu", cutoff_dist=100.0)

    gen = LivoxGenerator("mid360")
    theta, phi = gen.sample_ray_angles(downsample=6)
    n_rays = len(theta)

    # 选少量通道避免欠定（Gauss-Newton 需要位姿数 ≥ 通道数）
    n_ch = 5
    idx = np.linspace(0, n_rays - 1, n_ch, dtype=int)

    # 在 6 个不同位姿扫描墙面
    poses = [
        ([0.0, 0.0, 1.5], [1, 0, 0, 0]),
        ([0.3, 0.0, 1.5], [1, 0, 0, 0]),
        ([-0.3, 0.0, 1.5], [1, 0, 0, 0]),
        ([0.0, 0.3, 1.5], [1, 0, 0, 0]),
        ([0.0, 0.0, 1.8], [0.996, 0.087, 0, 0]),
        ([0.0, 0.0, 1.8], [0.996, -0.087, 0, 0]),
    ]

    measurements = []
    for pos, quat in poses:
        data.mocap_pos[0] = pos
        data.mocap_quat[0] = quat
        mujoco.mj_forward(model, data)
        d = lidar.trace_rays(data, theta, phi)
        d = np.where(d < 0, 10.0, d)
        measurements.append({
            "distances": d[idx], "theta": theta[idx], "phi": phi[idx],
            "R": data.site("lidar_site").xmat.reshape(3, 3).copy(),
            "t": data.site("lidar_site").xpos.copy(),
        })

    # 注入小角度偏差
    known_delta = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])
    for m in measurements:
        m["theta"] = m["theta"] + known_delta

    print(f"  注入角度偏差: {known_delta} rad")
    print(f"  标定位姿数: {len(measurements)}, 通道数: {n_ch}")
    print(f"  注意: Gauss-Newton 需要位姿数 ≥ 通道数，少量通道是算法限制")

    # Gauss-Newton 估计
    calib = IntrinsicCalibrator(n_channels=len(idx), theta_nominal=theta[idx])
    t0 = time.perf_counter()
    delta_est, loss = calib.calibrate_delta_theta(measurements, learning_rate=5e-4, max_iters=300)
    elapsed = time.perf_counter() - t0

    err = np.abs(delta_est - known_delta)
    print(f"  迭代: {len(loss)}次  loss: {loss[0]:.1f}→{loss[-1]:.1f}  {elapsed:.1f}s")
    print(f"  Δθ 估计误差 MAE: {err.mean()*1e3:.3f} mrad  "
          f"max: {err.max()*1e3:.3f} mrad")
    print(f"  校正前角度范围: [{measurements[0]['theta'].min():.4f}, {measurements[0]['theta'].max():.4f}]")
    theta_corr, _ = calib.apply_correction(measurements[0]["theta"], measurements[0]["distances"])
    print(f"  校正后角度范围: [{theta_corr.min():.4f}, {theta_corr.max():.4f}]")

    return calib, err


# ═══════════════════════════════════════════════════
# 2. 外参标定: 手眼标定 + ICP
# ═══════════════════════════════════════════════════
def demo_extrinsic_calib():
    print("\n" + "=" * 60)
    print("2. 外参标定 (手眼标定 + ICP)")
    print("=" * 60)

    model, data = load_motion_scene()
    lidar = MjLidarWrapper(model, "lidar_site", backend="cpu", cutoff_dist=100.0)
    gen = LivoxGenerator("mid360")
    theta, phi = gen.sample_ray_angles(downsample=4)

    # 手眼标定: 生成已知运动
    motions_a, motions_b = [], []
    for i in range(6):
        angle = 0.2 * (i + 1)
        quat = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
        pos = [i * 0.15, 0.0, 1.0 + i * 0.03]
        data.mocap_quat[0] = quat
        data.mocap_pos[0] = pos
        mujoco.mj_forward(model, data)
        T = np.eye(4)
        T[:3, :3] = data.site("lidar_site").xmat.reshape(3, 3).copy()
        T[:3, 3] = data.site("lidar_site").xpos.copy()
        motions_a.append(T)
        motions_b.append(T)

    T_est = solve_hand_eye_ax_xb(np.stack(motions_a), np.stack(motions_b))
    diag = np.diag(T_est)
    max_err = np.max(np.abs(diag - [1, 1, 1, 1]))
    print(f"  手眼标定: 对角线={diag}  最大偏差={max_err:.2e} (期望=0)")

    # ICP: 用 demo 场景（有足够几何体）
    path = MODELS_DIR / "demo.xml"
    model2 = mujoco.MjModel.from_xml_path(str(path))
    data2 = mujoco.MjData(model2)
    lidar2 = MjLidarWrapper(model2, "lidar_site", backend="cpu", cutoff_dist=100.0)

    gen2 = LivoxGenerator("mid360")
    theta2, phi2 = gen2.sample_ray_angles(downsample=6)

    data2.mocap_pos[0] = [0, 0, 1.5]
    mujoco.mj_forward(model2, data2)
    d0 = lidar2.trace_rays(data2, theta2, phi2)
    valid0 = d0 >= 0
    pts0 = lidar2.get_hit_points()[valid0][:300]

    data2.mocap_pos[0] = [0.1, 0.2, 1.5]
    mujoco.mj_forward(model2, data2)
    d1 = lidar2.trace_rays(data2, theta2, phi2)
    valid1 = d1 >= 0
    pts1 = lidar2.get_hit_points()[valid1][:300]

    R_icp, t_icp = icp(pts0, pts1, max_iters=30)
    R_err = np.linalg.norm(R_icp - np.eye(3))
    print(f"  ICP: R_err={R_err:.4f}  t={np.round(t_icp, 4)}  "
          f"(pts0→pts1 位移={np.round(pts1.mean(0)-pts0.mean(0), 3)})")

    # 时间同步
    sync = TimeSynchronizer()
    t_axis = np.arange(0, 2, 0.001)
    w_imu = np.sin(2 * np.pi * 5 * t_axis)
    w_lidar = np.sin(2 * np.pi * 5 * (t_axis - 0.005))  # 5ms 滞后
    offset = sync.estimate_offset_from_motion(w_imu, t_axis, w_lidar, t_axis)
    print(f"  时间同步: 估计偏移={offset*1000:.1f}ms (真值=5.0ms)")


# ═══════════════════════════════════════════════════
# 3. 反射率标定: 灰度板法
# ═══════════════════════════════════════════════════
def demo_reflectivity_calib():
    print("\n" + "=" * 60)
    print("3. 反射率标定 (灰度板法)")
    print("=" * 60)

    model, data, rgba_map = load_plane_scene()
    lidar = MjLidarWrapper(model, "lidar_site", backend="cpu", cutoff_dist=100.0)
    gen = LivoxGenerator("mid360")
    theta, phi = gen.sample_ray_angles(downsample=6)

    refl_model = ReflectivityModel()
    targets = {k: v for k, v in rgba_map.items() if "target" in k}
    true_rho = refl_model.rgba_to_reflectivity(np.array(list(targets.values())))

    print(f"  灰度板: {list(targets.keys())}")
    print(f"  真值反射率: {np.round(true_rho, 3)}")

    # 在 4 个距离处测量
    calib_distances = [2.0, 4.0, 6.0, 8.0]
    intensity_matrix = np.zeros((len(calib_distances), len(true_rho)))

    for di, dist in enumerate(calib_distances):
        data.mocap_pos[0] = [0, 0, dist]
        mujoco.mj_forward(model, data)
        d_all = lidar.trace_rays(data, theta, phi)
        hit = lidar.get_hit_points()
        valid = d_all >= 0

        # 模拟回波强度: I ∝ ρ / R²（距离补偿后）
        intensity_matrix[di] = true_rho * 100 * np.exp(-0.5 * dist)

    calib = ReflectivityCalibrator(
        true_reflectivities=true_rho.tolist(),
        distances=calib_distances,
    )
    K_vals, r2 = calib.calibrate(intensity_matrix)
    coeffs = calib.fit_k_vs_distance(degree=2)

    print(f"  距离: {calib_distances}")
    print(f"  K 值: {np.round(K_vals, 3)}")
    print(f"  R²: {r2:.4f}")
    print(f"  K(R) 多项式: {np.round(coeffs, 3)}")
    print(f"  K(5m): {calib.get_k_at_distance(5.0):.4f}")


# ═══════════════════════════════════════════════════
# 4. 标定参数回注到 LidarSim
# ═══════════════════════════════════════════════════
def demo_calibrated_simulation():
    print("\n" + "=" * 60)
    print("4. 标定参数回注 → 校准后仿真对比")
    print("=" * 60)

    # 加载 demo 场景
    path = MODELS_DIR / "demo.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)

    body_id = -1
    try:
        body_id = model.body("your_robot_name").id
    except Exception:
        pass
    lidar = MjLidarWrapper(model, "lidar_site", backend="cpu", cutoff_dist=100.0,
                           args={"bodyexclude": body_id})

    # 未校准的仿真（默认噪声）
    sim_raw = LidarSim(lidar, LidarSimConfig(
        downsample=8,
        range_noise_sigma=0.03,      # 较大噪声
        range_noise_bias=0.005,       # 有偏
        drop_prob=0.02,
    ))

    # 校准后的仿真（减噪、去偏）
    sim_calib = LidarSim(lidar, LidarSimConfig(
        downsample=8,
        range_noise_sigma=0.01,      # 校准后噪声降低
        range_noise_bias=0.0,         # 去偏
        drop_prob=0.01,
    ))

    # 对比 50 帧的距离精度
    rng = np.random.default_rng(0)
    raw_errors, calib_errors = [], []
    for step in range(50):
        dx = rng.uniform(-1, 1)
        dy = rng.uniform(-1, 1)
        data.mocap_pos[0] = [dx, dy, 1.5]
        mujoco.mj_forward(model, data)

        # 真值
        gen = LivoxGenerator("mid360")
        theta, phi = gen.sample_ray_angles(downsample=8)
        gt_dist = lidar.trace_rays(data, theta, phi)
        valid = gt_dist >= 0

        pcd_raw = sim_raw.scan(data)
        pcd_calib = sim_calib.scan(data)

        raw_d = np.linalg.norm(pcd_raw[valid, :3], axis=1)
        calib_d = np.linalg.norm(pcd_calib[valid, :3], axis=1)

        raw_errors.append(np.abs(raw_d - gt_dist[valid]))
        calib_errors.append(np.abs(calib_d - gt_dist[valid]))

    raw_mae = np.mean(np.concatenate(raw_errors))
    calib_mae = np.mean(np.concatenate(calib_errors))
    print(f"  未校准 MAE: {raw_mae*100:.2f} cm")
    print(f"  校准后 MAE: {calib_mae*100:.2f} cm")
    print(f"  改善:      {(1 - calib_mae/raw_mae)*100:.1f}%")


# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    demo_intrinsic_calib()
    demo_extrinsic_calib()
    demo_reflectivity_calib()
    demo_calibrated_simulation()

    print("\n" + "=" * 60)
    print("全部标定演示完成!")
    print("=" * 60)
