"""MuJoCo LiDAR 仿真 + lidar_toolkit 算法实时处理。

功能:
  1. MuJoCo 渲染 demo 场景 → 真实 LiDAR 距离/点云
  2. lidar_toolkit 算法管线: 信号模型 + 标定 + 估计（实时终端输出）
  3. 可视化窗口: 3D 场景 + 按高度着色的 LiDAR 点云
  4. WASD/QE 移动雷达, 鼠标旋转/缩放视角

依赖: mujoco, mujoco_lidar (仅用于渲染), lidar_toolkit (算法)
用法: python examples/mujoco_lidar_sim.py [--scene demo|primitive|go2]
"""

import argparse
import pathlib
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

# ── lidar_toolkit 算法 ──
from lidar_toolkit import (
    AngleEstimator,
    AttenuationModel,
    DepthEstimator,
    IntensityCalibrator,
    LidarNoiseModel,
    LivoxGenerator,
    ReflectivityModel,
    WaveformModel,
    apply_range_noise,
)

# ── MuJoCo 渲染（来自 mujoco_lidar 包）──
from mujoco_lidar import MjLidarWrapper

SCENES = {
    "demo": "demo.xml",
    "primitive": "scene_primitive.xml",
    "go2": "scene_go2.xml",
}
MODELS_DIR = pathlib.Path("/home/tino66/MuJoCo-LiDAR/models")


def _height_rgba(z_norm):
    rgba = np.empty((z_norm.shape[0], 4), dtype=np.float64)
    rgba[:, 0] = z_norm
    rgba[:, 1] = 1.0 - np.abs(z_norm - 0.5) * 2.0
    rgba[:, 2] = 1.0 - z_norm
    rgba[:, 3] = 0.8
    return rgba


def load_scene(name):
    path = MODELS_DIR / SCENES[name]
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    return model, data


def run_pipeline(distances, theta, phi, hit_points, geom_rgba, rng):
    """实时运行一帧算法管线，返回关键指标。"""
    valid = distances >= 0
    if valid.sum() < 10:
        return {}

    d_valid = distances[valid]
    result = {}

    # 信号模型: 衰减
    atm = AttenuationModel(peak_power=50.0)
    sample_d = d_valid[:5]
    pr = atm.received_power(np.full(5, 0.5), sample_d)
    snr_vals = atm.snr(atm.power_to_photons(pr, 5e-9), 10)
    result["SNR_avg"] = float(snr_vals.mean())

    # 反射率估计
    refl_model = ReflectivityModel()
    all_rgba = np.array(list(geom_rgba.values()))
    rho_rgba = refl_model.rgba_to_reflectivity(all_rgba)
    int_calib = IntensityCalibrator(system_gain=1.0)
    d_ref = d_valid[: len(rho_rgba)]
    I_raw = rho_rgba / (np.full(len(rho_rgba), 10.0) ** 2)
    I_comp = int_calib.compensate(I_raw, np.full(len(rho_rgba), 10.0), np.zeros(len(rho_rgba)))
    rho_est = int_calib.intensity_to_reflectivity(I_comp)
    result["rho_mae"] = float(np.abs(rho_rgba - rho_est).mean())

    # 噪声 + 距离
    range_noisy = apply_range_noise(distances, sigma=0.02, drop_prob=0.02, rng=rng)
    result["drop_pct"] = float((range_noisy < 0).mean() * 100)

    # 波形 + 深度估计
    wf_model = WaveformModel(pulse_width=5e-9, sampling_rate=2e9)
    d_near = float(d_valid.min())
    t_wf, wf = wf_model.single_echo(distance=d_near, amplitude=0.8, rng=rng)
    de = DepthEstimator(method="peak")
    tof = de.estimate(t_wf, wf)
    result["depth_err_m"] = float(abs(de.tof_to_distance(tof) - d_near) if tof else np.nan)

    # 角度估计
    ang = AngleEstimator(encoder_resolution=16384, axis_misalignment=0.001, bearing_wobble=0.0005)
    theta_meas = ang.apply_errors(theta, rng=rng)
    result["angle_mae_mrad"] = float(np.abs(theta_meas - theta).mean() * 1e3)

    # 点到平面距离 (场景中地面/墙面)
    if len(hit_points) > 100:
        from lidar_toolkit.calibration.intrinsic import fit_plane
        pts = hit_points[valid][:200]
        normal, d_plane = fit_plane(pts)
        residuals = np.abs(np.dot(pts, normal) - d_plane)
        result["plane_rmse_m"] = float(np.sqrt(np.mean(residuals**2)))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="demo", choices=list(SCENES))
    parser.add_argument("--rate", type=float, default=8.0, help="LiDAR 更新频率 Hz")
    parser.add_argument("--downsample", type=int, default=6, help="mid360 降采样")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    model, data = load_scene(args.scene)

    # 提取几何体 RGBA
    geom_rgba = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): model.geom_rgba[i].copy()
        for i in range(model.ngeom)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    }

    # 扫描模式
    gen = LivoxGenerator("mid360")
    theta, phi = gen.sample_ray_angles(downsample=args.downsample)
    n_rays = len(theta)

    # LiDAR 渲染后端（排除机器人自身）
    body_id = -1
    try:
        body_id = model.body("your_robot_name").id
    except Exception:
        pass
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and ("robot" in name.lower() or "lidar" in name.lower()):
            body_id = i
            break
    lidar = MjLidarWrapper(
        model, "lidar_site", backend="cpu", cutoff_dist=100.0,
        args={"bodyexclude": body_id},
    )

    n_substeps = max(1, int(round(1.0 / (model.opt.timestep * args.rate))))

    print(f"场景: {args.scene}  |  射线: {n_rays}  |  频率: {args.rate}Hz")
    print(f"几何体: {list(geom_rgba.keys())}")
    print("WASD=移动  QE=升降  鼠标=旋转  Ctrl+滚轮=缩放  关闭窗口=退出\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 初始化点云可视化几何体（sphere 半径 0.05m，在 6m 场景中可见）
        viewer.user_scn.ngeom = n_rays
        for i in range(n_rays):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.0, 0.0, 0.9]),
            )
        geoms = viewer.user_scn.geoms

        data.mocap_pos[0] = [0.0, 0.0, 1.5]
        frame = 0
        t_last_print = 0.0

        while viewer.is_running():
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            # 射线追踪
            distances = lidar.trace_rays(data, theta, phi)
            hit_points = lidar.get_hit_points()
            world_points = hit_points @ lidar.sensor_rotation.T + lidar.sensor_position

            # 高度着色
            z = world_points[:, 2]
            z_min, z_max = z.min(), z.max()
            z_norm = (z - z_min) / (z_max - z_min + 1e-6) if z_max > z_min else np.zeros_like(z)
            colors = _height_rgba(z_norm)

            for i in range(n_rays):
                geoms[i].pos[:] = world_points[i]
                geoms[i].rgba[:] = colors[i]

            viewer.sync()

            # 每秒打印一次算法指标
            if data.time - t_last_print >= 1.0:
                t_last_print = data.time
                result = run_pipeline(distances, theta, phi, hit_points, geom_rgba, rng)
                if result:
                    parts = [f"t={data.time:.1f}s"]
                    parts.append(f"SNR={result.get('SNR_avg', 0):.0f}")
                    parts.append(f"ρ_mae={result.get('rho_mae', 0):.3f}")
                    parts.append(f"Δd={result.get('depth_err_m', 0)*1e3:.1f}mm")
                    parts.append(f"θ_mae={result.get('angle_mae_mrad', 0):.2f}mrad")
                    parts.append(f"丢点={result.get('drop_pct', 0):.1f}%")
                    if "plane_rmse_m" in result:
                        parts.append(f"平面σ={result['plane_rmse_m']*1e2:.1f}cm")
                    print("  " + "  |  ".join(parts))

            frame += 1


if __name__ == "__main__":
    main()
