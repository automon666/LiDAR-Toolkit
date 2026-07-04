"""拟真 LiDAR + 策略集成示例。

模拟完整的 sim2sim 流程:
  1. MuJoCo 物理步进
  2. LidarSim 拟真扫描 → (x,y,z,intensity,ring,time) 点云
  3. 点云输入策略（ONNX 推理）
  4. 策略输出关节角 → 控制机器人

用法:
  python examples/lidar_policy_demo.py [--scene go2] [--onnx go2_policy.onnx]
"""

import argparse
import pathlib
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from lidar_toolkit import LidarSim, LidarSimConfig
from mujoco_lidar import MjLidarWrapper

SCENES = {
    "demo": ("demo.xml", "your_robot_name"),
    "primitive": ("scene_primitive.xml", "lidar_base"),
    "go2": ("scene_go2.xml", "trunk"),
    "g1": ("scene_g1.xml", "pelvis"),
}
MODELS_DIR = pathlib.Path("/home/tino66/MuJoCo-LiDAR/models")
POLICY_DIR = pathlib.Path("/home/tino66/MuJoCo-LiDAR/examples/onnx")

# ── 点云处理（策略输入预处理）──

def process_pointcloud(pcd: np.ndarray, max_points: int = 2048) -> np.ndarray:
    """将点云裁剪/填充到固定尺寸，用于策略输入。

    Args:
        pcd: (N, 6) = [x, y, z, intensity, ring, time]
        max_points: 策略需要的固定点数

    Returns:
        (max_points, 3) 裁剪/填充后的 xyz 点云
    """
    valid = pcd[:, 3] >= 0  # intensity >= 0 表示有效点
    pts = pcd[valid]

    if len(pts) >= max_points:
        # 均匀采样
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[idx]
    else:
        # 零填充
        padded = np.zeros((max_points, pts.shape[1]), dtype=pts.dtype)
        padded[: len(pts)] = pts
        pts = padded

    return pts[:, :3].astype(np.float32)


def load_policy(onnx_path: str | None):
    """加载 ONNX 策略（如果可用）。"""
    if onnx_path is None:
        return None
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path)
        print(f"  策略加载: {onnx_path}")
        return session
    except ImportError:
        print("  onnxruntime 未安装，跳过策略加载")
        return None
    except Exception as e:
        print(f"  策略加载失败: {e}")
        return None


# ── 主循环 ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="go2", choices=list(SCENES))
    parser.add_argument("--onnx", default=None, help="ONNX 策略路径")
    parser.add_argument("--rate", type=float, default=50.0, help="控制频率 Hz")
    parser.add_argument("--downsample", type=int, default=4,
                        help="LiDAR 降采样 (mid360: 4→6000 rays, 6→4000)")
    parser.add_argument("--no-viewer", action="store_true", help="无头模式")
    args = parser.parse_args()

    # 加载场景
    scene_file, body_name = SCENES[args.scene]
    scene_path = MODELS_DIR / scene_file
    if not scene_path.exists():
        print(f"场景 {scene_path} 不存在，回退到 demo")
        scene_path = MODELS_DIR / "demo.xml"
        body_name = "your_robot_name"

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    # 获取 body id（排除自身）
    body_id = -1
    try:
        body_id = model.body(body_name).id
    except Exception:
        pass

    # LiDAR 渲染后端
    lidar = MjLidarWrapper(
        model, "lidar_site", backend="cpu", cutoff_dist=100.0,
        args={"bodyexclude": body_id},
    )

    # 拟真仿真器
    sim = LidarSim(
        lidar,
        config=LidarSimConfig(
            lidar_model="mid360",
            downsample=args.downsample,
            range_noise_sigma=0.02,
            drop_prob=0.01,
        ),
    )

    # 策略
    onnx_path = args.onnx
    if onnx_path is None:
        auto_path = POLICY_DIR / f"{args.scene}_policy.onnx"
        if auto_path.exists():
            onnx_path = str(auto_path)
    policy = load_policy(onnx_path)

    n_substeps = max(1, int(round(1.0 / (model.opt.timestep * args.rate))))

    print(f"场景: {args.scene}  |  射线: {sim.n_rays}  |  频率: {args.rate}Hz")
    print(f"噪声: σ_range={sim.cfg.range_noise_sigma}m  drop={sim.cfg.drop_prob}")
    print(f"策略: {onnx_path or '无（仅仿真）'}")
    print("WASD=移动  QE=升降  鼠标=旋转  Ctrl+滚轮=缩放\n")

    if args.no_viewer:
        # 无头模式：打印统计
        for step in range(500):
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            pcd = sim.scan(data)
            valid = pcd[:, 3] >= 0

            if step % 50 == 0:
                print(f"  step={step:4d}  |  points={valid.sum():5d}  "
                      f"|  dist=[{pcd[valid,2].min():.2f}, {pcd[valid,2].max():.2f}]m  "
                      f"|  intensity=[{pcd[valid,3].min():.0f}, {pcd[valid,3].max():.0f}]")

            if policy is not None:
                pts = process_pointcloud(pcd, max_points=2048)
                # action = policy.run(None, {"obs": pts})[0]  # 伪代码
                # data.ctrl[:] = action

        print("无头仿真完成")
        return

    # ── 可视化模式 ──
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 点云可视化
        viewer.user_scn.ngeom = sim.n_rays
        for i in range(sim.n_rays):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.04, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([0.0, 1.0, 1.0, 0.7]),
            )
        geoms = viewer.user_scn.geoms

        step = 0
        t_last = 0.0

        while viewer.is_running():
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            # 拟真扫描
            pcd = sim.scan(data)
            valid = pcd[:, 3] >= 0

            # 世界坐标转换（用于可视化）
            xyz_local = pcd[:, :3].copy()
            xyz_world = xyz_local @ lidar.sensor_rotation.T + lidar.sensor_position

            # 按强度着色 (低=蓝, 高=红)
            intensities = np.clip(pcd[:, 3], 0, 255)
            i_norm = intensities / 255.0
            for i in range(sim.n_rays):
                if valid[i]:
                    geoms[i].pos[:] = xyz_world[i]
                    # 蓝→绿→红 按强度
                    geoms[i].rgba[:] = [i_norm[i], i_norm[i] * 0.5, 1.0 - i_norm[i], 0.7]
                else:
                    geoms[i].rgba[3] = 0.0  # 透明隐藏未命中点

            viewer.sync()

            # 策略推理（每步都用最新点云）
            if policy is not None and step % 1 == 0:
                pts = process_pointcloud(pcd, max_points=2048)
                # action = policy.run(None, {"obs": pts.reshape(1, -1)})[0]
                # data.ctrl[:] = action

            # 每秒打印状态
            if data.time - t_last >= 1.0:
                t_last = data.time
                d = np.linalg.norm(xyz_local[valid, :3], axis=1)
                print(f"  t={data.time:.1f}s  |  pts={valid.sum()}  "
                      f"|  dist=[{d.min():.1f}, {d.max():.1f}]m  "
                      f"|  I=[{intensities[valid].min():.0f}, {intensities[valid].max():.0f}]")

            step += 1


if __name__ == "__main__":
    main()
