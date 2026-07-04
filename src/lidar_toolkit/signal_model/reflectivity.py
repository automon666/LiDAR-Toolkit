"""反射率与 BRDF 模型：朗伯反射、Phong 反射、强度估计"""

import numpy as np


def lambertian_brdf(rho: np.ndarray) -> np.ndarray:
    """朗伯 BRDF: f_r = ρ/π (理想漫反射)"""
    return rho / np.pi


class ReflectivityModel:
    """从 MuJoCo 材质 RGBA 估算反射率并计算回波强度因子。

    MuJoCo 的 geom rgba 提供颜色信息，可近似映射到 905nm 反射率。
    典型值: 白墙~0.9, 沥青~0.1, 植被~0.3~0.5, 金属~0.6~0.8

    Args:
        default_reflectivity: 无材质信息时的默认反射率
        specular_fraction: Phong 模型镜面分量占比, 0~1
        shininess: Phong 高光指数, 越大越集中
    """

    def __init__(
        self,
        default_reflectivity: float = 0.5,
        specular_fraction: float = 0.0,
        shininess: float = 32.0,
    ):
        self.default_reflectivity = default_reflectivity
        self.specular_fraction = specular_fraction
        self.shininess = shininess

    @staticmethod
    def rgba_to_reflectivity(rgba: np.ndarray) -> np.ndarray:
        """RGBA → 反射率近似映射 (基于亮度加权)。

        Args:
            rgba: shape (N, 4) 或 (4,), 值域 0~1
        Returns:
            反射率估计, shape 与输入一致
        """
        # 加权亮度: 人眼感知权重 → 近似 905nm 反射率趋势
        luminance = 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]
        alpha = rgba[..., 3]
        return np.clip(luminance * alpha, 0.02, 0.98)

    def intensity_factor(
        self,
        incident_angle: np.ndarray,
        material_reflectivity: np.ndarray | None = None,
    ) -> np.ndarray:
        """计算回波强度因子 (0~1), 考虑入射角和反射率。

        Args:
            incident_angle: 入射角 (rad), 0=垂直入射
            material_reflectivity: 反射率数组, None 则用默认值
        Returns:
            强度因子, shape 同 incident_angle
        """
        rho = (
            material_reflectivity
            if material_reflectivity is not None
            else np.full_like(incident_angle, self.default_reflectivity)
        )

        # 朗伯分量: cos(θ) 衰减
        cos_theta = np.cos(np.clip(incident_angle, -np.pi / 2, np.pi / 2))
        lambert = rho * cos_theta / np.pi

        # Phong 镜面分量
        if self.specular_fraction > 0:
            specular = rho * self.specular_fraction * np.power(cos_theta, self.shininess)
            lambert *= (1 - self.specular_fraction)
        else:
            specular = 0.0

        # 法向归一到 0~1
        max_val = rho / np.pi
        max_val = np.where(max_val > 0, max_val, 1.0)
        return np.clip((lambert + specular) / max_val, 0.0, 1.0)

    def compute_from_geom_rgba(
        self,
        geom_rgba: np.ndarray,
        incident_angles: np.ndarray,
    ) -> np.ndarray:
        """从 MuJoCo geom_rgba 直接计算强度因子。

        Args:
            geom_rgba: shape (N, 4), MuJoCo 几何体颜色
            incident_angles: 每条射线命中的入射角 (rad)
        """
        rho = self.rgba_to_reflectivity(geom_rgba)
        return self.intensity_factor(incident_angles, rho)
