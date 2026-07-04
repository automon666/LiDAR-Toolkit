"""反射率标定：多级灰度板标定法 + 距离衰减曲线拟合"""

import numpy as np


class ReflectivityCalibrator:
    """通过多级灰度板 (标准反射率靶) 标定 LiDAR 强度→反射率映射。

    原理: 在不同距离下测量已知反射率靶标，拟合 I_corr = K × ρ 中的 K 值。
    K 值可能与距离有关 (非理想补偿)，可用多项式拟合。

    Args:
        true_reflectivities: 灰度板真实反射率列表, 如 [0.05, 0.25, 0.50, 0.80, 0.95]
        distances: 标定时使用的距离列表 (m), 如 [5, 10, 20, 50]
    """

    def __init__(self, true_reflectivities: list[float], distances: list[float]):
        self.refs = np.array(true_reflectivities)
        self.distances = np.array(distances)
        self.K_matrix: np.ndarray | None = None
        self.poly_coeffs: np.ndarray | None = None

    def calibrate(
        self, intensity_matrix: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """标定: 对每个距离-反射率组合的测量强度拟合 K。

        Args:
            intensity_matrix: (n_distances, n_refs) 补偿后强度 (已做距离+角度补偿)
        Returns:
            (K_per_distance, overall_r2)
        """
        n_d, n_r = intensity_matrix.shape
        self.K_matrix = np.zeros(n_d)

        for i in range(n_d):
            I_corr = intensity_matrix[i]
            self.K_matrix[i] = np.sum(self.refs * I_corr) / max(np.sum(self.refs**2), 1e-15)

        # 拟合优度
        I_pred = np.outer(self.K_matrix, self.refs)
        ss_res = np.sum((intensity_matrix - I_pred) ** 2)
        ss_tot = np.sum((intensity_matrix - intensity_matrix.mean()) ** 2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

        return self.K_matrix, r2

    def fit_k_vs_distance(self, degree: int = 2) -> np.ndarray:
        """拟合 K 值随距离的变化曲线。

        K(R) = c0 + c1×R + c2×R² + ...

        Returns:
            多项式系数 [c0, c1, ...]
        """
        if self.K_matrix is None:
            raise RuntimeError("请先调用 calibrate()")

        degree = min(degree, len(self.distances) - 1)
        self.poly_coeffs = np.polyfit(self.distances, self.K_matrix, degree)
        return self.poly_coeffs

    def get_k_at_distance(self, distance: float) -> float:
        """获取指定距离处的系统增益 K。"""
        if self.poly_coeffs is not None:
            return float(np.polyval(self.poly_coeffs, distance))
        if self.K_matrix is not None:
            # 线性插值
            return float(np.interp(distance, self.distances, self.K_matrix))
        return 1.0

    def intensity_to_reflectivity(
        self, intensity_corrected: np.ndarray, distances: np.ndarray
    ) -> np.ndarray:
        """批量将补偿后强度转为反射率, K 值按距离自适应。

        Args:
            intensity_corrected: 已补偿的强度 (距离+角度)
            distances: 每个点的距离 (m)
        """
        result = np.zeros_like(intensity_corrected)
        unique_d = np.unique(np.round(distances, decimals=1))

        for d in unique_d:
            mask = np.abs(distances - d) < 0.05
            if mask.any():
                K = self.get_k_at_distance(d)
                result[mask] = intensity_corrected[mask] / max(K, 1e-10)

        return np.clip(result, 0.0, 1.0)

    def calibration_report(self) -> str:
        """生成标定报告 (文本)。"""
        lines = ["=== 反射率标定报告 ===", ""]
        lines.append(f"灰度板反射率: {self.refs.tolist()}")
        lines.append(f"标定距离: {self.distances.tolist()}")

        if self.K_matrix is not None:
            lines.append("")
            lines.append("系统增益 K vs 距离:")
            for d, k in zip(self.distances, self.K_matrix):
                lines.append(f"  R={d:5.1f}m  K={k:.4f}")
            lines.append(f"  K 均值: {self.K_matrix.mean():.4f}")
            lines.append(f"  K 标准差: {self.K_matrix.std():.4f}")

        if self.poly_coeffs is not None:
            lines.append("")
            lines.append(f"K(R) 多项式: {self.poly_coeffs}")
            lines.append(f"  10m: K={self.get_k_at_distance(10):.4f}")
            lines.append(f"  50m: K={self.get_k_at_distance(50):.4f}")

        return "\n".join(lines)
