"""
形状先验建模模块

实现管状结构的形状先验约束，包括:
- 曲率约束
- 管径约束
- 连通性约束

形状先验用于引导分割过程，确保结果符合管状结构的物理特性。
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import measure, morphology


class TubularShapePrior:
    """
    管状结构形状先验
    
    建模管状结构的几何特性:
    - 长度远大于宽度
    - 局部近似圆柱形
    - 曲率变化平滑
    
    属性:
        min_tube_width: 最小管径
        max_tube_width: 最大管径
        min_tube_length: 最小管长
        max_curvature: 最大曲率
    """
    
    def __init__(
        self,
        min_tube_width: int = 3,
        max_tube_width: int = 20,
        min_tube_length: int = 20,
        max_curvature: float = 0.5,
        smoothness_weight: float = 1.0
    ):
        """
        初始化管状形状先验
        
        参数:
            min_tube_width: 最小管径 (像素)
            max_tube_width: 最大管径 (像素)
            min_tube_length: 最小管长 (像素)
            max_curvature: 最大允许曲率
            smoothness_weight: 光滑性权重
        """
        self.min_tube_width = min_tube_width
        self.max_tube_width = max_tube_width
        self.min_tube_length = min_tube_length
        self.max_curvature = max_curvature
        self.smoothness_weight = smoothness_weight
    
    def compute_energy(self, segmentation: np.ndarray) -> float:
        """
        计算形状能量
        
        能量越低，形状越符合管状结构特征。
        
        参数:
            segmentation: 二值分割图
            
        返回:
            形状能量值
            
        示例:
            >>> shape_prior = TubularShapePrior()
            >>> energy = shape_prior.compute_energy(segmentation)
            >>> print(f"形状能量: {energy:.4f}")
        """
        if not np.any(segmentation):
            return float('inf')
        
        energy = 0.0
        
        # 1. 管径约束能量
        energy += self._width_energy(segmentation)
        
        # 2. 曲率约束能量
        energy += self._curvature_energy(segmentation)
        
        # 3. 连通性约束
        energy += self._connectivity_energy(segmentation)
        
        # 4. 光滑性约束
        energy += self._smoothness_energy(segmentation)
        
        return energy
    
    def _width_energy(self, segmentation: np.ndarray) -> float:
        """
        计算管径约束能量
        
        鼓励管径在合理范围内。
        
        参数:
            segmentation: 二值分割图
            
        返回:
            管径能量
        """
        # 距离变换得到每个点到边界的距离
        dist = distance_transform_edt(segmentation)
        
        # 估算管径 (2倍距离)
        estimated_width = 2 * dist[segmentation > 0]
        
        if len(estimated_width) == 0:
            return 0.0
        
        # 惩罚过大或过小的管径
        energy = 0.0
        
        # 惩罚小于最小管径
        too_narrow = estimated_width < self.min_tube_width
        energy += np.sum((self.min_tube_width - estimated_width[too_narrow]) ** 2)
        
        # 惩罚大于最大管径
        too_wide = estimated_width > self.max_tube_width
        energy += np.sum((estimated_width[too_wide] - self.max_tube_width) ** 2)
        
        return energy
    
    def _curvature_energy(self, segmentation: np.ndarray) -> float:
        """
        计算曲率约束能量
        
        鼓励管状结构具有平滑的曲率变化。
        
        参数:
            segmentation: 二值分割图
            
        返回:
            曲率能量
        """
        # 提取骨架
        skeleton = morphology.skeletonize(segmentation)
        
        # 如果没有骨架，返回0
        if not np.any(skeleton):
            return 0.0
        
        # 计算骨架点的曲率
        energy = 0.0
        
        # 获取骨架坐标
        y_coords, x_coords = np.where(skeleton)
        
        if len(y_coords) < 3:
            return 0.0
        
        # 对每点计算局部曲率
        for i in range(1, len(y_coords) - 1):
            # 三点计算曲率
            y_prev, x_prev = y_coords[i-1], x_coords[i-1]
            y_curr, x_curr = y_coords[i], x_coords[i]
            y_next, x_next = y_coords[i+1], x_coords[i+1]
            
            # 向量
            v1 = np.array([x_curr - x_prev, y_curr - y_prev])
            v2 = np.array([x_next - x_curr, y_next - y_curr])
            
            # 归一化
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                v1_norm = v1 / norm1
                v2_norm = v2 / norm2
                
                # 角度变化近似曲率
                angle = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1))
                
                # 惩罚过大曲率
                if angle > self.max_curvature:
                    energy += (angle - self.max_curvature) ** 2
        
        return energy
    
    def _connectivity_energy(self, segmentation: np.ndarray) -> float:
        """
        计算连通性约束能量
        
        鼓励管状结构具有良好的连通性。
        
        参数:
            segmentation: 二值分割图
            
        返回:
            连通性能量
        """
        # 标记连通区域
        labeled = measure.label(segmentation)
        num_regions = labeled.max()
        
        if num_regions == 0:
            return 0.0
        
        # 计算每个区域的面积
        regions = measure.regionprops(labeled)
        
        # 找到最大区域
        max_area = max(r.area for r in regions)
        
        # 惩罚碎片化 (小区域的总面积)
        energy = 0.0
        for region in regions:
            if region.area < self.min_tube_length:
                energy += region.area
        
        # 惩罚过多的不连通区域
        energy += num_regions * 10
        
        return energy
    
    def _smoothness_energy(self, segmentation: np.ndarray) -> float:
        """
        计算光滑性约束能量
        
        通过边界的光滑性来衡量。
        
        参数:
            segmentation: 二值分割图
            
        返回:
            光滑性能量
        """
        # 提取边界
        boundary = segmentation - morphology.erosion(
            segmentation,
            morphology.disk(1)
        )
        
        # 边界的总变差
        grad_x = np.gradient(boundary.astype(float), axis=0)
        grad_y = np.gradient(boundary.astype(float), axis=1)
        
        tv = np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y))
        
        return self.smoothness_weight * tv
    
    def apply_constraint(
        self,
        segmentation: np.ndarray,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        应用形状约束到分割结果
        
        通过后处理使分割结果更符合管状结构。
        
        参数:
            segmentation: 输入分割图
            strength: 约束强度 (0-1)
            
        返回:
            约束后的分割图
            
        示例:
            >>> constrained = shape_prior.apply_constraint(segmentation, strength=0.5)
        """
        result = segmentation.copy()
        
        # 1. 移除小物体
        result = morphology.remove_small_objects(
            result.astype(bool),
            min_size=self.min_tube_length
        )
        
        # 2. 填充小洞
        result = morphology.remove_small_holes(result, area_threshold=20)
        
        # 3. 形态学操作平滑边界
        selem = morphology.disk(1)
        result = morphology.opening(result, selem)
        result = morphology.closing(result, selem)
        
        # 4. 保持管径范围
        result = self._adjust_width(result)
        
        return result.astype(np.uint8)
    
    def _adjust_width(self, segmentation: np.ndarray) -> np.ndarray:
        """
        调整管径到合理范围
        
        参数:
            segmentation: 输入分割图
            
        返回:
            调整后的分割图
        """
        # 距离变换
        dist = distance_transform_edt(segmentation)
        
        # 限制最大距离
        adjusted = (dist <= self.max_tube_width / 2).astype(np.uint8)
        
        return adjusted


class CurvaturePrior:
    """
    曲率先验
    
    专门用于建模管状结构的曲率特性。
    """
    
    def __init__(self, max_curvature: float = 0.5, penalty_order: int = 2):
        """
        初始化曲率先验
        
        参数:
            max_curvature: 最大允许曲率
            penalty_order: 曲率惩罚阶数
        """
        self.max_curvature = max_curvature
        self.penalty_order = penalty_order
    
    def compute_curvature(self, curve_points: np.ndarray) -> np.ndarray:
        """
        计算曲线各点的曲率
        
        参数:
            curve_points: 曲线点坐标，形状 (N, 2)
            
        返回:
            各点曲率值
            
        示例:
            >>> curve = np.array([[0, 0], [1, 1], [2, 1], [3, 0]])
            >>> curvature = prior.compute_curvature(curve)
        """
        n_points = len(curve_points)
        
        if n_points < 3:
            return np.zeros(n_points)
        
        curvatures = np.zeros(n_points)
        
        # 使用三点法计算曲率
        for i in range(1, n_points - 1):
            p_prev = curve_points[i - 1]
            p_curr = curve_points[i]
            p_next = curve_points[i + 1]
            
            # 计算切向量
            t1 = p_curr - p_prev
            t2 = p_next - p_curr
            
            # 计算曲率 (角度变化率)
            dt = t2 - t1
            curvature = np.linalg.norm(dt) / (np.linalg.norm(t1) + 1e-8)
            
            curvatures[i] = curvature
        
        # 边界点使用邻近点
        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        
        return curvatures
    
    def energy_from_curvature(self, curvatures: np.ndarray) -> float:
        """
        从曲率计算能量
        
        参数:
            curvatures: 曲率值数组
            
        返回:
            形状能量
        """
        # 惩罚超过阈值的曲率
        excess = np.maximum(np.abs(curvatures) - self.max_curvature, 0)
        
        if self.penalty_order == 2:
            return np.sum(excess ** 2)
        else:
            return np.sum(excess ** self.penalty_order)


if __name__ == "__main__":
    """
    形状先验测试
    """
    print("="*60)
    print("形状先验模块测试")
    print("="*60)
    
    # 创建测试图像
    print("\n1. 创建管状结构测试图像...")
    size = 128
    
    # 直线管
    straight_tube = np.zeros((size, size), dtype=np.uint8)
    straight_tube[60:68, 20:108] = 1
    
    # 弯曲管
    y_coords = np.arange(size)
    x_coords = 20 + 0.5 * y_coords + 10 * np.sin(0.1 * y_coords)
    curved_tube = np.zeros((size, size), dtype=np.uint8)
    for y, x in zip(y_coords, x_coords):
        if 0 <= int(x) < size and 0 <= y < size:
            curved_tube[y-2:y+2, int(x)-2:int(x)+2] = 1
    
    # 碎片化图像
    fragmented = np.zeros((size, size), dtype=np.uint8)
    for i in range(5):
        x, y = np.random.randint(0, size-20), np.random.randint(0, size-10)
        fragmented[y:y+5, x:x+15] = 1
    
    print(f"   图像大小: {size}x{size}")
    
    # 测试形状先验
    print("\n2. 测试管状形状先验...")
    shape_prior = TubularShapePrior(
        min_tube_width=4,
        max_tube_width=16,
        min_tube_length=20
    )
    
    # 计算能量
    energy_straight = shape_prior.compute_energy(straight_tube)
    energy_curved = shape_prior.compute_energy(curved_tube)
    energy_fragmented = shape_prior.compute_energy(fragmented)
    
    print(f"   直线管能量: {energy_straight:.4f}")
    print(f"   弯曲管能量: {energy_curved:.4f}")
    print(f"   碎片化能量: {energy_fragmented:.4f}")
    print(f"   (能量越低，形状越符合管状结构)")
    
    # 测试约束应用
    print("\n3. 测试形状约束...")
    constrained = shape_prior.apply_constraint(fragmented, strength=0.5)
    print(f"   约束前区域数: {measure.label(fragmented).max()}")
    print(f"   约束后区域数: {measure.label(constrained).max()}")
    
    # 测试曲率先验
    print("\n4. 测试曲率先验...")
    curvature_prior = CurvaturePrior(max_curvature=0.3)
    
    # 生成测试曲线
    t = np.linspace(0, 2*np.pi, 100)
    curve_straight = np.column_stack([t, np.zeros_like(t)])
    curve_sine = np.column_stack([t, 0.5*np.sin(t)])
    
    curv_straight = curvature_prior.compute_curvature(curve_straight)
    curv_sine = curvature_prior.compute_curvature(curve_sine)
    
    print(f"   直线平均曲率: {np.mean(np.abs(curv_straight)):.4f}")
    print(f"   正弦曲线平均曲率: {np.mean(np.abs(curv_sine)):.4f}")
    
    print("\n" + "="*60)
    print("✅ 形状先验测试完成!")
    print("="*60)
