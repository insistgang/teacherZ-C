"""
Neural Varifold 点云核实现
Neural Varifolds for 3D Point Cloud Processing

论文: IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022
作者: Xiaohao Cai, et al.

核心组件:
    - 位置核 (Position Kernel)
    - 法向量核 (Normal Kernel)
    - 变分范数 (Varifold Norm)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class PositionKernel(nn.Module):
    """
    位置核

    k(p, q) = exp(-||p - q||^2 / (2 * σ^2))

    高斯核计算点之间的空间相似性
    """

    def __init__(self, sigma: float = 1.0, learnable: bool = True):
        """
        参数:
            sigma: 高斯核带宽参数
            learnable: 是否可学习sigma
        """
        super().__init__()
        if learnable:
            self.sigma = nn.Parameter(torch.tensor(sigma))
        else:
            self.register_buffer("sigma", torch.tensor(sigma))

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        计算位置核矩阵

        参数:
            p: (B, N, 3) 或 (N, 3) 点集1
            q: (B, M, 3) 或 (M, 3) 点集2

        返回:
            kernel: (B, N, M) 或 (N, M) 核矩阵
        """
        if p.dim() == 2:
            p = p.unsqueeze(0)
            q = q.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        diff = p.unsqueeze(2) - q.unsqueeze(1)
        dist_sq = (diff**2).sum(dim=-1)

        kernel = torch.exp(-dist_sq / (2 * self.sigma**2))

        if squeeze:
            kernel = kernel.squeeze(0)

        return kernel


class NormalKernel(nn.Module):
    """
    法向量核

    k_n(n1, n2) = <n1, n2>^k

    测量法向量的对齐程度，k通常为1或2
    """

    def __init__(self, exponent: int = 1):
        """
        参数:
            exponent: 核指数k
        """
        super().__init__()
        self.exponent = exponent

    def forward(self, n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
        """
        计算法向量核矩阵

        参数:
            n1: (B, N, 3) 法向量集1 (需归一化)
            n2: (B, M, 3) 法向量集2 (需归一化)

        返回:
            kernel: (B, N, M) 核矩阵
        """
        if n1.dim() == 2:
            n1 = n1.unsqueeze(0)
            n2 = n2.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        inner = torch.bmm(n1, n2.transpose(1, 2))
        inner = torch.clamp(inner, -1, 1)

        kernel = inner**self.exponent

        if squeeze:
            kernel = kernel.squeeze(0)

        return kernel


class VarifoldRepresentation(nn.Module):
    """
    Varifold表示

    V = Σ_i w_i * φ(p_i, n_i) * δ_{p_i}

    将点云表示为带权重的几何测度
    """

    def __init__(
        self, feat_dim: int = 64, use_normals: bool = True, sigma_init: float = 0.5
    ):
        """
        参数:
            feat_dim: 特征维度
            use_normals: 是否使用法向量
            sigma_init: 位置核初始sigma
        """
        super().__init__()

        self.feat_dim = feat_dim
        self.use_normals = use_normals

        self.position_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, feat_dim)
        )

        if use_normals:
            self.normal_encoder = nn.Sequential(
                nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, feat_dim)
            )

        self.weight_net = nn.Sequential(
            nn.Linear(feat_dim * (2 if use_normals else 1), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

        self.pos_kernel = PositionKernel(sigma=sigma_init, learnable=True)
        self.normal_kernel = NormalKernel(exponent=1)

    def forward(
        self, positions: torch.Tensor, normals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码点云为Varifold表示

        参数:
            positions: (B, N, 3) 点位置
            normals: (B, N, 3) 点法向量 (可选)

        返回:
            features: (B, N, feat_dim) 点特征
            weights: (B, N, 1) Varifold权重
            encoded_positions: (B, N, 3) 编码后的位置
        """
        pos_feat = self.position_encoder(positions)

        if self.use_normals and normals is not None:
            normal_feat = self.normal_encoder(normals)
            combined_feat = torch.cat([pos_feat, normal_feat], dim=-1)
        else:
            combined_feat = pos_feat

        weights = self.weight_net(combined_feat)

        return pos_feat, weights, positions


class VarifoldKernel(nn.Module):
    """
    Varifold核

    K(V1, V2) = Σ_{i,j} w1_i * w2_j * k(p1_i, p2_j) * <f1_i, f2_j>

    计算两个Varifold之间的相似性
    """

    def __init__(self, sigma_pos: float = 0.5, sigma_feat: float = 1.0):
        """
        参数:
            sigma_pos: 位置核带宽
            sigma_feat: 特征核带宽
        """
        super().__init__()
        self.sigma_pos = sigma_pos
        self.sigma_feat = sigma_feat

    def forward(
        self,
        varifold1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        varifold2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算两个Varifold之间的核

        参数:
            varifold1: (positions1, features1, weights1)
            varifold2: (positions2, features2, weights2)

        返回:
            kernel: (B,) 核值
        """
        pos1, feat1, w1 = varifold1
        pos2, feat2, w2 = varifold2

        pos_diff = pos1.unsqueeze(2) - pos2.unsqueeze(1)
        pos_dist_sq = (pos_diff**2).sum(dim=-1)
        pos_kernel = torch.exp(-pos_dist_sq / (2 * self.sigma_pos**2))

        feat_inner = torch.bmm(feat1, feat2.transpose(1, 2))
        feat_kernel = torch.exp(-feat_inner / (2 * self.sigma_feat**2))

        weight_outer = torch.bmm(w1, w2.transpose(1, 2))

        kernel = (pos_kernel * feat_kernel * weight_outer).sum(dim=(1, 2))

        return kernel


class VarifoldDistance(nn.Module):
    """
    Varifold距离

    ||V1 - V2||^2_V = K(V1, V1) + K(V2, V2) - 2*K(V1, V2)
    """

    def __init__(self, sigma_pos: float = 0.5, sigma_feat: float = 1.0):
        super().__init__()
        self.kernel = VarifoldKernel(sigma_pos, sigma_feat)

    def forward(
        self,
        varifold1: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        varifold2: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        计算Varifold距离

        返回:
            distance: (B,) 距离值
        """
        k11 = self.kernel(varifold1, varifold1)
        k22 = self.kernel(varifold2, varifold2)
        k12 = self.kernel(varifold1, varifold2)

        distance = torch.sqrt(torch.clamp(k11 + k22 - 2 * k12, min=1e-8))

        return distance


class NeuralVarifoldNet(nn.Module):
    """
    完整的Neural Varifold网络

    用于点云分割/分类任务
    """

    def __init__(
        self, num_classes: int = 10, feat_dim: int = 64, use_normals: bool = True
    ):
        """
        参数:
            num_classes: 分割类别数
            feat_dim: 特征维度
            use_normals: 是否使用法向量
        """
        super().__init__()

        self.encoder = VarifoldRepresentation(
            feat_dim=feat_dim, use_normals=use_normals
        )

        self.varifold_layers = nn.ModuleList(
            [VarifoldLayer(feat_dim=feat_dim, sigma=0.5) for _ in range(3)]
        )

        self.seg_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self, positions: torch.Tensor, normals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        点云分割

        参数:
            positions: (B, N, 3) 点位置
            normals: (B, N, 3) 点法向量 (可选)

        返回:
            logits: (B, N, num_classes) 分割logits
        """
        features, weights, pos = self.encoder(positions, normals)

        for layer in self.varifold_layers:
            features = layer(pos, features, weights)

        logits = self.seg_head(features)

        return logits


class VarifoldLayer(nn.Module):
    """
    Varifold处理层

    基于Varifold核的特征更新
    """

    def __init__(self, feat_dim: int, sigma: float = 0.5):
        super().__init__()
        self.feat_dim = feat_dim
        self.sigma = sigma

        self.feature_transform = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, positions: torch.Tensor, features: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        基于Varifold核的特征更新

        参数:
            positions: (B, N, 3)
            features: (B, N, feat_dim)
            weights: (B, N, 1)

        返回:
            new_features: (B, N, feat_dim)
        """
        B, N, _ = positions.shape

        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist_sq = (diff**2).sum(dim=-1)
        pos_kernel = torch.exp(-dist_sq / (2 * self.sigma**2))

        weighted_kernel = pos_kernel * weights.squeeze(-1).unsqueeze(1)
        weighted_kernel = weighted_kernel / (
            weighted_kernel.sum(dim=-1, keepdim=True) + 1e-8
        )

        aggregated = torch.bmm(weighted_kernel, features)

        transformed = self.feature_transform(aggregated)

        combined = torch.cat([features, transformed], dim=-1)
        attention = self.attention(combined)

        new_features = features + attention * transformed

        return new_features


def compute_varifold_norm(
    positions: torch.Tensor,
    normals: torch.Tensor,
    weights: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    计算Varifold范数

    ||V||^2_V = Σ_{i,j} w_i * w_j * k(p_i, p_j) * <n_i, n_j>

    参数:
        positions: (B, N, 3)
        normals: (B, N, 3)
        weights: (B, N, 1)
        sigma: 位置核带宽

    返回:
        norm: (B,) Varifold范数
    """
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    dist_sq = (diff**2).sum(dim=-1)
    pos_kernel = torch.exp(-dist_sq / (2 * sigma**2))

    normal_inner = torch.bmm(normals, normals.transpose(1, 2))

    weight_outer = torch.bmm(weights, weights.transpose(1, 2))

    kernel_sum = (pos_kernel * normal_inner * weight_outer).sum(dim=(1, 2))

    return torch.sqrt(kernel_sum)


def demo_varifold():
    """Neural Varifold演示"""
    print("=" * 60)
    print("Neural Varifold点云核演示")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    B, N = 2, 256
    positions = torch.randn(B, N, 3)
    normals = F.normalize(torch.randn(B, N, 3), dim=-1)

    print(f"\n输入点云: {positions.shape}")
    print(f"法向量: {normals.shape}")

    print("\n--- 组件1: 位置核 ---")
    pos_kernel = PositionKernel(sigma=0.5)
    kernel_matrix = pos_kernel(positions[0], positions[0])
    print(f"位置核矩阵: {kernel_matrix.shape}")
    print(f"核值范围: [{kernel_matrix.min():.4f}, {kernel_matrix.max():.4f}]")

    print("\n--- 组件2: 法向量核 ---")
    normal_kernel = NormalKernel(exponent=1)
    normal_matrix = normal_kernel(normals[0], normals[0])
    print(f"法向量核矩阵: {normal_matrix.shape}")
    print(f"核值范围: [{normal_matrix.min():.4f}, {normal_matrix.max():.4f}]")

    print("\n--- 组件3: Varifold表示 ---")
    encoder = VarifoldRepresentation(feat_dim=64, use_normals=True)
    features, weights, _ = encoder(positions, normals)
    print(f"特征: {features.shape}")
    print(f"权重: {weights.shape}")
    print(f"权重范围: [{weights.min():.4f}, {weights.max():.4f}]")

    print("\n--- 组件4: Varifold核与距离 ---")
    varifold_kernel = VarifoldKernel(sigma_pos=0.5, sigma_feat=1.0)
    varifold_dist = VarifoldDistance(sigma_pos=0.5, sigma_feat=1.0)

    varifold1 = (positions[:1], features[:1], weights[:1])
    varifold2 = (positions[1:], features[1:], weights[1:])

    k_val = varifold_kernel(varifold1, varifold2)
    dist_val = varifold_dist(varifold1, varifold2)

    print(f"Varifold核值: {k_val.item():.6f}")
    print(f"Varifold距离: {dist_val.item():.6f}")

    print("\n--- 组件5: Varifold范数 ---")
    v_norm = compute_varifold_norm(positions, normals, weights, sigma=0.5)
    print(f"Varifold范数: {v_norm}")

    print("\n--- 组件6: 完整分割网络 ---")
    net = NeuralVarifoldNet(num_classes=10, feat_dim=64, use_normals=True)
    logits = net(positions, normals)
    print(f"输入: {positions.shape}")
    print(f"输出logits: {logits.shape}")

    pred = logits.argmax(dim=-1)
    print(f"预测类别: {pred.shape}")

    num_params = sum(p.numel() for p in net.parameters())
    print(f"网络参数: {num_params:,}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)

    return net


if __name__ == "__main__":
    demo_varifold()
