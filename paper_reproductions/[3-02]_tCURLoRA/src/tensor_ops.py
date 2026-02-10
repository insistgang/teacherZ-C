"""
张量操作模块

实现张量CUR分解和相关的张量运算。

核心功能:
- 矩阵CUR分解
- 张量CUR分解
- n-模乘积
- 张量秩近似

数学基础:
    CUR分解是矩阵低秩近似的一种有效方法，
    选择实际的行和列而非线性组合，提高可解释性。

参考文献:
    Mahoney & Drineas (2009). CUR matrix decompositions.
    Drineas et al. (2007). Tensor-CUR decompositions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional, Union


def cur_decomposition(
    A: Union[np.ndarray, torch.Tensor],
    rank: int,
    num_cols: Optional[int] = None,
    num_rows: Optional[int] = None,
    method: str = 'uniform'
) -> Tuple[Union[np.ndarray, torch.Tensor], ...]:
    """
    矩阵CUR分解
    
    将矩阵A分解为C、U、R三个矩阵，其中C和R是A的实际列和行。
    
    参数:
        A: 输入矩阵，形状 (m, n)
        rank: 目标秩
        num_cols: 选取的列数 (默认2*rank)
        num_rows: 选取的行数 (默认2*rank)
        method: 采样方法 ('uniform', 'leverage')
        
    返回:
        (C, U, R) 元组
        - C: 选取的列，形状 (m, c)
        - U: 连接矩阵，形状 (c, r)
        - R: 选取的行，形状 (r, n)
        
    示例:
        >>> A = np.random.randn(100, 50)
        >>> C, U, R = cur_decomposition(A, rank=10)
        >>> A_approx = C @ U @ R
        >>> error = np.linalg.norm(A - A_approx)
    """
    if num_cols is None:
        num_cols = 2 * rank
    if num_rows is None:
        num_rows = 2 * rank
    
    is_torch = isinstance(A, torch.Tensor)
    
    if not is_torch:
        A = torch.from_numpy(A)
    
    m, n = A.shape
    num_cols = min(num_cols, n)
    num_rows = min(num_rows, m)
    
    # 列采样
    if method == 'uniform':
        # 均匀采样
        col_indices = torch.randperm(n)[:num_cols]
    elif method == 'leverage':
        # 基于杠杆分数的采样
        col_scores = compute_leverage_scores(A, axis=1)
        col_indices = sample_by_scores(col_scores, num_cols)
    else:
        raise ValueError(f"未知采样方法: {method}")
    
    C = A[:, col_indices]
    
    # 行采样
    if method == 'uniform':
        row_indices = torch.randperm(m)[:num_rows]
    elif method == 'leverage':
        row_scores = compute_leverage_scores(A, axis=0)
        row_indices = sample_by_scores(row_scores, num_rows)
    
    R = A[row_indices, :]
    
    # 计算U矩阵
    # U = C⁺ @ A @ R⁺，其中 ⁺ 表示伪逆
    C_pinv = torch.linalg.pinv(C)
    R_pinv = torch.linalg.pinv(R)
    U = C_pinv @ A @ R_pinv
    
    if not is_torch:
        C = C.numpy()
        U = U.numpy()
        R = R.numpy()
    
    return C, U, R


def tensor_cur_decomposition(
    tensor: Union[np.ndarray, torch.Tensor],
    ranks: List[int],
    num_samples: Optional[List[int]] = None
) -> Tuple[List, List, List]:
    """
    张量CUR分解 (tCUR)
    
    将张量分解为核心张量和多个因子矩阵。
    
    参数:
        tensor: 输入张量，形状 (I₁, I₂, ..., Iₙ)
        ranks: 每模的秩 [r₁, r₂, ..., rₙ]
        num_samples: 每模采样数 (默认2*ranks)
        
    返回:
        (factors, core, indices) 元组
        - factors: 因子矩阵列表
        - core: 核心张量
        - indices: 采样索引列表
        
    示例:
        >>> tensor = np.random.randn(10, 20, 30)
        >>> factors, core, indices = tensor_cur_decomposition(
        ...     tensor, ranks=[3, 4, 5]
        ... )
    """
    is_torch = isinstance(tensor, torch.Tensor)
    
    if not is_torch:
        tensor = torch.from_numpy(tensor)
    
    ndim = tensor.ndim
    
    if len(ranks) != ndim:
        raise ValueError(f"ranks长度({len(ranks)})必须匹配张量阶数({ndim})")
    
    if num_samples is None:
        num_samples = [2 * r for r in ranks]
    
    factors = []
    indices_list = []
    
    # 对每个模态进行纤维采样
    for mode in range(ndim):
        # 将张量展开为矩阵
        unfolded = unfold_tensor(tensor, mode)
        
        # 进行CUR分解
        rank = ranks[mode]
        n_samples = min(num_samples[mode], unfolded.shape[1])
        
        # 采样列 (纤维)
        col_indices = torch.randperm(unfolded.shape[1])[:n_samples]
        C = unfolded[:, col_indices]
        
        # 采样行
        row_indices = torch.randperm(unfolded.shape[0])[:rank]
        R = unfolded[row_indices, :]
        
        # 计算U
        C_pinv = torch.linalg.pinv(C)
        R_pinv = torch.linalg.pinv(R)
        U = C_pinv @ unfolded @ R_pinv
        
        factors.append((C, U, R))
        indices_list.append((col_indices, row_indices))
    
    # 构建核心张量 (简化版本)
    core = tensor
    for mode in range(ndim - 1, -1, -1):
        _, U, _ = factors[mode]
        core = n_mode_product(core, U, mode)
    
    if not is_torch:
        factors = [(c.numpy(), u.numpy(), r.numpy()) for c, u, r in factors]
        core = core.numpy()
        indices_list = [(c.numpy(), r.numpy()) for c, r in indices_list]
    
    return factors, core, indices_list


def n_mode_product(
    tensor: Union[np.ndarray, torch.Tensor],
    matrix: Union[np.ndarray, torch.Tensor],
    mode: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    n-模乘积 (Tensor-Matrix Product)
    
    张量在第n模上与矩阵相乘。
    
    参数:
        tensor: 输入张量，形状 (I₁, ..., Iₙ, ..., Iₖ)
        matrix: 矩阵，形状 (J, Iₙ)
        mode: 模态索引 (0-based)
        
    返回:
        结果张量，形状 (I₁, ..., J, ..., Iₖ)
        
    示例:
        >>> tensor = np.random.randn(3, 4, 5)
        >>> matrix = np.random.randn(6, 4)
        >>> result = n_mode_product(tensor, matrix, mode=1)
        >>> print(result.shape)  # (3, 6, 5)
    """
    is_torch = isinstance(tensor, torch.Tensor)
    
    if not is_torch:
        tensor = torch.from_numpy(tensor)
        matrix = torch.from_numpy(matrix)
    
    # 调整张量维度以便矩阵乘法
    # 将mode移到最前
    tensor_permuted = tensor.transpose(mode, 0)
    
    # 重塑为矩阵
    shape = tensor_permuted.shape
    tensor_matrix = tensor_permuted.reshape(shape[0], -1)
    
    # 矩阵乘法
    result_matrix = matrix @ tensor_matrix
    
    # 重塑回张量
    new_shape = (matrix.shape[0],) + shape[1:]
    result = result_matrix.reshape(new_shape)
    
    # 恢复维度顺序
    if mode != 0:
        result = result.transpose(0, mode)
    
    if not is_torch:
        result = result.numpy()
    
    return result


def unfold_tensor(
    tensor: torch.Tensor,
    mode: int
) -> torch.Tensor:
    """
    张量展开 (Matricization)
    
    将张量沿指定模态展开为矩阵。
    
    参数:
        tensor: 输入张量，形状 (I₁, I₂, ..., Iₙ)
        mode: 模态索引
        
    返回:
        展开矩阵，形状 (I_mode, I₁*...*I_{mode-1}*I_{mode+1}*...*Iₙ)
    """
    ndim = tensor.ndim
    
    # 调整维度顺序
    dims = list(range(ndim))
    dims.pop(mode)
    dims.insert(0, mode)
    
    tensor_permuted = tensor.permute(dims)
    
    # 展开
    return tensor_permuted.reshape(tensor.shape[mode], -1)


def fold_tensor(
    matrix: torch.Tensor,
    mode: int,
    shape: Tuple[int, ...]
) -> torch.Tensor:
    """
    张量折叠 (逆展开操作)
    
    将矩阵折叠回张量。
    
    参数:
        matrix: 输入矩阵
        mode: 模态索引
        shape: 目标张量形状
        
    返回:
        折叠后的张量
    """
    ndim = len(shape)
    
    # 重塑为调整维度后的形状
    new_shape = [shape[mode]]
    for i in range(ndim):
        if i != mode:
            new_shape.append(shape[i])
    
    tensor = matrix.reshape(new_shape)
    
    # 恢复原始维度顺序
    dims = list(range(ndim))
    dims.pop(0)
    dims.insert(mode, 0)
    
    return tensor.permute(dims)


def compute_leverage_scores(
    A: torch.Tensor,
    axis: int = 0
) -> torch.Tensor:
    """
    计算杠杆分数 (Leverage Scores)

    用于重要列/行的重要性采样。
    
    参数:
        A: 输入矩阵
        axis: 0计算行分数，1计算列分数
        
    返回:
        杠杆分数
    """
    if axis == 1:
        A = A.T
    
    # 计算SVD
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    # 杠杆分数是U的行范数平方
    leverage = torch.sum(U ** 2, dim=1)
    
    # 归一化
    leverage = leverage / leverage.sum()
    
    return leverage


def sample_by_scores(
    scores: torch.Tensor,
    num_samples: int
) -> torch.Tensor:
    """
    基于分数的重要性采样
    
    参数:
        scores: 采样概率
        num_samples: 采样数
        
    返回:
        采样索引
    """
    n = len(scores)
    num_samples = min(num_samples, n)
    
    # 多项式采样
    indices = torch.multinomial(scores, num_samples, replacement=False)
    
    return indices


def tensor_train_decomposition(
    tensor: Union[np.ndarray, torch.Tensor],
    ranks: List[int]
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    张量列车分解 (Tensor Train)
    
    另一种张量分解方法，用于对比。
    
    参数:
        tensor: 输入张量
        ranks: TT秩
        
    返回:
        TT核列表
    """
    is_torch = isinstance(tensor, torch.Tensor)
    
    if not is_torch:
        tensor = torch.from_numpy(tensor)
    
    cores = []
    remainder = tensor
    
    for i in range(tensor.ndim - 1):
        # SVD分解
        shape = remainder.shape
        matrix = remainder.reshape(ranks[i] * shape[0], -1)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # 截断
        U = U[:, :ranks[i+1]]
        S = S[:ranks[i+1]]
        Vh = Vh[:ranks[i+1], :]
        
        # 保存核
        core = U.reshape(ranks[i], shape[0], ranks[i+1])
        cores.append(core)
        
        # 更新余量
        remainder = (torch.diag(S) @ Vh).reshape(ranks[i+1], *shape[1:])
    
    cores.append(remainder)
    
    if not is_torch:
        cores = [c.numpy() for c in cores]
    
    return cores


if __name__ == "__main__":
    """
    张量操作测试
    """
    print("="*60)
    print("张量操作模块测试")
    print("="*60)
    
    # 测试矩阵CUR分解
    print("\n1. 测试矩阵CUR分解...")
    A = torch.randn(100, 50)
    rank = 10
    C, U, R = cur_decomposition(A, rank=rank, method='uniform')
    A_approx = C @ U @ R
    error = torch.norm(A - A_approx).item() / torch.norm(A).item()
    print(f"   原始矩阵: {A.shape}")
    print(f"   C: {C.shape}, U: {U.shape}, R: {R.shape}")
    print(f"   相对重构误差: {error:.4f}")
    
    # 测试n-模乘积
    print("\n2. 测试n-模乘积...")
    tensor = torch.randn(3, 4, 5)
    matrix = torch.randn(6, 4)
    result = n_mode_product(tensor, matrix, mode=1)
    print(f"   输入张量: {tensor.shape}")
    print(f"   矩阵: {matrix.shape}")
    print(f"   结果: {result.shape}")
    
    # 测试张量展开
    print("\n3. 测试张量展开...")
    unfolded = unfold_tensor(tensor, mode=1)
    print(f"   展开矩阵: {unfolded.shape}")
    
    # 测试张量CUR分解
    print("\n4. 测试张量CUR分解...")
    tensor_3d = torch.randn(10, 20, 30)
    factors, core, indices = tensor_cur_decomposition(
        tensor_3d, ranks=[3, 4, 5]
    )
    print(f"   输入张量: {tensor_3d.shape}")
    print(f"   因子矩阵数: {len(factors)}")
    print(f"   核心张量: {core.shape}")
    
    # 测试杠杆分数
    print("\n5. 测试杠杆分数...")
    scores = compute_leverage_scores(A, axis=1)
    print(f"   分数范围: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"   分数和: {scores.sum():.4f}")
    
    print("\n" + "="*60)
    print("✅ 张量操作测试完成!")
    print("="*60)
