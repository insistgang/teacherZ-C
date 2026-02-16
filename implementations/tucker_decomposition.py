"""
Tucker分解加速算法实现
Low-Rank Tucker Approximation with Sketching

论文: SIAM Journal on Scientific Computing / Journal of Machine Learning Research
核心贡献: 随机Sketching + HOOI迭代

包含:
    - 随机Sketching
    - Leverage Score采样
    - HOOI (Higher-Order Orthogonal Iteration)
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.linalg import qr, svd


class TensorOperations:
    """张量操作工具类"""

    @staticmethod
    def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
        """
        Mode-n乘积: Y = X ×_n A

        参数:
            tensor: 输入张量，形状 (I_1, I_2, ..., I_N)
            matrix: 矩阵，形状 (J, I_n)
            mode: 模态索引 (0-indexed)

        返回:
            结果张量，形状 (I_1, ..., I_{n-1}, J, I_{n+1}, ..., I_N)
        """
        ndim = tensor.ndim
        perm = list(range(ndim))
        perm[0], perm[mode] = perm[mode], perm[0]

        tensor_t = np.transpose(tensor, perm)

        result = np.tensordot(matrix, tensor_t, axes=([1], [0]))

        inv_perm = np.argsort(perm)
        result = np.transpose(result, inv_perm)

        return result

    @staticmethod
    def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
        """
        Mode-n展开: 将张量展开为矩阵

        参数:
            tensor: 输入张量
            mode: 模态索引

        返回:
            展开矩阵，形状 (I_n, prod(I_k for k != n))
        """
        ndim = tensor.ndim
        perm = [mode] + [i for i in range(ndim) if i != mode]
        tensor_t = np.transpose(tensor, perm)

        return tensor_t.reshape(tensor.shape[mode], -1)

    @staticmethod
    def fold(matrix: np.ndarray, mode: int, shape: Tuple[int, ...]) -> np.ndarray:
        """
        矩阵折叠为张量 (unfold的逆操作)

        参数:
            matrix: 展开矩阵
            mode: 模态索引
            shape: 原始张量形状

        返回:
            张量
        """
        ndim = len(shape)
        new_shape = [shape[mode]] + [shape[i] for i in range(ndim) if i != mode]
        tensor_t = matrix.reshape(new_shape)

        perm = [0] * ndim
        perm[mode] = 0
        idx = 1
        for i in range(ndim):
            if i != mode:
                perm[i] = idx
                idx += 1

        inv_perm = np.argsort(perm)
        return np.transpose(tensor_t, inv_perm)


class RandomizedSVD:
    """随机SVD"""

    def __init__(self, n_oversamples: int = 10, n_power_iter: int = 2):
        self.n_oversamples = n_oversamples
        self.n_power_iter = n_power_iter

    def fit(
        self, A: np.ndarray, rank: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        随机SVD分解

        参数:
            A: 输入矩阵 (m, n)
            rank: 目标秩

        返回:
            U: 左奇异向量 (m, rank)
            S: 奇异值 (rank,)
            Vh: 右奇异向量转置 (rank, n)
        """
        m, n = A.shape
        k = min(rank + self.n_oversamples, min(m, n))

        Omega = np.random.randn(n, k)
        Y = A @ Omega

        for _ in range(self.n_power_iter):
            Y = A @ (A.T @ Y)

        Q, _ = qr(Y, mode="economic")

        B = Q.T @ A
        U_tilde, S, Vh = svd(B, full_matrices=False)

        U = Q @ U_tilde

        return U[:, :rank], S[:rank], Vh[:rank, :]


class SketchingTucker:
    """基于Sketching的Tucker分解"""

    def __init__(
        self, ranks: List[int], sketch_multipliers: float = 2.0, seed: int = 42
    ):
        """
        初始化

        参数:
            ranks: 目标Tucker秩 [R_1, R_2, ..., R_N]
            sketch_multipliers: Sketch尺寸 = multiplier × rank
            seed: 随机种子
        """
        self.ranks = ranks
        self.N = len(ranks)
        self.sketch_multipliers = sketch_multipliers
        np.random.seed(seed)

    def fit(
        self, tensor: np.ndarray, max_iter: int = 50, tol: float = 1e-6
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        执行基于Sketching的Tucker分解

        参数:
            tensor: 输入张量
            max_iter: HOOI最大迭代次数
            tol: 收敛阈值

        返回:
            core: 核张量
            factors: 因子矩阵列表
        """
        shape = tensor.shape
        sketch_sizes = [int(self.sketch_multipliers * r) for r in self.ranks]

        sketch_matrices = self._build_sketch_matrices(shape, sketch_sizes)

        range_sketches, core_sketch = self._build_sketches(tensor, sketch_matrices)

        factors = self._extract_factors_from_sketch(range_sketches)

        core, factors = self._hooi_refinement(tensor, factors, max_iter, tol)

        return core, factors

    def _build_sketch_matrices(
        self, shape: Tuple[int, ...], sketch_sizes: List[int]
    ) -> List[np.ndarray]:
        """构建Gaussian随机Sketching矩阵"""
        matrices = []
        for I_n, s_n in zip(shape, sketch_sizes):
            S = np.random.randn(s_n, I_n) / np.sqrt(s_n)
            matrices.append(S)
        return matrices

    def _build_sketches(
        self, tensor: np.ndarray, sketch_matrices: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        构建Range Sketches和Core Sketch

        Range Sketch Y^{(n)} = X ×_1 S_1 ... ×_{n-1} S_{n-1} ×_{n+1} S_{n+1} ... ×_N S_N
        Core Sketch Z = X ×_1 S_1 ×_2 S_2 ... ×_N S_N
        """
        N = tensor.ndim
        shape = tensor.shape
        sketch_sizes = [S.shape[0] for S in sketch_matrices]

        core_sketch = tensor.copy()
        for n in range(N):
            core_sketch = TensorOperations.mode_n_product(
                core_sketch, sketch_matrices[n], n
            )

        range_sketches = []
        for n in range(N):
            Y_n = tensor.copy()
            for m in range(N):
                if m != n:
                    Y_n = TensorOperations.mode_n_product(Y_n, sketch_matrices[m], m)
            range_sketches.append(Y_n)

        return range_sketches, core_sketch

    def _extract_factors_from_sketch(
        self, range_sketches: List[np.ndarray]
    ) -> List[np.ndarray]:
        """从Range Sketch中提取因子矩阵"""
        factors = []
        rand_svd = RandomizedSVD()

        for n, Y_n in enumerate(range_sketches):
            Y_matrix = TensorOperations.unfold(Y_n, n)

            U, _, _ = rand_svd.fit(Y_matrix, self.ranks[n])
            factors.append(U)

        return factors

    def _hooi_refinement(
        self,
        tensor: np.ndarray,
        factors_init: List[np.ndarray],
        max_iter: int,
        tol: float,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        HOOI迭代优化

        Higher-Order Orthogonal Iteration:
        交替优化每个因子矩阵，保持其他固定
        """
        factors = [F.copy() for F in factors_init]
        prev_error = float("inf")

        for iteration in range(max_iter):
            for n in range(self.N):
                Y = tensor.copy()
                for m in range(self.N):
                    if m != n:
                        Y = TensorOperations.mode_n_product(Y, factors[m].T, m)

                Y_matrix = TensorOperations.unfold(Y, n)
                U, _, _ = svd(Y_matrix, full_matrices=False)
                factors[n] = U[:, : self.ranks[n]]

            core = tensor.copy()
            for n in range(self.N):
                core = TensorOperations.mode_n_product(core, factors[n].T, n)

            reconstructed = core.copy()
            for n in range(self.N):
                reconstructed = TensorOperations.mode_n_product(
                    reconstructed, factors[n], n
                )

            error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

            if abs(prev_error - error) < tol:
                break
            prev_error = error

        return core, factors


class HOOIDecomposition:
    """标准HOOI分解"""

    def __init__(self, ranks: List[int], max_iter: int = 100, tol: float = 1e-6):
        self.ranks = ranks
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, tensor: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        执行HOOI分解

        参数:
            tensor: 输入张量

        返回:
            core: 核张量
            factors: 因子矩阵列表
        """
        N = tensor.ndim
        factors = self._initialize_factors(tensor)

        prev_error = float("inf")

        for iteration in range(self.max_iter):
            for n in range(N):
                Y = tensor.copy()
                for m in range(N):
                    if m != n:
                        Y = TensorOperations.mode_n_product(Y, factors[m].T, m)

                Y_matrix = TensorOperations.unfold(Y, n)
                U, _, _ = svd(Y_matrix, full_matrices=False)
                factors[n] = U[:, : self.ranks[n]]

            core = tensor.copy()
            for n in range(N):
                core = TensorOperations.mode_n_product(core, factors[n].T, n)

            reconstructed = core.copy()
            for n in range(N):
                reconstructed = TensorOperations.mode_n_product(
                    reconstructed, factors[n], n
                )

            error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)

            if abs(prev_error - error) < self.tol:
                print(f"HOOI收敛于第 {iteration + 1} 次迭代，误差: {error:.6f}")
                break
            prev_error = error

        return core, factors

    def _initialize_factors(self, tensor: np.ndarray) -> List[np.ndarray]:
        """使用HOSVD初始化因子矩阵"""
        N = tensor.ndim
        factors = []

        for n in range(N):
            X_n = TensorOperations.unfold(tensor, n)
            U, _, _ = svd(X_n, full_matrices=False)
            factors.append(U[:, : self.ranks[n]])

        return factors


def reconstruct_tucker(core: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    """
    从Tucker分解重构张量

    X ≈ G ×_1 A_1 ×_2 A_2 ... ×_N A_N
    """
    result = core.copy()
    for n, A in enumerate(factors):
        result = TensorOperations.mode_n_product(result, A, n)
    return result


def demo_tucker():
    """Tucker分解演示"""
    print("=" * 60)
    print("Tucker分解加速演示")
    print("=" * 60)

    np.random.seed(42)

    I, J, K = 50, 40, 30
    R = [5, 4, 3]

    print(f"\n原始张量尺寸: ({I}, {J}, {K})")
    print(f"目标Tucker秩: {R}")

    core_true = np.random.randn(*R)
    A1, _ = qr(np.random.randn(I, R[0]), mode="economic")
    A2, _ = qr(np.random.randn(J, R[1]), mode="economic")
    A3, _ = qr(np.random.randn(K, R[2]), mode="economic")

    X = core_true.copy()
    X = TensorOperations.mode_n_product(X, A1, 0)
    X = TensorOperations.mode_n_product(X, A2, 1)
    X = TensorOperations.mode_n_product(X, A3, 2)

    X += 0.01 * np.random.randn(*X.shape)

    print("\n--- 方法1: 标准HOOI分解 ---")
    import time

    hooi = HOOIDecomposition(ranks=R)
    start = time.time()
    core_hooi, factors_hooi = hooi.fit(X)
    time_hooi = time.time() - start

    X_hooi = reconstruct_tucker(core_hooi, factors_hooi)
    error_hooi = np.linalg.norm(X - X_hooi) / np.linalg.norm(X)

    print(f"时间: {time_hooi:.4f} 秒")
    print(f"相对重构误差: {error_hooi:.6f}")
    print(f"核张量形状: {core_hooi.shape}")

    print("\n--- 方法2: 基于Sketching的Tucker分解 ---")

    sketchy = SketchingTucker(ranks=R, sketch_multipliers=2.0)
    start = time.time()
    core_sketch, factors_sketch = sketchy.fit(X)
    time_sketch = time.time() - start

    X_sketch = reconstruct_tucker(core_sketch, factors_sketch)
    error_sketch = np.linalg.norm(X - X_sketch) / np.linalg.norm(X)

    print(f"时间: {time_sketch:.4f} 秒")
    print(f"相对重构误差: {error_sketch:.6f}")
    print(f"核张量形状: {core_sketch.shape}")

    print("\n--- 压缩比分析 ---")
    original_params = np.prod(X.shape)
    tucker_params = np.prod(core_hooi.shape) + sum(
        A.shape[0] * A.shape[1] for A in factors_hooi
    )
    compression_ratio = original_params / tucker_params

    print(f"原始参数: {original_params:,}")
    print(f"Tucker参数: {tucker_params:,}")
    print(f"压缩比: {compression_ratio:.2f}:1")

    return core_hooi, factors_hooi


if __name__ == "__main__":
    demo_tucker()
