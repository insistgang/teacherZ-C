"""
图割与凸松弛分割方法完整实现
对比三种方法：图割、Chan-Vese、Split Bregman
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color
from scipy import ndimage
import time
import sys
import io as sys_io

# Windows编码修复
if sys.platform.startswith('win'):
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = sys_io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# 方法1：图割分割（使用网络流）
# ============================================================================

def graph_cut_segmentation_simple(f, lambda_=1.0, sigma=0.1):
    """
    简化的图割分割（基于NetworkX）

    使用最小割/最大流进行二值分割

    Parameters:
    -----------
    f : ndarray
        输入灰度图像
    lambda_ : float
        平滑参数（控制边界长度）
    sigma : float
        用于计算边权重的参数

    Returns:
    --------
    segmentation : ndarray
        二值分割结果（0=背景，1=前景）
    """
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX未安装，使用: pip install networkx")
        return None

    H, W = f.shape
    n_pixels = H * W

    print("\n[方法1] 图割分割")
    print("  数学基础：组合优化，全局最优")
    print("  构造图：源点、汇点、像素节点")

    # 创建有向图
    G = nx.DiGraph()

    # 添加顶点
    G.add_node('s')  # 源点
    G.add_node('t')  # 汇点

    start_time = time.time()

    # 估计前景和背景均值（简单的阈值）
    threshold = np.mean(f)
    mu_bg = np.mean(f[f <= threshold])
    mu_fg = np.mean(f[f > threshold])

    # 添加像素节点和边
    print(f"  构造图...")

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            pixel_val = f[i, j]

            # 添加像素节点
            G.add_node(idx)

            # t-links：数据项（拟合误差）
            w_bg = (pixel_val - mu_bg) ** 2 + 1e-6
            w_fg = (pixel_val - mu_fg) ** 2 + 1e-6

            G.add_edge('s', idx, capacity=w_fg)
            G.add_edge(idx, 't', capacity=w_bg)

            # n-links：平滑项（只连接右和下邻居）
            if i < H - 1:  # 下邻居
                idx_down = (i + 1) * W + j
                weight = lambda_ * np.exp(-(pixel_val - f[i+1, j])**2 / (2*sigma**2))
                G.add_edge(idx, idx_down, capacity=weight)
                G.add_edge(idx_down, idx, capacity=weight)

            if j < W - 1:  # 右邻居
                idx_right = i * W + (j + 1)
                weight = lambda_ * np.exp(-(pixel_val - f[i, j+1])**2 / (2*sigma**2))
                G.add_edge(idx, idx_right, capacity=weight)
                G.add_edge(idx_right, idx, capacity=weight)

    # 计算最小割
    print(f"  计算最小割（图大小：{G.number_of_nodes()}节点，{G.number_of_edges()}边）...")

    cut_value, partition = nx.minimum_cut(G, 's', 't')

    reachable, _ = partition

    # 提取分割
    segmentation = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if idx in reachable:
                segmentation[i, j] = 1

    elapsed = time.time() - start_time
    print(f"  完成：{elapsed:.2f}秒")
    print(f"  最小割容量：{cut_value:.2f}")

    return segmentation


# ============================================================================
# 方法2：凸松弛（连续松弛 + Split Bregman）
# ============================================================================

def split_bregman_segmentation(f, lambda_=0.1, mu=1.0, max_iter=100):
    """
    使用Split Bregman求解ROF + 分割

    凸松弛方法：
    1. 松弛标签为连续值 [0,1]
    2. 使用Split Bregman优化
    3. 阈值化得到二值分割

    Parameters:
    -----------
    f : ndarray
        输入图像
    lambda_ : float
        保真参数
    mu : float
        增广拉格朗日参数
    max_iter : int
        最大迭代次数

    Returns:
    --------
    segmentation : ndarray
        二值分割结果
    """
    print("\n[方法2] Split Bregman凸松弛")
    print("  数学基础：凸优化 + 变量分裂")
    print("  算法：交替优化三个子问题")

    H, W = f.shape

    # 初始化
    u = f.copy()
    dx = np.zeros_like(f)
    dy = np.zeros_like(f)
    bx = np.zeros_like(f)
    by = np.zeros_like(f)

    start_time = time.time()

    for k in range(max_iter):
        u_old = u.copy()

        # u子问题：(λI - μΔ)u = λf + μ·div(dx-bx, dy-by)
        # 使用FFT快速求解
        from scipy.fft import fft2, ifft2

        # 计算散度
        div_term = np.zeros_like(f)
        div_term[:, :-1] += dx[:, :-1]
        div_term[:, 1:]  -= dx[:, :-1]
        div_term[:-1, :] += dy[:-1, :]
        div_term[1:, :]  -= dy[:-1, :]

        # bx和by的维度匹配
        bx_pad = np.zeros_like(f)
        bx_pad[:, :-1] = bx[:, :-1]
        by_pad = np.zeros_like(f)
        by_pad[:-1, :] = by[:-1, :]

        rhs = lambda_ * f + mu * (div_term - bx_pad - by_pad)

        # FFT求解
        rhs_hat = fft2(rhs)
        # 拉普拉斯算子的频域表示
        y, x = np.mgrid[:H, :W]
        denom = lambda_ + mu * (4 - 2*np.cos(2*np.pi*x/W) - 2*np.cos(2*np.pi*y/H))

        u_hat = rhs_hat / (denom + 1e-10)
        u = np.real(ifft2(u_hat))

        # v子问题：收缩
        ux = np.zeros_like(f)
        uy = np.zeros_like(f)
        ux[:, :-1] = u[:, 1:] - u[:, :-1]
        uy[:-1, :] = u[1:, :] - u[:-1, :]

        # 收缩函数
        shrink = lambda x, th: x / np.abs(x) * np.maximum(np.abs(x) - th, 0)

        dx_new = shrink(ux[:, :-1] + bx[:, :-1], 1/mu)
        dy_new = shrink(uy[:-1, :] + by[:-1, :], 1/mu)

        # b更新
        bx[:, :-1] += ux[:, :-1] - dx_new
        by[:-1, :] += uy[:-1, :] - dy_new

        dx[:, :-1] = dx_new
        dy[:-1, :] = dy_new

        # 检查收敛
        if k > 0 and k % 10 == 0:
            rel_change = np.linalg.norm(u - u_old) / np.linalg.norm(u_old)
            if rel_change < 1e-4:
                print(f"  收敛于迭代 {k}")
                break

    # 阈值化
    segmentation = (u > np.mean(u)).astype(np.uint8)

    elapsed = time.time() - start_time
    print(f"  完成：{elapsed:.2f}秒，{k}次迭代")

    return segmentation


# ============================================================================
# 方法3：简化的Chan-Vese（用于对比）
# ============================================================================

def simple_chan_vese(f, max_iter=50):
    """
    简化的Chan-Vese分割

    Parameters:
    -----------
    f : ndarray
        输入图像
    max_iter : int
        最大迭代次数

    Returns:
    --------
    segmentation : ndarray
        二值分割结果
    """
    print("\n[方法3] Chan-Vese活动轮廓")
    print("  数学基础：变分法 + 水平集")

    H, W = f.shape

    # 初始化水平集
    y, x = np.mgrid[:H, :W]
    phi = np.sqrt((x - W/2)**2 + (y - H/2)**2) - min(H, W) / 4

    start_time = time.time()

    for k in range(max_iter):
        # 计算区域均值
        mask = phi > 0
        c1 = np.mean(f[mask]) if np.any(mask) else 0
        c2 = np.mean(f[~mask]) if np.any(~mask) else 0

        # 计算曲率
        phi_y, phi_x = np.gradient(phi)
        phi_norm = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)

        # 简化的曲率计算
        kappa = (phi_xx := np.gradient(phi_x, axis=1)) * phi_y**2 - \
                2 * (phi_xy := np.gradient(phi_x, axis=0)) * phi_x * phi_y + \
                (phi_yy := np.gradient(phi_y, axis=0)) * phi_x**2
        kappa = kappa / (phi_norm**3 + 1e-10)

        # 演化
        delta = (1e-6) / (phi**2 + 1e-12)
        force = 0.01 * kappa - 1.0 * (f - c1)**2 + 1.0 * (f - c2)**2

        phi = phi + 0.1 * delta * force

        if k % 10 == 0 and k > 0:
            # 重新初始化
            phi = phi / np.sqrt(phi**2 + 1)

    segmentation = (phi > 0).astype(np.uint8)

    elapsed = time.time() - start_time
    print(f"  完成：{elapsed:.2f}秒，{max_iter}次迭代")

    return segmentation


# ============================================================================
# 主函数：对比实验
# ============================================================================

def main():
    print("=" * 70)
    print("图割与凸松弛方法对比实验")
    print("=" * 70)

    # 准备测试图像
    print("\n[准备] 加载测试图像...")
    f = img_as_float(data.coins())

    print(f"  图像：coins")
    print(f"  大小：{f.shape}")

    # 运行三种方法
    results = {}

    # 方法1：图割（如果NetworkX可用）
    try:
        seg_gc = graph_cut_segmentation_simple(f, lambda_=0.5, sigma=0.1)
        if seg_gc is not None:
            results['Graph Cut'] = seg_gc
    except Exception as e:
        print(f"  图割失败: {e}")

    # 方法2：Split Bregman
    seg_sb = split_bregman_segmentation(f, lambda_=0.1, mu=1.0, max_iter=50)
    results['Split Bregman'] = seg_sb

    # 方法3：Chan-Vese
    seg_cv = simple_chan_vese(f, max_iter=100)
    results['Chan-Vese'] = seg_cv

    # 可视化对比
    print("\n[可视化] 生成对比图...")
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))

    for idx, (name, seg) in enumerate(results.items()):
        # 原始图像
        axes[0, idx].imshow(f, cmap='gray')
        axes[0, idx].set_title(f'{name}\n原始图像', fontsize=12)
        axes[0, idx].axis('off')

        # 分割结果
        axes[1, idx].imshow(seg, cmap='gray')
        axes[1, idx].set_title('分割结果', fontsize=12)
        axes[1, idx].axis('off')

    plt.tight_layout()
    plt.savefig('convex_relaxation_comparison.png', dpi=150, bbox_inches='tight')
    print("  保存到：convex_relaxation_comparison.png")

    # 统计信息
    print("\n" + "=" * 70)
    print("方法对比")
    print("=" * 70)

    for name, seg in results.items():
        fg_pixels = np.sum(seg)
        bg_pixels = seg.size - fg_pixels
        fg_ratio = fg_pixels / seg.size * 100

        print(f"\n{name}:")
        print(f"  前景像素: {fg_pixels} ({fg_ratio:.1f}%)")
        print(f"  背景像素: {bg_pixels} ({100-fg_ratio:.1f}%)")

    print("\n" + "=" * 70)
    print("观察要点")
    print("=" * 70)
    print("1. 图割：全局最优，但需要构造图（内存大）")
    print("2. Split Bregman：凸优化，收敛快，数值稳定")
    print("3. Chan-Vese：变分方法，易于理解，但可能局部最优")

    print("\n结论:")
    print("  - 凸松弛提供全局最优的理论保证")
    print("  - 图割在小规模问题上效果好")
    print("  - Split Bregman适合大规模问题")


if __name__ == "__main__":
    main()
