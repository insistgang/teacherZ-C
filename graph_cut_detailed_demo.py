"""
图割算法完整演示
从基础到高级的完整实现
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
import time
import sys
import io as sys_io

# Windows编码修复
if sys.platform.startswith('win'):
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = sys_io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================================================
# 第一部分：基础的图类和Ford-Fulkerson算法
# ============================================================================

class SimpleGraph:
    """
    简单的图类，实现最大流算法
    使用邻接表表示
    """

    def __init__(self, n_nodes):
        """
        初始化图

        Parameters:
        -----------
        n_nodes : int
            节点数量（不包括源点和汇点）
            节点编号: 0, 1, ..., n_nodes-1
            源点s = -1
            汇点t = n_nodes
        """
        self.n = n_nodes
        self.s = -1
        self.t = n_nodes
        self.total_nodes = n_nodes + 2
        self.capacity = {}  # 容量字典
        self.flow = {}      # 流量字典

    def add_edge(self, u, v, cap):
        """
        添加有向边

        Parameters:
        -----------
        u, v : int
            节点编号
        cap : float
            边的容量
        """
        key = (u, v)
        if key not in self.capacity:
            self.capacity[key] = 0
            self.capacity[(v, u)] = 0  # 反向边，初始容量为0
            self.flow[key] = 0
            self.flow[(v, u)] = 0

        self.capacity[key] += cap

    def get_residual_capacity(self, u, v):
        """获取残量容量"""
        # 如果边不存在，容量为0
        if (u, v) not in self.capacity:
            return 0
        if (u, v) not in self.flow:
            return self.capacity[(u, v)]
        return self.capacity[(u, v)] - self.flow[(u, v)]

    def bfs_find_path(self, parent):
        """
        BFS寻找增广路径

        Parameters:
        -----------
        parent : dict
            父节点字典（输出）

        Returns:
        --------
        bool : 是否找到路径
        """
        # 初始化
        visited = {self.s}
        queue = [self.s]

        # 清空parent
        parent.clear()

        # BFS
        while queue:
            u = queue.pop(0)

            # 检查所有邻居
            for v in range(self.total_nodes):
                if v in visited:
                    continue

                # 检查是否有残量容量
                if self.get_residual_capacity(u, v) > 0:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)

                    if v == self.t:
                        return True  # 找到汇点

        return False  # 没有增广路径

    def find_augmenting_path(self):
        """寻找增广路径"""
        parent = {}
        if not self.bfs_find_path(parent):
            return None, 0  # 没有增广路径

        # 回溯计算路径上的最小残量容量
        path_flow = float('inf')
        v = self.t
        path = [v]

        while v != self.s:
            u = parent[v]
            path_flow = min(path_flow, self.get_residual_capacity(u, v))
            path.append(u)
            v = u

        path.reverse()
        return path, path_flow

    def max_flow(self):
        """
        Ford-Fulkerson算法计算最大流

        Returns:
        --------
        max_flow : float
            最大流的值
        """
        max_flow = 0
        iteration = 0

        print("\n[Ford-Fulkerson算法]")
        print("  寻找增广路径...")

        while True:
            # 寻找增广路径
            path, path_flow = self.find_augmenting_path()

            if path is None:
                break  # 没有增广路径，算法结束

            # 更新流量
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                self.flow[(u, v)] += path_flow
                self.flow[(v, u)] -= path_flow

            max_flow += path_flow

            if iteration % 10 == 0 or path_flow > 0:
                print(f"  迭代 {iteration}: 增广路径流量={path_flow:.2f}, 总流量={max_flow:.2f}")

            iteration += 1

        print(f"  收敛！总迭代: {iteration}, 最大流: {max_flow:.2f}")

        return max_flow

    def get_min_cut(self):
        """
        从最大流获取最小割

        Returns:
        --------
        S : set
            源点侧的节点集合
        T : set
            汇点侧的节点集合
        """
        # 从源点出发，找残量网络中可达的节点
        S = set()
        queue = [self.s]
        visited = {self.s}

        while queue:
            u = queue.pop(0)
            S.add(u)

            for v in range(self.total_nodes):
                if v in visited:
                    continue

                if self.get_residual_capacity(u, v) > 0:
                    visited.add(v)
                    queue.append(v)

        # T = V \ S
        T = set(range(self.total_nodes)) - S

        return S, T

    def get_cut_capacity(self, S, T):
        """计算割的容量"""
        capacity = 0
        for u in S:
            for v in T:
                if (u, v) in self.capacity:
                    capacity += self.capacity[(u, v)]
        return capacity


# ============================================================================
# 第二部分：图像分割的图构造
# ============================================================================

def build_graph_for_segmentation(image, lambda_=1.0):
    """
    为图像分割构造图

    Parameters:
    -----------
    image : ndarray (H, W)
        输入灰度图像
    lambda_ : float
        平滑参数

    Returns:
    --------
    graph : SimpleGraph
        构造好的图
    """
    H, W = image.shape
    n_pixels = H * W

    print(f"\n[构造图]")
    print(f"  图像大小: {H}×{W} = {n_pixels}像素")
    print(f"  节点数: {n_pixels + 2} (包括源点和汇点)")

    # 创建图
    graph = SimpleGraph(n_pixels)

    # 估计前景和背景均值
    threshold = np.mean(image)
    mu_bg = np.mean(image[image <= threshold])
    mu_fg = np.mean(image[image > threshold])

    print(f"  背景均值: {mu_bg:.3f}")
    print(f"  前景均值: {mu_fg:.3f}")

    # 添加边
    n_edges = 0

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            pixel_val = image[i, j]

            # t-links: 数据项
            w_bg = (pixel_val - mu_bg) ** 2 + 1e-6
            w_fg = (pixel_val - mu_fg) ** 2 + 1e-6

            graph.add_edge(graph.s, idx, w_fg)  # s -> 像素
            graph.add_edge(idx, graph.t, w_bg)  # 像素 -> t
            n_edges += 2

            # n-links: 平滑项（只连接右和下邻居）
            if i < H - 1:  # 下邻居
                idx_down = (i + 1) * W + j
                weight = lambda_ * np.exp(-(pixel_val - image[i+1, j])**2)
                graph.add_edge(idx, idx_down, weight)
                n_edges += 1

            if j < W - 1:  # 右邻居
                idx_right = i * W + (j + 1)
                weight = lambda_ * np.exp(-(pixel_val - image[i, j+1])**2)
                graph.add_edge(idx, idx_right, weight)
                n_edges += 1

    print(f"  边数: {n_edges}")

    return graph


def graph_cut_segmentation_custom(image, lambda_=1.0):
    """
    使用自定义图进行图像分割

    Parameters:
    -----------
    image : ndarray
        输入图像
    lambda_ : float
        平滑参数

    Returns:
    --------
    segmentation : ndarray
        二值分割结果
    """
    H, W = image.shape

    # 构造图
    graph = build_graph_for_segmentation(image, lambda_)

    # 计算最大流
    start_time = time.time()
    max_flow = graph.max_flow()
    elapsed = time.time() - start_time

    print(f"  用时: {elapsed:.2f}秒")

    # 获取最小割
    S, T = graph.get_min_cut()

    # 验证 max flow = min cut
    cut_capacity = graph.get_cut_capacity(S, T)
    print(f"\n[验证]")
    print(f"  最大流: {max_flow:.2f}")
    print(f"  最小割容量: {cut_capacity:.2f}")
    print(f"  差异: {abs(max_flow - cut_capacity):.6f}")

    # 提取分割
    segmentation = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        for j in range(W):
            idx = i * W + j
            if idx in S:  # 在源点侧
                segmentation[i, j] = 1
            else:  # 在汇点侧
                segmentation[i, j] = 0

    return segmentation


# ============================================================================
# 第三部分：使用PyMaxflow的快速实现
# ============================================================================

def graph_cut_with_pymaxflow(image, lambda_=1.0):
    """
    使用PyMaxflow库进行图割分割（推荐）

    Parameters:
    -----------
    image : ndarray
        输入图像
    lambda_ : float
        平滑参数

    Returns:
    --------
    segmentation : ndarray
        分割结果
    """
    try:
        import maxflow
    except ImportError:
        print("PyMaxflow未安装，使用: pip install PyMaxflow")
        return None

    H, W = image.shape

    print(f"\n[PyMaxflow算法]")
    print(f"  数学基础：优化的Push-Relabel算法")

    # 创建图
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((H, W))

    # 估计均值
    threshold = np.mean(image)
    mu_bg = np.mean(image[image <= threshold])
    mu_fg = np.mean(image[image > threshold])

    # 添加t-links和n-links
    structure = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    weights = np.zeros((3, 3))
    weights[0, 1] = lambda_
    weights[1, 0] = lambda_
    weights[1, 2] = lambda_
    weights[2, 1] = lambda_

    start_time = time.time()

    for i in range(H):
        for j in range(W):
            # 数据项
            w_bg = (image[i, j] - mu_bg) ** 2
            w_fg = (image[i, j] - mu_fg) ** 2
            g.add_tedge(nodeids[i, j], w_fg, w_bg)

    # 添加n-links
    g.add_grid_edges(nodeids, weights=weights, structure=structure,
                     symmetric=True)

    # 计算最大流
    g.maxflow()
    elapsed = time.time() - start_time

    print(f"  用时: {elapsed:.2f}秒")

    # 获取分割
    sgm = g.get_grid_segments(nodeids)
    segmentation = sgm.reshape((H, W)).astype(np.uint8)

    return segmentation


# ============================================================================
# 第四部分：对比实验
# ============================================================================

def compare_methods():
    """对比不同的图割实现"""
    print("=" * 70)
    print("图割算法对比实验")
    print("=" * 70)

    # 准备测试图像
    print("\n[准备] 加载测试图像...")
    f = img_as_float(data.camera())

    # 裁剪为小图像（自定义算法较慢）
    f_small = f[100:200, 100:200]

    print(f"  图像: camera (裁剪后 {f_small.shape})")

    results = {}

    # 方法1: 自定义Ford-Fulkerson
    print("\n" + "=" * 70)
    seg_custom = graph_cut_segmentation_custom(f_small, lambda_=0.5)
    results['Custom Ford-Fulkerson'] = seg_custom

    # 方法2: PyMaxflow
    print("\n" + "=" * 70)
    seg_pymaxflow = graph_cut_with_pymaxflow(f_small, lambda_=0.5)
    if seg_pymaxflow is not None:
        results['PyMaxflow'] = seg_pymaxflow

    # 可视化
    print("\n[可视化] 生成对比图...")
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5*(n_methods+1), 10))

    # 原始图像
    axes[0, 0].imshow(f_small, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')

    axes[1, 0].axis('off')

    for idx, (name, seg) in enumerate(results.items()):
        col = idx + 1

        axes[0, col].imshow(seg, cmap='gray')
        axes[0, col].set_title(f'{name}\n分割结果')
        axes[0, col].axis('off')

        # 叠加边界
        axes[1, col].imshow(f_small, cmap='gray')
        from skimage import segmentation as seg_module
        boundaries = seg_module.find_boundaries(seg, mode='thick')
        axes[1, col].imshow(boundaries, cmap='jet', alpha=0.5)
        axes[1, col].set_title('边界叠加')
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig('graph_cut_comparison.png', dpi=150, bbox_inches='tight')
    print("  保存到: graph_cut_comparison.png")

    # 统计
    print("\n" + "=" * 70)
    print("方法对比")
    print("=" * 70)

    for name, seg in results.items():
        fg = np.sum(seg)
        ratio = fg / seg.size * 100
        print(f"\n{name}:")
        print(f"  前景比例: {ratio:.1f}%")
        print(f"  前景像素: {fg}")


# ============================================================================
# 第五部分：教学示例：手动计算一个小网络
# ============================================================================

def manual_example():
    """手动计算一个简单的图割示例"""
    print("\n" + "=" * 70)
    print("教学示例：手动计算最大流")
    print("=" * 70)

    # 创建一个简单的图
    #     s
    #    / \
    #   3   2
    #  /     \
    # a       b
    #  \     /
    #   2   3
    #    \ /
    #     t

    print("\n网络结构:")
    print("    s")
    print("   / \\")
    print("  3   2")
    print(" /     \\")
    print("a       b")
    print(" \\     /")
    print("  2   3")
    print("   \\ /")
    print("    t")

    graph = SimpleGraph(2)  # 节点: a(0), b(1), s(-1), t(2)

    # 添加边
    graph.add_edge(graph.s, 0, 3)  # s -> a
    graph.add_edge(graph.s, 1, 2)  # s -> b
    graph.add_edge(0, graph.t, 2)  # a -> t
    graph.add_edge(1, graph.t, 3)  # b -> t

    print("\n边和容量:")
    print("  s -> a: 3")
    print("  s -> b: 2")
    print("  a -> t: 2")
    print("  b -> t: 3")

    # 计算最大流
    print("\n[Ford-Fulkerson步骤]")
    max_flow = graph.max_flow()

    # 获取最小割
    S, T = graph.get_min_cut()

    print(f"\n[结果]")
    print(f"  最大流: {max_flow}")
    print(f"  源点侧 S: {S}")
    print(f"  汇点侧 T: {T}")

    # 验证
    cut_capacity = graph.get_cut_capacity(S, T)
    print(f"\n[验证] max flow = min cut?")
    print(f"  {max_flow} = {cut_capacity}")
    print(f"  {'✓ 正确' if abs(max_flow - cut_capacity) < 1e-6 else '✗ 错误'}")

    return max_flow


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 教学示例
    manual_example()

    # 对比实验
    if input("\n是否运行完整对比实验？(y/n): ").lower() == 'y':
        compare_methods()
    else:
        print("\n跳过完整对比（速度较慢）")

    print("\n" + "=" * 70)
    print("学习要点")
    print("=" * 70)
    print("1. 最大流最小割定理：max flow = min cut")
    print("2. Ford-Fulkerson算法：通过增广路径迭代")
    print("3. 残量网络：包含前向和反向边")
    print("4. 图割能量：可以用网络流优化")
    print("\n下一步：")
    print("  - 实现α-扩展（多标签）")
    print("  - 学习3D图割")
    print("  - 探索立体匹配应用")


if __name__ == "__main__":
    main()
