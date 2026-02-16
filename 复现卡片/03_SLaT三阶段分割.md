# 复现卡片: SLaT 三阶段分割

> arXiv: 1506.00060 | 退化彩色图像分割 | 复现难度: ★★★☆☆

---

## 基本信息

| 项目 | 信息 |
|:---|:---|
| **标题** | A Three-Stage Approach for Segmenting Degraded Color Images |
| **作者** | Xiaohao Cai, Raymond Chan, Mila Nikolova, Tieyong Zeng |
| **年份** | 2015 (期刊版: 2017) |
| **期刊** | Journal of Scientific Computing |
| **领域** | 图像分割、变分方法 |

---

## 代码可用性

| 检查项 | 状态 | 详情 |
|:---|:---:|:---|
| **开源代码** | ⚠️ | 作者提供MATLAB参考代码 |
| **代码仓库** | ❌ | 无公开GitHub |
| **许可证** | - | 需联系作者 |
| **文档** | ⚠️ | 论文中有伪代码 |

### 编程语言建议

```
原始实现: MATLAB R2014a
推荐实现: Python 3.8+
依赖库:
├── numpy >= 1.20
├── scipy >= 1.6 (FFT, 优化)
├── opencv-python (颜色空间)
├── scikit-learn (K-means)
└── matplotlib (可视化)
```

---

## 数据集可用性

| 检查项 | 状态 | 详情 |
|:---|:---:|:---|
| **数据类型** | 合成+真实 | 论文提供示例 |
| **获取难度** | 简单 | 可自生成 |
| **预处理** | ❌ | 需自行编写 |

### 测试图像

| 类型 | 来源 | 说明 |
|:---|:---|:---|
| 合成彩色图 | 自生成 | 6相分割测试 |
| 退化图像 | 自生成 | 添加噪声/模糊 |
| 真实图像 | 标准数据集 | 可用BSD500等 |

---

## 实验复现步骤

### 完整Python实现

```python
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.ndimage import laplace

class SLATSegmentation:
    """
    SLaT三阶段彩色图像分割
    Stage 1: Smoothing (平滑/恢复)
    Stage 2: Lifting (维度提升: RGB+Lab)
    Stage 3: Thresholding (K-means阈值化)
    """
    
    def __init__(self, lambda_param=0.1, mu=1.0, max_iter=200, tol=1e-4):
        self.lambda_param = lambda_param  # 数据保真项权重
        self.mu = mu                      # TV权重 (论文固定为1)
        self.max_iter = max_iter
        self.tol = tol
    
    # ============ Stage 1: Smoothing ============
    def stage1_smoothing(self, f):
        """
        对每个通道独立求解ROF变分问题
        min (mu/2)||f-g||^2 + (lambda/2)||grad g||^2 + TV(g)
        使用Chambolle-Pock算法
        """
        h, w, c = f.shape
        g_smooth = np.zeros_like(f)
        
        for i in range(c):
            g_smooth[:,:,i] = self._solve_rof_cp(f[:,:,i])
        
        return np.clip(g_smooth, 0, 1)
    
    def _solve_rof_cp(self, f, n_iter=100):
        """
        Chambolle-Pock算法求解ROF模型
        """
        u = f.copy()
        p = np.zeros((2, *f.shape))  # 对偶变量
        tau = 0.1
        sigma = 0.1
        theta = 1.0
        
        for _ in range(n_iter):
            # 计算u的梯度
            grad_u = np.gradient(u)
            grad_u = np.array([grad_u[0], grad_u[1]])
            
            # 对偶更新: prox of l_infinity ball
            p_new = p + sigma * grad_u
            p_norm = np.sqrt(np.sum(p_new**2, axis=0))
            p = p_new / np.maximum(1, p_norm)
            
            # 计算p的散度
            div_p = np.gradient(p[0], axis=0) + np.gradient(p[1], axis=1)
            
            # 原始更新
            u_bar = (u + tau * (self.mu * f + div_p)) / (1 + tau * self.mu)
            u_bar = np.clip(u_bar, 0, 1)
            
            # 外推
            u = u_bar + theta * (u_bar - u)
            u = np.clip(u, 0, 1)
        
        return u
    
    # ============ Stage 2: Lifting ============
    def stage2_lifting(self, g_rgb):
        """
        维度提升: RGB -> RGB + Lab
        输出: 6通道向量图像
        """
        # RGB转Lab (需要uint8)
        g_uint8 = (g_rgb * 255).astype(np.uint8)
        g_lab = cv2.cvtColor(g_uint8, cv2.COLOR_RGB2LAB)
        
        # 归一化Lab到[0,1]
        g_lab = g_lab.astype(np.float32)
        g_lab_norm = (g_lab - g_lab.min()) / (g_lab.max() - g_lab.min() + 1e-8)
        
        # 拼接: (H, W, 6)
        g_lifted = np.concatenate([g_rgb, g_lab_norm], axis=2)
        
        return g_lifted
    
    # ============ Stage 3: Thresholding ============
    def stage3_thresholding(self, g_lifted, K):
        """
        多通道K-means聚类
        """
        h, w, c = g_lifted.shape
        pixels = g_lifted.reshape(-1, c)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # 计算聚类中心
        centers = kmeans.cluster_centers_
        
        # 最终分割
        segmentation = labels.reshape(h, w)
        
        return segmentation, centers
    
    # ============ 完整流程 ============
    def segment(self, f, K):
        """
        完整SLaT分割流程
        """
        # 归一化输入
        f = f.astype(np.float32) / 255.0 if f.max() > 1 else f
        
        # Stage 1: 平滑
        g_smooth = self.stage1_smoothing(f)
        
        # Stage 2: 维度提升
        g_lifted = self.stage2_lifting(g_smooth)
        
        # Stage 3: 阈值化
        segmentation, centers = self.stage3_thresholding(g_lifted, K)
        
        return segmentation, g_smooth, g_lifted
    
    def change_K(self, new_K):
        """
        更改相位数K - 无需重新计算Stage 1和2
        这是SLaT的核心优势之一
        """
        if self.g_lifted is None:
            raise ValueError("需要先运行segment()方法")
        return self.stage3_thresholding(self.g_lifted, new_K)


# ============ 使用示例 ============
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 读取图像
    image = cv2.imread("test_image.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建分割器
    slat = SLATSegmentation(lambda_param=0.1, mu=1.0)
    
    # 执行分割 (K=4相)
    seg, g_smooth, g_lifted = slat.segment(image, K=4)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("原图")
    axes[1].imshow(g_smooth)
    axes[1].set_title("Stage 1: 平滑")
    axes[2].imshow(seg, cmap='jet')
    axes[2].set_title("Stage 3: 分割")
    plt.savefig("slat_result.png")
```

---

## 超参数设置

| 参数 | 论文值 | 建议范围 | 说明 |
|:---|:---:|:---:|:---|
| `lambda` | 0.1 | 0.05-0.5 | 数据保真权重 |
| `mu` | 1.0 | 固定 | TV权重 |
| `K` | 2-15 | - | 分割相数 |
| `n_iter` | 100 | 50-200 | ROF迭代次数 |

---

## 结果验证

### 论文报告指标

| 实验 | SLaT | 对比最优 | 提升 |
|:---|:---:|:---:|:---:|
| 6相合成图 | 99.21% | 71.68% | +38.4% |
| 信息丢失 | 99.25% | 85.04% | +16.7% |
| 模糊+噪声 | 98.88% | 98.58% | +0.3% |

### 验证脚本

```python
def evaluate_segmentation(pred, gt):
    """
    计算分割准确率
    """
    # 确保标签匹配
    from scipy.optimize import linear_sum_assignment
    
    # 计算混淆矩阵
    n_classes = max(pred.max(), gt.max()) + 1
    conf_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            conf_matrix[i, j] = np.sum((pred == i) & (gt == j))
    
    # Hungarian匹配
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    accuracy = conf_matrix[row_ind, col_ind].sum() / pred.size
    
    return accuracy

# 验证
acc = evaluate_segmentation(segmentation, ground_truth)
print(f"分割准确率: {acc:.2%}")
assert acc > 0.95, "结果低于预期"
```

---

## 常见问题

### Q1: 颜色空间转换错误

```python
# 确保RGB顺序正确
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Lab转换需要uint8
lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
```

### Q2: ROF求解不收敛

- 减小时间步长: `tau = sigma = 0.05`
- 增加迭代次数: `n_iter = 200`
- 检查输入范围: 确保在[0,1]

### Q3: 分割结果差

- 调整lambda参数
- 检查K值选择
- 尝试不同的颜色空间组合

---

## 复现时间估计

| 任务 | 时间 |
|:---|:---:|
| 理解算法 | 2小时 |
| Python实现 | 4小时 |
| 调试优化 | 2小时 |
| 完整测试 | 2小时 |
| **总计** | **1天** |

---

## 参考文献

1. Cai, X., Chan, R., Nikolova, M., & Zeng, T. (2017). SLaT: A Three-stage Approach. JSC.
2. Chambolle, A., & Pock, T. (2011). A First-Order Primal-Dual Algorithm. JMIV.
3. Mumford, D., & Shah, J. (1989). Optimal Approximation. CPAM.

---

*最后更新: 2026-02-16*
