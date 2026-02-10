# [4-32] 稀疏贝叶斯质量映射峰值统计 - 精读笔记

> **论文标题**: Sparse Bayesian Mass Mapping: Peak Statistics
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (高)
> **重要性**: ⭐⭐⭐⭐ (天体统计)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Sparse Bayesian Mass Mapping: Peak Statistics |
| **作者** | Xiaohao Cai 等人 |
| **应用领域** | 天体物理学、宇宙学统计 |
| **关键词** | Peak Statistics, Mass Mapping, Cosmology, Random Field |
| **核心价值** | 峰值统计在宇宙学参数估计中的应用 |

---

## 🎯 核心问题

### 峰值统计的重要性

```
峰值统计在宇宙学中的作用:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

背景:
  - 质量映射中的峰值对应星系团
  - 峰值统计包含宇宙学信息
  - 可用于约束暗物质和暗能量

峰值统计量:
  1. 峰值数密度 (Peak Count)
  2. 峰值高度分布 (Peak Height Distribution)
  3. 峰值-峰值关联 (Peak-Peak Correlation)
  4. 空洞统计 (Void Statistics)

科学目标:
  - 从观测数据提取宇宙学参数
  - 检验ΛCDM模型
  - 探测偏离标准模型的信号
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 峰值统计 vs 传统统计

| 统计量 | 信息内容 | 计算复杂度 | 对噪声鲁棒性 |
|:---|:---|:---:|:---:|
| **功率谱** | 二阶统计 | 低 | 中 |
| **双谱** | 三阶统计 | 中 | 低 |
| **峰值计数** | 非高斯信息 | 中 | **高** |
| **Minkowski泛函** | 形态统计 | 高 | 中 |

---

## 🔬 方法论

### 峰值统计框架

```
峰值统计分析流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 质量映射重建
   ┌─────────────────────────────────────┐
   │  - 稀疏贝叶斯重建                    │
   │  - 获得后验样本                      │
   │  - 不确定性量化                      │
   └─────────────────────────────────────┘
            ↓
2. 峰值检测
   ┌─────────────────────────────────────┐
   │  - 局部极大值识别                    │
   │  - 显著性阈值                        │
   │  - 噪声峰值剔除                      │
   └─────────────────────────────────────┘
            ↓
3. 峰值特征提取
   ┌─────────────────────────────────────┐
   │  - 峰值高度                          │
   │  - 峰值曲率                          │
   │  - 峰值位置                          │
   └─────────────────────────────────────┘
            ↓
4. 统计量计算
   ┌─────────────────────────────────────┐
   │  - 峰值数密度                        │
   │  - 高度分布                          │
   │  - 关联函数                          │
   └─────────────────────────────────────┘
            ↓
5. 宇宙学推断
   ┌─────────────────────────────────────┐
   │  - 与理论模型比较                    │
   │  - 参数估计                          │
   │  - 模型选择                          │
   └─────────────────────────────────────┘
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 核心组件1: 峰值检测

```python
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

class PeakDetector:
    """
    峰值检测器

    在收敛场中识别显著峰值
    """

    def __init__(self, threshold=3.0, min_distance=5):
        """
        Args:
            threshold: 信噪比阈值 (sigma)
            min_distance: 峰值间最小距离 (像素)
        """
        self.threshold = threshold
        self.min_distance = min_distance

    def detect_peaks(self, kappa_map, noise_std=None):
        """
        检测峰值

        Args:
            kappa_map: 收敛场 (2D array)
            noise_std: 噪声标准差 (可选)

        Returns:
            peaks: 峰值列表 [{'position': (y, x), 'height': ..., 'snr': ...}, ...]
        """
        if noise_std is None:
            noise_std = self.estimate_noise(kappa_map)

        # 局部极大值
        local_max = self.find_local_maxima(kappa_map)

        # 信噪比阈值
        snr = kappa_map / noise_std
        significant = snr > self.threshold

        # 峰值位置
        peak_mask = local_max & significant
        peak_positions = np.argwhere(peak_mask)

        # 构建峰值列表
        peaks = []
        for pos in peak_positions:
            y, x = pos
            height = kappa_map[y, x]

            peaks.append({
                'position': (y, x),
                'height': height,
                'snr': snr[y, x]
            })

        # 最小距离筛选
        peaks = self.filter_by_distance(peaks)

        return peaks

    def find_local_maxima(self, data):
        """找到局部极大值"""
        # 使用最大值滤波
        max_filtered = ndimage.maximum_filter(data, size=3)
        local_max = (data == max_filtered)

        # 排除边界
        local_max[0, :] = False
        local_max[-1, :] = False
        local_max[:, 0] = False
        local_max[:, -1] = False

        return local_max

    def estimate_noise(self, kappa_map):
        """估计噪声水平"""
        # 使用边缘区域估计噪声
        edge_mask = np.zeros_like(kappa_map, dtype=bool)
        edge_mask[:10, :] = True
        edge_mask[-10:, :] = True
        edge_mask[:, :10] = True
        edge_mask[:, -10:] = True

        return np.std(kappa_map[edge_mask])

    def filter_by_distance(self, peaks):
        """按最小距离筛选峰值"""
        if len(peaks) <= 1:
            return peaks

        # 按高度排序
        peaks_sorted = sorted(peaks, key=lambda p: p['height'], reverse=True)

        filtered = [peaks_sorted[0]]

        for peak in peaks_sorted[1:]:
            pos = np.array(peak['position'])

            # 检查与已选峰值的距离
            too_close = False
            for selected in filtered:
                selected_pos = np.array(selected['position'])
                dist = np.linalg.norm(pos - selected_pos)

                if dist < self.min_distance:
                    too_close = True
                    break

            if not too_close:
                filtered.append(peak)

        return filtered

    def compute_peak_curvature(self, kappa_map, peak):
        """
        计算峰值处的曲率

        用于区分真实峰值和噪声
        """
        y, x = peak['position']

        # 提取局部区域
        window = 2
        local = kappa_map[y-window:y+window+1, x-window:x+window+1]

        if local.shape != (2*window+1, 2*window+1):
            return None

        # 计算Hessian矩阵
        # 二阶导数
        dyy = (local[0, 1] - 2*local[1, 1] + local[2, 1])
        dxx = (local[1, 0] - 2*local[1, 1] + local[1, 2])
        dxy = (local[0, 0] - local[0, 2] - local[2, 0] + local[2, 2]) / 4

        hessian = np.array([[dxx, dxy], [dxy, dyy]])

        # 特征值
        eigenvalues = np.linalg.eigvals(hessian)

        return {
            'eigenvalues': eigenvalues,
            'curvature': -np.sum(eigenvalues),  # 拉普拉斯
            'anisotropy': abs(eigenvalues[0] - eigenvalues[1])
        }
```

---

### 核心组件2: 峰值统计量计算

```python
class PeakStatistics:
    """
    峰值统计量计算
    """

    def __init__(self, peaks, field_size):
        """
        Args:
            peaks: 峰值列表
            field_size: 场大小 (deg^2)
        """
        self.peaks = peaks
        self.field_size = field_size

    def peak_count_histogram(self, kappa_bins=None):
        """
        峰值高度分布直方图

        宇宙学敏感统计量
        """
        if kappa_bins is None:
            heights = [p['height'] for p in self.peaks]
            kappa_bins = np.linspace(min(heights), max(heights), 10)

        counts, bin_edges = np.histogram(
            [p['height'] for p in self.peaks],
            bins=kappa_bins
        )

        # 归一化到单位面积
        density = counts / self.field_size

        return {
            'bin_edges': bin_edges,
            'counts': counts,
            'density': density
        }

    def peak_correlation_function(self, r_bins=None):
        """
        峰值-峰值关联函数
        """
        if len(self.peaks) < 2:
            return None

        positions = np.array([p['position'] for p in self.peaks])

        # 计算所有峰值对距离
        from scipy.spatial.distance import pdist
        distances = pdist(positions)

        if r_bins is None:
            r_bins = np.linspace(0, np.max(distances), 20)

        # 距离直方图
        counts, bin_edges = np.histogram(distances, bins=r_bins)

        # 归一化 (除以随机分布期望)
        area = np.pi * (bin_edges[1:]**2 - bin_edges[:-1]**2)
        density = len(self.peaks) / self.field_size
        expected = np.pi * density**2 * area

        correlation = counts / (expected + 1e-10) - 1

        return {
            'r_bins': (bin_edges[:-1] + bin_edges[1:]) / 2,
            'correlation': correlation
        }

    void_statistics(self, kappa_map, threshold=-0.5):
        """
        空洞统计

        分析低密度区域
        """
        # 二值化: 低于阈值为空洞
        void_mask = kappa_map < threshold

        # 标记连通区域
        labeled, num_voids = ndimage.label(void_mask)

        # 分析每个空洞
        voids = []
        for i in range(1, num_voids + 1):
            void_mask_i = labeled == i
            size = np.sum(void_mask_i)

            if size > 10:  # 最小尺寸
                voids.append({
                    'size': size,
                    'depth': np.abs(np.min(kappa_map[void_mask_i]))
                })

        return {
            'num_voids': len(voids),
            'mean_size': np.mean([v['size'] for v in voids]) if voids else 0,
            'size_distribution': [v['size'] for v in voids]
        }

    def minkowski_functionals(self, kappa_map, thresholds=None):
        """
        Minkowski泛函

        描述场的形态特征
        """
        if thresholds is None:
            thresholds = np.linspace(-0.5, 1.0, 20)

        V0 = []  # 面积
        V1 = []  # 周长
        V2 = []  # 欧拉示性数

        for thresh in thresholds:
            binary = kappa_map > thresh

            # 面积 (归一化)
            area = np.sum(binary) / kappa_map.size
            V0.append(area)

            # 周长估计
            from scipy.ndimage import binary_erosion
            boundary = binary & ~binary_erosion(binary)
            perimeter = np.sum(boundary)
            V1.append(perimeter)

            # 欧拉示性数
            labeled, num_features = ndimage.label(binary)
            holes = num_features - 1  # 简化估计
            V2.append(num_features - holes)

        return {
            'thresholds': thresholds,
            'V0_area': np.array(V0),
            'V1_perimeter': np.array(V1),
            'V2_euler': np.array(V2)
        }
```

---

### 核心组件3: 宇宙学推断

```python
class CosmologicalInference:
    """
    宇宙学参数推断

    从峰值统计约束宇宙学参数
    """

    def __init__(self, theory_model):
        """
        Args:
            theory_model: 理论模型 (如HALOFIT + 峰值理论)
        """
        self.theory = theory_model

    def compute_likelihood(self, data_stats, cosmological_params):
        """
        计算似然函数

        P(data | params)
        """
        # 理论预测
        theory_stats = self.theory.predict(cosmological_params)

        # 高斯似然 (简化)
        diff = data_stats - theory_stats

        # 协方差矩阵 (需要预先计算)
        cov = self.load_covariance_matrix()

        chi2 = diff @ np.linalg.inv(cov) @ diff

        log_likelihood = -0.5 * chi2

        return log_likelihood

    def mcmc_inference(self, data_stats, initial_params, n_samples=10000):
        """
        MCMC参数推断
        """
        import emcee

        n_params = len(initial_params)

        def log_probability(params):
            # 先验
            if not self.check_prior(params):
                return -np.inf

            # 似然
            return self.compute_likelihood(data_stats, params)

        # 初始化 walkers
        n_walkers = 4 * n_params
        pos = initial_params + 1e-4 * np.random.randn(n_walkers, n_params)

        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability)
        sampler.run_mcmc(pos, n_samples, progress=True)

        return sampler

    def fisher_forecast(self, cosmological_params):
        """
        Fisher矩阵预测

        预测参数约束精度
        """
        n_params = len(cosmological_params)
        fisher_matrix = np.zeros((n_params, n_params))

        # 数值计算导数
        delta = 0.01

        for i in range(n_params):
            for j in range(n_params):
                # 计算二阶导数
                params_pp = cosmological_params.copy()
                params_pp[i] += delta
                params_pp[j] += delta

                params_pm = cosmological_params.copy()
                params_pm[i] += delta
                params_pm[j] -= delta

                params_mp = cosmological_params.copy()
                params_mp[i] -= delta
                params_mp[j] += delta

                params_mm = cosmological_params.copy()
                params_mm[i] -= delta
                params_mm[j] -= delta

                # 中心差分
                f_pp = self.theory.predict(params_pp)
                f_pm = self.theory.predict(params_pm)
                f_mp = self.theory.predict(params_mp)
                f_mm = self.theory.predict(params_mm)

                fisher_matrix[i, j] = np.sum(
                    (f_pp - f_pm - f_mp + f_mm) / (4 * delta**2)
                )

        # 协方差矩阵
        covariance = np.linalg.inv(fisher_matrix)

        return {
            'fisher_matrix': fisher_matrix,
            'parameter_covariance': covariance,
            'parameter_constraints': np.sqrt(np.diag(covariance))
        }
```

---

## 📊 实验结果

### 峰值统计的宇宙学敏感性

| 参数 | 峰值计数敏感性 | 功率谱敏感性 |
|:---|:---:|:---:|
| **Ω_m** | 高 | 高 |
| **σ_8** | **很高** | 高 |
| **w** | 中 | 低 |
| **n_s** | 中 | 高 |

### 参数约束对比

| 方法 | σ(Ω_m) | σ(σ_8) | 退化程度 |
|:---|:---:|:---:|:---:|
| 功率谱 | 0.03 | 0.04 | 高 |
| 峰值计数 | 0.025 | 0.03 | 中 |
| **联合分析** | **0.02** | **0.025** | **低** |

---

## 💡 对井盖检测的启示

### 异常模式统计

```python
class AnomalyPatternStatistics:
    """
    异常模式统计

    借鉴峰值统计思想
    """

    def __init__(self):
        self.patterns = []

    def collect_patterns(self, detection_results):
        """
        收集检测到的异常模式

        类似峰值收集
        """
        for result in detection_results:
            if result['is_anomaly']:
                self.patterns.append({
                    'position': result['position'],
                    'severity': result['severity'],
                    'type': result['anomaly_type']
                })

    def pattern_statistics(self):
        """
        异常模式统计
        """
        # 严重程度分布
        severities = [p['severity'] for p in self.patterns]

        # 类型分布
        from collections import Counter
        type_counts = Counter([p['type'] for p in self.patterns])

        # 空间分布
        positions = np.array([p['position'] for p in self.patterns])

        # 聚类分析
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=50, min_samples=3).fit(positions)

        return {
            'total_anomalies': len(self.patterns),
            'severity_distribution': np.histogram(severities, bins=5),
            'type_distribution': type_counts,
            'spatial_clusters': len(set(clustering.labels_)) - 1
        }
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **峰值计数** | Peak Count | 超过阈值的峰值数量 |
| **关联函数** | Correlation Function | 两点空间关联 |
| **Minkowski泛函** | Minkowski Functionals | 形态描述量 |
| **Fisher矩阵** | Fisher Matrix | 参数约束预测 |
| **空洞统计** | Void Statistics | 低密度区域统计 |

---

## ✅ 复习检查清单

- [ ] 理解峰值统计的宇宙学意义
- [ ] 掌握峰值检测方法
- [ ] 了解峰值统计量计算
- [ ] 理解宇宙学参数推断流程

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
