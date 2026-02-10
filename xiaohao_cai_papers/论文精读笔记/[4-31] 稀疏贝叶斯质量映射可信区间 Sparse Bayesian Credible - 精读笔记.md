# [4-31] 稀疏贝叶斯质量映射可信区间 - 精读笔记

> **论文标题**: Sparse Bayesian Mass Mapping: Credible Intervals
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (高)
> **重要性**: ⭐⭐⭐⭐ (不确定性量化)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Sparse Bayesian Mass Mapping: Credible Intervals |
| **作者** | Xiaohao Cai 等人 |
| **应用领域** | 天体物理学、统计推断、不确定性量化 |
| **关键词** | Credible Intervals, Uncertainty Quantification, Bayesian Inference |
| **核心价值** | 为质量映射提供可靠的不确定性估计 |

---

## 🎯 核心问题

### 为什么需要可信区间?

```
不确定性量化重要性:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统点估计的问题:
  - 只给出一个"最佳"估计
  - 无法判断估计的可靠性
  - 难以区分信号和噪声

贝叶斯可信区间的优势:
  - 提供概率性不确定性
  - 识别高/低置信度区域
  - 支持科学决策

应用:
  1. 判断峰值是否真实
  2. 量化子结构质量
  3. 比较不同观测结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 可信区间 vs 置信区间

| 特性 | 频率学派置信区间 | 贝叶斯可信区间 |
|:---|:---|:---|
| **解释** | 重复实验覆盖真值的概率 | 参数落在区间内的概率 |
| **计算** | 基于采样分布 | 基于后验分布 |
| **先验** | 不使用 | 使用先验信息 |
| **稀疏问题** | 难以处理 | 自然处理 |

---

## 🔬 方法论

### 可信区间计算

```
贝叶斯可信区间:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

定义:
  对于参数κ, 100(1-α)%可信区间 [L, U] 满足:
  P(L ≤ κ ≤ U | data) = 1 - α

常用类型:
  1. 等尾区间 (Equal-tailed):
     P(κ < L) = P(κ > U) = α/2

  2. 最高后验密度 (HPD):
     包含最高概率密度的区域
     P(κ ∈ HPD) = 1 - α

  3. 联合可信区域:
     多参数的可信区域
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 核心组件1: 后验采样

```python
import numpy as np
from scipy import stats

class PosteriorSampler:
    """
    后验分布采样器

    用于估计可信区间
    """

    def __init__(self, log_posterior, grad_log_posterior):
        """
        Args:
            log_posterior: 对数后验函数
            grad_log_posterior: 对数后验梯度
        """
        self.log_posterior = log_posterior
        self.grad_log_posterior = grad_log_posterior

    def hmc_sample(self, initial_state, n_samples=1000, n_warmup=500,
                   step_size=0.01, n_leapfrog=10):
        """
        Hamiltonian Monte Carlo采样

        高效探索高维后验分布
        """
        samples = []
        current_state = initial_state.copy()
        current_log_prob = self.log_posterior(current_state)

        for i in range(n_warmup + n_samples):
            # 采样动量
            momentum = np.random.randn(len(current_state))

            # 保存当前状态
            old_state = current_state.copy()
            old_momentum = momentum.copy()
            old_log_prob = current_log_prob

            # Leapfrog积分
            # 半步动量更新
            momentum = momentum + 0.5 * step_size * self.grad_log_posterior(current_state)

            # 完整位置更新
            for _ in range(n_leapfrog):
                current_state = current_state + step_size * momentum
                if _ < n_leapfrog - 1:
                    momentum = momentum + step_size * self.grad_log_posterior(current_state)

            # 最后半步动量更新
            momentum = momentum + 0.5 * step_size * self.grad_log_posterior(current_state)

            # 计算接受概率
            current_log_prob = self.log_posterior(current_state)

            # Metropolis-Hastings接受
            log_accept_prob = (current_log_prob - old_log_prob -
                             0.5 * np.sum(momentum**2) +
                             0.5 * np.sum(old_momentum**2))

            if np.log(np.random.rand()) < log_accept_prob:
                # 接受
                pass
            else:
                # 拒绝,恢复旧状态
                current_state = old_state
                current_log_prob = old_log_prob

            # 保存样本 (warmup后)
            if i >= n_warmup:
                samples.append(current_state.copy())

        return np.array(samples)

    def gibbs_sample(self, initial_state, n_samples=1000, n_warmup=500):
        """
        Gibbs采样

        适用于条件分布易采样的情况
        """
        samples = []
        current = initial_state.copy()

        for i in range(n_warmup + n_samples):
            # 对每个维度依次采样
            for d in range(len(current)):
                # 从条件分布 P(κ_d | κ_{-d}, data) 采样
                current[d] = self.sample_conditional(d, current)

            if i >= n_warmup:
                samples.append(current.copy())

        return np.array(samples)

    def sample_conditional(self, dim, current_state):
        """采样条件分布 (需要具体实现)"""
        # 简化为高斯提议
        current_val = current_state[dim]
        proposal = current_val + 0.1 * np.random.randn()

        # 计算接受概率
        old_log_prob = self.log_posterior(current_state)

        new_state = current_state.copy()
        new_state[dim] = proposal
        new_log_prob = self.log_posterior(new_state)

        if np.log(np.random.rand()) < (new_log_prob - old_log_prob):
            return proposal
        else:
            return current_val

    def variational_posterior_sample(self, mean, var, n_samples=1000):
        """
        从变分后验采样

        假设高斯变分分布
        """
        std = np.sqrt(var)
        samples = mean + std * np.random.randn(n_samples, len(mean))
        return samples
```

---

### 核心组件2: 可信区间计算

```python
class CredibleIntervalCalculator:
    """
    可信区间计算器
    """

    def __init__(self, samples):
        """
        Args:
            samples: 后验样本 (n_samples, n_dims)
        """
        self.samples = samples

    def equal_tailed_interval(self, alpha=0.05):
        """
        等尾可信区间

        使用样本分位数
        """
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        lower = np.percentile(self.samples, lower_percentile, axis=0)
        upper = np.percentile(self.samples, upper_percentile, axis=0)

        return lower, upper

    def hpd_interval(self, alpha=0.05):
        """
        最高后验密度区间

        使用网格搜索找最高密度区域
        """
        n_dims = self.samples.shape[1]
        intervals = []

        for d in range(n_dims):
            samples_d = self.samples[:, d]

            # 核密度估计
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(samples_d)

            # 评估密度
            x_range = np.linspace(samples_d.min(), samples_d.max(), 1000)
            density = kde(x_range)

            # 找HPD区间
            sorted_indices = np.argsort(density)[::-1]
            cumulative_prob = np.cumsum(density[sorted_indices])
            cumulative_prob /= cumulative_prob[-1]

            # 找到覆盖1-alpha概率的阈值
            threshold_idx = np.where(cumulative_prob >= 1 - alpha)[0][0]
            density_threshold = density[sorted_indices[threshold_idx]]

            # HPD区间是密度大于阈值的区域
            hpd_mask = density >= density_threshold
            hpd_regions = x_range[hpd_mask]

            if len(hpd_regions) > 0:
                intervals.append((hpd_regions.min(), hpd_regions.max()))
            else:
                intervals.append((np.median(samples_d), np.median(samples_d)))

        lower = np.array([i[0] for i in intervals])
        upper = np.array([i[1] for i in intervals])

        return lower, upper

    def simultaneous_credible_region(self, alpha=0.05, method='bonferroni'):
        """
        联合可信区域

        考虑多参数的相关性
        """
        n_dims = self.samples.shape[1]

        if method == 'bonferroni':
            # Bonferroni校正
            alpha_adjusted = alpha / n_dims
            return self.equal_tailed_interval(alpha_adjusted)

        elif method == 'mvn':
            # 多元正态近似
            mean = np.mean(self.samples, axis=0)
            cov = np.cov(self.samples.T)

            from scipy.stats import chi2
            threshold = chi2.ppf(1 - alpha, df=n_dims)

            # 马氏距离椭球
            return mean, cov, threshold

    def credible_interval_width(self, interval_type='equal_tailed'):
        """计算可信区间宽度"""
        if interval_type == 'equal_tailed':
            lower, upper = self.equal_tailed_interval()
        else:
            lower, upper = self.hpd_interval()

        return upper - lower

    def coverage_probability(self, true_values):
        """
        计算覆盖概率 (验证用)

        检查可信区间是否包含真值
        """
        lower, upper = self.equal_tailed_interval()

        coverage = np.mean((true_values >= lower) & (true_values <= upper))

        return coverage
```

---

### 核心组件3: 空间可信区域

```python
class SpatialCredibleRegion:
    """
    空间可信区域计算

    针对2D/3D质量映射
    """

    def __init__(self, kappa_samples):
        """
        Args:
            kappa_samples: 后验样本 (n_samples, H, W)
        """
        self.samples = kappa_samples
        self.n_samples, self.H, self.W = kappa_samples.shape

    def pixelwise_credible_intervals(self, alpha=0.05):
        """
        逐像素可信区间
        """
        lower = np.percentile(self.samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(self.samples, 100 * (1 - alpha / 2), axis=0)
        median = np.median(self.samples, axis=0)

        return {
            'median': median,
            'lower': lower,
            'upper': upper,
            'width': upper - lower
        }

    def significant_regions(self, threshold=0.95):
        """
        识别显著偏离零的区域

        计算P(κ > 0 | data) 或 P(κ < 0 | data)
        """
        prob_positive = np.mean(self.samples > 0, axis=0)
        prob_negative = np.mean(self.samples < 0, axis=0)

        # 显著正区域
        sig_positive = prob_positive > threshold

        # 显著负区域
        sig_negative = prob_negative > threshold

        return {
            'prob_positive': prob_positive,
            'prob_negative': prob_negative,
            'significant_positive': sig_positive,
            'significant_negative': sig_negative
        }

    def cluster_credible_regions(self, min_cluster_size=10):
        """
        聚类可信区域

        识别连通的显著区域
        """
        from scipy import ndimage

        sig_regions = self.significant_regions()

        # 标记连通区域
        labeled_pos, n_pos = ndimage.label(sig_regions['significant_positive'])
        labeled_neg, n_neg = ndimage.label(sig_regions['significant_negative'])

        clusters = []

        # 分析正区域
        for i in range(1, n_pos + 1):
            cluster_mask = labeled_pos == i
            size = np.sum(cluster_mask)

            if size >= min_cluster_size:
                cluster_samples = self.samples[:, cluster_mask]
                clusters.append({
                    'type': 'positive',
                    'size': size,
                    'mean_kappa': np.mean(cluster_samples),
                    'std_kappa': np.std(cluster_samples),
                    'mask': cluster_mask
                })

        # 分析负区域
        for i in range(1, n_neg + 1):
            cluster_mask = labeled_neg == i
            size = np.sum(cluster_mask)

            if size >= min_cluster_size:
                cluster_samples = self.samples[:, cluster_mask]
                clusters.append({
                    'type': 'negative',
                    'size': size,
                    'mean_kappa': np.mean(cluster_samples),
                    'std_kappa': np.std(cluster_samples),
                    'mask': cluster_mask
                })

        return clusters

    def visualize_credible_intervals(self, save_path=None):
        """
        可视化可信区间
        """
        import matplotlib.pyplot as plt

        intervals = self.pixelwise_credible_intervals()
        sig_regions = self.significant_regions()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 中位数估计
        im1 = axes[0, 0].imshow(intervals['median'], cmap='RdBu_r')
        axes[0, 0].set_title('Posterior Median')
        plt.colorbar(im1, ax=axes[0, 0])

        # 下界
        im2 = axes[0, 1].imshow(intervals['lower'], cmap='RdBu_r')
        axes[0, 1].set_title('95% Credible Interval Lower')
        plt.colorbar(im2, ax=axes[0, 1])

        # 上界
        im3 = axes[0, 2].imshow(intervals['upper'], cmap='RdBu_r')
        axes[0, 2].set_title('95% Credible Interval Upper')
        plt.colorbar(im3, ax=axes[0, 2])

        # 区间宽度
        im4 = axes[1, 0].imshow(intervals['width'], cmap='viridis')
        axes[1, 0].set_title('Credible Interval Width')
        plt.colorbar(im4, ax=axes[1, 0])

        # 正显著性概率
        im5 = axes[1, 1].imshow(sig_regions['prob_positive'], cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title('P(κ > 0 | data)')
        plt.colorbar(im5, ax=axes[1, 1])

        # 负显著性概率
        im6 = axes[1, 2].imshow(sig_regions['prob_negative'], cmap='hot', vmin=0, vmax=1)
        axes[1, 2].set_title('P(κ < 0 | data)')
        plt.colorbar(im6, ax=axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()
```

---

## 📊 实验结果

### 可信区间覆盖验证

| 方法 | 名义覆盖 | 实际覆盖 | 平均宽度 |
|:---|:---:|:---:|:---:|
| 等尾区间 | 95% | 94.2% | 0.42 |
| HPD区间 | 95% | 95.1% | 0.38 |
| Bonferroni | 95% | 98.5% | 0.55 |

### 峰值位置不确定性

| 峰值类型 | 位置误差 | 质量误差 |
|:---|:---:|:---:|
| 孤立峰值 | ±0.5像素 | ±15% |
| 重叠峰值 | ±1.2像素 | ±25% |
| 弱信号 | ±2.0像素 | ±40% |

---

## 💡 对井盖检测的启示

### 检测不确定性量化

```python
class DetectionUncertaintyEstimator:
    """
    检测不确定性估计

    借鉴可信区间思想
    """

    def __init__(self, model):
        self.model = model

    def estimate_uncertainty(self, image, n_samples=100):
        """
        估计检测结果的不确定性

        使用MC Dropout或集成方法
        """
        predictions = []

        # MC Dropout采样
        self.model.train()  # 启用dropout
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(image)
                predictions.append(pred)

        predictions = torch.stack(predictions)

        # 计算统计量
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)

        # 可信区间
        lower = torch.quantile(predictions, 0.025, dim=0)
        upper = torch.quantile(predictions, 0.975, dim=0)

        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower': lower,
            'upper': upper
        }

    def reliable_detection(self, image, confidence_threshold=0.95):
        """
        可靠检测

        只在高置信度区域报告检测结果
        """
        uncertainty = self.estimate_uncertainty(image)

        # 区间宽度作为不确定性度量
        interval_width = uncertainty['upper'] - uncertainty['lower']

        # 选择低不确定性区域
        reliable_mask = interval_width < (1 - confidence_threshold)

        return {
            'detection': uncertainty['mean'],
            'reliable_mask': reliable_mask,
            'uncertainty': interval_width
        }
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **可信区间** | Credible Interval | 贝叶斯后验概率区间 |
| **HPD** | Highest Posterior Density | 最高后验密度 |
| **MCMC** | Markov Chain Monte Carlo | 马尔可夫链蒙特卡洛采样 |
| **HMC** | Hamiltonian Monte Carlo | 哈密顿蒙特卡洛 |
| **覆盖概率** | Coverage Probability | 区间包含真值的概率 |

---

## ✅ 复习检查清单

- [ ] 理解可信区间的贝叶斯解释
- [ ] 掌握等尾区间和HPD区间的计算
- [ ] 了解MCMC采样方法
- [ ] 理解空间可信区域的概念

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
