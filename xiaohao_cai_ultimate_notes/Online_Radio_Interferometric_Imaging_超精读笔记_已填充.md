# 在线无线电干涉成像：到达时同化与丢弃可见度数据

> **超精读笔记** | 5-Agent辩论分析系统
> 分析时间：2026-02-16
> 作者：Xiaohao Cai, Luke Pratley, Jason D. McEwen
> 来源：arXiv:1712.04462v1（2017-12-12 预印本）→ 正式刊出 MNRAS (2019)
> 第一作者：Xiaohao Cai（MSSL, UCL；通讯邮箱 x.cai@ucl.ac.uk）

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Online Radio Interferometric Imaging: Assimilating and Discarding Visibilities on Arrival |
| **作者** | Xiaohao Cai, Luke Pratley, Jason D. McEwen |
| **第一作者核验** | 是。PDF 首页作者行 `Xiaohao Cai^{1*}, Luke Pratley^{1*} and Jason D. McEwen^{1*}`，星号脚注为通讯邮箱 `x.cai@ucl.ac.uk`，上标 1 指向唯一机构 MSSL/UCL；Xiaohao Cai 为唯一第一作者 |
| **年份** | PDF/arXiv v1 为 2017（"Preprint 14 December 2017"，© 2017 The Authors）；正式刊出版本为 MNRAS 2019（dashboard `year=2019` 取刊出年） |
| **arXiv ID** | 1712.04462v1 [astro-ph.IM]（页边竖排标识 12 Dec 2017） |
| **期刊** | Monthly Notices of the Royal Astronomical Society (MNRAS) |
| **机构** | Mullard Space Science Laboratory (MSSL), University College London (UCL), Surrey RH5 6NT, UK |
| **资助** | EPSRC EP/M011089/1；STFC ST/N000811/1（见 Acknowledgements） |
| **领域** | 射电天文 (RI imaging)、在线优化 (online optimisation)、压缩感知 / 稀疏正则化 |

### 📝 摘要翻译

本文提出了一种在线稀疏正则化方法，用于射电干涉测量的实时图像重建。针对新一代射电望远镜（如SKA）产生的大数据挑战，传统方法需要等待所有数据采集完成后才能开始重建。本文方法实现了数据到达时的即时同化和处理后的丢弃，理论上与离线方法重建质量相同，同时显著降低存储需求和计算延迟。

**关键词**: 射电干涉测量、在线优化、稀疏正则化、前向-后向分裂、SKA

---

## 🎯 一句话总结

通过在线前向-后向算法实现射电干涉测量的流式成像，数据到达即处理、处理完即丢弃，在保持质量的同时大幅降低存储需求。

---

## 🔑 核心创新点

1. **在线成像框架**：首次将在线优化应用于射电干涉测量
2. **数据同化与丢弃**：处理完立即释放，存储需求从O(M)降至O(M_b)
3. **统一算法框架**：适用于各种迭代优化算法
4. **理论保证**：证明在线方法与离线方法的收敛等价性

---

## 📊 背景与动机

### 射电干涉测量基础

**测量方程**（连续形式）：

```
y(u) = ∫ A(l)x(l)e^(-2πiu·l) d²l
```

**离散化模型**：

```
y = Φx + n
```

其中：
- `y ∈ C^M`：M个观测可见度
- `x ∈ R^N`：N个像素的图像
- `Φ ∈ C^(M×N)`：测量算子
- `n ∈ C^M`：加性高斯噪声

### 大数据挑战

**SKA第一阶段数据率**：
- 数据率：约5 Tb/s
- 观测时长：通常≥10小时
- 数据存储：指数级增长

**传统离线方法局限**：
1. 必须等待所有观测数据采集完成
2. 需要存储全部可见度数据
3. 计算延迟大

---

## 💡 方法详解（含公式推导）

### 3.1 贝叶斯推断框架

**MAP估计器**：

```
x_map = argmax_x p(x|y)
```

**似然函数**（高斯噪声假设）：

```
p(y|x) ∝ exp(-||y - Φx||²₂ / 2σ²)
```

**稀疏促进先验**：

```
p(x) ∝ exp(-φ(Bx))
```

### 3.2 优化问题

**分析形式**（Analysis form）：

```
x_map = argmin_x { μ||Ψ†x||₁ + ||y - Φx||²₂ / 2σ² }
```

**综合形式**（Synthesis form）：

```
x_map = Ψ × argmin_a { μ||a||₁ + ||y - ΦΨa||²₂ / 2σ² }
```

### 3.3 在线优化理论

**问题分解**：将M个测量值分为B个块

```
y = [y₁ᵀ, ..., y_Bᵀ]ᵀ, y_k ∈ C^M_k
Φ = [Φ₁ᵀ, ..., Φ_Bᵀ]ᵀ, Φ_k ∈ C^(M_k×N)
```

**目标函数分离**：

```
F_y(x) = f(x) + Σ_{k=1}^B g_k(x)
```

其中：
- `f(x)`：正则化项（如μ||Ψ†x||₁）
- `g_k(x)`：第k个数据块的数据保真项

### 3.4 在线前向-后向算法

**标准前向-后向迭代**：

```
x^(i+1) = prox_{λ^(i)f}( x^(i) - λ^(i)∇g(x^(i)) )
```

**在线版本**（处理前b个块）：

```
x^(i+1) = prox_{λ^(i)f}( x^(i) - λ^(i)∇g_{1:b}(x^(i)) )
```

其中 `g_{1:b} = g₁ + ... + g_b`

**算法伪代码**：

```
Algorithm 1: Online Forward-Backward Algorithm

Input: x^(0) ∈ R^N, σ, λ^(b) ∈ (0, ∞)
Output: x*

i = 0, b = 0
do:
    b = b + 1
    load data y_b                    // 加载新数据块
    do:
        // 同化y_b并成像
        x^(i+1) = prox_{λ^(b)f}( x^(i) - λ^(b)∇g_{1:b}(x^(i)) )
        i = i + 1
    while Stopping criterion type II not reached
    delete y_b                        // 丢弃数据块
while Stopping criterion type I not reached
x* = x^(i)
```

> **两层停止准则的工程含义**（PDF §3.2）：
> - **type I（数据块级，外层）**：已知总块数时设为 `b = B`；未知（真实流式）时由"无新块到达"的反馈触发。
> - **type II（迭代级，内层）**：最大内迭代数，**实践中设为 1**（每块只做一次 FB，算力最省）；或用连续两次迭代的相对误差。
> 注意：当块数 B 较小（小于标准 FB 收敛所需迭代数）时，可在处理完最后一块后**追加**几次"额外迭代"（Algorithm 2/3 的可选 while 段），但论文 Figure 5 显示这点改进有限、可选。big-data 下 B 通常很大，无需额外迭代。

### 3.4b 在线 FB 的 analysis 显式实例（PDF §4.1.1, Algorithm 2）

把抽象 prox/梯度落到 RI 的 analysis 模型（式 10）上。设 $\bar f(x)=\mu\|\Psi^\dagger x\|_1$、$\bar g_k(x)=\|y_k-\Phi_k x\|_2^2/2\sigma^2$。当 $\Psi^\dagger\Psi=\mathrm I$（正交基，论文用 Daubechies-8）时：

**proximity（软阈值，式 38/40）**：
```
prox_{λf}(z) = z + Ψ( soft_{λμ}(Ψ†z) − Ψ†z )
soft_λ(z_k) = z_k(|z_k|−λ)/|z_k|  if |z_k|>λ  else 0
```

**部分梯度（只用前 b 块，式 39）**：
```
∇g_{1:b}(x) = Σ_{k=1}^b Φ_k†(Φ_k x − y_k) / σ²
```

合起来即式 41/42 的两步：先梯度步 `v = x − λ·∇g_{1:b}(x)`，再 prox 步 `x⁺ = v + Ψ(soft_{λμ}(Ψ†v) − Ψ†v)`。

> **关键算力优化**：dirty map 项 $\Phi_k^\dagger y_k$ 以及 $\Phi_k^\dagger\Phi_k$ 可**预计算一次**反复调用（Remark 4.2）。这样在线第 b 步只需触及 $b/B$ 比例的算子，而标准法每次迭代都要全部 B 块——这正是 §4.3 算力比 $\eta_c\approx(B+1)/(2 i_{\max})$ 的来源。synthesis 模型见 Algorithm 3（式 49），正交基下与 analysis 性能差异可忽略，论文只报告 analysis。

### 3.5 收敛性分析

**核心假设**（假设29）：

```
Σ_{k=b+1}^B g_k(x^(i)) ≥ Σ_{k=b+1}^B g_k(x^(i+1))
```

**直观解释**：随着更多数据的同化，中间重建应对未观测数据块拟合更好。

**收敛定理**（定理3.2）：

在假设(29)下，设x*为问题(24)的极小化子，则序列F_y(x^(i))单调递减至F_y(x*)。

**证明骨架（PDF 式 30–35）**：要证 $\forall i$，$\mathcal F_y(x^{(i)})\ge\mathcal F_y(x^{(i+1)})$。
- 若 $x^{(i+1)}$ 用了**全部** B 块（$b=B$），则式 30 直接由 splitting 方法标准收敛性给出。
- 若 $x^{(i+1)}$ 只用了 $b<B$ 块，则对**部分**目标有 $\mathcal F_{y_1^b}(x^{(i)})\ge\mathcal F_{y_1^b}(x^{(i+1)})$（式 31，部分目标的能量单调下降，仍由 splitting 收敛性保证）。
- 把完整目标拆成"已用块部分目标 + 未用块之和"（式 32）：$\mathcal F_y(\cdot)=\mathcal F_{y_1^b}(\cdot)+\sum_{k=b+1}^B g_k(\cdot)$。对前者用式 31、对后者用假设 29（未用块之和也下降），相加即得式 33–35 的 $\mathcal F_y(x^{(i)})\ge\mathcal F_y(x^{(i+1)})$。∎

> **直觉**：online 算法在任一时刻只"看见"前 b 块，但论文要保证的是对**完整**目标 $\mathcal F_y$ 的单调下降。桥梁就是假设 29——"随着同化更多数据，中间重建对尚未到达的块也拟合更好"。这是个**温和但非平凡**的假设：它不是无条件成立的，论文也把"放宽/验证假设 29"列为未来方向（见本笔记局限性）。

---

## 🧪 实验与结果

### 算法复杂度对比

| 操作 | 离线方法 | 在线方法 |
|------|----------|----------|
| 每次迭代梯度计算 | O(MN) | O(M_bN) |
| 总梯度计算（I次迭代） | O(IMN) | O(BN)（单次迭代） |
| 近端算子计算 | O(N) | O(N) |

### 存储复杂度对比

| 项目 | 离线方法 | 在线方法 |
|------|----------|----------|
| 可见度数据存储 | O(M) | O(M_b) |
| 中间变量 | O(N) | O(N) |
| 总存储 | O(M+N) | O(M_b+N) ≈ O(N) |

**关键优势**：存储从O(M)降至O(M_b)，通常M_b << M

### 实验设置（PDF §5.1，均可核实）

| 项 | 设置 |
|------|------|
| 测试图 | **M31**(HI region, 256×256)、**Cygnus A**(256×512)、**W28**(超新星遗迹, 256×256)、**3C288**(256×256) |
| 采样 | variable-density profile（Puy et al. 2011），**半个 Fourier 平面**取 **10%** 离散 Fourier 系数 |
| 噪声 | 零均值高斯，$\sigma=\|f\|_\infty 10^{-\mathrm{SNR}/20}$，输入 **SNR=30 dB** |
| sensing 算子 | $\Phi_k=M_k F$（FFT + masking，on-grid 简化；真实 off-grid 需 degridding） |
| 稀疏基 $\Psi$ | **Daubechies-8** wavelets（MATLAB `wavedec2`），$\Psi^\dagger\Psi=\mathrm I$ |
| 正则参数 | $\mu=10^4$（试错定） |
| 迭代 | 标准法 $i_{\max}=50$；在线法每块 1 次内迭代 |
| 块数 | $B=50$（默认；SNR 分析中 $B\in\{50,100,200,300,500\}$，每块约 2% 系数） |
| 硬件 | MacBook 2.2 GHz i7 / 16 GB / MATLAB R2015b |

> 论文是 **simulation**（从公开射电图的 ground-truth 生成 visibilities），不是真实 telescope 观测。

### 主要结果（PDF §5.2，定量数值均来自论文，禁止外推）

**重建质量 — SNR（式 53）**：$\mathrm{SNR}=20\log_{10}(\|x\|_2/\|x-x^*\|_2)$ dB。
- **M31**：online 与 standard 取得**完全相同** SNR **14.2946 dB**（analysis 模型，$B=50$）；用另一种 splitting（均匀随机 vs 按距原点）得 14.2943 dB，几乎不变 → 说明 online 对 splitting 策略稳健。
- **Table 1**（四图 × $B\in\{50,100,200,300,500\}$ 的相对差，式 54）：M31 量级约 **10⁻⁶~10⁻⁷**，3C288 约 **10⁻⁶~10⁻⁸**，Cygnus A/W28 约 **10⁻²~10⁻³**；正负号互现，**无实质差异**——量化印证"online ≈ offline"。

**存储节省（式 50）**：$\eta_s=\max_k\{M_k\}/M$，等块时 $\eta_s=1/B$。$B>100$ 时所需存储 **< 1%** 全量 visibilities。论文未给具体 PB/TB 数字，仅给比值 $\eta_s$ 与趋势曲线（Figure 2 蓝实线）。

**算力节省（式 51/52）**：$\eta_c\approx\frac{\sum_b b/B}{i_{\max}}=\frac{(B+1)/2}{i_{\max}}$；当 $i_{\max}$ 足够大且两法迭代数相近时 $\eta_c\approx 1/2$，即在线约**省一半计算**。

**时间优势**：
- 在线方法**边采集边重建**，在数据采集完成时已**接近完成**重建（Figure 5 显示前几次迭代因数据少而极快）。
- 离线方法**必须等采集完成**才能开始重建——这是 online 在"开始重建时刻"上永远获胜的根本原因。

---

## 📈 技术演进脉络

```
传统RI成像
  ↓ CLEAN算法
  ↓ 最大熵法(MEM)
  ↓ 压缩感知
2017: 在线稀疏正则化 (本文)
  ↓ 数据同化与丢弃
  ↓ 前向-后向分裂
  ↓ 在线优化理论
未来方向
  ↓ 分布式在线算法
  ↓ 深度学习结合
  ↓ 自适应参数选择
```

---

## 🔗 上下游关系

### 上游依赖

- **压缩感知理论**：稀疏正则化框架
- **前向-后向分裂**：优化算法基础
- **在线优化理论**：在线学习方法

### 下游影响

- 为SKA等大型射电望远镜提供实时成像方案
- 推动在线优化在天文成像中的应用

### 与本项目其余 14 篇的具体关系

- **RI UQ 系列（Cai et al. 2017a/b，本文引为 [Cai et al. 2017a]/[Cai et al. 2017b]）**：2017b 是 RI inverse problem + convex MAP 的方法学综述（本文 §2 多处引它"for further details"）；2017a 提出 MAP-UQ（local credible intervals / error bars）。本文专攻"在线重建 + 存储"，与 2017a 的"不确定性近似"**正交互补**——论文 Conclusion 明确说要把二者结合做 big-data 的"efficient imaging + UQ"。这构成 dashboard 中 relation links 13/12/11 的依据。
- **优化求解器谱系**：本文的 online FB 是把 forward-backward splitting（Combettes & Pesquet 2010）改造成"分块同化"；与本项目里用 tight-frame / framelet 软阈值的几篇（vessel、sphere、color SLaT 等）共享"$\mathcal A^\top\,\mathrm{soft}_\lambda\,\mathcal A$ 风格的 proximal 算子"骨架，区别在于本文把数据保真项**按块在线拆分**。
- **方法学定位**：相对 CLEAN/MEM/CS 这些 offline 重建，本文的贡献不是"更好的重建质量"（它刻意证明质量**等价**），而是"把求解器搬进数据获取流程"这一**范式**转变。

---

## ⚙️ 可复现性分析

### 算法复杂度

```
T_total = B × T_block
T_block = T_inner_iter × (T_gradient + T_proximal)

其中：
- B: 数据块数
- T_inner_iter: 每块的内迭代数（通常为1）
- T_gradient: O(M_bN)
- T_proximal: O(N)
```

### 停止准则

**类型I**（数据块级）：
- 最大数据块数（已知时）
- 无新数据块可用反馈

**类型II**（迭代级）：
- 最大迭代数（实践中设为1）
- 连续迭代的相对误差

---

## 📚 关键参考文献

1. Wiaux et al. "Compressed sensing imaging techniques for radio interferometry." MNRAS 2009.
2. McEwen & Wiaux. "Compressed sensing for wide-field radio interferometric imaging." MNRAS 2011.
3. Combettes & Pesquet. "Proximal splitting methods in signal processing." 2011.
4. Shalev-Shwartz. "Online learning and online convex optimization." 2012.

---

## 💻 代码实现要点

```python
import numpy as np
from scipy.fftpack import fft2, ifft2

class OnlineRIReconstructor:
    def __init__(self, block_size=100, mu=0.01, sigma=0.1):
        self.block_size = block_size
        self.mu = mu  # 正则化参数
        self.sigma = sigma  # 噪声水平
        self.current_image = None
        self.block_count = 0

    def process_block(self, new_visibilities, new_sampling, wavelet_op):
        """
        处理新的数据块

        参数:
            new_visibilities: 新到达的可见度数据
            new_sampling: 对应的采样算子
            wavelet_op: 小波变换算子
        """
        # 初始化
        if self.current_image is None:
            self.current_image = np.zeros(wavelet_op.image_shape)

        # 前向-后向迭代
        gradient = self._compute_gradient(
            self.current_image, new_visibilities, new_sampling
        )

        # 梯度步
        x_temp = self.current_image - 0.01 * gradient

        # 近端算子（软阈值）
        self.current_image = wavelet_op.soft_threshold(x_temp, self.mu * 0.01)

        self.block_count += 1
        return self.current_image

    def _compute_gradient(self, x, visibilities, sampling):
        """计算数据保真项的梯度"""
        # Φx: 前向投影
        forward_proj = sampling.forward(x)

        # 残差
        residual = forward_proj - visibilities

        # Φ†残差: 反向投影
        gradient = sampling.adjoint(residual)

        return gradient

    def get_reconstruction(self):
        """获取当前重建结果"""
        return self.current_image
```

---

## 🌟 应用与影响

### 应用场景

1. **射电天文实时成像（论文主战场）**
   - SKA / SKA precursor（ASKAP, MWA, LOFAR, EVLA）的 big-data RI imaging
   - 论文明确目标：把 RI imaging 推向 big-data era，并规划集成进 **PURIFY** 包（github.com/basp-group/purify）

2. **论文自述的可迁移性（PDF §1 Introduction，p.2 左栏）**
   - 论文指出该 online 框架"is generic and therefore can be directly applied to many other applications, **such as medical imaging**"
   - 注：这是论文一句**泛化性陈述**，论文本身**未**做 MRI/CT 实验；下面具体场景为合理外推，非论文结论
     - （外推）MRI 流式 k-space 重建、CT 在线成像等其他逆问题成像

> ⚠️ 修正：原笔记此处曾写"地球观测/卫星/灾害监测"与"节省数PB存储成本"等具体数字，**论文未给这些场景或 PB 级数字**。论文只给比值型结论（$\eta_s=1/B$、$\eta_c\approx1/2$），已在上文实验结果中据 PDF 修正。

### 价值定位（据论文，避免编造商业数字）

- **存储**：相对全量 visibilities 仅需 $\eta_s=1/B$（$B>100$ 时 < 1%），把存储从 $O(M)$ 降到 $O(M_b)\approx O(N)$。
- **算力**：迭代数相近时约省一半（$\eta_c\approx1/2$）。
- **时效**：与采集**同步**重建，采集结束即近完成——这是 online 相对 offline 的根本时序优势。
- **集成路径**：拟并入 PURIFY，并与 Cai et al. (2017a) 的 MAP-UQ 框架结合做 streaming imaging + UQ。

---

## ❓ 未解问题与展望

### 局限性

1. **假设依赖**：收敛性依赖于假设(29)，实际中难以验证
2. **单次迭代**：实践中仅用一次迭代，可能损失精度
3. **参数敏感**：块大小、步长等参数需要调优

### 未来方向

1. **非凸扩展**：非凸情况下的收敛性分析
2. **自适应参数**：数据驱动的参数选择
3. **分布式实现**：多节点协同在线成像
4. **深度学习结合**：学习数据同化策略

---

## 📝 分析笔记

```
个人理解：

1. 核心创新：
   - 首次将在线优化引入射电成像
   - 数据同化与丢弃策略巧妙
   - 理论与工程结合紧密

2. 技术亮点：
   - 存储需求按 η_s=1/B 下降，B>100 时降到全量 visibilities 的 <1%（实验 B=500 时约 0.2%）
   - 重建质量与离线方法等价
   - 统一的算法框架

3. 实际价值：
   - 直接解决SKA大数据挑战
   - 可扩展到其他流式成像任务
   - 工程实现可行

4. 改进方向：
   - 假设(29)的验证与放宽
   - 收敛速度的定量界
   - 更复杂先验的处理
```

---

## 综合评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 理论深度 | ★★★★☆ | 在线优化理论应用 |
| 方法创新 | ★★★★★ | 在线成像框架创新 |
| 实现难度 | ★★★☆☆ | 算法清晰 |
| 应用价值 | ★★★★★ | SKA等大项目需求强 |
| 论文质量 | ★★★★☆ | 理论实验充分 |

**总分：★★★★☆ (4.2/5.0)**

---

## 🪤 阅读陷阱（Reading Pitfalls）

1. **"online" 不是"实时低延迟"那种 online**：这里指**沿数据获取流程逐块处理**（streaming/online optimisation，Shalev-Shwartz 2011），核心是"用完即丢"，不是延迟指标。
2. **质量等价 ≠ 质量更好**：本文刻意证明 online 与 offline **重建质量相同**（M31 同为 14.2946 dB）。它的卖点是**存储与时序**，不是 SNR 提升。读 Table 1 时要看的是"相对差极小"，而非"谁更高"。
3. **假设 29 是收敛的命门**：Theorem 3.2 的单调下降**依赖**式 29（未观测块之和随同化下降）。这是温和但**非无条件**的假设，论文自己把"验证/放宽假设 29"列为未来方向。不要把收敛当作无前提的定理。
4. **storage 不只来自 visibilities**：PDF §4.2 提醒——baseline 坐标、权重、NUFFT interpolation kernel 也占存储（kernel 可达 measurement 的 16+ 倍）。$\eta_s=1/B$ 只描述 visibility 部分。
5. **simulation vs 真实观测**：实验是从公开射电图**模拟**生成 visibilities（on-grid masked-FFT），不是真实 uv-track 观测；真实 off-grid 要换 degridding/NUFFT。
6. **年份双口径**：PDF 是 2017（arXiv v1），刊出是 2019。引用时注明各自含义。

---

## 复现判断

> 本节为项目固定纪律小节：诚实标注本仓库对该论文的复现真实等级，**绝不把 synthetic/proxy 结果夸大为论文级**。

| 维度 | 判断 |
|------|------|
| 复现等级 (`reproductionLevel`) | **toy** |
| 真实性 (`reproductionTruthLevel`) | **toy-completed** |
| paper-level 进度 | **0/15**，本篇亦为 0 |
| runner | `reproduce/experiments/online_ri_toy.py`（experiment_id=`online_ri_toy`，指标确定性；wall-clock 为亚秒级、随运行波动，不作复现指标） |
| 当前实现 | **真实 online forward-backward（Algorithm 2, analysis form）**：$\Phi_k=M_k F$ 正交 masked-FFT + Daubechies-8 正交小波 Ψ（pywt）软阈值 prox + 按 B=8 块累加部分梯度 + 丢弃机制；并行 standard offline FB 基线 |
| 当前指标 | offline_snr_db=24.6404，online_snr_db=24.5580，dirty_snr_db=15.1086，online_offline_rel_diff=0.003347，B=8，M=1229，peak_stored=154，$\eta_s$=0.1253(≈1/B) |
| 用的 proxy（仍存在） | 64×64 synthetic 多结构图代替 M31/Cygnus A/W28/3C288；on-grid masked-FFT 代替真实 NUFFT/w-projection 算子；全平面 ~30% 覆盖代替半 Fourier 平面 10% 采样；数据项 $1/2\sigma^2$ 折进 $\mu$（$\mu=0.005$ 而非论文 $10^4$ 尺度）|
| 缺什么（到 paper-like） | (1) M31/Cygnus A/W28/3C288 公开图 + Puy variable-density 半 Fourier 平面 10% 采样；(2) NUFFT/degridding/w-projection（PURIFY 风格）真实 RI 算子；(3) 论文规模 $B\in\{50..500\}$、256×256+、$\mu=10^4$ 同尺度；(4) 算力比 $\eta_c$ 与扫 $B$ 曲线、Table 1 与 M31≈14.2946 dB 逐数值对照 |

**诚实结论**：本轮已把"假动作"换成**真算法**——求解器现在是真实的 ℓ1-Daubechies-8-正则化 online forward-backward（不再是 dirty image）。它**定性复现**了论文核心机制：online（24.56 dB）≈ offline（24.64 dB）（rel.diff≈3e-3，与论文 Cygnus A/W28 的 10⁻²~10⁻³ 同档），两者都显著优于 dirty image（15.11 dB），且峰值存储 $\eta_s=0.1253\approx1/B$ 精确印证论文式 50。但它仍是 **toy** 等级：用 synthetic 圆盘 + on-grid masked-FFT 近似真实 RI 算子，**不复现**论文 M31 的具体 14.2946 dB 与 Table 1 量级数值。真实 NUFFT/w-projection visibility 算子与真实观测数据仍是缺口。任何"论文级 online RI 已复现"的陈述仍是错误的。

---

## 完整复现流程

本篇的"完整复现流程 (Complete Reproduction Workflow)"规范文档已单独成文，覆盖论文身份核验、诚实分级、算法 step-by-step、所需数据集、基线、指标与论文报告数值、当前 toy 实现、差距分析、运行步骤与风险说明。

➡️ 详见：[`../reproduce/paper_like/workflows/online-ri_reproduction_workflow.md`](../reproduce/paper_like/workflows/online-ri_reproduction_workflow.md)

---

*本笔记由5-Agent辩论分析系统生成，结合了多智能体精读报告内容；本轮基于 arXiv:1712.04462v1 全文（14 页）做 grounding 增强，并按项目纪律修正了若干无 PDF 依据的数值/场景陈述。*
