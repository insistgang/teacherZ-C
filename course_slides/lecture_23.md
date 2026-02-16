# 第二十三讲：张量分解与网络压缩

## Tensor Decomposition and Network Compression

---

### 📋 本讲大纲

1. 网络压缩概述
2. 张量基础
3. CP分解
4. Tucker分解
5. 张量分解在CNN中的应用

---

### 23.1 网络压缩动机

#### 深度学习的挑战

```
模型参数：数亿到千亿
存储需求：GB级别
计算需求：高性能GPU
部署困难：移动端、边缘设备
```

#### 压缩方法

| 方法 | 原理 |
|------|------|
| 剪枝 | 移除不重要的连接 |
| 量化 | 降低精度 |
| 知识蒸馏 | 教师-学生 |
| 张量分解 | 低秩近似 |

---

### 23.2 张量基础

#### 定义

张量是向量和矩阵的推广：
- 标量：0阶张量
- 向量：1阶张量
- 矩阵：2阶张量
- 高阶张量：3阶及以上

#### 符号

$$\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$$

N阶张量

#### 模-n展开

$$\mathcal{X}_{(n)} \in \mathbb{R}^{I_n \times \prod_{k \neq n} I_k}$$

将张量展开为矩阵

---

### 23.3 张量分解概述

#### 目标

用低秩因子近似高阶张量：
$$\mathcal{X} \approx \hat{\mathcal{X}}$$

其中 $\hat{\mathcal{X}}$ 由少量参数表示

#### 主要方法

- **CP分解** (CANDECOMP/PARAFAC)
- **Tucker分解**
- **张量列分解** (TT)
- **张量环分解** (TR)

---

### 23.4 CP分解

#### 形式

$$\mathcal{X} \approx \sum_{r=1}^R \mathbf{a}_r^{(1)} \circ \mathbf{a}_r^{(2)} \circ \cdots \circ \mathbf{a}_r^{(N)}$$

- $R$：CP秩
- $\circ$：外积
- $\mathbf{a}_r^{(n)}$：第n模因子向量

#### 参数量

原始：$I_1 \times I_2 \times \cdots \times I_N$
CP分解后：$R \times (I_1 + I_2 + \cdots + I_N)$

---

### 23.5 Tucker分解

#### 形式

$$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{A}^{(1)} \times_2 \mathbf{A}^{(2)} \times \cdots \times_N \mathbf{A}^{(N)}$$

- $\mathcal{G}$：核心张量
- $\mathbf{A}^{(n)}$：因子矩阵
- $\times_n$：模-n乘积

#### 参数量

$$\prod_{n=1}^N R_n + \sum_{n=1}^N I_n R_n$$

其中 $R_n \leq I_n$ 是Tucker秩

---

### 23.6 CP vs Tucker

| 方面 | CP | Tucker |
|------|----|----|
| 结构 | 对称 | 非对称 |
| 秩 | 单一秩R | 多秩$(R_1, ..., R_N)$ |
| 参数量 | 较少 | 较多 |
| 表达能力 | 较弱 | 较强 |
| 优化难度 | 困难 | 相对容易 |

#### 特殊情况

- CP：核心张量为超对角
- Tucker：更一般的分解

---

### 23.7 卷积层的张量分解

#### 4D卷积核

$$\mathcal{W} \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$$

#### CP分解卷积

将$K \times K$卷积分解为：
$$1 \times K \text{ 卷积} + K \times 1 \text{ 卷积}$$

进一步分解通道维度

#### Tucker-2分解

只分解通道维度：
$$\mathcal{W} \approx \mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V}$$

---

### 23.8 分解的加速效果

#### 计算复杂度

| 方法 | 复杂度 |
|------|--------|
| 原始卷积 | $O(C_{out} \cdot C_{in} \cdot K^2 \cdot HW)$ |
| CP分解 | $O(R \cdot (C_{out} + C_{in} + K + K) \cdot HW)$ |
| Tucker | $O(R_1 R_2 K^2 HW + C_{out} R_1 HW + C_{in} R_2 HW)$ |

#### 加速比

当 $R \ll \min(C_{out}, C_{in})$ 时显著加速

---

### 23.9 张量列分解(TT)

#### 形式

$$\mathcal{X}(i_1, ..., i_N) = \sum_{r_1, ..., r_{N-1}} \mathbf{G}_1(i_1, r_1) \mathbf{G}_2(r_1, i_2, r_2) \cdots \mathbf{G}_N(r_{N-1}, i_N)$$

#### TT秩

$(r_1, r_2, ..., r_{N-1})$

#### 参数量

$$\sum_{n=1}^N r_{n-1} I_n r_n$$

通常 $r_n$ 较小，参数量大大减少

---

### 23.10 分解后微调

#### 流程

```
1. 预训练原始网络
2. 应用张量分解
3. 初始化分解后的层
4. 微调恢复精度
```

#### 微调策略

- 较低学习率
- 渐进式分解
- 知识蒸馏辅助

---

### 23.11 与其他压缩方法结合

#### 分解 + 剪枝

先分解，再剪枝小权重

#### 分解 + 量化

分解后使用低精度

#### 多方法组合

综合使用多种压缩技术

---

### 23.12 实践建议

#### 层选择

- 对大卷积层分解效果好
- 小层可能不划算

#### 秩选择

- 通过验证集选择
- 可使用自动秩搜索

#### 精度-效率权衡

- 分解秩越高，精度越好，加速越少
- 需要在目标设备上测试

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│       张量分解与网络压缩                         │
├─────────────────────────────────────────────────┤
│                                                 │
│   CP分解：                                      │
│   X ≈ Σᵣ a_r^(1) ◦ a_r^(2) ◦ ... ◦ a_r^(N)    │
│   参数量：R × ΣIₙ                              │
│                                                 │
│   Tucker分解：                                  │
│   X ≈ G ×₁ A^(1) ×₂ A^(2) × ... ×ₙ A^(N)      │
│   参数量：ΠRₙ + ΣIₙRₙ                          │
│                                                 │
│   TT分解：                                      │
│   X(i₁,...,iₙ) = G₁·G₂·...·Gₙ                 │
│   参数量：Σrₙ₋₁Iₙrₙ                            │
│                                                 │
│   应用：卷积层分解 → 加速 + 压缩               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **推导题**：推导CP分解的参数量公式

2. **实现题**：实现2D卷积的CP分解

3. **实验题**：对ResNet卷积层进行Tucker分解并微调

4. **分析题**：比较CP和Tucker分解在精度-压缩率上的权衡

---

### 📖 扩展阅读

1. **教材**：
   - Kolda & Bader, "Tensor Decompositions and Applications", SIAM Review, 2009

2. **经典论文**：
   - Lebedev et al., "Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition", ICLR 2015
   - Kim et al., "Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications", ICLR 2016

3. **工具**：
   - TensorLy (Python)
   - T3F (TensorFlow)

---

### 📖 参考文献

1. Kolda, T.G. & Bader, B.W. (2009). Tensor decompositions and applications. *SIAM Review*, 51(3), 455-500.

2. Lebedev, V., et al. (2015). Speeding-up convolutional neural networks using fine-tuned CP-decomposition. *ICLR*.

3. Kim, Y.D., et al. (2016). Compression of deep convolutional neural networks for fast and low power mobile applications. *ICLR*.

4. Novikov, A., et al. (2015). Tensorizing neural networks. *NeurIPS*.
