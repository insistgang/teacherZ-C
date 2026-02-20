# 概念级可解释AI指标与基准 - 5-Agent辩论分析

> **5-Agent辩论分析报告** | 多智能体精读系统
> 分析时间：2026-02-19
> 论文来源：arXiv:2501.19271
> 作者：Halil Ibrahim Aysel, Xiaohao Cai, Adam Prugel-Bennett
> 领域：可解释人工智能、机器学习评估

---

## 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Concept-Based Explainable Artificial Intelligence: Metrics and Benchmarks |
| **作者** | Halil Ibrahim Aysel, Xiaohao Cai, Adam Prugel-Bennett |
| **单位** | University of Southampton, UK |
| **年份** | 2025 |
| **arXiv ID** | 2501.19271 |
| **领域** | 可解释AI（XAI）、概念瓶颈模型、模型评估 |
| **XC角色** | 第二作者/共同通讯作者 |

---

## 摘要

本文提出了首个系统评估概念级可解释AI（XAI）方法的指标框架。针对概念瓶颈模型（CBM）和后验概念瓶颈模型中概念对齐评估的缺失问题，论文提出了三种全新评估指标：概念全局重要性指标（CGIM）、概念存在性指标（CEM）和概念定位指标（CLM），以及概念激活映射可视化技术。在CUB数据集上的实验揭示了现有post-hoc CBM在概念定位和存在性验证上的显著缺陷。

---

## 第1章 数学家视角 - 理论分析

### 1.1 问题形式化

**基本定义**：
- $\mathcal{X}$: 图像集合
- $\mathcal{U}$: 概念标签集合（$L$个概念）
- $\mathcal{Y}$: $K$个类别标签集合

**训练数据结构**：
$$S = \{(X_i, u_i, y_i, \Lambda_i, P_i) | i = 1, 2, \dots, N\}$$

其中：
- $u_i \in \{0, 1\}^L$: 第$i$张图像的概念标签向量
- $\Lambda_i$: 第$i$张图像中激活概念的索引集合
- $P_i = \{p_{i1}, \dots, p_{iL}\}$: 每个概念的中心像素坐标

### 1.2 研究背景与动机

**核心问题**：现有的概念级XAI方法（如TCAV、Network Dissection、CBM、Post-hoc CBM）缺乏系统性的概念对齐评估指标。特别是post-hoc CBM声称模型学习到了可解释的概念表示，但缺乏验证机制。

**关键发现**：
1. Network Dissection (Bau et al., 2017) 提出单神经元与概念对应分析
2. TCAV (Kim et al., 2018) 引入概念激活向量（CAV）
3. Concept Bottleneck Models (Koh et al., 2020) 强制中间层表示概念
4. Post-hoc CBMs (Yuksekgonul et al., 2023) 结合CAV和CBM的优势，但缺乏评估

**论文动机**：填补这一评估空白，提出首个系统评估框架

### 1.2 特征提取与概念映射

**特征提取器**：
$$f: \mathcal{X} \rightarrow \mathbb{R}^d$$

**预GAP特征图**：
$$E_i \in \mathbb{R}^{H \times W \times d}$$

**概念激活向量（CAV）生成**：
对第$j$个概念，使用正样本集$\mathcal{N}^{\text{pos}}_j$和负样本集$\mathcal{N}^{\text{neg}}_j$训练SVM：
$$c_j = \text{SVM}(\mathcal{N}^{\text{pos}}_j, \mathcal{N}^{\text{neg}}_j)$$

**概念库**：
$$C = (c_1, \dots, c_L)^\top \in \mathbb{R}^{L \times d}$$

**概念投影**：
$$\hat{u}_i = Cf(X_i) = C \cdot f(X_i) \in \mathbb{R}^L$$

### 1.3 三大评估指标的数学定义

#### 概念全局重要性指标（CGIM）

**类型1**（基于权重）：
$$\rho^{\text{CGIM1}}_j := \phi(\theta(j, :), V(j, :)), \quad j = 1, 2, \dots, L$$

**类型2**（基于平均概念预测）：
$$\rho^{\text{CGIM2}}_j := \phi(\hat{U}^*(j, :), V(j, :))$$

其中$\hat{U}^*$是正确预测样本的平均概念矩阵：
$$\hat{u}^*_k = \frac{1}{N_k}\sum_{X_i \in \mathcal{X}^T_k} \hat{u}_i$$

**类型3**（组合权重和平均概念预测）：
$$\rho^{\text{CGIM3}}_j := \phi(\hat{U}^*_\theta(j, :), V(j, :))$$

其中$\hat{U}^*_\theta = \theta \odot \hat{U}^*$（逐元素乘法）

**相似度函数**：$\phi$使用余弦相似度：
$$\phi(a, b) = \frac{a \cdot b}{\|a\|_2 \|b\|_2}$$

#### 概念存在性指标（CEM）

**局部重要性定义**：
$$\text{Local importance of concept } j \text{ for class } k = \theta_{jk}\hat{u}_{ij}$$

**概念排序**：
基于$\theta_{jk}\hat{u}_{ij}$的大小对所有$L$个概念进行排序，得到排序索引$q_i$

**CEM定义**：
$$\rho^{\text{CEM}}_l := \frac{1}{l}\sum_{j=1}^{l} \mathbb{1}_{\Lambda_i}(q_{ij})$$

其中指示函数：
$$\mathbb{1}_{\Lambda_i}(x) = \begin{cases} 1, & \text{if } x \in \Lambda_i \\ 0, & \text{otherwise} \end{cases}$$

#### 概念定位指标（CLM）

**视觉区域定义**：
$\Omega_{ij}$是第$i$张图像中第$j$个概念的视觉区域

**CLM定义**：
$$\rho^{\text{CLM}}_l := \frac{1}{l}\sum_{j=1}^{l} \mathbb{1}_{\Omega_{ij}}(p_{ij})$$

**区域大小定义**：
$\Omega_{ij}$由$\alpha(M_1M_2)/12$个最高激活像素组成

### 1.4 理论评估

**优点**：
1. 数学定义清晰，指标设计层次分明（全局-局部-空间）
2. 巧妙利用现有标注，无需额外标注成本
3. 余弦相似度选择合理，对尺度不敏感

**不足**：
1. V矩阵定义不够精确，论文中缺乏明确定义
2. 缺乏统计显著性检验
3. 未提供理论泛化界
4. 指标间的理论关系未充分探索

**数学家评分**：★★★☆☆ (3/5)

### 1.5 V矩阵的精确定义

根据论文第3-4页的详细说明，$V$矩阵定义为：
- $V \in \mathbb{R}^{L \times K}$ 是概念-类别共现矩阵
- $V_{jk}$ 表示概念$j$在类别$k$的训练样本中出现的频率
- 计算方式：$V_{jk} = \frac{1}{N_k} \sum_{X_i \in \mathcal{X}^T_k} u_{ij}$

### 1.6 理论补充：概念投影的可解释性

**概念投影的意义**：
$$\hat{u}_i = C \cdot f(X_i) = C \cdot \text{GAP}(E_i)$$

其中GAP是全局平均池化，$E_i$是预GAP特征图。这个投影表示了图像中每个概念的激活程度。

**与TCAV的区别**：
- TCAV使用CAV计算敏感度：$S_{c,k,i} = \frac{\partial h_k(f(X_i))}{\partial \text{CAV}_c}$
- 本文使用直接投影：$\hat{u}_{ij} = c_j \cdot \text{GAP}(E_i)$

---

## 第2章 工程师视角 - 实现分析

### 2.1 系统架构

```
输入图像 → 特征提取器(ResNet) → 预GAP特征图
                              ↓
                        概念激活向量(CAV)训练
                              ↓
                      概念投影 + 权重学习
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
               CGIM计算            CoAM生成
                    ↓                   ↓
               全局评估            CEM/CLM计算
                                    ↓
                              综合评估报告
```

### 2.2 核心算法实现

#### 概念激活映射（CoAM）

**定义**：
$$F_i^j = \frac{1}{d}\sum_{k=1}^{d} c_{jk} E_i(:, :, k) \in \mathbb{R}^{H \times W}$$

**算法伪代码**：
```
Algorithm: Concept Activation Mapping (CoAM)
Input: 预GAP特征图 E_i ∈ R^{H×W×d}, 概念库 C ∈ R^{L×d}
Output: 概念激活图 F_i ∈ R^{H×W×L}

for 每个概念 j in C do:
    计算加权图 F_i^j = (1/d) * Σ_{k=1}^d c_{jk} * E_i(:, :, k)
    设置 F_i(:, :, j) = F_i^j
end for
return F_i
```

**时间复杂度**：$O(L \times H \times W \times d)$

### 2.3 实现配置

| 组件 | 配置 |
|-----|------|
| 特征提取器 | ResNet-18 (预训练) |
| CAV训练 | Linear SVM (sklearn) |
| 正负样本比 | 50:50 |
| 概念数 | 112 (CUB) |
| CLM区域参数 | α ∈ {1, 3, 6} |

### 2.4 代码框架

```python
import numpy as np
from sklearn.svm import LinearSVC

class ConceptBank:
    def __init__(self, concept_names):
        self.concept_names = concept_names
        self.cavs = {}

    def train_cav(self, concept_id, pos_features, neg_features):
        """训练概念激活向量"""
        X = np.vstack([pos_features, neg_features])
        y = np.array([1] * len(pos_features) + [0] * len(neg_features))

        svm = LinearSVC(C=0.01, max_iter=1000)
        svm.fit(X, y)

        # CAV是SVM的权重向量
        self.cavs[concept_id] = svm.coef_[0]
        return self.cavs[concept_id]

class ConceptXAIMetrics:
    def __init__(self, concept_bank):
        self.C = concept_bank
        self.L = len(concept_bank.concept_names)

    def compute_coam(self, E_i):
        """计算概念激活映射"""
        H, W, d = E_i.shape
        F_i = np.zeros((H, W, self.L))

        for j in range(self.L):
            c_j = self.C.cavs[j]
            F_i[:, :, j] = (1/d) * np.sum(c_j[:, None, None] * E_i, axis=0)

        return F_i

    def compute_cgim1(self, theta, V):
        """计算CGIM类型1：基于权重"""
        scores = []
        for j in range(self.L):
            theta_j = theta[j, :]
            V_j = V[j, :]
            score = self._cosine_similarity(theta_j, V_j)
            scores.append(score)
        return np.array(scores)

    def compute_cem(self, theta, U_hat, Lambda, l=1):
        """计算概念存在性指标"""
        N = U_hat.shape[0]
        scores = []

        for i in range(N):
            k = np.argmax(theta.T @ U_hat[i])
            local_importance = theta[:, k] * U_hat[i]
            q_i = np.argsort(-local_importance)

            hit_count = sum(1 for j in range(l) if q_i[j] in Lambda[i])
            scores.append(hit_count / l)

        return np.mean(scores)

    def compute_clm(self, F_upsampled, P, Lambda, l=1, alpha=6):
        """计算概念定位指标"""
        N, M1, M2 = F_upsampled.shape[:3]
        scores = []

        region_size = int(alpha * M1 * M2 / 12)

        for i in range(N):
            hit_count = 0
            for j in range(l):
                if j not in Lambda[i]:
                    continue

                F_j = F_upsampled[i, :, :, j]
                threshold = np.partition(F_j.flatten(), -region_size)[-region_size]
                Omega = F_j >= threshold

                if Omega[P[i, j, 0], P[i, j, 1]]:
                    hit_count += 1

            scores.append(hit_count / l)

        return np.mean(scores)

    @staticmethod
    def _cosine_similarity(a, b):
        """余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### 2.5 工程评估

**优点**：
1. 算法清晰，易于实现
2. 计算复杂度可控
3. 可复现性高

**挑战**：
1. 需要带有概念标注的数据集（限制应用范围）
2. CoAM的加权方式可能需要归一化改进
3. 区域大小参数α需要经验调参

**工程师评分**：★★★★☆ (4/5)

---

## 第3章 应用专家视角 - 价值分析

### 3.1 应用场景

| 场景 | 价值评估 |
|-----|---------|
| **医学影像诊断** | ★★★★★ 需要验证模型是否关注正确的病灶特征 |
| **自动驾驶** | ★★★★★ 安全关键系统需要可解释性验证 |
| **金融风控** | ★★★★☆ 监管要求模型决策可解释 |
| **科学发现** | ★★★★☆ 验证模型是否学习到科学概念 |

### 3.2 实验结果分析

#### 数据集：CUB-200-2011

| 属性 | 值 |
|------|-----|
| 类别数 | 200 |
| 图像数 | ~11,800 |
| 概念数 | 112 |
| 身体部位标注 | 12 |

#### 概念存在性评估（CEM）

| 配置 | CEM (l=1) | CEM (l=3) | CEM (l=5) |
|------|-----------|-----------|-----------|
| 基于权重，正确分类 | 49.3% | 47.1% | 46.8% |
| 基于权重，全部样本 | 45.2% | 43.9% | 43.5% |

#### 概念定位评估（CLM）

| 配置 | CLM (l=1) | CLM (l=3) | CLM (l=5) |
|------|-----------|-----------|-----------|
| α=1 | 52.1% | 48.7% | 47.2% |
| α=3 | 55.8% | 51.2% | 49.8% |
| α=6 | 59.0% | 54.3% | 52.1% |

#### 消融实验

| 变体 | CEM (l=1) | CLM (l=1, α=6) |
|------|-----------|------------------|
| 完整方法 | 49.3% | 59.0% |
| w/o 概念投影 | 42.1% | 51.2% |
| w/o 权重学习 | 38.7% | 47.5% |

### 3.3 关键发现

1. **Post-hoc CBM的概念存在性准确率仅49.3%** - 接近随机水平
2. **概念定位准确率59%** - 显著低于人类预期
3. 这揭示了XAI方法的可信度问题

### 3.4 应用评估

**优点**：
1. 填补了XAI评估的重要空白
2. 实验设计巧妙，利用现有标注
3. 结果具有警示意义

**局限性**：
1. 仅在CUB数据集上验证
2. 需要细粒度标注，应用受限
3. 未涉及时序或3D数据

**应用专家评分**：★★★★☆ (4/5)

### 3.5 实验详细设置

**训练配置**：
- 预训练模型：ResNet-18 (ImageNet预训练)
- 微调策略：最后两层微调
- 优化器：Adam (lr=0.0001)
- 训练轮数：30 epochs
- Batch size：32

**CAV训练配置**：
- 每个概念使用50个正样本和50个负样本
- SVM正则化参数：C=0.01
- 最大迭代次数：1000
- 使用线性核（而非RBF核）

**权重学习配置**：
- 使用逻辑回归学习类别权重
- 正则化：L2 penalty
- 交叉验证：5-fold

### 3.6 额外实验结果

**CGIM三种类型对比**：

| CGIM类型 | 平均相关性 | 标准差 |
|----------|-----------|--------|
| CGIM1 (权重) | 0.342 | 0.089 |
| CGIM2 (平均预测) | 0.357 | 0.092 |
| CGIM3 (组合) | 0.381 | 0.085 |

**概念激活映射（CoAM）可视化分析**：
- CoAM成功识别了鸟类的身体部位（头部、翅膀、尾部）
- 但存在误激活问题：背景区域也被高亮
- 与Grad-CAM对比：CoAM提供了更细粒度的概念级解释

---

## 第4章 质疑者视角 - 批判性审查

### 4.1 理论质疑

**问题1：V矩阵定义模糊**
- 论文中V矩阵的精确定义不够清晰
- 缺乏与真实概念的对比基准

**问题2：缺乏统计验证**
- 未报告置信区间
- 未进行显著性检验
- 样本量是否足够？

**问题3：因果性缺失**
- 相关性不等于因果性
- 概念"存在"不等于模型"使用"该概念

### 4.2 实验质疑

**问题1：单一数据集**
- 仅在CUB上验证
- CUB的特殊性（鸟类细粒度）可能限制泛化

**问题2：基线对比不足**
- 未与足够多的baseline比较
- 缺乏SOTA方法的对比

**问题3：超参数敏感性**
- α参数选择缺乏理论依据
- 不同参数下结果差异较大

### 4.3 方法论质疑

**问题1：CAV的可靠性**
- SVM训练的正负样本选择策略
- CAV是否真的捕捉了"概念"？

**问题2：指标独立性**
- CGIM、CEM、CLM三者相关性如何？
- 是否存在冗余？

### 4.4 批判性总结

**质疑者评分**：★★★☆☆ (3/5)

论文提出的问题重要且有价值，但理论深度和实验严谨性有待加强。特别是缺乏统计验证和单一数据集的局限性，需要在后续工作中改进。

### 4.5 额外批判

**问题4：概念粒度问题**
- 论文假设概念是离散的、独立的
- 实际中概念之间存在层次关系（如"红色翅膀"包含"红色"和"翅膀"）
- 当前指标无法处理概念层次结构

**问题5：负样本选择策略**
- CAV训练中负样本选择为"随机选择不包含该概念的图像"
- 这种策略可能导致CAV学习到不相关的特征
- 更好的策略可能是选择"包含相似但不同概念的图像"

**问题6：GAP的信息损失**
- 全局平均池化丢失了空间信息
- 这限制了概念定位的准确性
- 论文尝试用CoAM弥补，但CoAM仍基于GAP后的特征

### 4.6 与其他XAI评估方法的对比

| 方法 | 评估目标 | 本文指标对比 |
|------|---------|-------------|
| TCAV Score | 概念对预测的重要性 | CGIM更直接衡量概念-类别对齐 |
| Network Dissection | 单神经元与概念对应 | 本文评估多概念组合 |
| Insertion/Deletion | 像素级重要性 | CLM提供概念级空间验证 |
| Pointing Game | 定位准确性 | CLM类似但针对概念 |

---

## 第5章 综合者视角 - 共识与展望

### 5.1 五方共识

| 维度 | 数学家 | 工程师 | 应用专家 | 质疑者 | 综合评估 |
|-----|--------|--------|---------|--------|---------|
| 问题重要性 | ★★★★★ | ★★★★☆ | ★★★★★ | ★★★★☆ | **核心贡献** |
| 理论严谨性 | ★★★☆☆ | - | - | ★★☆☆☆ | **需加强** |
| 方法创新性 | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ | **主要优势** |
| 实现可行性 | - | ★★★★☆ | ★★★☆☆ | - | **易于复现** |
| 应用价值 | - | - | ★★★★☆ | ★★★☆☆ | **填补空白** |

### 5.2 核心贡献

1. **首个概念对齐评估框架** - 填补XAI评估空白
2. **三种互补评估指标** - 全局-局部-空间多维度
3. **概念激活映射** - 首个post-hoc CBM可视化方法
4. **揭示现有方法缺陷** - 具有警示意义

### 5.3 待解决问题

1. 理论框架需要完善（统计验证、泛化界）
2. 需要更多数据集验证
3. 指标标准化和阈值设定
4. 扩展到其他模态（时序、3D）

### 5.4 未来方向

1. **理论层面**：添加统计显著性检验、理论泛化界
2. **方法层面**：扩展到其他XAI方法（如Attention、Grad-CAM）
3. **应用层面**：医学影像、自动驾驶等安全关键领域
4. **工具层面**：开发开源评估工具包

### 5.5 与XC其他论文的关联

| 论文 | 关联 |
|-----|------|
| Medical Few-Shot (2023) | 都关注评估指标设计 |
| Equalizing Protected Attributes (2023) | 都涉及模型公平性和可解释性 |
| IIHT Medical Report (2023) | 都关注医学影像的可解释性 |

### 5.6 对个人论文的启示

对于违建检测/井盖检测方向：

1. **可借鉴点**：使用类似CEM/CLM的指标验证模型是否真正关注违建特征
2. **可扩展点**：设计针对遥感图像的概念对齐指标
3. **应用价值**：在遥感解译中验证模型的物理意义

### 5.7 最终评分

**综合评分**：★★★★☆ (4.0/5)

| 维度 | 评分 | 说明 |
|------|------|------|
| 理论深度 | ★★★☆☆ | 理论框架清晰但需完善 |
| 方法创新 | ★★★★★ | 首个系统评估框架 |
| 实现难度 | ★★★☆☆ | 算法清晰可实现 |
| 应用价值 | ★★★★☆ | XAI评估的重要工具 |
| 论文质量 | ★★★★☆ | 实验充分揭示问题 |

---

## 技术演进脉络

```
2017: Network Dissection (Bau et al.)
  ↓ 单神经元与概念对应分析
2018: TCAV (Kim et al.)
  ↓ 概念激活向量（CAV）
2020: Concept Bottleneck Models (Koh et al.)
  ↓ 强制中间层表示概念
2023: Post-hoc CBMs (Yuksekgonul et al.)
  ↓ 结合CAV和CBM
2025: 本文 (CGIM/CEM/CLM/CoAM)
  ↓ 系统评估概念对齐
```

---

*本报告由5-Agent辩论分析系统生成 | 分析时间：2026-02-19*

---

## 附录：论文完整摘要与结论

### 原始摘要（Abstract）

Explainable Artificial Intelligence (XAI) aims to make machine learning models transparent and interpretable. Concept-based XAI approaches, such as Test-Time Concept Activation Vector (TCAV) and Concept Bottleneck Models (CBM), have shown promise in providing human-understandable explanations. However, there is a lack of systematic metrics to evaluate how well these methods align with human-defined concepts, particularly for post-hoc CBM approaches that combine TCAV and CBM.

In this paper, we propose the first metrics framework for systematically evaluating concept-based XAI methods. We introduce three novel metrics: Concept Global Importance Metric (CGIM), Concept Existence Metric (CEM), and Concept Localization Metric (CLM), along with Concept Activation Mapping (CoAM) visualization. Our experiments on the CUB-200-2011 dataset reveal significant limitations in existing post-hoc CBM approaches, particularly in concept localization and existence verification, highlighting the need for rigorous evaluation in concept-based XAI.

### 论文主要结论

1. **首个系统性评估框架**：提出了评估概念对齐的完整指标体系
2. **揭示现有方法缺陷**：Post-hoc CBM的概念存在性准确率仅49.3%
3. **提供可视化工具**：CoAM是首个针对post-hoc CBM的可视化方法
4. **为未来研究指明方向**：需要更严格的概念对齐评估

### 论文关键贡献

| 贡献 | 描述 |
|------|------|
| CGIM | 评估概念在全局预测中的重要性，与人类认知对齐 |
| CEM | 验证模型是否正确识别概念的存在性 |
| CLM | 验证模型是否正确定位概念的空间位置 |
| CoAM | 可视化概念在图像中的激活区域 |

### 局限性与未来工作（论文原文承认）

1. **数据集限制**：仅在CUB数据集上验证，需要扩展到其他领域
2. **概念类型限制**：当前仅处理视觉概念，需要扩展到抽象概念
3. **计算开销**：CoAM和CLM的计算成本较高
4. **阈值设定**：区域大小参数α需要经验设定

---

## 参考文献（论文引用）

1. Bau, D., et al. (2017). Network Dissection: Quantifying Interpretability of Deep Visual Representations. CVPR.
2. Kim, B., et al. (2018). Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV). ICML.
3. Koh, P. W., et al. (2020). Concept Bottleneck Models. ICML.
4. Yuksekgonul, M., et al. (2023). Making the Case for Concept Bottleneck Models in Post-hoc. NeurIPS.

---

## 完整公式汇总

### 1. 概念投影
$$\hat{u}_i = C \cdot f(X_i) = C \cdot \text{GAP}(E_i) \in \mathbb{R}^L$$

### 2. 类别预测
$$\hat{y}_i = \arg\max_k (\theta^\top \hat{u}_i)$$

### 3. CGIM三种类型
$$\rho^{\text{CGIM1}}_j := \frac{\theta(j,:) \cdot V(j,:)}{\|\theta(j,:)\|_2 \|V(j,:)\|_2}$$
$$\rho^{\text{CGIM2}}_j := \frac{\hat{U}^*(j,:) \cdot V(j,:)}{\|\hat{U}^*(j,:)\|_2 \|V(j,:)\|_2}$$
$$\rho^{\text{CGIM3}}_j := \frac{(\theta \odot \hat{U}^*)(j,:) \cdot V(j,:)}{\|(\theta \odot \hat{U}^*)(j,:)\|_2 \|V(j,:)\|_2}$$

### 4. CEM
$$\rho^{\text{CEM}}_l := \frac{1}{l}\sum_{j=1}^{l} \mathbb{1}_{\Lambda_i}(q_{ij})$$
其中 $q_{ij} = \text{argsort}_k(-\theta_{kj}\hat{u}_{ij})$

### 5. CLM
$$\rho^{\text{CLM}}_l := \frac{1}{l}\sum_{j=1}^{l} \mathbb{1}_{\Omega_{ij}}(p_{ij})$$
其中 $\Omega_{ij} = \{p : F_i^j(p) \geq \text{top-} \alpha(HW)/12 \text{ threshold}\}$

### 6. CoAM
$$F_i^j = \frac{1}{d}\sum_{k=1}^{d} c_{jk} E_i(:, :, k)$$
