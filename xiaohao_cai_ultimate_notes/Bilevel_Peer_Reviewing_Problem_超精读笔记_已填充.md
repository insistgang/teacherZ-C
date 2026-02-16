# A Bilevel Formalism for the Peer-Reviewing Problem 超精读笔记（已填充版）

## 论文元信息

| 属性 | 内容 |
|------|------|
| **论文标题** | A Bilevel Formalism for the Peer-Reviewing Problem |
| **作者** | Gennaro Auricchio, Ruixiao Zhang, Jie Zhang, Xiaohao Cai |
| **发表单位** | University of Bath, University of Southampton |
| **发表年份** | 2023 |
| **arXiv编号** | arXiv:2307.12248v1 |
| **关键词** | Peer-reviewing, Bilevel programming, Matching problem, Fair allocation |
| **代码** | https://github.com/Galaxy-ZRX/Bilevel-Review |

---

## 中文摘要翻译

由于越来越多的会议收到大量投稿，找到一种自动化的方式在审稿人之间很好地分配提交的论文已成为必要。我们将同行评审匹配问题建模为双层规划（BP）公式。我们的模型包括一个描述审稿人视角的下层问题和一个描述编辑视角的上层问题。每个审稿人都希望最小化他们的总工作量，而编辑者希望找到一种能最大化评审质量并尽可能遵循审稿人偏好的分配。据我们所知，所提出的模型是第一个通过考虑两个目标函数来表述同行评审匹配问题的模型，一个目标函数描述审稿人的观点，另一个描述编辑的观点。我们证明了上层和下层问题都是可行的，并且我们的BP模型在温和假设下承认一个解。在研究了解的性质后，我们提出了一个启发式方法来解决我们的模型，并将其性能与相关的最先进方法进行了比较。大量的数值结果表明，我们的方法可以找到更公平的解决方案，具有竞争力的质量和更少的审稿人工作量。

---

## 第一部分：数学家Agent（理论分析）

### 1.1 问题背景

#### 1.1.1 同行评审分配问题

**核心任务**：将n篇论文分配给m个审稿人
- 论文集：P = [n]
- 审稿人集：R = [m]
- 目标：在满足约束条件下最大化评审质量

#### 1.1.2 传统ILP模型（Problem 1）

```
max <W, X>
s.t.
    li ≤ Σj X_ij ≤ ui,  ∀i ∈ [n]
    Σi X_ij ≤ Uj,          ∀j ∈ [m]
    X ∈ {0,1}^(n×m)
```

其中：
- W：质量矩阵，W_ij表示审稿人j评审论文i的预期质量
- li, ui：论文i需要的最少/最多审稿人数
- Uj：审稿人j最多能审的论文数

### 1.2 双层规划模型（Problem 2）

#### 1.2.1 核心思想

**Stackelberg博弈框架**：
- **领导者（Leader）**：编辑（Editor）
- **跟随者（Follower）**：审稿人（Reviewers）

**三阶段过程**：
1. 编辑提出方案Z（编辑向每位审稿人建议论文）
2. 审稿人竞标Y（审稿人从建议中选择愿意评审的论文）
3. 编辑最终分配X（编辑确定最终分配方案）

#### 1.2.2 数学表述

**上层问题（ULp）**：
```
max <W_E, X> + <Y*, X>
s.t.
    li ≤ Σj X_ij ≤ ui
    Σi X_ij ≤ Uj
    Σi Z_ij = Uj + φj
    X ≤ E - Z + Y*
```

**下层问题（LLp）**：
```
Y* ∈ argmin <(W_R)j, Yj>
s.t.
    0 ≤ Y ≤ Z
    Σi Yij = Uj
```

其中：
- W_E：编辑视角的质量矩阵
- W_R：审稿人视角的努力矩阵
- φj：审稿人j的自由度（可拒绝的论文数）

### 1.3 关键理论结果

#### 1.3.1 定理1：下层问题的等价性

**定理**：对于任意给定的Z，松弛的LP问题（2）和原始ILP问题（3）达到相同的最优值。

**条件**：如果每个审稿人对论文有严格的偏好顺序，则解唯一且为二进制。

#### 1.3.2 定理2：分量最小化与全局最小化

**定理**：Y是下层问题的解当且仅当Y最小化全局审稿人努力函数。

#### 1.3.3 定理3：解的存在性

**定理**：如果满足
```
max_j φj + 2 max_j Uj ≤ n
```
则问题存在可行解。

### 1.4 公平性概念

#### 1.4.1 定义1：符合度百分比（Accordance Percentage）

```
AC(X, Y, Z) := <X, Y> / <X, X>
```

- AC ∈ [0, 1]
- AC = 1 当且仅当 X ≤ Y（完美方案）

#### 1.4.2 定义2：Φ-弱公平性

**定义**：分配X是Φ-弱公平的，如果每个审稿人j没有被分配其φj个最差选择。

**命题1**：存在最大化质量的完美解当且仅当ILP问题存在Φ-弱公平解。

---

## 第二部分：工程师Agent（实现分析）

### 2.1 贪婪启发式算法

#### 2.1.1 算法设计

由于BP问题的复杂性，提出贪婪启发式：

**步骤1：编辑提案Z_g**
```
Z_g = argmax_Z <W_E, Z>
s.t.
    Σ_i Z_ij = Uj + φj,  ∀j ∈ [m]
    Z ∈ {0,1}^(n×m)
```

**步骤2：审稿人响应Y***
```
Y* = argmin_Y <W_R, Y>
s.t.
    0 ≤ Y ≤ Z_g
    Σ_i Yij = Uj
```

**步骤3：最终分配X_Z_g,Y_g***
```
X = argmax_X <W_E, X>
s.t.
    容量约束
    X ≤ E - Z_g + Y*
```

### 2.2 Python实现框架

```python
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import product

class BilevelPeerReview:
    """
    Bilevel Programming formulation for Peer-Reviewing Problem
    """

    def __init__(self, quality_matrix, effort_matrix,
                 papers_min=None, papers_max=None,
                 reviewer_capacity=None, freedom_degree=None):
        """
        Parameters:
        -----------
        quality_matrix : ndarray (n_papers, n_reviewers)
            Editor's perspective quality matrix
        effort_matrix : ndarray (n_papers, n_reviewers)
            Reviewer's effort matrix (higher = more effort)
        papers_min : array-like, minimal reviews per paper
        papers_max : array-like, maximal reviews per paper
        reviewer_capacity : array-like, max papers per reviewer
        freedom_degree : array-like, papers reviewer can refuse
        """
        self.W_E = quality_matrix
        self.W_R = effort_matrix
        self.n, self.m = quality_matrix.shape

        # Set defaults
        self.papers_min = np.array([3] * self.n) if papers_min is None else np.array(papers_min)
        self.papers_max = np.array([5] * self.n) if papers_max is None else np.array(papers_max)
        self.U = np.array([reviewer_capacity] * self.m) if reviewer_capacity is None else np.array(reviewer_capacity)
        self.phi = np.array([freedom_degree] * self.m) if freedom_degree is None else np.array(freedom_degree)

    def greedy_proposal(self):
        """
        Step 1: Editor proposes papers to reviewers (Z_g)
        Maximizes total quality of proposals
        """
        Z = np.zeros((self.n, self.m), dtype=int)

        # For each reviewer, select U_j + phi_j papers with highest quality
        for j in range(self.m):
            n_proposals = int(self.U[j] + self.phi[j])
            # Select papers with highest quality (descending)
            proposals = np.argsort(-self.W_E[:, j])[:n_proposals]
            Z[proposals, j] = 1

        return Z

    def reviewer_response(self, Z):
        """
        Step 2: Reviewers bid for papers (Y*)
        Each reviewer minimizes their total effort
        """
        Y = np.zeros((self.n, self.m), dtype=int)

        for j in range(self.m):
            # Get papers proposed to this reviewer
            proposed = np.where(Z[:, j] == 1)[0]

            if len(proposed) > 0:
                # Select U_j papers with minimum effort
                # (lowest effort values)
                efforts = self.W_R[proposed, j]
                n_select = int(min(self.U[j], len(proposed)))

                # Select papers with minimum effort
                selected_indices = np.argsort(efforts)[:n_select]
                selected = proposed[selected_indices]

                Y[selected, j] = 1

        return Y

    def final_assignment(self, Z, Y):
        """
        Step 3: Editor makes final assignment (X)
        Maximum weight matching with constraints
        """
        # Create cost matrix for assignment problem
        # We want to maximize quality, so use negative for minimization
        cost = -self.W_E.copy()

        # Apply consistency constraint: can only assign if not refused
        refused = (Z - Y) > 0
        cost[refused] = -np.inf  # Cannot assign refused papers

        # Solve assignment problem using Hungarian algorithm
        # This is a simplified version - full implementation would handle
        # all constraints properly
        row_ind, col_ind = linear_sum_assignment(cost)

        X = np.zeros((self.n, self.m), dtype=int)
        X[row_ind, col_ind] = 1

        return X

    def greedy_solve(self):
        """
        Full greedy heuristic solution
        Returns (X, Y, Z)
        """
        # Step 1: Editor's proposal
        Z = self.greedy_proposal()

        # Step 2: Reviewers' response
        Y = self.reviewer_response(Z)

        # Step 3: Final assignment
        X = self.final_assignment(Z, Y)

        return X, Y, Z

    def compute_metrics(self, X, Y, Z, X_ilp=None):
        """
        Compute evaluation metrics
        """
        # Quality Percentage
        quality_X = np.sum(self.W_E * X)
        if X_ilp is not None:
            quality_max = np.sum(self.W_E * X_ilp)
            qp = quality_X / quality_max if quality_max > 0 else 0
        else:
            qp = None

        # Reviewers' Average Effort Ratio
        efforts = self.W_R * X
        active_reviewers = np.sum(np.any(X > 0, axis=0))
        if active_reviewers > 0:
            avg_effort = np.sum(efforts) / active_reviewers
        else:
            avg_effort = 0

        # Fairness Ratio (variance of efforts)
        reviewer_efforts = np.sum(efforts, axis=0)
        variance = np.var(reviewer_efforts[reviewer_efforts > 0])

        # Accordance Percentage
        accordance = np.sum(X * Y) / np.sum(X) if np.sum(X) > 0 else 1

        return {
            'quality_percentage': qp,
            'average_effort': avg_effort,
            'fairness_variance': variance,
            'accordance': accordance
        }


def generate_synthetic_data(n_papers=73, n_reviewers=189, random_state=42):
    """
    Generate synthetic quality and effort matrices for testing

    Scenarios:
    1. Aligned: W_R inversely related to W_E
    2. Random: W_R independent of W_E
    """
    np.random.seed(random_state)

    # Generate quality matrix (from topic vectors)
    # Simulate by random values in [0, 1]
    W_E = np.random.rand(n_papers, n_reviewers)

    # Scenario 1: Aligned
    # Higher quality -> lower effort (with noise)
    K = 1.0
    sigma = 0.1
    chi = np.random.normal(0, sigma, (n_papers, n_reviewers))
    W_R_aligned = K - W_E + chi
    W_R_aligned = np.maximum(W_R_aligned, 0.01)  # Ensure positive

    # Scenario 2: Random
    W_R_random_uniform = np.random.uniform(0, 1, (n_papers, n_reviewers))
    W_R_random_exp = np.random.exponential(0.5, (n_papers, n_reviewers))

    return W_E, W_R_aligned, W_R_random_uniform, W_R_random_exp


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    W_E, W_R_aligned, W_R_unif, W_R_exp = generate_synthetic_data(
        n_papers=20, n_reviewers=30
    )

    # Parameters
    reviewer_capacity = 6
    freedom_degree = 4

    # Solve with aligned effort matrix
    bp_model = BilevelPeerReview(
        quality_matrix=W_E,
        effort_matrix=W_R_aligned,
        reviewer_capacity=reviewer_capacity,
        freedom_degree=freedom_degree
    )

    X, Y, Z = bp_model.greedy_solve()

    # Compute metrics
    metrics = bp_model.compute_metrics(X, Y, Z)

    print("Bilevel Peer Review Allocation")
    print("=" * 40)
    print(f"Papers assigned: {np.sum(X)}")
    print(f"Reviewers utilized: {np.sum(np.any(X > 0, axis=0))}")
    print(f"Accordance: {metrics['accordance']:.3f}")
```

### 2.2 约束满足检查

```python
def check_feasibility(X, papers_min, papers_max, reviewer_capacity):
    """
    Check if assignment X satisfies all constraints
    """
    n, m = X.shape

    # Check paper constraints
    paper_review_counts = np.sum(X, axis=1)
    paper_valid = np.all((paper_review_counts >= papers_min) &
                        (paper_review_counts <= papers_max))

    # Check reviewer constraints
    reviewer_counts = np.sum(X, axis=0)
    reviewer_valid = np.all(reviewer_counts <= reviewer_capacity)

    return paper_valid and reviewer_valid
```

---

## 第三部分：应用专家Agent（价值分析）

### 3.1 应用场景

#### 3.1.1 大型学术会议

- **NeurIPS、ICML、CVPR**等顶级会议
- 数千篇论文需要在几天内分配
- 数百名审稿人参与

#### 3.1.2 期刊评审

- **Nature、Science**等期刊
- 持续的论文投稿流
- 需要快速、公平的分配

### 3.2 实验结果分析

#### 3.2.1 数据集

使用SIGIR 2007数据集：
- 73篇论文
- 189名潜在审稿人
- 25个主题标签

#### 3.2.2 评估指标

1. **质量百分比（QP）**：X_BP相对于X_ILP的质量
2. **审稿人平均努力比（RAER）**：相对于ILP的平均努力
3. **公平比（FR）**：方差比（越小越公平）

#### 3.2.3 主要发现（表2）

**对齐情况（质量与努力负相关）**：
- QP > 95%（质量接近最优）
- RAER = 0.86-0.89（审稿人努力降低10-14%）
- FR = 0.57-0.75（公平性显著提升）

**随机情况**：
- QP = 93-97%（质量略降）
- RAER = 0.30-0.87（努力大幅降低）
- FR = 0.07-0.61（公平性大幅提升）

### 3.3 与现有方法比较

| 方法 | 优点 | 缺点 |
|------|------|------|
| 纯ILP | 最大化质量 | 不考虑审稿人偏好 |
| t-调优ILP | 平衡质量和努力 | 需要调参t |
| BP模型 | 自动平衡，更公平 | 计算复杂度略高 |

### 3.4 实际价值

1. **更公平的分配**：方差降低15-75%
2. **更低的审稿人负担**：平均努力降低10-50%
3. **竞争性的质量**：质量保持90%以上
4. **无需调参**：自动平衡编辑和审稿人目标

---

## 第四部分：怀疑者Agent（批判分析）

### 4.1 论文优势

#### 4.1.1 理论创新

1. **首次BP建模**：第一个用双层规划建模同行评审问题
2. **双重视角**：同时考虑编辑和审稿人的目标
3. **理论保证**：解的存在性和可行性证明

#### 4.1.2 实验验证

1. **多种场景**：对齐、随机（均匀/指数）
2. **公平性量化**：引入公平性指标
3. **综合比较**：与ILP、t-调优ILP对比

### 4.2 潜在问题

#### 4.2.1 理论层面

1. **NP难问题**：BP问题本质上是NP难的，启发式可能不是最优
2. **假设限制**：
   - 需要预先知道W_E和W_R
   - 线性目标函数可能过于简化

3. **唯一性条件**：解的唯一性需要严格的偏好顺序

#### 4.2.2 实验层面

1. **合成数据**：努力矩阵W_R是合成的，非真实数据
2. **规模限制**：实验规模（73篇论文）相对较小
3. **缺乏真实验证**：没有在实际会议系统中验证

#### 4.2.3 实用性限制

1. **计算复杂度**：需要求解多个ILP问题
2. **参数敏感性**：φj的选择影响结果
3. **动态变化**：模型不处理审稿人动态退出

### 4.3 公平性问题

**何为"公平"？**
- 论文定义的"弱公平"（避免最差φj个选择）
- 但这可能与实际审稿人的公平感不同
- 未考虑历史公平性（过去审稿多的审稿人）

---

## 第五部分：综合理解Agent（Synthesizer）

### 5.1 核心贡献

#### 5.1.1 方法创新

**双层规划框架**：
```
编辑（上层）：max 质量 + 一致性
    ↓ 提议
审稿人（下层）：min 努力
    ↓ 竞标
编辑（上层）：最终分配
```

#### 5.1.2 关键洞察

1. **博弈论视角**：将审稿分配建模为Stackelberg博弈
2. **双向优化**：编辑和审稿人各自优化自己的目标
3. **内在公平性**：通过下层优化自然实现公平分配

### 5.2 与经典ILP的关系

**命题2**：当φj = 0时，BP解与ILP解等价。

这意味着：
- BP模型是ILP的推广
- 当审稿人无拒绝权时，退化为经典ILP
- 当φj > 0时，引入审稿人偏好

### 5.3 算法复杂度

| 步骤 | 问题类型 | 复杂度 |
|------|----------|--------|
| 编辑提案 | 最大权匹配 | O(n³) |
| 审稿人响应 | m个独立问题 | O(m·n·log(n)) |
| 最终分配 | 最大权匹配 | O(n³) |
| **总计** | - | O(n³ + m·n·log(n)) |

### 5.4 局限与未来方向

#### 5.4.1 当前局限

1. **静态模型**：不处理动态变化
2. **二进制分配**：论文不可分割
3. **单编辑假设**：只有一个编辑

#### 5.4.2 未来方向

1. **概率分配**：允许论文被多个审稿人部分评审
2. **多编辑场景**：扩展到多个编辑协调
3. **机器学习**：用ML预测质量矩阵和努力矩阵
4. **在线学习**：根据反馈动态调整

### 5.5 学术价值评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 理论创新 | ⭐⭐⭐⭐⭐ | 首创BP建模同行评审 |
| 算法设计 | ⭐⭐⭐⭐ | 贪婪启发式有效 |
| 实验验证 | ⭐⭐⭐⭐ | 多种场景验证 |
| 实用价值 | ⭐⭐⭐⭐ | 会议分配系统潜在应用 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，数学严谨 |
| **综合评分** | **4.2/5.0** | 优秀的理论贡献 |

### 5.6 关键要点总结

1. **双层规划框架**：建模编辑-审稿人博弈
2. **理论保证**：解的存在性和等价性证明
3. **贪婪启发式**：实用的求解方法
4. **公平性提升**：显著降低审稿人努力方差
5. **质量保持**：90%+的最优质量水平

---

*笔记生成时间：2024年*
*基于arXiv:2307.12248v1*
