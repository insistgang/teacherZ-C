# Reproduction Engineering Report — #3 iterated-rof

> 角色：论文复现工程师
> 审查日期：2026-06-06
> 工具基线：PyMuPDF 1.26.5（已装）、Python 3.9、`reproduce/sync_to_dashboard.mjs --check` 已 `match`
> 严格按 6 条原则 + 11-section 模板

---

## 1. Paper Identity Check

| 字段 | PDF 实际 | 项目记录（`docs/js/reading-data.js` + 笔记） | 核验 |
|---|---|---|---|
| 英文标题 | Multiclass Segmentation by Iterated ROF Thresholding | `paperNotesV2[2].titleEn` 同左 | ✅ |
| 作者顺序 | Xiaohao Cai and Gabriele Steidl | `paperNotesV2[2].titleEn` 注释 + `papers[2].authors` "Xiaohao Cai, Gabriele Steidl" | ✅ |
| 第一作者 | Xiaohao Cai | `papers[2].authors` 以 Xiaohao Cai 开头 | ✅ |
| 年份 | 2013 | `paperNotesV2[2].year: 2013` | ✅ |
| 出版 | LNCS 8081, pp. 237–250 (EMMCVPR 2013, Springer-Verlag) | `paperNotesV2[2].type: "LNCS / EMMCVPR"` + `papers[2].type: "LNCS / EMMCVPR"` | ✅ |
| 页数 | PDF 14 页（封面+13 正文） | `papers[2].pages: 14` | ✅ |
| 文件名 | `多类ROF分割 Iterated ROF.pdf` | `papers[2].file` 同左 | ✅ |
| 文件存在 | docs/00_papers_first_author_xiaohao_cai_deduped/多类ROF分割 Iterated ROF.pdf | validate.mjs 通过 | ✅ |

**Pass**。身份元信息无任何 issue。

---

## 2. Core Reproduction Target

**论文核心贡献**（PDF p.242 Algorithm T-ROF + Theorem 1）：

> Given a degraded image `f`, the T-ROF algorithm **first solves the ROF model once** to obtain a smoothed image `u`, then iteratively refines K-1 thresholds `τ_i = 1/2(m_{i-1} + m_i)` based on segment means, until the threshold sequence converges. Under assumption (A) (Eq. 16) and the projected modification (`Pn` projector), the threshold sequence `τ^(k)` converges (Theorem 1, p.244).

**核心算法结构**：
- **Step 1**：解一次 ROF: `min_u TV(u) + μ/2 ∫(u - f)² dx`  → 得 `u*`
- **Step 2**：迭代 K-1 阈值 `τ_i := {x : u*(x) > τ_i}` → 计算 `Ω_i := Σ_i \ Σ_{i+1}` → 算 `m_i := mean_f(Ω_i)` → 更新 `τ_i^(k+1) := 1/2(m_{i-1}^(k) + m_i^(k))`
- **投影版**：`τ^(k+1) := P_n(τ^(k+1))`，n 大到机器精度

**理论关键点**：
- **Proposition 3**（p.241）：`{x : u(x) > τ}` solves E(·,τ) iff `u` solves ROF model — 这就是"只解一次 ROF"的理论根据
- **Lemma 2**：阈值/均值序列单调性（`0 ≤ m_0 ≤ τ_1 ≤ m_1 ≤ ... ≤ τ_{K-1} ≤ m_{K-1}`）
- **Lemma 3**：sign changes `s_k` 单调不增
- **Theorem 1**：投影版 T-ROF 在 assumption (A) 下收敛

**实验对象**（PDF p.246+）：cartoon、texture、medical images；重点是 close-gray-value 多相分割；K=2 与 Chan-Vese 等价关系（Proposition 2）。

**复现该论文 = 必须实现**：
1. 一个 ROF solver（**不能是 Gaussian proxy**）
2. 阈值迭代更新 `τ_i = 1/2(m_{i-1} + m_i)`，用 `mean_f(Ω_i)` on **raw f**（per Eq. 15）
3. close-gray-value 多相 synthetic benchmark
4. 记录 Lemma 2/3 的单调性 / sign changes
5. 必须有 `fidelityWarning`：toy-to-partial，不能 paper-level

---

## 3. Current Project Repro Status

### 3.1 静态声明（`docs/js/reading-data.js` 行 1086–1108）

| 字段 | 当前值 | 核验 |
|---|---|---|
| `reproductionLevel` | `toy-to-partial` | ✅ valid 等级（reproduce/README.md 行列 37） |
| `reproductionTruthLevel`（经 mapping）| `partial-completed`（toy-to-partial includes "partial"） | ✅ |
| `resultStatus` | `completed` | ✅ 与 run 一致 |
| `runtimeSeconds` | `0.4396` | ✅ 与 run 一致 |
| `runMetrics` | `{raw_kmeans_accuracy: 0.659, trof_accuracy: 0.9799, threshold_iterations: 3}` | ✅ 与 run 一致 |
| `resultFiles` | `["assets/repro/sat_demo.png", "assets/repro/trof_thresholds.png"]` | ✅ 文件存在 |
| `fidelityWarning` | `"Uses proxy smoothing before threshold updates; strict T-ROF should solve ROF once."` | ✅ 明确 |
| `notes` | `"This synthetic toy implements the threshold update tau_i = 1/2(m_{i-1}+m_i) after proxy smoothing; strict T-ROF should solve ROF once."` | ✅ 明确 |
| `difficultyScore` | `3` | ✅ |
| `effectScore` | `5` | ⚠️ **偏激进**：toy 0.32 gain 配 5/5 不合理（见 §3.3） |
| `effectLabel` | `"很明显"` | ⚠️ 与 effectScore=4 也叫"很明显"重复，标尺无 5 |

### 3.2 实际运行结果（`reproduce/results/repro_results.json` mtime Jun 5 15:01）

```json
{
  "priority": 3, "id": "iterated-rof", "experiment_id": "sat_rof_trof",
  "reproductionLevel": "toy-to-partial", "status": "completed",
  "runtime_seconds": 0.4396,
  "metrics": {"raw_kmeans_accuracy": 0.659, "trof_accuracy": 0.9799, "threshold_iterations": 3},
  "resultFiles": ["assets/repro/sat_demo.png", "assets/repro/trof_thresholds.png"],
  "notes": "..."
}
```

### 3.3 sync 校验

`node reproduce/sync_to_dashboard.mjs --check` → `dashboard repro fields match latest run results` ✅

### 3.4 当前实现代码（`reproduce/experiments/sat_rof_trof.py` 行 23–52, 101–105）

**关键问题**：

| # | 论文元素 | 当前实现 | 缺口 | 严重度 |
|---|---|---|---|---|
| 1 | **求解 ROF（Step 1）** | `scipy.ndimage.gaussian_filter(image, sigma=1.0)` | Gaussian 没有 TV regularization，没有 data fidelity term。**这不是 ROF** | must-fix（要 partial 升级） |
| 2 | **K-1 阈值（Step 2.1）** | `np.digitize(smooth4, thresholds)` + nested Σ 用 `np.quantile` 初始化 | 用 quantile 不是论文里 "0 < τ_1^(0) < ... < τ_{K-1}^(0) < 1" 的任意初始化；nested Σ (Eq. 14) 显式实现为 digitize 略简化 | should-fix |
| 3 | **m_i := mean_f(Ω_i) Eq. 15** | `smooth4[trof_labels == k].mean()` — **用 smoothed image 算均值** | 论文 Eq. 15 是 `mean_f(Ω_i)` on **raw f**，不是 on smoothed `u` | must-fix（语义错误） |
| 4 | **τ_i = 1/2(m_{i-1} + m_i)** | ✅ `np.array([(means[i] + means[i+1]) / 2 ...])` | OK | — |
| 5 | **Projected T-ROF (Pn)** | ❌ 没用 projection | Lemma 2/3 和 Theorem 1 都需要 projection | should-fix（要 partial 升级） |
| 6 | **Assumption (A) 验证** | ❌ 没检查 | assumption (A) 是 Theorem 1 唯一前提 | should-fix（要 partial 升级） |
| 7 | **Lemma 2 单调性** | ❌ 没记录 | 这是 convergence 关键指标 | should-fix |
| 8 | **Lemma 3 sign changes** | ❌ 没记录 | `s_k` monotone decreasing | should-fix |
| 9 | **K=2 ↔ Chan-Vese Proposition 2** | ❌ 只跑 K=4 | 论文 Proposition 2 是 K=2 关键贡献 | should-fix |
| 10 | **close-gray-value 强度** | levels `[0.26, 0.36, 0.47, 0.58]`，相邻差 ≈ 0.1 | 论文强调灰度差很小的多相合成，差 0.1 偏大 | should-fix |
| 11 | **实验对象 cartoon/texture/medical** | ❌ 只有 synthetic 4-phase square | 没有真实数据 | **不补**（这正是 paper-level 缺口，partial 没必要补） |
| 12 | **效果评分 effectScore=5** | toy 0.32 gain 配 5/5 偏强 | | should-fix |

### 3.5 第一步审查小结

**诚实性**：
- resultStatus / runtimeSeconds / runMetrics / resultFiles / notes / fidelityWarning **全部与实际 run 一致** ✅
- reproductionLevel 命名 `toy-to-partial` 是 valid 等级，**不夸大** ✅
- 旧快照问题（`docs/assets/repro/repro_results.json` mtime Jun 5 15:01，已是 completed）— 我之前误读为 skipped 已纠正
- **唯一诚实性问题**：effectScore=5 配 0.32 toy gain 偏激进，effectLabel="很明显" 标尺无 5

**有效性**：
- 核心算法"只解一次 ROF + 迭代阈值"**没有真正实现** — Step 1 用 Gaussian proxy 完全绕过 ROF
- m_i 计算用 smoothed image 而不是 raw f，违反 Eq. 15
- 投影 / 单调性 / assumption (A) 都没验证
- **当前实现本质是"smoothing + quantile-init K-means"**，不是 T-ROF

---

## 4. Gap To Paper-Level

按 user 任务定义：
- toy：只验证机制，synthetic
- partial：实现核心算法一部分，缺原数据/完整 baseline
- paper-like：严肃 solver、参数、数据或公开替代、指标对照
- paper-level：原数据 + 原算法 + 主要 baseline + 主要指标

**iterated-rof 距离 paper-level 缺**：

| 缺口 | 描述 | partial 升级 | paper-like 升级 | paper-level |
|---|---|---|---|---|
| ROF solver | Gaussian proxy → 严肃 TV minimizer | Chambolle-Pock dual projection 或 Split Bregman | 加 Chambolle 2005 收敛步数 + 数据保真梯度步长分析 | 多组 μ 参数扫描 + 对比 [Goldstein-Osher split Bregman]、[Chambolle-Pock CP]、[Boyd proximal] |
| 数据 | synthetic 4-phase | 增强 close-gray-value 难度 | 加公开 texture 库（e.g. Brodatz） | 论文图复现（PDF Fig. 2-3 的 cartoon + texture + medical） |
| Baseline | 只有 K-means | 加 Otsu 多阈值 | 加 Chan-Vese (Mumford-Shah 凸松弛) + graph cuts | 论文 [20-30] 的方法列表 |
| Proposition 2 (K=2↔Chan-Vese) | 没验证 | 加 K=2 toy + 验证 λ = μ/(2(m_1 - m_0)) 关系 | K=2 segmentation 跟 Chan-Vese 公式 (4) 的 lambda 对照 | 完整理论数值重现 |
| Theorem 1 收敛性 | 跑通即停 | 记录 Lemma 2 单调性 + Lemma 3 sign changes + Assumption (A) | 加 projected T-ROF 与原版对比的收敛速度分析 | 完整收敛证明的数值重现（无法做到 paper-level，因为 Theorem 1 是数学证明，不是数值实验） |
| 实际图像 | 0 | 0 | medical 公开数据集（e.g. DRIVE 视网膜） | 论文 [35] 实验完整复现 |
| speed claim | 没量化 | 加 runtime 分解（ROF 求解 vs 阈值更新） | 加 solver 切换的 CPU time 对比 | paper Table 1-2 完整复现 |

**结论**：**iterated-rof 在 partial 升级是合理且可达的目标**；paper-like 需要外部 texture 数据 + 严肃 ROF solver 对比；**paper-level 需要论文图复现 + 论文 Table 1-2 baseline 完整对照 + 完整 medical 数据集**，**项目当前不应该冲 paper-level**（user 推荐顺序也明确把 #3 列为最优先 partial 升级目标）。

---

## 5. Proposed Minimal Reproduction (partial 升级)

### 5.1 目标分级

| 等级 | 实现内容 | 何时 |
|---|---|---|
| **partial**（本次） | 真 ROF solver（Chambolle-Pock） + K=2 验证 Proposition 2 + K=4 close-gray-value (差 0.04) + Lemma 2/3 指标 + projected T-ROF + Assumption (A) 检查 + warning 完整 | 立即可做 |
| paper-like（下次） | Brodatz texture + 1 个 medical 数据集 + 至少 1 个 baseline（Otsu 多阈值 / Chan-Vese convex relaxation） | 需外部数据 |
| paper-level（不冲） | 论文 Fig. 2-3 完整复现 + paper Table 1-2 baseline 对照 | 不建议 |

### 5.2 partial 实现方案

**核心思路**：保留现有 `sat_rof_trof.py` 的 `iterated-rof` 路径，**不替换** Gaussian proxy（作为 baseline 保留），而是**新增**一个 `sat_rof_trof.py` 内部的 `solver` 开关：

```python
solver = "chambolle_pock" | "split_bregman" | "gaussian_proxy"
```

跑出 3 组 metrics，分别标 fidelityWarning：
- Gaussian proxy：toy baseline（保留）
- Chambolle-Pock：partial 主结果
- Split Bregman：partial 备选

### 5.3 算法规格（partial）

```
输入: f ∈ [0,1]^{n×n}, K=4, μ=0.1, n_iter_max=12, eps_drift=1e-4
输出: labels ∈ {0,1,2,3}^{n×n}, metrics dict, figures

1. u = rof_solve(f, μ, solver="chambolle_pock")  # 严格 ROF
2. τ^(0) = np.linspace(1/(2K), 1 - 1/(2K), K-1)  # 论文允许任意 init，选均匀
3. repeat k = 0, 1, ..., n_iter_max:
   3.1. Σ_i^(k) := {x : u(x) > τ_i^(k)},  i = 1..K-1
   3.2. Ω_0 := Ω; Ω_i := Σ_i \ Σ_{i+1} for i=1..K-2; Ω_{K-1} := Σ_{K-1}
   3.3. m_i^(k) := mean_f(Ω_i^(k))   # 关键：用 raw f 算，不是 u
   3.4. τ_i^(k+1) := 1/2(m_{i-1}^(k) + m_i^(k))   # projected
   3.5. record drift_k = max|τ^(k+1) - τ^(k)|
   3.6. check monotonicity: assert 0 ≤ m_0 ≤ τ_1 ≤ m_1 ≤ ... ≤ τ_{K-1} ≤ m_{K-1}
   3.7. if drift_k < eps_drift: break
4. labels = argmin_i |f - m_i^(*)|  # 用 f 跟最终 m_i 比较分
5. compute metrics: accuracy, threshold_iterations, drift_history, monotonicity_violated, sign_changes
```

### 5.4 K=2 toy（验证 Proposition 2）

```
输入: f two-phase, μ, K=2
1. u = rof_solve(f, μ)
2. τ_1^(0) = 0.5
3. m_0 = mean_f(f < u_thresh), m_1 = mean_f(f >= u_thresh)
4. τ_1^* = 1/2(m_0 + m_1)
5. λ_derived := μ / (2(m_1 - m_0))   # per Proposition 2
6. compare against direct Chan-Vese with parameter λ_derived on same f
7. report dice_rof_threshold, dice_chan_vese, |λ_derived - λ_true|
```

---

## 6. Implementation Plan

### 6.1 代码修改

**主改文件**：`reproduce/experiments/sat_rof_trof.py`

**新增依赖检查**：
```python
def require_modules_for_rof(*solvers):
    # all 走 numpy + scipy 即可
    return require_modules("numpy", "matplotlib", "scipy")
```

**新增 ROF solver 函数**（独立函数，便于单测）：

```python
def rof_chambolle_pock(f, mu, n_iter=200, tau=0.25, sigma=1/tau/8, tol=1e-4):
    """
    Chambolle-Pock primal-dual algorithm for ROF.
    min_u TV(u) + (mu/2) ||u - f||^2
    
    Proximal of TV: p^{k+1} = (p^k + sigma * K u_bar^k) / (1 + sigma * |K u_bar^k|)
    Proximal of data: u^{k+1} = (u^k + tau * (-K^T p^{k+1}) + tau * mu * f) / (1 + tau * mu)
    """
    import numpy as np
    h, w = f.shape
    p_y = np.zeros_like(f)
    p_x = np.zeros_like(f)
    u = f.copy()
    for it in range(n_iter):
        u_bar = u.copy()
        # primal update (gradient descent on -K^T p)
        div_p = np.zeros_like(f)
        div_p[1:-1, :] += p_y[1:-1, :] - p_y[:-2, :]
        div_p[:, 1:-1] += p_x[:, 1:-1] - p_x[:, :-2]
        u_new = (u + tau * (-div_p) + tau * mu * f) / (1 + tau * mu)
        # dual update (project gradient)
        grad_u_y = np.zeros_like(f)
        grad_u_x = np.zeros_like(f)
        grad_u_y[:-1, :] = u_new[1:, :] - u_new[:-1, :]
        grad_u_x[:, :-1] = u_new[:, 1:] - u_new[:, :-1]
        p_y_new = (p_y + sigma * grad_u_y) / (1 + sigma * np.abs(grad_u_y))
        p_x_new = (p_x + sigma * grad_u_x) / (1 + sigma * np.abs(grad_u_x))
        if np.linalg.norm(u_new - u) / max(np.linalg.norm(u), 1e-9) < tol:
            u = u_new
            break
        u = u_new
        p_y, p_x = p_y_new, p_x_new
    return u
```

```python
def rof_split_bregman(f, mu, lam=0.01, n_iter=200, tol=1e-4):
    """
    Split Bregman for ROF via FFT.
    min_u TV(u) + (mu/2) ||u - f||^2
    
    频域闭式解 g = F^{-1} [F(mu*f - lam*div(d-b)) / (mu - lam*laplacian)]
    """
    import numpy as np
    h, w = f.shape
    dx = np.zeros_like(f)
    dy = np.zeros_like(f)
    bx = np.zeros_like(f)
    by = np.zeros_like(f)
    
    # 频域符号
    yy, xx = np.mgrid[:h, :w]
    laplacian = 2 * (np.cos(2*np.pi*xx/w) + np.cos(2*np.pi*yy/h)) - 4
    denom = mu - lam * laplacian
    denom[0, 0] = 1  # avoid div by zero
    
    u = f.copy()
    for it in range(n_iter):
        u_old = u.copy()
        # g 子问题 (FFT 闭式)
        div_db = np.zeros_like(f)
        div_db[1:-1, :] += (dx[1:-1, :] - bx[1:-1, :]) - (dx[:-2, :] - bx[:-2, :])
        div_db[:, 1:-1] += (dx[:, 1:-1] - bx[:, 1:-1]) - (dx[:, :-2] - bx[:, :-2])
        rhs = mu * f - lam * div_db
        u = np.real(np.fft.ifft2(np.fft.fft2(rhs) / denom))
        # d 子问题 (shrinkage)
        sy = grad_y(u) + by
        sx = grad_x(u) + bx
        s = np.sqrt(sy**2 + sx**2) + 1e-10
        dy = np.maximum(s - 1/lam, 0) * sy / s
        dx = np.maximum(s - 1/lam, 0) * sx / s
        # Bregman 更新
        by += grad_y(u) - dy
        bx += grad_x(u) - dx
        if np.linalg.norm(u - u_old) / max(np.linalg.norm(u), 1e-9) < tol:
            break
    return u
```

**重写 T-ROF 主函数**（partial 版本）：

```python
def run_iterated_rof_partial():
    """partial reproduction: Chambolle-Pock ROF + T-ROF with Lemma 2/3 checks."""
    import numpy as np
    elapsed = timer()
    rng = np.random.default_rng(SEED)
    
    # K=4 close-gray-value (差 0.04，比 current 0.1 更难)
    n = 96
    yy, xx = np.mgrid[:n, :n]
    truth4 = np.zeros((n, n), dtype=int)
    truth4[(yy < 48) & (xx >= 48)] = 1
    truth4[(yy >= 48) & (xx < 48)] = 2
    truth4[(yy >= 48) & (xx >= 48)] = 3
    levels = np.array([0.28, 0.32, 0.36, 0.40])  # close-gray-value
    image4 = levels[truth4] + rng.normal(0, 0.05, (n, n))  # 噪声 σ=0.05
    
    # 1. 严肃 ROF
    mu = 8.0
    u_rof = rof_chambolle_pock(image4, mu, n_iter=300)
    
    # 2. T-ROF with K-1=3 thresholds
    K = 4
    thresholds = np.linspace(1/(2*K), 1 - 1/(2*K), K-1)
    history = [thresholds.copy()]
    means_history = []
    monotonicity_violated = False
    sign_changes_history = []
    
    for it in range(20):
        # Σ_i = {x : u(x) > τ_i}
        # Ω_0 := Ω; Ω_i := Σ_i \ Σ_{i+1}; Ω_{K-1} := Σ_{K-1}
        sorted_tau = np.sort(np.concatenate([[0.0], thresholds, [1.0]]))
        labels = np.digitize(u_rof, sorted_tau[1:-1])  # 0..K-1
        # m_i := mean_f(Ω_i)  on RAW f
        means = np.array([image4[labels == k].mean() if (labels == k).any() else 0 for k in range(K)])
        means_history.append(means.copy())
        # 更新阈值 (Eq. 15)
        new_thresholds = np.array([(means[i] + means[i+1]) / 2 for i in range(K-1)])
        # 投影
        new_thresholds = np.clip(new_thresholds, 0, 1)
        # sign changes
        if it > 0:
            signs = np.sign(new_thresholds - history[-1])
            sign_changes = int(np.sum(np.diff(signs) != 0))
            sign_changes_history.append(sign_changes)
        # Lemma 2 单调性检查
        expected_mono = [means[i] for i in range(K-1)]
        mono_ok = all(means[i] <= new_thresholds[i] <= means[i+1] for i in range(K-1))
        if not mono_ok:
            monotonicity_violated = True
        history.append(new_thresholds.copy())
        if np.max(np.abs(new_thresholds - thresholds)) < 1e-4:
            break
        thresholds = new_thresholds
    
    # 3. 计算指标
    final_means = means_history[-1]
    final_labels = np.digitize(u_rof, np.sort(np.concatenate([[0.0], history[-1], [1.0]]))[1:-1])
    # 用 f 跟最终 m_i 比较分（更接近论文 Eq. 15 含义）
    dist = np.abs(image4[..., None] - final_means[None, None, :])
    pred_labels = dist.argmin(axis=2)
    
    from common import clustering_accuracy
    acc_rof_trof = clustering_accuracy(truth4, pred_labels)
    acc_raw_kmeans = clustering_accuracy(truth4, simple_kmeans(image4.reshape(-1, 1), K, seed=SEED).reshape(n, n))
    
    runtime = elapsed()
    return {
        "raw_kmeans_accuracy": round(acc_raw_kmeans, 4),
        "gaussian_proxy_trof_accuracy": <current 0.9799 for comparison>,
        "rof_trof_accuracy": round(acc_rof_trof, 4),
        "threshold_iterations": len(history) - 1,
        "max_threshold_drift": round(float(np.max(np.abs(np.diff(history, axis=0)))), 6),
        "monotonicity_violated": monotonicity_violated,
        "sign_changes_final": sign_changes_history[-1] if sign_changes_history else 0,
        "rof_iterations_chambolle_pock": <实际 ROF solver 步数>,
        "runtime_seconds_total": round(runtime, 4),
        "runtime_seconds_rof": <rof solver 单独计时>,
        "runtime_seconds_threshold": round(runtime - <rof 部分>, 4),
    }, {
        "sat_demo": <update sat_demo.png with 3 panels: raw / gaussian / rof>,
        "trof_thresholds": <update trof_thresholds.png with drift over iterations>,
        "trof_convergence": <new figure: threshold drift + monotonicity over iters>,
        "rop_residual": <new figure: Chambolle-Pock residual over iters>,
    }
```

**输出优先级**：
1. `repro_results.json` 字段更新（详见 §9）
2. `docs/assets/repro/iterated_rof_convergence.png`（新增，**thresholds 收敛轨迹 + 单调性指示**）
3. `docs/assets/repro/iterated_rof_chanvese.png`（新增，**K=2 case λ 关系 + Chan-Vese 对照**）

### 6.2 依赖

| 依赖 | 用途 | 必装 |
|---|---|---|
| `numpy` | 矩阵运算 | ✅ |
| `scipy.ndimage.gaussian_filter` | baseline (toy 保留) | ✅ |
| `scipy.fftpack` 或 `numpy.fft` | Split Bregman 频域求解 | ✅ (numpy 自带) |
| `matplotlib` | 出图 | ✅ |
| `scikit-image` | 暂不需要 | ❌ |
| `scikit-learn` | 暂不需要 | ❌ |

**不需要新增依赖**。partial 升级完全在 numpy + scipy 范围内。

### 6.3 单元测试

```python
# reproduce/tests/test_iterated_rof.py
def test_chanvese_relationship_k2():
    """Proposition 2: K=2 时, λ = μ/(2(m_1 - m_0))."""
    ...
def test_monotonicity_holds_under_assumption_a():
    """Lemma 2: 阈值/均值序列单调性."""
    ...
def test_sign_changes_monotone():
    """Lemma 3: sign changes s_k 单调不增."""
    ...
def test_rof_solver_converges_to_known_solution():
    """f = const 时, ROF 解应接近 f."""
    ...
```

### 6.4 工作流

```bash
# 1. 改 reproduce/experiments/sat_rof_trof.py
# 2. 加 reproduce/tests/test_iterated_rof.py
# 3. 跑测试
python -m pytest reproduce/tests/test_iterated_rof.py -v
# 4. 跑复现
python reproduce/run_all.py
# 5. 校验一致性
node reproduce/sync_to_dashboard.mjs --check
# 6. 跑项目校验
node docs/scripts/validate.mjs
```

---

## 7. Metrics And Expected Outputs

### 7.1 新增 runMetrics（partial 升级后）

| 字段 | 类型 | 含义 | 期望范围 |
|---|---|---|---|
| `raw_kmeans_accuracy` | float | K-means baseline accuracy | 0.50-0.70 (close-gray-value 难) |
| `gaussian_proxy_trof_accuracy` | float | 旧 toy T-ROF accuracy | 0.70-0.90 (Gaussian proxy 偏强) |
| `rof_trof_accuracy` | float | **新增**：Chambolle-Pock ROF + T-ROF | 0.80-0.95 |
| `threshold_iterations` | int | 阈值收敛迭代次数 | 3-10 |
| `max_threshold_drift` | float | max\|τ_new - τ_old\| 最终值 | < 1e-4 |
| `monotonicity_violated` | bool | Lemma 2 是否被违反 | False (在 assumption A 下) |
| `sign_changes_final` | int | 最终 sign changes s_k | 0-2 |
| `rof_iterations_chambolle_pock` | int | ROF solver 收敛步数 | 100-300 |
| `runtime_seconds_total` | float | 总耗时 | 1-3 秒 |
| `runtime_seconds_rof` | float | **新增**：ROF 单独耗时 | 0.5-2 秒 |
| `runtime_seconds_threshold` | float | **新增**：阈值更新耗时 | < 0.1 秒 |

### 7.2 期望输出（基于论文 Proposition 2/3 推导）

- **`rof_trof_accuracy` ≥ `gaussian_proxy_trof_accuracy` 的 80%**：ROF 求解是 partial 升级，不是为了数字上一定胜过 Gaussian（Gaussian 在 noise σ=0.05 上可能更稳）
- **`monotonicity_violated = False`**：在 close-gray-value 但分段均衡的合成上，assumption (A) 应该成立
- **`sign_changes_final` ≤ 2**：Lemma 3 期望 s_k 收敛到小值
- **`threshold_iterations` 在 5-10 之间**：比当前 toy 的 3 略多（更难的 case）

### 7.3 输出 figure

| 文件 | 内容 | 用途 |
|---|---|---|
| `assets/repro/sat_demo.png` | 3 列：raw K-means / Gaussian proxy T-ROF / ROF T-ROF + ground truth | dashboard 展示 |
| `assets/repro/trof_thresholds.png` | 4 条曲线：τ_1, τ_2, τ_3 over iterations + drift 收敛 | dashboard 展示 |
| **新增** `assets/repro/iterated_rof_convergence.png` | 双 y 轴：max drift (left) + sign changes (right) over iterations | dashboard 展示，验证 Lemma 2/3 |
| **新增** `assets/repro/iterated_rof_chanvese.png` | K=2 case：原始 f / ROF 解 / ROF-threshold segmentation / Chan-Vese with λ=μ/(2(m_1-m_0)) / diff | dashboard 展示，验证 Proposition 2 |

---

## 8. Required Files To Change

| 文件 | 操作 | 优先级 |
|---|---|---|
| `reproduce/experiments/sat_rof_trof.py` | 大改：新增 `rof_chambolle_pock` + `rof_split_bregman` + 重写 `run_iterated_rof_partial`；保留 `run_iterated_rof_toy` 作 baseline 对照 | high |
| `reproduce/experiments/common.py` | 小改：可能需要 `psnr` / `snr` 已有；可能加 `dice_score` 已有 | low |
| `reproduce/tests/test_iterated_rof.py` | 新增 | medium |
| `reproduce/README.md` | 更新：标注 #3 iterated-rof 当前 partial 升级状态 | low |
| `docs/js/reading-data.js` 行 1086-1108 | 更新 `reproDetails[3]`：新 metrics、新 resultFiles、新 fidelityWarning、新 effectScore | high |
| `xiaohao_cai_ultimate_notes/Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md` | 更新"复现判断" 段：从 toy-to-partial 升级到 partial，列新指标 | low |
| `docs/assets/repro/iterated_rof_convergence.png` | 新增 | high |
| `docs/assets/repro/iterated_rof_chanvese.png` | 新增 | high |
| `docs/assets/repro/sat_demo.png` | 覆盖 | high |
| `docs/assets/repro/trof_thresholds.png` | 覆盖 | high |

**不要修改**：
- `reproduce/run_all.py`（保持 runner list 不变，只改 sat_rof_trof.py）
- `reproduce/sync_to_dashboard.mjs`（保持校验逻辑不变）
- `docs/scripts/validate.mjs`（当前校验已足够，effectLabel 标尺问题不需要改 validate.mjs）

---

## 9. Dashboard Fields To Update

`docs/js/reading-data.js` 行 1086–1108 `reproDetails[3]`：

```js
3: {
  reproductionLevel: "partial",                                          // [改] 从 "toy-to-partial"
  difficultyScore: 3,
  difficultyLabel: "中",
  effectScore: 3,                                                         // [改] 从 5 降到 3
  effectLabel: "明显",                                                    // [改] 从 "很明显" 改 "明显"
  fullReproductionFeasibility: "偏难。paper-level 需要真实 cartoon / texture / medical 数据 + 多组 μ 参数 + paper Table 1-2 baseline 完整对照；当前 partial 已用 Chambolle-Pock 严格 ROF solver。",
  minimalExperiment: "close-gray-value (差 0.04) 4-phase synthetic image + K=2 case；用 Chambolle-Pock ROF 求解一次 u，再迭代更新 τ_i = 1/2(m_{i-1}+m_i)，m_i := mean_f(Ω_i) on raw f。",
  expectedOutcome: "partial 复现：Chambolle-Pock ROF 后再阈值化应接近 ground truth；K=2 验证 Proposition 2 的 λ = μ/(2(m_1 - m_0)) 关系；记录 Lemma 2 单调性 + Lemma 3 sign changes。",
  metrics: [                                                             // [改] 扩展
    "raw_kmeans_accuracy",
    "gaussian_proxy_trof_accuracy",                                       // [新增] 保留 baseline
    "rof_trof_accuracy",                                                  // [新增] partial 主指标
    "threshold_iterations",
    "max_threshold_drift",                                                // [新增]
    "monotonicity_violated",                                              // [新增]
    "sign_changes_final",                                                 // [新增]
    "rof_iterations_chambolle_pock",                                      // [新增]
    "runtime_seconds_total", "runtime_seconds_rof", "runtime_seconds_threshold"  // [新增]
  ],
  dependencies: ["numpy", "scipy", "matplotlib"],
  dataRequirement: "synthetic close-gray-value 4-phase image + K=2 synthetic two-phase image；不需要下载真实数据。",
  computeRequirement: "CPU，约 2-3 秒内（Chambolle-Pock 300 步 + T-ROF ≤20 步）。",
  implementationRisk: "partial 升级：使用 Chambolle-Pock dual projection 代替 Gaussian proxy；用 mean_f(Ω_i) on raw f (per Eq. 15)；仍然不验证 Theorem 1 完整证明，也不复现论文 cartoon/texture/medical 实验。",
  verificationPlan: "(1) 对比 raw K-means / Gaussian proxy T-ROF / ROF T-ROF 三者 accuracy；(2) 验证 Lemma 2 单调性不违反；(3) 验证 Lemma 3 sign changes 单调不增；(4) K=2 case 验证 λ = μ/(2(m_1 - m_0)) 与 Chan-Vese convex relaxation 对照；(5) 保存 convergence figure 显示 drift + sign changes 收敛。",
  resultStatus: "completed",                                              // 跑通后
  experimentId: "sat_rof_trof",
  runtimeSeconds: 1.5,                                                    // [改] 预计 1-3 秒
  runMetrics: { /* 实际跑后填 */ },                                       // [改] 见 §7.1
  resultFiles: [                                                          // [改] 新增 2 个 png
    "assets/repro/sat_demo.png",
    "assets/repro/trof_thresholds.png",
    "assets/repro/iterated_rof_convergence.png",                          // [新增]
    "assets/repro/iterated_rof_chanvese.png"                              // [新增]
  ],
  fidelityWarning: "Partial reproduction uses Chambolle-Pock dual projection as the ROF solver (strict TV minimizer, not Gaussian proxy). Threshold update uses mean_f(Ω_i) on raw f per Eq. (15). Lemma 2 monotonicity and Lemma 3 sign changes are checked but Theorem 1 projected-T-ROF full convergence proof is not numerically reproduced. Paper-level still requires real cartoon/texture/medical data and paper Table 1-2 baseline comparison.",
  notes: "Partial reproduction: Chambolle-Pock ROF + iterative threshold update with Lemma 2/3 checks; K=2 case validates Proposition 2's λ = μ/(2(m_1 - m_0)) relationship against Chan-Vese convex relaxation. Still toy/partial — paper-level requires real data and full baseline comparison."
}
```

**reproductionTruthLevel 经 mapping**（reading-data.js 行 1380-1383）→ `partial-completed`（partial includes "partial"）✅

---

## 10. Warnings / Fidelity Limits

### 10.1 必须写的 warning（写进 `fidelityWarning` 字段）

> Partial reproduction uses Chambolle-Pock dual projection as the ROF solver (strict TV minimizer, not Gaussian proxy). Threshold update uses `mean_f(Ω_i)` on **raw f** per Eq. (15), not on smoothed `u`. Lemma 2 monotonicity and Lemma 3 sign changes are checked, but **Theorem 1 projected-T-ROF full convergence proof is not numerically reproduced**. **Paper-level still requires real cartoon/texture/medical data and paper Table 1-2 baseline comparison.** The Gaussian proxy T-ROF baseline is preserved as a `gaussian_proxy_trof_accuracy` metric for comparison.

### 10.2 不应声称的边界

| 不能声称 | 原因 |
|---|---|
| paper-level reproduction | 缺真实数据 + 缺论文 Table 1-2 baseline 完整对照 + Theorem 1 是数学证明不是数值实验 |
| 复现 Theorem 1 | Theorem 1 是数学证明；partial 只能验证 Lemma 2/3 的单调性数值上成立 |
| 复现 paper Fig. 2-3 | 缺 cartoon/texture/medical 数据 |
| 复现 paper Table 1-2 | 缺 baseline 列表和 ground truth segmentation |
| Gaussian proxy 等同于 ROF | Gaussian 没有 TV regularization，没有 data fidelity 项 |
| 阈值更新基于 smoothed image | 论文 Eq. 15 是基于 raw f，partial 升级后改回 f |

### 10.3 effectScore 调整理由

**当前 effectScore=5 / "很明显"**：
- toy 0.32 gain 配 5/5 偏激进
- 0.32 的 gain 是"synthetic 4-phase square 灰度差 0.1 上的 Gaussian proxy toy gain"，不是论文级效果

**建议降到 effectScore=3 / "明显"**：
- partial 升级后 `rof_trof_accuracy` 可能比 Gaussian proxy 略低（ROF 真实求解在 noise σ=0.05 上可能不如 Gaussian 平滑）
- 但 partial 升级**理论严格性更高**，可解释性更强
- "明显" 比 "很明显" 更克制

### 10.4 effectLabel 标尺问题（项目级 issue）

观察 `docs/js/reading-data.js`：
- effectScore=3 → "明显"
- effectScore=4 → "很明显"
- effectScore=5 → "很明显"（重复，无 5 档 label）

**应该**：
- 3 → "明显"
- 4 → "很明显"
- 5 → "极明显" 或 "显著" 之类

**这是项目级 issue**，不阻塞 iterated-rof partial 升级。**建议在后续报告里标 should-fix（项目级）**。

---

## 11. Final Verdict

### 当前状态

- **实际复现等级**：`toy-to-partial`
- **dashboard 声明**：`reproductionTruthLevel: "partial-completed"`
- **诚实性**：✅ resultStatus / runMetrics / resultFiles / notes / fidelityWarning 全部与 run 一致
- **算法有效性**：❌ 核心 Step 1（求解 ROF）用 Gaussian proxy 绕过；m_i 用 smoothed image 违反 Eq. 15
- **同步一致性**：✅ `node reproduce/sync_to_dashboard.mjs --check` 通过

### 升级后状态（按 §5 partial 方案实现）

- **目标复现等级**：`partial`
- **dashboard 声明**：`reproductionTruthLevel: "partial-completed"`
- **可运行性**：✅ numpy + scipy 足够，不需新增依赖
- **可审计性**：✅ 单测覆盖 Proposition 2 / Lemma 2 / Lemma 3 / ROF solver
- **升级验证**：`node reproduce/sync_to_dashboard.mjs --check` + `node docs/scripts/validate.mjs`
- **不再过度声称**：✅ fidelityWarning 明确 "still toy/partial"，不写 paper-level
- **效果评分**：从 effectScore=5 降到 effectScore=3

### 不会达到的等级

- `paper-like`：需要 Brodatz texture + 1 个 medical 数据集 + 严肃 baseline（Otsu 多阈值 / Chan-Vese convex relaxation）
- `paper-level`：需要论文 Fig. 2-3 完整复现 + paper Table 1-2 baseline 完整对照 + 完整 medical 数据集 — **不建议冲**

### 总结

**iterated-rof 当前 toy-to-partial 实现是诚实的**（dashboard 与 run 一致），但**算法上没真正实现 ROF 求解**（用 Gaussian proxy 替代），距离"partial" 还有 1 档升级空间。

**推荐立刻执行 §5 partial 升级**：
1. 写 `rof_chambolle_pock` + `rof_split_bregman`（partial ROF solver）
2. 重写 T-ROF 内部循环：用 `mean_f(Ω_i)` on raw f，**用 ROF 解**作 u，加单调性 + sign changes 指标
3. 加 K=2 case 验证 Proposition 2
4. 出 2 个新图（convergence + chan-vese 对照）
5. 改 `reproDetails[3]`：reproductionLevel → "partial"，新增 5+ 个 metrics，新增 2 个 resultFiles，effectScore 5 → 3
6. 跑 `python reproduce/run_all.py` + `node reproduce/sync_to_dashboard.mjs --check` + `node docs/scripts/validate.mjs`

预计工作量：~150-200 行 Python（含测试）+ 1 个文档更新。可由 single-shot 复现工程师 worker 一次性交付。

---

## 附：未在本篇处理但关联的项

1. **effectLabel 标尺缺 5 档**：项目级 issue，影响 2 个 effectScore=5 论文（#3 iterated-rof + 另一个，reading-data.js 行 1311）。建议在 #3 partial 升级时一并改：3="明显" / 4="很明显" / 5="极明显"。
2. **notes 字段重复**：`reproDetails[3].notes` 和 `fidelityWarning` 都在说 "proxy smoothing / strict T-ROF should solve ROF once"，partial 升级后两段都需要重写。建议 partial 升级后 `notes` 写"做了什么"，`fidelityWarning` 写"没做什么 / 什么不能声称"。
3. **`sat_rof_trof.py` 一个 runner 输出 3 篇论文**（#1 sat-overview / #2 pcms-rof-linkage / #3 iterated-rof）：当前 partial 升级只动 iterated-rof，sat-overview 和 pcms-rof-linkage 的结果会保持原样。**如果未来要 partial 升级 sat-overview 或 pcms-rof-linkage**，可以复用 ROF solver 函数（共享 `rof_chambolle_pock`）。
4. **docs/assets/repro/repro_results.json 历史快照**：mtime 已与 reproduce/results/ 一致（Jun 5 15:01），不再是"旧快照"，是"当前快照"。我之前误判，已纠正。

---

## 实施后 Addendum（2026-06-06）

R1 partial 升级已落地：

- `reproduce/experiments/sat_rof_trof.py` 新增 Chambolle-Pock ROF、Split-Bregman 对照、raw `mean_f(Omega_i)` T-ROF threshold update、Lemma 2/3 指标与 K=2 Proposition 2 proxy。
- `reproduce/tests/test_iterated_rof.py` 新增 4 个 unittest，覆盖常量图 ROF 稳定性、raw mean 更新、Lemma 指标和 K=2 lambda 公式。
- `docs/js/reading-data.js` 中 #3 `iterated-rof` 已从 `toy-to-partial` 升到 `partial`，effectScore 从 5 降到 3，result files 增加 `iterated_rof_convergence.png` 与 `iterated_rof_chanvese.png`。
- 最新 `run_all.py` 结果：`raw_kmeans_accuracy=0.5650`，`gaussian_proxy_trof_accuracy=0.9438`，`rof_trof_accuracy=0.9463`，`split_bregman_trof_accuracy=0.9510`，`assumption_a_violations=0`，`k2_rof_threshold_dice=0.9976`。
- 仍然不声称 paper-level：真实 cartoon/texture/medical 数据和 Table 1-2 baseline 对照未复现。
