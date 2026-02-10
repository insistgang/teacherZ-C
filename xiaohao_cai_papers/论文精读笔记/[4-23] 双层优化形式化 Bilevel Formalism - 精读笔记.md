# [4-23] 双层优化形式化 Bilevel Formalism - 精读笔记

> **论文标题**: Bilevel Optimization: Theory and Applications
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (高)
> **重要性**: ⭐⭐⭐ (优化理论基础)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Bilevel Optimization: Theory and Applications |
| **作者** | Xiaohao Cai 等人 |
| **核心领域** | 双层优化、数学规划 |
| **关键词** | Bilevel Optimization, Stackelberg Game, Nested Optimization |
| **核心价值** | 双层优化的形式化理论与应用框架 |

---

## 🎯 双层优化核心问题

### 问题定义

```
双层优化问题定义:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

上层问题 (Leader):
  min_{x} F(x, y*(x))
  s.t. G(x, y*(x)) ≤ 0

其中 y*(x) 是下层问题的解:

下层问题 (Follower):
  y*(x) = argmin_{y} f(x, y)
          s.t. g(x, y) ≤ 0

特点:
  - 嵌套结构: 上层决策影响下层,下层反馈影响上层
  - 层次依赖: y*(x) 是x的隐函数
  - 非凸性: 即使上下层都凸,整体也可能非凸
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 与单层优化的对比

| 特性 | 单层优化 | 双层优化 |
|:---|:---|:---|
| **结构** | min f(x) | min F(x, y*(x)) |
| **变量** | 单层变量 x | 上层x + 下层y |
| **约束** | 显式约束 | 隐式约束 (下层最优性) |
| **求解** | 梯度下降等 | 需要特殊处理嵌套结构 |
| **应用** | 标准ML问题 | NAS、元学习、博弈论 |

---

## 🔬 双层优化方法论

### 数学形式化

```
标准形式:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

(P)  min_{x∈X} F(x, y)
     s.t. y ∈ S(x) = argmin_{y∈Y} {f(x, y) : g(x,y) ≤ 0}
          G(x, y) ≤ 0

其中:
  - x ∈ R^n: 上层决策变量
  - y ∈ R^m: 下层决策变量
  - F: R^n × R^m → R: 上层目标函数
  - f: R^n × R^m → R: 下层目标函数
  - G, g: 约束函数

解集映射 S(x):
  对每个固定的x, S(x)给出下层的最优解集
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 求解方法分类

```
双层优化求解方法:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 基于KKT条件的转化法
   - 将下层KKT条件作为上层约束
   - 转化为单层约束优化
   - 适用: 下层凸且满足约束规范

2. 隐函数梯度法
   - 利用隐函数定理计算 dy*/dx
   - 上层梯度: dF/dx = ∂F/∂x + ∂F/∂y · dy*/dx
   - 适用: 下层有唯一解且光滑

3. 迭代优化法
   - 交替更新上层和下层
   - 梯度下降-上升或固定点迭代
   - 适用: 大规模问题

4. 启发式方法
   - 进化算法
   - 代理模型优化
   - 适用: 非凸、不可微问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 核心组件1: KKT条件转化

```python
"""
双层优化 → 单层约束优化 (MPEC)

下层问题:
  min_y f(x, y)
  s.t. g(x, y) ≤ 0

KKT条件:
  ∇_y f(x, y) + λ^T ∇_y g(x, y) = 0  (平稳性)
  g(x, y) ≤ 0                         (原始可行性)
  λ ≥ 0                               (对偶可行性)
  λ_i · g_i(x, y) = 0                 (互补松弛)

转化为单层:
  min_{x,y,λ} F(x, y)
  s.t. KKT条件
       G(x, y) ≤ 0
"""

import numpy as np
from scipy.optimize import minimize

class KKTTransformation:
    """
    KKT条件转化求解器
    """

    def __init__(self, upper_obj, lower_obj, upper_constr, lower_constr):
        """
        Args:
            upper_obj: 上层目标函数 F(x, y)
            lower_obj: 下层目标函数 f(x, y)
            upper_constr: 上层约束 G(x, y)
            lower_constr: 下层约束 g(x, y)
        """
        self.F = upper_obj
        self.f = lower_obj
        self.G = upper_constr
        self.g = lower_constr

    def kkt_constraints(self, vars):
        """
        构建KKT约束

        Args:
            vars = [x, y, lambda]

        Returns:
            constraints: 等式和不等式约束列表
        """
        n_x = self.n_x
        n_y = self.n_y

        x = vars[:n_x]
        y = vars[n_x:n_x+n_y]
        lam = vars[n_x+n_y:]

        constraints = []

        # 1. 平稳性: ∇_y f + λ^T ∇_y g = 0
        grad_f_y = self.grad_lower_y(x, y)
        grad_g_y = self.grad_constr_y(x, y)
        stationarity = grad_f_y + lam @ grad_g_y

        for i, s in enumerate(stationarity):
            constraints.append({'type': 'eq', 'fun': lambda v, i=i: stationarity[i]})

        # 2. 原始可行性: g(x, y) ≤ 0
        g_val = self.g(x, y)
        for i, g_i in enumerate(g_val):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: -g_val[i]})

        # 3. 对偶可行性: λ ≥ 0
        for i, l in enumerate(lam):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: lam[i]})

        # 4. 互补松弛: λ_i · g_i = 0
        for i in range(len(lam)):
            compl = lam[i] * g_val[i]
            constraints.append({'type': 'eq', 'fun': lambda v, i=i: compl})

        return constraints

    def solve(self, x0, y0, lam0):
        """
        求解转化后的单层问题
        """
        z0 = np.concatenate([x0, y0, lam0])

        def objective(z):
            n_x = self.n_x
            n_y = self.n_y
            x = z[:n_x]
            y = z[n_x:n_x+n_y]
            return self.F(x, y)

        constraints = self.kkt_constraints(z0)

        result = minimize(objective, z0, method='SLSQP',
                         constraints=constraints)

        n_x = self.n_x
        n_y = self.n_y
        x_opt = result.x[:n_x]
        y_opt = result.x[n_x:n_x+n_y]

        return x_opt, y_opt, result.fun
```

---

### 核心组件2: 隐函数梯度法

```python
import torch
import torch.nn as nn

class ImplicitGradientBilevel:
    """
    隐函数梯度法求解双层优化

    适用于深度学习场景 (如元学习、NAS)
    """

    def __init__(self, upper_loss, lower_loss, lower_optimizer):
        """
        Args:
            upper_loss: 上层损失函数 L_val(θ, φ)
            lower_loss: 下层损失函数 L_train(θ, φ)
            lower_optimizer: 下层优化器
        """
        self.upper_loss = upper_loss
        self.lower_loss = lower_loss
        self.lower_opt = lower_optimizer

    def compute_hypergradient(self, theta, phi, train_data, val_data):
        """
        计算超梯度 dL_val/dtheta

        使用隐函数定理:
        dφ*/dθ = -(∇²_{φφ} L_train)^{-1} · ∇²_{θφ} L_train

        dL_val/dθ = ∇_θ L_val + ∇_φ L_val · dφ*/dθ
        """
        # 1. 求解下层问题 (得到最优φ*)
        phi_star = self.solve_lower(theta, phi, train_data)

        # 2. 计算上层梯度
        val_loss = self.upper_loss(theta, phi_star, val_data)
        grad_theta_val = torch.autograd.grad(val_loss, theta,
                                            create_graph=True)[0]
        grad_phi_val = torch.autograd.grad(val_loss, phi_star,
                                          create_graph=True)[0]

        # 3. 计算隐函数梯度 (使用共轭梯度法避免求逆)
        implicit_grad = self.implicit_gradient(theta, phi_star,
                                              train_data, grad_phi_val)

        # 4. 总梯度
        hypergradient = grad_theta_val + implicit_grad

        return hypergradient

    def solve_lower(self, theta, phi, train_data, num_steps=100):
        """求解下层优化问题"""
        phi_current = phi.clone().requires_grad_(True)

        for _ in range(num_steps):
            loss = self.lower_loss(theta, phi_current, train_data)
            grad = torch.autograd.grad(loss, phi_current)[0]

            with torch.no_grad():
                phi_current = phi_current - 0.01 * grad

        return phi_current

    def implicit_gradient(self, theta, phi, train_data, grad_phi_val):
        """
        计算隐函数梯度项

        求解: (∇²_{φφ} L_train) · v = grad_phi_val
        使用共轭梯度法
        """
        def hessian_vector_product(v):
            """计算Hessian-向量积"""
            loss = self.lower_loss(theta, phi, train_data)
            grad_phi = torch.autograd.grad(loss, phi,
                                          create_graph=True)[0]
            hvp = torch.autograd.grad(grad_phi, phi, v,
                                     retain_graph=True)[0]
            return hvp + 0.01 * v  # 添加正则化

        # 共轭梯度法求解线性系统
        v = self.conjugate_gradient(hessian_vector_product, grad_phi_val)

        # 计算 ∇²_{θφ} L_train · v
        loss = self.lower_loss(theta, phi, train_data)
        grad_theta = torch.autograd.grad(loss, theta,
                                        create_graph=True,
                                        allow_unused=True)[0]

        if grad_theta is None:
            return torch.zeros_like(theta)

        grad_grad = torch.autograd.grad(grad_theta, phi, v,
                                       retain_graph=True)[0]

        return -grad_grad

    def conjugate_gradient(self, A_func, b, max_iter=10, tol=1e-6):
        """
        共轭梯度法求解 Ax = b
        """
        x = torch.zeros_like(b)
        r = b - A_func(x)
        p = r.clone()
        rs_old = torch.sum(r * r)

        for _ in range(max_iter):
            Ap = A_func(p)
            alpha = rs_old / (torch.sum(p * Ap) + 1e-10)

            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = torch.sum(r * r)
            if torch.sqrt(rs_new) < tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x
```

---

### 核心组件3: 迭代优化法

```python
class IterativeBilevelOptimization:
    """
    迭代式双层优化

    交替更新上层和下层变量
    """

    def __init__(self, upper_obj, lower_obj, upper_lr=0.01, lower_lr=0.01):
        self.upper_obj = upper_obj
        self.lower_obj = lower_obj
        self.upper_lr = upper_lr
        self.lower_lr = lower_lr

    def solve(self, x_init, y_init, num_iterations=1000):
        """
        迭代求解

        算法:
        for t = 1, 2, ...:
          # 下层更新 (多步)
          for k = 1, ..., K:
            y_{k} = y_{k-1} - α_l · ∇_y f(x_t, y_{k-1})

          # 上层更新
          x_{t+1} = x_t - α_u · ∇_x F(x_t, y_K)
        """
        x = x_init.clone()
        y = y_init.clone()

        history = {'x': [], 'y': [], 'F': [], 'f': []}

        for t in range(num_iterations):
            # 下层优化 (内循环)
            y_current = y.clone()
            for k in range(10):  # K步下层更新
                grad_y = self.grad_lower_y(x, y_current)
                y_current = y_current - self.lower_lr * grad_y

            y = y_current

            # 上层优化
            grad_x = self.grad_upper_x(x, y)
            x = x - self.upper_lr * grad_x

            # 记录
            history['x'].append(x.clone())
            history['y'].append(y.clone())
            history['F'].append(self.upper_obj(x, y).item())
            history['f'].append(self.lower_obj(x, y).item())

            if t % 100 == 0:
                print(f"Iter {t}: F={history['F'][-1]:.4f}, f={history['f'][-1]:.4f}")

        return x, y, history

    def grad_upper_x(self, x, y):
        """上层关于x的梯度"""
        x_var = x.clone().requires_grad_(True)
        F_val = self.upper_obj(x_var, y)
        return torch.autograd.grad(F_val, x_var)[0]

    def grad_lower_y(self, x, y):
        """下层关于y的梯度"""
        y_var = y.clone().requires_grad_(True)
        f_val = self.lower_obj(x, y_var)
        return torch.autograd.grad(f_val, y_var)[0]
```

---

## 📊 应用案例

### 案例1: 超参数优化

```python
class HyperparameterOptimization:
    """
    双层优化用于超参数优化

    上层: 选择超参数 λ
    下层: 训练模型权重 w
    """

    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def upper_objective(self, lambda_reg, w_star):
        """
        上层目标: 验证集性能

        Args:
            lambda_reg: 正则化超参数
            w_star: 下层优化得到的最优权重
        """
        val_loss = 0
        for x, y in self.val_loader:
            pred = self.model(x, w_star)
            val_loss += F.cross_entropy(pred, y)

        return val_loss / len(self.val_loader)

    def lower_objective(self, lambda_reg, w):
        """
        下层目标: 训练集损失 + 正则化
        """
        train_loss = 0
        reg_loss = 0

        for x, y in self.train_loader:
            pred = self.model(x, w)
            train_loss += F.cross_entropy(pred, y)
            reg_loss += lambda_reg * torch.sum(w ** 2)

        return train_loss / len(self.train_loader) + reg_loss

    def optimize(self, lambda_init, w_init, num_outer=100):
        """双层优化"""
        lambda_reg = lambda_init
        w = w_init

        bilevel_opt = IterativeBilevelOptimization(
            upper_obj=lambda l, w: self.upper_objective(l, w),
            lower_obj=lambda l, w: self.lower_objective(l, w),
            upper_lr=0.001,
            lower_lr=0.01
        )

        lambda_opt, w_opt, _ = bilevel_opt.solve(lambda_reg, w, num_outer)

        return lambda_opt, w_opt
```

### 案例2: NAS中的双层优化

```python
class NASBilevelOptimization:
    """
    DARTS中的双层优化

    上层: 架构参数 α
    下层: 网络权重 w
    """

    def __init__(self, model):
        self.model = model

    def train_step(self, train_data, val_data, alpha, w,
                   alpha_lr=0.001, w_lr=0.01):
        """
        一步双层优化

        1. 下层: 在训练集上更新w
        2. 上层: 在验证集上更新α
        """
        # 下层更新 (近似)
        train_loss = self.model.loss(train_data, alpha, w)
        grad_w = torch.autograd.grad(train_loss, w)[0]
        w_prime = w - w_lr * grad_w

        # 上层更新 (使用w'近似w*)
        val_loss = self.model.loss(val_data, alpha, w_prime)
        grad_alpha = torch.autograd.grad(val_loss, alpha)[0]
        alpha = alpha - alpha_lr * grad_alpha

        # 实际更新w
        w = w - w_lr * torch.autograd.grad(train_loss, w)[0]

        return alpha, w
```

---

## 💡 可复用代码组件

### 通用双层优化求解器

```python
class BilevelOptimizer:
    """
    通用双层优化求解器

    支持多种求解策略
    """

    def __init__(self, method='implicit', **kwargs):
        """
        Args:
            method: 'kkt', 'implicit', 'iterative'
        """
        self.method = method
        self.kwargs = kwargs

    def solve(self, upper_obj, lower_obj, x0, y0):
        """
        求解双层优化问题

        Args:
            upper_obj: 上层目标函数
            lower_obj: 下层目标函数
            x0, y0: 初始值

        Returns:
            x_opt, y_opt: 最优解
        """
        if self.method == 'kkt':
            solver = KKTTransformation(upper_obj, lower_obj, None, None)
            return solver.solve(x0, y0, np.zeros(len(y0)))

        elif self.method == 'implicit':
            solver = ImplicitGradientBilevel(upper_obj, lower_obj, None)
            # ...

        elif self.method == 'iterative':
            solver = IterativeBilevelOptimization(
                upper_obj, lower_obj,
                self.kwargs.get('upper_lr', 0.01),
                self.kwargs.get('lower_lr', 0.01)
            )
            return solver.solve(x0, y0, self.kwargs.get('max_iter', 1000))

        else:
            raise ValueError(f"Unknown method: {self.method}")
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **Stackelberg博弈** | Stackelberg Game | 领导者-跟随者博弈模型 |
| **KKT条件** | Karush-Kuhn-Tucker | 约束优化的最优性条件 |
| **隐函数定理** | Implicit Function Theorem | 隐函数微分理论 |
| **超梯度** | Hypergradient | 上层关于上层变量的梯度 |
| **MPEC** | Mathematical Program with Equilibrium Constraints | 带均衡约束的数学规划 |

---

## ✅ 复习检查清单

- [ ] 理解双层优化的问题结构
- [ ] 掌握KKT条件转化方法
- [ ] 了解隐函数梯度计算
- [ ] 理解迭代优化法的原理
- [ ] 能够应用到NAS和元学习

---

## 🤔 思考问题

1. **双层优化为什么比单层更难求解？**
   - 提示: 嵌套结构、非凸性

2. **KKT转化的优缺点是什么？**
   - 提示: 互补约束的非光滑性

3. **隐函数梯度法的计算瓶颈在哪里？**
   - 提示: Hessian求逆

4. **如何选择合适的求解方法？**
   - 提示: 问题规模、光滑性、精度要求

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
