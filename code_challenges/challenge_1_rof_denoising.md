# Challenge 1: 最短ROF去噪 ⭐⭐⭐

## 题目描述

用最少的Python代码实现ROF(Rudin-Osher-Fatemi)图像去噪模型。

ROF模型通过最小化以下能量函数实现去噪：
```
min_u ∫|∇u| dx + λ∫(u-f)² dx
```

简化版本使用梯度下降迭代求解。

## 输入输出格式

**输入**:
```python
f: np.ndarray  # 噪声图像，shape=(H,W)或(H,W,C)，值域[0,1]
lambda_: float # 正则化参数，典型值0.1-1.0
```

**输出**:
```python
u: np.ndarray  # 去噪后图像，shape与输入相同
```

## 示例

```python
import numpy as np

f = np.array([[0.1, 0.9], [0.8, 0.2]])
lambda_ = 0.5
u = rof_denoise(f, lambda_)
# 期望输出近似: [[0.2, 0.8], [0.7, 0.3]]
```

## 数据范围

- 图像尺寸: 2×2 到 1024×1024
- λ值: 0.01 到 10.0
- 迭代次数: 自定，需保证收敛

## 限制条件

| 限制项 | 要求 |
|--------|------|
| 代码长度 | ≤100字符（不含import和def定义行） |
| Python版本 | 3.8+ |
| 禁止使用 | cv2.fastNlMeansDenoising, skimage.restoration等现成函数 |
| 允许使用 | numpy基础函数 |

## 评分标准

| 维度 | 分数 | 说明 |
|------|------|------|
| 代码长度 | 50分 | 100字符=25分，50字符=50分，线性插值 |
| 正确性 | 30分 | PSNR>25dB得满分 |
| 效率 | 20分 | 512×512图像<1秒得满分 |

## 参考答案

### 简化版 (85字符)
```python
import numpy as np
def r(f,l,n=99):
 u=f.copy()
 for _ in range(n):u-=.01*l*(u-f)
 return u
```

### 梯度下降版 (98字符)
```python
import numpy as np
def r(f,l,n=99):
 u=f.copy();g=lambda x:np.gradient(x)
 for _ in range(n):
  d=g(u);u-=.01*(l*(u-f)-np.sum([np.gradient(d[i])for i in range(2)],0))
 return u
```

## 优化提示

1. **字符压缩技巧**:
   - 用`l`代替`lambda_`
   - 省略空格：`u=f.copy()`而非`u = f.copy()`
   - 使用单字符变量名
   - 利用默认参数减少调用字符

2. **算法简化**:
   - 使用显式欧拉法代替隐式
   - 固定步长和迭代次数
   - 忽略边界条件处理

3. **性能优化**:
   - 使用numpy向量化
   - 避免中间变量
   - 考虑just-in-time编译

## 测试用例

### 测试1: 基本功能
```python
def test_basic():
    f = np.array([[0.1, 0.9], [0.8, 0.2]])
    u = rof_denoise(f, 0.5)
    assert u.shape == f.shape
    assert np.all(u >= 0) and np.all(u <= 1)
```

### 测试2: 平滑效果
```python
def test_smoothing():
    f = np.random.rand(64, 64)
    u = rof_denoise(f, 1.0)
    # 去噪后梯度应减小
    grad_f = np.sum(np.abs(np.diff(f)))
    grad_u = np.sum(np.abs(np.diff(u)))
    assert grad_u < grad_f
```

### 测试3: 代码长度
```python
def test_code_length():
    import inspect
    source = inspect.getsource(rof_denoise)
    # 排除import和def行
    code_lines = [l for l in source.split('\n') 
                  if l.strip() and not l.strip().startswith(('import','def'))]
    code = ''.join(code_lines)
    assert len(code) <= 100, f"代码长度{len(code)}超过100字符"
```

### 测试4: 性能测试
```python
def test_performance():
    import time
    f = np.random.rand(512, 512)
    start = time.time()
    u = rof_denoise(f, 0.5)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"耗时{elapsed:.2f}s超过1秒"
```
