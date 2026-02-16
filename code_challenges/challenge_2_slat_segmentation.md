# Challenge 2: SLaT三行实现 ⭐⭐⭐⭐

## 题目描述

用**恰好三行代码**实现SLaT(Spectral-Linear-Threshold)图像分割算法。

SLaT算法流程：
1. 构建图像的相似度矩阵
2. 计算Laplacian矩阵的特征向量
3. 对特征向量进行k-means聚类
4. 阈值化得到分割结果

## 输入输出格式

**输入**:
```python
image: np.ndarray  # 输入图像，shape=(H,W)或(H,W,3)
k: int            # 分割类别数，默认2
```

**输出**:
```python
labels: np.ndarray  # 分割标签，shape=(H,W)，值域{0,1,...,k-1}
```

## 示例

```python
import numpy as np

image = np.array([[0, 0, 255, 255],
                  [0, 0, 255, 255],
                  [255, 255, 0, 0],
                  [255, 255, 0, 0]], dtype=np.uint8)
labels = slat_segment(image, k=2)
# 期望: 左上和右下区域为同一类
```

## 数据范围

- 图像尺寸: 4×4 到 256×256
- 类别数k: 2 到 10
- 像素值: 0-255 (uint8)

## 限制条件

| 限制项 | 要求 |
|--------|------|
| 代码行数 | 恰好3行 |
| 每行长度 | ≤80字符 |
| Python版本 | 3.8+ |
| 允许使用 | numpy, scipy.sparse.linalg |

## 评分标准

| 维度 | 分数 | 说明 |
|------|------|------|
| 代码行数 | 30分 | 必须3行，否则0分 |
| 正确性 | 40分 | IoU>0.7得满分 |
| 代码简洁 | 30分 | 每行≤60字符加分 |

## 参考答案

### 标准版 (3行, 每行≤80字符)
```python
def slat_segment(I, k=2):
    W = np.exp(-np.square(np.linalg.norm(I[:,:,None]-I[:,None,:],axis=-1))/1000)
    L = np.diag(W.sum(1)) - W; _, V = scipy.sparse.linalg.eigsh(L, k, which='SM')
    return KMeans(k).fit(V).labels_.reshape(I.shape[:2])
```

### 紧凑版 (3行, 每行≤60字符)
```python
def slat_segment(I,k=2):
    d=np.sum((I-I.mean())**2);W=np.exp(-np.abs(I[:,:,None]-I[:,None,:])**2/d)
    return (scipy.cluster.vq.kmeans2(W.reshape(-1,1),k)[1]).reshape(I.shape)
```

### 近似版 (3行, 超轻量)
```python
def slat_segment(I,k=2):
    f=I.reshape(-1,1)if I.ndim==2 else I.reshape(-1,3)
    return KMeans(k,n_init=1).fit(f).labels_.reshape(I.shape[:2])
```

## 优化提示

1. **行合并技巧**:
   - 用分号`;`分隔多条语句
   - 利用Python的链式调用
   - 省略不必要的中间变量赋值

2. **矩阵构建**:
   - 使用广播避免显式循环
   - 考虑稀疏矩阵节省内存
   - 近似计算相似度

3. **特征分解**:
   - 只计算前k个最小特征值
   - 使用scipy.sparse.linalg加速
   - 考虑随机投影近似

4. **代码压缩**:
```python
# 80字符限制下的换行技巧
# 不推荐
very_long_line = something + something_else + more_stuff
# 推荐 (但仍算一行)
very_long_line=something+something_else+more_stuff
```

## 测试用例

### 测试1: 代码行数
```python
def test_line_count():
    import inspect
    source = inspect.getsource(slat_segment)
    lines = [l for l in source.split('\n') 
             if l.strip() and not l.strip().startswith(('import','def','@'))]
    assert len(lines) == 3, f"需要3行代码，实际{len(lines)}行"
```

### 测试2: 每行长度
```python
def test_line_length():
    import inspect
    source = inspect.getsource(slat_segment)
    lines = [l for l in source.split('\n') 
             if l.strip() and not l.strip().startswith(('import','def','@'))]
    for i, line in enumerate(lines):
        assert len(line) <= 80, f"第{i+1}行长度{len(line)}超过80"
```

### 测试3: 分割正确性
```python
def test_segmentation():
    # 创建简单测试图
    image = np.zeros((8, 8))
    image[:4, :4] = 1
    image[4:, 4:] = 1
    labels = slat_segment(image, k=2)
    # 检查对角区域标签相同
    assert labels[0, 0] == labels[7, 7]
    assert labels[0, 7] == labels[7, 0]
    assert labels[0, 0] != labels[0, 7]
```

### 测试4: 边界情况
```python
def test_edge_cases():
    # 单一颜色图像
    image = np.ones((16, 16)) * 128
    labels = slat_segment(image, k=2)
    assert labels.shape == (16, 16)
```

## 进阶挑战

- [ ] 实现彩色图像版本
- [ ] 添加超像素预处理
- [ ] 实现自适应k值选择
