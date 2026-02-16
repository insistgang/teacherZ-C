# 硕士学位论文图表规范

## 一、图片格式要求

### 1.1 基本要求

| 项目 | 要求 |
|------|------|
| 分辨率 | 不低于300dpi（打印），150dpi（屏幕） |
| 格式 | 矢量图：PDF、EPS、SVG；位图：TIFF、PNG、JPG |
| 色彩 | RGB模式（电子版），CMYK模式（印刷版） |
| 宽度 | 单栏：8cm；双栏：17cm |

### 1.2 图片类型与格式选择

| 图片类型 | 推荐格式 | 说明 |
|----------|----------|------|
| 曲线图、流程图 | PDF/EPS/SVG | 矢量格式，缩放不失真 |
| 灰度图像 | PNG/TIFF | 无损压缩 |
| 彩色图像 | PNG/JPG | 照片类用JPG，截图用PNG |
| 3D渲染图 | PNG | 支持透明背景 |

### 1.3 图片制作规范

#### 曲线图（使用Matplotlib/MATLAB）

```python
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图片大小和分辨率
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 绘制曲线
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax.plot(x, y1, 'b-', linewidth=2, label='方法A')
ax.plot(x, y2, 'r--', linewidth=2, label='方法B')

# 设置标签
ax.set_xlabel('迭代次数', fontsize=12)
ax.set_ylabel('Dice系数 (%)', fontsize=12)
ax.set_title('收敛曲线对比', fontsize=14)

# 设置图例
ax.legend(loc='lower right', fontsize=10)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.7)

# 保存图片
plt.savefig('convergence.pdf', bbox_inches='tight')
plt.savefig('convergence.png', dpi=300, bbox_inches='tight')
```

#### 分割结果对比图

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_segmentation(image, ground_truth, prediction, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)
    
    # 原图
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像', fontsize=12)
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(image, cmap='gray')
    axes[1].contour(ground_truth, colors='red', linewidths=1.5)
    axes[1].set_title('真实标注', fontsize=12)
    axes[1].axis('off')
    
    # 预测结果
    axes[2].imshow(image, cmap='gray')
    axes[2].contour(prediction, colors='blue', linewidths=1.5)
    axes[2].set_title('预测结果', fontsize=12)
    axes[2].axis('off')
    
    # 叠加对比
    axes[3].imshow(image, cmap='gray')
    axes[3].contour(ground_truth, colors='red', linewidths=1.5, linestyles='solid')
    axes[3].contour(prediction, colors='blue', linewidths=1.5, linestyles='dashed')
    axes[3].set_title('对比叠加', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
```

### 1.4 图片命名规范

```
推荐命名格式：章节_类型_序号_描述.扩展名

示例：
- ch3_fig_01_flowchart.pdf       # 第三章流程图
- ch3_fig_02_convergence.png     # 第三章收敛曲线
- ch4_fig_01_comparison.pdf      # 第四章对比图
- ch5_tab_01_results.tex         # 第五章表格
```

---

## 二、表格样式规范

### 2.1 三线表格式

**标准三线表**：顶线、栏目线、底线

```latex
\begin{table}[htbp]
    \centering
    \caption{不同方法在BraTS数据集上的分割结果比较}
    \label{tab:comparison}
    \begin{tabular}{lcccc}
        \toprule
        方法 & DSC(\%) & JI(\%) & HD(mm) & ASD(mm) \\
        \midrule
        C-V模型 & 78.32 & 64.51 & 12.45 & 2.31 \\
        DRLSE & 82.17 & 69.73 & 9.82 & 1.87 \\
        U-Net & 85.43 & 74.56 & 7.21 & 1.42 \\
        \textbf{本文方法} & \textbf{87.52} & \textbf{77.89} & \textbf{6.12} & \textbf{1.08} \\
        \bottomrule
    \end{tabular}
\end{table}
```

### 2.2 表格内容对齐

| 数据类型 | 对齐方式 | 示例 |
|----------|----------|------|
| 文字内容 | 左对齐 | `l` |
| 数字 | 右对齐 | `r` |
| 居中内容 | 居中 | `c` |
| 小数点对齐 | 特殊处理 | `S`（siunitx宏包） |

**小数点对齐示例**：
```latex
\usepackage{siunitx}

\begin{tabular}{lSSSS}
    \toprule
    方法 & {DSC(\%)} & {JI(\%)} & {HD(mm)} & {ASD(mm)} \\
    \midrule
    方法A & 78.32 & 64.51 & 12.45 & 2.31 \\
    方法B & 82.17 & 69.73 & 9.82 & 1.87 \\
    \bottomrule
\end{tabular}
```

### 2.3 复杂表格示例

**多行表头**：
```latex
\usepackage{multirow}

\begin{table}[htbp]
    \centering
    \caption{消融实验结果}
    \begin{tabular}{lccc}
        \toprule
        \multirow{2}{*}{配置} & \multicolumn{3}{c}{评价指标} \\
        \cmidrule{2-4}
        & DSC(\%) & HD(mm) & ASD(mm) \\
        \midrule
        基准模型 & 82.15 & 9.54 & 1.95 \\
        + 稀疏约束 & 84.37 & 8.12 & 1.62 \\
        + 紧框架变换 & 85.89 & 7.23 & 1.35 \\
        + 自适应权重 & \textbf{87.52} & \textbf{6.12} & \textbf{1.08} \\
        \bottomrule
    \end{tabular}
\end{table}
```

### 2.4 长表格处理

```latex
\usepackage{longtable}

\begin{longtable}{lcccc}
    \caption{各数据集详细实验结果} \label{tab:detail} \\
    \toprule
    数据集 & 方法 & DSC(\%) & HD(mm) & ASD(mm) \\
    \midrule
    \endfirsthead
    
    \multicolumn{5}{c}{续表 \thetable} \\
    \toprule
    数据集 & 方法 & DSC(\%) & HD(mm) & ASD(mm) \\
    \midrule
    \endhead
    
    \bottomrule
    \endfoot
    
    BraTS & 方法A & 85.43 & 7.21 & 1.42 \\
    BraTS & 方法B & 87.52 & 6.12 & 1.08 \\
    LiTS & 方法A & 83.21 & 8.54 & 1.67 \\
    LiTS & 方法B & 85.89 & 7.32 & 1.35 \\
    ... & ... & ... & ... & ... \\
\end{longtable}
```

---

## 三、编号规则

### 3.1 图表编号

**按章节编号**：
- 图3-1、图3-2、...
- 表3-1、表3-2、...
- 公式(3-1)、(3-2)、...

**跨章节引用**：
- "如图3-2所示"
- "参见表4-1"
- "由式(5-3)可得"

### 3.2 LaTeX自动编号

```latex
% 图片
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/ch3_flowchart.pdf}
    \caption{算法流程图}
    \label{fig:flowchart}
\end{figure}

% 引用
如图\ref{fig:flowchart}所示...

% 表格
\begin{table}[htbp]
    \centering
    \caption{实验参数设置}
    \label{tab:params}
    ...
\end{table}

% 引用
参见表\ref{tab:params}...

% 公式
\begin{equation}
    E(\phi) = \int_\Omega F(\phi) dx
    \label{eq:energy}
\end{equation}

% 引用
由式(\ref{eq:energy})可得...
```

### 3.3 编号注意事项

1. 编号应连续，不能跳号
2. 删除图表后重新编号
3. 引用编号与实际编号一致
4. 公式只在引用时才编号

---

## 四、图表内容要求

### 4.1 图片内容要求

#### 流程图
- 起止框：圆角矩形
- 处理框：矩形
- 判断框：菱形
- 流线：带箭头的直线
- 保持箭头方向一致（通常向下或向右）

#### 曲线图
- 坐标轴标签清晰
- 刻度设置合理
- 图例位置适当
- 线型/颜色区分明显
- 避免线条过多（≤5条）

#### 对比图
- 相同条件下的结果对比
- 标注清晰可辨
- 放大倍数一致
- 排列整齐美观

### 4.2 表格内容要求

- 表题简明扼要
- 列名清晰准确
- 单位标注明确
- 数据精度一致
- 重要数据突出显示

### 4.3 图表配合使用

- 定量数据用表格
- 定性比较用图片
- 趋势变化用曲线
- 空间分布用热图
- 结构关系用框图

---

## 五、常见问题

### Q1：图片模糊怎么办？

**A**：
1. 使用矢量格式（PDF/EPS）
2. 确保导出分辨率≥300dpi
3. 避免多次格式转换
4. 打印前检查预览效果

### Q2：表格太宽怎么办？

**A**：
1. 调整列宽或字体大小
2. 使用缩放：`\resizebox`
3. 旋转表格：`\rotating`宏包
4. 拆分表格

```latex
% 缩放示例
\begin{table}[htbp]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{...}
        ...
    \end{tabular}%
    }
\end{table}
```

### Q3：如何处理彩色图片？

**A**：
1. 确保彩色有意义（非装饰性）
2. 考虑黑白打印效果
3. 使用不同线型区分（实线、虚线等）
4. 避免使用相近颜色

### Q4：图表引用顺序有要求吗？

**A**：
1. 先见文，后见图表
2. 图表就近放置
3. 避免超前引用
4. 每个图表至少被引用一次

---

## 六、Word图表规范

### 6.1 插入图片

1. 选择"插入"→"图片"
2. 右键→"插入题注"
3. 设置标签和编号
4. 调整图片大小和位置

### 6.2 插入表格

1. 选择"插入"→"表格"
2. 设置行列数
3. 应用表格样式（三线表）
4. 添加题注

### 6.3 交叉引用

1. 插入题注后，选择"引用"→"交叉引用"
2. 选择引用类型和内容
3. 更新引用：Ctrl+A → F9

### 6.4 图表目录

1. 将光标放在目录位置
2. 选择"引用"→"插入表目录"
3. 选择图表标签
4. 更新目录：右键→"更新域"
