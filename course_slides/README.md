# 计算机视觉前沿方法

## 课程信息

- **课程名称**: 计算机视觉前沿方法
- **总学时**: 48学时 (24讲 × 2学时)
- **授课对象**: 研究生
- **理论基础**: Xiaohao Cai的68篇论文

---

## 课程目录

### 第一部分: 数学基础 (4讲)

| 讲次 | 主题 | 文件 |
|------|------|------|
| 第1讲 | 变分法基础 | [lecture_01.md](lecture_01.md) |
| 第2讲 | 凸优化理论 | [lecture_02.md](lecture_02.md) |
| 第3讲 | 全变分与图像处理 | [lecture_03.md](lecture_03.md) |
| 第4讲 | 小波与框架变换 | [lecture_04.md](lecture_04.md) |

### 第二部分: 图像分割方法 (6讲)

| 讲次 | 主题 | 文件 |
|------|------|------|
| 第5讲 | 经典分割方法 | [lecture_05.md](lecture_05.md) |
| 第6讲 | Mumford-Shah模型 | [lecture_06.md](lecture_06.md) |
| 第7讲 | 两阶段分割方法 | [lecture_07.md](lecture_07.md) |
| 第8讲 | SLaT三阶段方法 | [lecture_08.md](lecture_08.md) |
| 第9讲 | T-ROF与理论联系 | [lecture_09.md](lecture_09.md) |
| 第10讲 | 球面与非欧分割 | [lecture_10.md](lecture_10.md) |

### 第三部分: 逆问题与优化 (4讲)

| 讲次 | 主题 | 文件 |
|------|------|------|
| 第11讲 | 图像逆问题 | [lecture_11.md](lecture_11.md) |
| 第12讲 | 贝叶斯推断 | [lecture_12.md](lecture_12.md) |
| 第13讲 | MCMC方法 | [lecture_13.md](lecture_13.md) |
| 第14讲 | 近端算法 | [lecture_14.md](lecture_14.md) |

### 第四部分: 3D视觉 (4讲)

| 讲次 | 主题 | 文件 |
|------|------|------|
| 第15讲 | 点云处理基础 | [lecture_15.md](lecture_15.md) |
| 第16讲 | 图割与能量优化 | [lecture_16.md](lecture_16.md) |
| 第17讲 | Neural Varifolds | [lecture_17.md](lecture_17.md) |
| 第18讲 | 3D目标检测 | [lecture_18.md](lecture_18.md) |

### 第五部分: 医学影像 (3讲)

| 讲次 | 主题 | 文件 |
|------|------|------|
| 第19讲 | 医学影像模态 | [lecture_19.md](lecture_19.md) |
| 第20讲 | 图像重建 | [lecture_20.md](lecture_20.md) |
| 第21讲 | 分割与检测 | [lecture_21.md](lecture_21.md) |

### 第六部分: 深度学习前沿 (3讲)

| 讲次 | 主题 | 文件 |
|------|------|------|
| 第22讲 | 参数高效微调 | [lecture_22.md](lecture_22.md) |
| 第23讲 | 张量分解与网络压缩 | [lecture_23.md](lecture_23.md) |
| 第24讲 | 多模态学习 | [lecture_24.md](lecture_24.md) |

---

## 课件结构

每个课件包含以下内容：

- **标题页**: 讲次、主题、英文名称
- **大纲**: 本讲内容概览
- **详细内容**: 理论推导、算法描述、公式
- **表格与图示**: 方法对比、流程图
- **动画建议**: PPT制作时的动画效果建议
- **总结**: 核心概念框架图
- **课后作业**: 4道不同类型的作业题
- **扩展阅读**: 相关教材、论文、工具
- **参考文献**: 核心引用文献

---

## 核心理论贡献 (基于Cai的论文)

1. **SaT框架**: Smoothing and Thresholding两阶段分割
2. **SLaT框架**: Smoothing, Lifting, and Thresholding三阶段分割
3. **T-ROF**: Thresholded ROF模型
4. **PCMS-ROF等价性定理**: 变分分割与凸优化的理论联系
5. **球面框架变换**: 非欧几何上的图像处理
6. **Neural Varifolds**: 几何表示的神经方法

---

## 转换为PPT

这些Markdown文件可使用以下工具转换为PPT：

1. **Marp**: Markdown演示工具
2. **Pandoc**: 通用文档转换
3. **Slidev**: Vue驱动的演示工具
4. **Obsidian + 插件**: 笔记软件 + 演示插件

### Marp使用示例

```bash
# 安装
npm install -g @marp-team/marp-cli

# 转换
marp lecture_01.md -o lecture_01.pptx
marp lecture_01.md -o lecture_01.pdf
```

---

## 参考资料

### 教材
- Aubert & Kornprobst, *Mathematical Problems in Image Processing*
- Boyd & Vandenberghe, *Convex Optimization*
- Mallat, *A Wavelet Tour of Signal Processing*

### 期刊
- SIAM Journal on Imaging Sciences
- Journal of Mathematical Imaging and Vision
- IEEE Transactions on Pattern Analysis and Machine Intelligence

---

## 联系方式

如有问题或建议，请联系课程负责人。

---

*课件生成时间: 2026年2月*
