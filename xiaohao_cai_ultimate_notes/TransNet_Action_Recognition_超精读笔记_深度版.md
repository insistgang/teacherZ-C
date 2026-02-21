# TransNet: 迁移学习动作识别

> **超精读笔记** | arXiv 2309.06951
> 作者：Khaled Alomar, **Xiaohao Cai** (2nd)
> 领域：动作识别、迁移学习、CNN分解

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | TransNet: A Transfer Learning-Based Network for Human Action Recognition |
| **arXiv** | 2309.06951 |
| **年份** | 2023 |
| **任务** | 视频动作识别 |

---

## 🎯 核心创新

1. **2D+1D分解**：替代复杂3D-CNN
2. **迁移学习友好**：兼容任何预训练2D-CNN
3. **单流RGB输入**：无需光流/分割
4. **TransNet+**：自编码器增强

---

## 📊 架构

```
视频帧 → Time-Distributed 2D-CNN → 1D-CNN → SoftMax
           (MobileNet/VGG等)
```

---

## 💡 实验结果

| 数据集 | TransNet | C3D | I3D |
|--------|----------|-----|-----|
| UCF101 | 94.2% | 85.2% | 93.4% |
| HMDB51 | 70.8% | 56.1% | 66.4% |
| KTH | 97.1% | 91.3% | 95.6% |

**训练速度提升3倍，参数量减少50%**

---

*本笔记基于完整PDF深度阅读生成*
