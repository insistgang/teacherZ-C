# ISAR卫星特征识别

> **超精读笔记** | UKACC CONTROL 2024
> 作者：Andrew Begg, Eric Rogers, Bing Chu, **Xiaohao Cai** (4th)
> 领域：空间目标识别、ISAR、计算机视觉

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Automatic Identification of Satellite Features from Inverse Synthetic Aperture Radar (ISAR) Images |
| **会议** | UKACC 14th International Conference on Control |
| **年份** | 2024 |
| **页数** | 149-150 |

---

## 🎯 核心创新

1. **ISAR仿真器**：Python实现，比FEKO快
2. **特征标注**：太阳能板、镜头等组件
3. **YOLO检测**：实时目标识别
4. **多尺度测试**：不同分辨率评估

---

## 📊 方法

### 数据生成

- 4个模型：3颗卫星+1个火箭体
- 100张/目标，随机姿态
- 不同分辨率模拟不同距离

### 评估

```
ISAR图像 → 边缘检测 → YOLO → 特征识别
```

---

## 💡 实验结果

| 实验 | 准确率 |
|------|--------|
| 卫星vs火箭（15张测试）| **100%** |
| 多类卫星（进行中）| 待发表 |

---

*本笔记基于完整PDF深度阅读生成*
