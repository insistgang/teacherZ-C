# Mermaid 流程图集合

## 1. ROF去噪流程图
```mermaid
flowchart TD
    A[输入噪声图像 f] --> B[计算梯度 ∇f]
    B --> C[初始化 u = f]
    C --> D{迭代}
    D --> E[计算 TV梯度]
    E --> F[更新 u = f + λ·div p]
    F --> G{收敛?}
    G -->|否| D
    G -->|是| H[输出去噪图像 u]
```

## 2. SLaT三阶段流程
```mermaid
flowchart LR
    A[输入图像] --> B[Stage 1: ROF平滑]
    B --> C[Stage 2: RGB+Lab升维]
    C --> D[Stage 3: K-means阈值]
    D --> E[分割结果]
```

## 3. TV正则化原理
```mermaid
flowchart TD
    A[原始图像] --> B[添加噪声]
    B --> C[TV-L2模型]
    C --> D[梯度下降]
    D --> E[Primal-Dual求解]
    E --> F[去噪结果]
```

## 4. 图像分割通用流程
```mermaid
flowchart TD
    A[输入图像] --> B[预处理]
    B --> C[特征提取]
    C --> D{选择方法}
    D -->|传统| E[阈值/边缘/区域]
    D -->|深度学习| F[CNN/Transformer]
    E --> G[后处理]
    F --> G
    G --> H[分割结果]
```

## 5. Chambolle对偶算法
```mermaid
flowchart TD
    A[初始化 p=0] --> B[计算梯度 ∇u]
    B --> C[更新 p = p + τ∇u]
    C --> D[投影 p = p/max 1, |p|]
    D --> E[更新 u = f - λdiv p]
    E --> F{收敛?}
    F -->|否| B
    F -->|是| G[输出 u]
```

## 6. 深度学习分割流程
```mermaid
flowchart TD
    A[数据准备] --> B[数据增强]
    B --> C[模型设计]
    C --> D[损失函数]
    D --> E[训练]
    E --> F[验证]
    F --> G{性能达标?}
    G -->|否| H[调参]
    H --> C
    G -->|是| I[模型保存]
```

## 7. U-Net架构流程
```mermaid
flowchart TD
    A[输入图像] --> B[编码器下采样]
    B --> C[瓶颈层]
    C --> D[解码器上采样]
    D --> E[跳跃连接]
    E --> F[特征融合]
    F --> G[输出分割图]
```

## 8. 注意力机制流程
```mermaid
flowchart TD
    A[输入特征] --> B[Query投影]
    A --> C[Key投影]
    A --> D[Value投影]
    B --> E[注意力分数]
    C --> E
    E --> F[Softmax]
    F --> G[加权求和]
    D --> G
    G --> H[输出特征]
```

## 9. Transformer编码器
```mermaid
flowchart TD
    A[输入嵌入] --> B[位置编码]
    B --> C[多头注意力]
    C --> D[Add & Norm]
    D --> E[前馈网络]
    E --> F[Add & Norm]
    F --> G[输出]
```

## 10. Mamba状态空间模型
```mermaid
flowchart TD
    A[输入序列] --> B[线性投影]
    B --> C[SSM核心]
    C --> D[选择性扫描]
    D --> E[输出投影]
    E --> F[输出序列]
    
    subgraph SSM核心
        C1[状态矩阵 A]
        C2[输入矩阵 B]
        C3[输出矩阵 C]
    end
```

## 11. 语义分割评估流程
```mermaid
flowchart TD
    A[预测结果] --> B[计算混淆矩阵]
    B --> C[计算IoU]
    C --> D[计算mIoU]
    B --> E[计算Precision]
    B --> F[计算Recall]
    E --> G[计算F1]
    F --> G
    D --> H[评估报告]
    G --> H
```

## 12. 数据增强策略
```mermaid
flowchart LR
    A[原始图像] --> B[几何变换]
    A --> C[颜色变换]
    A --> D[噪声添加]
    B --> E[旋转/翻转]
    B --> F[缩放/裁剪]
    C --> G[亮度/对比度]
    C --> H[色调/饱和度]
    D --> I[高斯噪声]
    D --> J[椒盐噪声]
```

## 13. 多尺度特征融合
```mermaid
flowchart TD
    A[输入图像] --> B[尺度1]
    A --> C[尺度2]
    A --> D[尺度3]
    B --> E[特征提取]
    C --> F[特征提取]
    D --> G[特征提取]
    E --> H[上采样]
    F --> H
    G --> H
    H --> I[特征拼接]
    I --> J[融合输出]
```

## 14. 损失函数组合
```mermaid
flowchart TD
    A[预测] --> B[交叉熵损失]
    A --> C[Dice损失]
    A --> D[边界损失]
    B --> E[权重1]
    C --> F[权重2]
    D --> G[权重3]
    E --> H[总损失]
    F --> H
    G --> H
```

## 15. 模型推理流程
```mermaid
flowchart TD
    A[加载模型] --> B[图像预处理]
    B --> C[标准化]
    C --> D[批次处理]
    D --> E[前向传播]
    E --> F[后处理]
    F --> G[结果输出]
```

## 16. 迁移学习流程
```mermaid
flowchart TD
    A[预训练模型] --> B[冻结编码器]
    B --> C[修改解码器]
    C --> D[微调训练]
    D --> E{性能?}
    E -->|不足| F[解冻部分层]
    F --> G[继续训练]
    G --> E
    E -->|达标| H[最终模型]
```

## 17. 超参数优化
```mermaid
flowchart TD
    A[定义搜索空间] --> B{选择策略}
    B -->|网格搜索| C[遍历组合]
    B -->|随机搜索| D[随机采样]
    B -->|贝叶斯| E[模型指导]
    C --> F[训练评估]
    D --> F
    E --> F
    F --> G[记录结果]
    G --> H{最优?}
    H -->|否| B
    H -->|是| I[输出最优参数]
```

## 18. 模型压缩流程
```mermaid
flowchart TD
    A[原始模型] --> B{压缩方式}
    B -->|剪枝| C[移除冗余]
    B -->|量化| D[降低精度]
    B -->|蒸馏| E[知识迁移]
    C --> F[微调]
    D --> F
    E --> F
    F --> G[压缩模型]
```

## 19. 在线学习系统
```mermaid
flowchart TD
    A[新数据] --> B[预处理]
    B --> C[特征提取]
    C --> D[模型预测]
    D --> E[结果反馈]
    E --> F[标注收集]
    F --> G[模型更新]
    G --> H[部署]
```

## 20. 图像分类 vs 分割
```mermaid
flowchart LR
    subgraph 分类
        A1[图像] --> B1[特征]
        B1 --> C1[类别标签]
    end
    
    subgraph 分割
        A2[图像] --> B2[像素特征]
        B2 --> C2[像素标签]
    end
```
