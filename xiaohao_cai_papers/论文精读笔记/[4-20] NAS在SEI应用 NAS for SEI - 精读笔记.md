# [4-20] NAS在SEI应用 NAS for SEI - 精读笔记

> **论文标题**: Neural Architecture Search for Specific Emitter Identification
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐ (中)
> **重要性**: ⭐⭐⭐ (NAS在信号处理领域的应用)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Neural Architecture Search for Specific Emitter Identification |
| **作者** | Xiaohao Cai 等人 |
| **应用领域** | Specific Emitter Identification (SEI) |
| **关键词** | NAS, SEI, Signal Processing, Deep Learning |
| **核心价值** | 将NAS应用于无线信号识别任务 |

---

## 🎯 SEI核心问题

### 特定发射器识别问题

```
SEI问题定义:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

目标: 通过射频信号识别特定发射设备

输入: 接收到的射频信号 x(t)
输出: 发射器身份 ID

挑战:
  1. 信号噪声干扰
  2. 多径效应
  3. 设备间差异微小
  4. 实时性要求

传统方法:
  - 手工特征提取 (统计特征、高阶矩)
  - 专家知识依赖
  - 泛化能力有限

深度学习方法:
  - 自动特征学习
  - 端到端训练
  - 需要合适的网络架构
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### SEI信号处理流程

| 阶段 | 处理内容 | 输出 |
|:---|:---|:---|
| **信号采集** | 射频接收、ADC采样 | 原始I/Q数据 |
| **预处理** | 归一化、去噪、分帧 | 信号片段 |
| **特征提取** | 时频分析、深度学习 | 特征向量 |
| **分类识别** | 分类器、相似度匹配 | 设备ID |

---

## 🔬 NAS for SEI方法论

### 整体框架

```
┌─────────────────────────────────────────────────────────────┐
│              NAS for SEI 框架                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           SEI专用搜索空间                            │   │
│  │                                                     │   │
│  │  时域分支: 1D-Conv, LSTM, GRU                       │   │
│  │  频域分支: FFT, Spectrogram, 2D-Conv                │   │
│  │  融合操作: Concat, Attention, Bilinear              │   │
│  │  分类头: FC, Softmax                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           多模态特征融合 ⭐核心                       │   │
│  │                                                     │   │
│  │   I/Q信号 ──┬──→ 时域特征 ──┐                       │   │
│  │             │               ├──→ 融合 → 分类        │   │
│  │             └──→ 频域特征 ──┘                       │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           架构搜索策略                               │   │
│  │  - 强化学习 (Controller-RNN)                        │   │
│  │  - 进化算法                                         │   │
│  │  - 梯度优化 (DARTS)                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           SEI性能评估                                │   │
│  │  - 识别准确率                                       │   │
│  │  - 抗噪鲁棒性                                       │   │
│  │  - 计算效率                                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 核心组件1: SEI专用搜索空间

```python
class SEISearchSpace:
    """
    SEI任务的专用NAS搜索空间

    针对射频信号特点设计
    """

    def __init__(self):
        # 时域操作
        self.temporal_ops = [
            'conv1d_3',      # 1D卷积,核大小3
            'conv1d_5',      # 1D卷积,核大小5
            'conv1d_7',      # 1D卷积,核大小7
            'lstm_64',       # LSTM,隐藏层64
            'gru_64',        # GRU,隐藏层64
            'maxpool1d_2',   # 1D最大池化
            'avgpool1d_2',   # 1D平均池化
            'identity',      # 恒等连接
            'zero',          # 零连接
        ]

        # 频域操作
        self.spectral_ops = [
            'fft',           # 快速傅里叶变换
            'stft',          # 短时傅里叶变换
            'conv2d_3x3',    # 2D卷积
            'conv2d_5x5',    # 2D卷积
            'spectral_attn', # 频谱注意力
        ]

        # 融合操作
        self.fusion_ops = [
            'concat',        # 拼接
            'add',           # 相加
            'attention',     # 注意力融合
            'bilinear',      # 双线性融合
        ]

        self.num_layers = 8

    def sample_architecture(self):
        """随机采样一个架构"""
        arch = {
            'temporal_branch': [],
            'spectral_branch': [],
            'fusion_op': None,
            'classifier': None
        }

        # 采样时域分支
        for _ in range(self.num_layers):
            op = random.choice(self.temporal_ops)
            arch['temporal_branch'].append(op)

        # 采样频域分支
        for _ in range(self.num_layers // 2):
            op = random.choice(self.spectral_ops)
            arch['spectral_branch'].append(op)

        # 采样融合操作
        arch['fusion_op'] = random.choice(self.fusion_ops)

        # 采样分类头
        arch['classifier'] = random.choice(['fc_128', 'fc_256', 'fc_512'])

        return arch


class SEINetwork(nn.Module):
    """
    SEI网络架构

    双分支结构: 时域 + 频域
    """

    def __init__(self, arch, num_classes=10):
        super().__init__()

        self.temporal_branch = self._build_temporal_branch(arch['temporal_branch'])
        self.spectral_branch = self._build_spectral_branch(arch['spectral_branch'])
        self.fusion = self._build_fusion(arch['fusion_op'])
        self.classifier = self._build_classifier(arch['classifier'], num_classes)

    def _build_temporal_branch(self, ops):
        """构建时域分支"""
        layers = []
        in_channels = 2  # I/Q两通道

        for op in ops:
            if op.startswith('conv1d'):
                kernel_size = int(op.split('_')[1])
                out_channels = 64
                layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                ])
                in_channels = out_channels
            elif op.startswith('lstm'):
                hidden_size = int(op.split('_')[1])
                layers.append(nn.LSTM(in_channels, hidden_size, batch_first=True))
                in_channels = hidden_size
            elif op.startswith('gru'):
                hidden_size = int(op.split('_')[1])
                layers.append(nn.GRU(in_channels, hidden_size, batch_first=True))
                in_channels = hidden_size
            elif op == 'maxpool1d_2':
                layers.append(nn.MaxPool1d(2))
            elif op == 'avgpool1d_2':
                layers.append(nn.AvgPool1d(2))

        return nn.Sequential(*layers)

    def _build_spectral_branch(self, ops):
        """构建频域分支"""
        layers = []

        for op in ops:
            if op == 'fft':
                layers.append(FFTLayer())
            elif op == 'stft':
                layers.append(STFTLayer(n_fft=256))
            elif op.startswith('conv2d'):
                kernel_size = int(op.split('_')[1].split('x')[0])
                # 2D卷积层...

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, 2, L) I/Q信号, L为序列长度

        Returns:
            logits: (B, num_classes)
        """
        # 时域特征
        temporal_feat = self.temporal_branch(x)
        temporal_feat = temporal_feat.mean(dim=-1)  # 全局池化

        # 频域特征
        spectral_feat = self.spectral_branch(x)
        spectral_feat = spectral_feat.mean(dim=[-2, -1])  # 全局池化

        # 融合
        fused_feat = self.fusion(temporal_feat, spectral_feat)

        # 分类
        logits = self.classifier(fused_feat)

        return logits


class FFTLayer(nn.Module):
    """FFT层"""

    def forward(self, x):
        # x: (B, 2, L)
        # 转换为复数
        complex_signal = torch.view_as_complex(x.permute(0, 2, 1).contiguous())

        # FFT
        fft_result = torch.fft.fft(complex_signal, dim=-1)

        # 幅度谱
        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)

        # 拼接
        return torch.stack([magnitude, phase], dim=1)


class STFTLayer(nn.Module):
    """短时傅里叶变换层"""

    def __init__(self, n_fft=256, hop_length=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 4

    def forward(self, x):
        # x: (B, 2, L)
        batch_size = x.size(0)

        # 分别对I/Q做STFT
        spectrograms = []
        for i in range(2):
            spec = torch.stft(x[:, i], n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            return_complex=True)
            spectrograms.append(torch.abs(spec))

        # 堆叠 (B, 2, F, T)
        return torch.stack(spectrograms, dim=1)
```

---

### 核心组件2: 信号预处理

```python
class SEIPreprocessor:
    """
    SEI信号预处理器
    """

    def __init__(self, sample_rate=1e6, segment_length=1024):
        self.sample_rate = sample_rate
        self.segment_length = segment_length

    def preprocess(self, raw_signal):
        """
        预处理原始I/Q信号

        Args:
            raw_signal: (2, N) I/Q信号

        Returns:
            processed: (2, L) 预处理后的信号段
        """
        # 1. 归一化
        signal = self.normalize(raw_signal)

        # 2. 去噪 (可选)
        signal = self.denoise(signal)

        # 3. 分帧
        segments = self.segment(signal)

        # 4. 数据增强
        segments = self.augment(segments)

        return segments

    def normalize(self, signal):
        """能量归一化"""
        power = torch.mean(signal ** 2)
        return signal / torch.sqrt(power + 1e-10)

    def denoise(self, signal, method='wavelet'):
        """去噪"""
        if method == 'wavelet':
            # 小波去噪
            return self.wavelet_denoise(signal)
        elif method == 'spectral_gating':
            # 频谱门控
            return self.spectral_gating(signal)
        return signal

    def segment(self, signal):
        """分帧"""
        N = signal.size(1)
        num_segments = N // self.segment_length

        segments = []
        for i in range(num_segments):
            start = i * self.segment_length
            end = start + self.segment_length
            segments.append(signal[:, start:end])

        return torch.stack(segments)

    def augment(self, segments):
        """数据增强"""
        augmented = []

        for seg in segments:
            # 随机噪声
            if random.random() < 0.5:
                noise = torch.randn_like(seg) * 0.01
                seg = seg + noise

            # 随机相位旋转
            if random.random() < 0.3:
                phase = random.uniform(0, 2 * np.pi)
                seg = self.rotate_phase(seg, phase)

            augmented.append(seg)

        return torch.stack(augmented)

    def rotate_phase(self, signal, phase):
        """相位旋转"""
        # 转换为复数,旋转,转回
        complex_signal = signal[0] + 1j * signal[1]
        rotated = complex_signal * np.exp(1j * phase)
        return torch.stack([rotated.real, rotated.imag])
```

---

### 核心组件3: 架构搜索策略

```python
class SEINASSearcher:
    """
    SEI任务的NAS搜索器
    """

    def __init__(self, search_space, train_loader, val_loader):
        self.search_space = search_space
        self.train_loader = train_loader
        self.val_loader = val_loader

    def evaluate_architecture(self, arch, epochs=10):
        """
        评估架构性能

        Args:
            arch: 架构配置
            epochs: 训练轮数

        Returns:
            metrics: {'accuracy': ..., 'flops': ..., 'params': ...}
        """
        # 构建模型
        model = SEINetwork(arch, num_classes=10)

        # 计算FLOPs和参数量
        flops = self.count_flops(model)
        params = sum(p.numel() for p in model.parameters())

        # 训练
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for batch in self.train_loader:
                signals, labels = batch

                optimizer.zero_grad()
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 验证
        accuracy = self.evaluate(model)

        return {
            'accuracy': accuracy,
            'flops': flops,
            'params': params
        }

    def random_search(self, num_samples=100):
        """随机搜索"""
        results = []

        for i in range(num_samples):
            arch = self.search_space.sample_architecture()
            metrics = self.evaluate_architecture(arch)

            results.append({
                'architecture': arch,
                'metrics': metrics
            })

            print(f"Sample {i+1}/{num_samples}: Acc={metrics['accuracy']:.4f}")

        # 返回最佳架构
        best = max(results, key=lambda x: x['metrics']['accuracy'])
        return best

    def evolutionary_search(self, population_size=20, generations=10):
        """进化搜索"""
        # 初始化种群
        population = [self.search_space.sample_architecture()
                     for _ in range(population_size)]

        for gen in range(generations):
            # 评估
            fitness = [self.evaluate_architecture(arch)
                      for arch in population]

            # 选择
            sorted_indices = sorted(range(population_size),
                                  key=lambda i: fitness[i]['accuracy'],
                                  reverse=True)
            elites = [population[i] for i in sorted_indices[:population_size//2]]

            # 交叉和变异
            offspring = []
            while len(offspring) < population_size - len(elites):
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                offspring.append(child)

            population = elites + offspring

            best_acc = fitness[sorted_indices[0]]['accuracy']
            print(f"Generation {gen+1}: Best Acc={best_acc:.4f}")

        return population[sorted_indices[0]]
```

---

## 📊 实验结果

### SEI数据集性能

| 方法 | 识别准确率 | 参数量 | FLOPs |
|:---|:---:|:---:|:---:|
| 手工特征 + SVM | 78.5% | - | - |
| CNN基线 | 85.2% | 2.1M | 45M |
| LSTM基线 | 87.3% | 1.8M | 38M |
| ResNet-18 | 89.1% | 11M | 180M |
| **NAS-SEI** | **92.4%** | **1.2M** | **28M** |

### 消融实验

| 组件 | 准确率提升 | 说明 |
|:---|:---:|:---|
| 时域分支 | +3.2% | 捕捉瞬态特征 |
| 频域分支 | +4.1% | 捕捉频谱特征 |
| 双分支融合 | +5.8% | 互补信息 |
| NAS优化 | +3.3% | 架构搜索 |

---

## 💡 对井盖检测的启示

### 跨领域应用思路

```
SEI → 井盖检测的迁移:

SEI特点:
  - 双通道I/Q信号
  - 时频双分支处理
  - 微弱特征提取

井盖检测可借鉴:
  - 多模态输入 (RGB + 深度/红外)
  - 双分支架构
  - NAS自动搜索
```

### 多模态井盖检测

```python
class MultimodalManholeDetector:
    """
    多模态井盖检测器

    借鉴SEI的双分支思想
    """

    def __init__(self, searched_arch):
        # 可见光分支
        self.rgb_branch = self._build_rgb_branch(searched_arch['rgb'])

        # 深度/红外分支
        self.depth_branch = self._build_depth_branch(searched_arch['depth'])

        # 融合
        self.fusion = FusionModule(searched_arch['fusion'])

        # 检测头
        self.detector = YOLOHead()

    def forward(self, rgb, depth):
        """
        Args:
            rgb: (B, 3, H, W) 可见光图像
            depth: (B, 1, H, W) 深度图

        Returns:
            detections: 检测结果
        """
        # 各分支特征
        rgb_feat = self.rgb_branch(rgb)
        depth_feat = self.depth_branch(depth)

        # 融合
        fused_feat = self.fusion(rgb_feat, depth_feat)

        # 检测
        return self.detector(fused_feat)
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **SEI** | Specific Emitter Identification | 特定发射器识别 |
| **I/Q数据** | In-phase/Quadrature | 同相/正交信号 |
| **STFT** | Short-Time Fourier Transform | 短时傅里叶变换 |
| **双分支** | Two-Branch | 并行处理结构 |

---

## ✅ 复习检查清单

- [ ] 理解SEI问题的特点
- [ ] 了解射频信号处理方法
- [ ] 掌握双分支架构设计
- [ ] 理解NAS在信号处理中的应用

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
