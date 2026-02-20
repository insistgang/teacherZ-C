# D:\Documents\zx 读书笔记整理方案

> 生成时间: 2026-02-19

## 📊 现状分析

### 文件分布统计

| 目录 | 笔记数量 | 说明 |
|------|----------|------|
| `xiaohao_cai_ultimate_notes/` | 82篇 | 带编号的论文精读笔记（主要存储） |
| `01_notes/ultimate/` | 135篇 | 超精读笔记（含重复和已填充版本） |
| `01_notes/single_papers/` | 31篇 | 单篇论文笔记 |
| `01_notes/multi_agent/` | 30篇 | 多智能体分析报告 |
| 根目录散落 | 98篇 | 未整理的精读报告 |
| **总计** | **~376篇** | 含大量重复 |

### 问题诊断

1. **重复文件严重**: 同一论文有多个版本
   - 例如 `Mumford-Shah_and_ROF_Linkage_超精读笔记.md` 和 `Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md`
   - `01_notes/ultimate/` 和 `xiaohao_cai_ultimate_notes/` 存在大量重复

2. **命名不规范**:
   - 混用中英文命名
   - 编号格式不统一 (`[XX]_`, `论文精读_[X-XX]_`, 无编号)

3. **目录结构混乱**:
   - 根目录散落98个笔记文件
   - 分类不清晰

---

## 🎯 整理目标

### 1. 统一目录结构

```
D:/Documents/zx/
├── README.md                           # 项目总览
├── 01_notes/                          # 核心笔记目录
│   ├── papers/                        # 论文精读笔记（统一格式）
│   │   ├── 001_Mumford-Shah_ROF_Linkage.md
│   │   ├── 002_SLAT_Three_Stage_Segmentation.md
│   │   └── ...
│   ├── multi_agent/                   # 多智能体分析报告
│   └── summaries/                     # 分类汇总笔记
├── 02_learning/                       # 学习资料
├── 03_analysis/                       # 分析工具
├── 04_project/                        # 项目代码
└── 05_docs/                           # 文档资料
```

### 2. 统一命名规范

**论文笔记命名格式**: `{编号}_{主题关键词}_{年份}.md`
- 编号: 3位数字 (001-081)
- 主题: 英文关键词（下划线连接）
- 年份: 发表年份

**示例**:
- `001_Mumford_Shah_ROF_Linkage_2018.md`
- `022_SLAT_Three_Stage_Segmentation_2017.md`
- `074_Proximal_Nested_Sampling_2021.md`

### 3. 去重策略

| 保留 | 删除 |
|------|------|
| `xiaohao_cai_ultimate_notes/[XX]_*_详细版.md` | `01_notes/ultimate/*_超精读笔记_已填充.md` |
| 最新修改日期的版本 | 旧版本/重复版本 |
| 内容最完整的版本 | 空白/模板文件 |

---

## 📋 整理步骤

### Phase 1: 备份 (重要!)
```bash
# 创建备份
cp -r D:/Documents/zx D:/Documents/zx_backup_$(date +%Y%m%d)
```

### Phase 2: 文件去重
1. 对比 `xiaohao_cai_ultimate_notes/` 和 `01_notes/ultimate/`
2. 保留内容更完整的版本
3. 删除 `_已填充.md` 等重复后缀文件

### Phase 3: 移动散落文件
```bash
# 将根目录散落的笔记移动到 01_notes/papers/
mv D:/Documents/zx/*精读*.md D:/Documents/zx/01_notes/papers/
```

### Phase 4: 统一命名
运行重命名脚本，将所有笔记改为统一格式

### Phase 5: 重建索引
更新 `00_全部论文精读笔记索引_81篇.md`

---

## 🔧 自动化脚本

### 去重脚本 (de_duplicate_notes.py)
```python
import os
from pathlib import Path
import filecmp

def find_duplicates():
    """查找重复笔记"""
    # 实现文件内容对比逻辑
    pass

def remove_duplicates():
    """删除重复文件（先移动到备份目录）"""
    pass
```

### 重命名脚本 (rename_notes.py)
```python
def standardize_name(old_name):
    """将旧文件名转换为新格式"""
    # [34]_变分分割基础Mumford-Shah与ROF Mumford-Shah ROF_详细版.md
    # -> 034_Mumford_Shah_ROF_Linkage_2018.md
    pass
```

---

## ⚠️ 注意事项

1. **不要直接删除**: 先移动到 `to_delete/` 目录，确认无误后再删除
2. **保留原始PDF**: PDF文件不受影响
3. **更新引用**: 重组后需要更新索引文件中的链接
4. **Git追踪**: 整理过程分阶段提交，便于回滚

---

## 📈 预期效果

| 指标 | 整理前 | 整理后 |
|------|--------|--------|
| 笔记总数 | ~376篇 | 81篇（去重后） |
| 目录层级 | 混乱 | 3层清晰结构 |
| 命名规范 | 4种格式 | 1种统一格式 |
| 重复文件 | ~200篇 | 0篇 |

---

## 下一步

请确认整理方案后，我可以帮你：
1. 创建备份
2. 执行去重脚本
3. 重命名文件
4. 重建索引

是否开始执行整理？
