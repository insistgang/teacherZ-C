# 多智能体论文辩论系统

## 项目简介

这是一个多智能体协作系统，通过角色化辩论的方式深入理解学术论文（特别是Xiaohao Cai的论文）。每个Agent扮演特定角色，从不同视角分析论文，通过辩论交流达成深度共识。

## 系统架构

```
用户界面
    ↓
协调器层 (论文解析、辩论调度、状态管理)
    ↓
Agent管理器 + 消息队列 + LLM接口
    ↓
专业Agent (数学家、工程师、应用专家、质疑者、综合者)
```

## Agent角色

| 角色 | 中文名 | 关注焦点 |
|------|--------|----------|
| Mathematician | 数学家 | 公式推导、理论正确性、数学证明 |
| Engineer | 工程师 | 实现细节、复杂度分析、代码可行性 |
| ApplicationExpert | 应用专家 | 应用场景、数据需求、落地价值 |
| Skeptic | 质疑者 | 找漏洞、反例、批判性分析 |
| Synthesizer | 综合者 | 汇总观点、生成报告、判定共识 |

## 辩论流程

1. **第一轮：独立分析发言** - 每个Agent独立分析论文
2. **交互辩论轮次** - Agent间互相回应、质疑
3. **综合者判定** - 评估共识程度
4. **生成报告** - 输出辩论记录和最终理解报告

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```bash
# 使用模拟LLM进行快速测试
python debate_system.py --mock

# 从PDF文件进行辩论
python debate_system.py --paper papers/sample.pdf

# 指定论文内容和标题
python debate_system.py --title "论文标题" --content "论文内容..."

# 设置最大辩论轮次
python debate_system.py --mock --max-rounds 3
```

### Python API使用

```python
import asyncio
from debate_system import DebateSystem, Paper

async def main():
    # 创建论文对象
    paper = Paper(
        title="基于深度学习的图像分割方法",
        authors=["张三", "李四"],
        year=2024,
        abstract="论文摘要...",
        content="论文正文..."
    )

    # 创建系统并辩论
    system = DebateSystem()
    await system.debate_paper(paper)

asyncio.run(main())
```

## 配置文件

编辑 `config.yaml` 自定义系统行为：

```yaml
debate:
  max_rounds: 5          # 最大辩论轮次
  consensus_threshold: 0.7  # 共识阈值

agents:
  mathematician:
    enabled: true
    temperature: 0.3      # 较低温度，更确定性的输出
  skeptic:
    temperature: 0.7      # 较高温度，更多样化的质疑
```

## 输出文件

辩论结束后，系统会在 `outputs/` 目录生成：

- `debate_log_*.md` - 完整辩论记录（Markdown格式）
- `report_*.md` - 最终理解报告
- `debate_data_*.json` - 结构化数据（JSON格式）

## 使用真实LLM

要使用Claude API进行真实辩论，需要设置环境变量：

```bash
export ANTHROPIC_API_KEY="your-api-key"
python debate_system.py --paper papers/sample.pdf
```

## 示例输出

### 辩论记录格式

```markdown
# 论文辩论记录：基于 Mumford-Shah 泛函的凸松弛图像分割方法

**辩论时间**: 2026-02-16 02:53
**参与Agent**: 数学家, 工程师, 应用专家, 质疑者
**总轮次**: 5
**共识度**: 0.80

## 第1轮：独立分析发言

### 数学家发言
[详细分析内容...]

### 工程师发言
[详细分析内容...]
```

### 最终报告格式

```markdown
# 论文深度理解报告：[论文标题]

**共识度评分**: ⭐⭐⭐⭐ (0.85/1.0)

## 1. 核心贡献
## 2. 主要优势
## 3. 主要局限与问题
## 4. 改进方向
## 5. 应用建议
```

## 扩展性

### 添加新的Agent角色

```python
from debate_system import BaseAgent, AgentRole

class CustomAgent(BaseAgent):
    def __init__(self, llm_client, config=None):
        super().__init__(
            role=AgentRole.CUSTOM,
            name="自定义Agent",
            llm_client=llm_client,
            config=config
        )
```

### 自定义辩论流程

修改 `DebateScheduler` 类中的流程控制方法：
- `_conduct_first_round()` - 第一轮发言
- `_conduct_interaction_round()` - 交互轮次
- `_should_continue()` - 继续条件判断

## 技术栈

- **Python**: 3.10+
- **LLM**: Claude Opus 4.6 / GPT-4
- **PDF解析**: pdfplumber, PyPDF2
- **数据管理**: Pydantic dataclasses
- **配置**: YAML

## 文件结构

```
.
├── agent_debate_system.md   # 完整设计文档
├── debate_system.py         # 主系统实现
├── run_debate.py            # 使用示例
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖包
├── outputs/                 # 输出目录
│   ├── debate_logs/         # 辩论记录
│   ├── reports/             # 最终报告
│   └── data/                # JSON数据
└── README.md                # 本文件
```

## 设计文档

详细的设计文档请参考：`agent_debate_system.md`

包含内容：
- 完整系统架构
- 数据结构设计
- Agent角色详细定义
- 辩论流程图
- 配置管理
- 输出格式规范
- 扩展性设计
- 部署方案

## 许可证

MIT License
