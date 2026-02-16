# 多智能体论文辩论系统设计文档

## 1. 系统概述

### 1.1 设计目标
构建一个多智能体协作系统，通过角色化辩论的方式深入理解Xiaohao Cai的学术论文。每个Agent扮演特定角色，从不同视角分析论文，通过辩论交流达成深度共识。

### 1.2 核心价值
- **多视角分析**：不同角色关注论文的不同维度
- **批判性思维**：质疑者角色确保不遗漏问题
- **深度理解**：通过辩论而非单向阅读促进理解
- **知识沉淀**：生成结构化的辩论记录和最终报告

## 2. Agent角色设计

### 2.1 数学家Agent (Mathematician)
**关注焦点**：
- 公式推导的正确性
- 理论假设的合理性
- 数学证明的完整性
- 算法收敛性分析

**输入提示词模板**：
```
作为数学家，请从以下角度分析论文：
1. 核心数学公式是否推导正确？
2. 理论假设是否充分？有无隐藏假设？
3. 优化问题的目标函数设计是否合理？
4. 算法的收敛性是否有理论保证？
5. 数学符号是否统一、清晰？

请以严谨的数学态度进行分析，指出任何理论上的问题。
```

**输出格式**：
- 理论分析（公式、证明）
- 问题列表（理论缺陷）
- 改进建议（数学角度）

### 2.2 工程师Agent (Engineer)
**关注焦点**：
- 算法实现的可行性
- 计算复杂度分析
- 数值稳定性
- 代码可复现性

**输入提示词模板**：
```
作为工程师，请从以下角度分析论文：
1. 算法描述是否足够清晰以实现？
2. 时间/空间复杂度如何？是否可接受？
3. 数值计算是否稳定？有无潜在溢出/下溢问题？
4. 超参数敏感性如何？
5. 代码复现需要哪些关键信息？

请以工程实现的角度，评估论文的可落地性。
```

**输出格式**：
- 实现难度评估
- 复杂度分析
- 潜在工程问题
- 实现建议

### 2.3 应用专家Agent (ApplicationExpert)
**关注焦点**：
- 实际应用场景
- 数据需求与获取难度
- 与现有方法对比
- 行业落地价值

**输入提示词模板**：
```
作为应用专家，请从以下角度分析论文：
1. 该方法适用于哪些实际场景？
2. 数据需求是否现实？
3. 与SOTA方法相比的优势/劣势？
4. 落地的主要障碍是什么？
5. 商业/应用价值如何评估？

请从实际应用的角度，评估论文的实用价值。
```

**输出格式**：
- 应用场景分析
- 市场对比
- 落地挑战
- 应用建议

### 2.4 质疑者Agent (Skeptic)
**关注焦点**：
- 实验设计的合理性
- 结果的可信度
- 论点与证据的一致性
- 潜在反例

**输入提示词模板**：
```
作为质疑者，请批判性地分析论文：
1. 实验设置是否公平？有无 cherry-picking？
2. 结果是否足够显著？有无统计显著性检验？
3. 论文声称的贡献是否被实验充分支持？
4. 有无被忽略的反例或边缘情况？
5. 哪些声明是过度承诺？

请以批判的眼光，找出论文中的任何问题或夸大之处。
```

**输出格式**：
- 批判性问题列表
- 实验设计缺陷
- 过度声明指出
- 反例构造

### 2.5 综合者Agent (Synthesizer)
**关注焦点**：
- 汇总各Agent观点
- 识别共识与分歧
- 判定辩论是否充分
- 生成最终理解报告

**输入提示词模板**：
```
作为综合者，请完成以下任务：
1. 汇总所有Agent的观点，分类整理
2. 识别各角色之间的共识点
3. 识别分歧点，判断是否需要进一步辩论
4. 评估辩论是否充分（是否达成共识）
5. 生成最终理解报告，包含：
   - 论文核心贡献总结
   - 主要优势
   - 主要局限/问题
   - 改进方向
   - 应用建议

请综合各方观点，生成一份平衡、全面的分析报告。
```

**输出格式**：
- 辩论总结（共识/分歧）
- 共识判定（是否继续辩论）
- 最终理解报告

## 3. 辩论流程设计

### 3.1 流程图
```
┌─────────────────────────────────────────────────────────────┐
│                        论文输入                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              第一轮：独立分析发言                             │
│  数学家 → 工程师 → 应用专家 → 质疑者                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              交互辩论轮次（可选，多轮）                        │
│  每个Agent可选择：                                           │
│  1. 回应其他Agent的观点                                      │
│  2. 提出新问题                                               │
│  3. 支持或反对某观点                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              综合者判定                                      │
│  - 评估辩论充分性                                           │
│  - 决定：继续辩论 OR 生成报告                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │ 继续辩论    │         │ 生成报告    │
    │ (回到交互轮)│         └─────────────┘
    └─────────────┘               │
                                  ▼
                   ┌─────────────────────────────┐
                   │     输出：                  │
                   │ 1. 辩论记录（Markdown）     │
                   │ 2. 最终理解报告（Markdown） │
                   │ 3. 共识评分                 │
                   └─────────────────────────────┘
```

### 3.2 辩论规则

**发言规则**：
1. 每轮每个Agent最多发言一次
2. 发言必须基于论文内容或前序发言
3. 质疑者优先发言（确保问题不被掩盖）

**回应规则**：
1. 被点名质疑的Agent必须回应
2. 其他Agent可选择回应或保持沉默
3. 回应应直接针对问题，避免偏题

**终止规则**：
1. 达成共识（所有关键问题都有明确结论）
2. 连续两轮无新观点产生
3. 达到最大轮次限制（默认5轮）

## 4. 数据结构设计

### 4.1 论文数据结构
```python
@dataclass
class Paper:
    title: str
    authors: List[str]
    year: int
    abstract: str
    content: str  # 论文全文
    pdf_path: Optional[str] = None
    category: Optional[str] = None  # 论文分类
```

### 4.2 Agent发言数据结构
```python
@dataclass
class AgentMessage:
    agent_role: str  # Agent角色
    agent_name: str  # Agent名称
    content: str     # 发言内容
    timestamp: datetime
    round: int       # 辩论轮次
    reply_to: Optional[str] = None  # 回复的消息ID
    message_id: str = field(default_factory=lambda: str(uuid4()))
```

### 4.3 辩论状态数据结构
```python
@dataclass
class DebateState:
    paper: Paper
    messages: List[AgentMessage]
    current_round: int
    status: str  # 'initialized', 'in_progress', 'consensus', 'terminated'
    consensus_score: float  # 0-1，共识程度
    unresolved_issues: List[str]
    max_rounds: int = 5
```

### 4.4 最终报告数据结构
```python
@dataclass
class DebateReport:
    paper_title: str
    debate_summary: str
    consensus_points: List[str]
    disagreements: List[str]
    key_insights: Dict[str, List[str]]  # 按角色分类的见解
    final_assessment: str
    recommendations: List[str]
    consensus_score: float
    total_rounds: int
    timestamp: datetime
```

## 5. 技术架构

### 5.1 系统架构图
```
┌─────────────────────────────────────────────────────────────┐
│                        用户界面                              │
│  - 论文上传  - 辩论启动  - 进度监控  - 报告查看              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     协调器层                                  │
│  - 论文解析器  - 辩论调度器  - 状态管理器                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Agent管理器  │ │ 消息队列    │ │ LLM接口     │
└─────────────┘ └─────────────┘ └─────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ 数学家Agent │ │ 工程师Agent │ │ ...其他Agent │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 5.2 模块说明

**协调器层**：
- `PaperParser`: 解析PDF论文，提取文本
- `DebateScheduler`: 管理辩论流程，控制发言顺序
- `StateManager`: 维护辩论状态

**Agent管理器**：
- `AgentFactory`: 创建各角色Agent
- `AgentRegistry`: 管理已注册的Agent

**LLM接口**：
- `LLMClient`: 统一的LLM调用接口
- 支持多种模型：Claude, GPT-4, 等

**消息系统**：
- `MessageBus`: Agent间消息传递
- `MessageHistory`: 完整辩论历史记录

## 6. 实现细节

### 6.1 Agent基类设计
```python
class BaseAgent(ABC):
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.memory = []  # 记忆历史

    @abstractmethod
    def analyze(self, paper: Paper, context: List[AgentMessage]) -> str:
        """分析论文并生成发言"""
        pass

    @abstractmethod
    def respond(self, message: AgentMessage, context: List[AgentMessage]) -> str:
        """回应其他Agent"""
        pass
```

### 6.2 辩论调度器
```python
class DebateScheduler:
    def __init__(self, agents: List[BaseAgent], max_rounds: int = 5):
        self.agents = agents
        self.max_rounds = max_rounds
        self.state = None

    def start_debate(self, paper: Paper) -> DebateState:
        """启动辩论"""
        pass

    def next_round(self) -> bool:
        """进入下一轮，返回是否应该继续"""
        pass

    def check_consensus(self) -> float:
        """检查共识程度，返回0-1分数"""
        pass
```

### 6.3 报告生成器
```python
class ReportGenerator:
    def generate_debate_log(self, state: DebateState) -> str:
        """生成辩论记录Markdown"""
        pass

    def generate_final_report(self, state: DebateState) -> DebateReport:
        """生成最终理解报告"""
        pass
```

## 7. 配置管理

### 7.1 配置文件结构
```yaml
# config.yaml
debate:
  max_rounds: 5
  consensus_threshold: 0.7
  timeout_seconds: 300

agents:
  mathematician:
    enabled: true
    model: "claude-opus-4"
    temperature: 0.3
  engineer:
    enabled: true
    model: "claude-opus-4"
    temperature: 0.5
  application_expert:
    enabled: true
    model: "claude-opus-4"
    temperature: 0.5
  skeptic:
    enabled: true
    model: "claude-opus-4"
    temperature: 0.7
  synthesizer:
    enabled: true
    model: "claude-opus-4"
    temperature: 0.4

llm:
  provider: "anthropic"  # or "openai"
  api_key_env: "ANTHROPIC_API_KEY"
  max_tokens: 4000

output:
  debate_log_dir: "./outputs/debate_logs"
  report_dir: "./outputs/reports"
```

## 8. 输出格式

### 8.1 辩论记录格式
```markdown
# 论文辩论记录：[论文标题]

**辩论时间**: 2026-02-16
**参与Agent**: 数学家, 工程师, 应用专家, 质疑者, 综合者
**总轮次**: 3
**共识度**: 0.85

---

## 第一轮：独立分析发言

### 数学家发言
[内容...]

### 工程师发言
[内容...]

### 应用专家发言
[内容...]

### 质疑者发言
[内容...]

---

## 第二轮：交互辩论

### 数学家回应质疑者
[内容...]

### 工程师补充
[内容...]

---

## 共识总结
- 共识点1
- 共识点2

## 分歧点
- 分歧1
- 分歧2
```

### 8.2 最终报告格式
```markdown
# 论文深度理解报告：[论文标题]

**生成时间**: 2026-02-16
**论文作者**: [作者列表]
**共识度评分**: ⭐⭐⭐⭐ (0.85/1.0)

---

## 1. 核心贡献

### 1.1 理论贡献（数学家视角）
[内容...]

### 1.2 技术贡献（工程师视角）
[内容...]

### 1.3 应用价值（应用专家视角）
[内容...]

---

## 2. 主要优势

| 视角 | 优势描述 |
|------|----------|
| 数学 | ... |
| 工程 | ... |
| 应用 | ... |

---

## 3. 主要局限与问题

### 3.1 理论局限
[内容...]

### 3.2 工程挑战
[内容...]

### 3.3 应用障碍
[内容...]

---

## 4. 改进方向

### 4.1 理论改进
[内容...]

### 4.2 实现优化
[内容...]

### 4.3 应用拓展
[内容...]

---

## 5. 应用建议

[内容...]

---

## 6. 辩论统计

- **总发言数**: 15
- **共识点数**: 8
- **分歧点数**: 3
- **质疑总数**: 7
- **被回应质疑**: 6

---

*本报告由多智能体辩论系统自动生成*
```

## 9. 扩展性设计

### 9.1 新增Agent角色
系统支持动态添加新的Agent角色：
1. 继承`BaseAgent`基类
2. 实现`analyze`和`respond`方法
3. 在配置文件中注册

### 9.2 自定义辩论流程
支持配置不同的辩论流程模板：
- 标准流程（默认）
- 快速流程（只进行一轮）
- 深度流程（增加轮次和强制回应）

### 9.3 多论文对比辩论
支持多篇相关论文的对比辩论：
- 多个论文同时输入
- Agent进行对比分析
- 生成对比报告

## 10. 技术栈

- **语言**: Python 3.10+
- **LLM**: Claude Opus 4.6 / GPT-4
- **PDF解析**: PyPDF2 / pdfplumber
- **数据管理**: Pydantic
- **配置管理**: YAML
- **输出格式**: Markdown

## 11. 部署方案

### 11.1 本地部署
```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥
export ANTHROPIC_API_KEY="your-key"

# 运行
python debate_system.py --paper papers/sample.pdf
```

### 11.2 容器化部署
```dockerfile
FROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "debate_system.py"]
```

---

**文档版本**: 1.0
**最后更新**: 2026-02-16
