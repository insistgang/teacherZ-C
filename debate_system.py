"""
多智能体论文辩论系统
Multi-Agent Paper Debate System

功能：多个Agent协作辩论，深度理解学术论文
作者：系统设计
日期：2026-02-16
"""

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from uuid import uuid4
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import yaml


# =============================================================================
# 数据结构定义
# =============================================================================

class AgentRole(Enum):
    """Agent角色枚举"""
    MATHEMATICIAN = "mathematician"      # 数学家
    ENGINEER = "engineer"                # 工程师
    APPLICATION_EXPERT = "application_expert"  # 应用专家
    SKEPTIC = "skeptic"                  # 质疑者
    SYNTHESIZER = "synthesizer"          # 综合者


class DebateStatus(Enum):
    """辩论状态枚举"""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    AWAITING_CONSENSUS = "awaiting_consensus"
    CONSENSUS = "consensus"
    TERMINATED = "terminated"


@dataclass
class Paper:
    """论文数据结构"""
    title: str
    authors: List[str]
    year: int
    abstract: str
    content: str
    pdf_path: Optional[str] = None
    category: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def get_full_content(self) -> str:
        """获取完整内容用于分析"""
        parts = [f"# {self.title}"]
        parts.append(f"**作者**: {', '.join(self.authors)}")
        parts.append(f"**年份**: {self.year}")
        if self.abstract:
            parts.append(f"\n## 摘要\n{self.abstract}")
        parts.append(f"\n## 正文\n{self.content}")
        return "\n".join(parts)


@dataclass
class AgentMessage:
    """Agent发言消息"""
    agent_role: str
    agent_name: str
    content: str
    timestamp: datetime
    round: int
    message_id: str = field(default_factory=lambda: str(uuid4()))
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent_role": self.agent_role,
            "agent_name": self.agent_name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "round": self.round,
            "message_id": self.message_id,
            "reply_to": self.reply_to,
            "metadata": self.metadata
        }


@dataclass
class DebateState:
    """辩论状态"""
    paper: Paper
    messages: List[AgentMessage]
    current_round: int
    status: DebateStatus
    consensus_score: float
    unresolved_issues: List[str]
    max_rounds: int = 5
    start_time: datetime = field(default_factory=datetime.now)

    def add_message(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def get_messages_by_role(self, role: str) -> List[AgentMessage]:
        return [m for m in self.messages if m.agent_role == role]

    def get_messages_by_round(self, round_num: int) -> List[AgentMessage]:
        return [m for m in self.messages if m.round == round_num]

    def to_dict(self) -> Dict:
        return {
            "paper_title": self.paper.title,
            "current_round": self.current_round,
            "status": self.status.value,
            "consensus_score": self.consensus_score,
            "unresolved_issues": self.unresolved_issues,
            "message_count": len(self.messages),
            "start_time": self.start_time.isoformat()
        }


@dataclass
class DebateReport:
    """辩论报告"""
    paper_title: str
    debate_summary: str
    consensus_points: List[str]
    disagreements: List[str]
    key_insights: Dict[str, List[str]]
    final_assessment: str
    recommendations: List[str]
    consensus_score: float
    total_rounds: int
    timestamp: datetime = field(default_factory=datetime.now)
    debate_statistics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 配置管理
# =============================================================================

@dataclass
class AgentConfig:
    """Agent配置"""
    enabled: bool = True
    model: str = "claude-opus-4"
    temperature: float = 0.5
    max_tokens: int = 4000


@dataclass
class DebateConfig:
    """辩论配置"""
    max_rounds: int = 5
    consensus_threshold: float = 0.7
    timeout_seconds: int = 300
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    output_dir: str = "./outputs"


def load_config(config_path: str = "config.yaml") -> DebateConfig:
    """加载配置文件"""
    default_config = DebateConfig()

    # 默认Agent配置
    default_agents = {
        "mathematician": AgentConfig(
            enabled=True,
            model="claude-opus-4",
            temperature=0.3
        ),
        "engineer": AgentConfig(
            enabled=True,
            model="claude-opus-4",
            temperature=0.5
        ),
        "application_expert": AgentConfig(
            enabled=True,
            model="claude-opus-4",
            temperature=0.5
        ),
        "skeptic": AgentConfig(
            enabled=True,
            model="claude-opus-4",
            temperature=0.7
        ),
        "synthesizer": AgentConfig(
            enabled=True,
            model="claude-opus-4",
            temperature=0.4
        )
    }

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data:
                # 解析配置
                if 'debate' in data:
                    default_config.max_rounds = data['debate'].get('max_rounds', 5)
                    default_config.consensus_threshold = data['debate'].get('consensus_threshold', 0.7)
                    default_config.timeout_seconds = data['debate'].get('timeout_seconds', 300)

                if 'agents' in data:
                    for role, agent_data in data['agents'].items():
                        if role in default_agents:
                            default_agents[role].enabled = agent_data.get('enabled', True)
                            default_agents[role].model = agent_data.get('model', 'claude-opus-4')
                            default_agents[role].temperature = agent_data.get('temperature', 0.5)

    default_config.agents = default_agents
    return default_config


# =============================================================================
# LLM接口
# =============================================================================

class LLMClient(ABC):
    """LLM客户端基类"""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = "",
                      temperature: float = 0.5, max_tokens: int = 4000) -> str:
        """生成回复"""
        pass


class MockLLMClient(LLMClient):
    """模拟LLM客户端（用于测试）"""

    def __init__(self):
        self.call_count = 0

    async def generate(self, prompt: str, system_prompt: str = "",
                      temperature: float = 0.5, max_tokens: int = 4000) -> str:
        self.call_count += 1
        role = system_prompt.split("你是")[1].split("，")[0] if "你是" in system_prompt else "Agent"

        return f"""【{role}的分析】

基于对论文的深入分析，我认为：

1. **核心观点**：论文提出的方法在理论上是合理的，但在实际应用中需要考虑更多边界情况。

2. **主要问题**：
   - 缺乏对某些关键参数的理论分析
   - 实验数据集的多样性有待提高

3. **改进建议**：
   - 建议增加更多理论证明
   - 在更多数据集上进行验证

（这是模拟回复 - 第{self.call_count}次调用）"""


class ClaudeLLMClient(LLMClient):
    """Claude LLM客户端"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("请设置ANTHROPIC_API_KEY环境变量")

    async def generate(self, prompt: str, system_prompt: str = "",
                      temperature: float = 0.5, max_tokens: int = 4000) -> str:
        """调用Claude API生成回复"""
        try:
            # 这里使用import anthropic的方式
            # 实际部署时需要安装 anthropic 包
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.content[0].text

        except ImportError:
            raise ImportError("请先安装 anthropic 包: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Claude API调用失败: {e}")


# =============================================================================
# Agent基类和具体实现
# =============================================================================

class BaseAgent(ABC):
    """Agent基类"""

    # 角色提示词模板
    SYSTEM_PROMPTS = {
        AgentRole.MATHEMATICIAN: """你是数学家Agent，专门从数学理论角度分析学术论文。

你的关注焦点：
1. 公式推导的正确性和完整性
2. 理论假设的合理性
3. 数学证明的严谨性
4. 算法收敛性的理论保证
5. 数学符号的一致性和清晰度

请以严谨的数学态度进行分析，指出任何理论上的问题。""",

        AgentRole.ENGINEER: """你是工程师Agent，专门从工程实现角度分析学术论文。

你的关注焦点：
1. 算法描述是否足够清晰以实现
2. 时间/空间复杂度分析
3. 数值计算的稳定性
4. 超参数敏感性
5. 代码复现的可行性

请以工程实现的角度，评估论文的可落地性。""",

        AgentRole.APPLICATION_EXPERT: """你是应用专家Agent，专门从实际应用角度分析学术论文。

你的关注焦点：
1. 该方法适用的实际场景
2. 数据需求与获取难度
3. 与现有SOTA方法的对比
4. 行业落地的价值和障碍
5. 商业化潜力评估

请从实际应用的角度，评估论文的实用价值。""",

        AgentRole.SKEPTIC: """你是质疑者Agent，专门批判性地分析学术论文。

你的关注焦点：
1. 实验设置的公平性（有无cherry-picking）
2. 结果的统计显著性
3. 论文声称是否被实验充分支持
4. 被忽略的反例或边缘情况
5. 任何过度声明或不实之处

请以批判的眼光，找出论文中的任何问题或夸大之处。""",

        AgentRole.SYNTHESIZER: """你是综合者Agent，负责汇总各方观点并生成最终报告。

你的任务：
1. 汇总所有Agent的观点并分类整理
2. 识别各角色之间的共识点
3. 识别分歧点，判断是否需要进一步辩论
4. 评估辩论是否充分
5. 生成最终理解报告，包含：
   - 论文核心贡献总结
   - 主要优势
   - 主要局限/问题
   - 改进方向
   - 应用建议

请综合各方观点，生成一份平衡、全面的分析报告。"""
    }

    def __init__(self, role: AgentRole, name: str,
                 llm_client: LLMClient,
                 config: AgentConfig = None):
        self.role = role
        self.name = name
        self.llm_client = llm_client
        self.config = config or AgentConfig()
        self.system_prompt = self.SYSTEM_PROMPTS.get(role, "")
        self.memory: List[AgentMessage] = []

    async def analyze(self, paper: Paper, context: List[AgentMessage] = None) -> str:
        """分析论文并生成发言"""
        context = context or []

        # 构建分析提示
        prompt = self._build_analysis_prompt(paper, context)

        # 调用LLM
        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response

    async def respond(self, to_message: AgentMessage, context: List[AgentMessage]) -> str:
        """回应其他Agent的发言"""
        # 构建回应提示
        prompt = self._build_response_prompt(to_message, context)

        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return response

    def _build_analysis_prompt(self, paper: Paper, context: List[AgentMessage]) -> str:
        """构建分析提示词"""
        prompt_parts = [
            f"请从{self._get_role_name()}的角度分析以下论文：\n",
            paper.get_full_content(),
            "\n" + "="*50 + "\n"
        ]

        if context:
            prompt_parts.append("\n之前的讨论记录：\n")
            for msg in context:
                prompt_parts.append(f"**{msg.agent_name}**: {msg.content}\n")

        prompt_parts.append(f"\n现在请以{self._get_role_name()}的身份发表你的分析：")

        return "".join(prompt_parts)

    def _build_response_prompt(self, to_message: AgentMessage, context: List[AgentMessage]) -> str:
        """构建回应提示词"""
        prompt_parts = [
            f"{to_message.agent_name}发表了以下观点，请你作为{self._get_role_name()}进行回应：\n",
            to_message.content,
            "\n" + "="*50 + "\n"
        ]

        if context:
            prompt_parts.append("\n相关背景讨论：\n")
            for msg in context[-5:]:  # 最近5条
                if msg.message_id != to_message.message_id:
                    prompt_parts.append(f"**{msg.agent_name}**: {msg.content}\n")

        prompt_parts.append(f"\n请针对上述观点进行回应：")

        return "".join(prompt_parts)

    def _get_role_name(self) -> str:
        """获取角色中文名"""
        role_names = {
            AgentRole.MATHEMATICIAN: "数学家",
            AgentRole.ENGINEER: "工程师",
            AgentRole.APPLICATION_EXPERT: "应用专家",
            AgentRole.SKEPTIC: "质疑者",
            AgentRole.SYNTHESIZER: "综合者"
        }
        return role_names.get(self.role, "Agent")


class MathematicianAgent(BaseAgent):
    """数学家Agent"""

    def __init__(self, llm_client: LLMClient, config: AgentConfig = None):
        super().__init__(AgentRole.MATHEMATICIAN, "数学家", llm_client, config)


class EngineerAgent(BaseAgent):
    """工程师Agent"""

    def __init__(self, llm_client: LLMClient, config: AgentConfig = None):
        super().__init__(AgentRole.ENGINEER, "工程师", llm_client, config)


class ApplicationExpertAgent(BaseAgent):
    """应用专家Agent"""

    def __init__(self, llm_client: LLMClient, config: AgentConfig = None):
        super().__init__(AgentRole.APPLICATION_EXPERT, "应用专家", llm_client, config)


class SkepticAgent(BaseAgent):
    """质疑者Agent"""

    def __init__(self, llm_client: LLMClient, config: AgentConfig = None):
        super().__init__(AgentRole.SKEPTIC, "质疑者", llm_client, config)


class SynthesizerAgent(BaseAgent):
    """综合者Agent"""

    def __init__(self, llm_client: LLMClient, config: AgentConfig = None):
        super().__init__(AgentRole.SYNTHESIZER, "综合者", llm_client, config)

    async def generate_report(self, state: DebateState) -> DebateReport:
        """生成最终报告"""
        # 构建报告生成提示
        prompt = self._build_report_prompt(state)

        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.config.temperature,
            max_tokens=6000
        )

        # 解析响应生成报告
        return self._parse_report_response(state, response)

    def _build_report_prompt(self, state: DebateState) -> str:
        """构建报告生成提示"""
        prompt_parts = [
            f"辩论已完成，请基于以下辩论记录生成最终报告：\n",
            f"论文标题: {state.paper.title}\n",
            f"辩论轮次: {state.current_round}\n",
            f"共识度: {state.consensus_score:.2f}\n",
            "\n辩论记录：\n"
        ]

        for msg in state.messages:
            prompt_parts.append(
                f"[第{msg.round}轮] {msg.agent_name}: {msg.content}\n"
            )

        prompt_parts.append("\n请生成结构化的最终报告。")

        return "".join(prompt_parts)

    def _parse_report_response(self, state: DebateState, response: str) -> DebateReport:
        """解析LLM响应生成报告对象"""
        # 这里做简单解析，实际可以更复杂
        return DebateReport(
            paper_title=state.paper.title,
            debate_summary=response[:500],
            consensus_points=[],
            disagreements=[],
            key_insights={},
            final_assessment=response,
            recommendations=[],
            consensus_score=state.consensus_score,
            total_rounds=state.current_round,
            debate_statistics={
                "total_messages": len(state.messages),
                "messages_by_role": {
                    role.value: len(state.get_messages_by_role(role.value))
                    for role in AgentRole
                }
            }
        )


# =============================================================================
# Agent工厂
# =============================================================================

class AgentFactory:
    """Agent工厂"""

    _agent_classes = {
        AgentRole.MATHEMATICIAN: MathematicianAgent,
        AgentRole.ENGINEER: EngineerAgent,
        AgentRole.APPLICATION_EXPERT: ApplicationExpertAgent,
        AgentRole.SKEPTIC: SkepticAgent,
        AgentRole.SYNTHESIZER: SynthesizerAgent
    }

    @classmethod
    def create_agent(cls, role: AgentRole, llm_client: LLMClient,
                     config: AgentConfig = None) -> BaseAgent:
        """创建Agent实例"""
        agent_class = cls._agent_classes.get(role)
        if not agent_class:
            raise ValueError(f"未知的Agent角色: {role}")
        return agent_class(llm_client, config)

    @classmethod
    def create_all_agents(cls, llm_client: LLMClient,
                         config: DebateConfig) -> List[BaseAgent]:
        """创建所有启用的Agent"""
        agents = []
        for role, agent_config in config.agents.items():
            if agent_config.enabled:
                agent = cls.create_agent(
                    AgentRole(role),
                    llm_client,
                    agent_config
                )
                agents.append(agent)
        return agents


# =============================================================================
# 辩论调度器
# =============================================================================

class DebateScheduler:
    """辩论调度器"""

    def __init__(self, agents: List[BaseAgent], config: DebateConfig,
                 llm_client: LLMClient = None):
        self.agents = [a for a in agents if not isinstance(a, SynthesizerAgent)]
        self.synthesizer = next((a for a in agents if isinstance(a, SynthesizerAgent)), None)
        self.config = config
        self.llm_client = llm_client
        self.state: Optional[DebateState] = None
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def start_debate(self, paper: Paper) -> DebateState:
        """启动辩论"""
        # 初始化状态
        self.state = DebateState(
            paper=paper,
            messages=[],
            current_round=0,
            status=DebateStatus.INITIALIZED,
            consensus_score=0.0,
            unresolved_issues=[],
            max_rounds=self.config.max_rounds
        )

        print(f"\n{'='*60}")
        print(f"开始辩论：{paper.title}")
        print(f"参与Agent: {', '.join([a.name for a in self.agents])}")
        print(f"最大轮次: {self.config.max_rounds}")
        print(f"{'='*60}\n")

        # 第一轮：独立分析发言
        await self._conduct_first_round()

        # 交互辩论轮次
        while self._should_continue():
            await self._conduct_interaction_round()

        # 综合者生成报告
        await self._generate_final_report()

        return self.state

    async def _conduct_first_round(self) -> None:
        """进行第一轮独立分析发言"""
        self.state.current_round = 1
        self.state.status = DebateStatus.IN_PROGRESS

        print(f"\n{'='*60}")
        print(f"第1轮：独立分析发言")
        print(f"{'='*60}\n")

        for agent in self.agents:
            print(f"[{agent.name}] 正在分析...")
            try:
                content = await agent.analyze(self.state.paper)
                message = AgentMessage(
                    agent_role=agent.role.value,
                    agent_name=agent.name,
                    content=content,
                    timestamp=datetime.now(),
                    round=1
                )
                self.state.add_message(message)
                print(f"[{agent.name}] 分析完成\n")

                # 显示发言摘要
                self._print_message_summary(message)

            except Exception as e:
                print(f"[{agent.name}] 分析失败: {e}\n")

    async def _conduct_interaction_round(self) -> None:
        """进行交互辩论轮次"""
        self.state.current_round += 1
        print(f"\n{'='*60}")
        print(f"第{self.state.current_round}轮：交互辩论")
        print(f"{'='*60}\n")

        # 找出需要回应的消息（来自质疑者的，或上一轮有争议的）
        messages_to_respond = self._get_messages_requiring_response()

        for message in messages_to_respond:
            # 找到应该回应的Agent
            target_agent = self._find_agent_to_respond(message)
            if target_agent:
                print(f"[{target_agent.name}] 正在回应 {message.agent_name}...")
                try:
                    content = await target_agent.respond(
                        message,
                        self.state.messages
                    )
                    reply = AgentMessage(
                        agent_role=target_agent.role.value,
                        agent_name=target_agent.name,
                        content=content,
                        timestamp=datetime.now(),
                        round=self.state.current_round,
                        reply_to=message.message_id
                    )
                    self.state.add_message(reply)
                    print(f"[{target_agent.name}] 回应完成\n")
                    self._print_message_summary(reply)

                except Exception as e:
                    print(f"[{target_agent.name}] 回应失败: {e}\n")

        # 允许其他Agent补充发言
        for agent in self.agents:
            if self._should_agent_speak(agent, self.state.current_round):
                print(f"[{agent.name}] 正在补充发言...")
                try:
                    content = await agent.analyze(
                        self.state.paper,
                        self.state.messages[-5:]
                    )
                    message = AgentMessage(
                        agent_role=agent.role.value,
                        agent_name=agent.name,
                        content=content,
                        timestamp=datetime.now(),
                        round=self.state.current_round
                    )
                    self.state.add_message(message)
                    print(f"[{agent.name}] 发言完成\n")
                    self._print_message_summary(message)

                except Exception as e:
                    print(f"[{agent.name}] 发言失败: {e}\n")

    async def _generate_final_report(self) -> None:
        """生成最终报告"""
        print(f"\n{'='*60}")
        print("生成最终报告")
        print(f"{'='*60}\n")

        self.state.status = DebateStatus.AWAITING_CONSENSUS

        # 计算共识度
        self.state.consensus_score = self._calculate_consensus_score()

        if self.synthesizer:
            try:
                report = await self.synthesizer.generate_report(self.state)

                # 保存报告
                self._save_outputs(report)

                self.state.status = DebateStatus.CONSENSUS

                print(f"辩论完成！共识度: {self.state.consensus_score:.2f}")
                print(f"报告已保存到: {self.output_dir}")

            except Exception as e:
                print(f"报告生成失败: {e}")
                self.state.status = DebateStatus.TERMINATED
        else:
            print("警告: 没有综合者Agent，跳过报告生成")
            self.state.status = DebateStatus.TERMINATED

    def _get_messages_requiring_response(self) -> List[AgentMessage]:
        """获取需要回应的消息"""
        if self.state.current_round == 2:
            # 第二轮主要回应质疑者
            return self.state.get_messages_by_role(AgentRole.SKEPTIC.value)

        # 后续轮次回应上一轮的发言
        return [
            m for m in self.state.get_messages_by_round(self.state.current_round - 1)
            if m.reply_to is None  # 只回应原始发言，不回应回应
        ]

    def _find_agent_to_respond(self, message: AgentMessage) -> Optional[BaseAgent]:
        """找到应该回应消息的Agent"""
        # 质疑者的消息应该被所有相关Agent回应
        if message.agent_role == AgentRole.SKEPTIC.value:
            # 找一个还没回应过的Agent
            for agent in self.agents:
                if not any(m.reply_to == message.message_id
                          for m in self.state.messages):
                    return agent
        else:
            # 其他消息找其他角色的Agent
            for agent in self.agents:
                if agent.role.value != message.agent_role:
                    return agent
        return None

    def _should_agent_speak(self, agent: BaseAgent, round_num: int) -> bool:
        """判断Agent是否应该发言"""
        # 检查本轮是否已经发言
        already_spoken = any(
            m.agent_role == agent.role.value and m.round == round_num
            for m in self.state.messages
        )
        return not already_spoken and agent.role != AgentRole.SKEPTIC

    def _should_continue(self) -> bool:
        """判断是否继续辩论"""
        if self.state.current_round >= self.state.max_rounds:
            return False

        # 检查是否有新问题
        if self.state.current_round >= 2:
            recent_rounds = [
                self.state.get_messages_by_round(r)
                for r in [self.state.current_round, self.state.current_round - 1]
            ]
            if sum(len(r) for r in recent_rounds) < 3:
                return False

        return True

    def _calculate_consensus_score(self) -> float:
        """计算共识度分数"""
        # 简化版计算：基于质疑者发言数量和被回应数量
        skeptic_msgs = self.state.get_messages_by_role(AgentRole.SKEPTIC.value)
        if not skeptic_msgs:
            return 1.0

        responded_count = sum(
            1 for msg in skeptic_msgs
            if any(m.reply_to == msg.message_id for m in self.state.messages)
        )

        base_score = responded_count / len(skeptic_msgs) if skeptic_msgs else 1.0

        # 考虑轮次：轮次越多，分数越低（表示争议较多）
        round_penalty = (self.state.current_round - 1) * 0.05

        return max(0.0, min(1.0, base_score - round_penalty))

    def _print_message_summary(self, message: AgentMessage) -> None:
        """打印发言摘要"""
        content_preview = message.content[:200]
        if len(message.content) > 200:
            content_preview += "..."
        print(f"  发言摘要: {content_preview}\n")

    def _save_outputs(self, report: DebateReport) -> None:
        """保存输出文件"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[^\w\s-]', '', self.state.paper.title)[:30]

        # 保存辩论记录
        debate_log_path = self.output_dir / f"debate_log_{safe_title}_{timestamp_str}.md"
        with open(debate_log_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_debate_log_markdown())

        # 保存报告
        report_path = self.output_dir / f"report_{safe_title}_{timestamp_str}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_report_markdown(report))

        # 保存JSON数据
        json_path = self.output_dir / f"debate_data_{safe_title}_{timestamp_str}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "state": self.state.to_dict(),
                "messages": [m.to_dict() for m in self.state.messages],
                "report": {
                    "paper_title": report.paper_title,
                    "consensus_score": report.consensus_score,
                    "total_rounds": report.total_rounds,
                    "debate_statistics": report.debate_statistics
                }
            }, f, ensure_ascii=False, indent=2)

    def _generate_debate_log_markdown(self) -> str:
        """生成辩论记录Markdown"""
        lines = [
            f"# 论文辩论记录：{self.state.paper.title}\n",
            f"**辩论时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"**参与Agent**: {', '.join([a.name for a in self.agents])}\n",
            f"**总轮次**: {self.state.current_round}\n",
            f"**共识度**: {self.state.consensus_score:.2f}\n",
            "\n" + "="*60 + "\n"
        ]

        for round_num in range(1, self.state.current_round + 1):
            round_messages = self.state.get_messages_by_round(round_num)
            if not round_messages:
                continue

            round_name = "独立分析发言" if round_num == 1 else "交互辩论"
            lines.append(f"\n## 第{round_num}轮：{round_name}\n")

            for msg in round_messages:
                lines.append(f"\n### {msg.agent_name}发言\n")
                lines.append(f"{msg.content}\n")

        return "\n".join(lines)

    def _generate_report_markdown(self, report: DebateReport) -> str:
        """生成报告Markdown"""
        lines = [
            f"# 论文深度理解报告：{report.paper_title}\n",
            f"**生成时间**: {report.timestamp.strftime('%Y-%m-%d %H:%M')}\n",
            f"**共识度评分**: {'⭐'*int(report.consensus_score*5)} ({report.consensus_score:.2f}/1.0)\n",
            f"**辩论轮次**: {report.total_rounds}\n",
            "\n" + "="*60 + "\n",
            "\n## 1. 辩论总结\n",
            report.debate_summary,
            "\n## 2. 最终评估\n",
            report.final_assessment,
            "\n## 3. 辩论统计\n",
            f"- 总发言数: {report.debate_statistics.get('total_messages', 0)}\n"
        ]

        if report.debate_statistics.get('messages_by_role'):
            lines.append("\n### 各角色发言数\n")
            lines.append("| 角色 | 发言数 |\n|------|--------|\n")
            for role, count in report.debate_statistics['messages_by_role'].items():
                role_names = {
                    'mathematician': '数学家',
                    'engineer': '工程师',
                    'application_expert': '应用专家',
                    'skeptic': '质疑者',
                    'synthesizer': '综合者'
                }
                lines.append(f"| {role_names.get(role, role)} | {count} |\n")

        lines.append("\n---\n*本报告由多智能体辩论系统自动生成*\n")

        return "\n".join(lines)


# =============================================================================
# 论文解析器
# =============================================================================

class PaperParser:
    """论文解析器"""

    @staticmethod
    def parse_from_text(text: str, metadata: Dict = None) -> Paper:
        """从文本解析论文"""
        metadata = metadata or {}

        # 简单解析：假设文本包含标题、作者等信息
        lines = text.split('\n')

        title = metadata.get('title', lines[0] if lines else 'Untitled')
        authors = metadata.get('authors', [])
        year = metadata.get('year', 2024)
        abstract = metadata.get('abstract', '')
        content = text
        category = metadata.get('category')

        return Paper(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            content=content,
            category=category
        )

    @staticmethod
    def parse_from_pdf(pdf_path: str) -> Paper:
        """从PDF文件解析论文"""
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")

                full_text = "\n".join(text_parts)

                # 尝试提取元信息
                title = os.path.basename(pdf_path).replace('.pdf', '')
                authors = []
                year = 2024
                abstract = ""

                return Paper(
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    content=full_text,
                    pdf_path=pdf_path
                )

        except ImportError:
            raise ImportError("请安装 pdfplumber: pip install pdfplumber")
        except Exception as e:
            raise RuntimeError(f"PDF解析失败: {e}")

    @staticmethod
    def parse_from_markdown(md_path: str) -> Paper:
        """从Markdown文件解析论文"""
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析YAML front matter
        metadata = {}
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 2:
                try:
                    metadata = yaml.safe_load(parts[1])
                except:
                    pass
                content = parts[2] if len(parts) > 2 else ""

        return PaperParser.parse_from_text(content, metadata)


# =============================================================================
# 主系统
# =============================================================================

class DebateSystem:
    """多智能体论文辩论系统"""

    def __init__(self, config_path: str = "config.yaml",
                 llm_client: LLMClient = None):
        self.config = load_config(config_path)
        self.llm_client = llm_client or MockLLMClient()
        self.scheduler: Optional[DebateScheduler] = None

    def initialize(self) -> None:
        """初始化系统"""
        # 创建所有Agent
        agents = AgentFactory.create_all_agents(
            self.llm_client,
            self.config
        )

        # 创建调度器
        self.scheduler = DebateScheduler(
            agents=agents,
            config=self.config,
            llm_client=self.llm_client
        )

    async def debate_paper(self, paper: Paper) -> DebateState:
        """对论文进行辩论"""
        if not self.scheduler:
            self.initialize()

        return await self.scheduler.start_debate(paper)

    async def debate_from_file(self, file_path: str) -> DebateState:
        """从文件加载论文并进行辩论"""
        # 解析论文
        if file_path.endswith('.pdf'):
            paper = PaperParser.parse_from_pdf(file_path)
        elif file_path.endswith('.md'):
            paper = PaperParser.parse_from_markdown(file_path)
        else:
            paper = PaperParser.parse_from_text(
                Path(file_path).read_text(encoding='utf-8')
            )

        return await self.debate_paper(paper)

    def get_status(self) -> Dict:
        """获取系统状态"""
        if not self.scheduler or not self.scheduler.state:
            return {"status": "not_initialized"}

        return {
            "status": self.scheduler.state.status.value,
            "current_round": self.scheduler.state.current_round,
            "total_messages": len(self.scheduler.state.messages),
            "consensus_score": self.scheduler.state.consensus_score
        }


# =============================================================================
# 命令行接口
# =============================================================================

import argparse


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="多智能体论文辩论系统"
    )
    parser.add_argument(
        "--paper",
        type=str,
        help="论文文件路径 (PDF/Markdown/TXT)"
    )
    parser.add_argument(
        "--title",
        type=str,
        help="论文标题"
    )
    parser.add_argument(
        "--content",
        type=str,
        help="论文内容（直接输入）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用模拟LLM（用于测试）"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="最大辩论轮次"
    )

    args = parser.parse_args()

    # 创建LLM客户端
    if args.mock:
        llm_client = MockLLMClient()
        print("使用模拟LLM客户端\n")
    else:
        try:
            llm_client = ClaudeLLMClient()
            print("使用Claude LLM客户端\n")
        except ValueError as e:
            print(f"错误: {e}")
            print("使用 --mock 选项可使用模拟LLM进行测试")
            return

    # 加载配置
    config = load_config(args.config)
    config.max_rounds = args.max_rounds

    # 创建系统
    system = DebateSystem(config_path=args.config, llm_client=llm_client)
    system.config = config

    # 解析论文
    if args.paper:
        paper = PaperParser.parse_from_file(args.paper)
    elif args.title and args.content:
        paper = Paper(
            title=args.title,
            authors=[],
            year=2024,
            abstract="",
            content=args.content
        )
    else:
        # 使用示例论文
        paper = Paper(
            title="示例论文：基于深度学习的图像分割方法",
            authors=["张三", "李四"],
            year=2024,
            abstract="本文提出了一种新的图像分割方法...",
            content="""
# 引言

图像分割是计算机视觉的基础任务之一。本文提出了一种基于深度学习的新方法。

# 方法

我们的方法包含以下步骤：
1. 特征提取：使用ResNet-50作为骨干网络
2. 特征融合：采用FPN结构进行多尺度特征融合
3. 分割输出：使用卷积层生成分割掩码

数学公式如下：
L = L_ce + λL_dice

其中L_ce是交叉熵损失，L_dice是Dice损失，λ是权重参数。

# 实验

我们在COCO数据集上进行了实验，达到了45.2%的mIoU。

# 结论

本文提出的方法在图像分割任务上取得了良好的效果。
"""
        )
        print("使用示例论文进行辩论\n")

    # 启动辩论
    await system.debate_paper(paper)

    # 打印状态
    status = system.get_status()
    print(f"\n最终状态: {status}")


if __name__ == "__main__":
    asyncio.run(main())
