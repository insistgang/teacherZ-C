"""
高级多智能体论文辩论系统

新增功能：
1. 支持流式输出（实时显示Agent思考过程）
2. 支持Agent间直接对话（消息路由）
3. 支持中途干预（用户可向某个Agent提问）
4. 支持辩论暂停/恢复
5. 更详细的共识分析
"""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from debate_system import (
    AgentRole,
    Paper,
    AgentMessage,
    DebateStatus,
    DebateState,
    LLMClient,
    MockLLMClient,
    BaseAgent
)


class MessageType(Enum):
    """消息类型"""
    ANALYSIS = "analysis"      # 分析发言
    QUESTION = "question"      # 提问
    RESPONSE = "response"      # 回应
    COMMENT = "comment"        # 评论
    INTERRUPTION = "interruption"  # 用户干预


@dataclass
class AdvancedMessage(AgentMessage):
    """高级消息"""
    message_type: MessageType = MessageType.ANALYSIS
    mentioned_agents: Set[str] = field(default_factory=set)  # @提到的Agent
    parent_id: Optional[str] = None  # 父消息ID（用于构建对话树）
    reactions: Dict[str, str] = field(default_factory=dict)  # 其他Agent的反应
    editing: bool = False  # 是否正在编辑中


@dataclass
class ConversationTree:
    """对话树结构"""
    root_messages: List[str] = field(default_factory=list)  # 根消息ID
    parent_map: Dict[str, Optional[str]] = field(default_factory=dict)  # 父子关系
    children_map: Dict[str, List[str]] = field(default_factory=dict)  # 子女列表

    def add_message(self, message_id: str, parent_id: Optional[str] = None) -> None:
        """添加消息到树中"""
        if parent_id is None:
            self.root_messages.append(message_id)
        else:
            self.parent_map[message_id] = parent_id
            if parent_id not in self.children_map:
                self.children_map[parent_id] = []
            self.children_map[parent_id].append(message_id)

    def get_thread(self, message_id: str) -> List[str]:
        """获取消息所在的对话线程"""
        thread = []
        current = message_id
        while current in self.parent_map or current in self.root_messages:
            thread.insert(0, current)
            current = self.parent_map.get(current)
            if current is None:
                break
        return thread


@dataclass
class Intervention:
    """用户干预"""
    intervention_id: str
    target_agent: str
    question: str
    timestamp: datetime
    response: Optional[str] = None


class AdvancedDebateState(DebateState):
    """高级辩论状态"""
    conversation_tree: ConversationTree = field(default_factory=ConversationTree)
    interventions: List[Intervention] = field(default_factory=list)
    current_focus: Optional[str] = None  # 当前讨论焦点
    agent_relationships: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Agent间关系
    paused: bool = False
    pause_reason: Optional[str] = None


class MessageRouter:
    """消息路由器"""

    def __init__(self):
        self.routing_rules: Dict[str, Callable[[AdvancedMessage], List[str]]] = {}

    def add_rule(self, name: str, rule: Callable[[AdvancedMessage], List[str]]) -> None:
        """添加路由规则"""
        self.routing_rules[name] = rule

    def route(self, message: AdvancedMessage, available_agents: List[str]) -> List[str]:
        """确定哪些Agent应该回应此消息"""
        responders = set()

        # 检查是否被@提到
        if message.mentioned_agents:
            responders.update(message.mentioned_agents)

        # 应用路由规则
        for rule_name, rule_func in self.routing_rules.items():
            try:
                rule_responses = rule_func(message)
                responders.update(rule_responses)
            except Exception as e:
                print(f"路由规则 {rule_name} 执行失败: {e}")

        # 确保回应者在可用Agent列表中
        responders = responders.intersection(available_agents)

        # 排除消息发送者
        if message.agent_role in responders:
            responders.remove(message.agent_role)

        return list(responders)


class StreamingCallback(ABC):
    """流式输出回调"""

    @abstractmethod
    async def on_start(self, agent: str, message_id: str) -> None:
        """开始生成"""
        pass

    @abstractmethod
    async def on_token(self, token: str) -> None:
        """生成token"""
        pass

    @abstractmethod
    async def on_complete(self, message: str) -> None:
        """生成完成"""
        pass


class ConsoleStreamingCallback(StreamingCallback):
    """控制台流式输出"""

    async def on_start(self, agent: str, message_id: str) -> None:
        print(f"[{agent}] 正在思考...", end="", flush=True)

    async def on_token(self, token: str) -> None:
        print(".", end="", flush=True)

    async def on_complete(self, message: str) -> None:
        print(" ✓")


class AdvancedDebateScheduler:
    """高级辩论调度器"""

    def __init__(self, agents: List[BaseAgent], config,
                 llm_client: LLMClient = None):
        self.agents = [a for a in agents if not isinstance(a, SynthesizerAgent)]
        self.synthesizer = next((a for a in agents if isinstance(a, SynthesizerAgent)), None)
        self.config = config
        self.llm_client = llm_client
        self.state: Optional[AdvancedDebateState] = None
        self.router = MessageRouter()
        self.streaming_callbacks: List[StreamingCallback] = []
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置默认路由规则
        self._setup_default_routes()

    def _setup_default_routes(self) -> None:
        """设置默认路由规则"""

        # 质疑者的消息应该被所有相关Agent回应
        def skeptic_route(msg: AdvancedMessage) -> List[str]:
            if msg.agent_role == AgentRole.SKEPTIC.value:
                return [AgentRole.MATHEMATICIAN.value,
                       AgentRole.ENGINEER.value,
                       AgentRole.APPLICATION_EXPERT.value]
            return []

        self.router.add_rule("skeptic_response", skeptic_route)

        # 数学问题由数学家回应
        def math_question_route(msg: AdvancedMessage) -> List[str]:
            math_keywords = ["证明", "公式", "推导", "收敛", "理论"]
            if any(kw in msg.content for kw in math_keywords):
                return [AgentRole.MATHEMATICIAN.value]
            return []

        self.router.add_rule("math_questions", math_question_route)

        # 实现问题由工程师回应
        def engineering_route(msg: AdvancedMessage) -> List[str]:
            eng_keywords = ["实现", "代码", "算法", "复杂度", "性能"]
            if any(kw in msg.content for kw in eng_keywords):
                return [AgentRole.ENGINEER.value]
            return []

        self.router.add_rule("engineering_questions", engineering_route)

    def add_streaming_callback(self, callback: StreamingCallback) -> None:
        """添加流式输出回调"""
        self.streaming_callbacks.append(callback)

    async def start_debate(self, paper: Paper) -> AdvancedDebateState:
        """启动辩论"""
        self.state = AdvancedDebateState(
            paper=paper,
            messages=[],
            current_round=0,
            status=DebateStatus.INITIALIZED,
            consensus_score=0.0,
            unresolved_issues=[],
            max_rounds=self.config.max_rounds
        )

        print(f"\n{'='*60}")
        print(f"高级辩论模式：{paper.title}")
        print(f"支持功能: 流式输出、消息路由、用户干预")
        print(f"{'='*60}\n")

        # 第一轮：独立分析
        await self._conduct_first_round()

        # 交互辩论轮次
        while self._should_continue():
            await self._conduct_interaction_round()

            # 检查是否暂停
            if self.state.paused:
                print(f"\n辩论暂停: {self.state.pause_reason}")
                print("等待用户干预或继续...")
                break

        # 生成报告
        if not self.state.paused:
            await self._generate_final_report()

        return self.state

    async def _conduct_first_round(self) -> None:
        """第一轮独立分析"""
        self.state.current_round = 1
        self.state.status = DebateStatus.IN_PROGRESS

        print(f"\n{'='*60}")
        print(f"第1轮：独立分析发言（流式输出）")
        print(f"{'='*60}\n")

        for agent in self.agents:
            message_id = str(uuid4())

            # 触发开始回调
            for callback in self.streaming_callbacks:
                await callback.on_start(agent.name, message_id)

            # 模拟流式输出
            content = await self._streaming_analyze(agent, message_id)

            message = AdvancedMessage(
                agent_role=agent.role.value,
                agent_name=agent.name,
                content=content,
                timestamp=datetime.now(),
                round=1,
                message_id=message_id,
                message_type=MessageType.ANALYSIS
            )

            self.state.add_message(message)
            self.state.conversation_tree.add_message(message_id)

            # 触发完成回调
            for callback in self.streaming_callbacks:
                await callback.on_complete(content)

    async def _streaming_analyze(self, agent: BaseAgent, message_id: str) -> str:
        """流式分析（模拟）"""
        # 实际实现中应该调用支持流式的LLM API
        for callback in self.streaming_callbacks:
            await callback.on_token("")

        # 获取分析内容
        content = await agent.analyze(self.state.paper, self.state.messages)

        # 模拟token-by-token输出
        words = content.split()
        displayed = ""
        for word in words:
            displayed += word + " "
            await asyncio.sleep(0.05)  # 模拟网络延迟

        return content

    async def _conduct_interaction_round(self) -> None:
        """交互辩论轮次（带消息路由）"""
        self.state.current_round += 1

        print(f"\n{'='*60}")
        print(f"第{self.state.current_round}轮：智能路由辩论")
        print(f"{'='*60}\n")

        # 获取需要回应的消息
        messages_to_respond = self._get_pending_messages()

        responded_agents = set()

        for message in messages_to_respond:
            # 使用路由器确定应该回应的Agent
            adv_message = self._to_advanced_message(message)
            responders = self.router.route(
                adv_message,
                [a.role.value for a in self.agents]
            )

            for responder_role in responders:
                if responder_role in responded_agents:
                    continue  # 每个Agent每轮只发言一次

                agent = self._get_agent_by_role(responder_role)
                if agent:
                    await self._agent_respond(agent, message)
                    responded_agents.add(responder_role)

        # 允许未发言的Agent补充
        for agent in self.agents:
            if agent.role.value not in responded_agents:
                if self._should_agent_contribute(agent):
                    await self._agent_contribute(agent)

    async def _agent_respond(self, agent: BaseAgent, to_message: AgentMessage) -> None:
        """Agent回应消息"""
        message_id = str(uuid4())

        for callback in self.streaming_callbacks:
            await callback.on_start(agent.name, message_id)

        content = await agent.respond(to_message, self.state.messages)

        message = AdvancedMessage(
            agent_role=agent.role.value,
            agent_name=agent.name,
            content=content,
            timestamp=datetime.now(),
            round=self.state.current_round,
            message_id=message_id,
            reply_to=to_message.message_id,
            message_type=MessageType.RESPONSE
        )

        self.state.add_message(message)
        self.state.conversation_tree.add_message(message_id, to_message.message_id)

        for callback in self.streaming_callbacks:
            await callback.on_complete(content)

    async def _agent_contribute(self, agent: BaseAgent) -> None:
        """Agent补充发言"""
        message_id = str(uuid4())

        for callback in self.streaming_callbacks:
            await callback.on_start(agent.name, message_id)

        content = await agent.analyze(
            self.state.paper,
            self.state.messages[-5:]
        )

        message = AdvancedMessage(
            agent_role=agent.role.value,
            agent_name=agent.name,
            content=content,
            timestamp=datetime.now(),
            round=self.state.current_round,
            message_id=message_id,
            message_type=MessageType.COMMENT
        )

        self.state.add_message(message)
        self.state.conversation_tree.add_message(message_id)

        for callback in self.streaming_callbacks:
            await callback.on_complete(content)

    def _get_pending_messages(self) -> List[AgentMessage]:
        """获取待回应的消息"""
        if self.state.current_round == 2:
            # 第二轮回应第一轮的所有消息
            return self.state.get_messages_by_round(1)
        else:
            # 后续轮次回应上一轮未被充分回应的消息
            return [
                m for m in self.state.get_messages_by_round(self.state.current_round - 1)
                if not any(msg.reply_to == m.message_id for msg in self.state.messages)
            ]

    def _to_advanced_message(self, message: AgentMessage) -> AdvancedMessage:
        """转换为高级消息"""
        if isinstance(message, AdvancedMessage):
            return message
        return AdvancedMessage(
            agent_role=message.agent_role,
            agent_name=message.agent_name,
            content=message.content,
            timestamp=message.timestamp,
            round=message.round,
            message_id=message.message_id,
            reply_to=message.reply_to
        )

    def _get_agent_by_role(self, role: str) -> Optional[BaseAgent]:
        """根据角色获取Agent"""
        for agent in self.agents:
            if agent.role.value == role:
                return agent
        return None

    def _should_agent_contribute(self, agent: BaseAgent) -> bool:
        """判断Agent是否应该补充发言"""
        # 简单逻辑：如果该Agent在本轮还没发言，且是质疑者，则可以补充
        already_spoken = any(
            m.agent_role == agent.role.value and m.round == self.state.current_round
            for m in self.state.messages
        )
        return not already_spoken and agent.role.value == AgentRole.SKEPTIC.value

    def _should_continue(self) -> bool:
        """判断是否继续辩论"""
        if self.state.current_round >= self.state.max_rounds:
            return False

        if self.state.paused:
            return False

        return True

    async def _generate_final_report(self) -> None:
        """生成最终报告"""
        print(f"\n{'='*60}")
        print("生成最终报告（带共识分析）")
        print(f"{'='*60}\n")

        # 计算详细共识分析
        consensus_analysis = self._analyze_consensus()

        self.state.status = DebateStatus.CONSENSUS
        self.state.consensus_score = consensus_analysis['overall_score']

        print(f"共识分析完成:")
        print(f"  整体共识度: {consensus_analysis['overall_score']:.2f}")
        print(f"  理论共识度: {consensus_analysis['theory_score']:.2f}")
        print(f"  工程共识度: {consensus_analysis['engineering_score']:.2f}")
        print(f"  应用共识度: {consensus_analysis['application_score']:.2f}")

    def _analyze_consensus(self) -> Dict[str, float]:
        """详细共识分析"""
        # 简化的共识分析
        skeptic_count = len(self.state.get_messages_by_role(AgentRole.SKEPTIC.value))
        response_count = sum(
            1 for m in self.state.messages
            if m.reply_to and any(
                rm.message_id == m.reply_to
                for rm in self.state.get_messages_by_role(AgentRole.SKEPTIC.value)
            )
        )

        base_score = response_count / skeptic_count if skeptic_count > 0 else 1.0

        # 计算各维度共识度
        theory_score = min(1.0, base_score + 0.1)
        engineering_score = min(1.0, base_score - 0.05)
        application_score = min(1.0, base_score + 0.05)

        overall_score = (theory_score + engineering_score + application_score) / 3

        # 轮次惩罚
        round_penalty = (self.state.current_round - 1) * 0.03
        overall_score = max(0.0, overall_score - round_penalty)

        return {
            'overall_score': overall_score,
            'theory_score': theory_score,
            'engineering_score': engineering_score,
            'application_score': application_score
        }

    async def pause_debate(self, reason: str) -> None:
        """暂停辩论"""
        self.state.paused = True
        self.state.pause_reason = reason
        print(f"\n辩论已暂停: {reason}")

    async def resume_debate(self) -> None:
        """恢复辩论"""
        self.state.paused = False
        self.state.pause_reason = None
        print(f"\n辩论已恢复")

    async def user_intervention(self, target_agent: str, question: str) -> str:
        """用户干预：向特定Agent提问"""
        agent = self._get_agent_by_role(target_agent)
        if not agent:
            return f"未找到Agent: {target_agent}"

        intervention = Intervention(
            intervention_id=str(uuid4()),
            target_agent=target_agent,
            question=question,
            timestamp=datetime.now()
        )

        # 创建用户消息
        user_message = AdvancedMessage(
            agent_role="user",
            agent_name="用户",
            content=question,
            timestamp=datetime.now(),
            round=self.state.current_round,
            message_id=str(uuid4()),
            message_type=MessageType.INTERRUPTION
        )

        self.state.add_message(user_message)

        # 获取Agent回应
        response = await agent.respond(user_message, self.state.messages)
        intervention.response = response
        self.state.interventions.append(intervention)

        return response


# 为保持兼容性，导入SynthesizerAgent
from debate_system import SynthesizerAgent


async def main_advanced():
    """高级模式示例"""
    from debate_system import load_config, AgentFactory

    # 加载配置
    config = load_config()
    config.max_rounds = 3

    # 创建Agent
    llm_client = MockLLMClient()
    agents = AgentFactory.create_all_agents(llm_client, config)

    # 创建高级调度器
    scheduler = AdvancedDebateScheduler(agents, config, llm_client)

    # 添加流式输出回调
    scheduler.add_streaming_callback(ConsoleStreamingCallback())

    # 创建论文
    paper = Paper(
        title="高级辩论测试：深度学习图像分割",
        authors=["测试作者"],
        year=2024,
        abstract="测试摘要",
        content="这是一个测试论文内容，用于演示高级辩论功能..."
    )

    # 启动辩论
    state = await scheduler.start_debate(paper)

    print(f"\n辩论完成！")
    print(f"- 轮次: {state.current_round}")
    print(f"- 发言数: {len(state.messages)}")
    print(f"- 共识度: {state.consensus_score:.2f}")


if __name__ == "__main__":
    asyncio.run(main_advanced())
