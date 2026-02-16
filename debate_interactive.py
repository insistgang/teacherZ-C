"""
增强辩论交互系统 - Agent主动质疑与反驳
Enhanced Debate Interaction System

核心增强：
1. Agent可以主动发起质疑（Challenge）
2. Agent可以针对质疑进行反驳（Rebuttal）
3. 智能决策：何时质疑、质疑谁、质疑什么
4. 辩论收敛：共识检测 + 最大轮次
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from debate_system import (
    AgentRole,
    AgentMessage,
    DebateStatus,
    DebateState,
    LLMClient,
    MockLLMClient,
    Paper,
    BaseAgent,
    AgentConfig,
    DebateConfig,
    load_config,
    AgentFactory,
    SynthesizerAgent,
)


class InteractionType(Enum):
    ANALYSIS = "analysis"
    CHALLENGE = "challenge"
    REBUTTAL = "rebuttal"
    CLARIFICATION = "clarification"
    AGREEMENT = "agreement"
    SYNTHESIS = "synthesis"
    COMMENT = "comment"


class ConsensusLevel(Enum):
    FULL_AGREEMENT = "full_agreement"
    PARTIAL_AGREEMENT = "partial_agreement"
    DISAGREEMENT = "disagreement"
    UNRESOLVED = "unresolved"


@dataclass
class InteractionMessage(AgentMessage):
    interaction_type: InteractionType = InteractionType.ANALYSIS
    target_agent: Optional[str] = None
    target_message_id: Optional[str] = None
    challenge_points: List[str] = field(default_factory=list)
    consensus_level: Optional[ConsensusLevel] = None


@dataclass
class AgentOpinion:
    agent_role: str
    stance: str
    confidence: float
    key_points: List[str]
    concerns: List[str]


@dataclass
class DebateTopic:
    topic_id: str
    description: str
    raised_by: str
    round_raised: int
    status: str = "active"
    opinions: List[AgentOpinion] = field(default_factory=list)
    consensus_reached: bool = False


@dataclass
class InteractiveDebateState(DebateState):
    topics: List[DebateTopic] = field(default_factory=list)
    pending_challenges: List[InteractionMessage] = field(default_factory=list)
    agent_opinions: Dict[str, AgentOpinion] = field(default_factory=dict)
    consensus_matrix: Dict[Tuple[str, str], float] = field(default_factory=dict)


class AgentDecisionEngine:
    """Agent决策引擎 - 决定是否质疑、质疑谁、质疑什么"""

    CHALLENGE_KEYWORDS = {
        AgentRole.SKEPTIC: ["问题", "疑虑", "不足", "缺乏", "需要验证", "假设不成立"],
        AgentRole.MATHEMATICIAN: ["公式错误", "推导不严谨", "缺少证明", "边界条件"],
        AgentRole.ENGINEER: ["难以实现", "复杂度过高", "不稳定", "不可复现"],
        AgentRole.APPLICATION_EXPERT: ["不实用", "成本过高", "场景受限", "难以落地"],
    }

    def __init__(self, agent: BaseAgent, llm_client: LLMClient):
        self.agent = agent
        self.llm_client = llm_client

    async def should_challenge(
        self, message: InteractionMessage, context: List[InteractionMessage]
    ) -> Tuple[bool, float]:
        """判断是否应该质疑某条消息"""
        if message.agent_role == self.agent.role.value:
            return False, 0.0

        if message.interaction_type == InteractionType.CHALLENGE:
            target_is_me = message.target_agent == self.agent.role.value
            if not target_is_me:
                return False, 0.0

        challenge_score = await self._calculate_challenge_score(message, context)
        threshold = self._get_challenge_threshold()

        return challenge_score > threshold, challenge_score

    async def _calculate_challenge_score(
        self, message: InteractionMessage, context: List[InteractionMessage]
    ) -> float:
        """计算质疑分数"""
        prompt = f"""分析以下发言，判断是否需要从{self.agent.name}的角度提出质疑。

发言者: {message.agent_name}
内容: {message.content}

请以JSON格式返回：
{{
    "need_challenge": true/false,
    "confidence": 0.0-1.0,
    "reason": "原因说明",
    "challenge_points": ["质疑点1", "质疑点2"]
}}"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.agent.system_prompt,
                temperature=0.3,
                max_tokens=500,
            )
            result = self._parse_json_response(response)
            if result:
                return (
                    result.get("confidence", 0.0)
                    if result.get("need_challenge")
                    else 0.0
                )
        except:
            pass

        keywords = self.CHALLENGE_KEYWORDS.get(self.agent.role, [])
        keyword_score = sum(0.1 for kw in keywords if kw in message.content)
        return min(1.0, keyword_score)

    def _get_challenge_threshold(self) -> float:
        """获取质疑阈值"""
        thresholds = {
            AgentRole.SKEPTIC: 0.3,
            AgentRole.MATHEMATICIAN: 0.5,
            AgentRole.ENGINEER: 0.5,
            AgentRole.APPLICATION_EXPERT: 0.5,
        }
        return thresholds.get(self.agent.role, 0.6)

    async def select_challenge_target(
        self, messages: List[InteractionMessage]
    ) -> Optional[InteractionMessage]:
        """选择要质疑的目标消息"""
        candidates = []

        for msg in messages:
            if msg.agent_role == self.agent.role.value:
                continue
            if msg.interaction_type == InteractionType.REBUTTAL:
                continue

            should, score = await self.should_challenge(msg, messages)
            if should:
                candidates.append((msg, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def generate_challenge(
        self, target_message: InteractionMessage, context: List[InteractionMessage]
    ) -> Tuple[str, List[str]]:
        """生成质疑内容"""
        prompt = f"""作为{self.agent.name}，请对以下观点提出质疑。

原发言者: {target_message.agent_name}
原内容: {target_message.agent_name}说：{target_message.content}

辩论背景:
{self._format_context(context[-5:])}

请提出你的质疑，格式如下：
1. 明确指出你质疑的具体观点
2. 说明质疑的理由
3. 提出你认为正确或需要补充的内容

直接输出质疑内容，不要加前缀。"""

        response = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.agent.system_prompt,
            temperature=self.agent.config.temperature,
            max_tokens=1500,
        )

        challenge_points = self._extract_key_points(response)
        return response, challenge_points

    async def generate_rebuttal(
        self, challenge: InteractionMessage, context: List[InteractionMessage]
    ) -> str:
        """生成反驳"""
        prompt = f"""作为{self.agent.name}，请回应针对你观点的质疑。

质疑者: {challenge.agent_name}
质疑内容: {challenge.content}

质疑的具体点:
{chr(10).join(f"- {p}" for p in challenge.challenge_points)}

辩论背景:
{self._format_context(context[-5:])}

请进行反驳：
1. 针对每个质疑点逐一回应
2. 提供证据或论证支持你的观点
3. 如确实存在问题，可以部分承认并提出改进

直接输出反驳内容。"""

        return await self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.agent.system_prompt,
            temperature=self.agent.config.temperature,
            max_tokens=1500,
        )

    def _format_context(self, messages: List[InteractionMessage]) -> str:
        """格式化上下文"""
        lines = []
        for msg in messages:
            lines.append(f"[{msg.agent_name}]: {msg.content[:200]}...")
        return "\n".join(lines)

    def _extract_key_points(self, text: str) -> List[str]:
        """提取关键点"""
        points = []
        patterns = [
            r"\d+\.\s*(.+?)(?=\d+\.|$)",
            r"[-•]\s*(.+?)(?=[-•]|$)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            points.extend([m.strip() for m in matches if m.strip()])

        if not points:
            sentences = re.split(r"[。！？]", text)
            points = [s.strip() for s in sentences if len(s.strip()) > 10][:3]

        return points[:5]

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """解析JSON响应"""
        try:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None


class ConsensusDetector:
    """共识检测器"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def check_consensus(
        self, state: InteractiveDebateState
    ) -> Tuple[float, bool]:
        """检测是否达成共识"""
        if len(state.messages) < 4:
            return 0.0, False

        recent_messages = (
            state.messages[-6:] if len(state.messages) >= 6 else state.messages
        )

        prompt = f"""分析以下辩论记录，判断是否已达成共识。

辩论记录:
{self._format_debate(recent_messages)}

请返回JSON格式：
{{
    "consensus_score": 0.0-1.0,
    "consensus_reached": true/false,
    "remaining_issues": ["未解决问题1", "未解决问题2"],
    "agreed_points": ["共识点1", "共识点2"]
}}"""

        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt="你是一个辩论共识分析专家，客观判断辩论是否已经充分并达成共识。",
                temperature=0.2,
                max_tokens=500,
            )
            result = self._parse_json(response)
            if result:
                score = result.get("consensus_score", 0.0)
                reached = result.get("consensus_reached", False)
                state.unresolved_issues = result.get("remaining_issues", [])
                return score, reached
        except:
            pass

        return self._heuristic_consensus(state)

    def _heuristic_consensus(self, state: InteractiveDebateState) -> Tuple[float, bool]:
        """启发式共识检测"""
        challenge_count = sum(
            1
            for m in state.messages
            if isinstance(m, InteractionMessage)
            and m.interaction_type == InteractionType.CHALLENGE
        )
        rebuttal_count = sum(
            1
            for m in state.messages
            if isinstance(m, InteractionMessage)
            and m.interaction_type == InteractionType.REBUTTAL
        )

        if challenge_count == 0:
            return 0.8, True

        resolution_rate = (
            rebuttal_count / challenge_count if challenge_count > 0 else 1.0
        )
        score = min(1.0, resolution_rate * 0.8)

        return score, score > 0.7

    def _format_debate(self, messages: List[AgentMessage]) -> str:
        lines = []
        for msg in messages:
            lines.append(f"[{msg.agent_name}]: {msg.content[:150]}...")
        return "\n".join(lines)

    def _parse_json(self, text: str) -> Optional[Dict]:
        try:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None


class InteractiveDebateScheduler:
    """交互式辩论调度器"""

    def __init__(
        self,
        agents: List[BaseAgent],
        config: DebateConfig,
        llm_client: LLMClient = None,
    ):
        self.agents = [a for a in agents if not isinstance(a, SynthesizerAgent)]
        self.synthesizer = next(
            (a for a in agents if isinstance(a, SynthesizerAgent)), None
        )
        self.config = config
        self.llm_client = llm_client or MockLLMClient()
        self.state: Optional[InteractiveDebateState] = None

        self.decision_engines: Dict[str, AgentDecisionEngine] = {}
        for agent in self.agents:
            self.decision_engines[agent.role.value] = AgentDecisionEngine(
                agent, self.llm_client
            )

        self.consensus_detector = ConsensusDetector(self.llm_client)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def start_debate(self, paper: Paper) -> InteractiveDebateState:
        """启动辩论"""
        self.state = InteractiveDebateState(
            paper=paper,
            messages=[],
            current_round=0,
            status=DebateStatus.INITIALIZED,
            consensus_score=0.0,
            unresolved_issues=[],
            max_rounds=self.config.max_rounds,
        )

        self._print_header(paper)

        await self._round_initial_analysis()

        while self._should_continue():
            await self._round_challenge_rebuttal()

            (
                consensus_score,
                consensus_reached,
            ) = await self.consensus_detector.check_consensus(self.state)
            self.state.consensus_score = consensus_score

            if consensus_reached:
                print(f"\n[共识] 已达成共识！共识度: {consensus_score:.2f}")
                break

        await self._generate_final_report()

        return self.state

    def _print_header(self, paper: Paper):
        print(f"\n{'=' * 60}")
        print(f"交互式辩论系统：{paper.title}")
        print(f"参与Agent: {', '.join([a.name for a in self.agents])}")
        print(f"辩论模式: 主动质疑 + 反驳回应")
        print(f"{'=' * 60}\n")

    async def _round_initial_analysis(self):
        """第一轮：初始分析"""
        self.state.current_round = 1
        self.state.status = DebateStatus.IN_PROGRESS

        print(f"\n{'─' * 50}")
        print(f"第1轮：初始观点陈述")
        print(f"{'─' * 50}\n")

        for agent in self.agents:
            print(f"[{agent.name}] 正在分析...")
            content = await agent.analyze(self.state.paper, self.state.messages)

            message = InteractionMessage(
                agent_role=agent.role.value,
                agent_name=agent.name,
                content=content,
                timestamp=datetime.now(),
                round=1,
                message_id=str(uuid4()),
                interaction_type=InteractionType.ANALYSIS,
            )

            self.state.add_message(message)
            self._print_message(message)

    async def _round_challenge_rebuttal(self):
        """后续轮次：质疑与反驳"""
        self.state.current_round += 1

        print(f"\n{'─' * 50}")
        print(f"第{self.state.current_round}轮：质疑与反驳")
        print(f"{'─' * 50}\n")

        challenges = await self._collect_challenges()

        for challenge in challenges:
            self.state.add_message(challenge)
            self.state.pending_challenges.append(challenge)
            self._print_message(challenge)

            rebuttal = await self._generate_rebuttal_for_challenge(challenge)
            if rebuttal:
                self.state.add_message(rebuttal)
                self._print_message(rebuttal)

        await self._allow_follow_up_comments()

    async def _collect_challenges(self) -> List[InteractionMessage]:
        """收集所有Agent的质疑"""
        challenges = []

        for agent in self.agents:
            if agent.role == AgentRole.SYNTHESIZER:
                continue

            engine = self.decision_engines[agent.role.value]

            target = await engine.select_challenge_target(
                [m for m in self.state.messages if isinstance(m, InteractionMessage)]
            )

            if target:
                content, challenge_points = await engine.generate_challenge(
                    target,
                    [
                        m
                        for m in self.state.messages
                        if isinstance(m, InteractionMessage)
                    ],
                )

                challenge = InteractionMessage(
                    agent_role=agent.role.value,
                    agent_name=agent.name,
                    content=content,
                    timestamp=datetime.now(),
                    round=self.state.current_round,
                    message_id=str(uuid4()),
                    interaction_type=InteractionType.CHALLENGE,
                    target_agent=target.agent_role,
                    target_message_id=target.message_id,
                    challenge_points=challenge_points,
                )
                challenges.append(challenge)

        return challenges

    async def _generate_rebuttal_for_challenge(
        self, challenge: InteractionMessage
    ) -> Optional[InteractionMessage]:
        """为质疑生成反驳"""
        target_agent = self._get_agent_by_role(challenge.target_agent)
        if not target_agent:
            return None

        print(f"\n[{target_agent.name}] 正在准备反驳...")

        engine = self.decision_engines[target_agent.role.value]
        content = await engine.generate_rebuttal(
            challenge,
            [m for m in self.state.messages if isinstance(m, InteractionMessage)],
        )

        rebuttal = InteractionMessage(
            agent_role=target_agent.role.value,
            agent_name=target_agent.name,
            content=content,
            timestamp=datetime.now(),
            round=self.state.current_round,
            message_id=str(uuid4()),
            interaction_type=InteractionType.REBUTTAL,
            target_agent=challenge.agent_role,
            target_message_id=challenge.message_id,
        )

        if challenge in self.state.pending_challenges:
            self.state.pending_challenges.remove(challenge)

        return rebuttal

    async def _allow_follow_up_comments(self):
        """允许补充发言"""
        agents_spoken = set()
        for msg in self.state.get_messages_by_round(self.state.current_round):
            agents_spoken.add(msg.agent_role)

        for agent in self.agents:
            if agent.role.value not in agents_spoken:
                if await self._should_comment(agent):
                    print(f"\n[{agent.name}] 补充发言...")
                    content = await agent.analyze(
                        self.state.paper, self.state.messages[-3:]
                    )

                    message = InteractionMessage(
                        agent_role=agent.role.value,
                        agent_name=agent.name,
                        content=content,
                        timestamp=datetime.now(),
                        round=self.state.current_round,
                        message_id=str(uuid4()),
                        interaction_type=InteractionType.COMMENT,
                    )

                    self.state.add_message(message)
                    self._print_message(message)

    async def _should_comment(self, agent: BaseAgent) -> bool:
        """判断是否应该补充发言"""
        if self.state.current_round < 2:
            return False

        recent_challenges = [
            m
            for m in self.state.messages[-4:]
            if isinstance(m, InteractionMessage)
            and m.interaction_type == InteractionType.CHALLENGE
        ]

        return len(recent_challenges) > 0 and agent.role == AgentRole.SKEPTIC

    def _should_continue(self) -> bool:
        """判断是否继续辩论"""
        if self.state.current_round >= self.state.max_rounds:
            print(f"\n[终止] 达到最大轮次 ({self.config.max_rounds}轮)")
            return False

        if len(self.state.pending_challenges) == 0 and self.state.current_round >= 2:
            recent_activity = len(
                self.state.get_messages_by_round(self.state.current_round)
            )
            if recent_activity == 0:
                return False

        return True

    def _get_agent_by_role(self, role: str) -> Optional[BaseAgent]:
        """根据角色获取Agent"""
        for agent in self.agents:
            if agent.role.value == role:
                return agent
        return None

    def _print_message(self, message: InteractionMessage):
        """打印消息"""
        type_icons = {
            InteractionType.ANALYSIS: "[分析]",
            InteractionType.CHALLENGE: "[质疑]",
            InteractionType.REBUTTAL: "[反驳]",
            InteractionType.COMMENT: "[评论]",
            InteractionType.AGREEMENT: "[同意]",
        }
        icon = type_icons.get(message.interaction_type, "📌")

        preview = message.content[:150]
        if len(message.content) > 150:
            preview += "..."

        if message.interaction_type == InteractionType.CHALLENGE:
            print(
                f"{icon} [{message.agent_name}] → @{message.target_agent}: {preview}\n"
            )
        elif message.interaction_type == InteractionType.REBUTTAL:
            print(
                f"{icon} [{message.agent_name}] 回应 @{message.target_agent}: {preview}\n"
            )
        else:
            print(f"{icon} [{message.agent_name}]: {preview}\n")

    async def _generate_final_report(self):
        """生成最终报告"""
        print(f"\n{'=' * 60}")
        print("生成最终报告")
        print(f"{'=' * 60}\n")

        self.state.status = DebateStatus.CONSENSUS

        stats = self._calculate_statistics()
        self._save_outputs(stats)

        print(f"辩论统计:")
        print(f"  - 总轮次: {self.state.current_round}")
        print(f"  - 总发言数: {len(self.state.messages)}")
        print(f"  - 质疑数: {stats['challenge_count']}")
        print(f"  - 反驳数: {stats['rebuttal_count']}")
        print(f"  - 共识度: {self.state.consensus_score:.2f}")

    def _calculate_statistics(self) -> Dict:
        """计算统计信息"""
        messages = [m for m in self.state.messages if isinstance(m, InteractionMessage)]

        return {
            "challenge_count": sum(
                1 for m in messages if m.interaction_type == InteractionType.CHALLENGE
            ),
            "rebuttal_count": sum(
                1 for m in messages if m.interaction_type == InteractionType.REBUTTAL
            ),
            "analysis_count": sum(
                1 for m in messages if m.interaction_type == InteractionType.ANALYSIS
            ),
            "messages_by_role": {
                agent.role.value: len(
                    [m for m in messages if m.agent_role == agent.role.value]
                )
                for agent in self.agents
            },
        }

    def _save_outputs(self, stats: Dict):
        """保存输出"""
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = self.output_dir / f"interactive_debate_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "paper": self.state.paper.title,
                    "rounds": self.state.current_round,
                    "consensus_score": self.state.consensus_score,
                    "statistics": stats,
                    "messages": [
                        {
                            "round": m.round,
                            "agent": m.agent_name,
                            "type": m.interaction_type.value
                            if isinstance(m, InteractionMessage)
                            else "unknown",
                            "content": m.content,
                        }
                        for m in self.state.messages
                    ],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n报告已保存: {json_path}")


class GLM5Client(LLMClient):
    """GLM-5 API客户端"""

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY")
        self.base_url = base_url or os.getenv(
            "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        )
        if not self.api_key:
            raise ValueError("请设置GLM_API_KEY或ZHIPU_API_KEY环境变量")

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.5,
        max_tokens: int = 4000,
    ) -> str:
        import asyncio

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": "glm-4-plus",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            for attempt in range(3):
                try:
                    await asyncio.sleep(2)
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=payload,
                        )
                        if response.status_code == 429:
                            wait_time = 10 * (attempt + 1)
                            print(f"[API限流] 等待{wait_time}秒后重试...")
                            await asyncio.sleep(wait_time)
                            continue
                        response.raise_for_status()
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < 2:
                        continue
                    raise

        except ImportError:
            raise ImportError("请安装 httpx: pip install httpx")
        except Exception as e:
            raise RuntimeError(f"GLM-5 API调用失败: {e}")


import os


async def run_interactive_debate(
    paper_content: str,
    paper_title: str = "测试论文",
    max_rounds: int = 3,
    llm_type: str = "mock",
    api_key: str = None,
):
    """运行交互式辩论"""
    config = load_config()
    config.max_rounds = max_rounds

    if llm_type == "mock":
        llm_client = MockLLMClient()
    elif llm_type == "glm":
        llm_client = GLM5Client(api_key=api_key)
    elif llm_type == "claude":
        from debate_system import ClaudeLLMClient

        llm_client = ClaudeLLMClient()
    else:
        llm_client = MockLLMClient()

    agents = AgentFactory.create_all_agents(llm_client, config)
    scheduler = InteractiveDebateScheduler(agents, config, llm_client)

    paper = Paper(
        title=paper_title,
        authors=["待补充"],
        year=2024,
        abstract="",
        content=paper_content,
    )

    return await scheduler.start_debate(paper)


async def main():
    """示例运行"""
    sample_paper = """
# 基于变分方法的图像分割算法研究

## 摘要
本文提出了一种基于变分优化的图像分割方法，结合了全变分(TV)正则化和水平集方法。

## 方法
1. 能量函数定义：
   E(u) = ∫|∇u| dx + λ∫(u-f)² dx

2. 优化策略：采用梯度下降法求解

3. 实验验证：在标准数据集上达到92%的分割精度

## 结论
该方法在多类分割任务上表现优异，具有良好的理论保证。
"""

    await run_interactive_debate(sample_paper, "变分图像分割方法", max_rounds=3)


if __name__ == "__main__":
    asyncio.run(main())
