#!/usr/bin/env python3
"""
Xiaohao Cai 论文精读 - 多智能体Battle系统

系统设计:
- 5个专业智能体，各有专长和立场
- 多轮辩论: 初析 → 质疑 → 补充 → 综合
- 每个智能体会阅读其他智能体的观点并进行回应
"""

import os
import json
import fitz
from pathlib import Path
from datetime import datetime

# ============================================
# 配置
# ============================================
PAPERS_DIR = "xiaohao_cai_papers_final"
OUTPUT_DIR = "battle_notes"
BATTLE_ROUNDS = 4  # 辩论轮数

# ============================================
# 智能体定义
# ============================================
AGENTS = {
    "math_rigor": {
        "name": "数学 rigor",
        "role": "数学严谨性专家",
        "personality": "critical",
        "style": "严格、细致、不容忍模糊",
        "focus": ["定理证明", "数学推导", "假设条件", "边界情况"],
        "questions": [
            "定理的证明是否完整？",
            "假设条件是否过于严格？",
            "数学推导有没有漏洞？"
        ]
    },
    "algorithm_hunter": {
        "name": "算法猎手",
        "role": "算法分析专家",
        "personality": "skeptical",
        "style": "犀利、直接、追求效率",
        "focus": ["时间复杂度", "空间复杂度", "收敛速度", "算法创新"],
        "questions": [
            "算法的创新点在哪里？",
            "复杂度分析是否准确？",
            "有没有更高效的替代方案？"
        ]
    },
    "practitioner": {
        "name": "落地工程师",
        "role": "工程应用专家",
        "personality": "pragmatic",
        "style": "务实、关注可行性",
        "focus": ["实现难度", "计算资源", "参数敏感性", "适用场景"],
        "questions": [
            "这个方法能落地吗？",
            "需要多少计算资源？",
            "参数调优难度如何？"
        ]
    },
    "visionary": {
        "name": "远见者",
        "role": "趋势洞察专家",
        "personality": "enthusiastic",
        "style": "宏大、前瞻、寻找意义",
        "focus": ["研究意义", "领域影响", "未来方向", "跨学科连接"],
        "questions": [
            "这项工作的长远价值是什么？",
            "可能开启哪些新方向？",
            "与其他前沿工作的联系？"
        ]
    },
    "critic": {
        "name": "魔鬼代言人",
        "role": "批判性评论家",
        "personality": "contrarian",
        "style": "反其道而行、挑战权威",
        "focus": ["局限性", "反对意见", "潜在风险", "未被考虑的视角"],
        "questions": [
            "作者可能忽略了什么？",
            "在什么情况下这个方法会失败？",
            "有没有更简单的替代方案？"
        ]
    }
}

# ============================================
# 提示词模板
# ============================================
ROUND_PROMPTS = {
    1: """# 第1轮：初析 - 各自独立分析

你是{name}，一位{role}。
你的分析风格：{style}

请仔细阅读以下Xiaohao Cai的论文内容，从你的专业视角给出**初次分析**。

## 你的关注重点
{focus}

## 论文内容
{content}

## 输出要求（深度分析，约2000字）
```markdown
## {name}的初析

### 1. 核心摘要
用你的话概括论文的核心贡献（从你的视角）

### 2. 关键发现
列出你发现的3-5个关键点

### 3. 专业评价
从{role}的角度评价这篇论文：
- 创新性: /10
- 严谨性: /10
- 可用性: /10
- 影响力: /10

### 4. 深度洞察
只有{role}才能发现的深层见解

### 5. 待探讨问题
列出需要其他专家解答的问题
```
""",

    2: """# 第2轮：质疑 - 针对性提问

你是{name}，现在阅读其他专家的初析，然后**提出质疑**。

## 其他专家的观点
{other_views}

## 你的任务
1. **指出其他专家分析中的问题**
2. **对论文中不清晰的点提出尖锐问题**
3. **挑战其他专家的结论**

## 输出要求
```markdown
## {name}的质疑

### 对其他专家的回应
[逐一回应其他专家的观点，支持有价值的，反对有问题的]

### 对论文的质疑
[提出你对论文的疑问和挑战]

### 要求其他专家澄清的问题
[列出需要其他人回答的问题]
```
""",

    3: """# 第3轮：补充 - 深度挖掘

你是{name}，基于前两轮的讨论，进行**补充分析**。

## 前两轮讨论总结
{discussion_summary}

## 你的任务
1. **回应其他专家对你的质疑**
2. **补充你的分析细节**
3. **综合多方观点，给出更完整的评价**

## 输出要求
```markdown
## {name}的补充分析

### 回应质疑
[解释你的观点，修正可能的错误]

### 补充细节
[深入分析之前未展开的内容]

### 综合评价
[结合讨论，给出最终评价]
```
""",

    4: """# 第4轮：综合 - 达成共识

你是{name}，这是最后一轮，请**综合所有讨论**。

## 完整讨论记录
{full_discussion}

## 你的任务
1. **总结所有有价值的核心观点**
2. **指出最终的共识和分歧**
3. **给出这篇论文的最终评价**

## 输出要求
```markdown
## {name}的最终总结

### 核心共识
[所有专家认同的关键点]

### 仍有分歧
[尚未达成一致的观点]

### 最终评分
- 创新性: X/10 (说明理由)
- 价值: X/10 (说明理由)
- 实用性: X/10 (说明理由)

### 一句话总结
[用一句话概括这篇论文的价值]
```
"""
}

class PaperBattleSystem:
    """论文辩论系统"""

    def __init__(self):
        self.papers_dir = Path(PAPERS_DIR)
        self.output_dir = Path(OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)

    def extract_paper_content(self, pdf_path):
        """提取论文内容"""
        try:
            doc = fitz.open(pdf_path)
            content = []

            # 提取前15页的核心内容
            for i in range(min(15, len(doc))):
                page = doc[i]
                text = page.get_text()
                if text.strip():
                    content.append(f"## Page {i+1}\n{text[:3000]}")  # 每页限制3000字符

            doc.close()

            # 获取论文基本信息
            filename = Path(pdf_path).name
            title = filename.replace('.pdf', '')

            return {
                'filename': filename,
                'title': title,
                'content': '\n\n'.join(content),
                'total_pages': len(doc)
            }
        except Exception as e:
            return {'error': str(e)}

    def create_battle_session(self, pdf_path):
        """创建一场辩论会话"""
        # 提取论文内容
        paper = self.extract_paper_content(pdf_path)

        if 'error' in paper:
            print(f"❌ 无法读取论文: {paper['error']}")
            return None

        # 创建会话
        session = {
            'paper': paper,
            'agents': list(AGENTS.keys()),
            'rounds': {},
            'created_at': datetime.now().isoformat()
        }

        # 为每轮生成每个智能体的任务
        for round_num in range(1, BATTLE_ROUNDS + 1):
            session['rounds'][round_num] = {}

            for agent_id, agent_info in AGENTS.items():
                task = self._generate_agent_task(
                    agent_info,
                    paper,
                    round_num,
                    session
                )

                session['rounds'][round_num][agent_id] = {
                    'agent': agent_info,
                    'task': task,
                    'response': None  # 待填充
                }

        return session

    def _generate_agent_task(self, agent_info, paper, round_num, session):
        """为智能体生成任务提示"""

        if round_num == 1:
            # 第一轮：独立分析
            return ROUND_PROMPTS[1].format(
                name=agent_info['name'],
                role=agent_info['role'],
                style=agent_info['style'],
                focus='\n'.join([f"- {f}" for f in agent_info['focus']]),
                content=paper['content'][:15000]  # 限制长度
            )

        elif round_num == 2:
            # 第二轮：质疑 - 需要其他人的第一轮观点
            other_views = self._collect_round_responses(session, 1)
            return ROUND_PROMPTS[2].format(
                name=agent_info['name'],
                other_views=other_views
            )

        elif round_num == 3:
            # 第三轮：补充 - 需要前两轮的讨论
            discussion = self._collect_full_discussion(session, up_to=2)
            return ROUND_PROMPTS[3].format(
                name=agent_info['name'],
                discussion_summary=discussion
            )

        else:
            # 第四轮：综合
            discussion = self._collect_full_discussion(session, up_to=3)
            return ROUND_PROMPTS[4].format(
                name=agent_info['name'],
                full_discussion=discussion
            )

    def _collect_round_responses(self, session, round_num):
        """收集某轮的所有回复"""
        responses = []
        for agent_id, data in session['rounds'][round_num].items():
            if data.get('response'):
                responses.append(f"## {data['agent']['name']}\n{data['response']}")
        return '\n\n'.join(responses) if responses else "[暂无回复]"

    def _collect_full_discussion(self, session, up_to):
        """收集到某轮为止的所有讨论"""
        all_discussions = []
        for r in range(1, up_to + 1):
            if r in session['rounds']:
                all_discussions.append(f"\n=== 第{r}轮 ===\n")
                all_discussions.append(self._collect_round_responses(session, r))
        return '\n'.join(all_discussions)

    def save_session(self, session):
        """保存会话到文件"""
        filename = session['paper']['filename'].replace('.pdf', '') + '.json'
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

        return output_path

    def create_agent_prompts_file(self, session, round_num=None):
        """为人类或AI执行创建提示文件"""
        prompts_dir = self.output_dir / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        files_created = []

        rounds_to_process = [round_num] if round_num else range(1, BATTLE_ROUNDS + 1)

        for r in rounds_to_process:
            for agent_id, data in session['rounds'][r].items():
                filename = f"r{r}_{agent_id}_{session['paper']['filename'][:30]}.txt"
                prompt_path = prompts_dir / filename

                with open(prompt_path, 'w', encoding='utf-8') as f:
                    f.write(data['task'])

                files_created.append(prompt_path)

        return files_created


def main():
    """主函数"""
    system = PaperBattleSystem()

    # 获取所有论文
    papers = list(system.papers_dir.glob("*.pdf"))
    print(f"📚 发现 {len(papers)} 篇论文\n")

    # 显示论文列表
    for i, paper in enumerate(papers[:20], 1):  # 显示前20篇
        print(f"{i:2}. {paper.name}")

    if len(papers) > 20:
        print(f"... 还有 {len(papers) - 20} 篇")

    # 选择论文
    print("\n选择要辩论的论文编号 (1-{})，或按回车随机选择: ".format(len(papers)))
    # 默认随机选择
    import random
    selected = random.choice(papers)
    print(f"✅ 选中: {selected.name}\n")

    # 创建会话
    print("🎯 创建辩论会话...")
    session = system.create_battle_session(str(selected))

    if session:
        # 保存会话
        output_path = system.save_session(session)
        print(f"✅ 会话已保存: {output_path}")

        # 生成提示文件
        prompt_files = system.create_agent_prompts_file(session)
        print(f"✅ 已生成 {len(prompt_files)} 个提示文件到 {system.output_dir}/prompts/")

        print(f"\n📊 论文信息:")
        print(f"   标题: {session['paper']['title']}")
        print(f"   页数: {session['paper']['total_pages']}")
        print(f"   辩论轮数: {BATTLE_ROUNDS}")
        print(f"   参与智能体: {len(AGENTS)}")

        print(f"\n🚀 下一步:")
        print(f"   提示文件已生成，可以逐一执行各智能体的任务")

if __name__ == "__main__":
    main()
