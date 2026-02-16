#!/usr/bin/env python3
"""
多智能体论文精读Battle系统
让多个专业AI智能体互相辩论、质疑、补充，共同深入分析Xiaohao Cai的论文
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime

# 配置
PAPERS_DIR = "xiaohao_cai_papers_final"
OUTPUT_DIR = "battle_notes"

class Agent:
    """AI智能体基类"""
    def __init__(self, name, role, personality, focus_areas):
        self.name = name
        self.role = role
        self.personality = personality  # 性格：critical, supportive, curious, skeptical
        self.focus_areas = focus_areas  # 关注领域
        self.contributions = []  # 贡献记录

    def analyze(self, paper_content, other_agents_views):
        """分析论文并回应其他智能体的观点"""
        pass

class PaperBattleArena:
    """论文辩论竞技场"""
    def __init__(self):
        self.agents = [
            Agent(
                name="算法猎手",
                role="算法专家",
                personality="skeptical",
                focus_areas=["算法创新性", "复杂度分析", "收敛性证明"]
            ),
            Agent(
                name="数学 rigor",
                role="数学专家",
                personality="critical",
                focus_areas=["数学严谨性", "定理证明", "公式推导"]
            ),
            Agent(
                name="应用探路者",
                role="应用专家",
                personality="curious",
                focus_areas=["实际应用", "实验设计", "结果分析"]
            ),
            Agent(
                name="架构洞察者",
                role="系统架构专家",
                personality="supportive",
                focus_areas=["整体架构", "模块设计", "可扩展性"]
            ),
            Agent(
                name="批判思维者",
                role="批评家",
                personality="critical",
                focus_areas=["局限性", "假设合理性", "潜在问题"]
            )
        ]
        self.battle_rounds = 3  # 辩论轮数
        self.current_paper = None

    def extract_pdf_text(self, pdf_path, max_pages=20):
        """提取PDF文本"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for i, page in enumerate(doc[:max_pages]):
                text += f"\n=== Page {i+1} ===\n"
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"Error reading PDF: {e}"

    def get_paper_info(self, pdf_path):
        """获取论文基本信息"""
        filename = os.path.basename(pdf_path)
        name_parts = filename.replace('.pdf', '').split('_')

        # 解析论文信息
        info = {
            'filename': filename,
            'title': filename,
            'arxiv_id': None,
            'year': None
        }

        # 尝试解析arXiv ID
        for part in name_parts:
            if len(part) == 10 and part.count('.') == 1:
                try:
                    float(part)
                    info['arxiv_id'] = part
                    # 提取年份
                    year_prefix = part.split('.')[0]
                    if year_prefix.startswith('20') or year_prefix.startswith('19'):
                        info['year'] = '20' + year_prefix[2:4]
                except:
                    pass

        # 从文件名提取标题
        if len(name_parts) > 1:
            info['title'] = ' '.join(name_parts[1:]).replace('_', ' ')

        return info

    def generate_battle_prompt(self, paper_info, paper_content, round_num, previous_discussions):
        """生成辩论提示词"""

        prompt = f"""# Xiaohao Cai 论文深度分析 - 第{round_num}轮辩论

## 论文信息
- 标题: {paper_info['title']}
- 文件: {paper_info['filename']}
- arXiv ID: {paper_info.get('arxiv_id', 'N/A')}
- 年份: {paper_info.get('year', 'N/A')}

## 论文内容摘要
{paper_content[:5000]}

"""
        if round_num > 1 and previous_discussions:
            prompt += f"\n## 前几轮讨论要点\n{previous_discussions}\n"

        prompt += """
## 你的任务
作为{{agent_name}}({{role}})，你需要:

1. **从你的专业角度**深度分析这篇论文
2. **回应其他智能体的观点** - 支持有价值的观点，质疑薄弱之处
3. **提出你的独到见解** - 发现其他人没有注意到的问题

## 你的关注领域
{{focus_areas}}

## 你的性格特质
{{personality_description}}

## 输出格式
```markdown
## {{agent_name}}的分析 (第{round_num}轮)

### 核心发现
[你发现的最重要内容]

### 对其他观点的回应
[支持谁/反对谁，为什么]

### 你的独到见解
[只有你能发现的深度洞察]

### 待解答的问题
[需要进一步探讨的问题]
```
"""
        return prompt

    def run_battle(self, pdf_path):
        """运行一场论文辩论"""
        self.current_paper = self.get_paper_info(pdf_path)

        print(f"\n{'='*60}")
        print(f"🎯 论文辩论: {self.current_paper['title']}")
        print(f"{'='*60}\n")

        # 提取论文内容
        paper_content = self.extract_pdf_text(pdf_path)

        # 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 存储辩论记录
        battle_log = {
            'paper_info': self.current_paper,
            'rounds': [],
            'timestamp': datetime.now().isoformat()
        }

        previous_discussions = ""

        # 进行多轮辩论
        for round_num in range(1, self.battle_rounds + 1):
            print(f"\n--- 第 {round_num} 轮辩论 ---\n")

            round_discussions = []

            for agent in self.agents:
                # 为每个智能体生成提示
                personality_desc = {
                    'skeptical': '你持怀疑态度，总是寻找论文中的漏洞和过度声明',
                    'critical': '你严格批判，关注方法的局限性和假设的合理性',
                    'supportive': '你支持建设性分析，关注论文的贡献和价值',
                    'curious': '你充满好奇，探索论文的延伸应用和未来方向'
                }

                prompt = self.generate_battle_prompt(
                    self.current_paper,
                    paper_content,
                    round_num,
                    previous_discussions
                ).format(
                    agent_name=agent.name,
                    role=agent.role,
                    focus_areas=', '.join(agent.focus_areas),
                    personality_description=personality_desc.get(agent.personality, '')
                )

                # 这里会调用实际的AI模型
                # 暂时保存prompt供手动执行
                round_discussions.append({
                    'agent': agent.name,
                    'prompt': prompt
                })

                print(f"  [{agent.name}] 准备分析...")

            # 更新讨论历史（实际执行后会填充）
            previous_discussions += f"\n\n=== 第 {round_num} 轮 ===\n"

            battle_log['rounds'].append({
                'round': round_num,
                'discussions': round_discussions
            })

        # 保存辩论记录
        output_file = os.path.join(
            OUTPUT_DIR,
            f"battle_{self.current_paper['filename'].replace('.pdf', '')}.json"
        )
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(battle_log, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 辩论记录已保存: {output_file}")
        return battle_log

def main():
    """主函数"""
    arena = PaperBattleArena()

    # 获取所有论文
    papers = list(Path(PAPERS_DIR).glob("*.pdf"))
    print(f"发现 {len(papers)} 篇论文")

    # 随机选择一篇开始
    import random
    selected = random.choice(papers)
    print(f"\n选中论文: {selected.name}")

    # 运行辩论
    arena.run_battle(str(selected))

if __name__ == "__main__":
    main()
