"""
Xiaohao Cai论文专用辩论脚本

从现有的论文笔记文件中提取内容，进行多智能体辩论
"""

import asyncio
import json
import os
import re
from pathlib import Path
from debate_system import (
    DebateSystem,
    Paper,
    PaperParser,
    MockLLMClient,
    load_config
)


def extract_paper_from_note(note_path: str) -> Paper:
    """从论文笔记文件中提取论文信息"""

    with open(note_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取标题
    title_match = re.search(r'# (.+)', content)
    title = title_match.group(1) if title_match else Path(note_path).stem

    # 提取作者
    authors = []
    author_match = re.search(r'\*\*作者\*\*[:：]\s*(.+)', content)
    if author_match:
        authors_str = author_match.group(1)
        authors = [a.strip() for a in authors_str.split(',')]

    # 提取年份
    year = 2024
    year_match = re.search(r'\*\*年份\*\*[:：]\s*(\d{4})', content)
    if year_match:
        year = int(year_match.group(1))

    # 提取摘要（如果有）
    abstract = ""
    abstract_match = re.search(r'## 摘要\s*\n(.+?)(?=\n##|\n---|\Z)', content, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()

    # 提取正文（去掉元数据部分）
   正文_content = content

    return Paper(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        content=正文_content,
        pdf_path=note_path
    )


async def debate_single_note(note_path: str, use_real_llm: bool = False):
    """对单个笔记进行辩论"""

    print(f"\n{'='*60}")
    print(f"正在处理: {note_path}")
    print(f"{'='*60}\n")

    # 提取论文
    paper = extract_paper_from_note(note_path)

    print(f"论文标题: {paper.title}")
    print(f"作者: {', '.join(paper.authors) if paper.authors else '未知'}")
    print(f"年份: {paper.year}")
    print(f"内容长度: {len(paper.content)} 字符\n")

    # 创建系统
    if use_real_llm:
        try:
            from debate_system import ClaudeLLMClient
            llm_client = ClaudeLLMClient()
            print("使用Claude LLM客户端\n")
        except ValueError as e:
            print(f"错误: {e}")
            print("切换到模拟LLM\n")
            llm_client = MockLLMClient()
    else:
        llm_client = MockLLMClient()
        print("使用模拟LLM客户端\n")

    system = DebateSystem(llm_client=llm_client)
    system.initialize()

    # 执行辩论
    state = await system.debate_paper(paper)

    print(f"\n{'='*60}")
    print(f"辩论完成！")
    print(f"- 论文: {paper.title}")
    print(f"- 轮次: {state.current_round}")
    print(f"- 发言数: {len(state.messages)}")
    print(f"- 共识度: {state.consensus_score:.2f}")
    print(f"{'='*60}\n")

    return state


async def debate_multiple_notes(note_dir: str, pattern: str = "*.md",
                                max_papers: int = None,
                                use_real_llm: bool = False):
    """批量辩论多个笔记"""

    note_dir = Path(note_dir)

    # 查找所有笔记文件
    note_files = list(note_dir.glob(pattern))

    if max_papers:
        note_files = note_files[:max_papers]

    print(f"\n找到 {len(note_files)} 个笔记文件\n")

    results = []

    for i, note_path in enumerate(note_files, 1):
        print(f"\n[{i}/{len(note_files)}] 处理: {note_path.name}")

        try:
            state = await debate_single_note(str(note_path), use_real_llm)
            results.append({
                'note_path': str(note_path),
                'title': state.paper.title,
                'rounds': state.current_round,
                'messages': len(state.messages),
                'consensus': state.consensus_score
            })
        except Exception as e:
            print(f"处理失败: {e}")
            results.append({
                'note_path': str(note_path),
                'error': str(e)
            })

    # 汇总报告
    print(f"\n{'='*60}")
    print("批量辩论汇总报告")
    print(f"{'='*60}\n")

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}\n")

    if successful:
        avg_consensus = sum(r['consensus'] for r in successful) / len(successful)
        print(f"平均共识度: {avg_consensus:.2f}\n")

        print("论文共识度排名:")
        sorted_results = sorted(successful, key=lambda x: x['consensus'], reverse=True)
        for i, r in enumerate(sorted_results, 1):
            print(f"  {i}. {r['title'][:40]}... (共识度: {r['consensus']:.2f})")

    # 保存汇总
    summary_path = Path("outputs/batch_debate_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n汇总报告已保存到: {summary_path}")


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Xiaohao Cai论文辩论系统"
    )
    parser.add_argument(
        "--note",
        type=str,
        help="单个笔记文件路径"
    )
    parser.add_argument(
        "--note-dir",
        type=str,
        default=".",
        help="笔记目录路径"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="论文阐述_*.md",
        help="笔记文件匹配模式"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="最多处理论文数量"
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="使用真实LLM（需要API密钥）"
    )

    args = parser.parse_args()

    if args.note:
        # 单个笔记辩论
        await debate_single_note(args.note, args.real_llm)
    else:
        # 批量辩论
        await debate_multiple_notes(
            args.note_dir,
            args.pattern,
            args.max_papers,
            args.real_llm
        )


if __name__ == "__main__":
    asyncio.run(main())
