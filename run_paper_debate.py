"""
论文辩论启动器
快速启动多Agent辩论分析论文
"""

import asyncio
import os
import sys
from pathlib import Path

PAPERS_DIR = Path(r"D:\Documents\zx\xiaohao_cai_papers_final")
GLM_API_KEY = "c1680a1242154411aee50b6bc4abdefd.5Qj5WoDI2jFX4gAw"


def list_papers():
    """列出可选论文"""
    papers = []
    for f in PAPERS_DIR.iterdir():
        if f.suffix in [".pdf", ".md", ".txt"]:
            papers.append(f)
    papers.sort(key=lambda x: x.name)
    return papers


def print_paper_list(papers):
    """打印论文列表"""
    print("\n" + "=" * 60)
    print("可辩论论文列表")
    print("=" * 60)
    for i, p in enumerate(papers, 1):
        size = p.stat().st_size / 1024
        ext = p.suffix.upper()
        print(f"{i:3}. [{ext:4}] {size:6.1f}KB  {p.name}")
    print("=" * 60)


def extract_text_from_pdf(pdf_path):
    """从PDF提取文本"""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:15]:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts)
    except ImportError:
        print("错误: 需要安装 pdfplumber: pip install pdfplumber")
        return None
    except Exception as e:
        print(f"PDF解析错误: {e}")
        return None


def extract_text_from_file(file_path):
    """从文件提取文本"""
    path = Path(file_path)
    if path.suffix == ".pdf":
        return extract_text_from_pdf(path)
    else:
        try:
            return path.read_text(encoding="utf-8")
        except:
            return path.read_text(encoding="gbk")


async def run_debate(paper_path, max_rounds=3):
    """运行辩论"""
    from debate_interactive import run_interactive_debate, GLM5Client

    paper_path = Path(paper_path)
    title = paper_path.stem

    print(f"\n正在读取: {paper_path.name}")
    content = extract_text_from_file(paper_path)

    if not content:
        print("无法读取文件内容")
        return

    if len(content) > 30000:
        print(f"内容较长 ({len(content)}字符)，截取前30000字符...")
        content = content[:30000]

    print(f"论文标题: {title}")
    print(f"内容长度: {len(content)}字符")
    print(f"辩论轮次: {max_rounds}")
    print(f"LLM: GLM-5")
    print("\n启动辩论...\n")

    await run_interactive_debate(
        paper_content=content,
        paper_title=title,
        max_rounds=max_rounds,
        llm_type="glm",
        api_key=GLM_API_KEY,
    )


def main():
    papers = list_papers()
    print_paper_list(papers)

    while True:
        choice = input("\n输入序号选择论文 (q退出): ").strip()

        if choice.lower() == "q":
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(papers):
                rounds_input = input("辩论轮次 (默认3): ").strip()
                max_rounds = int(rounds_input) if rounds_input else 3

                asyncio.run(run_debate(papers[idx], max_rounds))
                break
            else:
                print("序号超出范围")
        except ValueError:
            print("请输入有效数字")
        except KeyboardInterrupt:
            print("\n已取消")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paper_path = sys.argv[1]
        rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        asyncio.run(run_debate(paper_path, rounds))
    else:
        main()
