"""
简化版辩论系统 - 减少API请求
只用2个Agent：质疑者 + 综合者
"""

import asyncio
import os
import re
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

PAPERS_DIR = Path(r"D:\Documents\zx\xiaohao_cai_papers_final")
GLM_API_KEY = "c1680a1242154411aee50b6bc4abdefd.5Qj5WoDI2jFX4gAw"
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


@dataclass
class Paper:
    title: str
    content: str


@dataclass
class Message:
    agent: str
    content: str
    round: int
    msg_type: str


async def call_glm(prompt: str, system: str = "", delay: float = 3.0) -> str:
    """调用GLM API"""
    await asyncio.sleep(delay)

    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "glm-4-flash",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000,
    }

    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{GLM_BASE_URL}/chat/completions", headers=headers, json=payload
                )
                if resp.status_code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"  [限流] 等待{wait}秒...")
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 4:
                await asyncio.sleep(10)
                continue
            raise RuntimeError(f"API调用失败: {e}")

    raise RuntimeError("API调用失败: 重试次数用尽")


SYSTEM_PROMPTS = {
    "skeptic": """你是质疑者Agent，专门批判性地分析学术论文。

你的任务：
1. 找出论文的潜在问题、假设的局限性
2. 质疑实验设置的合理性
3. 指出任何过度声明或缺乏证据的结论
4. 提出需要进一步验证的问题

请用简洁的中文回复，列出2-4个主要质疑点。""",
    "synthesizer": """你是综合者Agent，负责评估论文价值并生成总结。

你的任务：
1. 综合各方观点
2. 评估论文的实际贡献
3. 给出改进建议
4. 判断论文是否值得深入阅读

请用简洁的中文回复，给出结构化的评估报告。""",
}


async def debate_paper(paper: Paper, max_rounds: int = 2):
    """简化版辩论：质疑者提问 -> 综合者总结"""
    messages: List[Message] = []

    print(f"\n{'=' * 60}")
    print(f"简化版论文辩论")
    print(f"论文: {paper.title}")
    print(f"内容长度: {len(paper.content)}字符")
    print(f"{'=' * 60}\n")

    content_preview = paper.content[:15000]

    # 第1轮：质疑者分析
    print("[质疑者] 正在分析论文...")
    prompt = f"""请分析以下论文并指出主要问题：

标题: {paper.title}

内容摘要:
{content_preview[:5000]}

请列出2-4个主要质疑点或潜在问题。"""

    skeptic_response = await call_glm(prompt, SYSTEM_PROMPTS["skeptic"], delay=2.0)
    messages.append(Message("质疑者", skeptic_response, 1, "analysis"))
    print(f"\n[质疑者] 分析完成:\n{skeptic_response[:500]}...\n")

    # 第2轮：综合者总结
    print("[综合者] 正在评估论文...")
    prompt = f"""基于以下论文内容和质疑，请给出综合评估：

标题: {paper.title}

内容摘要:
{content_preview[:3000]}

质疑者的观点:
{skeptic_response}

请给出:
1. 论文核心贡献 (1-2点)
2. 主要局限 (1-2点)
3. 阅读建议 (是否值得深入阅读)"""

    synth_response = await call_glm(prompt, SYSTEM_PROMPTS["synthesizer"], delay=3.0)
    messages.append(Message("综合者", synth_response, 2, "synthesis"))
    print(f"\n[综合者] 评估完成:\n{synth_response}\n")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("outputs") / f"debate_simple_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    result = {
        "paper_title": paper.title,
        "timestamp": timestamp,
        "messages": [
            {
                "agent": m.agent,
                "round": m.round,
                "type": m.msg_type,
                "content": m.content,
            }
            for m in messages
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_file}")
    return messages


def extract_text(pdf_path: Path) -> Optional[str]:
    """从PDF提取文本"""
    try:
        import pdfplumber

        texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:12]:
                t = page.extract_text()
                if t:
                    texts.append(t)
        return "\n\n".join(texts)
    except Exception as e:
        print(f"PDF解析错误: {e}")
        return None


def list_pdfs():
    """列出PDF文件"""
    pdfs = sorted(PAPERS_DIR.glob("*.pdf"), key=lambda x: x.name)
    return pdfs


async def main():
    pdfs = list_pdfs()

    print("\n可选论文 (部分):")
    for i, p in enumerate(pdfs[:20], 1):
        size = p.stat().st_size / 1024
        print(f"  {i:2}. [{size:6.0f}KB] {p.name[:50]}")

    print("\n输入序号选择论文，或直接输入论文序号 (1-20):")

    try:
        choice = input("> ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(pdfs):
            pdf = pdfs[idx]
            content = extract_text(pdf)
            if content:
                paper = Paper(title=pdf.stem, content=content)
                await debate_paper(paper)
            else:
                print("无法读取PDF内容")
        else:
            print("序号无效")
    except ValueError:
        print("请输入数字")
    except KeyboardInterrupt:
        print("\n已取消")


if __name__ == "__main__":
    asyncio.run(main())
