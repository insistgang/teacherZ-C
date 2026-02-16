"""
多智能体论文辩论系统 - 使用示例

快速启动脚本，展示如何使用辩论系统
"""

import asyncio
from debate_system import (
    DebateSystem,
    Paper,
    PaperParser,
    MockLLMClient,
    ClaudeLLMClient
)


# 示例1：使用模拟LLM进行快速测试
async def example_with_mock():
    """使用模拟LLM的示例"""
    print("=" * 60)
    print("示例1: 使用模拟LLM进行辩论")
    print("=" * 60 + "\n")

    # 创建示例论文
    paper = Paper(
        title="基于 Mumford-Shah 泛函的凸松弛图像分割方法",
        authors=["Xiaohao Cai", "其他作者"],
        year=2015,
        abstract="""
        本文提出了一种基于Mumford-Shah泛函的凸松弛方法用于图像分割。
        通过将原始非凸问题转化为凸优化问题，我们能够获得全局最优解。
        方法在医学图像分割任务上表现出色。
        """,
        content="""
# 1. 引言

Mumford-Shah泛函是图像分割的经典方法，但原始问题是非凸的，
容易陷入局部最优。

# 2. 方法

## 2.1 Mumford-Shah泛函

原始MS泛函定义为：

E(u, K) = ∫_Ω\\K |∇u|^2 dx + μ ∫_Ω|f-u|^2 dx + ν H^{d-1}(K)

其中：
- u 是分段光滑函数
- K 是边缘集
- H^{d-1} 是Hausdorff测度

## 2.2 凸松弛

我们提出以下凸松弛形式：

min_{u∈BV(Ω)} ∫_Ω |∇u| + λ ∫_Ω (u-f)^2 dx

## 2.3 数值算法

使用原始对偶算法求解：
- 原始步：u^{k+1} = (I + λ∂F)^{-1}(f - p^k)
- 对偶步：p^{k+1} = (p^k + τ∇u^{k+1})/(1+τ|∇u^{k+1}|)

# 3. 实验

在医学图像数据集上评估：
- 脑部MRI分割：Dice系数0.92
- 前列腺CT分割：Dice系数0.88

# 4. 结论

本文提出的凸松弛方法有效解决了MS泛函的非凸问题。
        """
    )

    # 创建系统（使用模拟LLM）
    system = DebateSystem()
    system.initialize()

    # 执行辩论
    state = await system.debate_paper(paper)

    print(f"\n辩论完成！")
    print(f"- 轮次: {state.current_round}")
    print(f"- 发言数: {len(state.messages)}")
    print(f"- 共识度: {state.consensus_score:.2f}")


# 示例2：从文件加载论文
async def example_from_file(file_path: str):
    """从文件加载论文进行辩论"""
    print("\n" + "=" * 60)
    print("示例2: 从文件加载论文进行辩论")
    print("=" * 60 + "\n")

    system = DebateSystem()
    system.initialize()

    state = await system.debate_from_file(file_path)

    print(f"\n辩论完成！共识度: {state.consensus_score:.2f}")


# 示例3：使用真实LLM
async def example_with_real_llm():
    """使用真实Claude API的示例"""
    print("\n" + "=" * 60)
    print("示例3: 使用Claude API进行辩论")
    print("=" * 60 + "\n")

    try:
        # 创建Claude客户端
        llm_client = ClaudeLLMClient()

        # 创建系统
        system = DebateSystem(llm_client=llm_client)
        system.initialize()

        # 使用示例论文
        paper = Paper(
            title="测试论文：深度学习在医学图像分析中的应用",
            authors=["测试作者"],
            year=2024,
            abstract="这是一篇测试论文的摘要",
            content="这是一篇测试论文的内容..."
        )

        state = await system.debate_paper(paper)

        print(f"\n辩论完成！共识度: {state.consensus_score:.2f}")

    except ValueError as e:
        print(f"错误: {e}")
        print("请设置 ANTHROPIC_API_KEY 环境变量")


# 示例4：自定义Agent配置
async def example_custom_config():
    """使用自定义配置的示例"""
    print("\n" + "=" * 60)
    print("示例4: 自定义配置")
    print("=" * 60 + "\n")

    from debate_system import load_config, AgentFactory, MockLLMClient, DebateScheduler

    # 加载并修改配置
    config = load_config()
    config.max_rounds = 3  # 减少轮次

    # 禁用某些Agent
    config.agents["application_expert"].enabled = False

    # 创建Agent
    llm_client = MockLLMClient()
    agents = AgentFactory.create_all_agents(llm_client, config)

    print(f"启用的Agent: {[a.name for a in agents]}")

    # 创建调度器
    scheduler = DebateScheduler(agents, config, llm_client)

    # 创建论文
    paper = Paper(
        title="配置测试论文",
        authors=["测试"],
        year=2024,
        abstract="测试摘要",
        content="测试内容"
    )

    # 运行辩论
    state = await scheduler.start_debate(paper)

    print(f"\n辩论完成！")


# 主函数
async def main():
    """运行所有示例"""
    print("多智能体论文辩论系统 - 使用示例\n")

    # 示例1：模拟LLM
    await example_with_mock()

    # 示例2：从文件（如果存在）
    # await example_from_file("path/to/paper.pdf")

    # 示例3：真实LLM（需要API密钥）
    # await example_with_real_llm()

    # 示例4：自定义配置
    # await example_custom_config()


if __name__ == "__main__":
    asyncio.run(main())
