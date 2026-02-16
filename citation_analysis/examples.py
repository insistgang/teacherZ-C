"""
使用示例：论文引用分析系统
"""

from citation_analyzer import (
    CitationAnalyzer, Paper, Reference,
    InfluenceMetrics, TrendAnalyzer
)
import numpy as np


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===\n")
    
    analyzer = CitationAnalyzer(output_dir="./my_analysis")
    
    paper1 = Paper(
        paper_id="paper_001",
        title="Deep Learning for Image Recognition",
        authors=["Zhang San", "Li Si"],
        year=2020,
        abstract="This paper presents a novel deep learning approach...",
        keywords=["deep learning", "CNN", "image recognition"],
        citations_received=150,
        citation_years={2020: 20, 2021: 50, 2022: 80}
    )
    
    paper2 = Paper(
        paper_id="paper_002",
        title="Attention Mechanism in Neural Networks",
        authors=["Wang Wu", "Zhao Liu"],
        year=2019,
        keywords=["attention", "transformer", "neural network"],
        citations_received=300,
        citation_years={2019: 30, 2020: 100, 2021: 170}
    )
    
    paper1.references.append(Reference(
        ref_id="ref_1",
        authors=["Wang Wu", "Zhao Liu"],
        title="Attention Mechanism in Neural Networks",
        year=2019
    ))
    
    analyzer.add_papers([paper1, paper2])
    
    adj_matrix = analyzer.build_citation_matrix()
    print(f"引用矩阵:\n{adj_matrix}\n")
    
    pagerank = analyzer.compute_pagerank()
    print(f"PageRank: {pagerank}\n")
    
    h_index = analyzer.compute_h_index()
    print(f"h-index: {h_index}\n")
    
    return analyzer


def example_pdf_extraction():
    """PDF提取示例"""
    print("\n=== PDF提取示例 ===\n")
    
    from citation_analyzer import ReferenceExtractor
    
    extractor = ReferenceExtractor()
    
    pdf_path = "path/to/your/paper.pdf"
    
    print("使用方法:")
    print(f"  text, references = extractor.extract_from_pdf('{pdf_path}')")
    print("  for ref in references:")
    print("      print(f'  - {ref.title} ({ref.year})')\n")


def example_network_analysis():
    """网络分析示例"""
    print("\n=== 网络分析示例 ===\n")
    
    analyzer = CitationAnalyzer()
    
    papers = []
    for i in range(10):
        paper = Paper(
            paper_id=f"p{i}",
            title=f"Research Paper {i} on Machine Learning",
            authors=[f"Author {i}A", f"Author {i}B"],
            year=2018 + i % 5,
            keywords=["machine learning", f"topic_{i % 3}"],
            citations_received=np.random.randint(10, 200)
        )
        papers.append(paper)
    
    papers[0].references = [
        Reference(ref_id="r1", title="Research Paper 5 on Machine Learning", year=2020),
        Reference(ref_id="r2", title="Research Paper 3 on Machine Learning", year=2019)
    ]
    papers[1].references = [
        Reference(ref_id="r3", title="Research Paper 0 on Machine Learning", year=2018)
    ]
    
    analyzer.add_papers(papers)
    
    adj_matrix = analyzer.build_citation_matrix()
    print(f"构建了 {adj_matrix.shape[0]}x{adj_matrix.shape[1]} 引用矩阵\n")
    
    pagerank = analyzer.compute_pagerank()
    top_3 = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]
    print("PageRank Top 3:")
    for pid, score in top_3:
        print(f"  {pid}: {score:.4f}")
    print()
    
    hubs, authorities = analyzer.compute_hits()
    top_auth = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Authority Top 3:")
    for pid, score in top_auth:
        print(f"  {pid}: {score:.4f}")
    print()
    
    communities = analyzer.detect_communities()
    print(f"检测到 {len(set(communities.values()))} 个社区\n")
    
    return analyzer


def example_influence_metrics():
    """影响力指标示例"""
    print("\n=== 影响力指标示例 ===\n")
    
    citations = [100, 80, 60, 50, 40, 30, 25, 20, 15, 10, 8, 5, 3, 2, 1]
    
    h_index = InfluenceMetrics.compute_h_index(citations)
    g_index = InfluenceMetrics.compute_g_index(citations)
    i10_index = InfluenceMetrics.compute_i10_index(citations)
    
    print(f"引用列表: {citations}")
    print(f"h-index: {h_index}  (有{h_index}篇论文至少被引用{h_index}次)")
    print(f"g-index: {g_index}  (前{g_index}篇论文总引用 >= {g_index}²)")
    print(f"i10-index: {i10_index}  (被引用>=10次的论文数)\n")
    
    percentile = InfluenceMetrics.compute_citation_percentile(50, citations)
    print(f"引用50次的百分位: {percentile:.1f}%\n")


def example_trend_prediction():
    """趋势预测示例"""
    print("\n=== 趋势预测示例 ===\n")
    
    analyzer = CitationAnalyzer()
    
    paper = Paper(
        paper_id="trend_example",
        title="Trending Research Topic",
        authors=["Author"],
        year=2019,
        citations_received=200,
        citation_years={
            2019: 20,
            2020: 40,
            2021: 60,
            2022: 80
        }
    )
    
    analyzer.add_paper(paper)
    
    print("历史引用数据:")
    for year, count in sorted(paper.citation_years.items()):
        print(f"  {year}: {count}")
    print()
    
    predictions = analyzer.predict_citation_trend("trend_example", years=3)
    print("预测未来3年引用:")
    for year, pred in sorted(predictions.items()):
        print(f"  {year}: {pred:.1f}")
    print()


def example_full_pipeline():
    """完整流程示例"""
    print("\n=== 完整分析流程 ===\n")
    
    from citation_analyzer import create_sample_data
    
    analyzer = CitationAnalyzer(output_dir="./full_analysis")
    
    papers, _ = create_sample_data()
    analyzer.add_papers(papers)
    
    results = analyzer.analyze_all()
    
    print(f"分析完成!")
    print(f"  - 论文数: {results['paper_count']}")
    print(f"  - h-index: {results['metrics']['h_index']}")
    print(f"  - 社区数: {results['communities']['num_communities']}")
    print(f"  - 热点数: {len(results['hotspots'])}")
    print()
    
    report_path = analyzer.generate_report()
    print(f"报告已生成: {report_path}")
    
    figures = analyzer.generate_visualizations()
    print(f"可视化图表:")
    for name, path in figures.items():
        print(f"  - {name}: {path}")
    
    return analyzer


def example_custom_visualization():
    """自定义可视化示例"""
    print("\n=== 自定义可视化示例 ===\n")
    
    from citation_analyzer import CitationVisualizer
    
    visualizer = CitationVisualizer(output_dir="./my_figures")
    
    history = {2018: 10, 2019: 25, 2020: 50, 2021: 80, 2022: 120}
    predictions = {2023: 140, 2024: 155, 2025: 165}
    
    print("可视化工具使用方法:")
    print("  path = visualizer.plot_citation_trend(")
    print("      history=history,")
    print("      predictions=predictions,")
    print("      title='My Citation Trend',")
    print("      output_file='my_trend.png'")
    print("  )\n")


if __name__ == "__main__":
    print("=" * 60)
    print("论文引用分析系统 - 使用示例")
    print("=" * 60)
    
    example_basic_usage()
    example_influence_metrics()
    example_trend_prediction()
    example_network_analysis()
    example_full_pipeline()
    
    print("\n" + "=" * 60)
    print("示例运行完成!")
    print("=" * 60)
