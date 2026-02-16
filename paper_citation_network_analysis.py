# -*- coding: utf-8 -*-
"""
论文引用网络分析 - Xiaohao Cai 论文
基于方法继承的引用关系分析

生成内容：
1. 邻接矩阵
2. PageRank 排名
3. 社区检测
4. Graphviz DOT 可视化
5. NetworkX 绘图代码
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# ============================================================
# 1. 定义论文节点和引用关系（基于方法继承）
# ============================================================

papers = {
    # ID: (短名, 年份, 研究领域)
    "P01": ("Two-Stage Segmentation", 2013, "Segmentation"),
    "P02": ("Iterated ROF (T-ROF)", 2013, "Segmentation"),
    "P03": ("Tight-Frame Vessel", 2013, "Medical"),
    "P04": ("Variational Seg-Rest", 2015, "Segmentation"),
    "P05": ("SLaT Three-stage", 2015, "Segmentation"),
    "P06": ("Wavelet on Sphere", 2016, "Mathematical"),
    "P07": ("3D Tree Graph Cut", 2017, "RemoteSensing"),
    "P08": ("Radio Imaging I", 2017, "Astronomy"),
    "P09": ("Radio Imaging II", 2017, "Astronomy"),
    "P10": ("Online Radio Imaging", 2017, "Astronomy"),
    "P11": ("MS-ROF Linkage", 2018, "Mathematical"),
    "P12": ("3D Tree MCGC", 2019, "RemoteSensing"),
    "P13": ("High-dim Classification", 2019, "MachineLearning"),
    "P14": ("3D Orientation Field", 2020, "RemoteSensing"),
    "P15": ("Proximal Nested Sampling", 2021, "Astronomy"),
    "P16": ("Tucker Sketching", 2023, "MachineLearning"),
    "P17": ("Semantic Proportions", 2023, "Segmentation"),
    "P18": ("Few-shot Medical", 2023, "Medical"),
    "P19": ("Tensor Train", 2023, "MachineLearning"),
    "P20": ("Medical Report (IIHT)", 2023, "Medical"),
    "P21": ("TransNet HAR", 2023, "MachineLearning"),
    "P22": ("Non-negative Subspace", 2024, "MachineLearning"),
    "P23": ("Diffusion Brain MRI", 2024, "Medical"),
    "P24": ("CNNs RNNs Transformers", 2024, "MachineLearning"),
    "P25": ("Cross-Domain LiDAR", 2024, "RemoteSensing"),
    "P26": ("Talk2Radar", 2025, "Multimodal"),
    "P27": ("Neural Varifolds", 2025, "3DProcessing"),
    "P28": ("GAMED FakeNews", 2025, "Multimodal"),
    "P29": ("tCURLoRA", 2025, "MachineLearning"),
    "P30": ("Concept-Based XAI", 2025, "ExplainableAI"),
    "P31": ("LL4G Personality", 2025, "NLP"),
    "P32": ("CornerPoint3D", 2025, "3DProcessing"),
    "P33": ("HiFi-Mamba MRI", 2025, "Medical"),
    "P34": ("MOGO 3D Motion", 2025, "3DProcessing"),
    "P35": ("SaT Overview", 2023, "Segmentation"),
}

# 引用关系 (被引用者 -> 引用者): A -> B 表示 B 引用/继承 A 的方法
citations = [
    # SaT 方法演进
    ("P01", "P02"),  # Two-Stage -> Iterated ROF
    ("P01", "P04"),  # Two-Stage -> Variational Seg-Rest
    ("P01", "P05"),  # Two-Stage -> SLaT
    ("P02", "P11"),  # T-ROF -> MS-ROF Linkage
    ("P04", "P05"),  # Variational Seg-Rest -> SLaT
    ("P01", "P35"),  # Two-Stage -> SaT Overview
    ("P02", "P35"),  # T-ROF -> SaT Overview
    ("P05", "P35"),  # SLaT -> SaT Overview
    ("P04", "P35"),  # Variational -> SaT Overview
    # 理论联系
    ("P02", "P04"),  # T-ROF -> Variational
    ("P11", "P17"),  # MS-ROF -> Semantic Proportions (弱监督思想)
    # Tight-Frame 方法演进
    ("P03", "P06"),  # Tight-Frame -> Wavelet on Sphere
    # 3D 树分割演进
    ("P07", "P12"),  # Graph Cut -> MCGC
    ("P12", "P14"),  # MCGC -> 3D Orientation Field
    ("P14", "P25"),  # 3D Orientation -> Cross-Domain LiDAR
    # 射电天文系列
    ("P08", "P09"),  # Radio I -> Radio II
    ("P08", "P10"),  # Radio I -> Online
    ("P09", "P10"),  # Radio II -> Online
    ("P10", "P15"),  # Online -> Proximal Nested
    # 机器学习/张量方法演进
    ("P16", "P19"),  # Tucker -> Tensor Train
    ("P16", "P29"),  # Tucker -> tCURLoRA
    ("P19", "P29"),  # Tensor Train -> tCURLoRA
    # Few-shot 学习演进
    ("P18", "P22"),  # Few-shot Medical -> Non-negative Subspace
    # 医学图像演进
    ("P03", "P18"),  # Tight-Frame -> Few-shot Medical
    ("P18", "P23"),  # Few-shot -> Diffusion MRI
    ("P23", "P33"),  # Diffusion MRI -> HiFi-Mamba
    ("P20", "P33"),  # Medical Report -> HiFi-Mamba
    # HAR/行为识别
    ("P21", "P24"),  # TransNet -> CNNs RNNs Transformers Survey
    # 多模态方法
    ("P24", "P26"),  # Survey -> Talk2Radar (语言+雷达)
    ("P26", "P28"),  # Talk2Radar -> GAMED (多模态思想)
    # 3D处理演进
    ("P14", "P27"),  # 3D Orientation -> Neural Varifolds
    ("P25", "P32"),  # LiDAR -> CornerPoint3D
    ("P27", "P34"),  # Neural Varifolds -> MOGO 3D Motion
    ("P32", "P34"),  # CornerPoint3D -> MOGO 3D Motion
    # XAI/可解释性
    ("P30", "P31"),  # XAI -> Personality Detection
]

# ============================================================
# 2. 构建邻接矩阵
# ============================================================

n = len(papers)
paper_ids = list(papers.keys())
id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

# 邻接矩阵 A[i][j] = 1 表示 i 被 j 引用（有边从 i 指向 j）
adj_matrix = np.zeros((n, n), dtype=int)

for src, dst in citations:
    i, j = id_to_idx[src], id_to_idx[dst]
    adj_matrix[i][j] = 1

print("=" * 60)
print("1. 邻接矩阵 (Adjacency Matrix)")
print("=" * 60)
print(f"矩阵维度: {n} x {n}")
print(f"边数: {len(citations)}")
print()

# 创建 DataFrame 显示
df_adj = pd.DataFrame(
    adj_matrix,
    index=[f"{pid}" for pid in paper_ids],
    columns=[f"{pid}" for pid in paper_ids],
)
print("邻接矩阵 (部分展示):")
print(df_adj.iloc[:10, :10])
print()

# ============================================================
# 3. 计算 PageRank
# ============================================================


def compute_pagerank(adj_matrix, damping=0.85, max_iter=100, tol=1e-6):
    """计算 PageRank 值"""
    n = adj_matrix.shape[0]

    # 转置矩阵（因为 A[i][j]=1 表示 i->j，我们计算入度）
    # PageRank 需要的是：节点 j 指向哪些节点
    # 所以使用转置后的邻接矩阵

    # 出度：每行求和
    out_degree = adj_matrix.sum(axis=1)

    # 构建转移矩阵
    M = np.zeros((n, n))
    for i in range(n):
        if out_degree[i] > 0:
            M[i, :] = adj_matrix[i, :] / out_degree[i]

    # PageRank 迭代
    pr = np.ones(n) / n

    for iteration in range(max_iter):
        pr_new = (1 - damping) / n + damping * M.T @ pr
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new

    return pr


pagerank_scores = compute_pagerank(adj_matrix)

print("=" * 60)
print("2. PageRank 排名 (Top 10)")
print("=" * 60)

pr_ranking = sorted(zip(paper_ids, pagerank_scores), key=lambda x: x[1], reverse=True)

for rank, (pid, score) in enumerate(pr_ranking[:10], 1):
    name, year, field = papers[pid]
    print(f"{rank:2d}. {pid} | {name:25s} | PR={score:.4f} | {year} | {field}")

print()

# ============================================================
# 4. 社区检测 (基于标签传播)
# ============================================================


def label_propagation(adj_matrix, max_iter=100):
    """标签传播社区检测"""
    n = adj_matrix.shape[0]

    # 构建无向图
    undirected = adj_matrix + adj_matrix.T
    undirected = (undirected > 0).astype(int)

    # 初始化：每个节点有唯一标签
    labels = np.arange(n)

    for iteration in range(max_iter):
        changed = False
        order = np.random.permutation(n)

        for node in order:
            neighbors = np.where(undirected[node] > 0)[0]

            if len(neighbors) == 0:
                continue

            # 统计邻居标签
            neighbor_labels = labels[neighbors]
            unique, counts = np.unique(neighbor_labels, return_counts=True)

            # 选择最常见的标签
            max_count = counts.max()
            candidates = unique[counts == max_count]
            new_label = np.random.choice(candidates)

            if labels[node] != new_label:
                labels[node] = new_label
                changed = True

        if not changed:
            break

    return labels


np.random.seed(42)
communities = label_propagation(adj_matrix)

print("=" * 60)
print("3. 社区检测结果")
print("=" * 60)

# 组织社区
community_dict = defaultdict(list)
for i, label in enumerate(communities):
    community_dict[label].append(paper_ids[i])

# 重命名社区编号
sorted_communities = sorted(
    community_dict.items(), key=lambda x: len(x[1]), reverse=True
)

# 社区命名
community_names = {
    0: "SaT分割方法",
    1: "医学图像处理",
    2: "射电天文",
    3: "3D遥感",
    4: "机器学习/张量",
    5: "多模态/NLP",
    6: "孤立节点",
}

for idx, (label, members) in enumerate(sorted_communities):
    if len(members) > 0:
        print(f"\n社区 {idx + 1} ({len(members)} 篇):")
        for pid in members:
            name, year, field = papers[pid]
            print(f"  - {pid}: {name} ({year}) [{field}]")

print()

# ============================================================
# 5. Graphviz DOT 可视化代码
# ============================================================

print("=" * 60)
print("4. Graphviz DOT 可视化代码")
print("=" * 60)

dot_code = """// 论文引用网络 - Graphviz DOT 格式
// 使用方法: dot -Tpng citation_network.dot -o citation_network.png

digraph CitationNetwork {
    // 全局设置
    graph [
        rankdir=TB,
        splines=true,
        nodesep=0.6,
        ranksep=0.8,
        fontname="Arial",
        label="Xiaohao Cai 论文引用网络\\n(基于方法继承)",
        labelloc=t,
        fontsize=20
    ];
    node [
        shape=box,
        style="rounded,filled",
        fontname="Arial",
        fontsize=10,
        width=2.5,
        height=0.6
    ];
    edge [
        color="#666666",
        arrowsize=0.8,
        fontname="Arial",
        fontsize=8
    ];

    // ===== 子图：SaT分割方法社区 =====
    subgraph cluster_segmentation {
        label="SaT分割方法";
        style=filled;
        color="#E8F5E9";
        node [fillcolor="#4CAF50", fontcolor=white];
        
        P01 [label="Two-Stage\\n(2013)"];
        P02 [label="Iterated ROF\\n(2013)"];
        P04 [label="Var Seg-Rest\\n(2015)"];
        P05 [label="SLaT\\n(2015)"];
        P11 [label="MS-ROF Linkage\\n(2018)"];
        P17 [label="Semantic Prop\\n(2023)"];
        P35 [label="SaT Overview\\n(2023)"];
    }

    // ===== 子图：医学图像社区 =====
    subgraph cluster_medical {
        label="医学图像处理";
        style=filled;
        color="#E3F2FD";
        node [fillcolor="#2196F3", fontcolor=white];
        
        P03 [label="Tight-Frame\\n(2013)"];
        P06 [label="Wavelet Sphere\\n(2016)"];
        P18 [label="Few-shot Med\\n(2023)"];
        P20 [label="Med Report\\n(2023)"];
        P23 [label="Diffusion MRI\\n(2024)"];
        P33 [label="HiFi-Mamba\\n(2025)"];
    }

    // ===== 子图：射电天文社区 =====
    subgraph cluster_astronomy {
        label="射电天文";
        style=filled;
        color="#FFF3E0";
        node [fillcolor="#FF9800", fontcolor=white];
        
        P08 [label="Radio I\\n(2017)"];
        P09 [label="Radio II\\n(2017)"];
        P10 [label="Online Radio\\n(2017)"];
        P15 [label="Proximal Nested\\n(2021)"];
    }

    // ===== 子图：3D遥感社区 =====
    subgraph cluster_remote {
        label="3D遥感/点云";
        style=filled;
        color="#F3E5F5";
        node [fillcolor="#9C27B0", fontcolor=white];
        
        P07 [label="3D Tree GC\\n(2017)"];
        P12 [label="3D Tree MCGC\\n(2019)"];
        P14 [label="3D Orient\\n(2020)"];
        P25 [label="Cross-LiDAR\\n(2024)"];
        P27 [label="Neural Var\\n(2025)"];
        P32 [label="CornerPoint3D\\n(2025)"];
        P34 [label="MOGO 3D\\n(2025)"];
    }

    // ===== 子图：机器学习社区 =====
    subgraph cluster_ml {
        label="机器学习/张量";
        style=filled;
        color="#FFEBEE";
        node [fillcolor="#F44336", fontcolor=white];
        
        P13 [label="High-dim Class\\n(2019)"];
        P16 [label="Tucker Sketch\\n(2023)"];
        P19 [label="Tensor Train\\n(2023)"];
        P21 [label="TransNet HAR\\n(2023)"];
        P22 [label="Non-neg Sub\\n(2024)"];
        P24 [label="CNN/RNN/Trans\\n(2024)"];
        P29 [label="tCURLoRA\\n(2025)"];
    }

    // ===== 子图：多模态社区 =====
    subgraph cluster_multimodal {
        label="多模态/NLP/XAI";
        style=filled;
        color="#E0F7FA";
        node [fillcolor="#00BCD4", fontcolor=white];
        
        P26 [label="Talk2Radar\\n(2025)"];
        P28 [label="GAMED\\n(2025)"];
        P30 [label="XAI Concept\\n(2025)"];
        P31 [label="LL4G Person\\n(2025)"];
    }

    // ===== 引用关系边 =====
    // SaT 方法演进
    P01 -> P02 [label="扩展"];
    P01 -> P04 [label="框架"];
    P01 -> P05 [label="扩展"];
    P02 -> P11 [label="理论"];
    P04 -> P05 [label="结合"];
    P01 -> P35 [label="综述"];
    P02 -> P35 [label="综述"];
    P05 -> P35 [label="综述"];
    P04 -> P35 [label="综述"];
    P11 -> P17 [label="弱监督"];
    
    // Tight-Frame
    P03 -> P06 [label="推广"];
    
    // 3D 树
    P07 -> P12 [label="改进"];
    P12 -> P14 [label="扩展"];
    P14 -> P25 [label="跨域"];
    
    // 射电天文
    P08 -> P09 [label="续篇"];
    P08 -> P10 [label="在线"];
    P09 -> P10 [label="在线"];
    P10 -> P15 [label="采样"];
    
    // 张量方法
    P16 -> P19 [label="近似"];
    P16 -> P29 [label="分解"];
    P19 -> P29 [label="LoRA"];
    
    // Few-shot
    P18 -> P22 [label="子空间"];
    
    // 医学
    P03 -> P18 [label="应用"];
    P18 -> P23 [label="MRI"];
    P23 -> P33 [label="重建"];
    P20 -> P33 [label="报告"];
    
    // HAR
    P21 -> P24 [label="综述"];
    
    // 多模态
    P24 -> P26 [label="多模"];
    P26 -> P28 [label="融合"];
    
    // 3D处理
    P14 -> P27 [label="3D"];
    P25 -> P32 [label="检测"];
    P27 -> P34 [label="运动"];
    P32 -> P34 [label="运动"];
    
    // XAI
    P30 -> P31 [label="检测"];
}
"""

print(dot_code)

# ============================================================
# 6. Python NetworkX 绘图代码
# ============================================================

print("=" * 60)
print("5. Python NetworkX 绘图代码")
print("=" * 60)

networkx_code = '''# -*- coding: utf-8 -*-
"""
论文引用网络可视化 - NetworkX 版本
运行此代码生成网络图
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建有向图
G = nx.DiGraph()

# 添加节点（论文）
papers = {
    'P01': {'name': 'Two-Stage', 'year': 2013, 'field': 'Segmentation'},
    'P02': {'name': 'Iterated ROF', 'year': 2013, 'field': 'Segmentation'},
    'P03': {'name': 'Tight-Frame', 'year': 2013, 'field': 'Medical'},
    'P04': {'name': 'Var Seg-Rest', 'year': 2015, 'field': 'Segmentation'},
    'P05': {'name': 'SLaT', 'year': 2015, 'field': 'Segmentation'},
    'P06': {'name': 'Wavelet Sphere', 'year': 2016, 'field': 'Mathematical'},
    'P07': {'name': '3D Tree GC', 'year': 2017, 'field': 'RemoteSensing'},
    'P08': {'name': 'Radio I', 'year': 2017, 'field': 'Astronomy'},
    'P09': {'name': 'Radio II', 'year': 2017, 'field': 'Astronomy'},
    'P10': {'name': 'Online Radio', 'year': 2017, 'field': 'Astronomy'},
    'P11': {'name': 'MS-ROF Linkage', 'year': 2018, 'field': 'Mathematical'},
    'P12': {'name': '3D Tree MCGC', 'year': 2019, 'field': 'RemoteSensing'},
    'P13': {'name': 'High-dim Class', 'year': 2019, 'field': 'MachineLearning'},
    'P14': {'name': '3D Orient', 'year': 2020, 'field': 'RemoteSensing'},
    'P15': {'name': 'Proximal Nested', 'year': 2021, 'field': 'Astronomy'},
    'P16': {'name': 'Tucker Sketch', 'year': 2023, 'field': 'MachineLearning'},
    'P17': {'name': 'Semantic Prop', 'year': 2023, 'field': 'Segmentation'},
    'P18': {'name': 'Few-shot Med', 'year': 2023, 'field': 'Medical'},
    'P19': {'name': 'Tensor Train', 'year': 2023, 'field': 'MachineLearning'},
    'P20': {'name': 'Med Report', 'year': 2023, 'field': 'Medical'},
    'P21': {'name': 'TransNet HAR', 'year': 2023, 'field': 'MachineLearning'},
    'P22': {'name': 'Non-neg Sub', 'year': 2024, 'field': 'MachineLearning'},
    'P23': {'name': 'Diffusion MRI', 'year': 2024, 'field': 'Medical'},
    'P24': {'name': 'CNN/RNN/Trans', 'year': 2024, 'field': 'MachineLearning'},
    'P25': {'name': 'Cross-LiDAR', 'year': 2024, 'field': 'RemoteSensing'},
    'P26': {'name': 'Talk2Radar', 'year': 2025, 'field': 'Multimodal'},
    'P27': {'name': 'Neural Var', 'year': 2025, 'field': '3DProcessing'},
    'P28': {'name': 'GAMED', 'year': 2025, 'field': 'Multimodal'},
    'P29': {'name': 'tCURLoRA', 'year': 2025, 'field': 'MachineLearning'},
    'P30': {'name': 'XAI Concept', 'year': 2025, 'field': 'ExplainableAI'},
    'P31': {'name': 'LL4G Person', 'year': 2025, 'field': 'NLP'},
    'P32': {'name': 'CornerPoint3D', 'year': 2025, 'field': '3DProcessing'},
    'P33': {'name': 'HiFi-Mamba', 'year': 2025, 'field': 'Medical'},
    'P34': {'name': 'MOGO 3D', 'year': 2025, 'field': '3DProcessing'},
    'P35': {'name': 'SaT Overview', 'year': 2023, 'field': 'Segmentation'},
}

# 添加节点
for pid, attrs in papers.items():
    G.add_node(pid, **attrs)

# 添加边（引用关系）
citations = [
    ('P01', 'P02'), ('P01', 'P04'), ('P01', 'P05'), ('P02', 'P11'),
    ('P04', 'P05'), ('P01', 'P35'), ('P02', 'P35'), ('P05', 'P35'),
    ('P04', 'P35'), ('P11', 'P17'), ('P02', 'P04'),
    ('P03', 'P06'), ('P07', 'P12'), ('P12', 'P14'), ('P14', 'P25'),
    ('P08', 'P09'), ('P08', 'P10'), ('P09', 'P10'), ('P10', 'P15'),
    ('P16', 'P19'), ('P16', 'P29'), ('P19', 'P29'),
    ('P18', 'P22'), ('P03', 'P18'), ('P18', 'P23'), ('P23', 'P33'),
    ('P20', 'P33'), ('P21', 'P24'), ('P24', 'P26'), ('P26', 'P28'),
    ('P14', 'P27'), ('P25', 'P32'), ('P27', 'P34'), ('P32', 'P34'),
    ('P30', 'P31'),
]

G.add_edges_from(citations)

# 颜色映射
field_colors = {
    'Segmentation': '#4CAF50',      # 绿色
    'Medical': '#2196F3',           # 蓝色
    'Astronomy': '#FF9800',         # 橙色
    'RemoteSensing': '#9C27B0',     # 紫色
    'MachineLearning': '#F44336',   # 红色
    'Mathematical': '#795548',      # 棕色
    'Multimodal': '#00BCD4',        # 青色
    '3DProcessing': '#673AB7',      # 深紫色
    'ExplainableAI': '#009688',     # 青绿色
    'NLP': '#607D8B',               # 蓝灰色
}

# 计算 PageRank 用于节点大小
pagerank = nx.pagerank(G, alpha=0.85)
node_sizes = [pagerank[node] * 8000 + 200 for node in G.nodes()]

# 节点颜色
node_colors = [field_colors[papers[node]['field']] for node in G.nodes()]

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# 使用层级布局
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

# 绘制边
nx.draw_networkx_edges(G, pos, ax=ax,
                       edge_color='#AAAAAA',
                       arrows=True,
                       arrowsize=15,
                       alpha=0.6,
                       width=1.5,
                       connectionstyle='arc3,rad=0.1')

# 绘制节点
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.9,
                       edgecolors='white',
                       linewidths=2)

# 绘制标签
labels = {node: papers[node]['name'] for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, ax=ax,
                        font_size=8,
                        font_weight='bold')

# 添加图例
legend_patches = [
    mpatches.Patch(color='#4CAF50', label='SaT分割方法'),
    mpatches.Patch(color='#2196F3', label='医学图像'),
    mpatches.Patch(color='#FF9800', label='射电天文'),
    mpatches.Patch(color='#9C27B0', label='3D遥感'),
    mpatches.Patch(color='#F44336', label='机器学习'),
    mpatches.Patch(color='#00BCD4', label='多模态/NLP'),
]
ax.legend(handles=legend_patches, loc='upper left', fontsize=10)

# 添加标题
ax.set_title('Xiaohao Cai 论文引用网络\\n(节点大小=PageRank值)', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('D:/Documents/zx/citation_network.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

print("网络图已保存至: D:/Documents/zx/citation_network.png")

# ===== 附加：时间线视图 =====
fig2, ax2 = plt.subplots(1, 1, figsize=(18, 8))

# 按年份排列
pos_timeline = {}
for node in G.nodes():
    year = papers[node]['year']
    # 获取同年份论文的排名
    same_year = [n for n in G.nodes() if papers[n]['year'] == year]
    idx = same_year.index(node)
    pos_timeline[node] = (idx * 0.5, year)

nx.draw_networkx_edges(G, pos_timeline, ax=ax2,
                       edge_color='#AAAAAA',
                       arrows=True,
                       arrowsize=10,
                       alpha=0.5,
                       width=1)

nx.draw_networkx_nodes(G, pos_timeline, ax=ax2,
                       node_color=node_colors,
                       node_size=node_sizes,
                       alpha=0.9,
                       edgecolors='white',
                       linewidths=2)

nx.draw_networkx_labels(G, pos_timeline, labels, ax=ax2,
                        font_size=7,
                        font_weight='bold')

ax2.set_title('论文引用网络 - 时间线视图', fontsize=14, fontweight='bold')
ax2.set_xlabel('论文（同年份）')
ax2.set_ylabel('年份')
ax2.set_ylim(2012, 2026)
ax2.legend(handles=legend_patches, loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/Documents/zx/citation_network_timeline.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

print("时间线图已保存至: D:/Documents/zx/citation_network_timeline.png")
'''

print(networkx_code)

# ============================================================
# 7. 保存 DOT 文件
# ============================================================

with open("D:/Documents/zx/citation_network.dot", "w", encoding="utf-8") as f:
    f.write(dot_code)

print()
print("=" * 60)
print("文件已生成:")
print("  - D:/Documents/zx/citation_network.dot")
print("  - D:/Documents/zx/paper_citation_network_analysis.py")
print("=" * 60)

# ============================================================
# 8. 统计摘要
# ============================================================

print()
print("=" * 60)
print("6. 网络统计摘要")
print("=" * 60)

# 入度和出度
in_degree = adj_matrix.sum(axis=0)
out_degree = adj_matrix.sum(axis=1)

print(f"\n节点总数: {n}")
print(f"边总数: {len(citations)}")

print("\n被引用最多 (入度):")
top_cited = sorted(zip(paper_ids, in_degree), key=lambda x: x[1], reverse=True)[:5]
for pid, deg in top_cited:
    print(f"  {pid}: {papers[pid][0]} - 被引用 {deg} 次")

print("\n引用最多 (出度):")
top_citing = sorted(zip(paper_ids, out_degree), key=lambda x: x[1], reverse=True)[:5]
for pid, deg in top_citing:
    print(f"  {pid}: {papers[pid][0]} - 引用 {deg} 篇")

# 按年份统计
year_counts = defaultdict(int)
for pid, (name, year, field) in papers.items():
    year_counts[year] += 1

print("\n按年份统计:")
for year in sorted(year_counts.keys()):
    print(f"  {year}: {year_counts[year]} 篇")

# 按领域统计
field_counts = defaultdict(int)
for pid, (name, year, field) in papers.items():
    field_counts[field] += 1

print("\n按领域统计:")
for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {field}: {count} 篇")
