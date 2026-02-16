# Xiaohao Cai 论文知识图谱

基于 68 篇论文构建的完整知识图谱，涵盖变分方法、图像分割、张量分解、深度学习等领域。

## 图谱架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     知识图谱实体关系                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     USES       ┌──────────┐                      │
│   │  Paper   │───────────────▶│  Method  │                      │
│   └──────────┘                └──────────┘                      │
│        │                           │                            │
│        │ APPLIES_TO                │ INHERITS_FROM              │
│        ▼                           ▼                            │
│   ┌──────────┐                ┌──────────┐                      │
│   │  Domain  │                │  Method  │                      │
│   └──────────┘                └──────────┘                      │
│        │                                                      │
│        │ WRITES                                                │
│        ▼                                                      │
│   ┌──────────┐                                                │
│   │  Author  │                                                │
│   └──────────┘                                                │
│        │                                                      │
│        │ CITES                                                 │
│        ▼                                                      │
│   ┌──────────┐                                                │
│   │  Paper   │                                                │
│   └──────────┘                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 文件说明

| 文件 | 格式 | 用途 |
|------|------|------|
| `entities.json` | JSON | 实体定义（论文、作者、方法、领域、工具） |
| `relations.json` | JSON | 关系定义 |
| `ontology.ttl` | RDF/Turtle | OWL本体定义 |
| `knowledge_graph.jsonld` | JSON-LD | 完整图谱数据（语义网格式） |
| `neo4j_import.cypher` | Cypher | Neo4j图数据库导入脚本 |
| `queries.py` | Python | 查询示例代码 |

## 实体统计

| 实体类型 | 数量 |
|----------|------|
| 论文 (Paper) | 58 |
| 作者 (Author) | 8 |
| 方法 (Method) | 24 |
| 领域 (Domain) | 12 |
| 工具 (Tool) | 10 |

## 方法类别

| 类别 | 方法示例 |
|------|----------|
| Variational | ROF, Mumford-Shah |
| Segmentation | SLaT, SaT, T-ROF |
| Tensor | Tucker, Tensor-Train, CUR |
| Deep Learning | Mamba, Diffusion, LoRA |
| Optimization | Split-Bregman, Primal-Dual |

## 快速开始

### Python 查询

```python
from queries import KnowledgeGraphQuery

kg = KnowledgeGraphQuery()

# 1. 找出所有使用ROF方法的论文
papers = kg.find_papers_by_method('ROF')

# 2. 找出SLaT方法的演进路径
evolution = kg.find_method_evolution('SLaT')

# 3. 找出2020年后3D视觉论文
papers_3d = kg.find_papers_after_year(2020, domain='3D Vision')

# 4. 找出使用变分方法的医学图像论文
var_med = kg.find_variational_medical_papers()

# 5. 获取统计信息
stats = kg.get_statistics()
```

### Neo4j 导入

```bash
# 1. 启动 Neo4j
neo4j start

# 2. 运行导入脚本
cypher-shell -u neo4j -p password -f neo4j_import.cypher

# 3. 执行查询示例
cypher-shell -u neo4j -p password "
MATCH (p:Paper)-[:USES]->(m:Method)
WHERE m.name = 'ROF'
RETURN p.title, p.year
ORDER BY p.year
"
```

## 查询示例

### 1. 找出所有使用变分方法的医学图像论文

**Cypher:**
```cypher
MATCH (p:Paper)-[:USES]->(m:Method)
WHERE m.category = 'Variational'
AND (p)-[:APPLIES_TO]->(:Domain {name: 'Medical Imaging'})
RETURN p.title, p.year, collect(m.name) as methods
ORDER BY p.year;
```

**Python:**
```python
kg.find_variational_medical_papers()
```

### 2. ROF 方法的完整演进路径

**Cypher:**
```cypher
MATCH (m:Method)
WHERE m.name IN ['ROF', 'T-ROF', 'SaT', 'SLaT']
OPTIONAL MATCH (m)-[:INHERITS_FROM]->(parent:Method)
RETURN m.name, m.category, parent.name as inherits_from;
```

**Python:**
```python
kg.find_method_evolution('ROF')
```

**结果:**
```
ROF → T-ROF (Thresholded ROF)
ROF → SaT → SLaT (Smoothing, Lifting, Thresholding)
```

### 3. 2020年后发表的3D视觉论文

**Cypher:**
```cypher
MATCH (p:Paper)-[:APPLIES_TO]->(d:Domain)
WHERE d.name CONTAINS '3D'
AND p.year > 2020
RETURN p.title, p.year, p.arxiv
ORDER BY p.year;
```

**Python:**
```python
kg.find_papers_after_year(2020, domain='3D Vision')
```

## 方法演进图

```
                    ┌─────────────┐
                    │ Mumford-Shah│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌─────│     ROF     │─────┐
              │     └─────────────┘     │
              │                         │
       ┌──────▼──────┐           ┌──────▼──────┐
       │    T-ROF    │           │     SaT     │
       └─────────────┘           └──────┬──────┘
                                        │
                                 ┌──────▼──────┐
                                 │    SLaT     │
                                 └─────────────┘
```

## 领域分布

| 领域 | 论文数 |
|------|--------|
| Medical Imaging | 8 |
| Remote Sensing | 4 |
| Radio Astronomy | 3 |
| 3D Vision | 6 |
| Personality Detection | 5 |
| HAR | 3 |
| Motion Generation | 2 |

## 技术栈

- **存储格式**: JSON-LD, RDF/Turtle
- **图数据库**: Neo4j
- **查询语言**: Cypher, SPARQL
- **编程接口**: Python

## 参考

- 论文来源: Xiaohao Cai 完整文献库 (2011-2026)
- 本体设计: 基于 DCMI 和 Schema.org 标准
