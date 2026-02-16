"""
论文引用分析系统
包含：引用提取、网络分析、影响力指标、时序分析、可视化
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PDF处理
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# 网络分析
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# 机器学习
try:
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 可视化
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class Reference:
    """参考文献数据结构"""
    ref_id: str
    authors: List[str] = field(default_factory=list)
    title: str = ""
    year: Optional[int] = None
    journal: str = ""
    volume: str = ""
    pages: str = ""
    doi: str = ""
    raw_text: str = ""
    
    def to_dict(self):
        return asdict(self)


@dataclass
class Paper:
    """论文数据结构"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    citations_received: int = 0
    citation_years: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self):
        data = asdict(self)
        data['references'] = [r.to_dict() for r in self.references]
        return data


class ReferenceExtractor:
    """从PDF提取参考文献"""
    
    # 常见引用格式正则
    PATTERNS = {
        'apa': re.compile(
            r'([A-Z][a-zA-Z\-]+(?:,\s*[A-Z][a-zA-Z\-]+)*(?:,\s*&\s*[A-Z][a-zA-Z\-]+)?)\s*'
            r'\((\d{4})\)\.\s*'
            r'([^.]+(?:\.[^.]{0,20})?)\.\s*'
            r'([^.]+),\s*(\d+)(?:\((\d+)\))?,\s*(\d+-\d+)',
            re.MULTILINE
        ),
        'ieee': re.compile(
            r'\[(\d+)\]\s*([^.]+),\s*"([^"]+)",\s*'
            r'([^.]+),\s*vol\.\s*(\d+),\s*no\.\s*(\d+),\s*'
            r'pp\.\s*(\d+-\d+),\s*(\d{4})',
            re.MULTILINE
        ),
        'generic': re.compile(
            r'([A-Z][a-zA-Z\s,\.&]+?)\s*\(?(\d{4})\)?\s*\.?\s*'
            r'([^.]+)\.\s*([^.]+(?:Journal|Conference|Proceedings|arXiv)[^.]*)',
            re.MULTILINE | re.IGNORECASE
        )
    }
    
    def __init__(self):
        self.section_keywords = [
            'references', 'bibliography', 'references cited',
            '参考文献', 'literature cited'
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> Tuple[str, List[Reference]]:
        """从PDF提取文本和参考文献"""
        if not HAS_PYMUPDF:
            raise ImportError("需要安装 PyMuPDF: pip install pymupdf")
        
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page in doc:
            full_text += page.get_text()
        
        doc.close()
        
        references = self._parse_references(full_text)
        return full_text, references
    
    def _find_references_section(self, text: str) -> str:
        """定位参考文献部分"""
        text_lower = text.lower()
        
        for keyword in self.section_keywords:
            pattern = re.compile(
                rf'\n\s*{keyword}\s*\n',
                re.IGNORECASE
            )
            match = pattern.search(text_lower)
            if match:
                return text[match.end():]
        
        return ""
    
    def _parse_references(self, text: str) -> List[Reference]:
        """解析参考文献"""
        ref_section = self._find_references_section(text)
        if not ref_section:
            ref_section = text
        
        references = []
        
        for fmt_name, pattern in self.PATTERNS.items():
            matches = pattern.finditer(ref_section)
            for i, match in enumerate(matches):
                ref = self._create_reference(match.groups(), fmt_name, i)
                if ref and ref.title:
                    references.append(ref)
        
        if not references:
            references = self._fallback_parse(ref_section)
        
        return references
    
    def _create_reference(self, groups: Tuple, fmt: str, idx: int) -> Optional[Reference]:
        """根据格式创建参考文献"""
        try:
            if fmt == 'apa':
                authors = self._parse_authors(groups[0])
                year = int(groups[1]) if groups[1] else None
                title = groups[2].strip()
                journal = groups[3].strip()
                return Reference(
                    ref_id=f"ref_{idx}",
                    authors=authors,
                    title=title,
                    year=year,
                    journal=journal,
                    volume=groups[4],
                    pages=groups[6],
                    raw_text=' '.join(g for g in groups if g)
                )
            elif fmt == 'ieee':
                authors = self._parse_authors(groups[1])
                title = groups[2].strip()
                journal = groups[3].strip()
                year = int(groups[7]) if groups[7] else None
                return Reference(
                    ref_id=f"ref_{idx}",
                    authors=authors,
                    title=title,
                    year=year,
                    journal=journal,
                    volume=groups[4],
                    pages=groups[6],
                    raw_text=' '.join(g for g in groups if g)
                )
            else:
                authors = self._parse_authors(groups[0])
                year = int(groups[1]) if groups[1] else None
                title = groups[2].strip()
                journal = groups[3].strip() if len(groups) > 3 else ""
                return Reference(
                    ref_id=f"ref_{idx}",
                    authors=authors,
                    title=title,
                    year=year,
                    journal=journal,
                    raw_text=' '.join(g for g in groups if g)
                )
        except (ValueError, IndexError):
            return None
    
    def _parse_authors(self, author_str: str) -> List[str]:
        """解析作者列表"""
        author_str = author_str.strip()
        author_str = re.sub(r'\s+', ' ', author_str)
        
        separators = [';', ',', ' and ', ' & ', '、']
        authors = [author_str]
        
        for sep in separators:
            new_authors = []
            for a in authors:
                new_authors.extend(a.split(sep))
            authors = new_authors
        
        authors = [a.strip() for a in authors if a.strip()]
        return authors
    
    def _fallback_parse(self, text: str) -> List[Reference]:
        """备用的简单解析"""
        lines = text.split('\n')
        references = []
        
        ref_pattern = re.compile(r'^\s*\[?(\d+)\]?\s*(.+)', re.MULTILINE)
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 20:
                continue
            
            year_match = re.search(r'\b(19|20)\d{2}\b', line)
            year = int(year_match.group()) if year_match else None
            
            ref = Reference(
                ref_id=f"ref_{len(references)}",
                raw_text=line,
                year=year
            )
            references.append(ref)
        
        return references


class CitationNetwork:
    """引用网络分析"""
    
    def __init__(self):
        self.papers: Dict[str, Paper] = {}
        self.graph = None
        self.adj_matrix = None
        self.paper_to_idx: Dict[str, int] = {}
        self.idx_to_paper: Dict[int, str] = {}
    
    def add_paper(self, paper: Paper):
        """添加论文到网络"""
        self.papers[paper.paper_id] = paper
        self._rebuild_indices()
    
    def add_papers(self, papers: List[Paper]):
        """批量添加论文"""
        for paper in papers:
            self.papers[paper.paper_id] = paper
        self._rebuild_indices()
    
    def _rebuild_indices(self):
        """重建索引映射"""
        self.paper_to_idx = {pid: i for i, pid in enumerate(self.papers.keys())}
        self.idx_to_paper = {i: pid for pid, i in self.paper_to_idx.items()}
    
    def build_citation_matrix(self) -> np.ndarray:
        """构建引用矩阵
        
        adj_matrix[i][j] = 1 表示论文i引用了论文j
        """
        n = len(self.papers)
        self.adj_matrix = np.zeros((n, n), dtype=np.float64)
        
        title_to_id = {}
        for pid, paper in self.papers.items():
            title_key = paper.title.lower().strip()
            if title_key:
                title_to_id[title_key] = pid
            for author in paper.authors:
                author_key = author.lower().strip()
                if author_key and len(author_key) > 3:
                    title_to_id[author_key] = pid
        
        for pid, paper in self.papers.items():
            if pid not in self.paper_to_idx:
                continue
            i = self.paper_to_idx[pid]
            
            for ref in paper.references:
                matched_id = self._match_reference(ref, title_to_id, pid)
                
                if matched_id and matched_id in self.paper_to_idx:
                    j = self.paper_to_idx[matched_id]
                    self.adj_matrix[i][j] = 1.0
        
        return self.adj_matrix
    
    def _match_reference(self, ref: Reference, title_to_id: Dict, source_pid: str) -> Optional[str]:
        """匹配参考文献到已有论文"""
        if ref.title:
            title_key = ref.title.lower().strip()
            if title_key in title_to_id:
                matched = title_to_id[title_key]
                if matched != source_pid:
                    return matched
        
        for author in ref.authors:
            author_key = author.lower().strip()
            if author_key in title_to_id:
                matched = title_to_id[author_key]
                if matched != source_pid:
                    return matched
        
        return None
    
    def build_networkx_graph(self):
        """构建NetworkX图"""
        if not HAS_NETWORKX:
            raise ImportError("需要安装 networkx: pip install networkx")
        
        self.graph = nx.DiGraph()
        
        for pid, paper in self.papers.items():
            self.graph.add_node(
                pid,
                title=paper.title,
                year=paper.year,
                authors=', '.join(paper.authors[:3])
            )
        
        if self.adj_matrix is None:
            self.build_citation_matrix()
        
        n = len(self.papers)
        for i in range(n):
            for j in range(n):
                if self.adj_matrix[i][j] > 0:
                    self.graph.add_edge(
                        self.idx_to_paper[i],
                        self.idx_to_paper[j]
                    )
        
        return self.graph
    
    def compute_pagerank(self, damping: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """计算PageRank值"""
        if self.adj_matrix is None:
            self.build_citation_matrix()
        
        n = len(self.papers)
        if n == 0:
            return {}
        
        out_degree = self.adj_matrix.sum(axis=1)
        out_degree[out_degree == 0] = 1
        
        transition = self.adj_matrix / out_degree[:, np.newaxis]
        
        teleport = np.ones(n) / n
        
        pr = np.ones(n) / n
        
        for _ in range(max_iter):
            new_pr = (1 - damping) * teleport + damping * transition.T @ pr
            
            if np.allclose(pr, new_pr, atol=1e-8):
                break
            pr = new_pr
        
        return {self.idx_to_paper[i]: float(pr[i]) for i in range(n)}
    
    def compute_hits(self, max_iter: int = 100) -> Tuple[Dict[str, float], Dict[str, float]]:
        """计算HITS算法的Hub和Authority值"""
        if self.adj_matrix is None:
            self.build_citation_matrix()
        
        n = len(self.papers)
        if n == 0:
            return {}, {}
        
        hubs = np.ones(n) / np.sqrt(n)
        authorities = np.ones(n) / np.sqrt(n)
        
        for _ in range(max_iter):
            new_auth = self.adj_matrix.T @ hubs
            new_hubs = self.adj_matrix @ new_auth
            
            auth_norm = np.linalg.norm(new_auth)
            hub_norm = np.linalg.norm(new_hubs)
            
            if auth_norm > 0:
                authorities = new_auth / auth_norm
            if hub_norm > 0:
                hubs = new_hubs / hub_norm
        
        hub_scores = {self.idx_to_paper[i]: float(hubs[i]) for i in range(n)}
        auth_scores = {self.idx_to_paper[i]: float(authorities[i]) for i in range(n)}
        
        return hub_scores, auth_scores
    
    def detect_communities_louvain(self) -> Dict[str, int]:
        """使用Louvain算法进行社区检测"""
        if not HAS_NETWORKX:
            raise ImportError("需要安装 networkx: pip install networkx")
        
        if self.graph is None:
            self.build_networkx_graph()
        
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph.to_undirected())
            return partition
        except ImportError:
            return self._label_propagation_communities()
    
    def _label_propagation_communities(self) -> Dict[str, int]:
        """标签传播社区检测（不依赖community库）"""
        if self.graph is None:
            self.build_networkx_graph()
        
        undirected = self.graph.to_undirected()
        
        labels = {node: i for i, node in enumerate(undirected.nodes())}
        
        for _ in range(100):
            changed = False
            nodes = list(undirected.nodes())
            np.random.shuffle(nodes)
            
            for node in nodes:
                neighbors = list(undirected.neighbors(node))
                if not neighbors:
                    continue
                
                neighbor_labels = [labels[n] for n in neighbors]
                label_counts = Counter(neighbor_labels)
                max_count = max(label_counts.values())
                most_common = [l for l, c in label_counts.items() if c == max_count]
                
                new_label = most_common[0]
                if labels[node] != new_label:
                    labels[node] = new_label
                    changed = True
            
            if not changed:
                break
        
        unique_labels = {label: i for i, label in enumerate(set(labels.values()))}
        return {node: unique_labels[label] for node, label in labels.items()}
    
    def compute_centrality(self) -> Dict[str, Dict[str, float]]:
        """计算各种中心性指标"""
        if not HAS_NETWORKX:
            return self._compute_basic_centrality()
        
        if self.graph is None:
            self.build_networkx_graph()
        
        centrality = {}
        
        try:
            degree = nx.degree_centrality(self.graph)
            betweenness = nx.betweenness_centrality(self.graph)
            closeness = nx.closeness_centrality(self.graph)
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
            
            for node in self.graph.nodes():
                centrality[node] = {
                    'degree': degree.get(node, 0),
                    'betweenness': betweenness.get(node, 0),
                    'closeness': closeness.get(node, 0),
                    'eigenvector': eigenvector.get(node, 0)
                }
        except:
            return self._compute_basic_centrality()
        
        return centrality
    
    def _compute_basic_centrality(self) -> Dict[str, Dict[str, float]]:
        """计算基础中心性（无networkx）"""
        if self.adj_matrix is None:
            self.build_citation_matrix()
        
        n = len(self.papers)
        centrality = {}
        
        in_degree = self.adj_matrix.sum(axis=0)
        out_degree = self.adj_matrix.sum(axis=1)
        
        max_in = in_degree.max() if in_degree.max() > 0 else 1
        max_out = out_degree.max() if out_degree.max() > 0 else 1
        
        for pid in self.papers:
            i = self.paper_to_idx[pid]
            centrality[pid] = {
                'in_degree': float(in_degree[i]),
                'out_degree': float(out_degree[i]),
                'degree': float(in_degree[i] + out_degree[i]) / (2 * (n - 1)) if n > 1 else 0
            }
        
        return centrality


class InfluenceMetrics:
    """影响力指标计算"""
    
    @staticmethod
    def compute_h_index(citations: List[int]) -> int:
        """计算h-index
        
        h-index: 有h篇论文至少被引用h次
        """
        if not citations:
            return 0
        
        sorted_citations = sorted(citations, reverse=True)
        h_index = 0
        
        for i, c in enumerate(sorted_citations, 1):
            if c >= i:
                h_index = i
            else:
                break
        
        return h_index
    
    @staticmethod
    def compute_g_index(citations: List[int]) -> int:
        """计算g-index
        
        g-index: 前g篇论文的总引用次数 >= g^2
        """
        if not citations:
            return 0
        
        sorted_citations = sorted(citations, reverse=True)
        g_index = 0
        
        for i in range(1, len(sorted_citations) + 1):
            if sum(sorted_citations[:i]) >= i ** 2:
                g_index = i
            else:
                break
        
        return g_index
    
    @staticmethod
    def compute_i10_index(citations: List[int]) -> int:
        """计算i10-index: 被引用超过10次的论文数"""
        return sum(1 for c in citations if c >= 10)
    
    @staticmethod
    def compute_impact_factor(citations_per_year: Dict[int, int], 
                             current_year: int, 
                             window: int = 2) -> float:
        """计算影响因子（模拟期刊影响因子）
        
        IF = 近window年引用数 / 近window年论文数
        """
        recent_citations = sum(
            count for year, count in citations_per_year.items()
            if current_year - year < window
        )
        
        return float(recent_citations)
    
    @staticmethod
    def compute_citation_percentile(citations: int, all_citations: List[int]) -> float:
        """计算引用百分位"""
        if not all_citations:
            return 0.0
        
        below = sum(1 for c in all_citations if c < citations)
        return (below / len(all_citations)) * 100


class TrendAnalyzer:
    """时序分析和趋势预测"""
    
    def __init__(self):
        self.time_series_data = {}
    
    def build_time_series(self, papers: Dict[str, Paper]) -> pd.DataFrame:
        """构建时间序列数据"""
        records = []
        
        for pid, paper in papers.items():
            for year, count in paper.citation_years.items():
                records.append({
                    'paper_id': pid,
                    'year': year,
                    'citations': count
                })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        return df
    
    def predict_citation_trend(self, 
                               citation_history: Dict[int, int],
                               future_years: int = 3,
                               method: str = 'exp_smoothing') -> Dict[int, float]:
        """预测引用趋势"""
        if not citation_history:
            return {}
        
        years = sorted(citation_history.keys())
        citations = [citation_history[y] for y in years]
        
        if len(years) < 2:
            return {years[-1] + i: citations[-1] for i in range(1, future_years + 1)}
        
        if method == 'exp_smoothing':
            return self._exp_smoothing_predict(years, citations, future_years)
        elif method == 'linear':
            return self._linear_predict(years, citations, future_years)
        else:
            return self._arima_predict(years, citations, future_years)
    
    def _exp_smoothing_predict(self, years: List[int], citations: List[int], 
                               future_years: int, alpha: float = 0.3) -> Dict[int, float]:
        """指数平滑预测"""
        smoothed = [citations[0]]
        
        for c in citations[1:]:
            smoothed.append(alpha * c + (1 - alpha) * smoothed[-1])
        
        last_smooth = smoothed[-1]
        trend = (smoothed[-1] - smoothed[0]) / (len(smoothed) - 1) if len(smoothed) > 1 else 0
        
        predictions = {}
        last_year = years[-1]
        
        for i in range(1, future_years + 1):
            predictions[last_year + i] = max(0, last_smooth + trend * i)
        
        return predictions
    
    def _linear_predict(self, years: List[int], citations: List[int], 
                        future_years: int) -> Dict[int, float]:
        """线性回归预测"""
        x = np.array(years)
        y = np.array(citations)
        
        n = len(x)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x ** 2).sum()
        
        denom = n * sum_x2 - sum_x ** 2
        if denom == 0:
            slope = 0
            intercept = sum_y / n
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
        
        predictions = {}
        last_year = years[-1]
        
        for i in range(1, future_years + 1):
            predictions[last_year + i] = max(0, slope * (last_year + i) + intercept)
        
        return predictions
    
    def _arima_predict(self, years: List[int], citations: List[int],
                       future_years: int) -> Dict[int, float]:
        """简单AR预测（自回归）"""
        if len(citations) < 3:
            return self._linear_predict(years, citations, future_years)
        
        recent_avg = np.mean(citations[-3:])
        predictions = {}
        last_year = years[-1]
        
        for i in range(1, future_years + 1):
            decay = 0.9 ** i
            predictions[last_year + i] = max(0, recent_avg * decay)
        
        return predictions
    
    def detect_hotspots(self, papers: Dict[str, Paper], 
                       time_window: int = 3) -> List[Dict]:
        """检测研究热点"""
        current_year = datetime.now().year
        recent_papers = []
        
        for pid, paper in papers.items():
            if paper.year and current_year - paper.year <= time_window:
                recent_papers.append(paper)
        
        if not recent_papers:
            return []
        
        keyword_freq = Counter()
        for paper in recent_papers:
            for kw in paper.keywords:
                keyword_freq[kw.lower()] += 1
            
            words = re.findall(r'\b[a-zA-Z]{4,}\b', paper.title.lower())
            keyword_freq.update(words)
        
        stopwords = {'study', 'analysis', 'research', 'paper', 'using', 'based', 
                    'approach', 'method', 'model', 'system', 'new', 'novel'}
        
        for sw in stopwords:
            keyword_freq.pop(sw, None)
        
        hotspots = []
        for kw, freq in keyword_freq.most_common(20):
            related = [p for p in recent_papers if kw in p.title.lower()]
            hotspots.append({
                'keyword': kw,
                'frequency': freq,
                'growth_rate': self._calc_growth_rate(related),
                'papers': [p.paper_id for p in related[:5]]
            })
        
        return sorted(hotspots, key=lambda x: x['growth_rate'], reverse=True)
    
    def _calc_growth_rate(self, papers: List[Paper]) -> float:
        """计算增长率"""
        if len(papers) < 2:
            return 0.0
        
        year_counts = Counter(p.year for p in papers if p.year)
        if len(year_counts) < 2:
            return 0.0
        
        years = sorted(year_counts.keys())
        if len(years) < 2:
            return 0.0
        
        recent = sum(year_counts[y] for y in years[-2:])
        earlier = sum(year_counts[y] for y in years[:-2]) if len(years) > 2 else 1
        
        return (recent - earlier) / earlier if earlier > 0 else float(recent)


class TopicEvolution:
    """主题演化分析"""
    
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self.topic_model = None
        self.vectorizer = None
    
    def extract_topics(self, papers: Dict[str, Paper]) -> Dict[str, List[Tuple[int, float]]]:
        """提取论文主题分布"""
        if not HAS_SKLEARN:
            return self._keyword_based_topics(papers)
        
        texts = []
        paper_ids = []
        
        for pid, paper in papers.items():
            text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}"
            texts.append(text)
            paper_ids.append(pid)
        
        if not texts:
            return {}
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf = self.vectorizer.fit_transform(texts)
            
            self.topic_model = NMF(
                n_components=min(self.n_topics, len(texts)),
                random_state=42,
                max_iter=200
            )
            
            doc_topics = self.topic_model.fit_transform(tfidf)
            
            result = {}
            for i, pid in enumerate(paper_ids):
                topics = [(j, float(doc_topics[i, j])) 
                         for j in range(doc_topics.shape[1])]
                topics = sorted(topics, key=lambda x: x[1], reverse=True)[:3]
                result[pid] = topics
            
            return result
        except:
            return self._keyword_based_topics(papers)
    
    def _keyword_based_topics(self, papers: Dict[str, Paper]) -> Dict[str, List[Tuple[int, float]]]:
        """基于关键词的主题分配"""
        all_keywords = []
        for paper in papers.values():
            all_keywords.extend([kw.lower() for kw in paper.keywords])
        
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(self.n_topics)]
        
        result = {}
        for pid, paper in papers.items():
            topics = []
            for i, kw in enumerate(top_keywords):
                if kw in [k.lower() for k in paper.keywords]:
                    topics.append((i, 1.0))
            result[pid] = topics[:3] if topics else [(0, 0.5)]
        
        return result
    
    def get_topic_words(self, top_n: int = 10) -> Dict[int, List[str]]:
        """获取每个主题的代表性词语"""
        if self.topic_model is None or self.vectorizer is None:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for i, topic in enumerate(self.topic_model.components_):
            top_indices = topic.argsort()[-top_n:][::-1]
            topics[i] = [feature_names[j] for j in top_indices]
        
        return topics
    
    def track_topic_evolution(self, papers: Dict[str, Paper]) -> pd.DataFrame:
        """追踪主题随时间演化"""
        paper_topics = self.extract_topics(papers)
        
        records = []
        for pid, paper in papers.items():
            if not paper.year:
                continue
            
            for topic_id, weight in paper_topics.get(pid, []):
                records.append({
                    'year': paper.year,
                    'topic': topic_id,
                    'weight': weight
                })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        evolution = df.groupby(['year', 'topic'])['weight'].sum().reset_index()
        
        return evolution.pivot(index='year', columns='topic', values='weight').fillna(0)


class CitationVisualizer:
    """可视化模块"""
    
    def __init__(self, output_dir: str = "./figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = plt.cm.tab10.colors
    
    def plot_citation_network(self, 
                              network: CitationNetwork,
                              max_nodes: int = 100,
                              output_file: str = "citation_network.png"):
        """绘制引用网络图"""
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            print("需要安装 matplotlib 和 networkx")
            return None
        
        if network.graph is None:
            network.build_networkx_graph()
        
        G = network.graph
        
        if len(G.nodes()) > max_nodes:
            pagerank = nx.pagerank(G)
            top_nodes = sorted(pagerank.keys(), key=lambda x: pagerank[x], 
                             reverse=True)[:max_nodes]
            G = G.subgraph(top_nodes)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        pagerank = nx.pagerank(G)
        node_sizes = [3000 * pagerank.get(node, 0.01) + 100 for node in G.nodes()]
        
        communities = network.detect_communities_louvain()
        node_colors = [communities.get(node, 0) for node in G.nodes()]
        
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, 
                              arrowsize=10, ax=ax)
        
        scatter = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                        node_color=node_colors, 
                                        cmap=plt.cm.Set3, alpha=0.8, ax=ax)
        
        labels = {node: G.nodes[node].get('title', node)[:20] + '...' 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
        
        ax.set_title('Citation Network', fontsize=16)
        ax.axis('off')
        
        filepath = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_temporal_heatmap(self, 
                             papers: Dict[str, Paper],
                             output_file: str = "temporal_heatmap.png"):
        """绘制时序热力图"""
        if not HAS_MATPLOTLIB:
            print("需要安装 matplotlib")
            return None
        
        years = [p.year for p in papers.values() if p.year]
        if not years:
            return None
        
        min_year, max_year = min(years), max(years)
        year_range = list(range(min_year, max_year + 1))
        
        topics = TopicEvolution(n_topics=8)
        evolution = topics.track_topic_evolution(papers)
        
        if evolution.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(evolution.T, cmap='YlOrRd', annot=False, 
                   linewidths=0.5, ax=ax, cbar_kws={'label': 'Topic Weight'})
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Topic ID', fontsize=12)
        ax.set_title('Topic Evolution Over Time', fontsize=14)
        
        filepath = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_topic_stream(self,
                         papers: Dict[str, Paper],
                         output_file: str = "topic_stream.png"):
        """绘制主题河流图"""
        if not HAS_MATPLOTLIB:
            print("需要安装 matplotlib")
            return None
        
        topics = TopicEvolution(n_topics=6)
        evolution = topics.track_topic_evolution(papers)
        
        if evolution.empty:
            return None
        
        topic_words = topics.get_topic_words(top_n=3)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        years = evolution.index.values
        topic_data = evolution.values
        
        topic_data = np.maximum(topic_data, 0)
        
        cumulative = np.zeros(len(years))
        
        for i in range(topic_data.shape[1]):
            label = f"Topic {i}: " + ', '.join(topic_words.get(i, [])[:3])
            ax.fill_between(years, cumulative, cumulative + topic_data[:, i],
                          label=label, alpha=0.7)
            ax.plot(years, cumulative + topic_data[:, i], linewidth=0.5, color='white')
            cumulative += topic_data[:, i]
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Cumulative Topic Weight', fontsize=12)
        ax.set_title('Topic Stream Graph', fontsize=14)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        
        filepath = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_influence_ranking(self,
                               influence_scores: Dict[str, Dict[str, float]],
                               top_n: int = 20,
                               output_file: str = "influence_ranking.png"):
        """绘制影响力排名图"""
        if not HAS_MATPLOTLIB:
            print("需要安装 matplotlib")
            return None
        
        sorted_papers = sorted(influence_scores.items(),
                              key=lambda x: x[1].get('pagerank', 0),
                              reverse=True)[:top_n]
        
        if not sorted_papers:
            return None
        
        ids = [p[0][:15] + '...' for p in sorted_papers]
        scores = [p[1].get('pagerank', 0) for p in sorted_papers]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(ids)), scores, color=self.colors[0])
        
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids)
        ax.invert_yaxis()
        ax.set_xlabel('PageRank Score', fontsize=12)
        ax.set_title('Top Papers by PageRank', fontsize=14)
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', va='center', fontsize=8)
        
        filepath = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_citation_trend(self,
                           history: Dict[int, int],
                           predictions: Dict[int, float] = None,
                           title: str = "Citation Trend",
                           output_file: str = "citation_trend.png"):
        """绘制引用趋势图"""
        if not HAS_MATPLOTLIB:
            print("需要安装 matplotlib")
            return None
        
        years = sorted(history.keys())
        citations = [history[y] for y in years]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(years, citations, 'o-', color=self.colors[0], 
               linewidth=2, markersize=8, label='Actual')
        
        if predictions:
            pred_years = sorted(predictions.keys())
            pred_values = [predictions[y] for y in pred_years]
            
            ax.plot(pred_years, pred_values, '--', color=self.colors[1],
                   linewidth=2, marker='s', label='Predicted')
            
            ax.axvline(x=years[-1], color='gray', linestyle=':', alpha=0.5)
            ax.fill_between(pred_years, pred_values, alpha=0.2, color=self.colors[1])
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Citations', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        filepath = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_community_structure(self,
                                network: CitationNetwork,
                                output_file: str = "community_structure.png"):
        """绘制社区结构图"""
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            print("需要安装 matplotlib 和 networkx")
            return None
        
        if network.graph is None:
            network.build_networkx_graph()
        
        communities = network.detect_communities_louvain()
        
        community_sizes = Counter(communities.values())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        labels = [f'Community {c}' for c in community_sizes.keys()]
        sizes = list(community_sizes.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax1.set_title('Community Size Distribution', fontsize=12)
        
        ax2 = axes[1]
        G = network.graph
        pos = nx.spring_layout(G, seed=42)
        
        node_colors = [communities.get(node, 0) for node in G.nodes()]
        
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax2)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              cmap=plt.cm.Set3, node_size=50, ax=ax2)
        
        ax2.set_title('Network Community Structure', fontsize=12)
        ax2.axis('off')
        
        filepath = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(filepath)


class CitationAnalyzer:
    """论文引用分析主类"""
    
    def __init__(self, output_dir: str = "./analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = ReferenceExtractor()
        self.network = CitationNetwork()
        self.trend_analyzer = TrendAnalyzer()
        self.visualizer = CitationVisualizer(str(self.output_dir / "figures"))
        
        self.papers: Dict[str, Paper] = {}
        self.analysis_results: Dict = {}
    
    def extract_references(self, pdf_path: str) -> Tuple[str, List[Reference]]:
        """从PDF提取引用"""
        return self.extractor.extract_from_pdf(pdf_path)
    
    def add_paper(self, paper: Paper):
        """添加论文"""
        self.papers[paper.paper_id] = paper
        self.network.add_paper(paper)
    
    def add_papers(self, papers: List[Paper]):
        """批量添加论文"""
        for paper in papers:
            self.papers[paper.paper_id] = paper
        self.network.add_papers(papers)
    
    def build_citation_matrix(self) -> np.ndarray:
        """构建引用矩阵"""
        return self.network.build_citation_matrix()
    
    def compute_pagerank(self) -> Dict[str, float]:
        """计算PageRank"""
        return self.network.compute_pagerank()
    
    def compute_hits(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """计算HITS"""
        return self.network.compute_hits()
    
    def detect_communities(self) -> Dict[str, int]:
        """社区检测"""
        return self.network.detect_communities_louvain()
    
    def compute_h_index(self, author_papers: List[str] = None) -> int:
        """计算h-index"""
        if author_papers:
            citations = [self.papers[pid].citations_received 
                        for pid in author_papers if pid in self.papers]
        else:
            citations = [p.citations_received for p in self.papers.values()]
        
        return InfluenceMetrics.compute_h_index(citations)
    
    def predict_citation_trend(self, paper_id: str, years: int = 3) -> Dict[int, float]:
        """预测引用趋势"""
        if paper_id not in self.papers:
            return {}
        
        paper = self.papers[paper_id]
        return self.trend_analyzer.predict_citation_trend(paper.citation_years, years)
    
    def detect_hotspots(self, time_window: int = 3) -> List[Dict]:
        """检测研究热点"""
        return self.trend_analyzer.detect_hotspots(self.papers, time_window)
    
    def analyze_all(self) -> Dict:
        """执行完整分析"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'paper_count': len(self.papers),
            'metrics': {},
            'rankings': {},
            'communities': {},
            'hotspots': [],
            'trends': {}
        }
        
        print("构建引用矩阵...")
        adj_matrix = self.build_citation_matrix()
        results['metrics']['citation_density'] = float(
            adj_matrix.sum() / (len(self.papers) ** 2) if len(self.papers) > 0 else 0
        )
        
        print("计算PageRank...")
        pagerank = self.compute_pagerank()
        
        print("计算HITS...")
        hubs, authorities = self.compute_hits()
        
        print("计算中心性...")
        centrality = self.network.compute_centrality()
        
        print("检测社区...")
        communities = self.detect_communities()
        community_sizes = Counter(communities.values())
        results['communities'] = {
            'num_communities': len(community_sizes),
            'size_distribution': dict(community_sizes),
            'assignments': communities
        }
        
        print("计算影响力指标...")
        all_citations = [p.citations_received for p in self.papers.values()]
        
        results['metrics']['h_index'] = self.compute_h_index()
        results['metrics']['g_index'] = InfluenceMetrics.compute_g_index(all_citations)
        results['metrics']['i10_index'] = InfluenceMetrics.compute_i10_index(all_citations)
        results['metrics']['total_citations'] = sum(all_citations)
        results['metrics']['avg_citations'] = np.mean(all_citations) if all_citations else 0
        
        print("生成排名...")
        influence_scores = {}
        for pid in self.papers:
            influence_scores[pid] = {
                'pagerank': pagerank.get(pid, 0),
                'hub': hubs.get(pid, 0),
                'authority': authorities.get(pid, 0),
                'citations': self.papers[pid].citations_received,
                **centrality.get(pid, {})
            }
        
        results['rankings'] = {
            'by_pagerank': sorted(influence_scores.items(), 
                                 key=lambda x: x[1]['pagerank'], reverse=True)[:20],
            'by_authority': sorted(influence_scores.items(),
                                  key=lambda x: x[1]['authority'], reverse=True)[:20],
            'by_citations': sorted(influence_scores.items(),
                                  key=lambda x: x[1]['citations'], reverse=True)[:20]
        }
        
        print("检测热点...")
        results['hotspots'] = self.detect_hotspots()
        
        print("分析主题演化...")
        topics = TopicEvolution(n_topics=8)
        results['trends']['topic_evolution'] = topics.track_topic_evolution(self.papers).to_dict()
        results['trends']['topic_words'] = topics.get_topic_words()
        
        self.analysis_results = results
        self.influence_scores = influence_scores
        
        return results
    
    def generate_visualizations(self) -> Dict[str, str]:
        """生成所有可视化图表"""
        figures = {}
        
        print("生成引用网络图...")
        fig = self.visualizer.plot_citation_network(self.network)
        if fig:
            figures['citation_network'] = fig
        
        print("生成时序热力图...")
        fig = self.visualizer.plot_temporal_heatmap(self.papers)
        if fig:
            figures['temporal_heatmap'] = fig
        
        print("生成主题河流图...")
        fig = self.visualizer.plot_topic_stream(self.papers)
        if fig:
            figures['topic_stream'] = fig
        
        print("生成影响力排名图...")
        if hasattr(self, 'influence_scores'):
            fig = self.visualizer.plot_influence_ranking(self.influence_scores)
            if fig:
                figures['influence_ranking'] = fig
        
        print("生成社区结构图...")
        fig = self.visualizer.plot_community_structure(self.network)
        if fig:
            figures['community_structure'] = fig
        
        return figures
    
    def generate_report(self, filename: str = "citation_analysis_report.md") -> str:
        """生成分析报告"""
        if not self.analysis_results:
            self.analyze_all()
        
        report = []
        report.append("# 论文引用分析报告\n")
        report.append(f"生成时间: {self.analysis_results['timestamp']}\n\n")
        
        report.append("## 1. 数据概览\n")
        report.append(f"- 论文总数: {self.analysis_results['paper_count']}\n")
        report.append(f"- 引用矩阵密度: {self.analysis_results['metrics']['citation_density']:.4f}\n")
        report.append(f"- 总引用次数: {self.analysis_results['metrics']['total_citations']}\n")
        report.append(f"- 平均引用次数: {self.analysis_results['metrics']['avg_citations']:.2f}\n\n")
        
        report.append("## 2. 影响力指标\n")
        report.append(f"- h-index: {self.analysis_results['metrics']['h_index']}\n")
        report.append(f"- g-index: {self.analysis_results['metrics']['g_index']}\n")
        report.append(f"- i10-index: {self.analysis_results['metrics']['i10_index']}\n\n")
        
        report.append("## 3. 论文影响力排名 (Top 10 by PageRank)\n")
        for rank, (pid, scores) in enumerate(self.analysis_results['rankings']['by_pagerank'][:10], 1):
            paper = self.papers.get(pid)
            title = paper.title[:50] + '...' if paper and len(paper.title) > 50 else (paper.title if paper else pid)
            report.append(f"{rank}. **{title}**\n")
            report.append(f"   - PageRank: {scores['pagerank']:.4f}\n")
            report.append(f"   - 引用数: {scores['citations']}\n")
        report.append("\n")
        
        report.append("## 4. 研究社区结构\n")
        comm = self.analysis_results['communities']
        report.append(f"- 社区数量: {comm['num_communities']}\n")
        report.append(f"- 规模分布:\n")
        for cid, size in sorted(comm['size_distribution'].items()):
            report.append(f"  - 社区 {cid}: {size} 篇论文\n")
        report.append("\n")
        
        report.append("## 5. 研究热点\n")
        for i, hotspot in enumerate(self.analysis_results['hotspots'][:10], 1):
            report.append(f"{i}. **{hotspot['keyword']}**\n")
            report.append(f"   - 频次: {hotspot['frequency']}\n")
            report.append(f"   - 增长率: {hotspot['growth_rate']:.2%}\n")
        report.append("\n")
        
        report.append("## 6. 主题演化\n")
        if self.analysis_results['trends']['topic_words']:
            for tid, words in self.analysis_results['trends']['topic_words'].items():
                report.append(f"- 主题 {tid}: {', '.join(words[:5])}\n")
        report.append("\n")
        
        report_content = ''.join(report)
        
        filepath = self.output_dir / filename
        filepath.write_text(report_content, encoding='utf-8')
        
        return str(filepath)
    
    def save_results(self, filename: str = "analysis_results.json"):
        """保存分析结果"""
        filepath = self.output_dir / filename
        
        save_data = {
            'papers': {pid: p.to_dict() for pid, p in self.papers.items()},
            'analysis': self.analysis_results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        
        return str(filepath)
    
    def load_results(self, filename: str = "analysis_results.json"):
        """加载分析结果"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.papers = {}
        for pid, pdata in data['papers'].items():
            refs = [Reference(**r) for r in pdata.get('references', [])]
            paper = Paper(
                paper_id=pdata['paper_id'],
                title=pdata['title'],
                authors=pdata['authors'],
                year=pdata['year'],
                abstract=pdata.get('abstract', ''),
                keywords=pdata.get('keywords', []),
                references=refs,
                citations_received=pdata.get('citations_received', 0),
                citation_years=pdata.get('citation_years', {})
            )
            self.papers[pid] = paper
        
        self.network.add_papers(list(self.papers.values()))
        self.analysis_results = data['analysis']
        
        return data


def create_sample_data() -> List[Paper]:
    """创建示例数据用于测试"""
    papers_data = [
        {
            'id': 'p1', 'title': 'Deep Learning for Natural Language Processing',
            'authors': ['Smith J', 'Johnson A'], 'year': 2018,
            'keywords': ['deep learning', 'NLP', 'neural networks'],
            'citations': 150, 'citation_years': {2018: 20, 2019: 40, 2020: 50, 2021: 40}
        },
        {
            'id': 'p2', 'title': 'Transformer Architecture for Sequence Modeling',
            'authors': ['Vaswani A', 'Shazeer N'], 'year': 2017,
            'keywords': ['transformer', 'attention', 'sequence'],
            'citations': 500, 'citation_years': {2017: 50, 2018: 100, 2019: 150, 2020: 200}
        },
        {
            'id': 'p3', 'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'authors': ['Devlin J', 'Chang M'], 'year': 2019,
            'keywords': ['BERT', 'pre-training', 'transformer'],
            'citations': 300, 'citation_years': {2019: 80, 2020: 120, 2021: 100}
        },
        {
            'id': 'p4', 'title': 'Attention Is All You Need',
            'authors': ['Vaswani A', 'Shazeer N', 'Parmar N'], 'year': 2017,
            'keywords': ['attention', 'transformer', 'neural machine translation'],
            'citations': 800, 'citation_years': {2017: 100, 2018: 200, 2019: 250, 2020: 250}
        },
        {
            'id': 'p5', 'title': 'GPT-3: Language Models are Few-Shot Learners',
            'authors': ['Brown T', 'Mann B'], 'year': 2020,
            'keywords': ['GPT', 'language model', 'few-shot learning'],
            'citations': 400, 'citation_years': {2020: 150, 2021: 250}
        },
        {
            'id': 'p6', 'title': 'Image Recognition with Deep Convolutional Networks',
            'authors': ['Krizhevsky A', 'Hinton G'], 'year': 2012,
            'keywords': ['CNN', 'image recognition', 'deep learning'],
            'citations': 1000, 'citation_years': {2012: 50, 2013: 100, 2014: 150, 2015: 200, 2016: 200, 2017: 150, 2018: 150}
        },
        {
            'id': 'p7', 'title': 'ResNet: Deep Residual Learning',
            'authors': ['He K', 'Zhang X'], 'year': 2016,
            'keywords': ['ResNet', 'residual learning', 'CNN'],
            'citations': 700, 'citation_years': {2016: 100, 2017: 200, 2018: 200, 2019: 150, 2020: 50}
        },
        {
            'id': 'p8', 'title': 'Graph Neural Networks for Social Network Analysis',
            'authors': ['Wang H', 'Li Y'], 'year': 2019,
            'keywords': ['GNN', 'social network', 'graph learning'],
            'citations': 120, 'citation_years': {2019: 30, 2020: 50, 2021: 40}
        },
        {
            'id': 'p9', 'title': 'Knowledge Graph Embedding Methods',
            'authors': ['Bordes A', 'Usunier N'], 'year': 2013,
            'keywords': ['knowledge graph', 'embedding', 'TransE'],
            'citations': 450, 'citation_years': {2013: 30, 2014: 60, 2015: 80, 2016: 100, 2017: 80, 2018: 60, 2019: 40}
        },
        {
            'id': 'p10', 'title': 'Reinforcement Learning for Game Playing',
            'authors': ['Mnih V', 'Silver D'], 'year': 2015,
            'keywords': ['reinforcement learning', 'DQN', 'game AI'],
            'citations': 600, 'citation_years': {2015: 80, 2016: 150, 2017: 150, 2018: 120, 2019: 60, 2020: 40}
        }
    ]
    
    papers = []
    for data in papers_data:
        paper = Paper(
            paper_id=data['id'],
            title=data['title'],
            authors=data['authors'],
            year=data['year'],
            keywords=data['keywords'],
            citations_received=data['citations'],
            citation_years=data['citation_years']
        )
        papers.append(paper)
    
    citation_links = [
        ('p1', ['p2', 'p4']),
        ('p3', ['p2', 'p4']),
        ('p5', ['p2', 'p3', 'p4']),
        ('p7', ['p6']),
        ('p8', ['p7', 'p9']),
        ('p1', ['p6', 'p7'])
    ]
    
    return papers, citation_links


def main():
    """主函数 - 演示系统功能"""
    print("=" * 60)
    print("论文引用分析系统")
    print("=" * 60)
    
    analyzer = CitationAnalyzer(output_dir="./analysis_results")
    
    print("\n[1] 加载示例数据...")
    papers, citation_links = create_sample_data()
    
    for paper in papers:
        analyzer.add_paper(paper)
    
    print(f"已加载 {len(papers)} 篇论文")
    
    print("\n[2] 构建引用矩阵...")
    adj_matrix = analyzer.build_citation_matrix()
    print(f"引用矩阵形状: {adj_matrix.shape}")
    print(f"非零元素: {int(adj_matrix.sum())}")
    
    print("\n[3] 网络分析...")
    pagerank = analyzer.compute_pagerank()
    print(f"PageRank Top 3:")
    for pid, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  - {analyzer.papers[pid].title[:40]}... : {score:.4f}")
    
    hubs, authorities = analyzer.compute_hits()
    print(f"\nAuthority Top 3:")
    for pid, score in sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  - {analyzer.papers[pid].title[:40]}... : {score:.4f}")
    
    print("\n[4] 社区检测...")
    communities = analyzer.detect_communities()
    community_counts = Counter(communities.values())
    print(f"发现 {len(community_counts)} 个社区")
    for cid, count in community_counts.items():
        print(f"  - 社区 {cid}: {count} 篇论文")
    
    print("\n[5] 影响力指标...")
    h_index = analyzer.compute_h_index()
    citations = [p.citations_received for p in analyzer.papers.values()]
    g_index = InfluenceMetrics.compute_g_index(citations)
    i10_index = InfluenceMetrics.compute_i10_index(citations)
    
    print(f"h-index: {h_index}")
    print(f"g-index: {g_index}")
    print(f"i10-index: {i10_index}")
    
    print("\n[6] 趋势预测...")
    for pid in ['p2', 'p5']:
        trend = analyzer.predict_citation_trend(pid, years=3)
        paper = analyzer.papers[pid]
        print(f"\n{paper.title[:40]}...")
        print(f"  历史引用: {paper.citation_years}")
        print(f"  预测未来3年: {trend}")
    
    print("\n[7] 热点检测...")
    hotspots = analyzer.detect_hotspots(time_window=5)
    print("研究热点 (Top 5):")
    for i, hs in enumerate(hotspots[:5], 1):
        print(f"  {i}. {hs['keyword']} - 频次: {hs['frequency']}, 增长率: {hs['growth_rate']:.1%}")
    
    print("\n[8] 执行完整分析...")
    results = analyzer.analyze_all()
    
    print("\n[9] 生成可视化...")
    figures = analyzer.generate_visualizations()
    for name, path in figures.items():
        print(f"  - {name}: {path}")
    
    print("\n[10] 生成报告...")
    report_path = analyzer.generate_report()
    print(f"报告已保存: {report_path}")
    
    results_path = analyzer.save_results()
    print(f"结果已保存: {results_path}")
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
