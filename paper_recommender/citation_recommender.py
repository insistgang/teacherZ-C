"""
论文推荐系统 - 基于引用的推荐
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from data_loader import DataLoader, Paper
from content_recommender import RecommendationResult


@dataclass
class CitationNode:
    paper_id: str
    cited_by_count: int
    cites_count: int
    pagerank_score: float = 0.0
    is_foundational: bool = False


class CitationRecommender:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.citation_graph: dict[str, CitationNode] = {}
        self.influence_scores: dict[str, float] = {}
        self._build_graph()
        self._compute_pagerank()
        self._identify_foundational()

    def _build_graph(self) -> None:
        for paper_id, paper in self.loader.papers.items():
            self.citation_graph[paper_id] = CitationNode(
                paper_id=paper_id,
                cited_by_count=len(paper.cited_by),
                cites_count=len(paper.citations),
            )

    def _compute_pagerank(self, damping: float = 0.85, iterations: int = 20) -> None:
        nodes = list(self.citation_graph.keys())
        n = len(nodes)
        if n == 0:
            return

        scores = {node: 1.0 / n for node in nodes}

        for _ in range(iterations):
            new_scores: dict[str, float] = defaultdict(lambda: (1 - damping) / n)

            for node in nodes:
                paper = self.loader.papers.get(node)
                if not paper:
                    continue

                out_degree = len(paper.citations)
                if out_degree > 0:
                    contribution = damping * scores[node] / out_degree
                    for cited in paper.citations:
                        if cited in new_scores:
                            new_scores[cited] += contribution

            scores = dict(new_scores)

        for node_id, score in scores.items():
            if node_id in self.citation_graph:
                self.citation_graph[node_id].pagerank_score = score
                self.influence_scores[node_id] = score

    def _identify_foundational(self) -> None:
        for node_id, node in self.citation_graph.items():
            if node.cited_by_count >= 3:
                node.is_foundational = True

    def recommend_high_influence(self, top_k: int = 5) -> list[RecommendationResult]:
        sorted_nodes = sorted(
            self.citation_graph.values(),
            key=lambda x: (x.pagerank_score, x.cited_by_count),
            reverse=True,
        )[:top_k]

        return [
            RecommendationResult(
                paper=self.loader.papers[node.paper_id],
                score=node.pagerank_score * 100,
                reason=f"高影响力论文 (被引{node.cited_by_count}次)",
            )
            for node in sorted_nodes
            if node.paper_id in self.loader.papers
        ]

    def recommend_foundational(self, top_k: int = 5) -> list[RecommendationResult]:
        foundational = [
            node for node in self.citation_graph.values() if node.is_foundational
        ]
        foundational.sort(key=lambda x: x.year if hasattr(x, "year") else 0)

        results = []
        for node in foundational[:top_k]:
            if node.paper_id in self.loader.papers:
                paper = self.loader.papers[node.paper_id]
                results.append(
                    RecommendationResult(
                        paper=paper,
                        score=node.pagerank_score * 100,
                        reason=f"基础论文 (被引{node.cited_by_count}次)",
                    )
                )
        return results

    def recommend_citation_path(
        self, start_paper_id: str, end_paper_id: str, max_depth: int = 5
    ) -> list[RecommendationResult]:
        if (
            start_paper_id not in self.loader.papers
            or end_paper_id not in self.loader.papers
        ):
            return []

        queue: list[tuple[str, list[str]]] = [(start_paper_id, [start_paper_id])]
        visited: set[str] = {start_paper_id}
        paths: list[list[str]] = []

        while queue and len(paths) < 3:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue

            paper = self.loader.papers.get(current)
            if not paper:
                continue

            for cited in paper.citations:
                if cited == end_paper_id:
                    paths.append(path + [cited])
                elif cited not in visited:
                    visited.add(cited)
                    queue.append((cited, path + [cited]))

        results = []
        for path in paths:
            for i, paper_id in enumerate(path[1:], 1):
                if paper_id in self.loader.papers:
                    paper = self.loader.papers[paper_id]
                    results.append(
                        RecommendationResult(
                            paper=paper,
                            score=(len(path) - i) / len(path),
                            reason=f"引用路径上的关键论文 (第{i}步)",
                        )
                    )
        return results[:5]

    def recommend_next_read(
        self, paper_id: str, top_k: int = 5
    ) -> list[RecommendationResult]:
        if paper_id not in self.loader.papers:
            return []

        paper = self.loader.papers[paper_id]
        results: list[tuple[Paper, float, str]] = []

        for cited_id in paper.citations:
            if cited_id in self.loader.papers:
                cited_paper = self.loader.papers[cited_id]
                score = (
                    1.0
                    + self.citation_graph.get(
                        cited_id, CitationNode("", 0, 0)
                    ).pagerank_score
                )
                results.append((cited_paper, score, "本文引用"))

        for citing_id in paper.cited_by:
            if citing_id in self.loader.papers:
                citing_paper = self.loader.papers[citing_id]
                score = (
                    0.8
                    + self.citation_graph.get(
                        citing_id, CitationNode("", 0, 0)
                    ).pagerank_score
                    * 0.5
                )
                results.append((citing_paper, score, "引用本文"))

        related_pairs = self.loader.relations.get("paper_related_to_paper", [])
        for p1, p2 in related_pairs:
            if p1 == paper_id and p2 in self.loader.papers:
                related_paper = self.loader.papers[p2]
                score = 0.6
                results.append((related_paper, score, "相关论文"))
            elif p2 == paper_id and p1 in self.loader.papers:
                related_paper = self.loader.papers[p1]
                score = 0.6
                results.append((related_paper, score, "相关论文"))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            RecommendationResult(paper=paper, score=score, reason=reason)
            for paper, score, reason in results
        ]

    def get_citation_network_stats(self) -> dict[str, Any]:
        total_citations = sum(n.cites_count for n in self.citation_graph.values())
        total_cited_by = sum(n.cited_by_count for n in self.citation_graph.values())
        foundational_count = sum(
            1 for n in self.citation_graph.values() if n.is_foundational
        )

        return {
            "total_papers": len(self.citation_graph),
            "total_citation_relations": total_citations,
            "avg_citations_per_paper": total_citations / len(self.citation_graph)
            if self.citation_graph
            else 0,
            "foundational_papers": foundational_count,
            "top_influential": [
                (self.loader.papers[n.paper_id].title, n.pagerank_score)
                for n in sorted(
                    self.citation_graph.values(),
                    key=lambda x: x.pagerank_score,
                    reverse=True,
                )[:5]
                if n.paper_id in self.loader.papers
            ],
        }


if __name__ == "__main__":
    loader = DataLoader().load()
    recommender = CitationRecommender(loader)

    print("=== 高影响力论文 ===")
    for r in recommender.recommend_high_influence(3):
        print(f"  {r.paper.title} (score: {r.score:.2f})")
        print(f"    原因: {r.reason}")

    print("\n=== 引用网络统计 ===")
    stats = recommender.get_citation_network_stats()
    print(f"  总论文数: {stats['total_papers']}")
    print(f"  基础论文数: {stats['foundational_papers']}")
