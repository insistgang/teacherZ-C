"""
论文推荐系统 - 基于内容的推荐
"""

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
from data_loader import DataLoader, Paper


@dataclass
class RecommendationResult:
    paper: Paper
    score: float
    reason: str
    match_details: dict[str, Any] | None = None


class ContentRecommender:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.method_weights: dict[str, float] = {}
        self.domain_weights: dict[str, float] = {}
        self._build_idf()

    def _build_idf(self) -> None:
        total_papers = len(self.loader.papers)
        method_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}

        for paper in self.loader.papers.values():
            for method_id in paper.methods:
                method_counts[method_id] = method_counts.get(method_id, 0) + 1
            for domain_id in paper.domains:
                domain_counts[domain_id] = domain_counts.get(domain_id, 0) + 1

        import math

        for method_id, count in method_counts.items():
            self.method_weights[method_id] = math.log(total_papers / (1 + count))
        for domain_id, count in domain_counts.items():
            self.domain_weights[domain_id] = math.log(total_papers / (1 + count))

    def compute_similarity(self, paper1: Paper, paper2: Paper) -> float:
        score = 0.0
        method_overlap = set(paper1.methods) & set(paper2.methods)
        domain_overlap = set(paper1.domains) & set(paper2.domains)

        for method_id in method_overlap:
            score += self.method_weights.get(method_id, 1.0) * 2
        for domain_id in domain_overlap:
            score += self.domain_weights.get(domain_id, 1.0) * 1.5

        if paper1.year and paper2.year:
            year_diff = abs(paper1.year - paper2.year)
            year_bonus = max(0, 1 - year_diff / 10)
            score += year_bonus

        return score

    def recommend_similar(
        self, paper_id: str, top_k: int = 5
    ) -> list[RecommendationResult]:
        if paper_id not in self.loader.papers:
            return []

        target_paper = self.loader.papers[paper_id]
        results: list[tuple[Paper, float]] = []

        for pid, paper in self.loader.papers.items():
            if pid == paper_id:
                continue
            sim = self.compute_similarity(target_paper, paper)
            if sim > 0:
                results.append((paper, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        recommendations = []
        for paper, score in results:
            method_overlap = set(target_paper.methods) & set(paper.methods)
            domain_overlap = set(target_paper.domains) & set(paper.domains)

            method_names = [self.loader.get_method_name(m) for m in method_overlap]
            domain_names = [self.loader.get_domain_name(d) for d in domain_overlap]

            reason_parts = []
            if method_names:
                reason_parts.append(f"共同方法: {', '.join(method_names)}")
            if domain_names:
                reason_parts.append(f"共同领域: {', '.join(domain_names)}")

            recommendations.append(
                RecommendationResult(
                    paper=paper,
                    score=score,
                    reason="; ".join(reason_parts) if reason_parts else "相关论文",
                    match_details={
                        "methods": list(method_overlap),
                        "domains": list(domain_overlap),
                    },
                )
            )

        return recommendations

    def recommend_by_methods(
        self,
        method_ids: list[str],
        exclude_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RecommendationResult]:
        exclude_set = set(exclude_ids or [])
        results: list[tuple[Paper, float]] = []

        for paper in self.loader.papers.values():
            if paper.id in exclude_set:
                continue

            overlap = set(method_ids) & set(paper.methods)
            if not overlap:
                continue

            score = sum(self.method_weights.get(m, 1.0) for m in overlap)
            results.append((paper, score))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            RecommendationResult(
                paper=paper,
                score=score,
                reason=f"使用方法: {', '.join(self.loader.get_method_name(m) for m in (set(method_ids) & set(paper.methods)))}",
            )
            for paper, score in results
        ]

    def recommend_by_domains(
        self,
        domain_ids: list[str],
        exclude_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RecommendationResult]:
        exclude_set = set(exclude_ids or [])
        results: list[tuple[Paper, float]] = []

        for paper in self.loader.papers.values():
            if paper.id in exclude_set:
                continue

            overlap = set(domain_ids) & set(paper.domains)
            if not overlap:
                continue

            score = sum(self.domain_weights.get(d, 1.0) for d in overlap)
            results.append((paper, score))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            RecommendationResult(
                paper=paper,
                score=score,
                reason=f"应用领域: {', '.join(self.loader.get_domain_name(d) for d in (set(domain_ids) & set(paper.domains)))}",
            )
            for paper, score in results
        ]

    def recommend_by_keywords(
        self, keywords: list[str], top_k: int = 5
    ) -> list[RecommendationResult]:
        results: list[tuple[Paper, float]] = []
        keywords_lower = [k.lower() for k in keywords]

        for paper in self.loader.papers.values():
            score = 0.0
            matched_methods = []

            for method_id in paper.methods:
                if method_id in self.loader.methods:
                    method = self.loader.methods[method_id]
                    for kw in keywords_lower:
                        if (
                            kw in method.name.lower()
                            or kw in method.full_name.lower()
                            or kw in method.description.lower()
                        ):
                            score += 2.0
                            matched_methods.append(method.name)

            for domain_id in paper.domains:
                if domain_id in self.loader.domains:
                    domain = self.loader.domains[domain_id]
                    for kw in keywords_lower:
                        if (
                            kw in domain.name.lower()
                            or kw in domain.description.lower()
                        ):
                            score += 1.5

            for kw in keywords_lower:
                if kw in paper.title.lower():
                    score += 3.0

            if score > 0:
                results.append((paper, score, matched_methods))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            RecommendationResult(
                paper=paper,
                score=score,
                reason=f"关键词匹配: {', '.join(matched_methods[:3])}"
                if matched_methods
                else "标题匹配",
            )
            for paper, score, matched_methods in results
        ]

    def get_paper_feature_vector(self, paper: Paper) -> dict[str, float]:
        features: dict[str, float] = {}

        for method_id in paper.methods:
            features[f"method_{method_id}"] = self.method_weights.get(method_id, 1.0)
        for domain_id in paper.domains:
            features[f"domain_{domain_id}"] = self.domain_weights.get(domain_id, 1.0)

        features["year"] = paper.year / 2025.0
        features["citation_count"] = len(paper.cited_by)

        return features


if __name__ == "__main__":
    loader = DataLoader().load()
    recommender = ContentRecommender(loader)

    print("=== 测试: 与SLaT相似的论文 ===")
    results = recommender.recommend_similar("P005")
    for r in results:
        print(f"  {r.paper.title} (score: {r.score:.2f})")
        print(f"    原因: {r.reason}")
