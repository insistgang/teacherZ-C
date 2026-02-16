"""
论文推荐系统 - 数据加载器
"""

import json
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


@dataclass
class Paper:
    id: str
    title: str
    year: int
    venue: str
    arxiv: str | None = None
    methods: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    cited_by: list[str] = field(default_factory=list)
    difficulty: str = "medium"


@dataclass
class Method:
    id: str
    name: str
    full_name: str
    category: str
    description: str


@dataclass
class Domain:
    id: str
    name: str
    description: str


class DataLoader:
    def __init__(self, base_path: str = "D:/Documents/zx"):
        self.base_path = Path(base_path)
        self.entities: dict[str, Any] = {}
        self.relations: dict[str, Any] = {}
        self.papers: dict[str, Paper] = {}
        self.methods: dict[str, Method] = {}
        self.domains: dict[str, Domain] = {}
        self.method_index: dict[str, list[str]] = {}
        self.domain_index: dict[str, list[str]] = {}

    def load(self) -> "DataLoader":
        entities_path = self.base_path / "knowledge_graph" / "entities.json"
        relations_path = self.base_path / "knowledge_graph" / "relations.json"

        with open(entities_path, "r", encoding="utf-8") as f:
            self.entities = json.load(f)
        with open(relations_path, "r", encoding="utf-8") as f:
            self.relations = json.load(f)

        self._build_papers()
        self._build_methods()
        self._build_domains()
        self._build_relations()
        self._build_indices()
        self._compute_difficulty()

        return self

    def _build_papers(self) -> None:
        for p in self.entities.get("papers", []):
            self.papers[p["id"]] = Paper(
                id=p["id"],
                title=p["title"],
                year=p["year"],
                venue=p.get("venue", "Unknown"),
                arxiv=p.get("arxiv"),
            )

    def _build_methods(self) -> None:
        for m in self.entities.get("methods", []):
            self.methods[m["id"]] = Method(
                id=m["id"],
                name=m["name"],
                full_name=m["full_name"],
                category=m["category"],
                description=m["description"],
            )

    def _build_domains(self) -> None:
        for d in self.entities.get("domains", []):
            self.domains[d["id"]] = Domain(
                id=d["id"], name=d["name"], description=d["description"]
            )

    def _build_relations(self) -> None:
        for paper_id, method_id in self.relations.get("paper_uses_method", []):
            if paper_id in self.papers:
                self.papers[paper_id].methods.append(method_id)

        for paper_id, domain_id in self.relations.get("paper_applies_to_domain", []):
            if paper_id in self.papers:
                self.papers[paper_id].domains.append(domain_id)

        for citing, cited in self.relations.get("paper_cites_paper", []):
            if citing in self.papers:
                self.papers[citing].citations.append(cited)
            if cited in self.papers:
                self.papers[cited].cited_by.append(citing)

    def _build_indices(self) -> None:
        for paper_id, method_id in self.relations.get("paper_uses_method", []):
            if method_id not in self.method_index:
                self.method_index[method_id] = []
            self.method_index[method_id].append(paper_id)

        for paper_id, domain_id in self.relations.get("paper_applies_to_domain", []):
            if domain_id not in self.domain_index:
                self.domain_index[domain_id] = []
            self.domain_index[domain_id].append(paper_id)

    def _compute_difficulty(self) -> None:
        foundational_methods = {"M001", "M002", "M005", "M006"}
        advanced_methods = {"M015", "M016", "M017", "M018", "M019"}

        for paper_id, paper in self.papers.items():
            score = 0
            if any(m in foundational_methods for m in paper.methods):
                score -= 1
            if any(m in advanced_methods for m in paper.methods):
                score += 1
            if paper.year >= 2023:
                score += 1
            if len(paper.cited_by) > 2:
                score -= 1

            if score <= -1:
                paper.difficulty = "beginner"
            elif score >= 1:
                paper.difficulty = "advanced"
            else:
                paper.difficulty = "medium"

    def get_papers_by_method(self, method_id: str) -> list[Paper]:
        return [
            self.papers[pid]
            for pid in self.method_index.get(method_id, [])
            if pid in self.papers
        ]

    def get_papers_by_domain(self, domain_id: str) -> list[Paper]:
        return [
            self.papers[pid]
            for pid in self.domain_index.get(domain_id, [])
            if pid in self.papers
        ]

    def search_papers(self, query: str) -> list[Paper]:
        query_lower = query.lower()
        results = []
        for paper in self.papers.values():
            if (
                query_lower in paper.title.lower()
                or any(
                    query_lower in self.methods[m].name.lower()
                    for m in paper.methods
                    if m in self.methods
                )
                or any(
                    query_lower in self.domains[d].name.lower()
                    for d in paper.domains
                    if d in self.domains
                )
            ):
                results.append(paper)
        return results

    def get_method_name(self, method_id: str) -> str:
        return self.methods.get(method_id, Method("", "", "", "", "")).name

    def get_domain_name(self, domain_id: str) -> str:
        return self.domains.get(domain_id, Domain("", "", "")).name


if __name__ == "__main__":
    loader = DataLoader().load()
    print(
        f"Loaded {len(loader.papers)} papers, {len(loader.methods)} methods, {len(loader.domains)} domains"
    )
