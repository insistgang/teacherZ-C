"""
Knowledge Graph Query Examples for Xiaohao Cai Papers

This module provides query functions for the knowledge graph.
Supports both in-memory JSON-LD queries and Neo4j Cypher queries.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class KnowledgeGraphQuery:
    """In-memory knowledge graph query engine using JSON-LD data."""
    
    def __init__(self, jsonld_path: str = None):
        if jsonld_path is None:
            jsonld_path = Path(__file__).parent / "knowledge_graph.jsonld"
        
        with open(jsonld_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.graph = self.data.get('@graph', [])
        self._build_indices()
    
    def _build_indices(self):
        """Build indices for fast lookups."""
        self.papers = {}
        self.authors = {}
        self.methods = {}
        self.domains = {}
        self.tools = {}
        
        for entity in self.graph:
            entity_type = entity.get('type', '')
            entity_id = entity.get('id', '')
            
            if 'Paper' in entity_type:
                self.papers[entity_id] = entity
            elif 'Author' in entity_type:
                self.authors[entity_id] = entity
            elif 'Method' in entity_type:
                self.methods[entity_id] = entity
            elif 'Domain' in entity_type:
                self.domains[entity_id] = entity
            elif 'Tool' in entity_type:
                self.tools[entity_id] = entity
    
    def find_papers_by_method(self, method_name: str) -> List[Dict]:
        """
        Find all papers that use a specific method.
        
        Example: find_papers_by_method('ROF')
        """
        method_id = None
        for mid, m in self.methods.items():
            if m.get('name', '').lower() == method_name.lower():
                method_id = mid
                break
        
        if not method_id:
            return []
        
        results = []
        for pid, paper in self.papers.items():
            uses = paper.get('uses', [])
            if isinstance(uses, str):
                uses = [uses]
            if method_id in uses:
                results.append({
                    'id': pid,
                    'title': paper.get('title'),
                    'year': paper.get('year'),
                    'venue': paper.get('venue')
                })
        
        return sorted(results, key=lambda x: x.get('year', 0))
    
    def find_papers_by_domain(self, domain_name: str) -> List[Dict]:
        """
        Find all papers that apply to a specific domain.
        
        Example: find_papers_by_domain('Medical Imaging')
        """
        domain_id = None
        for did, d in self.domains.items():
            if domain_name.lower() in d.get('name', '').lower():
                domain_id = did
                break
        
        if not domain_id:
            return []
        
        results = []
        for pid, paper in self.papers.items():
            applies = paper.get('appliesTo', [])
            if isinstance(applies, str):
                applies = [applies]
            if domain_id in applies:
                results.append({
                    'id': pid,
                    'title': paper.get('title'),
                    'year': paper.get('year'),
                    'venue': paper.get('venue')
                })
        
        return sorted(results, key=lambda x: x.get('year', 0))
    
    def find_method_evolution(self, method_name: str) -> Dict:
        """
        Find the complete evolution path of a method.
        
        Example: find_method_evolution('SLaT')
        Returns: {
            'method': {...},
            'inherits_from': [...],
            'inherited_by': [...],
            'papers_using': [...]
        }
        """
        method_id = None
        method_info = None
        
        for mid, m in self.methods.items():
            if m.get('name', '').lower() == method_name.lower():
                method_id = mid
                method_info = m
                break
        
        if not method_id:
            return {'error': f'Method {method_name} not found'}
        
        ancestors = []
        current_id = method_id
        visited = {method_id}
        
        while True:
            current_method = self.methods.get(current_id)
            if not current_method:
                break
            parent = current_method.get('inheritsFrom')
            if parent and parent not in visited:
                visited.add(parent)
                parent_method = self.methods.get(parent, {})
                ancestors.append({
                    'id': parent,
                    'name': parent_method.get('name'),
                    'full_name': parent_method.get('full_name')
                })
                current_id = parent
            else:
                break
        
        descendants = []
        for mid, m in self.methods.items():
            if m.get('inheritsFrom') == method_id:
                descendants.append({
                    'id': mid,
                    'name': m.get('name'),
                    'full_name': m.get('full_name')
                })
        
        papers = self.find_papers_by_method(method_name)
        
        return {
            'method': {
                'id': method_id,
                'name': method_info.get('name'),
                'full_name': method_info.get('full_name'),
                'category': method_info.get('category'),
                'description': method_info.get('description')
            },
            'inherits_from': ancestors,
            'inherited_by': descendants,
            'papers_using': papers
        }
    
    def find_papers_after_year(self, year: int, domain: str = None) -> List[Dict]:
        """
        Find all papers published after a specific year.
        Optionally filter by domain.
        
        Example: find_papers_after_year(2020, domain='3D Vision')
        """
        results = []
        
        for pid, paper in self.papers.items():
            paper_year = paper.get('year', 0)
            if paper_year > year:
                if domain:
                    applies = paper.get('appliesTo', [])
                    if isinstance(applies, str):
                        applies = [applies]
                    
                    domain_match = False
                    for did in applies:
                        d = self.domains.get(did, {})
                        if domain.lower() in d.get('name', '').lower():
                            domain_match = True
                            break
                    
                    if not domain_match:
                        continue
                
                results.append({
                    'id': pid,
                    'title': paper.get('title'),
                    'year': paper_year,
                    'venue': paper.get('venue'),
                    'arxiv': paper.get('arxiv')
                })
        
        return sorted(results, key=lambda x: x.get('year', 0))
    
    def find_variational_medical_papers(self) -> List[Dict]:
        """
        Find all papers that use variational methods and apply to medical imaging.
        
        Query: "找出所有使用变分方法的医学图像论文"
        """
        variational_methods = []
        for mid, m in self.methods.items():
            if m.get('category', '').lower() == 'variational':
                variational_methods.append(mid)
        
        medical_domain = None
        for did, d in self.domains.items():
            if 'medical' in d.get('name', '').lower():
                medical_domain = did
                break
        
        if not medical_domain:
            return []
        
        results = []
        for pid, paper in self.papers.items():
            uses = paper.get('uses', [])
            if isinstance(uses, str):
                uses = [uses]
            
            applies = paper.get('appliesTo', [])
            if isinstance(applies, str):
                applies = [applies]
            
            has_variational = any(umid in variational_methods for umid in uses)
            has_medical = medical_domain in applies
            
            if has_variational and has_medical:
                method_names = [self.methods.get(m, {}).get('name', '') for m in uses if m in variational_methods]
                results.append({
                    'id': pid,
                    'title': paper.get('title'),
                    'year': paper.get('year'),
                    'variational_methods': method_names
                })
        
        return sorted(results, key=lambda x: x.get('year', 0))
    
    def find_citation_chain(self, paper_title: str) -> Dict:
        """
        Find the citation chain for a paper.
        
        Example: find_citation_chain('SLaT Three-stage Segmentation')
        """
        target_paper = None
        for pid, p in self.papers.items():
            if paper_title.lower() in p.get('title', '').lower():
                target_paper = {'id': pid, **p}
                break
        
        if not target_paper:
            return {'error': f'Paper {paper_title} not found'}
        
        cited_by = []
        cites = []
        
        target_id = target_paper['id']
        
        for pid, paper in self.papers.items():
            paper_cites = paper.get('cites', [])
            if isinstance(paper_cites, str):
                paper_cites = [paper_cites]
            
            if target_id in paper_cites:
                cited_by.append({
                    'id': pid,
                    'title': paper.get('title'),
                    'year': paper.get('year')
                })
        
        for cited_id in target_paper.get('cites', []):
            cited_paper = self.papers.get(cited_id, {})
            cites.append({
                'id': cited_id,
                'title': cited_paper.get('title'),
                'year': cited_paper.get('year')
            })
        
        return {
            'paper': {
                'id': target_id,
                'title': target_paper.get('title'),
                'year': target_paper.get('year')
            },
            'cites': cites,
            'cited_by': cited_by
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            'total_papers': len(self.papers),
            'total_authors': len(self.authors),
            'total_methods': len(self.methods),
            'total_domains': len(self.domains),
            'total_tools': len(self.tools),
            'papers_by_year': self._count_by_year(),
            'methods_by_category': self._count_methods_by_category(),
            'papers_by_domain': self._count_papers_by_domain()
        }
    
    def _count_by_year(self) -> Dict[int, int]:
        counts = {}
        for paper in self.papers.values():
            year = paper.get('year', 0)
            counts[year] = counts.get(year, 0) + 1
        return dict(sorted(counts.items()))
    
    def _count_methods_by_category(self) -> Dict[str, int]:
        counts = {}
        for method in self.methods.values():
            cat = method.get('category', 'Unknown')
            counts[cat] = counts.get(cat, 0) + 1
        return counts
    
    def _count_papers_by_domain(self) -> Dict[str, int]:
        counts = {}
        for paper in self.papers.values():
            applies = paper.get('appliesTo', [])
            if isinstance(applies, str):
                applies = [applies]
            for did in applies:
                domain_name = self.domains.get(did, {}).get('name', 'Unknown')
                counts[domain_name] = counts.get(domain_name, 0) + 1
        return counts


class Neo4jQueryExamples:
    """Neo4j Cypher query examples."""
    
    FIND_VARIATIONAL_MEDICAL = """
    // Query: 找出所有使用变分方法的医学图像论文
    MATCH (p:Paper)-[:USES]->(m:Method)
    WHERE m.category = 'Variational'
    AND (p)-[:APPLIES_TO]->(:Domain {name: 'Medical Imaging'})
    RETURN p.title, p.year, collect(m.name) as methods
    ORDER BY p.year;
    """
    
    FIND_ROF_EVOLUTION = """
    // Query: ROF 方法的完整演进路径
    MATCH path = (m:Method)-[:INHERITS_FROM*0..]->(ancestor:Method)
    WHERE m.name = 'ROF' OR ancestor.name = 'ROF'
    RETURN path;
    
    // Alternative: Get full evolution tree
    MATCH (m:Method)
    WHERE m.name IN ['ROF', 'T-ROF', 'SaT', 'SLaT']
    OPTIONAL MATCH (m)-[:INHERITS_FROM]->(parent:Method)
    RETURN m.name as method, parent.name as inherits_from;
    """
    
    FIND_3D_VISION_AFTER_2020 = """
    // Query: 2020年后发表的3D视觉论文
    MATCH (p:Paper)-[:APPLIES_TO]->(d:Domain)
    WHERE d.name CONTAINS '3D'
    AND p.year > 2020
    RETURN p.title, p.year, p.arxiv, d.name as domain
    ORDER BY p.year;
    """
    
    FIND_METHOD_PAPERS = """
    // Query: 找出使用特定方法的所有论文
    MATCH (p:Paper)-[:USES]->(m:Method {name: $method_name})
    RETURN p.id, p.title, p.year, p.venue
    ORDER BY p.year;
    """
    
    FIND_AUTHOR_COLLABORATION = """
    // Query: 作者合作网络
    MATCH (a1:Author)-[:WRITES]->(p:Paper)<-[:WRITES]-(a2:Author)
    WHERE a1.name = 'Xiaohao Cai' AND a2.name <> 'Xiaohao Cai'
    RETURN a2.name as collaborator, count(p) as collaboration_count
    ORDER BY collaboration_count DESC;
    """
    
    FIND_CITATION_NETWORK = """
    // Query: 论文引用网络
    MATCH (p1:Paper)-[:CITES]->(p2:Paper)
    RETURN p1.title as citing_paper, p1.year, p2.title as cited_paper, p2.year as cited_year
    ORDER BY p1.year;
    """
    
    STATISTICS = """
    // Query: 知识图谱统计信息
    MATCH (p:Paper)
    WITH count(p) as paper_count
    MATCH (a:Author)
    WITH paper_count, count(a) as author_count
    MATCH (m:Method)
    WITH paper_count, author_count, count(m) as method_count
    MATCH (d:Domain)
    RETURN paper_count, author_count, method_count, count(d) as domain_count;
    """


def demo():
    """Demo function showing query examples."""
    kg = KnowledgeGraphQuery()
    
    print("=" * 60)
    print("Knowledge Graph Query Examples")
    print("=" * 60)
    
    print("\n1. Statistics:")
    stats = kg.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n2. Papers using ROF method:")
    rof_papers = kg.find_papers_by_method('ROF')
    for p in rof_papers[:5]:
        print(f"  [{p['year']}] {p['title']}")
    
    print("\n3. Method evolution for SLaT:")
    slat_evolution = kg.find_method_evolution('SLaT')
    print(f"  Method: {slat_evolution['method']['name']}")
    print(f"  Inherits from: {[m['name'] for m in slat_evolution['inherits_from']]}")
    print(f"  Papers using: {len(slat_evolution['papers_using'])}")
    
    print("\n4. 3D Vision papers after 2020:")
    papers_3d = kg.find_papers_after_year(2020, domain='3D Vision')
    for p in papers_3d[:5]:
        print(f"  [{p['year']}] {p['title']}")
    
    print("\n5. Variational methods in Medical Imaging:")
    var_med = kg.find_variational_medical_papers()
    for p in var_med:
        print(f"  [{p['year']}] {p['title']} - Methods: {p['variational_methods']}")


if __name__ == "__main__":
    demo()
