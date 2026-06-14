from dataclasses import dataclass
from typing import List, Dict, Any
from src.services.macro_intelligence.query.executor import SearchResult

@dataclass
class SearchViewModel:
    query: str
    execution_time_ms: int
    total_hits: int
    results: List[Dict[str, Any]]

class SearchViewModelBuilder:
    @staticmethod
    def build(query: str, results: List[SearchResult], execution_time_ms: int) -> SearchViewModel:
        mapped_results = []
        for r in results:
            mapped_results.append({
                "event_id": r.event_id,
                "score": r.score,
                "matched_fields": r.matched_fields,
                "matched_terms": r.matched_terms,
                "expanded_by": r.expanded_by,
                "explanation": r.explanation
            })
            
        return SearchViewModel(
            query=query,
            execution_time_ms=execution_time_ms,
            total_hits=len(results),
            results=mapped_results
        )
