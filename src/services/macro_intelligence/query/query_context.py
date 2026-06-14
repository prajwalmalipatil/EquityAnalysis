from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class QueryContext:
    search_index: Dict[str, Any]
    relationships: List[Any]
    graph: Dict[str, Any]
    analytics: Dict[str, Any]
    dashboard: List[Any]
    
    def estimate_hits(self, term: str) -> int:
        """Returns cardinality estimation for a term using the search index."""
        if term in self.search_index:
            return len(self.search_index[term])
        return 0
