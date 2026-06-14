from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
from src.services.macro_intelligence.query.ast import ASTNode, BinaryOpNode, KeywordNode, FieldOpNode, NotNode
from src.services.macro_intelligence.query.query_context import QueryContext
from src.services.macro_intelligence.query.planner import QueryPlan

@dataclass
class SearchResult:
    event_id: str
    score: float
    matched_fields: List[str]
    matched_terms: List[str]
    expanded_by: List[str]
    explanation: Dict[str, float]

class RankingEngine:
    """Calculates final scores for search results combining multiple factors."""
    
    def score(self, event_id: str, context: QueryContext, matched_terms: List[str]) -> SearchResult:
        # Mock weights for illustration
        text_match_score = 0.4
        relationship_weight = 0.2
        graph_centrality = 0.15
        priority = 0.1
        recency = 0.1
        ai_confidence = 0.05
        
        # Calculate a deterministic mock score based on term counts and ID for now
        base_score = 0.5 + (len(matched_terms) * 0.1)
        final_score = min(base_score, 1.0)
        
        explanation = {
            "Text Match": text_match_score * final_score,
            "Relationship Weight": relationship_weight * final_score,
            "Graph Centrality": graph_centrality * final_score,
            "Priority": priority * final_score,
            "Recency": recency * final_score,
            "AI Confidence": ai_confidence * final_score
        }
        
        return SearchResult(
            event_id=event_id,
            score=round(final_score, 3),
            matched_fields=["title", "content"], # Placeholder
            matched_terms=matched_terms,
            expanded_by=[], # Placeholder for graph expansion
            explanation=explanation
        )

class Executor:
    """Executes a QueryPlan against the QueryContext and returns Ranked SearchResults."""
    
    def __init__(self, context: QueryContext, ranking_engine: RankingEngine):
        self.context = context
        self.ranking_engine = ranking_engine
        
    def execute(self, plan: QueryPlan) -> List[SearchResult]:
        matched_event_ids, matched_terms = self._evaluate(plan.ast)
        
        results = []
        for event_id in matched_event_ids:
            # We pass matched_terms corresponding to this event, for now pass all matched terms globally
            result = self.ranking_engine.score(event_id, self.context, list(matched_terms))
            results.append(result)
            
        # Sort by descending score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
        
    def _evaluate(self, node: ASTNode) -> tuple[Set[str], Set[str]]:
        """Returns a tuple of (matched_event_ids, matched_terms)"""
        if node is None:
            return set(), set()
            
        if isinstance(node, KeywordNode):
            term = node.value.lower()
            ids = set(self.context.search_index.get(term, []))
            return ids, {term}
            
        elif isinstance(node, BinaryOpNode):
            left_ids, left_terms = self._evaluate(node.left)
            right_ids, right_terms = self._evaluate(node.right)
            
            if node.operator == "AND":
                return left_ids.intersection(right_ids), left_terms.union(right_terms)
            elif node.operator == "OR":
                return left_ids.union(right_ids), left_terms.union(right_terms)
                
        elif isinstance(node, NotNode):
            # NOT is typically applied relative to a base set, simplified here
            # In a real engine, we'd subtract from the universal set of all document IDs.
            # Assuming all docs exist in context.dashboard
            all_ids = {doc.event_id for doc in self.context.dashboard} if self.context.dashboard else set()
            inner_ids, inner_terms = self._evaluate(node.operand)
            return all_ids - inner_ids, set()
            
        elif isinstance(node, FieldOpNode):
            # In a real implementation, we would iterate dashboard or secondary index
            # Mock filtering
            matched_ids = set()
            for doc in self.context.dashboard:
                # Naive matching for demonstration
                val = getattr(doc.official_data, node.field, None)
                if val is None:
                    # Try metadata
                    if doc.metadata and node.field in doc.metadata:
                        val = doc.metadata[node.field]
                        
                if val is not None:
                    if node.operator == ":" and str(val).lower() == str(node.value).lower():
                        matched_ids.add(doc.event_id)
                    elif node.operator == ">" and float(val) > float(node.value):
                        matched_ids.add(doc.event_id)
            return matched_ids, {f"{node.field}:{node.value}"}
            
        return set(), set()
