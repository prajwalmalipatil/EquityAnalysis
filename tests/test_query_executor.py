import pytest
from src.services.macro_intelligence.query.ast import BinaryOpNode, KeywordNode
from src.services.macro_intelligence.query.query_context import QueryContext
from src.services.macro_intelligence.query.optimizer import Optimizer
from src.services.macro_intelligence.query.planner import Planner
from src.services.macro_intelligence.query.executor import Executor, RankingEngine

def test_executor_and_ranking():
    # Context
    idx = {"repo": ["ev-1", "ev-2", "ev-3"], "policy": ["ev-2", "ev-4"]}
    context = QueryContext(search_index=idx, relationships=[], graph={}, analytics={}, dashboard=[])
    
    # query: repo AND policy
    ast = BinaryOpNode(operator="AND", left=KeywordNode("repo"), right=KeywordNode("policy"))
    
    planner = Planner(Optimizer(context))
    plan = planner.plan(ast)
    
    executor = Executor(context, RankingEngine())
    results = executor.execute(plan)
    
    # Only ev-2 has both
    assert len(results) == 1
    assert results[0].event_id == "ev-2"
    assert "repo" in results[0].matched_terms
    assert "policy" in results[0].matched_terms
    
    # Assert explainability structure
    assert "Text Match" in results[0].explanation
    assert results[0].score > 0
