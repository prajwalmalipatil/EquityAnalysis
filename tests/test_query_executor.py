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

def test_executor_field_operator():
    from src.services.macro_intelligence.models import MacroEvent, OfficialData, EventMetadata
    from src.services.macro_intelligence.query.ast import FieldOpNode
    
    # Setup mock event with metadata
    event = MacroEvent(
        event_id="ev-1",
        official_data=OfficialData(
            title="Policy Update",
            publication_date="2026-06-15",
            category="Notice",
            source="RBI",
            official_url="http://rbi/1"
        ),
        derived_data=None,
        metadata=EventMetadata(
            processing_state="COMPLETED",
            lifecycle_status="ACTIVE",
            created_at="2026-06-15",
            updated_at="2026-06-15"
        )
    )
    
    # Context
    context = QueryContext(search_index={}, relationships=[], graph={}, analytics={}, dashboard=[event])
    
    # query: lifecycle_status:ACTIVE
    ast = FieldOpNode(operator=":", field="lifecycle_status", value="ACTIVE")
    
    planner = Planner(Optimizer(context))
    plan = planner.plan(ast)
    
    executor = Executor(context, RankingEngine())
    results = executor.execute(plan)
    
    assert len(results) == 1
    assert results[0].event_id == "ev-1"
    
    # query: category:Notice
    ast2 = FieldOpNode(operator=":", field="category", value="Notice")
    plan2 = planner.plan(ast2)
    results2 = executor.execute(plan2)
    assert len(results2) == 1
    assert results2[0].event_id == "ev-1"
