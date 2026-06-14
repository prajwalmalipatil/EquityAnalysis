import pytest
from src.services.macro_intelligence.query.ast import BinaryOpNode, KeywordNode
from src.services.macro_intelligence.query.query_context import QueryContext
from src.services.macro_intelligence.query.optimizer import Optimizer
from src.services.macro_intelligence.query.planner import Planner

def test_optimizer_reordering():
    # Context where "repo" has 1000 hits and "policy" has 12 hits
    idx = {"repo": [1]*1000, "policy": [1]*12}
    context = QueryContext(search_index=idx, relationships=[], graph={}, analytics={}, dashboard=[])
    optimizer = Optimizer(context)
    
    # Original AST: repo AND policy (left=repo, right=policy)
    ast = BinaryOpNode(operator="AND", left=KeywordNode("repo"), right=KeywordNode("policy"))
    
    # Optimizer should flip it because policy has lower cost
    opt_ast = optimizer.optimize(ast)
    
    assert opt_ast.operator == "AND"
    assert opt_ast.left.value == "policy"
    assert opt_ast.right.value == "repo"

def test_planner_steps():
    idx = {"repo": [1]*1000, "policy": [1]*12}
    context = QueryContext(search_index=idx, relationships=[], graph={}, analytics={}, dashboard=[])
    optimizer = Optimizer(context)
    planner = Planner(optimizer)
    
    ast = BinaryOpNode(operator="AND", left=KeywordNode("repo"), right=KeywordNode("policy"))
    
    plan = planner.plan(ast)
    
    assert plan.estimated_hits == 12
    assert plan.execution_strategy == "COMPLEX_EVALUATION"
    
    # policy should be searched first
    assert plan.steps[0] == "Index lookup: 'policy'"
    assert plan.steps[1] == "Index lookup: 'repo'"
    assert plan.steps[2] == "Execute AND intersection/union"
