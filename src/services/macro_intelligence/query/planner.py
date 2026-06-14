from dataclasses import dataclass
from typing import List, Dict, Any
from src.services.macro_intelligence.query.ast import ASTNode

@dataclass
class QueryPlan:
    ast: ASTNode
    estimated_cost: int
    estimated_hits: int
    execution_strategy: str
    steps: List[str]

class Planner:
    """Transforms an optimized AST into a logical Execution Plan with diagnostics."""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def plan(self, ast: ASTNode) -> QueryPlan:
        optimized_ast = self.optimizer.optimize(ast)
        estimated_hits = self.optimizer._estimate_cost(optimized_ast)
        
        # Simple step generation for diagnostic purposes
        steps = self._generate_steps(optimized_ast)
        
        strategy = "INDEX_SCAN" if type(optimized_ast).__name__ == "KeywordNode" else "COMPLEX_EVALUATION"
        
        return QueryPlan(
            ast=optimized_ast,
            estimated_cost=estimated_hits, # simple proxy
            estimated_hits=estimated_hits,
            execution_strategy=strategy,
            steps=steps
        )
        
    def _generate_steps(self, node: ASTNode) -> List[str]:
        if node is None:
            return []
            
        name = type(node).__name__
        if name == "BinaryOpNode":
            steps = self._generate_steps(node.left)
            steps.extend(self._generate_steps(node.right))
            steps.append(f"Execute {node.operator} intersection/union")
            return steps
        elif name == "KeywordNode":
            return [f"Index lookup: '{node.value}'"]
        elif name == "FieldOpNode":
            return [f"Filter field '{node.field}' {node.operator} {node.value}"]
        elif name == "NotNode":
            steps = self._generate_steps(node.operand)
            steps.append("Invert result set (NOT)")
            return steps
        return []
