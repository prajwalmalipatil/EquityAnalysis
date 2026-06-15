from src.services.macro_intelligence.query.ast import ASTNode, BinaryOpNode, KeywordNode, FieldOpNode, NotNode
from src.services.macro_intelligence.query.query_context import QueryContext

class Optimizer:
    """Reorders AST nodes based on cardinality to optimize execution (e.g. smallest sets first)."""
    
    def __init__(self, context: QueryContext):
        self.context = context
        
    def optimize(self, node: ASTNode) -> ASTNode:
        if node is None:
            return None
            
        if isinstance(node, BinaryOpNode):
            left = self.optimize(node.left)
            right = self.optimize(node.right)
            
            # If it's an AND operation, put the node with lower estimated hits on the left.
            if node.operator == "AND":
                cost_l = self.estimate_cost(left)
                cost_r = self.estimate_cost(right)
                if cost_r < cost_l:
                    return BinaryOpNode(operator="AND", left=right, right=left)
                    
            return BinaryOpNode(operator=node.operator, left=left, right=right)
            
        elif isinstance(node, NotNode):
            return NotNode(operand=self.optimize(node.operand))
            
        return node
        
    def estimate_cost(self, node: ASTNode) -> int:
        """Rough estimation of how many documents this node matches."""
        if isinstance(node, KeywordNode):
            return self.context.estimate_hits(node.value.lower())
        elif isinstance(node, BinaryOpNode):
            cost_l = self.estimate_cost(node.left)
            cost_r = self.estimate_cost(node.right)
            if node.operator == "AND":
                return min(cost_l, cost_r) # Intersection bounds
            elif node.operator == "OR":
                return cost_l + cost_r # Union bounds
        elif isinstance(node, FieldOpNode):
            # We don't have secondary indices yet, so assume field ops are more expensive (full scan or large subset).
            return 1000 
        return 1000
