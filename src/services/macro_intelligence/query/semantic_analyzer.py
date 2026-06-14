from src.services.macro_intelligence.query.ast import ASTNode, KeywordNode, FieldOpNode, NotNode, BinaryOpNode

class SemanticError(Exception):
    pass

class SemanticAnalyzer:
    """Traverses the AST and validates field types, operators, and semantics."""
    
    # Define valid fields, their expected types, and allowed operators
    FIELD_SCHEMAS = {
        "confidence": {"type": "numeric", "operators": {">", "<", ">=", "<=", "="}},
        "category": {"type": "string", "operators": {":"}},
        "source": {"type": "string", "operators": {":"}},
        "priority": {"type": "string", "operators": {":"}},
        "type": {"type": "string", "operators": {":"}},
        "theme": {"type": "string", "operators": {":"}},
        "date": {"type": "date", "operators": {">", "<", ">=", "<=", "="}}
    }
    
    def analyze(self, node: ASTNode) -> ASTNode:
        """Validates the AST and returns a semantically checked AST (values may be casted)."""
        if node is None:
            return None
            
        if isinstance(node, BinaryOpNode):
            left = self.analyze(node.left)
            right = self.analyze(node.right)
            return BinaryOpNode(operator=node.operator, left=left, right=right)
            
        elif isinstance(node, NotNode):
            operand = self.analyze(node.operand)
            return NotNode(operand=operand)
            
        elif isinstance(node, FieldOpNode):
            field = node.field.lower()
            if field not in self.FIELD_SCHEMAS:
                raise SemanticError(f"Unknown field: '{field}'")
                
            schema = self.FIELD_SCHEMAS[field]
            
            if node.operator not in schema["operators"]:
                raise SemanticError(f"Operator '{node.operator}' not allowed for field '{field}'")
                
            # Type cast/validation
            val = node.value
            if schema["type"] == "numeric":
                try:
                    val = float(val)
                except ValueError:
                    raise SemanticError(f"Field '{field}' expects a numeric value, got '{val}'")
                    
            elif schema["type"] == "date":
                # Very simple date check for now
                if len(val) < 4:
                    raise SemanticError(f"Field '{field}' expects a valid date string, got '{val}'")
                    
            return FieldOpNode(field=field, operator=node.operator, value=val)
            
        elif isinstance(node, KeywordNode):
            return node
            
        else:
            raise SemanticError(f"Unknown AST node type: {type(node)}")
