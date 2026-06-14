from dataclasses import dataclass
from typing import Any

class ASTNode:
    """Base class for all Abstract Syntax Tree nodes in the query."""
    pass

@dataclass(frozen=True)
class KeywordNode(ASTNode):
    value: str

@dataclass(frozen=True)
class FieldOpNode(ASTNode):
    field: str
    operator: str
    value: Any

@dataclass(frozen=True)
class NotNode(ASTNode):
    operand: ASTNode

@dataclass(frozen=True)
class BinaryOpNode(ASTNode):
    operator: str  # "AND" or "OR"
    left: ASTNode
    right: ASTNode
