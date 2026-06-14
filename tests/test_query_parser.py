import pytest
from src.services.macro_intelligence.query.lexer import Lexer
from src.services.macro_intelligence.query.parser import Parser, ParserError
from src.services.macro_intelligence.query.ast import KeywordNode, FieldOpNode, NotNode, BinaryOpNode

def test_parser_basic_keywords():
    lexer = Lexer("repo liquidity")
    parser = Parser(lexer.tokenize())
    ast = parser.parse()
    
    assert isinstance(ast, BinaryOpNode)
    assert ast.operator == "AND"
    assert isinstance(ast.left, KeywordNode)
    assert ast.left.value == "repo"
    assert isinstance(ast.right, KeywordNode)
    assert ast.right.value == "liquidity"

def test_parser_field_ops():
    lexer = Lexer("confidence>90 AND category:RBI")
    parser = Parser(lexer.tokenize())
    ast = parser.parse()
    
    assert isinstance(ast, BinaryOpNode)
    assert ast.operator == "AND"
    
    assert isinstance(ast.left, FieldOpNode)
    assert ast.left.field == "confidence"
    assert ast.left.operator == ">"
    assert ast.left.value == "90"
    
    assert isinstance(ast.right, FieldOpNode)
    assert ast.right.field == "category"
    assert ast.right.operator == ":"
    assert ast.right.value == "RBI"

def test_parser_complex_grouping():
    lexer = Lexer("repo AND (policy OR NOT forex)")
    parser = Parser(lexer.tokenize())
    ast = parser.parse()
    
    # AST: AND(repo, OR(policy, NOT(forex)))
    assert isinstance(ast, BinaryOpNode)
    assert ast.operator == "AND"
    assert isinstance(ast.left, KeywordNode)
    assert ast.left.value == "repo"
    
    right_grp = ast.right
    assert isinstance(right_grp, BinaryOpNode)
    assert right_grp.operator == "OR"
    assert isinstance(right_grp.left, KeywordNode)
    assert right_grp.left.value == "policy"
    
    assert isinstance(right_grp.right, NotNode)
    assert isinstance(right_grp.right.operand, KeywordNode)
    assert right_grp.right.operand.value == "forex"
