import pytest
from src.services.macro_intelligence.query.lexer import Lexer
from src.services.macro_intelligence.query.parser import Parser
from src.services.macro_intelligence.query.ast import FieldOpNode
from src.services.macro_intelligence.query.semantic_analyzer import SemanticAnalyzer, SemanticError

def test_semantic_analyzer_valid():
    query = "confidence>90 AND category:RBI"
    ast = Parser(Lexer(query).tokenize()).parse()
    
    analyzer = SemanticAnalyzer()
    enriched_ast = analyzer.analyze(ast)
    
    # confidence 90 should be cast to float 90.0
    assert isinstance(enriched_ast.left, FieldOpNode)
    assert enriched_ast.left.field == "confidence"
    assert enriched_ast.left.value == 90.0
    
    # category RBI should remain string
    assert isinstance(enriched_ast.right, FieldOpNode)
    assert enriched_ast.right.field == "category"
    assert enriched_ast.right.value == "RBI"

def test_semantic_analyzer_invalid_type():
    query = "confidence>high"
    ast = Parser(Lexer(query).tokenize()).parse()
    
    analyzer = SemanticAnalyzer()
    with pytest.raises(SemanticError) as exc:
        analyzer.analyze(ast)
        
    assert "expects a numeric value" in str(exc.value)

def test_semantic_analyzer_invalid_operator():
    query = "category>RBI"
    ast = Parser(Lexer(query).tokenize()).parse()
    
    analyzer = SemanticAnalyzer()
    with pytest.raises(SemanticError) as exc:
        analyzer.analyze(ast)
        
    assert "not allowed for field" in str(exc.value)

def test_semantic_analyzer_unknown_field():
    query = "unknown:val"
    ast = Parser(Lexer(query).tokenize()).parse()
    
    analyzer = SemanticAnalyzer()
    with pytest.raises(SemanticError) as exc:
        analyzer.analyze(ast)
        
    assert "Unknown field" in str(exc.value)
